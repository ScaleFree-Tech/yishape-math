package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.vecidx.VecSearchOption;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 纯 Java 实现的 HNSW（Hierarchical Navigable Small World）单精度向量索引。
 *
 * <p>算法参考 Malkov &amp; Yashunin, 2018 及 hnswlib 开源实现。核心特征：</p>
 * <ul>
 *   <li>多层可导航小世界图，上层稀疏、下层密集；</li>
 *   <li>插入时通过概率分布决定节点层数，保持图的分形结构；</li>
 *   <li>贪心搜索 + beam search（ef 控制）保证近似精度；</li>
 *   <li>启发式邻居选择（heuristic pruning）避免所有节点连向同一中心，
 *       维持图的导航性能；</li>
 *   <li>Cosine 度量通过<strong>预归一化</strong>优化，内部退化为欧氏距离计算。</li>
 * </ul>
 *
 * <p>线程安全：{@link StampedLock} 实现多读单写。搜索可并发，
 * 插入/删除/清空互斥。当前不支持 {@code remove}（与 Rust FFI 版行为一致）。</p>
 */
public class RereHnswFloatVecIdx implements IFloatVecIdx, IMutableVecIdx<Float> {

    private static final long serialVersionUID = 1L;

    // ==================== 内部数据结构 ====================

    /**
     * HNSW 图节点。
     *
     * <p>并发策略（匹配 hnsw_rs 的 parking_lot::RwLock）：</p>
     * <ul>
     *   <li>连接列表预分配最大容量，add 不会触发数组扩容；</li>
     *   <li>并行构建期间：<strong>读不加锁</strong>（ArrayList.get 无扩容时线程安全），
     *       写通过 {@code synchronized(this)} 保护；</li>
     *   <li>构建完成后图只读，无需同步。</li>
     * </ul>
     */
    private static final class Node {
        final String id;
        final float[] vector;
        final float normSq;
        final int intId;
        /** connections[level] 为该节点在第 level 层的邻居列表（预分配最大容量，写需 synchronized） */
        final ArrayList<Node>[] connections;

        @SuppressWarnings("unchecked")
        Node(String id, int intId, float[] vector, int maxLayer, int layer0MaxConn, int layerMaxConn) {
            this.id = id;
            this.intId = intId;
            this.vector = vector;
            float ns = 0;
            for (float v : vector) ns += v * v;
            this.normSq = ns;
            this.connections = new ArrayList[maxLayer + 1];
            this.connections[0] = new ArrayList<>(layer0MaxConn);
            for (int i = 1; i <= maxLayer; i++) {
                this.connections[i] = new ArrayList<>(layerMaxConn);
            }
        }
    }

    /** 搜索候选，封装节点与到 query 的距离 */
    private static final class Candidate {
        final Node node;
        final float dist;

        Candidate(Node node, float dist) {
            this.node = node;
            this.dist = dist;
        }
    }

    // ==================== 配置与状态 ====================

    private final int dimensions;
    private final IDisMetric<Float> metric;
    private final VecSearchOption options;
    private final int M;
    private final int M0; // layer 0 的连接上限
    private final int efConstruction;
    private final int efSearch;
    private volatile int currentEfSearch; // 可动态调参（对应 Rust 的 AtomicUsize ef_search）
    private final double mL;
    private final boolean normalize; // cosine 模式下预先归一化
    private final boolean distanceIsSquared; // 外部度量是否为 squared_euclidean
    private final Random random;

    private final ConcurrentHashMap<String, Node> nodes = new ConcurrentHashMap<>();
    private volatile Node entryPoint;
    private volatile int maxLevel;
    private final AtomicInteger nextIntId = new AtomicInteger(0);
    private final StampedLock lock = new StampedLock();
    /** 并行构建期间的线程池（构建完成后置 null） */
    private volatile ExecutorService buildPool;

    // 每线程的 visited 标记：searchLayer 用 visitedStamp[intId] == currentStamp 判断节点是否已访问，
    // 避免 HashSet<String> 的 String 哈希与 Entry 分配开销
    private static final class VisitedBuf {
        int[] stamp;
        int currentStamp; // 每次 searchLayer 自增一次，整型溢出时清零
        VisitedBuf() {
            this.stamp = new int[256];
            this.currentStamp = 0;
        }
    }

    private final ThreadLocal<VisitedBuf> visitedTL = ThreadLocal.withInitial(VisitedBuf::new);

    /** 取出本线程的 visited 标记数组，必要时扩容；调用方需在 search/insert 期间使用 */
    private VisitedBuf acquireVisited() {
        VisitedBuf buf = visitedTL.get();
        int nid = nextIntId.get();
        if (buf.stamp.length < nid) {
            int newCap = Math.max(nid, buf.stamp.length * 2);
            buf.stamp = new int[newCap];
            buf.currentStamp = 0;
        }
        buf.currentStamp++;
        if (buf.currentStamp == Integer.MAX_VALUE) {
            java.util.Arrays.fill(buf.stamp, 0);
            buf.currentStamp = 1;
        }
        return buf;
    }

    // ==================== 构造函数 ====================

    public RereHnswFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        this.dimensions = validateAndGetDims(data, ids);
        this.metric = Objects.requireNonNull(metric, "metric");
        this.options = options != null ? options : VecSearchOption.DEFAULT;
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "Java HNSW only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;
        this.M = Math.max(2, this.options.hnswM());
        this.M0 = 2 * this.M;
        this.efConstruction = Math.max(1, this.options.hnswEfConstruction());
        this.efSearch = Math.max(1, this.options.hnswEfSearch());
        this.currentEfSearch = this.efSearch;
        this.mL = 1.0 / Math.log(this.M);
        this.random = new Random(42);

        if (ids.length == 0) {
            return;
        }

        // 预归一化（cosine 模式）
        float[][] processed = new float[ids.length][];
        for (int i = 0; i < ids.length; i++) {
            processed[i] = normalize ? normalize(data[i]) : data[i].clone();
        }

        // 少于阈值用串行构建，否则并行
        if (ids.length < 1000) {
            for (int i = 0; i < ids.length; i++) {
                insertNode(ids[i], processed[i]);
            }
        } else {
            parallelBatchInsert(ids, processed);
        }
    }

    private static int validateAndGetDims(float[][] data, String[] ids) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        if (data.length != ids.length) {
            throw new IllegalArgumentException("data rows must equal ids length");
        }
        if (ids.length == 0) {
            return 0;
        }
        int d = data[0].length;
        for (int i = 1; i < data.length; i++) {
            if (data[i] == null || data[i].length != d) {
                throw new IllegalArgumentException("all vectors must have consistent dimensions");
            }
        }
        return d;
    }

    // ==================== 核心算法：搜索一层 ====================

    /**
     * 在第 {@code level} 层搜索 query 的最近 {@code ef} 个邻居。
     *
     * @param parallel 并行构建模式。此时读不加锁（预分配容量保证无扩容），
     *                 仅写路径在 insertNodeInternal 中 synchronized。
     */
    private List<Candidate> searchLayer(float[] query, Node entry, int ef, int level, boolean parallel) {
        PriorityQueue<Candidate> candidates = new PriorityQueue<>(
                (a, b) -> Float.compare(a.dist, b.dist));
        PriorityQueue<Candidate> found = new PriorityQueue<>(
                (a, b) -> Float.compare(b.dist, a.dist));

        VisitedBuf vbuf = acquireVisited();
        int[] visitedStamp = vbuf.stamp;
        int qs = vbuf.currentStamp;

        float d0 = distance(query, entry);
        candidates.add(new Candidate(entry, d0));
        found.add(new Candidate(entry, d0));
        visitedStamp[entry.intId] = qs;

        while (!candidates.isEmpty()) {
            Candidate curr = candidates.poll();
            Candidate farthestFound = found.peek();

            if (curr.dist > farthestFound.dist && found.size() >= ef) {
                break;
            }

            // 读不加锁：ArrayList 预分配容量，get(i) 在无扩容时线程安全
            ArrayList<Node> neighbors = curr.node.connections[level];
            int nn = neighbors.size();
            for (int ni = 0; ni < nn; ni++) {
                Node neighbor = neighbors.get(ni);
                int nid = neighbor.intId;
                if (visitedStamp[nid] == qs) continue;
                visitedStamp[nid] = qs;
                float nd = distance(query, neighbor);
                farthestFound = found.peek();
                if (nd < farthestFound.dist || found.size() < ef) {
                    candidates.add(new Candidate(neighbor, nd));
                    found.add(new Candidate(neighbor, nd));
                    if (found.size() > ef) {
                        found.poll();
                    }
                }
            }
        }

        List<Candidate> result = new ArrayList<>(found);
        result.sort((a, b) -> Float.compare(a.dist, b.dist));
        return result;
    }

    // ==================== 核心算法：贪心搜索（单层最近点）====================

    /**
     * 在第 {@code level} 层贪心搜索 query 的最近点（ef=1）。
     *
     * @param parallel 未使用（读路径始终无锁），保留以统一调用签名。
     */
    private Candidate greedySearch(float[] query, Node entry, int level, boolean parallel) {
        Node curr = entry;
        float currDist = distance(query, curr);
        boolean improved;
        do {
            improved = false;
            // 索引遍历（非 Iterator），避免并行修改时 ConcurrentModificationException
            ArrayList<Node> neighbors = curr.connections[level];
            for (int ni = 0, nn = neighbors.size(); ni < nn; ni++) {
                Node neighbor = neighbors.get(ni);
                float nd = distance(query, neighbor);
                if (nd < currDist) {
                    curr = neighbor;
                    currDist = nd;
                    improved = true;
                }
            }
        } while (improved);
        return new Candidate(curr, currDist);
    }

    // ==================== 核心算法：启发式邻居选择 ====================

    /**
     * hnswlib 的启发式邻居选择：从候选集中选出最多 {@code m} 个连接，
     * 优先保留能维持图导航多样性的节点。
     *
     * <p>对于候选 c，若结果集中已存在 r 使得 dist(c,r) &lt; dist(c,query)，
     * 则 c 被 r "覆盖"，不加入结果。</p>
     */
    private List<Node> selectNeighbors(List<Candidate> candidates, int m, float[] queryVector, boolean parallel) {
        List<Node> result = new ArrayList<>(m);
        Set<Node> added = new HashSet<>(m * 2);
        for (Candidate c : candidates) {
            if (result.size() >= m) {
                break;
            }
            boolean dominated = false;
            for (Node r : result) {
                float dcr = distance(c.node, r);
                if (dcr < c.dist) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                result.add(c.node);
                added.add(c.node);
            }
        }
        // 启发式结果不足时，用最近节点补充
        if (result.size() < m) {
            for (Candidate c : candidates) {
                if (result.size() >= m) {
                    break;
                }
                if (added.add(c.node)) {
                    result.add(c.node);
                }
            }
        }
        return result;
    }

    // ==================== 核心算法：插入节点 ====================

    private void insertNode(String id, float[] vector) {
        Node newNode = new Node(id, nextIntId.getAndIncrement(), vector, randomLevel(), M0 * 2, M * 2);
        nodes.put(id, newNode);
        insertNodeInternal(newNode, false);
    }

    /**
     * 统一的节点插入逻辑（使用预创建的 Node，不再分配 intId）。
     *
     * @param parallel true 表示并行构建模式，读写连接列表时将 synchronized(node)；
     *                 false 为串行模式（无需同步）。
     */
    private void insertNodeInternal(Node newNode, boolean parallel) {
        int nodeLevel = newNode.connections.length - 1;

        if (entryPoint == null) {
            entryPoint = newNode;
            maxLevel = nodeLevel;
            return;
        }

        Node currEntry = entryPoint;
        int currLevel = maxLevel;

        for (int level = currLevel; level > nodeLevel; level--) {
            Candidate nearest = greedySearch(newNode.vector, currEntry, level, parallel);
            currEntry = nearest.node;
        }

        for (int level = Math.min(nodeLevel, currLevel); level >= 0; level--) {
            int ef = level == 0 ? Math.max(efConstruction, M0) : efConstruction;
            int mConn = level == 0 ? M0 : M;

            List<Candidate> candidates = searchLayer(newNode.vector, currEntry, ef, level, parallel);
            List<Node> neighbors = selectNeighbors(candidates, mConn, newNode.vector, parallel);

            for (Node neighbor : neighbors) {
                if (parallel) {
                    // 连接修改以 RCU（Read-Copy-Update）方式安全进行：
                    // 构建新列表然后原子替换引用，旧列表保留给正在读的线程继续使用。
                    synchronized (newNode) {
                        newNode.connections[level] = appendAndPrune(newNode.connections[level], neighbor, mConn, newNode.vector);
                    }
                    synchronized (neighbor) {
                        neighbor.connections[level] = appendAndPrune(neighbor.connections[level], newNode, mConn, neighbor.vector);
                    }
                } else {
                    newNode.connections[level].add(neighbor);
                    neighbor.connections[level].add(newNode);
                    if (neighbor.connections[level].size() > mConn) {
                        pruneConnections(neighbor, level, mConn);
                    }
                }
            }

            if (!candidates.isEmpty()) {
                currEntry = candidates.get(0).node;
            }
        }

        // CAS 更新全局 entryPoint（并行模式下可能有多线程竞争）
        if (nodeLevel > maxLevel) {
            synchronized (this) {
                if (nodeLevel > maxLevel) {
                    maxLevel = nodeLevel;
                    entryPoint = newNode;
                }
            }
        }
    }

    /**
     * 当节点在某层的连接数超过上限时，启发式裁剪只保留最近的 {@code mConn} 个。
     */
    private void pruneConnections(Node node, int level, int mConn) {
        List<Node> all = node.connections[level];
        // 构建候选列表（按到 node 的距离排序）
        List<Candidate> candidates = new ArrayList<>(all.size());
        for (Node neighbor : all) {
            candidates.add(new Candidate(neighbor, distance(node, neighbor)));
        }
        candidates.sort((a, b) -> Float.compare(a.dist, b.dist));

        List<Node> pruned = selectNeighbors(candidates, mConn, node.vector, false);
        all.clear();
        all.addAll(pruned);
    }

    /**
     * RCU 风格：复制当前连接列表，添加新邻居，若超限则用 selectNeighbors 启发式裁剪，
     * 返回新列表（旧列表不受影响，并发读线程仍可安全遍历）。
     */
    private static ArrayList<Node> appendAndPrune(ArrayList<Node> cur, Node toAdd, int mConn, float[] nodeVec) {
        ArrayList<Node> next = new ArrayList<>(cur.size() + 1);
        next.addAll(cur);
        next.add(toAdd);
        if (next.size() <= mConn) {
            return next;
        }
        // 构建候选列表，用 selectNeighbors 逻辑保持图导航多样性
        List<Candidate> candidates = new ArrayList<>(next.size());
        for (Node n : next) {
            candidates.add(new Candidate(n, distance(nodeVec, n)));
        }
        candidates.sort((a, b) -> Float.compare(a.dist, b.dist));
        // 启发式邻居选择（内联 selectNeighbors 的核心逻辑）
        ArrayList<Node> result = new ArrayList<>(mConn);
        boolean[] selected = new boolean[candidates.size()];
        for (int i = 0; i < candidates.size() && result.size() < mConn; i++) {
            Candidate c = candidates.get(i);
            boolean dominated = false;
            for (Node r : result) {
                if (distance(c.node, r) < c.dist) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) {
                result.add(c.node);
                selected[i] = true;
            }
        }
        // 启发式结果不足时，用最近节点补充
        if (result.size() < mConn) {
            for (int i = 0; i < candidates.size() && result.size() < mConn; i++) {
                if (!selected[i]) {
                    result.add(candidates.get(i).node);
                }
            }
        }
        return result;
    }

    // ==================== 并行批量构建 ====================

    /**
     * 并行批量插入（hnsw_rs 风格的 Rayon parallel_insert 的 Java 等价实现）。
     *
     * <p>策略：</p>
     * <ol>
     *   <li>预创建所有 Node 并分配随机层数（O(N)，串行，极快）；</li>
     *   <li>按层数降序排列，高层优先（确立上层导航结构）；</li>
     *   <li>先串行插入种子集（~sqrt(N)*2）以建立初始图拓扑；</li>
     *   <li>剩余节点用 {@link ExecutorService} 并行插入，每个线程独立计算距离、
     *       通过 per-node synchronized 安全修改连接列表。</li>
     * </ol>
     */
    private void parallelBatchInsert(String[] ids, float[][] vectors) {
        int n = ids.length;
        int parallelism = Runtime.getRuntime().availableProcessors();
        // hnsw_rs 风格：少量种子串行建立初始拓扑，剩余并行插入
        int seedSize = Math.min(n, Math.max(100, (int) Math.sqrt(n) * 2));

        // Step 1: 预创建所有 Node（获取 nextIntId 必须是顺序的）
        Node[] newNodes = new Node[n];
        int[] levels = new int[n];
        for (int i = 0; i < n; i++) {
            int level = randomLevel();
            levels[i] = level;
            newNodes[i] = new Node(ids[i], nextIntId.getAndIncrement(), vectors[i], level, M0 * 2, M * 2);
        }

        // Step 2: 按层数降序排列索引
        Integer[] sortedIdx = new Integer[n];
        for (int i = 0; i < n; i++) sortedIdx[i] = i;
        java.util.Arrays.sort(sortedIdx, (a, b) -> Integer.compare(levels[b], levels[a]));

        // Step 3: 串行插入种子集，建立初始图结构
        for (int si = 0; si < seedSize; si++) {
            int idx = sortedIdx[si];
            nodes.put(ids[idx], newNodes[idx]);
            insertNodeInternal(newNodes[idx], false);
        }

        // Step 4: 并行插入剩余节点
        int remaining = n - seedSize;
        if (remaining <= 0) return;

        ExecutorService pool = Executors.newWorkStealingPool(parallelism);
        this.buildPool = pool;
        try {
            int batchSize = Math.max(1, remaining / parallelism);
            java.util.concurrent.CountDownLatch latch =
                    new java.util.concurrent.CountDownLatch(parallelism);

            for (int t = 0; t < parallelism; t++) {
                final int start = seedSize + t * batchSize;
                final int end = (t == parallelism - 1) ? n : Math.min(start + batchSize, n);
                if (start >= n) {
                    latch.countDown();
                    continue;
                }
                pool.execute(() -> {
                    try {
                        for (int i = start; i < end; i++) {
                            int idx = sortedIdx[i];
                            Node node = newNodes[idx];
                            nodes.put(node.id, node);
                            insertNodeInternal(node, true);
                        }
                    } finally {
                        latch.countDown();
                    }
                });
            }

            try {
                if (!latch.await(10, TimeUnit.MINUTES)) {
                    throw new RuntimeException("Parallel HNSW build timed out");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Parallel HNSW build interrupted", e);
            }
        } finally {
            pool.shutdown();
            this.buildPool = null;
        }
    }

    // ==================== 层数随机选择 ====================

    private int randomLevel() {
        double r = random.nextDouble();
        int level = (int) Math.floor(-Math.log(r) * mL);
        return Math.max(0, level);
    }

    // ==================== 距离计算（内联优化）====================

    /** query 到 node 的 squared Euclidean distance（单次融合遍历）。 */
    private static float distance(float[] query, Node node) {
        float sum = 0;
        float[] v = node.vector;
        for (int i = 0; i < query.length; i++) {
            float d = query[i] - v[i];
            sum += d * d;
        }
        return sum;
    }

    /** node 到 node 的 squared Euclidean distance（利用缓存的 normSq，一次 dot 遍历）。 */
    private static float distance(Node a, Node b) {
        float dot = 0;
        float[] va = a.vector, vb = b.vector;
        for (int i = 0; i < va.length; i++) dot += va[i] * vb[i];
        return a.normSq + b.normSq - 2 * dot;
    }

    /**
     * 将内部 squared distance 转换为外部度量距离。
     */
    private double externalDistance(float squaredDist) {
        if (normalize) {
            // cosine: 对于归一化向量，cosine_distance = ||a-b||² / 2
            return squaredDist * 0.5;
        }
        if (distanceIsSquared) {
            return squaredDist;
        }
        // euclidean
        return Math.sqrt(squaredDist);
    }

    // ==================== 向量归一化（cosine 模式）====================

    private float[] normalize(float[] v) {
        return IVector.of(v).normalize().toFloatArray();
    }

    // ==================== IFloatVecIdx / IMutableVecIdx API ====================

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public int size() {
        return nodes.size();
    }

    @Override
    public IDisMetric<Float> metric() {
        return metric;
    }

    @Override
    public boolean isApproximate() {
        return true;
    }

    @Override
    public boolean isConcurrent() {
        return true;
    }

    /**
     * 动态设置查询时的 ef_search 参数（对应 Rust 的 AtomicUsize ef_search）。
     * 增大可提升召回率但增加查询延迟。
     */
    public void setEfSearch(int ef) {
        if (ef <= 0) {
            throw new IllegalArgumentException("efSearch 必须为正");
        }
        this.currentEfSearch = ef;
    }

    @Override
    public List<SearchHit> search(float[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        if (k <= 0) {
            return List.of();
        }

        float[] q = normalize ? normalize(query) : query.clone();

        Node ep = entryPoint;
        if (ep == null || nodes.isEmpty()) {
            return List.of();
        }

        // 1. 从最高层开始贪心搜索（图只读，无锁）
        Node curr = ep;
        for (int level = maxLevel; level > 0; level--) {
            Candidate nearest = greedySearch(q, curr, level, false);
            curr = nearest.node;
        }

        // 2. 在 layer 0 做 beam search
        int fetch = Math.max(k + (excludeIds != null ? excludeIds.size() : 0) + 16, currentEfSearch);
        fetch = Math.min(fetch, nodes.size());

        Set<String> ex = excludeIds instanceof Set<String> s ? s
                : new HashSet<>(excludeIds != null ? excludeIds : List.of());
        List<Candidate> candidates = searchLayer(q, curr, fetch, 0, false);

        // 3. 应用过滤和排除
        List<SearchHit> buf = new ArrayList<>(Math.min(k, candidates.size()));
        for (Candidate c : candidates) {
            if (buf.size() >= k) {
                break;
            }
            if (ex.contains(c.node.id) || (filter != null && !filter.test(c.node.id))) {
                continue;
            }
            buf.add(new SearchHit(c.node.id, externalDistance(c.dist)));
        }
        return buf;
    }

    @Override
    public List<SearchHit> search(IVector<Float> query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        Objects.requireNonNull(query, "query");
        if (query.length() != dimensions) {
            throw new IllegalArgumentException("query dimension must be " + dimensions);
        }
        float[] q = query.toFloatArray();
        return search(q, k, excludeIds, filter);
    }

    @Override
    public IVector<Float> getVector(String id) {
        Node node = nodes.get(id);
        if (node == null) {
            return null;
        }
        return IVector.of(node.vector.clone());
    }

    @Override
    public void add(String id, IVector<Float> vector) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(vector, "vector");
        if (vector.length() != dimensions) {
            throw new IllegalArgumentException("vector dimension must be " + dimensions);
        }
        long stamp = lock.writeLock();
        try {
            if (nodes.containsKey(id)) {
                throw new IllegalArgumentException("duplicate ID: " + id);
            }
            float[] vec = normalize ? normalize(vector.toFloatArray()) : vector.toFloatArray();
            insertNode(id, vec);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean remove(String id) {
        long stamp = lock.writeLock();
        try {
            Node node = nodes.remove(id);
            if (node == null) {
                return false;
            }

            int nodeLevel = node.connections.length - 1;
            // 从所有层的所有邻居的连接列表中移除该节点
            for (int level = 0; level <= nodeLevel; level++) {
                ArrayList<Node> neighbors = node.connections[level];
                for (Node neighbor : neighbors) {
                    neighbor.connections[level].remove(node);
                }
                // 清空被删节点自身的连接引用，协助 GC
                neighbors.clear();
            }

            // 如果被删除的是 entryPoint，需要更新
            if (node == entryPoint) {
                if (nodes.isEmpty()) {
                    entryPoint = null;
                    maxLevel = 0;
                } else {
                    Node newEntry = null;
                    int newMaxLevel = -1;
                    for (Node n : nodes.values()) {
                        int level = n.connections.length - 1;
                        if (level > newMaxLevel) {
                            newMaxLevel = level;
                            newEntry = n;
                        }
                    }
                    entryPoint = newEntry;
                    maxLevel = newMaxLevel;
                }
            } else if (nodeLevel == maxLevel && maxLevel > 0) {
                // 检查是否还有其他节点在该最高层
                boolean hasNodeAtMaxLevel = false;
                for (Node n : nodes.values()) {
                    if (n.connections.length - 1 >= maxLevel) {
                        hasNodeAtMaxLevel = true;
                        break;
                    }
                }
                if (!hasNodeAtMaxLevel) {
                    int newMaxLevel = 0;
                    for (Node n : nodes.values()) {
                        newMaxLevel = Math.max(newMaxLevel, n.connections.length - 1);
                    }
                    maxLevel = newMaxLevel;
                }
            }

            return true;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean contains(String id) {
        return nodes.containsKey(id);
    }

    @Override
    public void clear() {
        long stamp = lock.writeLock();
        try {
            nodes.clear();
            entryPoint = null;
            maxLevel = 0;
            nextIntId.set(0);
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public void close() {
        clear();
    }
}
