package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.vecidx.VecSearchOption;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 纯 Java 实现的 KD-Tree 单精度向量索引。
 *
 * <p>核心特征：</p>
 * <ul>
 *   <li><strong>方差最大化分裂</strong>：每层选择 spread 最大的维度作为分裂轴，
 *       树更平衡，查询剪枝更高效；</li>
 *   <li><strong>叶子节点阈值</strong>：子树规模 ≤ 16 时退化为线性数组，
 *       提升缓存局部性；</li>
 *   <li><strong>分支限界 k-NN</strong>：自定义最大堆维护当前 Top-k，
 *       超球面与分裂平面不相交时整棵子树剪枝；</li>
 *   <li><strong>惰性删除 + 自适应重建</strong>：增删标记为 pending，
 *       累积到一定比例后一次性重建平衡树；</li>
 *   <li><strong>原生范围查询</strong>：{@code rangeSearch} 直接在树上剪枝，
 *       不回落全量扫描；</li>
 *   <li><strong>Cosine 预归一化</strong>：同 HNSW 实现，内部退化为欧氏距离。</li>
 * </ul>
 *
 * <p>适用场景：低维（dim ≤ 20）精确检索，时间复杂度 O(log n) 典型，
 * 高维时退化至 O(n) 但仍有剪枝收益。</p>
 *
 * <p>线程安全：{@link StampedLock} 实现多读单写。搜索可并发，
 * 增删/重建/清空互斥。</p>
 */
public class KdTreeFloatVecIdx implements IFloatVecIdx, IMutableVecIdx<Float> {

    private static final long serialVersionUID = 1L;

    /** 叶子节点线性扫描阈值 */
    private static final int LEAF_SIZE = 16;
    /** 触发重建的修改比例（修改数 / 当前规模） */
    private static final double REBUILD_RATIO = 0.25;
    /** 最小重建间隔，防止小数据频繁重建 */
    private static final int MIN_REBUILD_DELTA = 64;

    // ==================== 内部数据结构 ====================

    /** 叶子条目 */
    private static final class LeafEntry {
        final String id;
        final float[] vector;
        boolean deleted;

        LeafEntry(String id, float[] vector) {
            this.id = id;
            this.vector = vector;
        }
    }

    /** KD-Tree 节点（内部节点或叶子） */
    private static final class Node {
        // 内部节点字段
        int splitDim;
        float splitValue;
        Node left, right;

        // 叶子节点字段
        boolean isLeaf;
        ArrayList<LeafEntry> entries;
    }

    /** k-NN 搜索专用最大堆（按 squared distance） */
    private static final class KnnHeap {
        private final String[] ids;
        private final float[] dists;
        private int size;
        private float worstDist;

        KnnHeap(int k) {
            this.ids = new String[k];
            this.dists = new float[k];
            this.worstDist = Float.MAX_VALUE;
        }

        boolean isFull() {
            return size == ids.length;
        }

        float worstDistance() {
            return isFull() ? worstDist : Float.MAX_VALUE;
        }

        void offer(String id, float distSq) {
            if (!isFull()) {
                ids[size] = id;
                dists[size] = distSq;
                size++;
                if (isFull()) {
                    buildMaxHeap();
                    worstDist = dists[0];
                }
            } else if (distSq < worstDist) {
                ids[0] = id;
                dists[0] = distSq;
                siftDown(0);
                worstDist = dists[0];
            }
        }

        private void buildMaxHeap() {
            for (int i = size / 2 - 1; i >= 0; i--) {
                siftDown(i);
            }
        }

        private void siftDown(int i) {
            while (true) {
                int left = 2 * i + 1;
                int right = 2 * i + 2;
                int largest = i;
                if (left < size && dists[left] > dists[largest]) {
                    largest = left;
                }
                if (right < size && dists[right] > dists[largest]) {
                    largest = right;
                }
                if (largest == i) {
                    break;
                }
                swap(i, largest);
                i = largest;
            }
        }

        private void swap(int i, int j) {
            String ti = ids[i];
            float td = dists[i];
            ids[i] = ids[j];
            dists[i] = dists[j];
            ids[j] = ti;
            dists[j] = td;
        }

        List<SearchHit> toSortedList(boolean normalize, boolean distanceIsSquared) {
            List<SearchHit> result = new ArrayList<>(size);
            for (int i = 0; i < size; i++) {
                float d = dists[i];
                double ext;
                if (normalize) {
                    ext = d * 0.5;
                } else if (distanceIsSquared) {
                    ext = d;
                } else {
                    ext = Math.sqrt(d);
                }
                result.add(new SearchHit(ids[i], ext));
            }
            result.sort((a, b) -> Double.compare(a.distance(), b.distance()));
            return result;
        }
    }

    // ==================== 配置与状态 ====================

    private final int dimensions;
    private final IDisMetric<Float> metric;
    private final boolean normalize;
    private final boolean distanceIsSquared;

    private final Map<String, LeafEntry> nodes = new HashMap<>();
    private Node root;
    private int activeSize;
    private int modCount;
    /** 上次重建后新增、尚未并入树的条目 */
    private final List<LeafEntry> recentAdds = new ArrayList<>();

    private final StampedLock lock = new StampedLock();

    // ==================== 构造函数 ====================

    /**
     * 从已有数据构建 KD-Tree。
     */
    public KdTreeFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        this.metric = Objects.requireNonNull(metric, "metric");
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "KD-Tree only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;

        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        if (data.length != ids.length) {
            throw new IllegalArgumentException("data rows must equal ids length");
        }
        if (ids.length == 0) {
            this.dimensions = 0;
            this.root = null;
            this.activeSize = 0;
            return;
        }
        this.dimensions = data[0].length;
        for (int i = 0; i < ids.length; i++) {
            float[] vec = normalize ? normalize(data[i]) : data[i].clone();
            LeafEntry e = new LeafEntry(ids[i], vec);
            nodes.put(ids[i], e);
        }
        this.activeSize = ids.length;
        rebuild();
    }

    /**
     * 构造空 KD-Tree（用于后续动态增删）。
     */
    public KdTreeFloatVecIdx(int dimensions, IDisMetric<Float> metric) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions 必须为正");
        }
        this.dimensions = dimensions;
        this.metric = Objects.requireNonNull(metric, "metric");
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "KD-Tree only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;
        this.root = null;
        this.activeSize = 0;
    }

    // ==================== 核心算法：建树 ====================

    private void rebuild() {
        recentAdds.clear();
        if (nodes.isEmpty()) {
            root = null;
            modCount = 0;
            return;
        }
        LeafEntry[] entries = new LeafEntry[activeSize];
        int idx = 0;
        for (LeafEntry e : nodes.values()) {
            if (!e.deleted) {
                entries[idx++] = e;
            }
        }
        root = buildTree(entries, 0, entries.length);
        modCount = 0;
    }

    private Node buildTree(LeafEntry[] entries, int from, int to) {
        int n = to - from;
        if (n <= LEAF_SIZE) {
            Node leaf = new Node();
            leaf.isLeaf = true;
            leaf.entries = new ArrayList<>(n);
            for (int i = from; i < to; i++) {
                leaf.entries.add(entries[i]);
            }
            return leaf;
        }

        int splitDim = selectSplitDim(entries, from, to);
        Arrays.sort(entries, from, to,
                (a, b) -> Float.compare(a.vector[splitDim], b.vector[splitDim]));
        int mid = from + n / 2;

        boolean allEqual = entries[from].vector[splitDim] == entries[to - 1].vector[splitDim];
        if (!allEqual) {
            while (mid < to - 1 && entries[mid].vector[splitDim] == entries[from].vector[splitDim]) {
                mid++;
            }
            if (mid == to) {
                mid = from + 1;
            }
        }

        Node node = new Node();
        node.splitDim = splitDim;
        node.splitValue = entries[mid].vector[splitDim];
        node.left = buildTree(entries, from, mid);
        node.right = buildTree(entries, mid, to);
        return node;
    }

    /**
     * 选择 spread（max - min）最大的维度作为分裂轴。
     */
    private int selectSplitDim(LeafEntry[] entries, int from, int to) {
        double maxSpread = -1.0;
        int bestDim = 0;
        for (int d = 0; d < dimensions; d++) {
            float min = Float.POSITIVE_INFINITY;
            float max = Float.NEGATIVE_INFINITY;
            for (int i = from; i < to; i++) {
                float v = entries[i].vector[d];
                if (v < min) {
                    min = v;
                }
                if (v > max) {
                    max = v;
                }
            }
            double spread = (double) max - (double) min;
            if (spread > maxSpread) {
                maxSpread = spread;
                bestDim = d;
            }
        }
        return bestDim;
    }

    // ==================== 核心算法：k-NN 搜索 ====================

    /**
     * KD-Tree branch-and-bound k-NN 搜索。
     *
     * <p>对于 far 侧子节点，仅使用分裂维度上的平方距离作为下界进行剪枝，
     * 当该距离 ≥ 当前第 k 近距离时剪枝整棵子树。</p>
     */
    private void searchKnn(float[] query, Node node, KnnHeap heap,
            Set<String> excluded, Predicate<String> filter) {
        if (node == null) {
            return;
        }

        if (node.isLeaf) {
            for (LeafEntry e : node.entries) {
                if (e.deleted) {
                    continue;
                }
                if (excluded != null && excluded.contains(e.id)) {
                    continue;
                }
                if (filter != null && !filter.test(e.id)) {
                    continue;
                }
                float d = distance(query, e.vector);
                heap.offer(e.id, d);
            }
            return;
        }

        int dim = node.splitDim;
        float diff = query[dim] - node.splitValue;
        Node first, second;
        if (diff < 0) {
            first = node.left;
            second = node.right;
        } else {
            first = node.right;
            second = node.left;
        }

        searchKnn(query, first, heap, excluded, filter);

        float diffSq = diff * diff;
        if (!heap.isFull() || diffSq < heap.worstDistance()) {
            searchKnn(query, second, heap, excluded, filter);
        }
    }

    // ==================== 核心算法：范围查询 ====================

    private void rangeSearch(float[] query, float radiusSq, Node node,
            List<SearchHit> result, Set<String> excluded, Predicate<String> filter) {
        if (node == null) {
            return;
        }

        if (node.isLeaf) {
            for (LeafEntry e : node.entries) {
                if (e.deleted) {
                    continue;
                }
                if (excluded != null && excluded.contains(e.id)) {
                    continue;
                }
                if (filter != null && !filter.test(e.id)) {
                    continue;
                }
                float d = distance(query, e.vector);
                if (d <= radiusSq) {
                    double ext;
                    if (normalize) {
                        ext = d * 0.5;
                    } else if (distanceIsSquared) {
                        ext = d;
                    } else {
                        ext = Math.sqrt(d);
                    }
                    result.add(new SearchHit(e.id, ext));
                }
            }
            return;
        }

        int dim = node.splitDim;
        float diff = query[dim] - node.splitValue;
        Node first, second;
        if (diff < 0) {
            first = node.left;
            second = node.right;
        } else {
            first = node.right;
            second = node.left;
        }

        rangeSearch(query, radiusSq, first, result, excluded, filter);

        float diffSq = diff * diff;
        if (diffSq <= radiusSq) {
            rangeSearch(query, radiusSq, second, result, excluded, filter);
        }
    }

    // ==================== 距离计算与向量工具 ====================

    private static float distance(float[] a, float[] b) {
        return Linalg.squaredDistance(a, b);
    }

    private float[] normalize(float[] v) {
        return IVector.of(v).normalize().toFloatArray();
    }

    private void maybeRebuild() {
        if (activeSize == 0) {
            root = null;
            recentAdds.clear();
            modCount = 0;
            return;
        }
        if (root == null) {
            rebuild();
            return;
        }
        int threshold = Math.max(MIN_REBUILD_DELTA, (int) (activeSize * REBUILD_RATIO));
        if (modCount >= threshold) {
            rebuild();
        }
    }

    // ==================== API 实现 ====================

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public int size() {
        long stamp = lock.readLock();
        try {
            return activeSize;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public IDisMetric<Float> metric() {
        return metric;
    }

    @Override
    public boolean isApproximate() {
        return false;
    }

    @Override
    public boolean isConcurrent() {
        return true;
    }

    @Override
    public List<SearchHit> search(float[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        if (k <= 0 || activeSize == 0) {
            return List.of();
        }
        float[] q = normalize ? normalize(query) : query.clone();

        long stamp = lock.readLock();
        try {
            if (root == null) {
                return List.of();
            }
            Set<String> ex = excludeIds instanceof Set<String> s ? s
                    : new HashSet<>(excludeIds != null ? excludeIds : List.of());
            KnnHeap heap = new KnnHeap(Math.min(k, activeSize));
            if (root != null) {
                searchKnn(q, root, heap, ex, filter);
            }
            // 扫描上次重建后新增的条目
            for (LeafEntry e : recentAdds) {
                if (e.deleted) {
                    continue;
                }
                if (ex.contains(e.id) || (filter != null && !filter.test(e.id))) {
                    continue;
                }
                float d = distance(q, e.vector);
                heap.offer(e.id, d);
            }
            return heap.toSortedList(normalize, distanceIsSquared);
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public List<SearchHit> rangeSearch(float[] query, double radius, Collection<String> excludeIds, Predicate<String> filter) {
        if (radius < 0 || activeSize == 0) {
            return List.of();
        }
        float[] q = normalize ? normalize(query) : query.clone();
        float radiusSq;
        if (normalize) {
            radiusSq = (float) (radius * 2.0);
        } else if (distanceIsSquared) {
            radiusSq = (float) radius;
        } else {
            radiusSq = (float) (radius * radius);
        }

        long stamp = lock.readLock();
        try {
            if (root == null) {
                return List.of();
            }
            Set<String> ex = excludeIds instanceof Set<String> s ? s
                    : new HashSet<>(excludeIds != null ? excludeIds : List.of());
            List<SearchHit> result = new ArrayList<>();
            if (root != null) {
                rangeSearch(q, radiusSq, root, result, ex, filter);
            }
            // 扫描上次重建后新增的条目
            for (LeafEntry e : recentAdds) {
                if (e.deleted) {
                    continue;
                }
                if (ex.contains(e.id) || (filter != null && !filter.test(e.id))) {
                    continue;
                }
                float d = distance(q, e.vector);
                if (d <= radiusSq) {
                    double ext;
                    if (normalize) {
                        ext = d * 0.5;
                    } else if (distanceIsSquared) {
                        ext = d;
                    } else {
                        ext = Math.sqrt(d);
                    }
                    result.add(new SearchHit(e.id, ext));
                }
            }
            result.sort((a, b) -> Double.compare(a.distance(), b.distance()));
            return result;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public IVector<Float> getVector(String id) {
        long stamp = lock.readLock();
        try {
            LeafEntry e = nodes.get(id);
            if (e == null || e.deleted) {
                return null;
            }
            return IVector.of(e.vector.clone());
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void add(String id, IVector<Float> vector) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(vector, "vector");
        if (dimensions > 0 && vector.length() != dimensions) {
            throw new IllegalArgumentException("vector dimension must be " + dimensions);
        }
        long stamp = lock.writeLock();
        try {
            if (nodes.containsKey(id)) {
                throw new IllegalArgumentException("duplicate ID: " + id);
            }
            float[] vec = normalize ? normalize(vector.toFloatArray()) : vector.toFloatArray();
            LeafEntry e = new LeafEntry(id, vec);
            nodes.put(id, e);
            recentAdds.add(e);
            activeSize++;
            modCount++;
            maybeRebuild();
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean remove(String id) {
        long stamp = lock.writeLock();
        try {
            LeafEntry e = nodes.get(id);
            if (e == null || e.deleted) {
                return false;
            }
            e.deleted = true;
            activeSize--;
            modCount++;
            maybeRebuild();
            return true;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean contains(String id) {
        long stamp = lock.readLock();
        try {
            LeafEntry e = nodes.get(id);
            return e != null && !e.deleted;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void clear() {
        long stamp = lock.writeLock();
        try {
            nodes.clear();
            root = null;
            recentAdds.clear();
            activeSize = 0;
            modCount = 0;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public void close() {
        clear();
    }
}
