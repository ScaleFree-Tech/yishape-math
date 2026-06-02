package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.SearchHit;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 暴力扫描单精度向量索引：对所有候选逐一点对点计算距离，排序后取 Top-k。
 *
 * <p>实现特点：</p>
 * <ul>
 *   <li>距离度量通过 {@link IDisMetric} 注入，支持任意自定义距离；</li>
 *   <li>对常见度量（euclidean、squared_euclidean、manhattan、cosine、inner_product）
 *       内联展开计算，避免 {@link IVector} 包装开销；</li>
 *   <li>标量过滤采用<strong>预过滤</strong>（在距离计算前排除），精确无遗漏；</li>
 *   <li>Top-k 选择根据候选规模自动切换排序 / 堆策略；</li>
 *   <li>可变实现基于 {@link StampedLock} 支持多读单写并发。</li>
 * </ul>
 *
 * <p>复杂度：单次查询 {@code O(n d + n log k)}，空间 {@code O(n d)}。
 * 适用于 {@code n} 较小（如 {@code < 1e5}）或精确性不可妥协的场景。</p>
 */
public class BruteForceFloatVecIdx implements IFloatVecIdx, IMutableVecIdx<Float> {

    private static final long serialVersionUID = 1L;

    private final int dimensions;
    private final IDisMetric<Float> metric;

    // 紧凑数组存储（替代 LinkedHashMap，避免迭代时 Map.Entry 分配）
    private float[][] vectors;        // vectors[i] = 第 i 个向量的 float[]
    private String[] ids;             // ids[i] = 第 i 个向量的 id
    private final Map<String, Integer> idToIndex = new HashMap<>();
    private int count;

    private final StampedLock lock = new StampedLock();

    /**
     * 从已有数据构建静态索引。
     */
    public BruteForceFloatVecIdx(float[][] data, String[] ids, IDisMetric<Float> metric) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        this.metric = Objects.requireNonNull(metric, "metric");
        if (ids.length == 0) {
            this.dimensions = data.length > 0 && data[0] != null ? data[0].length : 0;
            initArrays(0);
            return;
        }
        this.dimensions = data[0].length;
        initArrays(ids.length);
        for (int i = 0; i < ids.length; i++) {
            append(ids[i], data[i].clone());
        }
    }

    /**
     * 构造空的可变索引。
     */
    public BruteForceFloatVecIdx(int dimensions, IDisMetric<Float> metric) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions 必须为正");
        }
        this.dimensions = dimensions;
        this.metric = Objects.requireNonNull(metric, "metric");
        initArrays(0);
    }

    private void initArrays(int cap) {
        int c = Math.max(cap, 16);
        this.vectors = new float[c][];
        this.ids = new String[c];
        this.count = 0;
    }

    /** 追加一条向量（不检查重复，调用方负责） */
    private void append(String id, float[] vec) {
        if (count >= vectors.length) {
            int newCap = vectors.length + (vectors.length >> 1) + 1;
            vectors = Arrays.copyOf(vectors, newCap);
            ids = Arrays.copyOf(ids, newCap);
        }
        vectors[count] = vec;
        ids[count] = id;
        idToIndex.put(id, count);
        count++;
    }

    /** 交换删除（O(1)），调用前需持写锁 */
    private void swapRemove(int idx) {
        int last = count - 1;
        if (idx != last) {
            vectors[idx] = vectors[last];
            ids[idx] = ids[last];
            idToIndex.put(ids[idx], idx);
        }
        vectors[last] = null;
        ids[last] = null;
        count--;
    }

    // ==================== 距离计算 ====================

    /** 对第 idx 个存储向量计算到 query 的距离（避免 float[] 到 IVector 的包装） */
    private double computeToIdx(float[] query, int idx) {
        float[] b = vectors[idx];
        MetricType mt = metric.type();
        if (mt != null) {
            return switch (mt) {
                case EUCLIDEAN -> Math.sqrt(Linalg.squaredDistance(query, b));
                case SQUARED_EUCLIDEAN -> (double) Linalg.squaredDistance(query, b);
                case MANHATTAN -> manhattanDistance(query, b);
                case COSINE -> cosineDistance(query, b);
                case INNER_PRODUCT -> (double) Linalg.dot(query, b);
            };
        }
        return metric.compute(IVector.of(query), IVector.of(b));
    }

    private static double manhattanDistance(float[] a, float[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.abs((double) a[i] - (double) b[i]);
        }
        return sum;
    }

    private static double cosineDistance(float[] a, float[] b) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * (double) b[i];
            na += (double) a[i] * (double) a[i];
            nb += (double) b[i] * (double) b[i];
        }
        double cos = dot / (Math.sqrt(na) * Math.sqrt(nb));
        return 1.0 - Math.max(-1.0, Math.min(1.0, cos));
    }

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public int size() {
        long stamp = lock.readLock();
        try {
            return count;
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
    public IVector<Float> getVector(String id) {
        long stamp = lock.readLock();
        try {
            Integer idx = idToIndex.get(id);
            return idx != null ? IVector.of(vectors[idx].clone()) : null;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public List<SearchHit> search(float[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        if (k <= 0) {
            return List.of();
        }
        Set<String> ex = excludeIds instanceof Set<String> s ? s
                : new HashSet<>(excludeIds != null ? excludeIds : List.of());
        boolean sim = metric.isSimilarity();
        long stamp = lock.readLock();
        try {
            PriorityQueue<SearchHit> maxHeap = new PriorityQueue<>(k + 1,
                    sim
                            ? (a, b) -> Double.compare(a.distance(), b.distance())
                            : (a, b) -> Double.compare(b.distance(), a.distance()));
            for (int i = 0; i < count; i++) {
                String id = ids[i];
                if (ex.contains(id) || (filter != null && !filter.test(id))) {
                    continue;
                }
                double d = computeToIdx(query, i);
                if (maxHeap.size() < k) {
                    maxHeap.offer(new SearchHit(id, d));
                } else {
                    // max-heap peek 是 k 个中最差的；对 similarity 而言最差是最小，对 distance 是最远（最大）
                    double worst = maxHeap.peek().distance();
                    if ((sim && d > worst) || (!sim && d < worst)) {
                        maxHeap.poll();
                        maxHeap.offer(new SearchHit(id, d));
                    }
                }
            }
            List<SearchHit> result = new ArrayList<>(maxHeap);
            result.sort(sim
                    ? (a, b) -> Double.compare(b.distance(), a.distance())
                    : Comparator.comparingDouble(SearchHit::distance));
            return result;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public List<SearchHit> rangeSearch(float[] query, double radius, Collection<String> excludeIds, Predicate<String> filter) {
        if (radius < 0) {
            return List.of();
        }
        Set<String> ex = excludeIds instanceof Set<String> s ? s
                : new HashSet<>(excludeIds != null ? excludeIds : List.of());
        boolean sim = metric.isSimilarity();
        long stamp = lock.readLock();
        try {
            List<SearchHit> result = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                String id = ids[i];
                if (ex.contains(id) || (filter != null && !filter.test(id))) {
                    continue;
                }
                double d = computeToIdx(query, i);
                if (sim ? d >= radius : d <= radius) {
                    result.add(new SearchHit(id, d));
                }
            }
            result.sort(sim
                    ? (a, b) -> Double.compare(b.distance(), a.distance())
                    : Comparator.comparingDouble(SearchHit::distance));
            return result;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void add(String id, IVector<Float> vector) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(vector, "vector");
        if (vector.length() != dimensions) {
            throw new IllegalArgumentException(
                    "向量维度须为 " + dimensions + "，实为 " + vector.length());
        }
        long stamp = lock.writeLock();
        try {
            if (idToIndex.containsKey(id)) {
                throw new IllegalArgumentException("重复 ID: " + id);
            }
            append(id, vector.toFloatArray());
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean remove(String id) {
        long stamp = lock.writeLock();
        try {
            Integer idx = idToIndex.remove(id);
            if (idx == null) {
                return false;
            }
            swapRemove(idx);
            return true;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean contains(String id) {
        long stamp = lock.readLock();
        try {
            return idToIndex.containsKey(id);
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void clear() {
        long stamp = lock.writeLock();
        try {
            Arrays.fill(vectors, 0, count, null);
            Arrays.fill(ids, 0, count, null);
            idToIndex.clear();
            count = 0;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public void close() {
        clear();
    }
}
