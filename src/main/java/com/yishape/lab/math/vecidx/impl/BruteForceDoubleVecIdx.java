package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.SearchHit;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IDoubleVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 暴力扫描双精度向量索引：对所有候选逐一点对点计算距离，排序后取 Top-k。
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
public class BruteForceDoubleVecIdx implements IDoubleVecIdx, IMutableVecIdx<Double> {

    private static final long serialVersionUID = 1L;

    private final int dimensions;
    private final IDisMetric<Double> metric;
    private final Map<String, double[]> storage = new LinkedHashMap<>();
    private final StampedLock lock = new StampedLock();

    /**
     * 从已有数据构建静态索引。
     *
     * @param data   向量数组，{@code data[i]} 为 {@code ids[i]} 对应的向量
     * @param ids    标识数组，长度须等于 {@code data.length}
     * @param metric 距离度量
     */
    public BruteForceDoubleVecIdx(double[][] data, String[] ids, IDisMetric<Double> metric) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        this.metric = Objects.requireNonNull(metric, "metric");
        if (ids.length == 0) {
            this.dimensions = data.length > 0 && data[0] != null ? data[0].length : 0;
            return;
        }
        this.dimensions = data[0].length;
        for (int i = 0; i < ids.length; i++) {
            storage.put(ids[i], data[i].clone());
        }
    }

    /**
     * 构造空的可变索引。
     *
     * @param dimensions 向量维度
     * @param metric     距离度量
     */
    public BruteForceDoubleVecIdx(int dimensions, IDisMetric<Double> metric) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions 必须为正");
        }
        this.dimensions = dimensions;
        this.metric = Objects.requireNonNull(metric, "metric");
    }

    // ==================== 距离计算（委托 linalg SIMD/HPC 加速）====================

    private double compute(double[] a, double[] b) {
        MetricType mt = metric.type();
        if (mt != null) {
            return switch (mt) {
                case EUCLIDEAN -> euclideanDistance(a, b);
                case SQUARED_EUCLIDEAN -> squaredDistance(a, b);
                case MANHATTAN -> manhattanDistance(a, b);
                case COSINE -> cosineDistance(a, b);
                case INNER_PRODUCT -> innerProduct(a, b);
            };
        }
        return metric.compute(IVector.of(a), IVector.of(b));
    }

    private static double euclideanDistance(double[] a, double[] b) {
        return Math.sqrt(squaredDistance(a, b));
    }

    private static double squaredDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    private static double manhattanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.abs(a[i] - b[i]);
        }
        return sum;
    }

    private static double cosineDistance(double[] a, double[] b) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double cos = dot / (Math.sqrt(na) * Math.sqrt(nb));
        return 1.0 - Math.max(-1.0, Math.min(1.0, cos));
    }

    private static double innerProduct(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // ==================== Top-k 选择 ====================

    private static List<SearchHit> selectTopK(List<SearchHit> candidates, int k, boolean isSimilarity) {
        if (k <= 0 || candidates.isEmpty()) {
            return List.of();
        }
        if (candidates.size() <= k) {
            candidates.sort(comparator(isSimilarity));
            return List.copyOf(candidates);
        }
        if (candidates.size() <= k * 4) {
            candidates.sort(comparator(isSimilarity));
            return List.copyOf(candidates.subList(0, k));
        }
        Comparator<SearchHit> cmp = comparator(isSimilarity);
        PriorityQueue<SearchHit> pq = new PriorityQueue<>(k + 1, cmp.reversed());
        for (SearchHit h : candidates) {
            pq.offer(h);
            if (pq.size() > k) {
                pq.poll();
            }
        }
        List<SearchHit> result = new ArrayList<>(pq);
        result.sort(cmp);
        return result;
    }

    private static Comparator<SearchHit> comparator(boolean isSimilarity) {
        return isSimilarity
                ? (a, b) -> Double.compare(b.distance(), a.distance())
                : Comparator.comparingDouble(SearchHit::distance);
    }

    // ==================== IDoubleVecIdx / IMutableVecIdx API ====================

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public int size() {
        long stamp = lock.readLock();
        try {
            return storage.size();
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public IDisMetric<Double> metric() {
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
    public IVector<Double> getVector(String id) {
        long stamp = lock.readLock();
        try {
            double[] vec = storage.get(id);
            return vec != null ? IVector.of(vec.clone()) : null;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public List<SearchHit> search(double[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        if (k <= 0) {
            return List.of();
        }
        Set<String> ex = excludeIds instanceof Set<String> s ? s
                : new HashSet<>(excludeIds != null ? excludeIds : List.of());
        long stamp = lock.readLock();
        try {
            List<SearchHit> buf = new ArrayList<>(storage.size());
            for (Map.Entry<String, double[]> e : storage.entrySet()) {
                String id = e.getKey();
                if (ex.contains(id) || (filter != null && !filter.test(id))) {
                    continue;
                }
                double d = compute(query, e.getValue());
                buf.add(new SearchHit(id, d));
            }
            return selectTopK(buf, k, metric.isSimilarity());
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public List<SearchHit> rangeSearch(double[] query, double radius, Collection<String> excludeIds, Predicate<String> filter) {
        if (radius < 0) {
            return List.of();
        }
        Set<String> ex = excludeIds instanceof Set<String> s ? s
                : new HashSet<>(excludeIds != null ? excludeIds : List.of());
        long stamp = lock.readLock();
        try {
            List<SearchHit> all = new ArrayList<>();
            for (Map.Entry<String, double[]> e : storage.entrySet()) {
                String id = e.getKey();
                if (ex.contains(id) || (filter != null && !filter.test(id))) {
                    continue;
                }
                double d = compute(query, e.getValue());
                all.add(new SearchHit(id, d));
            }
            boolean sim = metric.isSimilarity();
            return all.stream()
                    .filter(h -> sim ? h.distance() >= radius : h.distance() <= radius)
                    .sorted(sim
                            ? (a, b) -> Double.compare(b.distance(), a.distance())
                            : Comparator.comparingDouble(SearchHit::distance))
            .toList();
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void add(String id, IVector<Double> vector) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(vector, "vector");
        if (vector.length() != dimensions) {
            throw new IllegalArgumentException(
                    "向量维度须为 " + dimensions + "，实为 " + vector.length());
        }
        long stamp = lock.writeLock();
        try {
            storage.put(id, vector.toDoubleArray());
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean remove(String id) {
        long stamp = lock.writeLock();
        try {
            return storage.remove(id) != null;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean contains(String id) {
        long stamp = lock.readLock();
        try {
            return storage.containsKey(id);
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void clear() {
        long stamp = lock.writeLock();
        try {
            storage.clear();
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public void close() {
        clear();
    }
}
