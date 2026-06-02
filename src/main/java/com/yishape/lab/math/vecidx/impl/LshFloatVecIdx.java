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
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 基于幂迭代学习投影的 Multi-Probe LSH 单精度向量索引。
 *
 * <p>内部使用 int 索引 + 数组存储，避免 String 哈希开销。
 * 参数（numTables / numBits / numProbes）均为数据驱动，自动适配规模。</p>
 */
public class LshFloatVecIdx implements IFloatVecIdx, IMutableVecIdx<Float> {

    private static final long serialVersionUID = 2L;

    // ==================== 配置 ====================

    protected final int dimensions;
    private final IDisMetric<Float> metric;
    private final boolean normalize;
    private final boolean distanceIsSquared;
    protected final int numTables;
    protected final int numBits;
    protected final int numProbes;

    // projections[table][bit] = float[dim]
    protected final float[][][] projections;
    protected final float[][] biases;
    protected final float[][] bucketWidths;

    // hashTables[table] -> Map<encodedBuckets, int[]>  (int[] = internal indices, zero-copy in hot path)
    @SuppressWarnings("unchecked")
    protected final Map<Long, int[]>[] hashTables;

    // ==================== 内部索引存储 ====================

    protected float[][] backingVectors;     // [internalIndex] -> float[dim]
    protected String[] indexToId;           // [internalIndex] -> String ID
    private Map<String, Integer> idToIndex; // String ID -> internalIndex
    protected int nextIndex;                // 下一个可用的内部索引

    // 每查询去重：stamp[intIdx] == queryStamp 表示本轮已访问
    private long[] seenStamp;
    private long queryStamp;

    // 有效向量计数（indexToId[i] != null）
    private int size;

    private final StampedLock lock = new StampedLock();

    // ==================== 构造函数 ====================

    public LshFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        this.dimensions = validateAndGetDims(data, ids);
        this.metric = Objects.requireNonNull(metric, "metric");
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "LSH only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;

        this.numTables = optimalTables(data.length);
        this.numBits = optimalBits(data.length);
        // 多探针：numBits*6 = 完整翻转覆盖 + 冗余，上限 32
        this.numProbes = Math.min(numBits * 6, 32);

        this.projections = new float[numTables][numBits][];
        this.biases = new float[numTables][numBits];
        this.bucketWidths = new float[numTables][numBits];
        @SuppressWarnings("unchecked")
        Map<Long, int[]>[] ht = new Map[numTables];
        this.hashTables = ht;
        for (int t = 0; t < numTables; t++) {
            hashTables[t] = new HashMap<>();
        }

        // 初始化内部存储
        this.backingVectors = new float[ids.length][];
        this.indexToId = new String[ids.length];
        this.idToIndex = new HashMap<>(ids.length);
        this.seenStamp = new long[ids.length];
        this.queryStamp = 0;
        this.nextIndex = 0;
        this.size = 0;

        Random rand = new Random(42);
        int parallelism = Runtime.getRuntime().availableProcessors();
        ExecutorService pool = Executors.newWorkStealingPool(parallelism);
        try {
            // 学习投影方向（并行）
            int totalProjections = numTables * numBits;
            float[][] learnedDirs = powerIteration(data, totalProjections, 5, rand, pool);
            int availableDirs = learnedDirs[0].length;
            for (int t = 0; t < numTables; t++) {
                for (int b = 0; b < numBits; b++) {
                    int dirIdx = (t * numBits + b) % availableDirs;
                    projections[t][b] = new float[dimensions];
                    for (int i = 0; i < dimensions; i++) {
                        projections[t][b][i] = learnedDirs[i][dirIdx];
                    }
                }
            }

            // 并行估计桶宽
            estimateBucketWidths(data, pool);

            // 串行：归一化向量并分配内部索引
            ensureCapacity(ids.length);
            for (int i = 0; i < ids.length; i++) {
                float[] vec = normalize ? normalize(data[i]) : data[i].clone();
                addInternal(ids[i], vec);
            }

            // 并行预计算所有桶 key
            long[][] allKeys = computeAllBucketKeys(pool);

            // 串行：构建每张表的 HashMap
            for (int t = 0; t < numTables; t++) {
                Map<Long, List<Integer>> grouped = new HashMap<>();
                for (int i = 0; i < ids.length; i++) {
                    long key = allKeys[t][i];
                    grouped.computeIfAbsent(key, k -> new ArrayList<>()).add(i);
                }
                for (Map.Entry<Long, List<Integer>> e : grouped.entrySet()) {
                    List<Integer> list = e.getValue();
                    int[] arr = new int[list.size()];
                    for (int j = 0; j < list.size(); j++) {
                        arr[j] = list.get(j);
                    }
                    hashTables[t].put(e.getKey(), arr);
                }
            }
        } finally {
            pool.shutdown();
            try {
                pool.awaitTermination(1, TimeUnit.MINUTES);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** 空索引构造（随机投影）。 */
    public LshFloatVecIdx(int dimensions, IDisMetric<Float> metric) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions must be positive");
        }
        this.dimensions = dimensions;
        this.metric = Objects.requireNonNull(metric, "metric");
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "LSH only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;

        this.numTables = 8;
        this.numBits = 4;
        this.numProbes = 8;

        this.projections = new float[numTables][numBits][];
        this.biases = new float[numTables][numBits];
        this.bucketWidths = new float[numTables][numBits];
        @SuppressWarnings("unchecked")
        Map<Long, int[]>[] ht = new Map[numTables];
        this.hashTables = ht;
        for (int t = 0; t < numTables; t++) {
            hashTables[t] = new HashMap<>();
        }

        this.backingVectors = new float[16][];
        this.indexToId = new String[16];
        this.idToIndex = new HashMap<>();
        this.seenStamp = new long[16];
        this.queryStamp = 0;
        this.nextIndex = 0;
        this.size = 0;

        Random rand = new Random(42);
        int totalProjections = numTables * numBits;
        float[][] randomDirs = randomOrthonormal(dimensions, totalProjections, rand);
        int availableDirs = randomDirs[0].length;
        for (int t = 0; t < numTables; t++) {
            for (int b = 0; b < numBits; b++) {
                int dirIdx = (t * numBits + b) % availableDirs;
                projections[t][b] = new float[dimensions];
                for (int i = 0; i < dimensions; i++) {
                    projections[t][b][i] = randomDirs[i][dirIdx];
                }
            }
        }

        for (int t = 0; t < numTables; t++) {
            for (int b = 0; b < numBits; b++) {
                biases[t][b] = 0.0f;
                bucketWidths[t][b] = 1.0f;
            }
        }
    }

    // ==================== 参数选择 ====================

    private static int optimalTables(int n) {
        if (n <= 0) return 2;
        return Math.min(12, Math.max(4, (int) Math.round(Math.log(n) / Math.log(2))));
    }

    private static int optimalBits(int n) {
        if (n <= 0) return 4;
        // 经验上 log2(n)/2 较好平衡桶大小与碰撞；编码使用每位 8 bit、最多 8 维（64-bit long），上限 8
        int k = (int) Math.round(Math.log(Math.max(n, 1)) / Math.log(2) / 2.0);
        return Math.min(8, Math.max(4, k));
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

    // ==================== 内部存储操作 ====================

    private int addInternal(String id, float[] vec) {
        ensureCapacity(nextIndex);
        int idx = nextIndex++;
        backingVectors[idx] = vec;
        indexToId[idx] = id;
        idToIndex.put(id, idx);
        size++;
        return idx;
    }

    private void ensureCapacity(int need) {
        if (need >= backingVectors.length) {
            int newCap = Math.max(backingVectors.length * 2, need + 1);
            backingVectors = Arrays.copyOf(backingVectors, newCap);
            indexToId = Arrays.copyOf(indexToId, newCap);
            seenStamp = Arrays.copyOf(seenStamp, newCap);
        }
    }

    // ==================== 幂迭代 ====================

    private static float[][] powerIteration(float[][] data, int k, int maxIter, Random rand,
            ExecutorService pool) {
        final int n = data.length;
        final int d = data[0].length;
        final int effK = Math.min(k, d);

        float[][] Q = new float[d][effK];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < effK; j++) {
                Q[i][j] = (float) rand.nextGaussian();
            }
        }
        orthonormalize(Q);

        int parallelism = Runtime.getRuntime().availableProcessors();
        for (int iter = 0; iter < maxIter; iter++) {
            int chunkSize = (n + parallelism - 1) / parallelism;
            List<Future<float[][]>> futures = new ArrayList<>(parallelism);
            for (int c = 0; c < parallelism; c++) {
                int start = c * chunkSize;
                int end = Math.min(start + chunkSize, n);
                if (start >= end) break;
                final float[][] Qf = Q;
                futures.add(pool.submit(() -> {
                    float[][] Zloc = new float[d][effK];
                    float[] vQ = new float[k];
                    for (int row = start; row < end; row++) {
                        float[] v = data[row];
                        for (int j = 0; j < effK; j++) {
                            double sum = 0.0;
                            for (int i = 0; i < d; i++) {
                                sum += v[i] * Qf[i][j];
                            }
                            vQ[j] = (float) sum;
                        }
                        for (int i = 0; i < d; i++) {
                            float vi = v[i];
                            for (int j = 0; j < effK; j++) {
                                Zloc[i][j] += vi * vQ[j];
                            }
                        }
                    }
                    return Zloc;
                }));
            }
            // sum-reduce into Q (reuse Q as accumulator)
            for (int i = 0; i < d; i++) {
                Arrays.fill(Q[i], 0.0f);
            }
            try {
                for (Future<float[][]> f : futures) {
                    float[][] Zloc = f.get();
                    for (int i = 0; i < d; i++) {
                        for (int j = 0; j < effK; j++) {
                            Q[i][j] += Zloc[i][j];
                        }
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException("power iteration failed", e);
            }
            orthonormalize(Q);
        }
        return Q;
    }

    private static float[][] randomOrthonormal(int d, int k, Random rand) {
        if (k > d) {
            k = d;
        }
        float[][] Q = new float[d][k];
        for (int j = 0; j < k; j++) {
            for (int i = 0; i < d; i++) {
                Q[i][j] = (float) rand.nextGaussian();
            }
        }
        orthonormalize(Q);
        return Q;
    }

    private static void orthonormalize(float[][] Q) {
        int d = Q.length;
        int k = Q[0].length;
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < j; l++) {
                double dot = 0.0;
                for (int i = 0; i < d; i++) {
                    dot += Q[i][l] * Q[i][j];
                }
                for (int i = 0; i < d; i++) {
                    Q[i][j] -= (float) (dot * Q[i][l]);
                }
            }
            double norm = 0.0;
            for (int i = 0; i < d; i++) {
                norm += Q[i][j] * Q[i][j];
            }
            norm = Math.sqrt(norm);
            if (norm > 1e-10) {
                for (int i = 0; i < d; i++) {
                    Q[i][j] /= (float) norm;
                }
            }
        }
    }

    // ==================== 桶宽估计 ====================

    private void estimateBucketWidths(float[][] data, ExecutorService pool) {
        int n = data.length;
        int total = numTables * numBits;
        int parallelism = Runtime.getRuntime().availableProcessors();
        List<Future<?>> futures = new ArrayList<>(Math.min(total, parallelism));
        // 并行：单次遍历计算所有 (t,b) 对的 sum/sumSq
        for (int c = 0; c < parallelism; c++) {
            int chunkStart = c * total / parallelism;
            int chunkEnd = (c + 1) * total / parallelism;
            if (chunkStart >= chunkEnd) break;
            futures.add(pool.submit(() -> {
                for (int idx = chunkStart; idx < chunkEnd; idx++) {
                    int t = idx / numBits;
                    int b = idx % numBits;
                    double sum = 0.0, sumSq = 0.0;
                    float[] proj = projections[t][b];
                    for (int i = 0; i < n; i++) {
                        double p = Linalg.dot(data[i], proj);
                        sum += p;
                        sumSq += p * p;
                    }
                    double mean = sum / n;
                    double var = sumSq / n - mean * mean;
                    double std = Math.sqrt(Math.max(var, 1e-12));
                    bucketWidths[t][b] = (float) (2.0 * std);
                    if (bucketWidths[t][b] < 1e-6f) {
                        bucketWidths[t][b] = 1e-6f;
                    }
                }
            }));
        }
        try {
            for (Future<?> f : futures) f.get();
        } catch (Exception e) {
            throw new RuntimeException("bucket width estimation failed", e);
        }
        // 串行：生成 biases（Random 非线程安全，宽度计算才是瓶颈）
        Random rand = new Random(42);
        for (int t = 0; t < numTables; t++) {
            for (int b = 0; b < numBits; b++) {
                biases[t][b] = (float) (rand.nextDouble() * bucketWidths[t][b]);
            }
        }
    }

    // ==================== 哈希编码 ====================

    protected static long encodeBuckets(int[] buckets) {
        long key = 0;
        for (int b : buckets) {
            int bb = Math.max(-128, Math.min(127, b)) + 128;
            key = (key << 8) | bb;
        }
        return key;
    }

    private long[][] computeAllBucketKeys(ExecutorService pool) {
        int n = size;
        int nt = numTables;
        long[][] allKeys = new long[nt][n];
        int parallelism = Runtime.getRuntime().availableProcessors();
        List<Future<?>> futures = new ArrayList<>(parallelism);
        int chunkSize = (n + parallelism - 1) / parallelism;
        for (int c = 0; c < parallelism; c++) {
            int start = c * chunkSize;
            int end = Math.min(start + chunkSize, n);
            if (start >= end) break;
            futures.add(pool.submit(() -> {
                int[] bucketBuf = new int[numBits];
                for (int i = start; i < end; i++) {
                    float[] vec = backingVectors[i];
                    for (int t = 0; t < nt; t++) {
                        computeBucketsInto(vec, t, bucketBuf);
                        allKeys[t][i] = encodeBuckets(bucketBuf);
                    }
                }
            }));
        }
        try {
            for (Future<?> f : futures) f.get();
        } catch (Exception e) {
            throw new RuntimeException("parallel bucket key computation failed", e);
        }
        return allKeys;
    }

    protected int[] computeBuckets(float[] vector, int table) {
        int[] buckets = new int[numBits];
        computeBucketsInto(vector, table, buckets);
        return buckets;
    }

    protected void computeBucketsInto(float[] vector, int table, int[] out) {
        for (int b = 0; b < numBits; b++) {
            double proj = Linalg.dot(vector, projections[table][b]) + biases[table][b];
            out[b] = (int) Math.floor(proj / bucketWidths[table][b]);
        }
    }

    // ==================== 多探针生成 ====================

    private List<Long> generateProbes(float[] query, int table) {
        int[] center = new int[numBits];
        float[] boundaryDists = new float[numBits];
        int[] flipDir = new int[numBits];

        for (int b = 0; b < numBits; b++) {
            double proj = Linalg.dot(query, projections[table][b]) + biases[table][b];
            double bw = bucketWidths[table][b];
            double bucketFloat = proj / bw;
            double floor = Math.floor(bucketFloat);
            center[b] = (int) floor;
            double delta = bucketFloat - floor;
            boundaryDists[b] = (float) Math.min(delta, 1.0 - delta);
            flipDir[b] = delta < 0.5 ? -1 : 1;
        }

        Integer[] order = new Integer[numBits];
        for (int i = 0; i < numBits; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(boundaryDists[a], boundaryDists[b]));

        List<Long> probes = new ArrayList<>(numProbes);
        Set<Long> seen = new HashSet<>(numProbes * 2);

        long centerKey = encodeBuckets(center);
        probes.add(centerKey);
        seen.add(centerKey);

        for (int i = 0; i < numBits && probes.size() < numProbes; i++) {
            int d = order[i];
            int[] perturbed = center.clone();
            perturbed[d] += flipDir[d];
            long key = encodeBuckets(perturbed);
            if (seen.add(key)) {
                probes.add(key);
            }
        }

        for (int i = 0; i < numBits && probes.size() < numProbes; i++) {
            for (int j = i + 1; j < numBits && probes.size() < numProbes; j++) {
                int d1 = order[i], d2 = order[j];
                int[] perturbed = center.clone();
                perturbed[d1] += flipDir[d1];
                perturbed[d2] += flipDir[d2];
                long key = encodeBuckets(perturbed);
                if (seen.add(key)) {
                    probes.add(key);
                }
            }
        }

        return probes;
    }

    // ==================== 距离计算 ====================

    private double externalDistance(float squaredDist) {
        if (normalize) {
            return squaredDist * 0.5;
        }
        if (distanceIsSquared) {
            return squaredDist;
        }
        return Math.sqrt(squaredDist);
    }

    private static float[] normalize(float[] v) {
        return IVector.of(v).normalize().toFloatArray();
    }

    // ==================== IFloatVecIdx / IMutableVecIdx API ====================

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public int size() {
        long stamp = lock.readLock();
        try {
            return size;
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
        return true;
    }

    @Override
    public boolean isConcurrent() {
        return true;
    }

    @Override
    public List<SearchHit> search(float[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        int total = size;
        if (k <= 0 || total == 0) {
            return List.of();
        }

        // 避免 clone：仅在 normalize 模式下创建新数组
        float[] q = query;
        if (normalize) {
            q = IVector.of(query).normalize().toFloatArray();
        }

        long stamp = lock.readLock();
        try {
            Set<String> ex = (excludeIds == null || excludeIds.isEmpty()) ? Set.of()
                    : excludeIds instanceof Set<String> s ? s : new HashSet<>(excludeIds);

            int heapCap = Math.min(k, total);
            int[] heapIdxs = new int[heapCap];
            float[] heapDists = new float[heapCap];
            int heapSize = 0;
            float worstDist = Float.MAX_VALUE;

            queryStamp++; // 递增查询标记，用于 O(1) 去重
            long qs = queryStamp;

            for (int t = 0; t < numTables; t++) {
                List<Long> probes = generateProbes(q, t);
                for (Long key : probes) {
                    int[] bucket = hashTables[t].get(key);
                    if (bucket == null) continue;
                    for (int idx : bucket) {
                        if (seenStamp[idx] == qs) continue; // 已访问
                        seenStamp[idx] = qs;
                        String id = indexToId[idx];
                        if (id == null) continue; // 已删除
                        if (!ex.isEmpty() && ex.contains(id)) continue;
                        if (filter != null && !filter.test(id)) continue;
                        float d = Linalg.squaredDistance(q, backingVectors[idx]);
                        if (heapSize < heapCap) {
                            heapIdxs[heapSize] = idx;
                            heapDists[heapSize] = d;
                            heapSize++;
                            if (heapSize == heapCap) {
                                buildMaxHeap(heapIdxs, heapDists, heapCap);
                                worstDist = heapDists[0];
                            }
                        } else if (d < worstDist) {
                            heapIdxs[0] = idx;
                            heapDists[0] = d;
                            siftDown(heapIdxs, heapDists, heapCap, 0);
                            worstDist = heapDists[0];
                        }
                    }
                }
            }

            List<SearchHit> result = new ArrayList<>(heapSize);
            for (int i = 0; i < heapSize; i++) {
                result.add(new SearchHit(indexToId[heapIdxs[i]], externalDistance(heapDists[i])));
            }
            result.sort((a, b) -> Double.compare(a.distance(), b.distance()));
            return result;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    private static void buildMaxHeap(int[] ids, float[] dists, int size) {
        for (int i = size / 2 - 1; i >= 0; i--) {
            siftDown(ids, dists, size, i);
        }
    }

    private static void siftDown(int[] ids, float[] dists, int size, int i) {
        while (true) {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int largest = i;
            if (left < size && dists[left] > dists[largest]) largest = left;
            if (right < size && dists[right] > dists[largest]) largest = right;
            if (largest == i) break;
            int ti = ids[i];
            float td = dists[i];
            ids[i] = ids[largest];
            dists[i] = dists[largest];
            ids[largest] = ti;
            dists[largest] = td;
            i = largest;
        }
    }

    @Override
    public List<SearchHit> search(IVector<Float> query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        Objects.requireNonNull(query, "query");
        if (query.length() != dimensions) {
            throw new IllegalArgumentException("query dimension must be " + dimensions);
        }
        return search(query.toFloatArray(), k, excludeIds, filter);
    }

    @Override
    public IVector<Float> getVector(String id) {
        long stamp = lock.readLock();
        try {
            Integer idx = idToIndex.get(id);
            if (idx == null || indexToId[idx] == null) {
                return null;
            }
            return IVector.of(backingVectors[idx].clone());
        } finally {
            lock.unlockRead(stamp);
        }
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
            if (idToIndex.containsKey(id)) {
                throw new IllegalArgumentException("duplicate ID: " + id);
            }
            float[] vec = normalize ? normalize(vector.toFloatArray()) : vector.toFloatArray();
            int idx = addInternal(id, vec);
            for (int t = 0; t < numTables; t++) {
                int[] buckets = computeBuckets(vec, t);
                long key = encodeBuckets(buckets);
                int[] existing = hashTables[t].get(key);
            int[] added;
            if (existing == null) {
                added = new int[] { idx };
            } else {
                added = Arrays.copyOf(existing, existing.length + 1);
                added[added.length - 1] = idx;
            }
            hashTables[t].put(key, added);
            }
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
            indexToId[idx] = null;
            backingVectors[idx] = null;
            size--;
            return true;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean contains(String id) {
        long stamp = lock.readLock();
        try {
            Integer idx = idToIndex.get(id);
            return idx != null && indexToId[idx] != null;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void clear() {
        long stamp = lock.writeLock();
        try {
            for (int t = 0; t < numTables; t++) {
                hashTables[t].clear();
            }
            Arrays.fill(backingVectors, 0, nextIndex, null);
            Arrays.fill(indexToId, 0, nextIndex, null);
            idToIndex.clear();
            nextIndex = 0;
            size = 0;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    /**
     * 重建所有哈希表。
     * <p>当 projections / biases / bucketWidths 被外部修改后（如监督学习训练后），
     * 调用此方法重新计算所有向量的桶编码并重建索引。</p>
     */
    protected void rebuildIndex() {
        long stamp = lock.writeLock();
        try {
            for (int t = 0; t < numTables; t++) {
                hashTables[t].clear();
            }
            int[] bucketBuf = new int[numBits];
            for (int i = 0; i < nextIndex; i++) {
                if (indexToId[i] == null) continue;
                float[] vec = backingVectors[i];
                for (int t = 0; t < numTables; t++) {
                    computeBucketsInto(vec, t, bucketBuf);
                    long key = encodeBuckets(bucketBuf);
                    int[] existing = hashTables[t].get(key);
                    int[] added;
                    if (existing == null) {
                        added = new int[] { i };
                    } else {
                        added = Arrays.copyOf(existing, existing.length + 1);
                        added[added.length - 1] = i;
                    }
                    hashTables[t].put(key, added);
                }
            }
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public void close() {
        clear();
    }
}
