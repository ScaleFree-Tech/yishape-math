package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.util.Tuple3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * PQ + HNSW 组合向量索引：Rust HNSW 图导航 + PQ 编码 + 精确重排序。
 *
 * <p>HNSW 图构建和搜索委托给 {@link RustHnswFloatVecIdx}（自动使用 Rust 后端），
 * PQ 编码用于内存压缩和未来的 ADC 快速距离近似。</p>
 */
public class PqHnswFloatVecIdx implements IFloatVecIdx, IMutableVecIdx<Float> {

    private static final long serialVersionUID = 1L;

    private static final int DEFAULT_CODEBOOK_SIZE = 256;
    private static final int OPQ_MAX_ITER = 5;
    private static final int KMEANS_MAX_ITER = 10;
    private static final int KMEANS_SAMPLE_LIMIT = 65536;

    // ==================== 配置 ====================

    private final int dimensions;
    private final IDisMetric<Float> metric;
    private final VecSearchOption options;
    private final boolean normalize;
    private final boolean distanceIsSquared;

    // PQ 配置
    private final int numSubspaces;
    private final int[] subspaceDims;
    private final int[] subspaceOffsets;
    private final int codebookSize;
    private final boolean useOpq;

    // PQ 训练后状态
    private float[][][] centroids;
    private float[][] rotation;
    private float[][] rotationT;

    // Rust HNSW 委托
    private RustHnswFloatVecIdx hnswIndex;

    // 紧凑数组形式的 PQ 码 + 对应 id（用于 ADC 全扫描，避免 HashMap 迭代分配）
    private byte[][] codesArr;
    private String[] codesIds;
    private int codesCount;

    // id -> 在 codesArr/codesIds 中的位置，用于 O(1) 删除
    private final Map<String, Integer> idToCodeIndex = new HashMap<>();

    // 未训练前的暂存向量
    private final Map<String, float[]> pendingVectors = new HashMap<>();

    // 原始（已归一化）向量本地缓存：用于 ADC 排序后做精确重排
    private final Map<String, float[]> rawVectors = new HashMap<>();

    private final StampedLock lock = new StampedLock();

    // ==================== 构造函数 ====================

    public PqHnswFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options, boolean useOpq) {
        this.dimensions = validateAndGetDims(data, ids);
        this.metric = Objects.requireNonNull(metric, "metric");
        this.options = options != null ? options : VecSearchOption.DEFAULT;
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "PQ+HNSW only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;
        this.useOpq = useOpq;

        int suggestedSubspaceDim = Math.max(1, Math.min(dimensions, this.options.hnswM()));
        if (suggestedSubspaceDim > dimensions / 2 && dimensions >= 8) {
            suggestedSubspaceDim = 8;
        }
        this.subspaceDims = computeSubspaceDims(dimensions, suggestedSubspaceDim);
        this.numSubspaces = subspaceDims.length;
        this.subspaceOffsets = computeOffsets(subspaceDims);
        this.codebookSize = Math.min(DEFAULT_CODEBOOK_SIZE, ids.length);

        // 拷贝并归一化原始向量
        float[][] normalizedData = new float[data.length][];
        for (int i = 0; i < data.length; i++) {
            normalizedData[i] = normalize ? normalize(data[i]) : data[i].clone();
        }

        // 训练 PQ 码本
        train(normalizedData);

        // 编码所有向量 + 本地保存原始（归一化）向量供精确重排使用
        this.codesArr = new byte[ids.length][];
        this.codesIds = new String[ids.length];
        for (int i = 0; i < ids.length; i++) {
            float[] vec = useOpq && rotationT != null ? rotate(normalizedData[i]) : normalizedData[i];
            byte[] code = encode(vec);
            codesArr[i] = code;
            codesIds[i] = ids[i];
            idToCodeIndex.put(ids[i], i);
            rawVectors.put(ids[i], normalizedData[i]);
        }
        this.codesCount = ids.length;

        // 委托 Rust HNSW 构建图（使用原始归一化向量 + 精确距离）
        this.hnswIndex = new RustHnswFloatVecIdx(normalizedData, ids, metric, this.options);
    }

    public PqHnswFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        this(data, ids, metric, options, false);
    }

    public PqHnswFloatVecIdx(int dimensions, IDisMetric<Float> metric,
            VecSearchOption options, boolean useOpq) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions 必须为正");
        }
        this.dimensions = dimensions;
        this.metric = Objects.requireNonNull(metric, "metric");
        this.options = options != null ? options : VecSearchOption.DEFAULT;
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "PQ+HNSW only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;
        this.useOpq = useOpq;

        int suggestedSubspaceDim = Math.max(1, Math.min(dimensions, 8));
        this.subspaceDims = computeSubspaceDims(dimensions, suggestedSubspaceDim);
        this.numSubspaces = subspaceDims.length;
        this.subspaceOffsets = computeOffsets(subspaceDims);
        this.codebookSize = DEFAULT_CODEBOOK_SIZE;
    }

    private float[] rotate(float[] vec) {
        IMatrix<Float> RT = Linalg.matrix(rotationT);
        return RT.mmul(IVector.of(vec)).toFloatArray();
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

    private static int[] computeSubspaceDims(int dim, int targetSubspaceDim) {
        if (dim <= 0) {
            return new int[0];
        }
        int numSubspaces = Math.max(1, (dim + targetSubspaceDim - 1) / targetSubspaceDim);
        int[] dims = new int[numSubspaces];
        int base = dim / numSubspaces;
        int rem = dim % numSubspaces;
        for (int i = 0; i < numSubspaces; i++) {
            dims[i] = base + (i < rem ? 1 : 0);
        }
        return dims;
    }

    private static int[] computeOffsets(int[] subspaceDims) {
        int[] offsets = new int[subspaceDims.length];
        for (int i = 1; i < subspaceDims.length; i++) {
            offsets[i] = offsets[i - 1] + subspaceDims[i - 1];
        }
        return offsets;
    }

    // ==================== PQ 训练 ====================

    private void train(float[][] data) {
        int n = data.length;
        if (n == 0) {
            return;
        }
        if (useOpq) {
            trainOpq(data);
        } else {
            trainPq(data);
        }
    }

    private void trainPq(float[][] data) {
        this.centroids = new float[numSubspaces][codebookSize][];
        for (int s = 0; s < numSubspaces; s++) {
            int dim = subspaceDims[s];
            int off = subspaceOffsets[s];
            float[][] subspaceData = extractSubspace(data, off, dim);
            centroids[s] = trainKMeans(subspaceData, codebookSize, KMEANS_MAX_ITER);
        }
    }

    private void trainOpq(float[][] data) {
        int n = data.length;
        int d = dimensions;

        rotation = new float[d][d];
        for (int i = 0; i < d; i++) {
            rotation[i][i] = 1.0f;
        }

        IMatrix<Float> dataMat = Linalg.matrix(data);

        for (int iter = 0; iter < OPQ_MAX_ITER; iter++) {
            float[][] rotatedData = dataMat.mmul(Linalg.matrix(rotation)).toFloatArray();
            trainPq(rotatedData);

            float[][] reconstructed = new float[n][d];
            for (int i = 0; i < n; i++) {
                decodeToVector(encode(rotatedData[i]), reconstructed[i]);
            }

            IMatrix<Float> reconMat = Linalg.matrix(reconstructed);
            IMatrix<Float> Mmat = dataMat.transpose().mmul(reconMat);
            Tuple3<IMatrix<Float>, IVector<Float>, IMatrix<Float>> svd = Mmat.svd();
            rotation = svd.getFirst().mmul(svd.getThird()).toFloatArray();
        }

        float[][] finalRotated = dataMat.mmul(Linalg.matrix(rotation)).toFloatArray();
        trainPq(finalRotated);
        rotationT = Linalg.matrix(rotation).transpose().toFloatArray();
    }

    private float[][] trainKMeans(float[][] data, int k, int maxIter) {
        int n = data.length;
        int dim = data[0].length;
        Random rand = new Random(42);

        float[][] sample;
        int sampleN;
        if (n <= KMEANS_SAMPLE_LIMIT) {
            sample = data;
            sampleN = n;
        } else {
            sampleN = KMEANS_SAMPLE_LIMIT;
            sample = new float[sampleN][dim];
            for (int i = 0; i < sampleN; i++) {
                System.arraycopy(data[rand.nextInt(n)], 0, sample[i], 0, dim);
            }
        }

        // K-means++ initialization (float32, no Double boxing)
        float[][] centroids = kmeansPlusPlusInit(sample, k, dim, rand);

        // Lloyd iterations
        int[] assignments = new int[sampleN];
        for (int iter = 0; iter < maxIter; iter++) {
            boolean changed = false;
            for (int i = 0; i < sampleN; i++) {
                int best = 0;
                float bestDist = squaredDistance(sample[i], centroids[0]);
                for (int c = 1; c < k; c++) {
                    float dist = squaredDistance(sample[i], centroids[c]);
                    if (dist < bestDist) {
                        bestDist = dist;
                        best = c;
                    }
                }
                if (assignments[i] != best) {
                    assignments[i] = best;
                    changed = true;
                }
            }
            if (!changed) {
                break;
            }

            float[][] newCentroids = new float[k][dim];
            int[] counts = new int[k];
            for (int i = 0; i < sampleN; i++) {
                int c = assignments[i];
                for (int j = 0; j < dim; j++) {
                    newCentroids[c][j] += sample[i][j];
                }
                counts[c]++;
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    for (int j = 0; j < dim; j++) {
                        newCentroids[c][j] /= counts[c];
                    }
                } else {
                    int idx = rand.nextInt(sampleN);
                    System.arraycopy(sample[idx], 0, newCentroids[c], 0, dim);
                }
            }
            centroids = newCentroids;
        }
        return centroids;
    }

    private static float[][] kmeansPlusPlusInit(float[][] sample, int k, int dim, Random rand) {
        int n = sample.length;
        float[][] centers = new float[k][dim];
        boolean[] chosen = new boolean[n];

        int first = rand.nextInt(n);
        System.arraycopy(sample[first], 0, centers[0], 0, dim);
        chosen[first] = true;

        float[] minDists = new float[n];
        Arrays.fill(minDists, Float.MAX_VALUE);
        for (int i = 0; i < n; i++) {
            if (!chosen[i]) {
                minDists[i] = squaredDistance(sample[i], centers[0]);
            }
        }

        for (int c = 1; c < k; c++) {
            float totalDist = 0;
            for (int i = 0; i < n; i++) {
                if (chosen[i]) continue;
                float d = squaredDistance(sample[i], centers[c - 1]);
                if (d < minDists[i]) minDists[i] = d;
                totalDist += minDists[i];
            }

            if (totalDist < 1e-8f) {
                for (int i = 0; i < n; i++) {
                    if (!chosen[i]) {
                        chosen[i] = true;
                        System.arraycopy(sample[i], 0, centers[c], 0, dim);
                        break;
                    }
                }
            } else {
                float target = rand.nextFloat() * totalDist;
                float cumulative = 0;
                boolean picked = false;
                for (int i = 0; i < n; i++) {
                    if (chosen[i]) continue;
                    cumulative += minDists[i];
                    if (cumulative >= target) {
                        chosen[i] = true;
                        System.arraycopy(sample[i], 0, centers[c], 0, dim);
                        picked = true;
                        break;
                    }
                }
                if (!picked) {
                    for (int i = n - 1; i >= 0; i--) {
                        if (!chosen[i]) {
                            chosen[i] = true;
                            System.arraycopy(sample[i], 0, centers[c], 0, dim);
                            break;
                        }
                    }
                }
            }
        }
        return centers;
    }

    // ==================== PQ 编码 / 解码 ====================

    private byte[] encode(float[] vector) {
        byte[] code = new byte[numSubspaces];
        for (int s = 0; s < numSubspaces; s++) {
            int off = subspaceOffsets[s];
            int dim = subspaceDims[s];
            int best = 0;
            float bestDist = subspaceSquaredDistance(vector, off, dim, centroids[s][0]);
            for (int c = 1; c < codebookSize; c++) {
                float d = subspaceSquaredDistance(vector, off, dim, centroids[s][c]);
                if (d < bestDist) {
                    bestDist = d;
                    best = c;
                }
            }
            code[s] = (byte) best;
        }
        return code;
    }

    private void decodeToVector(byte[] code, float[] out) {
        for (int s = 0; s < numSubspaces; s++) {
            int off = subspaceOffsets[s];
            int dim = subspaceDims[s];
            int c = code[s] & 0xFF;
            System.arraycopy(centroids[s][c], 0, out, off, dim);
        }
    }

    private static float[][] extractSubspace(float[][] data, int offset, int dim) {
        int n = data.length;
        float[][] result = new float[n][dim];
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], offset, result[i], 0, dim);
        }
        return result;
    }

    // ==================== ADC 距离计算 ====================

    /**
     * 为查询向量构建 ADC 距离表。
     */
    public float[][] buildDistanceTables(float[] query) {
        if (centroids == null) {
            throw new IllegalStateException("PQ 码本尚未训练");
        }
        float[][] tables = new float[numSubspaces][codebookSize];
        for (int s = 0; s < numSubspaces; s++) {
            int off = subspaceOffsets[s];
            int dim = subspaceDims[s];
            for (int c = 0; c < codebookSize; c++) {
                tables[s][c] = subspaceSquaredDistance(query, off, dim, centroids[s][c]);
            }
        }
        return tables;
    }

    /**
     * 通过 ADC 查表计算 query 与 code 的近似 squared Euclidean distance。
     */
    public float adcDistance(byte[] code, float[][] distanceTables) {
        float dist = 0.0f;
        for (int s = 0; s < numSubspaces; s++) {
            dist += distanceTables[s][code[s] & 0xFF];
        }
        return dist;
    }

    private float subspaceSquaredDistance(float[] vec, int offset, int dim, float[] centroid) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            double d = (double) vec[offset + i] - (double) centroid[i];
            sum += d * d;
        }
        return (float) sum;
    }

    // ==================== 辅助方法 ====================

    private double externalDistance(float squaredDist) {
        if (normalize) {
            return squaredDist * 0.5;
        }
        if (distanceIsSquared) {
            return squaredDist;
        }
        return Math.sqrt(squaredDist);
    }

    private float[] normalize(float[] v) {
        return IVector.of(v).normalize().toFloatArray();
    }

    private static float squaredDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    /** 未训练时对暂存向量做暴力扫描。 */
    private List<SearchHit> bruteForceSearch(float[] query, int k,
            Collection<String> excludeIds, Predicate<String> filter) {
        Set<String> ex = excludeIds instanceof Set<String> s ? s
                : new HashSet<>(excludeIds != null ? excludeIds : List.of());
        PriorityQueue<SearchHit> maxHeap = new PriorityQueue<>(
                (a, b) -> Double.compare(b.distance(), a.distance()));
        for (Map.Entry<String, float[]> e : pendingVectors.entrySet()) {
            String id = e.getKey();
            if (ex.contains(id) || (filter != null && !filter.test(id))) {
                continue;
            }
            float d = squaredDistance(query, e.getValue());
            maxHeap.offer(new SearchHit(id, externalDistance(d)));
            if (maxHeap.size() > k) {
                maxHeap.poll();
            }
        }
        List<SearchHit> result = new ArrayList<>(maxHeap);
        result.sort((a, b) -> Double.compare(a.distance(), b.distance()));
        return result;
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
            // hnswIndex 在我们这边镜像着 codesCount，避免 FFI 调用
            return pendingVectors.size() + codesCount;
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
        if (k <= 0) {
            return List.of();
        }

        float[] q = normalize ? normalize(query) : query;

        long stamp = lock.readLock();
        try {
            if (hnswIndex == null) {
                return bruteForceSearch(q, k, excludeIds, filter);
            }
            if (centroids == null || codesCount == 0) {
                return hnswIndex.search(q, k, excludeIds, filter);
            }

            Set<String> ex = (excludeIds == null || excludeIds.isEmpty()) ? Set.of()
                    : (excludeIds instanceof Set<String> s ? s : new HashSet<>(excludeIds));

            // 候选超采样倍率：HNSW + PQ ADC 两路各取 fetch 个
            int efSearch = Math.max(k, options.hnswEfSearch());
            int fetch = Math.max(k * 2, efSearch);

            // 1) HNSW 路径：取 fetch 个候选（含精确距离）
            List<SearchHit> hnswHits = hnswIndex.search(q, fetch, excludeIds, filter);

            // 2) PQ ADC 路径：基于查询向量构建距离表，全扫描所有 code
            float[] queryRot = useOpq && rotationT != null ? rotate(q) : q;
            float[][] distanceTables = buildDistanceTables(queryRot);

            // 用最大堆维护前 fetch 个 ADC 最小者（避免一次性排序所有 N 项）
            int adcLimit = Math.min(fetch, codesCount);
            float[] adcDist = new float[adcLimit];
            int[] adcIdx = new int[adcLimit];
            int adcSize = 0;
            float adcWorst = Float.MAX_VALUE;

            for (int i = 0; i < codesCount; i++) {
                String id = codesIds[i];
                if (!ex.isEmpty() && ex.contains(id)) {
                    continue;
                }
                if (filter != null && !filter.test(id)) {
                    continue;
                }
                byte[] code = codesArr[i];
                float d = 0;
                for (int s = 0; s < numSubspaces; s++) {
                    d += distanceTables[s][code[s] & 0xFF];
                }
                if (adcSize < adcLimit) {
                    adcIdx[adcSize] = i;
                    adcDist[adcSize] = d;
                    adcSize++;
                    if (adcSize == adcLimit) {
                        buildMaxHeap(adcIdx, adcDist, adcLimit);
                        adcWorst = adcDist[0];
                    }
                } else if (d < adcWorst) {
                    adcIdx[0] = i;
                    adcDist[0] = d;
                    siftDownHeap(adcIdx, adcDist, adcLimit, 0);
                    adcWorst = adcDist[0];
                }
            }

            // 3) 合并候选：HNSW 路径用其精确距离；PQ-only 路径用本地向量计算精确距离
            // 用 HashSet 去重，避免重复计算
            HashSet<String> seen = new HashSet<>(hnswHits.size() + adcSize);
            PriorityQueue<SearchHit> topK = new PriorityQueue<>(k + 1,
                    (a, b) -> Double.compare(b.distance(), a.distance()));

            for (SearchHit h : hnswHits) {
                // 过滤已从本地存储删除但 HNSW 图尚未同步的 ghost 条目
                if (!rawVectors.containsKey(h.id())) {
                    continue;
                }
                if (seen.add(h.id())) {
                    topK.offer(h);
                    if (topK.size() > k) {
                        topK.poll();
                    }
                }
            }

            for (int i = 0; i < adcSize; i++) {
                String id = codesIds[adcIdx[i]];
                if (!seen.add(id)) {
                    continue;
                }
                float[] vec = rawVectors.get(id);
                if (vec == null) {
                    // 极少见：本地缓存缺失，回退用 ADC 距离的近似值
                    topK.offer(new SearchHit(id, externalDistance(adcDist[i])));
                } else {
                    float exact = squaredDistance(q, vec);
                    topK.offer(new SearchHit(id, externalDistance(exact)));
                }
                if (topK.size() > k) {
                    topK.poll();
                }
            }

            List<SearchHit> result = new ArrayList<>(topK);
            result.sort((a, b) -> Double.compare(a.distance(), b.distance()));
            return result;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    /** 最大堆构建（按 dist[]）。 */
    private static void buildMaxHeap(int[] idx, float[] dist, int size) {
        for (int i = size / 2 - 1; i >= 0; i--) {
            siftDownHeap(idx, dist, size, i);
        }
    }

    private static void siftDownHeap(int[] idx, float[] dist, int size, int i) {
        while (true) {
            int l = 2 * i + 1;
            int r = 2 * i + 2;
            int largest = i;
            if (l < size && dist[l] > dist[largest]) {
                largest = l;
            }
            if (r < size && dist[r] > dist[largest]) {
                largest = r;
            }
            if (largest == i) {
                break;
            }
            int ti = idx[i];
            float td = dist[i];
            idx[i] = idx[largest];
            dist[i] = dist[largest];
            idx[largest] = ti;
            dist[largest] = td;
            i = largest;
        }
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
        long stamp = lock.readLock();
        try {
            float[] raw = rawVectors.get(id);
            if (raw != null) {
                return IVector.of(raw.clone());
            }
            float[] pending = pendingVectors.get(id);
            if (pending != null) {
                return IVector.of(pending.clone());
            }
            return null;
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
            if (pendingVectors.containsKey(id) || idToCodeIndex.containsKey(id)) {
                throw new IllegalArgumentException("duplicate ID: " + id);
            }
            float[] vec = normalize ? normalize(vector.toFloatArray()) : vector.toFloatArray();
            if (hnswIndex == null) {
                pendingVectors.put(id, vec);
                return;
            }
            float[] encodedVec = useOpq && rotationT != null ? rotate(vec) : vec;
            byte[] code = encode(encodedVec);
            appendCode(id, code, vec);
            hnswIndex.add(id, IVector.of(vec));
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    /** 追加一条 PQ 码到紧凑数组，必要时扩容；调用前需持写锁。 */
    private void appendCode(String id, byte[] code, float[] rawVec) {
        if (codesArr == null) {
            codesArr = new byte[16][];
            codesIds = new String[16];
            codesCount = 0;
        } else if (codesCount >= codesArr.length) {
            int newCap = codesArr.length + (codesArr.length >> 1) + 1;
            codesArr = Arrays.copyOf(codesArr, newCap);
            codesIds = Arrays.copyOf(codesIds, newCap);
        }
        codesArr[codesCount] = code;
        codesIds[codesCount] = id;
        idToCodeIndex.put(id, codesCount);
        rawVectors.put(id, rawVec);
        codesCount++;
    }

    @Override
    public boolean remove(String id) {
        long stamp = lock.writeLock();
        try {
            if (pendingVectors.remove(id) != null) {
                return true;
            }
            Integer idx = idToCodeIndex.remove(id);
            if (idx != null) {
                int i = idx;
                int last = codesCount - 1;
                if (i != last) {
                    codesArr[i] = codesArr[last];
                    codesIds[i] = codesIds[last];
                    idToCodeIndex.put(codesIds[i], i);
                }
                codesArr[last] = null;
                codesIds[last] = null;
                codesCount--;
                rawVectors.remove(id);
            }
            // Rust HNSW 后端可能不支持删除；忽略异常，已从本地存储移除即可（搜索时会过滤 ghost）
            boolean hnswRemoved = false;
            if (hnswIndex != null) {
                try {
                    hnswRemoved = hnswIndex.remove(id);
                } catch (UnsupportedOperationException ignored) {
                    hnswRemoved = false;
                }
            }
            return hnswRemoved || idx != null;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean contains(String id) {
        long stamp = lock.readLock();
        try {
            if (pendingVectors.containsKey(id)) {
                return true;
            }
            if (idToCodeIndex.containsKey(id)) {
                return true;
            }
            if (hnswIndex != null) {
                return hnswIndex.contains(id);
            }
            return false;
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void clear() {
        long stamp = lock.writeLock();
        try {
            pendingVectors.clear();
            idToCodeIndex.clear();
            rawVectors.clear();
            codesArr = null;
            codesIds = null;
            codesCount = 0;
            if (hnswIndex != null) {
                hnswIndex.close();
                hnswIndex = null;
            }
            centroids = null;
            rotation = null;
            rotationT = null;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public void close() {
        clear();
    }
}
