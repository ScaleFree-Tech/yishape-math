package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.util.Tuple3;
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
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 乘积量化（Product Quantization）单精度向量索引，支持 OPQ 旋转优化。
 *
 * <p>核心特征：</p>
 * <ul>
 *   <li><strong>子空间分解</strong>：向量划分为若干子空间，
 *       每个子空间独立训练 k-means 码本；</li>
 *   <li><strong>ADC 快速查询</strong>：查询时预先计算 query 与各子空间码本的距离表，
 *       通过查表累加即可得到近似距离，时间复杂度 O(n·numSubspaces)；</li>
 *   <li><strong>OPQ 旋转优化</strong>（可选）：学习正交旋转矩阵，
 *       最小化量化误差；通过幂迭代近似 SVD 求解 Procrustes 问题，
 *       避免完整 SVD 分解；</li>
 *   <li><strong>精确重排序</strong>：ADC 粗排后取超集，用原始向量精确计算距离并取 Top-k。</li>
 * </ul>
 *
 * <p>内存压缩比：原始 float[] 为 4·dim 字节/向量；
 * PQ 编码后为 numSubspaces 字节/向量（codebookSize=256 时），
 * 典型压缩比 4x-32x。</p>
 *
 * <p>线程安全：{@link StampedLock} 实现多读单写。</p>
 */
public class PqFloatVecIdx implements IFloatVecIdx, IMutableVecIdx<Float> {

    private static final long serialVersionUID = 1L;

    /** 默认码本大小（256 = 1 byte per subspace） */
    private static final int DEFAULT_CODEBOOK_SIZE = 256;
    /** OPQ 迭代次数 */
    private static final int OPQ_MAX_ITER = 5;
    /** k-means 最大迭代 */
    private static final int KMEANS_MAX_ITER = 10;
    /** k-means 训练采样上限 */
    private static final int KMEANS_SAMPLE_LIMIT = 65536;

    // ==================== 配置 ====================

    private final int dimensions;
    private final int numSubspaces;
    private final int[] subspaceDims; // 每个子空间的实际维度
    private final int[] subspaceOffsets; // 每个子空间在向量中的起始偏移
    private final int codebookSize;
    private final IDisMetric<Float> metric;
    private final boolean useOpq;
    private final boolean normalize; // cosine 预归一化
    private final boolean distanceIsSquared;
    private final int adcEf; // ADC 粗排候选数

    // ==================== 训练后状态 ====================

    /**
     * 码本：centroids[subspace][code][subspaceDim]
     * subspace ∈ [0, numSubspaces)，code ∈ [0, codebookSize)
     */
    private float[][][] centroids;

    /** OPQ 旋转矩阵：rotation[dim][dim]，仅 useOpq=true 时非 null */
    private float[][] rotation;
    /** OPQ 旋转矩阵转置，用于查询时快速旋转 */
    private float[][] rotationT;

    /** 紧凑数组形式的 PQ 码（用于 ADC 全扫描，避免 HashMap 迭代分配） */
    private byte[][] codesArr;
    /** 与 codesArr 对应的 id 数组 */
    private String[] codesIds;
    /** 已编码向量数量 */
    private int codesCount;
    /** id -> 在 codesArr/codesIds 中的位置（用于 O(1) 删除） */
    private final Map<String, Integer> idToCodeIndex = new HashMap<>();

    /** 原始向量存储，用于精确重排序与 getVector */
    private final Map<String, float[]> vectors;

    private final StampedLock lock = new StampedLock();

    // ==================== 构造函数 ====================

    /**
     * 从已有数据构建 PQ 索引（自动训练码本）。
     *
     * @param data      训练数据
     * @param ids       标识
     * @param metric    距离度量
     * @param options   构建选项；{@code m()} 作为子空间维度建议
     * @param useOpq    是否启用 OPQ 旋转优化
     */
    public PqFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options, boolean useOpq) {
        this.dimensions = validateAndGetDims(data, ids);
        this.metric = Objects.requireNonNull(metric, "metric");
        this.useOpq = useOpq;
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "PQ only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;
        VecSearchOption opts = options != null ? options : VecSearchOption.DEFAULT;

        int suggestedSubspaceDim = Math.max(1, Math.min(dimensions, opts.hnswM()));
        // 默认每子空间 8 维是 PQ 文献中的标准选择
        if (suggestedSubspaceDim > dimensions / 2 && dimensions >= 8) {
            suggestedSubspaceDim = 8;
        }
        this.subspaceDims = computeSubspaceDims(dimensions, suggestedSubspaceDim);
        this.numSubspaces = subspaceDims.length;
        this.subspaceOffsets = computeOffsets(subspaceDims);
        this.codebookSize = Math.min(DEFAULT_CODEBOOK_SIZE, ids.length);
        this.adcEf = Math.max(16, opts.hnswEfSearch());

        this.vectors = new HashMap<>();

        // 拷贝并归一化原始向量
        float[][] normalizedData = new float[data.length][];
        for (int i = 0; i < data.length; i++) {
            normalizedData[i] = normalize ? normalize(data[i]) : data[i].clone();
            vectors.put(ids[i], normalizedData[i].clone());
        }

        // 训练（仅训练码本，不写入 codes）
        train(normalizedData);

        // 训练完成后，将所有向量编码并填入紧凑数组
        encodeAllIntoArrays(normalizedData, ids);
    }

    /**
     * 从已有数据构建 PQ 索引（默认不启用 OPQ）。
     */
    public PqFloatVecIdx(float[][] data, String[] ids,
            IDisMetric<Float> metric, VecSearchOption options) {
        this(data, ids, metric, options, false);
    }

    /**
     * 构造空 PQ 索引（用于后续动态增删）。
     * 训练延迟到有足够数据时进行。
     */
    public PqFloatVecIdx(int dimensions, IDisMetric<Float> metric,
            boolean useOpq) {
        if (dimensions <= 0) {
            throw new IllegalArgumentException("dimensions 必须为正");
        }
        this.dimensions = dimensions;
        this.metric = Objects.requireNonNull(metric, "metric");
        this.useOpq = useOpq;
        MetricType mt = metric.type();
        if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
            throw new IllegalArgumentException(
                    "PQ only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
        }
        this.distanceIsSquared = mt == MetricType.SQUARED_EUCLIDEAN;
        this.normalize = mt == MetricType.COSINE;

        int suggestedSubspaceDim = Math.max(1, Math.min(dimensions, 8));
        this.subspaceDims = computeSubspaceDims(dimensions, suggestedSubspaceDim);
        this.numSubspaces = subspaceDims.length;
        this.subspaceOffsets = computeOffsets(subspaceDims);
        this.codebookSize = DEFAULT_CODEBOOK_SIZE;
        this.adcEf = 200;

        this.vectors = new HashMap<>();
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

    // ==================== 训练 ====================

    private void train(float[][] data) {
        int n = data.length;
        if (n == 0) {
            return;
        }

        if (useOpq) {
            trainOpq(data);
        } else {
            trainCentroids(data);
        }
    }

    /** 仅训练码本，不写入 codes 容器。 */
    private void trainCentroids(float[][] data) {
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

        // 初始化 R = I
        rotation = new float[d][d];
        for (int i = 0; i < d; i++) {
            rotation[i][i] = 1.0f;
        }

        IMatrix<Float> dataMat = Linalg.matrix(data);

        // OPQ 迭代：使用局部 byte[][] 缓存当轮编码，避免 HashMap 开销
        for (int iter = 0; iter < OPQ_MAX_ITER; iter++) {
            // 1. 旋转数据：rotated = data * R
            float[][] rotatedData = dataMat.mmul(Linalg.matrix(rotation)).toFloatArray();

            // 2. 在旋转空间训练码本
            trainCentroids(rotatedData);

            // 3. 编码 + 重建（仅本轮使用）
            float[][] reconstructed = new float[n][d];
            for (int i = 0; i < n; i++) {
                byte[] code = encode(rotatedData[i]);
                decodeToVector(code, reconstructed[i]);
            }

            // 4. 求解 Procrustes 问题更新 R
            IMatrix<Float> reconMat = Linalg.matrix(reconstructed);
            IMatrix<Float> Mmat = dataMat.transpose().mmul(reconMat);
            Tuple3<IMatrix<Float>, IVector<Float>, IMatrix<Float>> svd = Mmat.svd();
            IMatrix<Float> U = svd.getFirst();
            IMatrix<Float> VT = svd.getThird();
            rotation = U.mmul(VT).toFloatArray();
        }

        // 最终训练一次码本
        float[][] finalRotated = dataMat.mmul(Linalg.matrix(rotation)).toFloatArray();
        trainCentroids(finalRotated);

        // 预计算转置
        rotationT = Linalg.matrix(rotation).transpose().toFloatArray();
    }

    /** 将所有训练向量编码后填入紧凑数组。需先完成 train()。 */
    private void encodeAllIntoArrays(float[][] data, String[] ids) {
        int n = data.length;
        this.codesArr = new byte[n][];
        this.codesIds = new String[n];
        IMatrix<Float> RT = (useOpq && rotationT != null) ? Linalg.matrix(rotationT) : null;
        for (int i = 0; i < n; i++) {
            float[] vec = RT != null ? RT.mmul(IVector.of(data[i])).toFloatArray() : data[i];
            byte[] code = encode(vec);
            codesArr[i] = code;
            codesIds[i] = ids[i];
            idToCodeIndex.put(ids[i], i);
        }
        this.codesCount = n;
    }

    // ==================== k-means ====================

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
                    float d = squaredDistance(sample[i], centroids[c]);
                    if (d < bestDist) {
                        bestDist = d;
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

    // ==================== 编码 / 解码 ====================

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

    // ==================== 矩阵运算（OPQ）====================
    // 已迁移至 IMatrix / Linalg 层实现

    // ==================== 工具方法 ====================

    private static float[][] extractSubspace(float[][] data, int offset, int dim) {
        int n = data.length;
        float[][] result = new float[n][dim];
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], offset, result[i], 0, dim);
        }
        return result;
    }

    private static float squaredDistance(float[] a, float[] b) {
        return Linalg.squaredDistance(a, b);
    }

    private float subspaceSquaredDistance(float[] vec, int offset, int dim, float[] centroid) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            double d = (double) vec[offset + i] - (double) centroid[i];
            sum += d * d;
        }
        return (float) sum;
    }

    private float[] normalize(float[] v) {
        return IVector.of(v).normalize().toFloatArray();
    }

    private double externalDistance(float squaredDist) {
        if (normalize) {
            return squaredDist * 0.5;
        }
        if (distanceIsSquared) {
            return squaredDist;
        }
        return Math.sqrt(squaredDist);
    }

    // ==================== ADC 查询 ====================

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public int size() {
        long stamp = lock.readLock();
        try {
            return vectors.size();
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
        if (k <= 0 || vectors.isEmpty()) {
            return List.of();
        }

        float[] q = normalize ? normalize(query) : query;

        // OPQ：旋转查询向量
        if (useOpq && rotationT != null) {
            IMatrix<Float> RT = Linalg.matrix(rotationT);
            q = RT.mmul(IVector.of(q)).toFloatArray();
        }

        long stamp = lock.readLock();
        try {
            if (centroids == null || codesCount == 0) {
                // 尚未训练：退化为暴力扫描
                return bruteForceSearch(q, k, excludeIds, filter);
            }

            // 1. 构建 ADC 距离表
            float[][] distanceTables = buildDistanceTables(q);

            // 2. ADC 扫描取候选（直接基于紧凑数组 + 手写 maxheap）
            int fetch = Math.min(Math.max(k, adcEf), codesCount);
            int[] candIdx = adcScanArray(distanceTables, fetch, excludeIds, filter);

            // 3. 精确重排序（用原始向量 + 最大堆，只保留 top-k）
            return rerankToTopK(q, candIdx, k);
        } finally {
            lock.unlockRead(stamp);
        }
    }

    private float[][] buildDistanceTables(float[] query) {
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
     * 基于紧凑数组扫描所有 codes，返回 top-fetch 的位置索引（按 ADC 距离升序）。
     */
    private int[] adcScanArray(float[][] distanceTables, int fetch,
            Collection<String> excludeIds, Predicate<String> filter) {
        Set<String> ex = (excludeIds == null || excludeIds.isEmpty()) ? Set.of()
                : (excludeIds instanceof Set<String> s ? s : new HashSet<>(excludeIds));

        // 手写最大堆：dist[] 为键，idx[] 为对应 codesArr 下标
        int[] heapIdx = new int[fetch];
        float[] heapDist = new float[fetch];
        int heapSize = 0;
        float worst = Float.MAX_VALUE;

        for (int i = 0; i < codesCount; i++) {
            String id = codesIds[i];
            if (!ex.isEmpty() && ex.contains(id)) {
                continue;
            }
            if (filter != null && !filter.test(id)) {
                continue;
            }
            byte[] code = codesArr[i];
            float dist = 0.0f;
            for (int s = 0; s < numSubspaces; s++) {
                dist += distanceTables[s][code[s] & 0xFF];
            }
            if (heapSize < fetch) {
                heapIdx[heapSize] = i;
                heapDist[heapSize] = dist;
                heapSize++;
                if (heapSize == fetch) {
                    buildMaxHeap(heapIdx, heapDist, heapSize);
                    worst = heapDist[0];
                }
            } else if (dist < worst) {
                heapIdx[0] = i;
                heapDist[0] = dist;
                siftDownHeap(heapIdx, heapDist, heapSize, 0);
                worst = heapDist[0];
            }
        }

        // 截到实际大小；调用方只用其下标，距离已不再需要
        if (heapSize < fetch) {
            return Arrays.copyOf(heapIdx, heapSize);
        }
        return heapIdx;
    }

    /**
     * 对 ADC 候选用精确距离重排序，并通过最大堆只保留 top-k。
     */
    private List<SearchHit> rerankToTopK(float[] query, int[] candIdx, int k) {
        PriorityQueue<SearchHit> maxHeap = new PriorityQueue<>(k + 1,
                (a, b) -> Double.compare(b.distance(), a.distance()));
        double worst = Double.POSITIVE_INFINITY;
        for (int ci : candIdx) {
            String id = codesIds[ci];
            float[] vec = vectors.get(id);
            if (vec == null) {
                continue;
            }
            float d = squaredDistance(query, vec);
            double ed = externalDistance(d);
            if (maxHeap.size() < k) {
                maxHeap.offer(new SearchHit(id, ed));
                if (maxHeap.size() == k) {
                    worst = maxHeap.peek().distance();
                }
            } else if (ed < worst) {
                maxHeap.poll();
                maxHeap.offer(new SearchHit(id, ed));
                worst = maxHeap.peek().distance();
            }
        }
        List<SearchHit> result = new ArrayList<>(maxHeap);
        result.sort((a, b) -> Double.compare(a.distance(), b.distance()));
        return result;
    }

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

    private List<SearchHit> bruteForceSearch(float[] query, int k,
            Collection<String> excludeIds, Predicate<String> filter) {
        Set<String> ex = excludeIds instanceof Set<String> s ? s
                : new HashSet<>(excludeIds != null ? excludeIds : List.of());

        PriorityQueue<SearchHit> maxHeap = new PriorityQueue<>(
                (a, b) -> Double.compare(b.distance(), a.distance()));

        for (Map.Entry<String, float[]> e : vectors.entrySet()) {
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
            float[] vec = vectors.get(id);
            if (vec == null) {
                return null;
            }
            return IVector.of(vec.clone());
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
            if (vectors.containsKey(id)) {
                throw new IllegalArgumentException("duplicate ID: " + id);
            }
            float[] vec = normalize ? normalize(vector.toFloatArray()) : vector.toFloatArray();
            vectors.put(id, vec);
            if (centroids != null) {
                // 已训练：直接编码并写入紧凑数组
                float[] encodedVec = vec;
                if (useOpq && rotationT != null) {
                    IMatrix<Float> RT = Linalg.matrix(rotationT);
                    encodedVec = RT.mmul(IVector.of(vec)).toFloatArray();
                }
                appendCode(id, encode(encodedVec));
            }
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    /** 追加一条 PQ 码到紧凑数组（必要时扩容）；调用前需持写锁。 */
    private void appendCode(String id, byte[] code) {
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
        codesCount++;
    }

    @Override
    public boolean remove(String id) {
        long stamp = lock.writeLock();
        try {
            if (!vectors.containsKey(id)) {
                return false;
            }
            vectors.remove(id);
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
            }
            return true;
        } finally {
            lock.unlockWrite(stamp);
        }
    }

    @Override
    public boolean contains(String id) {
        long stamp = lock.readLock();
        try {
            return vectors.containsKey(id);
        } finally {
            lock.unlockRead(stamp);
        }
    }

    @Override
    public void clear() {
        long stamp = lock.writeLock();
        try {
            vectors.clear();
            idToCodeIndex.clear();
            codesArr = null;
            codesIds = null;
            codesCount = 0;
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
