package com.yishape.lab.math.ml.clu;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.IRichReport;
import com.yishape.lab.util.ReportBuilder;

import java.util.*;

/**
 * K-means++ clustering algorithm.
 *
 * <p>sklearn-style API: primary constructor accepts {@code k} (number of clusters).
 * Parameters can also be set via {@link #setParameters(Map)} with keys: {@code "k"} /
 * {@code "numClusters"}, {@code "maxIterations"}, {@code "tolerance"} /
 * {@code "convergenceThreshold"}, {@code "n_init"} / {@code "maxInitAttempts"},
 * {@code "random_state"} / {@code "seed"} / {@code "randomSeed"}.</p>
 *
 * <p><b>Thread safety:</b> This class is <strong>not</strong> thread-safe.
 * Concurrent calls to {@code fit} / {@code predict} are not supported.</p>
 *
 * @author reremouse
 */
public class KMeansPlusPlus implements IClustering {

    // ==================== 默认常量 ====================

    public static final int DEFAULT_K = 3;
    public static final int DEFAULT_MAX_ITERATIONS = 100;
    public static final double DEFAULT_TOLERANCE = 1e-6;
    public static final int DEFAULT_MAX_INIT_ATTEMPTS = 10;
    public static final long DEFAULT_SEED = 42L;

    private static final double NUMERICAL_STABILITY_EPS = 1e-12;

    // ==================== 可配置参数 ====================

    private long seed;
    private Random random;
    private int maxIterations;
    private double convergenceThreshold;
    private int maxInitAttempts;

    // ==================== 算法状态 ====================

    private int numClusters;
    private List<IVector<Double>> clusterCenters;
    private int[] labels;
    private double inertia;
    private boolean converged;
    private int iterations;
    private int dimension;
    private final Map<String, Object> parameters = new HashMap<>();

    // ==================== 构造函数 ====================

    /** 默认构造：k=3, seed=42L */
    public KMeansPlusPlus() {
        this(DEFAULT_K, DEFAULT_SEED);
    }

    /** @param k 聚类数量 */
    public KMeansPlusPlus(int k) {
        this(k, DEFAULT_SEED);
    }

    /** @param seed 随机种子（向后兼容旧版单参构造器） */
    public KMeansPlusPlus(long seed) {
        this(DEFAULT_K, seed);
    }

    /** @param k    聚类数量
     *  @param seed 随机种子 */
    public KMeansPlusPlus(int k, long seed) {
        this(k, seed, DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE);
    }

    /** @param k        聚类数量
     *  @param seed     随机种子
     *  @param maxIter  最大迭代次数
     *  @param tol      收敛阈值 */
    public KMeansPlusPlus(int k, long seed, int maxIter, double tol) {
        this(k, seed, maxIter, tol, DEFAULT_MAX_INIT_ATTEMPTS);
    }

    /** 完全参数构造器（主构造器）。
     *  @param k                聚类数量
     *  @param seed             随机种子
     *  @param maxIter          最大迭代次数
     *  @param tol              收敛阈值
     *  @param maxInitAttempts  K-means++ 初始化尝试次数 */
    public KMeansPlusPlus(int k, long seed, int maxIter, double tol, int maxInitAttempts) {
        validateK(k);
        this.numClusters = k;
        this.seed = seed;
        this.random = new Random(seed);
        this.maxIterations = maxIter;
        this.convergenceThreshold = tol;
        this.maxInitAttempts = maxInitAttempts;
    }

    /** CluWrapper 兼容构造器。
     *  @param k        聚类数量
     *  @param random   随机数生成器
     *  @param maxIter  最大迭代次数
     *  @param tol      收敛阈值 */
    public KMeansPlusPlus(int k, Random random, int maxIter, double tol) {
        validateK(k);
        this.numClusters = k;
        this.random = Objects.requireNonNull(random, "random");
        this.seed = 0L; // 外部 Random，seed 不可知
        this.maxIterations = maxIter;
        this.convergenceThreshold = tol;
        this.maxInitAttempts = DEFAULT_MAX_INIT_ATTEMPTS;
    }

    private static void validateK(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("聚类数量必须大于0，当前值: " + k);
        }
    }

    // ==================== K-means++ 初始化 ====================

    /**
     * K-means++ 初始化聚类中心（适配 Vector 列表接口）。
     */
    public List<IVector<Double>> initializeCenters(List<IVector<Double>> data, int k) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("数据列表不能为空");
        }
        IMatrix<Double> dataMatrix = convertToMatrix(data);
        IMatrix<Double> centerMatrix = initializeCenters(dataMatrix, k);
        return convertToVectorList(centerMatrix);
    }

    /**
     * K-means++ 初始化聚类中心。
     * 尝试多次初始化，选择分散度最好的一次。
     */
    public IMatrix<Double> initializeCenters(IMatrix<Double> data, int k) {
        if (data == null || data.getRowNum() == 0 || data.getColNum() == 0) {
            throw new IllegalArgumentException("数据矩阵不能为空");
        }
        if (k <= 0 || k > data.getRowNum()) {
            throw new IllegalArgumentException("聚类数量必须在1到数据点数量之间");
        }

        IMatrix<Double> bestCenters = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int attempt = 0; attempt < maxInitAttempts; attempt++) {
            try {
                IMatrix<Double> centers = performKMeansPlusPlusInit(data, k);
                double score = evaluateInitDispersion(data, centers);
                if (score > bestScore) {
                    bestScore = score;
                    bestCenters = centers;
                }
            } catch (RuntimeException e) {
                // 单次初始化失败，继续尝试
            }
        }

        if (bestCenters == null) {
            return randomInitialization(data, k);
        }
        return bestCenters;
    }

    /**
     * 执行一次 K-means++ 初始化（加权概率采样选择初始中心）。
     */
    private IMatrix<Double> performKMeansPlusPlusInit(IMatrix<Double> data, int k) {
        int n = data.getRowNum();
        int d = data.getColNum();

        IMatrix<Double> centers = Linalg.zeros(k, d);
        boolean[] chosen = new boolean[n];

        // 1. 随机选择第一个中心
        int firstCenter = random.nextInt(n);
        copyRow(data, firstCenter, centers, 0);
        chosen[firstCenter] = true;

        // 2. 依次选择剩余中心
        for (int centerIdx = 1; centerIdx < k; centerIdx++) {
            double[] distances = new double[n];
            double totalDistance = 0.0;

            for (int i = 0; i < n; i++) {
                if (chosen[i]) {
                    distances[i] = 0.0;
                    continue;
                }
                double minDistSq = Double.POSITIVE_INFINITY;
                for (int c = 0; c < centerIdx; c++) {
                    double distSq = computeSquaredDistance(data, i, centers, c);
                    minDistSq = Math.min(minDistSq, distSq);
                }
                distances[i] = Math.max(minDistSq, NUMERICAL_STABILITY_EPS);
                totalDistance += distances[i];
            }

            // 3. 按概率选择下一个中心
            if (totalDistance < NUMERICAL_STABILITY_EPS) {
                int nextCenter = selectRandomUnchosen(chosen);
                if (nextCenter == -1) break;
                copyRow(data, nextCenter, centers, centerIdx);
                chosen[nextCenter] = true;
            } else {
                double target = random.nextDouble() * totalDistance;
                double cumulative = 0.0;
                int selectedPoint = -1;

                for (int i = 0; i < n; i++) {
                    if (chosen[i]) continue;
                    cumulative += distances[i];
                    if (cumulative >= target) {
                        selectedPoint = i;
                        break;
                    }
                }

                if (selectedPoint == -1) {
                    selectedPoint = selectRandomUnchosen(chosen);
                }

                if (selectedPoint != -1) {
                    copyRow(data, selectedPoint, centers, centerIdx);
                    chosen[selectedPoint] = true;
                }
            }
        }

        return centers;
    }

    private static void copyRow(IMatrix<Double> src, int srcRow, IMatrix<Double> dst, int dstRow) {
        int d = src.getColNum();
        for (int j = 0; j < d; j++) {
            dst.set(dstRow, j, src.get(srcRow, j));
        }
    }

    /**
     * 随机初始化聚类中心（fallback）。
     */
    private IMatrix<Double> randomInitialization(IMatrix<Double> data, int k) {
        int n = data.getRowNum();
        int d = data.getColNum();

        IMatrix<Double> centers = Linalg.zeros(k, d);
        Set<Integer> selected = new HashSet<>();

        for (int i = 0; i < k; i++) {
            int idx;
            do {
                idx = random.nextInt(n);
            } while (selected.contains(idx));
            selected.add(idx);
            copyRow(data, idx, centers, i);
        }

        return centers;
    }

    /**
     * 评估初始化的分散度（所有点到最近中心的距离之和的负值，越大越好）。
     */
    private double evaluateInitDispersion(IMatrix<Double> data, IMatrix<Double> centers) {
        int n = data.getRowNum();
        int k = centers.getRowNum();

        double totalDistance = 0.0;
        for (int i = 0; i < n; i++) {
            double minDist = Double.POSITIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                double dist = computeSquaredDistance(data, i, centers, c);
                minDist = Math.min(minDist, dist);
            }
            totalDistance += minDist;
        }

        return -totalDistance;
    }

    /**
     * 计算数据点与聚类中心的平方欧氏距离。
     */
    private static double computeSquaredDistance(IMatrix<Double> data, int dataIdx,
                                                  IMatrix<Double> centers, int centerIdx) {
        double sum = 0.0;
        int d = data.getColNum();
        for (int j = 0; j < d; j++) {
            double diff = data.get(dataIdx, j) - centers.get(centerIdx, j);
            sum += diff * diff;
        }
        return sum;
    }

    private int selectRandomUnchosen(boolean[] chosen) {
        List<Integer> unchosen = new ArrayList<>();
        for (int i = 0; i < chosen.length; i++) {
            if (!chosen[i]) {
                unchosen.add(i);
            }
        }
        if (unchosen.isEmpty()) {
            return -1;
        }
        return unchosen.get(random.nextInt(unchosen.size()));
    }

    // ==================== K-means 聚类 ====================

    /**
     * 执行 K-means Lloyd 迭代。空聚类时重新初始化为随机数据点。
     */
    private KMeansResult performKMeansClustering(IMatrix<Double> data, IMatrix<Double> initialCenters) {
        int n = data.getRowNum();
        int d = data.getColNum();
        int k = initialCenters.getRowNum();

        IMatrix<Double> centers = initialCenters.copy();
        int[] assignments = new int[n];
        boolean localConverged = false;
        int localIterations = 0;
        double previousInertia = Double.POSITIVE_INFINITY;

        while (!localConverged && localIterations < maxIterations) {
            // 分配步骤
            for (int i = 0; i < n; i++) {
                double minDistance = Double.POSITIVE_INFINITY;
                int bestCluster = 0;
                for (int c = 0; c < k; c++) {
                    double distance = computeSquaredDistance(data, i, centers, c);
                    if (distance < minDistance) {
                        minDistance = distance;
                        bestCluster = c;
                    }
                }
                assignments[i] = bestCluster;
            }

            // 更新步骤
            IMatrix<Double> newCenters = Linalg.zeros(k, d);
            int[] clusterSizes = new int[k];

            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                clusterSizes[cluster]++;
                for (int j = 0; j < d; j++) {
                    newCenters.set(cluster, j, newCenters.get(cluster, j) + data.get(i, j));
                }
            }

            // 计算新中心 / 处理空聚类
            for (int c = 0; c < k; c++) {
                if (clusterSizes[c] > 0) {
                    for (int j = 0; j < d; j++) {
                        newCenters.set(c, j, newCenters.get(c, j) / clusterSizes[c]);
                    }
                } else {
                    // 空聚类：重新初始化为随机数据点
                    int randomIdx = random.nextInt(n);
                    copyRow(data, randomIdx, newCenters, c);
                }
            }

            // 检查收敛
            double currentInertia = 0.0;
            for (int i = 0; i < n; i++) {
                int cluster = assignments[i];
                currentInertia += computeSquaredDistance(data, i, newCenters, cluster);
            }

            double inertiaChange = Math.abs(previousInertia - currentInertia);
            localConverged = inertiaChange < convergenceThreshold;
            previousInertia = currentInertia;

            centers = newCenters;
            localIterations++;
        }

        return new KMeansResult(centers, assignments, previousInertia, localConverged, localIterations);
    }

    // ==================== 内部聚类流程 ====================

    private void clusterInternal(IMatrix<Double> data, int k) {
        if (data == null || data.getRowNum() == 0) {
            throw new IllegalArgumentException("数据矩阵不能为空");
        }

        int n = data.getRowNum();
        int d = data.getColNum();

        if (k <= 0 || k > n) {
            throw new IllegalArgumentException("聚类数量必须在1到数据点数量之间，k=" + k + ", n=" + n);
        }

        // 先验证，再设置状态
        this.dimension = d;
        this.numClusters = k;

        IMatrix<Double> initialCentersMatrix = initializeCenters(data, k);

        KMeansResult result = performKMeansClustering(data, initialCentersMatrix);

        this.clusterCenters = convertToVectorList(result.getCenters());
        this.labels = result.getAssignments().clone();
        this.inertia = result.getInertia();
        this.converged = result.isConverged();
        this.iterations = result.getIterations();
    }

    // ==================== 数据转换工具 ====================

    private static IMatrix<Double> convertToMatrix(List<IVector<Double>> vectors) {
        if (vectors == null || vectors.isEmpty()) {
            throw new IllegalArgumentException("向量列表不能为空");
        }
        int rows = vectors.size();
        int cols = vectors.get(0).size();
        IMatrix<Double> matrix = Linalg.zeros(rows, cols);
        for (int i = 0; i < rows; i++) {
            IVector<Double> row = vectors.get(i);
            if (row.size() != cols) {
                throw new IllegalArgumentException("所有向量必须具有相同的维度");
            }
            for (int j = 0; j < cols; j++) {
                matrix.set(i, j, row.get(j));
            }
        }
        return matrix;
    }

    private static List<IVector<Double>> convertToVectorList(IMatrix<Double> m) {
        List<IVector<Double>> vectors = new ArrayList<>();
        int rows = m.getRowNum();
        int cols = m.getColNum();
        for (int i = 0; i < rows; i++) {
            IVector<Double> vector = Linalg.zeros(cols);
            for (int j = 0; j < cols; j++) {
                vector.set(j, m.get(i, j));
            }
            vectors.add(vector);
        }
        return vectors;
    }

    // ==================== KMeansResult ====================

    /**
     * K-means 聚类结果。
     */
    public static class KMeansResult implements IRichReport {
        private final IMatrix<Double> centers;
        private final int[] assignments;
        private final double inertia;
        private final boolean converged;
        private final int iterations;

        public KMeansResult(IMatrix<Double> centers, int[] assignments, double inertia,
                            boolean converged, int iterations) {
            this.centers = centers;
            this.assignments = assignments;
            this.inertia = inertia;
            this.converged = converged;
            this.iterations = iterations;
        }

        public IMatrix<Double> getCenters() { return centers; }
        public int[] getAssignments() { return assignments; }
        public double getInertia() { return inertia; }
        public boolean isConverged() { return converged; }
        public int getIterations() { return iterations; }

        @Override
        public String toReport() {
            ReportBuilder rb = new ReportBuilder("K-Means Clustering Result");
            rb.kv("Clusters (k)", centers.getRowNum());
            rb.kv("Inertia", String.format("%.6f", inertia));
            rb.kv("Converged", converged);
            rb.kv("Iterations", iterations);
            rb.kv("Data points", assignments.length);
            return rb.build();
        }

        @Override
        public String toBriefReport() {
            return String.format("KMeans | k=%d | inertia=%.4f | converged=%s | iters=%d",
                    centers.getRowNum(), inertia, converged, iterations);
        }
    }

    // ==================== IClustering 接口实现 ====================

    @Override
    public IClustering fit(List<IVector<Double>> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("数据列表不能为空");
        }
        IMatrix<Double> dataMatrix = convertToMatrix(data);
        clusterInternal(dataMatrix, this.numClusters);
        return this;
    }

    @Override
    public IClustering fit(IMatrix<Double> data) {
        if (data == null || data.getRowNum() == 0) {
            throw new IllegalArgumentException("数据矩阵不能为空");
        }
        clusterInternal(data, this.numClusters);
        return this;
    }

    @Override
    public int[] fitPredict(List<IVector<Double>> data) {
        fit(data);
        return getLabels();
    }

    @Override
    public int[] fitPredict(IMatrix<Double> data) {
        fit(data);
        return getLabels();
    }

    @Override
    public int[] predict(List<IVector<Double>> data) {
        if (clusterCenters == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        if (data == null) {
            throw new IllegalArgumentException("数据不能为空");
        }
        int[] predictions = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            predictions[i] = predict(data.get(i));
        }
        return predictions;
    }

    @Override
    public int predict(IVector<Double> dataPoint) {
        if (clusterCenters == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        if (dataPoint == null) {
            throw new IllegalArgumentException("数据点不能为空");
        }

        double minDistSq = Double.POSITIVE_INFINITY;
        int bestCluster = 0;
        int d = dataPoint.size();

        for (int i = 0; i < clusterCenters.size(); i++) {
            IVector<Double> center = clusterCenters.get(i);
            double distSq = 0.0;
            for (int j = 0; j < d; j++) {
                double diff = dataPoint.get(j) - center.get(j);
                distSq += diff * diff;
            }
            if (distSq < minDistSq) {
                minDistSq = distSq;
                bestCluster = i;
            }
        }
        return bestCluster;
    }

    @Override
    public List<IVector<Double>> getClusterCenters() {
        if (clusterCenters == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        List<IVector<Double>> copy = new ArrayList<>(clusterCenters.size());
        for (IVector<Double> center : clusterCenters) {
            copy.add(center.copy());
        }
        return copy;
    }

    @Override
    public int[] getLabels() {
        if (labels == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        return labels.clone();
    }

    @Override
    public int getNumClusters() {
        return numClusters;
    }

    @Override
    public int getDimension() {
        return dimension;
    }

    @Override
    public double getInertia() {
        return inertia;
    }

    @Override
    public boolean isConverged() {
        return converged;
    }

    @Override
    public int getIterations() {
        return iterations;
    }

    @Override
    public ClusteringMetrics evaluateQuality(List<IVector<Double>> data) {
        if (clusterCenters == null || labels == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        return ClusteringMetrics.compute(data, clusterCenters, labels);
    }

    @Override
    public String getAlgorithmName() {
        return "K-Means++";
    }

    // ==================== 参数管理 ====================

    @Override
    public void setParameters(Map<String, Object> params) {
        this.parameters.putAll(params);

        // 提取并按类型更新已知参数
        Object kVal = params.getOrDefault("k", params.get("numClusters"));
        if (kVal instanceof Number n) {
            int newK = n.intValue();
            validateK(newK);
            this.numClusters = newK;
        }

        if (params.containsKey("maxIterations")) {
            Object v = params.get("maxIterations");
            if (v instanceof Number n) this.maxIterations = n.intValue();
        }

        Object tolVal = params.getOrDefault("tolerance", params.get("convergenceThreshold"));
        if (tolVal instanceof Number n) {
            this.convergenceThreshold = n.doubleValue();
        }

        Object nInitVal = params.getOrDefault("n_init", params.get("maxInitAttempts"));
        if (nInitVal instanceof Number n) {
            this.maxInitAttempts = n.intValue();
        }

        Object seedVal = params.getOrDefault("random_state",
                params.getOrDefault("seed", params.get("randomSeed")));
        if (seedVal instanceof Number n) {
            this.seed = n.longValue();
            this.random = new Random(this.seed);
        }
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("k", numClusters);
        params.put("numClusters", numClusters);
        params.put("maxIterations", maxIterations);
        params.put("tolerance", convergenceThreshold);
        params.put("convergenceThreshold", convergenceThreshold);
        params.put("n_init", maxInitAttempts);
        params.put("maxInitAttempts", maxInitAttempts);
        params.put("random_state", seed);
        params.put("seed", seed);
        params.put("algorithmName", getAlgorithmName());
        params.put("algorithm", getAlgorithmName());
        return params;
    }

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("seed", seed);
        p.put("maxIterations", maxIterations);
        p.put("convergenceThreshold", convergenceThreshold);
        p.put("maxInitAttempts", maxInitAttempts);
        p.put("numClusters", numClusters);
        p.put("inertia", inertia);
        p.put("converged", converged);
        p.put("iterations", iterations);
        p.put("dimension", dimension);
        // cluster centers
        if (clusterCenters != null) {
            double[][] centers = new double[clusterCenters.size()][];
            for (int i = 0; i < clusterCenters.size(); i++) {
                centers[i] = clusterCenters.get(i).toDoubleArray();
            }
            p.put("clusterCenters", centers);
        }
        if (labels != null) p.put("labels", labels.clone());
        return p;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void fromParams(Map<String, Object> p) {
        this.seed = ((Number) p.get("seed")).longValue();
        this.random = new Random(seed);
        this.maxIterations = ((Number) p.get("maxIterations")).intValue();
        this.convergenceThreshold = ((Number) p.get("convergenceThreshold")).doubleValue();
        this.maxInitAttempts = ((Number) p.get("maxInitAttempts")).intValue();
        this.numClusters = ((Number) p.get("numClusters")).intValue();
        this.inertia = ((Number) p.get("inertia")).doubleValue();
        this.converged = (Boolean) p.get("converged");
        this.iterations = ((Number) p.get("iterations")).intValue();
        this.dimension = ((Number) p.get("dimension")).intValue();
        double[][] centers = (double[][]) p.get("clusterCenters");
        if (centers != null) {
            this.clusterCenters = new ArrayList<>();
            for (double[] c : centers) {
                this.clusterCenters.add(Linalg.vector(c));
            }
        }
        this.labels = (int[]) p.get("labels");
    }
}
