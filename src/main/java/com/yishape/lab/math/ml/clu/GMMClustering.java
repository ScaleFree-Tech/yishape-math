package com.yishape.lab.math.ml.clu;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.model.GaussianMixtureModel;
import com.yishape.lab.math.stats.model.EMAlgorithm;
import com.yishape.lab.math.stats.distribution.multiv.MultivariateNormalDistribution;

import java.util.*;

/**
 * Gaussian Mixture Model (GMM) clustering via EM algorithm.
 *
 * <p>sklearn-style API: primary constructor accepts {@code k} (number of clusters).
 * Parameters can also be set via {@link #setParameters(Map)} with keys: {@code "k"} /
 * {@code "numClusters"}, {@code "maxIterations"}, {@code "tolerance"},
 * {@code "n_init"} / {@code "numRestarts"}, {@code "random_state"} / {@code "seed"} /
 * {@code "randomSeed"}, {@code "useKMeansInit"}, {@code "verbose"}.</p>
 *
 * <p><b>Thread safety:</b> This class is <strong>not</strong> thread-safe.
 * Concurrent calls to {@code fit} / {@code predict} are not supported.</p>
 *
 * @author reremouse
 */
public class GMMClustering implements IClustering {

    // ==================== 默认常量 ====================

    public static final int DEFAULT_K = 3;
    public static final int DEFAULT_MAX_ITERATIONS = 100;
    public static final double DEFAULT_TOLERANCE = 1e-6;
    public static final int DEFAULT_NUM_RESTARTS = 10;
    public static final boolean DEFAULT_USE_KMEANS_INIT = true;
    public static final long DEFAULT_SEED = 42L;
    public static final boolean DEFAULT_VERBOSE = false;

    // ==================== 可配置参数 ====================

    private int numClusters;
    private long seed;
    private Random random;
    private int maxIterations;
    private double tolerance;
    private int numRestarts;
    private boolean useKMeansInit;
    private boolean verbose;
    private EMAlgorithm emAlgorithm;

    // ==================== 算法状态 ====================

    private GaussianMixtureModel trainedModel;
    private List<IVector<Double>> clusterCenters;
    private int[] labels;
    private double inertia;
    private boolean converged;
    private int iterations;
    private int dimension;
    private final Map<String, Object> parameters = new HashMap<>();

    // ==================== 构造函数 ====================

    /** 默认构造：k=3, seed=42L */
    public GMMClustering() {
        this(DEFAULT_K);
    }

    /** @param k 聚类数量 */
    public GMMClustering(int k) {
        this(k, DEFAULT_SEED);
    }

    /** @param seed 随机种子（向后兼容旧版单参构造器） */
    public GMMClustering(long seed) {
        this(DEFAULT_K, seed);
    }

    /** @param k    聚类数量
     *  @param seed 随机种子 */
    public GMMClustering(int k, long seed) {
        this(k, seed, DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE);
    }

    /** @param k        聚类数量
     *  @param seed     随机种子
     *  @param maxIter  最大迭代次数
     *  @param tol      收敛阈值 */
    public GMMClustering(int k, long seed, int maxIter, double tol) {
        this(k, seed, maxIter, tol, DEFAULT_NUM_RESTARTS, DEFAULT_USE_KMEANS_INIT, DEFAULT_VERBOSE);
    }

    /** 完全参数构造器（主构造器）。
     *  @param k              聚类数量
     *  @param seed           随机种子
     *  @param maxIter        最大迭代次数
     *  @param tol            收敛容忍度
     *  @param numRestarts    重启次数
     *  @param useKMeansInit  是否用 K-means++ 初始化
     *  @param verbose        是否输出详细信息 */
    public GMMClustering(int k, long seed, int maxIter, double tol, int numRestarts,
                         boolean useKMeansInit, boolean verbose) {
        validateK(k);
        this.numClusters = k;
        this.seed = seed;
        this.random = new Random(seed);
        this.maxIterations = maxIter;
        this.tolerance = tol;
        this.numRestarts = numRestarts;
        this.useKMeansInit = useKMeansInit;
        this.verbose = verbose;
        this.emAlgorithm = new EMAlgorithm(maxIter, tol, verbose);
    }

    /**
     * 旧版六参数构造器（向后兼容）。
     *
     * @param maxIterations 最大迭代次数
     * @param tolerance     收敛容忍度
     * @param numRestarts   重启次数
     * @param useKMeansInit 是否使用 K-means++ 初始化
     * @param randomSeed    随机种子
     * @param verbose       是否输出详细信息
     * @deprecated 请使用 {@link #GMMClustering(int, long, int, double, int, boolean, boolean)} 指定 k
     */
    @Deprecated
    public GMMClustering(int maxIterations, double tolerance, int numRestarts,
                         boolean useKMeansInit, long randomSeed, boolean verbose) {
        this(DEFAULT_K, randomSeed, maxIterations, tolerance, numRestarts, useKMeansInit, verbose);
    }

    private static void validateK(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("聚类数量必须大于0，当前值: " + k);
        }
    }

    // ==================== IClustering 接口实现 ====================

    @Override
    public IClustering fit(List<IVector<Double>> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("数据不能为空");
        }

        if (this.numClusters <= 0) {
            throw new IllegalStateException("聚类数量必须大于0，当前值: " + this.numClusters);
        }
        if (this.numClusters > data.size()) {
            throw new IllegalArgumentException("聚类数量必须小于等于数据点数量");
        }

        this.dimension = data.get(0).size();

        for (IVector<Double> point : data) {
            if (point.size() != dimension) {
                throw new IllegalArgumentException("所有数据点必须具有相同的维度");
            }
        }

        // 多重启动策略
        GaussianMixtureModel bestGmm = null;
        EMAlgorithm.EMResult bestResult = null;
        double bestLogLikelihood = Double.NEGATIVE_INFINITY;

        GaussianMixtureModel fallbackGmm = null;
        EMAlgorithm.EMResult fallbackResult = null;
        double fallbackLogLikelihood = Double.NEGATIVE_INFINITY;

        for (int restart = 0; restart < this.numRestarts; restart++) {
            try {
                Random restartRandom = new Random(random.nextLong());
                GaussianMixtureModel gmm = new GaussianMixtureModel(numClusters, dimension, restartRandom);

                switch (restart % 4) {
                    case 0:
                        try {
                            gmm.initializeWithKMeansPlusPlus(data);
                        } catch (RuntimeException e) {
                            gmm.initializeWithSmartRandom(data);
                        }
                        break;
                    case 1:
                        gmm.initializeWithSmartRandom(data);
                        break;
                    case 2:
                        try {
                            initializeWithDataDrivenStrategy(gmm, data);
                        } catch (RuntimeException e) {
                            gmm.initializeWithSmartRandom(data);
                        }
                        break;
                    case 3:
                        gmm.initializeRandomly(data);
                        break;
                }

                EMAlgorithm.EMResult result = emAlgorithm.fit(data, gmm);

                if (result != null && Double.isFinite(result.logLikelihood)) {
                    int[] tempLabels = new int[data.size()];
                    for (int i = 0; i < data.size(); i++) {
                        tempLabels[i] = gmm.predictComponent(data.get(i));
                    }

                    if (isValidClustering(tempLabels, numClusters)) {
                        if (result.logLikelihood > bestLogLikelihood) {
                            bestLogLikelihood = result.logLikelihood;
                            bestResult = result;
                            bestGmm = gmm;
                        }
                    } else {
                        if (result.logLikelihood > fallbackLogLikelihood) {
                            fallbackLogLikelihood = result.logLikelihood;
                            fallbackResult = result;
                            fallbackGmm = gmm;
                        }
                    }
                }
            } catch (RuntimeException e) {
                if (verbose) {
                    log(String.format("重启 %d 失败: %s", restart + 1, e.getMessage()));
                }
            }
        }

        // 选择最终模型
        GaussianMixtureModel finalGmm;
        EMAlgorithm.EMResult finalResult;

        if (bestGmm != null) {
            finalGmm = bestGmm;
            finalResult = bestResult;
        } else if (fallbackGmm != null) {
            finalGmm = fallbackGmm;
            finalResult = fallbackResult;
        } else {
            // 最后的备选方案：强制使用 K-means
            KMeansPlusPlus kmeans = new KMeansPlusPlus(numClusters, seed);
            kmeans.fit(data);

            this.clusterCenters = kmeans.getClusterCenters();
            this.labels = kmeans.getLabels();
            this.inertia = kmeans.getInertia();
            this.converged = kmeans.isConverged();
            this.iterations = kmeans.getIterations();
            this.trainedModel = null;
            return this;
        }

        this.trainedModel = finalGmm;

        int[] clusterLabels = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            clusterLabels[i] = finalGmm.predictComponent(data.get(i));
        }

        List<IVector<Double>> centers = new ArrayList<>();
        for (int k = 0; k < numClusters; k++) {
            centers.add(finalGmm.getComponent(k).getMean().copy());
        }

        this.clusterCenters = centers;
        this.labels = clusterLabels;
        this.inertia = computeInertia(data, centers, clusterLabels);
        this.converged = finalResult.converged;
        this.iterations = finalResult.iterations;

        return this;
    }

    @Override
    public IClustering fit(IMatrix<Double> data) {
        return fit(convertMatrixToVectorList(data));
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
        checkFitted();
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
    public int predict(IVector<Double> point) {
        checkFitted();
        if (point == null) {
            throw new IllegalArgumentException("数据点不能为空");
        }

        if (trainedModel != null) {
            return trainedModel.predictComponent(point);
        }

        // fallback：最近中心（使用平方距离，无需开方）
        double minDistSq = Double.POSITIVE_INFINITY;
        int bestCluster = 0;
        int d = point.size();
        for (int i = 0; i < clusterCenters.size(); i++) {
            IVector<Double> center = clusterCenters.get(i);
            double distSq = 0.0;
            for (int j = 0; j < d; j++) {
                double diff = point.get(j) - center.get(j);
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
        checkFitted();
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
        return "Gaussian Mixture Model";
    }

    // ==================== 参数管理 ====================

    @Override
    public void setParameters(Map<String, Object> params) {
        if (params == null) return;
        this.parameters.putAll(params);

        Object kVal = params.getOrDefault("k",
                params.getOrDefault("numClusters", params.get("n_components")));
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
        if (tolVal instanceof Number n) this.tolerance = n.doubleValue();

        Object restartsVal = params.getOrDefault("n_init", params.get("numRestarts"));
        if (restartsVal instanceof Number n) this.numRestarts = n.intValue();

        if (params.containsKey("useKMeansInit")) {
            Object v = params.get("useKMeansInit");
            if (v instanceof Boolean b) this.useKMeansInit = b;
        }

        Object seedVal = params.getOrDefault("random_state",
                params.getOrDefault("seed", params.get("randomSeed")));
        if (seedVal instanceof Number n) {
            this.seed = n.longValue();
            this.random = new Random(this.seed);
        }

        if (params.containsKey("verbose")) {
            Object v = params.get("verbose");
            if (v instanceof Boolean b) this.verbose = b;
        }

        this.emAlgorithm = new EMAlgorithm(maxIterations, tolerance, verbose);
    }

    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new LinkedHashMap<>();
        params.put("k", numClusters);
        params.put("numClusters", numClusters);
        params.put("n_components", numClusters);
        params.put("maxIterations", maxIterations);
        params.put("tolerance", tolerance);
        params.put("convergenceThreshold", tolerance);
        params.put("n_init", numRestarts);
        params.put("numRestarts", numRestarts);
        params.put("useKMeansInit", useKMeansInit);
        params.put("random_state", seed);
        params.put("seed", seed);
        params.put("randomSeed", seed);
        params.put("verbose", verbose);
        params.put("algorithmName", getAlgorithmName());
        return params;
    }

    // ==================== GMM 特有 API ====================

    /** @return 训练好的高斯混合模型，未训练时返回 null */
    public GaussianMixtureModel getTrainedModel() {
        return trainedModel;
    }

    /** 计算数据点属于各分量的后验概率。 */
    public List<IVector<Double>> computePosteriorProbabilities(List<IVector<Double>> data) {
        if (trainedModel == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        List<IVector<Double>> posteriors = new ArrayList<>();
        for (IVector<Double> point : data) {
            posteriors.add(trainedModel.computePosteriors(point));
        }
        return posteriors;
    }

    /** 计算数据的对数似然。 */
    public double computeLogLikelihood(List<IVector<Double>> data) {
        if (trainedModel == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        double logLikelihood = 0.0;
        for (IVector<Double> point : data) {
            logLikelihood += trainedModel.logPdf(point);
        }
        return logLikelihood;
    }

    /** 从训练好的模型中采样。 */
    public List<IVector<Double>> sample(int numSamples) {
        if (trainedModel == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        return trainedModel.sample(numSamples);
    }

    // ==================== 内部工具方法 ====================

    private void checkFitted() {
        if (trainedModel == null && clusterCenters == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
    }

    /**
     * 计算惯性（数据点到聚类中心的平方距离之和，不做多余开方）。
     */
    private static double computeInertia(List<IVector<Double>> data,
                                          List<IVector<Double>> centers, int[] labels) {
        double inertia = 0.0;
        int d = data.get(0).size();
        for (int i = 0; i < data.size(); i++) {
            IVector<Double> point = data.get(i);
            IVector<Double> center = centers.get(labels[i]);
            double distSq = 0.0;
            for (int j = 0; j < d; j++) {
                double diff = point.get(j) - center.get(j);
                distSq += diff * diff;
            }
            inertia += distSq;
        }
        return inertia;
    }

    private static List<IVector<Double>> convertMatrixToVectorList(IMatrix<Double> matrix) {
        List<IVector<Double>> vectors = new ArrayList<>();
        for (int i = 0; i < matrix.getRowNum(); i++) {
            IVector<Double> vector = Linalg.zeros(matrix.getColNum());
            for (int j = 0; j < matrix.getColNum(); j++) {
                vector.set(j, matrix.get(i, j));
            }
            vectors.add(vector);
        }
        return vectors;
    }

    /**
     * 数据驱动的初始化策略：基于全局协方差分布采样初始中心。
     */
    private void initializeWithDataDrivenStrategy(GaussianMixtureModel gmm, List<IVector<Double>> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空");
        }

        int n = data.size();
        int d = data.get(0).size();
        int k = gmm.getNumComponents();

        // 计算均值
        IVector<Double> mean = Linalg.zeros(d);
        for (IVector<Double> point : data) {
            for (int i = 0; i < d; i++) {
                mean.set(i, mean.get(i) + point.get(i));
            }
        }
        for (int i = 0; i < d; i++) {
            mean.set(i, mean.get(i) / n);
        }

        // 计算协方差矩阵
        IMatrix<Double> covariance = Linalg.zeros(d, d);
        for (IVector<Double> point : data) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    double diff_i = point.get(i) - mean.get(i);
                    double diff_j = point.get(j) - mean.get(j);
                    covariance.set(i, j, covariance.get(i, j) + diff_i * diff_j);
                }
            }
        }
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                covariance.set(i, j, covariance.get(i, j) / (n - 1));
            }
        }

        // 沿主方向分散初始中心
        List<IVector<Double>> centers = new ArrayList<>();
        IVector<Double> firstCenter = sampleFromGaussian(mean, covariance, 0.1);
        centers.add(firstCenter);

        for (int c = 1; c < k; c++) {
            IVector<Double> newCenter = null;
            double maxMinDistance = 0;
            for (int trial = 0; trial < 50; trial++) {
                IVector<Double> candidate = sampleFromGaussian(mean, covariance, 0.5);
                double minDistance = Double.MAX_VALUE;
                for (IVector<Double> center : centers) {
                    double dist = 0;
                    for (int j = 0; j < d; j++) {
                        double diff = candidate.get(j) - center.get(j);
                        dist += diff * diff;
                    }
                    minDistance = Math.min(minDistance, dist);
                }
                if (minDistance > maxMinDistance) {
                    maxMinDistance = minDistance;
                    newCenter = candidate;
                }
            }
            if (newCenter != null) {
                centers.add(newCenter);
            } else {
                centers.add(data.get(random.nextInt(data.size())));
            }
        }

        double weight = 1.0 / k;
        for (int c = 0; c < k; c++) {
            IVector<Double> center = centers.get(c);
            IMatrix<Double> componentCovariance = Linalg.zeros(d, d);
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    componentCovariance.set(i, j, covariance.get(i, j) * 0.5 / k);
                }
            }
            for (int i = 0; i < d; i++) {
                componentCovariance.set(i, i, Math.max(componentCovariance.get(i, i), 1e-6));
            }
            MultivariateNormalDistribution component = new MultivariateNormalDistribution(center, componentCovariance);
            gmm.setComponent(c, component);
            gmm.setWeight(c, weight);
        }
    }

    private IVector<Double> sampleFromGaussian(IVector<Double> mean, IMatrix<Double> covariance, double scale) {
        int d = mean.size();
        IVector<Double> sample = Linalg.zeros(d);
        for (int i = 0; i < d; i++) {
            double variance = Math.max(covariance.get(i, i) * scale, 1e-6);
            double stddev = Math.sqrt(variance);
            sample.set(i, mean.get(i) + random.nextGaussian() * stddev);
        }
        return sample;
    }

    private boolean isValidClustering(int[] labels, int numClusters) {
        if (labels == null || labels.length == 0) return false;

        int[] clusterCounts = new int[numClusters];
        int validLabels = 0;
        for (int label : labels) {
            if (label >= 0 && label < numClusters) {
                clusterCounts[label]++;
                validLabels++;
            }
        }
        if (validLabels != labels.length) return false;

        for (int count : clusterCounts) {
            if (count == 0) return false;
        }

        int minPointsPerCluster = Math.max(1, labels.length / (numClusters * 10));
        for (int count : clusterCounts) {
            if (count < minPointsPerCluster) return false;
        }
        return true;
    }

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("numClusters", numClusters);
        p.put("seed", seed);
        p.put("maxIterations", maxIterations);
        p.put("tolerance", tolerance);
        p.put("numRestarts", numRestarts);
        p.put("useKMeansInit", useKMeansInit);
        p.put("verbose", verbose);
        p.put("inertia", inertia);
        p.put("converged", converged);
        p.put("iterations", iterations);
        p.put("dimension", dimension);
        if (clusterCenters != null) {
            double[][] centers = new double[clusterCenters.size()][];
            for (int i = 0; i < clusterCenters.size(); i++) {
                centers[i] = clusterCenters.get(i).toDoubleArray();
            }
            p.put("clusterCenters", centers);
        }
        if (labels != null) p.put("labels", labels.clone());
        // GMM components
        if (trainedModel != null) {
            List<Map<String, Object>> compList = new ArrayList<>();
            for (int i = 0; i < trainedModel.getNumComponents(); i++) {
                Map<String, Object> cm = new LinkedHashMap<>();
                cm.put("mean", trainedModel.getComponent(i).getMean().toDoubleArray());
                cm.put("covariance", ((IMatrix) trainedModel.getComponent(i).getCovariance()).toDoubleArray());
                cm.put("weight", trainedModel.getWeight(i));
                compList.add(cm);
            }
            p.put("components", compList);
        }
        return p;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void fromParams(Map<String, Object> p) {
        this.numClusters = ((Number) p.get("numClusters")).intValue();
        this.seed = ((Number) p.get("seed")).longValue();
        this.random = new Random(seed);
        this.maxIterations = ((Number) p.get("maxIterations")).intValue();
        this.tolerance = ((Number) p.get("tolerance")).doubleValue();
        this.numRestarts = ((Number) p.get("numRestarts")).intValue();
        this.useKMeansInit = (Boolean) p.get("useKMeansInit");
        if (p.containsKey("verbose")) this.verbose = (Boolean) p.get("verbose");
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
        // Reconstruct GMM
        List<Map<String, Object>> compList = (List<Map<String, Object>>) p.get("components");
        if (compList != null && !compList.isEmpty()) {
            List<com.yishape.lab.math.stats.distribution.multiv.MultivariateNormalDistribution> components = new ArrayList<>();
            List<Double> weights = new ArrayList<>();
            for (Map<String, Object> cm : compList) {
                double[] mean = (double[]) cm.get("mean");
                double[][] cov = (double[][]) cm.get("covariance");
                double weight = ((Number) cm.get("weight")).doubleValue();
                components.add(new com.yishape.lab.math.stats.distribution.multiv.MultivariateNormalDistribution(
                        Linalg.vector(mean), Linalg.matrix(cov)));
                weights.add(weight);
            }
            this.trainedModel = new com.yishape.lab.math.stats.model.GaussianMixtureModel(components, weights);
        }
    }

    private void log(String msg) {
        com.yishape.lab.util.YishapeLogger.getLogger(GMMClustering.class).info(msg);
    }
}
