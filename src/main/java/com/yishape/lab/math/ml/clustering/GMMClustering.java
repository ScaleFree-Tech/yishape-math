package com.yishape.lab.math.ml.clustering;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.model.GaussianMixtureModel;
import com.yishape.lab.math.stats.model.EMAlgorithm;
import com.yishape.lab.math.stats.distribution.multiv.MultivariateNormalDistribution;

import java.io.*;
import java.util.*;

/**
 * 基于高斯混合模型的聚类算法
 * Gaussian Mixture Model (GMM) based clustering algorithm
 * 
 * 使用EM算法训练高斯混合模型进行聚类
 * Uses EM algorithm to train Gaussian mixture model for clustering
 * 
 * @author reremouse
 */
public class GMMClustering implements IClustering, ISerializableModel {
    
    private static final long serialVersionUID = 1L;
    
    // 默认参数
    private static final int DEFAULT_MAX_ITERATIONS = 100;
    private static final double DEFAULT_TOLERANCE = 1e-6;
    private static final int DEFAULT_NUM_RESTARTS = 10;
    private static final boolean DEFAULT_USE_KMEANS_INIT = true;
    private static final long DEFAULT_RANDOM_SEED = 42L;
    
    // 算法参数
    private int maxIterations;
    private double tolerance;
    private int numRestarts;
    private boolean useKMeansInit;
    private long randomSeed;
    private boolean verbose;
    
    // 内部状态
    private Random random;
    private EMAlgorithm emAlgorithm;
    private GaussianMixtureModel trainedModel;
    private List<IVector<Double>> clusterCenters;
    private int[] labels;
    private double inertia;
    private boolean converged;
    private int iterations;
    private int dimension;
    private int numClusters;
    private Map<String, Object> parameters = new HashMap<>();
    
    /**
     * 默认构造函数
     * Default constructor
     */
    public GMMClustering() {
        this(DEFAULT_MAX_ITERATIONS, DEFAULT_TOLERANCE, DEFAULT_NUM_RESTARTS, 
             DEFAULT_USE_KMEANS_INIT, DEFAULT_RANDOM_SEED, false);
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param maxIterations 最大迭代次数 / Maximum iterations
     * @param tolerance 收敛容忍度 / Convergence tolerance
     * @param numRestarts 重启次数 / Number of restarts
     * @param useKMeansInit 是否使用K-means++初始化 / Whether to use K-means++ initialization
     * @param randomSeed 随机种子 / Random seed
     * @param verbose 是否输出详细信息 / Whether to output verbose information
     */
    public GMMClustering(int maxIterations, double tolerance, int numRestarts, 
                        boolean useKMeansInit, long randomSeed, boolean verbose) {
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.numRestarts = numRestarts;
        this.useKMeansInit = useKMeansInit;
        this.randomSeed = randomSeed;
        this.verbose = verbose;
        
        this.random = new Random(randomSeed);
        this.emAlgorithm = new EMAlgorithm(maxIterations, tolerance, verbose);
    }
    
    @Override
    public IClustering fit(List<IVector<Double>> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("数据不能为空");
        }
        
        // 确定聚类数量（如果未设置，则默认为3）
        if (this.numClusters <= 0) {
            this.numClusters = 3;
        }
        
        if (this.numClusters > data.size()) {
            throw new IllegalArgumentException("聚类数量必须小于等于数据点数量");
        }
        
        this.dimension = data.get(0).size();
        
        // 验证数据维度一致性
        for (IVector<Double> point : data) {
            if (point.size() != dimension) {
                throw new IllegalArgumentException("所有数据点必须具有相同的维度");
            }
        }
        
        if (verbose) {
            System.out.printf("开始GMM聚类: %d个数据点, %d维, %d个聚类\n", 
                            data.size(), dimension, numClusters);
        }
        
        // 多重启动策略
        int numRestarts = 10;  // 增加重启次数
        
        // 使用多重启动策略训练高斯混合模型
        GaussianMixtureModel bestGmm = null;
        EMAlgorithm.EMResult bestResult = null;
        double bestLogLikelihood = Double.NEGATIVE_INFINITY;
        
        // 备选方案：如果没有有效聚类，保存最好的无效聚类
        GaussianMixtureModel fallbackGmm = null;
        EMAlgorithm.EMResult fallbackResult = null;
        double fallbackLogLikelihood = Double.NEGATIVE_INFINITY;
        
        if (verbose) {
            System.out.printf("开始多重启动策略，计划进行 %d 次重启...\n", numRestarts);
        }
        
        for (int restart = 0; restart < numRestarts; restart++) {
            try {
                // 为每次重启创建新的随机种子
                Random restartRandom = new Random(random.nextLong());
                
                // 创建新的GMM实例
                GaussianMixtureModel gmm = new GaussianMixtureModel(numClusters, dimension, restartRandom);
                
                // 使用更多样化和鲁棒的初始化策略
                switch (restart % 4) {
                    case 0:
                        // K-means++初始化（优先使用）
                        try {
                            gmm.initializeWithKMeansPlusPlus(data);
                            if (verbose) {
                                System.out.printf("重启 %d: 使用K-means++初始化\n", restart + 1);
                            }
                        } catch (Exception e) {
                            // 如果K-means++失败，回退到智能随机初始化
                            gmm.initializeWithSmartRandom(data);
                            if (verbose) {
                                System.out.printf("重启 %d: K-means++失败，回退到智能随机初始化\n", restart + 1);
                            }
                        }
                        break;
                    case 1:
                        // 智能随机初始化
                        gmm.initializeWithSmartRandom(data);
                        if (verbose) {
                            System.out.printf("重启 %d: 使用智能随机初始化\n", restart + 1);
                        }
                        break;
                    case 2:
                        // 数据驱动初始化（基于数据分布）
                        try {
                            initializeWithDataDrivenStrategy(gmm, data);
                            if (verbose) {
                                System.out.printf("重启 %d: 使用数据驱动初始化\n", restart + 1);
                            }
                        } catch (Exception e) {
                            // 回退到智能随机初始化
                            gmm.initializeWithSmartRandom(data);
                            if (verbose) {
                                System.out.printf("重启 %d: 数据驱动初始化失败，回退到智能随机初始化\n", restart + 1);
                            }
                        }
                        break;
                    case 3:
                        // 随机初始化（作为基准）
                        gmm.initializeRandomly(data);
                        if (verbose) {
                            System.out.printf("重启 %d: 使用随机初始化\n", restart + 1);
                        }
                        break;
                }
                
                // 训练模型
                EMAlgorithm.EMResult result = emAlgorithm.fit(data, gmm);
                
                if (result != null && Double.isFinite(result.logLikelihood)) {
                    // 验证聚类结果是否有效
                    int[] tempLabels = new int[data.size()];
                    for (int i = 0; i < data.size(); i++) {
                        tempLabels[i] = gmm.predictComponent(data.get(i));
                    }
                    
                    boolean isValid = isValidClustering(tempLabels, numClusters);
                    
                    if (isValid) {
                        // 有效聚类
                        if (result.logLikelihood > bestLogLikelihood) {
                            bestLogLikelihood = result.logLikelihood;
                            bestResult = result;
                            bestGmm = gmm;
                            
                            if (verbose) {
                                System.out.printf("重启 %d: 找到更好的有效解，对数似然 = %.6f\n", 
                                                restart + 1, result.logLikelihood);
                            }
                        } else if (verbose) {
                            System.out.printf("重启 %d: 有效解，对数似然 = %.6f (当前最佳: %.6f)\n", 
                                            restart + 1, result.logLikelihood, bestLogLikelihood);
                        }
                    } else {
                        // 无效聚类，作为备选方案
                        if (result.logLikelihood > fallbackLogLikelihood) {
                            fallbackLogLikelihood = result.logLikelihood;
                            fallbackResult = result;
                            fallbackGmm = gmm;
                        }
                        
                        if (verbose) {
                            System.out.printf("重启 %d: 产生无效聚类（存在空聚类），对数似然 = %.6f\n", 
                                            restart + 1, result.logLikelihood);
                        }
                    }
                }
            } catch (Exception e) {
                if (verbose) {
                    System.out.printf("重启 %d 失败: %s\n", restart + 1, e.getMessage());
                }
            }
        }
        
        // 选择最终模型
        GaussianMixtureModel finalGmm;
        EMAlgorithm.EMResult finalResult;
        
        if (bestGmm != null) {
            // 有有效聚类
            finalGmm = bestGmm;
            finalResult = bestResult;
            if (verbose) {
                System.out.printf("多重启动完成，选择有效聚类，对数似然 = %.6f\n", bestLogLikelihood);
            }
        } else if (fallbackGmm != null) {
            // 没有有效聚类，使用备选方案
            finalGmm = fallbackGmm;
            finalResult = fallbackResult;
            if (verbose) {
                System.out.printf("多重启动完成，所有聚类都无效，选择最佳备选方案，对数似然 = %.6f\n", fallbackLogLikelihood);
                System.out.println("警告：选择的聚类可能包含空聚类");
            }
        } else {
             // 最后的备选方案：强制创建有效聚类
             if (verbose) {
                 System.out.println("所有重启都失败，使用强制有效聚类策略");
             }
             
             // 使用K-means算法作为强制策略
             KMeansPlusPlus kmeans = new KMeansPlusPlus(randomSeed);
             kmeans.fit(data);
             
             // 保存结果
             this.clusterCenters = kmeans.getClusterCenters();
             this.labels = kmeans.getLabels();
             this.inertia = kmeans.getInertia();
             this.converged = kmeans.isConverged();
             this.iterations = kmeans.getIterations();
             this.trainedModel = null;
             
             return this;
         }
        
        // 使用最终模型
        GaussianMixtureModel gmm = finalGmm;
        EMAlgorithm.EMResult emResult = finalResult;
        
        // 保存训练好的模型
        this.trainedModel = gmm;
        
        // 预测聚类标签
        int[] labels = new int[data.size()];
        for (int i = 0; i < data.size(); i++) {
            labels[i] = gmm.predictComponent(data.get(i));
        }
        
        // 提取聚类中心（高斯分量的均值）
        List<IVector<Double>> centers = new ArrayList<>();
        for (int k = 0; k < numClusters; k++) {
            centers.add(gmm.getComponent(k).getMean().copy());
        }
        
        // 计算惯性（使用高斯分量的均值作为中心）
        double inertia = computeInertia(data, centers, labels);
        
        // 保存结果
        this.clusterCenters = centers;
        this.labels = labels;
        this.inertia = inertia;
        this.converged = emResult.converged;
        this.iterations = emResult.iterations;
        
        return this;
    }
    
    @Override
    public IClustering fit(IMatrix<Double> data) {
        // 转换矩阵为向量列表
        List<IVector<Double>> dataList = convertMatrixToVectorList(data);
        return fit(dataList);
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
        if (trainedModel == null && clusterCenters == null) {
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
    public int predict(IVector<Double> point) {
        if (trainedModel == null && clusterCenters == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        if (point == null) {
            throw new IllegalArgumentException("数据点不能为空");
        }
        
        if (trainedModel != null) {
            // 使用GMM模型进行预测
            return trainedModel.predictComponent(point);
        } else {
            // 使用最近邻分类（简化版本，实际GMM应该使用概率分配）
            double minDistance = Double.POSITIVE_INFINITY;
            int bestCluster = 0;
            
            for (int i = 0; i < clusterCenters.size(); i++) {
                double distance = point.euclideanDistance(clusterCenters.get(i));
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = i;
                }
            }
            
            return bestCluster;
        }
    }
    
    @Override
    public List<IVector<Double>> getClusterCenters() {
        return clusterCenters;
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
//    
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
    
    @Override
    public void setParameters(Map<String, Object> parameters) {
        if (parameters == null) return;
        
        this.parameters.putAll(parameters);
        
        if (parameters.containsKey("numClusters")) {
            this.numClusters = (Integer) parameters.get("numClusters");
        }
        if (parameters.containsKey("maxIterations")) {
            this.maxIterations = (Integer) parameters.get("maxIterations");
        }
        if (parameters.containsKey("tolerance")) {
            this.tolerance = (Double) parameters.get("tolerance");
        }
        if (parameters.containsKey("numRestarts")) {
            this.numRestarts = (Integer) parameters.get("numRestarts");
        }
        if (parameters.containsKey("useKMeansInit")) {
            this.useKMeansInit = (Boolean) parameters.get("useKMeansInit");
        }
        if (parameters.containsKey("randomSeed")) {
            this.randomSeed = (Long) parameters.get("randomSeed");
            this.random = new Random(randomSeed);
        }
        if (parameters.containsKey("verbose")) {
            this.verbose = (Boolean) parameters.get("verbose");
        }
        
        // 重新创建EM算法实例
        this.emAlgorithm = new EMAlgorithm(maxIterations, tolerance, verbose);
    }
    
    @Override
    public Map<String, Object> getParameters() {
        Map<String, Object> params = new HashMap<>(parameters);
        params.put("numClusters", numClusters);
        params.put("maxIterations", maxIterations);
        params.put("tolerance", tolerance);
        params.put("convergenceThreshold", tolerance); // Alias for consistency with K-Means
        params.put("numRestarts", numRestarts);
        params.put("useKMeansInit", useKMeansInit);
        params.put("randomSeed", randomSeed);
        params.put("verbose", verbose);
        params.put("algorithmName", getAlgorithmName());
        return params;
    }
    /**
     * 获取训练好的高斯混合模型
     * Get the trained Gaussian mixture model
     * 
     * @return 高斯混合模型 / Gaussian mixture model
     */
    public GaussianMixtureModel getTrainedModel() {
        return trainedModel;
    }
    
    /**
     * 计算数据点属于各个分量的后验概率
     * Compute posterior probabilities of data points belonging to each component
     * 
     * @param data 数据点 / Data points
     * @return 后验概率矩阵 / Posterior probability matrix
     */
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
    
    /**
     * 计算数据的对数似然
     * Compute log-likelihood of data
     * 
     * @param data 数据点 / Data points
     * @return 对数似然 / Log-likelihood
     */
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
    
    /**
     * 从训练好的模型中采样
     * Sample from the trained model
     * 
     * @param numSamples 采样数量 / Number of samples
     * @return 采样数据 / Sampled data
     */
    public List<IVector<Double>> sample(int numSamples) {
        if (trainedModel == null) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        return trainedModel.sample(numSamples);
    }
    
    /**
     * 计算惯性（数据点到聚类中心的距离平方和）
     * Compute inertia (sum of squared distances from data points to cluster centers)
     */
    private double computeInertia(List<IVector<Double>> data, List<IVector<Double>> centers, int[] labels) {
        double inertia = 0.0;
        for (int i = 0; i < data.size(); i++) {
            IVector<Double> point = data.get(i);
            IVector<Double> center = centers.get(labels[i]);
            double distance = point.euclideanDistance(center);
            inertia += distance * distance;
        }
        return inertia;
    }
    
    /**
     * 将矩阵转换为向量列表
     * Convert matrix to vector list
     */
    private List<IVector<Double>> convertMatrixToVectorList(IMatrix<Double> matrix) {
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
     * 数据驱动的初始化策略
     * 基于数据的统计特性和分布来初始化GMM
     */
    private void initializeWithDataDrivenStrategy(GaussianMixtureModel gmm, List<IVector<Double>> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        
        int n = data.size();
        int d = data.get(0).size();
        int numClusters = gmm.getNumComponents();
        
        // 计算数据的统计信息
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
        
        // 使用主成分分析的思想来分散初始化中心
        List<IVector<Double>> centers = new ArrayList<>();
        
        // 第一个中心：数据均值附近的随机点
        IVector<Double> firstCenter = sampleFromGaussian(mean, covariance, 0.1);
        centers.add(firstCenter);
        
        // 后续中心：沿着主要变化方向分散
        for (int k = 1; k < numClusters; k++) {
            IVector<Double> newCenter = null;
            double maxMinDistance = 0;
            
            // 尝试多个候选点，选择距离现有中心最远的
            for (int trial = 0; trial < 50; trial++) {
                IVector<Double> candidate = sampleFromGaussian(mean, covariance, 0.5);
                
                // 计算到最近现有中心的距离
                double minDistance = Double.MAX_VALUE;
                for (IVector<Double> center : centers) {
                    double distance = candidate.euclideanDistance(center);
                    minDistance = Math.min(minDistance, distance);
                }
                
                if (minDistance > maxMinDistance) {
                    maxMinDistance = minDistance;
                    newCenter = candidate;
                }
            }
            
            if (newCenter != null) {
                centers.add(newCenter);
            } else {
                // 回退：在数据范围内随机选择
                centers.add(data.get(random.nextInt(data.size())));
            }
        }
        
        // 初始化GMM分量
         double weight = 1.0 / numClusters;
         for (int k = 0; k < numClusters; k++) {
             IVector<Double> center = centers.get(k);
             
             // 使用缩放的全局协方差作为初始协方差
             IMatrix<Double> componentCovariance = Linalg.zeros(d, d);
             for (int i = 0; i < d; i++) {
                 for (int j = 0; j < d; j++) {
                     componentCovariance.set(i, j, covariance.get(i, j) * 0.5 / numClusters);
                 }
             }
             
             // 确保协方差矩阵是正定的
             for (int i = 0; i < d; i++) {
                 componentCovariance.set(i, i, Math.max(componentCovariance.get(i, i), 1e-6));
             }
             
             // 创建MultivariateNormalDistribution并设置分量
             MultivariateNormalDistribution component = new MultivariateNormalDistribution(center, componentCovariance);
             gmm.setComponent(k, component);
             gmm.setWeight(k, weight);
         }
    }
    
    /**
      * 从多元高斯分布中采样
      */
     private IVector<Double> sampleFromGaussian(IVector<Double> mean, IMatrix<Double> covariance, double scale) {
         int d = mean.size();
         IVector<Double> sample = Linalg.zeros(d);
         
         // 简化的采样：使用对角协方差的近似
         for (int i = 0; i < d; i++) {
             double variance = Math.max(covariance.get(i, i) * scale, 1e-6);
             double stddev = Math.sqrt(variance);
             sample.set(i, mean.get(i) + random.nextGaussian() * stddev);
         }
         
         return sample;
     }
     

    /**
     * 验证聚类结果是否有效
     * Validate clustering result
     */
    private boolean isValidClustering(int[] labels, int numClusters) {
        if (labels == null || labels.length == 0) {
            return false;
        }
        
        int[] clusterCounts = new int[numClusters];
        int validLabels = 0;
        
        // 统计每个聚类的数据点数量
        for (int label : labels) {
            if (label >= 0 && label < numClusters) {
                clusterCounts[label]++;
                validLabels++;
            }
        }
        
        // 检查是否所有标签都有效
        if (validLabels != labels.length) {
            return false;
        }
        
        // 检查是否有空聚类
        for (int count : clusterCounts) {
            if (count == 0) {
                return false;
            }
        }
        
        // 检查聚类分布是否合理（每个聚类至少有2个数据点）
        // 这有助于避免退化的聚类结果
        int minPointsPerCluster = Math.max(1, labels.length / (numClusters * 10));
        for (int count : clusterCounts) {
            if (count < minPointsPerCluster) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * 将模型保存在本地
     * @param path 保存路径
     */
    @Override
    public void save(String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
}