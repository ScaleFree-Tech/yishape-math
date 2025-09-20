package com.reremouse.lab.math.stats.model;

import com.reremouse.lab.math.stats.distribution.multiv.MultivariateNormalDistribution;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.ml.clustering.KMeansPlusPlus;
import com.reremouse.lab.util.Tuple2;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * 高斯混合模型(GMM)实现
 * Gaussian Mixture Model (GMM) Implementation
 * 
 * 支持多个高斯分量的混合模型，用于聚类和密度估计
 * Supports mixture of multiple Gaussian components for clustering and density estimation
 */
public class GaussianMixtureModel {
    
    /** 高斯分量列表 / List of Gaussian components */
    private final List<MultivariateNormalDistribution> components;
    
    /** 混合权重 / Mixture weights */
    private final List<Double> weights;
    
    /** 分量数量 / Number of components */
    private final int numComponents;
    
    /** 数据维度 / Data dimensionality */
    private final int dimension;
    
    /** 随机数生成器 / Random number generator */
    private final Random random;
    
    /**
     * 构造函数
     * @param numComponents 分量数量
     * @param dimension 数据维度
     */
    public GaussianMixtureModel(int numComponents, int dimension) {
        this(numComponents, dimension, new Random(42));
    }
    
    /**
     * 构造函数（带随机种子）
     * @param numComponents 分量数量
     * @param dimension 数据维度
     * @param random 随机数生成器
     */
    public GaussianMixtureModel(int numComponents, int dimension, Random random) {
        if (numComponents <= 0 || dimension <= 0) {
            throw new IllegalArgumentException("分量数量和维度必须大于0");
        }
        
        this.numComponents = numComponents;
        this.dimension = dimension;
        this.random = random;
        this.components = new ArrayList<>(numComponents);
        this.weights = new ArrayList<>(numComponents);
        
        // 初始化分量和权重
        initializeComponents();
    }
    
    /**
     * 从已有的分量和权重构造GMM
     * @param components 高斯分量列表
     * @param weights 权重列表
     */
    public GaussianMixtureModel(List<MultivariateNormalDistribution> components, List<Double> weights) {
        if (components.isEmpty() || weights.isEmpty() || components.size() != weights.size()) {
            throw new IllegalArgumentException("分量和权重数量必须匹配且不能为空");
        }
        
        this.numComponents = components.size();
        this.dimension = components.get(0).getDimension();
        this.random = new Random(42);
        this.components = new ArrayList<>(components);
        this.weights = new ArrayList<>(weights);
        
        // 验证维度一致性
        for (MultivariateNormalDistribution component : components) {
            if (component.getDimension() != dimension) {
                throw new IllegalArgumentException("所有分量必须具有相同的维度");
            }
        }
        
        // 归一化权重
        normalizeWeights();
    }
    
    /**
     * 初始化分量
     */
    private void initializeComponents() {
        // 简单的权重初始化：均匀分布加小扰动
        for (int i = 0; i < numComponents; i++) {
            double baseWeight = 1.0 / numComponents;
            double perturbation = (random.nextDouble() - 0.5) * 0.1; // ±5%的扰动
            double weight = Math.max(0.01, baseWeight + perturbation);
            weights.add(weight);
        }
        
        // 归一化权重
        normalizeWeights();
        
        // 改进的分量初始化：使用数据驱动的方法
        // 先创建一个单位协方差矩阵作为默认值
        IMatrix<Double> defaultCovariance = Linalg.eye(dimension);
        
        // 随机初始化分量
        for (int i = 0; i < numComponents; i++) {
            // 随机均值
            IVector<Double> mean = Linalg.zeros(dimension);
            for (int d = 0; d < dimension; d++) {
                mean.set(d, random.nextGaussian());
            }
            
            // 使用改进的协方差矩阵初始化
            IMatrix<Double> covariance = generateRandomCovarianceMatrix();
            
            components.add(new MultivariateNormalDistribution(mean, covariance));
        }
    }
    
    /**
     * 生成随机协方差矩阵（确保正定性）
     * @return 正定的协方差矩阵
     */
    private IMatrix<Double> generateRandomCovarianceMatrix() {
        // 创建对称正定矩阵的更稳健方法
        // 生成对角占优的对称矩阵，确保正定性
        
        IMatrix<Double> matrix = Linalg.zeros(dimension, dimension);
        
        // 填充对称矩阵
        for (int i = 0; i < dimension; i++) {
            for (int j = i; j < dimension; j++) {
                if (i == j) {
                    // 对角线元素：确保为正数
                    double value = Math.abs(random.nextGaussian()) + 1.0;
                    matrix.set(i, j, value);
                } else {
                    // 非对角线元素
                    double value = random.nextGaussian() * 0.5;
                    matrix.set(i, j, value);
                    matrix.set(j, i, value);
                }
            }
        }
        
        // 确保对角占优以保证正定性
        for (int i = 0; i < dimension; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < dimension; j++) {
                if (i != j) {
                    rowSum += Math.abs(matrix.get(i, j));
                }
            }
            // 确保对角线元素大于该行其他元素绝对值之和
            double diagonalValue = matrix.get(i, i);
            if (diagonalValue <= rowSum) {
                matrix.set(i, i, rowSum + Math.abs(random.nextGaussian()) + 1.0);
            }
        }
        
        // 添加小的正数到对角线确保数值稳定性
        double epsilon = 1e-6;
        for (int i = 0; i < dimension; i++) {
            matrix.set(i, i, matrix.get(i, i) + epsilon);
        }
        
        return matrix;
    }
    
    /**
     * 使用K-means++算法初始化分量
     * @param data 训练数据
     */
    public void initializeWithKMeansPlusPlus(List<IVector<Double>> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        
        if (data.get(0).size() != dimension) {
            throw new IllegalArgumentException("数据维度与模型维度不匹配");
        }
        
        // 使用新的K-means++实现
        KMeansPlusPlus kMeansPlusPlus = new KMeansPlusPlus(random, 100, 1e-6);
        List<IVector<Double>> centers = kMeansPlusPlus.initializeCenters(data, numComponents);
        
        // 计算数据驱动的协方差矩阵
        IMatrix<Double> globalCovariance = computeGlobalCovariance(data, centers);
        
        // 使用选择的中心和数据驱动的协方差初始化分量
        components.clear();
        for (int i = 0; i < numComponents; i++) {
            // 为每个分量计算局部协方差矩阵
            IMatrix<Double> localCovariance = computeLocalCovariance(data, centers.get(i), globalCovariance);
            components.add(new MultivariateNormalDistribution(centers.get(i), localCovariance));
        }
        
        // 重新初始化权重
        weights.clear();
        double uniformWeight = 1.0 / numComponents;
        for (int i = 0; i < numComponents; i++) {
            weights.add(uniformWeight);
        }
    }
    
    /**
     * 随机初始化模型参数
     * @param data 训练数据
     */
    public void initializeRandomly(List<IVector<Double>> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        
        if (data.get(0).size() != dimension) {
            throw new IllegalArgumentException("数据维度与模型维度不匹配");
        }
        
        // 清空现有分量
        components.clear();
        weights.clear();
        
        // 使用改进的智能随机初始化策略
        initializeWithSmartRandom(data);
    }
    
    /**
     * 智能随机初始化策略 - 使用改进的分散策略
     * 结合了随机性和数据分布特征，提供比纯随机更好的初始化
     */
    public void initializeWithSmartRandom(List<IVector<Double>> data) {
        if (data.isEmpty()) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        
        // 清空现有分量和权重
        components.clear();
        weights.clear();
        
        // 计算数据的统计信息
        IVector<Double> dataMean = computeDataMean(data);
        IMatrix<Double> dataCovariance = computeDataCovariance(data, dataMean);
        
        // 计算数据的范围，用于更好的初始化
        double[] minValues = new double[dimension];
        double[] maxValues = new double[dimension];
        Arrays.fill(minValues, Double.POSITIVE_INFINITY);
        Arrays.fill(maxValues, Double.NEGATIVE_INFINITY);
        
        for (IVector<Double> point : data) {
            for (int d = 0; d < dimension; d++) {
                double value = point.get(d);
                minValues[d] = Math.min(minValues[d], value);
                maxValues[d] = Math.max(maxValues[d], value);
            }
        }
        
        // 均匀初始化权重
        for (int i = 0; i < numComponents; i++) {
            weights.add(1.0 / numComponents);
        }
        
        // 使用改进的分散初始化策略
        List<IVector<Double>> selectedCenters = new ArrayList<>();
        Set<Integer> usedIndices = new HashSet<>();
        
        // 第一个中心：选择距离数据中心较远的点
        double maxDistFromCenter = 0.0;
        int firstIndex = 0;
        for (int i = 0; i < data.size(); i++) {
            double dist = computeSquaredDistance(data.get(i), dataMean);
            if (dist > maxDistFromCenter) {
                maxDistFromCenter = dist;
                firstIndex = i;
            }
        }
        selectedCenters.add(data.get(firstIndex).copy());
        usedIndices.add(firstIndex);
        
        // 后续中心：使用改进的K-means++策略
        for (int i = 1; i < numComponents; i++) {
            double[] distances = new double[data.size()];
            double totalDistance = 0.0;
            
            // 计算每个数据点到最近已选中心的距离
            for (int j = 0; j < data.size(); j++) {
                if (usedIndices.contains(j)) {
                    distances[j] = 0.0;
                    continue;
                }
                
                double minDist = Double.POSITIVE_INFINITY;
                for (IVector<Double> center : selectedCenters) {
                    double dist = computeSquaredDistance(data.get(j), center);
                    minDist = Math.min(minDist, dist);
                }
                distances[j] = minDist;
                totalDistance += minDist;
            }
            
            // 如果总距离太小，使用随机选择
            if (totalDistance < 1e-10) {
                int selectedIndex;
                do {
                    selectedIndex = random.nextInt(data.size());
                } while (usedIndices.contains(selectedIndex));
                selectedCenters.add(data.get(selectedIndex).copy());
                usedIndices.add(selectedIndex);
            } else {
                // 使用距离权重随机选择下一个中心
                double threshold = random.nextDouble() * totalDistance;
                double cumulative = 0.0;
                int selectedIndex = 0;
                
                for (int j = 0; j < data.size(); j++) {
                    cumulative += distances[j];
                    if (cumulative >= threshold) {
                        selectedIndex = j;
                        break;
                    }
                }
                
                selectedCenters.add(data.get(selectedIndex).copy());
                usedIndices.add(selectedIndex);
            }
        }
        
        // 为每个选定的中心创建分量
        for (int i = 0; i < numComponents; i++) {
            IVector<Double> mean = selectedCenters.get(i);
            
            // 初始化协方差矩阵：使用数据驱动的策略
            IMatrix<Double> covariance = Linalg.zeros(dimension, dimension);
            
            // 计算局部协方差矩阵
            double scaleFactor = 0.5; // 适中的缩放因子
            
            // 使用对角协方差矩阵，基于数据范围
            for (int d = 0; d < dimension; d++) {
                double range = maxValues[d] - minValues[d];
                double variance = Math.max(range * range * scaleFactor / (numComponents * numComponents), 1e-6);
                covariance.set(d, d, variance);
            }
            
            components.add(new MultivariateNormalDistribution(mean, covariance));
        }
    }
    
    /**
     * 计算两个向量之间的平方距离
     */
    private double computeSquaredDistance(IVector<Double> v1, IVector<Double> v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.size(); i++) {
            double diff = v1.get(i) - v2.get(i);
            sum += diff * diff;
        }
        return sum;
    }
    
    /**
     * 计算数据均值
     */
    private IVector<Double> computeDataMean(List<IVector<Double>> data) {
        IVector<Double> mean = Linalg.zeros(dimension);
        for (IVector<Double> sample : data) {
            for (int d = 0; d < dimension; d++) {
                mean.set(d, mean.get(d) + sample.get(d));
            }
        }
        for (int d = 0; d < dimension; d++) {
            mean.set(d, mean.get(d) / data.size());
        }
        return mean;
    }
    
    /**
     * 计算数据协方差矩阵
     */
    private IMatrix<Double> computeDataCovariance(List<IVector<Double>> data, IVector<Double> mean) {
        IMatrix<Double> covariance = Linalg.zeros(dimension, dimension);
        
        for (IVector<Double> sample : data) {
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    double diff_i = sample.get(i) - mean.get(i);
                    double diff_j = sample.get(j) - mean.get(j);
                    covariance.set(i, j, covariance.get(i, j) + diff_i * diff_j);
                }
            }
        }
        
        // 归一化
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                covariance.set(i, j, covariance.get(i, j) / (data.size() - 1));
            }
        }
        
        return covariance;
    }
    
    /**
     * 使用EM算法训练模型
     * @param data 训练数据
     * @return EM算法结果
     */
    public EMAlgorithm.EMResult fit(List<IVector<Double>> data) {
        return fit(data, new EMAlgorithm());
    }
    
    /**
     * 使用指定的EM算法训练模型
     * @param data 训练数据
     * @param emAlgorithm EM算法实例
     * @return EM算法结果
     */
    public EMAlgorithm.EMResult fit(List<IVector<Double>> data, EMAlgorithm emAlgorithm) {
        // 使用K-means++初始化
        initializeWithKMeansPlusPlus(data);
        
        // 执行EM算法
        return emAlgorithm.fit(data, this);
    }
    
    /**
     * 计算样本的概率密度
     * @param x 输入向量
     * @return 概率密度值
     */
    public double pdf(IVector<Double> x) {
        if (x.size() != dimension) {
            throw new IllegalArgumentException("输入向量维度不匹配");
        }
        
        double density = 0.0;
        for (int k = 0; k < numComponents; k++) {
            density += weights.get(k) * components.get(k).pdf(x);
        }
        return density;
    }
    
    /**
     * 计算样本的对数概率密度（数值稳定版本）
     * @param x 输入向量
     * @return 对数概率密度值
     */
    public double logPdf(IVector<Double> x) {
        if (x.size() != dimension) {
            throw new IllegalArgumentException("输入向量维度不匹配");
        }
        
        // 使用数值稳定的log-sum-exp技巧计算对数概率密度
        double[] logComponentDensities = new double[numComponents];
        double maxLogDensity = Double.NEGATIVE_INFINITY;
        
        // 计算每个分量的对数密度并找到最大值
        for (int k = 0; k < numComponents; k++) {
            logComponentDensities[k] = Math.log(Math.max(weights.get(k), 1e-15)) + 
                                     components.get(k).logPdf(x);
            maxLogDensity = Math.max(maxLogDensity, logComponentDensities[k]);
        }
        
        // 使用log-sum-exp技巧
        double sumExp = 0.0;
        for (int k = 0; k < numComponents; k++) {
            sumExp += Math.exp(logComponentDensities[k] - maxLogDensity);
        }
        
        return maxLogDensity + Math.log(sumExp);
    }
    
    /**
     * 计算样本属于各个分量的后验概率
     * @param x 输入向量
     * @return 后验概率向量
     */
    public IVector<Double> computePosteriors(IVector<Double> x) {
        if (x.size() != dimension) {
            throw new IllegalArgumentException("输入向量维度不匹配");
        }
        
        IVector<Double> posteriors = Linalg.zeros(numComponents);
        double totalWeightedLikelihood = 0.0;
        
        // 计算加权似然
        for (int k = 0; k < numComponents; k++) {
            double weightedLikelihood = weights.get(k) * components.get(k).pdf(x);
            posteriors.set(k, weightedLikelihood);
            totalWeightedLikelihood += weightedLikelihood;
        }
        
        // 归一化
        if (totalWeightedLikelihood > 0) {
            posteriors = posteriors.divideByScalar(totalWeightedLikelihood);
        } else {
            // 处理数值下溢，使用均匀分布
            double uniformPosterior = 1.0 / numComponents;
            for (int k = 0; k < numComponents; k++) {
                posteriors.set(k, uniformPosterior);
            }
        }
        
        return posteriors;
    }
    
    /**
     * 预测样本属于各个分量的后验概率
     * @param x 输入向量
     * @return 后验概率向量
     */
    public IVector<Double> predict(IVector<Double> x) {
        return computePosteriors(x);
    }
    
    /**
     * 计算样本的概率密度（不归一化的后验概率）
     * @param x 输入向量
     * @return 未归一化的后验概率向量
     */
    public IVector<Double> computeUnnormalizedPosteriors(IVector<Double> x) {
        if (x.size() != dimension) {
            throw new IllegalArgumentException("输入向量维度不匹配");
        }
        
        IVector<Double> posteriors = Linalg.zeros(numComponents);
        
        // 计算加权似然（不归一化）
        for (int k = 0; k < numComponents; k++) {
            double weightedLikelihood = weights.get(k) * components.get(k).pdf(x);
            posteriors.set(k, weightedLikelihood);
        }
        
        return posteriors;
    }
    
    /**
     * 预测样本最可能属于的分量
     * @param x 输入向量
     * @return 分量索引
     */
    public int predictComponent(IVector<Double> x) {
        IVector<Double> posteriors = predict(x);
        int maxIndex = 0;
        double maxPosterior = posteriors.get(0);
        
        for (int k = 1; k < numComponents; k++) {
            if (posteriors.get(k) > maxPosterior) {
                maxPosterior = posteriors.get(k);
                maxIndex = k;
            }
        }
        
        return maxIndex;
    }
    
    /**
     * 从模型中采样
     * @param numSamples 采样数量
     * @return 采样结果列表
     */
    public List<IVector<Double>> sample(int numSamples) {
        List<IVector<Double>> samples = new ArrayList<>();
        
        for (int i = 0; i < numSamples; i++) {
            // 根据权重选择分量
            double rand = random.nextDouble();
            double cumWeight = 0.0;
            int selectedComponent = 0;
            
            for (int k = 0; k < numComponents; k++) {
                cumWeight += weights.get(k);
                if (rand <= cumWeight) {
                    selectedComponent = k;
                    break;
                }
            }
            
            // 从选择的分量中采样（使用正确的多元高斯采样）
            MultivariateNormalDistribution component = components.get(selectedComponent);
            IVector<Double> sample = component.sample();
            
            samples.add(sample);
        }
        
        return samples;
    }
    
    /**
     * 归一化权重
     */
    private void normalizeWeights() {
        double totalWeight = weights.stream().mapToDouble(Double::doubleValue).sum();
        if (totalWeight > 1e-12) {
            for (int i = 0; i < weights.size(); i++) {
                weights.set(i, weights.get(i) / totalWeight);
            }
        } else {
            // 如果总权重过小，重置为均匀分布
            double uniformWeight = 1.0 / numComponents;
            for (int i = 0; i < weights.size(); i++) {
                weights.set(i, uniformWeight);
            }
        }
    }
    
    /**
     * 计算欧几里得距离
     */
    private double euclideanDistance(IVector<Double> v1, IVector<Double> v2) {
        double sum = 0.0;
        for (int i = 0; i < v1.size(); i++) {
            double diff = v1.get(i) - v2.get(i);
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * 计算全局协方差矩阵（改进版，更稳定）
     */
    private IMatrix<Double> computeGlobalCovariance(List<IVector<Double>> data, List<IVector<Double>> centers) {
        // 计算数据的全局均值
        IVector<Double> globalMean = Linalg.zeros(dimension);
        for (IVector<Double> point : data) {
            for (int i = 0; i < dimension; i++) {
                globalMean.set(i, globalMean.get(i) + point.get(i));
            }
        }
        for (int i = 0; i < dimension; i++) {
            globalMean.set(i, globalMean.get(i) / data.size());
        }
        
        // 计算数据的方差范围，用于自适应正则化
        double[] variances = new double[dimension];
        for (int d = 0; d < dimension; d++) {
            double sum = 0.0;
            for (IVector<Double> point : data) {
                double diff = point.get(d) - globalMean.get(d);
                sum += diff * diff;
            }
            variances[d] = sum / (data.size() - 1);
        }
        
        // 计算全局协方差矩阵
        IMatrix<Double> covariance = Linalg.zeros(dimension, dimension);
        for (IVector<Double> point : data) {
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    double diff_i = point.get(i) - globalMean.get(i);
                    double diff_j = point.get(j) - globalMean.get(j);
                    covariance.set(i, j, covariance.get(i, j) + diff_i * diff_j);
                }
            }
        }
        
        // 归一化并添加自适应正则化
        double avgVariance = 0.0;
        for (double var : variances) {
            avgVariance += var;
        }
        avgVariance /= dimension;
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                covariance.set(i, j, covariance.get(i, j) / (data.size() - 1));
            }
        }
        
        // 添加自适应正则化到对角线
        for (int i = 0; i < dimension; i++) {
            double regularization = Math.max(avgVariance * 0.01, 1e-6);
            covariance.set(i, i, covariance.get(i, i) + regularization);
        }
        
        return covariance;
    }
    
    /**
     * 计算局部协方差矩阵（改进版，更稳定和保守）
     */
    private IMatrix<Double> computeLocalCovariance(List<IVector<Double>> data, IVector<Double> center, 
                                                 IMatrix<Double> globalCovariance) {
        // 更保守的邻居选择策略
        int k = Math.max(dimension * 3, Math.min(data.size() / 3, 30)); // 至少3*维度个点，最多30个
        
        // 计算所有点到中心的距离
        List<DistancePoint> distancePoints = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            IVector<Double> point = data.get(i);
            double distance = euclideanDistance(point, center);
            distancePoints.add(new DistancePoint(point, distance));
        }
        
        // 排序并选择最近的k个点
        distancePoints.sort((a, b) -> Double.compare(a.distance, b.distance));
        List<IVector<Double>> nearestPoints = new ArrayList<>();
        for (int i = 0; i < Math.min(k, distancePoints.size()); i++) {
            nearestPoints.add(distancePoints.get(i).point);
        }
        
        // 如果邻居点太少，直接返回全局协方差矩阵
        if (nearestPoints.size() < dimension + 1) {
            return globalCovariance.copy();
        }
        
        // 计算局部均值
        IVector<Double> localMean = Linalg.zeros(dimension);
        for (IVector<Double> point : nearestPoints) {
            for (int d = 0; d < dimension; d++) {
                localMean.set(d, localMean.get(d) + point.get(d));
            }
        }
        for (int d = 0; d < dimension; d++) {
            localMean.set(d, localMean.get(d) / nearestPoints.size());
        }
        
        // 计算局部协方差矩阵
        IMatrix<Double> localCovariance = Linalg.zeros(dimension, dimension);
        for (IVector<Double> point : nearestPoints) {
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    double diff_i = point.get(i) - localMean.get(i);
                    double diff_j = point.get(j) - localMean.get(j);
                    localCovariance.set(i, j, localCovariance.get(i, j) + diff_i * diff_j);
                }
            }
        }
        
        // 归一化
        int effectiveSize = nearestPoints.size() - 1;
        if (effectiveSize <= 0) {
            return globalCovariance.copy();
        }
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                localCovariance.set(i, j, localCovariance.get(i, j) / effectiveSize);
            }
        }
        
        // 检查局部协方差矩阵的健康状况
        boolean isHealthy = true;
        for (int i = 0; i < dimension; i++) {
            double diag = localCovariance.get(i, i);
            if (diag <= 0 || !Double.isFinite(diag)) {
                isHealthy = false;
                break;
            }
        }
        
        // 如果局部协方差矩阵不健康，使用更保守的混合策略
        double mixingRatio = isHealthy ? 0.5 : 0.2; // 健康时50%局部，不健康时20%局部
        
        // 与全局协方差矩阵混合
        IMatrix<Double> result = Linalg.zeros(dimension, dimension);
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                double localValue = localCovariance.get(i, j);
                double globalValue = globalCovariance.get(i, j);
                double mixedValue = mixingRatio * localValue + (1 - mixingRatio) * globalValue;
                result.set(i, j, mixedValue);
            }
        }
        
        // 添加额外的正则化以确保数值稳定性
        for (int i = 0; i < dimension; i++) {
            double currentDiag = result.get(i, i);
            double globalDiag = globalCovariance.get(i, i);
            double regularizedDiag = Math.max(currentDiag, globalDiag * 0.1);
            result.set(i, i, regularizedDiag);
        }
        
        return result;
    }
    
    /**
     * 距离点辅助类
     */
    private static class DistancePoint {
        final IVector<Double> point;
        final double distance;
        
        DistancePoint(IVector<Double> point, double distance) {
            this.point = point;
            this.distance = distance;
        }
    }
    
    // Getters and Setters
    public int getNumComponents() {
        return numComponents;
    }
    
    public int getDimension() {
        return dimension;
    }
    
    public MultivariateNormalDistribution getComponent(int index) {
        if (index < 0 || index >= numComponents) {
            throw new IndexOutOfBoundsException("分量索引超出范围");
        }
        return components.get(index);
    }
    
    public double getWeight(int index) {
        if (index < 0 || index >= numComponents) {
            throw new IndexOutOfBoundsException("权重索引超出范围");
        }
        return weights.get(index);
    }
    
    public void setWeight(int index, double weight) {
        if (index < 0 || index >= numComponents) {
            throw new IndexOutOfBoundsException("权重索引超出范围");
        }
        if (weight < 0) {
            throw new IllegalArgumentException("权重必须非负");
        }
        weights.set(index, weight);
    }
    
    public List<Double> getWeights() {
        return new ArrayList<>(weights);
    }
    
    public List<MultivariateNormalDistribution> getComponents() {
        return new ArrayList<>(components);
    }
    
    /**
     * 设置指定索引的分量
     * @param index 分量索引
     * @param component 新的分量
     */
    public void setComponent(int index, MultivariateNormalDistribution component) {
        if (index < 0 || index >= numComponents) {
            throw new IndexOutOfBoundsException("分量索引超出范围");
        }
        if (component.getDimension() != dimension) {
            throw new IllegalArgumentException("分量维度必须与模型维度匹配");
        }
        components.set(index, component);
    }
    
    /**
     * 获取所有分量的均值
     * @return 均值矩阵，每行是一个分量的均值
     */
    public IMatrix<Double> getMeans() {
        IMatrix<Double> means = Linalg.zeros(numComponents, dimension);
        for (int k = 0; k < numComponents; k++) {
            IVector<Double> mean = components.get(k).getMean();
            for (int d = 0; d < dimension; d++) {
                means.set(k, d, mean.get(d));
            }
        }
        return means;
    }
    
    /**
     * 获取所有分量的协方差矩阵
     * @return 协方差矩阵列表
     */
    public List<IMatrix<Double>> getCovariances() {
        List<IMatrix<Double>> covariances = new ArrayList<>();
        for (MultivariateNormalDistribution component : components) {
            covariances.add(component.getCovariance());
        }
        return covariances;
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("GaussianMixtureModel(components=%d, dimension=%d)%n", 
                               numComponents, dimension));
        for (int k = 0; k < numComponents; k++) {
            sb.append(String.format("  Component %d: weight=%.4f, %s%n", 
                                  k, weights.get(k), components.get(k).toString()));
        }
        return sb.toString();
    }
}