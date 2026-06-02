package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * 多元分布工厂类 / Multivariate Distributions Factory
 * 
 * <p>这个工厂类提供了创建各种多元统计分布的便捷方法，充分利用项目现有的功能：</p>
 * <ul>
 *   <li>使用 linalg 包的 IVector、IMatrix 和 Linalg 进行线性代数运算</li>
 *   <li>集成 stats 包的现有统计功能</li>
 *   <li>提供统一的多元分布创建接口</li>
 *   <li>支持参数估计和分布拟合</li>
 * </ul>
 * 
 * <p>This factory class provides convenient methods for creating various multivariate statistical
 * distributions, making full use of existing project functionality:</p>
 * <ul>
 *   <li>Uses IVector, IMatrix and Linalg from linalg package for linear algebra operations</li>
 *   <li>Integrates existing statistical functionality from stats package</li>
 *   <li>Provides unified interface for creating multivariate distributions</li>
 *   <li>Supports parameter estimation and distribution fitting</li>
 * </ul>
 * 
 * <h3>支持的分布类型 / Supported Distribution Types:</h3>
 * <ul>
 *   <li>多元正态分布 / Multivariate Normal Distribution</li>
 *   <li>多元t分布 / Multivariate t-Distribution</li>
 *   <li>多元均匀分布 / Multivariate Uniform Distribution</li>
 *   <li>多元指数分布 / Multivariate Exponential Distribution</li>
 *   <li>多元Beta分布（Dirichlet分布） / Multivariate Beta Distribution (Dirichlet)</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public final class MultivariateDistributions {
    
    /** 默认随机数生成器 / Default random number generator */
    private static final Random DEFAULT_RANDOM = new Random();
    
    // 私有构造函数，防止实例化
    private MultivariateDistributions() {
        throw new UnsupportedOperationException("工厂类不能被实例化");
    }
    
    // ==================== 多元正态分布 / Multivariate Normal Distribution ====================
    
    /**
     * 创建标准多元正态分布（零均值，单位协方差矩阵）
     * Create standard multivariate normal distribution (zero mean, identity covariance)
     * 
     * @param dimension 维度 / Dimension
     * @return 多元正态分布实例 / Multivariate normal distribution instance
     */
    public static MultivariateNormalDistribution standardNormal(int dimension) {
        IVector<Double> mean = Linalg.zeros(dimension);
        IMatrix<Double> covariance = Linalg.eye(dimension);
        return new MultivariateNormalDistribution(mean, covariance);
    }
    
    /**
     * 创建多元正态分布
     * Create multivariate normal distribution
     * 
     * @param mean 均值向量 / Mean vector
     * @param covariance 协方差矩阵 / Covariance matrix
     * @return 多元正态分布实例 / Multivariate normal distribution instance
     */
    public static MultivariateNormalDistribution normal(IVector<Double> mean, IMatrix<Double> covariance) {
        return new MultivariateNormalDistribution(mean, covariance);
    }
    
    /**
     * 创建多元正态分布（使用数组参数）
     * Create multivariate normal distribution (using array parameters)
     * 
     * @param mean 均值数组 / Mean array
     * @param covariance 协方差矩阵数组 / Covariance matrix array
     * @return 多元正态分布实例 / Multivariate normal distribution instance
     */
    public static MultivariateNormalDistribution normal(double[] mean, double[][] covariance) {
        IVector<Double> meanVector = Linalg.vector(mean);
        IMatrix<Double> covMatrix = Linalg.matrix(covariance);
        return new MultivariateNormalDistribution(meanVector, covMatrix);
    }
    
    /**
     * 创建独立多元正态分布（对角协方差矩阵）
     * Create independent multivariate normal distribution (diagonal covariance matrix)
     * 
     * @param mean 均值向量 / Mean vector
     * @param variances 方差向量 / Variance vector
     * @return 多元正态分布实例 / Multivariate normal distribution instance
     */
    public static MultivariateNormalDistribution independentNormal(IVector<Double> mean, IVector<Double> variances) {
        int dimension = mean.length();
        if (variances.length() != dimension) {
            throw new IllegalArgumentException("均值和方差向量维度必须相同");
        }
        
        double[][] covArray = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            covArray[i][i] = variances.get(i);
        }
        
        IMatrix<Double> covariance = Linalg.matrix(covArray);
        return new MultivariateNormalDistribution(mean, covariance);
    }
    
    /**
     * 从样本数据拟合多元正态分布
     * Fit multivariate normal distribution from sample data
     * 
     * @param samples 样本数据列表 / Sample data list
     * @return 拟合的多元正态分布 / Fitted multivariate normal distribution
     */
    public static MultivariateNormalDistribution fitNormal(List<IVector<Double>> samples) {
        return MultivariateNormalDistribution.fitFromSamples(samples);
    }
    
    /**
     * 从样本矩阵拟合多元正态分布
     * Fit multivariate normal distribution from sample matrix
     * 
     * @param sampleMatrix 样本矩阵（每行一个样本） / Sample matrix (each row is a sample)
     * @return 拟合的多元正态分布 / Fitted multivariate normal distribution
     */
    public static MultivariateNormalDistribution fitNormal(IMatrix<Double> sampleMatrix) {
        List<IVector<Double>> samples = matrixToSampleList(sampleMatrix);
        return fitNormal(samples);
    }
    
    // ==================== 多元t分布 / Multivariate t-Distribution ====================
    
    /**
     * 创建标准多元t分布
     * Create standard multivariate t-distribution
     * 
     * @param dimension 维度 / Dimension
     * @param degreesOfFreedom 自由度 / Degrees of freedom
     * @return 多元t分布实例 / Multivariate t-distribution instance
     */
    public static MultivariateTDistribution standardT(int dimension, double degreesOfFreedom) {
        IVector<Double> location = Linalg.zeros(dimension);
        IMatrix<Double> scale = Linalg.eye(dimension);
        return new MultivariateTDistribution(location, scale, degreesOfFreedom);
    }
    
    /**
     * 创建多元t分布
     * Create multivariate t-distribution
     * 
     * @param location 位置向量 / Location vector
     * @param scale 尺度矩阵 / Scale matrix
     * @param degreesOfFreedom 自由度 / Degrees of freedom
     * @return 多元t分布实例 / Multivariate t-distribution instance
     */
    public static MultivariateTDistribution t(IVector<Double> location, IMatrix<Double> scale, double degreesOfFreedom) {
        return new MultivariateTDistribution(location, scale, degreesOfFreedom);
    }
    
    /**
     * 从样本数据拟合多元t分布
     * Fit multivariate t-distribution from sample data
     * 
     * @param samples 样本数据列表 / Sample data list
     * @return 拟合的多元t分布 / Fitted multivariate t-distribution
     */
    public static MultivariateTDistribution fitT(List<IVector<Double>> samples) {
        return MultivariateTDistribution.fitFromSamples(samples);
    }
    
    // ==================== 多元均匀分布 / Multivariate Uniform Distribution ====================
    
    /**
     * 创建标准多元均匀分布（单位超立方体）
     * Create standard multivariate uniform distribution (unit hypercube)
     * 
     * @param dimension 维度 / Dimension
     * @return 多元均匀分布实例 / Multivariate uniform distribution instance
     */
    public static MultivariateUniformDistribution standardUniform(int dimension) {
        IVector<Double> lowerBounds = Linalg.zeros(dimension);
        IVector<Double> upperBounds = Linalg.ones(dimension);
        return new MultivariateUniformDistribution(lowerBounds, upperBounds);
    }
    
    /**
     * 创建多元均匀分布
     * Create multivariate uniform distribution
     * 
     * @param lowerBounds 下界向量 / Lower bounds vector
     * @param upperBounds 上界向量 / Upper bounds vector
     * @return 多元均匀分布实例 / Multivariate uniform distribution instance
     */
    public static MultivariateUniformDistribution uniform(IVector<Double> lowerBounds, IVector<Double> upperBounds) {
        return new MultivariateUniformDistribution(lowerBounds, upperBounds);
    }
    
    /**
     * 创建对称多元均匀分布
     * Create symmetric multivariate uniform distribution
     * 
     * @param dimension 维度 / Dimension
     * @param bound 对称边界 / Symmetric bound
     * @return 多元均匀分布实例 / Multivariate uniform distribution instance
     */
    public static MultivariateUniformDistribution symmetricUniform(int dimension, double bound) {
        double[] lowerArray = new double[dimension];
        double[] upperArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            lowerArray[i] = -bound;
            upperArray[i] = bound;
        }
        
        IVector<Double> lowerBounds = Linalg.vector(lowerArray);
        IVector<Double> upperBounds = Linalg.vector(upperArray);
        return new MultivariateUniformDistribution(lowerBounds, upperBounds);
    }
    
    // ==================== 多元指数分布 / Multivariate Exponential Distribution ====================
    
    /**
     * 创建标准多元指数分布（所有率参数为1）
     * Create standard multivariate exponential distribution (all rate parameters equal to 1)
     * 
     * @param dimension 维度 / Dimension
     * @return 多元指数分布实例 / Multivariate exponential distribution instance
     */
    public static MultivariateExponentialDistribution standardExponential(int dimension) {
        return MultivariateExponentialDistribution.standard(dimension);
    }
    
    /**
     * 创建多元指数分布
     * Create multivariate exponential distribution
     * 
     * @param rates 率参数向量 / Rate parameters vector
     * @return 多元指数分布实例 / Multivariate exponential distribution instance
     */
    public static MultivariateExponentialDistribution exponential(IVector<Double> rates) {
        return new MultivariateExponentialDistribution(rates);
    }
    
    /**
     * 创建均匀率参数的多元指数分布
     * Create multivariate exponential distribution with uniform rate parameters
     * 
     * @param dimension 维度 / Dimension
     * @param rate 率参数 / Rate parameter
     * @return 多元指数分布实例 / Multivariate exponential distribution instance
     */
    public static MultivariateExponentialDistribution uniformExponential(int dimension, double rate) {
        return MultivariateExponentialDistribution.uniform(dimension, rate);
    }
    
    /**
     * 从样本数据拟合多元指数分布
     * Fit multivariate exponential distribution from sample data
     * 
     * @param samples 样本数据列表 / Sample data list
     * @return 拟合的多元指数分布 / Fitted multivariate exponential distribution
     */
    public static MultivariateExponentialDistribution fitExponential(List<IVector<Double>> samples) {
        return MultivariateExponentialDistribution.fitFromSamples(samples);
    }
    
    // ==================== 多元Beta分布（Dirichlet分布） / Multivariate Beta Distribution (Dirichlet) ====================
    
    /**
     * 创建均匀Dirichlet分布（所有参数为1）
     * Create uniform Dirichlet distribution (all parameters equal to 1)
     * 
     * @param dimension 维度 / Dimension
     * @return Dirichlet分布实例 / Dirichlet distribution instance
     */
    public static MultivariateBetaDistribution uniformDirichlet(int dimension) {
        return MultivariateBetaDistribution.uniform(dimension);
    }
    
    /**
     * 创建对称Dirichlet分布
     * Create symmetric Dirichlet distribution
     * 
     * @param dimension 维度 / Dimension
     * @param alpha 形状参数 / Shape parameter
     * @return Dirichlet分布实例 / Dirichlet distribution instance
     */
    public static MultivariateBetaDistribution symmetricDirichlet(int dimension, double alpha) {
        return MultivariateBetaDistribution.symmetric(dimension, alpha);
    }
    
    /**
     * 创建Dirichlet分布
     * Create Dirichlet distribution
     * 
     * @param alpha 形状参数向量 / Shape parameters vector
     * @return Dirichlet分布实例 / Dirichlet distribution instance
     */
    public static MultivariateBetaDistribution dirichlet(IVector<Double> alpha) {
        return new MultivariateBetaDistribution(alpha);
    }
    
    /**
     * 从样本数据拟合Dirichlet分布
     * Fit Dirichlet distribution from sample data
     * 
     * @param samples 样本数据列表 / Sample data list
     * @return 拟合的Dirichlet分布 / Fitted Dirichlet distribution
     */
    public static MultivariateBetaDistribution fitDirichlet(List<IVector<Double>> samples) {
        return MultivariateBetaDistribution.fitFromSamples(samples);
    }
    
    // ==================== 工具方法 / Utility Methods ====================
    
    /**
     * 将样本矩阵转换为样本列表
     * Convert sample matrix to sample list
     * 
     * @param sampleMatrix 样本矩阵（每行一个样本） / Sample matrix (each row is a sample)
     * @return 样本列表 / Sample list
     */
    public static List<IVector<Double>> matrixToSampleList(IMatrix<Double> sampleMatrix) {
        List<IVector<Double>> samples = new ArrayList<>();
        for (int i = 0; i < sampleMatrix.rows(); i++) {
            samples.add(sampleMatrix.getRow(i));
        }
        return samples;
    }
    
    /**
     * 将样本列表转换为样本矩阵
     * Convert sample list to sample matrix
     * 
     * @param samples 样本列表 / Sample list
     * @return 样本矩阵（每行一个样本） / Sample matrix (each row is a sample)
     */
    public static IMatrix<Double> sampleListToMatrix(List<IVector<Double>> samples) {
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("样本列表不能为空");
        }
        
        int n = samples.size();
        int dimension = samples.get(0).length();
        double[][] sampleArray = new double[n][dimension];
        
        for (int i = 0; i < n; i++) {
            IVector<Double> sample = samples.get(i);
            if (sample.length() != dimension) {
                throw new IllegalArgumentException("所有样本维度必须一致");
            }
            for (int j = 0; j < dimension; j++) {
                sampleArray[i][j] = sample.get(j);
            }
        }
        
        return Linalg.matrix(sampleArray);
    }
    
    /**
     * 计算样本的经验均值向量
     * Compute empirical mean vector of samples
     * 
     * @param samples 样本列表 / Sample list
     * @return 经验均值向量 / Empirical mean vector
     */
    public static IVector<Double> empiricalMean(List<IVector<Double>> samples) {
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("样本列表不能为空");
        }
        
        int n = samples.size();
        int dimension = samples.get(0).length();
        double[] meanArray = new double[dimension];
        
        for (IVector<Double> sample : samples) {
            for (int i = 0; i < dimension; i++) {
                meanArray[i] += sample.get(i);
            }
        }
        
        for (int i = 0; i < dimension; i++) {
            meanArray[i] /= n;
        }
        
        return Linalg.vector(meanArray);
    }
    
    /**
     * 计算样本的经验协方差矩阵
     * Compute empirical covariance matrix of samples
     * 
     * @param samples 样本列表 / Sample list
     * @return 经验协方差矩阵 / Empirical covariance matrix
     */
    public static IMatrix<Double> empiricalCovariance(List<IVector<Double>> samples) {
        if (samples.isEmpty()) {
            throw new IllegalArgumentException("样本列表不能为空");
        }
        
        int n = samples.size();
        int dimension = samples.get(0).length();
        
        // 计算均值
        IVector<Double> mean = empiricalMean(samples);
        
        // 计算协方差矩阵
        double[][] covArray = new double[dimension][dimension];
        for (IVector<Double> sample : samples) {
            IVector<Double> diff = sample.sub(mean);
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    covArray[i][j] += diff.get(i) * diff.get(j);
                }
            }
        }
        
        // 除以 n-1（无偏估计）
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                covArray[i][j] /= (n - 1);
            }
        }
        
        return Linalg.matrix(covArray);
    }
    
    /**
     * 计算样本的经验相关矩阵
     * Compute empirical correlation matrix of samples
     * 
     * @param samples 样本列表 / Sample list
     * @return 经验相关矩阵 / Empirical correlation matrix
     */
    public static IMatrix<Double> empiricalCorrelation(List<IVector<Double>> samples) {
        IMatrix<Double> cov = empiricalCovariance(samples);
        int dimension = cov.rows();
        
        double[][] corrArray = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i == j) {
                    corrArray[i][j] = 1.0;
                } else {
                    double stdI = Math.sqrt(cov.get(i, i));
                    double stdJ = Math.sqrt(cov.get(j, j));
                    corrArray[i][j] = cov.get(i, j) / (stdI * stdJ);
                }
            }
        }
        
        return Linalg.matrix(corrArray);
    }
    
    /**
     * 生成多元正态分布的随机样本
     * Generate random samples from multivariate normal distribution
     * 
     * @param mean 均值向量 / Mean vector
     * @param covariance 协方差矩阵 / Covariance matrix
     * @param n 样本数量 / Number of samples
     * @return 样本列表 / Sample list
     */
    public static List<IVector<Double>> generateNormalSamples(IVector<Double> mean, IMatrix<Double> covariance, int n) {
        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(mean, covariance);
        return distribution.sample(n);
    }
    
    /**
     * 生成多元正态分布的随机样本矩阵
     * Generate random sample matrix from multivariate normal distribution
     * 
     * @param mean 均值向量 / Mean vector
     * @param covariance 协方差矩阵 / Covariance matrix
     * @param n 样本数量 / Number of samples
     * @return 样本矩阵 / Sample matrix
     */
    public static IMatrix<Double> generateNormalSampleMatrix(IVector<Double> mean, IMatrix<Double> covariance, int n) {
        MultivariateNormalDistribution distribution = new MultivariateNormalDistribution(mean, covariance);
        return distribution.sampleMatrix(n);
    }
    
    /**
     * 计算两个多元分布之间的KL散度
     * Compute KL divergence between two multivariate distributions
     * 
     * @param p 第一个分布 / First distribution
     * @param q 第二个分布 / Second distribution
     * @return KL散度 / KL divergence
     */
    public static double klDivergence(IMultivariateDistribution<Double> p, IMultivariateDistribution<Double> q) {
        return p.klDivergence(q);
    }
    
    /**
     * 计算两个多元分布之间的Wasserstein距离
     * Compute Wasserstein distance between two multivariate distributions
     * 
     * @param p 第一个分布 / First distribution
     * @param q 第二个分布 / Second distribution
     * @return Wasserstein距离 / Wasserstein distance
     */
    public static double wassersteinDistance(IMultivariateDistribution<Double> p, IMultivariateDistribution<Double> q) {
        return p.wassersteinDistance(q);
    }
    
    /**
     * 使用EM算法拟合高斯混合模型
     * Fit Gaussian mixture model using EM algorithm
     * 
     * @param samples 样本数据 / Sample data
     * @param k 混合成分数量 / Number of mixture components
     * @param maxIterations 最大迭代次数 / Maximum iterations
     * @param tolerance 收敛容忍度 / Convergence tolerance
     * @return 高斯混合模型 / Gaussian mixture model
     */
    public static GaussianMixtureModel fitGaussianMixture(List<IVector<Double>> samples, int k, 
                                                         int maxIterations, double tolerance) {
        return GaussianMixtureModel.fit(samples, k, maxIterations, tolerance);
    }
    
    /**
     * 高斯混合模型类
     * Gaussian Mixture Model class
     */
    public static class GaussianMixtureModel {
        private final List<MultivariateNormalDistribution> components;
        private final IVector<Double> weights;
        
        public GaussianMixtureModel(List<MultivariateNormalDistribution> components, IVector<Double> weights) {
            this.components = new ArrayList<>(components);
            this.weights = weights.copy();
        }
        
        public List<MultivariateNormalDistribution> getComponents() {
            return new ArrayList<>(components);
        }
        
        public IVector<Double> getWeights() {
            return weights.copy();
        }
        
        public double pdf(IVector<Double> x) {
            double density = 0.0;
            for (int i = 0; i < components.size(); i++) {
                density += weights.get(i) * components.get(i).pdf(x);
            }
            return density;
        }
        
        public IVector<Double> sample() {
            // 根据权重选择成分
            double u = DEFAULT_RANDOM.nextDouble();
            double cumWeight = 0.0;
            for (int i = 0; i < components.size(); i++) {
                cumWeight += weights.get(i);
                if (u <= cumWeight) {
                    return components.get(i).sample();
                }
            }
            return components.get(components.size() - 1).sample();
        }
        
        public List<IVector<Double>> sample(int n) {
            List<IVector<Double>> samples = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                samples.add(sample());
            }
            return samples;
        }
        
        /**
         * 使用EM算法拟合高斯混合模型
         * Fit Gaussian mixture model using EM algorithm
         */
        public static GaussianMixtureModel fit(List<IVector<Double>> samples, int k, 
                                             int maxIterations, double tolerance) {
            if (samples.isEmpty() || k <= 0) {
                throw new IllegalArgumentException("样本数据不能为空且混合成分数量必须大于0");
            }
            
            int n = samples.size();
            int dimension = samples.get(0).length();
            
            // 初始化参数
            List<MultivariateNormalDistribution> components = new ArrayList<>();
            double[] weightsArray = new double[k];
            
            // 使用K-means初始化
            for (int i = 0; i < k; i++) {
                // 简单初始化：使用随机样本作为均值
                IVector<Double> mean = samples.get(DEFAULT_RANDOM.nextInt(n));
                IMatrix<Double> covariance = Linalg.eye(dimension);
                components.add(new MultivariateNormalDistribution(mean, covariance));
                weightsArray[i] = 1.0 / k;
            }
            
            IVector<Double> weights = Linalg.vector(weightsArray);
            
            // EM迭代
            double prevLogLikelihood = Double.NEGATIVE_INFINITY;
            for (int iter = 0; iter < maxIterations; iter++) {
                // E步：计算后验概率
                double[][] responsibilities = new double[n][k];
                for (int i = 0; i < n; i++) {
                    IVector<Double> sample = samples.get(i);
                    double totalProb = 0.0;
                    
                    for (int j = 0; j < k; j++) {
                        responsibilities[i][j] = weights.get(j) * components.get(j).pdf(sample);
                        totalProb += responsibilities[i][j];
                    }
                    
                    // 归一化
                    for (int j = 0; j < k; j++) {
                        responsibilities[i][j] /= totalProb;
                    }
                }
                
                // M步：更新参数
                for (int j = 0; j < k; j++) {
                    // 更新权重
                    double nj = 0.0;
                    for (int i = 0; i < n; i++) {
                        nj += responsibilities[i][j];
                    }
                    weightsArray[j] = nj / n;
                    
                    // 更新均值
                    double[] meanArray = new double[dimension];
                    for (int i = 0; i < n; i++) {
                        IVector<Double> sample = samples.get(i);
                        for (int d = 0; d < dimension; d++) {
                            meanArray[d] += responsibilities[i][j] * sample.get(d);
                        }
                    }
                    for (int d = 0; d < dimension; d++) {
                        meanArray[d] /= nj;
                    }
                    IVector<Double> newMean = Linalg.vector(meanArray);
                    
                    // 更新协方差
                    double[][] covArray = new double[dimension][dimension];
                    for (int i = 0; i < n; i++) {
                        IVector<Double> sample = samples.get(i);
                        IVector<Double> diff = sample.sub(newMean);
                        for (int d1 = 0; d1 < dimension; d1++) {
                            for (int d2 = 0; d2 < dimension; d2++) {
                                covArray[d1][d2] += responsibilities[i][j] * diff.get(d1) * diff.get(d2);
                            }
                        }
                    }
                    for (int d1 = 0; d1 < dimension; d1++) {
                        for (int d2 = 0; d2 < dimension; d2++) {
                            covArray[d1][d2] /= nj;
                        }
                    }
                    IMatrix<Double> newCovariance = Linalg.matrix(covArray);
                    
                    // 更新成分
                    components.set(j, new MultivariateNormalDistribution(newMean, newCovariance));
                }
                
                weights = Linalg.vector(weightsArray);
                
                // 检查收敛
                double logLikelihood = 0.0;
                for (IVector<Double> sample : samples) {
                    double sampleProb = 0.0;
                    for (int j = 0; j < k; j++) {
                        sampleProb += weights.get(j) * components.get(j).pdf(sample);
                    }
                    logLikelihood += Math.log(sampleProb);
                }
                
                if (Math.abs(logLikelihood - prevLogLikelihood) < tolerance) {
                    break;
                }
                prevLogLikelihood = logLikelihood;
            }
            
            return new GaussianMixtureModel(components, weights);
        }
    }
    
    /**
     * 创建相关矩阵
     * Create correlation matrix
     * 
     * @param dimension 维度 / Dimension
     * @param correlations 相关系数数组（上三角部分） / Correlation coefficients array (upper triangular part)
     * @return 相关矩阵 / Correlation matrix
     */
    public static IMatrix<Double> createCorrelationMatrix(int dimension, double[] correlations) {
        if (correlations.length != dimension * (dimension - 1) / 2) {
            throw new IllegalArgumentException("相关系数数量不正确");
        }
        
        double[][] corrArray = new double[dimension][dimension];
        
        // 设置对角线为1
        for (int i = 0; i < dimension; i++) {
            corrArray[i][i] = 1.0;
        }
        
        // 设置上三角和下三角
        int index = 0;
        for (int i = 0; i < dimension; i++) {
            for (int j = i + 1; j < dimension; j++) {
                corrArray[i][j] = correlations[index];
                corrArray[j][i] = correlations[index];
                index++;
            }
        }
        
        return Linalg.matrix(corrArray);
    }
    
    /**
     * 从相关矩阵和标准差创建协方差矩阵
     * Create covariance matrix from correlation matrix and standard deviations
     * 
     * @param correlation 相关矩阵 / Correlation matrix
     * @param standardDeviations 标准差向量 / Standard deviations vector
     * @return 协方差矩阵 / Covariance matrix
     */
    public static IMatrix<Double> correlationToCovariance(IMatrix<Double> correlation, IVector<Double> standardDeviations) {
        int dimension = correlation.rows();
        if (standardDeviations.length() != dimension) {
            throw new IllegalArgumentException("标准差向量维度必须与相关矩阵维度一致");
        }
        
        double[][] covArray = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                covArray[i][j] = correlation.get(i, j) * standardDeviations.get(i) * standardDeviations.get(j);
            }
        }
        
        return Linalg.matrix(covArray);
    }
}