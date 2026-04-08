package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;

import java.io.Serializable;
import java.util.List;

/**
 * 多元分布接口 / Multivariate Distribution Interface
 * 
 * <p>定义了多元概率分布的基本操作，包括概率密度函数计算、统计量计算、随机采样等功能。
 * 该接口为所有多元分布提供统一的API，支持任意维度的多元分布。</p>
 * 
 * <p>Defines basic operations for multivariate probability distributions, including 
 * probability density function calculation, statistical computation, random sampling, etc.
 * This interface provides a unified API for all multivariate distributions, supporting 
 * multivariate distributions of arbitrary dimensions.</p>
 * 
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 *   <li><strong>概率密度计算</strong> - pdf() 和 logPdf() 方法</li>
 *   <li><strong>统计量计算</strong> - 均值向量、协方差矩阵、相关矩阵等</li>
 *   <li><strong>随机采样</strong> - 单个样本和批量样本生成</li>
 *   <li><strong>参数估计</strong> - 从样本数据估计分布参数</li>
 *   <li><strong>分布变换</strong> - 边际分布、条件分布等</li>
 *   <li><strong>距离度量</strong> - KL散度、Wasserstein距离等</li>
 * </ul>
 * 
 * <h3>使用示例 / Usage Examples:</h3>
 * <pre>{@code
 * // 创建多元正态分布
 * IVector<Double> mean = Linalg.vector(new double[]{0, 0});
 * IMatrix<Double> cov = Linalg.eye(2);
 * IMultivariateDistribution dist = new MultivariateNormalDistribution(mean, cov);
 * 
 * // 计算概率密度
 * IVector<Double> x = Linalg.vector(new double[]{1, 1});
 * double density = dist.pdf(x);
 * double logDensity = dist.logPdf(x);
 * 
 * // 随机采样
 * IVector<Double> sample = dist.sample();
 * List<IVector<Double>> samples = dist.sample(100);
 * 
 * // 获取统计量
 * IVector<Double> meanVec = dist.getMean();
 * IMatrix<Double> covMatrix = dist.getCovariance();
 * }</pre>
 * 
 * @param <T> 数值类型，通常为Double
 * @author lteb2
 * @version 1.0
 * @since 1.0
 * @see MultivariateNormalDistribution
 * @see MultivariateTDistribution
 * @see IVector
 * @see IMatrix
 */
public interface IMultivariateDistribution<T extends Number> extends Serializable {
    
    // ==================== 基本属性 / Basic Properties ====================
    
    /**
     * 获取分布的维度
     * Get the dimension of the distribution
     * 
     * @return 分布维度 / Distribution dimension
     */
    int getDimension();
    
    /**
     * 获取分布名称
     * Get the distribution name
     * 
     * @return 分布名称 / Distribution name
     */
    String getDistributionName();
    
    /**
     * 获取分布参数信息
     * Get distribution parameter information
     * 
     * @return 参数信息字符串 / Parameter information string
     */
    String getParameterInfo();
    
    // ==================== 概率密度函数 / Probability Density Functions ====================
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param x 输入向量 / Input vector
     * @return 概率密度值 / Probability density value
     * @throws IllegalArgumentException 如果输入向量维度不匹配 / If input vector dimension doesn't match
     */
    double pdf(IVector<T> x);
    
    /**
     * 计算对数概率密度函数值
     * Calculate log probability density function value
     * 
     * @param x 输入向量 / Input vector
     * @return 对数概率密度值 / Log probability density value
     * @throws IllegalArgumentException 如果输入向量维度不匹配 / If input vector dimension doesn't match
     */
    double logPdf(IVector<T> x);
    
    /**
     * 批量计算概率密度函数值
     * Calculate probability density function values for multiple inputs
     * 
     * @param samples 输入样本列表 / List of input samples
     * @return 概率密度值数组 / Array of probability density values
     * @throws IllegalArgumentException 如果任何输入向量维度不匹配 / If any input vector dimension doesn't match
     */
    double[] pdf(List<IVector<T>> samples);
    
    /**
     * 批量计算对数概率密度函数值
     * Calculate log probability density function values for multiple inputs
     * 
     * @param samples 输入样本列表 / List of input samples
     * @return 对数概率密度值数组 / Array of log probability density values
     * @throws IllegalArgumentException 如果任何输入向量维度不匹配 / If any input vector dimension doesn't match
     */
    double[] logPdf(List<IVector<T>> samples);
    
    // ==================== 统计量 / Statistical Properties ====================
    
    /**
     * 获取均值向量
     * Get the mean vector
     * 
     * @return 均值向量 / Mean vector
     */
    IVector<T> getMean();
    
    /**
     * 获取协方差矩阵
     * Get the covariance matrix
     * 
     * @return 协方差矩阵 / Covariance matrix
     */
    IMatrix<T> getCovariance();
    
    /**
     * 获取相关矩阵
     * Get the correlation matrix
     * 
     * @return 相关矩阵 / Correlation matrix
     */
    IMatrix<T> getCorrelation();
    
    /**
     * 获取精度矩阵（协方差矩阵的逆）
     * Get the precision matrix (inverse of covariance matrix)
     * 
     * @return 精度矩阵 / Precision matrix
     */
    IMatrix<T> getPrecision();
    
    /**
     * 获取标准差向量（协方差矩阵对角线元素的平方根）
     * Get the standard deviation vector (square root of covariance matrix diagonal)
     * 
     * @return 标准差向量 / Standard deviation vector
     */
    IVector<T> getStandardDeviation();
    
    /**
     * 计算马氏距离
     * Calculate Mahalanobis distance
     * 
     * @param x 输入向量 / Input vector
     * @return 马氏距离 / Mahalanobis distance
     * @throws IllegalArgumentException 如果输入向量维度不匹配 / If input vector dimension doesn't match
     */
    double mahalanobisDistance(IVector<T> x);
    
    /**
     * 计算马氏距离的平方
     * Calculate squared Mahalanobis distance
     * 
     * @param x 输入向量 / Input vector
     * @return 马氏距离的平方 / Squared Mahalanobis distance
     * @throws IllegalArgumentException 如果输入向量维度不匹配 / If input vector dimension doesn't match
     */
    double squaredMahalanobisDistance(IVector<T> x);
    
    // ==================== 随机采样 / Random Sampling ====================
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本向量 / Random sample vector
     */
    IVector<T> sample();
    
    /**
     * 生成多个随机样本
     * Generate multiple random samples
     * 
     * @param n 样本数量 / Number of samples
     * @return 随机样本列表 / List of random samples
     * @throws IllegalArgumentException 如果n小于等于0 / If n is less than or equal to 0
     */
    List<IVector<T>> sample(int n);
    
    /**
     * 生成随机样本矩阵（每行是一个样本）
     * Generate random sample matrix (each row is a sample)
     * 
     * @param n 样本数量 / Number of samples
     * @return 样本矩阵 / Sample matrix
     * @throws IllegalArgumentException 如果n小于等于0 / If n is less than or equal to 0
     */
    IMatrix<T> sampleMatrix(int n);
    
    // ==================== 边际分布和条件分布 / Marginal and Conditional Distributions ====================
    
    /**
     * 获取指定维度的边际分布
     * Get marginal distribution for specified dimensions
     * 
     * @param indices 维度索引数组 / Array of dimension indices
     * @return 边际分布 / Marginal distribution
     * @throws IllegalArgumentException 如果索引无效 / If indices are invalid
     */
    IMultivariateDistribution<T> getMarginal(int... indices);
    
    /**
     * 获取条件分布
     * Get conditional distribution
     * 
     * @param conditionIndices 条件变量的维度索引 / Dimension indices of conditioning variables
     * @param conditionValues 条件变量的值 / Values of conditioning variables
     * @return 条件分布 / Conditional distribution
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    IMultivariateDistribution<T> getConditional(int[] conditionIndices, IVector<T> conditionValues);
    
    // ==================== 分布变换 / Distribution Transformations ====================
    
    /**
     * 线性变换分布
     * Linear transformation of distribution
     * 
     * @param A 变换矩阵 / Transformation matrix
     * @param b 平移向量 / Translation vector
     * @return 变换后的分布 / Transformed distribution
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / If matrix dimensions don't match
     */
    IMultivariateDistribution<T> linearTransform(IMatrix<T> A, IVector<T> b);
    
    /**
     * 仿射变换分布（线性变换的特殊情况，b=0）
     * Affine transformation of distribution (special case of linear transformation with b=0)
     * 
     * @param A 变换矩阵 / Transformation matrix
     * @return 变换后的分布 / Transformed distribution
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / If matrix dimensions don't match
     */
    IMultivariateDistribution<T> affineTransform(IMatrix<T> A);
    
    // ==================== 分布距离和比较 / Distribution Distance and Comparison ====================
    
    /**
     * 计算与另一个多元分布的KL散度
     * Calculate Kullback-Leibler divergence with another multivariate distribution
     * 
     * @param other 另一个多元分布 / Another multivariate distribution
     * @return KL散度 / KL divergence
     * @throws IllegalArgumentException 如果分布不兼容 / If distributions are incompatible
     */
    double klDivergence(IMultivariateDistribution<T> other);
    
    /**
     * 计算与另一个多元分布的Wasserstein距离
     * Calculate Wasserstein distance with another multivariate distribution
     * 
     * @param other 另一个多元分布 / Another multivariate distribution
     * @return Wasserstein距离 / Wasserstein distance
     * @throws IllegalArgumentException 如果分布不兼容 / If distributions are incompatible
     */
    double wassersteinDistance(IMultivariateDistribution<T> other);
    
    // ==================== 参数估计 / Parameter Estimation ====================
    
    /**
     * 从样本数据估计分布参数（最大似然估计）
     * Estimate distribution parameters from sample data (Maximum Likelihood Estimation)
     * 
     * @param samples 样本数据列表 / List of sample data
     * @return 估计的分布 / Estimated distribution
     * @throws IllegalArgumentException 如果样本数据无效 / If sample data is invalid
     */
    IMultivariateDistribution<T> fit(List<IVector<T>> samples);
    
    /**
     * 从加权样本数据估计分布参数
     * Estimate distribution parameters from weighted sample data
     * 
     * @param samples 样本数据列表 / List of sample data
     * @param weights 权重列表 / List of weights
     * @return 估计的分布 / Estimated distribution
     * @throws IllegalArgumentException 如果样本数据或权重无效 / If sample data or weights are invalid
     */
    IMultivariateDistribution<T> fit(List<IVector<T>> samples, List<Double> weights);
    
    // ==================== 验证和检查 / Validation and Checking ====================
    
    /**
     * 检查分布是否是椭圆分布
     * Check if the distribution is elliptical
     * 
     * @return 是否是椭圆分布 / Whether it's an elliptical distribution
     */
    boolean isElliptical();
    
    /**
     * 检查分布是否是对称的
     * Check if the distribution is symmetric
     * 
     * @return 是否对称 / Whether symmetric
     */
    boolean isSymmetric();
    
    /**
     * 检查协方差矩阵是否是正定的
     * Check if the covariance matrix is positive definite
     * 
     * @return 是否正定 / Whether positive definite
     */
    boolean isPositiveDefinite();
    
    /**
     * 检查输入向量的维度是否有效
     * Check if the input vector dimension is valid
     * 
     * @param x 输入向量 / Input vector
     * @throws IllegalArgumentException 如果维度无效 / If dimension is invalid
     */
    void validateDimension(IVector<T> x);
    
    // ==================== 高级统计方法 / Advanced Statistical Methods ====================
    
    /**
     * 计算分布的熵
     * Calculate the entropy of the distribution
     * 
     * @return 熵值 / Entropy value
     */
    double entropy();
    
    /**
     * 计算分布的信息矩阵
     * Calculate the information matrix of the distribution
     * 
     * @return 信息矩阵 / Information matrix
     */
    IMatrix<T> informationMatrix();
    
    /**
     * 计算置信椭圆
     * Calculate confidence ellipse
     * 
     * @param confidence 置信水平 / Confidence level
     * @return 置信椭圆参数 / Confidence ellipse parameters
     * @throws IllegalArgumentException 如果置信水平无效 / If confidence level is invalid
     */
    ConfidenceEllipse getConfidenceEllipse(double confidence);
    
    // ==================== 贝叶斯分析方法 / Bayesian Analysis Methods ====================
    
    /**
     * 计算共轭先验更新
     * Calculate conjugate prior update
     * 
     * @param observations 观测数据 / Observed data
     * @return 更新后的分布 / Updated distribution
     */
    IMultivariateDistribution<T> conjugateUpdate(IVector<T> observations);
    
    /**
     * 计算边际似然（证据）
     * Calculate marginal likelihood (evidence)
     * 
     * @param observations 观测数据 / Observed data
     * @return 边际似然值 / Marginal likelihood value
     */
    double marginalLikelihood(IVector<T> observations);
    
    /**
     * 从后验分布采样
     * Sample from posterior distribution
     * 
     * @param observations 观测数据 / Observed data
     * @param n 采样数量 / Number of samples
     * @return 后验样本 / Posterior samples
     */
    List<IVector<T>> posteriorSample(IVector<T> observations, int n);
    
    /**
     * 置信椭圆参数类
     * Confidence ellipse parameters class
     */
    public static class ConfidenceEllipse {
        /** 椭圆中心 / Ellipse center */
        public final IVector<Double> center;
        /** 长轴长度 / Major axis length */
        public final double majorAxis;
        /** 短轴长度 / Minor axis length */
        public final double minorAxis;
        /** 旋转角度（弧度）/ Rotation angle (radians) */
        public final double angle;
        
        public ConfidenceEllipse(IVector<Double> center, double majorAxis, double minorAxis, double angle) {
            this.center = center;
            this.majorAxis = majorAxis;
            this.minorAxis = minorAxis;
            this.angle = angle;
        }
    }
}