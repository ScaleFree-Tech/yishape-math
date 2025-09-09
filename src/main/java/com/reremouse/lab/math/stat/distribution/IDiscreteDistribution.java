package com.reremouse.lab.math.stat.distribution;

import java.io.Serializable;

/**
 * 离散分布接口
 * 
 * 离散分布是指其取值是离散的，即取值是有限的或可数的。
 * 离散分布的分布函数是阶梯函数，即在每个取值点上有一个跳跃。
 * 与连续分布不同，离散分布使用概率质量函数(PMF)而不是概率密度函数(PDF)。
 * 
 * 该接口定义了离散概率分布的基本操作，包括：
 * - 统计量计算（均值、方差、偏度、峰度等）
 * - 概率计算（PMF、CDF、生存函数等）
 * - 随机采样
 * - 分位数计算
 * - 支持区间查询
 * 
 * Discrete distribution interface
 * 
 * A discrete distribution is one where the values are discrete, i.e., 
 * the values are finite or countable. The distribution function of a 
 * discrete distribution is a step function with jumps at each value point.
 * Unlike continuous distributions, discrete distributions use probability 
 * mass function (PMF) instead of probability density function (PDF).
 * 
 * @author lteb2
 */
public interface IDiscreteDistribution extends Serializable {
    
    // ==================== 基本统计量 / Basic Statistics ====================
    
    /**
     * 计算分布的均值（期望值）
     * Calculate the mean (expected value) of the distribution
     * 
     * @return 均值 / Mean value
     */
    public float mean();

    /**
     * 计算分布的方差
     * Calculate the variance of the distribution
     * 
     * @return 方差 / Variance
     */
    public float var();

    /**
     * 计算分布的标准差
     * Calculate the standard deviation of the distribution
     * 
     * @return 标准差 / Standard deviation
     */
    public float std();
    
    /**
     * 计算分布的中位数
     * Calculate the median of the distribution
     * 
     * @return 中位数 / Median
     */
    public float median();
    
    /**
     * 计算分布的众数（最频繁出现的值）
     * Calculate the mode of the distribution (most frequent value)
     * 
     * @return 众数，如果不存在唯一众数则返回NaN / Mode, or NaN if no unique mode exists
     */
    public float mode();
    
    /**
     * 计算第一四分位数（25%分位数）
     * Calculate the first quartile (25th percentile)
     * 
     * @return 第一四分位数 / First quartile
     */
    public float q1();
    
    /**
     * 计算第三四分位数（75%分位数）
     * Calculate the third quartile (75th percentile)
     * 
     * @return 第三四分位数 / Third quartile
     */    
    public float q3();
    
    /**
     * 计算分布的偏度
     * Calculate the skewness of the distribution
     * 
     * @return 偏度 / Skewness
     */
    public float skewness();
    
    /**
     * 计算分布的峰度
     * Calculate the kurtosis of the distribution
     * 
     * @return 峰度 / Kurtosis
     */
    public float kurtosis();
    
    // ==================== 概率计算 / Probability Calculations ====================
    
    /**
     * 计算概率质量函数值
     * Calculate probability mass function value
     * 
     * @param x 输入值 / Input value
     * @return 概率质量函数值 / PMF value
     * @throws IllegalArgumentException 如果x不在分布的支持区间内 / If x is not in the support of the distribution
     */
    public float pmf(int x);
    
    /**
     * 计算累积分布函数值
     * Calculate cumulative distribution function value
     * 
     * @param x 输入值 / Input value
     * @return 累积分布函数值 / CDF value
     */
    public float cdf(int x);
    
    /**
     * 计算百分点函数值（分位数函数）
     * Calculate percent point function value (quantile function)
     * 
     * @param prob 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 百分点函数值 / PPF value
     * @throws IllegalArgumentException 如果概率值不在[0,1]范围内 / If probability is not in range [0,1]
     */
    public int ppf(float prob);
    
    /**
     * 计算生存函数值（1 - CDF）
     * Calculate survival function value (1 - CDF)
     * 
     * @param x 输入值 / Input value
     * @return 生存函数值 / Survival function value
     */
    public float sf(int x);
    
    /**
     * 计算逆生存函数值
     * Calculate inverse survival function value
     * 
     * @param prob 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 逆生存函数值 / Inverse survival function value
     * @throws IllegalArgumentException 如果概率值不在[0,1]范围内 / If probability is not in range [0,1]
     */
    public int isf(float prob);
    
    // ==================== 随机采样 / Random Sampling ====================
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    public int sample();
    
    /**
     * 生成n个随机样本
     * Generate n random samples
     * 
     * @param n 样本数量 / Number of samples
     * @return 随机样本数组 / Array of random samples
     * @throws IllegalArgumentException 如果n小于等于0 / If n is less than or equal to 0
     */
    public int[] sample(int n);
    
    // ==================== 支持区间和验证 / Support and Validation ====================
    
    /**
     * 获取分布的最小支持值
     * Get the minimum support value of the distribution
     * 
     * @return 最小支持值 / Minimum support value
     */
    public int getMinSupport();
    
    /**
     * 获取分布的最大支持值
     * Get the maximum support value of the distribution
     * 
     * @return 最大支持值，如果无界则返回Integer.MAX_VALUE / Maximum support value, or Integer.MAX_VALUE if unbounded
     */
    public int getMaxSupport();
    
    /**
     * 检查值是否在分布的支持区间内
     * Check if value is within the support of the distribution
     * 
     * @param x 输入值 / Input value
     * @return 是否在支持区间内 / Whether within support
     */
    public boolean isInSupport(int x);
    
    /**
     * 检查分布是否有界
     * Check if the distribution is bounded
     * 
     * @return 是否有界 / Whether bounded
     */
    public boolean isBounded();
    
    // ==================== 高级统计方法 / Advanced Statistical Methods ====================
    
    /**
     * 计算分布的矩（k阶原点矩）
     * Calculate the k-th raw moment of the distribution
     * 
     * @param k 矩的阶数 / Order of the moment
     * @return k阶原点矩 / k-th raw moment
     * @throws IllegalArgumentException 如果k小于0 / If k is less than 0
     */
    public float moment(int k);
    
    /**
     * 计算分布的中心矩（k阶中心矩）
     * Calculate the k-th central moment of the distribution
     * 
     * @param k 矩的阶数 / Order of the moment
     * @return k阶中心矩 / k-th central moment
     * @throws IllegalArgumentException 如果k小于0 / If k is less than 0
     */
    public float centralMoment(int k);
    
    /**
     * 计算分布的标准化矩
     * Calculate the standardized moment of the distribution
     * 
     * @param k 矩的阶数 / Order of the moment
     * @return k阶标准化矩 / k-th standardized moment
     * @throws IllegalArgumentException 如果k小于0或标准差为0 / If k is less than 0 or standard deviation is 0
     */
    public float standardizedMoment(int k);
    
    /**
     * 计算分布的熵
     * Calculate the entropy of the distribution
     * 
     * @return 熵值 / Entropy value
     */
    public float entropy();
    
    /**
     * 计算分布的累积生成函数值
     * Calculate the cumulative generating function value
     * 
     * @param t 参数 / Parameter
     * @return 累积生成函数值 / CGF value
     */
    public float cgf(float t);
    
    // ==================== 分位数和区间估计 / Quantiles and Interval Estimation ====================
    
    /**
     * 计算指定概率的分位数
     * Calculate quantile for specified probability
     * 
     * @param prob 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 分位数 / Quantile
     * @throws IllegalArgumentException 如果概率值不在[0,1]范围内 / If probability is not in range [0,1]
     */
    public int quantile(float prob);
    
    /**
     * 计算置信区间
     * Calculate confidence interval
     * 
     * @param confidence 置信水平，范围[0,1] / Confidence level, range [0,1]
     * @return 置信区间，[下界, 上界] / Confidence interval [lower bound, upper bound]
     * @throws IllegalArgumentException 如果置信水平不在[0,1]范围内 / If confidence level is not in range [0,1]
     */
    public int[] confidenceInterval(float confidence);
    
    // ==================== 分布比较和距离 / Distribution Comparison and Distance ====================
    
    /**
     * 计算与另一个离散分布的KL散度
     * Calculate Kullback-Leibler divergence with another discrete distribution
     * 
     * @param other 另一个离散分布 / Another discrete distribution
     * @return KL散度 / KL divergence
     * @throws IllegalArgumentException 如果分布不兼容 / If distributions are incompatible
     */
    public float klDivergence(IDiscreteDistribution other);
    
    /**
     * 计算与另一个离散分布的JS散度
     * Calculate Jensen-Shannon divergence with another discrete distribution
     * 
     * @param other 另一个离散分布 / Another discrete distribution
     * @return JS散度 / JS divergence
     * @throws IllegalArgumentException 如果分布不兼容 / If distributions are incompatible
     */
    public float jsDivergence(IDiscreteDistribution other);
    
    /**
     * 计算与另一个离散分布的Wasserstein距离
     * Calculate Wasserstein distance with another discrete distribution
     * 
     * @param other 另一个离散分布 / Another discrete distribution
     * @return Wasserstein距离 / Wasserstein distance
     * @throws IllegalArgumentException 如果分布不兼容 / If distributions are incompatible
     */
    public float wassersteinDistance(IDiscreteDistribution other);
    
    // ==================== 分布信息 / Distribution Information ====================
    
    /**
     * 获取分布的名称
     * Get the name of the distribution
     * 
     * @return 分布名称 / Distribution name
     */
    public String getDistributionName();
    
    /**
     * 获取分布的参数信息
     * Get the parameter information of the distribution
     * 
     * @return 参数信息字符串 / Parameter information string
     */
    public String getParameterInfo();
    
    /**
     * 检查分布是否是对称的
     * Check if the distribution is symmetric
     * 
     * @return 是否对称 / Whether symmetric
     */
    public boolean isSymmetric();
    
    /**
     * 检查分布是否是无记忆的
     * Check if the distribution is memoryless
     * 
     * @return 是否无记忆 / Whether memoryless
     */
    public boolean isMemoryless();
    
}
