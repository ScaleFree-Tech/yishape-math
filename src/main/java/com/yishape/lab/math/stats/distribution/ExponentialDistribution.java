package com.yishape.lab.math.stats.distribution;

import java.io.Serializable;
import com.yishape.lab.math.linalg.IDoubleVector;

/**
 * 指数分布 (Exponential Distribution)
 * 
 * 指数分布是描述泊松过程中事件间隔时间的连续概率分布，具有无记忆性。
 * 其概率密度函数为：f(x) = λe^(-λx) for x ≥ 0, 0 otherwise
 * 其中λ是速率参数（λ > 0）。
 * 
 * Exponential distribution is a continuous probability distribution that describes the time between events
 * in a Poisson process, and has the memoryless property. Its probability density function is:
 * f(x) = λe^(-λx) for x ≥ 0, 0 otherwise
 * where λ is the rate parameter (λ > 0).
 * 
 * @author lteb2
 */
public class ExponentialDistribution implements IContinuousDistribution, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** 速率参数 / Rate parameter */
    private final double rate;
    
    /** 尺度参数（1/λ） / Scale parameter (1/λ) */
    private final double scale;
    
    /**
     * 构造函数，创建标准指数分布（速率参数为1）
     * Constructor for standard exponential distribution (rate parameter = 1)
     */
    public ExponentialDistribution() {
        this(1.0f);
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param rate 速率参数，必须大于0 / Rate parameter, must be greater than 0
     * @throws IllegalArgumentException 如果速率参数小于等于0 / If rate parameter is less than or equal to 0
     */
    public ExponentialDistribution(double rate) {
        if (rate <= 0) {
            throw new IllegalArgumentException("速率参数必须大于0 / Rate parameter must be greater than 0");
        }
        this.rate = rate;
        this.scale = 1.0f / rate;
    }
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param x 输入值 / Input value
     * @return 概率密度函数值 / PDF value
     */
    @Override
    public double pdf(double x) {
        if (x < 0) {
            return 0.0f;
        }
        return rate * (float) Math.exp(-rate * x);
    }
    
    /**
     * 计算累积分布函数值
     * Calculate cumulative distribution function value
     * 
     * @param x 输入值 / Input value
     * @return 累积分布函数值 / CDF value
     */
    @Override
    public double cdf(double x) {
        if (x < 0) {
            return 0.0f;
        }
        return 1.0f - (float) Math.exp(-rate * x);
    }
    
    /**
     * 计算百分点函数值（分位数函数）
     * Calculate percent point function value (quantile function)
     * 
     * @param p 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 百分点函数值 / PPF value
     */
    @Override
    public double ppf(double p) {
        if (p < 0.0f || p > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        
        if (p == 0.0f) return 0.0f;
        if (p == 1.0f) return Float.POSITIVE_INFINITY;
        
        return -(float) Math.log(1.0f - p) / rate;
    }
    
    /**
     * 计算生存函数值（1 - CDF）
     * Calculate survival function value (1 - CDF)
     * 
     * @param x 输入值 / Input value
     * @return 生存函数值 / Survival function value
     */
    @Override
    public double sf(double x) {
        if (x < 0) {
            return 1.0f;
        }
        return (float) Math.exp(-rate * x);
    }
    
    /**
     * 计算逆生存函数值
     * Calculate inverse survival function value
     * 
     * @param p 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 逆生存函数值 / Inverse survival function value
     */
    @Override
    public double isf(double p) {
        return ppf(1.0f - p);
    }
    
    /**
     * 获取速率参数
     * Get rate parameter
     * 
     * @return 速率参数 / Rate parameter
     */
    public double getRate() {
        return rate;
    }
    
    /**
     * 获取尺度参数
     * Get scale parameter
     * 
     * @return 尺度参数 / Scale parameter
     */
    public double getScale() {
        return scale;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    public double getMean() {
        return scale;
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    public double getVariance() {
        return scale * scale;
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    public double getStandardDeviation() {
        return scale;
    }
    
    /**
     * 获取中位数
     * Get median
     * 
     * @return 中位数 / Median
     */
    public double getMedian() {
        return scale * (float) Math.log(2.0);
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    public double getMode() {
        return 0.0f;
    }
    
    /**
     * 计算风险函数（hazard function）
     * Calculate hazard function
     * 
     * @param x 输入值 / Input value
     * @return 风险函数值 / Hazard function value
     */
    public double hazard(double x) {
        if (x < 0) {
            return 0.0f;
        }
        return rate; // 指数分布的风险函数是常数
    }
    
    /**
     * 计算累积风险函数
     * Calculate cumulative hazard function
     * 
     * @param x 输入值 / Input value
     * @return 累积风险函数值 / Cumulative hazard function value
     */
    public double cumulativeHazard(double x) {
        if (x < 0) {
            return 0.0f;
        }
        return rate * x;
    }
    
    /**
     * 检查值是否在分布的支持区间内
     * Check if value is within the support interval of the distribution
     * 
     * @param x 输入值 / Input value
     * @return 是否在支持区间内 / Whether within support interval
     */
    public boolean isInSupport(double x) {
        return x >= 0;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    @Override
    public double mean() {
        return getMean();
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    @Override
    public double var() {
        return getVariance();
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    @Override
    public double std() {
        return getStandardDeviation();
    }
    
    /**
     * 获取中位数
     * Get median
     * 
     * @return 中位数 / Median
     */
    @Override
    public double median() {
        return getMedian();
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    @Override
    public double mode() {
        return getMode();
    }
    
    /**
     * 获取第一四分位数（Q1）
     * Get first quartile (Q1)
     * 
     * @return 第一四分位数 / First quartile
     */
    @Override
    public double q1() {
        return ppf(0.25f);
    }
    
    /**
     * 获取第三四分位数（Q3）
     * Get third quartile (Q3)
     * 
     * @return 第三四分位数 / Third quartile
     */
    @Override
    public double q3() {
        return ppf(0.75f);
    }
    
    /**
     * 获取偏度
     * Get skewness
     * 
     * @return 偏度 / Skewness
     */
    @Override
    public double skewness() {
        return 2.0f; // 指数分布的偏度为2
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public double kurtosis() {
        return 6.0f; // 指数分布的峰度为6
    }
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public double sample() {
        // 使用逆变换采样生成指数分布随机数
        // Using inverse transform sampling to generate exponential random numbers
        double u = (float) Math.random();
        return -(float) Math.log(1.0f - u) / rate;
    }
    
    /**
     * 生成n个随机样本
     * Generate n random samples
     * 
     * @param n 样本数量 / Number of samples
     * @return 随机样本数组 / Array of random samples
     */
    @Override
    public double[] sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0 / Sample size must be greater than 0");
        }
        
        // 使用IVector进行数组操作
        // Using IDoubleVector for array operations
        IDoubleVector samples = IDoubleVector.zeros(n);
        for (int i = 0; i < n; i++) {
            samples.set(i, sample());
        }
        return samples.getData();
    }
    
    @Override
    public String toString() {
        return String.format("ExponentialDistribution(rate=%.3f)", rate);
    }
}
