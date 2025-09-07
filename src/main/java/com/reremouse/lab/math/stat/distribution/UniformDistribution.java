package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.IVector;
import java.io.Serializable;

/**
 * 均匀分布 (Uniform Distribution)
 * 
 * 均匀分布是概率论中最简单的连续概率分布之一，在区间[a,b]内所有值出现的概率相等。
 * 其概率密度函数为：f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
 * 其中a是下界，b是上界。
 * 
 * Uniform distribution is one of the simplest continuous probability distributions in probability theory,
 * where all values in the interval [a,b] have equal probability. Its probability density function is:
 * f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
 * where a is the lower bound and b is the upper bound.
 * 
 * @author lteb2
 */
public class UniformDistribution implements IContinuousDistribution, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** 下界 / Lower bound */
    private final float lowerBound;
    
    /** 上界 / Upper bound */
    private final float upperBound;
    
    /** 区间长度 / Interval length */
    private final float range;
    
    /** 概率密度值 / Probability density value */
    private final float density;
    
    /**
     * 构造函数，创建标准均匀分布（区间[0,1]）
     * Constructor for standard uniform distribution (interval [0,1])
     */
    public UniformDistribution() {
        this(0.0f, 1.0f);
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param lowerBound 下界 / Lower bound
     * @param upperBound 上界 / Upper bound
     * @throws IllegalArgumentException 如果上界小于等于下界 / If upper bound is less than or equal to lower bound
     */
    public UniformDistribution(float lowerBound, float upperBound) {
        if (upperBound <= lowerBound) {
            throw new IllegalArgumentException("上界必须大于下界 / Upper bound must be greater than lower bound");
        }
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.range = upperBound - lowerBound;
        this.density = 1.0f / range;
    }
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param x 输入值 / Input value
     * @return 概率密度函数值 / PDF value
     */
    @Override
    public float pdf(float x) {
        if (x < lowerBound || x > upperBound) {
            return 0.0f;
        }
        return density;
    }
    
    /**
     * 计算累积分布函数值
     * Calculate cumulative distribution function value
     * 
     * @param x 输入值 / Input value
     * @return 累积分布函数值 / CDF value
     */
    @Override
    public float cdf(float x) {
        if (x < lowerBound) {
            return 0.0f;
        } else if (x >= upperBound) {
            return 1.0f;
        } else {
            return (x - lowerBound) / range;
        }
    }
    
    /**
     * 计算百分点函数值（分位数函数）
     * Calculate percent point function value (quantile function)
     * 
     * @param p 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 百分点函数值 / PPF value
     */
    @Override
    public float ppf(float p) {
        if (p < 0.0f || p > 1.0f) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }
        
        if (p == 0.0f) return lowerBound;
        if (p == 1.0f) return upperBound;
        
        return lowerBound + p * range;
    }
    
    /**
     * 计算生存函数值（1 - CDF）
     * Calculate survival function value (1 - CDF)
     * 
     * @param x 输入值 / Input value
     * @return 生存函数值 / Survival function value
     */
    @Override
    public float sf(float x) {
        return 1.0f - cdf(x);
    }
    
    /**
     * 计算逆生存函数值
     * Calculate inverse survival function value
     * 
     * @param p 概率值，范围[0,1] / Probability value, range [0,1]
     * @return 逆生存函数值 / Inverse survival function value
     */
    @Override
    public float isf(float p) {
        return ppf(1.0f - p);
    }
    
    /**
     * 获取下界
     * Get lower bound
     * 
     * @return 下界 / Lower bound
     */
    public float getLowerBound() {
        return lowerBound;
    }
    
    /**
     * 获取上界
     * Get upper bound
     * 
     * @return 上界 / Upper bound
     */
    public float getUpperBound() {
        return upperBound;
    }
    
    /**
     * 获取区间长度
     * Get interval length
     * 
     * @return 区间长度 / Interval length
     */
    public float getRange() {
        return range;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    public float getMean() {
        return (lowerBound + upperBound) / 2.0f;
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    public float getVariance() {
        return (range * range) / 12.0f;
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    public float getStandardDeviation() {
        return range / (2.0f * (float) Math.sqrt(3.0));
    }
    
    /**
     * 检查值是否在分布的支持区间内
     * Check if value is within the support interval of the distribution
     * 
     * @param x 输入值 / Input value
     * @return 是否在支持区间内 / Whether within support interval
     */
    public boolean isInSupport(float x) {
        return x >= lowerBound && x <= upperBound;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    @Override
    public float mean() {
        return getMean();
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    @Override
    public float var() {
        return getVariance();
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    @Override
    public float std() {
        return getStandardDeviation();
    }
    
    /**
     * 获取中位数
     * Get median
     * 
     * @return 中位数 / Median
     */
    @Override
    public float median() {
        return getMean(); // 均匀分布的中位数等于均值
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    @Override
    public float mode() {
        return Float.NaN; // 均匀分布没有唯一的众数
    }
    
    /**
     * 获取第一四分位数（Q1）
     * Get first quartile (Q1)
     * 
     * @return 第一四分位数 / First quartile
     */
    @Override
    public float q1() {
        return ppf(0.25f);
    }
    
    /**
     * 获取第三四分位数（Q3）
     * Get third quartile (Q3)
     * 
     * @return 第三四分位数 / Third quartile
     */
    @Override
    public float q3() {
        return ppf(0.75f);
    }
    
    /**
     * 获取偏度
     * Get skewness
     * 
     * @return 偏度 / Skewness
     */
    @Override
    public float skewness() {
        return 0.0f; // 均匀分布是对称的，偏度为0
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public float kurtosis() {
        return -1.2f; // 均匀分布的峰度为-1.2（超额峰度）
    }
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public float sample() {
        return lowerBound + (float) Math.random() * range;
    }
    
    /**
     * 生成n个随机样本
     * Generate n random samples
     * 
     * @param n 样本数量 / Number of samples
     * @return 随机样本数组 / Array of random samples
     */
    @Override
    public float[] sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0 / Sample size must be greater than 0");
        }
        
        // 使用IVector进行数组操作
        // Using IVector for array operations
        IVector samples = IVector.zeros(n);
        for (int i = 0; i < n; i++) {
            samples.set(i, sample());
        }
        return samples.getData();
    }
    
    @Override
    public String toString() {
        return String.format("UniformDistribution([%.3f, %.3f])", lowerBound, upperBound);
    }
}
