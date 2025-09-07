package com.reremouse.lab.math.stat.distribution;

import java.io.Serializable;

/**
 * 正态分布 (Normal Distribution)
 * 
 * 正态分布是统计学中最重要的连续概率分布之一，也称为高斯分布。
 * 其概率密度函数为：f(x) = (1/σ√(2π)) * e^(-(x-μ)²/(2σ²))
 * 其中μ是均值，σ是标准差。
 * 
 * Normal distribution is one of the most important continuous probability distributions in statistics,
 * also known as Gaussian distribution. Its probability density function is:
 * f(x) = (1/σ√(2π)) * e^(-(x-μ)²/(2σ²))
 * where μ is the mean and σ is the standard deviation.
 * 
 * @author lteb2
 */
public class NormalDistribution implements IContinuousDistribution, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** 均值 / Mean */
    private final float mean;
    
    /** 标准差 / Standard deviation */
    private final float stdDev;
    
    /** 方差 / Variance */
    private final float variance;
    
    /** 1/√(2π) 的预计算值 / Precomputed value of 1/√(2π) */
    private static final float INV_SQRT_2PI = 0.3989422804014327f;
    
    /**
     * 构造函数，创建标准正态分布（均值为0，标准差为1）
     * Constructor for standard normal distribution (mean=0, stdDev=1)
     */
    public NormalDistribution() {
        this(0.0f, 1.0f);
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param mean 均值 / Mean
     * @param stdDev 标准差 / Standard deviation
     * @throws IllegalArgumentException 如果标准差小于等于0 / If standard deviation is less than or equal to 0
     */
    public NormalDistribution(float mean, float stdDev) {
        if (stdDev <= 0) {
            throw new IllegalArgumentException("标准差必须大于0 / Standard deviation must be greater than 0");
        }
        this.mean = mean;
        this.stdDev = stdDev;
        this.variance = stdDev * stdDev;
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
        float diff = x - mean;
        float exponent = -(diff * diff) / (2.0f * variance);
        return INV_SQRT_2PI / stdDev * (float) Math.exp(exponent);
    }
    
    /**
     * 计算累积分布函数值（使用近似方法）
     * Calculate cumulative distribution function value (using approximation)
     * 
     * @param x 输入值 / Input value
     * @return 累积分布函数值 / CDF value
     */
    @Override
    public float cdf(float x) {
        // 使用误差函数的近似公式
        // Using approximation formula for error function
        float z = (x - mean) / stdDev;
        return 0.5f * (1.0f + erf(z / (float) Math.sqrt(2.0)));
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
        
        if (p == 0.0f) return Float.NEGATIVE_INFINITY;
        if (p == 1.0f) return Float.POSITIVE_INFINITY;
        
        // 使用近似方法计算逆正态分布
        // Using approximation method to calculate inverse normal distribution
        float z = inverseNormalCDF(p);
        return mean + stdDev * z;
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
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    public float getMean() {
        return mean;
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    public float getStdDev() {
        return stdDev;
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    public float getVariance() {
        return variance;
    }
    
    /**
     * 误差函数的近似实现
     * Approximation implementation of error function
     * 
     * @param x 输入值 / Input value
     * @return 误差函数值 / Error function value
     */
    private float erf(float x) {
        // Abramowitz and Stegun approximation
        float a1 = 0.254829592f;
        float a2 = -0.284496736f;
        float a3 = 1.421413741f;
        float a4 = -1.453152027f;
        float a5 = 1.061405429f;
        float p = 0.3275911f;
        
        int sign = x >= 0 ? 1 : -1;
        x = Math.abs(x);
        
        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (float) Math.exp(-x * x);
        
        return sign * y;
    }
    
    /**
     * 逆正态累积分布函数的近似实现
     * Approximation implementation of inverse normal CDF
     * 
     * @param p 概率值 / Probability value
     * @return 逆正态CDF值 / Inverse normal CDF value
     */
    private float inverseNormalCDF(float p) {
        // Beasley-Springer-Moro algorithm approximation
        float[] a = {0.0f, -3.969683028665376e+01f, 2.209460984245205e+02f, -2.759285104469687e+02f, 1.383577518672690e+02f, -3.066479806614201e+01f, 2.506628277459239f};
        float[] b = {0.0f, -5.447609879822406e+01f, 1.615858368580409e+02f, -1.556989798598866e+02f, 6.680131188771972e+01f, -1.328068155288572e+01f};
        float[] c = {0.0f, -7.784894002430293e-03f, -3.223964580411365e-01f, -2.400758277161838f, -2.549732539343734f, 4.374664141464968f, 2.938163982698783f};
        float[] d = {0.0f, 7.784695709041462e-03f, 3.224671290700398e-01f, 2.445134137142996f, 3.754408661907416f};
        
        float x;
        if (p < 0.02425f) {
            // 左尾 / Left tail
            float q = (float) Math.sqrt(-2.0 * Math.log(p));
            x = (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / 
                ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0f);
        } else if (p <= 0.97575f) {
            // 中心区域 / Central region
            float q = p - 0.5f;
            float r = q * q;
            x = (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q / 
                (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0f);
        } else {
            // 右尾 / Right tail
            float q = (float) Math.sqrt(-2.0 * Math.log(1.0 - p));
            x = -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / 
                 ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0f);
        }
        
        return x;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    @Override
    public float mean() {
        return mean;
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    @Override
    public float var() {
        return variance;
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    @Override
    public float std() {
        return stdDev;
    }
    
    /**
     * 获取中位数
     * Get median
     * 
     * @return 中位数 / Median
     */
    @Override
    public float median() {
        return mean; // 正态分布的中位数等于均值
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    @Override
    public float mode() {
        return mean; // 正态分布的众数等于均值
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
        return 0.0f; // 正态分布的偏度为0
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public float kurtosis() {
        return 0.0f; // 正态分布的峰度为0（超额峰度）
    }
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public float sample() {
        // 使用Box-Muller变换生成正态分布随机数
        // Using Box-Muller transform to generate normal random numbers
        if (hasSpare) {
            hasSpare = false;
            return mean + stdDev * spare;
        }
        
        hasSpare = true;
        double u = Math.random();
        double v = Math.random();
        double mag = stdDev * Math.sqrt(-2.0 * Math.log(u));
        spare = (float) (mag * Math.sin(2.0 * Math.PI * v));
        return mean + (float) (mag * Math.cos(2.0 * Math.PI * v));
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
        
        float[] samples = new float[n];
        for (int i = 0; i < n; i++) {
            samples[i] = sample();
        }
        return samples;
    }
    
    // Box-Muller变换的辅助变量
    // Helper variables for Box-Muller transform
    private boolean hasSpare = false;
    private float spare = 0.0f;
    
    @Override
    public String toString() {
        return String.format("NormalDistribution(mean=%.3f, stdDev=%.3f)", mean, stdDev);
    }
}
