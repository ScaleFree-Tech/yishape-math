package com.yishape.lab.math.stats.distribution;

import com.yishape.lab.math.RereMathUtil;
import java.io.Serializable;
import com.yishape.lab.math.linalg.IDoubleVector;

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
    private final double mean;
    
    /** 标准差 / Standard deviation */
    private final double stdDev;
    
    /** 方差 / Variance */
    private final double variance;
    
    /** 1/√(2π) 的预计算值 / Precomputed value of 1/√(2π) */
    private static final double INV_SQRT_2PI = 0.3989422804014327;
    
    /**
     * 构造函数，创建标准正态分布（均值为0，标准差为1）
     * Constructor for standard normal distribution (mean=0, stdDev=1)
     */
    public NormalDistribution() {
        this(0.0, 1.0);
    }
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param mean 均值 / Mean
     * @param stdDev 标准差 / Standard deviation
     * @throws IllegalArgumentException 如果标准差小于等于0 / If standard deviation is less than or equal to 0
     */
    public NormalDistribution(double mean, double stdDev) {
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
    public double pdf(double x) {
        double diff = x - mean;
        double exponent = -(diff * diff) / (2.0 * variance);
        return INV_SQRT_2PI / stdDev * Math.exp(exponent);
    }
    
    /**
     * 计算累积分布函数值（使用近似方法）
     * Calculate cumulative distribution function value (using approximation)
     * 
     * @param x 输入值 / Input value
     * @return 累积分布函数值 / CDF value
     */
    @Override
    public double cdf(double x) {
        // 使用误差函数的近似公式
        // Using approximation formula for error function
        double z = (x - mean) / stdDev;
        return 0.5 * (1.0 + RereMathUtil.erf(z / Math.sqrt(2.0)));
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
        if (p < 0.0 || p > 1.0) {
            throw new IllegalArgumentException("概率值必须在[0,1]范围内 / Probability must be in range [0,1]");
        }

        if (p == 0.0) return Double.NEGATIVE_INFINITY;
        if (p == 1.0) return Double.POSITIVE_INFINITY;

        // 使用近似方法计算逆正态分布
        // Using approximation method to calculate inverse normal distribution
        double z = RereMathUtil.inverseNormalCDF(p);
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
    public double sf(double x) {
        return 1.0 - cdf(x);
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
        return ppf(1.0 - p);
    }
    
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    @Override
    public double mean() {
        return mean;
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    @Override
    public double var() {
        return variance;
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    @Override
    public double std() {
        return stdDev;
    }
    
    /**
     * 获取中位数
     * Get median
     * 
     * @return 中位数 / Median
     */
    @Override
    public double median() {
        return mean; // 正态分布的中位数等于均值
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    @Override
    public double mode() {
        return mean; // 正态分布的众数等于均值
    }
    
    /**
     * 获取第一四分位数（Q1）
     * Get first quartile (Q1)
     * 
     * @return 第一四分位数 / First quartile
     */
    @Override
    public double q1() {
        return ppf(0.25);
    }
    
    /**
     * 获取第三四分位数（Q3）
     * Get third quartile (Q3)
     * 
     * @return 第三四分位数 / Third quartile
     */
    @Override
    public double q3() {
        return ppf(0.75);
    }
    
    /**
     * 获取偏度
     * Get skewness
     * 
     * @return 偏度 / Skewness
     */
    @Override
    public double skewness() {
        return 0.0; // 正态分布的偏度为0
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public double kurtosis() {
        return 0.0; // 正态分布的峰度为0（超额峰度）
    }
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public double sample() {
        // 使用Box-Muller变换生成正态分布随机数
        // Using Box-Muller transform to generate normal random numbers
        if (hasSpare) {
            hasSpare = false;
            return mean + spare;
        }

        hasSpare = true;
        double u = java.util.concurrent.ThreadLocalRandom.current().nextDouble();
        double v = java.util.concurrent.ThreadLocalRandom.current().nextDouble();
        double mag = stdDev * Math.sqrt(-2.0 * Math.log(u));
        spare = mag * Math.sin(2.0 * Math.PI * v);
        return mean + mag * Math.cos(2.0 * Math.PI * v);
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
    
    // Box-Muller变换的辅助变量
    // Helper variables for Box-Muller transform
    private boolean hasSpare = false;
    private double spare = 0.0;
    
    @Override
    public String toString() {
        return String.format("NormalDistribution(mean=%.3f, stdDev=%.3f)", mean, stdDev);
    }
}
