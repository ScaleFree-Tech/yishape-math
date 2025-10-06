package com.yishape.lab.math.stats.distribution;

import com.yishape.lab.math.RereMathUtil;
import java.io.Serializable;
import com.yishape.lab.math.linalg.IDoubleVector;

/**
 * 卡方分布 (Chi-Squared Distribution)
 * 
 * 卡方分布是统计学中重要的连续概率分布，常用于假设检验和置信区间估计。
 * 其概率密度函数为：f(x) = (1/(2^(k/2) * Γ(k/2))) * x^(k/2-1) * e^(-x/2) for x > 0, 0 otherwise
 * 其中k是自由度。
 * 
 * Chi-squared distribution is an important continuous probability distribution in statistics,
 * commonly used in hypothesis testing and confidence interval estimation. Its probability density function is:
 * f(x) = (1/(2^(k/2) * Γ(k/2))) * x^(k/2-1) * e^(-x/2) for x > 0, 0 otherwise
 * where k is the degrees of freedom.
 * 
 * @author lteb2
 */
public class Chi2Distribution implements IContinuousDistribution, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** 自由度 / Degrees of freedom */
    private final double degreesOfFreedom;
    
    /** 预计算的常数 / Precomputed constants */
    private final double halfDof;
    private final double normalizationConstant;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param degreesOfFreedom 自由度，必须大于0 / Degrees of freedom, must be greater than 0
     * @throws IllegalArgumentException 如果自由度小于等于0 / If degrees of freedom is less than or equal to 0
     */
    public Chi2Distribution(double degreesOfFreedom) {
        if (degreesOfFreedom <= 0) {
            throw new IllegalArgumentException("自由度必须大于0 / Degrees of freedom must be greater than 0");
        }
        this.degreesOfFreedom = degreesOfFreedom;
        this.halfDof = degreesOfFreedom / 2.0f;
        
        // 计算归一化常数
        // Calculate normalization constant
        this.normalizationConstant = (1.0 / (Math.pow(2.0, halfDof) * RereMathUtil.gamma(halfDof)));
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
        if (x <= 0) {
            return 0.0f;
        }
        
        double power = halfDof - 1.0f;
        double exponent = -x / 2.0f;
        return normalizationConstant * (float) Math.pow(x, power) * (float) Math.exp(exponent);
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
        if (x <= 0) {
            return 0.0f;
        }
        
        // 使用不完全伽马函数
        // Using incomplete gamma function
        return RereMathUtil.incompleteGamma(halfDof, x / 2.0);
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
        
        // 使用数值方法求解
        // Using numerical method to solve
        return inverseChi2CDF(p);
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
    public double isf(double p) {
        return ppf(1.0f - p);
    }
    
    /**
     * 获取自由度
     * Get degrees of freedom
     * 
     * @return 自由度 / Degrees of freedom
     */
    public double getDegreesOfFreedom() {
        return degreesOfFreedom;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    public double getMean() {
        return degreesOfFreedom;
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    public double getVariance() {
        return 2.0f * degreesOfFreedom;
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    public double getStandardDeviation() {
        return (float) Math.sqrt(2.0 * degreesOfFreedom);
    }
    
    /**
     * 获取中位数（近似值）
     * Get median (approximate value)
     * 
     * @return 中位数 / Median
     */
    public double getMedian() {
        // Wilson-Hilferty近似
        // Wilson-Hilferty approximation
        double n = degreesOfFreedom;
        return n * (float) Math.pow(1.0 - 2.0 / (9.0 * n), 3.0);
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    public double getMode() {
        if (degreesOfFreedom >= 2) {
            return degreesOfFreedom - 2.0f;
        }
        return 0.0f;
    }
    
    // 使用RereMathUtil中的gamma函数
    // Using gamma function from RereMathUtil
    
    // 使用RereMathUtil中的incompleteGamma函数
    // Using incompleteGamma function from RereMathUtil
    
    /**
     * 逆卡方分布累积分布函数的数值求解
     * Numerical solution for inverse chi-squared distribution CDF
     */
    private double inverseChi2CDF(double p) {
        // 使用二分法求解
        // Using bisection method to solve
        double left = 0.0f;
        double right = 2.0f * degreesOfFreedom + 10.0f; // 初始右边界
        double tolerance = 1e-6f;
        int maxIter = 100;
        
        // 调整右边界直到CDF(right) >= p
        // Adjust right boundary until CDF(right) >= p
        while (cdf(right) < p && right < 1000.0f) {
            right *= 2.0f;
        }
        
        // 确保左边界CDF值小于p
        // Ensure left boundary CDF value is less than p
        while (cdf(left) >= p && left > 1e-10f) {
            left /= 2.0f;
        }
        
        for (int i = 0; i < maxIter; i++) {
            double mid = (left + right) / 2.0f;
            double cdfMid = cdf(mid);
            
            if (Math.abs(cdfMid - p) < tolerance) {
                return mid;
            }
            
            if (cdfMid < p) {
                left = mid;
            } else {
                right = mid;
            }
            
            // 检查收敛性
            // Check convergence
            if (Math.abs(right - left) < tolerance) {
                break;
            }
        }
        
        return (left + right) / 2.0f;
    }
    
    /**
     * 检查值是否在分布的支持区间内
     * Check if value is within the support interval of the distribution
     * 
     * @param x 输入值 / Input value
     * @return 是否在支持区间内 / Whether within support interval
     */
    public boolean isInSupport(double x) {
        return x > 0;
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
        return (float) Math.sqrt(8.0 / degreesOfFreedom);
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public double kurtosis() {
        return 12.0f / degreesOfFreedom;
    }
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public double sample() {
        // 使用伽马分布生成卡方分布随机数
        // Using gamma distribution to generate chi-squared random numbers
        double gammaSample = gammaSample(halfDof, 2.0f);
        return gammaSample;
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
    
    /**
     * 生成伽马分布随机数
     * Generate gamma distribution random number
     * 
     * @param shape 形状参数 / Shape parameter
     * @param scale 尺度参数 / Scale parameter
     * @return 伽马分布随机数 / Gamma distribution random number
     */
    private double gammaSample(double shape, double scale) {
        // 使用逆变换采样方法
        // Using inverse transform sampling method
        if (shape < 1.0f) {
            // 对于形状参数小于1的情况，使用变换
            // For shape parameter less than 1, use transformation
            return gammaSample(shape + 1.0f, scale) * (float) Math.pow(Math.random(), 1.0f / shape);
        }
        
        // 使用Box-Muller变换的变体生成伽马分布
        // Using variant of Box-Muller transform to generate gamma distribution
        double d = shape - 1.0f / 3.0f;
        double c = 1.0f / (float) Math.sqrt(9.0f * d);
        
        while (true) {
            // 生成两个独立的均匀随机数
            // Generate two independent uniform random numbers
            double u1 = (float) Math.random();
            double u2 = (float) Math.random();
            
            // 使用Box-Muller变换生成正态分布随机数
            // Use Box-Muller transform to generate normal random numbers
            double z1 = (float) Math.sqrt(-2.0 * Math.log(u1)) * (float) Math.cos(2.0 * Math.PI * u2);
            
            double v = 1.0f + c * z1;
            if (v <= 0) continue;
            
            v = v * v * v;
            double u = (float) Math.random();
            
            if (u < 1.0f - 0.0331f * z1 * z1 * z1 * z1) {
                return d * v * scale;
            }
            
            if (Math.log(u) < 0.5f * z1 * z1 + d * (1.0f - v + Math.log(v))) {
                return d * v * scale;
            }
        }
    }
    
    @Override
    public String toString() {
        return String.format("Chi2Distribution(df=%.3f)", degreesOfFreedom);
    }
}
