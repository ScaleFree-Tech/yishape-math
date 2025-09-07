package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import java.io.Serializable;

/**
 * t分布 (Student's t-Distribution)
 * 
 * t分布是统计学中用于小样本推断的重要分布，特别是在总体方差未知时。
 * 其概率密度函数为：f(x) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
 * 其中ν是自由度。
 * 
 * Student's t-distribution is an important distribution in statistics for small sample inference,
 * especially when the population variance is unknown. Its probability density function is:
 * f(x) = Γ((ν+1)/2) / (√(νπ) * Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
 * where ν is the degrees of freedom.
 * 
 * @author lteb2
 */
public class StudentDistribution implements IContinuousDistribution, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** 自由度 / Degrees of freedom */
    private final float degreesOfFreedom;
    
    /** 预计算的常数 / Precomputed constants */
    private final float normalizationConstant;
    private final float halfDof;
    private final float halfDofPlusHalf;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param degreesOfFreedom 自由度，必须大于0 / Degrees of freedom, must be greater than 0
     * @throws IllegalArgumentException 如果自由度小于等于0 / If degrees of freedom is less than or equal to 0
     */
    public StudentDistribution(float degreesOfFreedom) {
        if (degreesOfFreedom <= 0) {
            throw new IllegalArgumentException("自由度必须大于0 / Degrees of freedom must be greater than 0");
        }
        this.degreesOfFreedom = degreesOfFreedom;
        this.halfDof = degreesOfFreedom / 2.0f;
        this.halfDofPlusHalf = (degreesOfFreedom + 1.0f) / 2.0f;
        
        // 计算归一化常数
        // Calculate normalization constant
        this.normalizationConstant = (float) (RereMathUtil.gamma(halfDofPlusHalf) / 
            (Math.sqrt(degreesOfFreedom * Math.PI) * RereMathUtil.gamma(halfDof)));
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
        float power = -(halfDofPlusHalf);
        float base = 1.0f + (x * x) / degreesOfFreedom;
        return normalizationConstant * (float) Math.pow(base, power);
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
        if (degreesOfFreedom >= 30) {
            // 对于大自由度，使用正态分布近似
            // For large degrees of freedom, use normal distribution approximation
            return 0.5f * (1.0f + (float) RereMathUtil.erf(x / (float) Math.sqrt(2.0)));
        }
        
        // 使用不完全贝塔函数
        // Using incomplete beta function
        float t = x / (float) Math.sqrt(degreesOfFreedom + x * x);
        return 0.5f + 0.5f * sign(x) * (float) RereMathUtil.incompleteBeta(halfDof, 0.5f, t * t);
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
        
        if (degreesOfFreedom >= 30) {
            // 对于大自由度，使用正态分布近似
            // For large degrees of freedom, use normal distribution approximation
            return (float) RereMathUtil.inverseNormalCDF(p);
        }
        
        // 使用数值方法求解
        // Using numerical method to solve
        return inverseTCDF(p);
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
     * 获取自由度
     * Get degrees of freedom
     * 
     * @return 自由度 / Degrees of freedom
     */
    public float getDegreesOfFreedom() {
        return degreesOfFreedom;
    }
    
    // 使用RereMathUtil中的gamma函数
    // Using gamma function from RereMathUtil
    
    // 使用RereMathUtil中的incompleteBeta函数
    // Using incompleteBeta function from RereMathUtil
    
    // 使用RereMathUtil中的betaCF函数
    // Using betaCF function from RereMathUtil
    
    // 使用RereMathUtil中的erf和inverseNormalCDF函数
    // Using erf and inverseNormalCDF functions from RereMathUtil
    
    /**
     * 逆t分布累积分布函数的数值求解
     * Numerical solution for inverse t-distribution CDF
     */
    private float inverseTCDF(float p) {
        // 使用二分法求解
        // Using bisection method to solve
        float left = -10.0f;
        float right = 10.0f;
        float tolerance = 1e-6f;
        int maxIter = 100;
        
        for (int i = 0; i < maxIter; i++) {
            float mid = (left + right) / 2.0f;
            float cdfMid = cdf(mid);
            
            if (Math.abs(cdfMid - p) < tolerance) {
                return mid;
            }
            
            if (cdfMid < p) {
                left = mid;
            } else {
                right = mid;
            }
        }
        
        return (left + right) / 2.0f;
    }
    
    /**
     * 符号函数
     * Sign function
     */
    private float sign(float x) {
        return x >= 0 ? 1.0f : -1.0f;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    @Override
    public float mean() {
        if (degreesOfFreedom > 1) {
            return 0.0f; // t分布的均值为0
        }
        return Float.NaN; // 当自由度 <= 1 时均值不存在
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    @Override
    public float var() {
        if (degreesOfFreedom > 2) {
            return degreesOfFreedom / (degreesOfFreedom - 2.0f);
        }
        return Float.NaN; // 当自由度 <= 2 时方差不存在
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    @Override
    public float std() {
        float variance = var();
        if (Float.isNaN(variance)) {
            return Float.NaN;
        }
        return (float) Math.sqrt(variance);
    }
    
    /**
     * 获取中位数
     * Get median
     * 
     * @return 中位数 / Median
     */
    @Override
    public float median() {
        return 0.0f; // t分布的中位数为0
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    @Override
    public float mode() {
        return 0.0f; // t分布的众数为0
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
        return 0.0f; // t分布是对称的，偏度为0
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public float kurtosis() {
        if (degreesOfFreedom > 4) {
            return 6.0f / (degreesOfFreedom - 4.0f);
        }
        return Float.NaN; // 当自由度 <= 4 时峰度不存在
    }
    
    // 缓存的分布对象，避免重复创建
    // Cached distribution objects to avoid repeated creation
    private NormalDistribution normal;
    private Chi2Distribution chi2;
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public float sample() {
        // 使用正态分布和卡方分布生成t分布随机数
        // Using normal and chi-squared distributions to generate t-distribution random numbers
        if (normal == null) {
            normal = new NormalDistribution(0.0f, 1.0f);
        }
        if (chi2 == null) {
            chi2 = new Chi2Distribution(degreesOfFreedom);
        }
        
        float z = normal.sample();
        float chi2Sample = chi2.sample();
        
        return z / (float) Math.sqrt(chi2Sample / degreesOfFreedom);
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
        return String.format("StudentDistribution(df=%.3f)", degreesOfFreedom);
    }
}
