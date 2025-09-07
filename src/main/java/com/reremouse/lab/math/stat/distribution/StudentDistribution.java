package com.reremouse.lab.math.stat.distribution;

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
        this.normalizationConstant = (float) (gamma(halfDofPlusHalf) / 
            (Math.sqrt(degreesOfFreedom * Math.PI) * gamma(halfDof)));
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
            return 0.5f * (1.0f + erf(x / (float) Math.sqrt(2.0)));
        }
        
        // 使用不完全贝塔函数
        // Using incomplete beta function
        float t = x / (float) Math.sqrt(degreesOfFreedom + x * x);
        return 0.5f + 0.5f * sign(x) * incompleteBeta(halfDof, 0.5f, t * t);
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
            return inverseNormalCDF(p);
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
    
    /**
     * 伽马函数的近似实现
     * Approximation implementation of gamma function
     * 
     * @param x 输入值 / Input value
     * @return 伽马函数值 / Gamma function value
     */
    private double gamma(float x) {
        // Stirling's approximation for gamma function
        if (x < 0.5f) {
            return Math.PI / (Math.sin(Math.PI * x) * gamma(1.0f - x));
        }
        
        x = x - 1.0f;
        double result = 0.99999999999980993;
        double[] coefficients = {
            676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012,
            9.9843695780195716e-6, 1.5056327351493116e-7
        };
        
        for (int i = 0; i < coefficients.length; i++) {
            result += coefficients[i] / (x + i + 1);
        }
        
        double t = x + coefficients.length - 0.5;
        return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * result;
    }
    
    /**
     * 不完全贝塔函数的近似实现
     * Approximation implementation of incomplete beta function
     * 
     * @param a 参数a / Parameter a
     * @param b 参数b / Parameter b
     * @param x 输入值 / Input value
     * @return 不完全贝塔函数值 / Incomplete beta function value
     */
    private float incompleteBeta(float a, float b, float x) {
        if (x < 0.0f || x > 1.0f) {
            throw new IllegalArgumentException("x必须在[0,1]范围内 / x must be in range [0,1]");
        }
        
        if (x == 0.0f) return 0.0f;
        if (x == 1.0f) return 1.0f;
        
        // 使用连分数展开
        // Using continued fraction expansion
        float bt = (float) (Math.exp(gamma(a + b) - gamma(a) - gamma(b) + 
            a * Math.log(x) + b * Math.log(1.0 - x)));
        
        if (x < (a + 1.0f) / (a + b + 2.0f)) {
            return bt * betaCF(a, b, x) / a;
        } else {
            return 1.0f - bt * betaCF(b, a, 1.0f - x) / b;
        }
    }
    
    /**
     * 贝塔函数的连分数展开
     * Continued fraction expansion for beta function
     */
    private float betaCF(float a, float b, float x) {
        int maxIter = 100;
        float eps = 1e-10f;
        
        float qab = a + b;
        float qap = a + 1.0f;
        float qam = a - 1.0f;
        float c = 1.0f;
        float d = 1.0f - qab * x / qap;
        
        if (Math.abs(d) < eps) d = eps;
        d = 1.0f / d;
        float h = d;
        
        for (int m = 1; m <= maxIter; m++) {
            int m2 = 2 * m;
            float aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0f + aa * d;
            if (Math.abs(d) < eps) d = eps;
            c = 1.0f + aa / c;
            if (Math.abs(c) < eps) c = eps;
            d = 1.0f / d;
            h *= d * c;
            
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0f + aa * d;
            if (Math.abs(d) < eps) d = eps;
            c = 1.0f + aa / c;
            if (Math.abs(c) < eps) c = eps;
            d = 1.0f / d;
            float del = d * c;
            h *= del;
            
            if (Math.abs(del - 1.0f) < eps) break;
        }
        
        return h;
    }
    
    /**
     * 误差函数的近似实现
     * Approximation implementation of error function
     */
    private float erf(float x) {
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
     */
    private float inverseNormalCDF(float p) {
        // 简化的逆正态CDF近似
        // Simplified inverse normal CDF approximation
        float[] a = {0.0f, -3.969683028665376e+01f, 2.209460984245205e+02f, -2.759285104469687e+02f, 1.383577518672690e+02f, -3.066479806614201e+01f, 2.506628277459239f};
        float[] b = {0.0f, -5.447609879822406e+01f, 1.615858368580409e+02f, -1.556989798598866e+02f, 6.680131188771972e+01f, -1.328068155288572e+01f};
        
        float q = p - 0.5f;
        float r = q * q;
        float x = (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q / 
            (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0f);
        
        return x;
    }
    
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
        
        float[] samples = new float[n];
        for (int i = 0; i < n; i++) {
            samples[i] = sample();
        }
        return samples;
    }
    
    @Override
    public String toString() {
        return String.format("StudentDistribution(df=%.3f)", degreesOfFreedom);
    }
}
