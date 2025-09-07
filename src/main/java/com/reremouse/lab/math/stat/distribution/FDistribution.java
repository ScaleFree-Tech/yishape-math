package com.reremouse.lab.math.stat.distribution;

import java.io.Serializable;

/**
 * F分布 (F-Distribution)
 * 
 * F分布是统计学中重要的连续概率分布，常用于方差分析和回归分析中的假设检验。
 * 其概率密度函数为：f(x) = (Γ((d1+d2)/2) / (Γ(d1/2) * Γ(d2/2))) * (d1/d2)^(d1/2) * x^(d1/2-1) * (1 + d1*x/d2)^(-(d1+d2)/2)
 * 其中d1和d2是两个自由度参数。
 * 
 * F-distribution is an important continuous probability distribution in statistics,
 * commonly used in hypothesis testing in analysis of variance and regression analysis.
 * Its probability density function is:
 * f(x) = (Γ((d1+d2)/2) / (Γ(d1/2) * Γ(d2/2))) * (d1/d2)^(d1/2) * x^(d1/2-1) * (1 + d1*x/d2)^(-(d1+d2)/2)
 * where d1 and d2 are the two degrees of freedom parameters.
 * 
 * @author lteb2
 */
public class FDistribution implements IContinuousDistribution, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** 分子自由度 / Numerator degrees of freedom */
    private final float numeratorDof;
    
    /** 分母自由度 / Denominator degrees of freedom */
    private final float denominatorDof;
    
    /** 预计算的常数 / Precomputed constants */
    private final float halfNumeratorDof;
    private final float halfDenominatorDof;
    private final float halfSumDof;
    private final float normalizationConstant;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param numeratorDof 分子自由度，必须大于0 / Numerator degrees of freedom, must be greater than 0
     * @param denominatorDof 分母自由度，必须大于0 / Denominator degrees of freedom, must be greater than 0
     * @throws IllegalArgumentException 如果自由度小于等于0 / If degrees of freedom is less than or equal to 0
     */
    public FDistribution(float numeratorDof, float denominatorDof) {
        if (numeratorDof <= 0 || denominatorDof <= 0) {
            throw new IllegalArgumentException("自由度必须大于0 / Degrees of freedom must be greater than 0");
        }
        this.numeratorDof = numeratorDof;
        this.denominatorDof = denominatorDof;
        this.halfNumeratorDof = numeratorDof / 2.0f;
        this.halfDenominatorDof = denominatorDof / 2.0f;
        this.halfSumDof = (numeratorDof + denominatorDof) / 2.0f;
        
        // 计算归一化常数
        // Calculate normalization constant
        this.normalizationConstant = (float) (gamma(halfSumDof) / 
            (gamma(halfNumeratorDof) * gamma(halfDenominatorDof)) * 
            Math.pow(numeratorDof / denominatorDof, halfNumeratorDof));
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
        if (x <= 0) {
            return 0.0f;
        }
        
        float power1 = halfNumeratorDof - 1.0f;
        float power2 = -(halfSumDof);
        float base = 1.0f + (numeratorDof * x) / denominatorDof;
        
        return normalizationConstant * (float) Math.pow(x, power1) * (float) Math.pow(base, power2);
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
        if (x <= 0) {
            return 0.0f;
        }
        
        // 使用不完全贝塔函数
        // Using incomplete beta function
        float t = (numeratorDof * x) / (denominatorDof + numeratorDof * x);
        return (float) incompleteBeta(halfNumeratorDof, halfDenominatorDof, t);
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
        
        if (p == 0.0f) return 0.0f;
        if (p == 1.0f) return Float.POSITIVE_INFINITY;
        
        // 使用数值方法求解
        // Using numerical method to solve
        return inverseFCDF(p);
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
     * 获取分子自由度
     * Get numerator degrees of freedom
     * 
     * @return 分子自由度 / Numerator degrees of freedom
     */
    public float getNumeratorDof() {
        return numeratorDof;
    }
    
    /**
     * 获取分母自由度
     * Get denominator degrees of freedom
     * 
     * @return 分母自由度 / Denominator degrees of freedom
     */
    public float getDenominatorDof() {
        return denominatorDof;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    public float getMean() {
        if (denominatorDof > 2) {
            return denominatorDof / (denominatorDof - 2.0f);
        }
        return Float.NaN; // 当分母自由度 <= 2 时均值不存在
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    public float getVariance() {
        if (denominatorDof > 4) {
            float numerator = 2.0f * denominatorDof * denominatorDof * (numeratorDof + denominatorDof - 2.0f);
            float denominator = numeratorDof * (denominatorDof - 2.0f) * (denominatorDof - 2.0f) * (denominatorDof - 4.0f);
            return numerator / denominator;
        }
        return Float.NaN; // 当分母自由度 <= 4 时方差不存在
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    public float getStandardDeviation() {
        float variance = getVariance();
        if (Float.isNaN(variance)) {
            return Float.NaN;
        }
        return (float) Math.sqrt(variance);
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    public float getMode() {
        if (numeratorDof > 2) {
            return (denominatorDof * (numeratorDof - 2.0f)) / (numeratorDof * (denominatorDof + 2.0f));
        }
        return 0.0f;
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
    private double incompleteBeta(float a, float b, float x) {
        if (x < 0.0f || x > 1.0f) {
            throw new IllegalArgumentException("x必须在[0,1]范围内 / x must be in range [0,1]");
        }
        
        if (x == 0.0f) return 0.0;
        if (x == 1.0f) return 1.0;
        
        // 使用连分数展开
        // Using continued fraction expansion
        double bt = Math.exp(gamma(a + b) - gamma(a) - gamma(b) + 
            a * Math.log(x) + b * Math.log(1.0 - x));
        
        if (x < (a + 1.0f) / (a + b + 2.0f)) {
            return bt * betaCF(a, b, x) / a;
        } else {
            return 1.0 - bt * betaCF(b, a, 1.0f - x) / b;
        }
    }
    
    /**
     * 贝塔函数的连分数展开
     * Continued fraction expansion for beta function
     */
    private double betaCF(float a, float b, float x) {
        int maxIter = 100;
        double eps = 1e-10;
        
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - qab * x / qap;
        
        if (Math.abs(d) < eps) d = eps;
        d = 1.0 / d;
        double h = d;
        
        for (int m = 1; m <= maxIter; m++) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < eps) d = eps;
            c = 1.0 + aa / c;
            if (Math.abs(c) < eps) c = eps;
            d = 1.0 / d;
            h *= d * c;
            
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (Math.abs(d) < eps) d = eps;
            c = 1.0 + aa / c;
            if (Math.abs(c) < eps) c = eps;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            
            if (Math.abs(del - 1.0) < eps) break;
        }
        
        return h;
    }
    
    /**
     * 逆F分布累积分布函数的数值求解
     * Numerical solution for inverse F-distribution CDF
     */
    private float inverseFCDF(float p) {
        // 使用二分法求解
        // Using bisection method to solve
        float left = 0.0f;
        float right = 10.0f; // 初始右边界
        float tolerance = 1e-6f;
        int maxIter = 100;
        
        // 调整右边界直到CDF(right) >= p
        // Adjust right boundary until CDF(right) >= p
        while (cdf(right) < p && right < 1000.0f) {
            right *= 2.0f;
        }
        
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
     * 检查值是否在分布的支持区间内
     * Check if value is within the support interval of the distribution
     * 
     * @param x 输入值 / Input value
     * @return 是否在支持区间内 / Whether within support interval
     */
    public boolean isInSupport(float x) {
        return x > 0;
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
        return ppf(0.5f);
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    @Override
    public float mode() {
        return getMode();
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
        if (denominatorDof > 6) {
            float numerator = (2.0f * numeratorDof + denominatorDof - 2.0f) * 
                (float) Math.sqrt(8.0f * (denominatorDof - 4.0f));
            float denominator = (denominatorDof - 6.0f) * 
                (float) Math.sqrt(numeratorDof * (numeratorDof + denominatorDof - 2.0f));
            return numerator / denominator;
        }
        return Float.NaN; // 当分母自由度 <= 6 时偏度不存在
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public float kurtosis() {
        if (denominatorDof > 8) {
            float n1 = numeratorDof;
            float n2 = denominatorDof;
            float numerator = 12.0f * n1 * (n1 + n2 - 2.0f) * (n1 * (n2 - 2.0f) + n2 * (n2 - 4.0f));
            float denominator = n1 * (n2 - 6.0f) * (n2 - 8.0f) * (n1 + n2 - 2.0f);
            return numerator / denominator;
        }
        return Float.NaN; // 当分母自由度 <= 8 时峰度不存在
    }
    
    // 缓存的卡方分布对象，避免重复创建
    // Cached chi-squared distribution objects to avoid repeated creation
    private Chi2Distribution chi2Num;
    private Chi2Distribution chi2Den;
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public float sample() {
        // 使用卡方分布生成F分布随机数
        // Using chi-squared distributions to generate F-distribution random numbers
        if (chi2Num == null) {
            chi2Num = new Chi2Distribution(numeratorDof);
        }
        if (chi2Den == null) {
            chi2Den = new Chi2Distribution(denominatorDof);
        }
        
        float chi2NumSample = chi2Num.sample();
        float chi2DenSample = chi2Den.sample();
        
        return (chi2NumSample / numeratorDof) / (chi2DenSample / denominatorDof);
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
        return String.format("FDistribution(d1=%.3f, d2=%.3f)", numeratorDof, denominatorDof);
    }
}
