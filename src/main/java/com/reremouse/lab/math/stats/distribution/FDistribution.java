package com.reremouse.lab.math.stats.distribution;

import com.reremouse.lab.math.RereMathUtil;
import java.io.Serializable;
import com.reremouse.lab.math.linalg.IDoubleVector;

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
    private final double numeratorDof;
    
    /** 分母自由度 / Denominator degrees of freedom */
    private final double denominatorDof;
    
    /** 预计算的常数 / Precomputed constants */
    private final double halfNumeratorDof;
    private final double halfDenominatorDof;
    private final double halfSumDof;
    private final double normalizationConstant;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param numeratorDof 分子自由度，必须大于0 / Numerator degrees of freedom, must be greater than 0
     * @param denominatorDof 分母自由度，必须大于0 / Denominator degrees of freedom, must be greater than 0
     * @throws IllegalArgumentException 如果自由度小于等于0 / If degrees of freedom is less than or equal to 0
     */
    public FDistribution(double numeratorDof, double denominatorDof) {
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
        this.normalizationConstant =  (RereMathUtil.gamma(halfSumDof) / 
            (RereMathUtil.gamma(halfNumeratorDof) * RereMathUtil.gamma(halfDenominatorDof)) * 
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
    public double pdf(double x) {
        if (x <= 0) {
            return 0.0f;
        }
        
        double power1 = halfNumeratorDof - 1.0f;
        double power2 = -(halfSumDof);
        double base = 1.0f + (numeratorDof * x) / denominatorDof;
        
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
    public double cdf(double x) {
        if (x <= 0) {
            return 0.0f;
        }
        
        // 使用正则化不完全贝塔函数
        // Using regularized incomplete beta function
        double t = (numeratorDof * x) / (denominatorDof + numeratorDof * x);
        return RereMathUtil.regularizedIncompleteBeta(halfNumeratorDof, halfDenominatorDof, t);
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
        if (p == 1.0f) return Double.POSITIVE_INFINITY;
        
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
     * 获取分子自由度
     * Get numerator degrees of freedom
     * 
     * @return 分子自由度 / Numerator degrees of freedom
     */
    public double getNumeratorDof() {
        return numeratorDof;
    }
    
    /**
     * 获取分母自由度
     * Get denominator degrees of freedom
     * 
     * @return 分母自由度 / Denominator degrees of freedom
     */
    public double getDenominatorDof() {
        return denominatorDof;
    }
    
    /**
     * 获取均值
     * Get mean
     * 
     * @return 均值 / Mean
     */
    public double getMean() {
        if (denominatorDof > 2) {
            return denominatorDof / (denominatorDof - 2.0f);
        }
        return Double.NaN; // 当分母自由度 <= 2 时均值不存在
    }
    
    /**
     * 获取方差
     * Get variance
     * 
     * @return 方差 / Variance
     */
    public double getVariance() {
        if (denominatorDof > 4) {
            double numerator = 2.0f * denominatorDof * denominatorDof * (numeratorDof + denominatorDof - 2.0f);
            double denominator = numeratorDof * (denominatorDof - 2.0f) * (denominatorDof - 2.0f) * (denominatorDof - 4.0f);
            return numerator / denominator;
        }
        return Double.NaN; // 当分母自由度 <= 4 时方差不存在
    }
    
    /**
     * 获取标准差
     * Get standard deviation
     * 
     * @return 标准差 / Standard deviation
     */
    public double getStandardDeviation() {
        double variance = getVariance();
        if (Double.isNaN(variance)) {
            return Double.NaN;
        }
        return (float) Math.sqrt(variance);
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    public double getMode() {
        if (numeratorDof > 2) {
            return (denominatorDof * (numeratorDof - 2.0f)) / (numeratorDof * (denominatorDof + 2.0f));
        }
        return 0.0f;
    }
    
    // 使用RereMathUtil中的gamma函数
    // Using gamma function from RereMathUtil
    
    // 使用RereMathUtil中的incompleteBeta函数
    // Using incompleteBeta function from RereMathUtil
    
    // 使用RereMathUtil中的betaCF函数
    // Using betaCF function from RereMathUtil
    
    /**
     * 逆F分布累积分布函数的数值求解
     * Numerical solution for inverse F-distribution CDF
     */
    private double inverseFCDF(double p) {
        // 使用改进的二分法求解
        // Using improved bisection method to solve
        double left = 0.0f;
        double right = 1.0f; // 从较小的初始值开始
        double tolerance = 1e-8f;
        int maxIter = 200;
        
        // 调整右边界直到CDF(right) >= p
        // Adjust right boundary until CDF(right) >= p
        int attempts = 0;
        while (cdf(right) < p && attempts < 20) {
            right *= 2.0f;
            attempts++;
        }
        
        // 如果右边界调整失败，使用更大的初始值
        if (cdf(right) < p) {
            right = 100.0f;
            while (cdf(right) < p && right < 10000.0f) {
                right *= 2.0f;
            }
        }
        
        // 确保左边界CDF < p
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
            
            // 检查收敛
            if (right - left < tolerance) {
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
        return ppf(0.5f);
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
        if (denominatorDof > 6) {
            double numerator = (2.0f * numeratorDof + denominatorDof - 2.0f) * 
                (float) Math.sqrt(8.0f * (denominatorDof - 4.0f));
            double denominator = (denominatorDof - 6.0f) * 
                (float) Math.sqrt(numeratorDof * (numeratorDof + denominatorDof - 2.0f));
            return numerator / denominator;
        }
        return Double.NaN; // 当分母自由度 <= 6 时偏度不存在
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public double kurtosis() {
        if (denominatorDof > 8) {
            double n1 = numeratorDof;
            double n2 = denominatorDof;
            double numerator = 12.0f * n1 * (n1 + n2 - 2.0f) * (n1 * (n2 - 2.0f) + n2 * (n2 - 4.0f));
            double denominator = n1 * (n2 - 6.0f) * (n2 - 8.0f) * (n1 + n2 - 2.0f);
            return numerator / denominator;
        }
        return Double.NaN; // 当分母自由度 <= 8 时峰度不存在
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
    public double sample() {
        // 使用卡方分布生成F分布随机数
        // Using chi-squared distributions to generate F-distribution random numbers
        if (chi2Num == null) {
            chi2Num = new Chi2Distribution(numeratorDof);
        }
        if (chi2Den == null) {
            chi2Den = new Chi2Distribution(denominatorDof);
        }
        
        double chi2NumSample = chi2Num.sample();
        double chi2DenSample = chi2Den.sample();
        
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
        return String.format("FDistribution(d1=%.3f, d2=%.3f)", numeratorDof, denominatorDof);
    }
}
