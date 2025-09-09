package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import java.io.Serializable;

/**
 * Beta分布 (Beta Distribution)
 * 
 * Beta分布是定义在区间[0,1]上的连续概率分布，由两个形状参数α和β控制。
 * 其概率密度函数为：f(x) = (1/B(α,β)) * x^(α-1) * (1-x)^(β-1)
 * 其中B(α,β)是Beta函数，α和β是形状参数。
 * 
 * Beta distribution is a continuous probability distribution defined on the interval [0,1],
 * controlled by two shape parameters α and β. Its probability density function is:
 * f(x) = (1/B(α,β)) * x^(α-1) * (1-x)^(β-1)
 * where B(α,β) is the Beta function, and α and β are shape parameters.
 * 
 * @author lteb2
 */
public class BetaDistribution implements IContinuousDistribution, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** 形状参数α / Shape parameter α */
    private final float alpha;
    
    /** 形状参数β / Shape parameter β */
    private final float beta;
    
    /** Beta函数值B(α,β)的缓存 / Cached value of Beta function B(α,β) */
    private final float betaFunction;
    
    /** 均值 / Mean */
    private final float mean;
    
    /** 方差 / Variance */
    private final float variance;
    
    /** 标准差 / Standard deviation */
    private final float stdDev;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param alpha 形状参数α，必须大于0 / Shape parameter α, must be greater than 0
     * @param beta 形状参数β，必须大于0 / Shape parameter β, must be greater than 0
     * @throws IllegalArgumentException 如果参数小于等于0 / If parameters are less than or equal to 0
     */
    public BetaDistribution(float alpha, float beta) {
        if (alpha <= 0.0f) {
            throw new IllegalArgumentException("α参数必须大于0 / α parameter must be greater than 0");
        }
        if (beta <= 0.0f) {
            throw new IllegalArgumentException("β参数必须大于0 / β parameter must be greater than 0");
        }
        
        this.alpha = alpha;
        this.beta = beta;
        this.betaFunction = (float) RereMathUtil.beta(alpha, beta);
        
        // 计算统计量
        this.mean = alpha / (alpha + beta);
        this.variance = (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1.0f));
        this.stdDev = (float) Math.sqrt(variance);
    }
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param x 输入值，必须在[0,1]范围内 / Input value, must be in range [0,1]
     * @return 概率密度函数值 / PDF value
     */
    @Override
    public float pdf(float x) {
        if (x < 0.0f || x > 1.0f) {
            return 0.0f;
        }
        
        if (Float.isInfinite(x) || Float.isNaN(x)) {
            return 0.0f;
        }
        
        if (x == 0.0f && alpha < 1.0f) {
            return Float.POSITIVE_INFINITY;
        }
        if (x == 1.0f && beta < 1.0f) {
            return Float.POSITIVE_INFINITY;
        }
        
        if (x == 0.0f || x == 1.0f) {
            return 0.0f;
        }
        
        double logPdf = (alpha - 1.0f) * Math.log(x) + (beta - 1.0f) * Math.log(1.0f - x) - Math.log(betaFunction);
        return (float) Math.exp(logPdf);
    }
    
    /**
     * 计算累积分布函数值
     * Calculate cumulative distribution function value
     * 
     * @param x 输入值，必须在[0,1]范围内 / Input value, must be in range [0,1]
     * @return 累积分布函数值 / CDF value
     */
    @Override
    public float cdf(float x) {
        if (Float.isInfinite(x)) {
            if (x == Float.NEGATIVE_INFINITY) return 0.0f;
            if (x == Float.POSITIVE_INFINITY) return 1.0f;
        }
        if (Float.isNaN(x)) return Float.NaN;
        
        if (x < 0.0f) return 0.0f;
        if (x > 1.0f) return 1.0f;
        if (x == 0.0f) return 0.0f;
        if (x == 1.0f) return 1.0f;
        
        return (float) RereMathUtil.incompleteBeta(alpha, beta, x) / betaFunction;
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
        if (p == 1.0f) return 1.0f;
        
        // 使用二分法求解逆Beta分布
        // Using binary search to solve inverse Beta distribution
        return inverseBetaCDF(p);
    }
    
    /**
     * 使用二分法计算逆Beta CDF
     * Calculate inverse Beta CDF using binary search
     */
    private float inverseBetaCDF(float p) {
        float low = 0.0f;
        float high = 1.0f;
        float tolerance = 1e-6f;
        int maxIter = 100;
        
        for (int i = 0; i < maxIter; i++) {
            float mid = (low + high) / 2.0f;
            float cdfValue = cdf(mid);
            
            if (Math.abs(cdfValue - p) < tolerance) {
                return mid;
            }
            
            if (cdfValue < p) {
                low = mid;
            } else {
                high = mid;
            }
        }
        
        return (low + high) / 2.0f;
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
        if (alpha <= 1.0f && beta <= 1.0f) {
            // 当α≤1且β≤1时，众数在端点处
            return alpha < beta ? 0.0f : (alpha > beta ? 1.0f : Float.NaN);
        }
        return (alpha - 1.0f) / (alpha + beta - 2.0f);
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
        float numerator = 2.0f * (beta - alpha) * (float) Math.sqrt(alpha + beta + 1.0f);
        float denominator = (alpha + beta + 2.0f) * (float) Math.sqrt(alpha * beta);
        return numerator / denominator;
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public float kurtosis() {
        float aPlusB = alpha + beta;
        float numerator = 6.0f * ((alpha - beta) * (alpha - beta) * (aPlusB + 1.0f) - alpha * beta * (aPlusB + 2.0f));
        float denominator = alpha * beta * (aPlusB + 2.0f) * (aPlusB + 3.0f);
        return numerator / denominator;
    }
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public float sample() {
        // 使用Gamma分布生成Beta分布样本
        // Using Gamma distribution to generate Beta distribution samples
        float gamma1 = gammaSample(alpha, 1.0f);
        float gamma2 = gammaSample(beta, 1.0f);
        return gamma1 / (gamma1 + gamma2);
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
        
        IVector samples = IVector.zeros(n);
        for (int i = 0; i < n; i++) {
            samples.set(i, sample());
        }
        return samples.getData();
    }
    
    /**
     * 使用Box-Muller变换生成Gamma分布样本
     * Generate Gamma distribution sample using Box-Muller transform
     */
    private float gammaSample(float shape, float scale) {
        if (shape < 1.0f) {
            // 对于形状参数小于1的情况，使用变换
            return gammaSample(shape + 1.0f, scale) * (float) Math.pow(Math.random(), 1.0f / shape);
        }
        
        // 使用Marsaglia和Tsang的方法
        float d = shape - 1.0f / 3.0f;
        float c = 1.0f / (float) Math.sqrt(9.0f * d);
        
        while (true) {
            float x = (float) RereMathUtil.normalSample(0.0f, 1.0f);
            float v = 1.0f + c * x;
            
            if (v <= 0.0f) continue;
            
            v = v * v * v;
            float u = (float) Math.random();
            
            if (u < 1.0f - 0.0331f * x * x * x * x) {
                return d * v * scale;
            }
            
            if (Math.log(u) < 0.5f * x * x + d * (1.0f - v + Math.log(v))) {
                return d * v * scale;
            }
        }
    }
    
    /**
     * 获取形状参数α
     * Get shape parameter α
     * 
     * @return 形状参数α / Shape parameter α
     */
    public float getAlpha() {
        return alpha;
    }
    
    /**
     * 获取形状参数β
     * Get shape parameter β
     * 
     * @return 形状参数β / Shape parameter β
     */
    public float getBeta() {
        return beta;
    }
    
    @Override
    public String toString() {
        return String.format("BetaDistribution(α=%.3f, β=%.3f)", alpha, beta);
    }
}
