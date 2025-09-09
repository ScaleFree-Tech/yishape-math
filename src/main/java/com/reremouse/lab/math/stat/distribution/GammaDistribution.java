package com.reremouse.lab.math.stat.distribution;

import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.math.IVector;
import java.io.Serializable;

/**
 * Gamma分布 (Gamma Distribution)
 * 
 * Gamma分布是连续概率分布，由形状参数α和尺度参数β控制。
 * 其概率密度函数为：f(x) = (β^α / Γ(α)) * x^(α-1) * e^(-βx)
 * 其中Γ(α)是Gamma函数，α是形状参数，β是尺度参数。
 * 
 * Gamma distribution is a continuous probability distribution controlled by 
 * shape parameter α and scale parameter β. Its probability density function is:
 * f(x) = (β^α / Γ(α)) * x^(α-1) * e^(-βx)
 * where Γ(α) is the Gamma function, α is the shape parameter, and β is the scale parameter.
 * 
 * @author lteb2
 */
public class GammaDistribution implements IContinuousDistribution, Serializable {
    
    private static final long serialVersionUID = 1L;
    
    /** 形状参数α / Shape parameter α */
    private final float alpha;
    
    /** 尺度参数β / Scale parameter β */
    private final float beta;
    
    /** Gamma函数值Γ(α)的缓存 / Cached value of Gamma function Γ(α) */
    private final float gammaFunction;
    
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
     * @param beta 尺度参数β，必须大于0 / Scale parameter β, must be greater than 0
     * @throws IllegalArgumentException 如果参数小于等于0 / If parameters are less than or equal to 0
     */
    public GammaDistribution(float alpha, float beta) {
        if (alpha <= 0.0f) {
            throw new IllegalArgumentException("α参数必须大于0 / α parameter must be greater than 0");
        }
        if (beta <= 0.0f) {
            throw new IllegalArgumentException("β参数必须大于0 / β parameter must be greater than 0");
        }
        
        this.alpha = alpha;
        this.beta = beta;
        this.gammaFunction = (float) RereMathUtil.gamma(alpha);
        
        // 计算统计量
        this.mean = alpha / beta;
        this.variance = alpha / (beta * beta);
        this.stdDev = (float) Math.sqrt(variance);
    }
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param x 输入值，必须大于0 / Input value, must be greater than 0
     * @return 概率密度函数值 / PDF value
     */
    @Override
    public float pdf(float x) {
        if (x <= 0.0f) {
            return 0.0f;
        }
        
        if (Float.isInfinite(x) || Float.isNaN(x)) {
            return 0.0f;
        }
        
        if (x == 0.0f && alpha < 1.0f) {
            return Float.POSITIVE_INFINITY;
        }
        
        double logPdf = alpha * Math.log(beta) - Math.log(gammaFunction) + 
                       (alpha - 1.0f) * Math.log(x) - beta * x;
        return (float) Math.exp(logPdf);
    }
    
    /**
     * 计算累积分布函数值
     * Calculate cumulative distribution function value
     * 
     * @param x 输入值，必须大于等于0 / Input value, must be greater than or equal to 0
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
        if (x == 0.0f) return 0.0f;
        
        return (float) RereMathUtil.incompleteGamma(alpha, beta * x) / gammaFunction;
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
        
        // 使用二分法求解逆Gamma分布
        // Using binary search to solve inverse Gamma distribution
        return inverseGammaCDF(p);
    }
    
    /**
     * 使用二分法计算逆Gamma CDF
     * Calculate inverse Gamma CDF using binary search
     */
    private float inverseGammaCDF(float p) {
        float low = 0.0f;
        float high = mean + 10.0f * stdDev; // 使用均值+10倍标准差作为上界
        float tolerance = 1e-6f;
        int maxIter = 100;
        
        // 如果概率很小，调整上界
        if (p < 0.01f) {
            high = mean + 20.0f * stdDev;
        }
        
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
        if (alpha < 1.0f) {
            return 0.0f;
        }
        return (alpha - 1.0f) / beta;
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
        return 2.0f / (float) Math.sqrt(alpha);
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public float kurtosis() {
        return 6.0f / alpha;
    }
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public float sample() {
        return gammaSample(alpha, beta);
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
     * 使用Marsaglia和Tsang的方法生成Gamma分布样本
     * Generate Gamma distribution sample using Marsaglia and Tsang method
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
                return d * v / scale;
            }
            
            if (Math.log(u) < 0.5f * x * x + d * (1.0f - v + Math.log(v))) {
                return d * v / scale;
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
     * 获取尺度参数β
     * Get scale parameter β
     * 
     * @return 尺度参数β / Scale parameter β
     */
    public float getBeta() {
        return beta;
    }
    
    @Override
    public String toString() {
        return String.format("GammaDistribution(α=%.3f, β=%.3f)", alpha, beta);
    }
}
