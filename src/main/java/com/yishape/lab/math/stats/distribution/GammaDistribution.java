package com.yishape.lab.math.stats.distribution;

import com.yishape.lab.math.RereMathUtil;
import java.io.Serializable;
import com.yishape.lab.math.linalg.IDoubleVector;

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
    private final double alpha;
    
    /** 尺度参数β / Scale parameter β */
    private final double beta;
    
    /** Gamma函数值Γ(α)的缓存 / Cached value of Gamma function Γ(α) */
    private final double gammaFunction;
    
    /** 均值 / Mean */
    private final double mean;
    
    /** 方差 / Variance */
    private final double variance;
    
    /** 标准差 / Standard deviation */
    private final double stdDev;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param alpha 形状参数α，必须大于0 / Shape parameter α, must be greater than 0
     * @param beta 尺度参数β，必须大于0 / Scale parameter β, must be greater than 0
     * @throws IllegalArgumentException 如果参数小于等于0 / If parameters are less than or equal to 0
     */
    public GammaDistribution(double alpha, double beta) {
        if (alpha <= 0.0) {
            throw new IllegalArgumentException("α参数必须大于0 / α parameter must be greater than 0");
        }
        if (beta <= 0.0) {
            throw new IllegalArgumentException("β参数必须大于0 / β parameter must be greater than 0");
        }
        
        this.alpha = alpha;
        this.beta = beta;
        this.gammaFunction = RereMathUtil.gamma(alpha);
        
        // 计算统计量
        this.mean = alpha / beta;
        this.variance = alpha / (beta * beta);
        this.stdDev = Math.sqrt(variance);
    }
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param x 非负自变量；在 x=0 处按极限定义：α&lt;1 为 +∞，α=1 为 β，α&gt;1 为 0。
     * Non-negative argument; at x=0: limit +∞ if α&lt;1, β if α=1, 0 if α&gt;1.
     * @return 概率密度函数值 / PDF value
     */
    @Override
    public double pdf(double x) {
        if (Double.isNaN(x)) {
            return Double.NaN;
        }
        if (x < 0.0) {
            return 0.0;
        }
        if (Double.isInfinite(x)) {
            return x > 0.0 ? 0.0 : 0.0;
        }
        if (x == 0.0) {
            if (alpha < 1.0) {
                return Double.POSITIVE_INFINITY;
            }
            if (alpha == 1.0) {
                return beta;
            }
            return 0.0;
        }
        double logPdf = alpha * Math.log(beta) - Math.log(gammaFunction)
                + (alpha - 1.0) * Math.log(x) - beta * x;
        return Math.exp(logPdf);
    }
    
    /**
     * 计算累积分布函数值
     * Calculate cumulative distribution function value
     * 
     * @param x 输入值，必须大于等于0 / Input value, must be greater than or equal to 0
     * @return 累积分布函数值 / CDF value
     */
    @Override
    public double cdf(double x) {
        if (Double.isInfinite(x)) {
            if (x == Double.NEGATIVE_INFINITY) return 0.0;
            if (x == Double.POSITIVE_INFINITY) return 1.0;
        }
        if (Double.isNaN(x)) return Double.NaN;
        
        if (x < 0.0) return 0.0;
        if (x == 0.0) return 0.0;
        
        return RereMathUtil.regularizedIncompleteGamma(alpha, beta * x);
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
        
        if (p == 0.0) return 0.0;
        if (p == 1.0) return Double.POSITIVE_INFINITY;
        
        // 使用二分法求解逆Gamma分布
        // Using binary search to solve inverse Gamma distribution
        return inverseGammaCDF(p);
    }
    
    /**
     * 使用二分法计算逆Gamma CDF
     * Calculate inverse Gamma CDF using binary search
     */
    private double inverseGammaCDF(double p) {
        double low = 0.0;
        double high = mean + 10.0 * stdDev; // 使用均值+10倍标准差作为上界
        double tolerance = 1e-6;
        int maxIter = 100;
        
        // 如果概率很小，调整上界
        if (p < 0.01) {
            high = mean + 20.0 * stdDev;
        }
        
        for (int i = 0; i < maxIter; i++) {
            double mid = (low + high) / 2.0;
            double cdfValue = cdf(mid);
            
            if (Math.abs(cdfValue - p) < tolerance) {
                return mid;
            }
            
            if (cdfValue < p) {
                low = mid;
            } else {
                high = mid;
            }
        }
        
        return (low + high) / 2.0;
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
        return ppf(0.5);
    }
    
    /**
     * 获取众数
     * Get mode
     * 
     * @return 众数 / Mode
     */
    @Override
    public double mode() {
        if (alpha < 1.0) {
            return 0.0;
        }
        return (alpha - 1.0) / beta;
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
        return 2.0 / Math.sqrt(alpha);
    }
    
    /**
     * 获取峰度
     * Get kurtosis
     * 
     * @return 峰度 / Kurtosis
     */
    @Override
    public double kurtosis() {
        return 6.0 / alpha;
    }
    
    /**
     * 生成一个随机样本
     * Generate a random sample
     * 
     * @return 随机样本 / Random sample
     */
    @Override
    public double sample() {
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
    public double[] sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0 / Sample size must be greater than 0");
        }
        
        IDoubleVector samples = IDoubleVector.zeros(n);
        for (int i = 0; i < n; i++) {
            samples.set(i, sample());
        }
        return samples.getData();
    }
    
    /**
     * 使用Marsaglia和Tsang的方法生成Gamma分布样本
     * Generate Gamma distribution sample using Marsaglia and Tsang method
     */
    private double gammaSample(double shape, double scale) {
        if (shape < 1.0) {
            // 对于形状参数小于1的情况，使用变换
            return gammaSample(shape + 1.0, scale) * Math.pow(Math.random(), 1.0 / shape);
        }
        
        // 使用Marsaglia和Tsang的方法
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        
        while (true) {
            double x = RereMathUtil.normalSample(0.0, 1.0);
            double v = 1.0 + c * x;
            
            if (v <= 0.0) continue;
            
            v = v * v * v;
            double u = Math.random();
            
            if (u < 1.0 - 0.0331 * x * x * x * x) {
                return d * v / scale;
            }
            
            if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) {
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
    public double getAlpha() {
        return alpha;
    }
    
    /**
     * 获取尺度参数β
     * Get scale parameter β
     * 
     * @return 尺度参数β / Scale parameter β
     */
    public double getBeta() {
        return beta;
    }
    
    @Override
    public String toString() {
        return String.format("GammaDistribution(α=%.3f, β=%.3f)", alpha, beta);
    }
}
