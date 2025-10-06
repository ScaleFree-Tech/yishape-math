package com.yishape.lab.math.stats.bayes.conjugate;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.stats.distribution.GammaDistribution;
import com.yishape.lab.math.stats.distribution.NormalDistribution;

import java.util.Random;

/**
 * Inverse-Gamma共轭先验
 * Inverse-Gamma Conjugate Prior
 * 
 * <p>实现Inverse-Gamma分布作为正态分布方差的共轭先验。
 * 在观测到正态分布数据后，后验分布仍然是Inverse-Gamma分布。</p>
 * <p>Implements Inverse-Gamma distribution as conjugate prior for normal distribution variance.
 * After observing normal data, the posterior distribution remains Inverse-Gamma.</p>
 * 
 * <p>模型：
 * σ² ~ InverseGamma(α, β)
 * x_i | σ² ~ Normal(μ, σ²)
 * 后验：σ² | x ~ InverseGamma(α + n/2, β + Σ(x_i - μ)²/2)</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class InverseGammaConjugate {
    
    private double priorAlpha;    // α (形状参数)
    private double priorBeta;     // β (尺度参数)
    private double knownMean;     // 已知均值μ
    private boolean meanKnown;    // 是否已知均值
    
    private double posteriorAlpha;
    private double posteriorBeta;
    private boolean posteriorComputed;
    
    /**
     * 构造函数（已知均值）
     * Constructor (known mean)
     * 
     * @param priorAlpha 先验形状参数 α / Prior shape parameter α
     * @param priorBeta 先验尺度参数 β / Prior scale parameter β
     * @param knownMean 已知均值 μ / Known mean μ
     */
    public InverseGammaConjugate(double priorAlpha, double priorBeta, double knownMean) {
        if (priorAlpha <= 0 || priorBeta <= 0) {
            throw new IllegalArgumentException("Shape and scale parameters must be positive");
        }
        
        this.priorAlpha = priorAlpha;
        this.priorBeta = priorBeta;
        this.knownMean = knownMean;
        this.meanKnown = true;
        this.posteriorComputed = false;
    }
    
    /**
     * 构造函数（未知均值）
     * Constructor (unknown mean)
     * 
     * @param priorAlpha 先验形状参数 α / Prior shape parameter α
     * @param priorBeta 先验尺度参数 β / Prior scale parameter β
     */
    public InverseGammaConjugate(double priorAlpha, double priorBeta) {
        if (priorAlpha <= 0 || priorBeta <= 0) {
            throw new IllegalArgumentException("Shape and scale parameters must be positive");
        }
        
        this.priorAlpha = priorAlpha;
        this.priorBeta = priorBeta;
        this.meanKnown = false;
        this.posteriorComputed = false;
    }
    
    /**
     * 使用观测数据更新后验分布（已知均值）
     * Update posterior distribution with observed data (known mean)
     * 
     * @param observations 观测数据 / Observed data
     */
    public void updatePosterior(double[] observations) {
        if (!meanKnown) {
            throw new IllegalStateException("Mean is unknown. Use updatePosterior(observations, sampleMean) instead.");
        }
        
        if (observations == null || observations.length == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        int n = observations.length;
        
        // 计算平方偏差和
        double sumSquaredDeviations = 0.0;
        for (double obs : observations) {
            double deviation = obs - knownMean;
            sumSquaredDeviations += deviation * deviation;
        }
        
        // 更新后验参数
        this.posteriorAlpha = priorAlpha + n / 2.0;
        this.posteriorBeta = priorBeta + sumSquaredDeviations / 2.0;
        this.posteriorComputed = true;
    }
    
    /**
     * 使用观测数据更新后验分布（未知均值）
     * Update posterior distribution with observed data (unknown mean)
     * 
     * @param observations 观测数据 / Observed data
     * @param sampleMean 样本均值 / Sample mean
     */
    public void updatePosterior(double[] observations, double sampleMean) {
        if (observations == null || observations.length == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        int n = observations.length;
        
        // 计算平方偏差和
        double sumSquaredDeviations = 0.0;
        for (double obs : observations) {
            double deviation = obs - sampleMean;
            sumSquaredDeviations += deviation * deviation;
        }
        
        // 更新后验参数
        this.posteriorAlpha = priorAlpha + (n - 1) / 2.0;  // 自由度减1
        this.posteriorBeta = priorBeta + sumSquaredDeviations / 2.0;
        this.posteriorComputed = true;
    }
    
    /**
     * 使用观测数据更新后验分布（向量形式）
     * Update posterior distribution with observed data (vector form)
     * 
     * @param observations 观测数据向量 / Observed data vector
     */
    public void updatePosterior(IVector observations) {
        if (observations == null || observations.size() == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        double[] obsArray = new double[observations.size()];
        for (int i = 0; i < observations.size(); i++) {
            obsArray[i] = observations.get(i).doubleValue();
        }
        
        if (meanKnown) {
            updatePosterior(obsArray);
        } else {
            // 计算样本均值
            double sampleMean = 0.0;
            for (double obs : obsArray) {
                sampleMean += obs;
            }
            sampleMean /= obsArray.length;
            
            updatePosterior(obsArray, sampleMean);
        }
    }
    
    /**
     * 获取先验分布
     * Get prior distribution
     * 
     * @return Inverse-Gamma先验分布（通过Gamma分布的倒数实现） / Inverse-Gamma prior distribution
     */
    public InverseGammaDistribution getPriorDistribution() {
        return new InverseGammaDistribution(priorAlpha, priorBeta);
    }
    
    /**
     * 获取后验分布
     * Get posterior distribution
     * 
     * @return Inverse-Gamma后验分布 / Inverse-Gamma posterior distribution
     * @throws IllegalStateException 如果后验分布尚未计算 / If posterior not computed yet
     */
    public InverseGammaDistribution getPosteriorDistribution() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return new InverseGammaDistribution(posteriorAlpha, posteriorBeta);
    }
    
    /**
     * 从后验预测分布中采样
     * Sample from posterior predictive distribution
     * 
     * @param numSamples 采样数量 / Number of samples
     * @param random 随机数生成器 / Random number generator
     * @return 采样结果 / Sample results
     */
    public double[] samplePosteriorPredictive(int numSamples, Random random) {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        double[] samples = new double[numSamples];
        InverseGammaDistribution posteriorIG = getPosteriorDistribution();
        
        for (int i = 0; i < numSamples; i++) {
            // 1. 从后验Inverse-Gamma分布中采样σ²
            double variance = posteriorIG.sample(random);
            
            // 2. 从Normal(μ, σ²)中采样
            double mean = meanKnown ? knownMean : 0.0; // 如果均值未知，使用0作为默认值
            NormalDistribution normal = new NormalDistribution(mean, Math.sqrt(variance));
            samples[i] = normal.sample(1)[0];
        }
        
        return samples;
    }
    
    /**
     * 计算后验均值
     * Calculate posterior mean
     * 
     * @return 后验均值 / Posterior mean
     */
    public double getPosteriorMean() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        if (posteriorAlpha <= 1) {
            return Double.POSITIVE_INFINITY;
        }
        
        return posteriorBeta / (posteriorAlpha - 1);
    }
    
    /**
     * 计算后验方差
     * Calculate posterior variance
     * 
     * @return 后验方差 / Posterior variance
     */
    public double getPosteriorVariance() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        if (posteriorAlpha <= 2) {
            return Double.POSITIVE_INFINITY;
        }
        
        double numerator = posteriorBeta * posteriorBeta;
        double denominator = (posteriorAlpha - 1) * (posteriorAlpha - 1) * (posteriorAlpha - 2);
        
        return numerator / denominator;
    }
    
    /**
     * 计算后验众数
     * Calculate posterior mode
     * 
     * @return 后验众数 / Posterior mode
     */
    public double getPosteriorMode() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return posteriorBeta / (posteriorAlpha + 1);
    }
    
    /**
     * 计算可信区间
     * Calculate credible interval
     * 
     * @param alpha 显著性水平 / Significance level
     * @return 可信区间 [下界, 上界] / Credible interval [lower, upper]
     */
    public double[] getCredibleInterval(double alpha) {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        if (alpha <= 0 || alpha >= 1) {
            throw new IllegalArgumentException("Alpha must be between 0 and 1");
        }
        
        InverseGammaDistribution posterior = getPosteriorDistribution();
        double lowerQuantile = alpha / 2;
        double upperQuantile = 1 - alpha / 2;
        
        return new double[] {
            posterior.ppf(lowerQuantile),
            posterior.ppf(upperQuantile)
        };
    }
    
    /**
     * 计算边际似然（证据）
     * Calculate marginal likelihood (evidence)
     * 
     * @param observations 观测数据 / Observed data
     * @return 边际似然 / Marginal likelihood
     */
    public double calculateMarginalLikelihood(double[] observations) {
        if (observations == null || observations.length == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        int n = observations.length;
        double sumSquaredDeviations;
        
        if (meanKnown) {
            sumSquaredDeviations = 0.0;
            for (double obs : observations) {
                double deviation = obs - knownMean;
                sumSquaredDeviations += deviation * deviation;
            }
        } else {
            // 计算样本均值
            double sampleMean = 0.0;
            for (double obs : observations) {
                sampleMean += obs;
            }
            sampleMean /= n;
            
            sumSquaredDeviations = 0.0;
            for (double obs : observations) {
                double deviation = obs - sampleMean;
                sumSquaredDeviations += deviation * deviation;
            }
        }
        
        // 计算边际似然的对数
        double logMarginalLikelihood = 
            logGamma(priorAlpha + n / 2.0) - logGamma(priorAlpha) +
            priorAlpha * Math.log(priorBeta) -
            (priorAlpha + n / 2.0) * Math.log(priorBeta + sumSquaredDeviations / 2.0) -
            n / 2.0 * Math.log(2 * Math.PI);
        
        return Math.exp(logMarginalLikelihood);
    }
    
    /**
     * 重置为先验状态
     * Reset to prior state
     */
    public void reset() {
        this.posteriorComputed = false;
    }
    
    /**
     * 获取先验参数
     * Get prior parameters
     * 
     * @return [形状参数, 尺度参数] / [shape parameter, scale parameter]
     */
    public double[] getPriorParameters() {
        return new double[] { priorAlpha, priorBeta };
    }
    
    /**
     * 获取后验参数
     * Get posterior parameters
     * 
     * @return [形状参数, 尺度参数] / [shape parameter, scale parameter]
     * @throws IllegalStateException 如果后验分布尚未计算 / If posterior not computed yet
     */
    public double[] getPosteriorParameters() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return new double[] { posteriorAlpha, posteriorBeta };
    }
    
    /**
     * 是否已知均值
     * Whether mean is known
     */
    public boolean isMeanKnown() {
        return meanKnown;
    }
    
    /**
     * 获取已知均值
     * Get known mean
     */
    public double getKnownMean() {
        if (!meanKnown) {
            throw new IllegalStateException("Mean is unknown");
        }
        return knownMean;
    }
    
    /**
     * 是否已计算后验分布
     * Whether posterior distribution is computed
     */
    public boolean isPosteriorComputed() {
        return posteriorComputed;
    }
    
    /**
     * 计算对数Gamma函数
     * Calculate log Gamma function
     */
    private double logGamma(double x) {
        if (x <= 0) {
            throw new IllegalArgumentException("Gamma function undefined for non-positive values");
        }
        
        // 对于整数，使用阶乘
        if (x == Math.floor(x) && x <= 20) {
            return logFactorial((int) x - 1);
        }
        
        // Stirling近似
        return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI);
    }
    
    /**
     * 计算对数阶乘
     * Calculate log factorial
     */
    private double logFactorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("Factorial undefined for negative numbers");
        }
        
        if (n <= 1) {
            return 0.0;
        }
        
        double logFact = 0.0;
        for (int i = 2; i <= n; i++) {
            logFact += Math.log(i);
        }
        
        return logFact;
    }
    
    /**
     * Inverse-Gamma分布实现
     * Inverse-Gamma distribution implementation
     */
    public static class InverseGammaDistribution {
        private final double alpha;
        private final double beta;
        private final GammaDistribution gammaDistribution;
        
        public InverseGammaDistribution(double alpha, double beta) {
            if (alpha <= 0 || beta <= 0) {
                throw new IllegalArgumentException("Alpha and beta must be positive");
            }
            
            this.alpha = alpha;
            this.beta = beta;
            this.gammaDistribution = new GammaDistribution(alpha, 1.0 / beta);
        }
        
        /**
         * 从Inverse-Gamma分布中采样
         * Sample from Inverse-Gamma distribution
         */
        public double sample(Random random) {
            // 如果X ~ Gamma(α, β)，则1/X ~ InverseGamma(α, β)
            double gammaSample = gammaDistribution.sample(1)[0];
            return 1.0 / gammaSample;
        }
        
        /**
         * 计算概率密度函数
         * Calculate probability density function
         */
        public double pdf(double x) {
            if (x <= 0) {
                return 0.0;
            }
            
            double logPdf = alpha * Math.log(beta) - logGamma(alpha) - 
                          (alpha + 1) * Math.log(x) - beta / x;
            
            return Math.exp(logPdf);
        }
        
        /**
         * 计算累积分布函数
         * Calculate cumulative distribution function
         */
        public double cdf(double x) {
            if (x <= 0) {
                return 0.0;
            }
            
            // 使用Gamma分布的CDF：P(Y ≤ x) = P(1/X ≤ x) = P(X ≥ 1/x) = 1 - P(X < 1/x)
            return 1.0 - gammaDistribution.cdf(1.0 / x);
        }
        
        /**
         * 计算分位数函数
         * Calculate quantile function
         */
        public double ppf(double p) {
            if (p <= 0 || p >= 1) {
                throw new IllegalArgumentException("Probability must be between 0 and 1");
            }
            
            // 使用Gamma分布的PPF
            double gammaQuantile = gammaDistribution.ppf(1 - p);
            return 1.0 / gammaQuantile;
        }
        
        /**
         * 计算均值
         * Calculate mean
         */
        public double mean() {
            if (alpha <= 1) {
                return Double.POSITIVE_INFINITY;
            }
            return beta / (alpha - 1);
        }
        
        /**
         * 计算方差
         * Calculate variance
         */
        public double variance() {
            if (alpha <= 2) {
                return Double.POSITIVE_INFINITY;
            }
            return beta * beta / ((alpha - 1) * (alpha - 1) * (alpha - 2));
        }
        
        /**
         * 计算众数
         * Calculate mode
         */
        public double mode() {
            return beta / (alpha + 1);
        }
        
        private double logGamma(double x) {
            return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI);
        }
    }
}