package com.reremouse.lab.math.stats.bayes.conjugate;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.stats.Stats;
import com.reremouse.lab.math.stats.distribution.GammaDistribution;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import com.reremouse.lab.math.stats.distribution.StudentDistribution;
import java.util.Random;

/**
 * Normal-Gamma共轭先验
 * Normal-Gamma Conjugate Prior
 * 
 * <p>实现Normal-Gamma分布作为正态分布均值和精度的共轭先验。
 * 在观测到正态分布数据后，后验分布仍然是Normal-Gamma分布。</p>
 * <p>Implements Normal-Gamma distribution as conjugate prior for normal distribution mean and precision.
 * After observing normal data, the posterior distribution remains Normal-Gamma.</p>
 * 
 * <p>模型：
 * τ ~ Gamma(α, β)
 * μ | τ ~ Normal(μ₀, (λτ)⁻¹)
 * x_i | μ, τ ~ Normal(μ, τ⁻¹)
 * 后验：(μ, τ) | x ~ NormalGamma(μₙ, λₙ, αₙ, βₙ)</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class NormalGammaConjugate {
    
    // 先验参数
    private double priorMu0;      // μ₀ (先验均值)
    private double priorLambda;   // λ (先验精度参数)
    private double priorAlpha;    // α (Gamma形状参数)
    private double priorBeta;     // β (Gamma率参数)
    
    // 后验参数
    private double posteriorMu;
    private double posteriorLambda;
    private double posteriorAlpha;
    private double posteriorBeta;
    private boolean posteriorComputed;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param priorMu0 先验均值 μ₀ / Prior mean μ₀
     * @param priorLambda 先验精度参数 λ / Prior precision parameter λ
     * @param priorAlpha Gamma形状参数 α / Gamma shape parameter α
     * @param priorBeta Gamma率参数 β / Gamma rate parameter β
     */
    public NormalGammaConjugate(double priorMu0, double priorLambda, 
                               double priorAlpha, double priorBeta) {
        if (priorLambda <= 0 || priorAlpha <= 0 || priorBeta <= 0) {
            throw new IllegalArgumentException("Lambda, alpha, and beta parameters must be positive");
        }
        
        this.priorMu0 = priorMu0;
        this.priorLambda = priorLambda;
        this.priorAlpha = priorAlpha;
        this.priorBeta = priorBeta;
        this.posteriorComputed = false;
    }
    
    /**
     * 使用观测数据更新后验分布
     * Update posterior distribution with observed data
     * 
     * @param observations 观测数据 / Observed data
     */
    public void updatePosterior(double[] observations) {
        if (observations == null || observations.length == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        int n = observations.length;
        
        // 计算样本统计量
        double sampleMean = 0.0;
        for (double obs : observations) {
            sampleMean += obs;
        }
        sampleMean /= n;
        
        double sampleVariance = 0.0;
        for (double obs : observations) {
            sampleVariance += (obs - sampleMean) * (obs - sampleMean);
        }
        
        // 更新后验参数
        this.posteriorLambda = priorLambda + n;
        this.posteriorMu = (priorLambda * priorMu0 + n * sampleMean) / posteriorLambda;
        this.posteriorAlpha = priorAlpha + n / 2.0;
        this.posteriorBeta = priorBeta + 0.5 * sampleVariance + 
                           0.5 * priorLambda * n * (sampleMean - priorMu0) * (sampleMean - priorMu0) / posteriorLambda;
        
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
        
        updatePosterior(obsArray);
    }
    
    /**
     * 从先验分布中采样 (μ, τ)
     * Sample (μ, τ) from prior distribution
     * 
     * @param random 随机数生成器 / Random number generator
     * @return [μ, τ] 采样结果 / [μ, τ] sample result
     */
    public double[] samplePrior(Random random) {
        // 1. 从Gamma分布中采样τ
        GammaDistribution gammaDistribution = new GammaDistribution(priorAlpha, 1.0 / priorBeta);
        double tau = gammaDistribution.sample(1)[0];
        
        // 2. 从Normal分布中采样μ | τ
        double variance = 1.0 / (priorLambda * tau);
        NormalDistribution normalDistribution = new NormalDistribution(priorMu0, Math.sqrt(variance));
        double mu = normalDistribution.sample(1)[0];
        
        return new double[] { mu, tau };
    }
    
    /**
     * 从后验分布中采样 (μ, τ)
     * Sample (μ, τ) from posterior distribution
     * 
     * @param random 随机数生成器 / Random number generator
     * @return [μ, τ] 采样结果 / [μ, τ] sample result
     */
    public double[] samplePosterior(Random random) {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        // 1. 从后验Gamma分布中采样τ
        GammaDistribution gammaDistribution = new GammaDistribution(posteriorAlpha, 1.0 / posteriorBeta);
        double tau = gammaDistribution.sample(1)[0];
        
        // 2. 从后验Normal分布中采样μ | τ
        double variance = 1.0 / (posteriorLambda * tau);
        NormalDistribution normalDistribution = new NormalDistribution(posteriorMu, Math.sqrt(variance));
        double mu = normalDistribution.sample(1)[0];
        
        return new double[] { mu, tau };
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
        
        for (int i = 0; i < numSamples; i++) {
            // 1. 从后验分布中采样 (μ, τ)
            double[] params = samplePosterior(random);
            double mu = params[0];
            double tau = params[1];
            
            // 2. 从Normal(μ, τ⁻¹)中采样
            NormalDistribution normalDistribution = new NormalDistribution(mu, Math.sqrt(1.0 / tau));
            samples[i] = normalDistribution.sample(1)[0];
        }
        
        return samples;
    }
    
    /**
     * 计算后验预测分布（Student-t分布）
     * Calculate posterior predictive distribution (Student-t distribution)
     * 
     * @return Student-t分布 / Student-t distribution
     */
    public StudentDistribution getPosteriorPredictiveDistribution() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        // 后验预测分布是Student-t分布
        // 自由度：ν = 2α
        // 位置参数：μ
        // 尺度参数：√(β(λ+1)/(αλ))
        
        double nu = 2 * posteriorAlpha;
        double location = posteriorMu;
        double scale = Math.sqrt(posteriorBeta * (posteriorLambda + 1) / (posteriorAlpha * posteriorLambda));
        
        return new StudentDistribution(nu, location, scale);
    }
    
    /**
     * 计算后验均值的边际分布（Student-t分布）
     * Calculate marginal distribution of posterior mean (Student-t distribution)
     * 
     * @return Student-t分布 / Student-t distribution
     */
    public StudentDistribution getPosteriorMeanDistribution() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        double nu = 2 * posteriorAlpha;
        double location = posteriorMu;
        double scale = Math.sqrt(posteriorBeta / (posteriorAlpha * posteriorLambda));
        
        return Stats.t(nu);
    }
    
    /**
     * 计算后验精度的边际分布（Gamma分布）
     * Calculate marginal distribution of posterior precision (Gamma distribution)
     * 
     * @return Gamma分布 / Gamma distribution
     */
    public GammaDistribution getPosteriorPrecisionDistribution() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return new GammaDistribution(posteriorAlpha, 1.0 / posteriorBeta);
    }
    
    /**
     * 计算后验均值的期望
     * Calculate expectation of posterior mean
     * 
     * @return 后验均值的期望 / Expectation of posterior mean
     */
    public double getPosteriorMeanExpectation() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return posteriorMu;
    }
    
    /**
     * 计算后验精度的期望
     * Calculate expectation of posterior precision
     * 
     * @return 后验精度的期望 / Expectation of posterior precision
     */
    public double getPosteriorPrecisionExpectation() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return posteriorAlpha / posteriorBeta;
    }
    
    /**
     * 计算后验方差的期望
     * Calculate expectation of posterior variance
     * 
     * @return 后验方差的期望 / Expectation of posterior variance
     */
    public double getPosteriorVarianceExpectation() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return posteriorBeta / (posteriorAlpha - 1);
    }
    
    /**
     * 计算后验均值的方差
     * Calculate variance of posterior mean
     * 
     * @return 后验均值的方差 / Variance of posterior mean
     */
    public double getPosteriorMeanVariance() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        if (posteriorAlpha <= 1) {
            return Double.POSITIVE_INFINITY;
        }
        
        return posteriorBeta / ((posteriorAlpha - 1) * posteriorLambda);
    }
    
    /**
     * 计算可信区间
     * Calculate credible interval
     * 
     * @param alpha 显著性水平 / Significance level
     * @param parameter 参数类型："mean"或"precision" / Parameter type: "mean" or "precision"
     * @return 可信区间 [下界, 上界] / Credible interval [lower, upper]
     */
    public double[] getCredibleInterval(double alpha, String parameter) {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        if (alpha <= 0 || alpha >= 1) {
            throw new IllegalArgumentException("Alpha must be between 0 and 1");
        }
        
        double lowerQuantile = alpha / 2;
        double upperQuantile = 1 - alpha / 2;
        
        if ("mean".equalsIgnoreCase(parameter)) {
            StudentDistribution meanDist = getPosteriorMeanDistribution();
            return new double[] {
                meanDist.ppf(lowerQuantile),
                meanDist.ppf(upperQuantile)
            };
        } else if ("precision".equalsIgnoreCase(parameter)) {
            GammaDistribution precisionDist = getPosteriorPrecisionDistribution();
            return new double[] {
                precisionDist.ppf(lowerQuantile),
                precisionDist.ppf(upperQuantile)
            };
        } else {
            throw new IllegalArgumentException("Parameter must be 'mean' or 'precision'");
        }
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
        
        // 计算样本统计量
        double sampleMean = 0.0;
        for (double obs : observations) {
            sampleMean += obs;
        }
        sampleMean /= n;
        
        double sampleVariance = 0.0;
        for (double obs : observations) {
            sampleVariance += (obs - sampleMean) * (obs - sampleMean);
        }
        
        // 计算边际似然
        double logMarginalLikelihood = 
            0.5 * Math.log(priorLambda / (priorLambda + n)) +
            logGamma(priorAlpha + n / 2.0) - logGamma(priorAlpha) +
            priorAlpha * Math.log(priorBeta) -
            (priorAlpha + n / 2.0) * Math.log(priorBeta + 0.5 * sampleVariance + 
                0.5 * priorLambda * n * (sampleMean - priorMu0) * (sampleMean - priorMu0) / (priorLambda + n)) -
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
     * @return [μ₀, λ, α, β] / [μ₀, λ, α, β]
     */
    public double[] getPriorParameters() {
        return new double[] { priorMu0, priorLambda, priorAlpha, priorBeta };
    }
    
    /**
     * 获取后验参数
     * Get posterior parameters
     * 
     * @return [μₙ, λₙ, αₙ, βₙ] / [μₙ, λₙ, αₙ, βₙ]
     * @throws IllegalStateException 如果后验分布尚未计算 / If posterior not computed yet
     */
    public double[] getPosteriorParameters() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return new double[] { posteriorMu, posteriorLambda, posteriorAlpha, posteriorBeta };
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
     * 是否已计算后验分布
     * Whether posterior distribution is computed
     */
    public boolean isPosteriorComputed() {
        return posteriorComputed;
    }
}