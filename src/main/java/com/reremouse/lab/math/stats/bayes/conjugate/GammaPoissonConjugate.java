package com.reremouse.lab.math.stats.bayes.conjugate;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.stats.distribution.GammaDistribution;
import com.reremouse.lab.math.stats.distribution.PoissonDistribution;
import java.util.Random;

/**
 * Gamma-Poisson共轭先验
 * Gamma-Poisson Conjugate Prior
 * 
 * <p>实现Gamma分布作为Poisson分布参数的共轭先验。
 * 在观测到Poisson数据后，后验分布仍然是Gamma分布。</p>
 * <p>Implements Gamma distribution as conjugate prior for Poisson distribution parameter.
 * After observing Poisson data, the posterior distribution remains Gamma.</p>
 * 
 * <p>模型：
 * λ ~ Gamma(α, β)
 * x_i | λ ~ Poisson(λ)
 * 后验：λ | x ~ Gamma(α + Σx_i, β + n)</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GammaPoissonConjugate {
    
    private double priorShape;    // α (形状参数)
    private double priorRate;     // β (率参数)
    private double posteriorShape;
    private double posteriorRate;
    private boolean posteriorComputed;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param priorShape 先验形状参数 α / Prior shape parameter α
     * @param priorRate 先验率参数 β / Prior rate parameter β
     */
    public GammaPoissonConjugate(double priorShape, double priorRate) {
        if (priorShape <= 0 || priorRate <= 0) {
            throw new IllegalArgumentException("Shape and rate parameters must be positive");
        }
        
        this.priorShape = priorShape;
        this.priorRate = priorRate;
        this.posteriorComputed = false;
    }
    
    /**
     * 使用观测数据更新后验分布
     * Update posterior distribution with observed data
     * 
     * @param observations 观测数据（Poisson计数） / Observed data (Poisson counts)
     */
    public void updatePosterior(int[] observations) {
        if (observations == null || observations.length == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        // 计算观测数据的总和
        int sumObservations = 0;
        for (int obs : observations) {
            if (obs < 0) {
                throw new IllegalArgumentException("Poisson observations must be non-negative");
            }
            sumObservations += obs;
        }
        
        // 更新后验参数
        this.posteriorShape = priorShape + sumObservations;
        this.posteriorRate = priorRate + observations.length;
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
        
        int[] obsArray = new int[observations.size()];
        for (int i = 0; i < observations.size(); i++) {
            obsArray[i] = observations.get(i).intValue();
        }
        
        updatePosterior(obsArray);
    }
    
    /**
     * 获取先验分布
     * Get prior distribution
     * 
     * @return Gamma先验分布 / Gamma prior distribution
     */
    public GammaDistribution getPriorDistribution() {
        return new GammaDistribution(priorShape, 1.0 / priorRate); // 注意：GammaDistribution使用scale参数
    }
    
    /**
     * 获取后验分布
     * Get posterior distribution
     * 
     * @return Gamma后验分布 / Gamma posterior distribution
     * @throws IllegalStateException 如果后验分布尚未计算 / If posterior not computed yet
     */
    public GammaDistribution getPosteriorDistribution() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return new GammaDistribution(posteriorShape, 1.0 / posteriorRate);
    }
    
    /**
     * 计算后验预测分布的概率质量函数
     * Calculate posterior predictive distribution PMF
     * 
     * @param k 预测值 / Predicted value
     * @return 概率质量 / Probability mass
     */
    public double posteriorPredictivePmf(int k) {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        if (k < 0) {
            return 0.0;
        }
        
        // 负二项分布的概率质量函数
        // P(X = k) = Γ(k + α) / (k! * Γ(α)) * (β / (β + 1))^α * (1 / (β + 1))^k
        
        double logGammaKPlusAlpha = logGamma(k + posteriorShape);
        double logGammaAlpha = logGamma(posteriorShape);
        double logFactorialK = logFactorial(k);
        
        double p = posteriorRate / (posteriorRate + 1);
        double logP = Math.log(p);
        double log1MinusP = Math.log(1 - p);
        
        double logPmf = logGammaKPlusAlpha - logFactorialK - logGammaAlpha + 
                       posteriorShape * logP + k * log1MinusP;
        
        return Math.exp(logPmf);
    }
    
    /**
     * 从后验预测分布中采样
     * Sample from posterior predictive distribution
     * 
     * @param numSamples 采样数量 / Number of samples
     * @param random 随机数生成器 / Random number generator
     * @return 采样结果 / Sample results
     */
    public int[] samplePosteriorPredictive(int numSamples, Random random) {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        int[] samples = new int[numSamples];
        GammaDistribution posteriorGamma = getPosteriorDistribution();
        
        for (int i = 0; i < numSamples; i++) {
            // 1. 从后验Gamma分布中采样λ
            double lambda = posteriorGamma.sample(1)[0];
            
            // 2. 从Poisson(λ)中采样
            PoissonDistribution poisson = new PoissonDistribution(lambda);
            samples[i] = (int) poisson.sample(1)[0];
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
        
        return posteriorShape / posteriorRate;
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
        
        return posteriorShape / (posteriorRate * posteriorRate);
    }
    
    /**
     * 计算后验预测均值
     * Calculate posterior predictive mean
     * 
     * @return 后验预测均值 / Posterior predictive mean
     */
    public double getPosteriorPredictiveMean() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return posteriorShape / posteriorRate;
    }
    
    /**
     * 计算后验预测方差
     * Calculate posterior predictive variance
     * 
     * @return 后验预测方差 / Posterior predictive variance
     */
    public double getPosteriorPredictiveVariance() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        // 负二项分布的方差：α * (1 + β) / β^2
        return posteriorShape * (posteriorRate + 1) / (posteriorRate * posteriorRate);
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
        
        GammaDistribution posterior = getPosteriorDistribution();
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
    public double calculateMarginalLikelihood(int[] observations) {
        if (observations == null || observations.length == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        int n = observations.length;
        int sumObs = 0;
        double logFactorialProduct = 0.0;
        
        for (int obs : observations) {
            if (obs < 0) {
                throw new IllegalArgumentException("Poisson observations must be non-negative");
            }
            sumObs += obs;
            logFactorialProduct += logFactorial(obs);
        }
        
        // 边际似然 = Γ(α + Σx_i) * β^α * Γ(α) * (β + n)^(-(α + Σx_i)) / Π(x_i!)
        double logMarginalLikelihood = logGamma(priorShape + sumObs) + 
                                     priorShape * Math.log(priorRate) - 
                                     logGamma(priorShape) - 
                                     (priorShape + sumObs) * Math.log(priorRate + n) - 
                                     logFactorialProduct;
        
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
     * @return [形状参数, 率参数] / [shape parameter, rate parameter]
     */
    public double[] getPriorParameters() {
        return new double[] { priorShape, priorRate };
    }
    
    /**
     * 获取后验参数
     * Get posterior parameters
     * 
     * @return [形状参数, 率参数] / [shape parameter, rate parameter]
     * @throws IllegalStateException 如果后验分布尚未计算 / If posterior not computed yet
     */
    public double[] getPosteriorParameters() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return new double[] { posteriorShape, posteriorRate };
    }
    
    /**
     * 计算对数Gamma函数
     * Calculate log Gamma function
     */
    private double logGamma(double x) {
        // 使用Stirling近似或查表法
        // 这里使用简化的实现
        if (x <= 0) {
            throw new IllegalArgumentException("Gamma function undefined for non-positive values");
        }
        
        // 对于整数，使用阶乘
        if (x == Math.floor(x) && x <= 20) {
            return logFactorial((int) x - 1);
        }
        
        // Stirling近似：ln(Γ(x)) ≈ (x - 0.5) * ln(x) - x + 0.5 * ln(2π)
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