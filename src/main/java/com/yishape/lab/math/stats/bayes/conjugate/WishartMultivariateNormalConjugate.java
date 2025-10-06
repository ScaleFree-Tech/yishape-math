package com.yishape.lab.math.stats.bayes.conjugate;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.distribution.NormalDistribution;
import com.yishape.lab.math.stats.distribution.multiv.MultivariateNormalDistribution;

import java.util.Random;

/**
 * Wishart-Multivariate Normal共轭先验
 * Wishart-Multivariate Normal Conjugate Prior
 * 
 * <p>实现Normal-Wishart分布作为多元正态分布均值向量和精度矩阵的共轭先验。
 * 在观测到多元正态分布数据后，后验分布仍然是Normal-Wishart分布。</p>
 * <p>Implements Normal-Wishart distribution as conjugate prior for multivariate normal distribution 
 * mean vector and precision matrix. After observing multivariate normal data, 
 * the posterior distribution remains Normal-Wishart.</p>
 * 
 * <p>模型：
 * Λ ~ Wishart(ν₀, S₀⁻¹)
 * μ | Λ ~ MVN(μ₀, (κ₀Λ)⁻¹)
 * x_i | μ, Λ ~ MVN(μ, Λ⁻¹)
 * 后验：(μ, Λ) | X ~ NormalWishart(μₙ, κₙ, νₙ, Sₙ)</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WishartMultivariateNormalConjugate {
    
    private final int dimension;
    
    // 先验参数
    private IVector priorMu0;      // μ₀ (先验均值向量)
    private double priorKappa;     // κ₀ (先验精度参数)
    private double priorNu;        // ν₀ (Wishart自由度)
    private IMatrix priorS;        // S₀ (Wishart尺度矩阵)
    
    // 后验参数
    private IVector posteriorMu;
    private double posteriorKappa;
    private double posteriorNu;
    private IMatrix posteriorS;
    private boolean posteriorComputed;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param priorMu0 先验均值向量 μ₀ / Prior mean vector μ₀
     * @param priorKappa 先验精度参数 κ₀ / Prior precision parameter κ₀
     * @param priorNu Wishart自由度 ν₀ / Wishart degrees of freedom ν₀
     * @param priorS Wishart尺度矩阵 S₀ / Wishart scale matrix S₀
     */
    public WishartMultivariateNormalConjugate(IVector priorMu0, double priorKappa, 
                                            double priorNu, IMatrix priorS) {
        if (priorMu0 == null || priorS == null) {
            throw new IllegalArgumentException("Prior mean and scale matrix cannot be null");
        }
        
        if (priorKappa <= 0 || priorNu <= priorMu0.size() - 1) {
            throw new IllegalArgumentException("Invalid prior parameters");
        }
        
        if (priorS.rows() != priorS.cols() || priorS.rows() != priorMu0.size()) {
            throw new IllegalArgumentException("Scale matrix must be square and match mean vector dimension");
        }
        
        this.dimension = priorMu0.size();
        this.priorMu0 = priorMu0.copy();
        this.priorKappa = priorKappa;
        this.priorNu = priorNu;
        this.priorS = priorS.copy();
        this.posteriorComputed = false;
    }
    
    /**
     * 使用观测数据更新后验分布
     * Update posterior distribution with observed data
     * 
     * @param observations 观测数据矩阵（每行一个观测） / Observed data matrix (each row is an observation)
     */
    public void updatePosterior(IMatrix observations) {
        if (observations == null || observations.rows() == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        if (observations.cols() != dimension) {
            throw new IllegalArgumentException("Observation dimension must match prior dimension");
        }
        
        int n = observations.rows();
        
        // 计算样本均值
        IVector sampleMean = Linalg.vector(dimension);
        for (int i = 0; i < n; i++) {
            IVector row = observations.getRow(i);
            sampleMean = sampleMean.add(row);
        }
        sampleMean = sampleMean.multiplyScalar(1.0 / n);
        
        // 计算样本协方差矩阵
        IMatrix sampleCov = Linalg.zeros(dimension, dimension);
        for (int i = 0; i < n; i++) {
            IVector diff = observations.getRow(i).sub(sampleMean);
            IMatrix outerProduct = diff.outer(diff);
            sampleCov = sampleCov.add(outerProduct);
        }
        
        // 更新后验参数
        this.posteriorKappa = priorKappa + n;
        this.posteriorNu = priorNu + n;
        
        // 更新后验均值
        IVector kappaTimesmu0 = priorMu0.multiplyScalar(priorKappa);
        IVector nTimesSampleMean = sampleMean.multiplyScalar(n);
        this.posteriorMu = kappaTimesmu0.add(nTimesSampleMean).multiplyScalar(1.0 / posteriorKappa);
        
        // 更新后验尺度矩阵
        IVector muDiff = sampleMean.sub(priorMu0);
        IMatrix muDiffOuter = muDiff.outer(muDiff);
        double scaleFactor = (priorKappa * n) / posteriorKappa;
        IMatrix scaledMuDiffOuter = muDiffOuter.multiplyScalar(scaleFactor);
        
        this.posteriorS = priorS.add(sampleCov).add(scaledMuDiffOuter);
        this.posteriorComputed = true;
    }
    
    /**
     * 从先验分布中采样 (μ, Λ)
     * Sample (μ, Λ) from prior distribution
     * 
     * @param random 随机数生成器 / Random number generator
     * @return [μ, Λ] 采样结果 / [μ, Λ] sample result
     */
    public Object[] samplePrior(Random random) {
        // 1. 从Wishart分布中采样Λ
        IMatrix lambda = sampleWishart(priorNu, priorS, random);
        
        // 2. 从多元正态分布中采样μ | Λ
        IMatrix covarianceMatrix = lambda.multiplyScalar(priorKappa).inv();
        MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(priorMu0, covarianceMatrix);
        IVector mu = mvn.sample();
        
        return new Object[] { mu, lambda };
    }
    
    /**
     * 从后验分布中采样 (μ, Λ)
     * Sample (μ, Λ) from posterior distribution
     * 
     * @param random 随机数生成器 / Random number generator
     * @return [μ, Λ] 采样结果 / [μ, Λ] sample result
     */
    public Object[] samplePosterior(Random random) {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        // 1. 从后验Wishart分布中采样Λ
        IMatrix lambda = sampleWishart(posteriorNu, posteriorS, random);
        
        // 2. 从后验多元正态分布中采样μ | Λ
        IMatrix covarianceMatrix = lambda.multiplyScalar(posteriorKappa).inv();
        MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(posteriorMu, covarianceMatrix);
        IVector mu = mvn.sample();
        
        return new Object[] { mu, lambda };
    }
    
    /**
     * 从后验预测分布中采样
     * Sample from posterior predictive distribution
     * 
     * @param numSamples 采样数量 / Number of samples
     * @param random 随机数生成器 / Random number generator
     * @return 采样结果矩阵 / Sample result matrix
     */
    public IMatrix samplePosteriorPredictive(int numSamples, Random random) {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        IMatrix samples = Linalg.zeros(numSamples, dimension);
        
        for (int i = 0; i < numSamples; i++) {
            // 1. 从后验分布中采样 (μ, Λ)
            Object[] params = samplePosterior(random);
            IVector mu = (IVector) params[0];
            IMatrix lambda = (IMatrix) params[1];
            
            // 2. 从MVN(μ, Λ⁻¹)中采样
            IMatrix covariance = lambda.inv();
            MultivariateNormalDistribution mvn = new MultivariateNormalDistribution(mu, covariance);
            IVector sample = mvn.sample();
            
            // 设置样本行
            for (int j = 0; j < dimension; j++) {
                samples.set(i, j, sample.get(j));
            }
        }
        
        return samples;
    }
    
    /**
     * 计算后验预测分布（多元t分布）
     * Calculate posterior predictive distribution (multivariate t-distribution)
     * 
     * @return 后验预测分布参数 [自由度, 位置向量, 尺度矩阵] / Posterior predictive parameters [df, location, scale]
     */
    public Object[] getPosteriorPredictiveParameters() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        // 多元t分布参数
        double df = posteriorNu - dimension + 1;
        IVector location = posteriorMu.copy();
        
        double scaleFactor = (posteriorKappa + 1) / (posteriorKappa * df);
        IMatrix scale = posteriorS.multiplyScalar(scaleFactor);
        
        return new Object[] { df, location, scale };
    }
    
    /**
     * 计算后验均值的期望
     * Calculate expectation of posterior mean
     * 
     * @return 后验均值的期望 / Expectation of posterior mean
     */
    public IVector getPosteriorMeanExpectation() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return posteriorMu.copy();
    }
    
    /**
     * 计算后验精度矩阵的期望
     * Calculate expectation of posterior precision matrix
     * 
     * @return 后验精度矩阵的期望 / Expectation of posterior precision matrix
     */
    public IMatrix getPosteriorPrecisionExpectation() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return posteriorS.inv().multiplyScalar(posteriorNu);
    }
    
    /**
     * 计算后验协方差矩阵的期望
     * Calculate expectation of posterior covariance matrix
     * 
     * @return 后验协方差矩阵的期望 / Expectation of posterior covariance matrix
     */
    public IMatrix getPosteriorCovarianceExpectation() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        if (posteriorNu <= dimension + 1) {
            throw new IllegalStateException("Degrees of freedom too small for covariance expectation");
        }
        
        return posteriorS.multiplyScalar(1.0 / (posteriorNu - dimension - 1));
    }
    
    /**
     * 计算边际似然（证据）
     * Calculate marginal likelihood (evidence)
     * 
     * @param observations 观测数据 / Observed data
     * @return 边际似然 / Marginal likelihood
     */
    public double calculateMarginalLikelihood(IMatrix observations) {
        if (observations == null || observations.rows() == 0) {
            throw new IllegalArgumentException("Observations cannot be null or empty");
        }
        
        int n = observations.rows();
        
        // 计算边际似然的对数
        double logMarginalLikelihood = 
            // 常数项
            -n * dimension / 2.0 * Math.log(Math.PI) +
            // κ项
            dimension / 2.0 * Math.log(priorKappa / (priorKappa + n)) +
            // Gamma函数项
            logMultivariateGamma((priorNu + n) / 2.0, dimension) - 
            logMultivariateGamma(priorNu / 2.0, dimension) +
            // 行列式项
            priorNu / 2.0 * Math.log((double)priorS.det()) -
            (priorNu + n) / 2.0 * Math.log((double)posteriorS.det());
        
        return Math.exp(logMarginalLikelihood);
    }
    
    /**
     * 从Wishart分布中采样
     * Sample from Wishart distribution
     * 
     * @param nu 自由度 / Degrees of freedom
     * @param scale 尺度矩阵 / Scale matrix
     * @param random 随机数生成器 / Random number generator
     * @return Wishart采样结果 / Wishart sample
     */
    private IMatrix sampleWishart(double nu, IMatrix scale, Random random) {
        // 使用Bartlett分解方法
        int p = scale.rows();
        
        // 1. 生成下三角矩阵A
        IMatrix A = Linalg.zeros(p, p);
        
        for (int i = 0; i < p; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    // 对角元素：χ²分布
                    double chi2Sample = sampleChiSquared(nu - i, random);
                    A.set(i, j, Math.sqrt(chi2Sample));
                } else {
                    // 下三角元素：标准正态分布
                    NormalDistribution normal = new NormalDistribution(0, 1);
                    A.set(i, j, normal.sample());
                }
            }
        }
        
        // 2. 计算Cholesky分解 L，使得 scale = L * L^T
        IMatrix L = choleskyDecomposition(scale);
        
        // 3. 返回 L * A * A^T * L^T
        IMatrix LA = L.multiply(A);
        IMatrix AAT = A.multiply(A.transpose());
        IMatrix LAAT = L.multiply(AAT);
        return LAAT.multiply(L.transpose());
    }
    
    /**
     * 从卡方分布中采样
     * Sample from chi-squared distribution
     */
    private double sampleChiSquared(double df, Random random) {
        // 使用Gamma分布：χ²(k) = Gamma(k/2, 2)
        double shape = df / 2.0;
        double scale = 2.0;
        
        // 简化的Gamma采样（使用正态近似）
        if (df > 30) {
            NormalDistribution normal = new NormalDistribution(df, Math.sqrt(2 * df));
            return Math.max(0, normal.sample());
        } else {
            // 使用变换方法
            double sum = 0;
            for (int i = 0; i < (int) df; i++) {
                NormalDistribution normal = new NormalDistribution(0, 1);
                double z = normal.sample();
                sum += z * z;
            }
            return sum;
        }
    }
    
    /**
     * Cholesky分解
     * Cholesky decomposition
     */
    private IMatrix choleskyDecomposition(IMatrix matrix) {
        int n = matrix.rows();
        IMatrix L = Linalg.zeros(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    double sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum += L.get(j, k).doubleValue() * L.get(j, k).doubleValue();
                    }
                    L.set(j, j, Math.sqrt(matrix.get(j, j).doubleValue() - sum));
                } else {
                    double sum = 0;
                    for (int k = 0; k < j; k++) {
                        sum += L.get(i, k).doubleValue() * L.get(j, k).doubleValue();
                    }
                    L.set(i, j, (matrix.get(i, j).doubleValue() - sum) / L.get(j, j).doubleValue());
                }
            }
        }
        
        return L;
    }
    
    /**
     * 计算多元Gamma函数的对数
     * Calculate log multivariate Gamma function
     */
    private double logMultivariateGamma(double a, int p) {
        double result = p * (p - 1) / 4.0 * Math.log(Math.PI);
        
        for (int i = 0; i < p; i++) {
            result += logGamma(a - i / 2.0);
        }
        
        return result;
    }
    
    /**
     * 计算对数Gamma函数
     * Calculate log Gamma function
     */
    private double logGamma(double x) {
        if (x <= 0) {
            throw new IllegalArgumentException("Gamma function undefined for non-positive values");
        }
        
        // Stirling近似
        return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI);
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
     * @return [μ₀, κ₀, ν₀, S₀] / [μ₀, κ₀, ν₀, S₀]
     */
    public Object[] getPriorParameters() {
        return new Object[] { priorMu0.copy(), priorKappa, priorNu, priorS.copy() };
    }
    
    /**
     * 获取后验参数
     * Get posterior parameters
     * 
     * @return [μₙ, κₙ, νₙ, Sₙ] / [μₙ, κₙ, νₙ, Sₙ]
     * @throws IllegalStateException 如果后验分布尚未计算 / If posterior not computed yet
     */
    public Object[] getPosteriorParameters() {
        if (!posteriorComputed) {
            throw new IllegalStateException("Posterior distribution not computed. Call updatePosterior() first.");
        }
        
        return new Object[] { posteriorMu.copy(), posteriorKappa, posteriorNu, posteriorS.copy() };
    }
    
    /**
     * 是否已计算后验分布
     * Whether posterior distribution is computed
     */
    public boolean isPosteriorComputed() {
        return posteriorComputed;
    }
    
    /**
     * 获取维度
     * Get dimension
     */
    public int getDimension() {
        return dimension;
    }
}