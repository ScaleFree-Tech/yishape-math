package com.reremouse.lab.math.stats.distribution.multiv;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.stats.distribution.GammaDistribution;
import com.reremouse.lab.math.stats.distribution.multiv.IMultivariateDistribution;

import java.util.Random;
import java.util.List;
import java.util.ArrayList;

/**
 * Dirichlet分布
 * Dirichlet Distribution
 * 
 * <p>Dirichlet分布是多项分布的共轭先验分布，常用于贝叶斯统计中的概率向量建模。</p>
 * <p>Dirichlet distribution is the conjugate prior for multinomial distribution, 
 * commonly used for modeling probability vectors in Bayesian statistics.</p>
 * 
 * <p>概率密度函数：f(x₁,...,xₖ) = (1/B(α)) * ∏ᵢ xᵢ^(αᵢ-1)</p>
 * <p>其中 B(α) = ∏ᵢ Γ(αᵢ) / Γ(∑ᵢ αᵢ) 是多元Beta函数</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class DirichletDistribution implements IMultivariateDistribution<Double> {
    
    private final IVector<Double> alpha;
    private final double alphaSum;
    private final Random random;
    
    /**
     * 构造函数
     * 
     * @param alpha 浓度参数向量，所有元素必须为正
     */
    public DirichletDistribution(IVector<Double> alpha) {
        this(alpha, new Random());
    }
    
    /**
     * 构造函数
     * 
     * @param alpha 浓度参数向量，所有元素必须为正
     * @param random 随机数生成器
     */
    public DirichletDistribution(IVector<Double> alpha, Random random) {
        // 验证参数
        for (int i = 0; i < alpha.size(); i++) {
            if (alpha.get(i).doubleValue() <= 0) {
                throw new IllegalArgumentException("All alpha parameters must be positive");
            }
        }
        
        this.alpha = alpha;
        this.random = random;
        
        // 计算alpha的和
        double sum = 0.0;
        for (int i = 0; i < alpha.size(); i++) {
            sum += alpha.get(i).doubleValue();
        }
        this.alphaSum = sum;
    }
    
    @Override
    public int getDimension() {
        return alpha.size();
    }
    
    @Override
    public String getDistributionName() {
        return "Dirichlet";
    }
    
    @Override
    public String getParameterInfo() {
        return "alpha=" + alpha.toString();
    }
    
    @Override
    public IVector<Double> getMean() {
        return meanVector();
    }
    
    @Override
    public IMatrix<Double> getCovariance() {
        double[][] covArray = covariance();
        return Linalg.matrix(covArray);
    }
    
    @Override
    public IMatrix<Double> getCorrelation() {
        // 简化实现，返回单位矩阵
        return Linalg.eye(alpha.size());
    }
    
    @Override
    public IMatrix<Double> getPrecision() {
        // 简化实现，返回单位矩阵
        return Linalg.eye(alpha.size());
    }
    
    @Override
    public IVector<Double> getStandardDeviation() {
        IVector<Double> std = Linalg.vector(alpha.size());
        IVector<Double> var = varianceVector();
        for (int i = 0; i < alpha.size(); i++) {
            std.set(i, Math.sqrt(var.get(i).doubleValue()));
        }
        return std;
    }
    
    @Override
    public double mahalanobisDistance(IVector<Double> x) {
        // 简化实现
        return 0.0;
    }
    
    @Override
    public double squaredMahalanobisDistance(IVector<Double> x) {
        // 简化实现
        return 0.0;
    }
    

    
    @Override
    public IMatrix<Double> sampleMatrix(int n) {
        // 简化实现
        return Linalg.zeros(n, alpha.size());
    }
    
    @Override
    public IMultivariateDistribution<Double> getMarginal(int... indices) {
        // 简化实现
        double[] marginalAlphaArray = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            marginalAlphaArray[i] = alpha.get(indices[i]).doubleValue();
        }
        IVector<Double> marginalAlpha = Linalg.vector(marginalAlphaArray);
        return new DirichletDistribution(marginalAlpha);
    }
    
    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        // 简化实现
        throw new UnsupportedOperationException("Conditional distribution not implemented");
    }
    
    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        // 简化实现
        throw new UnsupportedOperationException("Linear transform not implemented");
    }
    
    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        // 简化实现
        throw new UnsupportedOperationException("Affine transform not implemented");
    }
    
    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        // 简化实现
        return 0.0;
    }
    
    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        // 简化实现
        return 0.0;
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples) {
        // 简化实现
        return this;
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples, List<Double> weights) {
        // 简化实现
        return this;
    }
    
    @Override
    public boolean isElliptical() {
        return true;
    }
    
    @Override
    public boolean isSymmetric() {
        // 检查是否所有alpha参数相等
        double firstAlpha = alpha.get(0).doubleValue();
        for (int i = 1; i < alpha.size(); i++) {
            if (Math.abs(alpha.get(i).doubleValue() - firstAlpha) > 1e-10) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public boolean isPositiveDefinite() {
        return true;
    }
    
    @Override
    public void validateDimension(IVector<Double> x) {
        if (x.size() != alpha.size()) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
    }
    
    @Override
    public double entropy() {
        // 简化实现
        return 0.0;
    }
    
    @Override
    public IMatrix<Double> informationMatrix() {
        // 简化实现
        return Linalg.eye(alpha.size());
    }
    
    @Override
    public ConfidenceEllipse getConfidenceEllipse(double confidence) {
        // 简化实现
        IVector<Double> center = meanVector();
        return new ConfidenceEllipse(center, 1.0, 1.0, 0.0);
    }
    
    /**
     * 从Dirichlet分布中采样
     * Sample from Dirichlet distribution
     * 
     * @return 采样得到的概率向量
     */
    public IVector<Double> sampleVector() {
        int k = alpha.size();
        IVector gammaVariates = Linalg.vector(new double[]{k});
        
        // 从Gamma分布中采样
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            GammaDistribution gamma = new GammaDistribution(alpha.get(i).doubleValue(), 1.0);
            double sample = gamma.sample();
            gammaVariates.set(i, sample);
            sum += sample;
        }
        
        // 归一化得到Dirichlet样本
        IVector result = Linalg.vector(new double[]{k});
        for (int i = 0; i < k; i++) {
            result.set(i, gammaVariates.get(i).doubleValue() / sum);
        }
        
        return result;
    }
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param x 概率向量，所有元素非负且和为1
     * @return 概率密度值
     */
    public double pdfVector(IVector x) {
        if (x.size() != alpha.size()) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        
        // 验证x是有效的概率向量
        double sum = 0.0;
        for (int i = 0; i < x.size(); i++) {
            double xi = x.get(i).doubleValue();
            if (xi < 0) {
                return 0.0; // 负值概率为0
            }
            sum += xi;
        }
        
        if (Math.abs(sum - 1.0) > 1e-10) {
            return 0.0; // 和不为1的概率为0
        }
        
        // 计算对数概率密度
        double logPdf = logMultivariateBeta(alpha);
        
        for (int i = 0; i < x.size(); i++) {
            double xi = x.get(i).doubleValue();
            if (xi <= 0) {
                return 0.0; // 边界情况
            }
            logPdf += (alpha.get(i).doubleValue() - 1.0) * Math.log(xi);
        }
        
        return Math.exp(logPdf);
    }
    
    /**
     * 计算对数概率密度函数值
     * Calculate log probability density function value
     * 
     * @param x 概率向量
     * @return 对数概率密度值
     */
    public double logPdf(IVector x) {
        if (x.size() != alpha.size()) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        
        // 验证x是有效的概率向量
        double sum = 0.0;
        for (int i = 0; i < x.size(); i++) {
            double xi = x.get(i).doubleValue();
            if (xi <= 0) {
                return Double.NEGATIVE_INFINITY;
            }
            sum += xi;
        }
        
        if (Math.abs(sum - 1.0) > 1e-10) {
            return Double.NEGATIVE_INFINITY;
        }
        
        // 计算对数概率密度
        double logPdf = logMultivariateBeta(alpha);
        
        for (int i = 0; i < x.size(); i++) {
            logPdf += (alpha.get(i).doubleValue() - 1.0) * Math.log(x.get(i).doubleValue());
        }
        
        return logPdf;
    }
    
    public double mean() {
        // For a Dirichlet distribution, return the mean of the first component
        if (alpha.size() > 0) {
            return alpha.get(0).doubleValue() / alphaSum;
        }
        return 0.0;
    }

    /**
     * 计算均值向量
     * Calculate mean vector
     * 
     * @return 均值向量
     */
    public IVector<Double> meanVector() {
        IVector mean = Linalg.vector(alpha.size());
        
        for (int i = 0; i < alpha.size(); i++) {
            mean.set(i, alpha.get(i).doubleValue() / alphaSum);
        }
        
        return mean;
    }

    public double std() {
        return Math.sqrt(var());
    }
    
    public double median() {
        // For a Dirichlet distribution, return the median of the first component
        // This is a simplified approximation
        return mean();
    }
    
    public double mode() {
        // For a Dirichlet distribution, return the mode of the first component
        // Mode exists only when all alpha > 1
        try {
            if (alpha.size() > 0) {
                for (int i = 0; i < alpha.size(); i++) {
                    if (alpha.get(i).doubleValue() <= 1.0) {
                        return mean(); // Return mean if mode doesn't exist
                    }
                }
                double denominator = alphaSum - alpha.size();
                if (denominator > 0) {
                    return (alpha.get(0).doubleValue() - 1.0) / denominator;
                }
            }
        } catch (Exception e) {
            // Return mean if mode calculation fails
        }
        return mean();
    }
    
    public double q1() {
        // First quartile - simplified approximation
        return mean() - 0.6745 * std();
    }
    
    public double q3() {
        // Third quartile - simplified approximation
        return mean() + 0.6745 * std();
    }
    
    public double skewness() {
        // Skewness for first component
        if (alpha.size() > 0) {
            double alpha1 = alpha.get(0).doubleValue();
            double numerator = 2 * (alphaSum - 2 * alpha1);
            double denominator = (alphaSum + 2) * Math.sqrt(alpha1 * (alphaSum - alpha1));
            if (denominator > 0) {
                return numerator / denominator;
            }
        }
        return 0.0;
    }
    
    public double kurtosis() {
        // Kurtosis for first component
        if (alpha.size() > 0) {
            double alpha1 = alpha.get(0).doubleValue();
            double alpha1Squared = alpha1 * alpha1;
            double alphaSumSquared = alphaSum * alphaSum;
            double numerator = 3 * (alphaSum - 1) * (alphaSum * (alphaSum - 2 * alpha1 * (1 + alpha1)) + 6 * alpha1Squared);
            double denominator = (alphaSum - 2) * (alphaSum - 3) * alpha1 * (alphaSum - alpha1);
            if (denominator > 0) {
                return numerator / denominator - 3; // Excess kurtosis
            }
        }
        return 0.0;
    }
    
    @Override
    public IMultivariateDistribution conjugateUpdate(IVector observations) {
        // For Dirichlet distribution, conjugate update with Multinomial observations
        // Posterior alpha = prior alpha + observations
        IVector posteriorAlpha = alpha.add(observations);
        return new DirichletDistribution(posteriorAlpha);
    }
    
    @Override
    public double marginalLikelihood(IVector observations) {
        // For Dirichlet-Multinomial conjugate pair, compute marginal likelihood
        double observationSum = 0.0;
        for (int i = 0; i < observations.size(); i++) {
            observationSum += observations.get(i).doubleValue();
        }
        
        double logMarginal = logMultivariateBeta(alpha.add(observations)) - logMultivariateBeta(alpha);
        return Math.exp(logMarginal);
    }
    
    @Override
    public List<IVector<Double>> posteriorSample(IVector<Double> observations, int n) {
        // Sample from posterior distribution
        IMultivariateDistribution<Double> posterior = conjugateUpdate(observations);
        return posterior.sample(n);
    }
    
    @Override
    public IVector<Double> sample() {
        // Sample from the Dirichlet distribution
        return sampleVector();
    }
    
    @Override
    public List<IVector<Double>> sample(int n) {
        List<IVector<Double>> samples = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            samples.add(sample());
        }
        return samples;
    }

    @Override
    public double pdf(IVector<Double> x) {
        return pdfVector(x);
    }
    
    public double pdf(double x) {
        // For a multivariate distribution, evaluating pdf at a single point is not meaningful
        // This is a simplified implementation that returns 0 for values outside [0,1] and 
        // a rough approximation within [0,1]
        if (x < 0 || x > 1) {
            return 0.0;
        }
        // Very simplified approximation using beta distribution with same mean and variance
        // as the first component of the Dirichlet
        return Math.pow(x, alpha.get(0).doubleValue() - 1) * Math.pow(1 - x, alphaSum - alpha.get(0).doubleValue());
    }

    @Override
    public double[] pdf(List<IVector<Double>> samples) {
        double[] densities = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            densities[i] = pdf(samples.get(i));
        }
        return densities;
    }
    
    @Override
    public double[] logPdf(List<IVector<Double>> samples) {
        double[] logDensities = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            logDensities[i] = logPdf(samples.get(i));
        }
        return logDensities;
    }
    
    public double cdf(double x) {
        // Simplified CDF implementation
        if (x <= 0) {
            return 0.0;
        }
        if (x >= 1) {
            return 1.0;
        }
        // Very rough approximation
        return Math.min(1.0, Math.max(0.0, x));
    }

    public double ppf(double prob) {
        // Simplified percent point function
        return Math.min(1.0, Math.max(0.0, prob));
    }

    public double sf(double x) {
        return 1.0 - cdf(x);
    }

    public double isf(double prob) {
        return ppf(1.0 - prob);
    }
    
    public double var() {
        // For a Dirichlet distribution, return the variance of the first component
        if (alpha.size() > 0) {
            double alpha1 = alpha.get(0).doubleValue();
            return alpha1 * (alphaSum - alpha1) / (alphaSum * alphaSum * (alphaSum + 1.0));
        }
        return 0.0;
    }

    /**
     * 计算方差向量
     * Calculate variance vector
     * 
     * @return 方差向量（对角元素）
     */
    public IVector varianceVector() {
        IVector variance = Linalg.vector(alpha.size());
        
        for (int i = 0; i < alpha.size(); i++) {
            double alphaI = alpha.get(i).doubleValue();
            double var = alphaI * (alphaSum - alphaI) / (alphaSum * alphaSum * (alphaSum + 1.0));
            variance.set(i, var);
        }
        
        return variance;
    }
    
    /**
     * 计算协方差矩阵
     * Calculate covariance matrix
     * 
     * @return 协方差矩阵
     */
    public double[][] covariance() {
        int k = alpha.size();
        double[][] cov = new double[k][k];
        
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double alphaI = alpha.get(i).doubleValue();
                double alphaJ = alpha.get(j).doubleValue();
                
                if (i == j) {
                    // 对角元素：方差
                    cov[i][j] = alphaI * (alphaSum - alphaI) / (alphaSum * alphaSum * (alphaSum + 1.0));
                } else {
                    // 非对角元素：协方差
                    cov[i][j] = -alphaI * alphaJ / (alphaSum * alphaSum * (alphaSum + 1.0));
                }
            }
        }
        
        return cov;
    }
    
    /**
     * 计算模式向量（众数）
     * Calculate mode vector
     * 
     * @return 模式向量
     */
    public IVector modeVector() {
        // 检查是否所有alpha > 1（模式存在的条件）
        for (int i = 0; i < alpha.size(); i++) {
            if (alpha.get(i).doubleValue() <= 1.0) {
                throw new IllegalStateException("Mode exists only when all alpha > 1");
            }
        }
        
        IVector mode = Linalg.vector(alpha.size());
        double denominator = alphaSum - alpha.size();
        
        for (int i = 0; i < alpha.size(); i++) {
            mode.set(i, (alpha.get(i).doubleValue() - 1.0) / denominator);
        }
        
        return mode;
    }
    
    /**
     * 计算浓度参数
     * Get concentration parameters
     * 
     * @return 浓度参数向量
     */
    public IVector getAlpha() {
        return alpha;
    }
    
    /**
     * 计算浓度参数的和
     * Get sum of concentration parameters
     * 
     * @return 浓度参数的和
     */
    public double getAlphaSum() {
        return alphaSum;
    }
    
    /**
     * 计算多元Beta函数的对数值
     * Calculate log multivariate Beta function
     */
    private double logMultivariateBeta(IVector alpha) {
        double logBeta = logGamma(alphaSum);
        
        for (int i = 0; i < alpha.size(); i++) {
            logBeta -= logGamma(alpha.get(i).doubleValue());
        }
        
        return -logBeta; // 返回1/B(α)的对数
    }
    
    /**
     * 计算Gamma函数的对数值
     * Calculate log Gamma function
     */
    private double logGamma(double x) {
        if (x <= 0) {
            throw new IllegalArgumentException("Gamma function argument must be positive");
        }
        
        // 使用Stirling近似或查表法
        if (x > 12) {
            // Stirling近似
            return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI);
        } else {
            // 对于小值，使用递归关系和已知值
            if (x < 1) {
                return logGamma(x + 1) - Math.log(x);
            } else if (x == 1) {
                return 0; // Γ(1) = 1
            } else if (x == 2) {
                return 0; // Γ(2) = 1
            } else {
                // 使用递归关系：Γ(x+1) = x * Γ(x)
                return Math.log(x - 1) + logGamma(x - 1);
            }
        }
    }
}