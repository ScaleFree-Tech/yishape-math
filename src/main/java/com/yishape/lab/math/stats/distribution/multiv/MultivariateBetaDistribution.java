package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * 多元Beta分布实现（Dirichlet分布） / Multivariate Beta Distribution Implementation (Dirichlet Distribution)
 * 
 * <p>多元Beta分布，也称为Dirichlet分布，是Beta分布在多维空间的推广。
 * 它定义在单纯形上，即所有分量非负且和为1的向量空间。
 * 概率密度函数为：</p>
 * <p>f(x) = [Γ(∑αᵢ) / ∏Γ(αᵢ)] * ∏xᵢ^(αᵢ-1)</p>
 * <p>其中 αᵢ > 0 是形状参数，x ∈ Δₙ₋₁（n-1维单纯形）。</p>
 * 
 * <p>Multivariate Beta distribution, also known as Dirichlet distribution, is a generalization
 * of the Beta distribution to multiple dimensions. It is defined on the simplex, i.e., the space
 * of vectors with non-negative components that sum to 1. The PDF is:</p>
 * <p>f(x) = [Γ(∑αᵢ) / ∏Γ(αᵢ)] * ∏xᵢ^(αᵢ-1)</p>
 * <p>where αᵢ > 0 are shape parameters and x ∈ Δₙ₋₁ (n-1 dimensional simplex).</p>
 * 
 * <h3>主要特性 / Key Properties:</h3>
 * <ul>
 *   <li>支撑在单纯形上 / Support on the simplex</li>
 *   <li>共轭先验性质 / Conjugate prior property</li>
 *   <li>可用于建模概率向量 / Used for modeling probability vectors</li>
 *   <li>边际分布为Beta分布 / Marginal distributions are Beta distributions</li>
 *   <li>广泛用于贝叶斯统计 / Widely used in Bayesian statistics</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MultivariateBetaDistribution implements IMultivariateDistribution<Double> {
    
    private static final long serialVersionUID = 1L;
    
    /** 形状参数向量 / Shape parameters vector */
    private final IVector<Double> alpha;
    
    /** 维度 / Dimensionality */
    private final int dimension;
    
    /** 参数总和 / Sum of parameters */
    private final double alphaSum;
    
    /** 对数归一化常数 / Log normalization constant */
    private final double logNormalizationConstant;
    
    /** 随机数生成器 / Random number generator */
    private final Random random;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param alpha 形状参数向量 / Shape parameters vector
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateBetaDistribution(IVector<Double> alpha) {
        this(alpha, new Random());
    }
    
    /**
     * 构造函数（带随机种子）
     * Constructor with random seed
     * 
     * @param alpha 形状参数向量 / Shape parameters vector
     * @param random 随机数生成器 / Random number generator
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateBetaDistribution(IVector<Double> alpha, Random random) {
        validateParameters(alpha);
        
        this.dimension = alpha.length();
        this.alpha = alpha.copy();
        this.random = random;
        
        // 计算参数总和
        this.alphaSum = computeAlphaSum();
        
        // 计算对数归一化常数
        this.logNormalizationConstant = computeLogNormalizationConstant();
    }
    
    /**
     * 验证参数有效性
     * Validate parameter validity
     */
    private void validateParameters(IVector<Double> alpha) {
        if (alpha == null) {
            throw new IllegalArgumentException("形状参数向量不能为null");
        }
        
        if (alpha.length() < 2) {
            throw new IllegalArgumentException("Dirichlet分布至少需要2个维度");
        }
        
        // 验证所有形状参数都为正
        for (int i = 0; i < alpha.length(); i++) {
            if (alpha.get(i) <= 0) {
                throw new IllegalArgumentException(
                    String.format("形状参数 %d 必须大于0，实际值: %.6f", i, alpha.get(i)));
            }
        }
    }
    
    /**
     * 计算参数总和
     * Compute sum of parameters
     */
    private double computeAlphaSum() {
        double sum = 0.0;
        for (int i = 0; i < dimension; i++) {
            sum += alpha.get(i);
        }
        return sum;
    }
    
    /**
     * 计算对数归一化常数
     * Compute log normalization constant
     */
    private double computeLogNormalizationConstant() {
        // log[Γ(∑αᵢ) / ∏Γ(αᵢ)] = logΓ(∑αᵢ) - ∑logΓ(αᵢ)
        double logNorm = logGamma(alphaSum);
        for (int i = 0; i < dimension; i++) {
            logNorm -= logGamma(alpha.get(i));
        }
        return logNorm;
    }
    
    /**
     * 对数Gamma函数的近似计算
     * Approximate computation of log Gamma function
     */
    private double logGamma(double x) {
        // 使用Stirling近似：log Γ(x) ≈ (x-0.5)log(x) - x + 0.5*log(2π)
        if (x < 1.0) {
            // 使用递推关系：Γ(x) = Γ(x+1)/x
            return logGamma(x + 1.0) - Math.log(x);
        }
        
        if (x == 1.0 || x == 2.0) {
            return 0.0; // Γ(1) = Γ(2) = 1
        }
        
        // Stirling近似
        return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2.0 * Math.PI);
    }
    
    /**
     * 检查点是否在单纯形上
     * Check if point is on the simplex
     */
    private boolean isOnSimplex(IVector<Double> x) {
        double sum = 0.0;
        for (int i = 0; i < dimension; i++) {
            double xi = x.get(i);
            if (xi < 0 || xi > 1) {
                return false;
            }
            sum += xi;
        }
        return Math.abs(sum - 1.0) < 1e-10; // 允许数值误差
    }
    
    // ==================== IMultivariateDistribution 接口实现 ====================
    
    @Override
    public int getDimension() {
        return dimension;
    }
    
    @Override
    public String getDistributionName() {
        return "Multivariate Beta Distribution (Dirichlet)";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("Dimension: %d, Alpha: %s, Sum: %.6f", 
                           dimension, alpha.toString(), alphaSum);
    }
    
    @Override
    public double pdf(IVector<Double> x) {
        validateDimension(x);
        if (!isOnSimplex(x)) {
            return 0.0;
        }
        return Math.exp(logPdf(x));
    }
    
    @Override
    public double logPdf(IVector<Double> x) {
        validateDimension(x);
        if (!isOnSimplex(x)) {
            return Double.NEGATIVE_INFINITY;
        }
        
        // log f(x) = log[Γ(∑αᵢ) / ∏Γ(αᵢ)] + ∑(αᵢ-1)log(xᵢ)
        double logDensity = logNormalizationConstant;
        for (int i = 0; i < dimension; i++) {
            double xi = x.get(i);
            if (xi <= 0) {
                return Double.NEGATIVE_INFINITY;
            }
            logDensity += (alpha.get(i) - 1.0) * Math.log(xi);
        }
        
        return logDensity;
    }
    
    @Override
    public double[] pdf(List<IVector<Double>> samples) {
        double[] results = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            results[i] = pdf(samples.get(i));
        }
        return results;
    }
    
    @Override
    public double[] logPdf(List<IVector<Double>> samples) {
        double[] results = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            results[i] = logPdf(samples.get(i));
        }
        return results;
    }
    
    @Override
    public IVector<Double> getMean() {
        // 均值是 αᵢ / ∑αⱼ
        double[] meanArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            meanArray[i] = alpha.get(i) / alphaSum;
        }
        return Linalg.vector(meanArray);
    }
    
    @Override
    public IMatrix<Double> getCovariance() {
        // 协方差矩阵：Cov(Xᵢ,Xⱼ) = -αᵢαⱼ/[α₀²(α₀+1)] (i≠j)
        //                      Var(Xᵢ) = αᵢ(α₀-αᵢ)/[α₀²(α₀+1)] (i=j)
        double[][] covArray = new double[dimension][dimension];
        double alpha0 = alphaSum;
        double denominator = alpha0 * alpha0 * (alpha0 + 1.0);
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i == j) {
                    // 方差
                    covArray[i][j] = alpha.get(i) * (alpha0 - alpha.get(i)) / denominator;
                } else {
                    // 协方差
                    covArray[i][j] = -alpha.get(i) * alpha.get(j) / denominator;
                }
            }
        }
        
        return Linalg.matrix(covArray);
    }
    
    @Override
    public IMatrix<Double> getCorrelation() {
        IMatrix<Double> cov = getCovariance();
        IVector<Double> stdDev = getStandardDeviation();
        
        double[][] corrArray = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                if (i == j) {
                    corrArray[i][j] = 1.0;
                } else {
                    corrArray[i][j] = cov.get(i, j) / (stdDev.get(i) * stdDev.get(j));
                }
            }
        }
        
        return Linalg.matrix(corrArray);
    }
    
    @Override
    public IMatrix<Double> getPrecision() {
        return getCovariance().inv();
    }
    
    @Override
    public IVector<Double> getStandardDeviation() {
        IMatrix<Double> cov = getCovariance();
        double[] stdDevArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            stdDevArray[i] = Math.sqrt(cov.get(i, i));
        }
        return Linalg.vector(stdDevArray);
    }
    
    @Override
    public double mahalanobisDistance(IVector<Double> x) {
        return Math.sqrt(squaredMahalanobisDistance(x));
    }
    
    @Override
    public double squaredMahalanobisDistance(IVector<Double> x) {
        validateDimension(x);
        
        IVector<Double> mean = getMean();
        IVector<Double> diff = x.sub(mean);
        IMatrix<Double> precision = getPrecision();
        IVector<Double> temp = precision.mmul(diff);
        return diff.dot(temp);
    }
    
    @Override
    public IVector<Double> sample() {
        // 使用Gamma分布采样方法
        double[] gammaValues = new double[dimension];
        double sum = 0.0;
        
        // 从Gamma(αᵢ, 1)分布采样
        for (int i = 0; i < dimension; i++) {
            gammaValues[i] = sampleGamma(alpha.get(i), 1.0);
            sum += gammaValues[i];
        }
        
        // 归一化到单纯形
        double[] sampleArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            sampleArray[i] = gammaValues[i] / sum;
        }
        
        return Linalg.vector(sampleArray);
    }
    
    /**
     * 从Gamma分布采样
     * Sample from Gamma distribution
     */
    private double sampleGamma(double shape, double scale) {
        // 使用Marsaglia and Tsang方法
        if (shape < 1.0) {
            // 对于shape < 1，使用变换方法
            double u = random.nextDouble();
            return sampleGamma(shape + 1.0, scale) * Math.pow(u, 1.0 / shape);
        }
        
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        
        while (true) {
            double x, v;
            do {
                x = random.nextGaussian();
                v = 1.0 + c * x;
            } while (v <= 0);
            
            v = v * v * v;
            double u = random.nextDouble();
            
            if (u < 1.0 - 0.0331 * x * x * x * x) {
                return d * v * scale;
            }
            
            if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) {
                return d * v * scale;
            }
        }
    }
    
    @Override
    public List<IVector<Double>> sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("样本数量必须大于0");
        }
        
        List<IVector<Double>> samples = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            samples.add(sample());
        }
        return samples;
    }
    
    @Override
    public IMatrix<Double> sampleMatrix(int n) {
        List<IVector<Double>> samples = sample(n);
        double[][] sampleArray = new double[n][dimension];
        
        for (int i = 0; i < n; i++) {
            IVector<Double> sample = samples.get(i);
            for (int j = 0; j < dimension; j++) {
                sampleArray[i][j] = sample.get(j);
            }
        }
        
        return Linalg.matrix(sampleArray);
    }
    
    @Override
    public IMultivariateDistribution<Double> getMarginal(int... indices) {
        if (indices == null || indices.length == 0) {
            throw new IllegalArgumentException("索引不能为空");
        }
        
        // 验证索引有效性
        for (int index : indices) {
            if (index < 0 || index >= dimension) {
                throw new IllegalArgumentException("索引超出范围: " + index);
            }
        }
        
        // Dirichlet分布的边际分布仍是Dirichlet分布
        double[] marginalAlphaArray = new double[indices.length];
        for (int i = 0; i < indices.length; i++) {
            marginalAlphaArray[i] = alpha.get(indices[i]);
        }
        
        IVector<Double> marginalAlpha = Linalg.vector(marginalAlphaArray);
        return new MultivariateBetaDistribution(marginalAlpha, random);
    }
    
    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        // Dirichlet分布的条件分布比较复杂，这里提供简化实现
        throw new UnsupportedOperationException("Dirichlet分布的条件分布计算较为复杂，暂不支持");
    }
    
    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        // 线性变换后的Dirichlet分布一般不再是Dirichlet分布
        throw new UnsupportedOperationException("Dirichlet分布的线性变换结果一般不再是Dirichlet分布");
    }
    
    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        return linearTransform(A, Linalg.zeros(A.rows()));
    }
    
    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        if (!(other instanceof MultivariateBetaDistribution)) {
            throw new IllegalArgumentException("只支持与其他Dirichlet分布计算KL散度");
        }
        
        MultivariateBetaDistribution otherDir = (MultivariateBetaDistribution) other;
        if (otherDir.dimension != this.dimension) {
            throw new IllegalArgumentException("分布维度必须相同");
        }
        
        // KL散度公式比较复杂，这里提供近似计算
        double klDiv = 0.0;
        
        // 使用对数归一化常数的差
        klDiv += otherDir.logNormalizationConstant - this.logNormalizationConstant;
        
        // 添加期望项
        for (int i = 0; i < dimension; i++) {
            double alphaP = this.alpha.get(i);
            double alphaQ = otherDir.alpha.get(i);
            klDiv += (alphaP - alphaQ) * (digamma(alphaP) - digamma(this.alphaSum));
        }
        
        return klDiv;
    }
    
    /**
     * Digamma函数的近似计算
     * Approximate computation of digamma function
     */
    private double digamma(double x) {
        // 使用渐近展开
        if (x > 6.0) {
            return Math.log(x) - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x);
        }
        
        // 使用递推关系
        return digamma(x + 1.0) - 1.0 / x;
    }
    
    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        throw new UnsupportedOperationException("Dirichlet分布的Wasserstein距离计算较为复杂，暂不支持");
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples) {
        return fitFromSamples(samples);
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples, List<Double> weights) {
        return fitFromWeightedSamples(samples, weights);
    }
    
    @Override
    public boolean isElliptical() {
        return false; // Dirichlet分布不是椭圆分布
    }
    
    @Override
    public boolean isSymmetric() {
        // 只有当所有αᵢ相等时才对称
        double firstAlpha = alpha.get(0);
        for (int i = 1; i < dimension; i++) {
            if (Math.abs(alpha.get(i) - firstAlpha) > 1e-10) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public boolean isPositiveDefinite() {
        return true; // 协方差矩阵是正定的（在单纯形约束下）
    }
    
    @Override
    public void validateDimension(IVector<Double> x) {
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x.length() != dimension) {
            throw new IllegalArgumentException(
                String.format("输入向量维度不匹配：期望 %d，实际 %d", dimension, x.length()));
        }
    }
    
    @Override
    public double entropy() {
        // 熵公式：H = log B(α) + (α₀ - k) ψ(α₀) - ∑(αᵢ - 1) ψ(αᵢ)
        // 其中 B(α) 是Beta函数，ψ是digamma函数
        double entropy = -logNormalizationConstant; // -log B(α)
        entropy += (alphaSum - dimension) * digamma(alphaSum);
        
        for (int i = 0; i < dimension; i++) {
            entropy -= (alpha.get(i) - 1.0) * digamma(alpha.get(i));
        }
        
        return entropy;
    }
    
    @Override
    public IMatrix<Double> informationMatrix() {
        // 信息矩阵计算较为复杂，这里提供简化版本
        return getPrecision();
    }
    
    @Override
    public IMultivariateDistribution<Double> conjugateUpdate(IVector<Double> observations) {
        // For Dirichlet distribution, conjugate update with Multinomial observations
        // Posterior alpha = prior alpha + observations
        IVector<Double> posteriorAlpha = alpha.add(observations);
        return new MultivariateBetaDistribution(posteriorAlpha);
    }
    
    @Override
    public double marginalLikelihood(IVector<Double> observations) {
        // For Dirichlet-Multinomial conjugate pair, compute marginal likelihood
        double observationSum = 0.0;
        for (int i = 0; i < observations.size(); i++) {
            observationSum += observations.get(i);
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
    
    /**
     * 计算多元Beta函数的对数值
     * Calculate log multivariate Beta function
     */
    private double logMultivariateBeta(IVector<Double> alpha) {
        double logBeta = logGamma(alphaSum);
        
        for (int i = 0; i < alpha.size(); i++) {
            logBeta -= logGamma(alpha.get(i));
        }
        
        return logBeta;
    }
    
    @Override
    public ConfidenceEllipse getConfidenceEllipse(double confidence) {
        if (dimension != 2) {
            throw new UnsupportedOperationException("置信椭圆只支持二维分布");
        }
        if (confidence <= 0 || confidence >= 1) {
            throw new IllegalArgumentException("置信水平必须在(0,1)范围内");
        }
        
        // 对于二维Dirichlet分布，置信区域不是椭圆，这里用椭圆近似
        IVector<Double> mean = getMean();
        IVector<Double> stdDev = getStandardDeviation();
        
        // 使用卡方分位数
        double chiSquareQuantile = 2.0 * Math.log(1.0 / (1.0 - confidence));
        double majorAxis = stdDev.get(0) * Math.sqrt(chiSquareQuantile);
        double minorAxis = stdDev.get(1) * Math.sqrt(chiSquareQuantile);
        
        return new ConfidenceEllipse(mean, majorAxis, minorAxis, 0.0);
    }
    
    // ==================== 静态工厂方法 ====================
    
    /**
     * 从样本数据估计Dirichlet分布参数
     * Estimate Dirichlet distribution parameters from sample data
     */
    public static MultivariateBetaDistribution fitFromSamples(List<IVector<Double>> samples) {
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("样本数据不能为空");
        }
        
        int n = samples.size();
        int dimension = samples.get(0).length();
        
        // 验证所有样本都在单纯形上
        for (IVector<Double> sample : samples) {
            if (sample.length() != dimension) {
                throw new IllegalArgumentException("所有样本维度必须一致");
            }
            
            double sum = 0.0;
            for (int i = 0; i < dimension; i++) {
                double xi = sample.get(i);
                if (xi < 0 || xi > 1) {
                    throw new IllegalArgumentException("Dirichlet分布的样本值必须在[0,1]范围内");
                }
                sum += xi;
            }
            if (Math.abs(sum - 1.0) > 1e-6) {
                throw new IllegalArgumentException("Dirichlet分布的样本值必须和为1");
            }
        }
        
        // 使用矩估计方法
        // 计算样本均值和方差
        double[] meanArray = new double[dimension];
        double[] varArray = new double[dimension];
        
        for (IVector<Double> sample : samples) {
            for (int i = 0; i < dimension; i++) {
                meanArray[i] += sample.get(i);
            }
        }
        for (int i = 0; i < dimension; i++) {
            meanArray[i] /= n;
        }
        
        for (IVector<Double> sample : samples) {
            for (int i = 0; i < dimension; i++) {
                double diff = sample.get(i) - meanArray[i];
                varArray[i] += diff * diff;
            }
        }
        for (int i = 0; i < dimension; i++) {
            varArray[i] /= (n - 1);
        }
        
        // 使用矩估计公式：αᵢ = μᵢ(μᵢ(1-μᵢ)/σᵢ² - 1)
        double[] alphaArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            double mu = meanArray[i];
            double sigma2 = varArray[i];
            if (sigma2 <= 0 || mu <= 0 || mu >= 1) {
                throw new IllegalArgumentException("样本统计量不满足Dirichlet分布的要求");
            }
            alphaArray[i] = mu * (mu * (1.0 - mu) / sigma2 - 1.0);
            if (alphaArray[i] <= 0) {
                alphaArray[i] = 0.1; // 设置最小值
            }
        }
        
        IVector<Double> alpha = Linalg.vector(alphaArray);
        return new MultivariateBetaDistribution(alpha);
    }
    
    /**
     * 从加权样本数据估计Dirichlet分布参数
     * Estimate Dirichlet distribution parameters from weighted sample data
     */
    public static MultivariateBetaDistribution fitFromWeightedSamples(List<IVector<Double>> samples, 
                                                                    List<Double> weights) {
        if (samples == null || samples.isEmpty() || weights == null || weights.isEmpty()) {
            throw new IllegalArgumentException("样本数据和权重不能为空");
        }
        if (samples.size() != weights.size()) {
            throw new IllegalArgumentException("样本数量和权重数量必须相等");
        }
        
        int n = samples.size();
        int dimension = samples.get(0).length();
        
        // 计算权重总和
        double weightSum = weights.stream().mapToDouble(Double::doubleValue).sum();
        
        // 计算加权均值和方差
        double[] meanArray = new double[dimension];
        double[] varArray = new double[dimension];
        
        for (int i = 0; i < n; i++) {
            IVector<Double> sample = samples.get(i);
            double weight = weights.get(i);
            for (int j = 0; j < dimension; j++) {
                meanArray[j] += weight * sample.get(j);
            }
        }
        for (int i = 0; i < dimension; i++) {
            meanArray[i] /= weightSum;
        }
        
        for (int i = 0; i < n; i++) {
            IVector<Double> sample = samples.get(i);
            double weight = weights.get(i);
            for (int j = 0; j < dimension; j++) {
                double diff = sample.get(j) - meanArray[j];
                varArray[j] += weight * diff * diff;
            }
        }
        for (int i = 0; i < dimension; i++) {
            varArray[i] /= weightSum;
        }
        
        // 使用矩估计公式
        double[] alphaArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            double mu = meanArray[i];
            double sigma2 = varArray[i];
            if (sigma2 <= 0 || mu <= 0 || mu >= 1) {
                throw new IllegalArgumentException("加权样本统计量不满足Dirichlet分布的要求");
            }
            alphaArray[i] = mu * (mu * (1.0 - mu) / sigma2 - 1.0);
            if (alphaArray[i] <= 0) {
                alphaArray[i] = 0.1; // 设置最小值
            }
        }
        
        IVector<Double> alpha = Linalg.vector(alphaArray);
        return new MultivariateBetaDistribution(alpha);
    }
    
    /**
     * 创建对称Dirichlet分布（所有参数相等）
     * Create symmetric Dirichlet distribution (all parameters equal)
     */
    public static MultivariateBetaDistribution symmetric(int dimension, double alpha) {
        if (dimension <= 1) {
            throw new IllegalArgumentException("维度必须大于1");
        }
        if (alpha <= 0) {
            throw new IllegalArgumentException("形状参数必须大于0");
        }
        
        double[] alphaArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            alphaArray[i] = alpha;
        }
        
        IVector<Double> alphaVector = Linalg.vector(alphaArray);
        return new MultivariateBetaDistribution(alphaVector);
    }
    
    /**
     * 创建均匀Dirichlet分布（所有参数为1）
     * Create uniform Dirichlet distribution (all parameters equal to 1)
     */
    public static MultivariateBetaDistribution uniform(int dimension) {
        return symmetric(dimension, 1.0);
    }
    
    /**
     * 获取形状参数向量
     * Get shape parameters vector
     */
    public IVector<Double> getAlpha() {
        return alpha.copy();
    }
    
    /**
     * 获取参数总和
     * Get sum of parameters
     */
    public double getAlphaSum() {
        return alphaSum;
    }
    
    /**
     * 计算浓度参数（参数总和）
     * Compute concentration parameter (sum of parameters)
     */
    public double getConcentration() {
        return alphaSum;
    }
    
    /**
     * 检查是否为对称分布
     * Check if distribution is symmetric
     */
    public boolean isSymmetricDistribution() {
        double firstAlpha = alpha.get(0);
        for (int i = 1; i < dimension; i++) {
            if (Math.abs(alpha.get(i) - firstAlpha) > 1e-10) {
                return false;
            }
        }
        return true;
    }
    
    @Override
    public String toString() {
        return String.format("MultivariateBetaDistribution(dimension=%d, alpha=%s, sum=%.6f)", 
                           dimension, alpha.toString(), alphaSum);
    }
}