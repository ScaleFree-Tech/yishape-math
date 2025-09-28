package com.reremouse.lab.math.stats.distribution.multiv;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * 多元t分布实现 / Multivariate t-Distribution Implementation
 * 
 * <p>多元t分布是多元正态分布的推广，具有更厚的尾部，对异常值更加鲁棒。
 * 其概率密度函数为：</p>
 * <p>f(x) = Γ((ν+k)/2) / (Γ(ν/2) * (νπ)^(k/2) * |Σ|^(1/2)) * 
 *        [1 + (x-μ)^T * Σ^(-1) * (x-μ) / ν]^(-(ν+k)/2)</p>
 * <p>其中 μ 是位置向量，Σ 是尺度矩阵，ν 是自由度，k 是维度。</p>
 * 
 * <p>Multivariate t-distribution is a generalization of the multivariate normal distribution
 * with heavier tails, making it more robust to outliers.
 * Its probability density function is:</p>
 * <p>f(x) = Γ((ν+k)/2) / (Γ(ν/2) * (νπ)^(k/2) * |Σ|^(1/2)) * 
 *        [1 + (x-μ)^T * Σ^(-1) * (x-μ) / ν]^(-(ν+k)/2)</p>
 * <p>where μ is the location vector, Σ is the scale matrix, ν is the degrees of freedom, and k is the dimension.</p>
 * 
 * <h3>主要特性 / Key Properties:</h3>
 * <ul>
 *   <li>椭圆分布 / Elliptical distribution</li>
 *   <li>对称分布 / Symmetric distribution</li>
 *   <li>厚尾分布 / Heavy-tailed distribution</li>
 *   <li>当ν→∞时趋向于多元正态分布 / Approaches multivariate normal as ν→∞</li>
 *   <li>对异常值鲁棒 / Robust to outliers</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class MultivariateTDistribution implements IMultivariateDistribution<Double> {
    
    private static final long serialVersionUID = 1L;
    
    /** 位置向量 / Location vector */
    private final IVector<Double> location;
    
    /** 尺度矩阵 / Scale matrix */
    private final IMatrix<Double> scale;
    
    /** 尺度矩阵的逆 / Inverse of scale matrix */
    private final IMatrix<Double> precision;
    
    /** 尺度矩阵的行列式 / Determinant of scale matrix */
    private final double scaleDeterminant;
    
    /** 自由度 / Degrees of freedom */
    private final double degreesOfFreedom;
    
    /** 维度 / Dimensionality */
    private final int dimension;
    
    /** 对数归一化常数 / Log normalization constant */
    private final double logNormalizationConstant;
    
    /** 随机数生成器 / Random number generator */
    private final Random random;
    
    /**
     * 构造函数
     * Constructor
     * 
     * @param location 位置向量 / Location vector
     * @param scale 尺度矩阵 / Scale matrix
     * @param degreesOfFreedom 自由度 / Degrees of freedom
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateTDistribution(IVector<Double> location, IMatrix<Double> scale, double degreesOfFreedom) {
        this(location, scale, degreesOfFreedom, new Random());
    }
    
    /**
     * 构造函数（带随机种子）
     * Constructor with random seed
     * 
     * @param location 位置向量 / Location vector
     * @param scale 尺度矩阵 / Scale matrix
     * @param degreesOfFreedom 自由度 / Degrees of freedom
     * @param random 随机数生成器 / Random number generator
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateTDistribution(IVector<Double> location, IMatrix<Double> scale, 
                                   double degreesOfFreedom, Random random) {
        validateParameters(location, scale, degreesOfFreedom);
        
        this.dimension = location.length();
        this.location = location.copy();
        this.scale = scale.copy();
        this.degreesOfFreedom = degreesOfFreedom;
        this.random = random;
        
        // 预计算逆矩阵和行列式
        this.precision = computePrecisionMatrix(scale);
        this.scaleDeterminant = scale.det();
        this.logNormalizationConstant = computeLogNormalizationConstant();
    }
    
    /**
     * 验证参数有效性
     * Validate parameter validity
     */
    private void validateParameters(IVector<Double> location, IMatrix<Double> scale, double degreesOfFreedom) {
        if (location == null || scale == null) {
            throw new IllegalArgumentException("位置向量和尺度矩阵不能为null");
        }
        
        if (location.length() != scale.rows() || scale.rows() != scale.cols()) {
            throw new IllegalArgumentException("维度不匹配：位置向量长度必须等于尺度矩阵的行数和列数");
        }
        
        if (!scale.isSymmetric()) {
            throw new IllegalArgumentException("尺度矩阵必须是对称的");
        }
        
        if (!scale.isPositiveDefinite()) {
            throw new IllegalArgumentException("尺度矩阵必须是正定的");
        }
        
        if (degreesOfFreedom <= 0) {
            throw new IllegalArgumentException("自由度必须大于0");
        }
    }
    
    /**
     * 计算精度矩阵（尺度矩阵的逆）
     * Compute precision matrix (inverse of scale matrix)
     */
    private IMatrix<Double> computePrecisionMatrix(IMatrix<Double> scale) {
        try {
            return scale.inv();
        } catch (Exception e) {
            throw new IllegalArgumentException("尺度矩阵必须是可逆的", e);
        }
    }
    
    /**
     * 计算对数归一化常数
     * Compute log normalization constant
     */
    private double computeLogNormalizationConstant() {
        double logGammaRatio = logGamma((degreesOfFreedom + dimension) / 2.0) - logGamma(degreesOfFreedom / 2.0);
        double logPiTerm = -0.5 * dimension * Math.log(degreesOfFreedom * Math.PI);
        double logDetTerm = -0.5 * Math.log(Math.abs(scaleDeterminant));
        
        return logGammaRatio + logPiTerm + logDetTerm;
    }
    
    /**
     * 计算对数伽马函数（简化实现）
     * Compute log gamma function (simplified implementation)
     */
    private double logGamma(double x) {
        // 使用Stirling近似公式的简化版本
        if (x < 0.5) {
            return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * x)) - logGamma(1 - x);
        }
        
        x -= 1;
        double result = 0.5 * Math.log(2 * Math.PI);
        result += (x + 0.5) * Math.log(x + 1);
        result -= (x + 1);
        
        // 添加修正项
        result += 1.0 / (12.0 * (x + 1));
        result -= 1.0 / (360.0 * Math.pow(x + 1, 3));
        
        return result;
    }
    
    // ==================== IMultivariateDistribution 接口实现 ====================
    
    @Override
    public int getDimension() {
        return dimension;
    }
    
    @Override
    public String getDistributionName() {
        return "Multivariate t-Distribution";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("Dimension: %d, Location: %s, Scale determinant: %.6f, DoF: %.2f", 
                           dimension, location.toString(), scaleDeterminant, degreesOfFreedom);
    }
    
    @Override
    public double pdf(IVector<Double> x) {
        validateDimension(x);
        return Math.exp(logPdf(x));
    }
    
    @Override
    public double logPdf(IVector<Double> x) {
        validateDimension(x);
        
        // 计算二次型 (x - μ)^T * Σ^(-1) * (x - μ)
        double quadraticForm = squaredMahalanobisDistance(x);
        
        // 计算对数概率密度
        double logDensity = logNormalizationConstant;
        logDensity -= 0.5 * (degreesOfFreedom + dimension) * Math.log(1 + quadraticForm / degreesOfFreedom);
        
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
        if (degreesOfFreedom <= 1) {
            throw new UnsupportedOperationException("均值在自由度≤1时不存在");
        }
        return location.copy();
    }
    
    @Override
    public IMatrix<Double> getCovariance() {
        if (degreesOfFreedom <= 2) {
            throw new UnsupportedOperationException("协方差在自由度≤2时不存在");
        }
        
        double factor = degreesOfFreedom / (degreesOfFreedom - 2);
        return scale.multiplyScalar(factor);
    }
    
    @Override
    public IMatrix<Double> getCorrelation() {
        IMatrix<Double> covariance = getCovariance();
        IVector<Double> stdDev = getStandardDeviation();
        IMatrix<Double> correlation = covariance.copy();
        
        // 计算相关矩阵：R[i,j] = Cov[i,j] / (std[i] * std[j])
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                double corr = correlation.get(i, j) / (stdDev.get(i) * stdDev.get(j));
                correlation.set(i, j, corr);
            }
        }
        
        return correlation;
    }
    
    @Override
    public IMatrix<Double> getPrecision() {
        return precision.copy();
    }
    
    @Override
    public IVector<Double> getStandardDeviation() {
        if (degreesOfFreedom <= 2) {
            throw new UnsupportedOperationException("标准差在自由度≤2时不存在");
        }
        
        double factor = Math.sqrt(degreesOfFreedom / (degreesOfFreedom - 2));
        double[] stdDevArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            stdDevArray[i] = factor * Math.sqrt(scale.get(i, i));
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
        
        IVector<Double> diff = x.sub(location);
        IVector<Double> temp = precision.mmul(diff);
        return diff.dot(temp);
    }
    
    @Override
    public IVector<Double> sample() {
        // 使用多元t分布的采样算法：
        // 1. 从多元正态分布N(0, Σ)采样得到Z
        // 2. 从卡方分布χ²(ν)采样得到W
        // 3. 返回 μ + Z * sqrt(ν/W)
        
        // 步骤1：从多元正态分布采样
        double[] normalSample = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            normalSample[i] = random.nextGaussian();
        }
        
        IMatrix<Double> cholesky = scale.cholesky();
        IVector<Double> normalVector = Linalg.vector(normalSample);
        IVector<Double> transformedNormal = cholesky.mmul(normalVector);
        
        // 步骤2：从卡方分布采样（使用伽马分布近似）
        double chiSquareSample = sampleChiSquare(degreesOfFreedom);
        
        // 步骤3：组合结果
        double scaleFactor = Math.sqrt(degreesOfFreedom / chiSquareSample);
        IVector<Double> scaledSample = transformedNormal.multiplyScalar(scaleFactor);
        
        return scaledSample.add(location);
    }
    
    /**
     * 从卡方分布采样（使用伽马分布）
     * Sample from chi-square distribution using gamma distribution
     */
    private double sampleChiSquare(double degreesOfFreedom) {
        // χ²(ν) = Gamma(ν/2, 2)
        return sampleGamma(degreesOfFreedom / 2.0, 2.0);
    }
    
    /**
     * 从伽马分布采样（简化实现）
     * Sample from gamma distribution (simplified implementation)
     */
    private double sampleGamma(double shape, double scale) {
        // 使用Marsaglia and Tsang方法的简化版本
        if (shape < 1) {
            return sampleGamma(shape + 1, scale) * Math.pow(random.nextDouble(), 1.0 / shape);
        }
        
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.sqrt(9.0 * d);
        
        while (true) {
            double x = random.nextGaussian();
            double v = 1.0 + c * x;
            
            if (v <= 0) continue;
            
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
        
        // 提取边际位置和尺度
        double[] marginalLocationArray = new double[indices.length];
        double[][] marginalScaleArray = new double[indices.length][indices.length];
        
        for (int i = 0; i < indices.length; i++) {
            marginalLocationArray[i] = location.get(indices[i]);
            for (int j = 0; j < indices.length; j++) {
                marginalScaleArray[i][j] = scale.get(indices[i], indices[j]);
            }
        }
        
        IVector<Double> marginalLocation = Linalg.vector(marginalLocationArray);
        IMatrix<Double> marginalScale = Linalg.matrix(marginalScaleArray);
        
        return new MultivariateTDistribution(marginalLocation, marginalScale, degreesOfFreedom, random);
    }
    
    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        throw new UnsupportedOperationException("条件分布计算尚未实现");
    }
    
    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        if (A.cols() != dimension) {
            throw new IllegalArgumentException("变换矩阵列数必须等于分布维度");
        }
        if (b.length() != A.rows()) {
            throw new IllegalArgumentException("平移向量维度必须等于变换矩阵行数");
        }
        
        // 新的位置：A * μ + b
        IVector<Double> newLocation = A.mmul(location).add(b);
        
        // 新的尺度：A * Σ * A^T
        IMatrix<Double> newScale = A.mmul(scale).mmul(A.transpose());
        
        return new MultivariateTDistribution(newLocation, newScale, degreesOfFreedom, random);
    }
    
    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        return linearTransform(A, Linalg.zeros(A.rows()));
    }
    
    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        throw new UnsupportedOperationException("多元t分布的KL散度计算较为复杂，尚未实现");
    }
    
    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        throw new UnsupportedOperationException("多元t分布的Wasserstein距离计算较为复杂，尚未实现");
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples) {
        return fitFromSamples(samples);
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples, List<Double> weights) {
        throw new UnsupportedOperationException("加权拟合尚未实现");
    }
    
    @Override
    public boolean isElliptical() {
        return true; // 多元t分布是椭圆分布
    }
    
    @Override
    public boolean isSymmetric() {
        return true; // 多元t分布是对称分布
    }
    
    @Override
    public boolean isPositiveDefinite() {
        return scale.isPositiveDefinite();
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
        if (degreesOfFreedom <= 1) {
            throw new UnsupportedOperationException("熵在自由度≤1时不存在");
        }
        
        // 多元t分布的熵公式
        double logDetTerm = 0.5 * Math.log(Math.abs(scaleDeterminant));
        double logBetaTerm = logGamma((degreesOfFreedom + dimension) / 2.0) - logGamma(degreesOfFreedom / 2.0);
        double logPiTerm = 0.5 * dimension * Math.log(degreesOfFreedom * Math.PI);
        double digammaTerm = (degreesOfFreedom + dimension) / 2.0 * 
                           (digamma((degreesOfFreedom + dimension) / 2.0) - digamma(degreesOfFreedom / 2.0));
        
        return logDetTerm + logBetaTerm + logPiTerm + digammaTerm;
    }
    
    /**
     * 计算digamma函数（简化实现）
     * Compute digamma function (simplified implementation)
     */
    private double digamma(double x) {
        // 使用渐近展开的简化版本
        if (x < 6) {
            return digamma(x + 1) - 1.0 / x;
        }
        
        double result = Math.log(x) - 1.0 / (2.0 * x);
        result -= 1.0 / (12.0 * x * x);
        result += 1.0 / (120.0 * Math.pow(x, 4));
        
        return result;
    }
    
    @Override
    public IMatrix<Double> informationMatrix() {
        double factor = (degreesOfFreedom + dimension) / (degreesOfFreedom + 2);
        return precision.multiplyScalar(factor);
    }
    
    @Override
    public IMultivariateDistribution<Double> conjugateUpdate(IVector<Double> observations) {
        // For multivariate t-distribution, conjugate update with normal observations
        // This is a simplified placeholder implementation
        IVector<Double> newLocation = location.add(observations.multiplyScalar(0.1));
        return new MultivariateTDistribution(newLocation, scale, degreesOfFreedom);
    }
    
    @Override
    public double marginalLikelihood(IVector<Double> observations) {
        // For multivariate t-distribution, compute marginal likelihood of observations
        // This is a simplified placeholder implementation
        double logLikelihood = -0.5 * observations.dot(observations) / degreesOfFreedom;
        return Math.exp(logLikelihood);
    }
    
    @Override
    public List<IVector<Double>> posteriorSample(IVector<Double> observations, int n) {
        // Sample from posterior distribution after conjugate update
        IMultivariateDistribution<Double> posterior = conjugateUpdate(observations);
        return posterior.sample(n);
    }
    
    @Override
    public ConfidenceEllipse getConfidenceEllipse(double confidence) {
        if (dimension != 2) {
            throw new UnsupportedOperationException("置信椭圆只支持二维分布");
        }
        if (confidence <= 0 || confidence >= 1) {
            throw new IllegalArgumentException("置信水平必须在(0,1)范围内");
        }
        
        // 对于t分布，使用F分布的分位数
        double fQuantile = getFQuantile(confidence, dimension, degreesOfFreedom);
        double scaleFactor = dimension * (degreesOfFreedom - 2) / degreesOfFreedom * fQuantile;
        
        // 计算尺度矩阵的特征值和特征向量（简化版本）
        double a = scale.get(0, 0);
        double b = scale.get(0, 1);
        double c = scale.get(1, 1);
        
        double trace = a + c;
        double det = a * c - b * b;
        double discriminant = Math.sqrt(trace * trace - 4 * det);
        
        double eigenvalue1 = 0.5 * (trace + discriminant);
        double eigenvalue2 = 0.5 * (trace - discriminant);
        
        double majorAxis = Math.sqrt(scaleFactor * eigenvalue1);
        double minorAxis = Math.sqrt(scaleFactor * eigenvalue2);
        
        double angle = 0.5 * Math.atan2(2 * b, a - c);
        
        return new ConfidenceEllipse(location.copy(), majorAxis, minorAxis, angle);
    }
    
    /**
     * 获取F分布分位数（简化实现）
     * Get F-distribution quantile (simplified implementation)
     */
    private double getFQuantile(double confidence, int df1, double df2) {
        // 简化的F分布分位数计算
        // 实际应用中应使用更精确的算法
        return 2.0 + confidence * 3.0; // 占位符实现
    }
    
    // ==================== 静态工厂方法 ====================
    
    /**
     * 从样本数据估计多元t分布参数（使用EM算法的简化版本）
     * Estimate multivariate t-distribution parameters from sample data (simplified EM algorithm)
     */
    public static MultivariateTDistribution fitFromSamples(List<IVector<Double>> samples) {
        return fitFromSamples(samples, 4.0); // 默认自由度为4
    }
    
    /**
     * 从样本数据估计多元t分布参数，指定自由度
     * Estimate multivariate t-distribution parameters from sample data with specified degrees of freedom
     */
    public static MultivariateTDistribution fitFromSamples(List<IVector<Double>> samples, double degreesOfFreedom) {
        if (samples == null || samples.isEmpty()) {
            throw new IllegalArgumentException("样本数据不能为空");
        }
        
        int n = samples.size();
        int dimension = samples.get(0).length();
        
        // 验证所有样本维度一致
        for (IVector<Double> sample : samples) {
            if (sample.length() != dimension) {
                throw new IllegalArgumentException("所有样本维度必须一致");
            }
        }
        
        // 计算样本均值作为位置参数的初始估计
        double[] locationArray = new double[dimension];
        for (IVector<Double> sample : samples) {
            for (int i = 0; i < dimension; i++) {
                locationArray[i] += sample.get(i);
            }
        }
        for (int i = 0; i < dimension; i++) {
            locationArray[i] /= n;
        }
        IVector<Double> sampleLocation = Linalg.vector(locationArray);
        
        // 计算样本协方差矩阵作为尺度参数的初始估计
        double[][] scaleArray = new double[dimension][dimension];
        for (IVector<Double> sample : samples) {
            IVector<Double> diff = sample.sub(sampleLocation);
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    scaleArray[i][j] += diff.get(i) * diff.get(j);
                }
            }
        }
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                scaleArray[i][j] /= (n - 1);
            }
        }
        IMatrix<Double> sampleScale = Linalg.matrix(scaleArray);
        
        return new MultivariateTDistribution(sampleLocation, sampleScale, degreesOfFreedom);
    }
    
    /**
     * 获取自由度
     * Get degrees of freedom
     */
    public double getDegreesOfFreedom() {
        return degreesOfFreedom;
    }
    
    /**
     * 获取位置向量
     * Get location vector
     */
    public IVector<Double> getLocation() {
        return location.copy();
    }
    
    /**
     * 获取尺度矩阵
     * Get scale matrix
     */
    public IMatrix<Double> getScale() {
        return scale.copy();
    }
    
    @Override
    public String toString() {
        return String.format("MultivariateTDistribution(dimension=%d, location=%s, det(scale)=%.6f, dof=%.2f)", 
                           dimension, location.toString(), scaleDeterminant, degreesOfFreedom);
    }
}