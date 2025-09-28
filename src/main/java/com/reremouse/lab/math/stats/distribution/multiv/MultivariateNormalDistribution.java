package com.reremouse.lab.math.stats.distribution.multiv;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * 多元正态分布实现 / Multivariate Normal Distribution Implementation
 * 
 * <p>多元正态分布（也称为多元高斯分布）是单变量正态分布在多维空间的推广。
 * 其概率密度函数为：</p>
 * <p>f(x) = (2π)^(-k/2) * |Σ|^(-1/2) * exp(-1/2 * (x-μ)^T * Σ^(-1) * (x-μ))</p>
 * <p>其中 μ 是均值向量，Σ 是协方差矩阵，k 是维度。</p>
 * 
 * <p>Multivariate normal distribution (also known as multivariate Gaussian distribution) 
 * is the generalization of the univariate normal distribution to multiple dimensions.
 * Its probability density function is:</p>
 * <p>f(x) = (2π)^(-k/2) * |Σ|^(-1/2) * exp(-1/2 * (x-μ)^T * Σ^(-1) * (x-μ))</p>
 * <p>where μ is the mean vector, Σ is the covariance matrix, and k is the dimension.</p>
 * 
 * <h3>主要特性 / Key Properties:</h3>
 * <ul>
 *   <li>椭圆分布 / Elliptical distribution</li>
 *   <li>对称分布 / Symmetric distribution</li>
 *   <li>边际分布仍为正态分布 / Marginal distributions are normal</li>
 *   <li>线性变换后仍为正态分布 / Linear transformations preserve normality</li>
 * </ul>
 * 
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class MultivariateNormalDistribution implements IMultivariateDistribution<Double> {
    
    private static final long serialVersionUID = 1L;
    
    /** 均值向量 / Mean vector */
    private final IVector<Double> mean;
    
    /** 协方差矩阵 / Covariance matrix */
    private final IMatrix<Double> covariance;
    
    /** 协方差矩阵的逆 / Inverse of covariance matrix */
    private final IMatrix<Double> precision;
    
    /** 协方差矩阵的行列式 / Determinant of covariance matrix */
    private final double covarianceDeterminant;
    
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
     * @param mean 均值向量 / Mean vector
     * @param covariance 协方差矩阵 / Covariance matrix
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateNormalDistribution(IVector<Double> mean, IMatrix<Double> covariance) {
        this(mean, covariance, new Random());
    }
    
    /**
     * 构造函数（带随机种子）
     * Constructor with random seed
     * 
     * @param mean 均值向量 / Mean vector
     * @param covariance 协方差矩阵 / Covariance matrix
     * @param random 随机数生成器 / Random number generator
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public MultivariateNormalDistribution(IVector<Double> mean, IMatrix<Double> covariance, Random random) {
        validateParameters(mean, covariance);
        
        this.dimension = mean.length();
        this.mean = mean.copy();
        this.covariance = covariance.copy();
        this.random = random;
        
        // 预计算逆矩阵和行列式
        this.precision = computePrecisionMatrix(covariance);
        this.covarianceDeterminant = covariance.det();
        this.logNormalizationConstant = computeLogNormalizationConstant();
    }
    
    /**
     * 验证参数有效性
     * Validate parameter validity
     */
    private void validateParameters(IVector<Double> mean, IMatrix<Double> covariance) {
        if (mean == null || covariance == null) {
            throw new IllegalArgumentException("均值向量和协方差矩阵不能为null");
        }
        
        if (mean.length() != covariance.rows() || covariance.rows() != covariance.cols()) {
            throw new IllegalArgumentException("维度不匹配：均值向量长度必须等于协方差矩阵的行数和列数");
        }
        
        if (!covariance.isSymmetric()) {
            throw new IllegalArgumentException("协方差矩阵必须是对称的");
        }
        
        if (!covariance.isPositiveDefinite()) {
            throw new IllegalArgumentException("协方差矩阵必须是正定的");
        }
    }
    
    /**
     * 计算精度矩阵（协方差矩阵的逆）
     * Compute precision matrix (inverse of covariance matrix)
     */
    private IMatrix<Double> computePrecisionMatrix(IMatrix<Double> covariance) {
        try {
            return covariance.inv();
        } catch (Exception e) {
            throw new IllegalArgumentException("协方差矩阵必须是可逆的", e);
        }
    }
    
    /**
     * 计算对数归一化常数
     * Compute log normalization constant
     */
    private double computeLogNormalizationConstant() {
        return -0.5 * (dimension * Math.log(2 * Math.PI) + Math.log(Math.abs(covarianceDeterminant)));
    }
    
    // ==================== IMultivariateDistribution 接口实现 ====================
    
    @Override
    public int getDimension() {
        return dimension;
    }
    
    @Override
    public String getDistributionName() {
        return "Multivariate Normal Distribution";
    }
    
    @Override
    public String getParameterInfo() {
        return String.format("Dimension: %d, Mean: %s, Covariance determinant: %.6f", 
                           dimension, mean.toString(), covarianceDeterminant);
    }
    
    @Override
    public double pdf(IVector<Double> x) {
        validateDimension(x);
        return Math.exp(logPdf(x));
    }
    
    @Override
    public double logPdf(IVector<Double> x) {
        validateDimension(x);
        
        // 计算 (x - μ)
        IVector<Double> diff = x.sub(mean);
        
        // 计算二次型 (x - μ)^T * Σ^(-1) * (x - μ)
        double quadraticForm = squaredMahalanobisDistance(x);
        
        // 返回对数概率密度
        return logNormalizationConstant - 0.5 * quadraticForm;
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
        return mean.copy();
    }
    
    @Override
    public IMatrix<Double> getCovariance() {
        return covariance.copy();
    }
    
    @Override
    public IMatrix<Double> getCorrelation() {
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
        double[] stdDevArray = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            stdDevArray[i] = Math.sqrt(covariance.get(i, i));
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
        
        IVector<Double> diff = x.sub(mean);
        IVector<Double> temp = precision.mmul(diff);
        return diff.dot(temp);
    }
    
    @Override
    public IVector<Double> sample() {
        // 使用Box-Muller变换生成标准正态分布样本
        double[] standardSample = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            standardSample[i] = random.nextGaussian();
        }
        
        // 使用Cholesky分解进行变换
        IMatrix<Double> cholesky = covariance.cholesky();
        IVector<Double> standardVector = Linalg.vector(standardSample);
        IVector<Double> transformedSample = cholesky.mmul(standardVector);
        
        return transformedSample.add(mean);
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
        
        // 提取边际均值和协方差
        double[] marginalMeanArray = new double[indices.length];
        double[][] marginalCovArray = new double[indices.length][indices.length];
        
        for (int i = 0; i < indices.length; i++) {
            marginalMeanArray[i] = mean.get(indices[i]);
            for (int j = 0; j < indices.length; j++) {
                marginalCovArray[i][j] = covariance.get(indices[i], indices[j]);
            }
        }
        
        IVector<Double> marginalMean = Linalg.vector(marginalMeanArray);
        IMatrix<Double> marginalCov = Linalg.matrix(marginalCovArray);
        
        return new MultivariateNormalDistribution(marginalMean, marginalCov, random);
    }
    
    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        // 实现条件分布的计算（较复杂，这里提供基本框架）
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
        
        // 新的均值：A * μ + b
        IVector<Double> newMean = A.mmul(mean).add(b);
        
        // 新的协方差：A * Σ * A^T
        IMatrix<Double> newCovariance = A.mmul(covariance).mmul(A.transpose());
        
        return new MultivariateNormalDistribution(newMean, newCovariance, random);
    }
    
    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        return linearTransform(A, Linalg.zeros(A.rows()));
    }
    
    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        if (!(other instanceof MultivariateNormalDistribution)) {
            throw new IllegalArgumentException("只支持与其他多元正态分布计算KL散度");
        }
        
        MultivariateNormalDistribution otherNormal = (MultivariateNormalDistribution) other;
        if (otherNormal.dimension != this.dimension) {
            throw new IllegalArgumentException("分布维度必须相同");
        }
        
        // KL散度公式实现
        IVector<Double> meanDiff = otherNormal.mean.sub(this.mean);
        IMatrix<Double> otherPrecision = otherNormal.precision;
        
        double trace = this.covariance.mmul(otherPrecision).trace();
        double quadratic = meanDiff.asColumnVector().mmul(otherPrecision).mmul(meanDiff.asColumnVector()).get(0, 0);
        double logDet = Math.log(otherNormal.covarianceDeterminant / this.covarianceDeterminant);
        
        return 0.5 * (trace + quadratic - dimension + logDet);
    }
    
    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        if (!(other instanceof MultivariateNormalDistribution)) {
            throw new IllegalArgumentException("只支持与其他多元正态分布计算Wasserstein距离");
        }
        
        MultivariateNormalDistribution otherNormal = (MultivariateNormalDistribution) other;
        if (otherNormal.dimension != this.dimension) {
            throw new IllegalArgumentException("分布维度必须相同");
        }
        
        // Wasserstein距离公式实现（2-Wasserstein距离）
        IVector<Double> meanDiff = this.mean.sub(otherNormal.mean);
        double meanTerm = Math.pow(meanDiff.norm2(),2);
        
        // 协方差项的计算较复杂，这里提供简化版本
        double covTerm = this.covariance.frobeniusNorm() + otherNormal.covariance.frobeniusNorm();
        
        return Math.sqrt(meanTerm + covTerm);
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
        return true; // 多元正态分布是椭圆分布
    }
    
    @Override
    public boolean isSymmetric() {
        return true; // 多元正态分布是对称分布
    }
    
    @Override
    public boolean isPositiveDefinite() {
        return covariance.isPositiveDefinite();
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
        return 0.5 * (dimension * (1 + Math.log(2 * Math.PI)) + Math.log(Math.abs(covarianceDeterminant)));
    }
    
    @Override
    public IMatrix<Double> informationMatrix() {
        return precision.copy();
    }
    
    @Override
    public IMultivariateDistribution<Double> conjugateUpdate(IVector<Double> observations) {
        // For multivariate normal distribution, conjugate update with normal observations
        // This would typically involve updating the mean and covariance based on observations
        // This is a simplified placeholder implementation
        IVector<Double> newMean = mean.add(observations.multiplyScalar(0.1));
        return new MultivariateNormalDistribution(newMean, covariance);
    }
    
    @Override
    public double marginalLikelihood(IVector<Double> observations) {
        // For multivariate normal distribution, compute marginal likelihood of observations
        // This is a simplified placeholder implementation
        double logLikelihood = -0.5 * observations.dot(observations);
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
        
        // 计算卡方分位数
        double chiSquareQuantile = -2 * Math.log(1 - confidence);
        
        // 计算协方差矩阵的特征值和特征向量
        // 这里需要实现特征值分解，简化版本
        double a = covariance.get(0, 0);
        double b = covariance.get(0, 1);
        double c = covariance.get(1, 1);
        
        double trace = a + c;
        double det = a * c - b * b;
        double discriminant = Math.sqrt(trace * trace - 4 * det);
        
        double eigenvalue1 = 0.5 * (trace + discriminant);
        double eigenvalue2 = 0.5 * (trace - discriminant);
        
        double majorAxis = Math.sqrt(chiSquareQuantile * eigenvalue1);
        double minorAxis = Math.sqrt(chiSquareQuantile * eigenvalue2);
        
        double angle = 0.5 * Math.atan2(2 * b, a - c);
        
        return new ConfidenceEllipse(mean.copy(), majorAxis, minorAxis, angle);
    }
    
    // ==================== 静态工厂方法 ====================
    
    /**
     * 从样本数据估计多元正态分布参数
     * Estimate multivariate normal distribution parameters from sample data
     */
    public static MultivariateNormalDistribution fitFromSamples(List<IVector<Double>> samples) {
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
        
        // 计算样本均值
        double[] meanArray = new double[dimension];
        for (IVector<Double> sample : samples) {
            for (int i = 0; i < dimension; i++) {
                meanArray[i] += sample.get(i);
            }
        }
        for (int i = 0; i < dimension; i++) {
            meanArray[i] /= n;
        }
        IVector<Double> sampleMean = Linalg.vector(meanArray);
        
        // 计算样本协方差矩阵
        double[][] covArray = new double[dimension][dimension];
        for (IVector<Double> sample : samples) {
            IVector<Double> diff = sample.sub(sampleMean);
            for (int i = 0; i < dimension; i++) {
                for (int j = 0; j < dimension; j++) {
                    covArray[i][j] += diff.get(i) * diff.get(j);
                }
            }
        }
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                covArray[i][j] /= (n - 1); // 无偏估计
            }
        }
        IMatrix<Double> sampleCovariance = Linalg.matrix(covArray);
        
        return new MultivariateNormalDistribution(sampleMean, sampleCovariance);
    }
    
    /**
     * 从加权样本数据估计多元正态分布参数
     * Estimate multivariate normal distribution parameters from weighted sample data
     */
    public static MultivariateNormalDistribution fitFromWeightedSamples(List<IVector<Double>> samples, 
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
        
        // 计算加权均值
        double[] meanArray = new double[dimension];
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
        IVector<Double> weightedMean = Linalg.vector(meanArray);
        
        // 计算加权协方差矩阵
        double[][] covArray = new double[dimension][dimension];
        for (int i = 0; i < n; i++) {
            IVector<Double> sample = samples.get(i);
            double weight = weights.get(i);
            IVector<Double> diff = sample.sub(weightedMean);
            for (int j = 0; j < dimension; j++) {
                for (int k = 0; k < dimension; k++) {
                    covArray[j][k] += weight * diff.get(j) * diff.get(k);
                }
            }
        }
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                covArray[i][j] /= weightSum;
            }
        }
        IMatrix<Double> weightedCovariance = Linalg.matrix(covArray);
        
        return new MultivariateNormalDistribution(weightedMean, weightedCovariance);
    }
    
    @Override
    public String toString() {
        return String.format("MultivariateNormalDistribution(dimension=%d, mean=%s, det(cov)=%.6f)", 
                           dimension, mean.toString(), covarianceDeterminant);
    }
}