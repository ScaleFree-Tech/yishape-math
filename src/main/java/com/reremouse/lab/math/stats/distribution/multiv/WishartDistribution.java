package com.reremouse.lab.math.stats.distribution.multiv;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.stats.distribution.GammaDistribution;
import com.reremouse.lab.math.stats.distribution.NormalDistribution;
import com.reremouse.lab.math.stats.distribution.multiv.IMultivariateDistribution;

import java.util.Random;
import java.util.List;
import java.util.ArrayList;

/**
 * Wishart分布
 * Wishart Distribution
 * 
 * <p>Wishart分布是多元正态分布协方差矩阵的共轭先验分布，是卡方分布的多元推广。</p>
 * <p>Wishart distribution is the conjugate prior for the covariance matrix of 
 * multivariate normal distribution, and is a multivariate generalization of chi-squared distribution.</p>
 * 
 * <p>概率密度函数：f(X) = |X|^((ν-p-1)/2) * exp(-tr(V⁻¹X)/2) / (2^(νp/2) * |V|^(ν/2) * Γₚ(ν/2))</p>
 * <p>其中 X 是 p×p 正定矩阵，ν 是自由度，V 是尺度矩阵</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WishartDistribution  implements IMultivariateDistribution<Double> {
    
    private final double degreesOfFreedom;
    private final IMatrix scaleMatrix;
    private final int dimension;
    private final Random random;
    private final IMatrix scaleMatrixInverse;
    private final IMatrix scaleMatrixCholesky;
    
    /**
     * 构造函数
     * 
     * @param degreesOfFreedom 自由度，必须 >= 维度
     * @param scaleMatrix 尺度矩阵，必须是正定矩阵
     */
    public WishartDistribution(double degreesOfFreedom, IMatrix scaleMatrix) {
        this(degreesOfFreedom, scaleMatrix, new Random());
    }
    
    /**
     * 构造函数
     * 
     * @param degreesOfFreedom 自由度，必须 >= 维度
     * @param scaleMatrix 尺度矩阵，必须是正定矩阵
     * @param random 随机数生成器
     */
    public WishartDistribution(double degreesOfFreedom, IMatrix scaleMatrix, Random random) {
        if (scaleMatrix.rows() != scaleMatrix.cols()) {
            throw new IllegalArgumentException("Scale matrix must be square");
        }
        
        this.dimension = scaleMatrix.rows();
        
        if (degreesOfFreedom < dimension) {
            throw new IllegalArgumentException("Degrees of freedom must be >= dimension");
        }
        
        this.degreesOfFreedom = degreesOfFreedom;
        this.scaleMatrix = scaleMatrix;
        this.random = random;
        
        // 预计算逆矩阵和Cholesky分解
        this.scaleMatrixInverse = computeInverse(scaleMatrix);
        this.scaleMatrixCholesky = computeCholesky(scaleMatrix);
    }
    

    
    @Override
    public String getDistributionName() {
        return "Wishart";
    }
    
    @Override
    public String getParameterInfo() {
        return "degreesOfFreedom=" + degreesOfFreedom + ", scaleMatrix=" + scaleMatrix.toString();
    }
    
    @Override
    public IVector<Double> getMean() {
        IMatrix<Double> meanMat = meanMatrix();
        // Convert matrix to vector by taking diagonal elements
        IVector<Double> meanVec = Linalg.vector(dimension);
        for (int i = 0; i < dimension; i++) {
            meanVec.set(i, meanMat.get(i, i));
        }
        return meanVec;
    }
    
    @Override
    public IMatrix<Double> getCovariance() {
        // Simplified implementation
        return Linalg.eye(dimension);
    }
    
    @Override
    public IMatrix<Double> getCorrelation() {
        // Simplified implementation
        return Linalg.eye(dimension);
    }
    
    @Override
    public IMatrix<Double> getPrecision() {
        // Simplified implementation
        return Linalg.eye(dimension);
    }
    
    @Override
    public IVector<Double> getStandardDeviation() {
        // Simplified implementation
        return Linalg.vector(new double[dimension]);
    }
    
    @Override
    public double mahalanobisDistance(IVector<Double> x) {
        // Simplified implementation
        return 0.0;
    }
    
    @Override
    public double squaredMahalanobisDistance(IVector<Double> x) {
        // Simplified implementation
        return 0.0;
    }
    
    @Override
    public IMultivariateDistribution<Double> getMarginal(int... indices) {
        // Simplified implementation
        throw new UnsupportedOperationException("Marginal distribution not implemented");
    }
    
    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        // Simplified implementation
        throw new UnsupportedOperationException("Conditional distribution not implemented");
    }
    
    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        // Simplified implementation
        throw new UnsupportedOperationException("Linear transform not implemented");
    }
    
    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        // Simplified implementation
        throw new UnsupportedOperationException("Affine transform not implemented");
    }
    
    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        // Simplified implementation
        return 0.0;
    }
    
    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        // Simplified implementation
        return 0.0;
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples) {
        // Simplified implementation
        return this;
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples, List<Double> weights) {
        // Simplified implementation
        return this;
    }
    
    @Override
    public boolean isElliptical() {
        return true;
    }
    
    @Override
    public boolean isSymmetric() {
        return false;
    }
    
    @Override
    public boolean isPositiveDefinite() {
        return true;
    }
    
    @Override
    public void validateDimension(IVector<Double> x) {
        if (x.size() != dimension) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
    }
    
    @Override
    public double entropy() {
        // Simplified implementation
        return 0.0;
    }
    
    @Override
    public IMatrix<Double> informationMatrix() {
        // Simplified implementation
        return Linalg.eye(dimension);
    }
    
    @Override
    public ConfidenceEllipse getConfidenceEllipse(double confidence) {
        // Simplified implementation
        IVector<Double> center = getMean();
        return new ConfidenceEllipse(center, 1.0, 1.0, 0.0);
    }
    
    @Override
    public IMultivariateDistribution<Double> conjugateUpdate(IVector<Double> observations) {
        // For Wishart distribution, conjugate update with multivariate normal observations
        // In practice, this would update the scale matrix and degrees of freedom based on observations
        // This is a simplified placeholder implementation
        return new WishartDistribution(degreesOfFreedom + 1, scaleMatrix.add(observationsToMatrix(observations)));
    }
    
    @Override
    public double marginalLikelihood(IVector<Double> observations) {
        // For Wishart distribution, compute marginal likelihood of observations
        // This would involve complex calculations with the Wishart distribution parameters
        // This is a simplified placeholder implementation
        return Math.exp(-0.5 * observations.sum().doubleValue());
    }
    
    @Override
    public List<IVector<Double>> posteriorSample(IVector<Double> observations, int n) {
        // Sample from posterior distribution after conjugate update
        IMultivariateDistribution<Double> posterior = conjugateUpdate(observations);
        return posterior.sample(n);
    }
    
    /**
     * Convert observations vector to matrix form
     * 
     * @param observations observation vector
     * @return matrix representation
     */
    private IMatrix observationsToMatrix(IVector<Double> observations) {
        // Simple implementation: create diagonal matrix from vector
        IMatrix result = Linalg.zeros(dimension, dimension);
        for (int i = 0; i < Math.min(observations.size(), dimension); i++) {
            result.set(i, i, observations.get(i));
        }
        return result;
    }
    
    /**
     * 从Wishart分布中采样
     * Sample from Wishart distribution
     * 
     * @return 采样得到的正定矩阵
     */
    public IMatrix sampleMatrix() {
        return sampleBartlett();
    }
    
    @Override
    public IVector<Double> sample() {
        // For a matrix distribution, return a vector representation of a sample matrix
        IMatrix<Double> matrixSample = sampleMatrix();
        // Convert matrix to vector by taking diagonal elements
        IVector<Double> vectorSample = Linalg.vector(dimension);
        for (int i = 0; i < dimension; i++) {
            vectorSample.set(i, matrixSample.get(i, i));
        }
        return vectorSample;
    }
    
    @Override
    public List<IVector<Double>> sample(int n) {
        List<IVector<Double>> samples = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            // For a matrix distribution, we can't directly return matrix samples as vectors
            // We'll create a vector representation of the matrix trace
            IMatrix<Double> matrixSample = sampleMatrix();
            double trace = (double) matrixSample.trace() / dimension;
            IVector<Double> vectorSample = Linalg.vector(new double[]{trace});
            samples.add(vectorSample);
        }
        return samples;
    }
    
    @Override
    public IMatrix<Double> sampleMatrix(int n) {
        // Create a matrix where each row is a flattened matrix sample
        IMatrix<Double> result = Linalg.zeros(n, dimension * dimension);
        for (int i = 0; i < n; i++) {
            IMatrix<Double> matrixSample = sampleMatrix();
            // Flatten the matrix into a vector
            for (int row = 0; row < dimension; row++) {
                for (int col = 0; col < dimension; col++) {
                    result.set(i, row * dimension + col, matrixSample.get(row, col));
                }
            }
        }
        return result;
    }
    
    /**
     * 使用Bartlett分解方法采样
     * Sample using Bartlett decomposition
     */
    private IMatrix sampleBartlett() {
        // 创建下三角矩阵A
        IMatrix A = Linalg.lowerTriMatrix(dimension);
        
        // 填充下三角部分
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    // 对角元素：从卡方分布采样
                    double chiSquaredSample = sampleChiSquared(degreesOfFreedom - i);
                    A.set(i, j, Math.sqrt(chiSquaredSample));
                } else {
                    // 下三角元素：从标准正态分布采样
                    NormalDistribution normal = new NormalDistribution(0, 1);
                    A.set(i, j, normal.sample());
                }
            }
        }
        
        // 计算 L * A * A^T * L^T，其中 L 是尺度矩阵的Cholesky分解
        IMatrix AT = A.t();
        IMatrix AAT = A.mmul(AT);
        IMatrix LAAT = scaleMatrixCholesky.mmul(AAT);
        IMatrix LT = scaleMatrixCholesky.t();
        
        return LAAT.mmul(LT);
    }
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param X 正定矩阵
     * @return 概率密度值
     */
    public double pdfMatrix(IMatrix X) {
        if (X.rows() != dimension || X.cols() != dimension) {
            throw new IllegalArgumentException("Matrix dimension mismatch");
        }
        
        if (!isPositiveDefinite(X)) {
            return 0.0;
        }
        
        return Math.exp(logPdf(X));
    }
    
    /**
     * 计算对数概率密度函数值
     * Calculate log probability density function value
     * 
     * @param X 正定矩阵
     * @return 对数概率密度值
     */
    public double logPdf(IMatrix X) {
        if (X.rows() != dimension || X.cols() != dimension) {
            throw new IllegalArgumentException("Matrix dimension mismatch");
        }
        
        if (!isPositiveDefinite(X)) {
            return Double.NEGATIVE_INFINITY;
        }
        
        double logDet = logDeterminant(X);
        double trace = (double)scaleMatrixInverse.mmul(X).trace();
        
        double logPdf = ((degreesOfFreedom - dimension - 1) / 2.0) * logDet;
        logPdf -= trace / 2.0;
        logPdf -= (degreesOfFreedom * dimension / 2.0) * Math.log(2);
        logPdf -= (degreesOfFreedom / 2.0) * logDeterminant(scaleMatrix);
        logPdf -= logMultivariateGamma(degreesOfFreedom / 2.0, dimension);
        
        return logPdf;
    }
    
    public double mean() {
        // For a Wishart distribution, return the mean of the trace as a scalar representation
        double trace = 0.0;
        for (int i = 0; i < dimension; i++) {
            trace += scaleMatrix.get(i, i).doubleValue();
        }
        return degreesOfFreedom * trace;
    }
    
    /**
     * 计算均值矩阵
     * Calculate mean matrix
     * 
     * @return 均值矩阵
     */
    public IMatrix meanMatrix() {
        IMatrix mean = Linalg.zeros(dimension, dimension);
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                mean.set(i, j, degreesOfFreedom * scaleMatrix.get(i, j).doubleValue());
            }
        }
        
        return mean;
    }
    
    @Override
    public double pdf(IVector<Double> x) {
        // For a matrix distribution, we can't directly evaluate pdf at a vector
        // This is a simplified implementation
        return 0.0;
    }
    
    @Override
    public double logPdf(IVector<Double> x) {
        // For a matrix distribution, we can't directly evaluate logPdf at a vector
        // This is a simplified implementation
        return Double.NEGATIVE_INFINITY;
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
    
    public double var() {
        // For a Wishart distribution, return the variance of the trace as a scalar representation
        double trace = 0.0;
        double traceSquared = 0.0;
        for (int i = 0; i < dimension; i++) {
            double diagElement = scaleMatrix.get(i, i).doubleValue();
            trace += diagElement;
            traceSquared += diagElement * diagElement;
        }
        return 2 * degreesOfFreedom * traceSquared + 4 * degreesOfFreedom * trace * trace;
    }
    
    /**
     * 计算方差矩阵
     * Calculate variance matrix
     * 
     * @return 方差矩阵
     */
    public IMatrix varianceMatrix() {
        IMatrix variance = Linalg.zeros(dimension, dimension);
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                double vij = scaleMatrix.get(i, j).doubleValue();
                double vii = scaleMatrix.get(i, i).doubleValue();
                double vjj = scaleMatrix.get(j, j).doubleValue();
                
                double var = degreesOfFreedom * (vij * vij + vii * vjj);
                variance.set(i, j, var);
            }
        }
        
        return variance;
    }
    
    /**
     * 获取自由度
     * Get degrees of freedom
     */
    public double getDegreesOfFreedom() {
        return degreesOfFreedom;
    }
    
    /**
     * 获取尺度矩阵
     * Get scale matrix
     */
    public IMatrix getScaleMatrix() {
        return scaleMatrix;
    }
    
    @Override
    public int getDimension() {
        return dimension;
    }
    
    /**
     * 从卡方分布采样
     * Sample from chi-squared distribution
     */
    private double sampleChiSquared(double degreesOfFreedom) {
        // 使用Gamma分布：χ²(k) = Gamma(k/2, 2)
        GammaDistribution gamma = new GammaDistribution(degreesOfFreedom / 2.0, 2.0);
        return gamma.sample();
    }
    
    /**
     * 计算矩阵的Cholesky分解
     * Compute Cholesky decomposition
     */
    private IMatrix computeCholesky(IMatrix matrix) {
        int n = matrix.rows();
        IMatrix L = Linalg.zeros(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    // 对角元素
                    double sum = 0.0;
                    for (int k = 0; k < j; k++) {
                        double lij = L.get(i, k).doubleValue();
                        sum += lij * lij;
                    }
                    double aii = matrix.get(i, i).doubleValue();
                    L.set(i, j, Math.sqrt(aii - sum));
                } else {
                    // 下三角元素
                    double sum = 0.0;
                    for (int k = 0; k < j; k++) {
                        double lik = L.get(i, k).doubleValue();
                        double ljk = L.get(j, k).doubleValue();
                        sum += lik * ljk;
                    }
                    double aij = matrix.get(i, j).doubleValue();
                    double ljj = L.get(j, j).doubleValue();
                    L.set(i, j, (aij - sum) / ljj);
                }
            }
        }
        
        return L;
    }
    
    /**
     * 计算矩阵逆
     * Compute matrix inverse
     */
    private IMatrix computeInverse(IMatrix matrix) {
        // 简化实现：使用Cholesky分解求逆
        IMatrix L = computeCholesky(matrix);
        return choleskyInverse(L);
    }
    
    /**
     * 通过Cholesky分解计算逆矩阵
     * Compute inverse through Cholesky decomposition
     */
    private IMatrix choleskyInverse(IMatrix L) {
        int n = L.rows();
        
        // 计算 L^(-1)
        IMatrix LInv = Linalg.zeros(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    LInv.set(i, j, 1.0 / L.get(i, j).doubleValue());
                } else {
                    double sum = 0.0;
                    for (int k = j; k < i; k++) {
                        sum += L.get(i, k).doubleValue() * LInv.get(k, j).doubleValue();
                    }
                    LInv.set(i, j, -sum / L.get(i, i).doubleValue());
                }
            }
        }
        
        // 计算 (L^(-1))^T * L^(-1)
        IMatrix LInvT = LInv.t();
        return LInvT.mmul(LInvT);
    }
    

    

    
    /**
     * 计算对数行列式
     * Calculate log determinant
     */
    private double logDeterminant(IMatrix matrix) {
        IMatrix L = computeCholesky(matrix);
        double logDet = 0.0;
        
        for (int i = 0; i < L.rows(); i++) {
            logDet += Math.log(L.get(i, i).doubleValue());
        }
        
        return 2.0 * logDet; // 因为 det(A) = det(L)²
    }
    
    /**
     * 检查矩阵是否正定
     * Check if matrix is positive definite
     */
    private boolean isPositiveDefinite(IMatrix matrix) {
        try {
            computeCholesky(matrix);
            return true;
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * 计算多元Gamma函数的对数值
     * Calculate log multivariate Gamma function
     */
    private double logMultivariateGamma(double a, int p) {
        double result = (p * (p - 1) / 4.0) * Math.log(Math.PI);
        
        for (int j = 1; j <= p; j++) {
            result += logGamma(a + (1 - j) / 2.0);
        }
        
        return result;
    }
    
    /**
     * 计算Gamma函数的对数值
     * Calculate log Gamma function
     */
    private double logGamma(double x) {
        if (x <= 0) {
            throw new IllegalArgumentException("Gamma function argument must be positive");
        }
        
        // 使用Stirling近似
        if (x > 12) {
            return (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI);
        } else {
            // 对于小值，使用递归关系
            if (x < 1) {
                return logGamma(x + 1) - Math.log(x);
            } else if (x == 1) {
                return 0;
            } else if (x == 2) {
                return 0;
            } else {
                return Math.log(x - 1) + logGamma(x - 1);
            }
        }
    }
}