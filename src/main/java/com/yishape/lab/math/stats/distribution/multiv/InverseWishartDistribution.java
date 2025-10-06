package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

import java.util.Random;

/**
 * 逆Wishart分布
 * Inverse Wishart Distribution
 * 
 * <p>逆Wishart分布是Wishart分布的逆，常用作多元正态分布协方差矩阵的共轭先验分布。</p>
 * <p>Inverse Wishart distribution is the inverse of Wishart distribution, 
 * commonly used as conjugate prior for covariance matrix of multivariate normal distribution.</p>
 * 
 * <p>概率密度函数：f(X) = |Ψ|^(ν/2) * |X|^(-(ν+p+1)/2) * exp(-tr(ΨX⁻¹)/2) / (2^(νp/2) * Γₚ(ν/2))</p>
 * <p>其中 X 是 p×p 正定矩阵，ν 是自由度，Ψ 是尺度矩阵</p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class InverseWishartDistribution  implements IMultivariateDistribution<Double> {
    
    private final double degreesOfFreedom;
    private final IMatrix scaleMatrix;
    private final int dimension;
    private final Random random;
    private final WishartDistribution wishartDistribution;
    
    /**
     * 构造函数
     * 
     * @param degreesOfFreedom 自由度，必须 > 维度 + 1
     * @param scaleMatrix 尺度矩阵，必须是正定矩阵
     */
    public InverseWishartDistribution(double degreesOfFreedom, IMatrix scaleMatrix) {
        this(degreesOfFreedom, scaleMatrix, new Random());
    }
    
    /**
     * 构造函数
     * 
     * @param degreesOfFreedom 自由度，必须 > 维度 + 1
     * @param scaleMatrix 尺度矩阵，必须是正定矩阵
     * @param random 随机数生成器
     */
    public InverseWishartDistribution(double degreesOfFreedom, IMatrix scaleMatrix, Random random) {
        if (scaleMatrix.rows() != scaleMatrix.cols()) {
            throw new IllegalArgumentException("Scale matrix must be square");
        }
        
        this.dimension = scaleMatrix.rows();
        
        if (degreesOfFreedom <= dimension + 1) {
            throw new IllegalArgumentException("Degrees of freedom must be > dimension + 1");
        }
        
        this.degreesOfFreedom = degreesOfFreedom;
        this.scaleMatrix = scaleMatrix;
        this.random = random;
        
        // 创建对应的Wishart分布用于采样
        IMatrix scaleInverse = computeInverse(scaleMatrix);
        this.wishartDistribution = new WishartDistribution(degreesOfFreedom, scaleInverse, random);
    }
    
    /**
     * 从逆Wishart分布中采样
     * Sample from Inverse Wishart distribution
     * 
     * @return 采样得到的正定矩阵
     */
    public IMatrix sampleMatrix() {
        // 从对应的Wishart分布采样，然后取逆
        IMatrix wishartSample = wishartDistribution.sampleMatrix();
        return computeInverse(wishartSample);
    }
    
    /**
     * 计算概率密度函数值
     * Calculate probability density function value
     * 
     * @param X 正定矩阵
     * @return 概率密度值
     */
    public double pdf(IMatrix X) {
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
        
        double logDetX = logDeterminant(X);
        double logDetScale = logDeterminant(scaleMatrix);
        IMatrix XInverse = computeInverse(X);
        double trace = trace(multiply(scaleMatrix, XInverse));
        
        double logPdf = (degreesOfFreedom / 2.0) * logDetScale;
        logPdf -= ((degreesOfFreedom + dimension + 1) / 2.0) * logDetX;
        logPdf -= trace / 2.0;
        logPdf -= (degreesOfFreedom * dimension / 2.0) * Math.log(2);
        logPdf -= logMultivariateGamma(degreesOfFreedom / 2.0, dimension);
        
        return logPdf;
    }
    
    /**
     * 计算均值
     * Calculate mean
     * 
     * @return 均值矩阵
     */
    public IMatrix mean() {
        if (degreesOfFreedom <= dimension + 1) {
            throw new IllegalStateException("Mean does not exist when degrees of freedom <= dimension + 1");
        }
        
        IMatrix mean = Linalg.zeros(dimension, dimension);
        double factor = 1.0 / (degreesOfFreedom - dimension - 1);
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                mean.set(i, j, factor * scaleMatrix.get(i, j).doubleValue());
            }
        }
        
        return mean;
    }
    
    /**
     * 计算众数
     * Calculate mode
     * 
     * @return 众数矩阵
     */
    public IMatrix mode() {
        IMatrix mode = Linalg.zeros(dimension, dimension);
        double factor = 1.0 / (degreesOfFreedom + dimension + 1);
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                mode.set(i, j, factor * scaleMatrix.get(i, j).doubleValue());
            }
        }
        
        return mode;
    }
    
    /**
     * 计算方差（对于矩阵元素）
     * Calculate variance (for matrix elements)
     * 
     * @return 方差矩阵
     */
    public IMatrix variance() {
        if (degreesOfFreedom <= dimension + 3) {
            throw new IllegalStateException("Variance does not exist when degrees of freedom <= dimension + 3");
        }
        
        IMatrix variance = Linalg.zeros(dimension, dimension);
        double nu = degreesOfFreedom;
        double p = dimension;
        
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                double psi_ij = scaleMatrix.get(i, j).doubleValue();
                double psi_ii = scaleMatrix.get(i, i).doubleValue();
                double psi_jj = scaleMatrix.get(j, j).doubleValue();
                
                double numerator = (nu - p + 1) * psi_ij * psi_ij + (nu - p - 1) * psi_ii * psi_jj;
                double denominator = (nu - p) * (nu - p - 1) * (nu - p - 1) * (nu - p - 3);
                
                variance.set(i, j, numerator / denominator);
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
    
    @Override
    public String getDistributionName() {
        return "InverseWishart";
    }
    
    @Override
    public String getParameterInfo() {
        return "degreesOfFreedom=" + degreesOfFreedom + ", scaleMatrix=" + scaleMatrix.toString();
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
    public double[] pdf(java.util.List<IVector<Double>> samples) {
        double[] densities = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            densities[i] = pdf(samples.get(i));
        }
        return densities;
    }
    
    @Override
    public double[] logPdf(java.util.List<IVector<Double>> samples) {
        double[] logDensities = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            logDensities[i] = logPdf(samples.get(i));
        }
        return logDensities;
    }
    
    @Override
    public IVector<Double> getMean() {
        // Convert matrix mean to vector by taking diagonal elements
        IMatrix meanMat = mean();
        IVector<Double> meanVec = Linalg.vector(dimension);
        for (int i = 0; i < dimension; i++) {
            meanVec.set(i, (Double) meanMat.get(i, i));
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
    public IVector<Double> sample() {
        // For a matrix distribution, return a vector representation of a sample matrix
        IMatrix matrixSample = sampleMatrix();
        // Convert matrix to vector by taking diagonal elements
        IVector<Double> vectorSample = Linalg.vector(dimension);
        for (int i = 0; i < dimension; i++) {
            vectorSample.set(i, (Double) matrixSample.get(i, i));
        }
        return vectorSample;
    }
    
    @Override
    public java.util.List<IVector<Double>> sample(int n) {
        java.util.List<IVector<Double>> samples = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            samples.add(sample());
        }
        return samples;
    }
    
    @Override
    public IMatrix<Double> sampleMatrix(int n) {
        // Create a matrix where each row is a flattened matrix sample
        IMatrix<Double> result = Linalg.zeros(n, dimension * dimension);
        for (int i = 0; i < n; i++) {
            IMatrix matrixSample = sampleMatrix();
            // Flatten the matrix into a vector
            for (int row = 0; row < dimension; row++) {
                for (int col = 0; col < dimension; col++) {
                    result.set(i, row * dimension + col, (Double) matrixSample.get(row, col));
                }
            }
        }
        return result;
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
    public IMultivariateDistribution<Double> fit(java.util.List<IVector<Double>> samples) {
        // Simplified implementation
        return this;
    }
    
    @Override
    public IMultivariateDistribution<Double> fit(java.util.List<IVector<Double>> samples, java.util.List<Double> weights) {
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
        // For Inverse Wishart distribution, conjugate update with multivariate normal observations
        // In practice, this would update the scale matrix and degrees of freedom based on observations
        // This is a simplified placeholder implementation
        return new InverseWishartDistribution(degreesOfFreedom + 1, scaleMatrix.add(observationsToMatrix(observations)));
    }
    
    @Override
    public double marginalLikelihood(IVector<Double> observations) {
        // For Inverse Wishart distribution, compute marginal likelihood of observations
        // This would involve complex calculations with the Inverse Wishart distribution parameters
        // This is a simplified placeholder implementation
        return Math.exp(-0.5 * observations.sum().doubleValue());
    }
    
    @Override
    public java.util.List<IVector<Double>> posteriorSample(IVector<Double> observations, int n) {
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
     * 计算矩阵逆
     * Compute matrix inverse
     */
    private IMatrix computeInverse(IMatrix matrix) {
        // 使用Cholesky分解求逆
        IMatrix L = computeCholesky(matrix);
        return choleskyInverse(L);
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
                    double value = aii - sum;
                    if (value <= 0) {
                        throw new IllegalArgumentException("Matrix is not positive definite");
                    }
                    L.set(i, j, Math.sqrt(value));
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
        IMatrix LInvT = transpose(LInv);
        return multiply(LInvT, LInv);
    }
    
    /**
     * 矩阵转置
     * Matrix transpose
     */
    private IMatrix transpose(IMatrix matrix) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        IMatrix result = Linalg.zeros(cols, rows);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(j, i, matrix.get(i, j));
            }
        }
        
        return result;
    }
    
    /**
     * 矩阵乘法
     * Matrix multiplication
     */
    private IMatrix multiply(IMatrix A, IMatrix B) {
        int rowsA = A.rows();
        int colsA = A.cols();
        int colsB = B.cols();
        
        IMatrix result = Linalg.zeros(rowsA, colsB);
        
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                double sum = 0.0;
                for (int k = 0; k < colsA; k++) {
                    sum += A.get(i, k).doubleValue() * B.get(k, j).doubleValue();
                }
                result.set(i, j, sum);
            }
        }
        
        return result;
    }
    
    /**
     * 计算矩阵的迹
     * Calculate matrix trace
     */
    private double trace(IMatrix matrix) {
        double sum = 0.0;
        int n = Math.min(matrix.rows(), matrix.cols());
        
        for (int i = 0; i < n; i++) {
            sum += matrix.get(i, i).doubleValue();
        }
        
        return sum;
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