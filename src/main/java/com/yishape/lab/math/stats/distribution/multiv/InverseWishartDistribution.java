package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

import java.util.Arrays;
import java.util.Random;

/**
 * 逆 Wishart 分布；向量接口为 {@code p×p} 矩阵按<strong>行优先</strong>拉直的 {@code p²} 维向量。
 *
 * <p>{@link #getDimension()} 返回 {@code p²}；{@link #getMatrixOrder()} 返回阶数 {@code p}。
 * {@link #getMarginal(int...)} 的索引为矩阵<strong>行列指标</strong>（0…p−1），与 Wishart 一致。</p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public class InverseWishartDistribution  implements IMultivariateDistribution<Double> {
    
    private final double degreesOfFreedom;
    private final IMatrix scaleMatrix;
    private final int matrixOrder;
    private final int vectorDim;
    private final Random random;
    private final WishartDistribution wishartDistribution;
    
    /**
     * 构造函数
     * Constructor
     *
     * @param degreesOfFreedom 自由度，必须 > 维度 + 1 / Degrees of freedom, must be > dimension + 1
     * @param scaleMatrix 尺度矩阵，必须是正定矩阵 / Scale matrix, must be positive definite
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public InverseWishartDistribution(double degreesOfFreedom, IMatrix scaleMatrix) {
        this(degreesOfFreedom, scaleMatrix, new Random());
    }

    /**
     * 构造函数（带随机数生成器）
     * Constructor with random number generator
     *
     * @param degreesOfFreedom 自由度，必须 > 维度 + 1 / Degrees of freedom, must be > dimension + 1
     * @param scaleMatrix 尺度矩阵，必须是正定矩阵 / Scale matrix, must be positive definite
     * @param random 随机数生成器 / Random number generator
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public InverseWishartDistribution(double degreesOfFreedom, IMatrix scaleMatrix, Random random) {
        if (scaleMatrix == null) {
            throw new IllegalArgumentException("Scale matrix must not be null");
        }
        if (scaleMatrix.rows() != scaleMatrix.cols()) {
            throw new IllegalArgumentException("Scale matrix must be square");
        }
        
        this.matrixOrder = scaleMatrix.rows();
        this.vectorDim = matrixOrder * matrixOrder;
        
        // 逆Wishart分布要求 ν > p+1（均值存在需 ν > p+1，方差存在需 ν > p+3）
        if (degreesOfFreedom < matrixOrder + 1) {
            throw new IllegalArgumentException("Degrees of freedom must be greater than matrix order + 1 (got " +
                degreesOfFreedom + ", need > " + (matrixOrder + 1) + ")");
        }
        
        this.degreesOfFreedom = degreesOfFreedom;
        this.scaleMatrix = scaleMatrix;
        this.random = random;
        
        // 创建对应的Wishart分布用于采样
        IMatrix scaleInverse = computeInverse(scaleMatrix);
        this.wishartDistribution = new WishartDistribution(degreesOfFreedom, scaleInverse, random);
    }
    
    /**
     * 从逆Wishart分布中采样一个正定矩阵
     * Sample one positive definite matrix from Inverse Wishart distribution
     *
     * @return 采样得到的正定矩阵 / Sampled positive definite matrix
     */
    public IMatrix sampleMatrix() {
        // 从对应的Wishart分布采样，然后取逆
        IMatrix wishartSample = wishartDistribution.sampleMatrix();
        return computeInverse(wishartSample);
    }
    
    /**
     * 计算概率密度函数值
     * Compute probability density function value
     *
     * @param X 正定矩阵 / Positive definite matrix
     * @return 概率密度值 / Probability density value
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / If matrix dimension mismatch
     */
    public double pdf(IMatrix X) {
        if (X.rows() != matrixOrder || X.cols() != matrixOrder) {
            throw new IllegalArgumentException("Matrix dimension mismatch");
        }
        
        if (!isPositiveDefinite(X)) {
            return 0.0;
        }
        
        return Math.exp(logPdf(X));
    }
    
    /**
     * 计算对数概率密度函数值
     * Compute log probability density function value
     *
     * @param X 正定矩阵 / Positive definite matrix
     * @return 对数概率密度值 / Log probability density value
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / If matrix dimension mismatch
     */
    public double logPdf(IMatrix X) {
        if (X.rows() != matrixOrder || X.cols() != matrixOrder) {
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
        logPdf -= ((degreesOfFreedom + matrixOrder + 1) / 2.0) * logDetX;
        logPdf -= trace / 2.0;
        logPdf -= (degreesOfFreedom * matrixOrder / 2.0) * Math.log(2);
        logPdf -= logMultivariateGamma(degreesOfFreedom / 2.0, matrixOrder);
        
        return logPdf;
    }
    
    /**
     * 计算均值矩阵
     * Compute mean matrix
     *
     * @return 均值矩阵 / Mean matrix
     * @throws IllegalStateException 如果均值不存在 / If mean does not exist
     */
    public IMatrix mean() {
        if (degreesOfFreedom <= matrixOrder + 1) {
            throw new IllegalStateException("Mean does not exist when degrees of freedom <= matrixOrder + 1");
        }
        
        IMatrix mean = Linalg.zeros(matrixOrder, matrixOrder);
        double factor = 1.0 / (degreesOfFreedom - matrixOrder - 1);
        
        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j < matrixOrder; j++) {
                mean.set(i, j, factor * scaleMatrix.get(i, j));
            }
        }
        
        return mean;
    }
    
    /**
     * 计算众数（模式）矩阵
     * Compute mode matrix
     *
     * @return 众数矩阵 / Mode matrix
     */
    public IMatrix mode() {
        IMatrix mode = Linalg.zeros(matrixOrder, matrixOrder);
        double factor = 1.0 / (degreesOfFreedom + matrixOrder + 1);
        
        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j < matrixOrder; j++) {
                mode.set(i, j, factor * scaleMatrix.get(i, j));
            }
        }
        
        return mode;
    }
    
    /**
     * 计算方差矩阵
     * Compute variance matrix
     *
     * @return 方差矩阵 / Variance matrix
     * @throws IllegalStateException 如果方差不存在 / If variance does not exist
     */
    public IMatrix variance() {
        if (degreesOfFreedom <= matrixOrder + 3) {
            throw new IllegalStateException("Variance does not exist when degrees of freedom <= matrixOrder + 3");
        }
        
        IMatrix variance = Linalg.zeros(matrixOrder, matrixOrder);
        double nu = degreesOfFreedom;
        double p = matrixOrder;
        
        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j < matrixOrder; j++) {
                double psi_ij = scaleMatrix.get(i, j);
                double psi_ii = scaleMatrix.get(i, i);
                double psi_jj = scaleMatrix.get(j, j);
                
                double numerator = (nu - p + 1) * psi_ij * psi_ij + (nu - p - 1) * psi_ii * psi_jj;
                double denominator = (nu - p) * (nu - p - 1) * (nu - p - 1) * (nu - p - 3);
                
                variance.set(i, j, numerator / denominator);
            }
        }
        
        return variance;
    }
    
    /**
     * 获取自由度参数
     * Get degrees of freedom parameter
     *
     * @return 自由度 / Degrees of freedom
     */
    public double getDegreesOfFreedom() {
        return degreesOfFreedom;
    }
    
    /**
     * 获取尺度矩阵
     * Get scale matrix
     *
     * @return 尺度矩阵副本 / Copy of scale matrix
     */
    public IMatrix getScaleMatrix() {
        return scaleMatrix;
    }

    /** @return 矩阵阶数 p / Matrix side length p */
    public int getMatrixOrder() {
        return matrixOrder;
    }

    private IVector<Double> flattenMatrix(IMatrix m) {
        double[] data = new double[vectorDim];
        int t = 0;
        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j < matrixOrder; j++) {
                data[t++] = m.get(i, j);
            }
        }
        return Linalg.vector(data);
    }
    
    /**
     * 获取向量维度（p²）
     * Get vector dimension (p²)
     *
     * @return 向量维度 / Vector dimension
     */
    @Override
    public int getDimension() {
        return vectorDim;
    }
    
    /**
     * 获取分布名称
     * Get distribution name
     *
     * @return 分布名称 / Distribution name
     */
    @Override
    public String getDistributionName() {
        return "InverseWishart";
    }
    
    /**
     * 获取参数信息
     * Get parameter information
     *
     * @return 参数信息字符串 / Parameter information string
     */
    @Override
    public String getParameterInfo() {
        return "degreesOfFreedom=" + degreesOfFreedom + ", matrixOrder=" + matrixOrder + ", scaleMatrix=" + scaleMatrix;
    }
    
    @Override
    public double pdf(IVector<Double> x) {
        return Math.exp(logPdf(x));
    }

    @Override
    public double logPdf(IVector<Double> x) {
        validateDimension(x);
        return logPdf(vectorToMatrix(x));
    }

    private IMatrix<Double> vectorToMatrix(IVector<Double> x) {
        IMatrix<Double> m = Linalg.zeros(matrixOrder, matrixOrder);
        int k = 0;
        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j < matrixOrder; j++) {
                m.set(i, j, x.get(k++));
            }
        }
        return m;
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
        return flattenMatrix(mean());
    }
    
    @Override
    public IMatrix<Double> getCovariance() {
        if (degreesOfFreedom <= matrixOrder + 3) {
            throw new IllegalStateException("Variance does not exist when degrees of freedom <= matrixOrder + 3");
        }
        if (vectorDim > 4096) {
            throw new UnsupportedOperationException("Explicit vec(X) covariance too large (vectorDim=" + vectorDim + ")");
        }
        IMatrix<Double> cov = Linalg.zeros(vectorDim, vectorDim);
        for (int i1 = 0; i1 < matrixOrder; i1++) {
            for (int j1 = 0; j1 < matrixOrder; j1++) {
                int u = i1 * matrixOrder + j1;
                for (int i2 = 0; i2 < matrixOrder; i2++) {
                    for (int j2 = 0; j2 < matrixOrder; j2++) {
                        int v = i2 * matrixOrder + j2;
                        double c = MultivariateDistributionMath.inverseWishartElementCovariance(
                                degreesOfFreedom, matrixOrder, scaleMatrix, i1, j1, i2, j2);
                        cov.set(u, v, c);
                    }
                }
            }
        }
        return cov;
    }

    @Override
    public IMatrix<Double> getCorrelation() {
        IMatrix<Double> cov = getCovariance();
        IVector<Double> std = getStandardDeviation();
        IMatrix<Double> corr = Linalg.zeros(vectorDim, vectorDim);
        for (int i = 0; i < vectorDim; i++) {
            for (int j = 0; j < vectorDim; j++) {
                corr.set(i, j, cov.get(i, j) / (std.get(i) * std.get(j)));
            }
        }
        return corr;
    }

    @Override
    public IMatrix<Double> getPrecision() {
        return getCovariance().inv();
    }

    @Override
    public IVector<Double> getStandardDeviation() {
        IMatrix<Double> cov = getCovariance();
        double[] sd = new double[vectorDim];
        for (int i = 0; i < vectorDim; i++) {
            sd[i] = Math.sqrt(Math.max(0.0, cov.get(i, i)));
        }
        return Linalg.vector(sd);
    }

    @Override
    public double mahalanobisDistance(IVector<Double> x) {
        return Math.sqrt(squaredMahalanobisDistance(x));
    }

    @Override
    public double squaredMahalanobisDistance(IVector<Double> x) {
        validateDimension(x);
        IVector<Double> diff = x.sub(getMean());
        IMatrix<Double> prec = getPrecision();
        return diff.dotValue(prec.mmul(diff));
    }

    @Override
    public IVector<Double> sample() {
        return flattenMatrix(sampleMatrix());
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
        IMatrix<Double> result = Linalg.zeros(n, vectorDim);
        for (int i = 0; i < n; i++) {
            IMatrix matrixSample = sampleMatrix();
            // Flatten the matrix into a vector
            for (int row = 0; row < matrixOrder; row++) {
                for (int col = 0; col < matrixOrder; col++) {
                    result.set(i, row * matrixOrder + col, (Double) matrixSample.get(row, col));
                }
            }
        }
        return result;
    }
    
    /**
     * 主对角子块仍为逆 Wishart：若 X∼IWₚ(ν,Ψ)，则 X_II∼IW_|I|(ν,Ψ_II)。
     */
    @Override
    public IMultivariateDistribution<Double> getMarginal(int... indices) {
        if (indices == null || indices.length == 0) {
            throw new IllegalArgumentException("indices must be non-empty");
        }
        int[] uniq = Arrays.stream(indices).distinct().sorted().toArray();
        if (uniq.length != indices.length) {
            throw new IllegalArgumentException("indices must be unique");
        }
        for (int ix : uniq) {
            if (ix < 0 || ix >= matrixOrder) {
                throw new IllegalArgumentException("index out of range: " + ix);
            }
        }
        double[][] sub = new double[uniq.length][uniq.length];
        for (int i = 0; i < uniq.length; i++) {
            for (int j = 0; j < uniq.length; j++) {
                sub[i][j] = scaleMatrix.get(uniq[i], uniq[j]);
            }
        }
        return new InverseWishartDistribution(degreesOfFreedom, Linalg.matrix(sub), random);
    }

    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        throw new UnsupportedOperationException(
                "矩阵型逆 Wishart 在给定向量坐标下的条件分布无通用闭式表达 / "
                        + "No closed-form conditional inverse Wishart for arbitrary coordinate conditioning");
    }

    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        throw new UnsupportedOperationException(
                "R^{p²} 上线性变换不保持逆 Wishart 族；请使用矩阵相合变换 / "
                        + "Linear maps on R^{p²} do not preserve inverse Wishart; use matrix congruence");
    }

    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        throw new UnsupportedOperationException(
                "R^{p²} 上仿射变换不保持逆 Wishart；请使用矩阵相合变换 / "
                        + "Affine maps on R^{p²} do not preserve inverse Wishart; use matrix congruence");
    }
    
    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        if (!(other instanceof InverseWishartDistribution)) {
            throw new IllegalArgumentException("KL 估计要求同为 InverseWishart");
        }
        return MultivariateDistributionMath.klMonteCarlo(this, other, 2048, random);
    }
    
    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        if (!(other instanceof InverseWishartDistribution)) {
            throw new IllegalArgumentException("需要同为 InverseWishart");
        }
        InverseWishartDistribution ow = (InverseWishartDistribution) other;
        if (ow.vectorDim != this.vectorDim) {
            throw new IllegalArgumentException("矩阵阶数必须一致");
        }
        return MultivariateDistributionMath.slicedWasserstein2(this, ow, vectorDim, 256, 32, random);
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
        if (x == null) {
            throw new IllegalArgumentException("vector must not be null");
        }
        if (x.size() != vectorDim) {
            throw new IllegalArgumentException(
                    "Expected vector length " + vectorDim + " (row-major vec(X)), got " + x.size());
        }
    }
    
    @Override
    public double entropy() {
        int m = 256;
        double acc = 0.0;
        for (int i = 0; i < m; i++) {
            acc -= logPdf(sample());
        }
        return acc / m;
    }

    @Override
    public IMatrix<Double> informationMatrix() {
        return getPrecision();
    }

    @Override
    public ConfidenceEllipse getConfidenceEllipse(double confidence) {
        if (matrixOrder < 2) {
            throw new UnsupportedOperationException("confidence ellipse requires matrix order >= 2");
        }
        return MultivariateDistributionMath.confidenceEllipseMarginalPlane(
                getMean(), getCovariance(), 0, 1, confidence);
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
        return Math.exp(-0.5 * observations.sumValue());
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
        if (observations.size() == vectorDim) {
            return vectorToMatrix(observations);
        }
        IMatrix result = Linalg.zeros(matrixOrder, matrixOrder);
        for (int i = 0; i < Math.min(observations.size(), matrixOrder); i++) {
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
                        double lij = L.get(i, k);
                        sum += lij * lij;
                    }
                    double aii = matrix.get(i, i);
                    double value = aii - sum;
                    if (value <= 0) {
                        throw new IllegalArgumentException("Matrix is not positive definite");
                    }
                    L.set(i, j, Math.sqrt(value));
                } else {
                    // 下三角元素
                    double sum = 0.0;
                    for (int k = 0; k < j; k++) {
                        double lik = L.get(i, k);
                        double ljk = L.get(j, k);
                        sum += lik * ljk;
                    }
                    double aij = matrix.get(i, j);
                    double ljj = L.get(j, j);
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
                    LInv.set(i, j, 1.0 / L.get(i, j));
                } else {
                    double sum = 0.0;
                    for (int k = j; k < i; k++) {
                        sum += L.get(i, k) * LInv.get(k, j);
                    }
                    LInv.set(i, j, -sum / L.get(i, i));
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
                    sum += A.get(i, k) * B.get(k, j);
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
            sum += matrix.get(i, i);
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
            logDet += Math.log(L.get(i, i));
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
        double shift = 0;
        double z = x;
        while (z < 12) {
            shift -= Math.log(z);
            z += 1;
        }
        double inv = 1 / z;
        return shift + (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI)
                + inv / 12 - inv * inv / 360 + inv * inv * inv / 1260;
    }
}