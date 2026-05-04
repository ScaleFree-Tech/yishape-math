package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.stats.distribution.GammaDistribution;
import com.yishape.lab.math.stats.distribution.NormalDistribution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Wishart 矩阵分布，向量接口下样本为 {@code p×p} 矩阵按<strong>行优先</strong>拉直的 {@code p²}
 * 维向量；{@link #getDimension()} 返回 {@code p²}，{@link #getMatrixOrder()} 返回阶数 {@code p}。
 *
 * <p>Wishart matrix law; vector API uses row-major {@code vec(X)} ∈ ℝ^{p²}. {@link #getDimension()}
 * is {@code p²}; {@link #getMatrixOrder()} returns matrix side length {@code p}.</p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class WishartDistribution implements IMultivariateDistribution<Double> {

    private final double degreesOfFreedom;
    private final IMatrix scaleMatrix;
    /** 正定矩阵阶数 p */
    private final int matrixOrder;
    /** vec(X) 的长度 p²（行优先） */
    private final int vectorDim;
    private final Random random;
    private final IMatrix scaleMatrixInverse;
    private final IMatrix scaleMatrixCholesky;

    /**
     * 构造函数
     * Constructor
     *
     * @param degreesOfFreedom 自由度，必须 >= 矩阵阶数 / Degrees of freedom, must be >= matrix order
     * @param scaleMatrix 尺度矩阵，必须是正定矩阵 / Scale matrix, must be positive definite
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public WishartDistribution(double degreesOfFreedom, IMatrix scaleMatrix) {
        this(degreesOfFreedom, scaleMatrix, new Random());
    }

    /**
     * 构造函数（带随机数生成器）
     * Constructor with random number generator
     *
     * @param degreesOfFreedom 自由度，必须 >= 矩阵阶数 / Degrees of freedom, must be >= matrix order
     * @param scaleMatrix 尺度矩阵，必须是正定矩阵 / Scale matrix, must be positive definite
     * @param random 随机数生成器 / Random number generator
     * @throws IllegalArgumentException 如果参数无效 / If parameters are invalid
     */
    public WishartDistribution(double degreesOfFreedom, IMatrix scaleMatrix, Random random) {
        if (scaleMatrix == null) {
            throw new IllegalArgumentException("Scale matrix must not be null");
        }
        if (scaleMatrix.rows() != scaleMatrix.cols()) {
            throw new IllegalArgumentException("Scale matrix must be square");
        }

        this.matrixOrder = scaleMatrix.rows();
        this.vectorDim = matrixOrder * matrixOrder;

        if (degreesOfFreedom < matrixOrder) {
            throw new IllegalArgumentException("Degrees of freedom must be >= matrix order");
        }

        this.degreesOfFreedom = degreesOfFreedom;
        this.scaleMatrix = scaleMatrix;
        this.random = random;

        this.scaleMatrixInverse = computeInverse(scaleMatrix);
        this.scaleMatrixCholesky = computeCholesky(scaleMatrix);
    }

    /**
     * 获取矩阵阶数 p
     * Get matrix order p
     *
     * @return 矩阵阶数 / Matrix order
     */
    public int getMatrixOrder() {
        return matrixOrder;
    }

    @Override
    public String getDistributionName() {
        return "Wishart";
    }

    @Override
    public String getParameterInfo() {
        return "degreesOfFreedom=" + degreesOfFreedom + ", matrixOrder=" + matrixOrder + ", scaleMatrix=" + scaleMatrix;
    }

    @Override
    public int getDimension() {
        return vectorDim;
    }

    private IVector<Double> flattenMatrix(IMatrix m) {
        double[] data = new double[vectorDim];
        int t = 0;
        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j < matrixOrder; j++) {
                data[t++] = m.get(i, j).doubleValue();
            }
        }
        return Linalg.vector(data);
    }

    @Override
    public IVector<Double> getMean() {
        return flattenMatrix(meanMatrix());
    }

    /**
     * Cov(vec(X))_{ab,cd} 对应元素顺序：{@code a=i*p+j}, {@code c=k*p+l}，
     * Cov(W_ij,W_kl)=ν(V_ik V_jl + V_il V_jk)。
     */
    @Override
    public IMatrix<Double> getCovariance() {
        if (vectorDim > 4096) {
            throw new UnsupportedOperationException("Explicit vec(W) covariance too large (vectorDim=" + vectorDim + ")");
        }
        IMatrix<Double> cov = Linalg.zeros(vectorDim, vectorDim);
        for (int i1 = 0; i1 < matrixOrder; i1++) {
            for (int j1 = 0; j1 < matrixOrder; j1++) {
                int u = i1 * matrixOrder + j1;
                for (int i2 = 0; i2 < matrixOrder; i2++) {
                    for (int j2 = 0; j2 < matrixOrder; j2++) {
                        int v = i2 * matrixOrder + j2;
                        double c = MultivariateDistributionMath.wishartElementCovariance(
                                degreesOfFreedom, scaleMatrix, i1, j1, i2, j2);
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
        return diff.dot(prec.mmul(diff));
    }

    /**
     * 主对角指标集上的 Wishart 子块：W_II ∼ W_{|I|}(ν, V_II)。
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
                sub[i][j] = scaleMatrix.get(uniq[i], uniq[j]).doubleValue();
            }
        }
        return new WishartDistribution(degreesOfFreedom, Linalg.matrix(sub), random);
    }

    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        throw new UnsupportedOperationException(
                "矩阵型 Wishart 在给定向量坐标下的条件分布无通用闭式表达 / "
                        + "No closed-form conditional Wishart for arbitrary coordinate conditioning");
    }

    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        throw new UnsupportedOperationException(
                "R^{p²} 上线性变换不保持 Wishart；请使用矩阵相合变换 X↦LXLᵀ / "
                        + "Linear maps on R^{p²} do not preserve Wishart; use congruence X ↦ L X Lᵀ");
    }

    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        return linearTransform(A, Linalg.zeros(A.rows()));
    }

    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        if (!(other instanceof WishartDistribution)) {
            throw new IllegalArgumentException("KL 估计要求同为 Wishart");
        }
        return MultivariateDistributionMath.klMonteCarlo(this, other, 2048, random);
    }

    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        if (!(other instanceof WishartDistribution)) {
            throw new IllegalArgumentException("需要同为 Wishart");
        }
        WishartDistribution ow = (WishartDistribution) other;
        if (ow.vectorDim != this.vectorDim) {
            throw new IllegalArgumentException("矩阵阶数必须一致");
        }
        return MultivariateDistributionMath.slicedWasserstein2(this, ow, vectorDim, 256, 32, random);
    }

    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples) {
        return this;
    }

    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples, List<Double> weights) {
        return this;
    }

    @Override
    public boolean isElliptical() {
        return false;
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
            throw new IllegalArgumentException("Expected vector length " + vectorDim + " (row-major vec(X)), got " + x.size());
        }
    }

    @Override
    public double entropy() {
        int m = 128;
        double acc = 0.0;
        for (int i = 0; i < m; i++) {
            acc -= logPdf(sampleMatrix());
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
            throw new UnsupportedOperationException("need matrix order >= 2 for marginal (W₁₁,W₁₂) ellipse");
        }
        return MultivariateDistributionMath.confidenceEllipseMarginalPlane(
                getMean(), getCovariance(), 0, 1, confidence);
    }

    @Override
    public IMultivariateDistribution<Double> conjugateUpdate(IVector<Double> observations) {
        return new WishartDistribution(degreesOfFreedom + 1, scaleMatrix.add(observationsToMatrix(observations)));
    }

    @Override
    public double marginalLikelihood(IVector<Double> observations) {
        return Math.exp(-0.5 * observations.sum().doubleValue());
    }

    @Override
    public List<IVector<Double>> posteriorSample(IVector<Double> observations, int n) {
        IMultivariateDistribution<Double> posterior = conjugateUpdate(observations);
        return posterior.sample(n);
    }

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
     * 从Wishart分布中采样一个正定矩阵
     * Sample one positive definite matrix from Wishart distribution
     *
     * @return 采样得到的正定矩阵 / Sampled positive definite matrix
     */
    public IMatrix sampleMatrix() {
        return sampleBartlett();
    }

    @Override
    public IVector<Double> sample() {
        return flattenMatrix(sampleMatrix());
    }

    @Override
    public List<IVector<Double>> sample(int n) {
        List<IVector<Double>> samples = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            samples.add(sample());
        }
        return samples;
    }

    /**
     * 采样多个正定矩阵并返回矩阵形式
     * Sample multiple positive definite matrices and return as matrix
     *
     * @param n 采样数量 / Number of samples
     * @return n×p² 采样矩阵 / n×p² sample matrix
     */
    @Override
    public IMatrix<Double> sampleMatrix(int n) {
        IMatrix<Double> result = Linalg.zeros(n, vectorDim);
        for (int i = 0; i < n; i++) {
            IMatrix<Double> matrixSample = sampleMatrix();
            int k = 0;
            for (int row = 0; row < matrixOrder; row++) {
                for (int col = 0; col < matrixOrder; col++) {
                    result.set(i, k++, matrixSample.get(row, col));
                }
            }
        }
        return result;
    }

    private IMatrix sampleBartlett() {
        IMatrix A = Linalg.lowerTriMatrix(matrixOrder);

        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    double chiSquaredSample = sampleChiSquared(degreesOfFreedom - i);
                    A.set(i, j, Math.sqrt(chiSquaredSample));
                } else {
                    NormalDistribution normal = new NormalDistribution(0, 1);
                    A.set(i, j, normal.sample());
                }
            }
        }

        IMatrix AT = A.t();
        IMatrix AAT = A.mmul(AT);
        IMatrix LAAT = scaleMatrixCholesky.mmul(AAT);
        IMatrix LT = scaleMatrixCholesky.t();

        return LAAT.mmul(LT);
    }

    /**
     * 计算矩阵形式的概率密度函数值
     * Compute probability density function value for matrix form
     *
     * @param X 正定矩阵 / Positive definite matrix
     * @return 概率密度值 / Probability density value
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / If matrix dimension mismatch
     */
    public double pdfMatrix(IMatrix X) {
        if (X.rows() != matrixOrder || X.cols() != matrixOrder) {
            throw new IllegalArgumentException("Matrix dimension mismatch");
        }

        if (!isPositiveDefinite(X)) {
            return 0.0;
        }

        return Math.exp(logPdf(X));
    }

    /**
     * 计算矩阵形式的对数概率密度函数值
     * Compute log probability density function value for matrix form
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

        double logDet = logDeterminant(X);
        double trace = (double) scaleMatrixInverse.mmul(X).trace();

        double logPdf = ((degreesOfFreedom - matrixOrder - 1) / 2.0) * logDet;
        logPdf -= trace / 2.0;
        logPdf -= (degreesOfFreedom * matrixOrder / 2.0) * Math.log(2);
        logPdf -= (degreesOfFreedom / 2.0) * logDeterminant(scaleMatrix);
        logPdf -= logMultivariateGamma(degreesOfFreedom / 2.0, matrixOrder);

        return logPdf;
    }

    /**
     * 计算标量均值（矩阵迹的倍数）
     * Compute scalar mean (multiple of matrix trace)
     *
     * @return 标量均值 / Scalar mean
     */
    public double mean() {
        double trace = 0.0;
        for (int i = 0; i < matrixOrder; i++) {
            trace += scaleMatrix.get(i, i).doubleValue();
        }
        return degreesOfFreedom * trace;
    }

    /**
     * 计算均值矩阵
     * Compute mean matrix
     *
     * @return 均值矩阵 / Mean matrix
     */
    public IMatrix meanMatrix() {
        return meanMatrixTyped();
    }

    private IMatrix<Double> meanMatrixTyped() {
        IMatrix<Double> mean = Linalg.zeros(matrixOrder, matrixOrder);
        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j < matrixOrder; j++) {
                mean.set(i, j, degreesOfFreedom * scaleMatrix.get(i, j).doubleValue());
            }
        }
        return mean;
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

    /**
     * 计算标量方差（基于迹）
     * Compute scalar variance (based on trace)
     *
     * @return 标量方差 / Scalar variance
     */
    public double var() {
        double trace = 0.0;
        double traceSquared = 0.0;
        for (int i = 0; i < matrixOrder; i++) {
            double diagElement = scaleMatrix.get(i, i).doubleValue();
            trace += diagElement;
            traceSquared += diagElement * diagElement;
        }
        return 2 * degreesOfFreedom * traceSquared + 4 * degreesOfFreedom * trace * trace;
    }

    /**
     * 计算方差矩阵
     * Compute variance matrix
     *
     * @return 方差矩阵 / Variance matrix
     */
    public IMatrix varianceMatrix() {
        IMatrix variance = Linalg.zeros(matrixOrder, matrixOrder);

        for (int i = 0; i < matrixOrder; i++) {
            for (int j = 0; j < matrixOrder; j++) {
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
     * @return 尺度矩阵 / Scale matrix
     */
    public IMatrix getScaleMatrix() {
        return scaleMatrix;
    }

    private double sampleChiSquared(double dof) {
        GammaDistribution gamma = new GammaDistribution(dof / 2.0, 2.0);
        return gamma.sample();
    }

    private IMatrix computeCholesky(IMatrix matrix) {
        int n = matrix.rows();
        IMatrix L = Linalg.zeros(n, n);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    double sum = 0.0;
                    for (int k = 0; k < j; k++) {
                        double lij = L.get(i, k).doubleValue();
                        sum += lij * lij;
                    }
                    double aii = matrix.get(i, i).doubleValue();
                    L.set(i, j, Math.sqrt(aii - sum));
                } else {
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

    private IMatrix computeInverse(IMatrix matrix) {
        IMatrix L = computeCholesky(matrix);
        return choleskyInverse(L);
    }

    private IMatrix choleskyInverse(IMatrix L) {
        int n = L.rows();

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

        IMatrix LInvT = LInv.t();
        return LInvT.mmul(LInv);
    }

    private double logDeterminant(IMatrix matrix) {
        IMatrix L = computeCholesky(matrix);
        double logDet = 0.0;

        for (int i = 0; i < L.rows(); i++) {
            logDet += Math.log(L.get(i, i).doubleValue());
        }

        return 2.0 * logDet;
    }

    private boolean isPositiveDefinite(IMatrix matrix) {
        try {
            computeCholesky(matrix);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private double logMultivariateGamma(double a, int p) {
        double result = (p * (p - 1) / 4.0) * Math.log(Math.PI);

        for (int j = 1; j <= p; j++) {
            result += logGamma(a + (1 - j) / 2.0);
        }

        return result;
    }

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
