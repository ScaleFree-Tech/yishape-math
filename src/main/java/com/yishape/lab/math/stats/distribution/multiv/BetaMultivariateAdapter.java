package com.yishape.lab.math.stats.distribution.multiv;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.distribution.BetaDistribution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * 将一元 {@link BetaDistribution} 适配为 {@link IMultivariateDistribution}（支撑在 [0,1]，向量长度 1）。
 * Adapts univariate Beta to {@link IMultivariateDistribution} (support [0,1], vector length 1).
 *
 * @author RereMouse
 * @version 1.0
 * @since 1.0
 */
public final class BetaMultivariateAdapter implements IMultivariateDistribution<Double> {

    private static final long serialVersionUID = 1L;

    private final BetaDistribution beta;

    /**
     * 构造函数
     * Constructor
     *
     * @param beta Beta 分布实例（不能为 null）/ Beta distribution instance (must not be null)
     * @throws IllegalArgumentException 如果 beta 为 null / If beta is null
     */
    public BetaMultivariateAdapter(BetaDistribution beta) {
        if (beta == null) {
            throw new IllegalArgumentException("beta must not be null");
        }
        this.beta = beta;
    }

    /**
     * 获取内部的 Beta 分布实例
     * Get the internal Beta distribution instance
     *
     * @return Beta 分布实例 / Beta distribution instance
     */
    public BetaDistribution getBeta() {
        return beta;
    }

    @Override
    public int getDimension() {
        return 1;
    }

    @Override
    public String getDistributionName() {
        return "Beta (1D multivariate adapter)";
    }

    @Override
    public String getParameterInfo() {
        return beta.toString();
    }

    /**
     * 计算概率密度函数值
     * Compute probability density function value
     *
     * @param x 概率向量（长度为1）/ Probability vector (length 1)
     * @return 概率密度值 / Probability density value
     */
    @Override
    public double pdf(IVector<Double> x) {
        validateDimension(x);
        return beta.pdf(x.get(0));
    }

    /**
     * 计算对数概率密度函数值
     * Compute log probability density function value
     *
     * @param x 概率向量（长度为1）/ Probability vector (length 1)
     * @return 对数概率密度值 / Log probability density value
     */
    @Override
    public double logPdf(IVector<Double> x) {
        validateDimension(x);
        double p = beta.pdf(x.get(0));
        return p <= 0 ? Double.NEGATIVE_INFINITY : Math.log(p);
    }

    @Override
    public double[] pdf(List<IVector<Double>> samples) {
        double[] r = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            r[i] = pdf(samples.get(i));
        }
        return r;
    }

    @Override
    public double[] logPdf(List<IVector<Double>> samples) {
        double[] r = new double[samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            r[i] = logPdf(samples.get(i));
        }
        return r;
    }

    /**
     * 获取均值向量
     * Get mean vector
     *
     * @return 均值向量（长度为1）/ Mean vector (length 1)
     */
    @Override
    public IVector<Double> getMean() {
        return Linalg.vector(new double[]{beta.mean()});
    }

    /**
     * 获取协方差矩阵
     * Get covariance matrix
     *
     * @return 协方差矩阵（1×1）/ Covariance matrix (1×1)
     */
    @Override
    public IMatrix<Double> getCovariance() {
        double[][] c = {{beta.var()}};
        return Linalg.matrix(c);
    }

    /**
     * 获取相关矩阵
     * Get correlation matrix
     *
     * @return 相关矩阵（1×1，单位矩阵）/ Correlation matrix (1×1, identity)
     */
    @Override
    public IMatrix<Double> getCorrelation() {
        return Linalg.matrix(new double[][]{{1.0}});
    }

    /**
     * 获取精度矩阵
     * Get precision matrix
     *
     * @return 精度矩阵（协方差矩阵的逆）/ Precision matrix (inverse of covariance matrix)
     * @throws UnsupportedOperationException 如果方差为0 / If variance is zero
     */
    @Override
    public IMatrix<Double> getPrecision() {
        double v = beta.var();
        if (v <= 0) {
            throw new UnsupportedOperationException("Beta degenerate variance");
        }
        return Linalg.matrix(new double[][]{{1.0 / v}});
    }

    /**
     * 获取标准差向量
     * Get standard deviation vector
     *
     * @return 标准差向量（长度为1）/ Standard deviation vector (length 1)
     */
    @Override
    public IVector<Double> getStandardDeviation() {
        return Linalg.vector(new double[]{beta.std()});
    }

    /**
     * 计算马氏距离
     * Compute Mahalanobis distance
     *
     * @param x 概率向量 / Probability vector
     * @return 马氏距离 / Mahalanobis distance
     */
    @Override
    public double mahalanobisDistance(IVector<Double> x) {
        return Math.sqrt(squaredMahalanobisDistance(x));
    }

    /**
     * 计算平方马氏距离
     * Compute squared Mahalanobis distance
     *
     * @param x 概率向量 / Probability vector
     * @return 平方马氏距离 / Squared Mahalanobis distance
     */
    @Override
    public double squaredMahalanobisDistance(IVector<Double> x) {
        validateDimension(x);
        double d = x.get(0) - beta.mean();
        double v = beta.var();
        return d * d / v;
    }

    /**
     * 从 Beta 分布中采样一个值
     * Sample one value from Beta distribution
     *
     * @return 采样结果（长度为1的向量）/ Sampled result (vector of length 1)
     */
    @Override
    public IVector<Double> sample() {
        return Linalg.vector(new double[]{beta.sample()});
    }

    /**
     * 从 Beta 分布中采样多个值
     * Sample multiple values from Beta distribution
     *
     * @param n 采样数量 / Number of samples
     * @return 采样结果列表 / List of sampled results
     * @throws IllegalArgumentException 若 {@code n} 非正 / If {@code n} is not positive
     */
    @Override
    public List<IVector<Double>> sample(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive");
        }
        List<IVector<Double>> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(sample());
        }
        return out;
    }

    /**
     * 采样多个值并返回矩阵形式
     * Sample multiple values and return as matrix
     *
     * @param n 采样数量 / Number of samples
     * @return n×1 采样矩阵 / n×1 sample matrix
     * @throws IllegalArgumentException 若 {@code n} 非正 / If {@code n} is not positive
     */
    @Override
    public IMatrix<Double> sampleMatrix(int n) {
        List<IVector<Double>> s = sample(n);
        IMatrix<Double> m = Linalg.zeros(n, 1);
        for (int i = 0; i < n; i++) {
            m.set(i, 0, s.get(i).get(0));
        }
        return m;
    }

    @Override
    public IMultivariateDistribution<Double> getMarginal(int... indices) {
        throw new UnsupportedOperationException("一维 Beta 无更低维边际 / Scalar Beta has no proper marginal");
    }

    @Override
    public IMultivariateDistribution<Double> getConditional(int[] conditionIndices, IVector<Double> conditionValues) {
        throw new UnsupportedOperationException("条件分布不适用 / Not applicable");
    }

    @Override
    public IMultivariateDistribution<Double> linearTransform(IMatrix<Double> A, IVector<Double> b) {
        throw new UnsupportedOperationException("Beta 在线性变换下不闭族 / Beta not closed under general linear maps");
    }

    @Override
    public IMultivariateDistribution<Double> affineTransform(IMatrix<Double> A) {
        return linearTransform(A, Linalg.zeros(A.rows()));
    }

    @Override
    public double klDivergence(IMultivariateDistribution<Double> other) {
        if (!(other instanceof BetaMultivariateAdapter)) {
            throw new IllegalArgumentException("KL 仅定义在两个 Beta 适配器之间");
        }
        BetaMultivariateAdapter o = (BetaMultivariateAdapter) other;
        double a1 = beta.getAlpha();
        double b1 = beta.getBeta();
        double a2 = o.beta.getAlpha();
        double b2 = o.beta.getBeta();
        double logB2 = MultivariateDistributionMath.logGamma(a2) + MultivariateDistributionMath.logGamma(b2)
                - MultivariateDistributionMath.logGamma(a2 + b2);
        double logB1 = MultivariateDistributionMath.logGamma(a1) + MultivariateDistributionMath.logGamma(b1)
                - MultivariateDistributionMath.logGamma(a1 + b1);
        double psiSum1 = MultivariateDistributionMath.digamma(a1 + b1);
        return logB2 - logB1
                + (a1 - a2) * (MultivariateDistributionMath.digamma(a1) - psiSum1)
                + (b1 - b2) * (MultivariateDistributionMath.digamma(b1) - psiSum1);
    }

    @Override
    public double wassersteinDistance(IMultivariateDistribution<Double> other) {
        if (!(other instanceof BetaMultivariateAdapter)) {
            throw new IllegalArgumentException("比较双方须为一维 Beta 适配器");
        }
        BetaMultivariateAdapter o = (BetaMultivariateAdapter) other;
        return empiricalWasserstein2Dim1(beta, o.beta, 512, new Random(1));
    }

    private static double empiricalWasserstein2Dim1(BetaDistribution p, BetaDistribution q, int n, Random rng) {
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = p.sample();
            ys[i] = q.sample();
        }
        Arrays.sort(xs);
        Arrays.sort(ys);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            double d = xs[i] - ys[i];
            sum += d * d;
        }
        return Math.sqrt(sum / n);
    }

    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples) {
        throw new UnsupportedOperationException("请直接拟合 BetaDistribution / Fit BetaDistribution directly");
    }

    @Override
    public IMultivariateDistribution<Double> fit(List<IVector<Double>> samples, List<Double> weights) {
        throw new UnsupportedOperationException("请直接拟合 BetaDistribution / Fit BetaDistribution directly");
    }

    @Override
    public boolean isElliptical() {
        return false;
    }

    @Override
    public boolean isSymmetric() {
        return true;
    }

    @Override
    public boolean isPositiveDefinite() {
        return beta.var() > 0;
    }

    @Override
    public void validateDimension(IVector<Double> x) {
        if (x == null || x.length() != 1) {
            throw new IllegalArgumentException("Expected vector length 1");
        }
    }

    @Override
    public double entropy() {
        double a = beta.getAlpha();
        double b = beta.getBeta();
        double logB = MultivariateDistributionMath.logGamma(a) + MultivariateDistributionMath.logGamma(b)
                - MultivariateDistributionMath.logGamma(a + b);
        return logB - (a - 1) * MultivariateDistributionMath.digamma(a) - (b - 1) * MultivariateDistributionMath.digamma(b)
                + (a + b - 2) * MultivariateDistributionMath.digamma(a + b);
    }

    @Override
    public IMatrix<Double> informationMatrix() {
        return getPrecision();
    }

    @Override
    public ConfidenceEllipse getConfidenceEllipse(double confidence) {
        throw new UnsupportedOperationException("一维分布无椭圆 / No ellipse in 1D");
    }

    @Override
    public IMultivariateDistribution<Double> conjugateUpdate(IVector<Double> observations) {
        throw new UnsupportedOperationException("未实现 / Not implemented");
    }

    @Override
    public double marginalLikelihood(IVector<Double> observations) {
        throw new UnsupportedOperationException("未实现 / Not implemented");
    }

    @Override
    public List<IVector<Double>> posteriorSample(IVector<Double> observations, int n) {
        throw new UnsupportedOperationException("未实现 / Not implemented");
    }
}
