package com.yishape.lab.math.ml.dml.itml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.MetricTransforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Information-Theoretic Metric Learning（ITML）：Bregman–LogDet 迭代，维护对称 PSD 的 {@code A}，
 * 对随机约束对施加同类 / 异类距离边界（松弛 {@code γ}）。
 *
 * <p>先验为 {@link PriorKind#INVERSE_COVARIANCE} 时，样本协方差对角上的 L2 扰动强度为
 * {@link #getPriorL2Weight() priorL2Weight}。</p>
 *
 * <p>本类实现 {@link ISupervisedDml}；超参数由链式 setter 配置。</p>
 *
 * @apiNote 应用代码优先使用 {@link com.yishape.lab.math.ml.ML#dml} 上与本算法对应的重载；
 * 仅当需要尚未在 {@link com.yishape.lab.math.ml.DmlWrapper} 中暴露的调参项时，再使用本类构造 + {@code setXxx}。
 * {@code static fit(...)} 为脚本式一行调用保留，非 API 首选路径。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Davis, J. V., Kulis, B., Jain, P., Sra, S., &amp; Dhillon, I. S. (2007). Information-theoretic
 *       metric learning. In <em>Proceedings of the 24th International Conference on Machine Learning (ICML)</em>,
 *       pp. 209–216.</li>
 * </ul>
 */
public final class ItmlDml implements ISupervisedDml {

    public enum PriorKind {
        /** A₀ = I */
        IDENTITY,
        /** A₀ ≈ (Cov + λI)⁻¹，Cov 为列均值中心化样本协方差；λ 为 {@link ItmlDml#getPriorL2Weight()} */
        INVERSE_COVARIANCE
    }

    private double gamma = 1.0;
    private int maxIter = 200;
    private double tol = 1e-3;
    /** 与 metric-learn 类似：约 20×C²； capped 以便小数据可控 */
    private Integer nConstraintPairs;
    private PriorKind priorKind = PriorKind.IDENTITY;
    /**
     * 当 {@link #priorKind} 为 {@link PriorKind#INVERSE_COVARIANCE} 时，加在样本协方差对角上的 L2 岭（须非负），
     * 与 Fisher/RCA 等处的 {@code l2Weight} 同属 Ridge 口径。
     */
    private double priorL2Weight = 1e-6;
    /** null：自动用样本对上欧氏平方距离的 5% / 95% 分位 */
    private double[] bounds;
    private Random random;

    public double getGamma() {
        return gamma;
    }

    public ItmlDml setGamma(double gamma) {
        this.gamma = gamma;
        return this;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public ItmlDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    public double getTol() {
        return tol;
    }

    public ItmlDml setTol(double tol) {
        this.tol = tol;
        return this;
    }

    public Integer getNConstraintPairs() {
        return nConstraintPairs;
    }

    /** 每种符号（同类/异类）约束条数上界的一半场景下：总约束约 2×该值 */
    public ItmlDml setNConstraintPairs(Integer nConstraintPairs) {
        this.nConstraintPairs = nConstraintPairs;
        return this;
    }

    public PriorKind getPriorKind() {
        return priorKind;
    }

    public ItmlDml setPriorKind(PriorKind priorKind) {
        this.priorKind = priorKind != null ? priorKind : PriorKind.IDENTITY;
        return this;
    }

    public double getPriorL2Weight() {
        return priorL2Weight;
    }

    public ItmlDml setPriorL2Weight(double priorL2Weight) {
        this.priorL2Weight = priorL2Weight;
        return this;
    }

    public double[] getBounds() {
        return bounds;
    }

    /** lengths=2：{同类上界平方目标, 异类下界平方目标}，与 vᵀAv 同量纲（平方欧氏当 A=I） */
    public ItmlDml setBounds(double[] bounds) {
        this.bounds = bounds;
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public ItmlDml setRandom(Random random) {
        this.random = random;
        return this;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        Objects.requireNonNull(labels, "labels");
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(labels, "labels");
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, ItmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, ItmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;
        if (n < 2 || d < 1) {
            throw new IllegalArgumentException("样本过少或维数为 0");
        }

        Random rnd = random != null ? random : ThreadLocalRandom.current();
        int numClasses = Arrays.stream(y).max().orElse(0) + 1;
        int nTarget = nConstraintPairs != null ? nConstraintPairs : Math.min(200, n * (n - 1) / 2);
        nTarget = Math.max(4, Math.min(nTarget, 20 * numClasses * numClasses));

        double[] bnds = bounds;
        if (bnds == null || bnds.length != 2) {
            bnds = percentileSquaredPairDistances(x, n, d, 0.05, 0.95);
        }
        if (bnds[0] <= 0) {
            bnds[0] = 1e-9;
        }
        if (bnds[1] <= 0) {
            bnds[1] = 1e-9;
        }

        List<double[]> posDiffs = sampleSameClassDiffs(x, y, n, d, nTarget / 2, rnd);
        List<double[]> negDiffs = sampleDiffClassDiffs(x, y, n, d, nTarget / 2, rnd);
        if (posDiffs.isEmpty() || negDiffs.isEmpty()) {
            throw new IllegalStateException("无法采样同类/异类约束对；请检查标签与类别数。");
        }

        double[][] A = initializePriorA(x, n, d);

        double gammaProj = Double.isInfinite(gamma) ? 1.0 : gamma / (gamma + 1.0);

        int np = posDiffs.size();
        int nn = negDiffs.size();
        double[] lambda = new double[np + nn];
        double[] lambdaOld = new double[np + nn];
        double[] posBhat = new double[np];
        Arrays.fill(posBhat, bnds[0]);
        double[] negBhat = new double[nn];
        Arrays.fill(negBhat, bnds[1]);

        double conv = Double.POSITIVE_INFINITY;
        int it;
        for (it = 0; it < maxIter; it++) {
            for (int i = 0; i < np; i++) {
                double[] v = posDiffs.get(i);
                double wtw = quadraticForm(A, v, d);
                if (wtw <= 1e-20) {
                    wtw = 1e-20;
                }
                double alpha = Math.min(lambda[i], gammaProj * (1.0 / wtw - 1.0 / posBhat[i]));
                if (alpha < 0) {
                    alpha = 0;
                }
                lambda[i] -= alpha;
                double beta = alpha / (1.0 - alpha * wtw);
                posBhat[i] = 1.0 / ((1.0 / posBhat[i]) + (alpha / gamma));
                double[] av = matVec(A, v, d);
                rank1Add(A, av, beta, d);
            }
            for (int i = 0; i < nn; i++) {
                double[] v = negDiffs.get(i);
                double wtw = quadraticForm(A, v, d);
                if (wtw <= 1e-20) {
                    wtw = 1e-20;
                }
                double alpha = Math.min(lambda[np + i], gammaProj * (1.0 / negBhat[i] - 1.0 / wtw));
                if (alpha < 0) {
                    alpha = 0;
                }
                lambda[np + i] -= alpha;
                double beta = -alpha / (1.0 + alpha * wtw);
                negBhat[i] = 1.0 / ((1.0 / negBhat[i]) - (alpha / gamma));
                double[] av = matVec(A, v, d);
                rank1Add(A, av, beta, d);
            }

            symmetrizeInPlace(A, d);
            double normSum = l1(lambda) + l1(lambdaOld);
            if (normSum == 0) {
                conv = Double.POSITIVE_INFINITY;
                break;
            }
            conv = l1Diff(lambda, lambdaOld) / normSum;
            if (conv < tol) {
                break;
            }
            System.arraycopy(lambda, 0, lambdaOld, 0, lambda.length);
        }

        IMatrix<Double> am = MetricTransforms.symmetrize(IMatrix.of(A));
        IMatrix<Double> w = MetricTransforms.whitenerFromPrecision(am);
        return DmlMetric.fullWhitening(w);
    }

    private double[][] initializePriorA(double[][] x, int n, int d) {
        double[][] A = new double[d][d];
        if (priorKind == PriorKind.INVERSE_COVARIANCE) {
            double[] mean = new double[d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    mean[j] += x[i][j];
                }
            }
            for (int j = 0; j < d; j++) {
                mean[j] /= n;
            }
            double[][] C = new double[d][d];
            for (int i = 0; i < n; i++) {
                for (int a = 0; a < d; a++) {
                    double da = x[i][a] - mean[a];
                    for (int b = 0; b < d; b++) {
                        C[a][b] += da * (x[i][b] - mean[b]);
                    }
                }
            }
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    C[a][b] /= n;
                }
            }
            for (int i = 0; i < d; i++) {
                C[i][i] += priorL2Weight;
            }
            IMatrix<Double> inv = IMatrix.of(C).inv();
            inv = MetricTransforms.symmetrize(inv);
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    A[i][j] = inv.get(i, j);
                }
            }
        } else {
            for (int i = 0; i < d; i++) {
                A[i][i] = 1.0;
            }
        }
        return A;
    }

    private static double l1(double[] a) {
        double s = 0;
        for (double v : a) {
            s += Math.abs(v);
        }
        return s;
    }

    private static double l1Diff(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s;
    }

    private static double[] percentileSquaredPairDistances(double[][] x, int n, int d, double pLo, double pHi) {
        List<Double> ds = new ArrayList<>(n * n / 2);
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = 0;
                for (int t = 0; t < d; t++) {
                    double u = x[i][t] - x[j][t];
                    s += u * u;
                }
                ds.add(s);
            }
        }
        Collections.sort(ds);
        int il = (int) Math.floor(pLo * (ds.size() - 1));
        int ih = (int) Math.ceil(pHi * (ds.size() - 1));
        il = Math.max(0, Math.min(il, ds.size() - 1));
        ih = Math.max(0, Math.min(ih, ds.size() - 1));
        return new double[] { ds.get(il), ds.get(ih) };
    }

    private static List<double[]> sampleSameClassDiffs(double[][] x, int[] y, int n, int d, int m, Random rnd) {
        ArrayList<double[]> out = new ArrayList<>();
        for (int t = 0; t < m * 8 && out.size() < m; t++) {
            int i = rnd.nextInt(n);
            int j = rnd.nextInt(n);
            if (i == j || y[i] != y[j]) {
                continue;
            }
            double[] v = new double[d];
            for (int k = 0; k < d; k++) {
                v[k] = x[i][k] - x[j][k];
            }
            out.add(v);
        }
        return out;
    }

    private static List<double[]> sampleDiffClassDiffs(double[][] x, int[] y, int n, int d, int m, Random rnd) {
        ArrayList<double[]> out = new ArrayList<>();
        for (int t = 0; t < m * 8 && out.size() < m; t++) {
            int i = rnd.nextInt(n);
            int j = rnd.nextInt(n);
            if (y[i] == y[j]) {
                continue;
            }
            double[] v = new double[d];
            for (int k = 0; k < d; k++) {
                v[k] = x[i][k] - x[j][k];
            }
            out.add(v);
        }
        return out;
    }

    private static double quadraticForm(double[][] A, double[] v, int d) {
        double s = 0;
        for (int i = 0; i < d; i++) {
            double ai = 0;
            for (int j = 0; j < d; j++) {
                ai += A[i][j] * v[j];
            }
            s += v[i] * ai;
        }
        return s;
    }

    private static double[] matVec(double[][] A, double[] v, int d) {
        double[] o = new double[d];
        for (int i = 0; i < d; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                s += A[i][j] * v[j];
            }
            o[i] = s;
        }
        return o;
    }

    private static void rank1Add(double[][] A, double[] av, double beta, int d) {
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                A[i][j] += beta * av[i] * av[j];
            }
        }
    }

    private static void symmetrizeInPlace(double[][] A, int d) {
        for (int i = 0; i < d; i++) {
            for (int j = i + 1; j < d; j++) {
                double m = 0.5 * (A[i][j] + A[j][i]);
                A[i][j] = m;
                A[j][i] = m;
            }
        }
    }
}
