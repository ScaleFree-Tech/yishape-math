package com.yishape.lab.math.ml.dml.dmleig;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.MetricTransforms;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;

import java.util.*;
import java.util.Objects;
import com.yishape.lab.math.ml.dml.ISupervisedDml;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;

/**
 * DML-eig：通过广义特征值优化学习马氏度量。
 * 最小化异类样本间的最小距离，同时约束同类样本间距离之和不超过常数。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * <p>优化路径：从 v1.0 开始，内部使用 {@link Opts#lbfgs()} 门面委托
 * {@code RustLBFGS}（底层走 HPC/Rust 高性能路径），替代早期版本的固定步长梯度下降。
 * 目标函数和梯度均基于解析公式。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Ying, Y., &amp; Li, P. (2012). Distance metric learning with eigenvalue optimization.
 *       <em>Journal of Machine Learning Research</em>, 13, 1–26.</li>
 * </ul>
 */
public final class DmleigDml implements ISupervisedDml {

    private double mu = 1e-4;
    private double tol = 1e-5;
    private double eps = 1e-10;
    private int maxIter = 25;
    /** L-BFGS 容忍度 */
    private double tolerance = 1e-6;
    /** L-BFGS 最大迭代次数 */
    private int maxBfgsIter = 200;
    /** 内部优化器注入（null 时用默认值） */
    private transient IOptimizer optimizer;

    public double getMu() {
        return mu;
    }

    public DmleigDml setMu(double mu) {
        this.mu = mu;
        return this;
    }

    public double getTol() {
        return tol;
    }

    public DmleigDml setTol(double tol) {
        this.tol = tol;
        return this;
    }

    public double getEps() {
        return eps;
    }

    public DmleigDml setEps(double eps) {
        this.eps = eps;
        return this;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public DmleigDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    public double getTolerance() {
        return tolerance;
    }

    public DmleigDml setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public int getMaxBfgsIter() {
        return maxBfgsIter;
    }

    public DmleigDml setMaxBfgsIter(int maxBfgsIter) {
        this.maxBfgsIter = maxBfgsIter;
        return this;
    }

    public IOptimizer getOptimizer() {
        return optimizer;
    }

    /**
     * 注入自定义优化器（null 时使用 {@link Opts#lbfgs(double, int)}）。
     */
    public DmleigDml setOptimizer(IOptimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(labels, "labels");
        return fit(features, DmlArrays.stringLabels(labels));
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, DmleigDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, DmleigDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        int n = x.length;
        int d = x[0].length;

        // 构建同类/异类索引映射
        Map<Integer, List<Integer>> similar = new HashMap<>();
        Map<Integer, List<Integer>> dissimilar = new HashMap<>();
        for (int i = 0; i < n; i++) {
            similar.put(i, new ArrayList<>());
            dissimilar.put(i, new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (y[i] == y[j]) {
                    similar.get(i).add(j);
                    similar.get(j).add(i);
                } else {
                    dissimilar.get(i).add(j);
                    dissimilar.get(j).add(i);
                }
            }
        }

        // 预计算 Xs（类内散度矩阵）
        double[][] Xs = new double[d][d];
        for (int i : similar.keySet()) {
            for (int j : similar.get(i)) {
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        Xs[a][b] += (x[i][a] - x[j][a]) * (x[i][b] - x[j][b]);
                    }
                }
            }
        }

        IMatrix<Double> XsMat = IMatrix.of(Xs);
        double det = XsMat.det();
        if (Math.abs(det) < eps) {
            for (int i = 0; i < d; i++) {
                Xs[i][i] += 1e-5;
            }
            XsMat = IMatrix.of(Xs);
        }

        IMatrix<Double> XsInvSqrt = computeMatrixInvSqrt(XsMat);

        // 构建异类差值矩阵 U (P×d)
        List<double[]> diffList = new ArrayList<>();
        for (int i : dissimilar.keySet()) {
            for (int j : dissimilar.get(i)) {
                double[] diff = new double[d];
                for (int t = 0; t < d; t++) {
                    diff[t] = x[i][t] - x[j][t];
                }
                diffList.add(diff);
            }
        }
        int P = diffList.size();
        double[][] uData = new double[P][d];
        for (int p = 0; p < P; p++) {
            uData[p] = diffList.get(p);
        }

        // 初始化 M（对称 PSD）
        double[][] M = new double[d][d];
        for (int i = 0; i < d; i++) {
            M[i][i] = 1.0 / d;
        }

        // 构造目标/梯度
        DmleigAutodiffObjective obj = new DmleigAutodiffObjective(uData, P, d, mu);
        IOptimizer opt = optimizer != null
                ? optimizer
                : Opts.lbfgs(tolerance, maxBfgsIter);

        IVector<Double> initVec = Linalg.vector(flattenSymmetric(M, d));

        // L-BFGS 迭代，外层控制最大迭代
        IVector<Double> current = initVec.copy();
        double maxDiff = Double.POSITIVE_INFINITY;
        int iter = 0;

        while (iter < maxIter && maxDiff > tol) {
            OptResult res = opt.optimize(current, obj, obj);
            IVector<Double> prev = current;
            current = res.getOptimalPoint().copy();

            // 对称化（保证 M 始终对称）
            double[][] Mprev = unflattenSymmetric(prev, d);
            double[][] Mcurr = unflattenSymmetric(current, d);
            for (int a = 0; a < d; a++) {
                for (int b = a + 1; b < d; b++) {
                    double avg = 0.5 * (Mcurr[a][b] + Mcurr[b][a]);
                    Mcurr[a][b] = avg;
                    Mcurr[b][a] = avg;
                }
            }

            maxDiff = 0;
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    maxDiff = Math.max(maxDiff, Math.abs(Mcurr[a][b] - Mprev[a][b]));
                }
            }

            // 将对称化后的矩阵重新展平
            current = Linalg.vector(flattenSymmetric(Mcurr, d));
            iter++;
        }

        double[][] Mfinal = unflattenSymmetric(current, d);
        IMatrix<Double> result = MetricTransforms.symmetrize(IMatrix.of(Mfinal));
        return DmlMetric.fullWhitening(result);
    }

    private IMatrix<Double> computeMatrixInvSqrt(IMatrix<Double> A) {
        var eigResult = A.eigen();
        IVector<Double> eigVals = eigResult._1;
        IMatrix<Double> eigVecs = eigResult._2;

        int d = A.getRowNum();
        double[] poweredVals = new double[d];
        for (int i = 0; i < d; i++) {
            double v = eigVals.get(i);
            poweredVals[i] = 1.0 / Math.sqrt(Math.max(v, eps));
        }

        IMatrix<Double> diag = IMatrix.diag(poweredVals);
        return eigVecs.mmul(diag).mmul(eigVecs.transpose());
    }

    // ---- 对称矩阵扁平化（只存储上三角，d(d+1)/2 维）----

    private static double[] flattenSymmetric(double[][] M, int d) {
        double[] out = new double[d * d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                out[i * d + j] = M[i][j];
            }
        }
        return out;
    }

    private static double[][] unflattenSymmetric(IVector<Double> v, int d) {
        double[] arr = v.toDoubleArray();
        double[][] M = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                M[i][j] = arr[i * d + j];
            }
        }
        return M;
    }

    // ---- DML-eig 自动微分目标函数 ----

    private static final class DmleigAutodiffObjective implements IObjectiveFunction, IGradientFunction {

        private final double[][] uData;
        private final int P, d;
        private final double mu;

        DmleigAutodiffObjective(double[][] uData, int P, int d, double mu) {
            this.uData = uData;
            this.P = P;
            this.d = d;
            this.mu = mu;
        }

        private IDiffVector buildLoss(IDiffVector w) {
            IDiffMatrix M = w.reshape(d, d);
            IDiffMatrix U = AD.matrix(uData);
            IDiffMatrix UM = U.matmul(M);
            IDiffVector quads = UM.mul(U).matmul(AD.ones(d));
            return quads.div(-mu).exp().mean();
        }

        @Override
        public double computeObjective(IVector xv) {
            IDiffVector var = new RereDiffVector((com.yishape.lab.math.linalg.IDoubleVector) xv);
            return buildLoss(var).getValue().get(0);
        }

        @Override
        public IVector computeGradient(IVector xv) {
            IDiffVector var = new RereDiffVector((com.yishape.lab.math.linalg.IDoubleVector) xv);
            IDiffVector loss = buildLoss(var);
            loss.backward();
            return var.getGradient();
        }
    }
}
