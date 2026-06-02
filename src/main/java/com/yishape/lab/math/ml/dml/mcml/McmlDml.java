package com.yishape.lab.math.ml.dml.mcml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.MetricEmbeddingOps;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.util.Tuple2;

import java.util.Arrays;
import java.util.Objects;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Maximally Collapsing Metric Learning（MCML）：与 NCA 共享 softmax 邻域结构，
 * 但将损失取为当前邻域分布与「质量在同类上均匀」的理想分布之间的 KL。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * <p>优化路径：从 v1.0 开始，内部使用 {@link Opts#lbfgs()} 门面委托
 * {@code RustLBFGS}（底层走 HPC/Rust 高性能路径），替代早期版本的固定学习率 SGD。
 * PSD 投影（{@code sdProject}）在 L-BFGS 每步后作为后处理步骤保留。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}；需未在
 * {@link com.yishape.lab.math.ml.DmlWrapper} 暴露的调参时再直接使用本类。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Globerson, A., &amp; Roweis, S. T. (2006). Metric learning by collapsing classes.
 *       In <em>Advances in Neural Information Processing Systems (NeurIPS) 18</em>, pp. 451–458.</li>
 * </ul>
 */
public final class McmlDml implements ISupervisedDml {

    private int rank = 2;
    private int maxIter = 120;
    /** L-BFGS 容忍度 */
    private double tolerance = 1e-6;
    /** L-BFGS 最大迭代次数 */
    private int maxBfgsIter = 200;
    /** 梯度裁剪阈值 */
    private double gradClip = 5.0;
    /** 内部优化器注入（null 时用 Opts.lbfgs() 默认） */
    private transient IOptimizer optimizer;

    public int getRank() {
        return rank;
    }

    public McmlDml setRank(int rank) {
        this.rank = rank;
        return this;
    }

    /**
     * @deprecated L-BFGS 为自适应步长，调用此方法无实际效果，仅为兼容旧代码。
     */
    @Deprecated
    public McmlDml setLearningRate(double lr) {
        return this;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public McmlDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    public double getTolerance() {
        return tolerance;
    }

    public McmlDml setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public int getMaxBfgsIter() {
        return maxBfgsIter;
    }

    public McmlDml setMaxBfgsIter(int maxBfgsIter) {
        this.maxBfgsIter = maxBfgsIter;
        return this;
    }

    public double getGradClip() {
        return gradClip;
    }

    public McmlDml setGradClip(double gradClip) {
        this.gradClip = gradClip;
        return this;
    }

    public IOptimizer getOptimizer() {
        return optimizer;
    }

    /**
     * 注入自定义优化器（null 时使用 {@link Opts#lbfgs(double, int)}）。
     */
    public McmlDml setOptimizer(IOptimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, McmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, McmlDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        Objects.requireNonNull(x, "x");
        int n = x.length;
        int d = x[0].length;
        int r = Math.min(Math.max(1, rank), d);

        // MCML 目标分布 q（同类均匀）
        double[][] q = new double[n][n];
        MetricEmbeddingOps.mcmlTargetConditional(y, n, q);

        // 初始 L：r×d
        double[][] L0 = new double[r][d];
        for (int i = 0; i < Math.min(r, d); i++) {
            L0[i][i] = 1.0;
        }

        IVector<Double> initVec = Linalg.vector(flatten(L0, r, d));

        IOptimizer opt = optimizer != null
                ? optimizer
                : Opts.lbfgs(tolerance, maxBfgsIter);

        McmlObjective objective = new McmlObjective(x, y, q, n, d, r, gradClip);

        OptResult result;
        if (maxIter <= 0) {
            // 直接 L-BFGS，无外层包装
            result = opt.optimize(initVec, objective, objective);
        } else {
            // 带最大迭代次数封装的 L-BFGS（作为早停上界）
            result = fitWithIterLimit(initVec, opt, objective, r, d);
        }

        IVector<Double> sol = result.getOptimalPoint();
        double[][] L = unflatten(sol, r, d);
        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    /**
     * 带迭代次数上限封装的 L-BFGS 循环：每轮外层迭代重新调用优化器，
     * 以兼容 PSD 投影（不可微后处理）以及最大迭代次数限制。
     */
    private OptResult fitWithIterLimit(IVector<Double> initVec,
            IOptimizer opt, McmlObjective obj, int r, int d) {
        IVector<Double> current = initVec.copy();
        OptResult lastResult = null;

        for (int iter = 0; iter < maxIter; iter++) {
            lastResult = opt.optimize(current, obj, obj);
            IVector<Double> sol = lastResult.getOptimalPoint();
            double[][] Lraw = unflatten(sol, r, d);
            // PSD 投影（与原版行为一致）
            double[][] Lproj = sdProject(Lraw, r, d);
            current = Linalg.vector(flatten(Lproj, r, d));
        }

        return lastResult != null ? lastResult
                : new OptResult.Builder(0.0, current).build();
    }

    // ---- 参数扁平化 / 反扁平化 ----

    private static double[] flatten(double[][] L, int r, int d) {
        double[] out = new double[r * d];
        for (int i = 0; i < r; i++) {
            System.arraycopy(L[i], 0, out, i * d, d);
        }
        return out;
    }

    private static double[][] unflatten(IVector<Double> v, int r, int d) {
        double[] arr = v.toDoubleArray();
        double[][] L = new double[r][d];
        for (int i = 0; i < r; i++) {
            System.arraycopy(arr, i * d, L[i], 0, d);
        }
        return L;
    }

    // ---- MCML 目标函数与梯度 ----

    private static final class McmlObjective implements IObjectiveFunction, IGradientFunction {

        private final double[][] x;
        private final int[] y;
        private final double[][] q;  // 目标分布
        final int n;
        final int d;
        final int r;
        private final double gradClip;

        McmlObjective(double[][] x, int[] y, double[][] q, int n, int d, int r, double gradClip) {
            this.x = x;
            this.y = y;
            this.q = q;
            this.n = n;
            this.d = d;
            this.r = r;
            this.gradClip = gradClip;
        }

        @Override
        public double computeObjective(IVector xv) {
            // MCML 最小化 KL(q || p) = -sum_ij q[i][j] * log(p[i][j])
            // 等价于最大化 -KL，即最小化
            double[][] L = unflatten(xv, r, d);
            double[][] emb = MetricEmbeddingOps.embed(x, n, d, L, r);
            double[][] distSq = MetricEmbeddingOps.pairwiseSquaredDistances(emb, n, r);
            double[][] p = new double[n][n];
            MetricEmbeddingOps.softmaxConditionalFromNegSqDist(distSq, n, p);
            return MetricEmbeddingOps.mcmlKlLoss(p, q, n);
        }

        @Override
        public IVector computeGradient(IVector xv) {
            double[][] L = unflatten(xv, r, d);
            double[][] emb = MetricEmbeddingOps.embed(x, n, d, L, r);
            double[][] distSq = MetricEmbeddingOps.pairwiseSquaredDistances(emb, n, r);
            double[][] p = new double[n][n];
            MetricEmbeddingOps.softmaxConditionalFromNegSqDist(distSq, n, p);
            double[][] weighted = new double[n][n];
            MetricEmbeddingOps.mcmlWeightedKernel(p, q, n, weighted);
            double[][] sym = new double[n][n];
            MetricEmbeddingOps.skewSymmetrizeForGradient(weighted, n, sym);
            double[][] grad = new double[r][d];
            MetricEmbeddingOps.gradientLinearTransform(emb, sym, x, n, d, r, grad);

            if (gradClip > 0) {
                clip(grad, r, d, gradClip);
            }

            return Linalg.vector(flatten(grad, r, d));
        }

        private static void clip(double[][] g, int r, int d, double m) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < d; j++) {
                    if (g[i][j] > m) {
                        g[i][j] = m;
                    } else if (g[i][j] < -m) {
                        g[i][j] = -m;
                    }
                }
            }
        }
    }

    /**
     * SDProject: 将 M = L^T L 投影到 PSD 矩阵，再恢复 L。
     * 与原版一致。
     */
    private static double[][] sdProject(double[][] L, int r, int d) {
        // M = L^T L
        double[][] M = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double sum = 0;
                for (int k = 0; k < r; k++) {
                    sum += L[k][i] * L[k][j];
                }
                M[i][j] = sum;
            }
        }

        // 特征分解
        IMatrix<Double> Mmat = IMatrix.of(M);
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = Mmat.eigen();
        IVector<Double> evals = eigenResult._1;
        IMatrix<Double> evecs = eigenResult._2;

        // 按特征值降序排列
        int numEvals = evals.length();
        Integer[] sortIdx = new Integer[numEvals];
        for (int k = 0; k < numEvals; k++) {
            sortIdx[k] = k;
        }
        Arrays.sort(sortIdx, (a, b) -> Double.compare(
                evals.get(b), evals.get(a)));

        double[][] Lnew = new double[r][d];
        for (int i = 0; i < r; i++) {
            int idx = sortIdx[i];
            double sqrtEval = Math.sqrt(Math.max(0.0, (Double) evals.get(idx)));
            for (int j = 0; j < d; j++) {
                Lnew[i][j] = sqrtEval * (Double) evecs.get(j, idx);
            }
        }
        return Lnew;
    }
}
