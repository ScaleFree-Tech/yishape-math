package com.yishape.lab.math.ml.dml.nca;

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

import java.util.Objects;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * Neighborhood Components Analysis（NCA）：学习秩为 {@code rank} 的线性映射 {@code L}，
 * 对每个样本定义留一法 softmax 邻域，<strong>最大化</strong>分配给同类邻居的概率质量。
 *
 * <p>本类实现 {@link ISupervisedDml}；超参数见链式 setter。</p>
 *
 * <p>优化路径：从 v1.0 开始，内部使用 {@link Opts#lbfgs()} 门面委托
 * {@code RustLBFGS}（底层走 HPC/Rust 高性能路径），替代早期版本的固定学习率 SGD。
 * 目标函数（留一法 softmax 邻域概率）和梯度均委托给 {@link MetricEmbeddingOps}。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}；需未在
 * {@link com.yishape.lab.math.ml.DmlWrapper} 暴露的调参时再直接使用本类。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Goldberger, J., Roweis, S. T., Hinton, G. E., &amp; Salakhutdinov, R. (2005). Neighbourhood
 *       components analysis. In <em>Advances in Neural Information Processing Systems (NeurIPS) 17</em>,
 *       pp. 513–520.</li>
 * </ul>
 */
public final class NcaDml implements ISupervisedDml {

    private int rank = 2;
    private int maxIter = 120;
    /** L-BFGS 容忍度（优化器收敛判据） */
    private double tolerance = 1e-6;
    /** L-BFGS 最大迭代次数 */
    private int maxBfgsIter = 200;
    /** L-BFGS 历史大小 */
    private int lbfgsM = 10;
    /** 梯度裁剪阈值 */
    private double gradClip = 5.0;
    /** 收敛判断的 loss 变化阈值；不设置（null）时使用固定 maxIter */
    private Double convergenceTol;
    /** 连续多少轮 loss 改善小于阈值认为收敛 */
    private int patience = 10;
    /** 内部优化器注入（可为 null，用默认值） */
    private transient IOptimizer optimizer;

    public int getRank() {
        return rank;
    }

    public NcaDml setRank(int rank) {
        this.rank = rank;
        return this;
    }

    /**
     * @deprecated L-BFGS 为自适应步长，调用此方法无实际效果，仅为兼容旧代码。
     */
    @Deprecated
    public NcaDml setLearningRate(double lr) {
        return this;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public NcaDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    public double getTolerance() {
        return tolerance;
    }

    public NcaDml setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public int getMaxBfgsIter() {
        return maxBfgsIter;
    }

    public NcaDml setMaxBfgsIter(int maxBfgsIter) {
        this.maxBfgsIter = maxBfgsIter;
        return this;
    }

    public int getLbfgsM() {
        return lbfgsM;
    }

    public NcaDml setLbfgsM(int lbfgsM) {
        this.lbfgsM = lbfgsM;
        return this;
    }

    public double getGradClip() {
        return gradClip;
    }

    public NcaDml setGradClip(double gradClip) {
        this.gradClip = gradClip;
        return this;
    }

    public Double getConvergenceTol() {
        return convergenceTol;
    }

    public NcaDml setConvergenceTol(Double convergenceTol) {
        this.convergenceTol = convergenceTol;
        return this;
    }

    public int getPatience() {
        return patience;
    }

    public NcaDml setPatience(int patience) {
        this.patience = patience;
        return this;
    }

    public IOptimizer getOptimizer() {
        return optimizer;
    }

    /**
     * 注入自定义优化器（用于高级调参或单元测试注入 mock）。
     * 为 null 时使用 {@link Opts#lbfgs(double, int)} 默认实例。
     */
    public NcaDml setOptimizer(IOptimizer optimizer) {
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

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, NcaDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, NcaDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        Objects.requireNonNull(x, "x");
        int n = x.length;
        int d = x[0].length;
        int r = Math.min(Math.max(1, rank), d);

        boolean[][] same = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                same[i][j] = i != j && y[i] == y[j];
            }
        }

        // 初始 L：r×d，单位子矩阵
        double[][] L0 = new double[r][d];
        for (int i = 0; i < Math.min(r, d); i++) {
            L0[i][i] = 1.0;
        }

        IVector<Double> initVec = Linalg.vector(flatten(L0, r, d));

        IOptimizer opt = optimizer != null
                ? optimizer
                : Opts.lbfgs(tolerance, maxBfgsIter);

        NcaObjective objective = new NcaObjective(x, y, same, n, d, r, gradClip);

        OptResult result;
        if (convergenceTol != null) {
            // 带早停的 L-BFGS 迭代包装
            result = fitWithConvergenceCheck(initVec, opt, objective, r, d);
        } else {
            // 直接调用 L-BFGS（内部自行控制迭代）
            result = opt.optimize(initVec, objective, objective);
        }

        IVector<Double> sol = result.getOptimalPoint();
        double[][] L = unflatten(sol, r, d);
        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    /**
     * 带早停检测的 L-BFGS 迭代：每次外层迭代构造新的 L-BFGS 调用，
     * 直到 loss 改善不足 patience 轮或达到 maxIter。
     */
    private OptResult fitWithConvergenceCheck(IVector<Double> initVec,
            IOptimizer opt, NcaObjective obj, int r, int d) {
        int n = obj.n;
        int d0 = obj.d;
        int r0 = obj.r;

        IVector<Double> current = initVec.copy();
        double prevLoss = -obj.computeObjective(current);
        int noImprovementCount = 0;

        for (int outer = 0; outer < maxIter; outer++) {
            OptResult res = opt.optimize(current, obj, obj);
            IVector<Double> sol = res.getOptimalPoint();
            current = sol.copy();

            double loss = -obj.computeObjective(current);
            double improvement = loss - prevLoss;

            if (improvement < convergenceTol) {
                noImprovementCount++;
                if (noImprovementCount >= patience) {
                    break;
                }
            } else {
                noImprovementCount = 0;
            }
            prevLoss = loss;
        }

        OptResult.Builder builder = new OptResult.Builder(-prevLoss, current)
                .converged(true)
                .convergenceReason("NCA early stopping: loss improvement below tolerance");
        return builder.build();
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

    // ---- NCA 目标函数与梯度（同时实现 IObjectiveFunction + IGradientFunction） ----

    private static final class NcaObjective implements IObjectiveFunction, IGradientFunction {

        private final double[][] x;   // n×d
        private final int[] y;
        private final boolean[][] same;
        final int n;
        final int d;
        final int r;
        private final double gradClip;

        NcaObjective(double[][] x, int[] y, boolean[][] same, int n, int d, int r, double gradClip) {
            this.x = x;
            this.y = y;
            this.same = same;
            this.n = n;
            this.d = d;
            this.r = r;
            this.gradClip = gradClip;
        }

        @Override
        public double computeObjective(IVector xv) {
            double[][] L = unflatten(xv, r, d);
            double[][] emb = MetricEmbeddingOps.embed(x, n, d, L, r);
            double[][] distSq = MetricEmbeddingOps.pairwiseSquaredDistances(emb, n, r);
            double[][] p = new double[n][n];
            MetricEmbeddingOps.softmaxConditionalFromNegSqDist(distSq, n, p);
            // NCA 目标：同类分配概率之和（越大越好）
            return MetricEmbeddingOps.ncaLoss(p, same, n);
        }

        @Override
        public IVector computeGradient(IVector xv) {
            double[][] L = unflatten(xv, r, d);
            double[][] emb = MetricEmbeddingOps.embed(x, n, d, L, r);
            double[][] distSq = MetricEmbeddingOps.pairwiseSquaredDistances(emb, n, r);
            double[][] p = new double[n][n];
            MetricEmbeddingOps.softmaxConditionalFromNegSqDist(distSq, n, p);
            double[][] weighted = new double[n][n];
            MetricEmbeddingOps.ncaWeightedKernel(p, same, n, weighted);
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
}
