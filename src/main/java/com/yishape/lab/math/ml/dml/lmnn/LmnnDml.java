package com.yishape.lab.math.ml.dml.lmnn;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import com.yishape.lab.math.ml.dml.ISupervisedDml;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;

/**
 * 基于<strong>三元组 hinge 损失</strong>的大间隔度量学习，在低秩矩阵 {@code L}（{@code r×d}）上做优化。
 * 本类实现 {@link ISupervisedDml}，超参数通过 {@link #setRank}、{@link #setMaxSteps} 等链式方法配置。
 *
 * <p>与文献 LMNN 的关系：本实现不包含半定规划求解器，而是用 {@code M = L^T L} 的因子
 * {@code L} 参数化，在批量采样的三元组上优化 hinge 损失：</p>
 * <ul>
 *   <li>优化路径：从 v1.0 开始，内部使用 {@link Opts#lbfgs()} 门面委托
 *       {@code RustLBFGS}（底层走 HPC/Rust 高性能路径），替代早期版本的固定学习率随机 SGD；
 *   <li>「目标近邻」在<strong>输入欧氏空间</strong>中预计算，与 LMNN 原文一致。</li>
 * </ul>
 *
 * <p>应用优先 {@link com.yishape.lab.math.ml.ML#dml} 门面；需细调未在
 * {@link com.yishape.lab.math.ml.DmlWrapper} 暴露的项时再直接使用本类。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Weinberger, K. Q., &amp; Saul, L. K. (2009). Distance metric learning for large margin nearest
 *       neighbor classification. <em>Journal of Machine Learning Research</em>, 10, 207–244.</li>
 * </ul>
 */
public final class LmnnDml implements ISupervisedDml {

    /** 嵌入行数 {@code r}，{@code 1 ≤ r ≤ d}（拟合时按维数截断）。 */
    private int rank = 2;
    /** 每样本在欧氏空间中保留的同类最近邻个数（作正样本端点）。 */
    private int targetNeighbors = 1;
    /** Hinge 中异类距离需比较大同类距离多出的裕量。 */
    private double margin = 1.0;
    /** 外层迭代步数上限（每步做一次 L-BFGS 批量优化）。 */
    private int maxSteps = 30;
    /** L-BFGS 容忍度 */
    private double tolerance = 1e-6;
    /** L-BFGS 最大迭代次数 */
    private int maxBfgsIter = 50;
    /** 梯度裁剪阈值 */
    private double gradClip = 5.0;
    /** 内部优化器注入（null 时用默认值） */
    private transient IOptimizer optimizer;
    private Random random;

    public LmnnDml() {
    }

    public int getRank() {
        return rank;
    }

    public LmnnDml setRank(int rank) {
        this.rank = rank;
        return this;
    }

    public int getTargetNeighbors() {
        return targetNeighbors;
    }

    public LmnnDml setTargetNeighbors(int targetNeighbors) {
        this.targetNeighbors = targetNeighbors;
        return this;
    }

    public double getMargin() {
        return margin;
    }

    public LmnnDml setMargin(double margin) {
        this.margin = margin;
        return this;
    }

    public int getMaxSteps() {
        return maxSteps;
    }

    public LmnnDml setMaxSteps(int maxSteps) {
        this.maxSteps = maxSteps;
        return this;
    }

    public double getTolerance() {
        return tolerance;
    }

    public LmnnDml setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public int getMaxBfgsIter() {
        return maxBfgsIter;
    }

    public LmnnDml setMaxBfgsIter(int maxBfgsIter) {
        this.maxBfgsIter = maxBfgsIter;
        return this;
    }

    public double getGradClip() {
        return gradClip;
    }

    public LmnnDml setGradClip(double gradClip) {
        this.gradClip = gradClip;
        return this;
    }

    /**
     * @deprecated L-BFGS 为自适应步长，调用此方法无实际效果，仅为兼容旧代码。
     */
    @Deprecated
    public LmnnDml setLearningRate(double lr) {
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public LmnnDml setRandom(Random random) {
        this.random = random;
        return this;
    }

    public IOptimizer getOptimizer() {
        return optimizer;
    }

    /**
     * 注入自定义优化器（null 时使用 {@link Opts#lbfgs(double, int)}）。
     */
    public LmnnDml setOptimizer(IOptimizer optimizer) {
        this.optimizer = optimizer;
        return this;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        double[][] x = DmlArrays.featureRows(features);
        int[] y = DmlArrays.classIndices(labels);
        return fitFromRows(x, y);
    }

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, LmnnDml learner) {
        return Objects.requireNonNull(learner, "learner").fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, LmnnDml learner) {
        return Objects.requireNonNull(learner, "learner").fit(features, labels);
    }

    /**
     * 核心训练循环：批量 L-BFGS 优化。
     */
    DmlMetric fitFromRows(double[][] x, int[] y) {
        Objects.requireNonNull(x, "x");
        if (x.length == 0) {
            throw new IllegalArgumentException("样本不能为空");
        }
        int n = x.length;
        int d = x[0].length;
        for (int i = 1; i < n; i++) {
            if (x[i].length != d) {
                throw new IllegalArgumentException("各特征行维数须一致");
            }
        }
        if (y.length != n) {
            throw new IllegalArgumentException("标签行数须与样本一致");
        }

        int r = Math.min(Math.max(1, rank), d);
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        // 预计算目标近邻
        List<int[]>[] targets = buildTargetNeighbors(x, y, n, d, targetNeighbors);

        // 预采样三元组批量数据（用于 L-BFGS 批量优化）
        List<TripletSample> batch = buildTripletBatch(x, y, n, d, targets, rnd);

        // 初始 L
        double[][] L0 = new double[r][d];
        for (int i = 0; i < Math.min(r, d); i++) {
            L0[i][i] = 1.0;
        }
        IVector<Double> initVec = Linalg.vector(flatten(L0, r, d));

        IOptimizer opt = optimizer != null
                ? optimizer
                : Opts.lbfgs(tolerance, maxBfgsIter);

        // 构建三元组差值矩阵
        int T = batch.size();
        double[][] u1Data = new double[T][d];
        double[][] u2Data = new double[T][d];
        for (int t = 0; t < T; t++) {
            TripletSample ts = batch.get(t);
            System.arraycopy(ts.u1(), 0, u1Data[t], 0, d);
            System.arraycopy(ts.u2(), 0, u2Data[t], 0, d);
        }

        LmnnAutodiffObjective obj = new LmnnAutodiffObjective(u1Data, u2Data, T, r, d, margin, gradClip);

        IVector<Double> current = initVec.copy();

        for (int step = 0; step < maxSteps; step++) {
            OptResult res = opt.optimize(current, obj, obj);
            current = res.getOptimalPoint().copy();
        }

        double[][] L = unflatten(current, r, d);
        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    /**
     * 预采样批量三元组（随机选择锚点和近邻/异类样本）。
     * 与原版 SGD 的随机单样本策略对齐：每个 epoch 采样 n 个锚点，各配一个同类近邻和一个异类样本。
     * 仅采样 3 个 epoch，总量约为 3n 个三元组，适合 L-BFGS 批量优化。
     */
    private List<TripletSample> buildTripletBatch(double[][] x, int[] y, int n, int d,
            List<int[]>[] targets, Random rnd) {
        List<TripletSample> batch = new ArrayList<>();
        for (int epoch = 0; epoch < 3; epoch++) {
            for (int i = 0; i < n; i++) {
                List<int[]> ti = targets[i];
                if (ti.isEmpty()) continue;
                int j = ti.get(rnd.nextInt(ti.size()))[0];
                int k = rnd.nextInt(n);
                int guard = 0;
                while (y[k] == y[i] && guard++ < n) {
                    k = rnd.nextInt(n);
                }
                if (y[k] == y[i]) continue;

                double[] xi = x[i];
                double[] xj = x[j];
                double[] xk = x[k];
                double[] u1 = new double[d];
                double[] u2 = new double[d];
                for (int t = 0; t < d; t++) {
                    u1[t] = xi[t] - xj[t];
                    u2[t] = xi[t] - xk[t];
                }
                batch.add(new TripletSample(i, j, k, u1, u2));
            }
        }
        return batch;
    }

    private record TripletSample(int i, int j, int k, double[] u1, double[] u2) {}

    // ---- 目标近邻计算 ----

    @SuppressWarnings("unchecked")
    private static List<int[]>[] buildTargetNeighbors(double[][] x, int[] y, int n, int d, int k) {
        List<int[]>[] targets = new List[n];
        for (int i = 0; i < n; i++) {
            targets[i] = new ArrayList<>();
        }
        for (int i = 0; i < n; i++) {
            List<double[]> cand = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i == j || y[j] != y[i]) continue;
                double dist = 0;
                for (int t = 0; t < d; t++) {
                    double u = x[i][t] - x[j][t];
                    dist += u * u;
                }
                cand.add(new double[]{j, dist});
            }
            cand.sort((a, b) -> Double.compare(a[1], b[1]));
            for (int t = 0; t < Math.min(k, cand.size()); t++) {
                targets[i].add(new int[]{(int) cand.get(t)[0]});
            }
        }
        return targets;
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

    // ---- LMNN 自动微分目标函数 ----

    private static final class LmnnAutodiffObjective implements IObjectiveFunction, IGradientFunction {

        private final double[][] u1Data;
        private final double[][] u2Data;
        private final int T, r, d;
        private final double margin;
        private final double gradClip;

        LmnnAutodiffObjective(double[][] u1Data, double[][] u2Data, int T, int r, int d,
                double margin, double gradClip) {
            this.u1Data = u1Data;
            this.u2Data = u2Data;
            this.T = T;
            this.r = r;
            this.d = d;
            this.margin = margin;
            this.gradClip = gradClip;
        }

        private IDiffVector buildLoss(IDiffVector w) {
            IDiffMatrix L = w.reshape(r, d);
            IDiffMatrix LT = L.transpose();
            IDiffMatrix U1 = AD.matrix(u1Data);
            IDiffMatrix U2 = AD.matrix(u2Data);
            IDiffVector ones_r = AD.ones(r);
            IDiffVector dist1 = U1.matmul(LT).square().matmul(ones_r);
            IDiffVector dist2 = U2.matmul(LT).square().matmul(ones_r);
            return dist1.add(margin).sub(dist2).relu().mean();
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
            com.yishape.lab.math.linalg.IDoubleVector g = var.getGradient();
            if (gradClip > 0) {
                double[] gd = g.getData();
                for (int i = 0; i < gd.length; i++) {
                    if (gd[i] > gradClip) gd[i] = gradClip;
                    else if (gd[i] < -gradClip) gd[i] = -gradClip;
                }
            }
            return g;
        }
    }
}
