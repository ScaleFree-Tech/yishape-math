package com.yishape.lab.math.ml.dml.ldml;

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

import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import com.yishape.lab.math.ml.dml.ISupervisedDml;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;

/**
 * 成对 logistic 度量学习（与文献 LDML 类比）：配对标签 {@code s=1} 表同类、{@code s=0} 表异类，
 * 模型 {@code P(s=1) = σ(b − ‖L(x−y)‖²)}，对抽样配对最小化交叉熵。
 *
 * <p>本类实现 {@link ISupervisedDml}。</p>
 *
 * <p>优化路径：从 v1.0 开始，内部使用 {@link Opts#lbfgs()} 门面委托
 * {@code RustLBFGS}（底层走 HPC/Rust 高性能路径），替代早期版本的固定学习率 SGD。
 * 目标函数（成对 logistic 交叉熵）和梯度均直接计算，不依赖采样。</p>
 *
 * @apiNote 应用代码优先 {@link com.yishape.lab.math.ml.ML#dml}；需本包中未在
 * {@link com.yishape.lab.math.ml.DmlWrapper} 暴露的调参时再直接使用本类。
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Guillaumin, M., Verbeek, J., &amp; Schmid, C. (2009). Is that you? Metric learning approaches
 *       for face identification. In <em>ICCV</em>, pp. 498–505.</li>
 *   <li>Chopra, S., Hadsell, R., &amp; LeCun, Y. (2005). Learning a similarity metric discriminatively,
 *       with application to face verification. In <em>CVPR</em>, pp. 539–546.</li>
 * </ul>
 */
public final class LdmlPairwiseDml implements ISupervisedDml {

    private int rank = 2;
    private double bias = 0.5;
    private int maxIter = 120;
    /** L-BFGS 容忍度 */
    private double tolerance = 1e-6;
    /** L-BFGS 最大迭代次数 */
    private int maxBfgsIter = 200;
    /** 梯度裁剪阈值 */
    private double gradClip = 5.0;
    /** 内部优化器注入（null 时用默认值） */
    private transient IOptimizer optimizer;
    private Random random;

    public int getRank() {
        return rank;
    }

    public LdmlPairwiseDml setRank(int rank) {
        this.rank = rank;
        return this;
    }

    public double getBias() {
        return bias;
    }

    public LdmlPairwiseDml setBias(double bias) {
        this.bias = bias;
        return this;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public LdmlPairwiseDml setMaxIter(int maxIter) {
        this.maxIter = maxIter;
        return this;
    }

    /**
     * @deprecated L-BFGS 为自适应步长，调用此方法无实际效果，仅为兼容旧代码。
     */
    @Deprecated
    public LdmlPairwiseDml setLearningRate(double lr) {
        return this;
    }

    /**
     * L-BFGS 外层迭代次数上限（别名，同 {@link #setMaxIter}）。
     */
    public LdmlPairwiseDml setMaxSteps(int steps) {
        this.maxIter = steps;
        return this;
    }

    public double getTolerance() {
        return tolerance;
    }

    public LdmlPairwiseDml setTolerance(double tolerance) {
        this.tolerance = tolerance;
        return this;
    }

    public int getMaxBfgsIter() {
        return maxBfgsIter;
    }

    public LdmlPairwiseDml setMaxBfgsIter(int maxBfgsIter) {
        this.maxBfgsIter = maxBfgsIter;
        return this;
    }

    public double getGradClip() {
        return gradClip;
    }

    public LdmlPairwiseDml setGradClip(double gradClip) {
        this.gradClip = gradClip;
        return this;
    }

    public Random getRandom() {
        return random;
    }

    public LdmlPairwiseDml setRandom(Random random) {
        this.random = random;
        return this;
    }

    public IOptimizer getOptimizer() {
        return optimizer;
    }

    /**
     * 注入自定义优化器（null 时使用 {@link Opts#lbfgs(double, int)}）。
     */
    public LdmlPairwiseDml setOptimizer(IOptimizer optimizer) {
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

    public static DmlMetric fit(IMatrix<Double> features, IVector<?> labels, LdmlPairwiseDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    public static DmlMetric fit(IMatrix<Double> features, String[] labels, LdmlPairwiseDml hyper) {
        return Objects.requireNonNull(hyper).fit(features, labels);
    }

    DmlMetric fitFromRows(double[][] x, int[] y) {
        Objects.requireNonNull(x, "x");
        int n = x.length;
        int d = x[0].length;
        int r = Math.min(Math.max(1, rank), d);
        Random rnd = random != null ? random : ThreadLocalRandom.current();

        // 初始 L：r×d
        double[][] L0 = new double[r][d];
        for (int i = 0; i < Math.min(r, d); i++) {
            L0[i][i] = 0.5;
        }

        IVector<Double> initVec = Linalg.vector(flatten(L0, r, d));

        IOptimizer opt = optimizer != null
                ? optimizer
                : Opts.lbfgs(tolerance, maxBfgsIter);

        // 预采样配对
        List<PairSample> pairs = samplePairs(x, y, n, d, rnd);
        int P = pairs.size();

        // 构建 U 矩阵 (P×d) 和标签数组
        double[][] uData = new double[P][d];
        double[] labelData = new double[P];
        for (int p = 0; p < P; p++) {
            PairSample ps = pairs.get(p);
            System.arraycopy(ps.diff(), 0, uData[p], 0, d);
            labelData[p] = ps.label();
        }

        LdmlAutodiffObjective objective = new LdmlAutodiffObjective(uData, labelData, P, r, d, bias, gradClip);

        OptResult result;
        if (maxIter <= 0) {
            result = opt.optimize(initVec, objective, objective);
        } else {
            result = fitWithIterLimit(initVec, opt, objective);
        }

        IVector<Double> sol = result.getOptimalPoint();
        double[][] L = unflatten(sol, r, d);
        return DmlMetric.lowRank(Linalg.matrix(L));
    }

    private OptResult fitWithIterLimit(IVector<Double> initVec,
            IOptimizer opt, LdmlAutodiffObjective obj) {
        IVector<Double> current = initVec.copy();
        OptResult lastResult = null;
        for (int iter = 0; iter < maxIter; iter++) {
            lastResult = opt.optimize(current, obj, obj);
            current = lastResult.getOptimalPoint().copy();
        }
        return lastResult != null ? lastResult
                : new OptResult.Builder(0.0, current).build();
    }

    private List<PairSample> samplePairs(double[][] x, int[] y, int n, int d, Random rnd) {
        List<PairSample> pairs = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int s = y[i] == y[j] ? 1 : 0;
                double[] u = new double[d];
                for (int t = 0; t < d; t++) {
                    u[t] = x[i][t] - x[j][t];
                }
                pairs.add(new PairSample(i, j, s, u));
            }
        }
        return pairs;
    }

    private record PairSample(int i, int j, int label, double[] diff) {}

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

    // ---- LDML 自动微分目标函数 ----

    private static final class LdmlAutodiffObjective implements IObjectiveFunction, IGradientFunction {

        private final double[][] uData;
        private final double[] labelData;
        private final int P, r, d;
        private final double bias;
        private final double gradClip;

        LdmlAutodiffObjective(double[][] uData, double[] labelData, int P, int r, int d,
                double bias, double gradClip) {
            this.uData = uData;
            this.labelData = labelData;
            this.P = P;
            this.r = r;
            this.d = d;
            this.bias = bias;
            this.gradClip = gradClip;
        }

        private IDiffVector buildLoss(IDiffVector w) {
            IDiffMatrix L = w.reshape(r, d);
            IDiffMatrix U = AD.matrix(uData);
            IDiffMatrix Lu = U.matmul(L.transpose());
            IDiffVector dist = Lu.square().matmul(AD.ones(r));
            IDiffVector z = dist.rsub(bias);
            IDiffVector sigZ = z.sigmoid();
            IDiffVector labels = AD.vector(labelData);
            IDiffVector logSig = sigZ.log();
            IDiffVector logOneMinusSig = sigZ.rsub(1.0).log();
            IDiffVector term1 = logSig.mul(labels);
            IDiffVector term2 = logOneMinusSig.mul(labels.rsub(1.0));
            return term1.add(term2).sum().mul(-1.0 / P);
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
