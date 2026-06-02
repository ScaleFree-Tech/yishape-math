package com.yishape.lab.math.ml.dml.ddml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.ml.dml.DmlArrays;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.ml.dml.triplet.Triplet;
import com.yishape.lab.math.ml.dml.triplet.TripletBuilder;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import com.yishape.lab.math.ml.dml.ISupervisedDml;

/**
 * 对角化距离度量学习（Diagonal DML），与 {@code refs/ddml/julia_src/rere_dml/DiagDml.jl} 中
 * 求解器选择一致；本类实现 {@link ISupervisedDml}，超参数由链式 setter（及若干 {@code get*}）维护。
 * {@link #fit} 与其它 DML 算法一致，返回 {@link DmlMetric}，
 * {@link DmlMetric#form()} 为 {@link com.yishape.lab.math.ml.dml.MetricForm#DIAGONAL}，
 * {@link DmlMetric#transformMatrix()} 为 d×d 对角矩阵（Mahalanobis 精度仅对角非零）。
 * 约束与求解分支与仓库内 {@code refs/ddml/...} 中 Julia 参考一致；三元组由
 * {@link TripletBuilder} 构建。
 * <p>
 * <strong>Julia 源文件对照：</strong>{@code DiagDml.jl}（{@code solve_diag_dml}、系数构造）；
 * {@code RereDmlLpSolver.jl}（单阶 LP）；{@code RereDiagDmlSolverLangMul.jl}（增广 Lagrangian）；
 * {@code RereDiagDmlADMMDistributed.jl}（ADMM）。路径均相对于仓库 {@code refs/ddml/julia_src/}。
 * </p>
 * <p>
 * 正则强度由 {@link #setL1Weight(double) l1Weight}、{@link #setL2Weight(double) l2Weight} 确定，规则与
 * {@link com.yishape.lab.math.ml.reg.RereLinearRegression} 的 {@code lambda1}/{@code lambda2} 一致：
 * 均为 0 则无正则；仅 L2、仅 L1、弹性网分别对应 (0,{@code >0})、({@code >0},0)、均{@code >0}。内部再折合为
 * Julia 参考实现所用的单尺度 {@code regWeight} 与混合系数 {@code alpha}（弹性网：
 * {@code regWeight=l1+l2}，{@code alpha=l1/(l1+l2)}）。若原先使用 Julia 的 {@code regWeight=W} 与 {@code a}，
 * 可设 {@code l1Weight=a·W}、{@code l2Weight=(1-a)·W}。
 * </p>
 * <p>
 * <strong>求解覆盖：</strong>
 * <ul>
 *   <li>无正则（两权重均为 0）：线性目标 DDML，{@code Ax >= b, x >= 0}。</li>
 *   <li>纯 L1：单阶 LP，{@code min c'x + l1Weight·Σ x特征}（{@link DiagDmlLpSolver}）。</li>
 *   <li>含 L2（纯 L2 或弹性网）：默认 {@link RereDiagDmlAdmmSolver}；{@link #setUseAdmm(boolean) useAdmm=false} 时
 *       走 {@link RereDiagDmlSolverLangMul}。内层默认 L-BFGS，光滑 L1/L2 混合与 {@link RereDiagDmlAdmmSolver} Z 步同式。</li>
 * </ul>
 * 学得的向量经非负截断、开方后用作特征逐维缩放（与 Julia {@code sqrt.(x)} 后右乘数据一致）。
 * </p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Li, T., Kou, G., Peng, Y., &amp; Yu, P. S. (2024). Feature selection and grouping effect analysis
 *       for credit evaluation via regularized diagonal distance metric learning.
 *       <em>INFORMS Journal on Computing</em>. DOI: <a href="https://doi.org/10.1287/ijoc.2023.0322">10.1287/ijoc.2023.0322</a>.</li>
 *   <li>Rosales, R., &amp; Fung, G. (2006). Learning sparse metrics via linear programming.
 *       In <em>Proceedings of ICML</em>, pp. 367–374.</li>
 *   <li>Schultz, M., &amp; Joachims, T. (2003). Learning a distance metric from relative comparisons.
 *       In <em>NeurIPS</em> 16.</li>
 *   <li>Weinberger, K. Q., &amp; Saul, L. K. (2009). Distance metric learning for large margin nearest
 *       neighbor classification. <em>JMLR</em>, 10, 207–244.</li>
 * </ul>
 *
 * @author lteb2
 */
public class RereDiagDml implements Serializable, ISupervisedDml {

    private static final double ALPHA_L1_THRESHOLD = 1.0 - 1e-9;

    private static final long serialVersionUID = 1L;

    /** L1（稀疏）正则强度；与 {@link com.yishape.lab.math.ml.reg.RereLinearRegression#getLambda1()} 判定规则一致。 */
    private double l1Weight;
    /** L2 正则强度；与 {@link com.yishape.lab.math.ml.reg.RereLinearRegression#getLambda2()} 判定规则一致。 */
    private double l2Weight;
    /** {@code huber} 或 {@code no_huber}。 */
    private String distance = "huber";
    /** 约束右端尺度，默认 {@code 2^8}。 */
    private double tau = 256.0;
    private int maxTriplets = 10_000_000;
    /** 松驰惩罚系数 {@code punishment_mu}，默认 5000。 */
    private double punishmentMu = 5000.0;
    private transient ILinProgSolver linProgSolver;
    /** L2 / Elastic：{@code true} 用 ADMM，{@code false} 用增广拉格朗日 LangMul。 */
    private boolean useAdmm = true;
    /** ADMM 块大小；对齐 Julia {@code num_each = 1000}。 */
    private int admmBlockSize = 1000;
    /** ADMM 外层迭代上限；对齐 Julia {@code max_itr = 30}。 */
    private int maxAdmmIterations = 30;
    private double rhoStart = 10.0;
    private double admmErrorTol = 1e-4;
    /** LangMul 外层步上限；对齐 Julia {@code maxStep = 200}。 */
    private int maxLangMulOuterIterations = 200;
    private double langMulResidualTol = 1e-4;
    /** 非 null：该 Random 用于每轮打乱；null：每轮用 {@code new Random(itr)}。 */
    private transient Random admmShuffleRandom;
    /** LangMul 内层与 ADMM Z 步共用的光滑优化器；{@code null} 时用求解器内置 L-BFGS。 */
    private transient IOptimizer innerOptimizer;
    /** 构建三元组时是否启用并行。 */
    private boolean parallelTripletBuild;
    /** 三元组采样时的向量近邻策略。 */
    private VecSearchOption vectorNnSearchOptions = VecSearchOption.DEFAULT;

    public double getL1Weight() {
        return l1Weight;
    }

    public RereDiagDml setL1Weight(double l1Weight) {
        this.l1Weight = l1Weight;
        return this;
    }

    public double getL2Weight() {
        return l2Weight;
    }

    public RereDiagDml setL2Weight(double l2Weight) {
        this.l2Weight = l2Weight;
        return this;
    }

    /** 与链式调用一致地设置 L1/L2 权重。 */
    public RereDiagDml setRegularization(double l1Weight, double l2Weight) {
        this.l1Weight = l1Weight;
        this.l2Weight = l2Weight;
        return this;
    }

    public String getDistance() {
        return distance;
    }

    public RereDiagDml setDistance(String distance) {
        this.distance = distance;
        return this;
    }

    public double getTau() {
        return tau;
    }

    public RereDiagDml setTau(double tau) {
        this.tau = tau;
        return this;
    }

    public int getMaxTriplets() {
        return maxTriplets;
    }

    public RereDiagDml setMaxTriplets(int maxTriplets) {
        this.maxTriplets = maxTriplets;
        return this;
    }

    public double getPunishmentMu() {
        return punishmentMu;
    }

    public RereDiagDml setPunishmentMu(double punishmentMu) {
        this.punishmentMu = punishmentMu;
        return this;
    }

    public ILinProgSolver getLinProgSolver() {
        return linProgSolver;
    }

    /** 若为 null，内部使用 {@link RereSimplexLinProgSolver}。 */
    public RereDiagDml setLinProgSolver(ILinProgSolver linProgSolver) {
        this.linProgSolver = linProgSolver;
        return this;
    }

    public boolean isUseAdmm() {
        return useAdmm;
    }

    public RereDiagDml setUseAdmm(boolean useAdmm) {
        this.useAdmm = useAdmm;
        return this;
    }

    public int getAdmmBlockSize() {
        return admmBlockSize;
    }

    public RereDiagDml setAdmmBlockSize(int admmBlockSize) {
        this.admmBlockSize = admmBlockSize;
        return this;
    }

    public int getMaxAdmmIterations() {
        return maxAdmmIterations;
    }

    public RereDiagDml setMaxAdmmIterations(int maxAdmmIterations) {
        this.maxAdmmIterations = maxAdmmIterations;
        return this;
    }

    public double getRhoStart() {
        return rhoStart;
    }

    public RereDiagDml setRhoStart(double rhoStart) {
        this.rhoStart = rhoStart;
        return this;
    }

    public double getAdmmErrorTol() {
        return admmErrorTol;
    }

    public RereDiagDml setAdmmErrorTol(double admmErrorTol) {
        this.admmErrorTol = admmErrorTol;
        return this;
    }

    public int getMaxLangMulOuterIterations() {
        return maxLangMulOuterIterations;
    }

    public RereDiagDml setMaxLangMulOuterIterations(int maxLangMulOuterIterations) {
        this.maxLangMulOuterIterations = maxLangMulOuterIterations;
        return this;
    }

    public double getLangMulResidualTol() {
        return langMulResidualTol;
    }

    public RereDiagDml setLangMulResidualTol(double langMulResidualTol) {
        this.langMulResidualTol = langMulResidualTol;
        return this;
    }

    public Random getAdmmShuffleRandom() {
        return admmShuffleRandom;
    }

    public RereDiagDml setAdmmShuffleRandom(Random admmShuffleRandom) {
        this.admmShuffleRandom = admmShuffleRandom;
        return this;
    }

    public IOptimizer getInnerOptimizer() {
        return innerOptimizer;
    }

    public RereDiagDml setInnerOptimizer(IOptimizer innerOptimizer) {
        this.innerOptimizer = innerOptimizer;
        return this;
    }

    public boolean isParallelTripletBuild() {
        return parallelTripletBuild;
    }

    public RereDiagDml setParallelTripletBuild(boolean parallelTripletBuild) {
        this.parallelTripletBuild = parallelTripletBuild;
        return this;
    }

    public VecSearchOption getVectorNnSearchOptions() {
        return vectorNnSearchOptions;
    }

    public RereDiagDml setVectorNnSearchOptions(VecSearchOption vectorNnSearchOptions) {
        this.vectorNnSearchOptions = vectorNnSearchOptions != null
                ? vectorNnSearchOptions : VecSearchOption.DEFAULT;
        return this;
    }

    /**
     * 当前超参数下是否可走「一次线性规划」路径（与 {@code DiagDml.jl}：L1 用 LP、L2/弹性用 ADMM/LangMul 的分支一致）。
     *
     * @return {@code true}：无正则，或 {@code α} 近 1（纯 L1，单阶 LP）；{@code false}：含 L2/弹性（ADMM 或 LangMul）
     */
    public boolean isLpSupported() {
        double rw = solverRegWeight();
        if (rw <= 0) {
            return true;
        }
        return solverAlpha() >= ALPHA_L1_THRESHOLD;
    }

    @Override
    public DmlMetric fit(IMatrix<Double> features, IVector<?> labels) {
        Objects.requireNonNull(labels, "labels");
        return fit(features, DmlArrays.stringLabels(labels));
    }

    /**
     * 拟合对角度量：采样三元组 → 构造 LP 系数 → 按正则类型选 LP / ADMM / LangMul → 取前 {@code m} 维作特征缩放（开方）。
     *
     * @return {@link DmlMetric#diagonal}，缩放因子为非负开方后的解
     */
    @Override
    public DmlMetric fit(IMatrix<Double> features, String[] labels) {
        Objects.requireNonNull(features, "features");
        Objects.requireNonNull(labels, "labels");
        validateRegularizationWeights();
        double regWeight = solverRegWeight();
        double alpha = solverAlpha();

        List<Triplet> triplets = TripletBuilder.build(features, labels, maxTriplets,
                parallelTripletBuild, vectorNnSearchOptions);
        if (triplets.isEmpty()) {
            throw new IllegalStateException("未能生成任何三元组；请检查每类至少 2 个样本且存在异类。");
        }

        DiagDmlCoefficients coef = DiagDmlCoefficients.fromTriplets(
                triplets, punishmentMu, distance, tau);
        int m = coef.featureDim();
        ILinProgSolver solver = linProgSolver != null ? linProgSolver : ILinProgSolver.of();

        IVector<Double> featureHead;
        // 对应 DiagDml.jl：无正则或 regType≈L1 时 RereDmlLpSolver.solveDmlLp；否则 ADMM 或 LangMul
        if (regWeight <= 0 || alpha >= ALPHA_L1_THRESHOLD) {
            IVector<Double> xFull = DiagDmlLpSolver.solve(coef, regWeight, solver);
            featureHead = xFull.slice(0, m);
        } else {
            if (useAdmm) {
                RereDiagDmlAdmmSolver.RegPathResult rr = RereDiagDmlAdmmSolver.iterate(
                        triplets,
                        regWeight,
                        alpha,
                        admmShuffleRandom,
                        admmBlockSize,
                        maxAdmmIterations,
                        solver,
                        innerOptimizer,
                        rhoStart,
                        admmErrorTol);
                featureHead = rr.z();
            } else {
                DiagDmlCoefficients.TruncatedLangMulProblem t = coef.truncateForLangMul();
                IVector<Double> xLm = RereDiagDmlSolverLangMul.solveVector(
                        t.cVector(),
                        t.aMatrix(),
                        t.bVector(),
                        regWeight,
                        alpha,
                        maxLangMulOuterIterations,
                        langMulResidualTol,
                        innerOptimizer);
                featureHead = xLm.slice(0, m);
            }
        }

        IVector<Double> scaling = sqrtScaleNonnegative(featureHead);
        return DmlMetric.diagonal(scaling, triplets.size(), coef.numTriplets());
    }

    /**
     * 列缩放：与已拟合 {@link DmlMetric#diagonalWeights()}（{@link com.yishape.lab.math.ml.dml.MetricForm#DIAGONAL}）一致。
     */
    public static IMatrix transform(IMatrix data, IVector scaling) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(scaling, "scaling");
        if (data.getColNum() != scaling.length()) {
            throw new IllegalArgumentException("列数须等于 scaling 长度");
        }
        int rows = data.getRowNum();
        int cols = data.getColNum();
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            IVector row = data.getRow(i);
            for (int j = 0; j < cols; j++) {
                out[i][j] = row.get(j) * scaling.get(j);
            }
        }
        return Linalg.matrix(out);
    }

    private void validateRegularizationWeights() {
        if (l1Weight < 0 || l2Weight < 0) {
            throw new IllegalArgumentException("l1Weight 与 l2Weight 须 >= 0");
        }
    }

    /** 折合为 Julia {@code regWeight}；弹性网为 {@code l1+l2}。 */
    private double solverRegWeight() {
        if (l1Weight <= 0 && l2Weight <= 0) {
            return 0;
        }
        if (l1Weight > 0 && l2Weight <= 0) {
            return l1Weight;
        }
        if (l1Weight <= 0) {
            return l2Weight;
        }
        return l1Weight + l2Weight;
    }

    /** 折合为 Julia {@code alpha}；无正则时返回值未使用。 */
    private double solverAlpha() {
        if (l1Weight <= 0 && l2Weight <= 0) {
            return 0.5;
        }
        if (l1Weight > 0 && l2Weight <= 0) {
            return 1.0;
        }
        if (l1Weight <= 0) {
            return 0.0;
        }
        return l1Weight / (l1Weight + l2Weight);
    }

    /** 与 {@code DiagDml.jl} 尾部一致：负分量置 0 后逐维开方，再数值舍入。 */
    private static IVector<Double> sqrtScaleNonnegative(IVector<Double> featurePart) {
        return featurePart.apply(x -> {
            double v = x;
            if (v < 0) {
                v = 0;
            }
            return round12(Math.sqrt(v));
        });
    }

    private static double round12(double v) {
        return Math.round(v * 1e12) / 1e12;
    }
}
