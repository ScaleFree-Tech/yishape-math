package com.yishape.lab.math.ml.dml.ddml;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.dml.triplet.Triplet;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.admm.ConsensusAdmm;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.regularization.RereL1Regularization;
import com.yishape.lab.math.optimize.regularization.RereL2Regularization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

/**
 * 对角 DDML 的 ADMM 求解器（单线程、按三元组分块），与
 * {@code refs/ddml/julia_src/solver/RereDiagDmlADMMDistributed.jl} 对齐。
 *
 * <p>外层共识 ADMM 循环委托 {@link ConsensusAdmm}；本类仅实现 DDML 专用的 W 步 LP 与 Z 步光滑正则。</p>
 *
 * <p><strong>一轮迭代</strong>：对各块解局部 {@code w}（W 步，化为 LP）、
 * 对偶变量 {@code y} 平均得 {@code ȳ}、各块 {@code w} 平均得 {@code w̄}；再解全局 Z 步更新共识 {@code z}；
 * 最后对每个块做 {@code y ← y + w - z}（与 Julia {@code admmUpdate} 同序）。</p>
 *
 * <p><strong>W 步</strong>（{@link #optimizeWInner}）：在块上调用 {@link DiagDmlCoefficients#fromTriplets} 时固定使用
 * {@code "no_huber"} 与 {@code tau=10}，与 Julia {@code optimizeW} 中 {@code "not_huber", tau=10.0} 一致；
 * 再将 ADMM 一致性约束 {@code |w - (z-y)| ≤ τ₂/ρ} 展开为线性不等式并引入附加松弛。</p>
 *
 * <p><strong>Z 步</strong>（{@link ZStepObjective}）：目标为 {@code regWeight⁴ · g(z) + (n_blocks·ρ/2)·‖z - (w̄+ȳ)‖²}，
 * 其中 {@code g(z) = α·L1_smooth + (1-α)·‖z‖²}，与 Julia {@code optimizeZ} 一致。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Boyd, S., Parikh, N., Chu, E., Peleato, B., &amp; Eckstein, J. (2011). Distributed optimization
 *       and statistical learning via the alternating direction method of multipliers.
 *       <em>Foundations and Trends in Machine Learning</em>, 3(1), 1–122.</li>
 *   <li>Rosales, R., &amp; Fung, G. (2006). Learning sparse metrics via linear programming. <em>ICML</em>.</li>
 * </ul>
 */
@SuppressWarnings("rawtypes")
public final class RereDiagDmlAdmmSolver {

    /** W 步 LP 与 Julia {@code optimizeW} 一致：{@code punishment_mu = 5000}。 */
    private static final double MU_PUNISH = 5000.0;
    /** Julia {@code tau2}：一致性球形约束半径相关。 */
    private static final double TAU2 = 10.0;
    /** W 步构造 {@link DiagDmlCoefficients} 的 {@code tau}，与 Julia {@code tau = 10.0} 一致。 */
    private static final double TAU_CONSTRAINT = 10.0;
    /** 与 Julia {@code optimizeZ}、罚参数放大上限同量级。 */
    private static final double RHO_CEILING = 1e24;
    /** 每轮Successful后 ρ 缩放，与 Julia 一致。 */
    private static final double RHO_MULTIPLIER = 10.0;

    private static final RereL1Regularization L1 = new RereL1Regularization();
    private static final RereL2Regularization L2 = new RereL2Regularization();

    private RereDiagDmlAdmmSolver() {
    }

    /** 共识向量 {@code z}（Z 步输出、非负截断后）及每轮残差序列；语义同 {@link ConsensusAdmm.Outcome}。 */
    public record RegPathResult(IVector<Double> z, List<Double> errors) {
    }

    /**
     * ADMM 主循环：委托 {@link ConsensusAdmm#run}；返回前对 {@code z} 做 {@code max(z,0)}。
     *
     * @param triplets           非空三元组序列（每轮会被原地打乱）
     * @param regWeight          外层正则尺度（Z 步内四次方）
     * @param alpha              L1 混合份额
     * @param random             每轮打乱用；{@code null} 时用 {@code new Random(itr)}
     * @param blockSize          每块三元组条数
     * @param maxAdmmItr         外层迭代上限
     * @param lpSolver           W 步 LP；{@code null} 时默认单纯形
     * @param innerOptimizerForZ Z 步优化器；{@code null} 时默认 L-BFGS
     * @param rhoStart           初始 {@code ρ}
     * @param errorTol           停机阈值（{@link ConsensusAdmm#censusPunishL2Squared}）
     */
    public static RegPathResult iterate(List<Triplet> triplets, double regWeight, double alpha, Random random,
            int blockSize, int maxAdmmItr, ILinProgSolver lpSolver,
            IOptimizer innerOptimizerForZ, double rhoStart, double errorTol) {
        Objects.requireNonNull(triplets, "triplets");
        if (triplets.isEmpty()) {
            throw new IllegalArgumentException("triplets 不能为空");
        }
        int m = triplets.get(0).dimension();
        ILinProgSolver lp = lpSolver != null ? lpSolver : new RereSimplexLinProgSolver();
        IOptimizer zOpt = innerOptimizerForZ != null ? innerOptimizerForZ : new RereLBFGS(10, 1e-7, 400);

        ConsensusAdmm.Config cfg = new ConsensusAdmm.Config(
                rhoStart, RHO_MULTIPLIER, RHO_CEILING, maxAdmmItr, errorTol);

        ConsensusAdmm.Outcome out = ConsensusAdmm.run(
                Linalg.ones(m).multiplyByScalar(2.0),
                cfg,
                (itr, rng) -> {
                    Random shuffleR = rng != null ? rng : new Random(itr);
                    Collections.shuffle(triplets, shuffleR);
                    return splitBlocks(triplets, blockSize);
                },
                (firstBlocks, wMap, yMap, zDim) -> {
                    for (Integer k : firstBlocks.keySet()) {
                        wMap.put(k, Linalg.ones(zDim));
                        yMap.put(k, Linalg.ones(zDim).multiplyByScalar(1.5));
                    }
                },
                (key, blockData, wInit, z, y, rho) -> optimizeWInner(wInit, z, y, rho, regWeight, blockData, m, lp),
                (zInit, wBar, yBar, rho, numBlocks) -> optimizeZInner(
                        zInit, wBar, yBar, rho, numBlocks, Math.pow(regWeight, 4), alpha, zOpt),
                zVec -> zVec.copy().apply(x -> x < 0 ? 0.0 : x),
                random);

        return new RegPathResult(out.z(), out.errors());
    }

    /**
     * Z 步：最多 20 次内层循环，每次用 {@link IOptimizer} 最小化
     * {@code regWtPow4*g(z) + (nBloc*ρ/2)·‖z-(w̄+ȳ)‖²}；内层残差为相邻两次 {@link ConsensusAdmm#censusPunishL2Squared} 之差。
     */
    private static IVector<Double> optimizeZInner(IVector<Double> initZ, IVector<Double> wBar, IVector<Double> yBar,
            double rho, int numBlocks, double regWtPow4, double alphaBlend, IOptimizer optimizer) {

        double scale = numBlocks * rho / 2.0;
        IVector<Double> wy = wBar.add(yBar);
        IVector<Double> zCurr = initZ.copy();
        double threshold = 1e-6;
        double residual = 100.0;
        int loops = 0;
        double errPrev = ConsensusAdmm.censusPunishL2Squared(wBar, zCurr, yBar);

        while (loops < 20 && residual > threshold) {
            loops++;
            ZStepObjective inner = new ZStepObjective(regWtPow4, alphaBlend, wy, scale);
            IVector<Double> zi = zCurr.copy();
            OptResult or = optimizer.optimize(zi, inner, inner);
            IVector<Double> sol = or.getOptimalPoint();
            zCurr = sol.copy();
            double errCurr = ConsensusAdmm.censusPunishL2Squared(wBar, zCurr, yBar);
            residual = Math.abs(errCurr - errPrev);
            errPrev = errCurr;
        }
        return zCurr;
    }

    /**
     * ADMM Z 子问题目标/梯度。
     */
    private static final class ZStepObjective implements IObjectiveFunction, IGradientFunction {
        private final double regWt;
        private final double alphaBlend;
        private final IVector<Double> wy;
        private final double scale;

        ZStepObjective(double regWt, double alphaBlend, IVector<Double> wy, double scale) {
            this.regWt = regWt;
            this.alphaBlend = alphaBlend;
            this.wy = wy;
            this.scale = scale;
        }

        private double blendedRegObjective(IVector z) {
            return alphaBlend * L1.computeObjective(z) + (1.0 - alphaBlend) * L2.computeObjective(z);
        }

        private IVector<Double> blendedRegGrad(IVector z) {
            return L1.computeGradient(z).multiplyByScalar(alphaBlend)
                    .add(L2.computeGradient(z).multiplyByScalar(1.0 - alphaBlend));
        }

        @Override
        public double computeObjective(IVector zz) {
            double regPart = regWt * blendedRegObjective(zz);
            IVector<Double> delta = zz.sub(wy);
            return regPart + scale * delta.innerProductValue(delta);
        }

        @Override
        public IVector computeGradient(IVector zz) {
            IVector<Double> gReg = blendedRegGrad(zz);
            IVector<Double> delta = zz.sub(wy);
            return gReg.multiplyByScalar(regWt).add(delta.multiplyByScalar(2 * scale));
        }
    }

    private static IVector<Double> optimizeWInner(IVector<Double> wInit, IVector<Double> z, IVector<Double> y,
            double rho, double regWeight, List<Triplet> block, int featureDim, ILinProgSolver lpSolver) {

        DiagDmlCoefficients coefFull = DiagDmlCoefficients.fromTriplets(block, MU_PUNISH, "no_huber", TAU_CONSTRAINT);
        double[][] af = coefFull.getA();
        double[] bf = coefFull.getB();
        double[] cf = coefFull.getC();
        int nb = coefFull.numTriplets();
        int ncolRed = featureDim + nb;

        double[][] ac = new double[nb][ncolRed];
        for (int i = 0; i < nb; i++) {
            System.arraycopy(af[i], 0, ac[i], 0, ncolRed);
        }
        double[] cc = Arrays.copyOf(cf, ncolRed);

        IVector<Double> zMinusY = z.sub(y);
        double c1 = -TAU2 / rho;
        double[] bExtend = Arrays.copyOf(bf, nb + 2 * featureDim);
        for (int i = 0; i < featureDim; i++) {
            bExtend[nb + i] = c1 + zMinusY.get(i);
            bExtend[nb + featureDim + i] = c1 - zMinusY.get(i);
        }

        int ncolFull = ncolRed + 2 * featureDim;
        int nrowFull = nb + 2 * featureDim;
        double[][] aBig = new double[nrowFull][ncolFull];

        for (int i = 0; i < nb; i++) {
            System.arraycopy(ac[i], 0, aBig[i], 0, ncolRed);
        }
        for (int r = nb; r < nb + featureDim; r++) {
            int jj = r - nb;
            aBig[r][jj] = 1.0;
            for (int c = ncolRed; c < ncolFull; c++) {
                aBig[r][c] = 1.0;
            }
        }
        for (int r = nb + featureDim; r < nrowFull; r++) {
            int jj = r - (nb + featureDim);
            aBig[r][jj] = -1.0;
            for (int c = ncolRed; c < ncolFull; c++) {
                aBig[r][c] = 1.0;
            }
        }

        double[] cBig = Arrays.copyOf(cc, ncolFull);
        Arrays.fill(cBig, ncolRed, ncolFull, MU_PUNISH);

        try {
            IVector<Double> solFull = DiagDmlLpSolver.solveRaw(cBig, aBig, bExtend, regWeight, lpSolver);
            double[] w = new double[featureDim];
            for (int i = 0; i < featureDim; i++) {
                w[i] = solFull.get(i);
            }
            return Linalg.vector(w);
        } catch (RuntimeException ex) {
            return wInit.copy();
        }
    }

    private static Map<Integer, List<Triplet>> splitBlocks(List<Triplet> triplets, int chunk) {
        int total = triplets.size();
        if (chunk <= 0 || chunk > total) {
            chunk = total;
        }
        int nb = total % chunk == 0 ? total / chunk : total / chunk + 1;
        Map<Integer, List<Triplet>> map = new LinkedHashMap<>();
        for (int b = 0; b < nb; b++) {
            int from = b * chunk;
            int to = Math.min(total, from + chunk);
            map.put(b + 1, new ArrayList<>(triplets.subList(from, to)));
        }
        return map;
    }

}