package com.yishape.lab.math.ml.dml.ddml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.regularization.RereL1Regularization;
import com.yishape.lab.math.optimize.regularization.RereL2Regularization;

import java.util.Arrays;

/**
 * 对角 DDML 的增广 Lagrangian / 乘子法外层迭代，对应
 * {@code refs/ddml/julia_src/solver/RereDiagDmlSolverLangMul.jl} 中 {@code solveDmlLp}。
 *
 * <p>内层将「LP 主目标 + 大罚因子的二次约束违反」与光滑正则
 * {@code regWeight·(α·L1_smooth+(1-α)·L2)} 一并无约束最小化（默认 {@link RereLBFGS}），外层更新乘子向量 {@code β}。</p>
 *
 * <p>{@link #bf(double[], double[], double, double[][], double[], int, int)} /
 * {@link #bfGrad} 对应 Julia 中罚项及其梯度；{@link #calConstraintItemValue} 对应 {@code cal_constraint_item_value}。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Bertsekas, D. P. (1999). <em>Nonlinear Programming</em>（增广 Lagrangian、乘子法）。</li>
 *   <li>Nocedal, J., &amp; Wright, S. J. (2006). <em>Numerical Optimization</em>（约束与罚方法）。</li>
 * </ul>
 *
 * @author lteb2
 */
public final class RereDiagDmlSolverLangMul {

    private static final RereL1Regularization L1 = new RereL1Regularization();
    private static final RereL2Regularization L2 = new RereL2Regularization();

    private RereDiagDmlSolverLangMul() {
    }

    /** 与 {@link RereDiagDmlAdmmSolver} Z 步相同：{@code alpha} 为 L1 份额，{@code 1-alpha} 为 L2 份额。 */
    private static double smoothedRegObjective(IVector<?> x, double alpha) {
        return alpha * L1.computeObjective(x) + (1.0 - alpha) * L2.computeObjective(x);
    }

    private static void addSmoothedRegGradientScaled(IVector<?> x, double alpha, double scale, double[] out) {
        if (scale == 0.0) {
            return;
        }
        IVector g1 = L1.computeGradient(x);
        IVector g2 = L2.computeGradient(x);
        for (int i = 0; i < out.length; i++) {
            out[i] += scale * (alpha * g1.get(i) + (1.0 - alpha) * g2.get(i));
        }
    }

    /**
     * 与 {@link #solve(double[], double[][], double[], double, double, int, double, IOptimizer)} 等价，
     * 入参使用 {@link IVector} / {@link IMatrix}。
     *
     * @return 与原 LP 向量同维最优解向量
     */
    public static IVector<Double> solveVector(IVector<Double> cReduced, IMatrix<Double> aReduced,
            IVector<Double> bVec,
            double regWeight, double alpha,
            int maxOuterSteps, double residualTol, IOptimizer innerOptimizer) {
        double[] xv = solve(cReduced.toDoubleArray(), aReduced.toDoubleArray(), bVec.toDoubleArray(),
                regWeight, alpha, maxOuterSteps, residualTol, innerOptimizer);
        return Linalg.vector(xv);
    }

    /**
     * @param cReduced  长度 m+n（主变量 + 松驰变量）
     * @param aReduced  n × (m+n)
     * @param bVec      n
     * @param regWeight 总正则尺度（乘在光滑 L1/L2 混合项前）
     * @param alpha     L1 份额；{@code 0} 纯 L2，{@code 1} 纯 L1，中间为弹性网（与外层 {@link RereDiagDml} 的 Julia {@code alpha} 一致）
     */
    public static double[] solve(double[] cReduced, double[][] aReduced, double[] bVec,
            double regWeight, double alpha,
            int maxOuterSteps, double residualTol, IOptimizer innerOptimizer) {
        int nInst = bVec.length;
        int nx = cReduced.length;
        if (aReduced.length != nInst || aReduced[0].length != nx) {
            throw new IllegalArgumentException("A,b,c 维数不一致");
        }
        int mFeature = nx - nInst;
        int betaLen = 2 * nInst + mFeature;
        double tho = 1.0e6;
        double[] betaLang = new double[betaLen];
        Arrays.fill(betaLang, 1.0);

        if (innerOptimizer == null) {
            innerOptimizer = new RereLBFGS(10, 1e-7, 800);
        }

        double[] x0 = new double[nx];
        Arrays.fill(x0, 1.0);

        double h0 = violNorm(calConstraintItemValue(x0, aReduced, bVec, nInst, mFeature));
        double residual = 100.0;
        int step = 0;

        // 外层：罚参数 tho 与乘子 betaLang 固定时解内层无约束子问题，再按违反量更新 betaLang（Julia LangMul 主循环）
        while (residual > residualTol && step < maxOuterSteps) {
            step++;
            LangMulInner inner = new LangMulInner(cReduced, aReduced, bVec, regWeight, alpha,
                    tho, betaLang, nInst, mFeature, nx);
            IVector<Double> init = Linalg.vector(Arrays.copyOf(x0, nx));
            OptResult or = innerOptimizer.optimize(init, inner, inner);
            IVector<Double> sol = or.getOptimalPoint();
            for (int i = 0; i < nx; i++) {
                x0[i] = sol.get(i);
            }

            double[] viol = calConstraintItemValue(x0, aReduced, bVec, nInst, mFeature);
            for (int k = 0; k < betaLen; k++) {
                betaLang[k] -= tho * viol[k];
            }
            double h1 = violNorm(viol);
            if (h0 > 0 && h1 / h0 >= 0.1) {
                tho *= 10.0;
            }
            h0 = h1;
            residual = h1;
        }

        return x0;
    }

    /** 约束违反的欧氏范数，驱动外层停机。 */
    private static double violNorm(double[] viol) {
        return Linalg.vector(viol).norm2Value();
    }

    /** {@code r[i] = (A x)_i - b_i}。 */
    private static double[] axMinusB(double[] x, double[][] a, double[] b) {
        int n = a.length;
        int cols = x.length;
        double[] r = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += a[i][j] * x[j];
            }
            r[i] = sum - b[i];
        }
        return r;
    }

    /**
     * 将 LP 不等式残差与变量边界压缩为长度 {@code 2·nInst+mFeature} 的违反向量（仅保留负分量），供外层残差范数使用；
     * 与 Julia {@code cal_constraint_item_value} 对齐。
     */
    static double[] calConstraintItemValue(double[] x, double[][] a, double[] b, int nInst, int mFeature) {
        int betaLen = 2 * nInst + mFeature;
        double[] y = new double[betaLen];
        double[] axb = axMinusB(x, a, b);
        System.arraycopy(axb, 0, y, 0, nInst);
        for (int k = 0; k < mFeature; k++) {
            y[nInst + k] = x[k];
        }
        y[nInst + mFeature] = x[mFeature];
        double[] y2 = new double[betaLen];
        for (int i = 0; i < betaLen; i++) {
            if (y[i] < 0) {
                y2[i] = y[i];
            }
        }
        return y2;
    }

    /**
     * 增广 Lagrangian 中与二次罚 {@code tho}、乘子 {@code betaLang} 对应的平滑项 {@code bf}（Julia 同名），
     * 在 {@code y_i = β_i - tho·(Ax-b)_i} 等为正的区间取 {@code y_i²/(2·tho)} 惩罚。
     */
    static double bf(double[] x, double[] betaLang, double tho, double[][] a, double[] b, int nInst, int mFeature) {
        int betaLen = betaLang.length;
        double[] y = new double[betaLen];
        double[] axb = axMinusB(x, a, b);
        for (int i = 0; i < nInst; i++) {
            y[i] = betaLang[i] - tho * axb[i];
        }
        for (int k = 0; k < mFeature; k++) {
            y[nInst + k] = betaLang[nInst + k] - tho * x[k];
        }
        y[nInst + mFeature] = betaLang[nInst + mFeature] - tho * x[mFeature];

        double[] y2 = new double[betaLen];
        for (int i = 0; i < betaLen; i++) {
            y2[i] = -betaLang[i] * betaLang[i];
        }
        for (int i = 0; i < betaLen; i++) {
            if (y[i] > 0) {
                y2[i] += y[i] * y[i];
            }
        }
        double sum = 0.0;
        for (double v : y2) {
            sum += v;
        }
        return sum / (tho * 2.0);
    }

    /** {@link #bf} 对 {@code x} 的梯度（Julia {@code bf_gra}）。 */
    static double[] bfGrad(double[] x, double[] betaLang, double tho, double[][] a, double[] b, int nInst,
            int mFeature) {
        int nx = x.length;
        double[] y1 = new double[nx];
        for (int j = 0; j < nx; j++) {
            double t = betaLang[nInst + j] - tho * x[j];
            if (t > 0) {
                y1[j] = 2.0 * t * (-tho);
            }
        }
        double[] axb = axMinusB(x, a, b);
        double[] y2 = new double[nx];
        for (int i = 0; i < nInst; i++) {
            double t = betaLang[i] - tho * axb[i];
            if (t > 0) {
                double vf = t;
                for (int j = 0; j < nx; j++) {
                    y2[j] += 2.0 * a[i][j] * vf * (-tho);
                }
            }
        }
        double[] g = new double[nx];
        for (int j = 0; j < nx; j++) {
            g[j] = (y1[j] + y2[j]) / (tho * 2.0);
        }
        return g;
    }

    /**
     * 内层无约束子问题：线性项 {@code c'x} + {@link #bf} + 光滑正则 {@code regWeight·(α L1 + (1-α)L2)}。
     */
    @SuppressWarnings("rawtypes")
    private static final class LangMulInner implements IObjectiveFunction, IGradientFunction {

        private final IVector<Double> cVec;
        private final double[] cRaw;
        private final double[][] a;
        private final double[] b;
        private final double regWeight;
        private final double alpha;
        private final double tho;
        private final double[] betaLang;
        private final int nInst;
        private final int mFeature;
        private final int nx;

        LangMulInner(double[] c, double[][] a, double[] b, double regWeight, double alpha,
                double tho, double[] betaLang, int nInst, int mFeature, int nx) {
            this.cRaw = c;
            this.cVec = Linalg.vector(c);
            this.a = a;
            this.b = b;
            this.regWeight = regWeight;
            this.alpha = alpha;
            this.tho = tho;
            this.betaLang = betaLang;
            this.nInst = nInst;
            this.mFeature = mFeature;
            this.nx = nx;
        }

        @Override
        public double computeObjective(IVector xv) {
            double linear = xv.innerProductValue(cVec);
            double[] x = xv.toDoubleArray();
            return linear + bf(x, betaLang, tho, a, b, nInst, mFeature)
                    + regWeight * smoothedRegObjective(xv, alpha);
        }

        @Override
        public IVector computeGradient(IVector xv) {
            double[] x = xv.toDoubleArray();
            double[] g = Arrays.copyOf(cRaw, nx);
            double[] bg = bfGrad(x, betaLang, tho, a, b, nInst, mFeature);
            for (int i = 0; i < nx; i++) {
                g[i] += bg[i];
            }
            addSmoothedRegGradientScaled(xv, alpha, regWeight, g);
            return Linalg.vector(g);
        }
    }
}
