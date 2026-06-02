package com.yishape.lab.math.ml.dml.ddml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;

import java.util.Objects;

/**
 * 对角化 DML 的线性规划（LP）子问题封装，与 Julia {@code RereDmlLpSolver.jl#solveDmlLp} 对齐。
 *
 * <p><strong>数学形式</strong>（全部决策变量非负）：</p>
 * <pre>
 *   minimize  c'x + regLinSumCoeff · Σ<sub>i</sub> x<sub>i</sub>
 *   s.t.      A x &gt;= b,  x &gt;= 0
 * </pre>
 * <p>实现上等价于在目标中对每个分量使用修改后的线性价 {@code (c<sub>i</sub> + regLinSumCoeff) x<sub>i</sub>}，
 * 再将 {@code A x &gt;= b} 改写为 {@code (-A) x &lt;= (-b)} 以调用 {@link ILinProgSolver}。</p>
 *
 * <p><strong>调用方</strong>：{@link RereDiagDml}（无正则 / 纯 L1 单阶 LP）、{@link RereDiagDmlAdmmSolver}（ADMM 的 W 步内层 LP）。</p>
 *
 * <p>Julia 实现见 {@code refs/ddml/julia_src/solver/RereDmlLpSolver.jl}（函数 {@code solveDmlLp}）；目标中
 * {@code regWeight·sum(x)} 通过抬高全体 {@code c_i} 实现，与注释「等价于 {@code (c_i+regWeight) x_i}」一致。</p>
 *
 * <h2>参考文献</h2>
 * <ul>
 *   <li>Rosales, R., &amp; Fung, G. (2006). Learning sparse metrics via linear programming. <em>ICML</em>.</li>
 * </ul>
 */
public final class DiagDmlLpSolver {

    private DiagDmlLpSolver() {
    }

    /**
     * 求解「≥ 约束 + 非负变量 + 目标含 {@code regLinSumCoeff·Σx}」的 LP；本重载使用容器类型，语义与
     * {@link #solveRaw(double[], double[][], double[], double, ILinProgSolver)} 相同。
     *
     * @param c                目标线性项 {@code c}，长度 = 变量维数 {@code n}；不得为 {@code null}
     * @param aGe              约束矩阵 {@code A}，{@code m×n}，行对应 {@code A x &gt;= b}；不得为 {@code null}
     * @param bGe              约束右端 {@code b}，长度 {@code m}；不得为 {@code null}
     * @param regLinSumCoeff  加在 {@code Σ x} 上的系数；{@code 0} 表示无此项（纯 {@code c'x}）
     * @param solver          线性规划求解器；{@code null} 时使用 {@link com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver}
     * @return 最优解向量 {@code x*}，长度 {@code n}，满足非负与约束（在求解器报告最优的前提下）
     * @throws NullPointerException     若 {@code c}、{@code aGe}、{@code bGe} 任一为 {@code null}
     * @throws IllegalStateException    若求解未返回 {@link OptResult#getOptimalPoint() 可行最优点}
     * @see #solveRaw(double[], double[][], double[], double, ILinProgSolver)
     */
    public static IVector<Double> solveRaw(IVector<Double> c, IMatrix<Double> aGe, IVector<Double> bGe,
            double regLinSumCoeff, ILinProgSolver solver) {
        Objects.requireNonNull(c, "c");
        Objects.requireNonNull(aGe, "A");
        Objects.requireNonNull(bGe, "b");
        return solveRaw(c.toDoubleArray(), aGe.toDoubleArray(), bGe.toDoubleArray(), regLinSumCoeff, solver);
    }

    /**
     * 原始 Java 数组形式的稠密 LP，目标为 {@code min c'x + regLinSumCoeff·Σx}，约束 {@code A x &gt;= b}，{@code x &gt;= 0}。
     *
     * @param c                长度 {@code n} 的线性目标系数；克隆后会在每个分量上加 {@code regLinSumCoeff}；不得为 {@code null}
     * @param aGe              {@code m×n}，{@code aGe[i]·x &gt;= bGe[i]}；行数须与 {@code bGe} 一致、列数须为 {@code n}；不得为 {@code null}
     * @param bGe              长度 {@code m} 的约束右端；不得为 {@code null}
     * @param regLinSumCoeff  加在目标 {@code Σ x} 上的系数；允许为 {@code 0}
     * @param solver          求解器；{@code null} 时默认 {@link com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver}
     * @return 长度 {@code n} 的最优决策向量（与传入 {@code c} 维数一致）
     * @throws NullPointerException     若 {@code c}、{@code aGe}、{@code bGe} 任一为 {@code null}
     * @throws IllegalStateException    若 {@link ILinProgSolver#solve} 未给出最优点
     */
    public static IVector<Double> solveRaw(double[] c, double[][] aGe, double[] bGe, double regLinSumCoeff,
            ILinProgSolver solver) {
        Objects.requireNonNull(c, "c");
        Objects.requireNonNull(aGe, "A");
        Objects.requireNonNull(bGe, "b");
        if (solver == null) {
            solver = ILinProgSolver.of();
        }
        double[] cEff = c.clone();
        for (int i = 0; i < cEff.length; i++) {
            cEff[i] += regLinSumCoeff;
        }
        int r = aGe.length;
        int cols = cEff.length;
        double[][] negA = new double[r][cols];
        double[] negB = new double[r];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < cols; j++) {
                negA[i][j] = -aGe[i][j];
            }
            negB[i] = -bGe[i];
        }
        IVector<Double> cVec = Linalg.vector(cEff);
        IMatrix<Double> aUb = Linalg.matrix(negA);
        IVector<Double> bUb = Linalg.vector(negB);
        OptResult or = solver.solve(cVec, aUb, bUb);
        if (or.getOptimalPoint() == null) {
            throw new IllegalStateException("LP 未返回最优解: " + or.getConvergenceReason());
        }
        return or.getOptimalPoint();
    }

    /**
     * 由三元组系数对象组装的 DDML LP：目标 {@code coef} 给出的 {@code c'x + regWeight·Σx}，约束与 {@link DiagDmlCoefficients} 一致。
     *
     * @param coef       由 {@link DiagDmlCoefficients#fromTriplets} 等构造的 LP 标准数据（{@code c,A,b}）；不得为 {@code null}
     * @param regWeight  加在目标 {@code sum(x)} 上的 L1 型权重；{@code 0} 对应无非负加权和正则项（纯线性主目标）
     * @param solver     线性规划求解器；{@code null} 时使用 {@link com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver}
     * @return 与 {@code coef} 变量维数一致的最优解 {@code x*}（含松弛变量等在 {@code coef} 中的全部分量）
     * @throws NullPointerException     若 {@code coef} 为 {@code null}
     * @throws IllegalStateException    若 LP 未收敛到可返回的最优点
     * @see #solveRaw(double[], double[][], double[], double, ILinProgSolver)
     */
    public static IVector<Double> solve(DiagDmlCoefficients coef, double regWeight, ILinProgSolver solver) {
        return solveRaw(coef.vectorC(), coef.matrixA(), coef.vectorB(), regWeight, solver);
    }
}
