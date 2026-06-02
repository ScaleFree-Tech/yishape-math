package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;

/**
 * 列主元 QR（QRCP，rank-revealing 风格）：{@code A · P = Q · R}，
 * {@code P} 为列置换、{@code Q} 正交、{@code R} 上三角（选列策略与 Apache Commons Math
 * {@code RRQRDecomposition} 一致）。
 * <p>{@link #getSolver()} 与 {@link IQRDecomposition} 共用 Householder 反演与三角回代：
 * 当 {@code R} 的某主元在阈值下为零（列秩亏损）时 {@code solve} 会失败；秩亏损最小二乘请走
 * {@link com.yishape.lab.math.linalg.decomposition.ISVDDecomposition} 等路径。</p>
 */
public interface IQrcpDecomposition extends IQRDecomposition {

    /**
     * 列主元记录：分解后第 {@code k} 列（在已置换矩阵中）来自原矩阵的第 {@code pivot[k]} 列，
     * 即 {@code (A·P)[:, k] = A[:, pivot[k]]}，其中 {@code P} 由 {@link #getColumnPermutation()} 给出。
     */
    int[] getColumnPivot();

    /**
     * {@code n×n} 列置换矩阵 {@code P}，满足 {@code A·P = Q·R}（{@code n} 为列数）。
     */
    IMatrix<Double> getColumnPermutation();

    /**
     * 在 QRCP 分解结果上构造 xGELSY 风格求解器解 {@code B}：按 {@code rankTolerance} 截断 {@code R}
     * 对角主元，尾随变量置零，再还原 {@code X = P·Z}。适用于最小二乘与秩亏损/近秩亏列空间。
     * 常用阈值见 {@link QrcpRankTolerance#forLeastSquares(IQrcpDecomposition, int, int)}、
     * {@link QrcpRankTolerance#fromRelativeRcond(IQrcpDecomposition, int, int, double)}。
     *
     * @param rankTolerance {@code |R_{ii}|} 不大于此值则视为数值秩截止（与 LAPACK {@code RCOND} 用法类似，为绝对阈值）
     */
    IDecompositionSolver createTolerantLeastSquaresSolver(double rankTolerance);
}
