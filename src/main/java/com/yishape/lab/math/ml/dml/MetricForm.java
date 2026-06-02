package com.yishape.lab.math.ml.dml;

/**
 * {@link DmlMetric} 内部左乘矩阵 {@code A} 的<strong>存储形态</strong>，
 * 决定 {@link DmlMetric#transformMatrix()} 的形状与语义。
 *
 * <p>统一约定：{@code A} 行数为 {@link DmlMetric#outputDimension()}，列数为
 * {@link DmlMetric#inputDimension()}（即「输出行 × 输入列」）。</p>
 *
 * <table border="1" cellpadding="4" summary="变换矩阵形状">
 *   <caption>各类型的变换矩阵 {@code A}（形状为 output×input，即 rows×cols）</caption>
 *   <tr><th>{@link #DIAGONAL}</th><td>{@code d×d} 对角矩阵，仅 {@code A[j,j]} 非零</td></tr>
 *   <tr><th>{@link #FULL_WHITENING}</th><td>{@code d×d} 方阵（whitener {@code W}）</td></tr>
 *   <tr><th>{@link #LOW_RANK}</th><td>{@code r×d} <strong>非方阵</strong>（线性嵌入矩阵 {@code L}，{@code r&lt;d} 时常用）</td></tr>
 * </table>
 */
public enum MetricForm {
    /**
     * 对角情形：{@code A} 为 {@code d×d}，仅对角线可能非零；马氏精度 {@code AᵀA} 亦为对角。
     */
    DIAGONAL,
    /**
     * 满秩 whitening：{@code A = W} 为 {@code d×d} 可逆方阵，平方距离 {@code d²(x,y)=‖W(x−y)‖²}。
     */
    FULL_WHITENING,
    /**
     * 低秩嵌入：{@code A = L} 为 {@code r×d} 长方形矩阵（非方阵），
     * {@code d²(x,y)=‖L(x−y)‖²}，变换后坐标维为 {@code r}。
     */
    LOW_RANK
}
