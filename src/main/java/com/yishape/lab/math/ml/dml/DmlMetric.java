package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.io.Serializable;
import java.util.Objects;

/**
 * 度量学习（或其他线性变换型度量）的<strong>拟合结果</strong>：封装左乘矩阵 {@code A} 与少量元数据，
 * 用法上对齐常见机器学习里 {@code fit → transform(X)} 与成对距离 API。
 *
 * <h2>线性模型</h2>
 * <p>对输入列向量 {@code x}（维数 {@link #inputDimension()}），变换与马氏型平方距离定义为：</p>
 * <pre>{@code z = A x,   d²(x,y) = ‖A(x−y)‖²}</pre>
 * <p>{@link #transformMatrix()} 返回的 {@code A} 形状为 {@code outputDimension() × inputDimension()}（行×列）。
 * 具体形状语义见 {@link MetricForm}。</p>
 *
 * <h2>构造方式</h2>
 * <ul>
 *   <li>{@link #diagonal} — 对角度量，仅对角线非零；</li>
 *   <li>{@link #fullWhitening} — 满秩方阵 whitener；</li>
 *   <li>{@link #lowRank} — 低秩嵌入矩阵 {@code L}（可为 {@code r×d} 非方阵）。</li>
 * </ul>
 *
 * <p>三元组计数字段仅部分算法（如对角 DML）有意义，其它算法多为零。</p>
 *
 * @see MetricForm
 * @see ISupervisedDml
 */
public final class DmlMetric implements Serializable {

    private static final long serialVersionUID = 1L;

    private final MetricForm form;
    private final IMatrix<Double> transformMatrix;
    private final int tripletCount;
    private final int usedTriplets;

    private DmlMetric(MetricForm form, IMatrix<Double> transformMatrix, int tripletCount, int usedTriplets) {
        this.form = Objects.requireNonNull(form, "form");
        this.transformMatrix = Objects.requireNonNull(transformMatrix, "transformMatrix").copy();
        this.tripletCount = tripletCount;
        this.usedTriplets = usedTriplets;
    }

    /**
     * 构造对角度量：{@code A = diag(w)}，故 {@code d²(x,y) = Σⱼ wⱼ² (xⱼ−yⱼ)²}。
     *
     * @param diagonalWeights 与各特征维一一对应的缩放 {@code wⱼ}（非负为宜，DDML 等处可做开方后的权重）
     * @param tripletCount    训练时采样的三元组总数（无可填 0）
     * @param usedTriplets    实际参与目标/约束的三元组计数（无可填 0）
     */
    public static DmlMetric diagonal(IVector<Double> diagonalWeights, int tripletCount, int usedTriplets) {
        Objects.requireNonNull(diagonalWeights, "diagonalWeights");
        int d = diagonalWeights.length();
        double[] dd = new double[d];
        for (int j = 0; j < d; j++) {
            dd[j] = diagonalWeights.get(j);
        }
        IMatrix<Double> a = IMatrix.diag(dd);
        return new DmlMetric(MetricForm.DIAGONAL, a, tripletCount, usedTriplets);
    }

    /**
     * 满秩情形：{@code A = W} 为 {@code d×d} 方阵（Mahalanobis 型 whitener），
     * {@code d²(x,y)=‖W(x−y)‖²} 对应某 PSD 精度阵的平方根形式。
     *
     * @param whitenerDxD 可逆方阵 {@code W}，行列均为特征维 {@code d}
     * @throws IllegalArgumentException 非方阵
     */
    public static DmlMetric fullWhitening(IMatrix<Double> whitenerDxD) {
        Objects.requireNonNull(whitenerDxD, "whitenerDxD");
        if (whitenerDxD.getRowNum() != whitenerDxD.getColNum()) {
            throw new IllegalArgumentException("whitener 须为方阵");
        }
        return new DmlMetric(MetricForm.FULL_WHITENING, whitenerDxD, 0, 0);
    }

    /**
     * 低秩嵌入：{@code A = L} 为 {@code r×d}，通常 {@code r ≤ d}；
     * 等价于在 {@code r} 维空间中欧氏距离，在原始空间为马氏型：{@code d²(x,y)=‖L(x−y)‖²}。
     *
     * @param linearTransformRD 左乘矩阵 {@code L}，行为输出维 {@code r}，列为输入维 {@code d}
     */
    public static DmlMetric lowRank(IMatrix<Double> linearTransformRD) {
        Objects.requireNonNull(linearTransformRD, "linearTransformRD");
        return new DmlMetric(MetricForm.LOW_RANK, linearTransformRD, 0, 0);
    }

    /** @return 本度量采用的三类存储形式之一 */
    public MetricForm form() {
        return form;
    }

    /** @return 输入特征维数 {@code d}（{@code A} 的列数） */
    public int inputDimension() {
        return transformMatrix.getColNum();
    }

    /** @return 线性变换输出维（{@code A} 的行数；低秩时可为 {@code r&lt;d}） */
    public int outputDimension() {
        return transformMatrix.getRowNum();
    }

    /**
     * 左乘矩阵 {@code A} 的<strong>防御性拷贝</strong>（{@code outputDimension()×inputDimension()}）。
     *
     * @return 与内部存储独立的 {@code A}
     */
    public IMatrix<Double> transformMatrix() {
        return transformMatrix.copy();
    }

    /**
     * 人类可读的矩阵类型与形状摘要，用于日志或异常信息。
     *
     * @return 含 {@link MetricForm} 与维度的英文短语
     */
    public String describeTransformMatrix() {
        int r = outputDimension();
        int c = inputDimension();
        return switch (form) {
            case DIAGONAL -> String.format(
                    "%s: diagonal matrix %d×%d (A=diag(w), off-diagonal entries are zero)",
                    form, r, c);
            case FULL_WHITENING -> String.format("%s: whitener (square) %d×%d", form, r, c);
            case LOW_RANK -> String.format(
                    "%s: low-rank embedding matrix %d×%d (non-square, r=%d, d=%d)",
                    form, r, c, r, c);
        };
    }

    /**
     * 仅当 {@link #form()} 为 {@link MetricForm#DIAGONAL}：
     * 返回对角 Mahalanobis 精度矩阵 {@code M = AᵀA}，此处 {@code M} 为对角，元素 {@code wⱼ²}。
     *
     * @return {@code d×d} 对角矩阵
     * @throws IllegalStateException 非对角形式
     */
    public IMatrix<Double> precisionDiagonalMatrix() {
        if (form != MetricForm.DIAGONAL) {
            throw new IllegalStateException("仅 DIAGONAL 形式存在对角精度矩阵");
        }
        int d = transformMatrix.getColNum();
        double[] m = new double[d];
        for (int j = 0; j < d; j++) {
            double w = transformMatrix.get(j, j);
            m[j] = w * w;
        }
        return IMatrix.diag(m);
    }

    /**
     * 仅当 {@link #form()} 为 {@link MetricForm#DIAGONAL}：返回 {@code wⱼ = A[j,j]}。
     *
     * @return 长度 {@code d} 的向量
     * @throws IllegalStateException 非对角形式
     */
    public IVector<Double> diagonalWeights() {
        if (form != MetricForm.DIAGONAL) {
            throw new IllegalStateException("仅 DIAGONAL 形式提供 diagonalWeights");
        }
        int d = transformMatrix.getColNum();
        double[] w = new double[d];
        for (int j = 0; j < d; j++) {
            w[j] = transformMatrix.get(j, j);
        }
        return Linalg.vector(w);
    }

    /**
     * 训练管线中见到的三元组总数（如 DDML 采样规模）；非三元组算法恒为 0。
     *
     * @return 三元组条数上界统计
     */
    public int tripletCount() {
        return tripletCount;
    }

    /**
     * 实际计入优化目标或约束的三元组条数；非三元组算法恒为 0。
     *
     * @return 有效三元组计数
     */
    public int usedTriplets() {
        return usedTriplets;
    }

    /**
     * 在已学习度量下两点的平方距离 {@code ‖A(x−y)‖²}。
     *
     * @param x 列向量，长度须为 {@link #inputDimension()}
     * @param y 列向量，与 {@code x} 同长
     * @return 非负标量（对极小负数值漂移可钳为 0）
     */
    public double squaredDistance(IVector<Double> x, IVector<Double> y) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        if (x.length() != y.length() || x.length() != inputDimension()) {
            throw new IllegalArgumentException("向量长度须等于 inputDimension");
        }
        IVector<Double> d = x.sub(y);
        IVector<Double> z = transformMatrix.mmul(d);
        double s = z.innerProductValue(z);
        if (s < 0 && s > -1e-10) {
            return 0.0;
        }
        return s;
    }

    /**
     * 对样本矩阵逐行左乘 {@code A}：结果行为样本数、列为 {@link #outputDimension()}。
     *
     * @param features 列为 {@link #inputDimension()} 的矩阵
     * @return 变换后矩阵 {@code Z = X Aᵀ}
     */
    public IMatrix<Double> transform(IMatrix<Double> features) {
        Objects.requireNonNull(features, "features");
        if (features.getColNum() != inputDimension()) {
            throw new IllegalArgumentException("列数须等于 inputDimension");
        }
        // Z = X Aᵀ；与 IMatrix#multiplyByTransposeOf 约定一致
        return features.multiplyByTransposeOf(transformMatrix);
    }

    @Override
    public String toString() {
        return "DmlMetric{" + describeTransformMatrix() + "}";
    }
}
