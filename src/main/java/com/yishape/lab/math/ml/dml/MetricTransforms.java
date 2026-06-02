package com.yishape.lab.math.ml.dml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;

import java.util.Objects;

/**
 * 度量相关的<strong>辅助变换</strong>：从精度矩阵构造 whitener、Mahalanobis / 低秩 / 对角距离，
 * 以及批量的行变换（与 {@link DmlMetric#transform} 同源的矩阵乘写法）。
 *
 * <p>应用侧优先使用已拟合的 {@link DmlMetric#transform(IMatrix)} 与
 * {@link DmlMetric#squaredDistance}；本类适用于 whitener 构造、一次性打分或未封装为
 * {@link DmlMetric} 的中间结果。</p>
 *
 * @see DmlMetric
 * @see IMatrix#multiplyByTransposeOf
 */
public final class MetricTransforms {

    private MetricTransforms() {
    }

    /**
     * 由对称（近似）PSD 精度矩阵 {@code M} 求得 whitener {@code W}，使
     * {@code ‖W v‖² ≈ vᵀ M v}（数值上经 Cholesky 等；失败时可抬升最小特征值后重试）。
     *
     * @param precision Mahalanobis 精度矩阵 {@code M}，须可对称为 PSD
     * @return 方阵 {@code W}，满足 {@link DmlMetric#fullWhitening} 所用形状
     */
    public static IMatrix<Double> whitenerFromPrecision(IMatrix<Double> precision) {
        Objects.requireNonNull(precision, "precision");
        IMatrix<Double> M = symmetrize(precision);
        return whitenerFromSymmetricPsdAttempt(M, false);
    }

    /**
     * 向量差分在 whitener 下的平方 Mahalanobis 距离。
     *
     * @param x                 列向量
     * @param y                 列向量，与 {@code x} 同维
     * @param whitenerForDiff   {@code d×d}，列数等于向量长度
     * @return 非负标量距离平方
     */
    public static double squaredMahalanobis(IVector<Double> x, IVector<Double> y,
            IMatrix<Double> whitenerForDiff) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(whitenerForDiff, "whitenerForDiff");
        if (x.length() != y.length() || x.length() != whitenerForDiff.getColNum()) {
            throw new IllegalArgumentException("维数与 whitener 列数须与向量长度一致");
        }
        IVector<Double> d = x.sub(y);
        IVector<Double> z = whitenerForDiff.mmul(d);
        double s = z.innerProductValue(z);
        if (s < 0 && s > -1e-10) {
            return 0.0;
        }
        return s;
    }

    /**
     * 低秩线性映射：{@code d²(x,y) = ‖L(x−y)‖²}，{@code L} 为 {@code r×d}。
     */
    public static double squaredLowRank(IVector<Double> x, IVector<Double> y, IMatrix<Double> L) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(L, "L");
        if (x.length() != y.length() || x.length() != L.getColNum()) {
            throw new IllegalArgumentException("维数须与 L 列数一致");
        }
        IVector<Double> d = x.sub(y);
        IVector<Double> z = L.mmul(d);
        return z.innerProductValue(z);
    }

    /**
     * 对角度量下的平方距离：{@code Σⱼ scaling[j]² (xⱼ−yⱼ)²}。
     *
     * @see IVector#diagonalWeightedSquaredDistanceTo
     */
    public static double squaredDiagonal(IVector<Double> x, IVector<Double> y, IVector<Double> scaling) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        Objects.requireNonNull(scaling, "scaling");
        return x.diagonalWeightedSquaredDistanceToValue(y, scaling);
    }

    /**
     * 对各行右乘对角等价于列方向逐元乘以 {@code scaling}：{@code out[i,j]=x[i,j]*scaling[j]}。
     *
     * @param x       行样本矩阵
     * @param scaling 长度等于 {@code x} 列数
     * @return 新矩阵
     */
    public static IMatrix<Double> transformRowsDiagonal(IMatrix<Double> x, IVector<Double> scaling) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(scaling, "scaling");
        if (x.getColNum() != scaling.length()) {
            throw new IllegalArgumentException("列数须等于 scaling 长度");
        }
        // 等价于 X·diag(scaling)；按列 broadcast
        return x.broadcastMultiplyColumn(scaling);
    }

    /**
     * 批量 whitener：每行 {@code xᵢ} 映射为 {@code W xᵢ}，等价于 {@code X Wᵀ}。
     */
    public static IMatrix<Double> transformRowsWhitener(IMatrix<Double> x, IMatrix<Double> whitenerForDiff) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(whitenerForDiff, "whitenerForDiff");
        if (x.getColNum() != whitenerForDiff.getColNum()) {
            throw new IllegalArgumentException("特征列数须等于 whitener 列数");
        }
        return x.multiplyByTransposeOf(whitenerForDiff);
    }

    /**
     * 批量低秩：{@code X Lᵀ}。
     */
    public static IMatrix<Double> transformRowsLowRank(IMatrix<Double> x, IMatrix<Double> L) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(L, "L");
        if (x.getColNum() != L.getColNum()) {
            throw new IllegalArgumentException("特征列数须等于 L 列数");
        }
        return x.multiplyByTransposeOf(L);
    }

    /**
     * 对称化 {@code (M+Mᵀ)/2}，供特征分解/Cholesky 前消除数值非对称噪音。
     */
    public static IMatrix<Double> symmetrize(IMatrix<Double> m) {
        IMatrix<Double> t = m.transpose();
        return m.add(t).multiplyByScalar(0.5);
    }

    private static IMatrix<Double> whitenerFromSymmetricPsdAttempt(IMatrix<Double> m, boolean retried) {
        try {
            IMatrix<Double> L = m.cholesky();
            return L.transpose();
        } catch (RuntimeException e) {
            if (retried) {
                throw new IllegalArgumentException("无法从精度矩阵构造 whitener（数值病态）", e);
            }
            IMatrix<Double> fixed = clampNegativeEigenvalues(m, 1e-10);
            return whitenerFromSymmetricPsdAttempt(fixed, true);
        }
    }

    private static IMatrix<Double> clampNegativeEigenvalues(IMatrix<Double> symmetric, double floor) {
        Tuple2<IVector<Double>, IMatrix<Double>> ev = symmetric.eigen();
        IVector<Double> lam = ev._1;
        IMatrix<Double> v = ev._2;
        int d = lam.length();
        double[] dd = new double[d];
        for (int i = 0; i < d; i++) {
            double l = lam.get(i);
            dd[i] = l < floor ? floor : l;
        }
        IMatrix<Double> diag = IMatrix.diag(dd);
        return v.mmul(diag).mmul(v.transpose());
    }
}
