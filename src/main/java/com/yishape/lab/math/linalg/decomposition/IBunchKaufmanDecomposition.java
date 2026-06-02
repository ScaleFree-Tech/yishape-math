package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;

/**
 * 对称不定矩阵的 Bunch–Kaufman 分解（L·D·Lᵀ，下三角存储），语义对齐 LAPACK {@code DSYTF2}/{@code DSYTRS}
 * 之 {@code UPLO='L'} 路径。
 */
public interface IBunchKaufmanDecomposition extends IMatrixDecomposition<IMatrix<Double>> {

    /**
     * 主元与块结构向量（与 {@link com.yishape.lab.math.linalg.decomposition.impl.BunchKaufmanLdltLower}
     * 中 {@code ipiv} 约定一致）。
     *
     * @return 副本；若尚未分解则为 {@code null}
     */
    int[] getIpiv();

    /**
     * 分解后的紧凑存储：下三角为 L 乘子及对角块 D（与 LAPACK 出站 {@code A} 一致），上三角未定义（当前实现中为 0）。
     */
    IMatrix<Double> getLdltFactor();
}
