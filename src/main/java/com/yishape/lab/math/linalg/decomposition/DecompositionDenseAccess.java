package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;

/**
 * 稠密 double 矩阵访问辅助：在 {@link IDoubleMatrix} 上优先用 {@link IDoubleMatrix#getData()}
 * 做一次 {@link System#arraycopy}，避免先 {@link IMatrix#toDoubleArray()} 再逐行拷贝的重复分配。
 */
public final class DecompositionDenseAccess {

    private DecompositionDenseAccess() {
    }

    /**
     * 将 {@code matrix} 的前 {@code rows}×{@code cols} 块拷入已分配好的 {@code dest}（每行长度 ≥ cols）。
     */
    public static void copyInto(IMatrix<Double> matrix, double[][] dest, int rows, int cols) {
        if (matrix instanceof IDoubleMatrix dm) {
            double[][] src = dm.getData();
            for (int i = 0; i < rows; i++) {
                System.arraycopy(src[i], 0, dest[i], 0, cols);
            }
            return;
        }
        double[][] tmp = matrix.toDoubleArray();
        for (int i = 0; i < rows; i++) {
            System.arraycopy(tmp[i], 0, dest[i], 0, cols);
        }
    }

    /**
     * 对 {@link IDoubleMatrix} 返回与矩阵共享的只读视角（调用方不得写入）；
     * 否则返回 {@link IMatrix#toDoubleArray()}（独立数组，可任意写入）。
     */
    public static double[][] rowsForReadOnly(IMatrix<Double> matrix) {
        if (matrix instanceof IDoubleMatrix dm) {
            return dm.getData();
        }
        return matrix.toDoubleArray();
    }
}
