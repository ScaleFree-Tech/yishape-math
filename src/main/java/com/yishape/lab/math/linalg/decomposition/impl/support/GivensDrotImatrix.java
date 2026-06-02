package com.yishape.lab.math.linalg.decomposition.impl.support;

import com.yishape.lab.math.linalg.IMatrix;

/**
 * Givens旋转矩阵应用工具类 / Givens Rotation Matrix Application Utility
 * <p>
 * Givens 旋转作用于 {@link IMatrix} 的行或列（与 LAPACK DROT 一致）。
 * Provides methods for applying Givens rotations to matrix rows and columns.
 * </p>
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public final class GivensDrotImatrix {

    private GivensDrotImatrix() {
    }

    /** 对列 j1, j2 做 BLAS {@code DROT}：{@code x := c*x + s*y}，{@code y := c*y - s*x}。 */
    public static void applyColumns(IMatrix<Double> m, int j1, int j2, double c, double s) {
        int rows = m.rows();
        for (int i = 0; i < rows; i++) {
            double x = m.get(i, j1);
            double y = m.get(i, j2);
            m.set(i, j1, c * x + s * y);
            m.set(i, j2, c * y - s * x);
        }
    }

    /** 对行 i1、i2 做 BLAS {@code DROT}（与 LAPACK 作用于 {@code VT} 行一致）。 */
    public static void applyRows(IMatrix<Double> m, int i1, int i2, double c, double s) {
        int cols = m.cols();
        for (int j = 0; j < cols; j++) {
            double x = m.get(i1, j);
            double y = m.get(i2, j);
            m.set(i1, j, c * x + s * y);
            m.set(i2, j, c * y - s * x);
        }
    }

    /**
     * 对 1-based 存储的稠密阵（忽略第 0 行/列）的两列 colA、colB 做 DROT；行 1..rowCount。
     */
    public static void applyColumns1Based(double[][] a, int rowCount, int colA, int colB, double c, double s) {
        for (int i = 1; i <= rowCount; i++) {
            double x = a[i][colA];
            double y = a[i][colB];
            a[i][colA] = c * x + s * y;
            a[i][colB] = c * y - s * x;
        }
    }

    /** 对行 rowA、rowB（1-based）、列 1..colCount 做 DROT */
    public static void applyRows1Based(double[][] a, int rowA, int rowB, int colCount, double c, double s) {
        for (int j = 1; j <= colCount; j++) {
            double x = a[rowA][j];
            double y = a[rowB][j];
            a[rowA][j] = c * x + s * y;
            a[rowB][j] = c * y - s * x;
        }
    }
}
