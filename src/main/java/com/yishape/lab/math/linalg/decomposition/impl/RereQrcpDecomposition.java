package com.yishape.lab.math.linalg.decomposition.impl;

/**
 * 列主元 QR：每步在未消元列中显式重算尾块 Frobenius 范数并选主元（{@code O(n^2 m)} 最坏），
 * 与 Commons Math {@code RRQRDecomposition} 等价。若需 LAPACK DGEQP3 式部分列范数递推，请用
 * {@link RereQrcpDgeqp3Decomposition}。
 */public class RereQrcpDecomposition extends AbstractRereQrcpDecomposition {

    public RereQrcpDecomposition() {
        super();
    }

    public RereQrcpDecomposition(double threshold) {
        super(threshold);
    }

    public RereQrcpDecomposition(double epsilon, int maxIterations) {
        super(epsilon, maxIterations);
    }

    @Override
    protected void performHouseholderReflection(int minor, double[][] matrix) {
        double normSqMax = -1.0;
        int jMax = minor;
        for (int j = minor; j < matrix.length; j++) {
            final double[] col = matrix[j];
            double s = 0.0;
            for (int row = minor; row < col.length; row++) {
                final double c = col[row];
                s += c * c;
            }
            if (s > normSqMax) {
                normSqMax = s;
                jMax = j;
            }
        }
        swapColumns(minor, jMax, matrix, columnPivot);
        super.performHouseholderReflection(minor, matrix);
    }
}
