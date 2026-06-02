package com.yishape.lab.math.linalg.decomposition.impl;

/**
 * 列主元 QR，选列代价接近 LAPACK DGEQP3：维护部分列范数估计 {@code vn1} 与“需时精确”的 {@code vn2}，
 * 每步 Householder 后按 DLAQP2/DGEQP3 的递推式更新（失效时 {@code dnrm2} 重算尾段）。
 */
public class RereQrcpDgeqp3Decomposition extends AbstractRereQrcpDecomposition {

    /** Partial column norms (see LAPACK VN1). */
    private double[] vn1;
    /** Workspace for exact tail norms when recurrence inaccurate (see LAPACK VN2). */
    private double[] vn2;

    /** Relative trigger for falling back to explicit tail norm (scaled machine epsilon). */
    private final double normUpdateEpsilon;

    public RereQrcpDgeqp3Decomposition() {
        this(1e-12, Math.ulp(1.0) * 64.0);
    }

    public RereQrcpDgeqp3Decomposition(double threshold) {
        this(threshold, Math.ulp(1.0) * 64.0);
    }

    public RereQrcpDgeqp3Decomposition(double epsilon, int maxIterations) {
        super(epsilon, maxIterations);
        this.normUpdateEpsilon = Math.ulp(1.0) * 64.0;
    }

    public RereQrcpDgeqp3Decomposition(double threshold, double normRecurrenceEpsilon) {
        super(threshold);
        this.normUpdateEpsilon = normRecurrenceEpsilon;
    }

    public RereQrcpDgeqp3Decomposition(double epsilon, int maxIterations, double normRecurrenceEpsilon) {
        super(epsilon, maxIterations);
        this.normUpdateEpsilon = normRecurrenceEpsilon;
    }

    @Override
    protected void onBeforeQrcpSteps(double[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        vn1 = new double[n];
        vn2 = new double[n];
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            double[] col = matrix[j];
            for (int i = 0; i < m; i++) {
                double v = col[i];
                s += v * v;
            }
            double norm = Math.sqrt(s);
            vn1[j] = norm;
            vn2[j] = norm;
        }
    }

    @Override
    protected void performHouseholderReflection(int minor, double[][] matrix) {
        int n = matrix.length;
        double best = -1.0;
        int jMax = minor;
        for (int j = minor; j < n; j++) {
            double s = vn1[j];
            if (s > best) {
                best = s;
                jMax = j;
            }
        }
        swapColumns(minor, jMax, matrix, columnPivot);
        double t1 = vn1[minor];
        vn1[minor] = vn1[jMax];
        vn1[jMax] = t1;
        double t2 = vn2[minor];
        vn2[minor] = vn2[jMax];
        vn2[jMax] = t2;

        super.performHouseholderReflection(minor, matrix);
        updatePartialColumnNormsLapackStyle(minor, matrix);
    }

    /**
     * DGEQP3 式：对 {@code j > minor} 更新列范数估计 {@code vn1}/{@code vn2}。
     * 子类可按 LAPACK DLAQPS 思路对 {@code j} 做分块遍历以改善缓存局部性。
     */
    protected void updatePartialColumnNormsLapackStyle(int minor, double[][] matrix) {
        int n = matrix.length;
        for (int j = minor + 1; j < n; j++) {
            LAPACK_updateColumnNormEstimateForIndex(j, minor, matrix);
        }
    }

    /** 与 LAPACK DLAQ2/3 递推一致的单列更新；{@code j} 为列索引（qrt 行）。 */
    protected final void LAPACK_updateColumnNormEstimateForIndex(int j, int minor, double[][] matrix) {
        int m = matrix[0].length;
        if (vn1[j] == 0.0) {
            return;
        }
        double akj = Math.abs(matrix[j][minor]);
        double v1 = vn1[j];
        double v2b = vn2[j];
        if (v2b <= 0.0) {
            vn2[j] = dnrm2Tail(j, minor + 1, m - 1, matrix);
            vn1[j] = vn2[j];
            return;
        }
        double temp = akj / v1;
        temp = Math.max(0.0, (1.0 + temp) * (1.0 - temp));
        double ratio = v1 / v2b;
        double temp2 = temp * ratio * ratio;
        if (Math.abs(temp2 - 1.0) <= normUpdateEpsilon) {
            vn2[j] = dnrm2Tail(j, minor + 1, m - 1, matrix);
            vn1[j] = vn2[j];
        } else {
            vn1[j] = v1 * Math.sqrt(temp);
        }
    }

    protected static double dnrm2Tail(int col, int rowLo, int rowHi, double[][] qrt) {
        if (rowLo > rowHi) {
            return 0.0;
        }
        double s = 0.0;
        double[] c = qrt[col];
        for (int r = rowLo; r <= rowHi; r++) {
            double v = c[r];
            s += v * v;
        }
        return Math.sqrt(s);
    }
}
