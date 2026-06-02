package com.yishape.lab.math.linalg.decomposition;

/**
 * 列主元 QR 分解后构造 xGELSY 风格容忍最小二乘求解器时使用的秩阈值（对 {@code |Rᵢᵢ|} 的绝对判据），
 * 及 LAPACK RCOND 风格的相对判据 {@link #fromRelativeRcond}。
 */
public final class QrcpRankTolerance {

    private QrcpRankTolerance() {
    }

    /**
     * 已分解 QRCP 的 {@code R} 前 {@code min(m,n)} 个对角元上 {@code max |Rᵢᵢ|}。
     */
    public static double maxLeadingDiagonalAbs(IQrcpDecomposition qrcp, int m, int n) {
        int lim = Math.min(m, n);
        double maxAbs = 0.0;
        var r = qrcp.getR();
        for (int i = 0; i < lim; i++) {
            maxAbs = Math.max(maxAbs, Math.abs(r.get(i, i)));
        }
        return maxAbs;
    }

    /**
     * LAPACK RCOND 风格：{@code max(|Rᵢᵢ|) * rcond}，且不低于分解器 {@link IQrcpDecomposition#getEpsilon()}。
     *
     * @param rcond 必須满足 {@code 0 < rcond <= 1}
     */
    public static double fromRelativeRcond(IQrcpDecomposition qrcp, int m, int n, double rcond) {
        if (!(rcond > 0.0 && rcond <= 1.0)) {
            throw new IllegalArgumentException("rcond must be in (0, 1], got " + rcond);
        }
        double maxAbs = maxLeadingDiagonalAbs(qrcp, m, n);
        return Math.max(maxAbs * rcond, qrcp.getEpsilon());
    }

    /**
     * 由已分解的 QRCP 估计默认绝对阈值：{@code max(|Rᵢᵢ|) × ε_machine × max(1, max(m,n))}，
     * 且不低于分解器 {@link IQrcpDecomposition#getEpsilon()}。
     */
    public static double forLeastSquares(IQrcpDecomposition qrcp, int m, int n) {
        double maxAbs = maxLeadingDiagonalAbs(qrcp, m, n);
        double eps = Math.ulp(1.0);
        int dim = Math.max(1, Math.max(m, n));
        double scaled = maxAbs * eps * dim;
        return Math.max(scaled, qrcp.getEpsilon());
    }
}
