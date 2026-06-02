package com.yishape.lab.math.compute.hpc;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.NonPositiveDefiniteMatrixException;
import com.yishape.lab.math.linalg.decomposition.NonSymmetricMatrixException;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

import com.yishape.lab.util.YishapeLogger;

import java.util.Arrays;

/**
 * LU / Cholesky / SVD / Eigen / QR / Inverse / Multi-RHS Solve 可选委托 yishape-math-hpc（经 {@link HpcOptionalRuntime} 反射，无扩展 JAR 时不加载）。
 */
public final class HpcLapackDecomps {

    private static final YishapeLogger log = YishapeLogger.getLogger(HpcLapackDecomps.class);

    private HpcLapackDecomps() {
    }

    private static boolean useCholeskyNative(IMatrix<Double> matrix) {
        if (!HpcConfig.allowAttempts() || matrix == null) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int n = matrix.rows();
        if (n <= 0 || n != matrix.cols()) {
            return false;
        }
        return n >= HpcConfig.choleskyMinDim();
    }

    private static boolean useSvdNative(IMatrix<Double> matrix) {
        if (!HpcConfig.allowAttempts() || matrix == null) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int r = matrix.rows();
        int c = matrix.cols();
        if (r <= 0 || c <= 0) {
            return false;
        }
        return (long) r * c >= HpcConfig.svdMinTotalElements();
    }

    private static boolean useEigenNative(IMatrix<Double> matrix) {
        if (!HpcConfig.allowAttempts() || matrix == null) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int n = matrix.rows();
        if (n <= 0 || n != matrix.cols()) {
            return false;
        }
        return n >= HpcConfig.eigenMinDim();
    }

    private static boolean useQrNative(IMatrix<Double> matrix) {
        if (!HpcConfig.allowAttempts() || matrix == null) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int r = matrix.rows();
        int c = matrix.cols();
        if (r <= 0 || c <= 0) {
            return false;
        }
        return (long) r * c >= HpcConfig.qrMinTotalElements();
    }

    private static boolean useLuNative(IMatrix<Double> matrix) {
        if (!HpcConfig.allowAttempts() || matrix == null) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int n = matrix.rows();
        if (n <= 0 || n != matrix.cols()) {
            return false;
        }
        return n >= HpcConfig.luMinDim();
    }

    private static boolean useInverseNative(IMatrix<Double> matrix) {
        if (!HpcConfig.allowAttempts() || matrix == null) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int n = matrix.rows();
        if (n <= 0 || n != matrix.cols()) {
            return false;
        }
        return n >= HpcConfig.inverseMinDim();
    }

    private static boolean rectMatrixOk(double[][] a, int rows, int cols) {
        if (a == null || a.length != rows) {
            return false;
        }
        for (int i = 0; i < rows; i++) {
            if (a[i] == null || a[i].length != cols) {
                return false;
            }
        }
        return true;
    }

    private static void assertSymmetricForCholesky(double[][] a, int n) {
        double rel = 1e-12;
        double maxAsym = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double asym = Math.abs(a[i][j] - a[j][i]);
                maxAsym = Math.max(maxAsym, asym);
                double scale = rel * Math.max(1.0, Math.max(Math.abs(a[i][j]), Math.abs(a[j][i])));
                if (asym > scale) {
                    throw new NonSymmetricMatrixException(
                            "Matrix must be symmetric for Cholesky decomposition",
                            "Cholesky Decomposition",
                            "Matrix " + n + "x" + n,
                            rel,
                            maxAsym);
                }
            }
        }
    }

    /**
     * 稠密方阵多右端项之一列：{@code YishapeHpc.solveSquare}；失败或未达规模时 {@code null}。
     */
    public static IMatrix<Double> trySolveSquareRhs(IMatrix<Double> matrix, IMatrix<Double> rhsColumn) {
        if (!HpcConfig.allowAttempts() || matrix == null || rhsColumn == null) {
            return null;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return null;
        }
        int n = matrix.rows();
        if (n != matrix.cols() || rhsColumn.rows() != n || rhsColumn.cols() != 1) {
            return null;
        }
        if (n < HpcConfig.solveMinDim()) {
            return null;
        }
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, n, n)) {
                return null;
            }
            double[] b0 = new double[n];
            for (int i = 0; i < n; i++) {
                b0[i] = rhsColumn.get(i, 0);
            }
            HpcOptionalRuntime.RSolveSquare r = HpcOptionalRuntime.solveSquare(a0, b0);
            if (r == null || !r.ok() || r.x() == null || r.x().length != n) {
                return null;
            }
            double[][] xCol = new double[n][1];
            for (int i = 0; i < n; i++) {
                xCol[i][0] = r.x()[i];
            }
            return Linalg.matrix(xCol);
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * 多右端项求解 {@code AX=B}（HPC 路径）。
     */
    public static IMatrix<Double> trySolveMultiRhs(IMatrix<Double> a, IMatrix<Double> b) {
        if (!HpcConfig.allowAttempts() || a == null || b == null) {
            return null;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return null;
        }
        int n = a.rows();
        if (n != a.cols() || b.rows() != n || b.cols() < 1) {
            return null;
        }
        if (n < HpcConfig.solveMinDim()) {
            return null;
        }
        try {
            double[][] a0 = a.toDoubleArray();
            double[][] b0 = b.toDoubleArray();
            if (!rectMatrixOk(a0, n, n) || !rectMatrixOk(b0, n, b.cols())) {
                return null;
            }
            HpcOptionalRuntime.RSolveMultiRhs r = HpcOptionalRuntime.solveMultiRhs(a0, b0);
            if (r == null || !r.ok() || r.x() == null) {
                return null;
            }
            return Linalg.matrix(r.x());
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * 矩阵求逆 HPC 路径。
     */
    public static IMatrix<Double> tryInverse(IMatrix<Double> matrix) {
        if (!useInverseNative(matrix)) {
            return null;
        }
        int n = matrix.rows();
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, n, n)) {
                return null;
            }
            HpcOptionalRuntime.RInverse r = HpcOptionalRuntime.inverse(a0);
            if (r == null || !r.ok() || r.inv() == null) {
                return null;
            }
            return Linalg.matrix(r.inv());
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * 对称矩阵特征值分解 HPC 路径。
     * <p>返回特征值（非降序）和特征向量矩阵。</p>
     */
    public static Tuple2<IVector<Double>, IMatrix<Double>> tryEigenSymmetric(IMatrix<Double> matrix) {
        if (!useEigenNative(matrix)) {
            return null;
        }
        int n = matrix.rows();
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, n, n)) {
                return null;
            }
            // 仅对接近对称的矩阵启用 HPC 路径（避免非对称矩阵走错误路径）
            double maxAsym = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    maxAsym = Math.max(maxAsym, Math.abs(a0[i][j] - a0[j][i]));
                }
            }
            if (maxAsym > 1e-8) {
                return null; // 明显非对称，回退 Java
            }
            HpcOptionalRuntime.REigenSymmetric r = HpcOptionalRuntime.eigenSymmetric(a0);
            if (r == null || !r.ok() || r.eigenvalues() == null || r.eigenvectors() == null) {
                return null;
            }
            // faer 返回非降序，Java 文档要求降序（从大到小），反转
            double[] w = r.eigenvalues();
            double[][] vecs = r.eigenvectors();
            int len = w.length;
            double[] wDesc = new double[len];
            double[][] vecsDesc = new double[n][n];
            for (int i = 0; i < len; i++) {
                wDesc[i] = w[len - 1 - i];
            }
            for (int j = 0; j < len; j++) {
                int srcJ = len - 1 - j;
                for (int i = 0; i < n; i++) {
                    vecsDesc[i][j] = vecs[i][srcJ];
                }
            }
            return new Tuple2<>(IVector.of(wDesc), Linalg.matrix(vecsDesc));
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * 非对称矩阵特征值分解 HPC 路径。
     * <p>返回实特征值（对复特征对取实部，与纯 Java 实现语义一致），排序为降序。</p>
     */
    public static Tuple2<IVector<Double>, IMatrix<Double>> tryEigenNonsymmetric(IMatrix<Double> matrix) {
        if (!HpcConfig.allowAttempts() || matrix == null) {
            return null;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return null;
        }
        int n = matrix.rows();
        if (n <= 0 || n != matrix.cols()) {
            return null;
        }
        if (n < HpcConfig.eigenNonsymmetricMinDim()) {
            return null;
        }
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, n, n)) {
                return null;
            }
            HpcOptionalRuntime.REigenNonsymmetric r = HpcOptionalRuntime.eigenNonsymmetric(a0);
            if (r == null || !r.ok()) {
                return null;
            }
            double[] er = r.eigenvaluesReal();
            double[][] vr = r.eigenvectorsReal();
            // 排序为降序（与 Java 实现语义一致）
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            java.util.Arrays.sort(idx, (i, j) -> Double.compare(er[j], er[i]));
            double[] wDesc = new double[n];
            double[][] vecsDesc = new double[n][n];
            for (int i = 0; i < n; i++) {
                wDesc[i] = er[idx[i]];
                for (int j = 0; j < n; j++) {
                    vecsDesc[j][i] = vr[j][idx[i]];
                }
            }
            return new Tuple2<>(IVector.of(wDesc), Linalg.matrix(vecsDesc));
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * QR 分解 HPC 路径。
     */
    public static Tuple2<IMatrix<Double>, IMatrix<Double>> tryQr(IMatrix<Double> matrix) {
        if (!useQrNative(matrix)) {
            return null;
        }
        int m = matrix.rows();
        int n = matrix.cols();
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, m, n)) {
                return null;
            }
            HpcOptionalRuntime.RQr r = HpcOptionalRuntime.qr(a0);
            if (r == null || !r.ok() || r.q() == null || r.r() == null) {
                return null;
            }
            return new Tuple2<>(Linalg.matrix(r.q()), Linalg.matrix(r.r()));
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * LU 分解 HPC 路径。
     */
    public static Tuple2<IMatrix<Double>, IMatrix<Double>> tryLu(IMatrix<Double> matrix) {
        if (!useLuNative(matrix)) {
            return null;
        }
        int n = matrix.rows();
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, n, n)) {
                return null;
            }
            HpcOptionalRuntime.RLu r = HpcOptionalRuntime.lu(a0);
            if (r == null || !r.ok() || r.l() == null || r.u() == null) {
                return null;
            }
            return new Tuple2<>(Linalg.matrix(r.l()), Linalg.matrix(r.u()));
        } catch (Throwable t) {
            return null;
        }
    }

    public static IMatrix<Double> tryCholeskyL(IMatrix<Double> matrix) {
        if (!useCholeskyNative(matrix)) {
            return null;
        }
        int n = matrix.rows();
        if (n != matrix.cols()) {
            return null;
        }
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, n, n)) {
                return null;
            }
            assertSymmetricForCholesky(a0, n);

            HpcOptionalRuntime.RCholesky r = HpcOptionalRuntime.cholesky(a0);
            if (r == null) {
                return null;
            }
            if (r.status() == HpcAbiCodes.NOT_POSITIVE_DEFINITE) {
                throw new NonPositiveDefiniteMatrixException(
                        "Matrix is not positive definite for Cholesky decomposition",
                        "Cholesky Decomposition",
                        "Matrix " + n + "x" + n,
                        0.0,
                        0.0);
            }
            if (!r.ok() || r.lLower() == null) {
                return null;
            }
            return Linalg.matrix(r.lLower());
        } catch (NonSymmetricMatrixException | NonPositiveDefiniteMatrixException e) {
            throw e;
        } catch (Throwable t) {
            return null;
        }
    }

    public static Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> trySvd(IMatrix<Double> matrix) {
        if (!useSvdNative(matrix)) {
            log.info("[HpcLapack] trySvd: useSvdNative returned false, falling back to Java");
            return null;
        }
        int m = matrix.rows();
        int n = matrix.cols();
        int k = Math.min(m, n);
        long uvt = (long) m * k + (long) n * n;
        if (uvt > HpcConfig.svdMaxUPlusVtElements()) {
            log.info("[HpcLapack] trySvd: uvt {} > max {} for matrix {}x{}, falling back to Java",
                    uvt, HpcConfig.svdMaxUPlusVtElements(), m, n);
            return null;
        }
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, m, n)) {
                log.info("[HpcLapack] trySvd: rectMatrixOk failed for {}x{}, falling back to Java", m, n);
                return null;
            }
            HpcOptionalRuntime.RSvd r = HpcOptionalRuntime.svd(a0);
            if (r == null || !r.ok() || r.u() == null || r.singularValues() == null || r.vt() == null) {
                log.info("[HpcLapack] trySvd: Rust SVD returned invalid result, falling back to Java. r={}, ok={}", r, r != null ? r.ok() : "null");
                return null;
            }
            log.info("[HpcLapack] trySvd: SUCCESS using Rust for {}x{}", m, n);
            return new Tuple3<>(
                    Linalg.matrix(r.u()),
                    IVector.of(Arrays.copyOf(r.singularValues(), k)),
                    Linalg.matrix(r.vt()));
        } catch (Throwable t) {
            log.info("[HpcLapack] trySvd: Exception calling Rust SVD, falling back to Java: {}", t.getMessage());
            return null;
        }
    }

    /**
     * 行列式 HPC 路径（借助 HPC LU 分解，含置换符号修正）。
     * <p>det(A) = det(P) · det(U)，其中 P 为置换阵，U 为上三角。</p>
     */
    public static Double tryDet(IMatrix<Double> matrix) {
        if (!useLuNative(matrix)) {
            return null;
        }
        int n = matrix.rows();
        try {
            double[][] a0 = matrix.toDoubleArray();
            if (!rectMatrixOk(a0, n, n)) {
                return null;
            }
            HpcOptionalRuntime.RLu r = HpcOptionalRuntime.lu(a0);
            if (r == null || !r.ok() || r.u() == null || r.p() == null) {
                return null;
            }
            double det = 1.0;
            double[][] u = r.u();
            for (int i = 0; i < n; i++) {
                det *= u[i][i];
            }
            // 置换符号：计算置换数组中的环数 → 交换次数
            int[] p = r.p();
            boolean[] seen = new boolean[n];
            int swaps = 0;
            for (int i = 0; i < n; i++) {
                if (!seen[i]) {
                    int j = i;
                    int cycleLen = 0;
                    while (!seen[j]) {
                        seen[j] = true;
                        j = p[j];
                        cycleLen++;
                    }
                    swaps += cycleLen - 1;
                }
            }
            if (swaps % 2 == 1) {
                det = -det;
            }
            return det;
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * 条件数 HPC 路径（借助 HPC SVD 计算 κ₂ = σ₁/σᵣ）。
     */
    public static Double tryCond(IMatrix<Double> matrix) {
        var svdResult = trySvd(matrix);
        if (svdResult == null) return null;
        IVector<Double> s = svdResult._2;
        double max = s.get(0);
        double min = s.get(s.length() - 1);
        if (min < 1e-15) return Double.POSITIVE_INFINITY;
        return max / min;
    }

    /**
     * 矩阵秩 HPC 路径（借助 HPC SVD 计算数值秩）。
     */
    public static Integer tryRank(IMatrix<Double> matrix) {
        var svdResult = trySvd(matrix);
        if (svdResult == null) return null;
        IVector<Double> s = svdResult._2;
        double max = s.get(0);
        int m = matrix.rows();
        int n = matrix.cols();
        double tol = Math.max(m, n) * Math.max(max, 1.0) * 1e-12;
        int rank = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.get(i) > tol) rank++;
        }
        return rank;
    }
}
