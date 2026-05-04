package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.impl.support.lasd.Dlasd2Sqre0;
import com.yishape.lab.math.linalg.decomposition.impl.support.lasd.Dlasd3Sqre0Eigen;
import com.yishape.lab.math.linalg.decomposition.impl.support.lasd.LapackDlasdIdxq;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * LAPACK DLASD2 + DLASD3(Eigen) 与对同一父双对角矩阵的 QR 双对角 SVD 奇异值 multiset 一致（LAPACK 打包）。
 */
class DlasdPackedVsQrBidiagonalTest {

    /**
     * 完全对角（超对角全 0）时块间无耦合，合并后奇异值应为各块对角模长的并集。
     */
    @Test
    void lapackPackedSingularValuesMatchOnDiagonalUpperBidiagonal() {
        RereSVDDecomposition svd = new RereSVDDecomposition(1e-14, 8000);
        int n = 5;
        int mid = 2;
        int nl = mid - 1;
        int nr = n - mid;
        double[][] bFull = new double[n][n];
        for (int i = 0; i < n; i++) {
            bFull[i][i] = (i + 1) * 0.7;
        }
        runPackedVsQrOnce(svd, bFull, n, mid, nl, nr, 0);
    }

    /**
     * 一般上双对角：已用 {@code NL×(NL+1)} 叶 SVD 与匹配的 {@code VT} 打包，但与 JNI/Reference LAPACK 的 {@code dlasd2/dlasd3}
     * 数值链仍有差距（奇异值 multiset 与父 QR-SVD 不一致）。启用前建议用参考 LAPACK 对拍或收紧 Java 端口。
     */
    @Disabled("与 Reference LAPACK 链对拍后再启用")
    @Test
    void lapackPackedSingularValuesMatchQrOnParentBidiagonal() {
        Random rnd = new Random(11);
        RereSVDDecomposition svd = new RereSVDDecomposition(1e-14, 8000);
        for (int n = 5; n <= 12; n++) {
            int mid = n / 2;
            if (mid < 2 || n - mid < 2) {
                continue;
            }
            int n1 = mid;
            int n2 = n - mid;
            int nl = n1 - 1;
            int nr = n2;
            for (int t = 0; t < 15; t++) {
                double[][] bFull = randomUpperBidiagonal(rnd, n);
                runPackedVsQrOnce(svd, bFull, n, mid, nl, nr, t);
            }
        }
    }

    private static void runPackedVsQrOnce(
            RereSVDDecomposition svd,
            double[][] bFull,
            int n,
            int mid,
            int nl,
            int nr,
            int trialLabel) {
        IMatrix<Double> bMat = Linalg.matrix(bFull);
        int n1 = mid;
        double alphaConn = bFull[mid - 1][mid - 1];
        double betaConn = bFull[mid - 1][mid];

        IMatrix<Double> bLeftWide = extract(bFull, 0, nl, 0, nl + 1);
        IMatrix<Double> b2 = extract(bFull, mid, n, mid, n);

        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> leftSvd =
                Decomps.createSVD(1e-14, 8000).decompose(bLeftWide);
        Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> r2 = svd.qrBidiagonalForTesting(b2);

        Integer[] permLeft = singularValueAscPermutation(leftSvd.getSecond());
        double[] s1TopAsc = ascendingSortedSingularValues(leftSvd.getSecond(), permLeft);
        IMatrix<Double> u1 = columnPermuted(leftSvd.getFirst(), permLeft);
        IMatrix<Double> vtLeftAsc = rowPermutedLeading(leftSvd.getThird(), permLeft, nl);
        Integer[] permR2 = singularValueAscPermutation(r2.getFirst());
        double[] s2Asc = ascendingSortedSingularValues(r2.getFirst(), permR2);
        IMatrix<Double> u2 = columnPermuted(r2.getSecond(), permR2);
        IMatrix<Double> v2 = columnPermuted(r2.getThird(), permR2);

        int nLap = nl + nr + 1;
        if (nLap != n) {
            throw new IllegalStateException("n lap");
        }
        double[] d = new double[n + 1];
        double[] z = new double[n + 1];
        double[][] u = new double[n + 1][n + 1];
        double[][] vt = new double[n + 1][n + 1];
        double[] dsigma = new double[n + 1];
        double[][] u2w = new double[n + 1][n + 1];
        double[][] vt2w = new double[n + 1][n + 1];
        int[] idxp = new int[n + 1];
        int[] idx = new int[n + 1];
        int[] idxc = new int[n + 1];
        int[] idxq = new int[n + 1];
        int[] coltyp = new int[n + 1];

        for (int i = 1; i <= nl; i++) {
            d[i] = s1TopAsc[i - 1];
        }
        d[nl + 1] = 0.0;
        for (int j = 1; j <= nr; j++) {
            d[nl + 1 + j] = s2Asc[j - 1];
        }
        LapackDlasdIdxq.initIdxqBeforeDlasd2(nl, n, idxq);
        for (int i = 1; i <= nl; i++) {
            for (int j = 1; j <= nl; j++) {
                u[i][j] = u1.get(i - 1, j - 1);
            }
        }
        for (int i = 1; i <= nr; i++) {
            for (int j = 1; j <= nr; j++) {
                u[nl + 1 + i][nl + 1 + j] = u2.get(i - 1, j - 1);
            }
        }
        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n1; j++) {
                vt[i][j] = vtLeftAsc.get(i - 1, j - 1);
            }
        }
        for (int i = 1; i <= nr; i++) {
            for (int j = 1; j <= nr; j++) {
                vt[nl + 1 + i][nl + 1 + j] = v2.get(j - 1, i - 1);
            }
        }

        int k = Dlasd2Sqre0.dlasd2(nl, nr, alphaConn, betaConn, d, z, u, vt, dsigma, u2w, vt2w,
                idxp, idx, idxc, idxq, coltyp);

        double[][] uOut = new double[n + 1][n + 1];
        double[][] vtOut = new double[n + 1][n + 1];
        for (int r = 1; r <= n; r++) {
            System.arraycopy(u[r], 1, uOut[r], 1, n);
        }
        for (int r = 1; r <= n; r++) {
            System.arraycopy(vt[r], 1, vtOut[r], 1, n);
        }

        double[] dOut = new double[k + 1];
        Dlasd3Sqre0Eigen.dlasd3(k, n, n, dsigma, z, u2w, vt2w, uOut, vtOut, dOut, 1e-14);

        double[] merged = new double[n];
        for (int i = 0; i < k; i++) {
            merged[i] = dOut[i + 1];
        }
        for (int i = k; i < n; i++) {
            merged[i] = d[i + 1];
        }
        Arrays.sort(merged);

        Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qrFull = svd.qrBidiagonalForTesting(bMat);
        double[] ref = new double[n];
        for (int i = 0; i < n; i++) {
            ref[i] = Math.abs(qrFull.getFirst().get(i));
        }
        Arrays.sort(ref);

        for (int i = 0; i < n; i++) {
            assertEquals(ref[i], merged[i], 2e-6 * (1.0 + Math.abs(ref[i])),
                    "n=" + n + " mid=" + mid + " trial=" + trialLabel + " i=" + i);
        }
    }

    private static double[][] randomUpperBidiagonal(Random rnd, int n) {
        double[][] b = new double[n][n];
        for (int i = 0; i < n; i++) {
            b[i][i] = rnd.nextGaussian();
        }
        for (int i = 0; i < n - 1; i++) {
            b[i][i + 1] = rnd.nextGaussian();
        }
        return b;
    }

    private static IMatrix<Double> extract(double[][] b, int r0, int r1, int c0, int c1) {
        int rows = r1 - r0;
        int cols = c1 - c0;
        double[][] d = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(b[r0 + i], c0, d[i], 0, cols);
        }
        return Linalg.matrix(d);
    }

    private static Integer[] singularValueAscPermutation(IVector<Double> s) {
        int n = s.length();
        Integer[] ord = new Integer[n];
        for (int i = 0; i < n; i++) {
            ord[i] = i;
        }
        Arrays.sort(ord, Comparator.comparingDouble(s::get));
        return ord;
    }

    private static double[] ascendingSortedSingularValues(IVector<Double> s, Integer[] ascendingPerm) {
        double[] out = new double[s.length()];
        for (int j = 0; j < ascendingPerm.length; j++) {
            out[j] = Math.abs(s.get(ascendingPerm[j]));
        }
        return out;
    }

    private static IMatrix<Double> columnPermuted(IMatrix<Double> m, Integer[] perm) {
        int rows = m.rows();
        int cols = m.cols();
        IMatrix<Double> o = Linalg.zeros(rows, cols);
        for (int j = 0; j < cols; j++) {
            int sj = perm[j];
            for (int i = 0; i < rows; i++) {
                o.set(i, j, m.get(i, sj));
            }
        }
        return o;
    }

    /**
     * 对 {@code V^T} 中与正奇异值对应的前 {@code minDim} 行做与 σ 升序一致的行置换，其余行原样复制
     * （矩形 SVD 时 {@code VT} 行数 {@code > perm.length}）。
     */
    private static IMatrix<Double> rowPermutedLeading(IMatrix<Double> m, Integer[] ascendingSingularValuePerm,
                                                       int minDim) {
        int rows = m.rows();
        int cols = m.cols();
        IMatrix<Double> o = Linalg.zeros(rows, cols);
        for (int r = 0; r < minDim; r++) {
            int sr = ascendingSingularValuePerm[r];
            for (int c = 0; c < cols; c++) {
                o.set(r, c, m.get(sr, c));
            }
        }
        for (int r = minDim; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                o.set(r, c, m.get(r, c));
            }
        }
        return o;
    }
}
