package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 验证双对角 QR-SVD 的约定：{@code B ≈ Q_left · diag(σ) · Q_right^T}（上双对角方阵），
 * {@code Q_left}、{@code Q_right} 正交；以及经双对角化后 {@code K ≈ (U_b Q_l) Σ (V_b Q_r)^T}。
 */
class BidiagonalQrSvdReconstructionTest {

    private static final Random RNG = new Random(20260503);
    private static final RereSVDDecomposition SVD = new RereSVDDecomposition(1e-14, 8000);
    private static final RereBidiagonalDecomposition BIDIAG = new RereBidiagonalDecomposition(1e-14, 8000);

    @Test
    void qrBidiagonalReconstructsRandomUpperBidiagonal() {
        for (int n = 1; n <= 48; n++) {
            for (int trial = 0; trial < 12; trial++) {
                double[][] data = randomUpperBidiagonal(RNG, n);
                IMatrix<Double> B = Linalg.matrix(data);
                Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qr = SVD.qrBidiagonalForTesting(B);
                assertReconstruction("n=" + n + " t=" + trial, B, qr.getFirst(), qr.getSecond(), qr.getThird());
                assertOrthogonal("Ql n=" + n + " t=" + trial, qr.getSecond(), 1e-10);
                assertOrthogonal("Qr n=" + n + " t=" + trial, qr.getThird(), 1e-10);
            }
        }
    }

    @Test
    void bidiagonalizeThenQrReconstructsOriginalDenseK() {
        for (int n = 3; n <= 24; n++) {
            for (int trial = 0; trial < 10; trial++) {
                IMatrix<Double> k = randomDense(RNG, n);
                Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bi = BIDIAG.decompose(k);
                IMatrix<Double> ub = bi.getFirst();
                IMatrix<Double> bbd = bi.getSecond();
                IMatrix<Double> vb = bi.getThird();
                Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qr = SVD.qrBidiagonalForTesting(bbd);
                IVector<Double> sigma = qr.getFirst();
                IMatrix<Double> ql = qr.getSecond();
                IMatrix<Double> qrMat = qr.getThird();
                IMatrix<Double> uK = ub.mmul(ql);
                IMatrix<Double> vK = vb.mmul(qrMat);
                assertReconstruction("pipeline n=" + n + " t=" + trial, k, sigma, uK, vK);
                assertOrthogonal("uK n=" + n + " t=" + trial, uK, 1e-9);
                assertOrthogonal("vK n=" + n + " t=" + trial, vK, 1e-9);
            }
        }
    }

    @Test
    void fixedSmallCasesMatchKnownStructure() {
        // n=2 显式上双对角
        double[][] b2 = {{3.0, 4.0}, {0.0, -5.0}};
        IMatrix<Double> B2 = Linalg.matrix(b2);
        Tuple3<IVector<Double>, IMatrix<Double>, IMatrix<Double>> qr2 = SVD.qrBidiagonalForTesting(B2);
        assertReconstruction("2x2", B2, qr2.getFirst(), qr2.getSecond(), qr2.getThird());
    }

    private static void assertReconstruction(
            String msg,
            IMatrix<Double> expected,
            IVector<Double> sigma,
            IMatrix<Double> qLeft,
            IMatrix<Double> qRight) {
        int n = sigma.length();
        IMatrix<Double> sMat = Linalg.zeros(n, n);
        for (int i = 0; i < n; i++) {
            sMat.set(i, i, sigma.get(i));
        }
        IMatrix<Double> recon = qLeft.mmul(sMat).mmul(qRight.transposeNew());
        double scale = 0.0;
        for (int i = 0; i < expected.rows(); i++) {
            for (int j = 0; j < expected.cols(); j++) {
                scale = Math.max(scale, Math.abs(expected.get(i, j)));
            }
        }
        double tol = Math.max(1e-9, 5e-11 * (1.0 + Math.max(1.0, scale)) * n);
        double err = maxAbsDiff(expected, recon);
        assertTrue(err <= tol, () -> msg + " recon max abs err=" + err + " tol=" + tol);
    }

    private static double maxAbsDiff(IMatrix<Double> a, IMatrix<Double> b) {
        double m = 0.0;
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                m = Math.max(m, Math.abs(a.get(i, j) - b.get(i, j)));
            }
        }
        return m;
    }

    private static void assertOrthogonal(String msg, IMatrix<Double> q, double tol) {
        int n = q.cols();
        IMatrix<Double> id = q.transposeNew().mmul(q);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double e = (i == j) ? 1.0 : 0.0;
                double a = id.get(i, j);
                final int fi = i;
                final int fj = j;
                final double diff = a - e;
                assertTrue(Math.abs(diff) <= tol,
                        () -> msg + " Q^T Q off/ diag err at (" + fi + "," + fj + "): " + diff);
            }
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

    private static IMatrix<Double> randomDense(Random rnd, int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextGaussian();
            }
        }
        return Linalg.matrix(a);
    }
}
