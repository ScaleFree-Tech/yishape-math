package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.util.RerePrecision;
import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 验证大规模分支：将阈值临时调低后，CGLS（最小二乘共轭梯度）与 QR 参考解一致。
 */
class LeastSquaresLargeScaleCgTest {

    @AfterEach
    void tearDown() {
        LeastSquaresSolver.resetHugeScaleThresholds();
    }

    @Test
    void cglsTinyOverDeterminedMatchesQrReference() {
        IMatrix<Double> A = Linalg.matrix(new double[][]{
            {1, 0},
            {0, 1},
            {1, 1}
        });
        IVector<Double> b = Linalg.vector(new double[]{1, 0, 1});
        IMatrix<Double> bCol = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        IVector<Double> xQr = qr.getSolver().solve(bCol).getColumn(0);

        IVector<Double> xCgls = LeastSquaresSolver.solveLeastSquaresCglsSingleRhs(A, b);
        assertEquals(xQr.get(0), xCgls.get(0), 1e-10);
        assertEquals(xQr.get(1), xCgls.get(1), 1e-10);
    }

    @Test
    void cglsVandermonde40x2DirectMatchesQr() {
        int m = 40;
        int n = 2;
        double[][] ad = new double[m][n];
        for (int i = 0; i < m; i++) {
            ad[i][0] = 1.0;
            ad[i][1] = i * 0.25;
        }
        IMatrix<Double> A = Linalg.matrix(ad);
        double[] bd = new double[m];
        for (int i = 0; i < m; i++) {
            bd[i] = 2.0 * ad[i][0] + (-0.5) * ad[i][1];
        }
        IVector<Double> b = Linalg.vector(bd);
        IMatrix<Double> bCol = Linalg.matrix(new double[][]{bd}).transpose();
        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        IVector<Double> xQr = qr.getSolver().solve(bCol).getColumn(0);
        IVector<Double> xCgls = LeastSquaresSolver.solveLeastSquaresCglsSingleRhs(A, b);
        assertEquals(xQr.get(0), xCgls.get(0), 1e-8);
        assertEquals(xQr.get(1), xCgls.get(1), 1e-8);
    }

    private static double residualNorm(IMatrix<Double> A, IMatrix<Double> x, IMatrix<Double> bCol) {
        return ((Number) A.mmul(x).sub(bCol).frobeniusNorm()).doubleValue();
    }

    @Test
    void cgPathSingleRhsMatchesQrReference() {
        int m = 40;
        int n = 2;
        double[][] ad = new double[m][n];
        for (int i = 0; i < m; i++) {
            ad[i][0] = 1.0;
            ad[i][1] = i * 0.25;
        }
        IMatrix<Double> A = Linalg.matrix(ad);
        double xTrue0 = 2.0;
        double xTrue1 = -0.5;
        double[] bd = new double[m];
        for (int i = 0; i < m; i++) {
            bd[i] = xTrue0 * ad[i][0] + xTrue1 * ad[i][1];
        }
        IMatrix<Double> bCol = Linalg.matrix(new double[][]{bd}).transpose();

        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        IMatrix<Double> xRef = qr.getSolver().solve(bCol);

        LeastSquaresSolver.setHugeScaleThresholds(30, Long.MAX_VALUE);
        IMatrix<Double> xCg = LeastSquaresSolver.solve(A, bCol);

        for (int i = 0; i < n; i++) {
            assertEquals(xRef.get(i, 0), xCg.get(i, 0), 1e-9,
                "CG x differs from QR reference at index " + i);
        }
        double rQr = residualNorm(A, xRef, bCol);
        double rCg = residualNorm(A, xCg, bCol);
        assertTrue(rCg <= rQr * 1.05 + 1e-9, "CG residual should not exceed QR optimum much");
    }

    @Test
    void cgPathMatrixRhsMatchesQrReference() {
        int m = 32;
        int n = 3;
        int k = 2;
        IMatrix<Double> A = Linalg.rand(m, n, 42L);
        IMatrix<Double> B = Linalg.rand(m, k, 43L);

        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        IMatrix<Double> xRef = qr.getSolver().solve(B);

        LeastSquaresSolver.setHugeScaleThresholds(28, Long.MAX_VALUE);
        IMatrix<Double> xCg = LeastSquaresSolver.solve(A, B);

        for (int c = 0; c < k; c++) {
            for (int r = 0; r < n; r++) {
                assertEquals(xRef.get(r, c), xCg.get(r, c), 1e-9,
                    "column " + c + " row " + r);
            }
        }
    }

    @Test
    void cgPathTriggeredByElementCount() {
        LeastSquaresSolver.setHugeScaleThresholds(Integer.MAX_VALUE / 2, 80);
        int rows = 24;
        int cols = 10;
        assertTrue((long) rows * cols >= 80);
        IMatrix<Double> A = Linalg.rand(rows, cols, 7L);
        IVector<Double> b = Linalg.ones(rows);
        IMatrix<Double> bCol = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();

        var qr = Decomps.createQR();
        qr.decompose(A.copy());
        IMatrix<Double> xRef = qr.getSolver().solve(bCol);
        IMatrix<Double> xCg = LeastSquaresSolver.solve(A, bCol);

        double rQr = residualNorm(A, xRef, bCol);
        double rCg = residualNorm(A, xCg, bCol);
        assertTrue(rCg <= rQr * 1.05 + 1e-9);
        for (int i = 0; i < cols; i++) {
            assertEquals(xRef.get(i, 0), xCg.get(i, 0), 1e-8,
                "element-threshold CG path at index " + i);
        }
    }

    @Test
    void solveWithResidualCgPathConsistentWithNorm() {
        int m = 28;
        int n = 2;
        double[][] ad = new double[m][n];
        for (int i = 0; i < m; i++) {
            ad[i][0] = 1 + i * 0.02;
            ad[i][1] = -0.5 + i * 0.015;
        }
        IMatrix<Double> A = Linalg.matrix(ad);
        double[] bd = new double[m];
        for (int i = 0; i < m; i++) {
            bd[i] = ad[i][0] * 3 - ad[i][1];
        }
        IVector<Double> b = Linalg.vector(bd);

        LeastSquaresSolver.setHugeScaleThresholds(25, Long.MAX_VALUE);
        Tuple2<IVector<Double>, Double> res = LeastSquaresSolver.solveWithResidual(A, b);
        IVector<Double> x = res.getFirst();
        double reported = res.getSecond();
        double raw = A.mmul(x).sub(b).norm2Value();
        double actualRounded = RerePrecision.roundToDecimalPlaces(raw, 6);
        assertEquals(reported, actualRounded, 0.0);
    }
}
