package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDecomposition;
import com.yishape.lab.math.linalg.solver.LeastSquaresSolver;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 列主元 QR（QRCP）：分解关系、最小二乘与秩亏损情形。
 */
public class QrcpDecompositionTest {

    private static double frobeniusNorm(IMatrix<Double> m) {
        double s = 0.0;
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                double v = m.get(i, j);
                s += v * v;
            }
        }
        return Math.sqrt(s);
    }

    private static double frobeniusDiff(IMatrix<Double> a, IMatrix<Double> b) {
        double s = 0.0;
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                double d = a.get(i, j) - b.get(i, j);
                s += d * d;
            }
        }
        return Math.sqrt(s);
    }

    @Test
    void factorizationApEqualsQr() {
        double[][] d = {
            {1, -2, 3},
            {4, 5, -6},
            {7, 8, 9},
            {-1, 0, 2}
        };
        IMatrix<Double> a = Linalg.matrix(d);
        var qrcp = new RereQrcpDecomposition(1e-14);
        qrcp.decompose(a);
        IMatrix<Double> q = qrcp.getQ();
        IMatrix<Double> r = qrcp.getR();
        IMatrix<Double> p = qrcp.getColumnPermutation();
        IMatrix<Double> qr = q.mmul(r);
        IMatrix<Double> reconstructed = qr.mmul(p.transposeNew());
        assertTrue(frobeniusDiff(a, reconstructed) < 1e-10);
    }

    @Test
    void permutationIsOrthogonal() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{3, 1, 4}, {1, 5, 9}, {2, 6, 5}});
        var qrcp = new RereQrcpDecomposition(1e-14);
        qrcp.decompose(a);
        IMatrix<Double> p = qrcp.getColumnPermutation();
        IMatrix<Double> shouldEye = p.transposeNew().mmul(p);
        for (int i = 0; i < shouldEye.rows(); i++) {
            for (int j = 0; j < shouldEye.cols(); j++) {
                double expect = i == j ? 1.0 : 0.0;
                assertEquals(expect, shouldEye.get(i, j), 1e-12);
            }
        }
    }

    @Test
    void leastSquaresFullRank_consistentWithPlainQr() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{1}, {0}, {0}});
        var plain = Decomps.createQR(1e-14);
        plain.decompose(a);
        var cp = Decomps.createQrcp(1e-14);
        cp.decompose(a);
        IMatrix<Double> xPlain = plain.getSolver().solve(b);
        IMatrix<Double> xCp = cp.getSolver().solve(b);
        assertTrue(frobeniusDiff(xPlain, xCp) < 1e-9);
    }

    @Test
    void rankRevealing_pivotsDuplicateColumnsApart() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{
            {1, 1, 0},
            {0, 0, 1},
            {1, 1, 1}
        });
        var qrcp = new RereQrcpDecomposition(1e-14);
        qrcp.decompose(a);
        IMatrix<Double> q = qrcp.getQ();
        IMatrix<Double> r = qrcp.getR();
        IMatrix<Double> p = qrcp.getColumnPermutation();
        IMatrix<Double> apMinusQr = a.mmul(p).sub(q.mmul(r));
        assertTrue(frobeniusNorm(apMinusQr) < 1e-10);
        assertTrue(qrcp.getRank() <= 2);
        assertTrue(Math.abs(r.get(2, 2)) < 1e-8);
    }

    @Test
    void leastSquaresSolverUsesQrcpPath_consistentRhs() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}, {7, 1}});
        IMatrix<Double> x0 = Linalg.matrix(new double[][]{{1.0}, {-0.5}});
        IMatrix<Double> b = a.mmul(x0);
        IMatrix<Double> x = LeastSquaresSolver.solve(a, b);
        assertTrue(frobeniusDiff(x, x0) < 1e-7);
        assertTrue(frobeniusDiff(a.mmul(x), b) < 1e-10);
    }

    @Test
    void randomTallMatchesOrdinaryWhenWellConditioned() {
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int t = 0; t < 15; t++) {
            int m = rnd.nextInt(8, 20);
            int n = rnd.nextInt(3, m);
            double[][] d = new double[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    d[i][j] = rnd.nextDouble(-1, 1);
                }
            }
            IMatrix<Double> a = Linalg.matrix(d);
            double[][] bd = new double[m][2];
            for (int i = 0; i < m; i++) {
                bd[i][0] = rnd.nextDouble(-1, 1);
                bd[i][1] = rnd.nextDouble(-1, 1);
            }
            IMatrix<Double> b = Linalg.matrix(bd);
            var qr = Decomps.createQR(1e-14);
            var qrcp = Decomps.createQrcp(1e-14);
            qr.decompose(a);
            qrcp.decompose(a);
            if (!qr.isNonSingular() || !qrcp.isNonSingular()) {
                continue;
            }
            IMatrix<Double> xQr = qr.getSolver().solve(b);
            IMatrix<Double> xCp = qrcp.getSolver().solve(b);
            assertTrue(frobeniusDiff(xQr, xCp) < 1e-8 * Math.max(1.0, frobeniusNorm(xQr)));
        }
    }
}
