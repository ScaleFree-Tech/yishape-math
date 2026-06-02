package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDgeqp3Decomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDlaqpsDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * DGEQP3 式范数递推与 xGELSY 容忍求解器。
 */
public class QrcpDgeqp3AndTolerantSolverTest {

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

    private static double reconstructionError(IQrcpDecomposition dec, IMatrix<Double> a) {
        dec.decompose(a);
        IMatrix<Double> q = dec.getQ();
        IMatrix<Double> r = dec.getR();
        IMatrix<Double> p = dec.getColumnPermutation();
        IMatrix<Double> recon = q.mmul(r).mmul(p.transposeNew());
        double err = 0.0;
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                double d = a.get(i, j) - recon.get(i, j);
                err += d * d;
            }
        }
        return Math.sqrt(err);
    }

    @Test
    void dgeqp3_sameAccuracyAsNaiveOnRandom() {
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int t = 0; t < 20; t++) {
            int m = rnd.nextInt(15, 40);
            int n = rnd.nextInt(5, 15);
            double[][] d = new double[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    d[i][j] = rnd.nextDouble(-3, 3);
                }
            }
            IMatrix<Double> a = Linalg.matrix(d);
            var naive = new RereQrcpDecomposition(1e-14);
            var fast = new RereQrcpDgeqp3Decomposition(1e-14);
            double e1 = reconstructionError(naive, a);
            double e2 = reconstructionError(fast, a);
            assertTrue(e1 < 1e-8 * Math.max(1.0, frobeniusNorm(a)));
            assertTrue(e2 < 1e-8 * Math.max(1.0, frobeniusNorm(a)));
        }
    }

    @Test
    void dlaqps_matchesDgeqp3Reconstruction() {
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int t = 0; t < 15; t++) {
            int m = rnd.nextInt(15, 40);
            int n = rnd.nextInt(10, 35);
            double[][] d = new double[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    d[i][j] = rnd.nextDouble(-3, 3);
                }
            }
            IMatrix<Double> a1 = Linalg.matrix(d);
            IMatrix<Double> a2 = Linalg.matrix(d);
            var dgeqp3 = new RereQrcpDgeqp3Decomposition(1e-14);
            var dlaqps = new RereQrcpDlaqpsDecomposition(1e-14, 13 + t % 11);
            double e2 = reconstructionError(dgeqp3, a1);
            double e3 = reconstructionError(dlaqps, a2);
            assertEquals(e2, e3, 1e-20 * Math.max(1.0, frobeniusNorm(a1)));
        }
    }

    @Test
    void tolerantSolver_rankDeficient_duplicateColumns() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{
            {1, 1, 0},
            {0, 0, 1},
            {1, 1, 1}
        });
        IMatrix<Double> b = Linalg.matrix(new double[][]{{1}, {1}, {2}});
        var qrcp = new RereQrcpDgeqp3Decomposition(1e-14);
        qrcp.decompose(a);
        IDecompositionSolver tol = qrcp.createTolerantLeastSquaresSolver(1e-10);
        IMatrix<Double> x = tol.solve(b);
        IMatrix<Double> ax = a.mmul(x);
        double res = frobeniusNorm(ax.sub(b));
        assertTrue(res < 1e-9);
    }

    @Test
    void tolerantSolver_underdetermined_sameResidualAsSvd() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{1}, {0}});
        var qrcp = new RereQrcpDgeqp3Decomposition(1e-15);
        qrcp.decompose(a);
        IMatrix<Double> xT = qrcp.createTolerantLeastSquaresSolver(1e-12).solve(b);
        var svd = Decomps.createSVD();
        svd.decompose(a);
        IMatrix<Double> xS = svd.getSolver().solve(b);
        double resT = frobeniusNorm(a.mmul(xT).sub(b));
        double resS = frobeniusNorm(a.mmul(xS).sub(b));
        assertEquals(resS, resT, 1e-10);
    }

    @Test
    void createTolerantFromNaiveQrcp_consistent() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 1}, {0, 1}, {1, 0}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{1}, {1}, {0}});
        var qrcp = new RereQrcpDecomposition(1e-14);
        qrcp.decompose(a);
        IMatrix<Double> x = qrcp.createTolerantLeastSquaresSolver(1e-12).solve(b);
        IMatrix<Double> ax = a.mmul(x);
        assertTrue(frobeniusNorm(ax.sub(b)) < 1e-10);
    }
}
