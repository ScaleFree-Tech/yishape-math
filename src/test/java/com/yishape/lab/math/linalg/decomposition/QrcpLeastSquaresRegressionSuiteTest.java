package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDecomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDgeqp3Decomposition;
import com.yishape.lab.math.linalg.decomposition.impl.RereQrcpDlaqpsDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.solver.LeastSquaresSolver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * QRCP（朴素 / DGEQP3 / DLAQPS 列条带）、秩容忍求解与 {@link LeastSquaresSolver} 主路径的
 * 系统性回归：供发布前及第三方性能对照前的功能正确性校验。
 */
@DisplayName("QRCP & least squares regression suite")
public class QrcpLeastSquaresRegressionSuiteTest {

    private static double frob(IMatrix<Double> m) {
        double s = 0.0;
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                double v = m.get(i, j);
                s += v * v;
            }
        }
        return Math.sqrt(s);
    }

    private static double reconErr(IQrcpDecomposition dec, IMatrix<Double> a) {
        dec.decompose(a);
        IMatrix<Double> q = dec.getQ();
        IMatrix<Double> r = dec.getR();
        IMatrix<Double> p = dec.getColumnPermutation();
        IMatrix<Double> recon = q.mmul(r).mmul(p.transposeNew());
        IMatrix<Double> d = a.sub(recon);
        return frob(d);
    }

    @AfterEach
    void resetHuge() {
        LeastSquaresSolver.resetHugeScaleThresholds();
    }

    @Test
    @DisplayName("QrcpRankTolerance: APIs finite and rcond validation")
    void rankToleranceApis() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{3, 1}, {1, 2}, {0, 1}});
        var q = Decomps.createQrcpDlaqps();
        q.decompose(a);
        double maxd = QrcpRankTolerance.maxLeadingDiagonalAbs(q, a.rows(), a.cols());
        assertTrue(maxd >= 0.0 && Double.isFinite(maxd));
        double tol = QrcpRankTolerance.forLeastSquares(q, a.rows(), a.cols());
        assertTrue(tol >= q.getEpsilon() && Double.isFinite(tol));
        double tRel = QrcpRankTolerance.fromRelativeRcond(q, a.rows(), a.cols(), 1e-6);
        assertTrue(tRel >= q.getEpsilon());
        assertThrows(IllegalArgumentException.class, () ->
                QrcpRankTolerance.fromRelativeRcond(q, a.rows(), a.cols(), 0.0));
        assertThrows(IllegalArgumentException.class, () ->
                QrcpRankTolerance.fromRelativeRcond(q, a.rows(), a.cols(), 1.1));
    }

    @Test
    @DisplayName("Three QRCP variants: same reconstruction on deterministic matrices")
    void threeVariantsReconstruction() {
        double[][][] cases = new double[][][]{
                {{1, 2, 3}, {4, 5, 6}, {7, 0, -1}},
                {{0.1, 0.2}, {0.3, -0.1}, {0.0, 0.5}},
                {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}
        };
        for (double[][] d : cases) {
            IMatrix<Double> a = Linalg.matrix(d);
            double e0 = reconErr(new RereQrcpDecomposition(1e-14), Linalg.matrix(copy2d(d)));
            double e1 = reconErr(new RereQrcpDgeqp3Decomposition(1e-14), Linalg.matrix(copy2d(d)));
            double e2 = reconErr(new RereQrcpDlaqpsDecomposition(1e-14, 11), Linalg.matrix(copy2d(d)));
            double scale = Math.max(1.0, frob(a));
            assertTrue(e0 < 1e-9 * scale, "naive err " + e0);
            assertTrue(e1 < 1e-9 * scale, "dgeqp3 err " + e1);
            assertEquals(e1, e2, 1e-20 * scale, "dlaqps vs dgeqp3");
        }
    }

    private static double[][] copy2d(double[][] d) {
        double[][] c = new double[d.length][];
        for (int i = 0; i < d.length; i++) {
            c[i] = d[i].clone();
        }
        return c;
    }

    @Test
    @DisplayName("LeastSquaresSolver tall: residual matches SVD on random full-rank overdetermined")
    void leastSquaresVersusSvdResidual() {
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int t = 0; t < 25; t++) {
            int m = rnd.nextInt(12, 28);
            int n = rnd.nextInt(3, Math.min(9, m));
            double[][] d = new double[m][n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    d[i][j] = rnd.nextDouble(-2, 2);
                }
            }
            if (n == m) {
                d[0][0] += 0.5;
            }
            IMatrix<Double> a = Linalg.matrix(d);
            IMatrix<Double> b = Linalg.matrix(new double[m][1]);
            for (int i = 0; i < m; i++) {
                b.put(i, 0, rnd.nextDouble(-1, 1));
            }
            IMatrix<Double> xLs = LeastSquaresSolver.solve(a, b);
            var svd = Decomps.createSVD(1e-15, 2000);
            svd.decompose(a);
            IMatrix<Double> xSvd = svd.getSolver().solve(b);
            double resLs = frob(a.mmul(xLs).sub(b));
            double resSvd = frob(a.mmul(xSvd).sub(b));
            assertEquals(resSvd, resLs, 1e-8 * Math.max(1e-12, resSvd));
            IVector<Double> atr = a.transpose().mmul(a.mmul(xLs).sub(b)).getColumn(0);
            assertTrue(atr.norm2Value() < 1e-6 * (1.0 + frob(b)),
                    "normal eq residual " + atr.norm2Value());
        }
    }

    @Test
    @DisplayName("LeastSquaresSolver: two RHS columns consistent")
    void leastSquaresMultiRhs() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{
                {1, 2}, {3, -1}, {0, 1}, {2, 2}
        });
        IMatrix<Double> b = Linalg.matrix(new double[][]{{1, 0}, {0, 1}, {1, 1}, {-1, 2}});
        IMatrix<Double> x = LeastSquaresSolver.solve(a, b);
        assertEquals(a.cols(), x.rows());
        assertEquals(2, x.cols());
        double res = frob(a.mmul(x).sub(b));
        var svd = Decomps.createSVD(1e-15, 2000);
        svd.decompose(a);
        double resSvd = frob(a.mmul(svd.getSolver().solve(b)).sub(b));
        assertEquals(resSvd, res, 1e-8 * Math.max(1e-12, resSvd));
    }

    @Test
    @DisplayName("Tolerant vs strict solver on full-rank small problem (should match)")
    void tolerantMatchesStrictOnFullRank() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {3, 4}, {2, 1}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{1}, {0}, {1}});
        var qrcp = new RereQrcpDgeqp3Decomposition(1e-14);
        qrcp.decompose(a);
        assertTrue(qrcp.getRank() >= a.cols());
        IDecompositionSolver strict = qrcp.getSolver();
        double tol = QrcpRankTolerance.forLeastSquares(qrcp, a.rows(), a.cols());
        IDecompositionSolver tolerant = qrcp.createTolerantLeastSquaresSolver(tol);
        IMatrix<Double> xs = strict.solve(b);
        IMatrix<Double> xt = tolerant.solve(b);
        assertTrue(frob(xs.sub(xt)) < 1e-10);
    }

    @Test
    @DisplayName("Factory createQrcpDlaqps: decomposition and least squares")
    void factoryDlaqpsSolve() {
        IQrcpDecomposition qr = Decomps.createQrcpDlaqps(7);
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 0}, {1, 1e-6}, {0, 1}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{1}, {1}, {0}});
        qr.decompose(a);
        assertNotNull(qr.getColumnPivot());
        assertEquals(2, qr.getColumnPivot().length);
        IMatrix<Double> x = qr.getSolver().solve(b);
        assertTrue(frob(a.mmul(x).sub(b)) < 1e-8);
    }

    @Test
    @DisplayName("Huge-scale path: CG still invoked when thresholds overridden")
    void cgPathStillWorksAfterQrcpChanges() {
        int m = 40;
        int n = 3;
        double[][] d = new double[m][n];
        for (int i = 0; i < m; i++) {
            d[i][0] = 1;
            d[i][1] = i * 0.1;
            d[i][2] = i * 0.01;
        }
        IMatrix<Double> a = Linalg.matrix(d);
        IMatrix<Double> x0 = Linalg.matrix(new double[][]{{1.0}, {-0.5}, {2.0}});
        IMatrix<Double> b = a.mmul(x0);
        LeastSquaresSolver.setHugeScaleThresholds(20, Long.MAX_VALUE);
        IMatrix<Double> x = LeastSquaresSolver.solve(a, b);
        double resCg = frob(a.mmul(x).sub(b));
        assertTrue(resCg < 1e-6 || frob(x.sub(x0)) / Math.max(1e-12, frob(x0)) < 0.1,
                "CG path residual " + resCg);
    }
}
