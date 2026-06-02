package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.BunchKaufmanLdltLower;
import com.yishape.lab.math.linalg.decomposition.impl.RereBunchKaufmanDecomposition;
import com.yishape.lab.math.linalg.solver.LinearSystemSolver;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.concurrent.ThreadLocalRandom;

/**
 * Bunch–Kaufman (SYTRF/DSYTRS 下三角) 正确性：残差与行列式、与 LU 对照。
 */
public class BunchKaufmanDecompositionTest {

    private static double frobeniusResidual(IMatrix<Double> a, IMatrix<Double> x, IMatrix<Double> b) {
        IMatrix<Double> ax = a.mmul(x);
        double s = 0.0;
        int rows = b.rows();
        int cols = b.cols();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double d = ax.get(i, j) - b.get(i, j);
                s += d * d;
            }
        }
        return Math.sqrt(s);
    }

    @Test
    public void indefinite2x2_matchesReference() {
        double[][] ad = {{1, 2}, {2, 1}};
        IMatrix<Double> a = Linalg.matrix(ad);
        double[][] bd = {{5}, {-1}};
        IMatrix<Double> b = Linalg.matrix(bd);

        var bk = new RereBunchKaufmanDecomposition();
        bk.decompose(a);
        assertTrue(bk.isNonSingular());
        IMatrix<Double> x = bk.getSolver().solve(b);

        assertEquals(-7.0 / 3.0, x.get(0, 0), 1e-12);
        assertEquals(11.0 / 3.0, x.get(1, 0), 1e-12);
        assertTrue(frobeniusResidual(a, x, b) < 1e-10);
        assertEquals(-3.0, bk.getDeterminant(), 1e-10);
    }

    @Test
    public void smallAlmostZeroDiagonal_forces2x2Pivot_solveOk() {
        double eps = 1e-8;
        double[][] ad = {
            {eps, 1, 0},
            {1, 2, 0},
            {0, 0, 3}
        };
        IMatrix<Double> a = Linalg.matrix(ad);
        IMatrix<Double> rhs = Linalg.matrix(new double[][]{{1}, {1}, {1}});

        var bk = new RereBunchKaufmanDecomposition(1e-14, 1e-14);
        bk.decompose(a);
        assertTrue(bk.isNonSingular());
        IMatrix<Double> x = bk.getSolver().solve(rhs);
        assertTrue(frobeniusResidual(a, x, rhs) < 1e-8);

        var lu = Decomps.createLU();
        lu.decompose(a);
        assertTrue(lu.isNonSingular());
        IMatrix<Double> xLu = lu.getSolver().solve(rhs);
        for (int i = 0; i < 3; i++) {
            assertEquals(xLu.get(i, 0), x.get(i, 0), 1e-8);
        }
    }

    @Test
    public void deterministicDenseSymmetric_randomVsLu() {
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int trial = 0; trial < 30; trial++) {
            int n = rnd.nextInt(4, 13);
            double[][] d = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    d[i][j] = rnd.nextDouble(-2, 2);
                }
            }
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    d[j][i] = d[i][j];
                }
            }
            IMatrix<Double> a = Linalg.matrix(d);
            double[][] br = new double[n][2];
            for (int i = 0; i < n; i++) {
                br[i][0] = rnd.nextDouble(-1, 1);
                br[i][1] = rnd.nextDouble(-1, 1);
            }
            IMatrix<Double> b = Linalg.matrix(br);

            var bk = new RereBunchKaufmanDecomposition();
            bk.decompose(a);
            if (!bk.isNonSingular()) {
                continue;
            }
            IMatrix<Double> xBk = bk.getSolver().solve(b);
            var lu = Decomps.createLU();
            lu.decompose(a);
            if (!lu.isNonSingular()) {
                continue;
            }
            IMatrix<Double> xLu = lu.getSolver().solve(b);
            assertTrue(frobeniusResidual(a, xBk, b) < 1e-9);
            for (int i = 0; i < n; i++) {
                assertEquals(xLu.get(i, 0), xBk.get(i, 0), 1e-8);
                assertEquals(xLu.get(i, 1), xBk.get(i, 1), 1e-8);
            }
        }
    }

    @Test
    public void linearSystemSolver_usesBunchKaufmanForSymmetricIndefinite() {
        IMatrix<Double> a = Linalg.matrix(new double[][]{{1, 2}, {2, 1}});
        IMatrix<Double> b = Linalg.matrix(new double[][]{{1}, {0}});
        IMatrix<Double> x = LinearSystemSolver.solve(a, b);
        assertTrue(frobeniusResidual(a, x, b) < 1e-10);
    }

    @Test
    public void rawDsytf2Dsytrs_roundTrip() {
        int n = 5;
        double[][] a = {
            {2, -1, 0, 0, 0},
            {-1, 2, -1, 0, 0},
            {0, -1, 2, -1, 0},
            {0, 0, -1, 2, -1},
            {0, 0, 0, -1, 2}
        };
        int[] ipiv = new int[n];
        BunchKaufmanLdltLower.dsytf2Lower(a, n, ipiv);
        assertEquals(6.0, BunchKaufmanLdltLower.determinantFromFactor(a, n, ipiv, 1e-10), 1e-8);

        double[][] b = new double[n][1];
        for (int i = 0; i < n; i++) {
            b[i][0] = 1;
        }
        BunchKaufmanLdltLower.dsytrsLower(a, n, ipiv, b, 1);
        double[][] acopy = {
            {2, -1, 0, 0, 0},
            {-1, 2, -1, 0, 0},
            {0, -1, 2, -1, 0},
            {0, 0, -1, 2, -1},
            {0, 0, 0, -1, 2}
        };
        IMatrix<Double> am = Linalg.matrix(acopy);
        IMatrix<Double> xm = Linalg.matrix(b);
        IMatrix<Double> rhs = Linalg.matrix(new double[n][1]);
        for (int i = 0; i < n; i++) {
            rhs.set(i, 0, 1.0);
        }
        assertTrue(frobeniusResidual(am, xm, rhs) < 1e-10);
    }
}
