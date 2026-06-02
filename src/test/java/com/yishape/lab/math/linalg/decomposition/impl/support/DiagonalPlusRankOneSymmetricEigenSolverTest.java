package com.yishape.lab.math.linalg.decomposition.impl.support;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DiagonalPlusRankOneSymmetricEigenSolverTest {

    private static final double TOL = 1e-8;

    @Test
    void twoByTwoHandVerified() {
        double[] d = {1.0, 4.0};
        double[] z = {1.0, 1.0};
        double rho = 1.0;
        var res = DiagonalPlusRankOneSymmetricEigenSolver.solve(d, z, rho, 1e-14);
        assertEquals(2, res.eigenvalues.length);
        double tr = 5.0 + rho * (z[0] * z[0] + z[1] * z[1]);
        double det = 4.0 + rho * (z[0] * z[0] * 4.0 + z[1] * z[1] * 1.0); // det(M) for this M
        double disc = tr * tr - 4 * det;
        double l0 = (tr + Math.sqrt(Math.max(0, disc))) / 2.0;
        double l1 = (tr - Math.sqrt(Math.max(0, disc))) / 2.0;
        double e0 = Math.max(l0, l1);
        double e1 = Math.min(l0, l1);
        assertEquals(e0, res.eigenvalues[0], TOL * Math.max(1, Math.abs(e0)));
        assertEquals(e1, res.eigenvalues[1], TOL * Math.max(1, Math.abs(e1)));
    }

    @Test
    void matchesDenseEigenRandomSmall() {
        Random rnd = new Random(42);
        for (int n = 1; n <= 12; n++) {
            for (int trial = 0; trial < 30; trial++) {
                double[] d = new double[n];
                double[] z = new double[n];
                for (int i = 0; i < n; i++) {
                    d[i] = rnd.nextGaussian();
                    z[i] = rnd.nextGaussian();
                }
                double rho = rnd.nextGaussian();
                IMatrix<Double> dense = buildDense(d, z, rho);
                var eigen = new RereEigenDecomposition(1e-14, 5000);
                Tuple2<com.yishape.lab.math.linalg.IVector<Double>, IMatrix<Double>> ref =
                        eigen.decompose(dense, 1e-14);
                double[] refVals = new double[n];
                for (int i = 0; i < n; i++) {
                    refVals[i] = ref.getFirst().get(i);
                }
                Arrays.sort(refVals);

                var got = DiagonalPlusRankOneSymmetricEigenSolver.solve(d, z, rho, 1e-12);
                double[] gval = Arrays.copyOf(got.eigenvalues, n);
                Arrays.sort(gval);

                for (int i = 0; i < n; i++) {
                    assertEquals(refVals[i], gval[i], 1e-6 * (1.0 + Math.abs(refVals[i])),
                            "n=" + n + " trial=" + trial + " i=" + i);
                }
            }
        }
    }

    @Test
    void ivectorOverloadMatchesArraySolve() {
        double[] d = {0.1, 0.5, 2.0, 5.0};
        double[] z = {0.2, -0.1, 0.4, 0.15};
        double rho = 0.7;
        IVector<Double> dv = Linalg.zeros(4);
        IVector<Double> zv = Linalg.zeros(4);
        for (int i = 0; i < 4; i++) {
            dv.set(i, d[i]);
            zv.set(i, z[i]);
        }
        var a = DiagonalPlusRankOneSymmetricEigenSolver.solve(d, z, rho, 1e-12);
        var b = DiagonalPlusRankOneSymmetricEigenSolver.solve(dv, zv, rho, 1e-12);
        for (int i = 0; i < 4; i++) {
            assertEquals(a.eigenvalues[i], b.eigenvalues[i], 1e-14 * (1 + Math.abs(a.eigenvalues[i])));
            for (int r = 0; r < 4; r++) {
                assertEquals(a.eigenvectors[r][i], b.eigenvectors[r][i], 1e-12);
            }
        }
    }

    @Test
    void deflationZeroZ() {
        double[] d = {1.0, 2.0, 5.0};
        double[] z = {0.4, 0.0, 0.3};
        double rho = 0.5;
        var res = DiagonalPlusRankOneSymmetricEigenSolver.solve(d, z, rho, 1e-14);
        assertEquals(3, res.eigenvalues.length);
        boolean has2 = false;
        for (double v : res.eigenvalues) {
            if (Math.abs(v - 2.0) < 1e-10) {
                has2 = true;
            }
        }
        assertTrue(has2);
    }

    @Test
    void orthogonalityColumns() {
        double[] d = new double[9];
        double[] z = new double[9];
        Random rnd = new Random(7);
        for (int i = 0; i < 9; i++) {
            d[i] = rnd.nextDouble() * 5;
            z[i] = rnd.nextDouble();
        }
        var res = DiagonalPlusRankOneSymmetricEigenSolver.solve(d, z, 0.3, 1e-12);
        int n = res.eigenvectors.length;
        int m = res.eigenvectors[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                double dot = 0;
                for (int r = 0; r < n; r++) {
                    dot += res.eigenvectors[r][i] * res.eigenvectors[r][j];
                }
                assertEquals(0.0, dot, 1e-7);
            }
        }
    }

    @Test
    void largeDimensionSecularPathMatchesDenseEigen() {
        Random rnd = new Random(202);
        int n = 320;
        double[] d = new double[n];
        double[] z = new double[n];
        for (int i = 0; i < n; i++) {
            d[i] = rnd.nextGaussian();
            z[i] = rnd.nextGaussian();
        }
        double rho = rnd.nextGaussian();
        IMatrix<Double> dense = buildDense(d, z, rho);
        var eigen = new RereEigenDecomposition(1e-12, 8000);
        var ref = eigen.decompose(dense, 1e-12);
        double[] refVals = new double[n];
        for (int i = 0; i < n; i++) {
            refVals[i] = ref.getFirst().get(i);
        }
        Arrays.sort(refVals);

        var got = DiagonalPlusRankOneSymmetricEigenSolver.solve(d, z, rho, 1e-10);
        double[] gval = Arrays.copyOf(got.eigenvalues, n);
        Arrays.sort(gval);

        for (int i = 0; i < n; i++) {
            assertEquals(refVals[i], gval[i], 1e-5 * (1.0 + Math.abs(refVals[i])), "i=" + i);
        }
    }

    private static IMatrix<Double> buildDense(double[] d, double[] z, double rho) {
        int n = d.length;
        double[][] m = new double[n][n];
        for (int i = 0; i < n; i++) {
            m[i][i] = d[i];
            for (int j = 0; j < n; j++) {
                m[i][j] += rho * z[i] * z[j];
            }
        }
        return Linalg.matrix(m);
    }

}
