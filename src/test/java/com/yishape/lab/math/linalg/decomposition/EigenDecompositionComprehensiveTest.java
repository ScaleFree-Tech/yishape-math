package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.util.Tuple2;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class EigenDecompositionComprehensiveTest {

    private static final double TOL = 1e-10;

    @Test
    public void testSymmetricMatrixEigenvalues() {
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;

        assertEquals(3, eigenvalues.length());
        assertEquals(3, eigenvectors.rows());
        assertEquals(3, eigenvectors.cols());

        // Verify A * v = λ * v for each eigenpair (columns of eigenvector matrix)
        for (int j = 0; j < 3; j++) {
            double lambda = eigenvalues.get(j);
            for (int i = 0; i < 3; i++) {
                double av = 0.0;
                for (int k = 0; k < 3; k++) {
                    av += A.get(i, k) * eigenvectors.get(k, j);
                }
                assertEquals(lambda * eigenvectors.get(i, j), av, TOL);
            }
        }
    }

    @Test
    public void testSymmetricMatrixReconstruction() {
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IMatrix<Double> V = result._2;
        IVector<Double> ev = result._1;

        IMatrix<Double> D = Linalg.zeros(3, 3);
        for (int i = 0; i < 3; i++) {
            D.put(i, i, ev.get(i));
        }

        // A ≈ V * D * V^(-1)
        IMatrix<Double> vInv = Linalg.solveLinearSystem(V, Linalg.eye(3));
        IMatrix<Double> reconstructed = V.mmul(D).mmul(vInv);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), 1e-8);
            }
        }
    }

    @Test
    public void testSymmetricSolve() {
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        eigen.decompose(A);
        IDecompositionSolver solver = eigen.getSolver();

        double[] b = {1.0, 2.0, 3.0};
        IVector<Double> bVec = Linalg.vector(b);
        IVector<Double> x = solver.solve(bVec);

        for (int i = 0; i < 3; i++) {
            double ax = 0.0;
            for (int k = 0; k < 3; k++) {
                ax += A.get(i, k) * x.get(k);
            }
            assertEquals(b[i], ax, 1e-8);
        }
    }

    @Test
    public void testEigenvalueOrderDescending() {
        double[][] data = {
            {2.0, 0.0, 0.0},
            {0.0, 5.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> ev = result._1;
        assertTrue(ev.get(0) >= ev.get(1), "Eigenvalues must be in descending order");
        assertTrue(ev.get(1) >= ev.get(2), "Eigenvalues must be in descending order");
    }

    @Test
    public void test2x2NonSymmetricEigenvalues() {
        // 2x2 non-symmetric matrix: uses qrAlgorithmForHessenberg (explicit QR)
        double[][] data = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;

        assertEquals(2, eigenvalues.length());

        // Verify A * v = λ * v for each eigenpair
        for (int j = 0; j < 2; j++) {
            double lambda = eigenvalues.get(j);
            for (int i = 0; i < 2; i++) {
                double av = 0.0;
                for (int k = 0; k < 2; k++) {
                    av += A.get(i, k) * eigenvectors.get(k, j);
                }
                assertEquals(lambda * eigenvectors.get(i, j), av, 1e-8);
            }
        }

        // Verify eigenvalues: trace = a+d = 5, det = ad-bc = -2
        double sum = eigenvalues.get(0) + eigenvalues.get(1);
        assertEquals(5.0, sum, 1e-8);
    }

    @Test
    public void test2x2NonSymmetricSolve() {
        double[][] data = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        eigen.decompose(A);
        IDecompositionSolver solver = eigen.getSolver();

        double[] b = {1.0, 2.0};
        IVector<Double> bVec = Linalg.vector(b);
        IVector<Double> x = solver.solve(bVec);

        for (int i = 0; i < 2; i++) {
            double ax = 0.0;
            for (int k = 0; k < 2; k++) {
                ax += A.get(i, k) * x.get(k);
            }
            assertEquals(b[i], ax, 1e-8);
        }
    }

    @Test
    public void test2x2NonSymmetricReconstruction() {
        double[][] data = {
            {1.0, 2.0},
            {3.0, 4.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IMatrix<Double> V = result._2;
        IVector<Double> ev = result._1;

        IMatrix<Double> D = Linalg.zeros(2, 2);
        for (int i = 0; i < 2; i++) {
            D.put(i, i, ev.get(i));
        }

        IMatrix<Double> vInv = Linalg.solveLinearSystem(V, Linalg.eye(2));
        IMatrix<Double> reconstructed = V.mmul(D).mmul(vInv);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), 1e-8);
            }
        }
    }

    @Test
    public void test2x2NonSymmetricSpecific() {
        // Test a specific 2x2 matrix with known analytic solution
        double[][] data = {
            {7.0, 2.0},
            {3.0, 6.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;

        // Trace = 13, det = 36, eigenvalues: (13 ± sqrt(169-144))/2 = (13 ± 5)/2 = 9 and 4
        assertEquals(9.0, eigenvalues.get(0), 1e-8);
        assertEquals(4.0, eigenvalues.get(1), 1e-8);

        // Verify A * v = λ * v
        for (int j = 0; j < 2; j++) {
            double lambda = eigenvalues.get(j);
            for (int i = 0; i < 2; i++) {
                double av = 0.0;
                for (int k = 0; k < 2; k++) {
                    av += A.get(i, k) * eigenvectors.get(k, j);
                }
                assertEquals(lambda * eigenvectors.get(i, j), av, 1e-8);
            }
        }
    }

    @Test
    public void test3x3NonSymmetricGeneral() {
        // General non-symmetric 3x3 matrix with all real eigenvalues
        // λ ≈ [4.3028, 3.0000, 0.6972], trace = 8
        double[][] data = {
            {3.0, 2.0, 0.0},
            {1.0, 2.0, 1.0},
            {0.0, 1.0, 3.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 5000);
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;

        assertEquals(3, eigenvalues.length());

        // Verify eigenvalues sum = trace
        double sum = eigenvalues.get(0) + eigenvalues.get(1) + eigenvalues.get(2);
        assertEquals(8.0, sum, 1e-8);

        // Verify A * v = λ * v for each eigenpair
        for (int j = 0; j < 3; j++) {
            double lambda = eigenvalues.get(j);
            for (int i = 0; i < 3; i++) {
                double av = 0.0;
                for (int k = 0; k < 3; k++) {
                    av += A.get(i, k) * eigenvectors.get(k, j);
                }
                assertEquals(lambda * eigenvectors.get(i, j), av, 1e-6);
            }
        }
    }

    @Test
    public void test3x3NonSymmetricReconstruction() {
        double[][] data = {
            {3.0, 2.0, 0.0},
            {1.0, 2.0, 1.0},
            {0.0, 1.0, 3.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 5000);
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IMatrix<Double> V = result._2;
        IVector<Double> ev = result._1;

        IMatrix<Double> D = Linalg.zeros(3, 3);
        for (int i = 0; i < 3; i++) {
            D.put(i, i, ev.get(i));
        }

        IMatrix<Double> vInv = Linalg.solveLinearSystem(V, Linalg.eye(3));
        IMatrix<Double> reconstructed = V.mmul(D).mmul(vInv);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), 1e-6);
            }
        }
    }

    @Test
    public void test3x3NonSymmetricSolve() {
        double[][] data = {
            {3.0, 2.0, 0.0},
            {1.0, 2.0, 1.0},
            {0.0, 1.0, 3.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 5000);
        eigen.decompose(A);

        double[] b = {1.0, 2.0, 3.0};
        IVector<Double> bVec = Linalg.vector(b);
        IVector<Double> x = eigen.getSolver().solve(bVec);

        for (int i = 0; i < 3; i++) {
            double ax = 0.0;
            for (int k = 0; k < 3; k++) {
                ax += A.get(i, k) * x.get(k);
            }
            assertEquals(b[i], ax, 1e-6);
        }
    }
}
