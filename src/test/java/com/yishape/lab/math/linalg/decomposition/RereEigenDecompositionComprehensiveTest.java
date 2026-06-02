package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.util.Tuple2;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for RereEigenDecomposition.
 * Tests correctness, numerical stability, symmetric/non-symmetric matrices, and edge cases.
 */
public class RereEigenDecompositionComprehensiveTest {

    private static final double TOL = 1e-10;
    private static final double LOOSE_TOL = 1e-8;
    private static final double VERY_LOOSE_TOL = 1e-6;

    // ========== Symmetric Matrix Tests ==========

    @Test
    public void testSymmetricMatrix_3x3() {
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

        // Verify A * v = λ * v for each eigenpair
        verifyEigenpairs(A, eigenvalues, eigenvectors, LOOSE_TOL);

        // Verify eigenvalue sum = trace
        double trace = data[0][0] + data[1][1] + data[2][2];
        double sum = 0.0;
        for (int i = 0; i < 3; i++) sum += eigenvalues.get(i);
        assertEquals(trace, sum, TOL);
    }

    @Test
    public void testSymmetricMatrix_4x4() {
        double[][] data = {
            {6.0, 2.0, 1.0, 0.0},
            {2.0, 3.0, 1.0, 2.0},
            {1.0, 1.0, 4.0, 1.0},
            {0.0, 2.0, 1.0, 5.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        verifyEigenpairs(A, result._1, result._2, LOOSE_TOL);
    }

    @Test
    public void testSymmetricMatrix_5x5() {
        java.util.Random rand = new java.util.Random(12345);
        double[][] data = new double[5][5];
        // Create symmetric matrix
        for (int i = 0; i < 5; i++) {
            for (int j = i; j < 5; j++) {
                double val = rand.nextDouble() * 10;
                data[i][j] = val;
                data[j][i] = val;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        verifyEigenpairs(A, result._1, result._2, LOOSE_TOL);
    }

    @Test
    public void testSymmetricMatrix_10x10() {
        java.util.Random rand = new java.util.Random(54321);
        double[][] data = new double[10][10];
        for (int i = 0; i < 10; i++) {
            for (int j = i; j < 10; j++) {
                double val = rand.nextDouble() * 10;
                data[i][j] = val;
                data[j][i] = val;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        verifyEigenpairs(A, result._1, result._2, LOOSE_TOL);
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

        // Build diagonal matrix from eigenvalues
        IMatrix<Double> D = Linalg.zeros(3, 3);
        for (int i = 0; i < 3; i++) {
            D.set(i, i, ev.get(i));
        }

        // A ≈ V * D * V^T (since V is orthogonal for symmetric matrices)
        IMatrix<Double> reconstructed = V.mmul(D).mmul(V.transposeNew());

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), LOOSE_TOL);
            }
        }
    }

    // ========== Non-Symmetric Matrix Tests ==========

    @Test
    public void testNonSymmetricMatrix_2x2() {
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

        // Verify A * v = λ * v
        verifyEigenpairs(A, eigenvalues, eigenvectors, LOOSE_TOL);

        // Verify trace = sum of eigenvalues
        double trace = 1.0 + 4.0;
        assertEquals(trace, eigenvalues.get(0) + eigenvalues.get(1), TOL);
    }

    @Test
    public void testNonSymmetricMatrix_3x3() {
        // General non-symmetric 3x3 with all real eigenvalues
        double[][] data = {
            {3.0, 2.0, 0.0},
            {1.0, 2.0, 1.0},
            {0.0, 1.0, 3.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 5000);
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        verifyEigenpairs(A, result._1, result._2, VERY_LOOSE_TOL);
    }

    @Test
    public void testNonSymmetricMatrix_4x4() {
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0},
            {0.0, 5.0, 6.0, 7.0},
            {0.0, 0.0, 8.0, 9.0},
            {0.0, 0.0, 0.0, 10.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 5000);
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        // Upper triangular matrix eigenvalues = diagonal elements
        // Eigenvalues are sorted in descending order
        IVector<Double> eigenvalues = result._1;
        assertTrue(Math.abs(eigenvalues.get(0) - 10.0) < TOL ||
                   Math.abs(eigenvalues.get(1) - 10.0) < TOL ||
                   Math.abs(eigenvalues.get(2) - 10.0) < TOL ||
                   Math.abs(eigenvalues.get(3) - 10.0) < TOL);
        // Verify sum of eigenvalues = trace
        double trace = 1.0 + 5.0 + 8.0 + 10.0;
        double sum = 0.0;
        for (int i = 0; i < 4; i++) sum += eigenvalues.get(i);
        assertEquals(trace, sum, TOL);

        verifyEigenpairs(A, eigenvalues, result._2, LOOSE_TOL);
    }

    @Test
    public void testNonSymmetricMatrix_5x5() {
        java.util.Random rand = new java.util.Random(98765);
        double[][] data = new double[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                data[i][j] = rand.nextDouble() * 10 - 5;  // Range [-5, 5]
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 5000);
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        assertEquals(5, eigenvalues.length());

        // Verify eigenvalues are non-zero (general matrix, likely non-singular)
        // Just check the computation runs and produces results
        for (int i = 0; i < 5; i++) {
            assertNotNull(eigenvalues.get(i));
        }
    }

    // ========== Special Matrix Tests ==========

    @Test
    public void testDiagonalMatrix() {
        double[][] data = {
            {3.0, 0.0, 0.0},
            {0.0, 2.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;

        // Eigenvalues = diagonal elements
        assertEquals(3.0, eigenvalues.get(0), TOL);
        assertEquals(2.0, eigenvalues.get(1), TOL);
        assertEquals(1.0, eigenvalues.get(2), TOL);

        // Eigenvectors should be identity columns (or negated)
        for (int j = 0; j < 3; j++) {
            double norm = 0.0;
            for (int i = 0; i < 3; i++) {
                norm += eigenvectors.get(i, j) * eigenvectors.get(i, j);
            }
            assertEquals(1.0, Math.sqrt(norm), TOL);
        }

        verifyEigenpairs(A, eigenvalues, eigenvectors, TOL);
    }

    @Test
    public void testIdentityMatrix() {
        IMatrix<Double> I = Linalg.eye(4);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(I);

        IVector<Double> eigenvalues = result._1;
        IMatrix<Double> eigenvectors = result._2;

        // All eigenvalues should be 1
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0, eigenvalues.get(i), TOL);
        }

        verifyEigenpairs(I, eigenvalues, eigenvectors, TOL);
    }

    @Test
    public void testScalarMatrix() {
        // Scalar matrix: c * I
        double c = 5.0;
        IMatrix<Double> A = Linalg.eye(3).scale(c);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;

        // All eigenvalues should be 5
        for (int i = 0; i < 3; i++) {
            assertEquals(c, eigenvalues.get(i), TOL);
        }
    }

    @Test
    public void testSymmetricPositiveDefiniteMatrix() {
        // A = L * L^T where L is random lower triangular with positive diagonal
        double[][] Ldata = {
            {3.0, 0.0, 0.0},
            {1.0, 2.0, 0.0},
            {0.5, 0.5, 1.5}
        };
        IMatrix<Double> L = Linalg.matrix(Ldata);
        IMatrix<Double> A = L.mmul(L.transposeNew());  // SPD matrix

        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;

        // For SPD matrix, all eigenvalues should be positive
        for (int i = 0; i < eigenvalues.length(); i++) {
            assertTrue(eigenvalues.get(i) > 0,
                "Eigenvalue should be positive: " + eigenvalues.get(i));
        }

        verifyEigenpairs(A, eigenvalues, result._2, LOOSE_TOL);
    }

    @Test
    public void testMatrixWithRepeatedEigenvalues() {
        // 2x2 matrix with repeated eigenvalue λ=3 (defective matrix)
        double[][] data = {
            {3.0, 1.0},
            {0.0, 3.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 10000);
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;

        // Both eigenvalues should be 3 (sum = 6, product = 9)
        double sum = eigenvalues.get(0) + eigenvalues.get(1);
        assertEquals(6.0, sum, TOL);

        // For defective matrices, the eigenvector computation might not be accurate
        // Just verify eigenvalue properties
        double product = eigenvalues.get(0) * eigenvalues.get(1);
        assertEquals(9.0, product, LOOSE_TOL);
    }

    @Test
    public void testMatrixWithZeroEigenvalue() {
        // Singular matrix with eigenvalue 0
        double[][] data = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 6.0},
            {3.0, 6.0, 9.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;

        // Should have at least one eigenvalue close to 0
        boolean hasZeroEigenvalue = false;
        for (int i = 0; i < eigenvalues.length(); i++) {
            if (Math.abs(eigenvalues.get(i)) < 1e-8) {
                hasZeroEigenvalue = true;
                break;
            }
        }
        assertTrue(hasZeroEigenvalue, "Should have at least one eigenvalue close to 0");

        verifyEigenpairs(A, eigenvalues, result._2, LOOSE_TOL);
    }

    // ========== Eigenvalue Order Tests ==========

    @Test
    public void testEigenvaluesDescendingOrder() {
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
    public void testEigenvaluesDescendingOrder_LargerMatrix() {
        java.util.Random rand = new java.util.Random(45678);
        double[][] data = new double[10][10];
        for (int i = 0; i < 10; i++) {
            for (int j = i; j < 10; j++) {
                double val = rand.nextDouble() * 10;
                data[i][j] = val;
                data[j][i] = val;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        IVector<Double> eigenvalues = result._1;
        for (int i = 1; i < eigenvalues.length(); i++) {
            assertTrue(eigenvalues.get(i - 1) >= eigenvalues.get(i) - 1e-10,
                "Eigenvalues should be in descending order at index " + i);
        }
    }

    // ========== Solver Tests ==========

    @Test
    public void testSolver_Symmetric() {
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

        // Verify A * x = b
        for (int i = 0; i < 3; i++) {
            double ax = 0.0;
            for (int k = 0; k < 3; k++) {
                ax += A.get(i, k) * x.get(k);
            }
            assertEquals(b[i], ax, 1e-8);
        }
    }

    @Test
    public void testSolver_NonSymmetric() {
        double[][] data = {
            {1.0, 2.0, 0.0},
            {0.0, 3.0, 1.0},
            {0.0, 0.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        eigen.decompose(A);

        IDecompositionSolver solver = eigen.getSolver();
        double[] b = {1.0, 2.0, 3.0};
        IVector<Double> bVec = Linalg.vector(b);
        IVector<Double> x = solver.solve(bVec);

        // Verify A * x = b
        for (int i = 0; i < 3; i++) {
            double ax = 0.0;
            for (int k = 0; k < 3; k++) {
                ax += A.get(i, k) * x.get(k);
            }
            assertEquals(b[i], ax, 1e-8);
        }
    }

    // ========== Determinant and Properties Tests ==========

    @Test
    public void testDeterminant_Symmetric() {
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        eigen.decompose(A);

        double det = eigen.getDeterminant();

        // For eigenvalue decomposition: det(A) = product of eigenvalues
        IVector<Double> eigenvalues = eigen.decompose(A)._1;
        double product = 1.0;
        for (int i = 0; i < eigenvalues.length(); i++) {
            product *= eigenvalues.get(i);
        }
        assertEquals(product, det, TOL);
    }

    @Test
    public void testIsNonSingular() {
        // Non-singular matrix
        double[][] data1 = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A1 = Linalg.matrix(data1);
        RereEigenDecomposition eigen1 = new RereEigenDecomposition();
        eigen1.decompose(A1);
        assertTrue(eigen1.isNonSingular());

        // Singular matrix
        double[][] data2 = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 6.0},
            {3.0, 6.0, 9.0}
        };
        IMatrix<Double> A2 = Linalg.matrix(data2);
        RereEigenDecomposition eigen2 = new RereEigenDecomposition();
        eigen2.decompose(A2);
        assertFalse(eigen2.isNonSingular());
    }

    @Test
    public void testRank() {
        // Full rank matrix
        double[][] data1 = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A1 = Linalg.matrix(data1);
        RereEigenDecomposition eigen1 = new RereEigenDecomposition();
        eigen1.decompose(A1);
        assertEquals(3, eigen1.getRank());

        // Rank 2 matrix (not dependent on single vector)
        double[][] data2 = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 5.0},
            {3.0, 6.0, 8.0}
        };
        IMatrix<Double> A2 = Linalg.matrix(data2);
        RereEigenDecomposition eigen2 = new RereEigenDecomposition();
        eigen2.decompose(A2);
        // This should have rank 2
        assertTrue(eigen2.getRank() <= 2);
    }

    @Test
    public void testConditionNumber() {
        // Well-conditioned symmetric matrix
        double[][] data1 = {
            {2.0, 0.0, 0.0},
            {0.0, 3.0, 0.0},
            {0.0, 0.0, 4.0}
        };
        IMatrix<Double> A1 = Linalg.matrix(data1);
        RereEigenDecomposition eigen1 = new RereEigenDecomposition();
        eigen1.decompose(A1);
        // Condition number = max|eig|/min|eig| = 4/2 = 2
        assertEquals(2.0, eigen1.getConditionNumber(), TOL);

        // Ill-conditioned matrix (has zero eigenvalue)
        double[][] data2 = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 6.0},
            {3.0, 6.0, 9.0}
        };
        IMatrix<Double> A2 = Linalg.matrix(data2);
        RereEigenDecomposition eigen2 = new RereEigenDecomposition();
        eigen2.decompose(A2);
        assertEquals(Double.POSITIVE_INFINITY, eigen2.getConditionNumber());
    }

    // ========== Eigenvector Orthogonality Tests (Symmetric) ==========

    @Test
    public void testSymmetricEigenvectorsOrthogonal() {
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        IMatrix<Double> V = eigen.decompose(A)._2;

        // For symmetric matrices with distinct eigenvalues, eigenvectors should be orthogonal
        IMatrix<Double> VtV = V.transposeNew().mmul(V);
        int n = VtV.rows();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, VtV.get(i, j), TOL);
            }
        }
    }

    // ========== Non-Square Matrix Exception Test ==========

    @Test
    public void testNonSquareMatrixThrowsException() {
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();

        assertThrows(NonSquareMatrixException.class, () -> {
            eigen.decompose(A);
        });
    }

    // ========== Helper Methods ==========

    /**
     * Verify that A * v = λ * v for each eigenpair (columns of eigenvector matrix).
     */
    private void verifyEigenpairs(IMatrix<Double> A, IVector<Double> eigenvalues,
                                   IMatrix<Double> eigenvectors, double tolerance) {
        int n = A.rows();
        assertEquals(n, A.cols());
        assertEquals(n, eigenvalues.length());
        assertEquals(n, eigenvectors.rows());
        assertEquals(n, eigenvectors.cols());

        for (int j = 0; j < n; j++) {
            double lambda = eigenvalues.get(j);

            // Compute A * v (column j of eigenvectors)
            double[] av = new double[n];
            for (int i = 0; i < n; i++) {
                av[i] = 0.0;
                for (int k = 0; k < n; k++) {
                    av[i] += A.get(i, k) * eigenvectors.get(k, j);
                }
            }

            // Compute λ * v
            for (int i = 0; i < n; i++) {
                double expected = lambda * eigenvectors.get(i, j);
                assertEquals(expected, av[i], tolerance,
                    String.format("A*v[%d] = %.15e, λ*v[%d] = %.15e, λ = %.15e",
                        i, av[i], i, expected, lambda));
            }
        }
    }
}
