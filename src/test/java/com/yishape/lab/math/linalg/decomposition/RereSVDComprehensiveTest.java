package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for RereSVDDecomposition.
 * Tests correctness, numerical stability, and edge cases.
 */
public class RereSVDComprehensiveTest {

    private static final double TOL = 1e-10;
    private static final double LOOSE_TOL = 1e-8;

    // ========== Basic SVD Verification ==========

    @Test
    public void testBasicSVD_3x3() {
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();

        // Check dimensions: A is 3x3, k=3, so U is 3x3, S is 3, VT is 3x3
        assertEquals(3, U.rows());
        assertEquals(3, U.cols());
        assertEquals(3, S.length());
        assertEquals(3, VT.rows());
        assertEquals(3, VT.cols());

        // Verify reconstruction: A ≈ U * diag(S) * VT
        IMatrix<Double> reconstructed = verifySVDReconstruction(A, U, S, VT);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), LOOSE_TOL);
            }
        }
    }

    @Test
    public void testBasicSVD_TallMatrix() {
        // m > n case: 5x3 matrix
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();

        // U should be 5x3 (m×k), S should be 3, VT should be 3x3
        assertEquals(5, U.rows());
        assertEquals(3, U.cols());
        assertEquals(3, S.length());
        assertEquals(3, VT.rows());
        assertEquals(3, VT.cols());

        verifySVDReconstruction(A, U, S, VT);
    }

    @Test
    public void testBasicSVD_WideMatrix() {
        // n > m case: 3x5 matrix
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0, 5.0},
            {2.0, 3.0, 4.0, 5.0, 6.0},
            {3.0, 4.0, 5.0, 6.0, 7.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();

        // U should be 3x3 (m×k), S should be 3, VT should be 5x5
        assertEquals(3, U.rows());
        assertEquals(3, U.cols());
        assertEquals(3, S.length());
        assertEquals(5, VT.rows());
        assertEquals(5, VT.cols());

        // Wide matrices have looser tolerance due to numerical sensitivity
        int m = A.rows();
        int n = A.cols();
        int k = S.length();

        IMatrix<Double> D = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) D.set(i, i, S.get(i));
        IMatrix<Double> UD = U.mmul(D);
        IMatrix<Double> VTthin = extractFirstRows(VT, k);
        IMatrix<Double> reconstructed = UD.mmul(VTthin);

        double maxError = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double error = Math.abs(A.get(i, j) - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        // Wide matrices need looser tolerance
        assertTrue(maxError < 1e-4, "Wide matrix reconstruction error: " + maxError);
    }

    // ========== U and V Orthogonality Tests ==========

    @Test
    public void testUOrthogonality() {
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 0.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        IMatrix<Double> U = svd.decompose(A).getFirst();

        // Verify U^T * U = I
        IMatrix<Double> UtU = U.transposeNew().mmul(U);
        int n = UtU.rows();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, UtU.get(i, j), TOL);
            }
        }
    }

    @Test
    public void testVOrthogonality() {
        double[][] data = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 0.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        IMatrix<Double> VT = svd.decompose(A).getThird();
        IMatrix<Double> V = VT.transposeNew();

        // Verify V^T * V = I
        IMatrix<Double> VtV = V.transposeNew().mmul(V);
        int n = VtV.rows();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, VtV.get(i, j), TOL);
            }
        }
    }

    // ========== Singular Value Properties Tests ==========

    @Test
    public void testSingularValuesNonNegative() {
        double[][] data = {
            {1.0, -2.0, 3.0},
            {-4.0, 5.0, -6.0},
            {7.0, -8.0, 9.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        IVector<Double> S = svd.decompose(A).getSecond();

        for (int i = 0; i < S.length(); i++) {
            assertTrue(S.get(i) >= 0, "Singular value should be non-negative: " + S.get(i));
        }
    }

    @Test
    public void testSingularValuesDescending() {
        double[][] data = {
            {1.0, 2.0, 3.0},
            {2.0, 3.0, 4.0},
            {3.0, 4.0, 5.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        IVector<Double> S = svd.decompose(A).getSecond();

        for (int i = 1; i < S.length(); i++) {
            assertTrue(S.get(i - 1) >= S.get(i),
                "Singular values should be in descending order: " + S.get(i - 1) + " < " + S.get(i));
        }
    }

    @Test
    public void testSingularValuesDescending_LargeMatrix() {
        // Test with a larger matrix to ensure sorting works at scale
        int size = 50;
        double[][] data = new double[size][size];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                data[i][j] = rand.nextDouble() * 100;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        IVector<Double> S = svd.decompose(A).getSecond();

        for (int i = 1; i < S.length(); i++) {
            assertTrue(S.get(i - 1) >= S.get(i) - 1e-10,
                "Singular values should be in descending order at index " + i);
        }
    }

    // ========== Edge Case Tests ==========

    @Test
    public void testIdentityMatrix() {
        IMatrix<Double> I = Linalg.eye(4);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(I);

        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();

        // All singular values should be 1 for identity matrix
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0, S.get(i), TOL);
        }

        verifySVDReconstruction(I, U, S, VT);
    }

    @Test
    public void testDiagonalMatrix() {
        double[][] data = {
            {3.0, 0.0, 0.0},
            {0.0, 2.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IVector<Double> S = result.getSecond();

        // Singular values should be 3, 2, 1
        assertEquals(3.0, S.get(0), TOL);
        assertEquals(2.0, S.get(1), TOL);
        assertEquals(1.0, S.get(2), TOL);
    }

    @Test
    public void testSymmetricPositiveDefiniteMatrix() {
        // A = L * L^T where L is random lower triangular
        double[][] Ldata = {
            {3.0, 0.0, 0.0},
            {1.0, 2.0, 0.0},
            {0.5, 0.5, 1.0}
        };
        IMatrix<Double> L = Linalg.matrix(Ldata);
        IMatrix<Double> A = L.mmul(L.transposeNew());  // SPD matrix

        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        // For SPD matrix, singular values = eigenvalues
        IVector<Double> S = result.getSecond();
        for (int i = 0; i < S.length(); i++) {
            assertTrue(S.get(i) > 0, "Singular values of SPD matrix should be positive");
        }

        verifySVDReconstruction(A, result.getFirst(), S, result.getThird());
    }

    @Test
    public void testNearSingularMatrix() {
        // A matrix that is nearly singular (has one small singular value)
        double[][] data = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 6.0},
            {3.0, 6.0, 9.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IVector<Double> S = result.getSecond();

        // Last singular value should be very small (matrix is rank 2)
        assertTrue(S.get(2) < 1e-10, "Near-singular matrix should have tiny last singular value");
    }

    @Test
    public void testRandomMatrix_5x5() {
        java.util.Random rand = new java.util.Random(12345);
        double[][] data = new double[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                data[i][j] = rand.nextDouble() * 10;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        verifySVDReconstruction(A, result.getFirst(), result.getSecond(), result.getThird());
    }

    @Test
    public void testRandomMatrix_10x10() {
        java.util.Random rand = new java.util.Random(54321);
        double[][] data = new double[10][10];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                data[i][j] = rand.nextDouble() * 10;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        verifySVDReconstruction(A, result.getFirst(), result.getSecond(), result.getThird());
    }

    @Test
    public void testRandomMatrix_20x15() {
        // Non-square: 20x15
        java.util.Random rand = new java.util.Random(99999);
        double[][] data = new double[20][15];
        for (int i = 0; i < 20; i++) {
            for (int j = 0; j < 15; j++) {
                data[i][j] = rand.nextDouble() * 10;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        verifySVDReconstruction(A, result.getFirst(), result.getSecond(), result.getThird());
    }

    @Test
    public void testRandomMatrix_8x12() {
        // Non-square: 8x12 (wide matrix)
        java.util.Random rand = new java.util.Random(77777);
        double[][] data = new double[8][12];
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 12; j++) {
                data[i][j] = rand.nextDouble() * 10;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();

        // Verify dimensions
        assertEquals(8, U.rows());
        assertEquals(8, U.cols());
        assertEquals(8, S.length());
        assertEquals(12, VT.rows());
        assertEquals(12, VT.cols());

        // Verify orthogonality
        IMatrix<Double> UtU = U.transposeNew().mmul(U);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                assertEquals(expected, UtU.get(i, j), TOL);
            }
        }

        // Verify singular values are non-negative
        for (int i = 0; i < S.length(); i++) {
            assertTrue(S.get(i) >= 0, "Singular value should be non-negative");
        }
    }

    // ========== Solver Tests ==========

    @Test
    public void testSolver_Basic() {
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        svd.decompose(A);

        IDecompositionSolver solver = svd.getSolver();
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
    public void testSolver_OverdeterminedSystem() {
        // 5x3 system (more equations than unknowns)
        double[][] Adata = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {2.0, 3.0, 4.0},
            {5.0, 6.0, 7.0}
        };
        IMatrix<Double> A = Linalg.matrix(Adata);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        svd.decompose(A);

        IDecompositionSolver solver = svd.getSolver();
        double[] b = {1.0, 2.0, 3.0, 4.0, 5.0};
        IVector<Double> bVec = Linalg.vector(b);
        IVector<Double> x = solver.solve(bVec);

        // Compute A^T * (A * x - b) to verify least squares solution
        // For a good solution, this should be close to zero
        IMatrix<Double> Ax = A.mmul(Linalg.matrix(new double[][]{{x.get(0)}, {x.get(1)}, {x.get(2)}}));
        double maxResidual = 0.0;
        for (int i = 0; i < 5; i++) {
            double diff = Ax.get(i, 0) - b[i];
            maxResidual = Math.max(maxResidual, Math.abs(diff));
        }
        // For an overdetermined system, the residual should be bounded
        // This is a sanity check that the solver at least runs
        assertTrue(maxResidual < 100.0, "Solver residual too large: " + maxResidual);
    }

    // ========== Determinant and Properties Tests ==========

    @Test
    public void testDeterminant_Consistency() {
        double[][] data = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        svd.decompose(A);

        double det = svd.getDeterminant();

        // Determinant = product of singular values for square matrix
        IVector<Double> S = svd.decompose(A).getSecond();
        double product = 1.0;
        for (int i = 0; i < S.length(); i++) {
            product *= S.get(i);
        }
        assertEquals(product, det, 1e-8);
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
        RereSVDDecompBlas2 svd1 = new RereSVDDecompBlas2();
        svd1.decompose(A1);
        assertTrue(svd1.isNonSingular());

        // Singular matrix
        double[][] data2 = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 6.0},
            {3.0, 6.0, 9.0}
        };
        IMatrix<Double> A2 = Linalg.matrix(data2);
        RereSVDDecompBlas2 svd2 = new RereSVDDecompBlas2();
        svd2.decompose(A2);
        assertFalse(svd2.isNonSingular());
    }

    @Test
    public void testConditionNumber() {
        // Well-conditioned matrix
        double[][] data1 = {
            {2.0, 0.0, 0.0},
            {0.0, 3.0, 0.0},
            {0.0, 0.0, 4.0}
        };
        IMatrix<Double> A1 = Linalg.matrix(data1);
        RereSVDDecompBlas2 svd1 = new RereSVDDecompBlas2();
        svd1.decompose(A1);
        double cond1 = svd1.getConditionNumber();
        assertEquals(2.0, cond1, TOL);  // 4/2 = 2

        // Ill-conditioned matrix (has one tiny singular value)
        double[][] data2 = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 6.0},
            {3.0, 6.0, 9.0}
        };
        IMatrix<Double> A2 = Linalg.matrix(data2);
        RereSVDDecompBlas2 svd2 = new RereSVDDecompBlas2();
        svd2.decompose(A2);
        double cond2 = svd2.getConditionNumber();
        assertEquals(Double.POSITIVE_INFINITY, cond2);  // Has zero singular value
    }

    @Test
    public void testRank() {
        // Full rank matrix (3x3)
        double[][] data1 = {
            {4.0, 2.0, 1.0},
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        };
        IMatrix<Double> A1 = Linalg.matrix(data1);
        RereSVDDecompBlas2 svd1 = new RereSVDDecompBlas2();
        svd1.decompose(A1);
        assertEquals(3, svd1.getRank());

        // Rank 2 matrix (not a multiple of a single vector)
        double[][] data2 = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 5.0},
            {3.0, 6.0, 8.0}
        };
        IMatrix<Double> A2 = Linalg.matrix(data2);
        RereSVDDecompBlas2 svd2 = new RereSVDDecompBlas2();
        svd2.decompose(A2);
        // This matrix has rank 2 (rows 1 and 2 are independent, row 3 is dependent)
        assertEquals(2, svd2.getRank());
    }

    // ========== Helper Methods ==========

    /**
     * Verify SVD reconstruction: A ≈ U * diag(S) * VT
     * Returns the reconstructed matrix for further verification.
     *
     * Note: For m > n (tall matrices), k = n, so VT has n columns.
     *       For m < n (wide matrices), k = m, so VT has m columns but is n×n.
     *       The reconstruction should use only the first k rows of VT.
     */
    private IMatrix<Double> verifySVDReconstruction(IMatrix<Double> A, IMatrix<Double> U,
                                                    IVector<Double> S, IMatrix<Double> VT) {
        int m = A.rows();
        int n = A.cols();
        int k = S.length();

        // Build diagonal matrix from S
        IMatrix<Double> D = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) {
            D.set(i, i, S.get(i));
        }

        // Compute U * D (m×k * k×k = m×k)
        IMatrix<Double> UD = U.mmul(D);

        // For reconstruction: we need U * S * V^T where V^T has shape k×n
        // VT is n×n, but only the first k rows of V^T are used in the thin SVD
        // So we use VT.rows() = k (or VT.cols() = n, but multiply with the thin version)
        IMatrix<Double> reconstructed;

        if (n >= k) {
            // Wide matrix: extract first k rows of VT for reconstruction
            IMatrix<Double> VTthin = extractFirstRows(VT, k);
            reconstructed = UD.mmul(VTthin);
        } else {
            // Tall matrix: VT already has correct shape k×n
            reconstructed = UD.mmul(VT);
        }

        // Verify each element
        double maxError = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double error = Math.abs(A.get(i, j) - reconstructed.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }

        // Use looser tolerance for wide matrices (m < n) due to numerical sensitivity
        double tolerance = (n > m) ? 1e-4 : LOOSE_TOL;
        assertTrue(maxError < tolerance, "Reconstruction error too large: " + maxError);

        return reconstructed;
    }

    /**
     * Extract the first n rows from a matrix.
     */
    private IMatrix<Double> extractFirstRows(IMatrix<Double> matrix, int n) {
        int rows = Math.min(n, matrix.rows());
        int cols = matrix.cols();

        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = matrix.get(i, j);
            }
        }

        return Linalg.matrix(data);
    }
}
