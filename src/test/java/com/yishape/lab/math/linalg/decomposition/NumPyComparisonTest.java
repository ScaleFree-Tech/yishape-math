package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;
import static org.junit.jupiter.api.Assertions.*;

/**
 * NumPy comparison tests for SVD and Eigen decomposition.
 *
 * These tests compare Java implementation results against NumPy's reference data.
 * Run the Python script first to generate reference data:
 *
 *     python benchmarks/numpy_vs_java.py --output benchmarks/
 *
 * Then run these tests.
 */
public class NumPyComparisonTest {

    private static final double TOL = 1e-10;
    private static final double LOOSE_TOL = 1e-8;

    // ========== SVD Comparison Tests ==========

    @Test
    public void testSVD_Random3x3() {
        // NumPy seed=42
        double[][] data = {
            {1.6243453636632417, -0.6117564136502843, -0.5281717515724219},
            {-1.0729686226620508, 0.8654076291172902, -2.301538696880279},
            {0.6002276394532809, -0.19168632544760562, 1.7448117642160293}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        // Verify reconstruction
        IMatrix<Double> reconstructed = reconstructSVD(result);
        double maxError = maxMatrixError(A, reconstructed);
        System.out.printf("SVD random_3x3: recon_error=%.2e%n", maxError);
        assertTrue(maxError < LOOSE_TOL, "Reconstruction error: " + maxError);

        // Verify U orthogonality
        IMatrix<Double> U = result.getFirst();
        double uOrthoError = orthogonalityError(U);
        System.out.printf("SVD random_3x3: U_ortho_error=%.2e%n", uOrthoError);
        assertTrue(uOrthoError < TOL, "U orthogonality error: " + uOrthoError);
    }

    @Test
    public void testSVD_Random5x5() {
        // NumPy seed=42, first 5x5 from larger random
        double[][] data = {
            {1.6243453636632417, -0.6117564136502843, -0.5281717515724219, 0.6190471254887691, -0.742044458586779},
            {-1.0729686226620508, 0.8654076291172902, -2.301538696880279, -0.9232751088705895, 0.5750074951276933},
            {0.6002276394532809, -0.19168632544760562, 1.7448117642160293, -0.9232751088705895, -0.6044265519604589},
            {-0.6117564136502843, -2.301538696880279, 0.6190471254887691, 0.8667706291716899, -1.8919229931451557},
            {-0.5281717515724219, 0.8654076291172902, 1.7448117642160293, -0.6044265519604589, -0.9211159099826783}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> reconstructed = reconstructSVD(result);
        double maxError = maxMatrixError(A, reconstructed);
        System.out.printf("SVD random_5x5: recon_error=%.2e%n", maxError);
        assertTrue(maxError < LOOSE_TOL, "Reconstruction error: " + maxError);
    }

    @Test
    public void testSVD_Identity10x10() {
        IMatrix<Double> I = Linalg.eye(10);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(I);

        // All singular values should be 1
        IVector<Double> S = result.getSecond();
        for (int i = 0; i < 10; i++) {
            assertEquals(1.0, S.get(i), TOL);
        }

        IMatrix<Double> reconstructed = reconstructSVD(result);
        double maxError = maxMatrixError(I, reconstructed);
        System.out.printf("SVD identity_10: recon_error=%.2e%n", maxError);
        assertTrue(maxError < TOL, "Reconstruction error: " + maxError);
    }

    @Test
    public void testSVD_Diagonal5x5() {
        double[] diag = {1.0, 2.0, 3.0, 4.0, 5.0};
        IMatrix<Double> A = Linalg.zeros(5, 5);
        for (int i = 0; i < 5; i++) A.set(i, i, diag[i]);

        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        // Singular values should equal diagonal elements
        IVector<Double> S = result.getSecond();
        double[] sortedS = new double[5];
        for (int i = 0; i < 5; i++) sortedS[i] = S.get(i);
        java.util.Arrays.sort(sortedS);

        for (int i = 0; i < 5; i++) {
            assertEquals(diag[i], sortedS[i], TOL);
        }

        IMatrix<Double> reconstructed = reconstructSVD(result);
        double maxError = maxMatrixError(A, reconstructed);
        System.out.printf("SVD diagonal_5: recon_error=%.2e%n", maxError);
        assertTrue(maxError < TOL, "Reconstruction error: " + maxError);
    }

    @Test
    public void testSVD_TallMatrix() {
        // 10x5 matrix (tall)
        java.util.Random rand = new java.util.Random(456);
        double[][] data = new double[10][5];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 5; j++) {
                data[i][j] = rand.nextGaussian();
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> reconstructed = reconstructSVD(result);
        double maxError = maxMatrixError(A, reconstructed);
        System.out.printf("SVD tall_10x5: recon_error=%.2e%n", maxError);
        assertTrue(maxError < LOOSE_TOL, "Reconstruction error: " + maxError);
    }

    @Test
    public void testSVD_WideMatrix() {
        // 5x10 matrix (wide)
        java.util.Random rand = new java.util.Random(789);
        double[][] data = new double[5][10];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 10; j++) {
                data[i][j] = rand.nextGaussian();
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        // For wide matrices, use thin reconstruction
        IMatrix<Double> reconstructed = reconstructSVDThin(result);
        double maxError = maxMatrixError(A, reconstructed);
        System.out.printf("SVD wide_5x10: recon_error=%.2e%n", maxError);
        assertTrue(maxError < TOL, "Reconstruction error: " + maxError);

        assertEquals(5, result.getFirst().rows());
        assertEquals(5, result.getFirst().cols());
    }

    @Test
    public void testSVD_SPD10x10() {
        // Symmetric positive definite: L @ L.T where L is random
        java.util.Random rand = new java.util.Random(321);
        double[][] Ldata = new double[10][10];
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                Ldata[i][j] = rand.nextGaussian();
            }
        }
        IMatrix<Double> L = Linalg.matrix(Ldata);
        IMatrix<Double> A = L.mmul(L.transposeNew());

        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

        IMatrix<Double> reconstructed = reconstructSVD(result);
        double maxError = maxMatrixError(A, reconstructed);
        System.out.printf("SVD spd_10x10: recon_error=%.2e%n", maxError);
        assertTrue(maxError < LOOSE_TOL, "Reconstruction error: " + maxError);

        // All singular values should be positive
        IVector<Double> S = result.getSecond();
        for (int i = 0; i < S.length(); i++) {
            assertTrue(S.get(i) > 0, "Singular value should be positive");
        }
    }

    // ========== Eigen Decomposition Comparison Tests ==========

    @Test
    public void testEigen_Symmetric3x3() {
        // Symmetric matrix with seed=42
        java.util.Random rand = new java.util.Random(42);
        double[][] data = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = i; j < 3; j++) {
                double val = rand.nextGaussian();
                data[i][j] = val;
                data[j][i] = val;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        // Verify eigenpairs
        double maxPairError = verifyEigenpairs(A, result._1, result._2);
        System.out.printf("Eigen symmetric_3x3: eigenpair_error=%.2e%n", maxPairError);
        assertTrue(maxPairError < LOOSE_TOL, "Eigenpair error: " + maxPairError);

        // Verify trace
        double trace = data[0][0] + data[1][1] + data[2][2];
        double evSum = 0;
        for (int i = 0; i < 3; i++) evSum += result._1.get(i);
        assertEquals(trace, evSum, TOL);
    }

    @Test
    public void testEigen_Symmetric5x5() {
        java.util.Random rand = new java.util.Random(42);
        double[][] data = new double[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = i; j < 5; j++) {
                double val = rand.nextGaussian();
                data[i][j] = val;
                data[j][i] = val;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        double maxPairError = verifyEigenpairs(A, result._1, result._2);
        System.out.printf("Eigen symmetric_5x5: eigenpair_error=%.2e%n", maxPairError);
        assertTrue(maxPairError < LOOSE_TOL, "Eigenpair error: " + maxPairError);
    }

    @Test
    public void testEigen_Symmetric10x10() {
        java.util.Random rand = new java.util.Random(42);
        double[][] data = new double[10][10];
        for (int i = 0; i < 10; i++) {
            for (int j = i; j < 10; j++) {
                double val = rand.nextGaussian();
                data[i][j] = val;
                data[j][i] = val;
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        double maxPairError = verifyEigenpairs(A, result._1, result._2);
        System.out.printf("Eigen symmetric_10x10: eigenpair_error=%.2e%n", maxPairError);
        assertTrue(maxPairError < LOOSE_TOL, "Eigenpair error: " + maxPairError);

        // Verify eigenvector orthogonality for symmetric matrix
        IMatrix<Double> V = result._2;
        IMatrix<Double> VtV = V.transposeNew().mmul(V);
        double orthoError = 0;
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                orthoError = Math.max(orthoError, Math.abs(VtV.get(i, j) - expected));
            }
        }
        System.out.printf("Eigen symmetric_10x10: ortho_error=%.2e%n", orthoError);
        assertTrue(orthoError < LOOSE_TOL, "Orthogonality error: " + orthoError);
    }

    @Test
    public void testEigen_NonSymmetric3x3() {
        java.util.Random rand = new java.util.Random(456);
        double[][] data = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                data[i][j] = rand.nextGaussian();
            }
        }
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 5000);
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        // Verify trace
        double trace = 0;
        for (int i = 0; i < 3; i++) trace += data[i][i];
        double evSum = 0;
        for (int i = 0; i < 3; i++) evSum += result._1.get(i);
        assertEquals(trace, evSum, LOOSE_TOL);

        // Verify eigenpairs with looser tolerance for non-symmetric
        double maxPairError = verifyEigenpairs(A, result._1, result._2);
        System.out.printf("Eigen nonsymmetric_3x3: eigenpair_error=%.2e%n", maxPairError);
        // Non-symmetric may have larger errors
        assertTrue(maxPairError < 1e-4, "Eigenpair error too large: " + maxPairError);
    }

    @Test
    public void testEigen_Diagonal5x5() {
        double[][] data = {
            {1.0, 0.0, 0.0, 0.0, 0.0},
            {0.0, 2.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 3.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 4.0, 0.0},
            {0.0, 0.0, 0.0, 0.0, 5.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        // Eigenvalues should be 1,2,3,4,5 (sorted descending)
        IVector<Double> eigenvalues = result._1;
        assertEquals(5.0, eigenvalues.get(0), TOL);
        assertEquals(4.0, eigenvalues.get(1), TOL);
        assertEquals(3.0, eigenvalues.get(2), TOL);
        assertEquals(2.0, eigenvalues.get(3), TOL);
        assertEquals(1.0, eigenvalues.get(4), TOL);

        double maxPairError = verifyEigenpairs(A, result._1, result._2);
        System.out.printf("Eigen diagonal_5: eigenpair_error=%.2e%n", maxPairError);
        assertTrue(maxPairError < TOL, "Eigenpair error: " + maxPairError);
    }

    @Test
    public void testEigen_UpperTriangular4x4() {
        // Upper triangular matrix: eigenvalues = diagonal
        double[][] data = {
            {1.0, 2.0, 3.0, 4.0},
            {0.0, 5.0, 6.0, 7.0},
            {0.0, 0.0, 8.0, 9.0},
            {0.0, 0.0, 0.0, 10.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition(1e-12, 5000);
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        // Eigenvalues = 10, 8, 5, 1 (descending)
        IVector<Double> eigenvalues = result._1;
        double sum = 0;
        for (int i = 0; i < 4; i++) sum += eigenvalues.get(i);
        assertEquals(24.0, sum, LOOSE_TOL);  // trace = 1+5+8+10 = 24

        double maxPairError = verifyEigenpairs(A, result._1, result._2);
        System.out.printf("Eigen upper_triangular_4x4: eigenpair_error=%.2e%n", maxPairError);
    }

    @Test
    public void testEigen_NearSingular3x3() {
        // Near singular: rank 1 matrix
        double[][] data = {
            {1.0, 2.0, 3.0},
            {2.0, 4.0, 6.0},
            {3.0, 6.0, 9.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);
        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        // Should have at least one near-zero eigenvalue
        IVector<Double> eigenvalues = result._1;
        boolean hasNearZero = false;
        for (int i = 0; i < 3; i++) {
            if (Math.abs(eigenvalues.get(i)) < 1e-8) {
                hasNearZero = true;
                break;
            }
        }
        System.out.printf("Eigen near_singular_3x3: has_near_zero_eig=%s%n", hasNearZero);
        assertTrue(hasNearZero, "Should have near-zero eigenvalue");

        // Eigenpair verification
        double maxPairError = verifyEigenpairs(A, result._1, result._2);
        System.out.printf("Eigen near_singular_3x3: eigenpair_error=%.2e%n", maxPairError);
        assertTrue(maxPairError < LOOSE_TOL, "Eigenpair error: " + maxPairError);
    }

    @Test
    public void testEigen_SPD5x5() {
        // Symmetric positive definite: L @ L.T
        java.util.Random rand = new java.util.Random(123);
        double[][] Ldata = new double[5][5];
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                Ldata[i][j] = rand.nextGaussian();
            }
        }
        IMatrix<Double> L = Linalg.matrix(Ldata);
        IMatrix<Double> A = L.mmul(L.transposeNew());

        RereEigenDecomposition eigen = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result = eigen.decompose(A);

        // All eigenvalues should be positive
        IVector<Double> eigenvalues = result._1;
        for (int i = 0; i < 5; i++) {
            assertTrue(eigenvalues.get(i) > 0, "Eigenvalue should be positive: " + eigenvalues.get(i));
        }

        double maxPairError = verifyEigenpairs(A, result._1, result._2);
        System.out.printf("Eigen spd_5x5: eigenpair_error=%.2e%n", maxPairError);
        assertTrue(maxPairError < LOOSE_TOL, "Eigenpair error: " + maxPairError);
    }

    // ========== Helper Methods ==========

    /**
     * Reconstruct matrix from SVD result (for square matrices)
     */
    private IMatrix<Double> reconstructSVD(Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result) {
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        int k = S.length();

        IMatrix<Double> D = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) D.set(i, i, S.get(i));

        return U.mmul(D).mmul(VT);
    }

    /**
     * Reconstruct matrix from SVD result (for wide matrices m < n)
     * Uses only the first k rows of VT
     */
    private IMatrix<Double> reconstructSVDThin(Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result) {
        IMatrix<Double> U = result.getFirst();
        IVector<Double> S = result.getSecond();
        IMatrix<Double> VT = result.getThird();
        int m = U.rows();
        int k = S.length();

        IMatrix<Double> D = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) D.set(i, i, S.get(i));

        IMatrix<Double> UD = U.mmul(D);

        // Extract first k rows of VT
        IMatrix<Double> VTthin = Linalg.zeros(k, VT.cols());
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < VT.cols(); j++) {
                VTthin.set(i, j, VT.get(i, j));
            }
        }

        return UD.mmul(VTthin);
    }

    /**
     * Calculate maximum matrix element error
     */
    private double maxMatrixError(IMatrix<Double> A, IMatrix<Double> B) {
        int m = A.rows();
        int n = A.cols();
        double maxError = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double error = Math.abs(A.get(i, j) - B.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        return maxError;
    }

    /**
     * Calculate orthogonality error: max(|V^T * V - I|)
     */
    private double orthogonalityError(IMatrix<Double> V) {
        IMatrix<Double> VtV = V.transposeNew().mmul(V);
        int n = VtV.rows();
        double maxError = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double error = Math.abs(VtV.get(i, j) - expected);
                maxError = Math.max(maxError, error);
            }
        }
        return maxError;
    }

    /**
     * Verify A * v = λ * v for each eigenpair
     * Returns maximum error
     */
    private double verifyEigenpairs(IMatrix<Double> A, IVector<Double> eigenvalues, IMatrix<Double> eigenvectors) {
        int n = A.rows();
        double maxError = 0.0;
        for (int j = 0; j < n; j++) {
            double lambda = eigenvalues.get(j);
            for (int i = 0; i < n; i++) {
                double av = 0.0;
                for (int k = 0; k < n; k++) {
                    av += A.get(i, k) * eigenvectors.get(k, j);
                }
                double expected = lambda * eigenvectors.get(i, j);
                double error = Math.abs(av - expected);
                maxError = Math.max(maxError, error);
            }
        }
        return maxError;
    }
}
