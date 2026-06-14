package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.NonSquareMatrixException;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for RereDoubleMatrix decomposition methods.
 * Tests eigen, SVD, QR, LU, and Cholesky decompositions with numerical verification.
 */
public class BetterRereDoubleMatrixDecompositionTest {

    private RereDoubleMatrix squareMatrix;
    private RereDoubleMatrix rectangularMatrix;
    private RereDoubleMatrix symmetricMatrix;
    private RereDoubleMatrix positiveDefiniteMatrix;
    private static final double TOL = 1e-9;

    @BeforeEach
    void setUp() {
        double[][] squareData = {
            {4.0, 2.0, 1.0},
            {2.0, 5.0, 3.0},
            {1.0, 3.0, 6.0}
        };
        squareMatrix = new RereDoubleMatrix(squareData);

        double[][] rectangularData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0},
            {7.0, 8.0, 9.0},
            {10.0, 11.0, 12.0}
        };
        rectangularMatrix = new RereDoubleMatrix(rectangularData);

        double[][] symmetricData = {
            {4.0, 2.0, 1.0},
            {2.0, 5.0, 3.0},
            {1.0, 3.0, 6.0}
        };
        symmetricMatrix = new RereDoubleMatrix(symmetricData);

        double[][] positiveDefiniteData = {
            {4.0, 2.0, 1.0},
            {2.0, 5.0, 3.0},
            {1.0, 3.0, 6.0}
        };
        positiveDefiniteMatrix = new RereDoubleMatrix(positiveDefiniteData);
    }

    /**
     * Verify reconstruction: A ≈ V*diag(λ)*V^T for symmetric matrices
     */
    private static void verifyEigenReconstruction(RereDoubleMatrix A,
            IVector<Double> eigenvalues, IMatrix<Double> eigenvectors) {
        int n = A.getRowNum();
        // Reconstruct: A ≈ Q * diag(λ) * Q^T
        IMatrix<Double> diag = Linalg.zeros(n, n);
        for (int i = 0; i < n; i++) diag.set(i, i, eigenvalues.get(i));
        IMatrix<Double> reconstructed = eigenvectors.mmul(diag).mmul(eigenvectors.transpose());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), TOL,
                    "Eigen reconstruction A[" + i + "," + j + "]");
            }
        }
    }

    /**
     * Verify SVD reconstruction: A = U * diag(σ) * V^T
     * Per ISVDDecomposition contract: svd() returns (U, S, V^T) where
     *   U is m×k (thin), S is length k, V^T is n×n (full)
     * So the third element IS V^T, and reconstruction is U @ diag(S) @ V^T directly.
     */
    private static void verifySVDReconstruction(RereDoubleMatrix A,
            IMatrix<Double> U, IVector<Double> sigma, IMatrix<Double> VT) {
        int m = A.getRowNum(), n = A.getColNum();
        int k = sigma.length();
        // diag is k×k; VT is n×n (full V^T from interface contract)
        // A = U(m×k) @ diag(k×k) @ VT(n×n) works because k=min(m,n)
        IMatrix<Double> diag = Linalg.zeros(k, k);
        for (int i = 0; i < k; i++) diag.set(i, i, sigma.get(i));
        // VT is already V^T, so reconstruction is U @ diag @ VT (no extra transpose!)
        IMatrix<Double> reconstructed = U.mmul(diag).mmul(VT);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), TOL,
                    "SVD reconstruction A[" + i + "," + j + "]");
            }
        }
    }

    /**
     * Verify QR reconstruction: A = Q * R
     */
    private static void verifyQRReconstruction(RereDoubleMatrix A,
            IMatrix<Double> Q, IMatrix<Double> R) {
        IMatrix<Double> reconstructed = Q.mmul(R);
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), TOL,
                    "QR reconstruction A[" + i + "," + j + "]");
            }
        }
    }

    /**
     * Verify LU reconstruction: A = L * U
     */
    private static void verifyLUReconstruction(RereDoubleMatrix A,
            IMatrix<Double> L, IMatrix<Double> U) {
        IMatrix<Double> reconstructed = L.mmul(U);
        for (int i = 0; i < A.getRowNum(); i++) {
            for (int j = 0; j < A.getColNum(); j++) {
                assertEquals(A.get(i, j), reconstructed.get(i, j), TOL,
                    "LU reconstruction A[" + i + "," + j + "]");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Eigen decomposition
    // ═══════════════════════════════════════════════════════════════

    @Test
    void testEigenDecomposition() {
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = symmetricMatrix.eigen();

        assertNotNull(eigenResult, "Eigen result should not be null");
        assertNotNull(eigenResult._1, "Eigenvalues should not be null");
        assertNotNull(eigenResult._2, "Eigenvectors should not be null");

        IVector<Double> eigenvalues = eigenResult._1;
        IMatrix<Double> eigenvectors = eigenResult._2;

        assertEquals(3, eigenvalues.length(), "Should have 3 eigenvalues for 3x3 matrix");
        assertEquals(3, eigenvectors.getRowNum(), "Eigenvector matrix should have 3 rows");
        assertEquals(3, eigenvectors.getColNum(), "Eigenvector matrix should have 3 columns");

        // Numerical verification: A ≈ V * diag(λ) * V^T
        verifyEigenReconstruction(symmetricMatrix, eigenvalues, eigenvectors);
    }

    @Test
    void testEigenDecompositionLargeMatrix() {
        double[][] largeSymmetricData = {
            {5.0, 2.0, 1.0, 0.0},
            {2.0, 6.0, 3.0, 1.0},
            {1.0, 3.0, 7.0, 2.0},
            {0.0, 1.0, 2.0, 8.0}
        };
        RereDoubleMatrix largeSymmetricMatrix = new RereDoubleMatrix(largeSymmetricData);

        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = largeSymmetricMatrix.eigen();

        assertNotNull(eigenResult, "Eigen result should not be null");
        assertEquals(4, eigenResult._1.length(), "Should have 4 eigenvalues for 4x4 matrix");
        assertEquals(4, eigenResult._2.getRowNum(), "Eigenvector matrix should have 4 rows");
        assertEquals(4, eigenResult._2.getColNum(), "Eigenvector matrix should have 4 columns");

        verifyEigenReconstruction(largeSymmetricMatrix, eigenResult._1, eigenResult._2);
    }

    // ═══════════════════════════════════════════════════════════════
    // SVD decomposition
    // ═══════════════════════════════════════════════════════════════

    @Test
    void testSVDDecomposition() {
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = rectangularMatrix.svd();

        assertNotNull(svdResult, "SVD result should not be null");
        assertNotNull(svdResult._1, "U matrix should not be null");
        assertNotNull(svdResult._2, "Singular values should not be null");
        assertNotNull(svdResult._3, "V^T matrix should not be null");

        IMatrix<Double> U = svdResult._1;
        IVector<Double> singularValues = svdResult._2;
        IMatrix<Double> VT = svdResult._3;  // Third element is V^T per ISVDDecomposition contract

        assertEquals(4, U.getRowNum(), "U matrix should have 4 rows");
        assertEquals(3, U.getColNum(), "U matrix should have min(m,n)=3 columns (thin SVD)");
        assertEquals(3, singularValues.length(), "Should have 3 singular values");
        assertEquals(3, VT.getRowNum(), "V^T matrix should have 3 rows (n=3)");
        assertEquals(3, VT.getColNum(), "V^T matrix should have 3 columns (n=3)");

        // Numerical verification: A = U * diag(σ) * V^T
        verifySVDReconstruction(rectangularMatrix, U, singularValues, VT);
    }

    @Test
    void testSVDSquareMatrix() {
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = squareMatrix.svd();

        assertNotNull(svdResult, "SVD result should not be null");
        assertEquals(3, svdResult._1.getRowNum(), "U matrix should have 3 rows");
        assertEquals(3, svdResult._1.getColNum(), "U matrix should have 3 columns");
        assertEquals(3, svdResult._2.length(), "Should have 3 singular values");
        assertEquals(3, svdResult._3.getRowNum(), "V^T matrix should have 3 rows (n=3)");
        assertEquals(3, svdResult._3.getColNum(), "V^T matrix should have 3 columns (n=3)");

        verifySVDReconstruction(squareMatrix, svdResult._1, svdResult._2, svdResult._3);
    }

    // ═══════════════════════════════════════════════════════════════
    // QR decomposition
    // ═══════════════════════════════════════════════════════════════

    @Test
    void testQRDecomposition() {
        Tuple2<IMatrix<Double>, IMatrix<Double>> qrResult = squareMatrix.qr();

        assertNotNull(qrResult, "QR result should not be null");
        assertNotNull(qrResult._1, "Q matrix should not be null");
        assertNotNull(qrResult._2, "R matrix should not be null");

        IMatrix<Double> Q = qrResult._1;
        IMatrix<Double> R = qrResult._2;

        assertEquals(3, Q.getRowNum(), "Q matrix should have 3 rows");
        assertEquals(3, Q.getColNum(), "Q matrix should have 3 columns");
        assertEquals(3, R.getRowNum(), "R matrix should have 3 rows");
        assertEquals(3, R.getColNum(), "R matrix should have 3 columns");

        // Numerical verification: A = Q * R
        verifyQRReconstruction(squareMatrix, Q, R);
    }

    // ═══════════════════════════════════════════════════════════════
    // LU decomposition
    // ═══════════════════════════════════════════════════════════════

    @Test
    void testLUDecomposition() {
        Tuple2<IMatrix<Double>, IMatrix<Double>> luResult = squareMatrix.lu();

        assertNotNull(luResult, "LU result should not be null");
        assertNotNull(luResult._1, "L matrix should not be null");
        assertNotNull(luResult._2, "U matrix should not be null");

        IMatrix<Double> L = luResult._1;
        IMatrix<Double> U = luResult._2;

        assertEquals(3, L.getRowNum(), "L matrix should have 3 rows");
        assertEquals(3, L.getColNum(), "L matrix should have 3 columns");
        assertEquals(3, U.getRowNum(), "U matrix should have 3 rows");
        assertEquals(3, U.getColNum(), "U matrix should have 3 columns");

        // Numerical verification: A = L * U
        verifyLUReconstruction(squareMatrix, L, U);
    }

    // ═══════════════════════════════════════════════════════════════
    // Cholesky decomposition
    // ═══════════════════════════════════════════════════════════════

    @Test
    void testCholeskyDecomposition() {
        IMatrix<Double> L = positiveDefiniteMatrix.cholesky();

        assertNotNull(L, "Cholesky result should not be null");
        assertEquals(3, L.getRowNum(), "L matrix should have 3 rows");
        assertEquals(3, L.getColNum(), "L matrix should have 3 columns");

        // Numerical verification: A = L * L^T
        IMatrix<Double> reconstructed = L.mmul(L.transpose());
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(positiveDefiniteMatrix.get(i, j), reconstructed.get(i, j), TOL,
                    "Cholesky reconstruction A[" + i + "," + j + "]");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Edge cases
    // ═══════════════════════════════════════════════════════════════

    @Test
    void testEigenDecompositionNonSquareMatrix() {
        assertThrows(NonSquareMatrixException.class, () -> {
            rectangularMatrix.eigen();
        }, "Eigen decomposition should throw exception for non-square matrix");
    }

    @Test
    void testDecompositionsIdentityMatrix() {
        double[][] identityData = {
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        };
        RereDoubleMatrix identityMatrix = new RereDoubleMatrix(identityData);

        // Eigen: eigenvalues should be [1,1,1]
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = identityMatrix.eigen();
        assertNotNull(eigenResult, "Eigen result should not be null");
        for (int i = 0; i < 3; i++) {
            assertEquals(1.0, eigenResult._1.get(i), TOL,
                "Identity eigenvalue[" + i + "] should be 1.0");
        }

        // QR: Q should be identity, R should be identity
        Tuple2<IMatrix<Double>, IMatrix<Double>> qrResult = identityMatrix.qr();
        assertNotNull(qrResult, "QR result should not be null");
        verifyQRReconstruction(identityMatrix, qrResult._1, qrResult._2);

        // SVD: singular values should be [1,1,1]
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = identityMatrix.svd();
        assertNotNull(svdResult, "SVD result should not be null");
        for (int i = 0; i < 3; i++) {
            assertEquals(1.0, svdResult._2.get(i), TOL,
                "Identity singular value[" + i + "] should be 1.0");
        }
        verifySVDReconstruction(identityMatrix, svdResult._1, svdResult._2, svdResult._3);
    }
}
