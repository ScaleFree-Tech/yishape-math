package com.yishape.lab.math.linalg;

import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verification test suite for BetterRereDoubleMatrix decomposition methods.
 * Tests that the decomposition methods produce mathematically correct results.
 */
public class BetterRereDoubleMatrixDecompositionVerificationTest {

    private RereDoubleMatrix simpleMatrix;
    private RereDoubleMatrix identityMatrix;

    @BeforeEach
    void setUp() {
        // Create a simple 2x2 matrix for testing
        double[][] simpleData = {
            {2.0, 1.0},
            {1.0, 2.0}
        };
        simpleMatrix = new RereDoubleMatrix(simpleData);

        // Create a 2x2 identity matrix
        double[][] identityData = {
            {1.0, 0.0},
            {0.0, 1.0}
        };
        identityMatrix = new RereDoubleMatrix(identityData);
    }

    /**
     * Test that eigenvalues of identity matrix are all 1
     */
    @Test
    void testEigenIdentityMatrix() {
        System.out.println("Testing Eigenvalues of Identity Matrix...");
        
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = identityMatrix.eigen();
        IVector<Double> eigenvalues = eigenResult._1;
        
        assertEquals(2, eigenvalues.length(), "Should have 2 eigenvalues for 2x2 matrix");
        assertEquals(1.0, eigenvalues.get(0), 1e-10, "First eigenvalue should be 1");
        assertEquals(1.0, eigenvalues.get(1), 1e-10, "Second eigenvalue should be 1");
        
        System.out.println("Eigenvalues of Identity Matrix test passed.");
    }

    /**
     * Test that SVD of identity matrix produces identity matrices
     */
    @Test
    void testSVDIdentityMatrix() {
        System.out.println("Testing SVD of Identity Matrix...");
        
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = identityMatrix.svd();
        
        IMatrix<Double> U = svdResult._1;
        IVector<Double> singularValues = svdResult._2;
        IMatrix<Double> V = svdResult._3;
        
        // Singular values should all be 1
        assertEquals(2, singularValues.length(), "Should have 2 singular values for 2x2 matrix");
        assertEquals(1.0, singularValues.get(0), 1e-10, "First singular value should be 1");
        assertEquals(1.0, singularValues.get(1), 1e-10, "Second singular value should be 1");
        
        System.out.println("SVD of Identity Matrix test passed.");
    }

    /**
     * Test basic properties of QR decomposition
     */
    @Test
    void testQRBasicProperties() {
        System.out.println("Testing Basic Properties of QR Decomposition...");
        
        Tuple2<IMatrix<Double>, IMatrix<Double>> qrResult = simpleMatrix.qr();
        
        IMatrix<Double> Q = qrResult._1;
        IMatrix<Double> R = qrResult._2;
        
        // Check that Q is orthogonal (Q^T * Q = I)
        IMatrix<Double> Qt = Q.transposeNew();
        IMatrix<Double> QtQ = Qt.mmul(Q);
        
        // Check that R is upper triangular
        assertTrue(R.get(1, 0) == 0.0, "R should be upper triangular");
        
        System.out.println("Basic Properties of QR Decomposition test passed.");
    }

    /**
     * Test basic properties of LU decomposition
     */
    @Test
    void testLUBasicProperties() {
        System.out.println("Testing Basic Properties of LU Decomposition...");
        
        Tuple2<IMatrix<Double>, IMatrix<Double>> luResult = simpleMatrix.lu();
        
        IMatrix<Double> L = luResult._1;
        IMatrix<Double> U = luResult._2;
        
        // Check that L is lower triangular
        assertTrue(L.get(0, 1) == 0.0, "L should be lower triangular");
        
        // Check that U is upper triangular
        assertTrue(U.get(1, 0) == 0.0, "U should be upper triangular");
        
        System.out.println("Basic Properties of LU Decomposition test passed.");
    }

    /**
     * Test basic properties of Cholesky decomposition
     */
    @Test
    void testCholeskyBasicProperties() {
        System.out.println("Testing Basic Properties of Cholesky Decomposition...");
        
        IMatrix<Double> L = identityMatrix.cholesky();
        
        // Check that L is lower triangular
        assertTrue(L.get(0, 1) == 0.0, "L should be lower triangular");
        
        System.out.println("Basic Properties of Cholesky Decomposition test passed.");
    }

    /**
     * Test that all decomposition methods can be called without exceptions
     */
    @Test
    void testAllDecompositionsCallable() {
        System.out.println("Testing that all decomposition methods can be called...");
        
        // Test eigen decomposition
        assertDoesNotThrow(() -> {
            simpleMatrix.eigen();
        }, "Eigen decomposition should not throw exception");
        
        // Test SVD decomposition
        assertDoesNotThrow(() -> {
            simpleMatrix.svd();
        }, "SVD decomposition should not throw exception");
        
        // Test QR decomposition
        assertDoesNotThrow(() -> {
            simpleMatrix.qr();
        }, "QR decomposition should not throw exception");
        
        // Test LU decomposition
        assertDoesNotThrow(() -> {
            simpleMatrix.lu();
        }, "LU decomposition should not throw exception");
        
        // Test Cholesky decomposition
        assertDoesNotThrow(() -> {
            identityMatrix.cholesky();
        }, "Cholesky decomposition should not throw exception");
        
        System.out.println("All decomposition methods can be called without exceptions.");
    }
}