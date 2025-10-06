package com.yishape.lab.math.linalg;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for matrix norms and solving functionality
 */
public class MatrixNormsAndSolveTest {

    private IMatrix<Double> testMatrix;
    private IMatrix<Double> lowerTriangularMatrix;
    private IMatrix<Double> upperTriangularMatrix;

    @BeforeEach
    void setUp() {
        // Create a 3x3 test matrix
        double[][] testData = {
            {4.0, 12.0, -16.0},
            {12.0, 37.0, -43.0},
            {-16.0, -43.0, 98.0}
        };
        testMatrix = Linalg.matrix(testData);

        // Create a lower triangular matrix for forward solve testing
        double[][] lowerTriData = {
            {2.0, 0.0, 0.0},
            {1.0, 3.0, 0.0},
            {2.0, 1.0, 1.0}
        };
        lowerTriangularMatrix = Linalg.matrix(lowerTriData);

        // Create an upper triangular matrix for backward solve testing
        double[][] upperTriData = {
            {2.0, 1.0, 2.0},
            {0.0, 3.0, 1.0},
            {0.0, 0.0, 1.0}
        };
        upperTriangularMatrix = Linalg.matrix(upperTriData);
    }

    /**
     * Test Frobenius norm calculation
     */
    @Test
    void testFrobeniusNorm() {
        System.out.println("Testing Frobenius norm calculation...");
        
        double frobeniusNorm = testMatrix.frobeniusNorm().doubleValue();
        // Calculate expected value manually: sqrt(4^2 + 12^2 + (-16)^2 + 12^2 + 37^2 + (-43)^2 + (-16)^2 + (-43)^2 + 98^2)
        double expected = Math.sqrt(16 + 144 + 256 + 144 + 1369 + 1849 + 256 + 1849 + 9604);
        
        assertEquals(expected, frobeniusNorm, 1e-10, "Frobenius norm should match expected value");
        
        System.out.println("Frobenius norm test passed. Norm: " + frobeniusNorm);
    }

    /**
     * Test entrywise L1 norm calculation (sum of absolute values of all elements)
     */
    @Test
    void testEntrywiseL1Norm() {
        System.out.println("Testing entrywise L1 norm calculation...");
        
        double l1Norm = testMatrix.norm1().doubleValue();
        // Calculate expected value: sum of absolute values of all elements
        // |4| + |12| + |-16| + |12| + |37| + |-43| + |-16| + |-43| + |98|
        double expected = 4 + 12 + 16 + 12 + 37 + 43 + 16 + 43 + 98;
        
        assertEquals(expected, l1Norm, 1e-10, "Entrywise L1 norm should match expected value");
        
        System.out.println("Entrywise L1 norm test passed. Norm: " + l1Norm);
    }

    /**
     * Test induced L1 norm calculation (maximum column sum)
     */
    @Test
    void testInducedL1Norm() {
        System.out.println("Testing induced L1 norm calculation...");
        
        double l1Norm = testMatrix.normL1().doubleValue();
        // Calculate expected value: max column sum of absolute values
        // Col 1: |4| + |12| + |-16| = 32
        // Col 2: |12| + |37| + |-43| = 92
        // Col 3: |-16| + |-43| + |98| = 157
        double expected = 157.0;
        
        assertEquals(expected, l1Norm, 1e-10, "Induced L1 norm should match expected value");
        
        System.out.println("Induced L1 norm test passed. Norm: " + l1Norm);
    }

    /**
     * Test infinity norm calculation (maximum row sum)
     */
    @Test
    void testInfinityNorm() {
        System.out.println("Testing infinity norm calculation...");
        
        double infNorm = testMatrix.normInf().doubleValue();
        // Calculate expected value: max row sum of absolute values
        // Row 1: |4| + |12| + |-16| = 32
        // Row 2: |12| + |37| + |-43| = 92
        // Row 3: |-16| + |-43| + |98| = 157
        double expected = 157.0;
        
        assertEquals(expected, infNorm, 1e-10, "Infinity norm should match expected value");
        
        System.out.println("Infinity norm test passed. Norm: " + infNorm);
    }

    /**
     * Test L2 norm calculation
     */
    @Test
    void testL2Norm() {
        System.out.println("Testing L2 norm calculation...");
        
        double l2Norm = testMatrix.norm2().doubleValue();
        // This should be the largest singular value
        // We'll just verify it's a reasonable positive value
        assertTrue(l2Norm > 0, "L2 norm should be positive");
        // For this positive definite matrix, it should be larger than the largest diagonal element
        assertTrue(l2Norm > 98, "L2 norm should be larger than largest diagonal element");
        
        System.out.println("L2 norm test passed. Norm: " + l2Norm);
    }

    /**
     * Test forward solve functionality
     */
    @Test
    void testForwardSolve() {
        System.out.println("Testing forward solve functionality...");
        
        // Create right-hand side matrix
        double[][] bData = {
            {1.0, 2.0},
            {2.0, 1.0},
            {3.0, 0.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Solve LX = B
        IMatrix<Double> X = Linalg.forwardSolve(lowerTriangularMatrix, B);
        
        assertNotNull(X, "Solution matrix should not be null");
        assertEquals(3, X.rows(), "Solution should have 3 rows");
        assertEquals(2, X.cols(), "Solution should have 2 columns");
        
        // Verify solution by checking LX = B
        IMatrix<Double> result = lowerTriangularMatrix.mmul(X);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(bData[i][j], result.get(i, j).doubleValue(), 1e-10, 
                    "Solution should satisfy LX = B at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Forward solve test passed.");
    }

    /**
     * Test backward solve functionality
     */
    @Test
    void testBackwardSolve() {
        System.out.println("Testing backward solve functionality...");
        
        // Create right-hand side matrix
        double[][] bData = {
            {1.0, 2.0},
            {2.0, 1.0},
            {3.0, 0.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Solve UX = B
        IMatrix<Double> X = Linalg.backwardSolve(upperTriangularMatrix, B);
        
        assertNotNull(X, "Solution matrix should not be null");
        assertEquals(3, X.rows(), "Solution should have 3 rows");
        assertEquals(2, X.cols(), "Solution should have 2 columns");
        
        // Verify solution by checking UX = B
        IMatrix<Double> result = upperTriangularMatrix.mmul(X);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(bData[i][j], result.get(i, j).doubleValue(), 1e-10, 
                    "Solution should satisfy UX = B at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Backward solve test passed.");
    }

    /**
     * Test bidiagonal matrix creation
     */
    @Test
    void testBidiagonalMatrix() {
        System.out.println("Testing bidiagonal matrix creation...");
        
        double[] diagonal = {1.0, 2.0, 3.0};
        double[] superDiagonal = {4.0, 5.0};
        
        IMatrix<Double> bidiagonal = Linalg.bidiagonalMatrix(3, diagonal, superDiagonal);
        
        assertNotNull(bidiagonal, "Bidiagonal matrix should not be null");
        assertEquals(3, bidiagonal.rows(), "Bidiagonal matrix should have 3 rows");
        assertEquals(3, bidiagonal.cols(), "Bidiagonal matrix should have 3 columns");
        
        // Check diagonal elements
        assertEquals(1.0, bidiagonal.get(0, 0).doubleValue(), 1e-10, "Diagonal element (0,0) should be 1.0");
        assertEquals(2.0, bidiagonal.get(1, 1).doubleValue(), 1e-10, "Diagonal element (1,1) should be 2.0");
        assertEquals(3.0, bidiagonal.get(2, 2).doubleValue(), 1e-10, "Diagonal element (2,2) should be 3.0");
        
        // Check superdiagonal elements
        assertEquals(4.0, bidiagonal.get(0, 1).doubleValue(), 1e-10, "Superdiagonal element (0,1) should be 4.0");
        assertEquals(5.0, bidiagonal.get(1, 2).doubleValue(), 1e-10, "Superdiagonal element (1,2) should be 5.0");
        
        // Check that other elements are zero
        assertEquals(0.0, bidiagonal.get(1, 0).doubleValue(), 1e-10, "Element (1,0) should be 0.0");
        assertEquals(0.0, bidiagonal.get(2, 0).doubleValue(), 1e-10, "Element (2,0) should be 0.0");
        assertEquals(0.0, bidiagonal.get(2, 1).doubleValue(), 1e-10, "Element (2,1) should be 0.0");
        
        System.out.println("Bidiagonal matrix test passed.");
    }

    /**
     * Test tridiagonal matrix creation
     */
    @Test
    void testTridiagonalMatrix() {
        System.out.println("Testing tridiagonal matrix creation...");
        
        double[] subDiagonal = {1.0, 2.0};
        double[] diagonal = {3.0, 4.0, 5.0};
        double[] superDiagonal = {6.0, 7.0};
        
        IMatrix<Double> tridiagonal = Linalg.tridiagonalMatrix(3, subDiagonal, diagonal, superDiagonal);
        
        assertNotNull(tridiagonal, "Tridiagonal matrix should not be null");
        assertEquals(3, tridiagonal.rows(), "Tridiagonal matrix should have 3 rows");
        assertEquals(3, tridiagonal.cols(), "Tridiagonal matrix should have 3 columns");
        
        // Check diagonal elements
        assertEquals(3.0, tridiagonal.get(0, 0).doubleValue(), 1e-10, "Diagonal element (0,0) should be 3.0");
        assertEquals(4.0, tridiagonal.get(1, 1).doubleValue(), 1e-10, "Diagonal element (1,1) should be 4.0");
        assertEquals(5.0, tridiagonal.get(2, 2).doubleValue(), 1e-10, "Diagonal element (2,2) should be 5.0");
        
        // Check subdiagonal elements
        assertEquals(1.0, tridiagonal.get(1, 0).doubleValue(), 1e-10, "Subdiagonal element (1,0) should be 1.0");
        assertEquals(2.0, tridiagonal.get(2, 1).doubleValue(), 1e-10, "Subdiagonal element (2,1) should be 2.0");
        
        // Check superdiagonal elements
        assertEquals(6.0, tridiagonal.get(0, 1).doubleValue(), 1e-10, "Superdiagonal element (0,1) should be 6.0");
        assertEquals(7.0, tridiagonal.get(1, 2).doubleValue(), 1e-10, "Superdiagonal element (1,2) should be 7.0");
        
        // Check that other elements are zero
        assertEquals(0.0, tridiagonal.get(0, 2).doubleValue(), 1e-10, "Element (0,2) should be 0.0");
        assertEquals(0.0, tridiagonal.get(2, 0).doubleValue(), 1e-10, "Element (2,0) should be 0.0");
        
        System.out.println("Tridiagonal matrix test passed.");
    }
}