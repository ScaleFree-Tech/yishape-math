package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ICholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.NonPositiveDefiniteMatrixException;
import com.yishape.lab.math.linalg.decomposition.NonSymmetricMatrixException;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for RereCholeskyDecomposition implementation
 */
public class RereCholeskyDecompositionTest {
    
    private IMatrix<Double> testMatrix;
    private IMatrix<Double> singularMatrix;
    private IMatrix<Double> nonSymmetricMatrix;
    
    @BeforeEach
    void setUp() {
        // Create a 3x3 symmetric positive definite test matrix
        // This is a typical test matrix for Cholesky decomposition
        double[][] testData = {
            { 4.0, 12.0, -16.0},
            { 12.0, 37.0, -43.0},
            { -16.0, -43.0, 98.0}
        };
        testMatrix = Linalg.matrix(testData);
        
        // Create a singular matrix (not positive definite)
        double[][] singularData = {
            { 1.0, 2.0 },
            { 2.0, 4.0 }  // This matrix is singular (determinant = 0)
        };
        singularMatrix = Linalg.matrix(singularData);
        
        // Create a non-symmetric matrix
        double[][] nonSymmetricData = {
            { 1.0, 2.0 },
            { 3.0, 4.0 }  // Not symmetric since 2 != 3
        };
        nonSymmetricMatrix = Linalg.matrix(nonSymmetricData);
    }
    
    @Test
    void testCholeskyDecomposition() {
        System.out.println("Testing Cholesky Decomposition...");
        
        ICholeskyDecomposition choleskyDecomposition = new RereCholeskyDecomposition();
        IMatrix<Double> L = choleskyDecomposition.decompose(testMatrix);
        
        assertNotNull(L, "L matrix should not be null");
        assertEquals(3, L.rows(), "L matrix should have 3 rows");
        assertEquals(3, L.cols(), "L matrix should have 3 columns");
        
        // Check that L is lower triangular
        assertEquals(0.0, L.get(0, 1), 1e-10, "L should be lower triangular");
        assertEquals(0.0, L.get(0, 2), 1e-10, "L should be lower triangular");
        assertEquals(0.0, L.get(1, 2), 1e-10, "L should be lower triangular");
        
        System.out.println("Cholesky Decomposition test passed.");
    }
    
    @Test
    void testLMatrixRetrieval() {
        System.out.println("Testing L matrix retrieval...");
        
        ICholeskyDecomposition choleskyDecomposition = new RereCholeskyDecomposition();
        choleskyDecomposition.decompose(testMatrix);
        
        IMatrix<Double> L = choleskyDecomposition.getL();
        assertNotNull(L, "L matrix should not be null");
        assertEquals(3, L.rows(), "L matrix should have 3 rows");
        assertEquals(3, L.cols(), "L matrix should have 3 columns");
        
        // Check that L is lower triangular
        assertEquals(0.0, L.get(0, 1), 1e-10, "L should be lower triangular");
        assertEquals(0.0, L.get(0, 2), 1e-10, "L should be lower triangular");
        assertEquals(0.0, L.get(1, 2), 1e-10, "L should be lower triangular");
        
        System.out.println("L matrix retrieval test passed.");
    }
    
    @Test
    void testLTMatrixRetrieval() {
        System.out.println("Testing LT matrix retrieval...");
        
        ICholeskyDecomposition choleskyDecomposition = new RereCholeskyDecomposition();
        choleskyDecomposition.decompose(testMatrix);
        
        IMatrix<Double> LT = choleskyDecomposition.getLT();
        assertNotNull(LT, "LT matrix should not be null");
        assertEquals(3, LT.rows(), "LT matrix should have 3 rows");
        assertEquals(3, LT.cols(), "LT matrix should have 3 columns");
        
        // Check that LT is upper triangular
        assertEquals(0.0, LT.get(1, 0), 1e-10, "LT should be upper triangular");
        assertEquals(0.0, LT.get(2, 0), 1e-10, "LT should be upper triangular");
        assertEquals(0.0, LT.get(2, 1), 1e-10, "LT should be upper triangular");
        
        System.out.println("LT matrix retrieval test passed.");
    }
    
    @Test
    void testDeterminantCalculation() {
        System.out.println("Testing determinant calculation...");
        
        ICholeskyDecomposition choleskyDecomposition = new RereCholeskyDecomposition();
        choleskyDecomposition.decompose(testMatrix);
        
        double determinant = choleskyDecomposition.getDeterminant();
        // For our test matrix, the determinant should be 36
        assertEquals(36.0, determinant, 1e-10, "Determinant should be 36");
        
        System.out.println("Determinant calculation test passed. Determinant: " + determinant);
    }
    
    @Test
    void testSolverRetrieval() {
        System.out.println("Testing solver retrieval...");
        
        ICholeskyDecomposition choleskyDecomposition = new RereCholeskyDecomposition();
        choleskyDecomposition.decompose(testMatrix);
        
        IDecompositionSolver solver = choleskyDecomposition.getSolver();
        assertNotNull(solver, "Solver should not be null");
        assertTrue(solver.isNonSingular(), "Test matrix should be non-singular");
        
        System.out.println("Solver retrieval test passed.");
    }
    
    @Test
    void testLLTProperty() {
        System.out.println("Testing L * LT = A property...");
        
        ICholeskyDecomposition choleskyDecomposition = new RereCholeskyDecomposition();
        IMatrix<Double> L = choleskyDecomposition.decompose(testMatrix);
        IMatrix<Double> LT = choleskyDecomposition.getLT();
        
        // Check that L * LT = A
        IMatrix<Double> result = L.mmul(LT);
        
        // Compare matrices element by element
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(testMatrix.get(i, j), result.get(i, j), 1e-10, 
                    "L * LT should equal A at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("L * LT = A property test passed.");
    }
    
    @Test
    void testNonSymmetricMatrixHandling() {
        System.out.println("Testing non-symmetric matrix handling...");
        
        ICholeskyDecomposition choleskyDecomposition = new RereCholeskyDecomposition();
        
        assertThrows(NonSymmetricMatrixException.class, () -> {
            choleskyDecomposition.decompose(nonSymmetricMatrix);
        }, "Non-symmetric matrix should throw NonSymmetricMatrixException");
        
        System.out.println("Non-symmetric matrix handling test passed.");
    }
    
    @Test
    void testSingularMatrixHandling() {
        System.out.println("Testing singular matrix handling...");
        
        ICholeskyDecomposition choleskyDecomposition = new RereCholeskyDecomposition();
        
        assertThrows(NonPositiveDefiniteMatrixException.class, () -> {
            choleskyDecomposition.decompose(singularMatrix);
        }, "Singular / non-PD matrix should throw NonPositiveDefiniteMatrixException");
        
        System.out.println("Singular matrix handling test passed.");
    }
    
    @Test
    void testConfigurableThresholds() {
        System.out.println("Testing configurable thresholds...");
        
        // Test with custom thresholds
        double relativeSymmetryThreshold = 1.0e-12;
        double absolutePositivityThreshold = 1.0e-8;
        
        ICholeskyDecomposition choleskyDecomposition = 
            new RereCholeskyDecomposition(relativeSymmetryThreshold, absolutePositivityThreshold);
            
        // Verify thresholds are set correctly
        assertEquals(relativeSymmetryThreshold, choleskyDecomposition.getRelativeSymmetryThreshold(), 1e-15);
        assertEquals(absolutePositivityThreshold, choleskyDecomposition.getAbsolutePositivityThreshold(), 1e-15);
        
        // Decomposition should still work with standard matrix
        IMatrix<Double> L = choleskyDecomposition.decompose(testMatrix);
        assertNotNull(L, "L matrix should not be null with custom thresholds");
        
        System.out.println("Configurable thresholds test passed.");
    }
}