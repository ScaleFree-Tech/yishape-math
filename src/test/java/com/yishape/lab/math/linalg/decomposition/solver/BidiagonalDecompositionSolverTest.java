package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for BidiagonalDecompositionSolver implementation
 */
public class BidiagonalDecompositionSolverTest {
    
    private IMatrix<Double> bMatrix;
    private IMatrix<Double> uMatrix;
    private IMatrix<Double> vMatrix;
    
    @BeforeEach
    void setUp() {
        // Create a simple 3x3 test case for bidiagonal decomposition
        // B matrix (bidiagonal)
        double[][] bData = {
            { 3.0, 1.0, 0.0},
            { 0.0, 2.0, 1.0},
            { 0.0, 0.0, 1.0}
        };
        bMatrix = Linalg.matrix(bData);
        
        // U matrix (orthogonal)
        double[][] uData = {
            { 1.0, 0.0, 0.0},
            { 0.0, 1.0, 0.0},
            { 0.0, 0.0, 1.0}
        };
        uMatrix = Linalg.matrix(uData);
        
        // V matrix (orthogonal)
        double[][] vData = {
            { 1.0, 0.0, 0.0},
            { 0.0, 1.0, 0.0},
            { 0.0, 0.0, 1.0}
        };
        vMatrix = Linalg.matrix(vData);
    }
    
    @Test
    void testMatrixSolve() {
        System.out.println("Testing matrix solve with bidiagonal decomposition...");
        
        // Create solver
        BidiagonalDecompositionSolver solver = new BidiagonalDecompositionSolver(bMatrix, uMatrix, vMatrix, 1e-12);
        
        // Create test matrix B
        double[][] bData = {
            {1.0, 2.0},
            {2.0, 1.0},
            {3.0, 0.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Solve BX = B (since U and V are identity matrices, A = B)
        IMatrix<Double> X = solver.solve(B);
        
        assertNotNull(X, "Solution matrix should not be null");
        assertEquals(3, X.rows(), "Solution matrix should have 3 rows");
        assertEquals(2, X.cols(), "Solution matrix should have 2 columns");
        
        // Verify solution by checking BX = B
        IMatrix<Double> result = bMatrix.mmul(X);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(bData[i][j], result.get(i, j).doubleValue(), 1e-10, 
                    "Solution should satisfy BX = B at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Matrix solve test passed.");
    }
    
    @Test
    void testVectorSolve() {
        System.out.println("Testing vector solve with bidiagonal decomposition...");
        
        // Create solver
        BidiagonalDecompositionSolver solver = new BidiagonalDecompositionSolver(bMatrix, uMatrix, vMatrix, 1e-12);
        
        // Create test vector b
        double[] bData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(bData);
        
        // Solve Bx = b
        IVector<Double> x = solver.solve(b);
        
        assertNotNull(x, "Solution vector should not be null");
        assertEquals(3, x.size(), "Solution vector should have 3 elements");
        
        // Verify solution by checking Bx = b
        IVector<Double> result = bMatrix.mmul(x);
        for (int i = 0; i < bData.length; i++) {
            assertEquals(bData[i], result.get(i).doubleValue(), 1e-10, 
                "Solution should satisfy Bx = b at index " + i);
        }
        
        System.out.println("Vector solve test passed.");
    }
    
    @Test
    void testInverse() {
        System.out.println("Testing matrix inverse computation with bidiagonal decomposition...");
        
        // Create solver
        BidiagonalDecompositionSolver solver = new BidiagonalDecompositionSolver(bMatrix, uMatrix, vMatrix, 1e-12);
        
        // Compute inverse
        IMatrix<Double> inverse = solver.getInverse();
        
        assertNotNull(inverse, "Inverse matrix should not be null");
        assertEquals(3, inverse.rows(), "Inverse matrix should have 3 rows");
        assertEquals(3, inverse.cols(), "Inverse matrix should have 3 columns");
        
        // Verify inverse by checking B * B^(-1) = I
        IMatrix<Double> identity = bMatrix.mmul(inverse);
        IMatrix<Double> expectedIdentity = Linalg.eye(3);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedIdentity.get(i, j).doubleValue(), identity.get(i, j).doubleValue(), 1e-10, 
                    "Inverse should satisfy B * B^(-1) = I at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Inverse computation test passed.");
    }
    
    @Test
    void testNonSingularProperty() {
        System.out.println("Testing non-singular property with bidiagonal decomposition...");
        
        // Create solver with non-singular B matrix
        BidiagonalDecompositionSolver solver = new BidiagonalDecompositionSolver(bMatrix, uMatrix, vMatrix, 1e-12);
        
        // Check that matrix is detected as non-singular
        assertTrue(solver.isNonSingular(), "Matrix with non-zero diagonal should be detected as non-singular");
        
        // Create a singular B matrix (with zero diagonal)
        double[][] singularBData = {
            { 3.0, 1.0, 0.0},
            { 0.0, 0.0, 1.0},  // Zero diagonal element
            { 0.0, 0.0, 1.0}
        };
        IMatrix<Double> singularB = Linalg.matrix(singularBData);
        BidiagonalDecompositionSolver singularSolver = new BidiagonalDecompositionSolver(singularB, uMatrix, vMatrix, 1e-12);
        
        // Check that matrix is detected as singular
        assertFalse(singularSolver.isNonSingular(), "Matrix with zero diagonal should be detected as singular");
        
        System.out.println("Non-singular property test passed.");
    }
    
    @Test
    void testGetters() {
        System.out.println("Testing getters with bidiagonal decomposition...");
        
        // Create solver
        BidiagonalDecompositionSolver solver = new BidiagonalDecompositionSolver(bMatrix, uMatrix, vMatrix, 1e-12);
        
        // Test getters
        Assertions.assertSame(bMatrix, solver.getB(), "B matrix should be returned correctly");
        Assertions.assertSame(uMatrix, solver.getU(), "U matrix should be returned correctly");
        Assertions.assertSame(vMatrix, solver.getV(), "V matrix should be returned correctly");
        
        System.out.println("Getters test passed.");
    }
}