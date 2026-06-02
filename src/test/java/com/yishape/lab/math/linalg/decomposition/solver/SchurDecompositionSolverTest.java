package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for SchurDecompositionSolver implementation
 */
public class SchurDecompositionSolverTest {
    
    private IMatrix<Double> tMatrix;
    private IMatrix<Double> uMatrix;
    
    @BeforeEach
    void setUp() {
        // Create a simple 3x3 test case for Schur decomposition
        // T matrix (quasi-upper triangular)
        double[][] tData = {
            { 3.0, 1.0, 0.0},
            { 0.0, 2.0, 1.0},
            { 0.0, 0.0, 1.0}
        };
        tMatrix = Linalg.matrix(tData);
        
        // U matrix (orthogonal)
        double[][] uData = {
            { 1.0, 0.0, 0.0},
            { 0.0, 1.0, 0.0},
            { 0.0, 0.0, 1.0}
        };
        uMatrix = Linalg.matrix(uData);
    }
    
    @Test
    void testMatrixSolve() {
        System.out.println("Testing matrix solve with Schur decomposition...");
        
        // Create solver
        SchurDecompositionSolver solver = new SchurDecompositionSolver(tMatrix, uMatrix, 1e-12);
        
        // Create test matrix B
        double[][] bData = {
            {1.0, 2.0},
            {2.0, 1.0},
            {3.0, 0.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Solve TX = B
        IMatrix<Double> X = solver.solve(B);
        
        assertNotNull(X, "Solution matrix should not be null");
        assertEquals(3, X.rows(), "Solution matrix should have 3 rows");
        assertEquals(2, X.cols(), "Solution matrix should have 2 columns");
        
        // Verify solution by checking TX = B
        IMatrix<Double> result = tMatrix.mmul(X);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(bData[i][j], result.get(i, j), 1e-10, 
                    "Solution should satisfy TX = B at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Matrix solve test passed.");
    }
    
    @Test
    void testVectorSolve() {
        System.out.println("Testing vector solve with Schur decomposition...");
        
        // Create solver
        SchurDecompositionSolver solver = new SchurDecompositionSolver(tMatrix, uMatrix, 1e-12);
        
        // Create test vector b
        double[] bData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(bData);
        
        // Solve Tx = b
        IVector<Double> x = solver.solve(b);
        
        assertNotNull(x, "Solution vector should not be null");
        assertEquals(3, x.size(), "Solution vector should have 3 elements");
        
        // Verify solution by checking Tx = b
        IVector<Double> result = tMatrix.mmul(x);
        for (int i = 0; i < bData.length; i++) {
            assertEquals(bData[i], result.get(i), 1e-10, 
                "Solution should satisfy Tx = b at index " + i);
        }
        
        System.out.println("Vector solve test passed.");
    }
    
    @Test
    void testInverse() {
        System.out.println("Testing matrix inverse computation with Schur decomposition...");
        
        // Create solver
        SchurDecompositionSolver solver = new SchurDecompositionSolver(tMatrix, uMatrix, 1e-12);
        
        // Compute inverse
        IMatrix<Double> inverse = solver.getInverse();
        
        assertNotNull(inverse, "Inverse matrix should not be null");
        assertEquals(3, inverse.rows(), "Inverse matrix should have 3 rows");
        assertEquals(3, inverse.cols(), "Inverse matrix should have 3 columns");
        
        // Verify inverse by checking T * T^(-1) = I
        IMatrix<Double> identity = tMatrix.mmul(inverse);
        IMatrix<Double> expectedIdentity = Linalg.eye(3);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedIdentity.get(i, j), identity.get(i, j), 1e-10, 
                    "Inverse should satisfy T * T^(-1) = I at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Inverse computation test passed.");
    }
    
    @Test
    void testNonSingularProperty() {
        System.out.println("Testing non-singular property with Schur decomposition...");
        
        // Create solver with non-singular T matrix
        SchurDecompositionSolver solver = new SchurDecompositionSolver(tMatrix, uMatrix, 1e-12);
        
        // Check that matrix is detected as non-singular
        assertTrue(solver.isNonSingular(), "Matrix with non-zero diagonal should be detected as non-singular");
        
        // Create a singular T matrix (with zero diagonal)
        double[][] singularTData = {
            { 3.0, 1.0, 0.0},
            { 0.0, 0.0, 1.0},  // Zero diagonal element
            { 0.0, 0.0, 1.0}
        };
        IMatrix<Double> singularT = Linalg.matrix(singularTData);
        SchurDecompositionSolver singularSolver = new SchurDecompositionSolver(singularT, uMatrix, 1e-12);
        
        // Check that matrix is detected as singular
        assertFalse(singularSolver.isNonSingular(), "Matrix with zero diagonal should be detected as singular");
        
        System.out.println("Non-singular property test passed.");
    }
    
    @Test
    void testConstructorWithRawData() {
        System.out.println("Testing constructor with raw data...");
        
        // Create solver using raw data constructor
        double[][] tData = {
            { 3.0, 1.0, 0.0},
            { 0.0, 2.0, 1.0},
            { 0.0, 0.0, 1.0}
        };
        double[][] uData = {
            { 1.0, 0.0, 0.0},
            { 0.0, 1.0, 0.0},
            { 0.0, 0.0, 1.0}
        };
        
        SchurDecompositionSolver solver = new SchurDecompositionSolver(tData, uData, 1e-12);
        
        // Verify that the matrices were correctly created
        Assertions.assertNotNull(solver.getT(), "T matrix should not be null");
        Assertions.assertNotNull(solver.getU(), "U matrix should not be null");
        Assertions.assertEquals(3, solver.getT().rows(), "T matrix should have 3 rows");
        Assertions.assertEquals(3, solver.getU().rows(), "U matrix should have 3 rows");
        
        System.out.println("Constructor with raw data test passed.");
    }
}