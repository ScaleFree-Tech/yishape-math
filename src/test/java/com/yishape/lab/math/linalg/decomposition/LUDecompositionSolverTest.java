package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereLUDecomposition;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for LUDecompositionSolver implementation
 */
public class LUDecompositionSolverTest {
    
    private IMatrix<Double> testMatrix;
    private IMatrix<Double> singularMatrix;
    
    @BeforeEach
    void setUp() {
        // Create a 3x3 test matrix
        double[][] testData = {
            { 1.0, 2.0, 3.0},
            { 2.0, 5.0, 3.0},
            { 1.0, 0.0, 8.0}
        };
        testMatrix = Linalg.matrix(testData);
        
        // Create a singular matrix
        double[][] singularData = {
            { 2.0, 3.0 },
            { 2.0, 3.0 }
        };
        singularMatrix = Linalg.matrix(singularData);
    }
    
    @Test
    void testVectorSolve() {
        System.out.println("Testing vector solve with LU decomposition...");
        
        // Create LU decomposition
        RereLUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        // Get solver
        IDecompositionSolver solver = luDecomposition.getSolver();
        
        // Create test vector b
        double[] bData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(bData);
        
        // Solve Ax = b
        IVector<Double> x = solver.solve(b);
        
        assertNotNull(x, "Solution vector should not be null");
        assertEquals(3, x.size(), "Solution vector should have 3 elements");
        
        // Verify solution by checking Ax = b
        IVector<Double> result = testMatrix.mmul(x);
        for (int i = 0; i < bData.length; i++) {
            assertEquals(bData[i], result.get(i), 1e-10, 
                "Solution should satisfy Ax = b at index " + i);
        }
        
        System.out.println("Vector solve test passed.");
    }
    
    @Test
    void testMatrixSolve() {
        System.out.println("Testing matrix solve with LU decomposition...");
        
        // Create LU decomposition
        RereLUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        // Get solver
        IDecompositionSolver solver = luDecomposition.getSolver();
        
        // Create test matrix B
        double[][] bData = {
            {1.0, 2.0},
            {2.0, 1.0},
            {3.0, 0.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Solve AX = B
        IMatrix<Double> X = solver.solve(B);
        
        assertNotNull(X, "Solution matrix should not be null");
        assertEquals(3, X.rows(), "Solution matrix should have 3 rows");
        assertEquals(2, X.cols(), "Solution matrix should have 2 columns");
        
        // Verify solution by checking AX = B
        IMatrix<Double> result = testMatrix.mmul(X);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(bData[i][j], result.get(i, j), 1e-10, 
                    "Solution should satisfy AX = B at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Matrix solve test passed.");
    }
    
    @Test
    void testInverse() {
        System.out.println("Testing matrix inverse computation...");
        
        // Create LU decomposition
        RereLUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        // Get solver
        IDecompositionSolver solver = luDecomposition.getSolver();
        
        // Compute inverse
        IMatrix<Double> inverse = solver.getInverse();
        
        assertNotNull(inverse, "Inverse matrix should not be null");
        assertEquals(3, inverse.rows(), "Inverse matrix should have 3 rows");
        assertEquals(3, inverse.cols(), "Inverse matrix should have 3 columns");
        
        // Verify inverse by checking A * A^(-1) = I
        IMatrix<Double> identity = testMatrix.mmul(inverse);
        IMatrix<Double> expectedIdentity = Linalg.eye(3);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(expectedIdentity.get(i, j), identity.get(i, j), 1e-10, 
                    "Inverse should satisfy A * A^(-1) = I at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Inverse computation test passed.");
    }
    
    @Test
    void testSingularityHandling() {
        System.out.println("Testing singularity handling...");
        
        // Create LU decomposition of singular matrix
        RereLUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(singularMatrix);
        
        // Get solver
        IDecompositionSolver solver = luDecomposition.getSolver();
        
        // Check that matrix is detected as singular
        assertFalse(solver.isNonSingular(), "Singular matrix should be detected as singular");
        
        // Try to solve - should throw exception
        double[] bData = {1.0, 2.0};
        IVector<Double> b = Linalg.vector(bData);
        
        assertThrows(RuntimeException.class, () -> {
            solver.solve(b);
        }, "Solving with singular matrix should throw RuntimeException");
        
        System.out.println("Singularity handling test passed.");
    }
}