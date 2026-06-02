package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.solver.LinearSystemSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for LinearSystemSolver implementation
 */
public class LinearSystemSolverTest {
    
    private IMatrix<Double> spdMatrix;  // Symmetric positive definite matrix
    private IMatrix<Double> symMatrix;  // Symmetric matrix
    private IMatrix<Double> genMatrix;  // General matrix
    
    @BeforeEach
    void setUp() {
        // Create a 3x3 symmetric positive definite test matrix
        double[][] spdData = {
            { 4.0, 12.0, -16.0},
            { 12.0, 37.0, -43.0},
            { -16.0, -43.0, 98.0}
        };
        spdMatrix = Linalg.matrix(spdData);
        
        // Create a 3x3 symmetric test matrix (not positive definite)
        double[][] symData = {
            { 2.0, -1.0, 0.0},
            { -1.0, 2.0, -1.0},
            { 0.0, -1.0, 2.0}
        };
        symMatrix = Linalg.matrix(symData);
        
        // Create a 3x3 general test matrix
        double[][] genData = {
            { 1.0, 2.0, 3.0},
            { 4.0, 5.0, 6.0},
            { 7.0, 8.0, 10.0}
        };
        genMatrix = Linalg.matrix(genData);
    }
    
    @Test
    void testSPDMatrixSolve() {
        System.out.println("Testing linear system solver with symmetric positive definite matrix...");
        
        // Create test vector b
        double[] bData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(bData);
        
        // Convert vector to matrix form
        IMatrix<Double> B = Linalg.matrix(new double[][]{bData}).transpose();
        
        // Solve Ax = b using LinearSystemSolver
        IMatrix<Double> X = LinearSystemSolver.solve(spdMatrix, B);
        
        assertNotNull(X, "Solution matrix should not be null");
        assertEquals(3, X.rows(), "Solution matrix should have 3 rows");
        assertEquals(1, X.cols(), "Solution matrix should have 1 column");
        
        // Verify solution by checking Ax = b
        IMatrix<Double> result = spdMatrix.mmul(X);
        for (int i = 0; i < bData.length; i++) {
            assertEquals(bData[i], result.get(i, 0), 1e-10, 
                "Solution should satisfy Ax = b at index " + i);
        }
        
        System.out.println("SPD matrix solve test passed.");
    }
    
    @Test
    void testSymmetricMatrixSolve() {
        System.out.println("Testing linear system solver with symmetric matrix...");
        
        // Create test matrix B
        double[][] bData = {
            {1.0, 2.0},
            {2.0, 1.0},
            {3.0, 0.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Solve AX = B using LinearSystemSolver
        IMatrix<Double> X = LinearSystemSolver.solve(symMatrix, B);
        
        assertNotNull(X, "Solution matrix should not be null");
        assertEquals(3, X.rows(), "Solution matrix should have 3 rows");
        assertEquals(2, X.cols(), "Solution matrix should have 2 columns");
        
        // Verify solution by checking AX = B
        IMatrix<Double> result = symMatrix.mmul(X);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(bData[i][j], result.get(i, j), 1e-10, 
                    "Solution should satisfy AX = B at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("Symmetric matrix solve test passed.");
    }
    
    @Test
    void testGeneralMatrixSolve() {
        System.out.println("Testing linear system solver with general matrix...");
        
        // Create test matrix B
        double[][] bData = {
            {1.0, 0.0},
            {0.0, 1.0},
            {0.0, 0.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Solve AX = B using LinearSystemSolver
        IMatrix<Double> X = LinearSystemSolver.solve(genMatrix, B);
        
        assertNotNull(X, "Solution matrix should not be null");
        assertEquals(3, X.rows(), "Solution matrix should have 3 rows");
        assertEquals(2, X.cols(), "Solution matrix should have 2 columns");
        
        // Verify solution by checking AX = B
        IMatrix<Double> result = genMatrix.mmul(X);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(bData[i][j], result.get(i, j), 1e-10, 
                    "Solution should satisfy AX = B at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("General matrix solve test passed.");
    }
    
    @Test
    void testSPDMatrixVectorSolve() {
        System.out.println("Testing linear system solver with symmetric positive definite matrix and vector RHS...");
        
        // Create test vector b
        double[] bData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(bData);
        
        // Solve Ax = b using LinearSystemSolver
        IVector<Double> x = LinearSystemSolver.solve(spdMatrix, b);
        
        assertNotNull(x, "Solution vector should not be null");
        assertEquals(3, x.size(), "Solution vector should have 3 elements");
        
        // Verify solution by checking Ax = b
        IVector<Double> result = spdMatrix.mmul(x);
        for (int i = 0; i < bData.length; i++) {
            assertEquals(bData[i], result.get(i), 1e-10, 
                "Solution should satisfy Ax = b at index " + i);
        }
        
        System.out.println("SPD matrix vector solve test passed.");
    }
    
    @Test
    void testSymmetricMatrixVectorSolve() {
        System.out.println("Testing linear system solver with symmetric matrix and vector RHS...");
        
        // Create test vector b
        double[] bData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(bData);
        
        // Solve Ax = b using LinearSystemSolver
        IVector<Double> x = LinearSystemSolver.solve(symMatrix, b);
        
        assertNotNull(x, "Solution vector should not be null");
        assertEquals(3, x.size(), "Solution vector should have 3 elements");
        
        // Verify solution by checking Ax = b
        IVector<Double> result = symMatrix.mmul(x);
        for (int i = 0; i < bData.length; i++) {
            assertEquals(bData[i], result.get(i), 1e-10, 
                "Solution should satisfy Ax = b at index " + i);
        }
        
        System.out.println("Symmetric matrix vector solve test passed.");
    }
    
    @Test
    void testGeneralMatrixVectorSolve() {
        System.out.println("Testing linear system solver with general matrix and vector RHS...");
        
        // Create test vector b
        double[] bData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(bData);
        
        // Solve Ax = b using LinearSystemSolver
        IVector<Double> x = LinearSystemSolver.solve(genMatrix, b);
        
        assertNotNull(x, "Solution vector should not be null");
        assertEquals(3, x.size(), "Solution vector should have 3 elements");
        
        // Verify solution by checking Ax = b
        IVector<Double> result = genMatrix.mmul(x);
        for (int i = 0; i < bData.length; i++) {
            assertEquals(bData[i], result.get(i), 1e-10, 
                "Solution should satisfy Ax = b at index " + i);
        }
        
        System.out.println("General matrix vector solve test passed.");
    }
    
    @Test
    void testInvalidMatrixDimensions() {
        System.out.println("Testing linear system solver with invalid matrix dimensions...");
        
        // Create test matrix B with wrong number of rows
        double[][] bData = {
            {1.0, 2.0},
            {2.0, 1.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Should throw IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystemSolver.solve(spdMatrix, B);
        }, "Should throw IllegalArgumentException for mismatched matrix dimensions");
        
        System.out.println("Invalid matrix dimensions test passed.");
    }
    
    @Test
    void testVectorInvalidDimensions() {
        System.out.println("Testing linear system solver with vector RHS and invalid dimensions...");
        
        // Create test vector b with wrong number of elements
        IVector<Double> b = Linalg.vector(new double[]{1.0, 2.0});
        
        // Should throw IllegalArgumentException
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystemSolver.solve(spdMatrix, b);
        }, "Should throw IllegalArgumentException for mismatched dimensions");
        
        System.out.println("Vector invalid dimensions test passed.");
    }
    
    @Test
    void testVectorNullInput() {
        System.out.println("Testing linear system solver with null vector input...");
        
        // Create test vector b
        IVector<Double> b = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        
        // Should throw IllegalArgumentException for null A
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystemSolver.solve((IMatrix<Double>) null, b);
        }, "Should throw IllegalArgumentException for null A matrix");
        
        // Should throw IllegalArgumentException for null b
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystemSolver.solve(spdMatrix, (IVector<Double>) null);
        }, "Should throw IllegalArgumentException for null b vector");
        
        System.out.println("Vector null input test passed.");
    }
    
    @Test
    void testNullMatrixInput() {
        System.out.println("Testing linear system solver with null matrix input...");
        
        // Create test matrix B
        double[][] bData = {
            {1.0},
            {2.0},
            {3.0}
        };
        IMatrix<Double> B = Linalg.matrix(bData);
        
        // Should throw IllegalArgumentException for null A
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystemSolver.solve((IMatrix<Double>) null, B);
        }, "Should throw IllegalArgumentException for null A matrix");
        
        // Should throw IllegalArgumentException for null B
        assertThrows(IllegalArgumentException.class, () -> {
            LinearSystemSolver.solve(spdMatrix, (IMatrix<Double>) null);
        }, "Should throw IllegalArgumentException for null B matrix");
        
        System.out.println("Null matrix input test passed.");
    }

    /**
     * n &gt; 100 dense nonsymmetric systems should use LU-first path (not Hessenberg),
     * consistent with Weka / Commons Math; residual should be small for well-behaved randn.
     */
    @Test
    void largeNonsymmetricRandomSolveResidualSmall() {
        int n = 120;
        IMatrix<Double> A = IMatrix.randn(n, n, 42L);
        IVector<Double> b = Linalg.ones(n);
        IVector<Double> x = LinearSystemSolver.solve(A, b);
        IVector<Double> ax = A.mmul(x);
        double maxAbsRes = 0.0;
        for (int i = 0; i < n; i++) {
            maxAbsRes = Math.max(maxAbsRes, Math.abs(ax.get(i) - b.get(i)));
        }
        assertTrue(maxAbsRes < 1e-8 * n,
                "||Ax - b||_inf should be small; got max residual " + maxAbsRes);
    }

    @Test
    void symmetricIndefinite2x2ResidualSmall() {
        IMatrix<Double> A = Linalg.matrix(new double[][]{{1.0, 2.0}, {2.0, 1.0}});
        IVector<Double> b = Linalg.vector(new double[]{1.0, 0.0});
        IVector<Double> x = LinearSystemSolver.solve(A, b);
        IVector<Double> ax = A.mmul(x);
        assertEquals(1.0, ax.get(0), 1e-10);
        assertEquals(0.0, ax.get(1), 1e-10);
    }
}