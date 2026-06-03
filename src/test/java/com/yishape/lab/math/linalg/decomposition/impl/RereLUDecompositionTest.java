package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ILUDecomposition;
import com.yishape.lab.math.linalg.decomposition.SingularMatrixException;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for RereLUDecomposition implementation
 */
public class RereLUDecompositionTest {
    
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
    void testLUDecomposition() {
        System.out.println("Testing LU Decomposition with partial pivoting...");
        
        ILUDecomposition luDecomposition = new RereLUDecomposition();
        Tuple2<IMatrix<Double>, IMatrix<Double>> result = luDecomposition.decompose(testMatrix);
        
        assertNotNull(result, "LU result should not be null");
        assertNotNull(result._1, "L matrix should not be null");
        assertNotNull(result._2, "U matrix should not be null");
        
        IMatrix<Double> L = result._1;
        IMatrix<Double> U = result._2;
        
        assertEquals(3, L.rows(), "L matrix should have 3 rows");
        assertEquals(3, L.cols(), "L matrix should have 3 columns");
        assertEquals(3, U.rows(), "U matrix should have 3 rows");
        assertEquals(3, U.cols(), "U matrix should have 3 columns");
        
        // Check that L is lower triangular with unit diagonal
        assertEquals(1.0, L.get(0, 0), 1e-10, "L diagonal should be 1");
        assertEquals(1.0, L.get(1, 1), 1e-10, "L diagonal should be 1");
        assertEquals(1.0, L.get(2, 2), 1e-10, "L diagonal should be 1");
        assertEquals(0.0, L.get(0, 1), 1e-10, "L should be lower triangular");
        assertEquals(0.0, L.get(0, 2), 1e-10, "L should be lower triangular");
        assertEquals(0.0, L.get(1, 2), 1e-10, "L should be lower triangular");
        
        // Check that U is upper triangular
        assertEquals(0.0, U.get(1, 0), 1e-10, "U should be upper triangular");
        assertEquals(0.0, U.get(2, 0), 1e-10, "U should be upper triangular");
        assertEquals(0.0, U.get(2, 1), 1e-10, "U should be upper triangular");
        
        System.out.println("LU Decomposition test passed.");
    }
    
    @Test
    void testLMatrixRetrieval() {
        System.out.println("Testing L matrix retrieval...");
        
        ILUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        IMatrix<Double> L = luDecomposition.getL();
        assertNotNull(L, "L matrix should not be null");
        assertEquals(3, L.rows(), "L matrix should have 3 rows");
        assertEquals(3, L.cols(), "L matrix should have 3 columns");
        
        // Check that L is lower triangular with unit diagonal
        assertEquals(1.0, L.get(0, 0), 1e-10, "L diagonal should be 1");
        assertEquals(1.0, L.get(1, 1), 1e-10, "L diagonal should be 1");
        assertEquals(1.0, L.get(2, 2), 1e-10, "L diagonal should be 1");
        
        System.out.println("L matrix retrieval test passed.");
    }
    
    @Test
    void testUMatrixRetrieval() {
        System.out.println("Testing U matrix retrieval...");
        
        ILUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        IMatrix<Double> U = luDecomposition.getU();
        assertNotNull(U, "U matrix should not be null");
        assertEquals(3, U.rows(), "U matrix should have 3 rows");
        assertEquals(3, U.cols(), "U matrix should have 3 columns");
        
        System.out.println("U matrix retrieval test passed.");
    }
    
    @Test
    void testPMatrixRetrieval() {
        System.out.println("Testing P matrix retrieval...");
        
        ILUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        IMatrix<Double> P = luDecomposition.getP();
        assertNotNull(P, "P matrix should not be null");
        assertEquals(3, P.rows(), "P matrix should have 3 rows");
        assertEquals(3, P.cols(), "P matrix should have 3 columns");
        
        // Check that P is a permutation matrix
        // Sum of each row should be 1
        for (int i = 0; i < 3; i++) {
            double rowSum = 0;
            for (int j = 0; j < 3; j++) {
                rowSum += P.get(i, j);
            }
            assertEquals(1.0, rowSum, 1e-10, "Each row of P should sum to 1");
        }
        
        // Sum of each column should be 1
        for (int j = 0; j < 3; j++) {
            double colSum = 0;
            for (int i = 0; i < 3; i++) {
                colSum += P.get(i, j);
            }
            assertEquals(1.0, colSum, 1e-10, "Each column of P should sum to 1");
        }
        
        System.out.println("P matrix retrieval test passed.");
    }
    
    @Test
    void testPivotRetrieval() {
        System.out.println("Testing pivot vector retrieval...");
        
        ILUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        int[] pivot = luDecomposition.getPivot();
        assertNotNull(pivot, "Pivot vector should not be null");
        assertEquals(3, pivot.length, "Pivot vector should have 3 elements");
        
        System.out.println("Pivot vector retrieval test passed.");
    }
    
    @Test
    void testDeterminantCalculation() {
        System.out.println("Testing determinant calculation...");
        
        ILUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        double determinant = luDecomposition.getDeterminant();
        // Expected determinant for the test matrix is -1
        assertEquals(-1.0, determinant, 1e-10, "Determinant should be -1");
        
        System.out.println("Determinant calculation test passed. Determinant: " + determinant);
    }
    
    @Test
    void testSolverRetrieval() {
        System.out.println("Testing solver retrieval...");
        
        ILUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        IDecompositionSolver solver = luDecomposition.getSolver();
        assertNotNull(solver, "Solver should not be null");
        assertTrue(solver.isNonSingular(), "Test matrix should be non-singular");
        
        System.out.println("Solver retrieval test passed.");
    }
    
    @Test
    void testSingularityDetection() {
        System.out.println("Testing singularity detection...");

        ILUDecomposition luDecomposition = new RereLUDecomposition();
        try {
            luDecomposition.decompose(singularMatrix);
        } catch (SingularMatrixException e) {
            // Expected: decomposition throws for singular matrices
        }

        Assertions.assertFalse(luDecomposition.getSolver().isNonSingular(), "Singular matrix should be detected as singular");

        // Determinant of singular matrix should be 0
        assertEquals(0.0, luDecomposition.getDeterminant(), 1e-10, "Determinant of singular matrix should be 0");

        System.out.println("Singularity detection test passed.");
    }
    
    @Test
    void testPAPartialLU() {
        System.out.println("Testing PA = LU property...");
        
        ILUDecomposition luDecomposition = new RereLUDecomposition();
        luDecomposition.decompose(testMatrix);
        
        IMatrix<Double> L = luDecomposition.getL();
        IMatrix<Double> U = luDecomposition.getU();
        IMatrix<Double> P = luDecomposition.getP();
        
        // Check that P * A = L * U
        IMatrix<Double> PA = P.mmul(testMatrix);
        IMatrix<Double> LU = L.mmul(U);
        
        // Compare matrices element by element
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(PA.get(i, j), LU.get(i, j), 1e-10, 
                    "PA should equal LU at position (" + i + "," + j + ")");
            }
        }
        
        System.out.println("PA = LU property test passed.");
    }
}