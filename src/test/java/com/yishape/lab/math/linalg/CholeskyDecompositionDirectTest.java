package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.impl.RereCholeskyDecomposition;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CholeskyDecompositionDirectTest {

    @Test
    public void testCholeskyDecompositionDirect() {
        // Create a 3x3 symmetric positive definite test matrix (same as in RereCholeskyDecompositionTest)
        double[][] testData = {
            { 4.0, 12.0, -16.0},
            { 12.0, 37.0, -43.0},
            { -16.0, -43.0, 98.0}
        };
        IMatrix<Double> matrix = Linalg.matrix(testData);
        
        // Perform Cholesky decomposition
        RereCholeskyDecomposition cholesky = new RereCholeskyDecomposition();
        
        // This should work without throwing an exception
        IMatrix<Double> L = cholesky.decompose(matrix);
        
        // Verify that L is not null
        assertNotNull(L, "L matrix should not be null");
        assertEquals(3, L.rows(), "L matrix should have 3 rows");
        assertEquals(3, L.cols(), "L matrix should have 3 columns");
    }
}