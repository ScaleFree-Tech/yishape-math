package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.impl.RereCholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.CholeskyDecompositionSolver;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CholeskySolverApiTest {

    @Test
    public void testCholeskySolverUsesNewApiMethods() {
        // Create a 3x3 symmetric positive definite test matrix (same as in RereCholeskyDecompositionTest)
        double[][] testData = {
            { 4.0, 12.0, -16.0},
            { 12.0, 37.0, -43.0},
            { -16.0, -43.0, 98.0}
        };
        IMatrix<Double> matrix = Linalg.matrix(testData);
        
        // Perform Cholesky decomposition
        RereCholeskyDecomposition cholesky = new RereCholeskyDecomposition();
        cholesky.decompose(matrix);
        
        // Get the LT data directly to create a solver
        double[][] lTData = new double[3][3];
        IMatrix<Double> ltMatrix = cholesky.getLT();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                lTData[i][j] = ltMatrix.get(i, j).doubleValue();
            }
        }
        
        // Create a solver directly
        CholeskyDecompositionSolver solver = new CholeskyDecompositionSolver(lTData);
        
        // Create a right-hand side vector
        double[] rhsData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(rhsData);
        
        // Solve the system - this should use the updated implementation with forwardSolve and backwardSolve
        IVector<Double> x = solver.solve(b);
        
        // Verify the solution exists and is reasonable
        assertNotNull(x);
        assertEquals(3, x.size());
        
        // Verify by checking that Ax = b
        IMatrix<Double> xMatrix = Linalg.matrix(new double[][]{x.toDoubleArray()}).t();
        IMatrix<Double> result = matrix.mmul(xMatrix);
        
        // Check that Ax = b (within numerical tolerance)
        for (int i = 0; i < rhsData.length; i++) {
            assertEquals(rhsData[i], result.get(i, 0).doubleValue(), 1e-10);
        }
    }
}