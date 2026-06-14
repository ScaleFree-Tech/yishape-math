package com.yishape.lab.math.linalg;

import com.yishape.lab.math.linalg.decomposition.impl.RereCholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CholeskySolverTest {

    @Test
    public void testCholeskySolverWithForwardAndBackwardSolve() {
        // Create a simple positive definite matrix directly
        // This is a known positive definite matrix
        double[][] testData = {
            {4.0, 1.0, 2.0},
            {1.0, 5.0, 3.0},
            {2.0, 3.0, 6.0}
        };
        
        IMatrix<Double> matrix = Linalg.matrix(testData);
        
        // Perform Cholesky decomposition — must succeed for this positive-definite matrix
        RereCholeskyDecomposition cholesky = new RereCholeskyDecomposition(1e-10, 1e-10);
        cholesky.decompose(matrix);

        // Get the solver
        IDecompositionSolver solver = cholesky.getSolver();

        // Create a right-hand side vector
        double[] rhsData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(rhsData);

        // Solve the system
        IVector<Double> x = solver.solve(b);

        // Verify the solution: Ax = b (within numerical tolerance)
        IMatrix<Double> xMatrix = Linalg.matrix(new double[][]{x.toDoubleArray()}).transpose();
        IMatrix<Double> result = matrix.mmul(xMatrix);

        for (int i = 0; i < rhsData.length; i++) {
            assertEquals(rhsData[i], result.get(i, 0), 1e-10,
                "Cholesky solve: Ax should equal b at row " + i);
        }
    }

    @Test
    public void testCholeskySolverMatrixEquation() {
        // Create a simple positive definite matrix directly
        double[][] testData = {
            {4.0, 1.0, 2.0},
            {1.0, 5.0, 3.0},
            {2.0, 3.0, 6.0}
        };

        IMatrix<Double> matrix = Linalg.matrix(testData);

        // Perform Cholesky decomposition — must succeed for this positive-definite matrix
        RereCholeskyDecomposition cholesky = new RereCholeskyDecomposition(1e-10, 1e-10);
        cholesky.decompose(matrix);

        // Get the solver
        IDecompositionSolver solver = cholesky.getSolver();

        // Create a right-hand side matrix
        double[][] rhsData = {
            {1.0, 2.0},
            {2.0, 1.0},
            {3.0, 0.0}
        };
        IMatrix<Double> b = Linalg.matrix(rhsData);

        // Solve the system
        IMatrix<Double> x = solver.solve(b);

        // Verify the solution: AX = B (within numerical tolerance)
        IMatrix<Double> result = matrix.mmul(x);

        for (int i = 0; i < b.rows(); i++) {
            for (int j = 0; j < b.cols(); j++) {
                assertEquals(rhsData[i][j], result.get(i, j), 1e-10,
                    "Cholesky solve: AX should equal B at [" + i + "," + j + "]");
            }
        }
    }
}