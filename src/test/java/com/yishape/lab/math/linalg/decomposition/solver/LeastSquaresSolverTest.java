package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.solver.LeastSquaresSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LeastSquaresSolverTest {

    @Test
    public void testSolveWithResidual() {
        // Create an overdetermined system Ax = b
        // A = [[1, 2], [3, 4], [5, 6]]
        // b = [1, 2, 3]
        IMatrix<Double> A = Linalg.matrix(new double[][]{
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        });
        
        IVector<Double> b = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        
        // Solve with residual
        Tuple2<IVector<Double>, Double> result = LeastSquaresSolver.solveWithResidual(A, b);
        IVector<Double> x = result.getFirst();
        Double residual = result.getSecond();
        
        // Verify that we got a solution
        assertNotNull(x);
        assertNotNull(residual);
        assertEquals(2, x.size()); // Solution should have 2 elements
        
        // Verify the result
        IVector<Double> Ax = A.mmul(x);
        IVector<Double> diff = Ax.sub(b);
        double computedResidual = diff.norm2().doubleValue();
        
        // The residual should be non-negative
        assertTrue(residual >= 0);
        // The computed residual should match the returned residual
        assertEquals(computedResidual, residual, 1e-10);
    }
}