package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.solver.LeastSquaresSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.util.Tuple2;
import org.junit.jupiter.api.Test;

public class LeastSquaresExampleTest {
    
    @Test
    public void testLeastSquaresExample() {
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
        
        System.out.println("Solution: " + x);
        System.out.println("Residual: " + residual);
        
        // Verify the result
        IVector<Double> Ax = A.mmul(x);
        System.out.println("A * x: " + Ax);
        System.out.println("b: " + b);
        IVector<Double> diff = Ax.sub(b);
        System.out.println("A * x - b: " + diff);
        double computedResidual = diff.norm2Value();
        System.out.println("Computed residual: " + computedResidual);
    }
}