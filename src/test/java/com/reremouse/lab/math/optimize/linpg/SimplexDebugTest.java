package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

/**
 * Debug test for SimplexLinProgSolver
 */
public class SimplexDebugTest {
    public static void main(String[] args) {
        // Objective function: minimize x1 => c = [1, 0]
        IVector c = Linalg.vector(new double[]{1, 0});
        
        // Constraint: x1 + s1 = 1 => A = [[1, 1]], b = [1]
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{1});
        
        // Create solver
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        try {
            // Solve
            Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            System.out.println("Success! Result: " + result);
        } catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}