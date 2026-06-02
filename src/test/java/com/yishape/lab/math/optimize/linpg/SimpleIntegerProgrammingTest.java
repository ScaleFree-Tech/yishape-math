package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Simple test to verify that the integer programming solver is working correctly
 */
public class SimpleIntegerProgrammingTest {
    public static void main(String[] args) {
        // Simple test problem: maximize x1 + x2 subject to x1 + x2 = 3, x1,x2 >= 0, x1,x2 integer
        // Expected solution: x1=3, x2=0 or x1=0, x2=3 with objective value -3 (since we minimize)
        
        IVector c = Linalg.vector(new double[]{1.0, 1.0});  // Coefficients for minimization
        IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});  // Equality constraint matrix
        IVector b_eq = Linalg.vector(new double[]{3.0});  // Equality constraint values
        
        RereIntegerProg solver = new RereIntegerProg();
        solver.setAllVariablesInteger();
        solver.setVerbose(true);
        
        System.out.println("Solving simple integer programming problem:");
        System.out.println("Minimize: x1 + x2");
        System.out.println("Subject to: x1 + x2 = 3");
        System.out.println("x1, x2 >= 0 and integer");
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        if (result != null) {
            System.out.println("Solution found:");
            System.out.println("Objective value: " + result.getOptimalValue());
            System.out.println("Solution vector: " + result.getOptimalPoint());
            System.out.println("Converged: " + result.isConverged());
        } else {
            System.out.println("No solution found");
        }
    }
}