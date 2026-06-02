package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Verbose test to see what's happening in the integer programming solver
 */
public class VerboseKnapsackTest {
    public static void main(String[] args) {
        // Simple knapsack problem:
        // Maximize: 10*x1 + 20*x2
        // Subject to: 5*x1 + 10*x2 <= 15
        // x1, x2 in {0, 1}
        // Expected solution: x1=1, x2=1 with value 30
        
        // Objective function coefficients (negative because we minimize)
        IVector c = Linalg.vector(new double[]{-10.0, -20.0});
        
        // Constraint matrix (weights)
        IMatrix A_ub = IMatrix.of(new double[][]{
            {5.0, 10.0}  // Weight constraint coefficients
        });
        
        // Constraint vector (capacity)
        IVector b_ub = Linalg.vector(new double[]{15.0});
        
        RereIntegerProg solver = new RereIntegerProg();
        solver.setAllVariablesBinary();  // Set all variables as binary (0-1)
        solver.setVerbose(true);
        solver.setMaxIterations(1000);
        solver.setGapTolerance(1e-9);
        solver.setTolerance(1e-9);
        
        System.out.println("Solving simple knapsack problem with verbose output:");
        System.out.println("Maximize: 10*x1 + 20*x2");
        System.out.println("Subject to: 5*x1 + 10*x2 <= 15");
        System.out.println("x1, x2 in {0, 1}");
        
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result != null) {
            System.out.println("Solution found:");
            System.out.println("Objective value (negative): " + result.getOptimalValue());
            System.out.println("Objective value (positive): " + (-result.getOptimalValue()));
            System.out.println("Solution vector: " + result.getOptimalPoint());
            System.out.println("Converged: " + result.isConverged());
        } else {
            System.out.println("No solution found");
        }
    }
}