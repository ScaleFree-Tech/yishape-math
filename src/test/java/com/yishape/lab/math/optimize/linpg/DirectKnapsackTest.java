package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Direct test to verify that the integer programming solver is working correctly for a small knapsack problem
 * by directly calling the solveWithNonNegativeEqualConstraints method
 */
public class DirectKnapsackTest {
    public static void main(String[] args) {
        // Simple knapsack problem:
        // Maximize: 10*x1 + 20*x2
        // Subject to: 5*x1 + 10*x2 <= 15
        // x1, x2 in {0, 1}
        // Expected solution: x1=1, x2=1 with value 30
        
        // First convert to equality constraints with slack variables
        // 5*x1 + 10*x2 + s1 = 15
        // s1 >= 0
        // x1, x2 in {0, 1}
        
        // Objective function coefficients (negative because we minimize)
        // Extended with slack variable coefficients (0)
        IVector c = Linalg.vector(new double[]{-10.0, -20.0, 0.0});
        
        // Constraint matrix (weights with slack variable)
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {5.0, 10.0, 1.0}  // Weight constraint coefficients with slack variable
        });
        
        // Constraint vector (capacity)
        IVector b_eq = Linalg.vector(new double[]{15.0});
        
        RereIntegerProg solver = new RereIntegerProg();
        solver.setBinaryVariable(0);  // Set x1 as binary
        solver.setBinaryVariable(1);  // Set x2 as binary
        // Note: We don't set the slack variable as binary
        solver.setVerbose(true);
        solver.setMaxIterations(1000);
        solver.setGapTolerance(1e-9);
        solver.setTolerance(1e-9);
        
        System.out.println("Solving simple knapsack problem with direct equality constraints:");
        System.out.println("Minimize: -10*x1 - 20*x2 + 0*s1");
        System.out.println("Subject to: 5*x1 + 10*x2 + s1 = 15");
        System.out.println("x1, x2 in {0, 1}, s1 >= 0");
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        if (result != null) {
            System.out.println("Solution found:");
            System.out.println("Objective value (negative): " + result.getOptimalValue());
            System.out.println("Objective value (positive): " + (-result.getOptimalValue()));
            System.out.println("Solution vector: " + result.getOptimalPoint());
            System.out.println("Converged: " + result.isConverged());
            
            // Verify that solution is binary for the original variables
            IVector solution = result.getOptimalPoint();
            boolean isBinary = true;
            // Check only the original variables (first 2)
            for (int i = 0; i < 2; i++) {
                double value = (Double) solution.get(i);
                if (Math.abs(value) > 1e-6 && Math.abs(value - 1.0) > 1e-6) {
                    isBinary = false;
                    System.out.println("Variable x" + (i+1) + " = " + value + " is not binary");
                }
            }
            System.out.println("Solution is binary for original variables: " + isBinary);
            
            // Calculate total weight and value
            double[] weights = {5.0, 10.0};
            double[] values = {10.0, 20.0};
            double totalWeight = 0;
            double totalValue = 0;
            for (int i = 0; i < 2; i++) {
                double value = (Double) solution.get(i);
                if (Math.abs(value - 1.0) < 1e-6) {
                    totalWeight += weights[i];
                    totalValue += values[i];
                }
            }
            System.out.println("Total weight: " + totalWeight);
            System.out.println("Total value: " + totalValue);
        } else {
            System.out.println("No solution found");
        }
    }
}