package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Test to verify that the integer programming solver is working correctly for the knapsack problem
 */
public class KnapsackTest {
    public static void main(String[] args) {
        // Knapsack problem data from the original problem
        // Values: {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0}
        // Weights: {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0}
        // Capacity: 100.0
        
        // Objective function coefficients (negative because we minimize)
        IVector c = Linalg.vector(new double[]{-60.0, -100.0, -120.0, -80.0, -150.0, -200.0, -50.0});
        
        // Constraint matrix (weights)
        IMatrix A_ub = IMatrix.of(new double[][]{
            {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0}  // Weight constraint coefficients
        });
        
        // Constraint vector (capacity)
        IVector b_ub = Linalg.vector(new double[]{100.0});
        
        RereIntegerProg solver = new RereIntegerProg();
        solver.setAllVariablesBinary();  // Set all variables as binary (0-1)
        solver.setVerbose(true);
        solver.setMaxIterations(50000);  // Increase iterations for complex problem
        solver.setGapTolerance(1e-9);    // Tighten the gap tolerance
        solver.setTolerance(1e-9);       // Tighten the tolerance
        
        System.out.println("Solving knapsack problem:");
        System.out.println("Maximize: 60*x1 + 100*x2 + 120*x3 + 80*x4 + 150*x5 + 200*x6 + 50*x7");
        System.out.println("Subject to: 10*x1 + 20*x2 + 30*x3 + 40*x4 + 50*x5 + 60*x6 + 10*x7 <= 100");
        System.out.println("x1, x2, x3, x4, x5, x6, x7 in {0, 1}");
        
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result != null) {
            System.out.println("Solution found:");
            System.out.println("Objective value (negative): " + result.getOptimalValue());
            System.out.println("Objective value (positive): " + (-result.getOptimalValue()));
            System.out.println("Solution vector: " + result.getOptimalPoint());
            System.out.println("Converged: " + result.isConverged());
            
            // Verify that solution is binary
            IVector solution = result.getOptimalPoint();
            boolean isBinary = true;
            for (int i = 0; i < solution.length(); i++) {
                double value = (Double) solution.get(i);
                if (Math.abs(value) > 1e-6 && Math.abs(value - 1.0) > 1e-6) {
                    isBinary = false;
                    System.out.println("Variable x" + (i+1) + " = " + value + " is not binary");
                }
            }
            System.out.println("Solution is binary: " + isBinary);
            
            // Calculate total weight and value
            double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
            double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
            double totalWeight = 0;
            double totalValue = 0;
            for (int i = 0; i < solution.length(); i++) {
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