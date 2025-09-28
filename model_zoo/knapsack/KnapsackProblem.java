package model_zoo.knapsack;

import com.reremouse.lab.math.optimize.linpg.*;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.OptResult;

/**
 * 0-1 Knapsack Problem Example Using Integer Programming
 * 
 * This example shows how to solve a 0-1 knapsack problem using integer programming.
 * In the 0-1 knapsack problem, each item can either be taken (1) or not taken (0).
 */
public class KnapsackProblem {
    public static void main(String[] args) {
        System.out.println("=== 0-1 Knapsack Problem Example ===");
        System.out.println();
        
        // Define the knapsack problem data
        String[] itemNames = {
            "Jewelry", 
            "Books", 
            "Camera", 
            "Watch", 
            "Laptop",
            "Tent",
            "Food"
        };
        
        // Value of each item
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        
        // Weight of each item
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        
        // Knapsack capacity
        double capacity = 100.0;
        
        // Display problem information
        System.out.println("Knapsack Problem Data:");
        System.out.println("Item Name\t\tValue\tWeight");
        System.out.println("--------------------------------");
        for (int i = 0; i < itemNames.length; i++) {
            System.out.printf("%-15s\t%.1f\t%.1f\n", itemNames[i], values[i], weights[i]);
        }
        System.out.println("--------------------------------");
        System.out.println("Knapsack Capacity: " + capacity);
        System.out.println();
        
        // Formulate the problem as 0-1 integer programming
        System.out.println("Problem Formulation:");
        System.out.println("Objective Function:");
        System.out.println("  Maximize: 60*x1 + 100*x2 + 120*x3 + 80*x4 + 150*x5 + 200*x6 + 50*x7");
        System.out.println();
        System.out.println("Constraints:");
        System.out.println("  Weight Constraint:");
        System.out.println("    10*x1 + 20*x2 + 30*x3 + 40*x4 + 50*x5 + 60*x6 + 10*x7 <= 100");
        System.out.println();
        System.out.println("  0-1 Variable Constraints:");
        System.out.println("    x1, x2, x3, x4, x5, x6, x7 ∈ {0, 1}");
        System.out.println();
        
        // Convert to minimization problem (solver minimizes)
        // Minimize: -sum(values[i] * x[i])
        IVector<Double> c = Linalg.vector(new double[]{-60.0, -100.0, -120.0, -80.0, -150.0, -200.0, -50.0});
        
        // Constraint matrix (weight constraint)
        IMatrix<Double> A_ub = IMatrix.of(new double[][]{
            {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0}  // Weight constraint
        });
        
        // Constraint vector
        IVector<Double> b_ub = Linalg.vector(new double[]{capacity});
        
        // Create integer programming solver
        RereIntegerProg solver = new RereIntegerProg();
        
        // Mark ALL variables as binary variables (0-1 variables)
        solver.setAllVariablesBinary();
        
        System.out.println("Solving 0-1 integer programming problem...");
        System.out.println();
        
        // Solve the 0-1 integer programming problem
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        // Check if solution exists
        if (result == null) {
            System.out.println("No feasible solution found");
            return;
        }
        
        // Extract solution
        IVector solution = result.getOptimalPoint();
        double optimalValue = -result.getOptimalValue(); // Convert back to maximization
        
        // Output results
        System.out.println("=== Optimal Solution ===");
        System.out.println("Solution vector: " + solution);
        System.out.println("Maximum total value: " + optimalValue);
        System.out.println();
        
        // Detailed solution analysis
        System.out.println("=== Solution Analysis ===");
        double totalWeight = 0;
        double totalValue = 0;
        
        System.out.println("Selected items:");
        System.out.println("Item Name\t\tSelect\tValue\tWeight");
        System.out.println("----------------------------------------");
        for (int i = 0; i < solution.size(); i++) {
            // Round to handle numerical precision issues
            int selected = (int) Math.round(solution.get(i).doubleValue());
            if (selected == 1) {
                System.out.printf("%-15s\t%d\t%.1f\t%.1f\n", itemNames[i], selected, values[i], weights[i]);
                totalWeight += weights[i];
                totalValue += values[i];
            }
        }
        System.out.println("----------------------------------------");
        System.out.println("Total weight: " + totalWeight + " <= " + capacity);
        System.out.println("Total value: " + totalValue);
        System.out.println();
        
        // Verify 0-1 constraints
        System.out.println("=== 0-1 Constraint Verification ===");
        boolean allBinary = true;
        for (int i = 0; i < solution.size(); i++) {
            double value = solution.get(i).doubleValue();
            // Check if value is 0 or 1 (considering numerical error)
            boolean isBinary = Math.abs(value) < 1e-6 || Math.abs(value - 1.0) < 1e-6;
            allBinary &= isBinary;
            System.out.printf("x%d = %.6f (is 0-1: %s)\n", i+1, value, isBinary ? "Yes" : "No");
        }
        System.out.println("All variables are 0-1: " + (allBinary ? "Yes" : "No"));
        System.out.println();
        
        // Summary
        System.out.println("=== Summary ===");
        System.out.println("This is a classic 0-1 integer programming problem (0-1 knapsack problem).");
        System.out.println();
        System.out.println("Key characteristics:");
        System.out.println("1. Each variable can only be 0 or 1");
        System.out.println("2. Objective is to maximize value");
        System.out.println("3. Subject to weight constraint");
        System.out.println("4. Solved using branch and bound method");
    }
}