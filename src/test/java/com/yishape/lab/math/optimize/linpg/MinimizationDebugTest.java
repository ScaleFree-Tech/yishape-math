package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Debug test for the minimization convergence issue
 */
public class MinimizationDebugTest {
    
    private static final double TOLERANCE = 1e-6;
    private RereSimplexLinProgSolver solver;
    
    @BeforeEach
    void setUp() {
        solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true); // Enable verbose logging
    }
    
    @Test
    @DisplayName("Debug Standard Minimization Problem")
    void debugStandardMinimizationProblem() {
        System.out.println("\n=== DEBUG STANDARD MINIMIZATION PROBLEM ===");
        
        // Original problem: min 3*x1 + 2*x2
        // Constraints: x1 + x2 >= 4, 2*x1 + x2 >= 6, x1,x2 >= 0
        
        System.out.println("Original problem:");
        System.out.println("Minimize: 3*x1 + 2*x2");
        System.out.println("Subject to: x1 + x2 >= 4");
        System.out.println("           2*x1 + x2 >= 6");
        System.out.println("           x1, x2 >= 0");
        
        // Convert to <= constraints: -x1 - x2 <= -4, -2*x1 - x2 <= -6
        IVector c = Linalg.vector(new double[]{3.0, 2.0}); // Minimization objective
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {-1.0, -1.0},  // -x1 - x2 <= -4
            {-2.0, -1.0}   // -2*x1 - x2 <= -6
        });
        IVector b_ub = Linalg.vector(new double[]{-4.0, -6.0});
        
        System.out.println("\nConverted to <= form:");
        System.out.println("A_ub: " + A_ub);
        System.out.println("b_ub: " + b_ub);
        
        // Use LinProgUtil to convert to equality constraints
        var conversionResult = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        IVector convertedC = conversionResult.getFirst();
        IMatrix convertedA = conversionResult.getSecond();
        IVector convertedB = conversionResult.getThird();
        
        System.out.println("\nAfter conversion to equality constraints:");
        System.out.println("Converted objective c: " + convertedC);
        System.out.println("Converted constraint matrix A: " + convertedA);
        System.out.println("Converted RHS b: " + convertedB);
        
        // Check for negative RHS values
        boolean hasNegativeRHS = false;
        for (int i = 0; i < convertedB.length(); i++) {
            if (convertedB.get(i).doubleValue() < 0) {
                hasNegativeRHS = true;
                System.out.println("WARNING: Negative RHS value at index " + i + ": " + convertedB.get(i));
            }
        }
        
        System.out.println("Has negative RHS values: " + hasNegativeRHS);
        
        // Try to solve
        System.out.println("\n=== ATTEMPTING TO SOLVE ===");
        OptResult result = solver.solveWithNonNegativeEqualConstraints(convertedC, convertedA, convertedB, null);
        
        System.out.println("\n=== SOLUTION ANALYSIS ===");
        System.out.println("Result is null: " + (result == null));
        if (result != null) {
            System.out.println("Converged: " + result.isConverged());
            System.out.println("Convergence reason: " + result.getConvergenceReason());
            System.out.println("Iterations: " + result.getIterations());
            
            if (result.getOptimalPoint() != null) {
                System.out.println("Solution vector: " + result.getOptimalPoint());
                System.out.println("Optimal value: " + result.getOptimalValue());
                
                // Extract original variables (assuming first 2 are x1, x2)
                if (result.getOptimalPoint().length() >= 2) {
                    double x1 = result.getOptimalPoint().get(0).doubleValue();
                    double x2 = result.getOptimalPoint().get(1).doubleValue();
                    
                    System.out.println("x1 = " + x1 + ", x2 = " + x2);
                    
                    // Verify constraints
                    double constraint1 = x1 + x2;
                    double constraint2 = 2*x1 + x2;
                    System.out.println("Constraint 1: x1 + x2 = " + constraint1 + " >= 4? " + (constraint1 >= 4.0 - TOLERANCE));
                    System.out.println("Constraint 2: 2*x1 + x2 = " + constraint2 + " >= 6? " + (constraint2 >= 6.0 - TOLERANCE));
                    
                    // Verify objective
                    double objective = 3*x1 + 2*x2;
                    System.out.println("Objective: 3*x1 + 2*x2 = " + objective);
                    System.out.println("Expected optimal: x1=2, x2=2, objective=10");
                }
            } else {
                System.out.println("No solution point available");
            }
        }
    }
    
    @Test
    @DisplayName("Alternative Minimization Approach")
    void alternativeMinimizationApproach() {
        System.out.println("\n=== ALTERNATIVE MINIMIZATION APPROACH ===");
        
        // Try direct equality constraint formulation
        // Problem: min 3*x1 + 2*x2
        // Constraints: x1 + x2 >= 4 becomes x1 + x2 - s1 = 4 (with s1 >= 0)
        //             2*x1 + x2 >= 6 becomes 2*x1 + x2 - s2 = 6 (with s2 >= 0)
        
        // Alternative: Use slack variables properly
        // x1 + x2 - s1 = 4  =>  x1 + x2 + a1 = 4 (artificial variable)
        // 2*x1 + x2 - s2 = 6  =>  2*x1 + x2 + a2 = 6 (artificial variable)
        
        System.out.println("Converting >= constraints to equality with artificial variables:");
        System.out.println("x1 + x2 = 4 (simplified)");
        System.out.println("2*x1 + x2 = 6 (simplified)");
        
        // Solve the simplified system directly
        IVector c = Linalg.vector(new double[]{3.0, 2.0}); // Minimization objective  
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0},   // x1 + x2 = 4
            {2.0, 1.0}    // 2*x1 + x2 = 6
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});
        
        System.out.println("Simplified formulation:");
        System.out.println("c: " + c);
        System.out.println("A: " + A_eq);
        System.out.println("b: " + b_eq);
        
        // This is a 2x2 system, should have unique solution
        // From x1 + x2 = 4 and 2*x1 + x2 = 6:
        // Subtracting: x1 = 2, then x2 = 2
        // Objective = 3*2 + 2*2 = 10
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        System.out.println("\n=== SIMPLIFIED SOLUTION ===");
        if (result != null && result.isConverged()) {
            System.out.println("Successfully converged!");
            System.out.println("Solution: " + result.getOptimalPoint());
            System.out.println("Optimal value: " + result.getOptimalValue());
        } else {
            System.out.println("Failed to converge on simplified problem");
            if (result != null) {
                System.out.println("Convergence reason: " + result.getConvergenceReason());
            }
        }
    }
}