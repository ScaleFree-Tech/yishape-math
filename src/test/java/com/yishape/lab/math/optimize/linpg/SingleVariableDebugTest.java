package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;

/**
 * Debug the single variable problem in detail
 */
public class SingleVariableDebugTest {
    
    @Test
    public void debugSingleVariableProblem() {
        System.out.println("=== Single Variable Problem Debug ===");
        
        // From the failing test: min -2*x1, subject to x1 + s1 = 5
        // This should maximize 2*x1 subject to the same constraint
        // Expected result: x1 = 5, objective = -10 (for minimization)
        
        IVector c = Linalg.vector(new double[]{-2.0, 0.0}); // minimize -2*x1
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0}  // x1 + s1 = 5
        });
        IVector b_eq = Linalg.vector(new double[]{5.0});
        
        System.out.println("Original problem (minimization):");
        System.out.println("c = " + c);
        System.out.println("A_eq = " + A_eq);
        System.out.println("b_eq = " + b_eq);
        System.out.println("Problem: minimize -2*x1 + 0*s1 subject to x1 + s1 = 5, x1,s1 >= 0");
        System.out.println("Expected solution: x1 = 5, s1 = 0, objective = -10");
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true); // Enable detailed logging
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        System.out.println("\\n=== SOLUTION ANALYSIS ===");
        System.out.println("Converged: " + result.isConverged());
        System.out.println("Objective value: " + result.getOptimalValue());
        System.out.println("Solution vector: " + result.getOptimalPoint());
        
        if (result.getOptimalPoint() != null) {
            double x1 = result.getOptimalPoint().get(0).doubleValue();
            double s1 = result.getOptimalPoint().get(1).doubleValue();
            
            System.out.println("x1 = " + x1);
            System.out.println("s1 = " + s1);
            
            // Verify constraint
            double constraint_lhs = x1 + s1;
            System.out.println("Constraint verification: x1 + s1 = " + constraint_lhs + " (should be 5.0)");
            
            // Verify objective
            double computed_obj = -2.0 * x1 + 0.0 * s1;
            System.out.println("Computed objective: " + computed_obj + " (should be -10.0 for optimal)");
            
            // Check if we have the optimal solution
            boolean is_optimal = Math.abs(x1 - 5.0) < 1e-6 && Math.abs(s1 - 0.0) < 1e-6;
            System.out.println("Is optimal solution? " + is_optimal);
        }
    }
}