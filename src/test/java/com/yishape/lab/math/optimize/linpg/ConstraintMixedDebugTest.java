package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;

/**
 * Debug test for mixed inequality and equality constraints
 * 调试混合不等式和等式约束问题
 * 
 * @author RereMouse
 */
public class ConstraintMixedDebugTest {
    
    public static void main(String[] args) {
        System.out.println("=== Debug Mixed Constraints Problem ===");
        System.out.println("Maximize: 2*x1 + 3*x2");
        System.out.println("Subject to: x1 + x2 <= 4, 2*x1 + x2 = 5, x1,x2 >= 0");
        
        // Set up the problem
        IVector c = IVector.of(2.0, 3.0);  // objective coefficients
        
        // Inequality constraints: A_ub * x <= b_ub
        IMatrix A_ub = IMatrix.of(new double[][]{
            {1.0, 1.0}   // x1 + x2 <= 4
        });
        IVector b_ub = IVector.of(4.0);
        
        // Equality constraints: A_eq * x = b_eq
        IMatrix A_eq = IMatrix.of(new double[][]{
            {2.0, 1.0}   // 2*x1 + x2 = 5
        });
        IVector b_eq = IVector.of(5.0);
        
        System.out.println("\n=== MANUAL FEASIBLE REGION ANALYSIS ===");
        System.out.println("Constraints:");
        System.out.println("1. x1 + x2 <= 4");
        System.out.println("2. 2*x1 + x2 = 5  =>  x2 = 5 - 2*x1");
        System.out.println("3. x1, x2 >= 0");
        
        System.out.println("\nSubstituting constraint 2 into constraint 1:");
        System.out.println("x1 + (5 - 2*x1) <= 4");
        System.out.println("x1 + 5 - 2*x1 <= 4");
        System.out.println("5 - x1 <= 4");
        System.out.println("x1 >= 1");
        
        System.out.println("\nWith x2 = 5 - 2*x1 >= 0:");
        System.out.println("5 - 2*x1 >= 0");
        System.out.println("x1 <= 2.5");
        
        System.out.println("\nFeasible region: 1 <= x1 <= 2.5, x2 = 5 - 2*x1");
        System.out.println("Corner points:");
        System.out.println("  x1 = 1: x2 = 5 - 2*1 = 3, objective = 2*1 + 3*3 = 11");
        System.out.println("  x1 = 2.5: x2 = 5 - 2*2.5 = 0, objective = 2*2.5 + 3*0 = 5");
        System.out.println("Expected optimal: x1 = 1, x2 = 3, objective = 11");
        
        // Solve
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        
        OptResult result = solver.maximize(c, A_ub, b_ub, A_eq, b_eq, null);
        
        System.out.println("\n=== RESULT ANALYSIS ===");
        if (result.isConverged()) {
            System.out.println("✓ CONVERGED");
            IVector solution = result.getOptimalPoint();
            double objectiveValue = result.getOptimalValue();
            
            System.out.printf("Solution: x1 = %.6f, x2 = %.6f\n", 
                solution.get(0).doubleValue(), solution.get(1).doubleValue());
            System.out.printf("Objective value: %.6f\n", objectiveValue);
            
            // Verify constraints
            double x1 = solution.get(0).doubleValue();
            double x2 = solution.get(1).doubleValue();
            
            // Inequality constraint: x1 + x2 <= 4
            double ineqValue = x1 + x2;
            System.out.printf("Inequality check: %.6f + %.6f = %.6f <= 4: %s\n", 
                x1, x2, ineqValue, (ineqValue <= 4.000001) ? "✓" : "✗");
            
            // Equality constraint: 2*x1 + x2 = 5
            double eqValue = 2*x1 + x2;
            System.out.printf("Equality check: 2*%.6f + %.6f = %.6f = 5: %s\n", 
                x1, x2, eqValue, (Math.abs(eqValue - 5.0) < 1e-6) ? "✓" : "✗");
                
        } else {
            System.out.println("✗ FAILED TO CONVERGE");
            System.out.println("Reason: " + result.getConvergenceReason());
        }
    }
}