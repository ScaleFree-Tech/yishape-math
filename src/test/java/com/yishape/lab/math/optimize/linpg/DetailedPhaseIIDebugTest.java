package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Detailed debugging of Phase II optimization issue
 */
public class DetailedPhaseIIDebugTest {
    
    private RereSimplexLinProgSolver solver;
    private static final double TOLERANCE = 1e-6;

    @BeforeEach
    void setUp() {
        solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true); // Enable detailed output
    }

    @Test
    @DisplayName("Debug Phase II optimization step by step")
    void testDetailedPhaseIIDebugging() {
        System.out.println("=== DETAILED PHASE II DEBUGGING ===");
        
        // Problem: min 3*x1 + 2*x2 subject to x1+x2>=4, 2*x1+x2>=6, x1,x2>=0
        // Expected optimal: x1=2, x2=2, objective=10
        // Current bug: solver finds x1=6, x2=0, objective=18
        
        double bigM = 10000.0;
        
        // Variables: [x1, x2, s1, s2, a1, a2]
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, bigM, bigM});
        
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 1.0, 0.0},   // x1 + x2 - s1 + a1 = 4
            {2.0, 1.0, 0.0, -1.0, 0.0, 1.0}    // 2*x1 + x2 - s2 + a2 = 6
        });
        
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});
        
        System.out.println("Problem setup:");
        System.out.println("  Objective: " + c);
        System.out.println("  Constraints: " + A_eq);
        System.out.println("  RHS: " + b_eq);
        System.out.println("  Big M: " + bigM);
        System.out.println();
        
        System.out.println("Expected optimal solution analysis:");
        System.out.println("  Point (2,2): constraints 2+2=4>=4✓, 2*2+2=6>=6✓, objective=3*2+2*2=10");
        System.out.println("  Point (6,0): constraints 6+0=6>=4✓, 2*6+0=12>=6✓, objective=3*6+2*0=18");
        System.out.println("  Point (0,6): constraints 0+6=6>=4✓, 2*0+6=6>=6✓, objective=3*0+2*6=12");
        System.out.println("  Therefore (2,2) with objective=10 should be optimal (minimum)");
        System.out.println();
        
        // Solve with detailed output
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        System.out.println("=== SOLVER RESULT ANALYSIS ===");
        System.out.println("Converged: " + result.isConverged());
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            double x1 = solution.get(0).doubleValue();
            double x2 = solution.get(1).doubleValue();
            double s1 = solution.get(2).doubleValue();
            double s2 = solution.get(3).doubleValue();
            double a1 = solution.get(4).doubleValue();
            double a2 = solution.get(5).doubleValue();
            
            System.out.println("Solution found:");
            System.out.println("  x1 = " + x1 + ", x2 = " + x2);
            System.out.println("  s1 = " + s1 + ", s2 = " + s2);
            System.out.println("  a1 = " + a1 + ", a2 = " + a2);
            System.out.println("  Reported objective = " + result.getOptimalValue());
            
            double trueObjective = 3*x1 + 2*x2;
            System.out.println("  True objective (3*x1 + 2*x2) = " + trueObjective);
            
            // Check constraint satisfaction
            double constraint1 = x1 + x2;
            double constraint2 = 2*x1 + x2;
            System.out.println("Constraint verification:");
            System.out.println("  x1 + x2 = " + constraint1 + " >= 4? " + (constraint1 >= 4.0 - TOLERANCE));
            System.out.println("  2*x1 + x2 = " + constraint2 + " >= 6? " + (constraint2 >= 6.0 - TOLERANCE));
            
            // Identify the issue
            if (Math.abs(x1 - 6.0) < TOLERANCE && Math.abs(x2 - 0.0) < TOLERANCE) {
                System.out.println("ISSUE IDENTIFIED: Solver found vertex (6,0) instead of optimal (2,2)");
                System.out.println("This suggests the Phase II optimization terminated prematurely");
                System.out.println("The vertex (6,0) is on constraint boundary 2*x1 + x2 = 6 and x2 = 0");
                System.out.println("The optimal vertex (2,2) is at intersection of x1+x2=4 and 2*x1+x2=6");
                System.out.println();
                
                // Test manual optimization direction
                testManualOptimizationDirection(x1, x2);
            }
            
            // Test if optimality conditions are correctly checked
            testOptimalityConditions(x1, x2, trueObjective);
            
        } else {
            System.out.println("SOLVER FAILED TO CONVERGE");
            fail("Solver should converge for this problem");
        }
    }
    
    private void testManualOptimizationDirection(double currentX1, double currentX2) {
        System.out.println("=== MANUAL OPTIMIZATION DIRECTION TEST ===");
        
        // From current point (6,0), test if we can improve by moving toward (2,2)
        double directionX1 = 2.0 - currentX1;  // -4
        double directionX2 = 2.0 - currentX2;  // +2
        
        System.out.println("Direction from (" + currentX1 + "," + currentX2 + ") to (2,2): (" + directionX1 + "," + directionX2 + ")");
        
        // Test different step sizes
        for (double step : new double[]{0.1, 0.25, 0.5, 1.0}) {
            double testX1 = currentX1 + step * directionX1;
            double testX2 = currentX2 + step * directionX2;
            
            double testObjective = 3*testX1 + 2*testX2;
            double constraint1 = testX1 + testX2;
            double constraint2 = 2*testX1 + testX2;
            
            boolean feasible = (constraint1 >= 4.0 - TOLERANCE) && 
                              (constraint2 >= 6.0 - TOLERANCE) &&
                              (testX1 >= -TOLERANCE) && (testX2 >= -TOLERANCE);
            
            System.out.println("  Step " + step + ": (" + testX1 + "," + testX2 + ") objective=" + testObjective + " feasible=" + feasible);
        }
    }
    
    private void testOptimalityConditions(double x1, double x2, double objectiveValue) {
        System.out.println("=== OPTIMALITY CONDITIONS TEST ===");
        
        // For the point (x1,x2), check if it satisfies KKT conditions
        System.out.println("Testing KKT conditions at point (" + x1 + "," + x2 + "):");
        
        // Gradient of objective function: ∇f = [3, 2]
        double[] gradient = {3.0, 2.0};
        System.out.println("  Objective gradient: [" + gradient[0] + ", " + gradient[1] + "]");
        
        // Active constraints at current point
        double constraint1 = x1 + x2;
        double constraint2 = 2*x1 + x2;
        
        boolean constraint1Active = Math.abs(constraint1 - 4.0) < TOLERANCE;
        boolean constraint2Active = Math.abs(constraint2 - 6.0) < TOLERANCE;
        boolean x1BoundActive = Math.abs(x1) < TOLERANCE;
        boolean x2BoundActive = Math.abs(x2) < TOLERANCE;
        
        System.out.println("  Active constraints:");
        System.out.println("    x1 + x2 >= 4: " + constraint1Active + " (value=" + constraint1 + ")");
        System.out.println("    2*x1 + x2 >= 6: " + constraint2Active + " (value=" + constraint2 + ")");
        System.out.println("    x1 >= 0: " + x1BoundActive + " (value=" + x1 + ")");
        System.out.println("    x2 >= 0: " + x2BoundActive + " (value=" + x2 + ")");
        
        // For optimality, gradient should be expressible as non-negative combination of active constraint normals
        if (constraint2Active && x2BoundActive) {
            System.out.println("  At (6,0): active constraints are 2*x1+x2>=6 and x2>=0");
            System.out.println("  Constraint normals: [2,1] for first, [0,1] for second");
            System.out.println("  For optimality: [3,2] = λ1*[2,1] + λ2*[0,1] with λ1,λ2>=0");
            System.out.println("  This gives: 3 = 2*λ1, 2 = λ1 + λ2");
            System.out.println("  Solution: λ1 = 1.5, λ2 = 0.5 (both >= 0)");
            System.out.println("  RESULT: (6,0) satisfies KKT conditions - it's a local minimum!");
            System.out.println("  But (2,2) has lower objective value, so solver should continue!");
        }
    }
}