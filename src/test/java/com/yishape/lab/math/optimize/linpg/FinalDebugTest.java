package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Final debug test for the remaining minimization convergence issue
 */
public class FinalDebugTest {
    
    private RereSimplexLinProgSolver solver;
    private static final double TOLERANCE = 1e-6;

    @BeforeEach
    void setUp() {
        solver = new RereSimplexLinProgSolver();
    }

    @Test
    @DisplayName("Debug the exact failing test case")
    void testFailingMinimizationCase() {
        System.out.println("=== DEBUG: Exact failing minimization test case ===");
        
        // Exact same setup as the failing test
        IVector c = Linalg.vector(new double[]{3.0, 2.0}); // min 3*x1 + 2*x2
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {-1.0, -1.0},  // -x1 - x2 <= -4  =>  x1 + x2 >= 4
            {-2.0, -1.0}   // -2*x1 - x2 <= -6  =>  2*x1 + x2 >= 6
        });
        IVector b_ub = Linalg.vector(new double[]{-4.0, -6.0});
        
        System.out.println("Original problem:");
        System.out.println("min 3*x1 + 2*x2");
        System.out.println("subject to: x1 + x2 >= 4, 2*x1 + x2 >= 6, x1,x2 >= 0");
        System.out.println("Expected solution: x1=2, x2=2, objective=10");
        System.out.println();
        
        // Manual constraint conversion
        var conversionResult = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        IVector convertedC = conversionResult.getFirst();
        IMatrix convertedA = conversionResult.getSecond();
        IVector convertedB = conversionResult.getThird();
        
        System.out.println("After constraint conversion:");
        System.out.println("Objective c: " + convertedC);
        System.out.println("Constraint matrix A:");
        for (int i = 0; i < convertedA.rows(); i++) {
            System.out.println("  Row " + i + ": " + convertedA.getRow(i));
        }
        System.out.println("RHS vector b: " + convertedB);
        System.out.println();
        
        // Check if all b values are non-negative (required for Phase I)
        boolean allBNonNegative = true;
        for (int i = 0; i < convertedB.length(); i++) {
            if (convertedB.get(i) < 0) {
                allBNonNegative = false;
                System.out.println("WARNING: b[" + i + "] = " + convertedB.get(i) + " is negative!");
            }
        }
        System.out.println("All b values non-negative: " + allBNonNegative);
        System.out.println();
        
        // Try solving
        System.out.println("=== Starting solver ===");
        OptResult result = solver.solveWithNonNegativeEqualConstraints(convertedC, convertedA, convertedB, null);
        
        System.out.println("Solver result:");
        System.out.println("  Converged: " + result.isConverged());
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            System.out.println("  Full solution: " + solution);
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            System.out.println("  x1 = " + x1 + ", x2 = " + x2);
            System.out.println("  Objective value = " + result.getOptimalValue());
            
            // Verify constraints
            System.out.println("  Constraint verification:");
            System.out.println("    x1 + x2 = " + (x1 + x2) + " >= 4? " + (x1 + x2 >= 4.0 - TOLERANCE));
            System.out.println("    2*x1 + x2 = " + (2*x1 + x2) + " >= 6? " + (2*x1 + x2 >= 6.0 - TOLERANCE));
            System.out.println("    x1 >= 0? " + (x1 >= -TOLERANCE));
            System.out.println("    x2 >= 0? " + (x2 >= -TOLERANCE));
        } else {
            System.out.println("  ERROR: Solver did not converge!");
            System.out.println("  This suggests either:");
            System.out.println("    1. Unbounded solution detection error");
            System.out.println("    2. Infeasible problem detection error");
            System.out.println("    3. Numerical precision issues");
        }
        
        // Test alternative approach: Try with manual tableau construction
        System.out.println();
        System.out.println("=== Testing alternative manual approach ===");
        testAlternativeApproach();
    }
    
    private void testAlternativeApproach() {
        // Manual constraint setup: min 3*x1 + 2*x2 subject to x1+x2>=4, 2*x1+x2>=6, x1,x2>=0
        // Convert to standard form by adding slack variables with negative coefficients
        // x1 + x2 - s1 = 4  where s1 >= 0 (excess variable)
        // 2*x1 + x2 - s2 = 6  where s2 >= 0 (excess variable)
        
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0}); // min 3*x1 + 2*x2 + 0*s1 + 0*s2
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0},   // x1 + x2 - s1 = 4
            {2.0, 1.0, 0.0, -1.0}    // 2*x1 + x2 - s2 = 6
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});
        
        System.out.println("Alternative formulation:");
        System.out.println("Objective c: " + c);
        System.out.println("Constraint matrix A:");
        for (int i = 0; i < A_eq.rows(); i++) {
            System.out.println("  Row " + i + ": " + A_eq.getRow(i));
        }
        System.out.println("RHS vector b: " + b_eq);
        
        OptResult result2 = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        System.out.println("Alternative result:");
        System.out.println("  Converged: " + result2.isConverged());
        
        if (result2.isConverged()) {
            IVector solution = result2.getOptimalPoint();
            System.out.println("  Full solution: " + solution);
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            double s1 = solution.get(2);
            double s2 = solution.get(3);
            System.out.println("  x1 = " + x1 + ", x2 = " + x2 + ", s1 = " + s1 + ", s2 = " + s2);
            System.out.println("  Objective value = " + result2.getOptimalValue());
        }
    }
}