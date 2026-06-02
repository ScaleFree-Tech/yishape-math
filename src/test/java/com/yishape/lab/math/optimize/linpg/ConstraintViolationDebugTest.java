package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Debug test to analyze constraint violation issues in RereSimplexLinProgSolver
 */
public class ConstraintViolationDebugTest {
    
    private static final double TOLERANCE = 1e-6;
    private RereSimplexLinProgSolver solver;
    
    @BeforeEach
    void setUp() {
        solver = new RereSimplexLinProgSolver();
    }
    
    @Test
    @DisplayName("Debug Standard Maximization Problem")
    void debugStandardMaximizationProblem() {
        System.out.println("\n=== DEBUG STANDARD MAXIMIZATION PROBLEM ===");
        
        // Problem: max 2*x1 + 3*x2
        // Constraints: x1 + 2*x2 <= 8, 2*x1 + x2 <= 10, x1,x2 >= 0
        // Convert to minimization: min -2*x1 - 3*x2
        // Add slack variables: x1 + 2*x2 + s1 = 8, 2*x1 + x2 + s2 = 10
        
        IVector c = Linalg.vector(new double[]{-2.0, -3.0, 0.0, 0.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 2.0, 1.0, 0.0},  // x1 + 2*x2 + s1 = 8
            {2.0, 1.0, 0.0, 1.0}   // 2*x1 + x2 + s2 = 10
        });
        IVector b_eq = Linalg.vector(new double[]{8.0, 10.0});
        
        System.out.println("Objective vector c: " + c);
        System.out.println("Constraint matrix A_eq:");
        for (int i = 0; i < A_eq.rows(); i++) {
            System.out.print("  Row " + i + ": ");
            for (int j = 0; j < A_eq.cols(); j++) {
                System.out.printf("%.1f ", A_eq.get(i, j));
            }
            System.out.println();
        }
        System.out.println("Constraint vector b_eq: " + b_eq);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "Result should not be null");
        assertTrue(result.isConverged(), "Problem should converge");
        
        IVector solution = result.getOptimalPoint();
        assertNotNull(solution, "Solution should not be null");
        
        double x1 = solution.get(0);
        double x2 = solution.get(1);
        double s1 = solution.get(2);
        double s2 = solution.get(3);
        
        System.out.println("\n=== SOLUTION ANALYSIS ===");
        System.out.println("Solution vector: " + solution);
        System.out.println("x1 = " + x1);
        System.out.println("x2 = " + x2);
        System.out.println("s1 = " + s1);
        System.out.println("s2 = " + s2);
        System.out.println("Optimal value (minimization): " + result.getOptimalValue());
        System.out.println("Optimal value (maximization): " + (-result.getOptimalValue()));
        
        // Verify constraints manually
        double constraint1_lhs = x1 + 2*x2;
        double constraint2_lhs = 2*x1 + x2;
        
        System.out.println("\n=== CONSTRAINT VERIFICATION ===");
        System.out.println("Constraint 1: x1 + 2*x2 = " + constraint1_lhs + " <= 8.0? " + (constraint1_lhs <= 8.0 + TOLERANCE));
        System.out.println("Constraint 2: 2*x1 + x2 = " + constraint2_lhs + " <= 10.0? " + (constraint2_lhs <= 10.0 + TOLERANCE));
        System.out.println("Non-negativity: x1 >= 0? " + (x1 >= -TOLERANCE) + ", x2 >= 0? " + (x2 >= -TOLERANCE));
        
        // Also verify using slack variables
        System.out.println("Slack variable verification:");
        System.out.println("s1 = 8 - (x1 + 2*x2) = " + (8.0 - constraint1_lhs) + " (should equal s1=" + s1 + ")");
        System.out.println("s2 = 10 - (2*x1 + x2) = " + (10.0 - constraint2_lhs) + " (should equal s2=" + s2 + ")");
        
        // Calculate objective manually
        double manual_objective = 2*x1 + 3*x2; // Original maximization objective
        System.out.println("Manual objective calculation: 2*" + x1 + " + 3*" + x2 + " = " + manual_objective);
        
        // Verify optimality (expected solution: x1=4, x2=2, objective=14)
        System.out.println("\n=== OPTIMALITY CHECK ===");
        System.out.println("Expected optimal solution: x1=4, x2=2, objective=14");
        System.out.println("Actual solution: x1=" + x1 + ", x2=" + x2 + ", objective=" + manual_objective);
        
        // Test assertions
        assertTrue(constraint1_lhs <= 8.0 + TOLERANCE, "First constraint should be satisfied");
        assertTrue(constraint2_lhs <= 10.0 + TOLERANCE, "Second constraint should be satisfied");
        assertTrue(x1 >= -TOLERANCE && x2 >= -TOLERANCE, "Non-negativity should be satisfied");
    }
    
    @Test
    @DisplayName("Debug Three Variable Problem")
    void debugThreeVariableProblem() {
        System.out.println("\n=== DEBUG THREE VARIABLE PROBLEM ===");
        
        // Problem: max x1 + 2*x2 + 3*x3
        // Constraints: x1 + x2 + x3 <= 6, 2*x1 + x2 <= 8, x2 + 2*x3 <= 10
        // Convert to minimization: min -x1 - 2*x2 - 3*x3
        
        IVector c = Linalg.vector(new double[]{-1.0, -2.0, -3.0, 0.0, 0.0, 0.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0, 1.0, 0.0, 0.0},  // x1 + x2 + x3 + s1 = 6
            {2.0, 1.0, 0.0, 0.0, 1.0, 0.0},  // 2*x1 + x2 + s2 = 8
            {0.0, 1.0, 2.0, 0.0, 0.0, 1.0}   // x2 + 2*x3 + s3 = 10
        });
        IVector b_eq = Linalg.vector(new double[]{6.0, 8.0, 10.0});
        
        System.out.println("Objective vector c: " + c);
        System.out.println("Constraint matrix A_eq:");
        for (int i = 0; i < A_eq.rows(); i++) {
            System.out.print("  Row " + i + ": ");
            for (int j = 0; j < A_eq.cols(); j++) {
                System.out.printf("%.1f ", A_eq.get(i, j));
            }
            System.out.println();
        }
        System.out.println("Constraint vector b_eq: " + b_eq);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "Result should not be null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "Solution should not be null");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            double x3 = solution.get(2);
            double s1 = solution.get(3);
            double s2 = solution.get(4);
            double s3 = solution.get(5);
            
            System.out.println("\n=== SOLUTION ANALYSIS ===");
            System.out.println("Solution vector: " + solution);
            System.out.println("x1 = " + x1 + ", x2 = " + x2 + ", x3 = " + x3);
            System.out.println("s1 = " + s1 + ", s2 = " + s2 + ", s3 = " + s3);
            
            // Verify constraints manually
            double constraint1_lhs = x1 + x2 + x3;
            double constraint2_lhs = 2*x1 + x2;
            double constraint3_lhs = x2 + 2*x3;
            
            System.out.println("\n=== CONSTRAINT VERIFICATION ===");
            System.out.println("Constraint 1: x1 + x2 + x3 = " + constraint1_lhs + " <= 6.0? " + (constraint1_lhs <= 6.0 + TOLERANCE));
            System.out.println("Constraint 2: 2*x1 + x2 = " + constraint2_lhs + " <= 8.0? " + (constraint2_lhs <= 8.0 + TOLERANCE));
            System.out.println("Constraint 3: x2 + 2*x3 = " + constraint3_lhs + " <= 10.0? " + (constraint3_lhs <= 10.0 + TOLERANCE));
            
            assertTrue(constraint1_lhs <= 6.0 + TOLERANCE, "First constraint should be satisfied");
            assertTrue(constraint2_lhs <= 8.0 + TOLERANCE, "Second constraint should be satisfied");
            assertTrue(constraint3_lhs <= 10.0 + TOLERANCE, "Third constraint should be satisfied");
        } else {
            System.out.println("Three variable problem did not converge");
            fail("Three variable problem should converge");
        }
    }
    
    @Test
    @DisplayName("Debug Degenerate Problem")
    void debugDegenerateProblem() {
        System.out.println("\n=== DEBUG DEGENERATE PROBLEM ===");
        
        // Problem: max x1 + x2
        // Constraints: x1 <= 2, x2 <= 2, x1 + x2 <= 3
        // Convert to minimization: min -x1 - x2
        
        IVector c = Linalg.vector(new double[]{-1.0, -1.0, 0.0, 0.0, 0.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 0.0, 1.0, 0.0, 0.0},  // x1 + s1 = 2
            {0.0, 1.0, 0.0, 1.0, 0.0},  // x2 + s2 = 2
            {1.0, 1.0, 0.0, 0.0, 1.0}   // x1 + x2 + s3 = 3
        });
        IVector b_eq = Linalg.vector(new double[]{2.0, 2.0, 3.0});
        
        System.out.println("Objective vector c: " + c);
        System.out.println("Constraint matrix A_eq:");
        for (int i = 0; i < A_eq.rows(); i++) {
            System.out.print("  Row " + i + ": ");
            for (int j = 0; j < A_eq.cols(); j++) {
                System.out.printf("%.1f ", A_eq.get(i, j));
            }
            System.out.println();
        }
        System.out.println("Constraint vector b_eq: " + b_eq);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "Result should not be null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "Solution should not be null");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            
            System.out.println("\n=== SOLUTION ANALYSIS ===");
            System.out.println("Solution vector: " + solution);
            System.out.println("x1 = " + x1 + ", x2 = " + x2);
            
            // Verify constraints manually
            System.out.println("\n=== CONSTRAINT VERIFICATION ===");
            System.out.println("Constraint 1: x1 = " + x1 + " <= 2.0? " + (x1 <= 2.0 + TOLERANCE));
            System.out.println("Constraint 2: x2 = " + x2 + " <= 2.0? " + (x2 <= 2.0 + TOLERANCE));
            System.out.println("Constraint 3: x1 + x2 = " + (x1 + x2) + " <= 3.0? " + ((x1 + x2) <= 3.0 + TOLERANCE));
            
            assertTrue(x1 <= 2.0 + TOLERANCE, "x1 constraint should be satisfied");
            assertTrue(x2 <= 2.0 + TOLERANCE, "x2 constraint should be satisfied");
            assertTrue(x1 + x2 <= 3.0 + TOLERANCE, "Sum constraint should be satisfied");
        } else {
            System.out.println("Degenerate problem did not converge");
            fail("Degenerate problem should converge");
        }
    }
}