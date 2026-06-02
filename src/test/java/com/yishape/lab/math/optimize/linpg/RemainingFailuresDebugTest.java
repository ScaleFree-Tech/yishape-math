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
 * Debug test for remaining failing tests
 */
public class RemainingFailuresDebugTest {
    
    private static final double TOLERANCE = 1e-6;
    private RereSimplexLinProgSolver solver;
    
    @BeforeEach
    void setUp() {
        solver = new RereSimplexLinProgSolver();
    }
    
    @Test
    @DisplayName("Debug Large Numbers Problem")
    void debugLargeNumbersProblem() {
        System.out.println("\n=== DEBUG LARGE NUMBERS PROBLEM ===");
        
        // Copy exact code from failing test
        IVector c = Linalg.vector(new double[]{-1000.0, -2000.0, 0.0, 0.0}); // 最小化负目标函数
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {100.0, 200.0, 1.0, 0.0},  // 100*x1 + 200*x2 + s1 = 10000
            {300.0, 100.0, 0.0, 1.0}   // 300*x1 + 100*x2 + s2 = 15000
        });
        IVector b_eq = Linalg.vector(new double[]{10000.0, 15000.0});
        
        System.out.println("Problem setup (large numbers):");
        System.out.println("Objective: " + c);
        System.out.println("Constraints: " + A_eq);
        System.out.println("RHS: " + b_eq);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "解向量不应为null");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            
            System.out.println("\n=== SOLUTION VERIFICATION ===");
            System.out.println("Solution: x1=" + x1 + ", x2=" + x2);
            System.out.println("Full solution vector: " + solution);
            
            // The exact constraints from the failing test
            double constraint1_lhs = 100*x1 + 200*x2;
            double constraint2_lhs = 300*x1 + 100*x2;
            double adjustedTolerance = TOLERANCE * 1000; // Same as in original test
            
            System.out.println("Constraint 1: 100*x1 + 200*x2 = " + constraint1_lhs + " <= 10000.0? " + (constraint1_lhs <= 10000.0 + adjustedTolerance));
            System.out.println("Constraint 2: 300*x1 + 100*x2 = " + constraint2_lhs + " <= 15000.0? " + (constraint2_lhs <= 15000.0 + adjustedTolerance));
            System.out.println("Adjusted tolerance = " + adjustedTolerance);
            System.out.println("Constraint 1 violation: " + (constraint1_lhs - 10000.0));
            System.out.println("Constraint 2 violation: " + (constraint2_lhs - 15000.0));
            
            // 验证约束满足 - EXACT same code as failing test
            assertTrue(100*x1 + 200*x2 <= 10000.0 + TOLERANCE*1000, "第一个约束应满足");
            assertTrue(300*x1 + 100*x2 <= 15000.0 + TOLERANCE*1000, "第二个约束应满足");
            assertTrue(x1 >= -TOLERANCE && x2 >= -TOLERANCE, "非负约束应满足");
            
            System.out.println("大数值问题解: x1=" + x1 + ", x2=" + x2);
        } else {
            System.out.println("大数值问题未收敛，可能存在数值稳定性问题");
            fail("Large numbers problem should converge");
        }
    }
}