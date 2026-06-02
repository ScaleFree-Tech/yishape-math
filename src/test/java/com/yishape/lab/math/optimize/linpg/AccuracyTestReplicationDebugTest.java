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
 * Replicate the exact failing test case from BetterSimplexLinProgSolverAccuracyTest
 */
public class AccuracyTestReplicationDebugTest {
    
    private static final double TOLERANCE = 1e-6;
    private RereSimplexLinProgSolver solver;
    
    @BeforeEach
    void setUp() {
        solver = new RereSimplexLinProgSolver();
    }
    
    @Test
    @DisplayName("Exact Replication of testStandardMaximizationProblem")
    void replicateStandardMaximizationProblem() {
        System.out.println("\n=== EXACT REPLICATION OF STANDARD MAXIMIZATION TEST ===");
        
        // Copy the exact code from BetterSimplexLinProgSolverAccuracyTest
        IVector c = Linalg.vector(new double[]{-2.0, -3.0, 0.0, 0.0}); // 最小化负目标函数，包含松弛变量
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 2.0, 1.0, 0.0},  // x1 + 2*x2 + s1 = 8
            {2.0, 1.0, 0.0, 1.0}   // 2*x1 + x2 + s2 = 10
        });
        IVector b_eq = Linalg.vector(new double[]{8.0, 10.0});
        
        System.out.println("Problem setup (copied from failing test):");
        System.out.println("Objective: " + c);
        System.out.println("Constraints: " + A_eq);
        System.out.println("RHS: " + b_eq);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        assertTrue(result.isConverged(), "应该收敛到最优解");
        
        IVector solution = result.getOptimalPoint();
        assertNotNull(solution, "解向量不应为null");
        assertEquals(4, solution.length(), "解向量维度应为4（包括松弛变量）");
        
        // 验证解的正确性 (允许一定误差)
        double x1 = solution.get(0);
        double x2 = solution.get(1);
        
        System.out.println("\n=== SOLUTION VERIFICATION ===");
        System.out.println("Solution: x1=" + x1 + ", x2=" + x2);
        System.out.println("Full solution vector: " + solution);
        
        // The EXACT assertions from the failing test
        double constraint1_lhs = x1 + 2*x2;
        double constraint2_lhs = 2*x1 + x2;
        
        System.out.println("Constraint 1: x1 + 2*x2 = " + constraint1_lhs + " <= 8.0? " + (constraint1_lhs <= 8.0 + TOLERANCE));
        System.out.println("Constraint 2: 2*x1 + x2 = " + constraint2_lhs + " <= 10.0? " + (constraint2_lhs <= 10.0 + TOLERANCE));
        System.out.println("TOLERANCE = " + TOLERANCE);
        
        // 验证约束满足
        assertTrue(x1 + 2*x2 <= 8.0 + TOLERANCE, "第一个约束应满足");
        assertTrue(2*x1 + x2 <= 10.0 + TOLERANCE, "第二个约束应满足");
        assertTrue(x1 >= -TOLERANCE && x2 >= -TOLERANCE, "非负约束应满足");
        
        // 验证目标函数值（注意是最小化负值，所以要取负号）
        double actualObjective = -result.getOptimalValue(); // 转换回最大化问题的目标值
        double expectedObjective = 2*x1 + 3*x2;
        assertEquals(expectedObjective, actualObjective, TOLERANCE, "目标函数值应一致");
        
        System.out.println("标准最大化问题解: x1=" + x1 + ", x2=" + x2 + ", 目标值=" + actualObjective);
    }
    
    @Test
    @DisplayName("Exact Replication of testThreeVariableProblem")
    void replicateThreeVariableProblem() {
        System.out.println("\n=== EXACT REPLICATION OF THREE VARIABLE TEST ===");
        
        // Copy exact code from failing test
        IVector c = Linalg.vector(new double[]{-1.0, -2.0, -3.0, 0.0, 0.0, 0.0}); // 最小化负目标函数，包含松弛变量
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0, 1.0, 0.0, 0.0},  // x1 + x2 + x3 + s1 = 6
            {2.0, 1.0, 0.0, 0.0, 1.0, 0.0},  // 2*x1 + x2 + s2 = 8
            {0.0, 1.0, 2.0, 0.0, 0.0, 1.0}   // x2 + 2*x3 + s3 = 10
        });
        IVector b_eq = Linalg.vector(new double[]{6.0, 8.0, 10.0});
        
        System.out.println("Problem setup (copied from failing test):");
        System.out.println("Objective: " + c);
        System.out.println("Constraints: " + A_eq);
        System.out.println("RHS: " + b_eq);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "解向量不应为null");
            assertEquals(6, solution.length(), "解向量维度应为6（包括松弛变量）");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            double x3 = solution.get(2);
            
            System.out.println("\n=== SOLUTION VERIFICATION ===");
            System.out.println("Solution: x1=" + x1 + ", x2=" + x2 + ", x3=" + x3);
            System.out.println("Full solution vector: " + solution);
            
            // The EXACT constraints from the failing test
            double constraint1_lhs = x1 + x2 + x3;
            double constraint2_lhs = 2*x1 + x2;
            double constraint3_lhs = x2 + 2*x3;
            
            System.out.println("Constraint 1: x1 + x2 + x3 = " + constraint1_lhs + " <= 6.0? " + (constraint1_lhs <= 6.0 + TOLERANCE));
            System.out.println("Constraint 2: 2*x1 + x2 = " + constraint2_lhs + " <= 8.0? " + (constraint2_lhs <= 8.0 + TOLERANCE));
            System.out.println("Constraint 3: x2 + 2*x3 = " + constraint3_lhs + " <= 10.0? " + (constraint3_lhs <= 10.0 + TOLERANCE));
            
            // 验证约束满足
            assertTrue(x1 + x2 + x3 <= 6.0 + TOLERANCE, "第一个约束应满足");
            assertTrue(2*x1 + x2 <= 8.0 + TOLERANCE, "第二个约束应满足");
            assertTrue(x2 + 2*x3 <= 10.0 + TOLERANCE, "第三个约束应满足");
            assertTrue(x1 >= -TOLERANCE && x2 >= -TOLERANCE && x3 >= -TOLERANCE, "非负约束应满足");
            
            // 验证目标函数值
            double actualObjective = -result.getOptimalValue(); // 转换回最大化问题的目标值
            double expectedObjective = x1 + 2*x2 + 3*x3;
            assertEquals(expectedObjective, actualObjective, TOLERANCE, "目标函数值应一致");
            
            System.out.println("三变量问题解: x1=" + x1 + ", x2=" + x2 + ", x3=" + x3 + ", 目标值=" + actualObjective);
        } else {
            System.out.println("三变量问题未收敛");
            fail("三变量问题应该收敛");
        }
    }
    
    @Test
    @DisplayName("Exact Replication of testDegenerateProblem")
    void replicateDegenerateProblem() {
        System.out.println("\n=== EXACT REPLICATION OF DEGENERATE TEST ===");
        
        // Copy exact code from failing test
        IVector c = Linalg.vector(new double[]{-1.0, -1.0, 0.0, 0.0, 0.0}); // 最小化负目标函数
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 0.0, 1.0, 0.0, 0.0},  // x1 + s1 = 2
            {0.0, 1.0, 0.0, 1.0, 0.0},  // x2 + s2 = 2
            {1.0, 1.0, 0.0, 0.0, 1.0}   // x1 + x2 + s3 = 3
        });
        IVector b_eq = Linalg.vector(new double[]{2.0, 2.0, 3.0});
        
        System.out.println("Problem setup (copied from failing test):");
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
            
            // The EXACT constraints from the failing test
            System.out.println("Constraint 1: x1 = " + x1 + " <= 2.0? " + (x1 <= 2.0 + TOLERANCE));
            System.out.println("Constraint 2: x2 = " + x2 + " <= 2.0? " + (x2 <= 2.0 + TOLERANCE));
            System.out.println("Constraint 3: x1 + x2 = " + (x1 + x2) + " <= 3.0? " + ((x1 + x2) <= 3.0 + TOLERANCE));
            
            // 验证约束满足
            assertTrue(x1 <= 2.0 + TOLERANCE, "x1约束应满足");
            assertTrue(x2 <= 2.0 + TOLERANCE, "x2约束应满足");
            assertTrue(x1 + x2 <= 3.0 + TOLERANCE, "和约束应满足");
            assertTrue(x1 >= -TOLERANCE && x2 >= -TOLERANCE, "非负约束应满足");
            
            // 目标函数值应为3
            double actualObjective = -result.getOptimalValue(); // 转换回最大化问题的目标值
            assertEquals(3.0, actualObjective, TOLERANCE, "目标函数值应为3");
            
            System.out.println("退化问题解: x1=" + x1 + ", x2=" + x2 + ", 目标值=" + actualObjective);
        } else {
            System.out.println("退化问题未收敛");
            fail("退化问题应该收敛");
        }
    }
}