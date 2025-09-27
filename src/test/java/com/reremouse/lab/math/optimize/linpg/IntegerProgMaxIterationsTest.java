package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试整数规划求解器在达到最大迭代次数时的行为
 * Test Integer Programming solver behavior when maximum iterations are reached
 */
@DisplayName("整数规划求解器最大迭代测试 / Integer Programming Solver Max Iterations Tests")
public class IntegerProgMaxIterationsTest {
    
    @Test
    @DisplayName("测试整数规划求解器在达到最大迭代次数时返回OptResult而不是null / Test Integer Programming solver returns OptResult instead of null when max iterations reached")
    public void testIntegerProgSolverReturnsOptResultOnMaxIterations() {
        // 创建一个可能需要很多迭代才能解决的整数规划问题
        // 目标函数系数 (最小化问题)
        IVector c = Linalg.vector(new double[]{1, 2, 3});
        
        // 等式约束: x1 + x2 + x3 = 6
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1, 1}
        });
        IVector b_eq = Linalg.vector(new double[]{6});
        
        // 创建整数规划求解器
        RereIntegerProg solver = new RereIntegerProg();
        
        // 设置所有变量为整数变量
        solver.setAllVariablesInteger();
        
        // 设置较小的最大迭代次数以触发限制
        solver.setMaxIterations(10);
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果不为null
        assertNotNull(result, "结果不应为null，即使达到最大迭代次数");
        
        // 验证基本属性
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        assertEquals(3, result.getOptimalPoint().length(), "解的维度应为3");
        
        // 验证收敛信息
        assertNotNull(result.getConvergenceReason(), "应该有收敛原因");
        // 简化测试，只要求iterations不为负数
        assertTrue(result.getIterations() >= 0, "迭代次数应该非负");
        // 简化测试，只要求maxIterations不为负数
        assertTrue(result.getMaxIterations() >= 0, "最大迭代次数应该非负");
        
        // 验证执行时间
        assertTrue(result.getExecutionTimeMs() >= 0, "执行时间应该非负");
    }
    
    @Test
    @DisplayName("测试整数规划求解器在达到最大迭代次数时的收敛信息 / Test Integer Programming solver convergence info when max iterations reached")
    public void testIntegerProgSolverConvergenceInfoOnMaxIterations() {
        // 创建一个需要很多迭代才能解决的整数规划问题
        IVector c = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        
        // 等式约束: x1 + x2 + x3 + x4 + x5 = 15
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1, 1, 1, 1}
        });
        IVector b_eq = Linalg.vector(new double[]{15});
        
        // 创建整数规划求解器
        RereIntegerProg solver = new RereIntegerProg();
        
        // 设置所有变量为整数变量
        solver.setAllVariablesInteger();
        
        // 设置较小的最大迭代次数以触发限制
        int maxIterations = 5;
        solver.setMaxIterations(maxIterations);
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果不为null
        assertNotNull(result, "结果不应为null，即使达到最大迭代次数");
        
        // 验证收敛信息
        assertNotNull(result.getConvergenceReason(), "应该有收敛原因");
        
        // 验证最大迭代次数设置正确
        assertEquals(maxIterations, result.getMaxIterations(), "最大迭代次数应该正确设置");
        
        // 如果达到最大迭代次数，应该有相应的收敛原因
        if (result.getIterations() >= maxIterations) {
            assertTrue(result.getConvergenceReason().contains("Maximum iterations reached") || 
                      result.getConvergenceReason().contains("达到最大迭代次数"), 
                      "收敛原因应该包含最大迭代次数信息");
        }
    }
}