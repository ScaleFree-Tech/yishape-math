package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试单纯形法求解器在达到最大迭代次数时的行为
 * Test Simplex solver behavior when maximum iterations are reached
 */
@DisplayName("单纯形法求解器最大迭代测试 / Simplex Solver Max Iterations Tests")
public class SimplexMaxIterationsTest {
    
    @Test
    @DisplayName("测试单纯形法求解器在达到最大迭代次数时返回OptResult而不是null / Test Simplex solver returns OptResult instead of null when max iterations reached")
    public void testSimplexSolverReturnsOptResultOnMaxIterations() {
        // 创建一个可能需要很多迭代才能解决的问题
        // 目标函数系数 (最小化问题)
        IVector c = Linalg.vector(new double[]{1, 1, 1, 1, 1});
        
        // 等式约束: x1 + x2 + x3 + x4 + x5 = 5
        //          x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 = 15
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1, 1, 1, 1},
            {1, 2, 3, 4, 5}
        });
        IVector b_eq = Linalg.vector(new double[]{5, 15});
        
        // 创建求解器
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果不为null
        assertNotNull(result, "结果不应为null，即使达到最大迭代次数");
        
        // 验证基本属性
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        assertEquals(5, result.getOptimalPoint().length(), "解的维度应为5");
        
        // 验证收敛信息
        assertNotNull(result.getConvergenceReason(), "应该有收敛原因");
        assertTrue(result.getIterations() >= 0, "迭代次数应该非负");
        assertTrue(result.getMaxIterations() > 0, "最大迭代次数应该大于0");
        
        // 验证执行时间
        assertTrue(result.getExecutionTimeMs() >= 0, "执行时间应该非负");
        
        // 验证评估计数
        assertTrue(result.getFunctionEvaluations() > 0, "函数评估次数应该大于0");
        
        // 验证历史信息
        assertNotNull(result.getFunctionValueHistory(), "应该有函数值历史");
        assertNotNull(result.getParameterHistory(), "应该有参数历史");
        assertFalse(result.getFunctionValueHistory().isEmpty(), "函数值历史不应该为空");
        assertFalse(result.getParameterHistory().isEmpty(), "参数历史不应该为空");
        
        // 验证约束满足
        IVector constraintCheck = A_eq.mmul(result.getOptimalPoint());
        assertEquals(5.0, (double)constraintCheck.get(0), 1e-3, "第一个约束应该满足");
        assertEquals(15.0, (double)constraintCheck.get(1), 1e-3, "第二个约束应该满足");
        
        // 验证非负约束
        for (int i = 0; i < result.getOptimalPoint().length(); i++) {
            assertTrue((double)result.getOptimalPoint().get(i) >= -1e-6, "变量应该非负");
        }
    }
}