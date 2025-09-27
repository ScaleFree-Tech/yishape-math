package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试线性规划求解器增强功能
 * Test linear programming solver enhancement features
 */
@DisplayName("线性规划求解器增强功能测试 / Linear Programming Solver Enhancement Tests")
public class LinProgSolverEnhancementTest {
    
    @Test
    @DisplayName("测试单纯形法求解器返回丰富的OptResult信息 / Test Simplex solver returns rich OptResult information")
    public void testSimplexSolverRichOptResultInformation() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1, 1});
        
        // 等式约束: x1 + x2 = 2
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        // 创建求解器
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果不为null
        assertNotNull(result, "结果不应为null");
        
        // 验证基本结果
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证增强信息
        assertTrue(result.getIterations() >= 0, "迭代次数应该非负");
        assertTrue(result.getMaxIterations() > 0, "最大迭代次数应该大于0");
        
        assertTrue(result.getFunctionEvaluations() > 0, "函数评估次数应该大于0");
        
        assertTrue(result.getExecutionTimeMs() >= 0, "执行时间应该非负");
        
        assertNotNull(result.getConvergenceReason(), "应该有收敛原因");
        
        assertNotNull(result.getFunctionValueHistory(), "应该有函数值历史");
        assertFalse(result.getFunctionValueHistory().isEmpty(), "函数值历史不应该为空");
        
        assertNotNull(result.getParameterHistory(), "应该有参数历史");
        assertFalse(result.getParameterHistory().isEmpty(), "参数历史不应该为空");
    }
    
    @Test
    @DisplayName("测试内点法求解器返回丰富的OptResult信息 / Test Interior Point solver returns rich OptResult information")
    public void testInteriorPointSolverRichOptResultInformation() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1, 1});
        
        // 等式约束: x1 + x2 = 2
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        // 创建求解器
        InteriorPointLinProgSolver solver = new InteriorPointLinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果不为null
        assertNotNull(result, "结果不应为null");
        
        // 验证基本结果
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证增强信息
        assertTrue(result.getIterations() >= 0, "迭代次数应该非负");
        assertTrue(result.getMaxIterations() > 0, "最大迭代次数应该大于0");
        
        assertTrue(result.getFunctionEvaluations() > 0, "函数评估次数应该大于0");
        assertTrue(result.getGradientEvaluations() > 0, "梯度评估次数应该大于0");
        
        assertTrue(result.getExecutionTimeMs() >= 0, "执行时间应该非负");
        
        assertNotNull(result.getConvergenceReason(), "应该有收敛原因");
        
        assertNotNull(result.getFunctionValueHistory(), "应该有函数值历史");
        assertFalse(result.getFunctionValueHistory().isEmpty(), "函数值历史不应该为空");
        
        assertNotNull(result.getParameterHistory(), "应该有参数历史");
        assertFalse(result.getParameterHistory().isEmpty(), "参数历史不应该为空");
    }
    
    @Test
    @DisplayName("测试拉格朗日乘子法求解器返回丰富的OptResult信息 / Test Lagrange Multiplier solver returns rich OptResult information")
    public void testLagrangeMultiplierSolverRichOptResultInformation() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1, 1});
        
        // 等式约束: x1 + x2 = 2
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        // 创建求解器
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果不为null
        assertNotNull(result, "结果不应为null");
        
        // 验证基本结果
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证增强信息
        assertTrue(result.getIterations() >= 0, "迭代次数应该非负");
        assertTrue(result.getMaxIterations() > 0, "最大迭代次数应该大于0");
        
        assertTrue(result.getFunctionEvaluations() > 0, "函数评估次数应该大于0");
        assertTrue(result.getGradientEvaluations() > 0, "梯度评估次数应该大于0");
        
        assertTrue(result.getExecutionTimeMs() >= 0, "执行时间应该非负");
        
        assertNotNull(result.getConvergenceReason(), "应该有收敛原因");
        
        assertNotNull(result.getFunctionValueHistory(), "应该有函数值历史");
        assertFalse(result.getFunctionValueHistory().isEmpty(), "函数值历史不应该为空");
        
        assertNotNull(result.getParameterHistory(), "应该有参数历史");
        assertFalse(result.getParameterHistory().isEmpty(), "参数历史不应该为空");
    }
}