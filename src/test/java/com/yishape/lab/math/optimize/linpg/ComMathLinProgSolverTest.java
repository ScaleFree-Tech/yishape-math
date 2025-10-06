package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * ComMathLinProgSolver测试类
 */
public class ComMathLinProgSolverTest {

    /**
     * 测试简单的线性规划问题
     * 最小化: x1 + x2
     * 约束: x1 + x2 = 2
     *       x1, x2 >= 0
     * 期望解: x1 = 0, x2 = 2 或 x1 = 2, x2 = 0 (目标值 = 2)
     */
    @Test
    public void testSimpleLinearProgramming() {
        // 目标函数系数 (最小化 x1 + x2)
        IVector c = Linalg.vector(new double[]{1.0, 1.0});
        
        // 等式约束: x1 + x2 = 2
        IMatrix A_eq = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector b_eq = Linalg.vector(new double[]{2.0});
        
        // 初始值
        IVector initX = Linalg.vector(new double[]{1.0, 1.0});
        
        // 创建求解器
        ComMath4LinProgSolver solver = new ComMath4LinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, initX);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertTrue(result.isConverged(), "应收敛");
        assertEquals(2.0, result.getOptimalValue(), 1e-6, "目标值应为2");
        
        // 验证解的维度
        Assertions.assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        Assertions.assertEquals(2, result.getOptimalPoint().length(), "解应有2个变量");
        
        // 验证约束满足
        IVector solution = result.getOptimalPoint();
        double x1 = (Double) solution.get(0);
        double x2 = (Double) solution.get(1);
        assertEquals(2.0, x1 + x2, 1e-6, "应满足 x1 + x2 = 2");
        assertTrue(x1 >= 0, "x1应非负");
        assertTrue(x2 >= 0, "x2应非负");
    }
    
    /**
     * 测试无解的情况
     * 最小化: x1 + x2
     * 约束: x1 + x2 = 2
     *       x1 + x2 = 3 (矛盾约束)
     *       x1, x2 >= 0
     */
    @Test
    public void testInfeasibleProblem() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1.0, 1.0});
        
        // 矛盾的等式约束
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0},
            {1.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{2.0, 3.0});
        
        // 创建求解器
        ComMath4LinProgSolver solver = new ComMath4LinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        // 对于无解情况，可以返回null或不收敛的结果
        // 这取决于具体实现，但我们至少要确保不会抛出异常
        assertTrue(result == null || !result.isConverged(), 
                  "对于无解问题，应返回null或不收敛的结果");
    }
    
    /**
     * 测试只有目标函数没有约束的情况
     * 最小化: x1 + x2
     * 约束: x1, x2 >= 0
     * 期望解: x1 = 0, x2 = 0 (目标值 = 0)
     */
    @Test
    public void testUnconstrainedProblem() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1.0, 1.0});
        
        // 无等式约束
        IMatrix A_eq = null;
        IVector b_eq = null;
        
        // 创建求解器
        ComMath4LinProgSolver solver = new ComMath4LinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertTrue(result.isConverged(), "应收敛");
        assertEquals(0.0, result.getOptimalValue(), 1e-6, "目标值应为0");
        
        // 验证解
        Assertions.assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        Assertions.assertEquals(2, result.getOptimalPoint().length(), "解应有2个变量");
        
        IVector solution = result.getOptimalPoint();
        double x1 = (Double) solution.get(0);
        double x2 = (Double) solution.get(1);
        assertEquals(0.0, x1, 1e-6, "x1应为0");
        assertEquals(0.0, x2, 1e-6, "x2应为0");
    }
}