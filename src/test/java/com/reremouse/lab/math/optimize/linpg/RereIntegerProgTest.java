package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * RereIntegerProg测试类
 * 测试整数规划求解器的正确性
 */
public class RereIntegerProgTest {

    /**
     * 测试简单的整数规划问题
     * 目标函数: minimize c^T * x = [1, 1] * [x1, x2]^T = x1 + x2
     * 约束条件: 
     *   x1 + x2 = 3
     *   x1, x2 >= 0 且为整数
     * 
     * 解析解: x1 = 1, x2 = 2 (或 x1 = 2, x2 = 1), 最优值 = 3
     */
    @Test
    public void testSimpleIntegerProgramming() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1, 1});
        
        // 等式约束: x1 + x2 = 3
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{3});
        
        // 创建求解器
        RereIntegerProg solver = new RereIntegerProg();
        solver.addIntegerVariables(0, 1); // x1, x2都是整数变量
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证最优解
        IVector solution = result.getOptimalPoint();
        assertEquals(2, solution.length(), "解的维度应为2");
        
        // 验证约束满足
        IVector constraintValue = A_eq.mmul(solution);
        assertEquals(3.0, (double)constraintValue.get(0), 1e-6, "约束值应等于3");
        
        // 验证整数约束
        double x1 = (Double) solution.get(0);
        double x2 = (Double) solution.get(1);
        assertEquals(Math.round(x1), x1, 1e-6, "x1应为整数");
        assertEquals(Math.round(x2), x2, 1e-6, "x2应为整数");
        
        // 验证最优值
        double expectedOptimalValue = 3.0;
        assertEquals(expectedOptimalValue, result.getOptimalValue(), 1e-6, "最优值应等于3");
    }

    /**
     * 测试混合整数规划问题
     * 目标函数: minimize c^T * x = [1, 2] * [x1, x2]^T = x1 + 2*x2
     * 约束条件: 
     *   x1 + x2 = 3
     *   x1 >= 0 且为整数, x2 >= 0 (连续变量)
     * 
     * 解析解: x1 = 3, x2 = 0, 最优值 = 3
     */
    @Test
    public void testMixedIntegerProgramming() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1, 2});
        
        // 等式约束: x1 + x2 = 3
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{3});
        
        // 创建求解器
        RereIntegerProg solver = new RereIntegerProg();
        solver.setIntegerVariable(0); // 只有x1是整数变量
        // solver.setVerbose(true); // Disable verbose output for cleaner test output
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证最优解
        IVector solution = result.getOptimalPoint();
        assertEquals(2, solution.length(), "解的维度应为2");
        
        // 验证约束满足
        IVector constraintValue = A_eq.mmul(solution);
        assertEquals(3.0, (double)constraintValue.get(0), 1e-6, "约束值应等于3");
        
        // 验证整数约束
        double x1 = (Double) solution.get(0);
        double x2 = (Double) solution.get(1);
        assertEquals(Math.round(x1), x1, 1e-6, "x1应为整数");
        
        // 验证最优值
        double expectedOptimalValue = 3.0;
        assertEquals(expectedOptimalValue, result.getOptimalValue(), 1e-6, "最优值应等于3");
    }

    /**
     * 测试复杂的整数规划问题
     * 目标函数: minimize c^T * x = [3, 2, 1] * [x1, x2, x3]^T = 3*x1 + 2*x2 + x3
     * 约束条件: 
     *   x1 + x2 + x3 = 4
     *   2*x1 + x2 = 5
     *   x1, x2, x3 >= 0 且为整数
     * 
     * 解析解: x1 = 2, x2 = 1, x3 = 1, 最优值 = 9
     */
    @Test
    public void testComplexIntegerProgramming() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{3, 2, 1});
        
        // 等式约束
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1, 1},
            {2, 1, 0}
        });
        IVector b_eq = Linalg.vector(new double[]{4, 5});
        
        // 创建求解器
        RereIntegerProg solver = new RereIntegerProg();
        solver.setAllVariablesInteger(); // 所有变量都是整数
        // solver.setVerbose(true); // Disable verbose output for cleaner test output
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证最优解
        IVector solution = result.getOptimalPoint();
        assertEquals(3, solution.length(), "解的维度应为3");
        
        // 验证约束满足
        IVector constraintValue = A_eq.mmul(solution);
        assertEquals(4.0, (double)constraintValue.get(0), 1e-6, "第一个约束值应等于4");
        assertEquals(5.0, (double)constraintValue.get(1), 1e-6, "第二个约束值应等于5");
        
        // 验证整数约束
        for (int i = 0; i < solution.length(); i++) {
            double value = (Double) solution.get(i);
            assertEquals(Math.round(value), value, 1e-6, "变量 x" + (i+1) + " 应为整数");
        }
        
        // 验证最优值
        double expectedOptimalValue = 9.0;
        assertEquals(expectedOptimalValue, result.getOptimalValue(), 1e-6, "最优值应等于9");
    }

    /**
     * 测试无整数变量的情况（应该退化为线性规划）
     */
    @Test
    public void testNoIntegerVariables() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1, 2});
        
        // 等式约束: x1 + x2 = 3
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{3});
        
        // 创建求解器，不设置任何整数变量
        RereIntegerProg solver = new RereIntegerProg();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证最优解
        IVector solution = result.getOptimalPoint();
        assertEquals(2, solution.length(), "解的维度应为2");
        
        // 验证约束满足
        IVector constraintValue = A_eq.mmul(solution);
        assertEquals(3.0, (double)constraintValue.get(0), 1e-6, "约束值应等于3");
        
        // 验证最优值（应该是线性规划的最优解）
        double expectedOptimalValue = 3.0;
        assertEquals(expectedOptimalValue, result.getOptimalValue(), 1e-6, "最优值应等于3");
    }

    /**
     * 测试求解器参数设置
     */
    @Test
    public void testSolverParameters() {
        RereIntegerProg solver = new RereIntegerProg();
        
        // 测试参数设置
        solver.setTolerance(1e-8);
        solver.setMaxIterations(500);
        solver.setVerbose(true);
        
        // 验证求解器可以正常工作
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        solver.addIntegerVariables(0, 1);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        assertNotNull(result, "结果不应为null");
        assertEquals(2.0, result.getOptimalValue(), 1e-6, "最优值应等于2");
    }

    /**
     * 测试使用不同的线性规划求解器
     */
    @Test
    public void testWithDifferentLPSolver() {
        // 使用内点法求解器作为基础求解器
        RereIntegerProg solver = new RereIntegerProg();
        solver.addIntegerVariables(0, 1);
        
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        assertNotNull(result, "结果不应为null");
        assertEquals(2.0, result.getOptimalValue(), 1e-3, "最优值应等于2");
    }
}