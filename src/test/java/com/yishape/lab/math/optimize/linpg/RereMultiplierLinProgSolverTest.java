package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * RereMultiplierLinProgSolver测试类
 */
public class RereMultiplierLinProgSolverTest {

    /**
     * 测试简单的线性规划问题
     * 目标函数: minimize c^T * x = [1, 1] * [x1, x2]^T = x1 + x2
     * 约束条件: 
     *   x1 + x2 = 2
     *   x1, x2 >= 0
     * 
     * 解析解: x1 = 1, x2 = 1, 最优值 = 2
     */
    @Test
    public void testSimpleLinearProgramming() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{1, 1});
        
        // 等式约束: x1 + x2 = 2
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        // 创建求解器
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        Assertions.assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证最优解
        IVector solution = result.getOptimalPoint();
        assertEquals(2, solution.length(), "解的维度应为2");
        
        // 验证约束满足
        IVector constraintValue = A_eq.mmul(solution);
        assertEquals(2.0, (double)constraintValue.get(0), 1e-6, "约束值应等于2");
        
        // 验证最优值（理论最优值为2）
        double expectedOptimalValue = 2.0;
        assertEquals(expectedOptimalValue, result.getOptimalValue(), 1e-3, "最优值应接近2");
    }
    
    /**
     * 测试更复杂的线性规划问题
     * 目标函数: minimize c^T * x = [2, 3] * [x1, x2]^T = 2*x1 + 3*x2
     * 约束条件: 
     *   x1 + 2*x2 = 4
     *   2*x1 + x2 = 5
     * 
     * 解析解: x1 = 2, x2 = 1, 最优值 = 7
     */
    @Test
    public void testComplexLinearProgramming() {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{2, 3});
        
        // 等式约束
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 2},
            {2, 1}
        });
        IVector b_eq = Linalg.vector(new double[]{4, 5});
        
        // 创建求解器
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        Assertions.assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证最优解
        IVector solution = result.getOptimalPoint();
        assertEquals(2, solution.length(), "解的维度应为2");
        
        // 验证约束满足
        IVector constraintValue = A_eq.mmul(solution);
        assertEquals(4.0, (double)constraintValue.get(0), 1e-6, "第一个约束值应等于4");
        assertEquals(5.0, (double)constraintValue.get(1), 1e-6, "第二个约束值应等于5");
        
        // 验证最优值（理论最优值为7）
        double expectedOptimalValue = 7.0;
        assertEquals(expectedOptimalValue, result.getOptimalValue(), 1e-3, "最优值应接近7");
        
        // 验证最优解（理论解为x1=2, x2=1）
        assertEquals(2.0, (double)solution.get(0), 1e-3, "x1应接近2");
        assertEquals(1.0, (double)solution.get(1), 1e-3, "x2应接近1");
    }
    
    /**
     * 测试带非负约束的线性规划问题
     * 目标函数: minimize c^T * x = [-1, -1] * [x1, x2]^T = -x1 - x2
     * 约束条件: 
     *   x1 + x2 = 2
     *   x1, x2 >= 0
     * 
     * 解析解: x1 = 2, x2 = 0 或 x1 = 0, x2 = 2, 最优值 = -2
     */
    @Test
    public void testNonNegativeConstraints() {
        // 目标函数系数（最大化 x1 + x2 等价于最小化 -x1 - x2）
        IVector c = Linalg.vector(new double[]{-1, -1});
        
        // 等式约束: x1 + x2 = 2
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        // 创建求解器
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        
        // 求解带非负约束的问题
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        // 验证结果
        assertNotNull(result, "结果不应为null");
        assertNotNull(result.getOptimalValue(), "最优值不应为null");
        Assertions.assertNotNull(result.getOptimalPoint(), "最优解不应为null");
        
        // 验证最优解
        IVector solution = result.getOptimalPoint();
        assertEquals(2, solution.length(), "解的维度应为2");
        
        // 验证约束满足
        IVector constraintValue = A_eq.mmul(solution);
        assertEquals(2.0, (double)constraintValue.get(0), 1e-6, "约束值应等于2");
        
        // 验证非负约束满足
        assertTrue((double)solution.get(0) >= 0, "x1应非负");
        assertTrue((double)solution.get(1) >= 0, "x2应非负");
        
        // 验证最优值（理论最优值为-2）
        double expectedOptimalValue = -2.0;
        assertEquals(expectedOptimalValue, result.getOptimalValue(), 1e-3, "最优值应接近-2");
    }
}