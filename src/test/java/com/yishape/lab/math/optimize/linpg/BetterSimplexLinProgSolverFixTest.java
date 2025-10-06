package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试BetterSimplexLinProgSolver的修复
 * 专门测试等式约束下c和A_eq列数匹配的情况
 */
public class BetterSimplexLinProgSolverFixTest {
    
    private RereSimplexLinProgSolver solver;
    
    @BeforeEach
    public void setUp() {
        solver = new RereSimplexLinProgSolver();
    }
    
    /**
     * 测试简单的等式约束线性规划问题
     * 最大化: 3x1 + 2x2
     * 约束: x1 + x2 = 4
     *       2x1 + x2 = 6
     *       x1, x2 >= 0
     * 
     * 期望解: x1 = 2, x2 = 2, 目标函数值 = 10
     */
    @Test
    public void testSimpleEqualityConstraints() {
        System.out.println("=== 测试简单等式约束问题 ===");
        
        // 目标函数系数: 最大化 3x1 + 2x2
        IVector c = Linalg.vector(3.0, 2.0);
        
        // 等式约束矩阵 A_eq
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0},  // x1 + x2 = 4
            {2.0, 1.0}   // 2x1 + x2 = 6
        });
        
        // 等式约束右侧向量 b_eq
        IVector b_eq = Linalg.vector(4.0, 6.0);
        
        System.out.println("目标函数系数 c: " + c);
        System.out.println("约束矩阵 A_eq: " + A_eq);
        System.out.println("约束向量 b_eq: " + b_eq);
        System.out.println("c的长度: " + c.length() + ", A_eq的列数: " + A_eq.cols());
        
        // 验证c和A_eq的列数匹配
        assertEquals(c.length(), A_eq.cols(), "c的长度必须等于A_eq的列数");
        
        // 求解
        OptResult result = solver.maximize(c,null,null, A_eq, b_eq, null);
        
        // 验证结果
        assertNotNull(result, "求解结果不应为null");
        assertTrue(result.isConverged(), "求解应该成功");
        
        IVector solution = result.getOptimalPoint();
        assertNotNull(solution, "解向量不应为null");
        assertEquals(2, solution.length(), "解向量长度应为2");
        
        System.out.println("求解结果:");
        System.out.println("解向量: " + solution);
        System.out.println("目标函数值: " + result.getOptimalValue());
        System.out.println("迭代次数: " + result.getIterations());
        
        // 验证解的正确性（允许数值误差）
        double x1 = solution.get(0).doubleValue();
        double x2 = solution.get(1).doubleValue();
        
        System.out.println("x1 = " + x1 + ", x2 = " + x2);
        
        // 验证约束满足
        double constraint1 = x1 + x2;
        double constraint2 = 2*x1 + x2;
        
        System.out.println("约束1验证: x1 + x2 = " + constraint1 + " (期望: 4.0)");
        System.out.println("约束2验证: 2*x1 + x2 = " + constraint2 + " (期望: 6.0)");
        
        assertEquals(4.0, constraint1, 1e-6, "第一个约束应该满足");
        assertEquals(6.0, constraint2, 1e-6, "第二个约束应该满足");
        
        // 验证非负性
        assertTrue(x1 >= -1e-6, "x1应该非负");
        assertTrue(x2 >= -1e-6, "x2应该非负");
        
        // 验证目标函数值
        double expectedObjective = 3*x1 + 2*x2;
        System.out.println("计算的目标函数值: " + expectedObjective);
        assertEquals(expectedObjective, result.getOptimalValue(), 1e-6, "目标函数值应该正确");
    }
    
    /**
     * 测试维度不匹配的情况
     */
    @Test
    public void testDimensionMismatch() {
        System.out.println("=== 测试维度不匹配情况 ===");
        
        // 目标函数系数: 3个变量
        IVector c = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        
        // 约束矩阵: 只有2列
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0},
            {2.0, 1.0}
        });
        
        IVector b_eq = Linalg.vector(4.0, 6.0);
        
        System.out.println("c的长度: " + c.length() + ", A_eq的列数: " + A_eq.cols());
        
        // 应该抛出异常或返回失败结果
        assertThrows(IllegalArgumentException.class, () -> {
            solver.maximize(c, null,null,A_eq, b_eq, null);
        }, "维度不匹配应该抛出异常");
    }
    
    /**
     * 测试单个变量的简单情况
     */
    @Test
    public void testSingleVariable() {
        System.out.println("=== 测试单变量情况 ===");
        
        // 最大化: 5x
        IVector c = Linalg.vector(5.0);
        
        // 约束: x = 3
        IMatrix A_eq = Linalg.matrix(new double[][]{{1.0}});
        IVector b_eq = Linalg.vector(3.0);
        
        System.out.println("c的长度: " + c.length() + ", A_eq的列数: " + A_eq.cols());
        assertEquals(c.length(), A_eq.cols(), "c的长度必须等于A_eq的列数");
        
        OptResult result = solver.maximize(c,null,null, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        assertTrue(result.isConverged(), "求解应该成功");
        
        IVector solution = result.getOptimalPoint();
        assertEquals(1, solution.length(), "解向量长度应为1");
        
        double x = solution.get(0).doubleValue();
        System.out.println("解: x = " + x);
        System.out.println("目标函数值: " + result.getOptimalValue());
        
        assertEquals(3.0, x, 1e-6, "解应该是x=3");
        assertEquals(15.0, result.getOptimalValue(), 1e-6, "目标函数值应该是15");
    }
}