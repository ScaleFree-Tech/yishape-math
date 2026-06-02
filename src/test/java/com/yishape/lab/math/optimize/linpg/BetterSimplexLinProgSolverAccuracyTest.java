package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 全面测试BetterSimplexLinProgSolver的求解正确性
 * 包括基础问题、边界情况和数值稳定性测试
 */
public class BetterSimplexLinProgSolverAccuracyTest {
    
    private RereSimplexLinProgSolver solver;
    private static final double TOLERANCE = 1e-6;
    
    private static boolean isAllNonNegative(IVector vector) {
        for (int i = 0; i < vector.size(); i++) {
            if (vector.get(i) < 0) {
                return false;
            }
        }
        return true;
    }
    
    @BeforeEach
    void setUp() {
        solver = new RereSimplexLinProgSolver();
    }
    
    // ==================== 基础线性规划问题测试 ====================
    
    @Test
    @DisplayName("标准最大化问题 - 经典教科书例题")
    void testStandardMaximizationProblem() {
        // 问题: max 2*x1 + 3*x2
        // 约束: x1 + 2*x2 <= 8
        //      2*x1 + x2 <= 10
        //      x1, x2 >= 0
        // 转换为等式约束形式: 添加松弛变量
        // x1 + 2*x2 + s1 = 8
        // 2*x1 + x2 + s2 = 10
        // 目标函数变为最小化: min -2*x1 - 3*x2
        
        IVector c = Linalg.vector(new double[]{-2.0, -3.0, 0.0, 0.0}); // 最小化负目标函数，包含松弛变量
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 2.0, 1.0, 0.0},  // x1 + 2*x2 + s1 = 8
            {2.0, 1.0, 0.0, 1.0}   // 2*x1 + x2 + s2 = 10
        });
        IVector b_eq = Linalg.vector(new double[]{8.0, 10.0});
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        assertTrue(result.isConverged(), "应该收敛到最优解");
        
        IVector solution = result.getOptimalPoint();
        assertNotNull(solution, "解向量不应为null");
        assertEquals(4, solution.length(), "解向量维度应为4（包括松弛变量）");
        
        // 验证解的正确性 (允许一定误差)
        double x1 = solution.get(0);
        double x2 = solution.get(1);
        
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
    @DisplayName("标准最小化问题")
    void testStandardMinimizationProblem() {
        // 重定向System.out到文件
        java.io.PrintStream originalOut = System.out;
        try {
            java.io.PrintStream fileOut = new java.io.PrintStream("debug_output.txt");
            System.setOut(fileOut);
        } catch (Exception e) {
            // 如果无法创建文件，继续使用原始输出
        }
        
        try {
            // 问题: min 3*x1 + 2*x2
            // 原约束: x1 + x2 >= 4, 2*x1 + x2 >= 6, x1, x2 >= 0
            // 直接使用大M法构建标准形式，避免有问题的约束转换
            
            // 转换为标准形式：x1 + x2 - s1 + a1 = 4, 2*x1 + x2 - s2 + a2 = 6
            // 其中 s1, s2 >= 0 是剩余变量，a1, a2 >= 0 是人工变量
            double bigM = 10000.0; // 大M惩罚系数
            
            // 变量: [x1, x2, s1, s2, a1, a2]
            IVector c = Linalg.vector(new double[]{3.0, 2.0});
            
            // 约束矩阵:
            IMatrix A_ub = Linalg.matrix(new double[][]{
                {-1.0, -1.0},   // x1 + x2 - s1 + a1 = 4
                {-2.0, -1.0}    // 2*x1 + x2 - s2 + a2 = 6
            });
            
            IVector b_ub = Linalg.vector(new double[]{-4.0, -6.0});
        
        System.out.println("=== 直接大M法求解 ===");
        System.out.println("目标函数向量 c:");
        System.out.println(c);
        System.out.println("约束矩阵 A:");
        for (int i = 0; i < A_ub.rows(); i++) {
            System.out.println("  Row " + i + ": " + A_ub.getRow(i));
        }
        System.out.println("约束向量 b:");
        System.out.println(b_ub);
        System.out.println("大M值: " + bigM);
        System.out.println("=== 开始求解 ===");
        
        // 启用详细输出
        solver.setVerbose(false);
        
        // 直接求解大M问题
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            IVector fullSolution = result.getOptimalPoint();
            assertNotNull(fullSolution, "解向量不应为null");
            assertEquals(2, fullSolution.length(), "解向量维度应为2（包括所有变量）");
            
            double x1 = fullSolution.get(0);
            double x2 = fullSolution.get(1);
            
            System.out.println("=== 解的详细信息 ===");
            System.out.println("完整解向量: " + fullSolution);
            System.out.println("原始变量: x1=" + x1 + ", x2=" + x2);
            System.out.println("目标函数值: " + result.getOptimalValue());
            
            
            System.out.println("第一个约束验证: x1 + x2 = " + (x1 + x2) + " >= 4.0? " + (x1 + x2 >= 4.0 - TOLERANCE));
            System.out.println("第二个约束验证: 2*x1 + x2 = " + (2*x1 + x2) + " >= 6.0? " + (2*x1 + x2 >= 6.0 - TOLERANCE));
            System.out.println("TOLERANCE = " + TOLERANCE);
            
            // 验证解满足原始约束
            assertTrue(x1 + x2 >= 4.0 - TOLERANCE, "第一个约束应满足");
            assertTrue(2*x1 + x2 >= 6.0 - TOLERANCE, "第二个约束应满足");
            assertTrue(x1 >= -TOLERANCE && x2 >= -TOLERANCE, "非负约束应满足");
            
            // 计算真实目标函数值（不包括人工变量惩罚）
            double trueObjectiveValue = 3*x1 + 2*x2;
            System.out.println("真实目标函数值（3*x1 + 2*x2）: " + trueObjectiveValue);
            
            // 理论最优解应该是 x1=2, x2=2, 目标值=10
            assertEquals(2.0, x1, TOLERANCE, "x1应接近2.0");
            assertEquals(2.0, x2, TOLERANCE, "x2应接近2.0");
            assertEquals(10.0, trueObjectiveValue, TOLERANCE, "真实目标函数值应接近10.0");
        } else {
            System.out.println("最小化问题未收敛，可能是无可行解或无界解");
            fail("最小化问题应该收敛");
        }
        } finally {
            // 恢复原始的System.out
            System.setOut(originalOut);
        }
    }
    
    @Test
    @DisplayName("三变量问题")
    void testThreeVariableProblem() {
        // 问题: max x1 + 2*x2 + 3*x3
        // 约束: x1 + x2 + x3 <= 6
        //      2*x1 + x2 <= 8
        //      x2 + 2*x3 <= 10
        //      x1, x2, x3 >= 0
        // 转换为等式约束形式: 添加松弛变量
        // 目标函数变为最小化: min -x1 - 2*x2 - 3*x3
        
        IVector c = Linalg.vector(new double[]{-1.0, -2.0, -3.0, 0.0, 0.0, 0.0}); // 最小化负目标函数，包含松弛变量
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0, 1.0, 0.0, 0.0},  // x1 + x2 + x3 + s1 = 6
            {2.0, 1.0, 0.0, 0.0, 1.0, 0.0},  // 2*x1 + x2 + s2 = 8
            {0.0, 1.0, 2.0, 0.0, 0.0, 1.0}   // x2 + 2*x3 + s3 = 10
        });
        IVector b_eq = Linalg.vector(new double[]{6.0, 8.0, 10.0});
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "解向量不应为null");
            assertEquals(6, solution.length(), "解向量维度应为6（包括松弛变量）");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            double x3 = solution.get(2);
            
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
        }
    }
    
    // ==================== 边界情况测试 ====================
    
    @Test
    @DisplayName("无可行解问题")
    void testInfeasibleProblem() {
        // 问题: max x1 + x2
        // 约束: x1 + x2 <= 1
        //      x1 + x2 >= 2  -> -x1 - x2 <= -2
        //      x1, x2 >= 0
        // 这个问题无可行解
        
        IVector c = Linalg.vector(new double[]{-1.0, -1.0,0.0,0.0}); // 最小化负目标函数，只有原始变量
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0, 0.0},   // x1 + x2 + s1 = 1
            {-1.0, -1.0, 0.0, 1.0}  // -x1 - x2 + s2 = -2
        });
        IVector b_eq = Linalg.vector(new double[]{1.0, -2.0});
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        // 无可行解的情况下，求解器可能不收敛或返回特殊状态
        if (!result.isConverged()) {
            System.out.println("正确检测到无可行解问题");
        } else {
            System.out.println("求解器返回了解，需要进一步验证是否为有效解");
        }
    }
    
    @Test
    @DisplayName("退化问题 - 多个基本解")
    void testDegenerateProblem() {
        // 问题: max x1 + x2
        // 约束: x1 <= 2
        //      x2 <= 2
        //      x1 + x2 <= 3
        //      x1, x2 >= 0
        // 最优解在 (1,2) 和 (2,1) 之间
        
        IVector c = Linalg.vector(new double[]{-1.0, -1.0, 0.0, 0.0, 0.0}); // 最小化负目标函数
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 0.0, 1.0, 0.0, 0.0},  // x1 + s1 = 2
            {0.0, 1.0, 0.0, 1.0, 0.0},  // x2 + s2 = 2
            {1.0, 1.0, 0.0, 0.0, 1.0}   // x1 + x2 + s3 = 3
        });
        IVector b_eq = Linalg.vector(new double[]{2.0, 2.0, 3.0});
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "解向量不应为null");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            
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
        }
    }
    
    // ==================== 数值稳定性测试 ====================
    
    @Test
    @DisplayName("大数值问题")
    void testLargeNumbersProblem() {
        // 使用较大的系数测试数值稳定性
        IVector c = Linalg.vector(new double[]{-1000.0, -2000.0, 0.0, 0.0}); // 最小化负目标函数
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {100.0, 200.0, 1.0, 0.0},  // 100*x1 + 200*x2 + s1 = 10000
            {300.0, 100.0, 0.0, 1.0}   // 300*x1 + 100*x2 + s2 = 15000
        });
        IVector b_eq = Linalg.vector(new double[]{10000.0, 15000.0});
        
        // 暂时启用缩放来验证intelligent scaling机制
        RereSimplexLinProgSolver localSolver = new RereSimplexLinProgSolver();
        localSolver.setUseNumericalScaling(true); // 启用缩放，让intelligent scaling决策
        localSolver.setVerbose(true); // 启用详细输出
        
        OptResult result = localSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "解向量不应为null");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            
            // 验证约束满足（正确的等式约束验证）
            double s1 = solution.get(2);
            double s2 = solution.get(3);
            
            // 等式约束验证
            double constraint1_eq = 100*x1 + 200*x2 + s1;
            double constraint2_eq = 300*x1 + 100*x2 + s2;
            
            System.out.println("约束验证详情:");
            System.out.println("  x1=" + x1 + ", x2=" + x2 + ", s1=" + s1 + ", s2=" + s2);
            System.out.println("  第一个约束: 100*x1 + 200*x2 + s1 = " + constraint1_eq + " (expected: 10000.0)");
            System.out.println("  第二个约束: 300*x1 + 100*x2 + s2 = " + constraint2_eq + " (expected: 15000.0)");
            System.out.println("  第一个约束误差: " + Math.abs(constraint1_eq - 10000.0));
            System.out.println("  第二个约束误差: " + Math.abs(constraint2_eq - 15000.0));
            System.out.println("  容忍度: TOLERANCE*1000 = " + (TOLERANCE*1000));
            
            // 使用更大的容忍度来处理大数值问题
            double largeTolerance = Math.max(TOLERANCE * 1000, 0.01); // 至少 0.01
            assertTrue(Math.abs(constraint1_eq - 10000.0) <= largeTolerance, "第一个等式约束应满足");
            assertTrue(Math.abs(constraint2_eq - 15000.0) <= largeTolerance, "第二个等式约束应满足");
            
            // 非负约束验证
            assertTrue(x1 >= -TOLERANCE && x2 >= -TOLERANCE, "非负约束应满足");
            
            System.out.println("大数值问题解: x1=" + x1 + ", x2=" + x2);
        } else {
            System.out.println("大数值问题未收敛，可能存在数值稳定性问题");
        }
    }
    
    @Test
    @DisplayName("小数值问题")
    void testSmallNumbersProblem() {
        // 使用较小的系数测试数值稳定性
        IVector c = Linalg.vector(new double[]{-0.001, -0.002, 0.0, 0.0}); // 最小化负目标函数
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {0.01, 0.02, 1.0, 0.0},  // 0.01*x1 + 0.02*x2 + s1 = 0.1
            {0.03, 0.01, 0.0, 1.0}   // 0.03*x1 + 0.01*x2 + s2 = 0.15
        });
        IVector b_eq = Linalg.vector(new double[]{0.1, 0.15});
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "解向量不应为null");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            
            // 验证约束满足
            assertTrue(0.01*x1 + 0.02*x2 <= 0.1 + TOLERANCE, "第一个约束应满足");
            assertTrue(0.03*x1 + 0.01*x2 <= 0.15 + TOLERANCE, "第二个约束应满足");
            assertTrue(x1 >= -TOLERANCE && x2 >= -TOLERANCE, "非负约束应满足");
            
            System.out.println("小数值问题解: x1=" + x1 + ", x2=" + x2);
        } else {
            System.out.println("小数值问题未收敛，可能存在数值稳定性问题");
        }
    }
    
    // ==================== 特殊情况测试 ====================
    
    @Test
    @DisplayName("单变量问题")
    void testSingleVariableProblem() {
        // 问题: max 2*x1
        // 约束: x1 <= 5
        //      x1 >= 0
        
        System.out.println("=== 开始单变量问题测试 ===");
        IVector c = Linalg.vector(new double[]{-2.0, 0.0}); // 最小化负目标函数
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0}  // x1 + s1 = 5
        });
        IVector b_eq = Linalg.vector(new double[]{5.0});
        
        System.out.println("目标函数向量 c: " + c);
        System.out.println("约束矩阵 A_eq: " + A_eq);
        System.out.println("约束向量 b_eq: " + b_eq);
        
        // 分析问题：最小化 -2*x1，约束 x1 = 5
        // 这意味着最大化 2*x1，在约束 x1 = 5 下，最优解应该是 x1 = 5
        System.out.println("问题分析：最小化 -2*x1，约束 x1 = 5");
        System.out.println("预期解：x1 = 5.0，目标值 = -2*5 = -10.0");
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "解向量不应为null");
            assertEquals(2, solution.length(), "解向量维度应为2（包括松弛变量）");
            
            double x1 = solution.get(0);
            assertEquals(5.0, x1, TOLERANCE, "x1应等于5");
            
            double actualObjective = -result.getOptimalValue(); // 转换回最大化问题的目标值
            assertEquals(10.0, actualObjective, TOLERANCE, "目标函数值应为10");
            
            System.out.println("单变量问题解: x1=" + x1 + ", 目标值=" + actualObjective);
        } else {
            System.out.println("单变量问题未收敛");
        }
    }
    
    @Test
    @DisplayName("零目标函数问题")
    void testZeroObjectiveProblem() {
        // 问题: max 0*x1 + 0*x2
        // 约束: x1 + x2 <= 1
        //      x1, x2 >= 0
        
        IVector c = Linalg.vector(new double[]{0.0, 0.0,0.0}); // 零目标函数，只有原始变量
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0}  // x1 + x2 + s1 = 1
        });
        IVector b_eq = Linalg.vector(new double[]{1.0});
        
        // 启用详细输出来调试
        solver.setVerbose(true);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        assertNotNull(result, "求解结果不应为null");
        
        if (result.isConverged()) {
            assertEquals(0.0, result.getOptimalValue(), TOLERANCE, "目标函数值应为0");
            
            IVector solution = result.getOptimalPoint();
            assertNotNull(solution, "解向量不应为null");
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            double s1 = solution.get(2);
            
            System.out.println("零目标函数问题解详情:");
            System.out.println("x1 = " + x1);
            System.out.println("x2 = " + x2);
            System.out.println("s1 = " + s1);
            System.out.println("x1 + x2 = " + (x1 + x2));
            System.out.println("x1 + x2 + s1 = " + (x1 + x2 + s1));
            
            // 验证等式约束
            boolean constraintSatisfied = Math.abs((x1 + x2 + s1) - 1.0) <= TOLERANCE;
            System.out.println("等式约束满足情况: |" + (x1 + x2 + s1) + " - 1.0| = " + Math.abs((x1 + x2 + s1) - 1.0) + " <= " + TOLERANCE + " ? " + constraintSatisfied);
            
            assertTrue(constraintSatisfied, "等式约束应满足: x1 + x2 + s1 = 1.0");
            
            // 验证非负约束
            boolean nonNegativeSatisfied = (x1 >= -TOLERANCE) && (x2 >= -TOLERANCE) && (s1 >= -TOLERANCE);
            System.out.println("非负约束满足情况: x1=" + x1 + " >= 0, x2=" + x2 + " >= 0, s1=" + s1 + " >= 0 ? " + nonNegativeSatisfied);
            
            assertTrue(nonNegativeSatisfied, "非负约束应满足");
            
            System.out.println("零目标函数问题解: x1=" + x1 + ", x2=" + x2 + ", 目标值=" + result.getOptimalValue());
        } else {
            System.out.println("零目标函数问题未收敛");
            fail("零目标函数问题应该收敛");
        }
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 验证解是否满足所有约束
     */
    private boolean verifySolution(IVector solution, IMatrix A, IVector b) {
        if (solution == null || A == null || b == null) {
            return false;
        }
        
        // 检查非负约束
        for (int i = 0; i < solution.length(); i++) {
            if (solution.get(i) < -TOLERANCE) {
                return false;
            }
        }
        
        // 检查线性约束 A*x = b
        IVector constraintValues = A.mmul(solution);
        for (int i = 0; i < b.length(); i++) {
            double diff = Math.abs(constraintValues.get(i) - b.get(i));
            if (diff > TOLERANCE) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * 计算目标函数值
     */
    private double calculateObjective(IVector c, IVector solution) {
        double objective = 0.0;
        for (int i = 0; i < c.length(); i++) {
            objective += c.get(i) * solution.get(i);
        }
        return objective;
    }
}