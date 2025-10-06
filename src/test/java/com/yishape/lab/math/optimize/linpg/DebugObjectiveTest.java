package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;

/**
 * 调试目标函数行处理的测试
 */
public class DebugObjectiveTest {
    
    @Test
    public void testObjectiveFunctionHandling() {
        System.out.println("=== 调试目标函数行处理 ===");
        
        // 创建最简单的问题：
        // 最小化: -x1 (等价于最大化 x1)
        // 约束: x1 + s1 = 1
        //       x1, s1 >= 0
        // 期望解: x1 = 1, s1 = 0
        // 期望目标函数值: -1 (最小化) 或 1 (最大化)
        
        IVector c = Linalg.vector(new double[]{-1, 0}); // 最小化 -x1
        
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1}  // x1 + s1 = 1
        });
        
        IVector b_eq = Linalg.vector(new double[]{1});
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        
        System.out.println("输入:");
        System.out.println("目标函数系数 c: " + java.util.Arrays.toString(c.toDoubleArray()));
        System.out.println("约束矩阵 A_eq: " + java.util.Arrays.deepToString(A_eq.toDoubleArray()));
        System.out.println("约束右端 b_eq: " + java.util.Arrays.toString(b_eq.toDoubleArray()));
        
        System.out.println("\n=== 开始求解 ===");
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        System.out.println("=== 求解完成 ===");
        
        System.out.println("\n=== 求解结果 ===");
        System.out.println("收敛状态: " + result.isConverged());
        System.out.println("目标函数值: " + result.getOptimalValue());
        
        if (result.getOptimalPoint() != null) {
            System.out.println("最优解: " + java.util.Arrays.toString(result.getOptimalPoint().toDoubleArray()));
            
            double[] solution = result.getOptimalPoint().toDoubleArray();
            if (solution.length >= 2) {
                System.out.println("x1 = " + solution[0]);
                System.out.println("s1 = " + solution[1]);
                
                // 验证约束
                double constraint1 = solution[0] + solution[1];
                System.out.println("约束验证: x1 + s1 = " + constraint1 + " (期望 1.0)");
                
                // 验证目标函数
                double objectiveValue = -solution[0]; // 最小化 -x1
                System.out.println("目标函数验证: -x1 = " + objectiveValue + " (期望 -1.0)");
                
                // 检查是否正确
                boolean correct = Math.abs(solution[0] - 1.0) < 1e-6 && 
                                Math.abs(solution[1] - 0.0) < 1e-6 &&
                                Math.abs(result.getOptimalValue() - (-1.0)) < 1e-6;
                
                System.out.println("\n=== 测试结果 ===");
                if (correct) {
                    System.out.println("✅ 测试通过!");
                } else {
                    System.out.println("❌ 测试失败!");
                    System.out.println("期望: x1=1.0, s1=0.0, 目标函数值=-1.0");
                    System.out.println("实际: x1=" + solution[0] + ", s1=" + solution[1] + ", 目标函数值=" + result.getOptimalValue());
                }
            }
        } else {
            System.out.println("❌ 没有找到解");
        }
    }
}