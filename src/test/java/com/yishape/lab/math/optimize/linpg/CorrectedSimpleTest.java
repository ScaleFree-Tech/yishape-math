package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

public class CorrectedSimpleTest {
    public static void main(String[] args) {
        System.out.println("=== 修正的测试 BetterSimplexLinProgSolver ===");
        
        // 创建一个正确的问题
        // 目标函数: maximize 2*x1 + 3*x2
        IVector c = Linalg.vector(new double[]{2, 3});
        
        // 约束条件 (不包含松弛变量):
        // x1 + x2 = 4
        // 2*x1 + x2 = 6
        // x1, x2 >= 0
        
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1},  // x1 + x2 = 4
            {2, 1}   // 2*x1 + x2 = 6
        });
        
        IVector b_eq = Linalg.vector(new double[]{4, 6});
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        
        try {
            OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
            
            System.out.println("求解完成");
            System.out.println("收敛: " + result.isConverged());
            System.out.println("最优目标函数值: " + result.getOptimalValue());
            System.out.println("最优解: " + result.getOptimalPoint());
            
            if (result.getOptimalPoint() != null) {
                IVector solution = result.getOptimalPoint();
                System.out.println("解的维度: " + solution.length());
                for (int i = 0; i < solution.length(); i++) {
                    System.out.println("x[" + i + "] = " + solution.get(i));
                }
            }
            
            System.out.println("\n期望结果:");
            System.out.println("x1 = 2, x2 = 2, 目标函数值 = 10");
            
        } catch (Exception e) {
            System.err.println("求解失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}