package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;

public class SimpleMinimizationDebug {
    public static void main(String[] args) {
        System.out.println("=== 简单最小化问题调试 ===");
        
        try {
            // 创建求解器
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            
            // 问题: min 3*x1 + 2*x2
            // 约束: x1 + x2 >= 4  -> -x1 - x2 <= -4
            //      2*x1 + x2 >= 6 -> -2*x1 - x2 <= -6
            //      x1, x2 >= 0
            // 转换为等式约束形式: 添加松弛变量
            // -x1 - x2 + s1 = -4
            // -2*x1 - x2 + s2 = -6
            
            IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0}); // 最小化目标函数，包含松弛变量
            IMatrix A_eq = Linalg.matrix(new double[][]{
                {-1.0, -1.0, 1.0, 0.0},  // -x1 - x2 + s1 = -4
                {-2.0, -1.0, 0.0, 1.0}   // -2*x1 - x2 + s2 = -6
            });
            IVector b_eq = Linalg.vector(new double[]{-4.0, -6.0});
            
            System.out.println("目标函数系数: " + c);
            System.out.println("约束矩阵 A_eq:");
            System.out.println(A_eq);
            System.out.println("右侧向量 b_eq: " + b_eq);
            
            // 求解
            OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
            
            System.out.println("求解结果:");
            System.out.println("是否收敛: " + result.isConverged());
            
            if (result.isConverged()) {
                IVector solution = result.getOptimalPoint();
                double x1 = solution.get(0).doubleValue();
                double x2 = solution.get(1).doubleValue();
                double s1 = solution.get(2).doubleValue();
                double s2 = solution.get(3).doubleValue();
                
                System.out.println("解: x1=" + x1 + ", x2=" + x2 + ", s1=" + s1 + ", s2=" + s2);
                System.out.println("目标函数值: " + result.getOptimalValue());
                
                // 验证原始约束
                double constraint1 = x1 + x2;
                double constraint2 = 2*x1 + x2;
                
                System.out.println("约束验证:");
                System.out.println("x1 + x2 = " + constraint1 + " >= 4? " + (constraint1 >= 4.0));
                System.out.println("2*x1 + x2 = " + constraint2 + " >= 6? " + (constraint2 >= 6.0));
                
                // 验证等式约束
                double eq1 = -x1 - x2 + s1;
                double eq2 = -2*x1 - x2 + s2;
                System.out.println("等式约束验证:");
                System.out.println("-x1 - x2 + s1 = " + eq1 + " (应等于-4)");
                System.out.println("-2*x1 - x2 + s2 = " + eq2 + " (应等于-6)");
                
                // 理论最优解应该是 x1=2, x2=2, 目标值=10
                System.out.println("理论最优解: x1=2, x2=2, 目标值=10");
            } else {
                System.out.println("求解失败");
            }
            
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}