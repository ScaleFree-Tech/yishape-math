package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 调试最小化问题的求解
 */
public class DebugMinimizationProblem {
    public static void main(String[] args) {
        System.out.println("=== 调试最小化问题求解 ===");
        
        // 测试标准最小化问题：min 3*x1 + 2*x2
        // 约束: x1 + x2 >= 4  -> x1 + x2 - s1 + a1 = 4
        //      2*x1 + x2 >= 6 -> 2*x1 + x2 - s2 + a2 = 6
        //      x1, x2 >= 0
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        
        // 使用大M法处理>=约束
        double M = 1000.0; // 大M值
        
        // 目标函数：min 3*x1 + 2*x2 + 0*s1 + 0*s2 + M*a1 + M*a2
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, M, M});
        
        // 约束矩阵：
        // x1 + x2 - s1 + a1 = 4
        // 2*x1 + x2 - s2 + a2 = 6
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 1.0, 0.0},  // x1 + x2 - s1 + a1 = 4
            {2.0, 1.0, 0.0, -1.0, 0.0, 1.0}   // 2*x1 + x2 - s2 + a2 = 6
        });
        
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});
        
        System.out.println("目标函数系数: " + c);
        System.out.println("约束矩阵:");
        for (int i = 0; i < A_eq.rows(); i++) {
            System.out.print("  [");
            for (int j = 0; j < A_eq.cols(); j++) {
                System.out.printf("%6.1f", A_eq.get(i, j));
            }
            System.out.println(" ]");
        }
        System.out.println("右端向量: " + b_eq);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        if (result != null) {
            System.out.println("\n=== 求解结果 ===");
            System.out.println("收敛状态: " + result.isConverged());
            System.out.println("目标函数值: " + String.format("%.2f", result.getOptimalValue()));
            
            IVector solution = result.getOptimalPoint();
            System.out.println("解向量: " + solution);
            
            if (solution.length() >= 6) {
                double x1 = solution.get(0);
                double x2 = solution.get(1);
                double s1 = solution.get(2);
                double s2 = solution.get(3);
                double a1 = solution.get(4);
                double a2 = solution.get(5);
                
                System.out.println(String.format("x1 = %.2f, x2 = %.2f", x1, x2));
                System.out.println(String.format("s1 = %.2f, s2 = %.2f", s1, s2));
                System.out.println(String.format("a1 = %.2f, a2 = %.2f", a1, a2));
                
                // 验证原始约束
                System.out.println("\n=== 约束验证 ===");
                double constraint1 = x1 + x2;
                double constraint2 = 2*x1 + x2;
                
                System.out.println(String.format("x1 + x2 = %.2f >= 4? %s", constraint1, constraint1 >= 4.0 - 1e-6));
                System.out.println(String.format("2*x1 + x2 = %.2f >= 6? %s", constraint2, constraint2 >= 6.0 - 1e-6));
                System.out.println(String.format("x1 >= 0? %s", x1 >= -1e-6));
                System.out.println(String.format("x2 >= 0? %s", x2 >= -1e-6));
                
                // 计算原始目标函数值
                double originalObjective = 3*x1 + 2*x2;
                System.out.println(String.format("原始目标函数值: 3*%.2f + 2*%.2f = %.2f", x1, x2, originalObjective));
                
                // 检查人工变量是否被消除
                if (Math.abs(a1) < 1e-6 && Math.abs(a2) < 1e-6) {
                    System.out.println("✓ 人工变量已被消除，找到可行解");
                } else {
                    System.out.println("✗ 人工变量未被消除，可能无可行解");
                }
                
                // 理论最优解分析
                System.out.println("\n=== 理论分析 ===");
                System.out.println("理论最优解: x1=2, x2=2 (约束交点)");
                System.out.println("理论最优值: 3*2 + 2*2 = 10");
            }
        } else {
            System.out.println("求解失败，返回null");
        }
    }
}