package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.ISimplexLinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

public class CorrectedTest {
    public static void main(String[] args) {
        System.out.println("=== 修正的测试 BetterSimplexLinProgSolver ===");
        
        // 创建一个正确的问题设置
        // 原始问题:
        // 目标函数: minimize 2*x1 + 3*x2
        // 约束条件:
        // x1 + x2 = 4
        // 2*x1 + x2 = 6
        // x1, x2 >= 0
        
        // 这个问题的解析解:
        // 从 x1 + x2 = 4 得到 x2 = 4 - x1
        // 代入 2*x1 + x2 = 6 得到 2*x1 + (4 - x1) = 6
        // 解得 x1 = 2, x2 = 2
        // 目标函数值: 2*2 + 3*2 = 10
        
        // 目标函数向量（最小化形式）
        IVector c = Linalg.vector(new double[]{2, 3});
        
        // 约束矩阵（全等式约束）
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1},  // x1 + x2 = 4
            {2, 1}   // 2*x1 + x2 = 6
        });
        
        IVector b_eq = Linalg.vector(new double[]{4, 6});
        
        // ILinProgSolver solver = new SimplexLinProgSolver();
        ISimplexLinProgSolver solver = new RereSimplexLinProgSolver();
//         ILinProgSolver solver = new ComMath4LinProgSolver();
        
        try {
            OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
            
            System.out.println("求解完成");
            System.out.println("收敛: " + result.isConverged());
            System.out.println("最优目标函数值: " + result.getOptimalValue());
            System.out.println("最优解: " + result.getOptimalPoint());
            
            // 验证结果
            boolean testPassed = true;
            
            if (!result.isConverged()) {
                System.out.println("❌ 测试失败: 求解未收敛");
                testPassed = false;
            } else {
                System.out.println("✅ 求解收敛");
            }
            
            // 验证目标函数值（最小化问题）
            double expectedObjective = 10.0;
            if (Math.abs(result.getOptimalValue() - expectedObjective) > 1e-6) {
                System.out.println("❌ 目标函数值错误: 期望 " + expectedObjective + ", 实际 " + result.getOptimalValue());
                testPassed = false;
            } else {
                System.out.println("✅ 目标函数值正确");
            }
            
            if (result.getOptimalPoint() != null) {
                IVector solution = result.getOptimalPoint();
                System.out.println("解的维度: " + solution.length());
                
                // 验证解向量
                if (solution.length() >= 2) {
                    double x1 = solution.get(0).doubleValue();
                    double x2 = solution.get(1).doubleValue();
                    
                    if (Math.abs(x1 - 2.0) > 1e-6) {
                        System.out.println("❌ x1错误: 期望 2.0, 实际 " + x1);
                        testPassed = false;
                    } else {
                        System.out.println("✅ x1正确: " + x1);
                    }
                    
                    if (Math.abs(x2 - 2.0) > 1e-6) {
                        System.out.println("❌ x2错误: 期望 2.0, 实际 " + x2);
                        testPassed = false;
                    } else {
                        System.out.println("✅ x2正确: " + x2);
                    }
                    
                    // 验证约束条件
                    double constraint1 = x1 + x2;
                    double constraint2 = 2*x1 + x2;
                    
                    if (Math.abs(constraint1 - 4.0) > 1e-6) {
                        System.out.println("❌ 约束1违反: x1 + x2 = " + constraint1 + ", 期望 4.0");
                        testPassed = false;
                    } else {
                        System.out.println("✅ 约束1满足: x1 + x2 = " + constraint1);
                    }
                    
                    if (Math.abs(constraint2 - 6.0) > 1e-6) {
                        System.out.println("❌ 约束2违反: 2*x1 + x2 = " + constraint2 + ", 期望 6.0");
                        testPassed = false;
                    } else {
                        System.out.println("✅ 约束2满足: 2*x1 + x2 = " + constraint2);
                    }
                }
                
                for (int i = 0; i < solution.length(); i++) {
                    System.out.println("x[" + i + "] = " + solution.get(i));
                }
            } else {
                System.out.println("❌ 未得到解");
                testPassed = false;
            }
            
            System.out.println("\n期望结果:");
            System.out.println("x1 = 2, x2 = 2, 目标函数值 = 10 (最小化形式)");
            
            System.out.println("\n=== 测试结果 ===");
            if (testPassed) {
                System.out.println("🎉 所有测试通过!");
            } else {
                System.out.println("💥 测试失败!");
            }
            
        } catch (Exception e) {
            System.err.println("求解失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}