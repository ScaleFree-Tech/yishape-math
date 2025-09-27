package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.*;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 调试混合整数规划问题的测试类
 */
public class DebugMixedIntegerTest {
    
    @Test
    public void debugMixedIntegerProgramming() {
        System.out.println("=== 调试混合整数规划问题 ===");
        
        // 测试问题：
        // minimize 2*x1 + x2
        // subject to x1 + x2 = 2.5
        // x1 >= 0 且为整数, x2 >= 0 (连续变量)
        // 期望解: x1 = 0, x2 = 2.5, 最优值 = 2.5
        
        IVector c = Linalg.vector(new double[]{2, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2.5});
        
        System.out.println("目标函数系数: " + c);
        System.out.println("约束矩阵: " + A_eq);
        System.out.println("约束右端: " + b_eq);
        
        // 首先测试线性规划松弛解
        System.out.println("\n=== 线性规划松弛解 ===");
        SimplexLinProgSolver lpSolver = new SimplexLinProgSolver();
        try {
            Tuple2<Double, IVector> lpResult = lpSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            if (lpResult != null) {
                System.out.println("LP松弛解: " + lpResult.getSecond());
                System.out.println("LP松弛最优值: " + lpResult.getFirst());
                
                // 验证解
                double x1 = (Double) lpResult.getSecond().get(0);
                double x2 = (Double) lpResult.getSecond().get(1);
                System.out.println("x1 = " + x1 + ", x2 = " + x2);
                System.out.println("约束验证: x1 + x2 = " + (x1 + x2) + " (应等于2.5)");
                System.out.println("目标函数值: 2*x1 + x2 = " + (2*x1 + x2));
                
                // 验证LP松弛解
                System.out.println("约束验证结果: " + (Math.abs(x1 + x2 - 2.5) < 1e-6 ? "通过" : "失败"));
                System.out.println("期望最优值: 1.0, 实际最优值: " + lpResult.getFirst());
            } else {
                fail("LP松弛问题应该有解");
            }
        } catch (Exception e) {
            fail("LP松弛求解失败: " + e.getMessage());
        }
        
        // 然后测试整数规划
        System.out.println("\n=== 整数规划求解 ===");
        RereIntegerProg solver = new RereIntegerProg();
        solver.setIntegerVariable(0); // 只有x1是整数变量
        solver.setVerbose(true);
        solver.setTolerance(1e-9);
        solver.setMaxIterations(100);
        solver.setMaxDepth(20);
        solver.setGapTolerance(1e-9);
        
        try {
            Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            assertNotNull(result, "整数规划应该有解");
            
            System.out.println("\n整数规划解: " + result.getSecond());
            System.out.println("整数规划最优值: " + result.getFirst());
            
            // 验证解
            double x1 = (Double) result.getSecond().get(0);
            double x2 = (Double) result.getSecond().get(1);
            System.out.println("x1 = " + x1 + " (应为整数), x2 = " + x2);
            System.out.println("约束验证: x1 + x2 = " + (x1 + x2) + " (应等于2.5)");
            System.out.println("目标函数值: 2*x1 + x2 = " + (2*x1 + x2));
            System.out.println("x1是否为整数: " + (Math.abs(x1 - Math.round(x1)) < 1e-6));
            
            // 验证约束满足
            System.out.println("约束验证结果: " + (Math.abs(x1 + x2 - 2.5) < 1e-6 ? "通过" : "失败"));
            
            // 验证整数约束
            System.out.println("整数约束验证: " + (Math.abs(x1 - Math.round(x1)) < 1e-6 ? "通过" : "失败"));
            
            // 期望结果分析
            System.out.println("\n期望结果分析:");
            System.out.println("如果x1=0, x2=2.5: 目标值 = 2*0 + 2.5 = 2.5");
            System.out.println("如果x1=1, x2=1.5: 目标值 = 2*1 + 1.5 = 3.5");
            System.out.println("如果x1=2, x2=0.5: 目标值 = 2*2 + 0.5 = 4.5");
            System.out.println("最优解应该是 x1=0, x2=2.5, 最优值=2.5");
            
            // 验证最优值 - 这里是问题所在
            double expectedOptimalValue = 2.5;
            System.out.println("期望最优值: " + expectedOptimalValue);
            System.out.println("实际最优值: " + result.getFirst());
            
            // 暂时不断言，先看看实际结果
            // assertEquals(expectedOptimalValue, result.getFirst(), 1e-6, "最优值应等于2.5");
            
        } catch (Exception e) {
            fail("整数规划求解失败: " + e.getMessage());
        }
    }
}