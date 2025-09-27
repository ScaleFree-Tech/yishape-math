package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.*;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;

/**
 * 调试LP问题的测试类
 */
public class DebugLPTest {
    
    @Test
    public void debugSimpleLP() {
        System.out.println("=== 调试简单LP问题 ===");
        
        // 问题：minimize 2*x1 + x2
        // subject to x1 + x2 = 2.5
        // x1 >= 0, x2 >= 0
        // 最优解应该是 x1=0, x2=2.5, 目标值=2.5
        
        IVector c = Linalg.vector(new double[]{2, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2.5});
        
        System.out.println("目标函数系数: " + c);
        System.out.println("约束矩阵: " + A_eq);
        System.out.println("约束右端: " + b_eq);
        
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        // 手动分析：
        // 对于 minimize 2*x1 + x2 subject to x1 + x2 = 2.5
        // 可行域是线段 x1 + x2 = 2.5, x1 >= 0, x2 >= 0
        // 即从 (0, 2.5) 到 (2.5, 0) 的线段
        // 目标函数在 (0, 2.5) 处取最小值 2.5
        // 目标函数在 (2.5, 0) 处取值 5.0
        
        System.out.println("\n手动分析:");
        System.out.println("在 (0, 2.5): 目标值 = 2*0 + 2.5 = 2.5");
        System.out.println("在 (2.5, 0): 目标值 = 2*2.5 + 0 = 5.0");
        System.out.println("最优解应该是 (0, 2.5)");
        
        try {
            Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            if (result != null) {
                System.out.println("\n求解器结果:");
                System.out.println("解: " + result.getSecond());
                System.out.println("目标值: " + result.getFirst());
                
                double x1 = (Double) result.getSecond().get(0);
                double x2 = (Double) result.getSecond().get(1);
                System.out.println("x1 = " + x1 + ", x2 = " + x2);
                System.out.println("验证约束: x1 + x2 = " + (x1 + x2));
                System.out.println("验证目标: 2*x1 + x2 = " + (2*x1 + x2));
                
                // 分析结果
                if (Math.abs(x1 - 0) < 1e-6 && Math.abs(x2 - 2.5) < 1e-6) {
                    System.out.println("✓ 找到了正确的最优解");
                } else if (Math.abs(x1 - 2.5) < 1e-6 && Math.abs(x2 - 0) < 1e-6) {
                    System.out.println("✗ 找到了错误的解 - 这是可行解但不是最优解");
                } else {
                    System.out.println("? 找到了其他解");
                }
            } else {
                System.out.println("求解器返回null");
            }
        } catch (Exception e) {
            System.out.println("求解失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}