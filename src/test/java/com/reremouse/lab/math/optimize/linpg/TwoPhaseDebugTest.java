package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;

public class TwoPhaseDebugTest {
    
    @Test
    public void testTwoPhaseMethod() {
        // 测试问题: min 2*x1 + x2
        // 约束: x1 + x2 = 2.5
        //       x1, x2 >= 0
        // 最优解应该是: x1=0, x2=2.5, 目标值=2.5
        
        IVector c = IVector.of(new double[]{2.0, 1.0});
        IMatrix A_eq = IMatrix.of(new double[][]{{1.0, 1.0}});
        IVector b_eq = IVector.of(new double[]{2.5});
        
        System.out.println("=== 两阶段法调试 ===");
        System.out.println("原问题: min 2*x1 + x2");
        System.out.println("约束: x1 + x2 = 2.5");
        System.out.println("期望最优解: x1=0, x2=2.5, 目标值=2.5");
        System.out.println();
        
        // 手动分析两阶段法
        System.out.println("第一阶段分析:");
        System.out.println("添加人工变量s1，问题变为:");
        System.out.println("min s1");
        System.out.println("约束: x1 + x2 + s1 = 2.5");
        System.out.println("初始基解: x1=0, x2=0, s1=2.5 (目标值=2.5)");
        System.out.println();
        
        System.out.println("第一阶段初始表格:");
        System.out.println("约束行: [1, 1, 1, 2.5]");
        System.out.println("目标行: [0, 0, 1, 0]");
        System.out.println("消除人工变量系数后:");
        System.out.println("目标行: [0, 0, 1, 0] - [1, 1, 1, 2.5] = [-1, -1, 0, -2.5]");
        System.out.println();
        
        System.out.println("第一阶段迭代:");
        System.out.println("入基变量选择: x1和x2的系数都是-1，选择哪个都可以");
        System.out.println("如果选择x1入基: 比率测试 2.5/1 = 2.5，s1出基");
        System.out.println("如果选择x2入基: 比率测试 2.5/1 = 2.5，s1出基");
        System.out.println("关键问题: 应该选择x2入基以获得更好的第二阶段起点！");
        System.out.println();
        
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        System.out.println("求解器结果:");
        System.out.println("最优解: " + result.getSecond());
        System.out.println("目标值: " + result.getFirst());
        
        // 验证解的正确性
        double x1 = (Double) result.getSecond().get(0);
        double x2 = (Double) result.getSecond().get(1);
        System.out.println();
        System.out.println("解的验证:");
        System.out.println("约束检查: x1 + x2 = " + (x1 + x2) + " (应该等于2.5)");
        System.out.println("目标值检查: 2*x1 + x2 = " + (2*x1 + x2) + " (应该等于2.5)");
        
        if (Math.abs(x1 - 0.0) < 1e-6 && Math.abs(x2 - 2.5) < 1e-6) {
            System.out.println("✓ 找到了正确的最优解！");
        } else {
            System.out.println("✗ 找到了错误的解");
            System.out.println("问题分析: 第一阶段选择了错误的入基变量");
        }
    }
}