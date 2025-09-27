package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;

public class SimpleBigMTest {
    
    @Test
    public void testSimpleBigM() {
        // 测试问题: min 2*x1 + x2
        // 约束: x1 + x2 = 2.5
        //       x1, x2 >= 0
        // 最优解应该是: x1=0, x2=2.5, 目标值=2.5
        
        IVector c = IVector.of(new double[]{2.0, 1.0});
        IMatrix A_eq = IMatrix.of(new double[][]{{1.0, 1.0}});
        IVector b_eq = IVector.of(new double[]{2.5});
        
        System.out.println("原问题:");
        System.out.println("目标函数: min 2*x1 + x2");
        System.out.println("约束: x1 + x2 = 2.5");
        System.out.println("期望最优解: x1=0, x2=2.5, 目标值=2.5");
        System.out.println();
        
        // 手动构建初始表格来验证大M法
        // 表格结构: [A_eq | I | b_eq]
        //          [c^T  | M | 0  ]
        double bigM = 1e6;
        
        // 约束行: [1, 1, 1, 2.5] (x1 + x2 + s1 = 2.5)
        // 目标行: [-2, -1, M, 0] (最小化 2*x1 + x2 + M*s1)
        System.out.println("初始表格构建:");
        System.out.println("约束行: [1, 1, 1, 2.5]");
        System.out.println("目标行: [-2, -1, " + bigM + ", 0]");
        
        // 消除人工变量在目标函数中的系数
        // 目标行 = 目标行 - M * 约束行
        // 新目标行: [-2, -1, M, 0] - M * [1, 1, 1, 2.5]
        //         = [-2-M, -1-M, M-M, 0-2.5M]
        //         = [-2-M, -1-M, 0, -2.5M]
        System.out.println("消除人工变量系数后:");
        System.out.println("目标行: [" + (-2-bigM) + ", " + (-1-bigM) + ", 0, " + (-2.5*bigM) + "]");
        
        // 现在选择入基变量：应该选择系数最负的
        // x1的系数: -2-M ≈ -1000002
        // x2的系数: -1-M ≈ -1000001
        // 所以应该选择x1入基，但这是错误的！
        System.out.println();
        System.out.println("入基变量选择:");
        System.out.println("x1系数: " + (-2-bigM) + " (更负)");
        System.out.println("x2系数: " + (-1-bigM));
        System.out.println("按最负系数规则，会选择x1，但这导致错误解！");
        
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        System.out.println();
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
    }
}