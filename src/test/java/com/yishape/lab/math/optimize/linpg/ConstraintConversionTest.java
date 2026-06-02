package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;

public class ConstraintConversionTest {
    
    @Test
    public void testConstraintConversionForStandardMinimization() {
        // 问题: min 3*x1 + 2*x2
        // 约束: x1 + x2 >= 4, 2*x1 + x2 >= 6, x1, x2 >= 0
        // 转换为<=约束: -x1 - x2 <= -4, -2*x1 - x2 <= -6
        
        IVector c = Linalg.vector(new double[]{3.0, 2.0}); // 最小化目标函数
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {-1.0, -1.0},  // -x1 - x2 <= -4
            {-2.0, -1.0}   // -2*x1 - x2 <= -6
        });
        IVector b_ub = Linalg.vector(new double[]{-4.0, -6.0});
        
        System.out.println("原始问题:");
        System.out.println("目标函数: min " + c);
        System.out.println("约束矩阵 A_ub: " + A_ub);
        System.out.println("约束向量 b_ub: " + b_ub);
        
        // 手动调用转换方法来查看转换结果
        var conversionResult = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        IVector convertedC = conversionResult.getFirst();
        IMatrix convertedA = conversionResult.getSecond();
        IVector convertedB = conversionResult.getThird();
        
        System.out.println("\n转换后:");
        System.out.println("目标函数: " + convertedC);
        System.out.println("约束矩阵 A_eq: " + convertedA);
        System.out.println("约束向量 b_eq: " + convertedB);
        
        // 验证转换结果
        // 约束应该是:
        // -x1 - x2 + s1 = -4  =>  x1 + x2 - s1 = 4
        // -2*x1 - x2 + s2 = -6  =>  2*x1 + x2 - s2 = 6
        
        // 创建求解器并求解
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        OptResult result = solver.solveWithNonNegativeEqualConstraints(convertedC, convertedA, convertedB, null);
        
        System.out.println("\n求解结果: " + result);
        if (result != null && result.isConverged()) {
            System.out.println("最优解: " + result.getOptimalPoint());
            System.out.println("最优值: " + result.getOptimalValue());
            
            // 验证理论解: x1=2, x2=2, 目标值=10
            IVector solution = result.getOptimalPoint();
            if (solution.length() >= 2) {
                double x1 = solution.get(0);
                double x2 = solution.get(1);
                System.out.println("x1 = " + x1 + ", x2 = " + x2);
                System.out.println("目标函数值: " + (3*x1 + 2*x2));
                System.out.println("约束验证:");
                System.out.println("  x1 + x2 = " + (x1 + x2) + " >= 4? " + (x1 + x2 >= 4));
                System.out.println("  2*x1 + x2 = " + (2*x1 + x2) + " >= 6? " + (2*x1 + x2 >= 6));
            }
        }
    }
}