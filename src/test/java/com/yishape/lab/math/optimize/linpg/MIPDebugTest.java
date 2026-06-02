package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;

public class MIPDebugTest {
    public static void main(String[] args) {
        // 目标函数系数
        IVector c = Linalg.vector(new double[]{2, 1});
        
        // 等式约束: x1 + x2 = 2.5
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2.5});
        
        // 创建求解器
        RereIntegerProg solver = new RereIntegerProg();
        solver.setIntegerVariable(0); // 只有x1是整数变量
        solver.setVerbose(true);
        
        // 求解
        try {
            OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            System.out.println("最优值: " + result.getOptimalValue());
            System.out.println("最优解: " + result.getOptimalPoint());
            
            // 验证解
            double x1 = (Double) result.getOptimalPoint().get(0);
            double x2 = (Double) result.getOptimalPoint().get(1);
            System.out.println("x1 = " + x1 + ", x2 = " + x2);
            System.out.println("验证: x1 + x2 = " + (x1 + x2));
            System.out.println("目标值: 2*" + x1 + " + 1*" + x2 + " = " + (2*x1 + x2));
        } catch (Exception e) {
            System.out.println("求解失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}