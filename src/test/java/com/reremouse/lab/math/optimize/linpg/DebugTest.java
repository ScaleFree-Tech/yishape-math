package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.OptResult;

public class DebugTest {
    public static void main(String[] args) {
        // 测试复杂的整数规划问题
        // 目标函数: minimize c^T * x = [3, 2, 1] * [x1, x2, x3]^T = 3*x1 + 2*x2 + x3
        // 约束条件: 
        //   x1 + x2 + x3 = 4
        //   2*x1 + x2 = 5
        //   x1, x2, x3 >= 0 且为整数
        System.out.println("=== 测试复杂的整数规划问题 ===");
        IVector c = Linalg.vector(new double[]{3, 2, 1});
        
        // 等式约束
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1, 1},
            {2, 1, 0}
        });
        IVector b_eq = Linalg.vector(new double[]{4, 5});
        
        // 创建求解器
        RereIntegerProg solver = new RereIntegerProg();
        solver.setAllVariablesInteger(); // 所有变量都是整数
        solver.setVerbose(true); // Enable verbose output
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        if (result == null) {
            System.out.println("未找到解");
        } else {
            System.out.println("解: " + result.getOptimalPoint() + ", 最优值: " + result.getOptimalValue());
        }
    }
}