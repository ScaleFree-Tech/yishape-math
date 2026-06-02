package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;

public class SimpleOptimizationTest {
    
    @Test
    public void testSimpleMaximization() {
        // 简单的线性规划问题：
        // max x1 + x2
        // s.t. x1 + x2 <= 3
        //      x1, x2 >= 0
        
        // 目标函数系数向量（最大化问题，所以是负的）
        IVector c = Linalg.vector(new double[]{-1.0, -1.0, 0.0}); // 最小化负目标函数，包含松弛变量
        
        // 等式约束矩阵和向量
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0}  // x1 + x2 + s1 = 3
        });
        IVector b_eq = Linalg.vector(new double[]{3.0});
        
        // 创建求解器
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        System.out.println("求解结果: " + result);
        if (result != null && result.isConverged()) {
            System.out.println("最优解: " + result.getOptimalPoint());
            System.out.println("最优值: " + result.getOptimalValue());
            
            // 验证解
            IVector solution = result.getOptimalPoint();
            if (solution.length() >= 2) {
                double x1 = solution.get(0);
                double x2 = solution.get(1);
                System.out.println("x1 = " + x1 + ", x2 = " + x2);
                System.out.println("目标函数值: " + (x1 + x2));
                System.out.println("约束验证: x1 + x2 = " + (x1 + x2) + " <= 3? " + (x1 + x2 <= 3));
            }
        }
    }
}