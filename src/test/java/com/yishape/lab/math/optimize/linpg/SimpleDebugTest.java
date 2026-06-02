package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.Test;

/**
 * 专门用于调试失败测试的简化版本
 */
public class SimpleDebugTest {
    
    @Test
    public void testMinimizationProblem() {
        System.out.println("=== 测试最小化问题 ===");
        
        // 问题: min 3*x1 + 2*x2
        // 约束: x1 + x2 >= 4, 2*x1 + x2 >= 6, x1, x2 >= 0
        // 转换为标准形式：x1 + x2 - s1 + a1 = 4, 2*x1 + x2 - s2 + a2 = 6
        double bigM = 10000.0;
        
        // 变量: [x1, x2, s1, s2, a1, a2]
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, bigM, bigM});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 1.0, 0.0},
            {2.0, 1.0, 0.0, -1.0, 0.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        System.out.println("求解结果:");
        System.out.println("是否收敛: " + result.isConverged());
        if (result.isConverged()) {
            System.out.println("目标函数值: " + result.getOptimalValue());
            System.out.println("解向量: " + result.getOptimalPoint());
        } else {
            System.out.println("收敛原因: " + result.getConvergenceReason());
        }
    }
    
    @Test 
    public void testLargeNumbersProblem() {
        System.out.println("=== 测试大数值问题 ===");
        
        // 使用较大的系数测试数值稳定性
        IVector c = Linalg.vector(new double[]{-1000.0, -2000.0, 0.0, 0.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {100.0, 200.0, 1.0, 0.0},
            {300.0, 100.0, 0.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{10000.0, 15000.0});
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        System.out.println("求解结果:");
        System.out.println("是否收敛: " + result.isConverged());
        if (result.isConverged()) {
            System.out.println("目标函数值: " + result.getOptimalValue());
            IVector solution = result.getOptimalPoint();
            System.out.println("解向量: " + solution);
            
            double x1 = solution.get(0);
            double x2 = solution.get(1);
            System.out.println("x1=" + x1 + ", x2=" + x2);
            
            // 验证约束
            double constraint1 = 100*x1 + 200*x2;
            double constraint2 = 300*x1 + 100*x2;
            System.out.println("约束1: 100*x1 + 200*x2 = " + constraint1 + " (期望: <= 10000)");
            System.out.println("约束2: 300*x1 + 100*x2 = " + constraint2 + " (期望: <= 15000)");
        } else {
            System.out.println("收敛原因: " + result.getConvergenceReason());
        }
    }
}