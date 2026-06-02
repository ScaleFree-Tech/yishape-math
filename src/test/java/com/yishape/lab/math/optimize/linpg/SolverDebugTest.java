package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.Test;

public class SolverDebugTest {
    
    @Test
    public void debugLargerProblem() {
        // 测试问题：3变量2约束
        // maximize 2*x1 + 3*x2 + x3
        // subject to: x1 + 2*x2 + x3 = 5
        //            2*x1 + x2 + 2*x3 = 6
        //            x1, x2, x3 >= 0
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        
        IVector c = Linalg.vector(new double[]{2.0, 3.0, 1.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 2.0, 1.0},
            {2.0, 1.0, 2.0}
        });
        IVector b_eq = Linalg.vector(new double[]{5.0, 6.0});
        
        System.out.println("=== 问题设置 ===");
        System.out.println("目标函数: maximize 2*x1 + 3*x2 + x3");
        System.out.println("约束1: x1 + 2*x2 + x3 = 5");
        System.out.println("约束2: 2*x1 + x2 + 2*x3 = 6");
        
        OptResult result = solver.maximize(c, null, null, A_eq, b_eq, null);
        
        System.out.println("\n=== TABLEAU DEBUG ===");
        // 打印tableau结构
        if (result.getOptimalPoint() != null) {
            System.out.println("Tableau dimensions: " + result.getOptimalPoint().size() + " variables");
        }
        
        System.out.println("\n=== 求解结果 ===");
        System.out.println("是否收敛: " + result.isConverged());
        System.out.println("目标函数值: " + result.getOptimalValue());
        
        if (result.getOptimalPoint() != null) {
            IVector solution = result.getOptimalPoint();
            System.out.println("解向量:");
            for (int i = 0; i < solution.size(); i++) {
                System.out.println("  x" + (i+1) + " = " + solution.get(i));
            }
            
            System.out.println("\n=== 约束验证 ===");
            double constraint1 = solution.get(0) + 2*solution.get(1) + solution.get(2);
            double constraint2 = 2*solution.get(0) + solution.get(1) + 2*solution.get(2);
            
            System.out.println("约束1计算值: " + constraint1 + " (期望: 5.0)");
            System.out.println("约束2计算值: " + constraint2 + " (期望: 6.0)");
        }
        
        System.out.println("收敛原因: " + result.getConvergenceReason());
    }
}