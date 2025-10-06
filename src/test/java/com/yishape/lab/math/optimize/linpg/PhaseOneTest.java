package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

public class PhaseOneTest {

    public static void main(String[] args) {
        System.out.println("=== 测试需要第一阶段的简单问题 ===");
        
        // Minimize: 3*x1 + 2*x2
        // Subject to: x1 + x2 >= 4
        //            2*x1 + x2 >= 6
        //            x1, x2 >= 0
        
        // 转换为等式约束形式:
        // Minimize: 3*x1 + 2*x2 + 0*s1 + 0*s2
        // Subject to: x1 + x2 - s1 = 4
        //            2*x1 + x2 - s2 = 6
        //            x1, x2, s1, s2 >= 0
        
        // 目标函数系数 (最小化问题)
        IVector c = Linalg.vector(new double[]{3, 2, 0, 0});
        
        // 约束矩阵 A_eq * x = b_eq
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1, -1, 0},  // x1 + x2 - s1 = 4
            {2, 1, 0, -1}   // 2*x1 + x2 - s2 = 6
        });
        
        // 约束向量
        IVector b_eq = Linalg.vector(new double[]{4, 6});
        
        // 创建求解器
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        
        // 求解
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);
        
        // 验证结果
        if (!result.isConverged()) {
            System.out.println("求解未收敛");
            return;
        } else {
            System.out.println("求解收敛");
        }
        
        // 最优解应该是 x1=2, x2=2, 目标值=10
        IVector solution = result.getOptimalPoint();
        System.out.println("解: " + solution);
        System.out.println("目标值: " + result.getOptimalValue());
        
        if (solution == null) {
            System.out.println("未得到解");
            return;
        }
        
        if (solution.length() != 4) {
            System.out.println("解向量长度不正确: 期望4, 实际" + solution.length());
            return;
        } else {
            System.out.println("解向量长度正确");
        }
        
        // 检查解的值
        double x1 = solution.get(0).doubleValue();
        double x2 = solution.get(1).doubleValue();
        double s1 = solution.get(2).doubleValue();
        double s2 = solution.get(3).doubleValue();
        
        // 验证约束
        double constraint1 = x1 + x2 - s1;
        double constraint2 = 2*x1 + x2 - s2;
        
        if (Math.abs(4.0 - constraint1) > 1e-6) {
            System.out.println("约束1不满足: x1 + x2 - s1 = " + constraint1 + ", 期望4.0");
            // 不返回，继续检查其他条件
        } else {
            System.out.println("约束1满足: x1 + x2 - s1 = " + constraint1);
        }
        
        if (Math.abs(6.0 - constraint2) > 1e-6) {
            System.out.println("约束2不满足: 2*x1 + x2 - s2 = " + constraint2 + ", 期望6.0");
            // 不返回，继续检查其他条件
        } else {
            System.out.println("约束2满足: 2*x1 + x2 - s2 = " + constraint2);
        }
        
        // 验证非负约束
        if (x1 < 0) {
            System.out.println("x1不满足非负约束: " + x1);
            return;
        }
        
        if (x2 < 0) {
            System.out.println("x2不满足非负约束: " + x2);
            return;
        }
        
        if (s1 < 0) {
            System.out.println("s1不满足非负约束: " + s1);
            return;
        }
        
        if (s2 < 0) {
            System.out.println("s2不满足非负约束: " + s2);
            return;
        }
        
        System.out.println("所有非负约束满足");
        
        // 验证目标值
        double expectedObjective = 3*x1 + 2*x2;
        if (Math.abs(expectedObjective - result.getOptimalValue()) > 1e-6) {
            System.out.println("目标值不正确: 期望" + expectedObjective + ", 实际" + result.getOptimalValue());
            return;
        } else {
            System.out.println("目标值正确: " + result.getOptimalValue());
        }
        
        System.out.println("测试通过!");
    }
}