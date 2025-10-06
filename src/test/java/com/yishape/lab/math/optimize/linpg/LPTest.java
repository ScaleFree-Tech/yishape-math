package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;

public class LPTest {

    public static void main(String[] args) {
        // Test the LP relaxation for the mixed integer problem
        // 目标函数: minimize [2, 1] * [x1, x2] = 2*x1 + x2
        // 约束条件: x1 + x2 = 2.5
        // x1, x2 >= 0
        System.out.println("=== 测试线性规划松弛问题 ===");
        IVector c = Linalg.vector(new double[]{2, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2.5});

//        ILinProgSolver solver = new SimplexLinProgSolver();
//        ILinProgSolver solver = new ComMath4LinProgSolver();
//        ILinProgSolver solver = new InteriorPointLinProgSolver();
        ILinProgSolver solver = new RereSimplexLinProgSolver();
//        ILinProgSolver solver = new LangMultiplierLinProgSolver();
        OptResult result = solver.solve(c, A_eq, b_eq);

        if (result == null) {
            System.out.println("线性规划问题无解");
        } else {
            System.out.println("线性规划解: " + result.getOptimalPoint() + ", 最优值: " + result.getOptimalValue());
        }

        // Test with x1 = 0, x2 = 2.5
        IVector testSolution = Linalg.vector(new double[]{0, 2.5});
        double testValue = (Double) c.innerProduct(testSolution);
        System.out.println("测试解 x1=0, x2=2.5 的目标值: " + testValue);

        // Test with x1 = 2, x2 = 0.5
        testSolution = Linalg.vector(new double[]{2, 0.5});
        testValue = (Double) c.innerProduct(testSolution);
        System.out.println("测试解 x1=2, x2=0.5 的目标值: " + testValue);
    }
}
