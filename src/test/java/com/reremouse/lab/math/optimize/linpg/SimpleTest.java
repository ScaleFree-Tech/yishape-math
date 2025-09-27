package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

public class SimpleTest {
    public static void main(String[] args) {
        // 测试简单的线性规划问题
        // 目标函数: minimize x1 + x2
        // 约束条件: x1 + x2 = 2
        // x1, x2 >= 0
        System.out.println("=== 测试简单的线性规划问题 ===");
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});
        
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        if (result == null) {
            System.out.println("线性规划问题无解");
        } else {
            System.out.println("线性规划解: " + result.getSecond() + ", 最优值: " + result.getFirst());
        }
        
        // 测试整数规划问题
        System.out.println("\n=== 测试简单的整数规划问题 ===");
        RereIntegerProg intSolver = new RereIntegerProg(solver);
        intSolver.addIntegerVariables(0, 1);
        result = intSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        if (result == null) {
            System.out.println("整数规划问题无解");
        } else {
            System.out.println("整数规划解: " + result.getSecond() + ", 最优值: " + result.getFirst());
        }
    }
}