package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.util.Tuple2;

public class LPSolverTest {
    public static void main(String[] args) {
        // Test the LP solver directly
        // 目标函数: minimize [2, 1] * [x1, x2] = 2*x1 + x2
        // 约束条件: x1 + x2 = 2.5
        // x1, x2 >= 0
        System.out.println("=== 测试线性规划求解器 ===");
        IVector c = Linalg.vector(new double[]{2, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2.5});
        
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        
        if (result == null) {
            System.out.println("线性规划问题无解");
        } else {
            System.out.println("线性规划解: " + result.getSecond() + ", 最优值: " + result.getFirst());
        }
        
        // Expected optimal solution: x1 = 0, x2 = 2.5, objective = 2.5
        // The solver is incorrectly returning: x1 = 2.5, x2 = 0, objective = 5.0
    }
}