package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * Simple 0-1 Integer Programming Example
 * 
 * This example shows the basic concept of 0-1 integer programming with a small problem:
 * 
 * Maximize: 3*x1 + 2*x2 + 5*x3
 * Subject to: 
 *   2*x1 + 1*x2 + 3*x3 <= 5  (Resource constraint)
 *   x1, x2, x3 ∈ {0, 1}      (Binary variables)
 * 
 * This is a simple knapsack problem with 3 items.
 */
public class SimpleZeroOneExample {
    public static void main(String[] args) {
        System.out.println("=== 简单0-1整数规划示例 / Simple 0-1 Integer Programming Example ===");
        System.out.println();
        
        // Problem data
        // Items:     1    2    3
        double[] values = {3.0, 2.0, 5.0};    // Value of each item
        double[] weights = {2.0, 1.0, 3.0};   // Weight of each item
        double capacity = 5.0;                // Knapsack capacity
        
        System.out.println("背包问题数据 (Knapsack Problem Data):");
        System.out.println("物品\t价值\t重量");
        System.out.println("Item\tValue\tWeight");
        for (int i = 0; i < values.length; i++) {
            System.out.printf("%d\t%.1f\t%.1f\n", i+1, values[i], weights[i]);
        }
        System.out.println("背包容量 (Knapsack capacity): " + capacity);
        System.out.println();
        
        // Formulate as 0-1 integer programming problem
        // Maximize: sum(values[i] * x[i])
        // Subject to: sum(weights[i] * x[i]) <= capacity
        //            x[i] ∈ {0, 1}
        
        // Convert to minimization (solver minimizes)
        IVector<Double> c = Linalg.vector(new double[]{-3.0, -2.0, -5.0});
        
        // Constraint: 2*x1 + 1*x2 + 3*x3 <= 5
        IMatrix<Double> A_ub = IMatrix.of(new double[][]{
            {2.0, 1.0, 3.0}
        });
        IVector<Double> b_ub = Linalg.vector(new double[]{capacity});
        
        // Create solver and specify 0-1 variables
        RereIntegerProg solver = new RereIntegerProg();
        solver.addBinaryVariables(0, 1, 2); // All variables are 0-1
        
        System.out.println("问题公式化 (Problem Formulation):");
        System.out.println("最大化: 3*x1 + 2*x2 + 5*x3");
        System.out.println("约束条件: 2*x1 + 1*x2 + 3*x3 <= 5");
        System.out.println("变量约束: x1, x2, x3 ∈ {0, 1}");
        System.out.println();
        
        // Solve
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result == null) {
            System.out.println("无可行解 (No feasible solution)");
            return;
        }
        
        IVector solution = result.getOptimalPoint();
        double maxValue = -result.getOptimalValue(); // Convert back to maximization
        
        System.out.println("=== 求解结果 / Solution ===");
        System.out.println("最优解 (Optimal solution): " + solution);
        System.out.println("最大价值 (Maximum value): " + maxValue);
        System.out.println();
        
        // Analyze solution
        System.out.println("=== 解的解释 / Solution Interpretation ===");
        double totalWeight = 0;
        System.out.println("选中的物品 (Selected items):");
        System.out.println("物品\t选择\t价值\t重量");
        System.out.println("Item\tSel.\tValue\tWeight");
        for (int i = 0; i < solution.size(); i++) {
            int selected = (int) Math.round(solution.get(i).doubleValue());
            if (selected == 1) {
                System.out.printf("%d\t%d\t%.1f\t%.1f\n", i+1, selected, values[i], weights[i]);
                totalWeight += weights[i];
            }
        }
        System.out.println("总重量 (Total weight): " + totalWeight + " <= " + capacity);
        System.out.println();
        
        // Verify 0-1 constraints
        System.out.println("=== 0-1约束验证 / 0-1 Constraint Verification ===");
        for (int i = 0; i < solution.size(); i++) {
            double value = solution.get(i).doubleValue();
            boolean isBinary = Math.abs(value) < 1e-6 || Math.abs(value - 1.0) < 1e-6;
            System.out.printf("x%d = %.6f (0-1变量: %s)\n", i+1, value, isBinary ? "是" : "否");
        }
    }
}