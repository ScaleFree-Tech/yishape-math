package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 0-1 Integer Programming Example Using the Enhanced RereIntegerProg
 * 
 * This example demonstrates how to use the new binary variable methods
 * in the RereIntegerProg class to solve 0-1 integer programming problems.
 */
public class BinaryIntegerProgrammingExample {
    public static void main(String[] args) {
        System.out.println("=== 0-1整数规划示例 - 使用新的二进制变量方法 ===");
        System.out.println("=== 0-1 Integer Programming Example - Using New Binary Variable Methods ===");
        System.out.println();
        
        // Define a simple 0-1 programming problem:
        // Maximize: 3*x1 + 2*x2 + 4*x3
        // Subject to:
        //   2*x1 + 3*x2 + x3 <= 5
        //   x1 + x2 + 2*x3 <= 4
        //   x1, x2, x3 ∈ {0, 1}
        
        // Convert to minimization problem (solver minimizes)
        IVector<Double> c = Linalg.vector(new double[]{-3.0, -2.0, -4.0});
        
        // Constraint matrix (<= constraints)
        IMatrix<Double> A_ub = IMatrix.of(new double[][]{
            {2.0, 3.0, 1.0},
            {1.0, 1.0, 2.0}
        });
        
        // Constraint vector
        IVector<Double> b_ub = Linalg.vector(new double[]{5.0, 4.0});
        
        // Create integer programming solver
        RereIntegerProg solver = new RereIntegerProg();
        
        // Specify that ALL variables are 0-1 binary variables using the new method
        solver.setAllVariablesBinary(); // 3 variables
        
        // Alternative way to set individual binary variables:
        // solver.setBinaryVariable(0);
        // solver.setBinaryVariable(1);
        // solver.setBinaryVariable(2);
        
        System.out.println("正在求解0-1整数规划问题...");
        System.out.println("Solving 0-1 integer programming problem...");
        System.out.println();
        
        // Solve the 0-1 integer programming problem
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        // Check if solution exists
        if (result == null) {
            System.out.println("未找到可行解 / No feasible solution found");
            return;
        }
        
        // Extract solution
        IVector solution = result.getOptimalPoint();
        double optimalValue = -result.getOptimalValue(); // Convert back to maximization
        
        // Output results
        System.out.println("=== 最优解 / Optimal Solution ===");
        System.out.println("解向量 / Solution vector: " + solution);
        System.out.println("最大目标值 / Maximum objective value: " + optimalValue);
        System.out.println();
        
        // Verify 0-1 constraints
        System.out.println("=== 0-1约束验证 / 0-1 Constraint Verification ===");
        boolean allBinary = true;
        for (int i = 0; i < solution.size(); i++) {
            double value = solution.get(i).doubleValue();
            // Check if value is 0 or 1 (considering numerical error)
            boolean isBinary = Math.abs(value) < 1e-6 || Math.abs(value - 1.0) < 1e-6;
            allBinary &= isBinary;
            System.out.printf("x%d = %.6f (是0-1变量: %s)\n", i+1, value, isBinary ? "是/Yes" : "否/No");
        }
        System.out.println("所有变量都是0-1变量: " + (allBinary ? "是 (满足0-1约束)" : "否 (不满足0-1约束)"));
        System.out.println();
        
        // Verify constraint satisfaction
        System.out.println("=== 约束满足验证 / Constraint Satisfaction Verification ===");
        IVector Ax = A_ub.mmul(solution);
        boolean constraintsSatisfied = true;
        for (int i = 0; i < Ax.size(); i++) {
            boolean satisfied = Ax.get(i).doubleValue() <= b_ub.get(i).doubleValue() + 1e-9;
            constraintsSatisfied &= satisfied;
            System.out.printf("约束 %d: %.6f <= %.6f (%s)\n", i+1, Ax.get(i).doubleValue(), b_ub.get(i).doubleValue(), satisfied ? "满足/Satisfied" : "不满足/Not satisfied");
        }
        System.out.println("所有约束满足: " + (constraintsSatisfied ? "是/Yes" : "否/No"));
        System.out.println();
        
        System.out.println("=== 总结 / Summary ===");
        System.out.println("成功使用新的setBinaryVariable方法实现了0-1整数规划!");
        System.out.println("Successfully implemented 0-1 integer programming using the new setBinaryVariable methods!");
    }
}