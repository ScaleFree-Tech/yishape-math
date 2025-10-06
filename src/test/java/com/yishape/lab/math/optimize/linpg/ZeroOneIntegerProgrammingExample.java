package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 0-1 Integer Programming Example (Binary Integer Programming)
 * This example demonstrates how to solve a classic 0-1 knapsack problem
 * 
 * Problem:
 * A company has 5 projects to choose from, each with a profit and required resources.
 * The company has limited resources and wants to maximize profit by selecting projects.
 * Each project can either be selected (1) or not selected (0).
 * 
 * @author lteb2
 */
public class ZeroOneIntegerProgrammingExample {
    public static void main(String[] args) {
        System.out.println("=== 0-1整数规划示例 / 0-1 Integer Programming Example ===");
        System.out.println("问题：项目选择问题 (Project Selection Problem)");
        System.out.println("目标：在资源限制下最大化利润");
        System.out.println();
        
        // Define the problem data
        // Projects: 0    1    2    3    4
        double[] profits = {10.0, 15.0, 8.0, 20.0, 12.0};    // Profit for each project
        double[] resources = {3.0, 5.0, 2.0, 7.0, 4.0};     // Resource requirement for each project
        double totalResources = 10.0;                        // Total available resources
        
        System.out.println("项目信息 (Project Information):");
        System.out.println("项目编号\t利润\t资源需求");
        System.out.println("Project\tProfit\tResources");
        for (int i = 0; i < profits.length; i++) {
            System.out.printf("%d\t\t%.1f\t%.1f\n", i+1, profits[i], resources[i]);
        }
        System.out.println("总资源限制 (Total resource limit): " + totalResources);
        System.out.println();
        
        // Formulate as 0-1 integer programming problem
        // Maximize: sum(profits[i] * x[i]) for i = 0 to 4
        // Subject to: sum(resources[i] * x[i]) <= totalResources
        //            x[i] ∈ {0, 1} for all i
        
        // Convert to minimization problem (solver minimizes)
        // Minimize: -sum(profits[i] * x[i]) for i = 0 to 4
        IVector<Double> c = Linalg.vector(new double[]{-10.0, -15.0, -8.0, -20.0, -12.0});
        
        // Constraint matrix (resource constraint)
        IMatrix<Double> A_ub = IMatrix.of(new double[][]{
            {3.0, 5.0, 2.0, 7.0, 4.0}  // Resource constraint
        });
        
        // Constraint vector
        IVector<Double> b_ub = Linalg.vector(new double[]{totalResources});
        
        // Create integer programming solver
        RereIntegerProg solver = new RereIntegerProg();
        
        // Specify that ALL variables are 0-1 integer variables
        // In 0-1 programming, all variables must be binary (0 or 1)
        for (int i = 0; i < profits.length; i++) {
            solver.setIntegerVariable(i);
        }
        
        // Set solver parameters
        solver.setMaxDepth(20);
        solver.setGapTolerance(1e-6);
        solver.setTolerance(1e-9);
        
        System.out.println("正在求解0-1整数规划问题...");
        System.out.println("目标函数: maximize 10*x1 + 15*x2 + 8*x3 + 20*x4 + 12*x5");
        System.out.println("约束条件: 3*x1 + 5*x2 + 2*x3 + 7*x4 + 4*x5 <= 10");
        System.out.println("变量约束: x1, x2, x3, x4, x5 ∈ {0, 1}");
        System.out.println();
        
        // Solve the problem
        OptResult result = solver.solve(c, A_ub, null, A_ub, b_ub);
        
        if (result == null) {
            System.out.println("未找到可行解 (No feasible solution found)");
            return;
        }
        
        IVector solution = result.getOptimalPoint();
        double optimalValue = -result.getOptimalValue(); // Convert back to maximization
        
        // Output results
        System.out.println("=== 最优解 / Optimal Solution ===");
        System.out.println("解向量 (Solution vector): " + solution);
        System.out.println("最优利润 (Optimal profit): " + optimalValue);
        System.out.println();
        
        // Analyze the solution
        System.out.println("=== 解的分析 / Solution Analysis ===");
        double totalResourceUsed = 0;
        double totalProfit = 0;
        
        System.out.println("选中的项目 (Selected projects):");
        System.out.println("项目\t选择\t利润\t资源");
        System.out.println("Proj\tSel.\tProfit\tRes.");
        for (int i = 0; i < solution.size(); i++) {
            int selected = (int) Math.round(solution.get(i).doubleValue());
            if (selected == 1) {
                System.out.printf("%d\t%d\t%.1f\t%.1f\n", i+1, selected, profits[i], resources[i]);
                totalResourceUsed += resources[i];
                totalProfit += profits[i];
            }
        }
        
        System.out.println();
        System.out.println("资源使用情况 (Resource usage): " + totalResourceUsed + " <= " + totalResources);
        System.out.println("总利润 (Total profit): " + totalProfit);
        
        // Verify 0-1 constraints
        System.out.println();
        System.out.println("=== 0-1约束验证 / 0-1 Constraint Verification ===");
        boolean allBinary = true;
        for (int i = 0; i < solution.size(); i++) {
            double value = solution.get(i).doubleValue();
            // Check if value is 0 or 1 (considering numerical error)
            boolean isBinary = Math.abs(value) < 1e-6 || Math.abs(value - 1.0) < 1e-6;
            allBinary &= isBinary;
            System.out.printf("x%d = %.6f (是0-1变量: %s)\n", i+1, value, isBinary ? "是" : "否");
        }
        System.out.println("所有变量都是0-1变量: " + (allBinary ? "是" : "否"));
        
        // Verify constraint satisfaction
        IVector Ax = A_ub.mmul(solution);
        boolean constraintSatisfied = Ax.get(0).doubleValue() <= b_ub.get(0).doubleValue() + 1e-9;
        System.out.println("约束满足: " + (constraintSatisfied ? "是" : "否"));
    }
}