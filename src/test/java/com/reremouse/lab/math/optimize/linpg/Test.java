package com.reremouse.lab.math.optimize.linpg;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.optimize.OptResult;

/**
 *
 * @author lteb2
 */
public class Test {
      public static void main(String[] args) {
        System.out.println("=== 复杂整数规划示例 / Complex Integer Programming Example ===");
        
        // 修改为一个更简单的整数规划问题，确保有可行解
        // 简化的背包问题示例
        // 物品价值: [3, 4, 5]
        // 物品重量: [2, 3, 4]
        // 背包容量: 5
        // 目标：最大化价值
        
        // 构造目标函数（最大化价值转换为最小化问题）
        // 由于求解器是最小化问题，需要将价值系数取负数
        IVector<Double> c = Linalg.vector(new double[]{-3.0, -4.0, -5.0});
        
        // 构造约束矩阵（重量约束）
        IMatrix<Double> A_eq = IMatrix.of(new double[][]{
            {2.0, 3.0, 4.0}  // 重量约束
        });
        
        // 构造约束右端向量（背包容量）
        IVector<Double> b_eq = Linalg.vector(new double[]{5.0});
        
        // 创建整数规划求解器
        RereIntegerProg solver = new RereIntegerProg();
        
        // 设置所有变量为整数变量（0-1变量）
        solver.setAllVariablesBinary();
        
        // 设置算法参数
        solver.setMaxDepth(20);
        solver.setGapTolerance(1e-6);
        solver.setTolerance(1e-9);
        
        System.out.println("正在求解整数规划问题...");
        System.out.println("目标函数: maximize 3*x1 + 4*x2 + 5*x3");
        System.out.println("约束条件: 2*x1 + 3*x2 + 4*x3 <= 5");
        System.out.println("变量约束: x1, x2, x3 为 0-1 变量");
        
        // 求解
        var result = solver.solve(c, A_eq, b_eq);
        
        if (result == null) {
            System.out.println("未找到可行解");
            return;
        }
        
        IVector solution = result.getOptimalPoint();
        double optimalValue = -result.getOptimalValue(); // 转换回正值
        
        // 输出结果
        System.out.println("最优解: " + solution);
        System.out.println("最优价值: " + optimalValue);
        
        // 分析解的含义
        System.out.println("\n解的分析:");
        String[] items = {"物品1", "物品2", "物品3"};
        double[] values = {3.0, 4.0, 5.0};
        double[] weights = {2.0, 3.0, 4.0};
        
        double totalWeight = 0;
        double totalValue = 0;
        
        for (int i = 0; i < solution.size(); i++) {
            int selected = (int) Math.round(solution.get(i).doubleValue());
            if (selected == 1) {
                System.out.println("  选择 " + items[i] + " (价值: " + values[i] + ", 重量: " + weights[i] + ")");
                totalWeight += weights[i];
                totalValue += values[i];
            }
        }
        
        System.out.println("\n总重量: " + totalWeight + " <= 5");
        System.out.println("总价值: " + totalValue);
        
        // 验证约束
        IVector Ax = A_eq.mmul(solution);
        boolean feasible = Ax.get(0).doubleValue() <= b_eq.get(0).doubleValue() + 1e-9;
        System.out.println("约束满足: " + feasible);
        
        // 验证0-1约束
        System.out.println("\n0-1约束验证:");
        boolean allBinary = true;
        for (int i = 0; i < solution.size(); i++) {
            double value = solution.get(i).doubleValue();
            boolean isBinary = Math.abs(value) < 1e-9 || Math.abs(value - 1.0) < 1e-9;
            allBinary &= isBinary;
            System.out.println("  x" + (i+1) + " = " + value + " (0-1变量: " + isBinary + ")");
        }
        System.out.println("所有变量都是0-1变量: " + allBinary);
    }
}
