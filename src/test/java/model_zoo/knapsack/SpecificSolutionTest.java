package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 测试特定解是否可达
 */
public class SpecificSolutionTest {
    public static void main(String[] args) {
        System.out.println("🔍 特定解可达性测试");
        System.out.println("========================================");
        
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 测试理论最优解
        int[] optimalSolution = {1, 1, 0, 0, 0, 1, 1};
        System.out.println("🎯 理论最优解: [1,1,0,0,0,1,1]");
        
        // 验证解的可行性
        double totalValue = 0, totalWeight = 0;
        for (int i = 0; i < optimalSolution.length; i++) {
            if (optimalSolution[i] == 1) {
                totalValue += values[i];
                totalWeight += weights[i];
            }
        }
        System.out.printf("  总价值: %.0f\n", totalValue);
        System.out.printf("  总重量: %.1f/%.1f\n", totalWeight, capacity);
        System.out.printf("  可行性: %s\n", totalWeight <= capacity ? "✅" : "❌");
        System.out.println();
        
        // 使用热启动测试该解是否可以被求解器找到
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        double[] initValues = new double[optimalSolution.length];
        for (int i = 0; i < optimalSolution.length; i++) {
            initValues[i] = optimalSolution[i];
        }
        IVector initX = Linalg.vector(initValues);
        
        System.out.println("🧪 使用理论最优解作为热启动:");
        System.out.println("  初始解: " + initX);
        
        try {
            RereIntegerProg solver = new RereIntegerProg(new RereSimplexLinProgSolver());
            solver.setAllVariablesBinary();
            solver.setInitialX(initX);  // 设置热启动
            solver.setMaxDepth(50);
            solver.setGapTolerance(1e-10);
            solver.setTolerance(1e-10);
            solver.setMaxIterations(1000);
            solver.setVerbose(true);
            
            OptResult result = solver.solve(c, A_ub, b_ub);
            
            if (result != null && result.getOptimalPoint() != null) {
                double objectiveValue = -result.getOptimalValue();
                System.out.printf("  ✅ 求解结果: 目标值=%.0f\n", objectiveValue);
                System.out.printf("  📐 解: %s\n", result.getOptimalPoint());
                
                if (Math.abs(objectiveValue - 410.0) < 1e-6) {
                    System.out.println("  🎉 找到理论最优解！");
                } else {
                    System.out.printf("  ⚠️ 与理论最优解不符，差距: %.0f\n", 410.0 - objectiveValue);
                }
            } else {
                System.out.println("  ❌ 求解失败");
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
        
        // 测试所有可能的可行解（暴力搜索验证）
        System.out.println("🔍 暴力搜索所有可行解（验证）:");
        findAllFeasibleSolutions(values, weights, capacity);
    }
    
    private static void findAllFeasibleSolutions(double[] values, double[] weights, double capacity) {
        int n = values.length;
        int totalSolutions = 1 << n; // 2^n
        
        double bestValue = 0;
        int[] bestSolution = null;
        int feasibleCount = 0;
        
        for (int mask = 0; mask < totalSolutions; mask++) {
            double totalValue = 0, totalWeight = 0;
            
            for (int i = 0; i < n; i++) {
                if ((mask & (1 << i)) != 0) {
                    totalValue += values[i];
                    totalWeight += weights[i];
                }
            }
            
            if (totalWeight <= capacity + 1e-6) {
                feasibleCount++;
                if (totalValue > bestValue) {
                    bestValue = totalValue;
                    bestSolution = new int[n];
                    for (int i = 0; i < n; i++) {
                        bestSolution[i] = (mask & (1 << i)) != 0 ? 1 : 0;
                    }
                }
                
                // 输出高价值的可行解
                if (totalValue >= 380) {
                    System.out.printf("  解: [");
                    for (int i = 0; i < n; i++) {
                        System.out.print((mask & (1 << i)) != 0 ? "1" : "0");
                        if (i < n-1) System.out.print(",");
                    }
                    System.out.printf("], 价值=%.0f, 重量=%.1f\n", totalValue, totalWeight);
                }
            }
        }
        
        System.out.printf("\n  📊 统计: 可行解总数=%d, 全解空间=%d\n", feasibleCount, totalSolutions);
        System.out.printf("  🏆 最优解: [");
        if (bestSolution != null) {
            for (int i = 0; i < bestSolution.length; i++) {
                System.out.print(bestSolution[i]);
                if (i < bestSolution.length-1) System.out.print(",");
            }
            System.out.printf("], 价值=%.0f\n", bestValue);
        }
    }
}