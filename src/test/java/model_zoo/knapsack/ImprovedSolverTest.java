package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 测试改进后的RereIntegerProg求解器性能
 */
public class ImprovedSolverTest {
    
    public static void main(String[] args) {
        System.out.println("🔬 改进后的RereIntegerProg求解器测试");
        System.out.println("========================================");
        
        // 标准背包问题
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("🎯 理论最优解: [1,1,0,0,0,1,1] = 410");
        System.out.println("📊 物品价值密度排序:");
        for (int i = 0; i < values.length; i++) {
            System.out.printf("  x%d: 价值%.0f, 重量%.0f, 密度%.2f\n", 
                i+1, values[i], weights[i], values[i]/weights[i]);
        }
        System.out.println();
        
        // 测试1: 使用改进后的BetterSimplexLinProgSolver + 增强的RereIntegerProg
        System.out.println("🧪 测试1: 改进后的求解器 (详细输出)");
        testImprovedSolver(c, A_ub, b_ub, true, 100, 5000);
        
        System.out.println();
        
        // 测试2: 使用改进后的求解器 (无详细输出，更大搜索空间)
        System.out.println("🧪 测试2: 改进后的求解器 (扩大搜索)");
        testImprovedSolver(c, A_ub, b_ub, false, 200, 20000);
        
        System.out.println();
        
        // 测试3: 多次运行验证稳定性
        System.out.println("🧪 测试3: 稳定性测试 (5次运行)");
        testStability(c, A_ub, b_ub);
        
        System.out.println();
        
        // 测试4: 更大规模的背包问题
        System.out.println("🧪 测试4: 更大规模背包问题");
        testLargerProblem();
    }
    
    private static void testImprovedSolver(IVector c, IMatrix A_ub, IVector b_ub, 
                                         boolean verbose, int maxDepth, int maxIterations) {
        try {
            RereIntegerProg solver = new RereIntegerProg(new RereSimplexLinProgSolver());
            
            // 配置求解器参数
            solver.setAllVariablesBinary();
            solver.setMaxDepth(maxDepth);
            solver.setGapTolerance(1e-12);
            solver.setTolerance(1e-12);
            solver.setMaxIterations(maxIterations);
            solver.setVerbose(verbose);
            
            System.out.printf("  参数: 深度=%d, 迭代=%d, 详细=%s\n", 
                maxDepth, maxIterations, verbose);
            
            long startTime = System.currentTimeMillis();
            OptResult result = solver.solve(c, A_ub, b_ub);
            long duration = System.currentTimeMillis() - startTime;
            
            if (result != null && result.getOptimalPoint() != null) {
                double objectiveValue = -result.getOptimalValue();
                System.out.printf("  ✅ 结果: 目标值=%.0f, 时间=%dms\n", objectiveValue, duration);
                System.out.printf("  📐 解: %s\n", formatSolution(result.getOptimalPoint()));
                System.out.printf("  🔄 收敛: %s (%s)\n", 
                    result.isConverged() ? "是" : "否", 
                    result.getConvergenceReason());
                
                // 检查是否找到最优解
                if (Math.abs(objectiveValue - 410.0) < 1e-6) {
                    System.out.println("  🎉 找到理论最优解！");
                } else {
                    System.out.printf("  ⚠️ 距离最优解差距: %.0f (%.1f%%)\n", 
                        410.0 - objectiveValue, (410.0 - objectiveValue) / 410.0 * 100);
                }
                
                // 验证解的详细信息
                validateSolution(result.getOptimalPoint(), A_ub, b_ub);
                
            } else {
                System.out.println("  ❌ 求解失败");
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testStability(IVector c, IMatrix A_ub, IVector b_ub) {
        int runs = 5;
        int successes = 0;
        long totalTime = 0;
        double bestValue = 0;
        
        for (int run = 0; run < runs; run++) {
            try {
                RereIntegerProg solver = new RereIntegerProg(new RereSimplexLinProgSolver());
                solver.setAllVariablesBinary();
                solver.setMaxDepth(150);
                solver.setGapTolerance(1e-10);
                solver.setTolerance(1e-10);
                solver.setMaxIterations(15000);
                solver.setVerbose(false);
                
                long startTime = System.currentTimeMillis();
                OptResult result = solver.solve(c, A_ub, b_ub);
                long duration = System.currentTimeMillis() - startTime;
                totalTime += duration;
                
                if (result != null && result.getOptimalPoint() != null) {
                    double objectiveValue = -result.getOptimalValue();
                    bestValue = Math.max(bestValue, objectiveValue);
                    
                    System.out.printf("  运行 %d: 目标值=%.0f, 时间=%dms\n", 
                        run + 1, objectiveValue, duration);
                    
                    if (Math.abs(objectiveValue - 410.0) < 1e-6) {
                        successes++;
                    }
                } else {
                    System.out.printf("  运行 %d: 失败\n", run + 1);
                }
            } catch (Exception e) {
                System.out.printf("  运行 %d: 异常 - %s\n", run + 1, e.getMessage());
            }
        }
        
        System.out.printf("  📊 成功率: %d/%d (%.1f%%), 平均时间: %dms, 最佳值: %.0f\n", 
            successes, runs, (double) successes / runs * 100, totalTime / runs, bestValue);
    }
    
    private static void testLargerProblem() {
        // 创建一个更大的背包问题（10个物品）
        double[] values = {60, 100, 120, 80, 150, 200, 50, 90, 130, 110};
        double[] weights = {10, 20, 30, 40, 50, 60, 10, 25, 35, 45};
        double capacity = 150.0;
        
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.printf("  📋 问题规模: %d个物品, 容量%.0f\n", values.length, capacity);
        
        // 计算贪心解作为参考
        double greedyValue = calculateGreedySolution(values, weights, capacity);
        System.out.printf("  🎯 贪心解参考值: %.0f\n", greedyValue);
        
        try {
            RereIntegerProg solver = new RereIntegerProg(new RereSimplexLinProgSolver());
            solver.setAllVariablesBinary();
            solver.setMaxDepth(200);
            solver.setGapTolerance(1e-10);
            solver.setTolerance(1e-10);
            solver.setMaxIterations(25000);
            solver.setVerbose(false);
            
            long startTime = System.currentTimeMillis();
            OptResult result = solver.solve(c, A_ub, b_ub);
            long duration = System.currentTimeMillis() - startTime;
            
            if (result != null && result.getOptimalPoint() != null) {
                double objectiveValue = -result.getOptimalValue();
                System.out.printf("  ✅ 结果: 目标值=%.0f, 时间=%dms\n", objectiveValue, duration);
                System.out.printf("  📐 解: %s\n", formatSolution(result.getOptimalPoint()));
                
                double improvement = objectiveValue / greedyValue;
                System.out.printf("  📈 相对贪心解改进: %.1f%%\n", (improvement - 1) * 100);
                
                validateSolution(result.getOptimalPoint(), A_ub, b_ub);
            } else {
                System.out.println("  ❌ 求解失败");
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
        }
    }
    
    private static double calculateGreedySolution(double[] values, double[] weights, double capacity) {
        // 简单的贪心算法：按价值密度排序
        int n = values.length;
        java.util.List<Integer> items = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            items.add(i);
        }
        
        // 按价值密度降序排序
        items.sort((a, b) -> Double.compare(values[b]/weights[b], values[a]/weights[a]));
        
        double totalValue = 0;
        double totalWeight = 0;
        
        for (int i : items) {
            if (totalWeight + weights[i] <= capacity) {
                totalValue += values[i];
                totalWeight += weights[i];
            }
        }
        
        return totalValue;
    }
    
    private static String formatSolution(IVector solution) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < solution.length(); i++) {
            double val = solution.get(i).doubleValue();
            int intVal = (int) Math.round(val);
            sb.append(intVal);
            if (i < solution.length() - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }
    
    private static void validateSolution(IVector solution, IMatrix A_ub, IVector b_ub) {
        // 验证约束满足
        double totalWeight = 0;
        for (int i = 0; i < solution.length(); i++) {
            totalWeight += A_ub.get(0, i).doubleValue() * solution.get(i).doubleValue();
        }
        
        boolean constraintsSatisfied = totalWeight <= b_ub.get(0).doubleValue() + 1e-6;
        System.out.printf("  ⚖️ 约束检查: %.1f ≤ %.1f (%s)\n", 
            totalWeight, b_ub.get(0).doubleValue(), constraintsSatisfied ? "✅" : "❌");
        
        // 验证二进制性
        boolean allBinary = true;
        for (int i = 0; i < solution.length(); i++) {
            double val = solution.get(i).doubleValue();
            if (Math.abs(val) > 1e-6 && Math.abs(val - 1.0) > 1e-6) {
                allBinary = false;
                break;
            }
        }
        System.out.printf("  🔢 二进制检查: %s\n", allBinary ? "✅" : "❌");
    }
}