package model_zoo.knapsack;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.OptResult;

import java.util.Arrays;

/**
 * 测试 RereIntegerProg 的高级启发式算法
 * 验证所有新增的优化策略是否有效提升求解性能
 */
public class AdvancedHeuristicTest {

    /**
     * 背包问题测试数据
     */
    private static class KnapsackData {
        double[] values;      // 物品价值
        double[] weights;     // 物品重量
        double capacity;      // 背包容量
        double optimalValue;  // 理论最优值
        int[] optimalSolution; // 理论最优解

        KnapsackData(double[] values, double[] weights, double capacity, double optimalValue, int[] optimalSolution) {
            this.values = values;
            this.weights = weights;
            this.capacity = capacity;
            this.optimalValue = optimalValue;
            this.optimalSolution = optimalSolution;
        }
    }

    public static void main(String[] args) {
        System.out.println("🚀 高级启发式算法测试");
        System.out.println();

        // 测试用例1：原始7物品背包问题
        KnapsackData problem1 = new KnapsackData(
            new double[]{60, 100, 120, 80, 150, 200, 50},    // 价值
            new double[]{10, 20, 30, 40, 50, 60, 10},        // 重量
            100.0,                                            // 容量
            410.0,                                           // 理论最优值
            new int[]{1, 1, 0, 0, 0, 1, 1}                  // 理论最优解
        );

        // 测试用例2：10物品背包问题
        KnapsackData problem2 = new KnapsackData(
            new double[]{60, 100, 120, 80, 150, 200, 50, 90, 110, 70},
            new double[]{10, 20, 30, 40, 50, 60, 10, 15, 25, 12},
            150.0,
            580.0,
            new int[]{1, 1, 0, 0, 1, 0, 1, 1, 1, 0}
        );

        // 测试用例3：难解的紧密背包问题
        KnapsackData problem3 = new KnapsackData(
            new double[]{50, 50, 64, 46, 50, 5},
            new double[]{56, 59, 80, 64, 75, 17},
            190.0,
            150.0,
            new int[]{1, 1, 0, 0, 0, 1}
        );

        // 运行所有测试
        testProblem("标准7物品背包", problem1, true);
        testProblem("扩展10物品背包", problem2, false);
        testProblem("紧密背包问题", problem3, false);

        System.out.println();
        System.out.println("📊 性能对比测试");
        performanceComparison(problem1);
    }

    /**
     * 测试单个背包问题
     */
    private static void testProblem(String problemName, KnapsackData data, boolean verbose) {
        System.out.println("========================================");
        System.out.println("🎯 " + problemName);
        
        if (verbose) {
            System.out.println("🎯 理论最优解: " + Arrays.toString(data.optimalSolution) + " = " + data.optimalValue);
            printValueDensityRanking(data.values, data.weights);
        }

        // 构造线性规划问题
        IVector c = Linalg.vector(Arrays.stream(data.values).map(v -> -v).toArray()); // 转最小化
        IMatrix A_ub = Linalg.matrix(new double[][]{data.weights});
        IVector b_ub = Linalg.vector(new double[]{data.capacity});

        // 创建高级启发式求解器
        RereSimplexLinProgSolver lpSolver = new RereSimplexLinProgSolver();
        RereIntegerProg integerSolver = new RereIntegerProg(lpSolver);
        
        // 配置高级参数
        integerSolver.setMaxDepth(200);
        integerSolver.setMaxIterations(10000);
        integerSolver.setVerbose(verbose);
        
        // 设置0-1变量
        integerSolver.setAllVariablesBinary();

        long startTime = System.currentTimeMillis();
        OptResult result = integerSolver.solve(c, A_ub, b_ub);
        long endTime = System.currentTimeMillis();

        if (result != null && result.getOptimalPoint() != null) {
            double objectiveValue = -result.getOptimalValue(); // 转回最大化
            IVector solution = result.getOptimalPoint();
            
            System.out.println("  ✅ 结果: 目标值=" + objectiveValue + ", 时间=" + (endTime - startTime) + "ms");
            System.out.println("  🎯 解: " + formatSolution(solution));
            
            // 验证解的正确性
            validateSolution(solution, data, objectiveValue);
            
            // 计算与理论最优的差距
            double gap = Math.abs(data.optimalValue - objectiveValue);
            double gapPercent = (gap / data.optimalValue) * 100;
            System.out.println("  ⚠️ 距离最优解差距: " + gap + " (" + String.format("%.1f", gapPercent) + "%)");
            
        } else {
            System.out.println("  ❌ 未找到可行解");
        }
        System.out.println();
    }

    /**
     * 性能对比测试：比较不同配置的性能
     */
    private static void performanceComparison(KnapsackData data) {
        System.out.println("对比不同搜索深度和迭代次数的性能:");
        
        int[][] configs = {
            {50, 1000},    // 基础配置
            {100, 5000},   // 中等配置
            {200, 10000},  // 高级配置
            {300, 20000}   // 极限配置
        };
        
        for (int[] config : configs) {
            int maxDepth = config[0];
            int maxIterations = config[1];
            
            System.out.println("  配置: 深度=" + maxDepth + ", 迭代=" + maxIterations);
            
            // 构造问题
            IVector c = Linalg.vector(Arrays.stream(data.values).map(v -> -v).toArray());
            IMatrix A_ub = Linalg.matrix(new double[][]{data.weights});
            IVector b_ub = Linalg.vector(new double[]{data.capacity});

            RereSimplexLinProgSolver lpSolver = new RereSimplexLinProgSolver();
            RereIntegerProg integerSolver = new RereIntegerProg(lpSolver);
            integerSolver.setMaxDepth(maxDepth);
            integerSolver.setMaxIterations(maxIterations);
            integerSolver.setVerbose(false);
            
            integerSolver.setAllVariablesBinary();

            long startTime = System.currentTimeMillis();
            OptResult result = integerSolver.solve(c, A_ub, b_ub);
            long endTime = System.currentTimeMillis();

            if (result != null && result.getOptimalPoint() != null) {
                double objectiveValue = -result.getOptimalValue();
                double gap = Math.abs(data.optimalValue - objectiveValue);
                double gapPercent = (gap / data.optimalValue) * 100;
                
                System.out.println("    目标值=" + objectiveValue + 
                                 ", 时间=" + (endTime - startTime) + "ms" +
                                 ", 差距=" + String.format("%.1f", gapPercent) + "%");
            } else {
                System.out.println("    未找到解, 时间=" + (endTime - startTime) + "ms");
            }
        }
    }

    /**
     * 打印物品价值密度排序
     */
    private static void printValueDensityRanking(double[] values, double[] weights) {
        System.out.println("🎯 物品价值密度排序:");
        
        for (int i = 0; i < values.length; i++) {
            double density = values[i] / weights[i];
            System.out.println("  x" + (i+1) + ": 价值" + (int)values[i] + 
                             ", 重量" + (int)weights[i] + 
                             ", 密度" + String.format("%.2f", density));
        }
    }

    /**
     * 格式化解向量
     */
    private static String formatSolution(IVector solution) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < solution.length(); i++) {
            if (i > 0) sb.append(",");
            sb.append(Math.round((Double) solution.get(i)));
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * 验证解的约束满足情况
     */
    private static void validateSolution(IVector solution, KnapsackData data, double objectiveValue) {
        // 验证容量约束
        double totalWeight = 0.0;
        double totalValue = 0.0;
        
        for (int i = 0; i < Math.min(solution.length(), data.weights.length); i++) {
            double value = (Double) solution.get(i);
            totalWeight += value * data.weights[i];
            totalValue += value * data.values[i];
        }
        
        System.out.println("  ⚖️ 约束检查: " + String.format("%.1f", totalWeight) + 
                         " ≤ " + data.capacity + " (" + 
                         (totalWeight <= data.capacity + 1e-6 ? "✅" : "❌") + ")");
        
        // 验证二进制约束
        boolean binaryValid = true;
        for (int i = 0; i < solution.length(); i++) {
            double value = (Double) solution.get(i);
            if (Math.abs(value - Math.round(value)) > 1e-6) {
                binaryValid = false;
                break;
            }
        }
        System.out.println("  🔘 二进制检查: " + (binaryValid ? "✅" : "❌"));
        
        // 验证目标值计算
        double calculatedValue = 0.0;
        for (int i = 0; i < Math.min(solution.length(), data.values.length); i++) {
            calculatedValue += (Double) solution.get(i) * data.values[i];
        }
        boolean valueMatch = Math.abs(calculatedValue - objectiveValue) < 1e-6;
        System.out.println("  📊 目标值检查: 计算=" + String.format("%.1f", calculatedValue) + 
                         ", 报告=" + String.format("%.1f", objectiveValue) + 
                         " (" + (valueMatch ? "✅" : "❌") + ")");
    }
}