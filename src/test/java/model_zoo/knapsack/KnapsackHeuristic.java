package model_zoo.knapsack;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.optimize.OptResult;
import java.util.*;

/**
 * 简单的背包问题启发式求解器
 */
public class KnapsackHeuristic {
    
    /**
     * 使用贪心算法求解背包问题
     * @param values 物品价值数组
     * @param weights 物品重量数组
     * @param capacity 背包容量
     * @return 最优解
     */
    public static OptResult solveGreedy(double[] values, double[] weights, double capacity) {
        int n = values.length;
        
        // 计算价值密度（价值/重量）
        List<Item> items = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            items.add(new Item(i, values[i], weights[i], values[i] / weights[i]));
        }
        
        // 按价值密度降序排序
        items.sort((a, b) -> Double.compare(b.density, a.density));
        
        // 贪心选择物品
        double totalValue = 0;
        double totalWeight = 0;
        double[] solution = new double[n];
        
        for (Item item : items) {
            if (totalWeight + item.weight <= capacity) {
                totalWeight += item.weight;
                totalValue += item.value;
                solution[item.index] = 1.0;
            }
        }
        
        IVector optimalPoint = Linalg.vector(solution);
        return new OptResult(totalValue, optimalPoint);
    }
    
    /**
     * 使用动态规划求解0-1背包问题（精确解）
     * @param values 物品价值数组
     * @param weights 物品重量数组
     * @param capacity 背包容量
     * @return 最优解
     */
    public static OptResult solveDP(double[] values, double[] weights, double capacity) {
        int n = values.length;
        int intCapacity = (int) Math.round(capacity);
        
        // dp[i][w] 表示前i个物品在容量为w时的最大价值
        double[][] dp = new double[n + 1][intCapacity + 1];
        
        // 填充dp表
        for (int i = 1; i <= n; i++) {
            int weight = (int) Math.round(weights[i - 1]);
            double value = values[i - 1];
            
            for (int w = 0; w <= intCapacity; w++) {
                // 不选择第i个物品
                dp[i][w] = dp[i - 1][w];
                
                // 如果能选择第i个物品
                if (weight <= w) {
                    dp[i][w] = Math.max(dp[i][w], dp[i - 1][w - weight] + value);
                }
            }
        }
        
        // 回溯找出选择的物品
        double[] solution = new double[n];
        int w = intCapacity;
        for (int i = n; i > 0; i--) {
            int weight = (int) Math.round(weights[i - 1]);
            if (dp[i][w] != dp[i - 1][w]) {
                solution[i - 1] = 1.0;
                w -= weight;
            }
        }
        
        IVector optimalPoint = Linalg.vector(solution);
        return new OptResult(dp[n][intCapacity], optimalPoint);
    }
    
    /**
     * 物品类
     */
    private static class Item {
        int index;
        double value;
        double weight;
        double density;
        
        Item(int index, double value, double weight, double density) {
            this.index = index;
            this.value = value;
            this.weight = weight;
            this.density = density;
        }
    }
    
    public static void main(String[] args) {
        // 测试背包问题
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        System.out.println("=== 背包问题测试 ===");
        System.out.println("物品价值: " + Arrays.toString(values));
        System.out.println("物品重量: " + Arrays.toString(weights));
        System.out.println("背包容量: " + capacity);
        System.out.println("理论最优解: [1,1,0,0,0,1,1] = 410 (选择物品0,1,5,6)");
        System.out.println();
        
        // 贪心算法求解
        System.out.println("=== 贪心算法求解 ===");
        OptResult greedyResult = solveGreedy(values, weights, capacity);
        System.out.println("目标值: " + greedyResult.getOptimalValue());
        System.out.println("解: " + greedyResult.getOptimalPoint());
        System.out.println();
        
        // 动态规划求解
        System.out.println("=== 动态规划求解 ===");
        OptResult dpResult = solveDP(values, weights, capacity);
        System.out.println("目标值: " + dpResult.getOptimalValue());
        System.out.println("解: " + dpResult.getOptimalPoint());
    }
}