package model_zoo.knapsack;

/**
 * 背包问题暴力搜索测试
 * 用于找到背包问题的最优解
 */
public class KnapsackBruteForceTest {
    public static void main(String[] args) {
        // 物品价值和重量
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        System.out.println("=== 背包问题暴力搜索 ===");
        System.out.println("物品清单:");
        for (int i = 0; i < values.length; i++) {
            System.out.printf("物品%d: 价值=%.1f, 重量=%.1f\n", i+1, values[i], weights[i]);
        }
        System.out.println("背包容量: " + capacity);
        System.out.println();
        
        // 暴力搜索所有可能的组合 (2^7 = 128种)
        double max_value = 0;
        int best_combination = 0;
        
        for (int i = 0; i < (1 << values.length); i++) {
            double total_value = 0;
            double total_weight = 0;
            boolean[] selected = new boolean[values.length];
            
            // 检查当前组合
            for (int j = 0; j < values.length; j++) {
                if ((i & (1 << j)) != 0) {
                    selected[j] = true;
                    total_value += values[j];
                    total_weight += weights[j];
                }
            }
            
            // 检查是否满足约束
            if (total_weight <= capacity) {
                if (total_value > max_value) {
                    max_value = total_value;
                    best_combination = i;
                }
            }
        }
        
        // 输出最优解
        System.out.println("=== 最优解 ===");
        System.out.println("最大价值: " + max_value);
        System.out.print("选择的物品: ");
        double total_weight = 0;
        for (int j = 0; j < 7; j++) {
            if ((best_combination & (1 << j)) != 0) {
                System.out.print("物品" + (j+1) + " ");
                total_weight += weights[j];
            }
        }
        System.out.println();
        System.out.println("总重量: " + total_weight);
        System.out.println();
        
        // 显示最优组合的详细信息
        System.out.println("=== 详细信息 ===");
        System.out.println("物品\t价值\t重量\t选择");
        System.out.println("========================");
        for (int j = 0; j < 7; j++) {
            String selected = ((best_combination & (1 << j)) != 0) ? "是" : "否";
            System.out.printf("物品%d\t%.1f\t%.1f\t%s\n", j+1, values[j], weights[j], selected);
        }
    }
}