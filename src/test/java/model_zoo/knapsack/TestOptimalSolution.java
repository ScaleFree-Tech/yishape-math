package model_zoo.knapsack;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;

/**
 * 测试理论最优解是否满足约束
 */
public class TestOptimalSolution {
    public static void main(String[] args) {
        // 创建背包问题数据
        String[] itemNames = {
            "珠宝💎", 
            "古籍📚", 
            "相机📷", 
            "手表⌚", 
            "笔记本💻",
            "帐篷⛺",
            "食物🍎"
        };
        
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 理论最优解：选择物品1,2,6,7 (索引0,1,5,6)
        double[] optimalSolution = {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0};
        IVector solution = Linalg.vector(optimalSolution);
        
        System.out.println("=== 测试理论最优解 ===");
        System.out.println("解向量: " + solution);
        
        // 验证解的有效性
        double totalWeight = 0;
        double totalValue = 0;
        boolean isValid = true;
        
        // 检查0-1约束
        for (int i = 0; i < solution.size(); i++) {
            double value = solution.get(i).doubleValue();
            boolean isBinary = Math.abs(value) < 1e-6 || Math.abs(value - 1.0) < 1e-6;
            if (!isBinary) {
                System.out.println("  变量x" + (i+1) + " = " + value + " 违反0-1约束");
                isValid = false;
            }
            
            totalWeight += value * weights[i];
            totalValue += value * values[i];
        }
        
        System.out.println("  总重量: " + totalWeight + " (容量: " + capacity + ")");
        System.out.println("  总价值: " + totalValue);
        System.out.println("  是否满足重量约束: " + (totalWeight <= capacity + 1e-6));
        System.out.println("  是否满足0-1约束: " + isValid);
        System.out.println("  解是否有效: " + (isValid && totalWeight <= capacity + 1e-6));
        
        if (isValid && totalWeight <= capacity + 1e-6) {
            System.out.println("✅ 理论最优解是可行的！");
            System.out.println("最优选择:");
            for (int i = 0; i < solution.size(); i++) {
                if (Math.abs(solution.get(i).doubleValue() - 1.0) < 1e-6) {
                    System.out.println("  " + itemNames[i] + " (价值: " + values[i] + ", 重量: " + weights[i] + ")");
                }
            }
        } else {
            System.out.println("❌ 理论最优解不可行！");
        }
    }
}