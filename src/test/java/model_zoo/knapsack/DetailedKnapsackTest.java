package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 详细测试RereIntegerProg在背包问题中的行为
 */
public class DetailedKnapsackTest {
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
        
        System.out.println("=== 背包问题详细测试 ===");
        System.out.println("物品列表:");
        for (int i = 0; i < itemNames.length; i++) {
            System.out.printf("  %s: 价值=%.1f, 重量=%.1f\n", itemNames[i], values[i], weights[i]);
        }
        System.out.println("背包容量: " + capacity);
        
        // 理论最优解
        double[] optimalSolution = {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0};
        IVector optimal = Linalg.vector(optimalSolution);
        double optimalValue = 0;
        double optimalWeight = 0;
        for (int i = 0; i < optimal.size(); i++) {
            optimalValue += optimal.get(i).doubleValue() * values[i];
            optimalWeight += optimal.get(i).doubleValue() * weights[i];
        }
        System.out.println("\n理论最优解:");
        System.out.println("  解向量: " + optimal);
        System.out.println("  总价值: " + optimalValue);
        System.out.println("  总重量: " + optimalWeight);
        
        // 转换为最小化问题
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        IMatrix A_ub = Linalg.matrix(new double[][]{weights});
        IVector b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("\n=== 使用RereIntegerProg求解 ===");
        try {
            RereSimplexLinProgSolver base = new RereSimplexLinProgSolver();
            RereIntegerProg solver = new RereIntegerProg(base);
            solver.setAllVariablesBinary();
            
            // 设置详细输出以便调试
            solver.setVerbose(true);
            solver.setMaxDepth(100);
            solver.setGapTolerance(1e-10);
            solver.setTolerance(1e-10);
            solver.setMaxIterations(10000);
            
            OptResult result = solver.solve(c, A_ub, b_ub);
            
            if (result != null) {
                System.out.println("\nRereIntegerProg结果:");
                System.out.println("  目标值: " + result.getOptimalValue());
                System.out.println("  解: " + result.getOptimalPoint());
                System.out.println("  转换后的最大化值: " + (-result.getOptimalValue()));
                
                // 验证解的有效性
                IVector solution = result.getOptimalPoint();
                if (solution != null) {
                    double totalWeight = 0;
                    double totalValue = 0;
                    boolean isBinary = true;
                    
                    for (int i = 0; i < solution.size(); i++) {
                        double value = solution.get(i).doubleValue();
                        // 检查0-1约束
                        if (Math.abs(value) > 1e-6 && Math.abs(value - 1.0) > 1e-6) {
                            isBinary = false;
                        }
                        
                        totalWeight += value * weights[i];
                        totalValue += value * values[i];
                    }
                    
                    System.out.println("  总重量: " + totalWeight + " (容量: " + capacity + ")");
                    System.out.println("  总价值: " + totalValue);
                    System.out.println("  是否满足重量约束: " + (totalWeight <= capacity + 1e-6));
                    System.out.println("  是否为0-1解: " + isBinary);
                    
                    // 比较与理论最优解的差距
                    double gap = optimalValue - totalValue;
                    System.out.println("  与理论最优解的差距: " + gap);
                }
            } else {
                System.out.println("RereIntegerProg求解失败");
            }
        } catch (Exception e) {
            System.out.println("RereIntegerProg异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
}