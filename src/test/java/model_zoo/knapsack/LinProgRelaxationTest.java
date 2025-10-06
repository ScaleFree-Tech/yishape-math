package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 线性规划松弛问题测试
 * 用于检查BetterSimplexLinProgSolver在背包问题松弛问题上的行为
 */
public class LinProgRelaxationTest {
    public static void main(String[] args) {
        // 物品价值和重量
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 转换为目标函数（最小化负价值）
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        
        // 约束矩阵和向量
        IMatrix A_ub = Linalg.matrix(new double[][]{weights});
        IVector b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("=== 线性规划松弛问题测试 ===");
        System.out.println("物品数量: " + values.length);
        System.out.println("背包容量: " + capacity);
        System.out.println();
        
        // 测试BetterSimplexLinProgSolver求解松弛问题
        System.out.println("=== 使用BetterSimplexLinProgSolver求解松弛问题 ===");
        RereSimplexLinProgSolver betterSolver = new RereSimplexLinProgSolver();
        betterSolver.setVerbose(true); // 启用详细输出
        OptResult betterResult = betterSolver.solve(c, A_ub, b_ub);
        
        if (betterResult != null) {
            double betterValue = -betterResult.getOptimalValue(); // 转换回最大化
            System.out.println("松弛问题结果: 最大价值 = " + betterValue);
            System.out.println("解向量: " + betterResult.getOptimalPoint());
            System.out.println("原始目标值: " + betterResult.getOptimalValue());
            
            // 验证解的约束满足情况
            IVector solution = betterResult.getOptimalPoint();
            double totalWeight = 0;
            for (int i = 0; i < solution.size(); i++) {
                totalWeight += solution.get(i).doubleValue() * weights[i];
            }
            System.out.println("解的总重量: " + totalWeight);
            System.out.println("是否满足约束: " + (totalWeight <= capacity + 1e-6));
        } else {
            System.out.println("BetterSimplex求解失败");
        }
        
        // 测试最优整数解作为初始解
        System.out.println("\n=== 测试最优整数解 ===");
        // 最优解: 物品1,2,6,7 -> [1,1,0,0,0,1,1]
        double[] optimalSolution = {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0};
        IVector initialSolution = Linalg.vector(optimalSolution);
        
        // 验证最优解的目标值
        double optimalValue = 0;
        double optimalWeight = 0;
        for (int i = 0; i < values.length; i++) {
            optimalValue += optimalSolution[i] * values[i];
            optimalWeight += optimalSolution[i] * weights[i];
        }
        System.out.println("理论最优整数解:");
        System.out.println("  解向量: " + initialSolution);
        System.out.println("  总价值: " + optimalValue);
        System.out.println("  总重量: " + optimalWeight);
        System.out.println("  是否满足约束: " + (optimalWeight <= capacity));
    }
}