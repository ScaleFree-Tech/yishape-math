package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 纯测试BetterSimplexLinProgSolver在背包问题中的表现
 */
public class PureBetterSimplexTest {
    public static void main(String[] args) {
        // 创建背包问题数据
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 转换为最小化问题
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        IMatrix A_ub = Linalg.matrix(new double[][]{weights});
        IVector b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("=== 测试BetterSimplexLinProgSolver ===");
        try {
            RereSimplexLinProgSolver betterSolver = new RereSimplexLinProgSolver();
            
            OptResult betterResult = betterSolver.solve(c, A_ub, b_ub);
            
            if (betterResult != null) {
                System.out.println("BetterSimplex结果:");
                System.out.println("  目标值: " + betterResult.getOptimalValue());
                System.out.println("  解: " + betterResult.getOptimalPoint());
                System.out.println("  转换后的最大化值: " + (-betterResult.getOptimalValue()));
                
                // 验证解的有效性
                IVector solution = betterResult.getOptimalPoint();
                if (solution != null) {
                    // 检查是否满足重量约束
                    double totalWeight = 0;
                    double totalValue = 0;
                    for (int i = 0; i < solution.size(); i++) {
                        double value = solution.get(i).doubleValue();
                        totalWeight += value * weights[i];
                        totalValue += value * values[i];
                    }
                    System.out.println("  总重量: " + totalWeight + " (容量: " + capacity + ")");
                    System.out.println("  总价值: " + totalValue);
                    System.out.println("  是否满足重量约束: " + (totalWeight <= capacity + 1e-6));
                }
            } else {
                System.out.println("BetterSimplex求解失败");
            }
        } catch (Exception e) {
            System.out.println("BetterSimplex异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
}