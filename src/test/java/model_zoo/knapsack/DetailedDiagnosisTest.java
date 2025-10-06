package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 详细诊断测试
 * 用于深入分析BetterSimplexLinProgSolver在背包问题上的行为
 */
public class DetailedDiagnosisTest {
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
        
        System.out.println("=== 详细诊断测试 ===");
        System.out.println("物品清单:");
        for (int i = 0; i < values.length; i++) {
            System.out.printf("物品%d: 价值=%.1f, 重量=%.1f\n", i+1, values[i], weights[i]);
        }
        System.out.println("背包容量: " + capacity);
        System.out.println();
        
        // 手动构造最优解的松弛问题
        System.out.println("=== 手动构造最优解的松弛问题 ===");
        // 最优整数解: 物品1,2,6,7 -> [1,1,0,0,0,1,1]
        // 对应的约束: 10*1 + 20*1 + 60*1 + 10*1 = 100 <= 100 ✓
        // 对应的价值: 60 + 100 + 200 + 50 = 410
        
        // 构造一个分数解来测试
        // 物品1: 1.0, 物品2: 1.0, 物品6: 1.0, 物品7: 1.0
        // 但这是整数解，我们想要一个真正的松弛解
        // 让我们尝试: 物品6: 1.0, 物品2: 1.0, 物品1: 1.0, 物品7: 1.0 (总重量100)
        // 或者: 物品6: 1.0, 物品5: 1.0 (重量110, 超重)
        // 或者: 物品6: 1.0, 物品2: 1.0, 物品1: 0.5, 物品7: 0.5 (重量20+10+5+5=40, 总重量90)
        
        // 实际上，线性规划松弛解应该是找到最大价值/重量比的组合
        System.out.println("计算价值/重量比:");
        for (int i = 0; i < values.length; i++) {
            double ratio = values[i] / weights[i];
            System.out.printf("物品%d: 价值/重量比 = %.3f\n", i+1, ratio);
        }
        System.out.println();
        
        // 按价值/重量比排序: 物品6(3.333), 物品7(5.0), 物品2(5.0), 物品3(4.0), 物品5(3.0), 物品1(6.0), 物品4(2.0)
        // 最优的松弛解应该是尽可能装入价值/重量比高的物品
        // 物品1: 6.0, 物品7: 5.0, 物品2: 5.0, 物品3: 4.0, 物品6: 3.333, 物品5: 3.0, 物品4: 2.0
        
        // 贪心解:
        // 1. 物品1 (重量10, 价值60) - 剩余90
        // 2. 物品7 (重量10, 价值50) - 剩余80  
        // 3. 物品2 (重量20, 价值100) - 剩余60
        // 4. 物品3 (重量30, 价值120) - 剩余30
        // 5. 物品6 (重量60, 价值200) - 无法完全装入，可以装入0.5个 -> 价值100
        // 总价值: 60 + 50 + 100 + 120 + 100 = 430
        
        // 但线性规划可以做得更好，它会找到真正的最优解
        
        // 测试BetterSimplexLinProgSolver
        System.out.println("=== 测试BetterSimplexLinProgSolver ===");
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result != null) {
            double value = -result.getOptimalValue(); // 转换回最大化
            System.out.println("线性规划解:");
            System.out.println("  最大价值 = " + value);
            System.out.println("  解向量 = " + result.getOptimalPoint());
            
            // 验证解
            IVector solution = result.getOptimalPoint();
            double totalWeight = 0;
            double totalValue = 0;
            for (int i = 0; i < solution.size(); i++) {
                double x = solution.get(i).doubleValue();
                totalWeight += x * weights[i];
                totalValue += x * values[i];
            }
            System.out.println("  验证总重量 = " + totalWeight);
            System.out.println("  验证总价值 = " + totalValue);
            System.out.println("  是否满足约束 = " + (totalWeight <= capacity + 1e-6));
        } else {
            System.out.println("求解失败");
        }
    }
}