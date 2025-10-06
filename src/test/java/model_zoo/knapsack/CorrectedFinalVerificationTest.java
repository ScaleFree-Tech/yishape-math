package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 修正的最终验证测试
 * 正确构造带变量界限约束的线性规划问题
 */
public class CorrectedFinalVerificationTest {
    public static void main(String[] args) {
        System.out.println("=== 修正的最终验证测试 ===");
        
        // 测试带变量界限约束的正确构造
        testWithCorrectVariableBounds();
    }
    
    private static void testWithCorrectVariableBounds() {
        System.out.println("\n=== 测试: 正确构造带变量界限的背包问题 ===");
        
        // 物品价值和重量
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 转换为目标函数（最小化负价值）
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        
        // 构造包含原始约束和变量界限约束的矩阵
        // 我们有 7个变量 + 7个下界约束 + 7个上界约束 = 21个约束
        // 但我们需要更聪明地构造矩阵
        
        // 实际上，我们应该构造:
        // 1. 原始重量约束: 10*x1 + 20*x2 + ... + 10*x7 <= 100
        // 2. 下界约束: xi >= 0 (7个)
        // 3. 上界约束: xi <= 1 (7个)
        
        // 构造约束矩阵 (15 x 7):
        // 行0: 重量约束 [10, 20, 30, 40, 50, 60, 10] <= 100
        // 行1-7: 下界约束 [-1, 0, 0, 0, 0, 0, 0] <= 0, [0, -1, 0, ...] <= 0, ...
        // 行8-14: 上界约束 [1, 0, 0, 0, 0, 0, 0] <= 1, [0, 1, 0, ...] <= 1, ...
        
        int n = values.length;
        double[][] constraints = new double[1 + 2 * n][n];
        double[] rhs = new double[1 + 2 * n];
        
        // 原始重量约束
        System.arraycopy(weights, 0, constraints[0], 0, n);
        rhs[0] = capacity;
        
        // 变量下界约束 (xi >= 0 转换为 -xi <= 0)
        for (int i = 0; i < n; i++) {
            constraints[1 + i][i] = -1.0;
            rhs[1 + i] = 0.0;
        }
        
        // 变量上界约束 (xi <= 1)
        for (int i = 0; i < n; i++) {
            constraints[1 + n + i][i] = 1.0;
            rhs[1 + n + i] = 1.0;
        }
        
        IMatrix A_ub = Linalg.matrix(constraints);
        IVector b_ub = Linalg.vector(rhs);
        
        System.out.println("约束矩阵形状: " + A_ub.rows() + " x " + A_ub.cols());
        System.out.println("RHS向量长度: " + b_ub.length());
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result != null) {
            double value = -result.getOptimalValue(); // 转换回最大化
            System.out.println("带变量界限的解:");
            System.out.println("  最大价值 = " + value);
            System.out.println("  解向量 = " + result.getOptimalPoint());
            
            // 验证解的合法性
            IVector solution = result.getOptimalPoint();
            boolean isValid = true;
            double totalWeight = 0;
            System.out.println("  各变量值和重量:");
            for (int i = 0; i < n; i++) {
                double x = solution.get(i).doubleValue();
                double weight = x * weights[i];
                totalWeight += weight;
                System.out.printf("    x%d = %.3f, 重量 = %.3f\n", i+1, x, weight);
                
                // 检查变量界限
                if (x < -1e-6 || x > 1.0 + 1e-6) {
                    System.out.println("    变量x" + (i+1) + "超出界限[0,1]");
                    isValid = false;
                }
            }
            
            System.out.println("  总重量 = " + totalWeight);
            System.out.println("  是否满足重量约束 = " + (totalWeight <= capacity + 1e-6));
            System.out.println("  解是否完全合法 = " + (isValid && totalWeight <= capacity + 1e-6));
            
            // 计算实际价值
            double actualValue = 0;
            for (int i = 0; i < n; i++) {
                actualValue += solution.get(i).doubleValue() * values[i];
            }
            System.out.println("  实际计算价值 = " + actualValue);
        } else {
            System.out.println("求解失败");
        }
    }
}