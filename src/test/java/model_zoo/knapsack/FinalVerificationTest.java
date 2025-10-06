package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 最终验证测试
 * 验证BetterSimplexLinProgSolver在有变量界限约束时的行为
 */
public class FinalVerificationTest {
    public static void main(String[] args) {
        System.out.println("=== 最终验证测试 ===");
        
        // 测试1: 原始背包问题（无变量界限）
        testOriginalProblem();
        
        // 测试2: 添加变量界限约束
        testWithVariableBounds();
    }
    
    private static void testOriginalProblem() {
        System.out.println("\n=== 测试1: 原始背包问题 ===");
        // 物品价值和重量
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 转换为目标函数（最小化负价值）
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        
        // 约束矩阵和向量
        IMatrix A_ub = Linalg.matrix(new double[][]{weights});
        IVector b_ub = Linalg.vector(new double[]{capacity});
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(false);
        OptResult result = solver.solve(c, A_ub, b_ub);
        
        if (result != null) {
            double value = -result.getOptimalValue(); // 转换回最大化
            System.out.println("原始问题解:");
            System.out.println("  最大价值 = " + value);
            System.out.println("  解向量 = " + result.getOptimalPoint());
        }
    }
    
    private static void testWithVariableBounds() {
        System.out.println("\n=== 测试2: 添加变量界限约束 ===");
        // 为每个变量添加 0 <= xi <= 1 约束
        // 这需要将界限约束转换为不等式约束
        
        // 物品价值和重量
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 转换为目标函数（最小化负价值）
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        
        // 构造包含原始约束和变量界限约束的矩阵
        // 原始约束: weights * x <= capacity
        // 变量界限: 0 <= xi <= 1  =>  -xi <= 0 且  xi <= 1
        
        // 构造扩展的约束矩阵
        // 行1: 原始重量约束
        // 行2-8: -xi <= 0 (即 xi >= 0)
        // 行9-15: xi <= 1
        
        int n = values.length;
        double[][] extendedConstraints = new double[1 + 2 * n][n];
        double[] extendedRHS = new double[1 + 2 * n];
        
        // 原始重量约束
        System.arraycopy(weights, 0, extendedConstraints[0], 0, n);
        extendedRHS[0] = capacity;
        
        // 变量下界约束 (xi >= 0  =>  -xi <= 0)
        for (int i = 0; i < n; i++) {
            extendedConstraints[1 + i][i] = -1.0;
            extendedRHS[1 + i] = 0.0;
        }
        
        // 变量上界约束 (xi <= 1)
        for (int i = 0; i < n; i++) {
            extendedConstraints[1 + n + i][i] = 1.0;
            extendedRHS[1 + n + i] = 1.0;
        }
        
        IMatrix A_ub = Linalg.matrix(extendedConstraints);
        IVector b_ub = Linalg.vector(extendedRHS);
        
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(false);
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
            for (int i = 0; i < n; i++) {
                double x = solution.get(i).doubleValue();
                if (x < -1e-6 || x > 1.0 + 1e-6) {
                    isValid = false;
                    break;
                }
                totalWeight += x * weights[i];
            }
            
            System.out.println("  总重量 = " + totalWeight);
            System.out.println("  解是否合法 = " + (isValid && totalWeight <= capacity + 1e-6));
        }
    }
}