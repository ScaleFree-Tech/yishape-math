package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.linpg.ComMath4LinProgSolver;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 背包问题求解器对比测试
 * 用于诊断BetterSimplexLinProgSolver和ComMath4LinProgSolver在背包问题上的差异
 */
public class KnapsackSolverComparisonTest {
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
        
        System.out.println("=== 背包问题求解器对比测试 ===");
        System.out.println("物品数量: " + values.length);
        System.out.println("背包容量: " + capacity);
        System.out.println("目标函数: " + c);
        System.out.println("约束矩阵: " + A_ub);
        System.out.println("约束向量: " + b_ub);
        System.out.println();
        
        // 测试ComMath4LinProgSolver
        System.out.println("=== 使用ComMath4LinProgSolver ===");
        ILinProgSolver comMath4Solver = new ComMath4LinProgSolver();
        OptResult comMath4Result = comMath4Solver.solve(c, A_ub, b_ub);
        if (comMath4Result != null) {
            double comMath4Value = -comMath4Result.getOptimalValue(); // 转换回最大化
            System.out.println("ComMath4结果: 最大价值 = " + comMath4Value);
            System.out.println("解向量: " + comMath4Result.getOptimalPoint());
            System.out.println("原始目标值: " + comMath4Result.getOptimalValue());
        } else {
            System.out.println("ComMath4求解失败");
        }
        System.out.println();
        
        // 测试BetterSimplexLinProgSolver
        System.out.println("=== 使用BetterSimplexLinProgSolver ===");
        ILinProgSolver betterSolver = new RereSimplexLinProgSolver();
        ((RereSimplexLinProgSolver) betterSolver).setVerbose(true); // 启用详细输出
        OptResult betterResult = betterSolver.solve(c, A_ub, b_ub);
        if (betterResult != null) {
            double betterValue = -betterResult.getOptimalValue(); // 转换回最大化
            System.out.println("BetterSimplex结果: 最大价值 = " + betterValue);
            System.out.println("解向量: " + betterResult.getOptimalPoint());
            System.out.println("原始目标值: " + betterResult.getOptimalValue());
        } else {
            System.out.println("BetterSimplex求解失败");
        }
        System.out.println();
        
        // 比较结果
        if (comMath4Result != null && betterResult != null) {
            double comMath4Value = -comMath4Result.getOptimalValue();
            double betterValue = -betterResult.getOptimalValue();
            System.out.println("=== 结果对比 ===");
            System.out.println("ComMath4最大价值: " + comMath4Value);
            System.out.println("BetterSimplex最大价值: " + betterValue);
            System.out.println("差异: " + Math.abs(comMath4Value - betterValue));
        }
    }
}