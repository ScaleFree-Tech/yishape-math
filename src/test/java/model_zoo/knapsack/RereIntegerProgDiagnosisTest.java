package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.ComMath4LinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 诊断RereIntegerProg在背包问题中的行为
 */
public class RereIntegerProgDiagnosisTest {
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
        
        System.out.println("=== RereIntegerProg背包问题诊断 ===");
        System.out.println("物品清单:");
        for (int i = 0; i < values.length; i++) {
            System.out.printf("物品%d: 价值=%.1f, 重量=%.1f\n", i+1, values[i], weights[i]);
        }
        System.out.println("背包容量: " + capacity);
        System.out.println();
        
        // 测试使用ComMath4LinProgSolver的RereIntegerProg
        System.out.println("=== 使用ComMath4LinProgSolver ===");
        RereIntegerProg comMath4Solver = new RereIntegerProg(new ComMath4LinProgSolver());
        comMath4Solver.setAllVariablesBinary();
        comMath4Solver.setVerbose(true);
        comMath4Solver.setMaxIterations(10000);
        comMath4Solver.setGapTolerance(1e-10);
        OptResult comMath4Result = comMath4Solver.solve(c, A_ub, b_ub);
        
        if (comMath4Result != null) {
            double value = -comMath4Result.getOptimalValue(); // 转换回最大化
            System.out.println("ComMath4结果:");
            System.out.println("  最大价值 = " + value);
            System.out.println("  解向量 = " + comMath4Result.getOptimalPoint());
            System.out.println("  是否收敛 = " + comMath4Result.isConverged());
        } else {
            System.out.println("ComMath4求解失败");
        }
        
        System.out.println();
        
        // 测试使用BetterSimplexLinProgSolver的RereIntegerProg
        System.out.println("=== 使用BetterSimplexLinProgSolver ===");
        RereIntegerProg betterSolver = new RereIntegerProg(new RereSimplexLinProgSolver());
        betterSolver.setAllVariablesBinary();
        betterSolver.setVerbose(true);
        betterSolver.setMaxIterations(10000);
        betterSolver.setGapTolerance(1e-10);
        OptResult betterResult = betterSolver.solve(c, A_ub, b_ub);
        
        if (betterResult != null) {
            double value = -betterResult.getOptimalValue(); // 转换回最大化
            System.out.println("BetterSimplex结果:");
            System.out.println("  最大价值 = " + value);
            System.out.println("  解向量 = " + betterResult.getOptimalPoint());
            System.out.println("  是否收敛 = " + betterResult.isConverged());
        } else {
            System.out.println("BetterSimplex求解失败");
        }
        
        // 验证理论最优解
        System.out.println("\n=== 理论最优解验证 ===");
        // 最优解: 物品1,2,6,7 -> [1,1,0,0,0,1,1]
        // 价值: 60 + 100 + 200 + 50 = 410
        // 重量: 10 + 20 + 60 + 10 = 100
        double[] optimalSolution = {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0};
        double optimalValue = 0;
        double optimalWeight = 0;
        for (int i = 0; i < values.length; i++) {
            optimalValue += optimalSolution[i] * values[i];
            optimalWeight += optimalSolution[i] * weights[i];
        }
        System.out.println("理论最优解:");
        System.out.println("  解向量 = " + Linalg.vector(optimalSolution));
        System.out.println("  最大价值 = " + optimalValue);
        System.out.println("  总重量 = " + optimalWeight);
    }
}