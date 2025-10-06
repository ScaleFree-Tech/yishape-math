package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.linpg.ComMath4LinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 测试BetterSimplexLinProgSolver在背包问题中的表现
 */
public class BetterSimplexOnlyTest {
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
        RereSimplexLinProgSolver betterSolver = new RereSimplexLinProgSolver();
        OptResult betterResult = betterSolver.solve(c, A_ub, b_ub);
        
        if (betterResult != null) {
            System.out.println("BetterSimplex结果:");
            System.out.println("  目标值: " + betterResult.getOptimalValue());
            System.out.println("  解: " + betterResult.getOptimalPoint());
            System.out.println("  转换后的最大化值: " + (-betterResult.getOptimalValue()));
        } else {
            System.out.println("BetterSimplex求解失败");
        }
        
        System.out.println("\n=== 测试ComMath4LinProgSolver ===");
        ComMath4LinProgSolver comSolver = new ComMath4LinProgSolver();
        OptResult comResult = comSolver.solve(c, A_ub, b_ub);
        
        if (comResult != null) {
            System.out.println("ComMath4结果:");
            System.out.println("  目标值: " + comResult.getOptimalValue());
            System.out.println("  解: " + comResult.getOptimalPoint());
            System.out.println("  转换后的最大化值: " + (-comResult.getOptimalValue()));
        } else {
            System.out.println("ComMath4求解失败");
        }
    }
}