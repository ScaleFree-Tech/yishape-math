package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 调试RereIntegerProg在背包问题中的行为
 */
public class DebugRereIntegerProg {
    public static void main(String[] args) {
        // 创建背包问题数据
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 转换为最小化问题
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        IMatrix A_ub = Linalg.matrix(new double[][]{weights});
        IVector b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("=== 测试RereIntegerProg with BetterSimplexLinProgSolver ===");
        try {
            RereSimplexLinProgSolver base = new RereSimplexLinProgSolver();
            RereIntegerProg solver = new RereIntegerProg(base);
            solver.setAllVariablesBinary();
            solver.setMaxDepth(100);
            solver.setGapTolerance(1e-10);
            solver.setTolerance(1e-10);
            solver.setMaxIterations(10000);
            solver.setVerbose(true);  // 开启详细输出
            
            OptResult result = solver.solve(c, A_ub, b_ub);
            
            if (result != null) {
                System.out.println("RereIntegerProg结果:");
                System.out.println("  目标值: " + result.getOptimalValue());
                System.out.println("  解: " + result.getOptimalPoint());
                System.out.println("  转换后的最大化值: " + (-result.getOptimalValue()));
                
                // 验证解的有效性
                IVector solution = result.getOptimalPoint();
                if (solution != null) {
                    // 检查是否满足重量约束
                    double totalWeight = 0;
                    double totalValue = 0;
                    for (int i = 0; i < solution.size(); i++) {
                        double value = Math.round(solution.get(i).doubleValue());  // 四舍五入到最近的整数
                        totalWeight += value * weights[i];
                        totalValue += value * values[i];
                    }
                    System.out.println("  总重量: " + totalWeight + " (容量: " + capacity + ")");
                    System.out.println("  总价值: " + totalValue);
                    System.out.println("  是否满足重量约束: " + (totalWeight <= capacity + 1e-6));
                    
                    // 检查是否为0-1解
                    boolean isBinary = true;
                    System.out.print("  解向量检查: ");
                    for (int i = 0; i < solution.size(); i++) {
                        double value = solution.get(i).doubleValue();
                        boolean isZeroOrOne = Math.abs(value) < 1e-6 || Math.abs(value - 1.0) < 1e-6;
                        isBinary &= isZeroOrOne;
                        System.out.print(String.format("x%d=%.3f(%s) ", i+1, value, isZeroOrOne ? "✓" : "✗"));
                    }
                    System.out.println();
                    System.out.println("  是否为0-1解: " + isBinary);
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