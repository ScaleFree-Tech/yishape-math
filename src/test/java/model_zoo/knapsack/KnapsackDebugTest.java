package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.linpg.ComMath4LinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 调试背包问题求解器，找出为什么结果是320而不是410
 */
public class KnapsackDebugTest {
    public static void main(String[] args) {
        // 创建背包问题数据
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 理论最优解: [1,1,0,0,0,1,1] = 60+100+200+50 = 410
        // 重量: 10+20+60+10 = 100 <= 100 (满足约束)
        
        System.out.println("=== 背包问题调试测试 ===");
        System.out.println("物品价值: " + java.util.Arrays.toString(values));
        System.out.println("物品重量: " + java.util.Arrays.toString(weights));
        System.out.println("背包容量: " + capacity);
        System.out.println("理论最优解: [1,1,0,0,0,1,1] = 410");
        System.out.println();
        
        // 转换为最小化问题
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        IMatrix A_ub = Linalg.matrix(new double[][]{weights});
        IVector b_ub = Linalg.vector(new double[]{capacity});
        
        // 测试ComMath4LinProgSolver
        System.out.println("=== 测试ComMath4LinProgSolver ===");
        try {
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
        } catch (Exception e) {
            System.out.println("ComMath4异常: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
        
        // 测试BetterSimplexLinProgSolver
        System.out.println("=== 测试BetterSimplexLinProgSolver ===");
        try {
            RereSimplexLinProgSolver base = new RereSimplexLinProgSolver();
            base.setVerbose(true);
            OptResult betterResult = base.solve(c, A_ub, b_ub);
            
            if (betterResult != null) {
                System.out.println("BetterSimplex结果:");
                System.out.println("  目标值: " + betterResult.getOptimalValue());
                System.out.println("  解: " + betterResult.getOptimalPoint());
                System.out.println("  转换后的最大化值: " + (-betterResult.getOptimalValue()));
            } else {
                System.out.println("BetterSimplex求解失败");
            }
        } catch (Exception e) {
            System.out.println("BetterSimplex异常: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println();
        
        // 测试RereIntegerProg with RereSimplexLinProgSolver
        System.out.println("=== 测试RereIntegerProg with BetterSimplexLinProgSolver ===");
        try {
            RereSimplexLinProgSolver base = new RereSimplexLinProgSolver();
            base.setVerbose(true);
            RereIntegerProg solver = new RereIntegerProg(base);
            solver.setAllVariablesBinary();
            solver.setMaxDepth(100);
            solver.setGapTolerance(1e-10);
            solver.setTolerance(1e-10);
            solver.setMaxIterations(10000);
            solver.setVerbose(true);
            
            OptResult result = solver.solve(c, A_ub, b_ub);
            
            if (result != null) {
                System.out.println("RereIntegerProg结果:");
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