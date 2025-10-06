package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.OptResult;

public class TestKnapsack {
    public static void main(String[] args) {
        // 创建背包问题数据
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        System.out.println("=== 简单背包问题测试 ===");
        System.out.println("物品价值: " + java.util.Arrays.toString(values));
        System.out.println("物品重量: " + java.util.Arrays.toString(weights));
        System.out.println("背包容量: " + capacity);
        System.out.println("理论最优解: [1,1,0,0,0,1,1] = 410 (选择物品0,1,5,6)");
        System.out.println("重量总和: " + (10+20+60+10) + " = 100");
        System.out.println();
        
        // 转换为最小化问题
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        IMatrix A_ub = Linalg.matrix(new double[][]{weights});
        IVector b_ub = Linalg.vector(new double[]{capacity});
        
        // 测试RereIntegerProg with RereSimplexLinProgSolver
        System.out.println("=== 测试RereIntegerProg ===");
        try {
            RereSimplexLinProgSolver base = new RereSimplexLinProgSolver();
            RereIntegerProg solver = new RereIntegerProg(base);
            solver.setAllVariablesBinary();
            solver.setVerbose(true);
            solver.setMaxIterations(100000); // 增加最大迭代次数
            solver.setMaxDepth(300); // 增加最大深度
            solver.setGapTolerance(1e-6); // 减小间隙容差
            
            OptResult result = solver.solve(c, A_ub, b_ub);
            
            if (result != null) {
                System.out.println("结果:");
                System.out.println("  目标值: " + result.getOptimalValue());
                System.out.println("  解: " + result.getOptimalPoint());
                System.out.println("  转换后的最大化值: " + (-result.getOptimalValue()));
                
                // 验证解的正确性
                if (result.getOptimalPoint() != null) {
                    IVector solution = result.getOptimalPoint();
                    double totalWeight = 0;
                    double totalValue = 0;
                    System.out.print("  选择的物品: ");
                    for (int i = 0; i < solution.length() && i < values.length; i++) {
                        double val = (Double) solution.get(i);
                        if (val > 0.5) { // 二进制变量，大于0.5认为是1
                            totalWeight += weights[i];
                            totalValue += values[i];
                            System.out.print("物品" + i + "(价值" + values[i] + ",重量" + weights[i] + ") ");
                        }
                    }
                    System.out.println();
                    System.out.println("  总价值: " + totalValue);
                    System.out.println("  总重量: " + totalWeight);
                    
                    // 检查是否为理论最优解
                    if (Math.abs(totalValue - 410) < 1e-6) {
                        System.out.println("  *** 找到理论最优解! ***");
                    } else {
                        System.out.println("  未找到理论最优解，差距: " + (410 - totalValue));
                    }
                }
            } else {
                System.out.println("求解失败");
            }
        } catch (Exception e) {
            System.out.println("异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
}