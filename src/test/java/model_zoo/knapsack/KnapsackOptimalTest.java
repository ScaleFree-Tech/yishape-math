package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 测试不同参数设置下的背包问题求解
 */
public class KnapsackOptimalTest {
    public static void main(String[] args) {
        System.out.println("🔍 背包问题最优解搜索测试");
        System.out.println("========================================");
        
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("🎯 理论最优解: [1,1,0,0,0,1,1] = 410");
        System.out.println();
        
        // 测试1: 增加搜索深度和迭代次数
        System.out.println("🧪 测试1: 增加搜索参数");
        testWithParameters(c, A_ub, b_ub, 200, 50000, 1e-12, true);
        
        // 测试2: 更严格的精度
        System.out.println("🧪 测试2: 更严格的数值精度");
        testWithParameters(c, A_ub, b_ub, 100, 10000, 1e-15, false);
        
        // 测试3: 启用详细输出观察分支过程
        System.out.println("🧪 测试3: 观察分支过程 (限制5次迭代)");
        testWithParameters(c, A_ub, b_ub, 10, 5, 1e-10, true);
    }
    
    private static void testWithParameters(IVector c, IMatrix A_ub, IVector b_ub, 
                                         int maxDepth, int maxIterations, 
                                         double tolerance, boolean verbose) {
        try {
            RereIntegerProg solver = new RereIntegerProg(new RereSimplexLinProgSolver());
            solver.setAllVariablesBinary();
            solver.setMaxDepth(maxDepth);
            solver.setGapTolerance(tolerance);
            solver.setTolerance(tolerance);
            solver.setMaxIterations(maxIterations);
            solver.setVerbose(verbose);
            
            System.out.printf("  参数: 深度=%d, 迭代=%d, 精度=%.0e, 详细=%s\n", 
                maxDepth, maxIterations, tolerance, verbose);
            
            long startTime = System.currentTimeMillis();
            OptResult result = solver.solve(c, A_ub, b_ub);
            long duration = System.currentTimeMillis() - startTime;
            
            if (result != null && result.getOptimalPoint() != null) {
                double objectiveValue = -result.getOptimalValue();
                System.out.printf("  ✅ 结果: 目标值=%.0f, 时间=%dms\n", objectiveValue, duration);
                System.out.printf("  📐 解: %s\n", result.getOptimalPoint());
                
                // 检查是否找到最优解
                if (Math.abs(objectiveValue - 410.0) < 1e-6) {
                    System.out.println("  🎉 找到理论最优解！");
                } else {
                    System.out.printf("  ⚠️ 距离最优解差距: %.0f\n", 410.0 - objectiveValue);
                }
                
                // 验证解的可行性
                IVector solution = result.getOptimalPoint();
                double totalWeight = 0;
                for (int i = 0; i < solution.length(); i++) {
                    totalWeight += A_ub.get(0, i).doubleValue() * solution.get(i).doubleValue();
                }
                System.out.printf("  ⚖️ 重量检查: %.1f ≤ %.1f (%s)\n", 
                    totalWeight, 100.0, totalWeight <= 100.1 ? "✅" : "❌");
                    
            } else {
                System.out.println("  ❌ 求解失败");
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
        }
        System.out.println();
    }
}