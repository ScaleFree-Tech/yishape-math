package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 背包问题固定版本 - 使用修复后的BetterSimplexLinProgSolver
 */
public class KnapsackProblemFixed {
    public static void main(String[] args) {
        System.out.println("🎒=== 修复后的背包问题求解 ===🎒");
        System.out.println("========================================");
        
        // 背包问题数据
        String[] itemNames = {
            "珠宝💎", "古籍📚", "相机📷", "手表⌚", "笔记本💻", "帐篷⛺", "食物🍎"
        };
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 展示物品信息
        System.out.println("🎯 探险家的物品清单：");
        System.out.println("物品名称\t\t\t价值\t重量\t价值密度");
        System.out.println("================================================");
        for (int i = 0; i < itemNames.length; i++) {
            double density = values[i] / weights[i];
            System.out.printf("%-20s\t%.1f\t%.1f\t%.2f\n", itemNames[i], values[i], weights[i], density);
        }
        System.out.println("================================================");
        System.out.println("🎒 背包最大承重: " + capacity + " 公斤");
        System.out.println();
        
        // 转换为优化问题
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        // 使用BetterSimplexLinProgSolver
        System.out.println("🔍 使用修复后的BetterSimplexLinProgSolver:");
        testIntegerProgramming("BetterSimplexLinProgSolver", new RereSimplexLinProgSolver(), c, A_ub, b_ub, itemNames, values, weights);
        
        // 手动验证理论最优解
        System.out.println("\n🧮 理论最优解验证:");
        System.out.println("按价值密度排序的贪心策略应该是：");
        System.out.println("1. 珠宝(6.0) - 重量10，价值60");
        System.out.println("2. 古籍(5.0) - 重量20，价值100");  
        System.out.println("3. 食物(5.0) - 重量10，价值50");
        System.out.println("4. 相机(4.0) - 重量30，价值120");
        System.out.println("5. 帐篷(3.33) - 重量60，价值200");
        System.out.println("6. 笔记本(3.0) - 重量50，价值150");
        System.out.println("7. 手表(2.0) - 重量40，价值80");
        System.out.println();
        
        // 验证几个理论候选解
        verifyCandidate("贪心解1: [1,1,0,0,0,1,1]", new int[]{1,1,0,0,0,1,1}, values, weights, capacity);
        verifyCandidate("贪心解2: [1,1,1,0,0,0,1]", new int[]{1,1,1,0,0,0,1}, values, weights, capacity);
        verifyCandidate("高价值解: [0,1,1,0,1,1,0]", new int[]{0,1,1,0,1,1,0}, values, weights, capacity);
        verifyCandidate("当前解: [1,0,1,0,1,0,1]", new int[]{1,0,1,0,1,0,1}, values, weights, capacity);
    }
    
    private static void testIntegerProgramming(String solverName, ILinProgSolver baseSolver,
                                             IVector c, IMatrix A_ub, IVector b_ub,
                                             String[] itemNames, double[] values, double[] weights) {
        try {
            RereIntegerProg solver = new RereIntegerProg(baseSolver);
            solver.setAllVariablesBinary();
            solver.setMaxDepth(100);
            solver.setGapTolerance(1e-10);
            solver.setTolerance(1e-10);
            solver.setMaxIterations(10000);
            solver.setVerbose(false);
            
            OptResult result = solver.solve(c, A_ub, b_ub);
            
            if (result != null && result.getOptimalPoint() != null) {
                System.out.println("  ✅ 求解完成");
                System.out.println("  💰 目标函数值（最大化）: " + (-result.getOptimalValue()));
                System.out.println("  📐 解向量: " + result.getOptimalPoint());
                System.out.println("  🔄 收敛状态: " + result.isConverged());
                
                // 详细分析解
                IVector solution = result.getOptimalPoint();
                double totalWeight = 0;
                double totalValue = 0;
                
                System.out.println("\n  📊 详细选择分析:");
                for (int i = 0; i < solution.length() && i < itemNames.length; i++) {
                    int selected = (int) Math.round(solution.get(i).doubleValue());
                    if (selected == 1) {
                        totalWeight += weights[i];
                        totalValue += values[i];
                        System.out.printf("  ✅ %s: 价值%.0f, 重量%.0f\n", itemNames[i], values[i], weights[i]);
                    }
                }
                System.out.printf("  📦 总重量: %.1f/%.1f, 总价值: %.0f\n", totalWeight, 100.0, totalValue);
                
            } else {
                System.out.println("  ❌ 求解失败");
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void verifyCandidate(String name, int[] solution, double[] values, double[] weights, double capacity) {
        double totalValue = 0;
        double totalWeight = 0;
        
        for (int i = 0; i < solution.length; i++) {
            if (solution[i] == 1) {
                totalValue += values[i];
                totalWeight += weights[i];
            }
        }
        
        boolean feasible = totalWeight <= capacity + 1e-6;
        System.out.printf("%-30s: 价值=%.0f, 重量=%.1f, 可行=%s\n", 
            name, totalValue, totalWeight, feasible ? "✅" : "❌");
    }
}