package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.ComMath4LinProgSolver;
import com.yishape.lab.math.optimize.linpg.LinProgUtil;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.util.Tuple3;

/**
 * 详细调试BetterSimplexLinProgSolver在背包问题上的线性松弛求解
 */
public class DetailedSolverDebugTest {
    
    public static void main(String[] args) {
        System.out.println("🔍 详细求解器调试分析");
        System.out.println("========================================");
        
        // 背包问题数据
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 转换为最小化目标函数（取负值）
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("📋 原始问题设置:");
        System.out.println("价值向量: " + java.util.Arrays.toString(values));
        System.out.println("重量向量: " + java.util.Arrays.toString(weights));
        System.out.println("容量: " + capacity);
        System.out.println("目标函数c (最小化): " + c);
        System.out.println("约束矩阵A_ub: " + displayMatrix(A_ub));
        System.out.println("约束向量b_ub: " + b_ub);
        System.out.println();
        
        // 检查约束转换过程
        System.out.println("🔄 约束转换过程分析:");
        Tuple3<IVector, IMatrix, IVector> converted = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        IVector c_eq = converted.getFirst();
        IMatrix A_eq = converted.getSecond(); 
        IVector b_eq = converted.getThird();
        
        System.out.println("转换后的目标函数c_eq: " + c_eq);
        System.out.println("转换后的约束矩阵A_eq: " + displayMatrix(A_eq));
        System.out.println("转换后的约束向量b_eq: " + b_eq);
        System.out.println();
        
        // 手动计算理论最优解
        System.out.println("🧮 理论分析:");
        System.out.println("对于线性松弛，我们要最大化价值，即最小化 -Σ(价值[i] * x[i])");
        System.out.println("约束: Σ(重量[i] * x[i]) ≤ 容量");
        System.out.println("价值重量比: ");
        for (int i = 0; i < values.length; i++) {
            System.out.printf("  x%d: 价值%.0f/重量%.0f = %.2f\n", i+1, values[i], weights[i], values[i]/weights[i]);
        }
        System.out.println("按价值重量比排序: x6(3.33) > x3(4.0) > x2(5.0) > x5(3.0) > x1(6.0) > x4(2.0) > x7(5.0)");
        System.out.println("理论最优线性松弛解应该选择价值重量比最高的变量...");
        System.out.println();
        
        // 测试BetterSimplexLinProgSolver
        System.out.println("🔍 BetterSimplexLinProgSolver 详细调试:");
        testBetterSimplex(c_eq, A_eq, b_eq);
        
        System.out.println();
        
        // 测试ComMath4LinProgSolver
        System.out.println("🔍 ComMath4LinProgSolver 详细调试:");
        testComMath4(c_eq, A_eq, b_eq);
        
        System.out.println();
        
        // 手动验证最优解
        System.out.println("🧪 手动验证最优解:");
        manualOptimalSolutionVerification(c, A_ub, b_ub);
    }
    
    private static String displayMatrix(IMatrix matrix) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n");
        for (int i = 0; i < matrix.rows(); i++) {
            sb.append("  [");
            for (int j = 0; j < matrix.cols(); j++) {
                sb.append(String.format("%8.2f", matrix.get(i, j).doubleValue()));
                if (j < matrix.cols() - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        return sb.toString();
    }
    
    private static void testBetterSimplex(IVector c_eq, IMatrix A_eq, IVector b_eq) {
        try {
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            
            System.out.println("开始求解等式约束问题...");
            OptResult result = solver.solveWithNonNegativeEqualConstraints(c_eq, A_eq, b_eq, null);
            
            if (result != null && result.isConverged()) {
                System.out.println("✅ 求解成功");
                System.out.println("目标函数值: " + result.getOptimalValue());
                System.out.println("解向量: " + result.getOptimalPoint());
                
                // 验证解
                IVector solution = result.getOptimalPoint();
                if (solution.length() >= 7) {
                    System.out.println("原始变量部分: ");
                    for (int i = 0; i < 7; i++) {
                        System.out.printf("  x%d = %.6f\n", i+1, solution.get(i).doubleValue());
                    }
                    
                    // 计算约束值
                    double constraintValue = 0.0;
                    for (int i = 0; i < 7; i++) {
                        constraintValue += A_eq.get(0, i).doubleValue() * solution.get(i).doubleValue();
                    }
                    System.out.println("约束检查: " + constraintValue + " = " + b_eq.get(0));
                }
                
                if (solution.length() > 7) {
                    System.out.println("松弛变量: ");
                    for (int i = 7; i < solution.length(); i++) {
                        System.out.printf("  s%d = %.6f\n", i-6, solution.get(i).doubleValue());
                    }
                }
                
            } else {
                System.out.println("❌ 求解失败");
                if (result != null) {
                    System.out.println("失败原因: " + result.getConvergenceReason());
                }
            }
        } catch (Exception e) {
            System.out.println("❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testComMath4(IVector c_eq, IMatrix A_eq, IVector b_eq) {
        try {
            ComMath4LinProgSolver solver = new ComMath4LinProgSolver();
            
            System.out.println("开始求解等式约束问题...");
            OptResult result = solver.solveWithNonNegativeEqualConstraints(c_eq, A_eq, b_eq, null);
            
            if (result != null && result.isConverged()) {
                System.out.println("✅ 求解成功");
                System.out.println("目标函数值: " + result.getOptimalValue());
                System.out.println("解向量: " + result.getOptimalPoint());
                
                // 验证解
                IVector solution = result.getOptimalPoint();
                if (solution.length() >= 7) {
                    System.out.println("变量值: ");
                    for (int i = 0; i < Math.min(7, solution.length()); i++) {
                        System.out.printf("  x%d = %.6f\n", i+1, solution.get(i).doubleValue());
                    }
                    
                    // 计算约束值 (需要从转换后的约束矩阵A_eq中提取原始约束)
                    double constraintValue = 0.0;
                    double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
                    for (int i = 0; i < Math.min(7, solution.length()); i++) {
                        constraintValue += weights[i] * solution.get(i).doubleValue();
                    }
                    System.out.println("原始约束检查: " + constraintValue + " ≤ 100.0");
                }
                
            } else {
                System.out.println("❌ 求解失败");
                if (result != null) {
                    System.out.println("失败原因: " + result.getConvergenceReason());
                }
            }
        } catch (Exception e) {
            System.out.println("❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void manualOptimalSolutionVerification(IVector c, IMatrix A_ub, IVector b_ub) {
        System.out.println("验证不同可行解的目标函数值:");
        
        // 候选解1: BetterSimplex产生的线性松弛解 [10,0,0,0,0,0,0]
        double[] sol1 = {10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        verifySolution("BetterSimplex线性松弛解", sol1, c, A_ub, b_ub);
        
        // 候选解2: 合理的线性松弛解 [0,0,0,0,0,1.67,0] (选择价值重量比最高的x6)
        double[] sol2 = {0.0, 0.0, 0.0, 0.0, 0.0, 100.0/60.0, 0.0};
        verifySolution("价值重量比最优解", sol2, c, A_ub, b_ub);
        
        // 候选解3: 另一个合理解 [0,5,0,0,0,0,0] (选择x2)
        double[] sol3 = {0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        verifySolution("选择x2的解", sol3, c, A_ub, b_ub);
        
        // 候选解4: 混合解 [0,0,3.33,0,0,0,0] (选择x3)
        double[] sol4 = {0.0, 0.0, 100.0/30.0, 0.0, 0.0, 0.0, 0.0};
        verifySolution("选择x3的解", sol4, c, A_ub, b_ub);
    }
    
    private static void verifySolution(String name, double[] solution, IVector c, IMatrix A_ub, IVector b_ub) {
        System.out.printf("\n📊 %s: [", name);
        for (int i = 0; i < solution.length; i++) {
            System.out.printf("%.2f", solution[i]);
            if (i < solution.length - 1) System.out.print(", ");
        }
        System.out.print("]\n");
        
        // 计算目标函数值
        double objValue = 0.0;
        for (int i = 0; i < solution.length; i++) {
            objValue += c.get(i).doubleValue() * solution[i];
        }
        System.out.printf("  目标函数值（最小化）: %.2f\n", objValue);
        System.out.printf("  目标函数值（最大化）: %.2f\n", -objValue);
        
        // 检查约束
        double constraintValue = 0.0;
        for (int i = 0; i < solution.length; i++) {
            constraintValue += A_ub.get(0, i).doubleValue() * solution[i];
        }
        System.out.printf("  约束值: %.2f ≤ %.1f\n", constraintValue, b_ub.get(0).doubleValue());
        System.out.printf("  约束满足: %s\n", (constraintValue <= b_ub.get(0).doubleValue() + 1e-6) ? "✅" : "❌");
    }
}