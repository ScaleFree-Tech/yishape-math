import com.reremouse.lab.math.linalg.*;
import com.reremouse.lab.math.optimize.linpg.RereIntegerProg;
import com.reremouse.lab.math.optimize.OptResult;

public class DebugIntegerProgramming {
    public static void main(String[] args) {
        System.out.println("=== 整数规划求解器测试 ===");
        
        // 测试案例1：简单的0-1背包问题
        testSimpleKnapsack();
        
        // 测试案例2：混合整数规划问题
        testMixedIntegerProgramming();
    }
    
    /**
     * 测试案例1：简单的0-1背包问题
     * 最大化: 3x1 + 2x2 + 4x3
     * 约束: x1 + 2x2 + x3 <= 4
     *       2x1 + x2 + 2x3 <= 5
     *       x1, x2, x3 ∈ {0, 1}
     */
    public static void testSimpleKnapsack() {
        System.out.println("\n=== 测试案例1：0-1背包问题 ===");
        
        try {
            // 目标函数系数 (最大化问题，所以取负数)
            IVector<Double> c = Linalg.vector(new double[]{-3.0, -2.0, -4.0});
            
            // 不等式约束 A_ub * x <= b_ub
            IMatrix<Double> A_ub = Linalg.matrix(new double[][]{
                {1.0, 2.0, 1.0},
                {2.0, 1.0, 2.0}
            });
            IVector<Double> b_ub = Linalg.vector(new double[]{4.0, 5.0});
            
            // 等式约束 (无)
            IMatrix<Double> A_eq = null;
            IVector<Double> b_eq = null;
            
            // 变量界限 (所有变量都是0-1变量)
            IVector<Double> bounds_lower = Linalg.vector(new double[]{0.0, 0.0, 0.0});
            IVector<Double> bounds_upper = Linalg.vector(new double[]{1.0, 1.0, 1.0});
            
            // 创建求解器
            RereIntegerProg solver = new RereIntegerProg();
            
            // 设置所有变量为0-1变量
            solver.addBinaryVariables(0, 1, 2);
            
            // 求解
            System.out.println("开始求解0-1背包问题...");
            OptResult result = solver.solve(c, A_ub, b_ub, A_eq, b_eq);
            
            if (result != null) {
                System.out.println("求解成功！");
                IVector<Double> solution = result.getOptimalPoint();
                double objValue = result.getOptimalValue();
                
                System.out.println("最优解: " + solution);
                System.out.println("目标函数值: " + (-objValue)); // 取负数因为我们求的是最大化
                
                // 验证约束
                System.out.println("\n约束验证:");
                for (int i = 0; i < A_ub.rows(); i++) {
                    double lhs = 0.0;
                    for (int j = 0; j < A_ub.cols(); j++) {
                        lhs += A_ub.get(i, j).doubleValue() * solution.get(j).doubleValue();
                    }
                    System.out.println("约束 " + (i+1) + ": " + lhs + " <= " + b_ub.get(i).doubleValue() + 
                                     " (" + (lhs <= b_ub.get(i).doubleValue() ? "满足" : "违反") + ")");
                }
                
                // 验证整数性
                System.out.println("\n整数性验证:");
                for (int i = 0; i < solution.length(); i++) {
                    double val = solution.get(i).doubleValue();
                    boolean isInteger = Math.abs(val - Math.round(val)) < 1e-6;
                    System.out.println("x" + (i+1) + " = " + val + " (" + (isInteger ? "整数" : "非整数") + ")");
                }
                
            } else {
                System.out.println("求解失败或无可行解");
            }
            
        } catch (Exception e) {
            System.out.println("求解过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试案例2：混合整数规划问题
     * 最大化: 2x1 + 3x2 + x3
     * 约束: x1 + x2 + x3 <= 6
     *       2x1 + x2 <= 8
     *       x1, x2 ∈ {0, 1, 2, ...} (整数)
     *       x3 >= 0 (连续变量)
     */
    public static void testMixedIntegerProgramming() {
        System.out.println("\n=== 测试案例2：混合整数规划问题 ===");
        
        try {
            // 目标函数系数 (最大化问题，所以取负数)
            IVector<Double> c = Linalg.vector(new double[]{-2.0, -3.0, -1.0});
            
            // 不等式约束 A_ub * x <= b_ub
            IMatrix<Double> A_ub = Linalg.matrix(new double[][]{
                {1.0, 1.0, 1.0},
                {2.0, 1.0, 0.0}
            });
            IVector<Double> b_ub = Linalg.vector(new double[]{6.0, 8.0});
            
            // 等式约束 (无)
            IMatrix<Double> A_eq = null;
            IVector<Double> b_eq = null;
            
            // 变量界限
            IVector<Double> bounds_lower = Linalg.vector(new double[]{0.0, 0.0, 0.0});
            IVector<Double> bounds_upper = Linalg.vector(new double[]{10.0, 10.0, 10.0});
            
            // 创建求解器
            RereIntegerProg solver = new RereIntegerProg();
            
            // 设置x1和x2为整数变量，x3保持连续
            solver.addIntegerVariables(0, 1);
            
            // 求解
            System.out.println("开始求解混合整数规划问题...");
            OptResult result = solver.solve(c, A_ub, b_ub, A_eq, b_eq);
            
            if (result != null) {
                System.out.println("求解成功！");
                IVector<Double> solution = result.getOptimalPoint();
                double objValue = result.getOptimalValue();
                
                System.out.println("最优解: " + solution);
                System.out.println("目标函数值: " + (-objValue)); // 取负数因为我们求的是最大化
                
                // 验证约束
                System.out.println("\n约束验证:");
                for (int i = 0; i < A_ub.rows(); i++) {
                    double lhs = 0.0;
                    for (int j = 0; j < A_ub.cols(); j++) {
                        lhs += A_ub.get(i, j).doubleValue() * solution.get(j).doubleValue();
                    }
                    System.out.println("约束 " + (i+1) + ": " + lhs + " <= " + b_ub.get(i).doubleValue() + 
                                     " (" + (lhs <= b_ub.get(i).doubleValue() ? "满足" : "违反") + ")");
                }
                
                // 验证整数性
                System.out.println("\n整数性验证:");
                for (int i = 0; i < solution.length(); i++) {
                    double val = solution.get(i).doubleValue();
                    boolean shouldBeInteger = (i == 0 || i == 1); // x1和x2应为整数
                    if (shouldBeInteger) {
                        boolean isInteger = Math.abs(val - Math.round(val)) < 1e-6;
                        System.out.println("x" + (i+1) + " = " + val + " (应为整数: " + (isInteger ? "是" : "否") + ")");
                    } else {
                        System.out.println("x" + (i+1) + " = " + val + " (连续变量)");
                    }
                }
                
            } else {
                System.out.println("求解失败或无可行解");
            }
            
        } catch (Exception e) {
            System.out.println("求解过程中发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}