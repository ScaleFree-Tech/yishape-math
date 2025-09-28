import com.reremouse.lab.math.linalg.*;
import com.reremouse.lab.math.optimize.linpg.RereIntegerProg;
import com.reremouse.lab.math.optimize.OptResult;

public class SimpleIntegerTest {
    public static void main(String[] args) {
        System.out.println("=== 简单整数规划测试 ===\n");
        
        // 测试一个非常简单的问题：
        // 最大化 x1 + x2
        // 约束：x1 + x2 <= 2.5
        //       x1, x2 >= 0
        //       x1, x2 是整数
        // 期望解：x1 = 1, x2 = 1, 目标值 = 2
        
        // 目标函数：最大化 x1 + x2 (转换为最小化 -x1 - x2)
        IVector<Double> c = Linalg.vector(new double[]{-1.0, -1.0});
        
        // 不等式约束：x1 + x2 <= 2.5
        IMatrix<Double> A_ub = Linalg.matrix(new double[][]{{1.0, 1.0}});
        IVector<Double> b_ub = Linalg.vector(new double[]{2.5});
        
        // 没有等式约束，使用null
        IMatrix<Double> A_eq = null;
        IVector<Double> b_eq = null;
        
        // 创建求解器
        RereIntegerProg solver = new RereIntegerProg();
        
        // 设置所有变量为整数
        solver.addIntegerVariables(0, 1);
        
        // 启用详细输出
        solver.setVerbose(true);
        
        // 设置较小的最大迭代次数以便观察
        solver.setMaxIterations(50);
        
        System.out.println("问题设置：");
        System.out.println("目标函数：最小化 -x1 - x2 (即最大化 x1 + x2)");
        System.out.println("约束：x1 + x2 <= 2.5");
        System.out.println("      x1, x2 >= 0 (非负约束)");
        System.out.println("      x1, x2 是整数");
        System.out.println("期望解：x1 = 1, x2 = 1, 目标值 = 2\n");
        
        // 求解
        System.out.println("开始求解...\n");
        OptResult result = solver.solve(c, A_ub, b_ub, A_eq, b_eq);
        
        if (result != null) {
            System.out.println("\n=== 求解结果 ===");
            IVector<Double> solution = result.getOptimalPoint();
            double objValue = result.getOptimalValue();
            
            System.out.println("最优解: " + solution);
            System.out.println("目标函数值: " + (-objValue)); // 取负数因为我们求的是最大化
            System.out.println("是否收敛: " + result.isConverged());
            System.out.println("收敛原因: " + result.getConvergenceReason());
            
            // 验证约束
            System.out.println("\n约束验证:");
            double lhs = solution.get(0).doubleValue() + solution.get(1).doubleValue();
            System.out.println("x1 + x2 = " + lhs + " <= 2.5 (" + (lhs <= 2.5 ? "满足" : "违反") + ")");
            
            // 验证整数性
            System.out.println("\n整数性验证:");
            for (int i = 0; i < solution.length(); i++) {
                double val = solution.get(i).doubleValue();
                boolean isInteger = Math.abs(val - Math.round(val)) < 1e-6;
                System.out.println("x" + (i+1) + " = " + val + " (整数: " + (isInteger ? "是" : "否") + ")");
            }
        } else {
            System.out.println("求解失败！");
        }
    }
}