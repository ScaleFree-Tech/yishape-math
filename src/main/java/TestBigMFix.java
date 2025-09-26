import com.reremouse.lab.math.linalg.*;
import com.reremouse.lab.math.optimize.linpg.SimplexLinProgSolver;
import com.reremouse.lab.util.Tuple2;

public class TestBigMFix {
    public static void main(String[] args) {
        // 问题定义
        IVector c = IVector.of(new double[]{1, 2, 3});
        IMatrix A_eq = IMatrix.of(new double[][]{
            {1, 1, 1},
            {2, 1, 0}
        });
        IVector b_eq = IVector.of(new double[]{3, 2});
        
        System.out.println("=== 测试大M法修复 ===");
        System.out.println("目标函数: min x0 + 2*x1 + 3*x2");
        System.out.println("约束条件:");
        System.out.println("  x0 + x1 + x2 = 3");
        System.out.println("  2*x0 + x1 = 2");
        System.out.println("  x0, x1, x2 >= 0");
        
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        try {
            Tuple2<Double, IVector> result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            
            System.out.println("\n=== 求解结果 ===");
            System.out.println("最优值: " + result.getFirst());
            IVector solution = result.getSecond();
            System.out.println("最优解: [" + solution.get(0) + ", " + solution.get(1) + ", " + solution.get(2) + "]");
            
            // 验证约束
            double x0 = (Double) solution.get(0);
            double x1 = (Double) solution.get(1);
            double x2 = (Double) solution.get(2);
            
            System.out.println("\n=== 约束验证 ===");
            double constraint1 = x0 + x1 + x2;
            double constraint2 = 2*x0 + x1;
            
            System.out.printf("约束1: %.6f = 3 (差值: %.6f)\n", constraint1, Math.abs(constraint1 - 3));
            System.out.printf("约束2: %.6f = 2 (差值: %.6f)\n", constraint2, Math.abs(constraint2 - 2));
            
            // 验证目标函数值
            double objValue = x0 + 2*x1 + 3*x2;
            System.out.printf("目标函数值: %.6f (报告值: %.6f, 差值: %.6f)\n", 
                objValue, result.getFirst(), Math.abs(objValue - result.getFirst()));
            
        } catch (Exception e) {
            System.out.println("求解失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}