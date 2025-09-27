import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.optimize.linpg.SimplexLinProgSolver;
import com.reremouse.lab.util.Tuple2;

public class SimpleTest {
    public static void main(String[] args) {
        // 测试问题：
        // 最大化 x1 + 2*x2
        // 约束：x1 + x2 <= 2.5
        // x1, x2 >= 0
        
        // 转换为标准形式：
        // 最小化 -x1 - 2*x2
        // 约束：x1 + x2 = 2.5 (添加松弛变量)
        
        IVector c = IVector.of(-1.0, -2.0);  // 目标函数系数
        IMatrix A_eq = IMatrix.of(new double[][]{{1.0, 1.0}});  // 约束矩阵
        IVector b_eq = IVector.of(2.5);  // 约束右端
        
        SimplexLinProgSolver solver = new SimplexLinProgSolver();
        
        try {
            var result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
            
            System.out.println("求解结果:");
            System.out.println("目标函数值: " + result.getOptimalValue());
            System.out.println("最优解: " + result.getOptimalPoint());
            
            // 验证解
            IVector solution = result.getOptimalPoint();
            double x1 = (Double) solution.get(0);
            double x2 = (Double) solution.get(1);
            
            System.out.println("x1 = " + x1);
            System.out.println("x2 = " + x2);
            System.out.println("约束验证: x1 + x2 = " + (x1 + x2) + " (应该等于2.5)");
            System.out.println("原始目标函数值: " + (x1 + 2*x2) + " (应该等于5.0)");
            
        } catch (Exception e) {
            System.err.println("求解失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
}