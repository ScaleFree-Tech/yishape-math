package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.OptResult;

/**
 * 测试边界约束处理
 */
public class TestBounds {
    public static void main(String[] args) {
        // 创建背包问题数据
        double[] values = {60.0, 100.0, 120.0, 80.0, 150.0, 200.0, 50.0};
        double[] weights = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0};
        double capacity = 100.0;
        
        // 转换为最小化问题
        IVector c = Linalg.vector(values).multiplyScalar(-1.0);
        IMatrix A_ub = Linalg.matrix(new double[][]{weights});
        IVector b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("=== 测试添加边界约束 ===");
        
        // 手动添加0-1边界约束
        // 对于每个变量xi，添加约束 0 <= xi <= 1
        int n = c.length();
        double[][] boundsMatrix = new double[n*2][n];
        double[] boundsValues = new double[n*2];
        
        for (int i = 0; i < n; i++) {
            // xi <= 1
            boundsMatrix[i][i] = 1.0;
            boundsValues[i] = 1.0;
            
            // -xi <= 0 (即 xi >= 0)
            boundsMatrix[i+n][i] = -1.0;
            boundsValues[i+n] = 0.0;
        }
        
        IMatrix A_bounds = Linalg.matrix(boundsMatrix);
        IVector b_bounds = Linalg.vector(boundsValues);
        
        System.out.println("原不等式约束数: " + A_ub.rows());
        System.out.println("边界约束数: " + A_bounds.rows());
        
        // 合并约束
        IMatrix A_combined = A_ub.vstack(A_bounds);
        IVector b_combined = b_ub.concat(b_bounds);
        
        System.out.println("合并后约束数: " + A_combined.rows());
        
        System.out.println("\n=== 使用BetterSimplexLinProgSolver求解 ===");
        try {
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            
            OptResult result = solver.solve(c, A_combined, b_combined);
            
            if (result != null) {
                System.out.println("结果:");
                System.out.println("  目标值: " + result.getOptimalValue());
                System.out.println("  解: " + result.getOptimalPoint());
                System.out.println("  转换后的最大化值: " + (-result.getOptimalValue()));
                
                // 验证解的有效性
                IVector solution = result.getOptimalPoint();
                if (solution != null) {
                    // 检查是否满足重量约束
                    double totalWeight = 0;
                    double totalValue = 0;
                    boolean isValid = true;
                    
                    for (int i = 0; i < solution.size(); i++) {
                        double value = solution.get(i).doubleValue();
                        // 检查0-1约束
                        if (value < -1e-6 || value > 1 + 1e-6) {
                            System.out.println("  变量x" + (i+1) + " = " + value + " 违反0-1约束");
                            isValid = false;
                        }
                        
                        totalWeight += value * weights[i];
                        totalValue += value * values[i];
                    }
                    
                    System.out.println("  总重量: " + totalWeight + " (容量: " + capacity + ")");
                    System.out.println("  总价值: " + totalValue);
                    System.out.println("  是否满足重量约束: " + (totalWeight <= capacity + 1e-6));
                    System.out.println("  解是否有效: " + isValid);
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