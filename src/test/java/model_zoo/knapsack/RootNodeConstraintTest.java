package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.ComMath4LinProgSolver;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.linpg.LinProgUtil;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.util.Tuple3;

/**
 * 测试RereIntegerProg根节点约束设置
 */
public class RootNodeConstraintTest {
    
    public static void main(String[] args) {
        System.out.println("🔍 根节点约束测试");
        System.out.println("========================================");
        
        // 创建一个简单的测试问题
        double[] values = {60.0, 100.0};  // 只用两个变量简化问题
        double[] weights = {10.0, 20.0};
        double capacity = 100.0;
        
        var c = Linalg.vector(values).multiplyScalar(-1.0);  // 最小化 -60x1 - 100x2
        var A_ub = Linalg.matrix(new double[][]{weights});   // 10x1 + 20x2 <= 100
        var b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("📋 简化测试问题:");
        System.out.println("目标函数c: " + c);
        System.out.println("约束矩阵A_ub: " + displayMatrix(A_ub));
        System.out.println("约束向量b_ub: " + b_ub);
        System.out.println();
        
        // 转换为等式约束
        Tuple3<IVector, IMatrix, IVector> converted = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        IVector c_eq = converted.getFirst();
        IMatrix A_eq = converted.getSecond(); 
        IVector b_eq = converted.getThird();
        
        System.out.println("🔄 转换后:");
        System.out.println("c_eq: " + c_eq);
        System.out.println("A_eq: " + displayMatrix(A_eq));
        System.out.println("b_eq: " + b_eq);
        System.out.println();
        
        // 测试1: 无约束的线性松弛
        System.out.println("🧪 测试1: 无约束线性松弛 (BetterSimplexLinProgSolver)");
        testUnconstrainedLP(c_eq, A_eq, b_eq, new RereSimplexLinProgSolver());
        
        System.out.println("🧪 测试1: 无约束线性松弛 (ComMath4LinProgSolver)");
        testUnconstrainedLP(c_eq, A_eq, b_eq, new ComMath4LinProgSolver());
        
        // 测试2: 手动添加0-1约束的线性松弛
        System.out.println("🧪 测试2: 手动添加0-1约束的线性松弛");
        testConstrainedLP(c, A_ub, b_ub);
        
        // 测试3: 使用RereIntegerProg的详细分析
        System.out.println("🧪 测试3: RereIntegerProg根节点分析");
        testRereIntegerProgRoot(c, A_ub, b_ub);
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
    
    private static void testUnconstrainedLP(IVector c_eq, IMatrix A_eq, IVector b_eq, ILinProgSolver solver) {
        try {
            System.out.println("  求解器: " + solver.getClass().getSimpleName());
            OptResult result = solver.solveWithNonNegativeEqualConstraints(c_eq, A_eq, b_eq, null);
            
            if (result != null && result.isConverged()) {
                System.out.println("  ✅ 求解成功");
                System.out.println("  目标函数值: " + result.getOptimalValue());
                System.out.println("  解向量: " + result.getOptimalPoint());
                
                IVector solution = result.getOptimalPoint();
                if (solution.length() >= 2) {
                    System.out.printf("  x1 = %.6f, x2 = %.6f\n", 
                        solution.get(0).doubleValue(), solution.get(1).doubleValue());
                }
            } else {
                System.out.println("  ❌ 求解失败");
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
        }
        System.out.println();
    }
    
    private static void testConstrainedLP(IVector c, IMatrix A_ub, IVector b_ub) {
        try {
            // 手动添加0-1约束：0 <= x1 <= 1, 0 <= x2 <= 1
            // 构建约束矩阵，包含原始约束和0-1约束
            double[][] constraintMatrix = {
                {10.0, 20.0},  // 原始约束: 10x1 + 20x2 <= 100
                {1.0, 0.0},    // x1 <= 1
                {0.0, 1.0},    // x2 <= 1
                {-1.0, 0.0},   // -x1 <= 0 即 x1 >= 0  
                {0.0, -1.0}    // -x2 <= 0 即 x2 >= 0
            };
            
            double[] constraintValues = {100.0, 1.0, 1.0, 0.0, 0.0};
            
            var A_ub_extended = Linalg.matrix(constraintMatrix);
            var b_ub_extended = Linalg.vector(constraintValues);
            
            System.out.println("  扩展约束矩阵A_ub: " + displayMatrix(A_ub_extended));
            System.out.println("  扩展约束向量b_ub: " + b_ub_extended);
            
            // 使用BetterSimplexLinProgSolver测试
            System.out.println("  使用BetterSimplexLinProgSolver:");
            RereSimplexLinProgSolver solver1 = new RereSimplexLinProgSolver();
            OptResult result1 = solver1.solve(c, A_ub_extended, b_ub_extended);
            
            if (result1 != null && result1.isConverged()) {
                System.out.println("    ✅ 求解成功");
                System.out.println("    目标函数值: " + result1.getOptimalValue());
                System.out.println("    解向量: " + result1.getOptimalPoint());
            } else {
                System.out.println("    ❌ 求解失败");
            }
            
            // 使用ComMath4LinProgSolver测试  
            System.out.println("  使用ComMath4LinProgSolver:");
            ComMath4LinProgSolver solver2 = new ComMath4LinProgSolver();
            OptResult result2 = solver2.solve(c, A_ub_extended, b_ub_extended);
            
            if (result2 != null && result2.isConverged()) {
                System.out.println("    ✅ 求解成功");
                System.out.println("    目标函数值: " + result2.getOptimalValue());
                System.out.println("    解向量: " + result2.getOptimalPoint());
            } else {
                System.out.println("    ❌ 求解失败");
            }
            
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
        System.out.println();
    }
    
    private static void testRereIntegerProgRoot(IVector c, IMatrix A_ub, IVector b_ub) {
        try {
            // 创建RereIntegerProg实例进行调试
            System.out.println("  使用BetterSimplexLinProgSolver作为基求解器:");
            RereIntegerProg solver1 = new RereIntegerProg(new RereSimplexLinProgSolver());
            solver1.setAllVariablesBinary();
            solver1.setVerbose(true);
            solver1.setMaxIterations(10);  // 限制迭代次数便于观察
            
            OptResult result1 = solver1.solve(c, A_ub, b_ub);
            
            if (result1 != null && result1.getOptimalPoint() != null) {
                System.out.println("    ✅ 求解完成");
                System.out.println("    目标函数值: " + result1.getOptimalValue());
                System.out.println("    解向量: " + result1.getOptimalPoint());
            } else {
                System.out.println("    ❌ 求解失败");
            }
            
            System.out.println();
            
            System.out.println("  使用ComMath4LinProgSolver作为基求解器:");
            RereIntegerProg solver2 = new RereIntegerProg(new ComMath4LinProgSolver());
            solver2.setAllVariablesBinary();
            solver2.setVerbose(true);
            solver2.setMaxIterations(10);
            
            OptResult result2 = solver2.solve(c, A_ub, b_ub);
            
            if (result2 != null && result2.getOptimalPoint() != null) {
                System.out.println("    ✅ 求解完成");
                System.out.println("    目标函数值: " + result2.getOptimalValue());
                System.out.println("    解向量: " + result2.getOptimalPoint());
            } else {
                System.out.println("    ❌ 求解失败");
            }
            
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
    }
}