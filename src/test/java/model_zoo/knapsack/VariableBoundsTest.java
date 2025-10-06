package model_zoo.knapsack;

import com.yishape.lab.math.optimize.linpg.LinProgUtil;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import java.util.HashMap;
import java.util.Map;

/**
 * 测试变量边界约束的添加和求解
 */
public class VariableBoundsTest {
    
    public static void main(String[] args) {
        System.out.println("🔍 变量边界约束测试");
        System.out.println("========================================");
        
        // 简单的2变量背包问题
        double[] values = {60.0, 100.0};
        double[] weights = {10.0, 20.0};
        double capacity = 100.0;
        
        var c = Linalg.vector(values).multiplyScalar(-1.0);
        var A_ub = Linalg.matrix(new double[][]{weights});
        var b_ub = Linalg.vector(new double[]{capacity});
        
        System.out.println("📋 原始问题:");
        System.out.println("c: " + c);
        System.out.println("A_ub: " + displayMatrix(A_ub));
        System.out.println("b_ub: " + b_ub);
        System.out.println();
        
        // 转换为等式约束
        Tuple3<IVector, IMatrix, IVector> converted = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        IVector c_eq = converted.getFirst();
        IMatrix A_eq = converted.getSecond();
        IVector b_eq = converted.getThird();
        
        System.out.println("🔄 转换后的等式约束:");
        System.out.println("c_eq: " + c_eq);
        System.out.println("A_eq: " + displayMatrix(A_eq));
        System.out.println("b_eq: " + b_eq);
        System.out.println();
        
        // 测试1: 无边界约束
        System.out.println("🧪 测试1: 无边界约束");
        testNoBounds(c_eq, A_eq, b_eq);
        
        // 测试2: 添加0-1边界约束
        System.out.println("🧪 测试2: 添加0-1边界约束");
        testWithBounds(c_eq, A_eq, b_eq);
        
        // 测试3: 手动构建约束对比
        System.out.println("🧪 测试3: 手动构建约束对比");
        testManualConstraints(c, A_ub, b_ub);
    }
    
    private static String displayMatrix(IMatrix matrix) {
        if (matrix == null) return "null";
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
    
    private static void testNoBounds(IVector c_eq, IMatrix A_eq, IVector b_eq) {
        try {
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            OptResult result = solver.solveWithNonNegativeEqualConstraints(c_eq, A_eq, b_eq, null);
            
            if (result != null && result.isConverged()) {
                System.out.println("  ✅ 无边界约束求解成功");
                System.out.println("  目标值: " + result.getOptimalValue());
                System.out.println("  解: " + result.getOptimalPoint());
            } else {
                System.out.println("  ❌ 无边界约束求解失败");
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
        }
        System.out.println();
    }
    
    private static void testWithBounds(IVector c_eq, IMatrix A_eq, IVector b_eq) {
        try {
            // 创建变量边界：x1 ∈ [0,1], x2 ∈ [0,1]
            Map<Integer, Tuple2<Double, Double>> bounds = new HashMap<>();
            bounds.put(0, new Tuple2<>(0.0, 1.0));  // x1 bounds
            bounds.put(1, new Tuple2<>(0.0, 1.0));  // x2 bounds
            
            System.out.println("  添加边界约束: x1 ∈ [0,1], x2 ∈ [0,1]");
            
            // 模拟addVariableBounds方法的逻辑
            Tuple3<IVector, IMatrix, IVector> bounded = addVariableBounds(c_eq, A_eq, b_eq, bounds, 2);
            
            System.out.println("  边界约束添加后:");
            System.out.println("  c_bounded: " + bounded.getFirst());
            System.out.println("  A_bounded: " + displayMatrix(bounded.getSecond()));
            System.out.println("  b_bounded: " + bounded.getThird());
            
            // 求解带边界约束的问题
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            OptResult result = solver.solveWithNonNegativeEqualConstraints(
                bounded.getFirst(), bounded.getSecond(), bounded.getThird(), null);
            
            if (result != null && result.isConverged()) {
                System.out.println("  ✅ 边界约束求解成功");
                System.out.println("  目标值: " + result.getOptimalValue());
                System.out.println("  解: " + result.getOptimalPoint());
                
                // 检查原始变量是否满足边界
                IVector solution = result.getOptimalPoint();
                if (solution.length() >= 2) {
                    double x1 = solution.get(0).doubleValue();
                    double x2 = solution.get(1).doubleValue();
                    System.out.printf("  x1 = %.6f (在[0,1]内: %s)\n", x1, (x1 >= -1e-6 && x1 <= 1 + 1e-6));
                    System.out.printf("  x2 = %.6f (在[0,1]内: %s)\n", x2, (x2 >= -1e-6 && x2 <= 1 + 1e-6));
                }
            } else {
                System.out.println("  ❌ 边界约束求解失败");
                if (result != null) {
                    System.out.println("  失败原因: " + result.getConvergenceReason());
                }
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
        System.out.println();
    }
    
    private static void testManualConstraints(IVector c, IMatrix A_ub, IVector b_ub) {
        try {
            // 手动添加边界约束: 0 ≤ x1 ≤ 1, 0 ≤ x2 ≤ 1  
            // 现有约束: 10x1 + 20x2 ≤ 100
            // 添加约束: x1 ≤ 1, x2 ≤ 1, x1 ≥ 0, x2 ≥ 0 (后两个由非负约束自动满足)
            
            double[][] allConstraints = {
                {10.0, 20.0},  // 原始约束
                {1.0, 0.0},    // x1 ≤ 1
                {0.0, 1.0}     // x2 ≤ 1
            };
            double[] allBounds = {100.0, 1.0, 1.0};
            
            IMatrix A_ub_extended = Linalg.matrix(allConstraints);
            IVector b_ub_extended = Linalg.vector(allBounds);
            
            System.out.println("  手动构建的约束:");
            System.out.println("  A_ub_extended: " + displayMatrix(A_ub_extended));
            System.out.println("  b_ub_extended: " + b_ub_extended);
            
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            OptResult result = solver.solve(c, A_ub_extended, b_ub_extended);
            
            if (result != null && result.isConverged()) {
                System.out.println("  ✅ 手动约束求解成功");
                System.out.println("  目标值: " + result.getOptimalValue());
                System.out.println("  解: " + result.getOptimalPoint());
                
                IVector solution = result.getOptimalPoint();
                if (solution.length() >= 2) {
                    double x1 = solution.get(0).doubleValue();
                    double x2 = solution.get(1).doubleValue();
                    System.out.printf("  x1 = %.6f, x2 = %.6f\n", x1, x2);
                    System.out.printf("  目标值验证: %.6f (期望: %.6f)\n", 
                        -result.getOptimalValue(), 60*x1 + 100*x2);
                }
            } else {
                System.out.println("  ❌ 手动约束求解失败");
                if (result != null) {
                    System.out.println("  失败原因: " + result.getConvergenceReason());
                }
            }
        } catch (Exception e) {
            System.out.println("  ❌ 异常: " + e.getMessage());
            e.printStackTrace();
        }
        System.out.println();
    }
    
    // 简化版的addVariableBounds方法用于测试
    private static Tuple3<IVector, IMatrix, IVector> addVariableBounds(
            IVector c, IMatrix A_eq, IVector b_eq, 
            Map<Integer, Tuple2<Double, Double>> bounds, 
            int originalVariableCount) {
        
        if (bounds.isEmpty()) {
            return new Tuple3<>(c, A_eq, b_eq);
        }

        // 构建上界约束
        java.util.List<double[]> ubConstraints = new java.util.ArrayList<>();
        java.util.List<Double> ubValues = new java.util.ArrayList<>();

        for (Map.Entry<Integer, Tuple2<Double, Double>> entry : bounds.entrySet()) {
            int varIndex = entry.getKey();
            double lowerBound = entry.getValue().getFirst();
            double upperBound = entry.getValue().getSecond();

            if (varIndex >= originalVariableCount) {
                continue;
            }
            
            // 添加下界约束：-x_i <= -lowerBound
            if (!Double.isInfinite(lowerBound)) {
                double[] constraint = new double[c.length()];  // 使用完整的列数
                constraint[varIndex] = -1.0;
                ubConstraints.add(constraint);
                ubValues.add(-lowerBound);
            }

            // 添加上界约束：x_i <= upperBound  
            if (!Double.isInfinite(upperBound)) {
                double[] constraint = new double[c.length()];  // 使用完整的列数
                constraint[varIndex] = 1.0;
                ubConstraints.add(constraint);
                ubValues.add(upperBound);
            }
        }

        if (ubConstraints.isEmpty()) {
            return new Tuple3<>(c, A_eq, b_eq);
        }

        // 构建不等式约束矩阵
        double[][] ubMatrix = ubConstraints.toArray(new double[0][]);
        IMatrix A_ub = Linalg.matrix(ubMatrix);
        IVector b_ub = Linalg.vector(ubValues.stream().mapToDouble(Double::doubleValue).toArray());

        // 使用LinProgUtil转换为等式约束
        return LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, A_eq, b_eq);
    }
}