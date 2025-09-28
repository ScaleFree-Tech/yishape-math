import com.reremouse.lab.math.linalg.*;
import com.reremouse.lab.math.optimize.linpg.*;
import com.reremouse.lab.math.optimize.OptResult;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import java.util.*;

public class DebugLPRelaxation {
    public static void main(String[] args) {
        System.out.println("=== LP松弛求解调试程序 ===");
        
        // 定义原始问题
        IVector c = Linalg.vector(new double[]{-2, -3, -1});
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {1, 2, 1},
            {2, 1, 3}
        });
        IVector b_ub = Linalg.vector(new double[]{4, 6});
        
        System.out.println("=== 原始问题 ===");
        System.out.println("目标函数向量 c: " + c);
        System.out.println("不等式约束矩阵 A_ub: " + A_ub);
        System.out.println("不等式约束向量 b_ub: " + b_ub);
        System.out.println();
        
        // 创建基础LP求解器
        InteriorPointLinProgSolver baseSolver = new InteriorPointLinProgSolver();
        
        System.out.println("=== 1. 不添加二进制约束的LP求解 ===");
        // 转换不等式约束为等式约束
        Tuple3<IVector, IMatrix, IVector> converted = LinProgUtil.convertUbEqToEqConstraits(c, A_ub, b_ub, null, null);
        IVector c_eq = converted.getFirst();
        IMatrix A_eq = converted.getSecond();
        IVector b_eq = converted.getThird();
        
        System.out.println("转换后的目标函数向量 c_eq: " + c_eq);
        System.out.println("转换后的等式约束矩阵 A_eq: " + A_eq);
        System.out.println("转换后的等式约束向量 b_eq: " + b_eq);
        
        OptResult result1 = baseSolver.solveWithNonNegativeEqualConstraints(c_eq, A_eq, b_eq, null);
        if (result1 != null) {
            System.out.println("求解成功");
            System.out.println("最优值: " + result1.getOptimalValue());
            System.out.println("最优解: " + result1.getOptimalPoint());
            System.out.println("解长度: " + result1.getOptimalPoint().length());
        } else {
            System.out.println("求解失败");
        }
        System.out.println();
        
        System.out.println("=== 2. 添加二进制约束的LP求解 ===");
        
        // 手动添加二进制约束
        Map<Integer, Tuple2<Double, Double>> bounds = new HashMap<>();
        bounds.put(0, new Tuple2<>(0.0, 1.0));
        bounds.put(1, new Tuple2<>(0.0, 1.0));
        bounds.put(2, new Tuple2<>(0.0, 1.0));
        
        System.out.println("二进制约束: " + bounds);
        System.out.println("原始变量数量: " + c.length());
        
        try {
            // 构建变量界限约束
            List<double[]> ubConstraints = new ArrayList<>();
            List<Double> ubValues = new ArrayList<>();
            
            for (Map.Entry<Integer, Tuple2<Double, Double>> entry : bounds.entrySet()) {
                int varIndex = entry.getKey();
                double lowerBound = entry.getValue().getFirst();
                double upperBound = entry.getValue().getSecond();
                
                System.out.println("变量 " + varIndex + ": [" + lowerBound + ", " + upperBound + "]");
                
                // 添加下界约束：-x_i <= -lowerBound
                if (!Double.isInfinite(lowerBound)) {
                    double[] constraint = new double[c.length()];
                    constraint[varIndex] = -1.0;
                    ubConstraints.add(constraint);
                    ubValues.add(-lowerBound);
                    System.out.println("  下界约束: " + Arrays.toString(constraint) + " <= " + (-lowerBound));
                }
                
                // 添加上界约束：x_i <= upperBound
                if (!Double.isInfinite(upperBound)) {
                    double[] constraint = new double[c.length()];
                    constraint[varIndex] = 1.0;
                    ubConstraints.add(constraint);
                    ubValues.add(upperBound);
                    System.out.println("  上界约束: " + Arrays.toString(constraint) + " <= " + upperBound);
                }
            }
            
            // 合并原有的不等式约束和新的变量界限约束
            List<double[]> allUbConstraints = new ArrayList<>();
            List<Double> allUbValues = new ArrayList<>();
            
            // 添加原有的不等式约束
             for (int i = 0; i < A_ub.rows(); i++) {
                 double[] row = new double[A_ub.cols()];
                 for (int j = 0; j < A_ub.cols(); j++) {
                     row[j] = A_ub.get(i, j).doubleValue();
                 }
                 allUbConstraints.add(row);
                 allUbValues.add(b_ub.get(i).doubleValue());
             }
            
            // 添加变量界限约束
            allUbConstraints.addAll(ubConstraints);
            allUbValues.addAll(ubValues);
            
            System.out.println("合并后的不等式约束数量: " + allUbConstraints.size());
            
            // 转换为矩阵
            IMatrix newA_ub = Linalg.matrix(allUbConstraints.toArray(new double[0][]));
            IVector newB_ub = Linalg.vector(allUbValues.stream().mapToDouble(Double::doubleValue).toArray());
            
            System.out.println("新的不等式约束矩阵 A_ub: " + newA_ub);
            System.out.println("新的不等式约束向量 b_ub: " + newB_ub);
            
            // 转换为等式约束
            Tuple3<IVector, IMatrix, IVector> newConverted = LinProgUtil.convertUbEqToEqConstraits(c, newA_ub, newB_ub, null, null);
            IVector newC_eq = newConverted.getFirst();
            IMatrix newA_eq = newConverted.getSecond();
            IVector newB_eq = newConverted.getThird();
            
            System.out.println("添加变量界限后的等式约束矩阵 A_eq: " + newA_eq);
            System.out.println("添加变量界限后的等式约束向量 b_eq: " + newB_eq);
            
            // 求解
            OptResult result2 = baseSolver.solveWithNonNegativeEqualConstraints(newC_eq, newA_eq, newB_eq, null);
            if (result2 != null) {
                System.out.println("求解成功");
                System.out.println("最优值: " + result2.getOptimalValue());
                System.out.println("最优解: " + result2.getOptimalPoint());
                System.out.println("解长度: " + result2.getOptimalPoint().length());
                
                // 提取原始变量的解
                IVector originalSolution = result2.getOptimalPoint().slice(0, c.length());
                System.out.println("原始变量解: " + originalSolution);
            } else {
                System.out.println("求解失败");
            }
            
        } catch (Exception e) {
            System.out.println("添加二进制约束时发生错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
}