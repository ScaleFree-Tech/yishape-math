package com.yishape.lab.math.linalg;

import com.yishape.lab.util.Tuple2;

/**
 * 演示和测试Linalg.lstsq方法的正确性
 */
public class LstsqDemo {

    public static void main(String[] args) {
        System.out.println("=== 测试Linalg.lstsq方法 ===");
        
        // 创建一个简单的超定系统进行测试
        // Test with a simple overdetermined system
        // Ax = b where A is 3x2 and b is 3x1
        // We want to solve for x which is 2x1
        
        // 系数矩阵 A (3x2)
        // Coefficient matrix A (3x2)
        double[][] aData = {
            {1.0, 1.0},
            {1.0, 2.0},
            {1.0, 3.0}
        };
        IMatrix<Double> A = Linalg.matrix(aData);
        
        // 常数向量 b (3x1)
        // Constant vector b (3x1)
        double[] bData = {1.0, 2.0, 3.0};
        IVector<Double> b = Linalg.vector(bData);
        
        System.out.println("系数矩阵 A:");
        System.out.println(A);
        
        System.out.println("常数向量 b:");
        System.out.println(b);
        
        // 调用lstsq方法求解
        Tuple2<IVector<Double>, Double> result = Linalg.lstsq(A, b);
        IVector<Double> solution = result.getFirst();
        Double residual = result.getSecond();
        
        System.out.println("最小二乘解:");
        System.out.println(solution);
        System.out.println("残差:");
        System.out.println(residual);

    }

}