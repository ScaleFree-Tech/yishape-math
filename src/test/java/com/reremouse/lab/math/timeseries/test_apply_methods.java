package com.reremouse.lab.math.timeseries;
import com.reremouse.lab.math.linalg.RereDoubleMatrix;
import com.reremouse.lab.math.linalg.RereDoubleVector;
import com.reremouse.lab.math.linalg.IDoubleMatrix;
import com.reremouse.lab.math.linalg.IDoubleVector;
import java.util.function.Function;

public class test_apply_methods {
    public static void main(String[] args) {
        // 测试RereDoubleVector的apply方法
        System.out.println("测试RereDoubleVector的apply方法:");
        double[] vectorData = {1.0, 2.0, 3.0, 4.0};
        RereDoubleVector vector = new RereDoubleVector(vectorData);
        
        // 测试平方函数
        IDoubleVector squared = (IDoubleVector) vector.apply(x -> x * x);
        System.out.println("原向量: " + java.util.Arrays.toString(vector.getData()));
        System.out.println("平方后: " + java.util.Arrays.toString(squared.getData()));
        
        // 测试绝对值函数
        double[] vectorData2 = {-1.0, -2.0, 3.0, -4.0};
        RereDoubleVector vector2 = new RereDoubleVector(vectorData2);
        IDoubleVector abs = (IDoubleVector) vector2.apply(Math::abs);
        System.out.println("原向量: " + java.util.Arrays.toString(vector2.getData()));
        System.out.println("绝对值: " + java.util.Arrays.toString(abs.getData()));
        
        System.out.println();
        
        // 测试RereDoubleMatrix的apply方法
        System.out.println("测试RereDoubleMatrix的apply方法:");
        double[][] matrixData = {{1.0, 2.0}, {3.0, 4.0}};
        RereDoubleMatrix matrix = new RereDoubleMatrix(matrixData);
        
        // 测试平方函数
        IDoubleMatrix squaredMatrix = (IDoubleMatrix) matrix.apply(x -> x * x);
        System.out.println("原矩阵:");
        printMatrix(matrix.getData());
        System.out.println("平方后:");
        printMatrix(squaredMatrix.getData());
        
        // 测试指数函数
        IDoubleMatrix expMatrix = (IDoubleMatrix) matrix.apply(Math::exp);
        System.out.println("指数函数:");
        printMatrix(expMatrix.getData());
        
        System.out.println("所有测试完成！");
    }
    
    private static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            System.out.println(java.util.Arrays.toString(row));
        }
    }
}
