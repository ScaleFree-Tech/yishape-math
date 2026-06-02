package com.yishape.lab.math.timeseries;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IFloatMatrix;
import com.yishape.lab.math.linalg.IMatrix;

public class test_negative_index {
    public static void main(String[] args) {
        // 测试RereDoubleMatrix的负数索引功能
        System.out.println("=== 测试RereDoubleMatrix负数索引功能 ===");
        
        // 创建一个4x4的测试矩阵
        double[][] testData = {
            {1, 2, 3, 4},
            {5, 6, 7, 8},
            {9, 10, 11, 12},
            {13, 14, 15, 16}
        };
        
        IDoubleMatrix matrix = IDoubleMatrix.of(testData);
        System.out.println("原始矩阵:");
        System.out.println(matrix);
        
        // 测试subMatrix方法 - 使用负数索引获取最后2行最后2列
        System.out.println("\n使用负数索引 subMatrix(-2, -1, -2, -1):");
        IMatrix<Double> subMatrix = matrix.subMatrix(-2, -1, -2, -1);
        System.out.println(subMatrix);
        
        // 测试setSubMatrix方法 - 使用负数索引设置子矩阵
        System.out.println("\n使用负数索引 setSubMatrix(-2, -1, -2, -1, newMatrix):");
        IDoubleMatrix newSubMatrix = IDoubleMatrix.of(new double[][]{{99, 88}, {77, 66}});
        matrix.setSubMatrix(-2, -1, -2, -1, newSubMatrix);
        System.out.println("修改后的矩阵:");
        System.out.println(matrix);
        
        // 测试RereFloatMatrix的负数索引功能
        System.out.println("\n=== 测试RereFloatMatrix负数索引功能 ===");
        
        float[][] testDataFloat = {
            {1.1f, 2.2f, 3.3f, 4.4f},
            {5.5f, 6.6f, 7.7f, 8.8f},
            {9.9f, 10.1f, 11.2f, 12.3f},
            {13.4f, 14.5f, 15.6f, 16.7f}
        };
        
        IFloatMatrix matrixFloat = IFloatMatrix.of(testDataFloat);
        System.out.println("原始Float矩阵:");
        System.out.println(matrixFloat);
        
        // 测试subMatrix方法 - 使用负数索引获取最后2行最后2列
        System.out.println("\n使用负数索引 subMatrix(-2, -1, -2, -1):");
        IMatrix<Float> subMatrixFloat = matrixFloat.subMatrix(-2, -1, -2, -1);
        System.out.println(subMatrixFloat);
        
        // 测试setSubMatrix方法 - 使用负数索引设置子矩阵
        System.out.println("\n使用负数索引 setSubMatrix(-2, -1, -2, -1, newMatrix):");
        IFloatMatrix newSubMatrixFloat = IFloatMatrix.of(new float[][]{{99.9f, 88.8f}, {77.7f, 66.6f}});
        matrixFloat.setSubMatrix(-2, -1, -2, -1, newSubMatrixFloat);
        System.out.println("修改后的Float矩阵:");
        System.out.println(matrixFloat);
        
        System.out.println("\n=== 测试完成 ===");
    }
}
