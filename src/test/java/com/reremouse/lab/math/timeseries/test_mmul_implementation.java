package com.reremouse.lab.math.timeseries;
import com.reremouse.lab.math.linalg.*;

/**
 * 测试矩阵与向量乘法实现
 * Test matrix-vector multiplication implementation
 */
public class test_mmul_implementation {
    public static void main(String[] args) {
        // 测试 Double 类型
        System.out.println("=== 测试 Double 类型矩阵与向量乘法 ===");
        testDoubleMatrixVectorMultiplication();
        
        // 测试 Float 类型
        System.out.println("\n=== 测试 Float 类型矩阵与向量乘法 ===");
        testFloatMatrixVectorMultiplication();
    }
    
    public static void testDoubleMatrixVectorMultiplication() {
        // 创建测试矩阵 2x3
        double[][] matrixData = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        RereDoubleMatrix matrix = new RereDoubleMatrix(matrixData);
        
        // 创建测试向量 3x1
        double[] vectorData = {1.0, 2.0, 3.0};
        IVector<Double> vector = IDoubleVector.of(vectorData);
        
        // 执行矩阵与向量乘法
        IVector<Double> result = matrix.mmul(vector);
        
        // 预期结果: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3] = [14, 32]
        System.out.println("矩阵: " + java.util.Arrays.deepToString(matrixData));
        System.out.println("向量: " + java.util.Arrays.toString(vectorData));
        System.out.println("结果: " + java.util.Arrays.toString(result.toDoubleArray()));
        System.out.println("预期: [14.0, 32.0]");
        
        // 验证结果
        double[] expected = {14.0, 32.0};
        double[] actual = result.toDoubleArray();
        boolean isCorrect = java.util.Arrays.equals(expected, actual);
        System.out.println("测试结果: " + (isCorrect ? "通过" : "失败"));
    }
    
    public static void testFloatMatrixVectorMultiplication() {
        // 创建测试矩阵 3x2
        float[][] matrixData = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
        RereFloatMatrix matrix = new RereFloatMatrix(matrixData);
        
        // 创建测试向量 2x1
        float[] vectorData = {2.0f, 3.0f};
        IVector<Float> vector = IFloatVector.of(vectorData);
        
        // 执行矩阵与向量乘法
        IVector<Float> result = matrix.mmul(vector);
        
        // 预期结果: [1*2 + 2*3, 3*2 + 4*3, 5*2 + 6*3] = [8, 18, 28]
        System.out.println("矩阵: " + java.util.Arrays.deepToString(matrixData));
        System.out.println("向量: " + java.util.Arrays.toString(vectorData));
        System.out.println("结果: " + java.util.Arrays.toString(result.toFloatArray()));
        System.out.println("预期: [8.0, 18.0, 28.0]");
        
        // 验证结果
        float[] expected = {8.0f, 18.0f, 28.0f};
        float[] actual = result.toFloatArray();
        boolean isCorrect = java.util.Arrays.equals(expected, actual);
        System.out.println("测试结果: " + (isCorrect ? "通过" : "失败"));
    }
}
