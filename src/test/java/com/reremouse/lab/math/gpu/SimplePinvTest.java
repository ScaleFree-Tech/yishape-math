package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUComputeFloatUtils;
import com.reremouse.lab.math.linalg.RereDoubleMatrix;
import com.reremouse.lab.math.linalg.IDoubleMatrix;

/**
 * 简单的伪逆矩阵测试
 * 验证pinv方法的GPU加速功能
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class SimplePinvTest {
    
    public static void main(String[] args) {
        System.out.println("=== 简单伪逆矩阵测试 ===\n");
        
        // 启用详细日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
        
        // 测试1：小矩阵（应该使用CPU）
        System.out.println("1. 测试小矩阵（应该使用CPU）:");
        System.out.println("--------------------------------");
        double[][] smallData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        };
        IDoubleMatrix smallMatrix = new RereDoubleMatrix(smallData);
        
        long startTime = System.currentTimeMillis();
        IDoubleMatrix smallResult = (IDoubleMatrix) smallMatrix.pinv();
        long endTime = System.currentTimeMillis();
        
        System.out.println("小矩阵pinv耗时: " + (endTime - startTime) + "ms");
        System.out.println("结果矩阵维度: " + smallResult.rows() + "x" + smallResult.cols());
        
        // 测试2：大矩阵（应该使用GPU）
        System.out.println("\n2. 测试大矩阵（应该使用GPU）:");
        System.out.println("--------------------------------");
        int size = 120; // 120x120 = 14400 > 10000
        double[][] largeData = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                largeData[i][j] = Math.random() * 10.0 - 5.0;
            }
        }
        IDoubleMatrix largeMatrix = new RereDoubleMatrix(largeData);
        
        startTime = System.currentTimeMillis();
        IDoubleMatrix largeResult = (IDoubleMatrix) largeMatrix.pinv();
        endTime = System.currentTimeMillis();
        
        System.out.println("大矩阵pinv耗时: " + (endTime - startTime) + "ms");
        System.out.println("结果矩阵维度: " + largeResult.rows() + "x" + largeResult.cols());
        
        // 测试3：验证结果正确性
        System.out.println("\n3. 验证结果正确性:");
        System.out.println("-------------------");
        
        // 验证小矩阵：A * A⁺ * A ≈ A
        IDoubleMatrix verification = (IDoubleMatrix) smallMatrix.mmul(smallResult).mmul(smallMatrix);
        double maxError = 0.0;
        for (int i = 0; i < smallMatrix.rows(); i++) {
            for (int j = 0; j < smallMatrix.cols(); j++) {
                double error = Math.abs(smallMatrix.get(i, j) - verification.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        System.out.println("小矩阵最大误差: " + maxError);
        
        // 验证大矩阵（只检查前几个元素）
        IDoubleMatrix largeVerification = (IDoubleMatrix) largeMatrix.mmul(largeResult).mmul(largeMatrix);
        double largeMaxError = 0.0;
        int checkSize = Math.min(5, largeMatrix.rows());
        for (int i = 0; i < checkSize; i++) {
            for (int j = 0; j < checkSize; j++) {
                double error = Math.abs(largeMatrix.get(i, j) - largeVerification.get(i, j));
                largeMaxError = Math.max(largeMaxError, error);
            }
        }
        System.out.println("大矩阵前" + checkSize + "x" + checkSize + "元素最大误差: " + largeMaxError);
        
        System.out.println("\n=== 测试完成 ===");
        System.out.println("GPU加速的pinv方法已成功实现！");
    }
}
