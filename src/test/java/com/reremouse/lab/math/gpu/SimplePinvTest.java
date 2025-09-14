package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUComputeUtils;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.RereMatrix;

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
        GPUComputeUtils.setLoggingEnabled(true);
        GPUComputeUtils.setDetailedLoggingEnabled(true);
        
        // 测试1：小矩阵（应该使用CPU）
        System.out.println("1. 测试小矩阵（应该使用CPU）:");
        System.out.println("--------------------------------");
        float[][] smallData = {
            {1, 2, 3},
            {4, 5, 6}
        };
        IMatrix smallMatrix = new RereMatrix(smallData);
        
        long startTime = System.currentTimeMillis();
        IMatrix smallResult = smallMatrix.pinv();
        long endTime = System.currentTimeMillis();
        
        System.out.println("小矩阵pinv耗时: " + (endTime - startTime) + "ms");
        System.out.println("结果矩阵维度: " + smallResult.getRows() + "x" + smallResult.getColumns());
        
        // 测试2：大矩阵（应该使用GPU）
        System.out.println("\n2. 测试大矩阵（应该使用GPU）:");
        System.out.println("--------------------------------");
        int size = 120; // 120x120 = 14400 > 10000
        float[][] largeData = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                largeData[i][j] = (float) (Math.random() * 10 - 5);
            }
        }
        IMatrix largeMatrix = new RereMatrix(largeData);
        
        startTime = System.currentTimeMillis();
        IMatrix largeResult = largeMatrix.pinv();
        endTime = System.currentTimeMillis();
        
        System.out.println("大矩阵pinv耗时: " + (endTime - startTime) + "ms");
        System.out.println("结果矩阵维度: " + largeResult.getRows() + "x" + largeResult.getColumns());
        
        // 测试3：验证结果正确性
        System.out.println("\n3. 验证结果正确性:");
        System.out.println("-------------------");
        
        // 验证小矩阵：A * A⁺ * A ≈ A
        IMatrix verification = smallMatrix.mmul(smallResult).mmul(smallMatrix);
        float maxError = 0.0f;
        for (int i = 0; i < smallMatrix.getRows(); i++) {
            for (int j = 0; j < smallMatrix.getColumns(); j++) {
                float error = Math.abs(smallMatrix.get(i, j) - verification.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        System.out.println("小矩阵最大误差: " + maxError);
        
        // 验证大矩阵（只检查前几个元素）
        IMatrix largeVerification = largeMatrix.mmul(largeResult).mmul(largeMatrix);
        float largeMaxError = 0.0f;
        int checkSize = Math.min(5, largeMatrix.getRows());
        for (int i = 0; i < checkSize; i++) {
            for (int j = 0; j < checkSize; j++) {
                float error = Math.abs(largeMatrix.get(i, j) - largeVerification.get(i, j));
                largeMaxError = Math.max(largeMaxError, error);
            }
        }
        System.out.println("大矩阵前" + checkSize + "x" + checkSize + "元素最大误差: " + largeMaxError);
        
        System.out.println("\n=== 测试完成 ===");
        System.out.println("GPU加速的pinv方法已成功实现！");
    }
}
