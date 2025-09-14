package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUComputeUtils;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.RereMatrix;
import com.reremouse.lab.math.linalg.RereVector;

/**
 * GPUComputeUtils简单测试类
 * 不使用JUnit，直接运行测试
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GPUComputeUtilsSimpleTest {
    
    public static void main(String[] args) {
        System.out.println("=== GPUComputeUtils 功能测试 ===");
        
        // 测试GPU可用性
        testGPUAvailability();
        
        // 测试GPU设备信息
        testGPUDeviceInfo();
        
        // 测试GPU矩阵乘法
        testGPUMatrixMultiply();
        
        // 测试GPU向量加法
        testGPUVectorAdd();
        
        // 测试GPU向量内积
        testGPUVectorDot();
        
        // 测试GPU资源清理
        testGPUCleanup();
        
        // 测试大矩阵性能
        testLargeMatrixPerformance();
        
        System.out.println("\n=== 所有测试完成 ===");
    }
    
    /**
     * 测试GPU可用性检查
     */
    private static void testGPUAvailability() {
        System.out.println("\n--- 测试GPU可用性检查 ---");
        
        boolean gpuAvailable = GPUComputeUtils.isGPUAvailable();
        System.out.println("GPU可用性: " + gpuAvailable);
        
        String gpuInfo = GPUComputeUtils.getGPUInfo();
        System.out.println("GPU信息: " + gpuInfo);
        
        if (gpuAvailable) {
            System.out.println("✓ GPU可用性检查通过");
        } else {
            System.out.println("⚠ GPU不可用，将使用CPU计算");
        }
    }
    
    /**
     * 测试GPU设备信息
     */
    private static void testGPUDeviceInfo() {
        System.out.println("\n--- 测试GPU设备信息 ---");
        
        String deviceInfo = GPUComputeUtils.getGPUDeviceInfo();
        System.out.println("设备信息:\n" + deviceInfo);
        
        System.out.println("✓ GPU设备信息检查完成");
    }
    
    /**
     * 测试GPU矩阵乘法
     */
    private static void testGPUMatrixMultiply() {
        System.out.println("\n--- 测试GPU矩阵乘法 ---");
        
        if (!GPUComputeUtils.isGPUAvailable()) {
            System.out.println("GPU不可用，跳过GPU矩阵乘法测试");
            return;
        }
        
        try {
            // 创建测试矩阵A (3x3)
            float[][] testMatrixA = {
                {1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f},
                {7.0f, 8.0f, 9.0f}
            };
            
            // 创建测试矩阵B (3x2)
            float[][] testMatrixB = {
                {1.0f, 2.0f},
                {3.0f, 4.0f},
                {5.0f, 6.0f}
            };
            
            IMatrix matrixA = new RereMatrix(testMatrixA);
            IMatrix matrixB = new RereMatrix(testMatrixB);
            
            IMatrix result = GPUComputeUtils.gpuMatrixMultiply(matrixA, matrixB);
            
            System.out.println("GPU矩阵乘法结果:");
            System.out.println("结果矩阵大小: " + result.getRowNum() + "x" + result.getColNum());
            
            // 显示结果矩阵
            for (int i = 0; i < result.getRowNum(); i++) {
                for (int j = 0; j < result.getColNum(); j++) {
                    System.out.print(String.format("%.2f ", result.get(i, j)));
                }
                System.out.println();
            }
            
            // 验证计算结果
            float expected00 = 1.0f * 1.0f + 2.0f * 3.0f + 3.0f * 5.0f; // 22
            float expected01 = 1.0f * 2.0f + 2.0f * 4.0f + 3.0f * 6.0f; // 28
            
            if (Math.abs(result.get(0, 0) - expected00) < 0.001f && 
                Math.abs(result.get(0, 1) - expected01) < 0.001f) {
                System.out.println("✓ GPU矩阵乘法测试通过");
            } else {
                System.out.println("✗ GPU矩阵乘法测试失败");
            }
            
        } catch (Exception e) {
            System.out.println("GPU矩阵乘法测试失败: " + e.getMessage());
            if (e.getMessage().contains("GPU不可用")) {
                System.out.println("这是预期的，因为GPU不可用");
            }
        }
    }
    
    /**
     * 测试GPU向量加法
     */
    private static void testGPUVectorAdd() {
        System.out.println("\n--- 测试GPU向量加法 ---");
        
        if (!GPUComputeUtils.isGPUAvailable()) {
            System.out.println("GPU不可用，跳过GPU向量加法测试");
            return;
        }
        
        try {
            float[] testVectorA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            float[] testVectorB = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            
            IVector vectorA = new RereVector(testVectorA);
            IVector vectorB = new RereVector(testVectorB);
            
            IVector result = GPUComputeUtils.gpuVectorAdd(vectorA, vectorB);
            
            System.out.println("GPU向量加法结果:");
            System.out.print("向量A: ");
            for (int i = 0; i < testVectorA.length; i++) {
                System.out.print(testVectorA[i] + " ");
            }
            System.out.println();
            
            System.out.print("向量B: ");
            for (int i = 0; i < testVectorB.length; i++) {
                System.out.print(testVectorB[i] + " ");
            }
            System.out.println();
            
            System.out.print("结果:  ");
            for (int i = 0; i < result.length(); i++) {
                System.out.print(result.get(i) + " ");
            }
            System.out.println();
            
            // 验证计算结果
            boolean correct = true;
            for (int i = 0; i < testVectorA.length; i++) {
                float expected = testVectorA[i] + testVectorB[i];
                if (Math.abs(result.get(i) - expected) > 0.001f) {
                    correct = false;
                    break;
                }
            }
            
            if (correct) {
                System.out.println("✓ GPU向量加法测试通过");
            } else {
                System.out.println("✗ GPU向量加法测试失败");
            }
            
        } catch (Exception e) {
            System.out.println("GPU向量加法测试失败: " + e.getMessage());
            if (e.getMessage().contains("GPU不可用")) {
                System.out.println("这是预期的，因为GPU不可用");
            }
        }
    }
    
    /**
     * 测试GPU向量内积
     */
    private static void testGPUVectorDot() {
        System.out.println("\n--- 测试GPU向量内积 ---");
        
        if (!GPUComputeUtils.isGPUAvailable()) {
            System.out.println("GPU不可用，跳过GPU向量内积测试");
            return;
        }
        
        try {
            float[] testVectorA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            float[] testVectorB = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            
            IVector vectorA = new RereVector(testVectorA);
            IVector vectorB = new RereVector(testVectorB);
            
            float result = GPUComputeUtils.gpuVectorDot(vectorA, vectorB);
            
            // 计算预期结果
            float expected = 0.0f;
            for (int i = 0; i < testVectorA.length; i++) {
                expected += testVectorA[i] * testVectorB[i];
            }
            
            System.out.println("GPU向量内积结果: " + result);
            System.out.println("预期结果: " + expected);
            
            if (Math.abs(result - expected) < 0.001f) {
                System.out.println("✓ GPU向量内积测试通过");
            } else {
                System.out.println("✗ GPU向量内积测试失败");
            }
            
        } catch (Exception e) {
            System.out.println("GPU向量内积测试失败: " + e.getMessage());
            if (e.getMessage().contains("GPU不可用")) {
                System.out.println("这是预期的，因为GPU不可用");
            }
        }
    }
    
    /**
     * 测试GPU资源清理
     */
    private static void testGPUCleanup() {
        System.out.println("\n--- 测试GPU资源清理 ---");
        
        try {
            GPUComputeUtils.cleanup();
            System.out.println("✓ GPU资源清理测试通过");
        } catch (Exception e) {
            System.out.println("✗ GPU资源清理测试失败: " + e.getMessage());
        }
    }
    
    /**
     * 测试大矩阵性能
     */
    private static void testLargeMatrixPerformance() {
        System.out.println("\n--- 测试大矩阵GPU性能 ---");
        
        if (!GPUComputeUtils.isGPUAvailable()) {
            System.out.println("GPU不可用，跳过大矩阵性能测试");
            return;
        }
        
        try {
            // 创建较大的矩阵进行性能测试
            int size = 50; // 使用较小的矩阵避免内存问题
            System.out.println("创建 " + size + "x" + size + " 矩阵...");
            
            float[][] largeMatrixA = createRandomMatrix(size, size);
            float[][] largeMatrixB = createRandomMatrix(size, size);
            
            IMatrix matrixA = new RereMatrix(largeMatrixA);
            IMatrix matrixB = new RereMatrix(largeMatrixB);
            
            System.out.println("开始GPU大矩阵乘法性能测试...");
            long startTime = System.currentTimeMillis();
            
            IMatrix result = GPUComputeUtils.gpuMatrixMultiply(matrixA, matrixB);
            
            long endTime = System.currentTimeMillis();
            long duration = endTime - startTime;
            
            System.out.println("GPU大矩阵乘法完成，耗时: " + duration + " ms");
            System.out.println("结果矩阵大小: " + result.getRowNum() + "x" + result.getColNum());
            
            // 计算性能指标
            long operations = (long) size * size * size * 2; // 乘法和加法
            double gflops = (operations / 1e9) / (duration / 1000.0);
            System.out.println("性能: " + String.format("%.2f", gflops) + " GFLOPS");
            
            System.out.println("✓ GPU大矩阵性能测试完成");
            
        } catch (Exception e) {
            System.out.println("GPU大矩阵性能测试失败: " + e.getMessage());
            if (e.getMessage().contains("GPU不可用")) {
                System.out.println("这是预期的，因为GPU不可用");
            }
        }
    }
    
    /**
     * 创建随机矩阵
     */
    private static float[][] createRandomMatrix(int rows, int cols) {
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (float) (Math.random() * 2.0 - 1.0); // -1 到 1 之间的随机数
            }
        }
        return matrix;
    }
}
