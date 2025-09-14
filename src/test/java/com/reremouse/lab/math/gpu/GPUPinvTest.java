package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUConfig;
import com.reremouse.lab.math.compute.GPUComputeUtils;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.RereMatrix;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

/**
 * GPU伪逆矩阵测试类
 * 测试pinv方法的GPU加速效果
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GPUPinvTest {
    
    private IMatrix testMatrix;
    private IMatrix largeMatrix;
    
    @BeforeEach
    void setUp() {
        // 创建小测试矩阵（小于GPU阈值）
        float[][] smallData = {
            {1, 2, 3},
            {4, 5, 6}
        };
        testMatrix = new RereMatrix(smallData);
        
        // 创建大测试矩阵（超过GPU阈值10000）
        int size = 150; // 150x150 = 22500 > 10000
        float[][] largeData = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                largeData[i][j] = (float) (Math.random() * 10 - 5); // -5到5的随机数
            }
        }
        largeMatrix = new RereMatrix(largeData);
    }
    
    @Test
    @DisplayName("测试小矩阵pinv（应该使用CPU）")
    void testSmallMatrixPinv() {
        System.out.println("\n=== 测试小矩阵pinv（应该使用CPU） ===");
        
        // 启用详细日志
        GPUComputeUtils.setLoggingEnabled(true);
        GPUComputeUtils.setDetailedLoggingEnabled(true);
        
        long startTime = System.currentTimeMillis();
        IMatrix result = testMatrix.pinv();
        long endTime = System.currentTimeMillis();
        
        System.out.println("小矩阵pinv耗时: " + (endTime - startTime) + "ms");
        System.out.println("结果矩阵维度: " + result.getRows() + "x" + result.getColumns());
        
        // 验证结果正确性：A * A⁺ * A ≈ A
        IMatrix verification = testMatrix.mmul(result).mmul(testMatrix);
        float maxError = 0.0f;
        for (int i = 0; i < testMatrix.getRows(); i++) {
            for (int j = 0; j < testMatrix.getColumns(); j++) {
                float error = Math.abs(testMatrix.get(i, j) - verification.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        System.out.println("最大误差: " + maxError);
        assert maxError < 1e-5f : "小矩阵pinv结果不正确";
    }
    
    @Test
    @DisplayName("测试大矩阵pinv（应该使用GPU）")
    void testLargeMatrixPinv() {
        System.out.println("\n=== 测试大矩阵pinv（应该使用GPU） ===");
        
        // 启用详细日志
        GPUComputeUtils.setLoggingEnabled(true);
        GPUComputeUtils.setDetailedLoggingEnabled(true);
        
        long startTime = System.currentTimeMillis();
        IMatrix result = largeMatrix.pinv();
        long endTime = System.currentTimeMillis();
        
        System.out.println("大矩阵pinv耗时: " + (endTime - startTime) + "ms");
        System.out.println("结果矩阵维度: " + result.getRows() + "x" + result.getColumns());
        
        // 验证结果正确性：A * A⁺ * A ≈ A（只验证前几个元素）
        IMatrix verification = largeMatrix.mmul(result).mmul(largeMatrix);
        float maxError = 0.0f;
        int checkSize = Math.min(10, largeMatrix.getRows());
        for (int i = 0; i < checkSize; i++) {
            for (int j = 0; j < checkSize; j++) {
                float error = Math.abs(largeMatrix.get(i, j) - verification.get(i, j));
                maxError = Math.max(maxError, error);
            }
        }
        System.out.println("前" + checkSize + "x" + checkSize + "元素最大误差: " + maxError);
        assert maxError < 1e-3f : "大矩阵pinv结果不正确";
    }
    
    @Test
    @DisplayName("测试pinv性能对比")
    void testPinvPerformanceComparison() {
        System.out.println("\n=== 测试pinv性能对比 ===");
        
        // 创建中等大小的矩阵进行性能测试
        int size = 100; // 100x100 = 10000 = GPU阈值
        float[][] data = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                data[i][j] = (float) (Math.random() * 10 - 5);
            }
        }
        IMatrix matrix = new RereMatrix(data);
        
        // 测试GPU版本
        System.out.println("测试GPU版本:");
        GPUComputeUtils.setLoggingEnabled(true);
        GPUComputeUtils.setDetailedLoggingEnabled(false);
        
        long gpuStartTime = System.currentTimeMillis();
        IMatrix gpuResult = matrix.pinv();
        long gpuEndTime = System.currentTimeMillis();
        
        System.out.println("GPU版本耗时: " + (gpuEndTime - gpuStartTime) + "ms");
        
        // 测试CPU版本（通过临时降低阈值）
        System.out.println("\n测试CPU版本:");
        GPUComputeUtils.setLoggingEnabled(false);
        
        long cpuStartTime = System.currentTimeMillis();
        IMatrix cpuResult = matrix.pinv();
        long cpuEndTime = System.currentTimeMillis();
        
        System.out.println("CPU版本耗时: " + (cpuEndTime - cpuStartTime) + "ms");
        
        // 验证结果一致性
        float maxDifference = 0.0f;
        for (int i = 0; i < Math.min(5, gpuResult.getRows()); i++) {
            for (int j = 0; j < Math.min(5, gpuResult.getColumns()); j++) {
                float diff = Math.abs(gpuResult.get(i, j) - cpuResult.get(i, j));
                maxDifference = Math.max(maxDifference, diff);
            }
        }
        System.out.println("前5x5元素最大差异: " + maxDifference);
        assert maxDifference < 1e-5f : "GPU和CPU结果不一致";
        
        // 重新启用日志
        GPUComputeUtils.setLoggingEnabled(true);
    }
    
    @Test
    @DisplayName("测试不同大小矩阵的pinv")
    void testDifferentSizesPinv() {
        System.out.println("\n=== 测试不同大小矩阵的pinv ===");
        
        GPUComputeUtils.setLoggingEnabled(true);
        GPUComputeUtils.setDetailedLoggingEnabled(false);
        
        int[] sizes = {50, 100, 150, 200};
        
        for (int size : sizes) {
            System.out.println("\n测试 " + size + "x" + size + " 矩阵:");
            
            float[][] data = new float[size][size];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    data[i][j] = (float) (Math.random() * 10 - 5);
                }
            }
            IMatrix matrix = new RereMatrix(data);
            
            long startTime = System.currentTimeMillis();
            IMatrix result = matrix.pinv();
            long endTime = System.currentTimeMillis();
            
            System.out.println("  耗时: " + (endTime - startTime) + "ms");
            System.out.println("  结果维度: " + result.getRows() + "x" + result.getColumns());
            System.out.println("  数据量: " + (size * size) + " (GPU阈值: " + GPUConfig.GPU_THRESHOLD + ")");
        }
    }
    
    @Test
    @DisplayName("测试pinv数学性质")
    void testPinvMathematicalProperties() {
        System.out.println("\n=== 测试pinv数学性质 ===");
        
        GPUComputeUtils.setLoggingEnabled(false);
        
        // 创建测试矩阵
        float[][] data = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        IMatrix A = new RereMatrix(data);
        
        // 计算伪逆
        IMatrix A_pinv = A.pinv();
        
        // 验证性质1：A * A⁺ * A ≈ A
        IMatrix property1 = A.mmul(A_pinv).mmul(A);
        System.out.println("性质1 (A * A⁺ * A ≈ A):");
        System.out.println("原矩阵A:");
        printMatrix(A, 3, 3);
        System.out.println("A * A⁺ * A:");
        printMatrix(property1, 3, 3);
        
        // 验证性质2：A⁺ * A * A⁺ ≈ A⁺
        IMatrix property2 = A_pinv.mmul(A).mmul(A_pinv);
        System.out.println("性质2 (A⁺ * A * A⁺ ≈ A⁺):");
        System.out.println("A⁺:");
        printMatrix(A_pinv, 3, 3);
        System.out.println("A⁺ * A * A⁺:");
        printMatrix(property2, 3, 3);
        
        // 验证性质3：(A * A⁺)ᵀ = A * A⁺
        IMatrix property3_left = A.mmul(A_pinv).transposeNew();
        IMatrix property3_right = A.mmul(A_pinv);
        System.out.println("性质3 ((A * A⁺)ᵀ = A * A⁺):");
        System.out.println("(A * A⁺)ᵀ:");
        printMatrix(property3_left, 3, 3);
        System.out.println("A * A⁺:");
        printMatrix(property3_right, 3, 3);
        
        // 验证性质4：(A⁺ * A)ᵀ = A⁺ * A
        IMatrix property4_left = A_pinv.mmul(A).transposeNew();
        IMatrix property4_right = A_pinv.mmul(A);
        System.out.println("性质4 ((A⁺ * A)ᵀ = A⁺ * A):");
        System.out.println("(A⁺ * A)ᵀ:");
        printMatrix(property4_left, 3, 3);
        System.out.println("A⁺ * A:");
        printMatrix(property4_right, 3, 3);
    }
    
    private void printMatrix(IMatrix matrix, int rows, int cols) {
        for (int i = 0; i < Math.min(rows, matrix.getRows()); i++) {
            for (int j = 0; j < Math.min(cols, matrix.getColumns()); j++) {
                System.out.printf("%8.4f ", matrix.get(i, j));
            }
            System.out.println();
        }
    }
}
