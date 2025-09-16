package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUConfig;
import com.reremouse.lab.math.compute.GPUComputeFloatUtils;
import com.reremouse.lab.math.linalg.RereFloatMatrix;
import com.reremouse.lab.util.Tuple2;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 * GPU性能优化验证测试
 * 验证CPU算法借鉴策略是否提升了GPU特征分解性能
 */
public class GPUOptimizationVerificationTest {
    
    @Test
    @DisplayName("验证GPU特征分解性能优化")
    void testGPUEigenDecompositionOptimization() {
        // 启用日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        
        System.out.println("=== GPU特征分解性能优化验证测试 ===");
        System.out.println("GPU阈值: " + GPUConfig.GPU_THRESHOLD);
        System.out.println();
        
        // 测试不同大小的矩阵
        int[] sizes = {10, 50, 100, 200};
        
        for (int size : sizes) {
            System.out.println("--- 测试矩阵大小: " + size + "x" + size + " (数据量: " + (size*size) + ") ---");
            
            // 创建对称矩阵进行特征分解测试
            float[][] data = createSymmetricMatrix(size);
            IMatrix<Float> matrix = new RereFloatMatrix(data);
            
            // 测试GPU特征分解
            long startTime = System.currentTimeMillis();
            Tuple2<IVector<Float>, IMatrix<Float>> result = GPUComputeFloatUtils.gpuEigenDecomposition(matrix);
            long endTime = System.currentTimeMillis();
            
            long duration = endTime - startTime;
            System.out.println("GPU特征分解耗时: " + duration + "ms");
            System.out.println("特征值数量: " + result._1.length());
            System.out.println("特征向量矩阵大小: " + result._2.rows() + "x" + result._2.cols());
            System.out.println();
        }
        
        System.out.println("=== 测试完成 ===");
    }
    
    /**
     * 创建对称矩阵用于测试
     */
    private float[][] createSymmetricMatrix(int size) {
        float[][] matrix = new float[size][size];
        
        // 创建随机对称矩阵
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {
                float value = (float) (Math.random() * 10 - 5); // -5到5的随机数
                matrix[i][j] = value;
                matrix[j][i] = value; // 保证对称性
            }
            // 增强对角优势以确保数值稳定性
            matrix[i][i] += size;
        }
        
        return matrix;
    }
    
    @Test
    @DisplayName("验证小数据CPU优化策略")
    void testSmallDataCPUStrategy() {
        // 启用日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        
        System.out.println("=== 小数据CPU优化策略验证 ===");
        
        // 创建小矩阵（远小于GPU阈值）
        float[][] smallData = {
            {4.0f, 1.0f, 0.0f},
            {1.0f, 3.0f, 1.0f}, 
            {0.0f, 1.0f, 2.0f}
        };
        IMatrix<Float> smallMatrix = new RereFloatMatrix(smallData);
        
        System.out.println("矩阵大小: 3x3, 数据量: 9");
        System.out.println("GPU阈值: " + GPUConfig.GPU_THRESHOLD);
        System.out.println("期望行为: 使用CPU算法，避免GPU设备访问");
        System.out.println();
        
        // 测试特征分解
        Tuple2<IVector<Float>, IMatrix<Float>> eigenResult = GPUComputeFloatUtils.gpuEigenDecomposition(smallMatrix);
        System.out.println("特征分解完成，特征值: " + eigenResult._1);
        
        // 测试SVD分解
        var svdResult = GPUComputeFloatUtils.gpuSVD(smallMatrix);
        System.out.println("SVD分解完成，奇异值: " + svdResult._2);
        
        System.out.println("=== 小数据测试完成 ===");
    }
}