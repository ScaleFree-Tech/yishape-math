package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUConfig;
import com.reremouse.lab.math.compute.GPUComputeFloatUtils;
import com.reremouse.lab.math.linalg.RereFloatMatrix;
import com.reremouse.lab.math.linalg.RereFloatVector;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 * GPU阈值测试类
 * 验证小数据时是否正确使用CPU而不访问GPU设备
 */
public class GPUThresholdTest {
    
    @BeforeEach
    void setUp() {
        // 启用日志以便观察CPU/GPU选择
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
    }
    
    @Test
    @DisplayName("测试小向量运算使用CPU")
    void testSmallVectorOperationsUseCPU() {
        // 创建小向量（远小于GPU阈值）
        float[] data1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] data2 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        IVector<Float> v1 = new RereFloatVector(data1);
        IVector<Float> v2 = new RereFloatVector(data2);
        
        System.out.println("=== 测试小向量运算（长度: " + v1.length() + "，阈值: " + GPUConfig.GPU_THRESHOLD + "）===");
        
        // 测试向量加法
        IVector<Float> result1 = GPUComputeFloatUtils.gpuVectorAdd(v1, v2);
        assertNotNull(result1);
        assertEquals(5, result1.length());
        
        // 测试向量内积
        float dotResult = GPUComputeFloatUtils.gpuVectorDot(v1, v2);
        assertTrue(dotResult > 0);
        
        // 测试向量减法
        IVector<Float> result2 = GPUComputeFloatUtils.gpuVectorSub(v1, v2);
        assertNotNull(result2);
        assertEquals(5, result2.length());
        
        // 测试向量标量运算
        IVector<Float> result3 = GPUComputeFloatUtils.gpuVectorScalarMultiply(v1, 2.0f);
        assertNotNull(result3);
        assertEquals(5, result3.length());
        
        System.out.println("小向量运算测试完成");
    }
    
    @Test
    @DisplayName("测试小矩阵运算使用CPU")
    void testSmallMatrixOperationsUseCPU() {
        // 创建小矩阵（远小于GPU阈值）
        float[][] data1 = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f}
        };
        float[][] data2 = {
            {2.0f, 3.0f, 4.0f},
            {5.0f, 6.0f, 7.0f},
            {8.0f, 9.0f, 1.0f}
        };
        IMatrix<Float> m1 = new RereFloatMatrix(data1);
        IMatrix<Float> m2 = new RereFloatMatrix(data2);
        
        int dataSize = m1.rows() * m1.cols();
        System.out.println("=== 测试小矩阵运算（大小: " + dataSize + "，阈值: " + GPUConfig.GPU_THRESHOLD + "）===");
        
        // 测试矩阵加法
        IMatrix<Float> result1 = GPUComputeFloatUtils.gpuMatrixAdd(m1, m2);
        assertNotNull(result1);
        assertEquals(3, result1.rows());
        assertEquals(3, result1.cols());
        
        // 测试矩阵乘法
        IMatrix<Float> result2 = GPUComputeFloatUtils.gpuMatrixMultiply(m1, m2);
        assertNotNull(result2);
        assertEquals(3, result2.rows());
        assertEquals(3, result2.cols());
        
        // 测试矩阵转置
        IMatrix<Float> result3 = GPUComputeFloatUtils.gpuMatrixTranspose(m1);
        assertNotNull(result3);
        assertEquals(3, result3.rows());
        assertEquals(3, result3.cols());
        
        System.out.println("小矩阵运算测试完成");
    }
    
    @Test
    @DisplayName("测试小矩阵特征分解使用CPU")
    void testSmallMatrixEigenDecompositionUsesCPU() {
        // 创建对称矩阵用于特征分解
        float[][] symmetricData = {
            {4.0f, 1.0f, 0.0f},
            {1.0f, 3.0f, 1.0f},
            {0.0f, 1.0f, 2.0f}
        };
        IMatrix<Float> symmetricMatrix = new RereFloatMatrix(symmetricData);
        
        int dataSize = symmetricMatrix.rows() * symmetricMatrix.cols();
        System.out.println("=== 测试小矩阵特征分解（大小: " + dataSize + "，阈值: " + GPUConfig.GPU_THRESHOLD + "）===");
        
        // 测试GPU特征分解（应该使用CPU）
        Tuple2<IVector<Float>, IMatrix<Float>> result = GPUComputeFloatUtils.gpuEigenDecomposition(symmetricMatrix);
        assertNotNull(result);
        assertNotNull(result._1); // 特征值
        assertNotNull(result._2); // 特征向量
        assertEquals(3, result._1.length());
        assertEquals(3, result._2.rows());
        assertEquals(3, result._2.cols());
        
        System.out.println("小矩阵特征分解测试完成");
    }
    
    @Test
    @DisplayName("测试小矩阵SVD分解使用CPU")
    void testSmallMatrixSVDUsesCPU() {
        // 创建测试矩阵
        float[][] data = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        };
        IMatrix<Float> matrix = new RereFloatMatrix(data);
        
        int dataSize = matrix.rows() * matrix.cols();
        System.out.println("=== 测试小矩阵SVD分解（大小: " + dataSize + "，阈值: " + GPUConfig.GPU_THRESHOLD + "）===");
        
        // 测试GPU SVD分解（应该使用CPU）
        Tuple3<IMatrix<Float>, IVector<Float>, IMatrix<Float>> result = GPUComputeFloatUtils.gpuSVD(matrix);
        assertNotNull(result);
        assertNotNull(result._1); // U矩阵
        assertNotNull(result._2); // 奇异值
        assertNotNull(result._3); // V转置矩阵
        
        System.out.println("小矩阵SVD分解测试完成");
    }
    
    @Test
    @DisplayName("测试优化方法对小数据使用CPU")
    void testOptimizedMethodsUsesCPUForSmallData() {
        // 创建小矩阵
        float[][] data = {
            {2.0f, 1.0f},
            {1.0f, 2.0f}
        };
        IMatrix<Float> matrix = new RereFloatMatrix(data);
        
        int dataSize = matrix.rows() * matrix.cols();
        System.out.println("=== 测试优化方法（大小: " + dataSize + "，阈值: " + GPUConfig.GPU_THRESHOLD + "）===");
        
        // 测试优化特征分解
        Tuple2<IVector<Float>, IMatrix<Float>> eigenResult = GPUComputeFloatUtils.optimizedEigenDecomposition(matrix);
        assertNotNull(eigenResult);
        assertNotNull(eigenResult._1);
        assertNotNull(eigenResult._2);
        
        // 测试优化SVD分解
        Tuple3<IMatrix<Float>, IVector<Float>, IMatrix<Float>> svdResult = GPUComputeFloatUtils.optimizedSVD(matrix);
        assertNotNull(svdResult);
        assertNotNull(svdResult._1);
        assertNotNull(svdResult._2);
        assertNotNull(svdResult._3);
        
        System.out.println("优化方法测试完成");
    }
    
    @Test
    @DisplayName("验证GPU阈值配置")
    void testGPUThresholdConfiguration() {
        System.out.println("=== GPU阈值配置信息 ===");
        System.out.println("GPU_THRESHOLD: " + GPUConfig.GPU_THRESHOLD);
        System.out.println("GPU可用性: " + GPUComputeFloatUtils.isGPUAvailable());
        System.out.println("GPU信息: " + GPUComputeFloatUtils.getGPUInfo());
        
        // 确保阈值是合理的
        assertTrue(GPUConfig.GPU_THRESHOLD > 0);
        assertTrue(GPUConfig.GPU_THRESHOLD >= 1000); // 至少1000个元素
        
        System.out.println("GPU阈值配置验证完成");
    }
}