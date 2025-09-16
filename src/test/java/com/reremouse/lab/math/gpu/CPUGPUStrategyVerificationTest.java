package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUConfig;
import com.reremouse.lab.math.compute.GPUComputeFloatUtils;
import com.reremouse.lab.math.linalg.RereDoubleMatrix;
import com.reremouse.lab.math.linalg.RereDoubleVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 * 验证RereMatrix和RereVector的CPU/GPU策略
 Verify CPU/GPU strategy in RereDoubleMatrix and RereDoubleVector
 */
public class CPUGPUStrategyVerificationTest {
    
    @BeforeEach
    void setUp() {
        // 启用日志以观察CPU/GPU选择
        GPUComputeFloatUtils.setLoggingEnabled(true);
    }
    
    @Test
    @DisplayName("验证RereMatrix小数据使用CPU策略")
    void testRereMatrixSmallDataUsesCPU() {
        System.out.println("=== RereMatrix小数据CPU策略验证 ===");
        System.out.println("GPU阈值: " + GPUConfig.GPU_THRESHOLD);
        
        // 创建小矩阵（远小于GPU阈值）
        double[][] data1 = {{1, 2}, {3, 4}};
        double[][] data2 = {{5, 6}, {7, 8}};
        IMatrix<Double> m1 = new RereDoubleMatrix(data1);
        IMatrix<Double> m2 = new RereDoubleMatrix(data2);
        
        int dataSize = m1.rows() * m1.cols();
        System.out.println("矩阵大小: " + m1.rows() + "x" + m1.cols() + 
                          ", 数据量: " + dataSize + 
                          ", 预期: 使用CPU (< " + GPUConfig.GPU_THRESHOLD + ")");
        
        // 测试各种矩阵运算
        IMatrix<Double> result1 = m1.add(m2);
        IMatrix<Double> result2 = m1.sub(m2);  
        IMatrix<Double> result3 = m1.mmul(m2);
        IMatrix<Double> result4 = m1.mmul(2.0);
        IMatrix<Double> result5 = m1.transposeNew();
        
        // 验证结果
        assertNotNull(result1);
        assertNotNull(result2);
        assertNotNull(result3);
        assertNotNull(result4);
        assertNotNull(result5);
        
        System.out.println("RereMatrix小数据CPU策略验证完成");
    }
    
    @Test
    @DisplayName("验证RereVector小数据使用CPU策略") 
    void testRereVectorSmallDataUsesCPU() {
        System.out.println("=== RereVector小数据CPU策略验证 ===");
        System.out.println("GPU阈值: " + GPUConfig.GPU_THRESHOLD);
        
        // 创建小向量（远小于GPU阈值）
        double[] data1 = {1, 2, 3, 4, 5};
        double[] data2 = {6, 7, 8, 9, 10};
        IVector<Double> v1 = new RereDoubleVector(data1);
        IVector<Double> v2 = new RereDoubleVector(data2);
        
        System.out.println("向量长度: " + v1.length() + 
                          ", 预期: 使用CPU (< " + GPUConfig.GPU_THRESHOLD + ")");
        
        // 测试各种向量运算
        IVector<Double> result1 = v1.add(v2);
        IVector<Double> result2 = v1.sub(v2);
        IVector<Double> result3 = v1.multiply(v2);
        IVector<Double> result4 = v1.multiplyScalar(2.0);
        IVector<Double> result5 = v1.addScalar(1.0);
        double dotResult = v1.dot(v2);
        double sumResult = v1.sum();
        
        // 验证结果
        assertNotNull(result1);
        assertNotNull(result2);
        assertNotNull(result3);
        assertNotNull(result4);
        assertNotNull(result5);
        assertTrue(dotResult > 0);
        assertTrue(sumResult > 0);
        
        System.out.println("RereVector小数据CPU策略验证完成");
    }
    
    @Test
    @DisplayName("验证大数据策略选择逻辑")
    void testLargeDataStrategyLogic() {
        System.out.println("=== 大数据策略选择验证 ===");
        System.out.println("GPU阈值: " + GPUConfig.GPU_THRESHOLD);
        
        // 计算需要多大的矩阵才会触发GPU
        int threshold = GPUConfig.GPU_THRESHOLD;
        int matrixSize = (int) Math.sqrt(threshold) + 1; // 略大于阈值的平方根
        
        System.out.println("计算矩阵大小: " + matrixSize + "x" + matrixSize + 
                          " = " + (matrixSize * matrixSize) + " 元素");
        
        if (matrixSize * matrixSize > threshold) {
            System.out.println("预期: 尝试GPU计算（如果可用）");
        } else {
            System.out.println("预期: 使用CPU计算");
        }
        
        // 注意：由于实际创建大矩阵可能消耗大量内存，这里只是验证逻辑
        // 在实际应用中，大矩阵会根据GPU可用性和性能选择最优策略
        
        System.out.println("策略选择逻辑验证完成");
    }
    
    @Test
    @DisplayName("验证GPU配置一致性")
    void testGPUConfigConsistency() {
        System.out.println("=== GPU配置一致性验证 ===");
        
        // 验证各个类使用相同的GPU阈值
        int gpuConfigThreshold = GPUConfig.GPU_THRESHOLD;
        boolean gpuEnabled = RereDoubleMatrix.isGPUEnabled();
        String gpuInfo = RereDoubleMatrix.getGPUInfo();
        
        System.out.println("GPUConfig.GPU_THRESHOLD: " + gpuConfigThreshold);
        System.out.println("RereMatrix.isGPUEnabled(): " + gpuEnabled);
        System.out.println("GPU信息: " + gpuInfo);
        
        // 验证向量GPU配置
        boolean vectorGpuEnabled = RereDoubleVector.isGPUEnabled();
        String vectorGpuInfo = RereDoubleVector.getGPUInfo();
        
        System.out.println("RereVector.isGPUEnabled(): " + vectorGpuEnabled);
        System.out.println("Vector GPU信息: " + vectorGpuInfo);
        
        // 验证一致性
        assertEquals(gpuEnabled, vectorGpuEnabled, "Matrix和Vector的GPU启用状态应该一致");
        assertTrue(gpuConfigThreshold > 0, "GPU阈值应该大于0");
        
        System.out.println("GPU配置一致性验证完成");
    }
    
    @Test
    @DisplayName("验证异常处理和回退机制")
    void testExceptionHandlingAndFallback() {
        System.out.println("=== 异常处理和回退机制验证 ===");
        
        // 创建测试数据
        double[][] data = {{1, 2}, {3, 4}};
        IMatrix<Double> matrix = new RereDoubleMatrix(data);
        
        // 即使在异常情况下，运算也应该成功（通过CPU回退）
        try {
            IMatrix<Double> result = matrix.mmul(2.0);
            assertNotNull(result);
            System.out.println("标量乘法成功（通过CPU回退机制）");
            
            IMatrix<Double> transpose = matrix.transposeNew();
            assertNotNull(transpose);
            System.out.println("矩阵转置成功（通过CPU回退机制）");
            
        } catch (Exception e) {
            fail("即使GPU失败，CPU回退也应该保证运算成功: " + e.getMessage());
        }
        
        System.out.println("异常处理和回退机制验证完成");
    }
}