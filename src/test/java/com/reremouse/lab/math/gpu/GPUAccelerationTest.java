package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUComputeUtils;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.RereMatrix;
import com.reremouse.lab.math.linalg.RereVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

/**
 * GPU加速功能测试类
 * 测试RereMatrix和RereVector中新增的GPU加速功能
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GPUAccelerationTest {
    
    private IMatrix largeMatrixA;
    private IMatrix largeMatrixB;
    private IVector largeVectorA;
    private IVector largeVectorB;
    
    @BeforeEach
    void setUp() {
        // 创建大矩阵用于测试GPU加速（超过GPU阈值10000）
        int size = 200; // 200x200 = 40000 > 10000
        float[][] dataA = new float[size][size];
        float[][] dataB = new float[size][size];
        float[] vectorDataA = new float[size * size];
        float[] vectorDataB = new float[size * size];
        
        // 初始化测试数据
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                dataA[i][j] = (float) (Math.random() * 10);
                dataB[i][j] = (float) (Math.random() * 10);
                vectorDataA[i * size + j] = (float) (Math.random() * 10);
                vectorDataB[i * size + j] = (float) (Math.random() * 10);
            }
        }
        
        largeMatrixA = new RereMatrix(dataA);
        largeMatrixB = new RereMatrix(dataB);
        largeVectorA = new RereVector(vectorDataA);
        largeVectorB = new RereVector(vectorDataB);
    }
    
    @Test
    @DisplayName("测试GPU矩阵加法")
    void testGPUMatrixAdd() {
        System.out.println("测试GPU矩阵加法...");
        
        long startTime = System.currentTimeMillis();
        IMatrix result = largeMatrixA.add(largeMatrixB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵加法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.getRows(), result.getRows());
        assertEquals(largeMatrixA.getColumns(), result.getColumns());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float expected = largeMatrixA.get(i, j) + largeMatrixB.get(i, j);
                assertEquals(expected, result.get(i, j), 1e-6f);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU矩阵减法")
    void testGPUMatrixSub() {
        System.out.println("测试GPU矩阵减法...");
        
        long startTime = System.currentTimeMillis();
        IMatrix result = largeMatrixA.sub(largeMatrixB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵减法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.getRows(), result.getRows());
        assertEquals(largeMatrixA.getColumns(), result.getColumns());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float expected = largeMatrixA.get(i, j) - largeMatrixB.get(i, j);
                assertEquals(expected, result.get(i, j), 1e-6f);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU矩阵标量乘法")
    void testGPUMatrixScalarMultiply() {
        System.out.println("测试GPU矩阵标量乘法...");
        
        float scalar = 2.5f;
        long startTime = System.currentTimeMillis();
        IMatrix result = largeMatrixA.mmul(scalar);
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵标量乘法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.getRows(), result.getRows());
        assertEquals(largeMatrixA.getColumns(), result.getColumns());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float expected = largeMatrixA.get(i, j) * scalar;
                assertEquals(expected, result.get(i, j), 1e-6f);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU矩阵标量减法")
    void testGPUMatrixScalarSub() {
        System.out.println("测试GPU矩阵标量减法...");
        
        float scalar = 1.5f;
        long startTime = System.currentTimeMillis();
        IMatrix result = largeMatrixA.sub(scalar);
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵标量减法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.getRows(), result.getRows());
        assertEquals(largeMatrixA.getColumns(), result.getColumns());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                float expected = largeMatrixA.get(i, j) - scalar;
                assertEquals(expected, result.get(i, j), 1e-6f);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU矩阵转置")
    void testGPUMatrixTranspose() {
        System.out.println("测试GPU矩阵转置...");
        
        long startTime = System.currentTimeMillis();
        IMatrix result = largeMatrixA.transposeNew();
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵转置耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.getColumns(), result.getRows());
        assertEquals(largeMatrixA.getRows(), result.getColumns());
        
        // 验证转置的正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                assertEquals(largeMatrixA.get(i, j), result.get(j, i), 1e-6f);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU向量加法")
    void testGPUVectorAdd() {
        System.out.println("测试GPU向量加法...");
        
        long startTime = System.currentTimeMillis();
        IVector result = largeVectorA.add(largeVectorB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量加法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            float expected = largeVectorA.get(i) + largeVectorB.get(i);
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量减法")
    void testGPUVectorSub() {
        System.out.println("测试GPU向量减法...");
        
        long startTime = System.currentTimeMillis();
        IVector result = largeVectorA.sub(largeVectorB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量减法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            float expected = largeVectorA.get(i) - largeVectorB.get(i);
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量乘法")
    void testGPUVectorMultiply() {
        System.out.println("测试GPU向量乘法...");
        
        long startTime = System.currentTimeMillis();
        IVector result = largeVectorA.multiply(largeVectorB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量乘法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            float expected = largeVectorA.get(i) * largeVectorB.get(i);
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量标量加法")
    void testGPUVectorScalarAdd() {
        System.out.println("测试GPU向量标量加法...");
        
        float scalar = 3.14f;
        long startTime = System.currentTimeMillis();
        IVector result = largeVectorA.addScalar(scalar);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量标量加法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            float expected = largeVectorA.get(i) + scalar;
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量标量乘法")
    void testGPUVectorScalarMultiply() {
        System.out.println("测试GPU向量标量乘法...");
        
        float scalar = 2.71f;
        long startTime = System.currentTimeMillis();
        IVector result = largeVectorA.multiplyScalar(scalar);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量标量乘法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            float expected = largeVectorA.get(i) * scalar;
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量内积")
    void testGPUVectorInnerProduct() {
        System.out.println("测试GPU向量内积...");
        
        long startTime = System.currentTimeMillis();
        float result = largeVectorA.innerProduct(largeVectorB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量内积耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性（计算整个向量的内积进行验证）
        float expected = 0.0f;
        for (int i = 0; i < largeVectorA.length(); i++) {
            expected += largeVectorA.get(i) * largeVectorB.get(i);
        }
        
        // 验证内积结果
        assertEquals(expected, result, 1e-6f, "内积结果不匹配");
    }
    
    @Test
    @DisplayName("测试GPU向量求和")
    void testGPUVectorSum() {
        System.out.println("测试GPU向量求和...");
        
        long startTime = System.currentTimeMillis();
        float result = largeVectorA.sum();
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量求和耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性（计算整个向量的和进行验证）
        float expected = 0.0f;
        for (int i = 0; i < largeVectorA.length(); i++) {
            expected += largeVectorA.get(i);
        }
        
        // 验证求和结果
        assertEquals(expected, result, 1e-6f, "求和结果不匹配");
    }
    
    @Test
    @DisplayName("测试GPU向量平方")
    void testGPUVectorSquare() {
        System.out.println("测试GPU向量平方...");
        
        long startTime = System.currentTimeMillis();
        IVector result = largeVectorA.squre();
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量平方耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            float expected = largeVectorA.get(i) * largeVectorA.get(i);
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量开方")
    void testGPUVectorSqrt() {
        System.out.println("测试GPU向量开方...");
        
        // 确保向量元素都是正数
        IVector positiveVector = largeVectorA.multiplyScalar(0.1f).addScalar(1.0f);
        
        long startTime = System.currentTimeMillis();
        IVector result = positiveVector.sqrt();
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量开方耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(positiveVector.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            float expected = (float) Math.sqrt(positiveVector.get(i));
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU信息")
    void testGPUInfo() {
        System.out.println("GPU信息:");
        System.out.println("GPU可用: " + GPUComputeUtils.isGPUAvailable());
        System.out.println("GPU信息: " + GPUComputeUtils.getGPUInfo());
        System.out.println("GPU设备信息: " + GPUComputeUtils.getGPUDeviceInfo());
        
        // 验证GPU信息不为空
        assertNotNull(GPUComputeUtils.getGPUInfo());
        assertNotNull(GPUComputeUtils.getGPUDeviceInfo());
    }
    
    @Test
    @DisplayName("测试小数据量使用CPU")
    void testSmallDataUsesCPU() {
        System.out.println("测试小数据量使用CPU...");
        
        // 创建小矩阵（小于GPU阈值）
        float[][] smallData = {{1, 2}, {3, 4}};
        IMatrix smallMatrix = new RereMatrix(smallData);
        IMatrix smallMatrix2 = new RereMatrix(smallData);
        
        // 创建小向量
        float[] smallVectorData = {1, 2, 3, 4};
        IVector smallVector = new RereVector(smallVectorData);
        IVector smallVector2 = new RereVector(smallVectorData);
        
        // 这些操作应该使用CPU（因为数据量小）
        IMatrix matrixResult = smallMatrix.add(smallMatrix2);
        IVector vectorResult = smallVector.add(smallVector2);
        
        // 验证结果正确性
        assertNotNull(matrixResult);
        assertNotNull(vectorResult);
        
        // 验证矩阵加法结果
        assertEquals(2, matrixResult.get(0, 0), 1e-6f);
        assertEquals(4, matrixResult.get(0, 1), 1e-6f);
        assertEquals(6, matrixResult.get(1, 0), 1e-6f);
        assertEquals(8, matrixResult.get(1, 1), 1e-6f);
        
        // 验证向量加法结果
        assertEquals(2, vectorResult.get(0), 1e-6f);
        assertEquals(4, vectorResult.get(1), 1e-6f);
        assertEquals(6, vectorResult.get(2), 1e-6f);
        assertEquals(8, vectorResult.get(3), 1e-6f);
    }
}
