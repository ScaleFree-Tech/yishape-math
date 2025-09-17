package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUComputeDoubleUtils;
import com.reremouse.lab.math.linalg.RereDoubleMatrix;
import com.reremouse.lab.math.linalg.RereDoubleVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IDoubleVector;
import com.reremouse.lab.math.linalg.IVector;

/**
 * GPU加速功能测试类
 * 测试RereMatrix和RereVector中新增的GPU加速功能
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GPUAccelerationTest {
    
    private IMatrix<Double> largeMatrixA;
    private IMatrix<Double> largeMatrixB;
    private IVector<Double> largeVectorA;
    private IVector<Double> largeVectorB;
    
    @BeforeEach
    void setUp() {
        // 创建大矩阵用于测试GPU加速（超过GPU阈值10000）
        int size = 200; // 200x200 = 40000 > 10000
        double[][] dataA = new double[size][size];
        double[][] dataB = new double[size][size];
        double[] vectorDataA = new double[size * size];
        double[] vectorDataB = new double[size * size];
        
        // 初始化测试数据
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                dataA[i][j] = (double) (Math.random() * 10);
                dataB[i][j] = (double) (Math.random() * 10);
                vectorDataA[i * size + j] = (double) (Math.random() * 10);
                vectorDataB[i * size + j] = (double) (Math.random() * 10);
            }
        }
        
        largeMatrixA = new RereDoubleMatrix(dataA);
        largeMatrixB = new RereDoubleMatrix(dataB);
        largeVectorA = new RereDoubleVector(vectorDataA);
        largeVectorB = new RereDoubleVector(vectorDataB);
    }
    
    @Test
    @DisplayName("测试GPU矩阵加法")
    void testGPUMatrixAdd() {
        System.out.println("测试GPU矩阵加法...");
        
        long startTime = System.currentTimeMillis();
        IMatrix<Double> result = largeMatrixA.add(largeMatrixB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵加法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.rows(), result.rows());
        assertEquals(largeMatrixA.cols(), result.cols());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                double expected = largeMatrixA.get(i, j) + largeMatrixB.get(i, j);
                assertEquals(expected, result.get(i, j), 1e-6d);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU矩阵减法")
    void testGPUMatrixSub() {
        System.out.println("测试GPU矩阵减法...");
        
        long startTime = System.currentTimeMillis();
        IMatrix<Double> result = largeMatrixA.sub(largeMatrixB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵减法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.rows(), result.rows());
        assertEquals(largeMatrixA.cols(), result.cols());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                double expected = largeMatrixA.get(i, j) - largeMatrixB.get(i, j);
                assertEquals(expected, result.get(i, j), 1e-6f);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU矩阵标量乘法")
    void testGPUMatrixScalarMultiply() {
        System.out.println("测试GPU矩阵标量乘法...");
        
        double scalar = 2.5f;
        long startTime = System.currentTimeMillis();
        IMatrix<Double> result = largeMatrixA.multiplyScalar(scalar);
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵标量乘法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.rows(), result.rows());
        assertEquals(largeMatrixA.cols(), result.cols());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                double expected = largeMatrixA.get(i, j) * scalar;
                assertEquals(expected, result.get(i, j), 1e-6f);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU矩阵标量减法")
    void testGPUMatrixScalarSub() {
        System.out.println("测试GPU矩阵标量减法...");
        
        double scalar = 1.5f;
        long startTime = System.currentTimeMillis();
        IMatrix<Double> result = largeMatrixA.sub(scalar);
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵标量减法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.rows(), result.rows());
        assertEquals(largeMatrixA.cols(), result.cols());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                double expected = largeMatrixA.get(i, j) - scalar;
                assertEquals(expected, result.get(i, j), 1e-6f);
            }
        }
    }
    
    @Test
    @DisplayName("测试GPU矩阵转置")
    void testGPUMatrixTranspose() {
        System.out.println("测试GPU矩阵转置...");
        
        long startTime = System.currentTimeMillis();
        IMatrix<Double> result = largeMatrixA.transposeNew();
        long endTime = System.currentTimeMillis();
        
        System.out.println("矩阵转置耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeMatrixA.cols(), result.rows());
        assertEquals(largeMatrixA.rows(), result.cols());
        
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
        IVector<Double> result = largeVectorA.add(largeVectorB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量加法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            double expected = largeVectorA.get(i) + largeVectorB.get(i);
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量减法")
    void testGPUVectorSub() {
        System.out.println("测试GPU向量减法...");
        
        long startTime = System.currentTimeMillis();
        IVector<Double> result = largeVectorA.sub(largeVectorB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量减法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            double expected = largeVectorA.get(i) - largeVectorB.get(i);
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量乘法")
    void testGPUVectorMultiply() {
        System.out.println("测试GPU向量乘法...");
        
        long startTime = System.currentTimeMillis();
        IVector<Double> result = largeVectorA.multiply(largeVectorB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量乘法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            double expected = largeVectorA.get(i) * largeVectorB.get(i);
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量标量加法")
    void testGPUVectorScalarAdd() {
        System.out.println("测试GPU向量标量加法...");
        
        double scalar = 3.14f;
        long startTime = System.currentTimeMillis();
        IVector<Double> result = largeVectorA.addScalar(scalar);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量标量加法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            double expected = largeVectorA.get(i) + scalar;
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量标量乘法")
    void testGPUVectorScalarMultiply() {
        System.out.println("测试GPU向量标量乘法...");
        
        double scalar = 2.71f;
        long startTime = System.currentTimeMillis();
        IVector<Double> result = largeVectorA.multiplyScalar(scalar);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量标量乘法耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            double expected = largeVectorA.get(i) * scalar;
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量内积")
    void testGPUVectorInnerProduct() {
        System.out.println("测试GPU向量内积...");
        
        long startTime = System.currentTimeMillis();
        double result = largeVectorA.innerProduct(largeVectorB);
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量内积耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性（计算整个向量的内积进行验证）
        double expected = 0.0f;
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
        double result = largeVectorA.sum();
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量求和耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性（计算整个向量的和进行验证）
        double expected = 0.0f;
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
        IVector<Double> result = largeVectorA.square();
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量平方耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(largeVectorA.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            double expected = largeVectorA.get(i) * largeVectorA.get(i);
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU向量开方")
    void testGPUVectorSqrt() {
        System.out.println("测试GPU向量开方...");
        
        // 确保向量元素都是正数
        IVector<Double> positiveVector = largeVectorA.multiplyScalar(0.1).addScalar(1.0);
        
        long startTime = System.currentTimeMillis();
        IVector<Double> result = positiveVector.sqrt();
        long endTime = System.currentTimeMillis();
        
        System.out.println("向量开方耗时: " + (endTime - startTime) + "ms");
        
        // 验证结果正确性
        assertNotNull(result);
        assertEquals(positiveVector.length(), result.length());
        
        // 验证几个元素的计算正确性
        for (int i = 0; i < 10; i++) {
            double expected = (double) Math.sqrt(positiveVector.get(i));
            assertEquals(expected, result.get(i), 1e-6f);
        }
    }
    
    @Test
    @DisplayName("测试GPU信息")
    void testGPUInfo() {
        System.out.println("GPU信息:");
        System.out.println("GPU可用: " + GPUComputeDoubleUtils.isGPUAvailable());
        System.out.println("GPU信息: " + GPUComputeDoubleUtils.getGPUInfo());
        System.out.println("GPU设备信息: " + GPUComputeDoubleUtils.getGPUDeviceInfo());
        
        // 验证GPU信息不为空
        assertNotNull(GPUComputeDoubleUtils.getGPUInfo());
        assertNotNull(GPUComputeDoubleUtils.getGPUDeviceInfo());
    }
    
    @Test
    @DisplayName("测试小数据量使用CPU")
    void testSmallDataUsesCPU() {
        System.out.println("测试小数据量使用CPU...");
        
        // 创建小矩阵（小于GPU阈值）
        double[][] smallData = {{1, 2}, {3, 4}};
        IMatrix smallMatrix = new RereDoubleMatrix(smallData);
        IMatrix smallMatrix2 = new RereDoubleMatrix(smallData);
        
        // 创建小向量
        double[] smallVectorData = {1, 2, 3, 4};
        IDoubleVector smallVector = new RereDoubleVector(smallVectorData);
        IDoubleVector smallVector2 = new RereDoubleVector(smallVectorData);
        
        // 这些操作应该使用CPU（因为数据量小）
        IMatrix<Double> matrixResult = smallMatrix.add(smallMatrix2);
        IVector<Double> vectorResult = smallVector.add(smallVector2);
        
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
