package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUComputeFloatUtils;
import com.reremouse.lab.math.linalg.RereFloatMatrix;
import com.reremouse.lab.math.linalg.RereFloatVector;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IFloatVector;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * GPU日志控制测试类
 * 演示GPUComputeUtils的日志控制功能
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GPULoggingTest {
    
    private IMatrix testMatrixA;
    private IMatrix testMatrixB;
    private IVector testVectorA;
    private IVector testVectorB;
    
    @BeforeEach
    void setUp() {
        // 创建测试数据（超过GPU阈值10000）
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
        
        testMatrixA = Linalg.matrix(dataA);
        testMatrixB = Linalg.matrix(dataB);
        testVectorA = Linalg.vector(vectorDataA);
        testVectorB = Linalg.vector(vectorDataB);
    }
    
    @Test
    @DisplayName("测试关闭日志")
    void testDisableLogging() {
        System.out.println("\n=== 测试关闭日志 ===");
        
        // 关闭日志
        GPUComputeFloatUtils.setLoggingEnabled(false);
        
        // 执行一些GPU操作
        testMatrixA.add(testMatrixB);
        testVectorA.add(testVectorB);
        
        System.out.println("日志已关闭，应该看不到GPU/CPU日志");
        
        // 重新启用日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
    }
    
    @Test
    @DisplayName("测试基本日志")
    void testBasicLogging() {
        System.out.println("\n=== 测试基本日志 ===");
        
        // 启用基本日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(false);
        
        // 执行一些GPU操作
        testMatrixA.add(testMatrixB);
        testVectorA.add(testVectorB);
        testMatrixA.multiplyScalar(2.0f);
        
        System.out.println("基本日志模式：只显示GPU/CPU标识");
    }
    
    @Test
    @DisplayName("测试详细日志")
    void testDetailedLogging() {
        System.out.println("\n=== 测试详细日志 ===");
        
        // 启用详细日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
        
        // 执行一些GPU操作
        testMatrixA.add(testMatrixB);
        testVectorA.add(testVectorB);
        testMatrixA.multiplyScalar(2.0f);
        testVectorA.sum();
        
        System.out.println("详细日志模式：显示GPU/CPU标识、操作详情和性能统计");
    }
    
    @Test
    @DisplayName("测试小数据量CPU回退")
    void testSmallDataCPUFallback() {
        System.out.println("\n=== 测试小数据量CPU回退 ===");
        
        // 启用详细日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
        
        // 创建小矩阵（小于GPU阈值）
        double[][] smallData = {{1, 2}, {3, 4}};
        IMatrix smallMatrix = Linalg.matrix(smallData);
        IMatrix smallMatrix2 = Linalg.matrix(smallData);
        
        // 创建小向量
        double[] smallVectorData = {1, 2, 3, 4};
        IVector smallVector = Linalg.vector(smallVectorData);
        IVector smallVector2 = Linalg.vector(smallVectorData);
        
        // 这些操作应该使用CPU（因为数据量小）
        smallMatrix.add(smallMatrix2);
        smallVector.add(smallVector2);
        
        System.out.println("小数据量操作应该显示CPU回退日志");
    }
    
    @Test
    @DisplayName("测试GPU失败回退")
    void testGPUFailureFallback() {
        System.out.println("\n=== 测试GPU失败回退 ===");
        
        // 启用详细日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
        
        // 模拟GPU不可用的情况
        boolean originalGPUState = GPUComputeFloatUtils.isGPUAvailable();
        
        // 注意：这里我们无法直接设置GPU不可用，因为它是静态初始化的
        // 但我们可以通过日志看到正常的GPU操作
        
        // 执行一些操作
        testMatrixA.add(testMatrixB);
        testVectorA.add(testVectorB);
        
        System.out.println("正常情况下应该显示GPU操作日志");
    }
    
    @Test
    @DisplayName("测试日志控制方法")
    void testLoggingControlMethods() {
        System.out.println("\n=== 测试日志控制方法 ===");
        
        // 测试日志状态
        System.out.println("初始日志状态:");
        System.out.println("日志启用: " + GPUComputeFloatUtils.isLoggingEnabled());
        System.out.println("详细日志启用: " + GPUComputeFloatUtils.isDetailedLoggingEnabled());
        
        // 关闭日志
        GPUComputeFloatUtils.setLoggingEnabled(false);
        System.out.println("关闭日志后:");
        System.out.println("日志启用: " + GPUComputeFloatUtils.isLoggingEnabled());
        
        // 重新启用日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
        System.out.println("启用详细日志后:");
        System.out.println("日志启用: " + GPUComputeFloatUtils.isLoggingEnabled());
        System.out.println("详细日志启用: " + GPUComputeFloatUtils.isDetailedLoggingEnabled());
        
        // 执行一个操作来验证日志
        testMatrixA.add(testMatrixB);
    }
    
    @Test
    @DisplayName("测试所有GPU操作日志")
    void testAllGPUOperationsLogging() {
        System.out.println("\n=== 测试所有GPU操作日志 ===");
        
        // 启用详细日志
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
        
        System.out.println("执行各种GPU操作:");
        
        // 矩阵操作
        System.out.println("\n--- 矩阵操作 ---");
        testMatrixA.add(testMatrixB);
        testMatrixA.sub(testMatrixB);
        testMatrixA.multiplyScalar(2.0f);
        testMatrixA.sub(1.0f);
        testMatrixA.transposeNew();
        
        // 向量操作
        System.out.println("\n--- 向量操作 ---");
        testVectorA.add(testVectorB);
        testVectorA.sub(testVectorB);
        testVectorA.multiply(testVectorB);
        testVectorA.addScalar(1.0f);
        testVectorA.subScalar(1.0f);
        testVectorA.multiplyScalar(2.0f);
        testVectorA.sum();
        testVectorA.square();
        testVectorA.sqrt();
        testVectorA.innerProduct(testVectorB);
        
        System.out.println("\n所有操作完成，应该看到详细的GPU操作日志");
    }
}
