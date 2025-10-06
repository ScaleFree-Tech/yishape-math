package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import static org.junit.jupiter.api.Assertions.*;

/**
 * SIMD特定操作优化测试
 * 专门测试求和和向量标量乘法的性能优化
 */
public class SIMDSpecificOptimizationTest {
    
    private static SIMDDoubleComputer simdComputer;
    private static SISDDoubleComputer sisdComputer;
    
    @BeforeAll
    static void setUp() {
        simdComputer = new SIMDDoubleComputer();
        sisdComputer = new SISDDoubleComputer();
        
        // 预热JVM
        warmupOperations();
    }
    
    private static void warmupOperations() {
        System.out.println("预热JVM...");
        double[] warmupData = createTestVector(1000, 1.0);
        
        // 预热求和操作
        for (int i = 0; i < 50; i++) {
            simdComputer.reduceOperate(warmupData, IDoubleVectorComputer.ReduceOperation.SUM);
            sisdComputer.reduceOperate(warmupData, IDoubleVectorComputer.ReduceOperation.SUM);
        }
        
        // 预热向量标量乘法
        for (int i = 0; i < 50; i++) {
            simdComputer.binaryOperate(warmupData, 2.5, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
            sisdComputer.binaryOperate(warmupData, 2.5, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        }
        
        System.out.println("预热完成");
    }
    
    @Test
    public void testSumOperationPerformance() {
        System.out.println("\\n=== 优化后的求和操作性能测试 ===");
        
        int[] testSizes = {500, 1000, 2000, 5000};
        
        for (int size : testSizes) {
            double[] testData = createTestVector(size, 1.5);
            
            // 多次测试取平均值
            long simdTotalTime = 0;
            long sisdTotalTime = 0;
            int iterations = 30;
            
            // SIMD测试
            double simdResult = 0;
            for (int iter = 0; iter < iterations; iter++) {
                long start = System.nanoTime();
                simdResult = simdComputer.reduceOperate(testData, IDoubleVectorComputer.ReduceOperation.SUM);
                simdTotalTime += System.nanoTime() - start;
            }
            
            // SISD测试
            double sisdResult = 0;
            for (int iter = 0; iter < iterations; iter++) {
                long start = System.nanoTime();
                sisdResult = sisdComputer.reduceOperate(testData, IDoubleVectorComputer.ReduceOperation.SUM);
                sisdTotalTime += System.nanoTime() - start;
            }
            
            // 验证结果正确性
            assertEquals(sisdResult, simdResult, 1e-8);
            
            double avgSimdTime = simdTotalTime / (double) iterations;
            double avgSisdTime = sisdTotalTime / (double) iterations;
            double speedup = avgSisdTime / avgSimdTime;
            
            System.out.printf("向量大小: %d, SIMD: %.0f ns, SISD: %.0f ns, 加速比: %.2fx%n", 
                size, avgSimdTime, avgSisdTime, speedup);
        }
    }
    
    @Test
    public void testVectorScalarMultiplicationPerformance() {
        System.out.println("\\n=== 优化后的向量标量乘法性能测试 ===");
        
        int[] testSizes = {500, 1000, 2000, 5000};
        double scalar = 3.14159;
        
        for (int size : testSizes) {
            double[] testData = createTestVector(size, 2.0);
            
            // 多次测试取平均值
            long simdTotalTime = 0;
            long sisdTotalTime = 0;
            int iterations = 30;
            
            // SIMD测试
            double[] simdResult = null;
            for (int iter = 0; iter < iterations; iter++) {
                long start = System.nanoTime();
                simdResult = simdComputer.binaryOperate(testData, scalar, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
                simdTotalTime += System.nanoTime() - start;
            }
            
            // SISD测试
            double[] sisdResult = null;
            for (int iter = 0; iter < iterations; iter++) {
                long start = System.nanoTime();
                sisdResult = sisdComputer.binaryOperate(testData, scalar, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
                sisdTotalTime += System.nanoTime() - start;
            }
            
            // 验证结果正确性
            assertNotNull(simdResult);
            assertNotNull(sisdResult);
            assertEquals(simdResult.length, sisdResult.length);
            for (int i = 0; i < Math.min(5, simdResult.length); i++) {
                assertEquals(sisdResult[i], simdResult[i], 1e-10);
            }
            
            double avgSimdTime = simdTotalTime / (double) iterations;
            double avgSisdTime = sisdTotalTime / (double) iterations;
            double speedup = avgSisdTime / avgSimdTime;
            
            System.out.printf("向量大小: %d, SIMD: %.0f ns, SISD: %.0f ns, 加速比: %.2fx%n", 
                size, avgSimdTime, avgSisdTime, speedup);
        }
    }
    
    @Test
    public void testOptimalVectorSizes() {
        System.out.println("\\n=== 寻找最优向量大小 ===");
        
        // 测试不同的向量大小，寻找SIMD优势最明显的区间
        int[] sizes = {200, 500, 1000, 2000, 5000, 10000};
        
        System.out.println("向量大小\\t求和加速比\\t标量乘法加速比");
        for (int size : sizes) {
            double[] testData = createTestVector(size, 1.0);
            
            // 测试求和
            long simdSumTime = measureSumOperation(simdComputer, testData, 15);
            long sisdSumTime = measureSumOperation(sisdComputer, testData, 15);
            double sumSpeedup = (double) sisdSumTime / simdSumTime;
            
            // 测试标量乘法
            long simdMulTime = measureScalarMultiplication(simdComputer, testData, 2.0, 15);
            long sisdMulTime = measureScalarMultiplication(sisdComputer, testData, 2.0, 15);
            double mulSpeedup = (double) sisdMulTime / simdMulTime;
            
            System.out.printf("%d\\t\\t%.2fx\\t\\t%.2fx%n", size, sumSpeedup, mulSpeedup);
        }
    }
    
    private long measureSumOperation(IDoubleVectorComputer computer, double[] data, int iterations) {
        long totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.SUM);
            totalTime += System.nanoTime() - start;
        }
        return totalTime / iterations;
    }
    
    private long measureScalarMultiplication(IDoubleVectorComputer computer, double[] data, double scalar, int iterations) {
        long totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            long start = System.nanoTime();
            computer.binaryOperate(data, scalar, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
            totalTime += System.nanoTime() - start;
        }
        return totalTime / iterations;
    }
    
    /**
     * 创建测试向量
     */
    private static double[] createTestVector(int length, double baseValue) {
        double[] vector = new double[length];
        for (int i = 0; i < length; i++) {
            vector[i] = baseValue + i * 0.001; // 添加一些变化避免编译器优化
        }
        return vector;
    }
}