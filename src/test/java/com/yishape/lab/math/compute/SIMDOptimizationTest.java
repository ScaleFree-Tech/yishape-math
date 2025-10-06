package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import static org.junit.jupiter.api.Assertions.*;

/**
 * SIMD优化效果验证测试
 * 专门测试长度1000+向量的性能优化
 * 
 * @author 系统优化
 */
public class SIMDOptimizationTest {
    
    private static SIMDDoubleComputer simdComputer;
    private static SISDDoubleComputer sisdComputer;
    
    @BeforeAll
    static void setUp() {
        // 启用性能监控
        System.setProperty("simd.performance.monitoring", "true");
        System.setProperty("simd.detailed.logging", "false");
        
        simdComputer = new SIMDDoubleComputer();
        sisdComputer = new SISDDoubleComputer();
        
        // 检查SIMD支持
        boolean simdSupported = SIMDDoubleComputer.checkIfSupport();
        System.out.println("SIMD支持状态: " + simdSupported);
    }
    
    @Test
    public void testLargeVectorAddition() {
        System.out.println("\n=== 测试大向量加法性能 ===");
        
        // 创建长度为1500的向量（针对1000+向量优化）
        int length = 1500;
        double[] x1 = createTestVector(length, 1.0);
        double[] x2 = createTestVector(length, 2.0);
        
        // SIMD计算
        long simdStart = System.nanoTime();
        double[] simdResult = simdComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        long simdTime = System.nanoTime() - simdStart;
        
        // SISD计算（对照组）
        long sisdStart = System.nanoTime();
        double[] sisdResult = sisdComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        long sisdTime = System.nanoTime() - sisdStart;
        
        // 验证结果正确性
        assertArrayEquals(sisdResult, simdResult, 1e-10);
        
        // 性能分析
        double speedup = (double) sisdTime / simdTime;
        System.out.printf("向量长度: %d%n", length);
        System.out.printf("SIMD时间: %.2f μs%n", simdTime / 1000.0);
        System.out.printf("SISD时间: %.2f μs%n", sisdTime / 1000.0);
        System.out.printf("加速比: %.2fx%n", speedup);
        
        // 对于长向量，SIMD应该有显著优势
        if (SIMDDoubleComputer.checkIfSupport()) {
            assertTrue(speedup > 1.0, "SIMD应该比SISD更快");
        }
    }
    
    @Test
    public void testLargeVectorDotProduct() {
        System.out.println("\n=== 测试大向量点积性能 ===");
        
        int length = 2048; // 更大的向量
        double[] x1 = createTestVector(length, 1.5);
        double[] x2 = createTestVector(length, 0.5);
        
        // SIMD计算
        long simdStart = System.nanoTime();
        double simdResult = simdComputer.binaryReduceOperate(x1, x2, IDoubleVectorComputer.BinaryReduceOperation.DOT);
        long simdTime = System.nanoTime() - simdStart;
        
        // SISD计算
        long sisdStart = System.nanoTime();
        double sisdResult = sisdComputer.binaryReduceOperate(x1, x2, IDoubleVectorComputer.BinaryReduceOperation.DOT);
        long sisdTime = System.nanoTime() - sisdStart;
        
        // 验证结果正确性
        assertEquals(sisdResult, simdResult, 1e-8);
        
        double speedup = (double) sisdTime / simdTime;
        System.out.printf("点积向量长度: %d%n", length);
        System.out.printf("SIMD时间: %.2f μs%n", simdTime / 1000.0);
        System.out.printf("SISD时间: %.2f μs%n", sisdTime / 1000.0);
        System.out.printf("点积加速比: %.2fx%n", speedup);
        
        if (SIMDDoubleComputer.checkIfSupport()) {
            assertTrue(speedup > 1.0, "点积运算SIMD应该更快");
        }
    }
    
    @Test
    public void testMatrixMultiplicationOptimization() {
        System.out.println("\n=== 测试矩阵乘法优化 ===");
        
        // 创建中等大小的矩阵
        int m = 100, n = 150, p = 120;
        double[][] a = createTestMatrix(m, n, 1.0);
        double[][] b = createTestMatrix(n, p, 2.0);
        
        // SIMD矩阵乘法
        long simdStart = System.nanoTime();
        double[][] simdResult = simdComputer.mmul(a, b);
        long simdTime = System.nanoTime() - simdStart;
        
        // SISD矩阵乘法
        long sisdStart = System.nanoTime();
        double[][] sisdResult = sisdComputer.mmul(a, b);
        long sisdTime = System.nanoTime() - sisdStart;
        
        // 验证结果正确性
        assertEquals(simdResult.length, sisdResult.length);
        assertEquals(simdResult[0].length, sisdResult[0].length);
        
        for (int i = 0; i < simdResult.length; i++) {
            assertArrayEquals(sisdResult[i], simdResult[i], 1e-8);
        }
        
        double speedup = (double) sisdTime / simdTime;
        System.out.printf("矩阵维度: %dx%d * %dx%d%n", m, n, n, p);
        System.out.printf("SIMD时间: %.2f ms%n", simdTime / 1_000_000.0);
        System.out.printf("SISD时间: %.2f ms%n", sisdTime / 1_000_000.0);
        System.out.printf("矩阵乘法加速比: %.2fx%n", speedup);
        
        if (SIMDDoubleComputer.checkIfSupport()) {
            assertTrue(speedup > 1.0, "矩阵乘法SIMD应该更快");
        }
    }
    
    @Test
    public void testPerformanceMonitoring() {
        System.out.println("\n=== 性能监控测试 ===");
        
        // 执行一些操作来生成统计数据
        int length = 1000;
        double[] x1 = createTestVector(length, 1.0);
        double[] x2 = createTestVector(length, 2.0);
        
        // 执行多次操作
        for (int i = 0; i < 5; i++) {
            simdComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
            simdComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
            simdComputer.reduceOperate(x1, IDoubleVectorComputer.ReduceOperation.SUM);
        }
        
        // 输出性能统计
        String stats = SIMDDoubleComputer.getPerformanceStats();
        System.out.println("性能统计:");
        System.out.println(stats);
        
        assertNotNull(stats);
        assertTrue(stats.contains("Total Operations"));
    }
    
    /**
     * 创建测试向量
     */
    private double[] createTestVector(int length, double baseValue) {
        double[] vector = new double[length];
        for (int i = 0; i < length; i++) {
            vector[i] = baseValue + i * 0.001; // 添加一些变化
        }
        return vector;
    }
    
    /**
     * 创建测试矩阵
     */
    private double[][] createTestMatrix(int rows, int cols, double baseValue) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = baseValue + i * 0.01 + j * 0.001;
            }
        }
        return matrix;
    }
}