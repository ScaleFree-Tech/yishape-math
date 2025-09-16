package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUComputeFloatUtils;
import com.reremouse.lab.math.compute.CPUComputeFloatUtils;
import com.reremouse.lab.math.linalg.RereFloatMatrix;
import com.reremouse.lab.math.linalg.RereFloatVector;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;

import java.util.ArrayList;
import java.util.List;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 * GPU vs CPU Performance Comparison Test Class

 Comprehensive performance benchmarks comparing GPUComputeFloatUtils and CPUComputeFloatUtils
 for identical mathematical operations across different data sizes.
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class GPUvsCPUPerformanceComparisonTest {
    
    // 测试数据精度容差
    private static final float TOLERANCE = 1e-5f;
    
    // 性能测量数据
    private static final List<PerformanceResult> performanceResults = new ArrayList<>();
    private static final int[] VECTOR_SIZES = {1000, 10000, 50000, 100000};
    private static final int[] MATRIX_SIZES = {50, 100, 200, 300};
    private static final int[] DECOMPOSITION_SIZES = {50, 100, 150}; // 分解运算使用较小的矩阵
    private static final int BENCHMARK_ITERATIONS = 5;
    
    @BeforeEach
    void setUp() {
        System.out.println("=== 初始化GPU vs CPU性能对比测试 ===");
        System.out.println("GPU可用性: " + GPUComputeFloatUtils.isGPUAvailable());
        performJVMWarmup();
    }
    
    @Test
    @Order(1)
    @DisplayName("向量运算性能对比")
    void testVectorOperationsPerformance() {
        System.out.println("\n=== 向量运算性能对比 ===");
        
        for (int size : VECTOR_SIZES) {
            System.out.printf("\n向量大小: %d\n", size);
            
            IVector<Float> v1 = generateRandomVector(size);
            IVector<Float> v2 = generateRandomVector(size);
            float scalar = 3.14f;
            
            // Vector Addition
            testVectorAddition(v1, v2, size);
            
            // Vector Dot Product
            testVectorDotProduct(v1, v2, size);
            
            // Vector Scalar Multiply
            testVectorScalarMultiply(v1, scalar, size);
        }
    }
    
    @Test
    @Order(2)
    @DisplayName("矩阵运算性能对比")
    void testMatrixOperationsPerformance() {
        System.out.println("\n=== 矩阵运算性能对比 ===");
        
        for (int size : MATRIX_SIZES) {
            System.out.printf("\n矩阵大小: %dx%d\n", size, size);
            
            IMatrix<Float> m1 = generateRandomMatrix(size, size);
            IMatrix<Float> m2 = generateRandomMatrix(size, size);
            float scalar = 2.5f;
            
            // Matrix Addition
            testMatrixAddition(m1, m2, size * size);
            
            // Matrix Multiplication
            testMatrixMultiplication(m1, m2, size * size);
            
            // Matrix Scalar Multiply
            testMatrixScalarMultiply(m1, scalar, size * size);
            
            // Matrix Transpose
            testMatrixTranspose(m1, size * size);
        }
    }
    
    @Test
    @Order(3)
    @DisplayName("高级运算性能对比")
    void testAdvancedOperationsPerformance() {
        System.out.println("\n=== 高级运算性能对比 ===");
        
        int[] sizes = {50, 100, 150};
        
        for (int size : sizes) {
            System.out.printf("\n高级运算矩阵大小: %dx%d\n", size, size);
            
            IMatrix<Float> matrix = generateWellConditionedMatrix(size, size);
            
            // Pseudo Inverse
            testPseudoInverse(matrix, size * size);
            
            // Eigenvalue Decomposition
            testEigenDecomposition(matrix, size * size);
            
            // SVD Decomposition
            testSVDDecomposition(matrix, size * size);
        }
    }
    
    @Test
    @Order(4)
    @DisplayName("性能分析报告")
    void generatePerformanceReport() {
        System.out.println("\n=== 性能分析报告 ===");
        
        if (performanceResults.isEmpty()) {
            System.out.println("无性能数据");
            return;
        }
        
        System.out.println("\n操作类型\t\t数据大小\tCPU(ms)\t\tGPU(ms)\t\t加速比");
        System.out.println("=".repeat(70));
        
        for (PerformanceResult result : performanceResults) {
            System.out.printf("%-20s\t%8d\t%8.2f\t%8.2f\t%8.2fx\n", 
                result.operation, result.dataSize, 
                result.cpuTime / 1_000_000.0, result.gpuTime / 1_000_000.0, result.speedup);
        }
        
        generateSummaryAnalysis();
    }
    
    @Test
    @Order(5)
    @DisplayName("清理资源")
    void cleanup() {
        System.out.println("\n=== 清理GPU资源 ===");
        try {
            GPUComputeFloatUtils.cleanup();
            System.out.println("GPU资源清理完成");
        } catch (Exception e) {
            System.out.println("GPU资源清理失败: " + e.getMessage());
        }
    }
    
    // =========================== 具体测试方法 ===========================
    
    private void testVectorAddition(IVector<Float> v1, IVector<Float> v2, int size) {
        long cpuTime = measureTime(() -> CPUComputeFloatUtils.vectorAdd(v1, v2));
        long gpuTime = measureTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuVectorAdd(v1, v2);
            } catch (Exception e) {
                return CPUComputeFloatUtils.vectorAdd(v1, v2);
            }
        });
        recordResult("Vector Addition", size, cpuTime, gpuTime);
    }
    
    private void testVectorDotProduct(IVector<Float> v1, IVector<Float> v2, int size) {
        long cpuTime = measureScalarTime(() -> CPUComputeFloatUtils.vectorDot(v1, v2));
        long gpuTime = measureScalarTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuVectorDot(v1, v2);
            } catch (Exception e) {
                return CPUComputeFloatUtils.vectorDot(v1, v2);
            }
        });
        recordResult("Vector Dot Product", size, cpuTime, gpuTime);
    }
    
    private void testVectorScalarMultiply(IVector<Float> v, float scalar, int size) {
        long cpuTime = measureTime(() -> CPUComputeFloatUtils.vectorScalarMultiply(v, scalar));
        long gpuTime = measureTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuVectorScalarMultiply(v, scalar);
            } catch (Exception e) {
                return CPUComputeFloatUtils.vectorScalarMultiply(v, scalar);
            }
        });
        recordResult("Vector Scalar Multiply", size, cpuTime, gpuTime);
    }
    
    private void testMatrixAddition(IMatrix<Float> m1, IMatrix<Float> m2, int dataSize) {
        long cpuTime = measureTime(() -> CPUComputeFloatUtils.matrixAdd(m1.toFloatArray(), m2.toFloatArray()));
        long gpuTime = measureTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuMatrixAdd(m1, m2);
            } catch (Exception e) {
                return CPUComputeFloatUtils.matrixAdd(m1.toFloatArray(), m2.toFloatArray());
            }
        });
        recordResult("Matrix Addition", dataSize, cpuTime, gpuTime);
    }
    
    private void testMatrixMultiplication(IMatrix<Float> m1, IMatrix<Float> m2, int dataSize) {
        long cpuTime = measureTime(() -> CPUComputeFloatUtils.matrixMultiply(m1.toFloatArray(), m2.toFloatArray()));
        long gpuTime = measureTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuMatrixMultiply(m1, m2);
            } catch (Exception e) {
                return CPUComputeFloatUtils.matrixMultiply(m1.toFloatArray(), m2.toFloatArray());
            }
        });
        recordResult("Matrix Multiplication", dataSize, cpuTime, gpuTime);
    }
    
    private void testMatrixScalarMultiply(IMatrix<Float> m, float scalar, int dataSize) {
        long cpuTime = measureTime(() -> CPUComputeFloatUtils.matrixScalarMultiply(m.toFloatArray(), scalar));
        long gpuTime = measureTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuMatrixScalarMultiply(m, scalar);
            } catch (Exception e) {
                return CPUComputeFloatUtils.matrixScalarMultiply(m.toFloatArray(), scalar);
            }
        });
        recordResult("Matrix Scalar Multiply", dataSize, cpuTime, gpuTime);
    }
    
    private void testMatrixTranspose(IMatrix<Float> m, int dataSize) {
        long cpuTime = measureTime(() -> CPUComputeFloatUtils.matrixTranspose(m.toFloatArray()));
        long gpuTime = measureTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuMatrixTranspose(m);
            } catch (Exception e) {
                return CPUComputeFloatUtils.matrixTranspose(m.toFloatArray());
            }
        });
        recordResult("Matrix Transpose", dataSize, cpuTime, gpuTime);
    }
    
    private void testPseudoInverse(IMatrix<Float> m, int dataSize) {
        long cpuTime = measureTime(() -> CPUComputeFloatUtils.pseudoInverse(m));
        long gpuTime = measureTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuPseudoInverse(m);
            } catch (Exception e) {
                return CPUComputeFloatUtils.pseudoInverse(m);
            }
        });
        recordResult("Pseudo Inverse", dataSize, cpuTime, gpuTime);
    }
    
    private void testEigenDecomposition(IMatrix<Float> m, int dataSize) {
        long cpuTime = measureEigenTime(() -> CPUComputeFloatUtils.eigen(m));
        long gpuTime = measureEigenTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuEigenDecomposition(m);
            } catch (Exception e) {
                return CPUComputeFloatUtils.eigen(m);
            }
        });
        recordResult("Eigen Decomposition", dataSize, cpuTime, gpuTime);
    }
    
    private void testSVDDecomposition(IMatrix<Float> m, int dataSize) {
        long cpuTime = measureSVDTime(() -> CPUComputeFloatUtils.svd(m));
        long gpuTime = measureGPUSVDTime(() -> {
            try {
                return GPUComputeFloatUtils.gpuSVD(m);
            } catch (Exception e) {
                var cpuResult = CPUComputeFloatUtils.svd(m);
                return cpuResult;
            }
        });
        recordResult("SVD Decomposition", dataSize, cpuTime, gpuTime);
    }
    
    // =========================== 辅助方法 ===========================
    
    private void performJVMWarmup() {
        System.out.println("执行JVM预热...");
        IVector<Float> warmupVector = generateRandomVector(100);
        IMatrix<Float> warmupMatrix = generateRandomMatrix(10, 10);
        
        for (int i = 0; i < 3; i++) {
            try {
                CPUComputeFloatUtils.vectorAdd(warmupVector, warmupVector);
                GPUComputeFloatUtils.gpuVectorAdd(warmupVector, warmupVector);
            } catch (Exception ignored) {}
        }
    }
    
    private IVector<Float> generateRandomVector(int size) {
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = (float) (Math.random() * 10.0 - 5.0);
        }
        return new RereFloatVector(data);
    }
    
    private IMatrix<Float> generateRandomMatrix(int rows, int cols) {
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (float) (Math.random() * 10.0 - 5.0);
            }
        }
        return new RereFloatMatrix(data);
    }
    
    private IMatrix<Float> generateWellConditionedMatrix(int rows, int cols) {
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (i == j) {
                    data[i][j] = 5.0f + (float) (Math.random() * 5.0);
                } else {
                    data[i][j] = (float) (Math.random() * 2.0 - 1.0);
                }
            }
        }
        return new RereFloatMatrix(data);
    }
    
    private long measureTime(Supplier<Object> operation) {
        long totalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            operation.get();
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        return totalTime / BENCHMARK_ITERATIONS;
    }
    
    private long measureScalarTime(ScalarSupplier operation) {
        long totalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            operation.get();
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        return totalTime / BENCHMARK_ITERATIONS;
    }
    
    private long measureSVDTime(SVDSupplier operation) {
        long totalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            operation.get();
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        return totalTime / BENCHMARK_ITERATIONS;
    }
    
    private long measureEigenTime(EigenSupplier operation) {
        long totalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            operation.get();
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        return totalTime / BENCHMARK_ITERATIONS;
    }
    
    private long measureGPUSVDTime(GPUSVDSupplier operation) {
        long totalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            operation.get();
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        return totalTime / BENCHMARK_ITERATIONS;
    }
    
    private void recordResult(String operation, int dataSize, long cpuTime, long gpuTime) {
        double speedup = (double) cpuTime / gpuTime;
        PerformanceResult result = new PerformanceResult(operation, dataSize, cpuTime, gpuTime, speedup);
        performanceResults.add(result);
        
        System.out.printf("%s: CPU %.2fms, GPU %.2fms, 加速比 %.2fx\n", 
            operation, cpuTime / 1_000_000.0, gpuTime / 1_000_000.0, speedup);
    }
    
    private void generateSummaryAnalysis() {
        System.out.println("\n=== 总结分析 ===");
        
        double avgSpeedup = performanceResults.stream()
            .mapToDouble(r -> r.speedup)
            .average().orElse(1.0);
        
        System.out.printf("平均加速比: %.2fx\n", avgSpeedup);
        
        long gpuWins = performanceResults.stream()
            .mapToLong(r -> r.speedup > 1.0 ? 1 : 0)
            .sum();
        
        System.out.printf("GPU优势操作: %d/%d (%.1f%%)\n", 
            gpuWins, performanceResults.size(), 
            (double) gpuWins / performanceResults.size() * 100);
        
        System.out.println("\n推荐使用GPU的场景:");
        performanceResults.stream()
            .filter(r -> r.speedup > 1.5)
            .forEach(r -> System.out.printf("- %s (数据量 >= %d): %.2fx加速\n", 
                r.operation, r.dataSize, r.speedup));
        
        System.out.println("\n推荐使用CPU的场景:");
        performanceResults.stream()
            .filter(r -> r.speedup < 0.8)
            .forEach(r -> System.out.printf("- %s (数据量 %d): CPU快%.2fx\n", 
                r.operation, r.dataSize, 1.0 / r.speedup));
        
        System.out.println("\n高级运算分析:");
        performanceResults.stream()
            .filter(r -> r.operation.contains("Decomposition") || r.operation.contains("Eigen") || r.operation.contains("SVD"))
            .forEach(r -> System.out.printf("- %s: %.2fx加速，适合数据量 >= %d\n", 
                r.operation, r.speedup, r.dataSize));
    }
    
    // =========================== 内部类和接口 ===========================
    
    private static class PerformanceResult {
        final String operation;
        final int dataSize;
        final long cpuTime;
        final long gpuTime;
        final double speedup;
        
        PerformanceResult(String operation, int dataSize, long cpuTime, long gpuTime, double speedup) {
            this.operation = operation;
            this.dataSize = dataSize;
            this.cpuTime = cpuTime;
            this.gpuTime = gpuTime;
            this.speedup = speedup;
        }
    }
    
    @FunctionalInterface
    private interface Supplier<T> {
        T get();
    }
    
    @FunctionalInterface
    private interface ScalarSupplier {
        float get();
    }
    
    @FunctionalInterface
    private interface SVDSupplier {
        Tuple3<IMatrix<Float>, IVector<Float>, IMatrix<Float>> get();
    }
    
    @FunctionalInterface
    private interface EigenSupplier {
        Tuple2<IVector<Float>, IMatrix<Float>> get();
    }
    
    @FunctionalInterface
    private interface GPUSVDSupplier {
        Tuple3<IMatrix<Float>, IVector<Float>, IMatrix<Float>> get();
    }
}