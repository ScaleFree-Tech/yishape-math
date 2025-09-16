package com.reremouse.lab.math.gpu;

import com.reremouse.lab.math.compute.GPUConfig;
import com.reremouse.lab.math.compute.GPUComputeFloatUtils;
import com.reremouse.lab.math.linalg.RereFloatMatrix;
import com.reremouse.lab.math.compute.CPUComputeFloatUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.BeforeEach;
import com.reremouse.lab.math.linalg.IMatrix;

/**
 * GPU Matrix Scalar Multiply Performance Analysis Test
 * 分析GPU矩阵标量乘法性能瓶颈
 */
public class GPUMatrixScalarMultiplyPerformanceTest {
    
    @BeforeEach
    void setUp() {
        // 启用详细日志以观察性能行为
        GPUComputeFloatUtils.setLoggingEnabled(true);
        GPUComputeFloatUtils.setDetailedLoggingEnabled(true);
    }
    
    @Test
    @DisplayName("GPU矩阵标量乘法性能瓶颈分析")
    void analyzeGPUMatrixScalarMultiplyBottlenecks() {
        System.out.println("=== GPU矩阵标量乘法性能瓶颈分析 ===");
        System.out.println("GPU阈值: " + GPUConfig.GPU_THRESHOLD);
        System.out.println();
        
        // 测试不同大小的矩阵
        int[] sizes = {100, 500, 1000, 1500, 2000}; // 从小到大测试
        float scalar = 2.5f;
        
        for (int size : sizes) {
            long dataSize = (long) size * size;
            System.out.println("--- 测试矩阵大小: " + size + "x" + size + 
                              " (数据量: " + dataSize + ") ---");
            
            // 创建测试矩阵
            float[][] data = createTestMatrix(size);
            IMatrix<Float> matrix = new RereFloatMatrix(data);
            
            // 1. 分析数据传输成本
            analyzeDataTransferCost(matrix, size);
            
            // 2. CPU vs GPU性能对比
            compareCPUvsGPUPerformance(matrix, scalar, size, dataSize);
            
            // 3. 分析GPU内存开销
            analyzeGPUMemoryOverhead(matrix, size);
            
            System.out.println();
        }
        
        // 4. 分析GPU操作的各个阶段
        analyzeGPUOperationStages();
        
        // 5. 提供优化建议
        provideOptimizationRecommendations();
    }
    
    /**
     * 创建测试矩阵
     */
    private float[][] createTestMatrix(int size) {
        float[][] data = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                data[i][j] = (float) (Math.random() * 100); // 随机值0-100
            }
        }
        return data;
    }
    
    /**
     * 分析数据传输成本
     */
    private void analyzeDataTransferCost(IMatrix<Float> matrix, int size) {
        System.out.println("📊 数据传输成本分析:");
        
        long startTime = System.nanoTime();
        
        // 模拟GPU计算中的数据转换步骤
        float[][] originalData = matrix.toFloatArray();
        
        // 1. 2D -> 1D 转换 (flattenMatrix)
        long flattenStart = System.nanoTime();
        float[] flatA = flattenMatrix(originalData);
        long flattenTime = System.nanoTime() - flattenStart;
        
        // 2. 创建结果数组
        long allocStart = System.nanoTime();
        float[] flatResult = new float[size * size];
        float[][] resultData = new float[size][size];
        long allocTime = System.nanoTime() - allocStart;
        
        // 3. 1D -> 2D 转换 (unflattenMatrix) 
        long unflattenStart = System.nanoTime();
        resultData = unflattenMatrix(flatResult, size, size);
        long unflattenTime = System.nanoTime() - unflattenStart;
        
        long totalTransferTime = System.nanoTime() - startTime;
        
        System.out.println("  - Flatten操作耗时: " + (flattenTime / 1_000_000.0) + " ms");
        System.out.println("  - 内存分配耗时: " + (allocTime / 1_000_000.0) + " ms");
        System.out.println("  - Unflatten操作耗时: " + (unflattenTime / 1_000_000.0) + " ms");
        System.out.println("  - 数据传输总耗时: " + (totalTransferTime / 1_000_000.0) + " ms");
        System.out.println("  - 内存使用: " + (size * size * 4 * 3 / 1024 / 1024) + " MB (原始+平化+结果)");
    }
    
    /**
     * CPU vs GPU性能对比
     */
    private void compareCPUvsGPUPerformance(IMatrix<Float> matrix, float scalar, int size, long dataSize) {
        System.out.println("⚡ CPU vs GPU性能对比:");
        
        // CPU性能测试
        long cpuStart = System.nanoTime();
        IMatrix<Float> cpuResult = CPUComputeFloatUtils.matrixScalarMultiply(matrix.toFloatArray(), scalar);
        long cpuTime = System.nanoTime() - cpuStart;
        
        // GPU性能测试
        long gpuStart = System.nanoTime();
        IMatrix<Float> gpuResult;
        boolean usedGPU = false;
        try {
            if (dataSize >= GPUConfig.GPU_THRESHOLD) {
                gpuResult = GPUComputeFloatUtils.gpuMatrixScalarMultiply(matrix, scalar);
                usedGPU = true;
            } else {
                gpuResult = CPUComputeFloatUtils.matrixScalarMultiply(matrix.toFloatArray(), scalar);
                usedGPU = false;
            }
        } catch (Exception e) {
            gpuResult = CPUComputeFloatUtils.matrixScalarMultiply(matrix.toFloatArray(), scalar);
            usedGPU = false;
        }
        long gpuTime = System.nanoTime() - gpuStart;
        
        double cpuMs = cpuTime / 1_000_000.0;
        double gpuMs = gpuTime / 1_000_000.0;
        double speedup = cpuMs / gpuMs;
        
        System.out.println("  - CPU耗时: " + String.format("%.3f", cpuMs) + " ms");
        System.out.println("  - GPU耗时: " + String.format("%.3f", gpuMs) + " ms " + 
                          (usedGPU ? "(实际使用GPU)" : "(回退到CPU)"));
        System.out.println("  - 加速比: " + String.format("%.2f", speedup) + "x " + 
                          (speedup > 1 ? "(GPU更快)" : "(CPU更快)"));
        
        // 验证结果正确性
        boolean resultsMatch = compareMatrices(cpuResult, gpuResult);
        System.out.println("  - 结果一致性: " + (resultsMatch ? "✅ 一致" : "❌ 不一致"));
        
        // 分析性能问题
        if (usedGPU && speedup < 1.0) {
            System.out.println("  ⚠️  性能问题检测: GPU比CPU慢 " + String.format("%.2f", 1/speedup) + "x");
            analyzePerformanceIssues(size, cpuMs, gpuMs);
        }
    }
    
    /**
     * 分析GPU内存开销
     */
    private void analyzeGPUMemoryOverhead(IMatrix<Float> matrix, int size) {
        System.out.println("💾 GPU内存开销分析:");
        
        long elements = (long) size * size;
        long originalMemory = elements * 4; // float = 4 bytes
        long flattenedMemory = elements * 4; // flattened array
        long resultMemory = elements * 4;   // result array
        long totalMemory = originalMemory + flattenedMemory + resultMemory;
        
        System.out.println("  - 原始矩阵内存: " + (originalMemory / 1024 / 1024) + " MB");
        System.out.println("  - 平化数组内存: " + (flattenedMemory / 1024 / 1024) + " MB");
        System.out.println("  - 结果数组内存: " + (resultMemory / 1024 / 1024) + " MB");
        System.out.println("  - 总内存开销: " + (totalMemory / 1024 / 1024) + " MB");
        System.out.println("  - 内存放大倍数: " + String.format("%.1f", (double)totalMemory / originalMemory) + "x");
        
        // 分析Aparapi开销
        System.out.println("  - Aparapi框架开销: Kernel创建、编译、执行、清理");
        System.out.println("  - GPU设备切换开销: CPU->GPU数据传输");
    }
    
    /**
     * 分析GPU操作的各个阶段
     */
    private void analyzeGPUOperationStages() {
        System.out.println("🔍 GPU操作阶段分析:");
        System.out.println("  1. 数据准备阶段:");
        System.out.println("     - matrix.getData() 获取2D数组");
        System.out.println("     - flattenMatrix() 转换为1D数组");
        System.out.println("     - 分配结果数组内存");
        System.out.println("  2. GPU计算阶段:");
        System.out.println("     - 创建Aparapi Kernel对象");
        System.out.println("     - Range.create2D() 创建2D范围");
        System.out.println("     - kernel.execute() 执行GPU计算");
        System.out.println("  3. 数据回收阶段:");
        System.out.println("     - unflattenMatrix() 转换回2D数组");
        System.out.println("     - new RereMatrix() 创建结果对象");
        System.out.println("     - kernel.dispose() 清理GPU资源");
        System.out.println();
        System.out.println("💡 关键发现:");
        System.out.println("  - 矩阵标量乘法是极简单操作: result[i] = input[i] * scalar");
        System.out.println("  - CPU可以直接在2D数组上操作，无需数据转换");
        System.out.println("  - GPU需要额外的flatten/unflatten开销");
        System.out.println("  - Aparapi JTP模式实际上是多线程CPU计算");
    }
    
    /**
     * 提供优化建议
     */
    private void provideOptimizationRecommendations() {
        System.out.println("🚀 优化建议:");
        System.out.println("  1. 提高GPU阈值:");
        System.out.println("     - 当前阈值: " + GPUConfig.GPU_THRESHOLD + " 元素");
        System.out.println("     - 建议阈值: 10,000,000+ 元素 (对于标量运算)");
        System.out.println("     - 原因: 数据转换开销超过计算收益");
        System.out.println();
        System.out.println("  2. 算法优化:");
        System.out.println("     - 避免不必要的2D<->1D转换");
        System.out.println("     - 考虑使用1D Range而非2D Range");
        System.out.println("     - 减少内存分配次数");
        System.out.println();
        System.out.println("  3. 应用场景优化:");
        System.out.println("     - 矩阵标量运算适合CPU (简单操作)");
        System.out.println("     - GPU更适合复杂的矩阵乘法运算");
        System.out.println("     - 考虑批量操作以摊销GPU初始化成本");
        System.out.println();
        System.out.println("  4. 架构改进:");
        System.out.println("     - 实现GPU数据缓存机制");
        System.out.println("     - 提供专门的1D数组GPU操作");
        System.out.println("     - 考虑使用更高效的GPU计算框架");
    }
    
    /**
     * 分析具体的性能问题
     */
    private void analyzePerformanceIssues(int size, double cpuMs, double gpuMs) {
        System.out.println("  🔬 性能问题详细分析:");
        
        double overhead = gpuMs - cpuMs;
        double overheadPercentage = (overhead / gpuMs) * 100;
        
        System.out.println("    - GPU额外开销: " + String.format("%.3f", overhead) + " ms");
        System.out.println("    - 开销占比: " + String.format("%.1f", overheadPercentage) + "%");
        
        // 估算开销来源
        double estimatedDataTransfer = (size * size * 4 * 3) / (1024.0 * 1024.0) * 0.1; // 假设10ms/MB
        double estimatedKernelOverhead = 2.0; // 假设2ms内核开销
        
        System.out.println("    - 估算数据传输开销: " + String.format("%.3f", estimatedDataTransfer) + " ms");
        System.out.println("    - 估算内核创建开销: " + String.format("%.3f", estimatedKernelOverhead) + " ms");
        System.out.println("    - 其他开销: " + String.format("%.3f", overhead - estimatedDataTransfer - estimatedKernelOverhead) + " ms");
    }
    
    // 辅助方法
    private float[] flattenMatrix(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, flat, i * cols, cols);
        }
        return flat;
    }
    
    private float[][] unflattenMatrix(float[] flat, int rows, int cols) {
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, matrix[i], 0, cols);
        }
        return matrix;
    }
    
    private boolean compareMatrices(IMatrix<Float> a, IMatrix<Float> b) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            return false;
        }
        
        float tolerance = 1e-6f;
        for (int i = 0; i < a.rows(); i++) {
            for (int j = 0; j < a.cols(); j++) {
                if (Math.abs(a.get(i, j) - b.get(i, j)) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }
}