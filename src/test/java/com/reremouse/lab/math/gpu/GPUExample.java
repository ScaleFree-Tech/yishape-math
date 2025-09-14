package com.reremouse.lab.math.gpu;


import com.reremouse.lab.math.linalg.RereMatrix;
import com.reremouse.lab.math.linalg.RereVector;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

/**
 * GPU加速使用示例
 * 
 * 本示例展示如何使用GPU加速的RereMatrix和RereVector进行大规模数值计算。
 * 当GPU不可用时，系统会自动回退到CPU优化算法。
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GPUExample {
    
    public static void main(String[] args) {
        System.out.println("=== GPU加速示例 ===");
        
        // 检查GPU状态
        checkGPUStatus();
        
        // 矩阵乘法GPU加速示例
        matrixMultiplicationExample();
        
        // 向量运算GPU加速示例
        vectorOperationsExample();
        
        // 性能对比示例
        performanceComparisonExample();
        
        // 清理资源
        cleanup();
    }
    
    /**
     * 检查GPU状态
     */
    private static void checkGPUStatus() {
        System.out.println("\n--- GPU状态检查 ---");
        System.out.println("矩阵GPU支持: " + RereMatrix.isGPUEnabled());
        System.out.println("向量GPU支持: " + RereVector.isGPUEnabled());
        System.out.println("GPU信息: " + RereMatrix.getGPUInfo());
    }
    
    /**
     * 矩阵乘法GPU加速示例
     */
    private static void matrixMultiplicationExample() {
        System.out.println("\n--- 矩阵乘法GPU加速示例 ---");
        
        // 创建大矩阵进行测试
        int size = 1000;
        System.out.println("创建 " + size + "x" + size + " 矩阵...");
        
        float[][] dataA = createRandomMatrix(size, size);
        float[][] dataB = createRandomMatrix(size, size);
        
        RereMatrix matrixA = new RereMatrix(dataA);
        RereMatrix matrixB = new RereMatrix(dataB);
        
        // 执行矩阵乘法（自动选择GPU或CPU）
        System.out.println("执行矩阵乘法...");
        long startTime = System.currentTimeMillis();
        
        IMatrix result = matrixA.mmul(matrixB);
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        System.out.println("矩阵乘法完成!");
        System.out.println("计算时间: " + duration + " ms");
        System.out.println("结果矩阵大小: " + result.getRowNum() + "x" + result.getColNum());
        
        // 计算性能指标
        long operations = (long) size * size * size * 2; // 乘法和加法
        double gflops = (operations / 1e9) / (duration / 1000.0);
        System.out.println("性能: " + String.format("%.2f", gflops) + " GFLOPS");
    }
    
    /**
     * 向量运算GPU加速示例
     */
    private static void vectorOperationsExample() {
        System.out.println("\n--- 向量运算GPU加速示例 ---");
        
        // 创建大向量进行测试
        int length = 500000;
        System.out.println("创建长度为 " + length + " 的向量...");
        
        float[] dataA = createRandomVector(length);
        float[] dataB = createRandomVector(length);
        
        RereVector vectorA = new RereVector(dataA);
        RereVector vectorB = new RereVector(dataB);
        
        // 向量加法
        System.out.println("执行向量加法...");
        long startTime = System.currentTimeMillis();
        
        IVector sumResult = vectorA.add(vectorB);
        
        long endTime = System.currentTimeMillis();
        System.out.println("向量加法完成! 耗时: " + (endTime - startTime) + " ms");
        
        // 向量内积
        System.out.println("执行向量内积...");
        startTime = System.currentTimeMillis();
        
        float dotProduct = vectorA.innerProduct(vectorB);
        
        endTime = System.currentTimeMillis();
        System.out.println("向量内积完成! 耗时: " + (endTime - startTime) + " ms");
        System.out.println("内积结果: " + String.format("%.6f", dotProduct));
    }
    
    /**
     * 性能对比示例
     */
    private static void performanceComparisonExample() {
        System.out.println("\n--- 性能对比示例 ---");
        
        int[] sizes = {100, 500, 1000, 2000};
        
        for (int size : sizes) {
            System.out.println("\n测试 " + size + "x" + size + " 矩阵乘法:");
            
            float[][] dataA = createRandomMatrix(size, size);
            float[][] dataB = createRandomMatrix(size, size);
            
            RereMatrix matrixA = new RereMatrix(dataA);
            RereMatrix matrixB = new RereMatrix(dataB);
            
            // 执行多次测试取平均值
            int iterations = 3;
            long totalTime = 0;
            
            for (int i = 0; i < iterations; i++) {
                long startTime = System.currentTimeMillis();
                IMatrix result = matrixA.mmul(matrixB);
                long endTime = System.currentTimeMillis();
                totalTime += (endTime - startTime);
            }
            
            long avgTime = totalTime / iterations;
            long operations = (long) size * size * size * 2;
            double gflops = (operations / 1e9) / (avgTime / 1000.0);
            
            System.out.println("  平均时间: " + avgTime + " ms");
            System.out.println("  性能: " + String.format("%.2f", gflops) + " GFLOPS");
        }
    }
    
    /**
     * 创建随机矩阵
     */
    private static float[][] createRandomMatrix(int rows, int cols) {
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (float) (Math.random() * 2.0 - 1.0); // -1 到 1 之间的随机数
            }
        }
        return matrix;
    }
    
    /**
     * 创建随机向量
     */
    private static float[] createRandomVector(int length) {
        float[] vector = new float[length];
        for (int i = 0; i < length; i++) {
            vector[i] = (float) (Math.random() * 2.0 - 1.0); // -1 到 1 之间的随机数
        }
        return vector;
    }
    
    /**
     * 清理资源
     */
    private static void cleanup() {
        System.out.println("\n--- 清理资源 ---");
        RereMatrix.shutdown();
        RereVector.shutdown();
        System.out.println("资源清理完成!");
    }
    
    /**
     * 高级GPU使用示例
     */
    public static void advancedGPUExample() {
        System.out.println("\n=== 高级GPU使用示例 ===");
        
        // 检查GPU是否可用
        if (!RereMatrix.isGPUEnabled()) {
            System.out.println("GPU不可用，将使用CPU优化算法");
            return;
        }
        
        // 创建超大矩阵测试GPU极限
        int[] testSizes = {2000, 3000, 4000};
        
        for (int size : testSizes) {
            System.out.println("\n测试 " + size + "x" + size + " 超大矩阵:");
            
            try {
                float[][] dataA = createRandomMatrix(size, size);
                float[][] dataB = createRandomMatrix(size, size);
                
                RereMatrix matrixA = new RereMatrix(dataA);
                RereMatrix matrixB = new RereMatrix(dataB);
                
                long startTime = System.currentTimeMillis();
                IMatrix result = matrixA.mmul(matrixB);
                long endTime = System.currentTimeMillis();
                
                long duration = endTime - startTime;
                long operations = (long) size * size * size * 2;
                double gflops = (operations / 1e9) / (duration / 1000.0);
                
                System.out.println("  计算时间: " + duration + " ms");
                System.out.println("  性能: " + String.format("%.2f", gflops) + " GFLOPS");
                System.out.println("  内存使用: " + String.format("%.2f", (size * size * 4 * 3) / 1024.0 / 1024.0) + " MB");
                
            } catch (OutOfMemoryError e) {
                System.out.println("  矩阵太大，内存不足: " + e.getMessage());
                break;
            } catch (Exception e) {
                System.out.println("  计算失败: " + e.getMessage());
            }
        }
    }
    
    /**
     * 向量GPU性能测试
     */
    public static void vectorGPUPerformanceTest() {
        System.out.println("\n=== 向量GPU性能测试 ===");
        
        int[] lengths = {100000, 500000, 1000000, 2000000};
        
        for (int length : lengths) {
            System.out.println("\n测试长度为 " + length + " 的向量:");
            
            try {
                float[] dataA = createRandomVector(length);
                float[] dataB = createRandomVector(length);
                
                RereVector vectorA = new RereVector(dataA);
                RereVector vectorB = new RereVector(dataB);
                
                // 向量加法测试
                long startTime = System.currentTimeMillis();
                IVector sumResult = vectorA.add(vectorB);
                long endTime = System.currentTimeMillis();
                
                System.out.println("  向量加法: " + (endTime - startTime) + " ms");
                
                // 向量内积测试
                startTime = System.currentTimeMillis();
                float dotProduct = vectorA.innerProduct(vectorB);
                endTime = System.currentTimeMillis();
                
                System.out.println("  向量内积: " + (endTime - startTime) + " ms");
                System.out.println("  内积结果: " + String.format("%.6f", dotProduct));
                
            } catch (OutOfMemoryError e) {
                System.out.println("  向量太大，内存不足: " + e.getMessage());
                break;
            } catch (Exception e) {
                System.out.println("  计算失败: " + e.getMessage());
            }
        }
    }
}
