package com.yishape.lab.math.compute;

import com.yishape.lab.math.linalg.RereDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.viz.RerePlot;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Performance comparison test for GPUFloatComputer, SIMDFloatComputer, and SISDFloatComputer
 * This test benchmarks the performance of different computer implementations across various operations
 * and data sizes, then visualizes the results using grouped bar charts.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ComputerPerformanceComparisonTest {
    
    // Computer instances
    private static final GPUFloatComputer gpuComputer = new GPUFloatComputer();
    private static final SIMDFloatComputer simdComputer = new SIMDFloatComputer();
    private static final SISDFloatComputer sisdComputer = new SISDFloatComputer();
    
    // Test data sizes
    private static final int[] VECTOR_SIZES = {1000, 10000, 100000, 1000000};
    private static final int[] MATRIX_SIZES = {50, 100, 200, 500};
    
    // Performance results storage
    private static final List<PerformanceResult> performanceResults = new ArrayList<>();
    
    // Number of iterations for averaging
    private static final int ITERATIONS = 5;
    
    @BeforeAll
    static void setUp() {
        System.out.println("=== Computer Performance Comparison Test ===");
        System.out.println("GPU Available: " + GPUFloatComputer.isGPUAvailable());
        System.out.println("SIMD Support: " + SIMDFloatComputer.checkIfSupport());
        
        // Warm up JVM
        performJVMWarmup();
    }
    
    private static void performJVMWarmup() {
        System.out.println("Performing JVM warmup...");
        
        // Generate test data
        float[] warmupVector1 = generateRandomArray(1000);
        float[] warmupVector2 = generateRandomArray(1000);
        
        // Run a few iterations of each operation to warm up
        for (int i = 0; i < 3; i++) {
            // Vector addition
            gpuComputer.binaryOperate(warmupVector1, warmupVector2, IFloatVectorComputer.BinaryOperation.ADD);
            simdComputer.binaryOperate(warmupVector1, warmupVector2, IFloatVectorComputer.BinaryOperation.ADD);
            sisdComputer.binaryOperate(warmupVector1, warmupVector2, IFloatVectorComputer.BinaryOperation.ADD);
            
            // Vector multiplication
            gpuComputer.binaryOperate(warmupVector1, warmupVector2, IFloatVectorComputer.BinaryOperation.MULTIPLY);
            simdComputer.binaryOperate(warmupVector1, warmupVector2, IFloatVectorComputer.BinaryOperation.MULTIPLY);
            sisdComputer.binaryOperate(warmupVector1, warmupVector2, IFloatVectorComputer.BinaryOperation.MULTIPLY);
        }
        
        System.out.println("JVM warmup completed.");
    }
    
    private static float[] generateRandomArray(int size) {
        Random random = new Random(42); // Fixed seed for reproducibility
        float[] array = new float[size];
        for (int i = 0; i < size; i++) {
            array[i] = random.nextFloat() * 100.0f;
        }
        return array;
    }
    
    private static float[][] generateRandomMatrix(int rows, int cols) {
        Random random = new Random(42); // Fixed seed for reproducibility
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextFloat() * 100.0f;
            }
        }
        return matrix;
    }
    
    private long measureTime(Runnable operation) {
        // Warm up
        for (int i = 0; i < 2; i++) {
            operation.run();
        }
        
        // Measure
        long totalTime = 0;
        for (int i = 0; i < ITERATIONS; i++) {
            long startTime = System.nanoTime();
            operation.run();
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        
        return totalTime / ITERATIONS;
    }
    
    private void recordResult(String operation, String computerType, int dataSize, long time) {
        PerformanceResult result = new PerformanceResult(operation, computerType, dataSize, time);
        performanceResults.add(result);
        System.out.printf("%-20s | %-10s | Size: %7d | Time: %8.2f ms%n", 
            operation, computerType, dataSize, time / 1_000_000.0);
    }
    
    @Test
    @DisplayName("Vector Addition Performance Comparison")
    void testVectorAdditionPerformance() {
        System.out.println("\n=== Vector Addition Performance ===");
        
        for (int size : VECTOR_SIZES) {
            float[] vector1 = generateRandomArray(size);
            float[] vector2 = generateRandomArray(size);
            
            // GPU Computer
            long gpuTime = measureTime(() -> 
                gpuComputer.binaryOperate(vector1, vector2, IFloatVectorComputer.BinaryOperation.ADD));
            recordResult("Vector Add", "GPU", size, gpuTime);
            
            // SIMD Computer
            long simdTime = measureTime(() -> 
                simdComputer.binaryOperate(vector1, vector2, IFloatVectorComputer.BinaryOperation.ADD));
            recordResult("Vector Add", "SIMD", size, simdTime);
            
            // SISD Computer
            long sisdTime = measureTime(() -> 
                sisdComputer.binaryOperate(vector1, vector2, IFloatVectorComputer.BinaryOperation.ADD));
            recordResult("Vector Add", "SISD", size, sisdTime);
        }
    }
    
    @Test
    @DisplayName("Vector Multiplication Performance Comparison")
    void testVectorMultiplicationPerformance() {
        System.out.println("\n=== Vector Multiplication Performance ===");
        
        for (int size : VECTOR_SIZES) {
            float[] vector1 = generateRandomArray(size);
            float[] vector2 = generateRandomArray(size);
            
            // GPU Computer
            long gpuTime = measureTime(() -> 
                gpuComputer.binaryOperate(vector1, vector2, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Vector Mul", "GPU", size, gpuTime);
            
            // SIMD Computer
            long simdTime = measureTime(() -> 
                simdComputer.binaryOperate(vector1, vector2, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Vector Mul", "SIMD", size, simdTime);
            
            // SISD Computer
            long sisdTime = measureTime(() -> 
                sisdComputer.binaryOperate(vector1, vector2, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Vector Mul", "SISD", size, sisdTime);
        }
    }
    
    @Test
    @DisplayName("Vector Scalar Multiplication Performance Comparison")
    void testVectorScalarMultiplicationPerformance() {
        System.out.println("\n=== Vector Scalar Multiplication Performance ===");
        
        float scalar = 3.14159f;
        
        for (int size : VECTOR_SIZES) {
            float[] vector = generateRandomArray(size);
            
            // GPU Computer
            long gpuTime = measureTime(() -> 
                gpuComputer.binaryOperate(vector, scalar, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Vector Scalar Mul", "GPU", size, gpuTime);
            
            // SIMD Computer
            long simdTime = measureTime(() -> 
                simdComputer.binaryOperate(vector, scalar, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Vector Scalar Mul", "SIMD", size, simdTime);
            
            // SISD Computer
            long sisdTime = measureTime(() -> 
                sisdComputer.binaryOperate(vector, scalar, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Vector Scalar Mul", "SISD", size, sisdTime);
        }
    }
    
    @Test
    @DisplayName("Matrix Addition Performance Comparison")
    void testMatrixAdditionPerformance() {
        System.out.println("\n=== Matrix Addition Performance ===");
        
        for (int size : MATRIX_SIZES) {
            float[][] matrix1 = generateRandomMatrix(size, size);
            float[][] matrix2 = generateRandomMatrix(size, size);
            
            // GPU Computer
            long gpuTime = measureTime(() -> 
                gpuComputer.binaryOperate(matrix1, matrix2, IFloatVectorComputer.BinaryOperation.ADD));
            recordResult("Matrix Add", "GPU", size * size, gpuTime);
            
            // SIMD Computer
            long simdTime = measureTime(() -> 
                simdComputer.binaryOperate(matrix1, matrix2, IFloatVectorComputer.BinaryOperation.ADD));
            recordResult("Matrix Add", "SIMD", size * size, simdTime);
            
            // SISD Computer
            long sisdTime = measureTime(() -> 
                sisdComputer.binaryOperate(matrix1, matrix2, IFloatVectorComputer.BinaryOperation.ADD));
            recordResult("Matrix Add", "SISD", size * size, sisdTime);
        }
    }
    
    @Test
    @DisplayName("Matrix Multiplication Performance Comparison")
    void testMatrixMultiplicationPerformance() {
        System.out.println("\n=== Matrix Multiplication Performance ===");
        
        for (int size : MATRIX_SIZES) {
            float[][] matrix1 = generateRandomMatrix(size, size);
            float[][] matrix2 = generateRandomMatrix(size, size);
            
            // GPU Computer
            long gpuTime = measureTime(() -> 
                gpuComputer.binaryOperate(matrix1, matrix2, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Matrix Mul", "GPU", size * size, gpuTime);
            
            // SIMD Computer
            long simdTime = measureTime(() -> 
                simdComputer.binaryOperate(matrix1, matrix2, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Matrix Mul", "SIMD", size * size, simdTime);
            
            // SISD Computer
            long sisdTime = measureTime(() -> 
                sisdComputer.binaryOperate(matrix1, matrix2, IFloatVectorComputer.BinaryOperation.MULTIPLY));
            recordResult("Matrix Mul", "SISD", size * size, sisdTime);
        }
    }
    
    @Test
    @DisplayName("Universal Operation (SQRT) Performance Comparison")
    void testUniversalOperationPerformance() {
        System.out.println("\n=== Universal Operation (SQRT) Performance ===");
        
        for (int size : VECTOR_SIZES) {
            float[] vector = generateRandomArray(size);
            
            // GPU Computer
            long gpuTime = measureTime(() -> 
                gpuComputer.universalOperate(vector, IFloatVectorComputer.UniversalOperation.SQRT, 0.0f));
            recordResult("Universal SQRT", "GPU", size, gpuTime);
            
            // SIMD Computer
            long simdTime = measureTime(() -> 
                simdComputer.universalOperate(vector, IFloatVectorComputer.UniversalOperation.SQRT, 0.0f));
            recordResult("Universal SQRT", "SIMD", size, simdTime);
            
            // SISD Computer
            long sisdTime = measureTime(() -> 
                sisdComputer.universalOperate(vector, IFloatVectorComputer.UniversalOperation.SQRT, 0.0f));
            recordResult("Universal SQRT", "SISD", size, sisdTime);
        }
    }
    
    @Test
    @DisplayName("Reduce Operation (SUM) Performance Comparison")
    void testReduceOperationPerformance() {
        System.out.println("\n=== Reduce Operation (SUM) Performance ===");
        
        for (int size : VECTOR_SIZES) {
            float[] vector = generateRandomArray(size);
            
            // GPU Computer
            long gpuTime = measureTime(() -> 
                gpuComputer.reduceOperate(vector, IFloatVectorComputer.ReduceOperation.SUM));
            recordResult("Reduce SUM", "GPU", size, gpuTime);
            
            // SIMD Computer
            long simdTime = measureTime(() -> 
                simdComputer.reduceOperate(vector, IFloatVectorComputer.ReduceOperation.SUM));
            recordResult("Reduce SUM", "SIMD", size, simdTime);
            
            // SISD Computer
            long sisdTime = measureTime(() -> 
                sisdComputer.reduceOperate(vector, IFloatVectorComputer.ReduceOperation.SUM));
            recordResult("Reduce SUM", "SISD", size, sisdTime);
        }
    }
    
    @Test
    @DisplayName("Generate Performance Comparison Charts")
    void generatePerformanceCharts() {
        System.out.println("\n=== Generating Performance Comparison Charts ===");
        
        // Group results by operation
        List<String> operations = new ArrayList<>();
        for (PerformanceResult result : performanceResults) {
            if (!operations.contains(result.operation)) {
                operations.add(result.operation);
            }
        }
        
        // Create a chart for each operation
        for (String operation : operations) {
            createOperationChart(operation);
        }
        
        // Create summary chart
        createSummaryChart();
    }
    
    private void createOperationChart(String operation) {
        // Filter results for this operation
        List<PerformanceResult> opResults = new ArrayList<>();
        for (PerformanceResult result : performanceResults) {
            if (result.operation.equals(operation)) {
                opResults.add(result);
            }
        }
        
        if (opResults.isEmpty()) {
            return;
        }
        
        // Get unique data sizes
        List<Integer> sizes = new ArrayList<>();
        for (PerformanceResult result : opResults) {
            if (!sizes.contains(result.dataSize)) {
                sizes.add(result.dataSize);
            }
        }
        sizes.sort(Integer::compareTo);
        
        // Create data for chart
        // For grouped bar charts, we need to organize data by computer type
        List<Float> gpuTimes = new ArrayList<>();
        List<Float> simdTimes = new ArrayList<>();
        List<Float> sisdTimes = new ArrayList<>();
        
        // Collect data organized by computer type
        for (Integer size : sizes) {
            float gpuTime = 0;
            float simdTime = 0;
            float sisdTime = 0;
            
            for (PerformanceResult result : opResults) {
                if (result.dataSize == size) {
                    switch (result.computerType) {
                        case "GPU":
                            gpuTime = result.time / 1_000_000.0f; // Convert to milliseconds
                            break;
                        case "SIMD":
                            simdTime = result.time / 1_000_000.0f;
                            break;
                        case "SISD":
                            sisdTime = result.time / 1_000_000.0f;
                            break;
                    }
                }
            }
            
            gpuTimes.add(gpuTime);
            simdTimes.add(simdTime);
            sisdTimes.add(sisdTime);
        }
        
        // Create grouped bar chart
        generateGroupedBarChart(sizes, gpuTimes, simdTimes, sisdTimes, operation);
    }
    
    private void generateGroupedBarChart(List<Integer> sizes, List<Float> gpuTimes, List<Float> simdTimes, List<Float> sisdTimes, String operation) {
        try {
            // Create grouped bar chart
            // For proper grouped bar charts, we need to structure the data correctly
            // The x-axis should show data sizes (the groups)
            // Each group should contain bars for GPU, SIMD, and SISD
            
            // Create x-axis labels (one for each data size)
            List<String> xLabels = new ArrayList<>();
            for (int i = 0; i < sizes.size(); i++) {
                xLabels.add(sizes.get(i).toString());
            }
            
            // After analyzing the RerePlot library implementation more carefully,
            // I realize that the current approach is not working correctly.
            // Let's try a different approach by creating separate series for each computer type.
            
            // Create data vectors for each computer type
            double[] gpuData = new double[sizes.size()];
            double[] simdData = new double[sizes.size()];
            double[] sisdData = new double[sizes.size()];
            
            // Fill in the data for each computer type
            for (int i = 0; i < sizes.size(); i++) {
                gpuData[i] = gpuTimes.get(i);
                simdData[i] = simdTimes.get(i);
                sisdData[i] = sisdTimes.get(i);
            }
            
            // Convert to IVector format
            IVector<Double> gpuVector = new RereDoubleVector(gpuData);
            IVector<Double> simdVector = new RereDoubleVector(simdData);
            IVector<Double> sisdVector = new RereDoubleVector(sisdData);
            
            // Create the chart using the proper API for grouped bar charts
            RerePlot plot = new RerePlot(1000, 600);
            
            // For grouped bar charts, we need to provide all the data in a specific format
            // Let's try creating a single vector with all the data and corresponding hue labels
            List<Float> allTimes = new ArrayList<>();
            List<String> hueLabels = new ArrayList<>();
            
            // Add data in the correct order for grouped bars
            for (int i = 0; i < sizes.size(); i++) {
                allTimes.add(gpuTimes.get(i));
                hueLabels.add("GPU");
                
                allTimes.add(simdTimes.get(i));
                hueLabels.add("SIMD");
                
                allTimes.add(sisdTimes.get(i));
                hueLabels.add("SISD");
            }
            
            // Convert to IVector format
            IVector<Double> allTimesVector = IVector.of(allTimes.stream().mapToDouble(Float::doubleValue).toArray());
            
            // Create the chart using the proper API for grouped bar charts
            plot.bar(xLabels, allTimesVector, hueLabels)
                .title(operation + " Performance Comparison")
                .xlabel("Data Size")
                .ylabel("Time (ms)");
            
            // Save the chart
            String filename = "temp/" + operation.replace(" ", "_") + "_performance_comparison.html";
            plot.saveAsHtml(filename);
            System.out.println("Chart saved to: " + filename);
        } catch (Exception e) {
            System.err.println("Error generating chart for " + operation + ": " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void createSummaryChart() {
        // Calculate average performance for each computer type
        double gpuTotal = 0;
        double simdTotal = 0;
        double sisdTotal = 0;
        int gpuCount = 0;
        int simdCount = 0;
        int sisdCount = 0;
        
        for (PerformanceResult result : performanceResults) {
            switch (result.computerType) {
                case "GPU":
                    gpuTotal += result.time / 1_000_000.0;
                    gpuCount++;
                    break;
                case "SIMD":
                    simdTotal += result.time / 1_000_000.0;
                    simdCount++;
                    break;
                case "SISD":
                    sisdTotal += result.time / 1_000_000.0;
                    sisdCount++;
                    break;
            }
        }
        
        double gpuAvg = gpuCount > 0 ? gpuTotal / gpuCount : 0;
        double simdAvg = simdCount > 0 ? simdTotal / simdCount : 0;
        double sisdAvg = sisdCount > 0 ? sisdTotal / sisdCount : 0;
        
        // Create summary chart
        try {
            IVector<Double> avgTimes = new RereDoubleVector(new double[]{gpuAvg, simdAvg, sisdAvg});
            List<String> labels = List.of("GPU", "SIMD", "SISD");
            List<String> xLabels = List.of("GPU", "SIMD", "SISD");
            
            RerePlot plot = new RerePlot(800, 500);
            plot.bar(xLabels, avgTimes, labels)
                .title("Average Performance Comparison Across All Operations")
                .xlabel("Computer Type")
                .ylabel("Average Time (ms)");
            
            String filename = "temp/average_performance_comparison.html";
            plot.saveAsHtml(filename);
            System.out.println("Summary chart saved to: " + filename);
            
        } catch (Exception e) {
            System.err.println("Error creating summary chart: " + e.getMessage());
        }
    }
    
    /**
     * Inner class to store performance results
     */
    private static class PerformanceResult {
        final String operation;
        final String computerType;
        final int dataSize;
        final long time;
        
        PerformanceResult(String operation, String computerType, int dataSize, long time) {
            this.operation = operation;
            this.computerType = computerType;
            this.dataSize = dataSize;
            this.time = time;
        }
    }
}