package com.yishape.lab.math.compute;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.Random;

/**
 * Small performance test to check GPU functionality with larger data sets
 */
public class SmallPerformanceTest {
    
    // Computer instances
    private static final GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
    private static final SIMDDoubleComputer simdComputer = new SIMDDoubleComputer();
    private static final SISDDoubleComputer sisdComputer = new SISDDoubleComputer();
    
    @BeforeAll
    static void setUp() {
        System.out.println("=== Small Performance Test ===");
        System.out.println("GPU Available: " + GPUDoubleComputer.isGPUAvailable());
        System.out.println("SIMD Support: " + SIMDDoubleComputer.checkIfSupport());
    }
    
    private static double[] generateRandomArray(int size) {
        Random random = new Random(42); // Fixed seed for reproducibility
        double[] array = new double[size];
        for (int i = 0; i < size; i++) {
            array[i] = random.nextDouble() * 100.0;
        }
        return array;
    }
    
    private long measureTime(Runnable operation) {
        // Warm up
        for (int i = 0; i < 2; i++) {
            operation.run();
        }
        
        // Measure
        long totalTime = 0;
        int iterations = 3;
        for (int i = 0; i < iterations; i++) {
            long startTime = System.nanoTime();
            operation.run();
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
        }
        
        return totalTime / iterations;
    }
    
    @Test
    @DisplayName("Small Vector Addition Performance Comparison")
    void testSmallVectorAdditionPerformance() {
        System.out.println("\n=== Small Vector Addition Performance ===");
        
        int[] sizes = {10000, 50000}; // Smaller sizes for quick testing
        
        for (int size : sizes) {
            double[] vector1 = generateRandomArray(size);
            double[] vector2 = generateRandomArray(size);
            
            // GPU Computer
            long gpuTime = measureTime(() -> 
                gpuComputer.binaryOperate(vector1, vector2, IDoubleVectorComputer.BinaryOperation.ADD));
            System.out.printf("GPU    | Size: %6d | Time: %8.2f ms%n", size, gpuTime / 1_000_000.0);
            
            // SIMD Computer
            long simdTime = measureTime(() -> 
                simdComputer.binaryOperate(vector1, vector2, IDoubleVectorComputer.BinaryOperation.ADD));
            System.out.printf("SIMD   | Size: %6d | Time: %8.2f ms%n", size, simdTime / 1_000_000.0);
            
            // SISD Computer
            long sisdTime = measureTime(() -> 
                sisdComputer.binaryOperate(vector1, vector2, IDoubleVectorComputer.BinaryOperation.ADD));
            System.out.printf("SISD   | Size: %6d | Time: %8.2f ms%n", size, sisdTime / 1_000_000.0);
        }
        
        System.out.println("Small performance test completed successfully!");
    }
}