package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SimpleDirectGPUArrayTest {
    
    @Test
    public void testGPUSqrtWithDirectArrays() {
        // This test demonstrates that GPUDoubleComputer works with direct double arrays
        // without any IVector interfaces or complex dependencies
        
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test SQRT operation which was causing issues
        double[] input = {4.0, 9.0, 16.0, 25.0, 36.0};
        double[] expected = {2.0, 3.0, 4.0, 5.0, 6.0};
        
        // This should work without crashing
        double[] result = gpuComputer.universalOperate(input, IDoubleVectorComputer.UniversalOperation.SQRT, 0.0);
        
        // Verify results
        assertEquals(expected.length, result.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], result[i], 0.001, "SQRT(" + input[i] + ") should be " + expected[i]);
        }
        
        System.out.println("GPU SQRT operation with direct arrays: SUCCESS");
        System.out.println("Input: " + java.util.Arrays.toString(input));
        System.out.println("Output: " + java.util.Arrays.toString(result));
    }
    
    @Test
    public void testGPUVectorOperationsWithDirectArrays() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test vector addition
        double[] x1 = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] x2 = {5.0, 4.0, 3.0, 2.0, 1.0};
        double[] expected = {6.0, 6.0, 6.0, 6.0, 6.0};
        
        double[] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        assertArrayEquals(expected, result, 0.001, "Vector addition failed");
        
        System.out.println("GPU Vector addition with direct arrays: SUCCESS");
    }
    
    @Test
    public void testGPUMatrixOperationsWithDirectArrays() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test 2x2 matrix addition
        double[][] m1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] m2 = {{5.0, 6.0}, {7.0, 8.0}};
        double[][] expected = {{6.0, 8.0}, {10.0, 12.0}};
        
        double[][] result = gpuComputer.binaryOperate(m1, m2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        assertEquals(expected.length, result.length, "Matrix row count mismatch");
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], result[i], 0.001, "Matrix addition failed at row " + i);
        }
        
        System.out.println("GPU Matrix addition with direct arrays: SUCCESS");
    }
}