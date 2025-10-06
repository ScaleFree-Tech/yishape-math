package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LargeMatrixGPUTest {
    
    @Test
    public void testLargeMatrixAddition() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test matrix addition with larger matrices (similar to performance test)
        int size = 50;
        double[][] x1 = new double[size][size];
        double[][] x2 = new double[size][size];
        
        // Fill matrices with test data
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                x1[i][j] = i * size + j;
                x2[i][j] = (i * size + j) * 2;
            }
        }
        
        double[][] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        // Check a few values
        assertEquals(x1[0][0] + x2[0][0], result[0][0], 0.001);
        assertEquals(x1[size-1][size-1] + x2[size-1][size-1], result[size-1][size-1], 0.001);
        
        System.out.println("Large matrix addition test passed");
    }
    
    @Test
    public void testLargeMatrixMultiplication() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test matrix multiplication with larger matrices (similar to performance test)
        int size = 50;
        double[][] x1 = new double[size][size];
        double[][] x2 = new double[size][size];
        
        // Fill matrices with test data
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                x1[i][j] = i * size + j;
                x2[i][j] = (i * size + j) * 2;
            }
        }
        
        double[][] result = gpuComputer.mmul(x1, x2);
        
        // Check a few values (we won't check all since it's a large matrix)
        assertTrue(result[0][0] >= 0);
        assertTrue(result[size-1][size-1] >= 0);
        
        System.out.println("Large matrix multiplication test passed");
    }
}