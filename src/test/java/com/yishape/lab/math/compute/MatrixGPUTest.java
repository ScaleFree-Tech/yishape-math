package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixGPUTest {
    
    @Test
    public void testMatrixAddition() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test matrix addition with small matrices
        double[][] x1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] x2 = {{5.0, 6.0}, {7.0, 8.0}};
        
        double[][] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        double[][] expected = {{6.0, 8.0}, {10.0, 12.0}};
        assertEquals(expected.length, result.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], result[i], 0.001);
        }
        
        System.out.println("Matrix addition test passed");
    }
    
    @Test
    public void testMatrixMultiplication() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test matrix multiplication with small matrices
        double[][] x1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] x2 = {{5.0, 6.0}, {7.0, 8.0}};
        
        double[][] result = gpuComputer.mmul(x1, x2);
        
        double[][] expected = {{19.0, 22.0}, {43.0, 50.0}};
        assertEquals(expected.length, result.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], result[i], 0.001);
        }
        
        System.out.println("Matrix multiplication test passed");
    }
    
    @Test
    public void testMatrixTranspose() {
        GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
        
        // Test matrix transpose
        double[][] x = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        
        double[][] result = gpuComputer.transpose(x);
        
        double[][] expected = {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}};
        assertEquals(expected.length, result.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], result[i], 0.001);
        }
        
        System.out.println("Matrix transpose test passed");
    }
}