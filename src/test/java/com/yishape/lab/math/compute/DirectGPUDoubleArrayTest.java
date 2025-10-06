package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeAll;
import static org.junit.jupiter.api.Assertions.*;

public class DirectGPUDoubleArrayTest {
    
    private static GPUDoubleComputer gpuComputer;
    
    @BeforeAll
    static void setUp() {
        gpuComputer = new GPUDoubleComputer();
        System.out.println("GPU Available: " + GPUDoubleComputer.isGPUAvailable());
    }
    
    @Test
    public void testVectorAddition() {
        System.out.println("\n=== Testing Direct Vector Addition ===");
        
        // Test with small arrays
        double[] x1 = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] x2 = {5.0, 4.0, 3.0, 2.0, 1.0};
        double[] expected = {6.0, 6.0, 6.0, 6.0, 6.0};
        
        double[] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        assertArrayEquals(expected, result, 0.001, "Vector addition failed");
        System.out.println("Small vector addition test passed");
        
        // Test with larger arrays
        int size = 1000;
        double[] largeX1 = new double[size];
        double[] largeX2 = new double[size];
        double[] largeExpected = new double[size];
        
        for (int i = 0; i < size; i++) {
            largeX1[i] = i * 1.5;
            largeX2[i] = i * 0.5;
            largeExpected[i] = i * 2.0;
        }
        
        double[] largeResult = gpuComputer.binaryOperate(largeX1, largeX2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        assertArrayEquals(largeExpected, largeResult, 0.001, "Large vector addition failed");
        System.out.println("Large vector addition test passed (size: " + size + ")");
    }
    
    @Test
    public void testVectorSubtraction() {
        System.out.println("\n=== Testing Direct Vector Subtraction ===");
        
        double[] x1 = {10.0, 8.0, 6.0, 4.0, 2.0};
        double[] x2 = {1.0, 2.0, 3.0, 4.0, 5.0};
        double[] expected = {9.0, 6.0, 3.0, 0.0, -3.0};
        
        double[] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        
        assertArrayEquals(expected, result, 0.001, "Vector subtraction failed");
        System.out.println("Vector subtraction test passed");
    }
    
    @Test
    public void testVectorMultiplication() {
        System.out.println("\n=== Testing Direct Vector Multiplication ===");
        
        double[] x1 = {2.0, 3.0, 4.0, 5.0};
        double[] x2 = {3.0, 4.0, 5.0, 6.0};
        double[] expected = {6.0, 12.0, 20.0, 30.0};
        
        double[] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        
        assertArrayEquals(expected, result, 0.001, "Vector multiplication failed");
        System.out.println("Vector multiplication test passed");
    }
    
    @Test
    public void testVectorDivision() {
        System.out.println("\n=== Testing Direct Vector Division ===");
        
        double[] x1 = {12.0, 15.0, 20.0, 25.0};
        double[] x2 = {3.0, 5.0, 4.0, 5.0};
        double[] expected = {4.0, 3.0, 5.0, 5.0};
        
        double[] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.DIVIDE);
        
        assertArrayEquals(expected, result, 0.001, "Vector division failed");
        System.out.println("Vector division test passed");
    }
    
    @Test
    public void testVectorScalarOperations() {
        System.out.println("\n=== Testing Direct Vector Scalar Operations ===");
        
        double[] x = {2.0, 4.0, 6.0, 8.0};
        double scalar = 3.0;
        
        // Test scalar multiplication
        double[] expectedMul = {6.0, 12.0, 18.0, 24.0};
        double[] resultMul = gpuComputer.binaryOperate(x, scalar, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        assertArrayEquals(expectedMul, resultMul, 0.001, "Vector scalar multiplication failed");
        System.out.println("Vector scalar multiplication test passed");
        
        // Test scalar addition
        double[] expectedAdd = {5.0, 7.0, 9.0, 11.0};
        double[] resultAdd = gpuComputer.binaryOperate(x, scalar, IDoubleVectorComputer.BinaryOperation.ADD);
        assertArrayEquals(expectedAdd, resultAdd, 0.001, "Vector scalar addition failed");
        System.out.println("Vector scalar addition test passed");
    }
    
    @Test
    public void testUniversalOperations() {
        System.out.println("\n=== Testing Direct Universal Operations ===");
        
        // Test SQRT
        double[] x1 = {4.0, 9.0, 16.0, 25.0};
        double[] expectedSqrt = {2.0, 3.0, 4.0, 5.0};
        double[] resultSqrt = gpuComputer.universalOperate(x1, IDoubleVectorComputer.UniversalOperation.SQRT, 0.0);
        assertArrayEquals(expectedSqrt, resultSqrt, 0.001, "SQRT operation failed");
        System.out.println("SQRT operation test passed");
        
        // Test EXP
        double[] x2 = {0.0, 1.0, 2.0};
        double[] expectedExp = {1.0, Math.E, Math.E * Math.E};
        double[] resultExp = gpuComputer.universalOperate(x2, IDoubleVectorComputer.UniversalOperation.EXP, 0.0);
        assertArrayEquals(expectedExp, resultExp, 0.001, "EXP operation failed");
        System.out.println("EXP operation test passed");
        
        // Test SIN
        double[] x3 = {0.0, Math.PI/2, Math.PI};
        double[] expectedSin = {0.0, 1.0, 0.0}; // Approximately
        double[] resultSin = gpuComputer.universalOperate(x3, IDoubleVectorComputer.UniversalOperation.SIN, 0.0);
        assertArrayEquals(expectedSin, resultSin, 0.001, "SIN operation failed");
        System.out.println("SIN operation test passed");
    }
    
    @Test
    public void testMatrixOperations() {
        System.out.println("\n=== Testing Direct Matrix Operations ===");
        
        // Test 2x2 matrix addition
        double[][] m1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] m2 = {{5.0, 6.0}, {7.0, 8.0}};
        double[][] expectedAdd = {{6.0, 8.0}, {10.0, 12.0}};
        
        double[][] resultAdd = gpuComputer.binaryOperate(m1, m2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        assertEquals(expectedAdd.length, resultAdd.length, "Matrix row count mismatch");
        for (int i = 0; i < expectedAdd.length; i++) {
            assertArrayEquals(expectedAdd[i], resultAdd[i], 0.001, "Matrix addition failed at row " + i);
        }
        System.out.println("Matrix addition test passed");
        
        // Test 3x3 matrix multiplication (element-wise)
        double[][] m3 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};
        double[][] m4 = {{2.0, 2.0, 2.0}, {3.0, 3.0, 3.0}, {4.0, 4.0, 4.0}};
        double[][] expectedMul = {{2.0, 4.0, 6.0}, {12.0, 15.0, 18.0}, {28.0, 32.0, 36.0}};
        
        double[][] resultMul = gpuComputer.binaryOperate(m3, m4, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        
        assertEquals(expectedMul.length, resultMul.length, "Matrix row count mismatch");
        for (int i = 0; i < expectedMul.length; i++) {
            assertArrayEquals(expectedMul[i], resultMul[i], 0.001, "Matrix multiplication failed at row " + i);
        }
        System.out.println("Matrix multiplication test passed");
    }
    
    @Test
    public void testLargeVectorPerformance() {
        System.out.println("\n=== Testing Large Vector Performance ===");
        
        int size = 100000;
        double[] x1 = new double[size];
        double[] x2 = new double[size];
        
        // Fill with test data
        for (int i = 0; i < size; i++) {
            x1[i] = Math.random() * 100;
            x2[i] = Math.random() * 100;
        }
        
        // Test that it doesn't crash and returns correct size
        double[] result = gpuComputer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        
        assertEquals(size, result.length, "Result array size mismatch");
        assertTrue(result[0] >= 0, "Result should be non-negative");
        System.out.println("Large vector performance test passed (size: " + size + ")");
    }
}