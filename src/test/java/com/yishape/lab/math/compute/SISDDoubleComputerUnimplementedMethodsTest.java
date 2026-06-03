package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.ops.LogicalOperation;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for verifying the implementation of previously unimplemented methods in SISDDoubleComputer
 */
public class SISDDoubleComputerUnimplementedMethodsTest {

    private final SISDDoubleComputer computer = new SISDDoubleComputer();

    @Test
    public void testLogicalOperateDoubleArrayLogicalOperation() {
        // Test NOT operation on double array
        double[] input = {0.0, 1.0, -1.0, 2.5, 0.0};
        boolean[] expected = {true, false, false, false, true};
        boolean[] result = computer.logicalOperate(input, LogicalOperation.NOT);
        
        assertArrayEquals(expected, result);
        
        // Test with null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate((double[]) null, LogicalOperation.NOT);
        });
    }

    @Test
    public void testTransposeRowVector() {
        // Test transposing a row vector to a column vector
        double[] rowVector = {1.0, 2.0, 3.0};
        double[][] expected = {{1.0}, {2.0}, {3.0}};
        double[][] result = computer.transpose(rowVector);
        
        assertArrayEquals(expected, result);
        
        // Test with null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.transpose((double[]) null);
        });
    }

    @Test
    public void testOuter() {
        // Test outer product of two vectors
        double[] a = {1.0, 2.0};
        double[] b = {3.0, 4.0, 5.0};
        double[][] expected = {{3.0, 4.0, 5.0}, {6.0, 8.0, 10.0}};
        double[][] result = computer.outer(a, b);
        
        assertArrayEquals(expected, result);
        
        // Test with null inputs
        assertThrows(IllegalArgumentException.class, () -> {
            computer.outer(null, b);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.outer(a, null);
        });
    }

    @Test
    public void testSignDoubleArray() {
        // Test sign function on double array
        double[] input = {-2.0, -1.0, 0.0, 1.0, 2.0};
        double[] expected = {-1.0, -1.0, 0.0, 1.0, 1.0};
        double[] result = computer.sign(input);
        
        assertArrayEquals(expected, result);
    }

    @Test
    public void testSignDoubleArrayArray() {
        // Test sign function on double matrix
        double[][] input = {{-2.0, -1.0, 0.0}, {1.0, 2.0, 3.0}};
        double[][] expected = {{-1.0, -1.0, 0.0}, {1.0, 1.0, 1.0}};
        double[][] result = computer.sign(input);
        
        // Compare each row
        assertEquals(expected.length, result.length);
        for (int i = 0; i < expected.length; i++) {
            assertArrayEquals(expected[i], result[i]);
        }
        
        // Test with null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.sign((double[][]) null);
        });
    }

    @Test
    public void testDiff() {
        // Test diff function
        double[] input = {1.0, 2.0, 4.0, 7.0, 11.0};
        double[] expected = {1.0, 2.0, 3.0, 4.0}; // diff with stride 1
        double[] result = computer.diff(input, 1);
        
        assertArrayEquals(expected, result);
        
        // Test with stride 2
        double[] expected2 = {3.0, 5.0, 7.0}; // diff with stride 2
        double[] result2 = computer.diff(input, 2);
        
        assertArrayEquals(expected2, result2);
        
        // Test with stride larger than array length
        double[] result3 = computer.diff(input, 10);
        assertEquals(0, result3.length);
        
        // Test with null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.diff(null, 1);
        });
        
        // Test with invalid stride
        assertThrows(IllegalArgumentException.class, () -> {
            computer.diff(input, 0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.diff(input, -1);
        });
    }


}