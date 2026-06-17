package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.ops.BinaryOperation;
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

    // ==================== Phase 3.2: In-place binary ops ====================

    @Test
    public void testBinaryOperateInPlaceAdd() {
        double[] target = {1.0, 2.0, 3.0, 4.0};
        double[] source = {5.0, 6.0, 7.0, 8.0};
        computer.binaryOperateInPlace(target, source, BinaryOperation.ADD);
        assertArrayEquals(new double[]{6.0, 8.0, 10.0, 12.0}, target, 1e-12);
        // Source is unchanged
        assertArrayEquals(new double[]{5.0, 6.0, 7.0, 8.0}, source, 1e-12);
    }

    @Test
    public void testBinaryOperateInPlaceSubtract() {
        double[] target = {10.0, 9.0, 8.0, 7.0};
        double[] source = {1.0, 2.0, 3.0, 4.0};
        computer.binaryOperateInPlace(target, source, BinaryOperation.SUBTRACT);
        assertArrayEquals(new double[]{9.0, 7.0, 5.0, 3.0}, target, 1e-12);
    }

    @Test
    public void testBinaryOperateInPlaceAllocationFree() {
        // Verify that in-place ADD does NOT allocate a new array (returns same target reference)
        double[] target = {1.0, 2.0, 3.0};
        double[] source = {0.5, 0.5, 0.5};
        double[] originalRef = target;
        computer.binaryOperateInPlace(target, source, BinaryOperation.ADD);
        assertSame(originalRef, target, "In-place op must not replace the target array reference");
    }

    @Test
    public void testBinaryOperateInPlaceEquivalence() {
        // Verify in-place result matches allocating binaryOperate result
        double[] a = {1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5};
        double[] b = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

        // Allocating path
        IDoubleVectorComputer dvc = new DoubleVectorComputer();
        double[] allocResult = dvc.binaryOperate(a.clone(), b, BinaryOperation.ADD);

        // In-place path
        DoubleVectorComputer dvc2 = new DoubleVectorComputer();
        double[] a2 = a.clone();
        dvc2.binaryOperateInPlace(a2, b, BinaryOperation.ADD);

        assertArrayEquals(allocResult, a2, 1e-12,
            "In-place ADD must produce identical result to allocating binaryOperate");
    }

    @Test
    public void testBinaryOperateInPlaceScalar() {
        double[] target = {1.0, 2.0, 3.0, 4.0};
        computer.binaryOperateInPlace(target, 10.0, BinaryOperation.ADD);
        assertArrayEquals(new double[]{11.0, 12.0, 13.0, 14.0}, target, 1e-12);

        computer.binaryOperateInPlace(target, 2.0, BinaryOperation.MULTIPLY);
        assertArrayEquals(new double[]{22.0, 24.0, 26.0, 28.0}, target, 1e-12);
    }

    @Test
    public void testBinaryOperateInPlaceNullRejected() {
        assertThrows(IllegalArgumentException.class, () ->
            computer.binaryOperateInPlace(null, new double[]{1.0}, BinaryOperation.ADD));
        assertThrows(IllegalArgumentException.class, () ->
            computer.binaryOperateInPlace(new double[]{1.0}, null, BinaryOperation.ADD));
    }

    @Test
    public void testBinaryOperateInPlaceLengthMismatch() {
        assertThrows(IllegalArgumentException.class, () ->
            computer.binaryOperateInPlace(new double[]{1.0, 2.0}, new double[]{1.0}, BinaryOperation.ADD));
    }

}