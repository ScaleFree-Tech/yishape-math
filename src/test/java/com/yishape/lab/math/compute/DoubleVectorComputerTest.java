package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for DoubleVectorComputer
 */
public class DoubleVectorComputerTest {

    private final DoubleVectorComputer computer = new DoubleVectorComputer();

    @Test
    public void testBinaryOperateVectorVector() {
        double[] x1 = {1.0, 2.0, 3.0};
        double[] x2 = {4.0, 5.0, 6.0};
        
        // Test addition
        double[] result = computer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        assertArrayEquals(new double[]{5.0, 7.0, 9.0}, result, 1e-10);
        
        // Test subtraction
        result = computer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        assertArrayEquals(new double[]{-3.0, -3.0, -3.0}, result, 1e-10);
        
        // Test multiplication
        result = computer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        assertArrayEquals(new double[]{4.0, 10.0, 18.0}, result, 1e-10);
        
        // Test division
        result = computer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.DIVIDE);
        assertArrayEquals(new double[]{0.25, 0.4, 0.5}, result, 1e-10);
    }

    @Test
    public void testBinaryOperateVectorScalar() {
        double[] x1 = {1.0, 2.0, 3.0};
        double x2 = 2.0;
        
        // Test addition
        double[] result = computer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.ADD);
        assertArrayEquals(new double[]{3.0, 4.0, 5.0}, result, 1e-10);
        
        // Test subtraction
        result = computer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        assertArrayEquals(new double[]{-1.0, 0.0, 1.0}, result, 1e-10);
        
        // Test multiplication
        result = computer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, result, 1e-10);
        
        // Test division
        result = computer.binaryOperate(x1, x2, IDoubleVectorComputer.BinaryOperation.DIVIDE);
        assertArrayEquals(new double[]{0.5, 1.0, 1.5}, result, 1e-10);
    }

    @Test
    public void testUniversalOperate() {
        double[] x = {1.0, 4.0, 9.0};
        
        // Test square root
        double[] result = computer.universalOperate(x, IDoubleVectorComputer.UniversalOperation.SQRT, 0.0);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, result, 1e-10);
        
        // Test power
        result = computer.universalOperate(x, IDoubleVectorComputer.UniversalOperation.POW, 2.0);
        assertArrayEquals(new double[]{1.0, 16.0, 81.0}, result, 1e-10);
    }

    @Test
    public void testReduceOperate() {
        double[] x = {1.0, 2.0, 3.0, 4.0};
        
        // Test sum
        double result = computer.reduceOperate(x, IDoubleVectorComputer.ReduceOperation.SUM);
        assertEquals(10.0, result, 1e-10);
        
        // Test mean
        result = computer.reduceOperate(x, IDoubleVectorComputer.ReduceOperation.MEAN);
        assertEquals(2.5, result, 1e-10);
        
        // Test min
        result = computer.reduceOperate(x, IDoubleVectorComputer.ReduceOperation.MIN);
        assertEquals(1.0, result, 1e-10);
        
        // Test max
        result = computer.reduceOperate(x, IDoubleVectorComputer.ReduceOperation.MAX);
        assertEquals(4.0, result, 1e-10);
    }

    @Test
    public void testElementWiseMin() {
        double[] x1 = {1.0, 5.0, 3.0};
        double[] x2 = {4.0, 2.0, 6.0};
        
        double[] result = computer.elementWiseMin(x1, x2);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, result, 1e-10);
    }

    @Test
    public void testElementWiseMax() {
        double[] x1 = {1.0, 5.0, 3.0};
        double[] x2 = {4.0, 2.0, 6.0};
        
        double[] result = computer.elementWiseMax(x1, x2);
        assertArrayEquals(new double[]{4.0, 5.0, 6.0}, result, 1e-10);
    }

    @Test
    public void testNegate() {
        double[] x = {1.0, -2.0, 3.0};
        
        double[] result = computer.negate(x);
        assertArrayEquals(new double[]{-1.0, 2.0, -3.0}, result, 1e-10);
    }

    @Test
    public void testLogicalOperate() {
        double[] x1 = {1.0, 2.0, 3.0};
        double[] x2 = {1.0, 3.0, 2.0};
        
        // Test equals
        boolean[] result = computer.logicalCompare(x1, x2, IDoubleVectorComputer.LogicalCompare.EQUALS);
        assertArrayEquals(new boolean[]{true, false, false}, result);
        
        // Test less than
        result = computer.logicalCompare(x1, x2, IDoubleVectorComputer.LogicalCompare.LESS_THAN);
        assertArrayEquals(new boolean[]{false, true, false}, result);
    }

    @Test
    public void testBinaryReduceOperate() {
        double[] x1 = {1.0, 2.0, 3.0};
        double[] x2 = {4.0, 5.0, 6.0};
        
        // Test dot product
        double result = computer.binaryReduceOperate(x1, x2, IDoubleVectorComputer.BinaryReduceOperation.DOT);
        assertEquals(32.0, result, 1e-10); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }
}