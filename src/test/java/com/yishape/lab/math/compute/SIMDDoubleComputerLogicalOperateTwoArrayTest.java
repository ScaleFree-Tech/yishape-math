package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SIMDDoubleComputerLogicalOperateTwoArrayTest {

    private final SIMDDoubleComputer computer = new SIMDDoubleComputer();

    @Test
    public void testLogicalOperateAnd() {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        double[] x2 = {0.0, 0.0, 3.0, 4.0, 5.0};
        // For AND operation: both non-zero -> true, otherwise false
        boolean[] expected = {false, false, true, false, true};
        
        boolean[] result = computer.logicalOperate(x1, x2, IDoubleVectorComputer.LogicalOperation.AND);
        
        assertArrayEquals(expected, result);
    }

    @Test
    public void testLogicalOperateOr() {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        double[] x2 = {0.0, 0.0, 3.0, 4.0, 5.0};
        // For OR operation: at least one non-zero -> true, otherwise false
        boolean[] expected = {false, true, true, true, true};
        
        boolean[] result = computer.logicalOperate(x1, x2, IDoubleVectorComputer.LogicalOperation.OR);
        
        assertArrayEquals(expected, result);
    }

    @Test
    public void testLogicalOperateXor() {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        double[] x2 = {0.0, 0.0, 3.0, 4.0, 5.0};
        // For XOR operation: exactly one non-zero -> true, otherwise false
        boolean[] expected = {false, true, false, true, false};
        
        boolean[] result = computer.logicalOperate(x1, x2, IDoubleVectorComputer.LogicalOperation.XOR);
        
        assertArrayEquals(expected, result);
    }

    @Test
    public void testLogicalOperateNotUnsupported() {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        double[] x2 = {0.0, 0.0, 3.0, 4.0, 5.0};
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(x1, x2, IDoubleVectorComputer.LogicalOperation.NOT);
        });
        
        assertTrue(exception.getMessage().contains("不支持的操作"));
        assertTrue(exception.getMessage().contains("logicalOperate方法仅支持AND、OR、XOR操作"));
    }

    @Test
    public void testLogicalOperateWithNullFirstVector() {
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(null, new double[]{1.0, 2.0}, IDoubleVectorComputer.LogicalOperation.AND);
        });
    }

    @Test
    public void testLogicalOperateWithNullSecondVector() {
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[]{1.0, 2.0}, null, IDoubleVectorComputer.LogicalOperation.AND);
        });
    }

    @Test
    public void testLogicalOperateWithDifferentLengths() {
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[]{1.0, 2.0}, new double[]{1.0, 2.0, 3.0}, IDoubleVectorComputer.LogicalOperation.AND);
        });
    }

    @Test
    public void testLogicalOperateWithEmptyVectors() {
        double[] x1 = {};
        double[] x2 = {};
        boolean[] expected = {};
        
        boolean[] result = computer.logicalOperate(x1, x2, IDoubleVectorComputer.LogicalOperation.AND);
        
        assertArrayEquals(expected, result);
    }
}