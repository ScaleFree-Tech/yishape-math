package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class SIMDDoubleComputerLogicalOperateTest {

    private final SIMDDoubleComputer computer = new SIMDDoubleComputer();

    @Test
    public void testLogicalOperateNot() {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        // For NOT operation: 0 becomes true, non-zero becomes false
        boolean[] expected = {true, false, false, true, false};
        
        boolean[] result = computer.logicalOperate(x1, IDoubleVectorComputer.LogicalOperation.NOT);
        
        assertArrayEquals(expected, result);
    }

    @Test
    public void testLogicalOperateAndUnsupported() {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(x1, IDoubleVectorComputer.LogicalOperation.AND);
        });
        
        assertTrue(exception.getMessage().contains("不支持的操作"));
        assertTrue(exception.getMessage().contains("logicalOperate方法仅支持NOT操作"));
    }

    @Test
    public void testLogicalOperateOrUnsupported() {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(x1, IDoubleVectorComputer.LogicalOperation.OR);
        });
        
        assertTrue(exception.getMessage().contains("不支持的操作"));
        assertTrue(exception.getMessage().contains("logicalOperate方法仅支持NOT操作"));
    }

    @Test
    public void testLogicalOperateXorUnsupported() {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(x1, IDoubleVectorComputer.LogicalOperation.XOR);
        });
        
        assertTrue(exception.getMessage().contains("不支持的操作"));
        assertTrue(exception.getMessage().contains("logicalOperate方法仅支持NOT操作"));
    }

    @Test
    public void testLogicalOperateWithNullVector() {
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate((double[]) null, IDoubleVectorComputer.LogicalOperation.NOT);
        });
    }

    @Test
    public void testLogicalOperateWithEmptyVector() {
        double[] x1 = {};
        boolean[] expected = {};
        
        boolean[] result = computer.logicalOperate(x1, IDoubleVectorComputer.LogicalOperation.NOT);
        
        assertArrayEquals(expected, result);
    }
}