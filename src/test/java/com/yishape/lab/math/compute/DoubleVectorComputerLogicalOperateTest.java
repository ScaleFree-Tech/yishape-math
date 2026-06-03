package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.ops.LogicalOperation;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class DoubleVectorComputerLogicalOperateTest {

    @Test
    public void testSIMDLogicalOperateTwoArrays() {
        SIMDDoubleComputer computer = new SIMDDoubleComputer();
        testLogicalOperateTwoArrays(computer);
    }

    @Test
    public void testSISDLogicalOperateTwoArrays() {
        SISDDoubleComputer computer = new SISDDoubleComputer();
        testLogicalOperateTwoArrays(computer);
    }

    private void testLogicalOperateTwoArrays(IDoubleVectorComputer computer) {
        double[] x1 = {0.0, 1.0, -2.0, 0.0, 3.5};
        double[] x2 = {0.0, 0.0, 3.0, 4.0, 5.0};

        // Test AND operation
        boolean[] andResult = computer.logicalOperate(x1, x2, LogicalOperation.AND);
        boolean[] expectedAnd = {false, false, true, false, true};
        assertArrayEquals(expectedAnd, andResult);

        // Test OR operation
        boolean[] orResult = computer.logicalOperate(x1, x2, LogicalOperation.OR);
        boolean[] expectedOr = {false, true, true, true, true};
        assertArrayEquals(expectedOr, orResult);

        // Test XOR operation
        boolean[] xorResult = computer.logicalOperate(x1, x2, LogicalOperation.XOR);
        boolean[] expectedXor = {false, true, false, true, false};
        assertArrayEquals(expectedXor, xorResult);
    }

    @Test
    public void testLogicalOperateTwoArraysInvalidInputs() {
        SIMDDoubleComputer computer = new SIMDDoubleComputer();

        // Test with null first array
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(null, new double[]{1.0, 2.0}, LogicalOperation.AND);
        });

        // Test with null second array
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[]{1.0, 2.0}, null, LogicalOperation.AND);
        });

        // Test with different length arrays
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[]{1.0, 2.0}, new double[]{1.0, 2.0, 3.0}, LogicalOperation.AND);
        });
    }
}