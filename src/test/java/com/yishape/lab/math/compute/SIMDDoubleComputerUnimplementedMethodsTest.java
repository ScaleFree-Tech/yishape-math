package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for the previously unimplemented methods in SIMDDoubleComputer
 */
public class SIMDDoubleComputerUnimplementedMethodsTest {

    private final SIMDDoubleComputer computer = new SIMDDoubleComputer();


    @Test
    public void testSign2D() {
        // Prepare test data
        double[][] array = {{-1.0, 0.0, 2.0}, {3.0, -4.0, 0.0}};
        
        // Test sign operation
        double[][] result = computer.sign((double[][]) array);
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(3, result[0].length);
        assertEquals(3, result[1].length);
        assertEquals(-1.0, result[0][0], 0.001); // sign(-1.0) = -1.0
        assertEquals(0.0, result[0][1], 0.001);  // sign(0.0) = 0.0
        assertEquals(1.0, result[0][2], 0.001);  // sign(2.0) = 1.0
        assertEquals(1.0, result[1][0], 0.001);  // sign(3.0) = 1.0
        assertEquals(-1.0, result[1][1], 0.001); // sign(-4.0) = -1.0
        assertEquals(0.0, result[1][2], 0.001);  // sign(0.0) = 0.0
    }

    @Test
    public void testLogicalCompare2DExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalCompare(null, new double[0][0], IDoubleVectorComputer.LogicalCompare.EQUALS);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalCompare(new double[0][0], null, IDoubleVectorComputer.LogicalCompare.EQUALS);
        });
        
        // Test dimension mismatch
        double[][] x1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] x2 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalCompare(x1, x2, IDoubleVectorComputer.LogicalCompare.EQUALS);
        });
    }

    @Test
    public void testLogicalOperate2DUnaryExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate((double[][]) null, IDoubleVectorComputer.LogicalOperation.NOT);
        });
    }

    @Test
    public void testLogicalOperate2DBinaryExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(null, new double[0][0], IDoubleVectorComputer.LogicalOperation.AND);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[0][0], null, IDoubleVectorComputer.LogicalOperation.AND);
        });
        
        // Test dimension mismatch
        double[][] x1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] x2 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(x1, x2, IDoubleVectorComputer.LogicalOperation.AND);
        });
    }

    @Test
    public void testSign2DExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.sign((double[][]) null);
        });
    }
}