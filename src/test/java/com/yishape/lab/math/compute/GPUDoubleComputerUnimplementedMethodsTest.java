package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for the previously unimplemented methods in GPUDoubleComputer
 */
public class GPUDoubleComputerUnimplementedMethodsTest {

    private final GPUDoubleComputer computer = new GPUDoubleComputer();

    @Test
    public void testLogicalOperateUnary() {
        // Prepare test data
        double[] x1 = {0.0, 2.0, 0.0, 4.0, 0.0, 6.0};
        
        // Test NOT operation
        boolean[] result = computer.logicalOperate(x1, IDoubleVectorComputer.LogicalOperation.NOT);
        assertNotNull(result);
        assertEquals(6, result.length);
        assertTrue(result[0]);  // NOT 0.0 = true
        assertFalse(result[1]); // NOT 2.0 = false
        assertTrue(result[2]);  // NOT 0.0 = true
        assertFalse(result[3]); // NOT 4.0 = false
        assertTrue(result[4]);  // NOT 0.0 = true
        assertFalse(result[5]); // NOT 6.0 = false
    }

    @Test
    public void testTransposeRowVector() {
        // Prepare test data
        double[] rowVector = {1.0, 2.0, 3.0, 4.0};
        
        // Test transpose operation
        double[][] result = computer.transpose(rowVector);
        assertNotNull(result);
        assertEquals(4, result.length);
        assertEquals(1, result[0].length);
        assertEquals(1.0, result[0][0], 0.001);
        assertEquals(2.0, result[1][0], 0.001);
        assertEquals(3.0, result[2][0], 0.001);
        assertEquals(4.0, result[3][0], 0.001);
    }

    @Test
    public void testMmul() {
        // Prepare test data
        double[][] a = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] b = {{2.0, 0.0}, {1.0, 2.0}};
        
        // Test matrix multiplication
        double[][] result = computer.mmul(a, b);
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(2, result[0].length);
        assertEquals(4.0, result[0][0], 0.001); // 1*2 + 2*1 = 4
        assertEquals(4.0, result[0][1], 0.001); // 1*0 + 2*2 = 4
        assertEquals(10.0, result[1][0], 0.001); // 3*2 + 4*1 = 10
        assertEquals(8.0, result[1][1], 0.001); // 3*0 + 4*2 = 8
    }

    @Test
    public void testOuter() {
        // Prepare test data
        double[] a = {1.0, 2.0};
        double[] b = {3.0, 4.0, 5.0};
        
        // Test outer product
        double[][] result = computer.outer(a, b);
        assertNotNull(result);
        assertEquals(2, result.length);
        assertEquals(3, result[0].length);
        assertEquals(3.0, result[0][0], 0.001); // 1*3 = 3
        assertEquals(4.0, result[0][1], 0.001); // 1*4 = 4
        assertEquals(5.0, result[0][2], 0.001); // 1*5 = 5
        assertEquals(6.0, result[1][0], 0.001); // 2*3 = 6
        assertEquals(8.0, result[1][1], 0.001); // 2*4 = 8
        assertEquals(10.0, result[1][2], 0.001); // 2*5 = 10
    }

    @Test
    public void testSign1D() {
        // Prepare test data
        double[] array = {-1.0, 0.0, 2.0, -4.0, 0.0, 5.0};
        
        // Test sign operation
        double[] result = computer.sign(array);
        assertNotNull(result);
        assertEquals(6, result.length);
        assertEquals(-1.0, result[0], 0.001); // sign(-1.0) = -1.0
        assertEquals(0.0, result[1], 0.001);  // sign(0.0) = 0.0
        assertEquals(1.0, result[2], 0.001);  // sign(2.0) = 1.0
        assertEquals(-1.0, result[3], 0.001); // sign(-4.0) = -1.0
        assertEquals(0.0, result[4], 0.001);  // sign(0.0) = 0.0
        assertEquals(1.0, result[5], 0.001);  // sign(5.0) = 1.0
    }

    @Test
    public void testSign2D() {
        // Prepare test data
        double[][] array = {{-1.0, 0.0, 2.0}, {3.0, -4.0, 0.0}};
        
        // Test sign operation
        double[][] result = computer.sign(array);
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
    public void testDiff() {
        // Prepare test data
        double[] array = {1.0, 3.0, 6.0, 10.0, 15.0};
        
        // Test diff operation with stride 1
        double[] result = computer.diff(array, 1);
        assertNotNull(result);
        assertEquals(4, result.length);
        assertEquals(2.0, result[0], 0.001); // 3.0 - 1.0 = 2.0
        assertEquals(3.0, result[1], 0.001); // 6.0 - 3.0 = 3.0
        assertEquals(4.0, result[2], 0.001); // 10.0 - 6.0 = 4.0
        assertEquals(5.0, result[3], 0.001); // 15.0 - 10.0 = 5.0
        
        // Test diff operation with stride 2
        result = computer.diff(array, 2);
        assertNotNull(result);
        assertEquals(3, result.length);
        assertEquals(5.0, result[0], 0.001); // 6.0 - 1.0 = 5.0
        assertEquals(7.0, result[1], 0.001); // 10.0 - 3.0 = 7.0
        assertEquals(9.0, result[2], 0.001); // 15.0 - 6.0 = 9.0
    }

    @Test
    public void testLogicalOperateUnaryExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate((double[]) null, IDoubleVectorComputer.LogicalOperation.NOT);
        });
    }

    @Test
    public void testTransposeRowVectorExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.transpose((double[]) null);
        });
    }

    @Test
    public void testMmulExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.mmul(null, new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.mmul(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, null);
        });
        
        // Test empty matrix
        assertThrows(IllegalArgumentException.class, () -> {
            computer.mmul(new double[0][0], new double[][] {{1.0, 2.0}, {3.0, 4.0}});
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.mmul(new double[][] {{1.0, 2.0}, {3.0, 4.0}}, new double[0][0]);
        });
        
        // Test dimension mismatch
        double[][] a = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] b = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        assertThrows(IllegalArgumentException.class, () -> {
            computer.mmul(a, b);
        });
    }

    @Test
    public void testOuterExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.outer(null, new double[] {1.0, 2.0});
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.outer(new double[] {1.0, 2.0}, null);
        });
        
        // Test empty vector
        assertThrows(IllegalArgumentException.class, () -> {
            computer.outer(new double[0], new double[] {1.0, 2.0});
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.outer(new double[] {1.0, 2.0}, new double[0]);
        });
    }

    @Test
    public void testSign1DExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.sign((double[]) null);
        });
    }

    @Test
    public void testSign2DExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.sign((double[][]) null);
        });
    }

    @Test
    public void testDiffExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.diff(null, 1);
        });
    }

    @Test
    public void testLogicalCompare2DExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalCompare(null, new double[][] {{1.0, 2.0}}, IDoubleVectorComputer.LogicalCompare.EQUALS);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalCompare(new double[][] {{1.0, 2.0}}, null, IDoubleVectorComputer.LogicalCompare.EQUALS);
        });
        
        // Test empty matrix
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalCompare(new double[0][0], new double[][] {{1.0, 2.0}}, IDoubleVectorComputer.LogicalCompare.EQUALS);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalCompare(new double[][] {{1.0, 2.0}}, new double[0][0], IDoubleVectorComputer.LogicalCompare.EQUALS);
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
        
        // Test empty matrix
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[0][0], IDoubleVectorComputer.LogicalOperation.NOT);
        });
    }

    @Test
    public void testLogicalOperate2DBinaryExceptions() {
        // Test null input
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(null, new double[][] {{1.0, 2.0}}, IDoubleVectorComputer.LogicalOperation.AND);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[][] {{1.0, 2.0}}, null, IDoubleVectorComputer.LogicalOperation.AND);
        });
        
        // Test empty matrix
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[0][0], new double[][] {{1.0, 2.0}}, IDoubleVectorComputer.LogicalOperation.AND);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(new double[][] {{1.0, 2.0}}, new double[0][0], IDoubleVectorComputer.LogicalOperation.AND);
        });
        
        // Test dimension mismatch
        double[][] x1 = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] x2 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
        assertThrows(IllegalArgumentException.class, () -> {
            computer.logicalOperate(x1, x2, IDoubleVectorComputer.LogicalOperation.AND);
        });
    }
}