package com.yishape.lab.math.compute;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import org.junit.jupiter.api.Test;

/**
 * Test class for verifying the mmul method in SISDDoubleComputer
 */
public class SISDDoubleComputerMmulTest {

    private final SISDDoubleComputer computer = new SISDDoubleComputer();

    @Test
    public void testMmulBasic() {
        // Test basic matrix multiplication
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //       = [[19, 22], [43, 50]]
        double[][] a = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] b = {{5.0, 6.0}, {7.0, 8.0}};
        double[][] expected = {{19.0, 22.0}, {43.0, 50.0}};
        double[][] result = computer.mmul(a, b);
        
        assertArrayEquals(expected, result);
    }

    @Test
    public void testMmulDifferentDimensions() {
        // Test matrix multiplication with different dimensions
        // A = [[1, 2, 3]]
        // B = [[4], [5], [6]]
        // A * B = [[1*4 + 2*5 + 3*6]] = [[32]]
        double[][] a = {{1.0, 2.0, 3.0}};
        double[][] b = {{4.0}, {5.0}, {6.0}};
        double[][] expected = {{32.0}};
        double[][] result = computer.mmul(a, b);
        
        assertArrayEquals(expected, result);
    }

    @Test
    public void testMmulLargerMatrix() {
        // Test larger matrix multiplication
        // A = [[1, 2], [3, 4], [5, 6]]
        // B = [[7, 8, 9], [10, 11, 12]]
        // A * B = [[1*7+2*10, 1*8+2*11, 1*9+2*12], 
        //          [3*7+4*10, 3*8+4*11, 3*9+4*12],
        //          [5*7+6*10, 5*8+6*11, 5*9+6*12]]
        //       = [[27, 30, 33], [61, 68, 75], [95, 106, 117]]
        double[][] a = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
        double[][] b = {{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
        double[][] expected = {{27.0, 30.0, 33.0}, {61.0, 68.0, 75.0}, {95.0, 106.0, 117.0}};
        double[][] result = computer.mmul(a, b);
        
        assertArrayEquals(expected, result);
    }
}