package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for verifying consistency of mmul implementation across all three computer classes:
 * GPUDoubleComputer, SISDDoubleComputer, and SIMDDoubleComputer
 */
public class MmulConsistencyTest {

    private final GPUDoubleComputer gpuComputer = new GPUDoubleComputer();
    private final SISDDoubleComputer sisdComputer = new SISDDoubleComputer();
    private final SIMDDoubleComputer simdComputer = new SIMDDoubleComputer();

    @Test
    public void testMmulConsistencyBasic() {
        // Test basic matrix multiplication consistency across all implementations
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //       = [[19, 22], [43, 50]]
        double[][] a = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] b = {{5.0, 6.0}, {7.0, 8.0}};
        double[][] expected = {{19.0, 22.0}, {43.0, 50.0}};

        // Test SISD implementation
        double[][] sisdResult = sisdComputer.mmul(a, b);
        assertArrayEquals(expected, sisdResult);

        // Test SIMD implementation
        double[][] simdResult = simdComputer.mmul(a, b);
        assertArrayEquals(expected, simdResult);

        // Test GPU implementation (may have slight floating point differences)
        try {
            double[][] gpuResult = gpuComputer.mmul(a, b);
            assertEquals(expected.length, gpuResult.length);
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i].length, gpuResult[i].length);
                for (int j = 0; j < expected[i].length; j++) {
                    assertEquals(expected[i][j], gpuResult[i][j], 1e-10, 
                        "GPU result mismatch at [" + i + "][" + j + "]");
                }
            }
        } catch (Exception e) {
            // GPU might not be available, skip this test
            System.out.println("GPU test skipped: " + e.getMessage());
        }
    }

    @Test
    public void testMmulConsistencyDifferentDimensions() {
        // Test matrix multiplication with different dimensions
        // A = [[1, 2, 3]]
        // B = [[4], [5], [6]]
        // A * B = [[1*4 + 2*5 + 3*6]] = [[32]]
        double[][] a = {{1.0, 2.0, 3.0}};
        double[][] b = {{4.0}, {5.0}, {6.0}};
        double[][] expected = {{32.0}};

        // Test SISD implementation
        double[][] sisdResult = sisdComputer.mmul(a, b);
        assertArrayEquals(expected, sisdResult);

        // Test SIMD implementation
        double[][] simdResult = simdComputer.mmul(a, b);
        assertArrayEquals(expected, simdResult);

        // Test GPU implementation
        try {
            double[][] gpuResult = gpuComputer.mmul(a, b);
            assertEquals(expected.length, gpuResult.length);
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i].length, gpuResult[i].length);
                for (int j = 0; j < expected[i].length; j++) {
                    assertEquals(expected[i][j], gpuResult[i][j], 1e-10,
                        "GPU result mismatch at [" + i + "][" + j + "]");
                }
            }
        } catch (Exception e) {
            // GPU might not be available, skip this test
            System.out.println("GPU test skipped: " + e.getMessage());
        }
    }

    @Test
    public void testMmulConsistencyLargerMatrix() {
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

        // Test SISD implementation
        double[][] sisdResult = sisdComputer.mmul(a, b);
        assertArrayEquals(expected, sisdResult);

        // Test SIMD implementation
        double[][] simdResult = simdComputer.mmul(a, b);
        assertArrayEquals(expected, simdResult);

        // Test GPU implementation
        try {
            double[][] gpuResult = gpuComputer.mmul(a, b);
            assertEquals(expected.length, gpuResult.length);
            for (int i = 0; i < expected.length; i++) {
                assertEquals(expected[i].length, gpuResult[i].length);
                for (int j = 0; j < expected[i].length; j++) {
                    assertEquals(expected[i][j], gpuResult[i][j], 1e-10,
                        "GPU result mismatch at [" + i + "][" + j + "]");
                }
            }
        } catch (Exception e) {
            // GPU might not be available, skip this test
            System.out.println("GPU test skipped: " + e.getMessage());
        }
    }
}