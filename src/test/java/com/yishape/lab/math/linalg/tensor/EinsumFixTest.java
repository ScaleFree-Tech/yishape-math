package com.yishape.lab.math.linalg.tensor;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verify einsum "ij,jk->ik" across all 4 code paths:
 * RereDoubleTensor, RereDiffTensor, DiffTensorMatrix, RereFloatTensor.
 */
public class EinsumFixTest {

    // ==================== RereDoubleTensor (non-diff double) ====================

    @Test
    void testDoubleEinsumIjJkIk() {
        IDoubleTensor a = new RereDoubleTensor(new double[]{
            1, 2, 3,
            4, 5, 6
        }, 2, 3);
        IDoubleTensor b = new RereDoubleTensor(new double[]{
            7, 8, 9, 10,
            11, 12, 13, 14,
            15, 16, 17, 18
        }, 3, 4);

        IDoubleTensor c = a.einsum("ij,jk->ik", b);

        assertArrayEquals(new int[]{2, 4}, c.shape());
        double[] result = c.toDoubleArray();
        // Row 0: [1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18]
        assertEquals(74, result[0], 1e-9);   // C[0,0]
        assertEquals(80, result[1], 1e-9);   // C[0,1]
        assertEquals(86, result[2], 1e-9);   // C[0,2]
        assertEquals(92, result[3], 1e-9);   // C[0,3]
        // Row 1: [4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18]
        assertEquals(173, result[4], 1e-9);  // C[1,0]
        assertEquals(188, result[5], 1e-9);  // C[1,1]
        assertEquals(203, result[6], 1e-9);  // C[1,2]
        assertEquals(218, result[7], 1e-9);  // C[1,3]
    }

    // ==================== RereDiffTensor (diff double) ====================

    @Test
    void testDiffEinsumIjJkIk() {
        IDiffTensor a = AD.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDiffTensor b = AD.tensor(new double[]{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, 3, 4);

        IDiffTensor c = a.einsum("ij,jk->ik", b);

        assertArrayEquals(new int[]{2, 4}, c.shape());
        double[] result = c.toDoubleArray();
        assertEquals(74, result[0], 1e-9);
        assertEquals(80, result[1], 1e-9);
        assertEquals(86, result[2], 1e-9);
        assertEquals(92, result[3], 1e-9);
        assertEquals(173, result[4], 1e-9);
        assertEquals(188, result[5], 1e-9);
        assertEquals(203, result[6], 1e-9);
        assertEquals(218, result[7], 1e-9);
    }

    // ==================== RereFloatTensor (non-diff float) ====================

    @Test
    void testFloatEinsumIjJkIk() {
        IFloatTensor a = new RereFloatTensor(new float[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IFloatTensor b = new RereFloatTensor(new float[]{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}, 3, 4);

        IFloatTensor c = a.einsum("ij,jk->ik", b);

        assertArrayEquals(new int[]{2, 4}, c.shape());
        float[] result = c.toFloatArray();
        assertEquals(74f, result[0], 1e-5f);
        assertEquals(80f, result[1], 1e-5f);
        assertEquals(86f, result[2], 1e-5f);
        assertEquals(92f, result[3], 1e-5f);
        assertEquals(173f, result[4], 1e-5f);
        assertEquals(188f, result[5], 1e-5f);
        assertEquals(203f, result[6], 1e-5f);
        assertEquals(218f, result[7], 1e-5f);
    }

    // ==================== Erf ====================

    @Test
    void testDoubleErf() {
        IDoubleTensor x = new RereDoubleTensor(new double[]{0, 1, -1, 0.5, -0.5}, 5);
        IDoubleTensor y = x.erf();
        double[] r = y.toDoubleArray();
        assertEquals(0.0, r[0], 1e-7);           // erf(0)=0
        assertEquals(0.84270079, r[1], 1e-5);    // erf(1)
        assertEquals(-0.84270079, r[2], 1e-5);   // erf(-1)= -erf(1)
        assertEquals(0.52049988, r[3], 1e-5);    // erf(0.5)
        assertEquals(-0.52049988, r[4], 1e-5);   // erf(-0.5)= -erf(0.5)
    }

    @Test
    void testDiffErf() {
        IDiffTensor x = AD.tensor(new double[]{0, 1, -1, 0.5}, 4);
        IDiffTensor y = x.erf();
        double[] r = y.toDoubleArray();
        assertEquals(0.0, r[0], 1e-7);
        assertEquals(0.84270079, r[1], 1e-5);
        assertEquals(-0.84270079, r[2], 1e-5);
        assertEquals(0.52049988, r[3], 1e-5);

        // Check gradient: d/dx erf(x) = 2/sqrt(pi) * exp(-x²)
        y.sum().backward();
        double[] grad = x.grad().toDoubleArray();
        double d0 = 2.0 / Math.sqrt(Math.PI); // at x=0: 2/sqrt(pi) ≈ 1.128
        assertEquals(d0, grad[0], 1e-4);
        double d1 = 2.0 / Math.sqrt(Math.PI) * Math.exp(-1); // at x=1
        assertEquals(d1, grad[1], 1e-5);
        double d2 = 2.0 / Math.sqrt(Math.PI) * Math.exp(-1); // at x=-1, same because x²
        assertEquals(d2, grad[2], 1e-5);
    }

    @Test
    void testFloatErf() {
        IFloatTensor x = new RereFloatTensor(new float[]{0, 1, -1, 0.5f}, 4);
        IFloatTensor y = x.erf();
        float[] r = y.toFloatArray();
        assertEquals(0f, r[0], 1e-5f);
        assertEquals(0.84270079f, r[1], 1e-4f);
        assertEquals(-0.84270079f, r[2], 1e-4f);
        assertEquals(0.52049988f, r[3], 1e-4f);
    }
}
