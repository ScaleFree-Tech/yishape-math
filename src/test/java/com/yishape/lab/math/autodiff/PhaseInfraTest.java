package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.FloatDiffTensor;
import com.yishape.lab.math.compute.FlatGemm;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.tensor.ContiguousPool;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

/**
 * Regression tests for Phase 1-6 infrastructure features.
 */
public class PhaseInfraTest {

    private static final double TOL = 1e-10;

    // ==================== 6A: ContiguousPool ====================

    @Test
    void testContiguousPoolAcquireRelease() {
        double[] buf = ContiguousPool.acquire(100);
        assertNotNull(buf);
        assertTrue(buf.length >= 100);
        ContiguousPool.release(buf);
        double[] buf2 = ContiguousPool.acquire(100);
        assertNotNull(buf2);
        assertTrue(buf2.length >= 100);
        ContiguousPool.clear();
    }

    @Test
    void testContiguousPoolPowerOfTwoBuckets() {
        double[] b1 = ContiguousPool.acquire(1);
        double[] b2 = ContiguousPool.acquire(100);
        double[] b3 = ContiguousPool.acquire(1000);
        double[] b4 = ContiguousPool.acquire(10000);
        assertTrue(b1.length >= 1);
        assertTrue(b2.length >= 100);
        assertTrue(b3.length >= 1000);
        assertTrue(b4.length >= 10000);
        ContiguousPool.release(b1);
        ContiguousPool.release(b2);
        ContiguousPool.release(b3);
        ContiguousPool.release(b4);
        ContiguousPool.clear();
    }

    @Test
    void testContiguousPoolClear() {
        double[] buf = ContiguousPool.acquire(500);
        ContiguousPool.release(buf);
        ContiguousPool.clear();
        double[] buf2 = ContiguousPool.acquire(500);
        assertNotNull(buf2);
        ContiguousPool.clear();
    }

    @Test
    void testContiguousPoolNullRelease() {
        ContiguousPool.release(null);
        ContiguousPool.clear();
    }

    @Test
    void testContiguousWithPool() {
        IDoubleTensor t = new RereDoubleTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        IDoubleTensor permuted = t.permute(1, 0);
        assertFalse(permuted.isContiguous());
        IDoubleTensor contig = permuted.contiguous();
        assertTrue(contig.isContiguous());
        assertArrayEquals(new int[]{3, 2}, contig.shape());
        assertEquals(1.0, contig.get(0, 0), TOL);
        assertEquals(4.0, contig.get(0, 1), TOL);
    }

    // ==================== 6B: Broadcast binary ops ====================

    @Test
    void testBroadcastAddForward() {
        IDiffTensor a = IDiffTensor.fromTensor(ITensor.tensor(new double[]{1, 2, 3}, 1, 3), true);
        IDiffTensor b = IDiffTensor.fromTensor(ITensor.tensor(new double[]{10, 20, 30, 40, 50, 60}, 2, 3), true);
        IDiffTensor c = a.add(b);
        assertArrayEquals(new int[]{2, 3}, c.shape());
        assertEquals(11.0, c.get(0, 0), TOL);
        assertEquals(63.0, c.get(1, 2), TOL);
    }

    @Test
    void testBroadcastAddBackward() {
        IDiffTensor a = IDiffTensor.fromTensor(ITensor.tensor(new double[]{1, 2, 3}, 1, 3), true);
        IDiffTensor b = IDiffTensor.fromTensor(ITensor.tensor(new double[]{10, 20, 30, 40, 50, 60}, 2, 3), true);
        IDiffTensor c = a.add(b);
        c.backward();
        // add backward: gradOut (all 1s) reduced to each input shape
        // Shape [1,3] reduced from [2,3]: sum over batch → 2 per element
        double[] gradA = a.flattenGrad().getData();
        assertEquals(2.0, gradA[0], TOL);
        assertEquals(2.0, gradA[1], TOL);
        assertEquals(2.0, gradA[2], TOL);
        // Shape [2,3]: no reduction needed, all 1s
        double[] gradB = b.flattenGrad().getData();
        assertEquals(1.0, gradB[0], TOL);
        assertEquals(1.0, gradB[5], TOL);
    }

    @Test
    void testBroadcastNonDiffOther() {
        IDiffTensor a = IDiffTensor.fromTensor(ITensor.tensor(new double[]{1, 2, 3}, 1, 3), true);
        IDoubleTensor b = ITensor.tensor(new double[]{10, 20, 30, 40, 50, 60}, 2, 3);
        IDiffTensor c = a.add(b);
        assertArrayEquals(new int[]{2, 3}, c.shape());
        assertEquals(11.0, c.get(0, 0), TOL);
        assertEquals(63.0, c.get(1, 2), TOL);
    }

    // ==================== Phase 5: AD tensor factories ====================

    @Test
    void testTensorFactory() {
        IDiffTensor t = AD.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        assertTrue(t.requiresGrad());
        assertArrayEquals(new int[]{2, 3}, t.shape());
        assertEquals(1.0, t.get(0, 0), TOL);
        assertEquals(6.0, t.get(1, 2), TOL);
    }

    @Test
    void testZerosTensorFactory() {
        IDiffTensor t = AD.zerosTensor(2, 3);
        assertTrue(t.requiresGrad());
        assertArrayEquals(new int[]{2, 3}, t.shape());
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                assertEquals(0.0, t.get(i, j), TOL);
    }

    @Test
    void testOnesTensorFactory() {
        IDiffTensor t = AD.onesTensor(2, 2);
        assertEquals(1.0, t.get(0, 0), TOL);
        assertEquals(1.0, t.get(1, 1), TOL);
    }

    @Test
    void testFullTensorFactory() {
        IDiffTensor t = AD.fullTensor(7.5, 3, 2);
        assertEquals(7.5, t.get(0, 0), TOL);
        assertEquals(7.5, t.get(2, 1), TOL);
    }

    @Test
    void testArangeFactory() {
        IDiffVector v = AD.arange(5);
        assertEquals(5, v.size());
        assertEquals(0.0, v.get(0), TOL);
        assertEquals(4.0, v.get(4), TOL);

        IDiffVector v2 = AD.arange(2, 5);
        assertEquals(3, v2.size());
        assertEquals(2.0, v2.get(0), TOL);
        assertEquals(4.0, v2.get(2), TOL);

        IDiffVector v3 = AD.arange(0, 10, 3);
        assertEquals(4, v3.size());
        assertEquals(0.0, v3.get(0), TOL);
        assertEquals(3.0, v3.get(1), TOL);
        assertEquals(9.0, v3.get(3), TOL);
    }

    @Test
    void testEyeFactory() {
        IDiffTensor t = AD.eye(3);
        assertArrayEquals(new int[]{3, 3}, t.shape());
        assertEquals(1.0, t.get(0, 0), TOL);
        assertEquals(0.0, t.get(0, 1), TOL);
        assertEquals(1.0, t.get(1, 1), TOL);
        assertEquals(0.0, t.get(2, 0), TOL);
    }

    // ==================== Phase 5: indexSelect / argsort ====================

    @Test
    void testIndexSelect() {
        IDiffTensor t = IDiffTensor.fromTensor(
            ITensor.tensor(new double[]{10, 20, 30, 40, 50, 60, 70, 80}, 4, 2), true);
        IDiffTensor idx = IDiffTensor.fromTensor(ITensor.tensor(new double[]{0, 2}, 2), false);
        IDiffTensor result = t.indexSelect(0, idx);
        assertArrayEquals(new int[]{2, 2}, result.shape());
        assertEquals(10.0, result.get(0, 0), TOL);
        assertEquals(20.0, result.get(0, 1), TOL);
        assertEquals(50.0, result.get(1, 0), TOL);
        assertEquals(60.0, result.get(1, 1), TOL);
    }

    @Test
    void testArgsort() {
        // Data [4,2]: [[3,1],[4,1],[5,9],[2,6]]; descending along dim 0
        // col 0 values: [3,4,5,2] → descending rank: 2,1,0,3
        // col 1 values: [1,1,9,6] → descending rank: 2,3,0,1
        IDiffTensor t = IDiffTensor.fromTensor(
            ITensor.tensor(new double[]{3, 1, 4, 1, 5, 9, 2, 6}, 4, 2), true);
        IDiffTensor result = t.argsort(0, true);
        // Verify we get 4x2 indices back
        assertArrayEquals(new int[]{4, 2}, result.shape());
        // Row with max col0 value (5) = row 2
        assertEquals(2.0, result.get(0, 0), TOL);
        // Row with max col1 value (9) = row 2
        assertEquals(2.0, result.get(0, 1), TOL);
    }

    // ==================== Phase 4: FlatGemm transp ====================

    @Test
    void testFlatGemmTranspNN() {
        int m = 2, k = 3, n = 2;
        double[] a = {1, 2, 3, 4, 5, 6};
        double[] b = {7, 8, 9, 10, 11, 12};
        double[] c = FlatGemm.flatMmulTransp(a, m, k, b, n, 0);
        assertEquals(58.0, c[0], TOL);
        assertEquals(64.0, c[1], TOL);
        assertEquals(139.0, c[2], TOL);
        assertEquals(154.0, c[3], TOL);
    }

    @Test
    void testFlatGemmTranspTN() {
        int m = 2, k = 3, n = 2;
        double[] a = {1, 2, 3, 4, 5, 6};
        double[] b = {7, 8, 9, 10, 11, 12};
        double[] c = FlatGemm.flatMmulTransp(a, m, k, b, n, 1);
        assertEquals(89.0, c[0], TOL);
        assertEquals(98.0, c[1], TOL);
        assertEquals(116.0, c[2], TOL);
        assertEquals(128.0, c[3], TOL);
    }

    @Test
    void testFlatGemmTranspNT() {
        int m = 2, k = 3, n = 2;
        double[] a = {1, 2, 3, 4, 5, 6};
        double[] b = {7, 8, 9, 10, 11, 12};
        double[] c = FlatGemm.flatMmulTransp(a, m, k, b, n, 2);
        assertEquals(50.0, c[0], TOL);
        assertEquals(68.0, c[1], TOL);
        assertEquals(122.0, c[2], TOL);
        assertEquals(167.0, c[3], TOL);
    }

    @Test
    void testFlatGemmTranspTT() {
        int m = 2, k = 3, n = 2;
        double[] a = {1, 2, 3, 4, 5, 6};
        double[] b = {7, 8, 9, 10, 11, 12};
        double[] c = FlatGemm.flatMmulTransp(a, m, k, b, n, 3);
        assertEquals(76.0, c[0], TOL);
        assertEquals(103.0, c[1], TOL);
        assertEquals(100.0, c[2], TOL);
        assertEquals(136.0, c[3], TOL);
    }

    // ==================== Phase 3: FloatDiffTensor ====================

    @Test
    void testFloatDiffTensorCreation() {
        FloatDiffTensor ft = new FloatDiffTensor(
            new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
        assertArrayEquals(new int[]{2, 2}, ft.shape());
        assertTrue(ft.requiresGrad());
        assertEquals(1.0, ft.get(0, 0), TOL);
        assertEquals(4.0, ft.get(1, 1), TOL);
        float[] floatData = ft.getFloatData();
        assertEquals(1.0f, floatData[0], 1e-6f);
        assertEquals(4.0f, floatData[3], 1e-6f);
    }

    @Test
    void testFloatDiffTensorGradient() {
        FloatDiffTensor ft = new FloatDiffTensor(
            new float[]{1.0f, 2.0f, 3.0f}, 3);
        IDiffTensor squared = ft.square();
        assertArrayEquals(new int[]{3}, squared.shape());
        squared.backward();
        IDiffVector g = ft.flattenGrad();
        assertNotNull(g);
        assertEquals(2.0, g.get(0), TOL);
        assertEquals(4.0, g.get(1), TOL);
        assertEquals(6.0, g.get(2), TOL);
    }

    // ==================== Contiguous gradient preservation ====================

    @Test
    void testContiguousPreservesGradient() {
        IDiffTensor t = IDiffTensor.fromTensor(
            ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3), true);
        IDiffTensor permuted = t.permute(1, 0);
        assertFalse(permuted.isContiguous());
        IDiffTensor contig = permuted.contiguous();
        assertTrue(contig.isContiguous());
        IDiffTensor result = contig.sum();
        result.backward();
        IDiffVector grad = t.flattenGrad();
        assertNotNull(grad);
        double[] g = grad.getData();
        assertEquals(1.0, g[0], TOL);
        assertEquals(1.0, g[5], TOL);
    }

    @Test
    void testReshapeAfterPermute() {
        IDiffTensor t = IDiffTensor.fromTensor(
            ITensor.tensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 4), true);
        IDiffTensor permuted = t.permute(1, 0);
        IDiffTensor reshaped = permuted.reshape(2, 6);
        assertArrayEquals(new int[]{2, 6}, reshaped.shape());
        IDiffTensor s = reshaped.sum();
        s.backward();
        double[] grad = t.flattenGrad().getData();
        assertEquals(1.0, grad[0], TOL);
        assertEquals(1.0, grad[11], TOL);
    }
}
