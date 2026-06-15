package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Defensive regression test for DiffTensorNormNN.rope() batch-dim absorption bug.
 *
 * The bug: seqLen was computed as totalSize / headDim, which for batched 3D input
 * [B, seqLen, headDim] gives totalSize/headDim = B * seqLen (not seqLen).
 * This caused incorrect position indexing and wrong rotary embeddings.
 *
 * Fix: compute seqLen from shape[rank-2] directly, and wrap forward+backward
 * in a batch loop.
 *
 * These tests verify:
 * 1. Unbatched 2D input [seqLen, headDim] produces correct values
 * 2. Batched 3D input [B, seqLen, headDim] produces same per-batch results as unbatched
 * 3. Backward through rope with batched input propagates gradients correctly
 */
public class RopeBatchDefenseTest {

    /**
     * Reference rope implementation that exactly mirrors DiffTensorNormNN.rope().
     * Uses: consecutive elements as pairs (2*i, 2*i+1), theta = pos / base^(2*i/halfDim).
     */
    private static double[] referenceRope(double[] x, int seqLen, int headDim, int halfDim, double base) {
        double[] result = new double[seqLen * headDim];
        double[] cosTable = new double[seqLen * halfDim];
        double[] sinTable = new double[seqLen * halfDim];
        for (int pos = 0; pos < seqLen; pos++) {
            int tblOff = pos * halfDim;
            for (int i = 0; i < halfDim; i++) {
                double theta = pos / Math.pow(base, 2.0 * i / halfDim);
                cosTable[tblOff + i] = Math.cos(theta);
                sinTable[tblOff + i] = Math.sin(theta);
            }
        }
        for (int pos = 0; pos < seqLen; pos++) {
            int posOff = pos * headDim;
            int tblOff = pos * halfDim;
            for (int i = 0; i < halfDim; i++) {
                int idx2i = posOff + 2 * i;
                int idx2i1 = idx2i + 1;
                double x1 = x[idx2i];
                double x2 = x[idx2i1];
                double c = cosTable[tblOff + i];
                double s = sinTable[tblOff + i];
                result[idx2i]  = x1 * c - x2 * s;
                result[idx2i1] = x1 * s + x2 * c;
            }
        }
        return result;
    }

    @Test
    void testRope2DUnbatched() {
        // seqLen=4, headDim=4 — no batch dimension
        int seqLen = 4, headDim = 4;
        double[] raw = new double[seqLen * headDim];
        for (int i = 0; i < raw.length; i++) raw[i] = i + 1.0;

        RereDiffTensor t = new RereDiffTensor(raw, seqLen, headDim);
        t.setRequiresGrad(true);
        IDiffTensor result = t.rope(headDim / 2, 100, 10000.0);

        assertArrayEquals(new int[]{seqLen, headDim}, result.shape());

        double[] expected = referenceRope(raw, seqLen, headDim, headDim / 2, 10000.0);
        double[] actual = result.toDoubleArray();
        assertArrayEquals(expected, actual, 1e-8);
    }

    @Test
    void testRope3DBatched() {
        // batch=2, seqLen=4, headDim=4
        int batch = 2, seqLen = 4, headDim = 4;
        int perBatch = seqLen * headDim;
        double[] raw = new double[batch * perBatch];
        for (int i = 0; i < raw.length; i++) raw[i] = i + 1.0;

        RereDiffTensor t = new RereDiffTensor(raw, batch, seqLen, headDim);
        t.setRequiresGrad(true);
        IDiffTensor result = t.rope(headDim / 2, 100, 10000.0);

        assertArrayEquals(new int[]{batch, seqLen, headDim}, result.shape());

        double[] actual = result.toDoubleArray();
        double[] expected;

        // Each batch should produce the same result as if processed independently
        for (int b = 0; b < batch; b++) {
            double[] batchSlice = new double[perBatch];
            System.arraycopy(raw, b * perBatch, batchSlice, 0, perBatch);
            expected = referenceRope(batchSlice, seqLen, headDim, headDim / 2, 10000.0);
            for (int i = 0; i < perBatch; i++) {
                assertEquals(expected[i], actual[b * perBatch + i], 1e-8,
                    "batch " + b + " pos " + (i / headDim) + " dim " + (i % headDim));
            }
        }
    }

    @Test
    void testRope3DBatchedBackward() {
        // Verify that backward through batched rope produces correct gradients
        int batch = 2, seqLen = 3, headDim = 2;
        double[] raw = new double[batch * seqLen * headDim];
        for (int i = 0; i < raw.length; i++) raw[i] = i + 1.0;

        RereDiffTensor t = new RereDiffTensor(raw, batch, seqLen, headDim);
        t.setRequiresGrad(true);
        IDiffTensor result = t.rope(1, 100, 10000.0);
        result.backward();

        // Gradient should exist and have correct shape
        double[] grad = t.gradData();
        assertNotNull(grad, "gradient should not be null");
        assertEquals(raw.length, grad.length, "gradient length should match input");

        // Gradient should NOT be all zeros (rope is differentiable)
        boolean hasNonZero = false;
        for (double v : grad) {
            if (Math.abs(v) > 1e-12) { hasNonZero = true; break; }
        }
        assertTrue(hasNonZero, "gradient should have non-zero values");
    }

    @Test
    void testRopeBatchIndependence() {
        // Verify that batches are processed independently:
        // changing data in batch 1 should not affect batch 0 output
        int batch = 2, seqLen = 3, headDim = 4;

        double[] raw = new double[batch * seqLen * headDim];
        for (int i = 0; i < raw.length; i++) raw[i] = 1.0;

        // Reference: all-ones input
        RereDiffTensor t1 = new RereDiffTensor(raw.clone(), batch, seqLen, headDim);
        t1.setRequiresGrad(true);
        double[] out1 = t1.rope(headDim / 2, 100, 10000.0).toDoubleArray();

        // Change batch 1 data only
        for (int i = seqLen * headDim; i < raw.length; i++) raw[i] = 100.0;
        RereDiffTensor t2 = new RereDiffTensor(raw, batch, seqLen, headDim);
        t2.setRequiresGrad(true);
        double[] out2 = t2.rope(headDim / 2, 100, 10000.0).toDoubleArray();

        // Batch 0 output should be identical in both cases
        int perBatch = seqLen * headDim;
        for (int i = 0; i < perBatch; i++) {
            assertEquals(out1[i], out2[i], 1e-10,
                "batch 0 should be unaffected by batch 1 data change");
        }
        // Batch 1 output should differ
        boolean batch1Differs = false;
        for (int i = perBatch; i < 2 * perBatch; i++) {
            if (Math.abs(out1[i] - out2[i]) > 1e-10) {
                batch1Differs = true;
                break;
            }
        }
        assertTrue(batch1Differs, "batch 1 should reflect data change");
    }

    @Test
    void testRopeSingleBatchRank3() {
        // Edge case: batch=1, seqLen=4, headDim=6 — should behave like 2D
        int seqLen = 4, headDim = 6;
        double[] raw2d = new double[seqLen * headDim];
        double[] raw3d = new double[seqLen * headDim];
        for (int i = 0; i < raw2d.length; i++) {
            raw2d[i] = i + 1.0;
            raw3d[i] = i + 1.0;
        }

        RereDiffTensor t2d = new RereDiffTensor(raw2d, seqLen, headDim);
        t2d.setRequiresGrad(true);
        double[] out2d = t2d.rope(headDim / 2, 100, 10000.0).toDoubleArray();

        RereDiffTensor t3d = new RereDiffTensor(raw3d, 1, seqLen, headDim);
        t3d.setRequiresGrad(true);
        double[] out3d = t3d.rope(headDim / 2, 100, 10000.0).toDoubleArray();

        assertArrayEquals(out2d, out3d, 1e-10,
            "single-batch 3D should produce same output as 2D");
    }
}
