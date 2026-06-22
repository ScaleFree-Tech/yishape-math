package com.yishape.lab.math.autodiff;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;

/**
 * Regression tests for C2 (2026-06-22 deep audit): {@link BatchedDiffTensor#einsum}
 * must prepend batch labels to the user subscript and broadcast shared (non-batched)
 * operands, so einsum is usable inside vmap.
 *
 * <p>Before the fix, the user subscript was passed through unchanged to the
 * underlying tensor whose physical shape carries the batch dim, causing either an
 * axis-count validation error or silently wrong results.</p>
 */
public class BatchedEinsumTest {

    // ==================== helpers ====================

    /** Build a BatchedDiffTensor (depth 1) whose data has shape [B, ...userShape]. */
    private static BatchedDiffTensor batched(double[] data, int... fullShape) {
        return new BatchedDiffTensor(new RereDiffTensor(data, fullShape), 1);
    }

    // ==================== 1. shared constant weight ====================

    /** x=[B,M,K] batched, w=[K,N] shared constant → "mk,kn->mn" gives [B,M,N]. */
    @Test
    void testSharedConstantWeightMatmul() {
        // x: 2 samples of [[1,2],[3,4]] and [[5,6],[7,8]]
        double[] xData = {1, 2, 3, 4, 5, 6, 7, 8};
        BatchedDiffTensor x = batched(xData, 2, 2, 2);
        // w = [[1,1],[1,1]]: each output element = row sum of input
        IDoubleTensor w = new RereDoubleTensor(new double[]{1, 1, 1, 1}, 2, 2);

        IDiffTensor result = x.einsum("mk,kn->mn", w);

        assertInstanceOf(BatchedDiffTensor.class, result);
        assertEquals(1, ((BatchedDiffTensor) result).nestingDepth());
        assertArrayEquals(new int[]{2, 2, 2}, result.shape());
        // sample 0: [[3,3],[7,7]], sample 1: [[11,11],[15,15]]
        assertArrayEquals(new double[]{3, 3, 7, 7, 11, 11, 15, 15},
                result.toDoubleArray(), 1e-12);
    }

    // ==================== 2. shared differentiable weight + gradient flow ====================

    /** w is a differentiable shared parameter; gradient must sum-reduce over batch. */
    @Test
    void testSharedDifferentiableWeightGradient() {
        double[] xData = {1, 2, 3, 4, 5, 6, 7, 8};
        BatchedDiffTensor x = batched(xData, 2, 2, 2);
        RereDiffTensor w = new RereDiffTensor(new double[]{1, 1, 1, 1}, 2, 2); // requiresGrad=true

        IDiffTensor result = x.einsum("mk,kn->mn", w);
        result.sum().backward();

        // dL/dw[k,j] = sum_b sum_i x[b,i,k] (independent of j)
        // k=0: (1+3)+(5+7)=16 ; k=1: (2+4)+(6+8)=20  →  [[16,16],[20,20]]
        assertArrayEquals(new double[]{16, 16, 20, 20}, w.grad().toDoubleArray(), 1e-10);
    }

    /** Cross-check the gradient by central differences on one weight element. */
    @Test
    void testSharedWeightGradientFiniteDifference() {
        double[] xData = {1, 2, 3, 4, 5, 6, 7, 8};
        BatchedDiffTensor x = batched(xData, 2, 2, 2);
        double[] wData = {1.0, 0.5, -0.5, 2.0};
        RereDiffTensor w = new RereDiffTensor(wData.clone(), 2, 2);

        IDiffTensor result = x.einsum("mk,kn->mn", w);
        result.sum().backward();
        double analytical = w.grad().toDoubleArray()[1]; // w[0,1]

        // Central difference on w[0,1]; loss = total sum of all result elements.
        double eps = 1e-6;
        double[] wp = wData.clone(); wp[1] += eps;
        double[] wm = wData.clone(); wm[1] -= eps;
        double lp = totalSum(batched(xData, 2, 2, 2).einsum("mk,kn->mn", new RereDiffTensor(wp, 2, 2)));
        double lm = totalSum(batched(xData, 2, 2, 2).einsum("mk,kn->mn", new RereDiffTensor(wm, 2, 2)));
        double numeric = (lp - lm) / (2 * eps);
        assertEquals(analytical, numeric, 1e-6, "einsum grad to shared weight must match finite diff");
    }

    /** Sum all elements of a tensor to a scalar (loss for finite-difference checks). */
    private static double totalSum(IDiffTensor t) {
        double[] d = t.toDoubleArray();
        double s = 0;
        for (double v : d) s += v;
        return s;
    }

    // ==================== 3. both operands batched ====================

    @Test
    void testBothOperandsBatched() {
        // a=[2,2,2], b=[2,2,2] both batched
        double[] aData = {1, 2, 3, 4, 5, 6, 7, 8};
        double[] bData = {1, 0, 0, 1, 1, 0, 0, 1}; // identity per sample
        BatchedDiffTensor a = batched(aData, 2, 2, 2);
        BatchedDiffTensor b = batched(bData, 2, 2, 2);

        IDiffTensor result = a.einsum("mk,kn->mn", b);

        assertArrayEquals(new int[]{2, 2, 2}, result.shape());
        // identity matmul → result == a
        assertArrayEquals(aData, result.toDoubleArray(), 1e-12);
    }

    // ==================== 4. single-input reduction ====================

    /** "ij->j" on batched [B,M,N] → per-sample column sums [B,N]. */
    @Test
    void testSingleInputReduction() {
        double[] xData = {1, 2, 3, 4, 5, 6, 7, 8}; // 2 samples [[1,2],[3,4]], [[5,6],[7,8]]
        BatchedDiffTensor x = batched(xData, 2, 2, 2);

        IDiffTensor result = x.einsum("ij->j");

        assertArrayEquals(new int[]{2, 2}, result.shape());
        // sample 0 col sums [4,6], sample 1 [12,14]
        assertArrayEquals(new double[]{4, 6, 12, 14}, result.toDoubleArray(), 1e-12);
    }

    /** "ij->" (scalar per sample) is NOT supported: EinsumParser rejects empty
     *  output (pre-existing limitation, out of scope for the C2 batch-shift fix). */

    // ==================== 5. nested vmap (nestingDepth=2) ====================

    /** Multi-batch label prepend: data [B2,B1,M,K], depth 2. */
    @Test
    void testNestedVmapDepth2() {
        // x = 1..16 shaped [2,2,2,2] (B2=2,B1=2,M=2,K=2)
        double[] xData = new double[16];
        for (int i = 0; i < 16; i++) xData[i] = i + 1;
        BatchedDiffTensor x = new BatchedDiffTensor(new RereDiffTensor(xData, 2, 2, 2, 2), 2);
        // w = [[1,1],[1,1]]: C[b2,b1,i,j] = sum_k x[b2,b1,i,k]
        IDoubleTensor w = new RereDoubleTensor(new double[]{1, 1, 1, 1}, 2, 2);

        IDiffTensor result = x.einsum("mk,kn->mn", w);

        assertInstanceOf(BatchedDiffTensor.class, result);
        assertEquals(2, ((BatchedDiffTensor) result).nestingDepth());
        assertArrayEquals(new int[]{2, 2, 2, 2}, result.shape());
        // Each [b2,b1] block: row sums.
        // x[0,0]=[[1,2],[3,4]]→[[3,3],[7,7]]; x[0,1]=[[5,6],[7,8]]→[[11,11],[15,15]]
        // x[1,0]=[[9,10],[11,12]]→[[19,19],[23,23]]; x[1,1]=[[13,14],[15,16]]→[[27,27],[31,31]]
        assertArrayEquals(new double[]{
                3, 3, 7, 7, 11, 11, 15, 15,
                19, 19, 23, 23, 27, 27, 31, 31
        }, result.toDoubleArray(), 1e-12);
    }

    // ==================== 6. label collision avoidance ====================

    /** User subscript uses 'a' (the first batch-label candidate); picker must skip it. */
    @Test
    void testLabelCollisionAvoidance() {
        // "am,an->mn" contracts 'a'; batch label must NOT be 'a' (used) → picks 'b'.
        double[] xData = {1, 2, 3, 4, 5, 6, 7, 8}; // [2,2,2]: M=2, A=2
        BatchedDiffTensor x = batched(xData, 2, 2, 2);
        // y shared [A,N]=[2,2] identity → result = x per sample
        IDoubleTensor y = new RereDoubleTensor(new double[]{1, 0, 0, 1}, 2, 2);

        IDiffTensor result = x.einsum("am,an->mn", y);

        assertArrayEquals(new int[]{2, 2, 2}, result.shape());
        assertArrayEquals(xData, result.toDoubleArray(), 1e-12);
    }

    // ==================== 7. batch context preserved downstream ====================

    @Test
    void testBatchContextPreservedForDownstreamOps() {
        double[] xData = {1, 2, 3, 4, 5, 6, 7, 8};
        BatchedDiffTensor x = batched(xData, 2, 2, 2);
        IDoubleTensor w = new RereDoubleTensor(new double[]{1, 1, 1, 1}, 2, 2);

        IDiffTensor result = x.einsum("mk,kn->mn", w);
        assertInstanceOf(BatchedDiffTensor.class, result);

        // Downstream dim-shift op (sum over user dim 0) must work and keep batch.
        IDiffTensor reduced = result.sum(0, false); // [B, N]
        assertInstanceOf(BatchedDiffTensor.class, reduced);
        assertArrayEquals(new int[]{2, 2}, reduced.shape());
        // sample 0: row sums of [[3,3],[7,7]] = [10,10]; sample 1: [26,26]
        assertArrayEquals(new double[]{10, 10, 26, 26}, reduced.toDoubleArray(), 1e-12);
    }

    // ==================== 8. end-to-end via vmapT ====================

    /** einsum inside a real vmapT call — per-sample matmul with a shared weight. */
    @Test
    void testEinsumInsideVmapT() {
        // Two input samples [2,2] each; shared weight [2,2]
        IDiffTensor s0 = AD.leafTensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor s1 = AD.leafTensor(new double[]{5, 6, 7, 8}, 2, 2);
        IDiffTensor w = AD.leafTensor(new double[]{1, 1, 1, 1}, 2, 2);

        IDiffTensor[] out = AD.vmapT(
                x -> x.einsum("mk,kn->mn", w),
                java.util.List.of(s0, s1));

        assertEquals(2, out.length);
        // sample 0: [[3,3],[7,7]], sample 1: [[11,11],[15,15]]
        assertArrayEquals(new double[]{3, 3, 7, 7}, out[0].toDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{11, 11, 15, 15}, out[1].toDoubleArray(), 1e-12);
    }
}
