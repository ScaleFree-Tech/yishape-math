package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Contract tests for {@link BatchedDiffTensor} exit methods that previously
 * leaked batch context (split/chunk/unbind/unstack/detach) plus the new
 * isBatched()/batchDimCount() interface flags.
 *
 * <p>These define the desired post-fix contract: results of structural splits
 * and detach must remain wrapped so batch context flows through.
 */
public class BatchedDiffTensorContractTest {

    /** [3, 4] batched tensor, depth 1 — user-visible shape [4]. */
    private BatchedDiffTensor batched3x4() {
        double[] raw = new double[12];
        for (int i = 0; i < 12; i++) raw[i] = i + 1;
        RereDiffTensor data = new RereDiffTensor(raw, 3, 4);
        return new BatchedDiffTensor(data, 1);
    }

    // ==================== isBatched / batchDimCount ====================

    @Test
    void isBatchedTrueForBatchedTensor() {
        BatchedDiffTensor bdt = batched3x4();
        assertTrue(bdt.isBatched());
        assertEquals(1, bdt.batchDimCount());
    }

    @Test
    void isBatchedFalseForPlainTensor() {
        RereDiffTensor plain = new RereDiffTensor(new double[]{1, 2, 3, 4}, 4);
        assertFalse(plain.isBatched());
        assertEquals(0, plain.batchDimCount());
    }

    @Test
    void batchDimCountMatchesNestingDepth() {
        // depth 2: [2, 3, 4, 5], user sees [4, 5]
        RereDiffTensor data = new RereDiffTensor(new double[2 * 3 * 4 * 5], 2, 3, 4, 5);
        BatchedDiffTensor bdt = new BatchedDiffTensor(data, 2);
        assertTrue(bdt.isBatched());
        assertEquals(2, bdt.batchDimCount());
    }

    // ==================== split / chunk / unbind / unstack ====================

    @Test
    void splitPreservesBatchWrapping() {
        BatchedDiffTensor bdt = batched3x4();
        // user sees [4], split size 2 along user dim 0 → 2 chunks each [3, 2]
        IDiffTensor[] parts = bdt.split(2, 0);
        assertEquals(2, parts.length);
        for (IDiffTensor p : parts) {
            assertInstanceOf(BatchedDiffTensor.class, p,
                "split elements must preserve batch wrapping");
            assertEquals(1, ((BatchedDiffTensor) p).nestingDepth());
            assertArrayEquals(new int[]{3, 2}, p.shape());
        }
        // Verify values: first chunk cols 0-1, second cols 2-3
        assertArrayEquals(new double[]{1, 2, 5, 6, 9, 10}, parts[0].toDoubleArray());
        assertArrayEquals(new double[]{3, 4, 7, 8, 11, 12}, parts[1].toDoubleArray());
    }

    @Test
    void splitSizesPreservesBatchWrapping() {
        BatchedDiffTensor bdt = batched3x4();
        IDiffTensor[] parts = bdt.split(new int[]{1, 3}, 0);
        assertEquals(2, parts.length);
        assertArrayEquals(new int[]{3, 1}, parts[0].shape());
        assertArrayEquals(new int[]{3, 3}, parts[1].shape());
        for (IDiffTensor p : parts) {
            assertInstanceOf(BatchedDiffTensor.class, p);
            assertEquals(1, ((BatchedDiffTensor) p).nestingDepth());
        }
    }

    @Test
    void chunkPreservesBatchWrapping() {
        BatchedDiffTensor bdt = batched3x4();
        IDiffTensor[] parts = bdt.chunk(2, 0);
        assertEquals(2, parts.length);
        for (IDiffTensor p : parts) {
            assertInstanceOf(BatchedDiffTensor.class, p);
            assertEquals(1, ((BatchedDiffTensor) p).nestingDepth());
            assertArrayEquals(new int[]{3, 2}, p.shape());
        }
    }

    @Test
    void unbindPreservesBatchWrapping() {
        BatchedDiffTensor bdt = batched3x4();
        // unbind along user dim 0 (the [4] dim) → 4 slices each [3]
        IDiffTensor[] slices = bdt.unbind(0);
        assertEquals(4, slices.length);
        for (IDiffTensor s : slices) {
            assertInstanceOf(BatchedDiffTensor.class, s);
            assertEquals(1, ((BatchedDiffTensor) s).nestingDepth());
            assertArrayEquals(new int[]{3}, s.shape());
        }
        // Slice i holds column i across the 3 samples
        assertArrayEquals(new double[]{1, 5, 9}, slices[0].toDoubleArray());
        assertArrayEquals(new double[]{4, 8, 12}, slices[3].toDoubleArray());
    }

    @Test
    void unstackPreservesBatchWrapping() {
        BatchedDiffTensor bdt = batched3x4();
        List<IDoubleTensor> slices = bdt.unstack(0);
        assertEquals(4, slices.size());
        for (IDoubleTensor s : slices) {
            assertInstanceOf(BatchedDiffTensor.class, s);
            assertEquals(1, ((BatchedDiffTensor) s).nestingDepth());
            assertArrayEquals(new int[]{3}, s.shape());
        }
    }

    // ==================== detach ====================

    @Test
    void detachPreservesBatchWrapping() {
        RereDiffTensor data = new RereDiffTensor(new double[12], 3, 4);
        data.setRequiresGrad(true);
        BatchedDiffTensor bdt = new BatchedDiffTensor(data, 1);
        IDiffTensor detached = bdt.detach();
        assertInstanceOf(BatchedDiffTensor.class, detached,
            "detach must preserve batch wrapping");
        assertEquals(1, ((BatchedDiffTensor) detached).nestingDepth());
        assertArrayEquals(new int[]{3, 4}, detached.shape());
        assertFalse(detached.requiresGrad());
        // Detached tensor's dim-shifted ops still work (batch context intact)
        IDiffTensor summed = detached.sum(0, false);
        assertArrayEquals(new int[]{3}, summed.shape());
    }

    // ==================== depth-2 split sanity ====================

    @Test
    void splitDepth2PreservesBatchWrapping() {
        // [2, 3, 4, 5], depth 2, user sees [4, 5]
        double[] raw = new double[2 * 3 * 4 * 5];
        for (int i = 0; i < raw.length; i++) raw[i] = i + 1;
        RereDiffTensor data = new RereDiffTensor(raw, 2, 3, 4, 5);
        BatchedDiffTensor bdt = new BatchedDiffTensor(data, 2);

        // split user dim 1 (the [5] dim) into size 2 → 3 parts each [2,3,4,2]
        IDiffTensor[] parts = bdt.split(2, 1);
        assertEquals(3, parts.length); // 5 = 2+2+1
        for (IDiffTensor p : parts) {
            assertInstanceOf(BatchedDiffTensor.class, p);
            assertEquals(2, ((BatchedDiffTensor) p).nestingDepth());
        }
        assertArrayEquals(new int[]{2, 3, 4, 2}, parts[0].shape());
        assertArrayEquals(new int[]{2, 3, 4, 1}, parts[2].shape());
    }
}
