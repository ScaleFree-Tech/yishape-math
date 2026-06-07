package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import java.util.function.Function;

/**
 * Tests for nested vmap via BatchedDiffTensor with nestingDepth.
 */
public class NestedVmapTest {

    // ==================== Basic vmapStackedT ====================

    @Test
    void testVmapStackedTReturnsBatchedDiffTensor() {
        RereDiffTensor x1 = new RereDiffTensor(new double[]{1, 2, 3, 4}, 4);
        RereDiffTensor x2 = new RereDiffTensor(new double[]{5, 6, 7, 8}, 4);
        RereDiffTensor x3 = new RereDiffTensor(new double[]{9, 10, 11, 12}, 4);

        Function<IDiffTensor, IDiffTensor> fn = x -> x.mul(2.0);

        IDiffTensor result = AD.vmapStackedT(fn, List.of(x1, x2, x3));
        assertInstanceOf(BatchedDiffTensor.class, result);
        BatchedDiffTensor bdt = (BatchedDiffTensor) result;
        assertEquals(1, bdt.nestingDepth());
        assertArrayEquals(new int[]{3, 4}, bdt.shape());
    }

    @Test
    void testVmapStackedTSum() {
        RereDiffTensor x1 = new RereDiffTensor(new double[]{1, 2, 3, 4}, 4);
        RereDiffTensor x2 = new RereDiffTensor(new double[]{5, 6, 7, 8}, 4);
        RereDiffTensor x3 = new RereDiffTensor(new double[]{9, 10, 11, 12}, 4);

        Function<IDiffTensor, IDiffTensor> fn = x -> x.sum();

        IDiffTensor result = AD.vmapStackedT(fn, List.of(x1, x2, x3));
        BatchedDiffTensor bdt = (BatchedDiffTensor) result;
        assertArrayEquals(new int[]{3}, bdt.shape());
        double[] vals = bdt.toDoubleArray();
        assertEquals(10.0, vals[0], 1e-12);
        assertEquals(26.0, vals[1], 1e-12);
        assertEquals(42.0, vals[2], 1e-12);
    }

    @Test
    void testVmapStackedTWithRequiresGrad() {
        RereDiffTensor x1 = new RereDiffTensor(new double[]{1, 2, 3, 4}, 4);
        x1.setRequiresGrad(true);
        RereDiffTensor x2 = new RereDiffTensor(new double[]{5, 6, 7, 8}, 4);
        x2.setRequiresGrad(true);

        Function<IDiffTensor, IDiffTensor> fn = x -> x.sum();

        IDiffTensor result = AD.vmapStackedT(fn, List.of(x1, x2));
        result.backward();

        assertArrayEquals(new double[]{1, 1, 1, 1}, x1.gradData(), 1e-12);
        assertArrayEquals(new double[]{1, 1, 1, 1}, x2.gradData(), 1e-12);
    }

    // ==================== Nested vmap depth=2 (data: [B2, B1, D1, D2] = [2, 3, 4, 5]) ====================

    /**
     * Helper: create [2, 3, 4, 5] data and wrap with depth=2.
     * User sees [4, 5] with two batch dims [2, 3].
     */
    private BatchedDiffTensor depth2Data() {
        double[] raw = new double[2 * 3 * 4 * 5];
        for (int i = 0; i < raw.length; i++) raw[i] = i + 1;
        RereDiffTensor data = new RereDiffTensor(raw, 2, 3, 4, 5);
        return new BatchedDiffTensor(data, 2);
    }

    @Test
    void testNestingDepth2Construction() {
        BatchedDiffTensor bdt = depth2Data();
        assertEquals(2, bdt.nestingDepth());
        assertArrayEquals(new int[]{2, 3, 4, 5}, bdt.shape());
    }

    @Test
    void testNestingDepth2Reshape() {
        BatchedDiffTensor bdt = depth2Data();
        // User sees [4, 5], reshapes to [2, 2, 5]
        // Full: prepend [2,3] → [2, 3, 2, 2, 5], total = 2*3*2*2*5 = 120 = 2*3*4*5
        IDiffTensor reshaped = bdt.reshape(2, 2, 5);
        assertArrayEquals(new int[]{2, 3, 2, 2, 5}, reshaped.shape());
    }

    @Test
    void testNestingDepth2Permute() {
        BatchedDiffTensor bdt = depth2Data();
        // User sees [4, 5], permutes to [5, 4] with permute(1, 0)
        // shiftPermute: batch [0,1], shifted user dims: shift(1)=3, shift(0)=2 → [0,1,3,2]
        IDiffTensor permuted = bdt.permute(1, 0);
        assertArrayEquals(new int[]{2, 3, 5, 4}, permuted.shape());
    }

    @Test
    void testNestingDepth2Transpose() {
        BatchedDiffTensor bdt = depth2Data();
        // User transposes dims 0 and 1 → actual dims 2 and 3
        IDiffTensor transposed = bdt.transpose(0, 1);
        assertArrayEquals(new int[]{2, 3, 5, 4}, transposed.shape());
    }

    @Test
    void testNestingDepth2Sum() {
        BatchedDiffTensor bdt = depth2Data();
        // User sees [4, 5], sum over dim 0 → reduce actual dim 2
        IDiffTensor summed = bdt.sum(0, false);
        // Shape [2, 3, 5]: batch dims + remaining feature dim
        assertArrayEquals(new int[]{2, 3, 5}, summed.shape());

        // Sum over dim 1 → reduce actual dim 3
        IDiffTensor summed2 = bdt.sum(1, false);
        assertArrayEquals(new int[]{2, 3, 4}, summed2.shape());
    }

    @Test
    void testNestingDepth2SumAll() {
        BatchedDiffTensor bdt = depth2Data();
        // Reduces all non-batch dims (dims 2 and 3)
        IDiffTensor result = bdt.sum();
        assertArrayEquals(new int[]{2, 3}, result.shape());
    }

    @Test
    void testNestingDepth2Squeeze() {
        // Create [2, 3, 1, 5] data
        RereDiffTensor data = new RereDiffTensor(new double[2 * 3 * 1 * 5], 2, 3, 1, 5);
        BatchedDiffTensor bdt = new BatchedDiffTensor(data, 2);

        // User sees [1, 5], squeeze dim 0 → squeeze actual dim 2
        IDiffTensor squeezed = bdt.squeeze(0);
        assertArrayEquals(new int[]{2, 3, 5}, squeezed.shape());
    }

    @Test
    void testNestingDepth2Unsqueeze() {
        BatchedDiffTensor bdt = depth2Data();
        // User sees [4, 5], unsqueeze at dim 0 → insert at actual dim 2
        IDiffTensor result = bdt.unsqueeze(0);
        assertArrayEquals(new int[]{2, 3, 1, 4, 5}, result.shape());
    }

    @Test
    void testNestingDepth2Select() {
        BatchedDiffTensor bdt = depth2Data();
        // Select index 1 along user dim 0 (actual dim 2)
        IDiffTensor selected = bdt.select(0, 1);
        assertArrayEquals(new int[]{2, 3, 5}, selected.shape());
    }

    @Test
    void testNestingDepth2Tile() {
        BatchedDiffTensor bdt = depth2Data();
        // User sees [4, 5], tile by [2, 1] — repeat dim 0 twice
        IDiffTensor tiled = bdt.tile(2, 1);
        assertArrayEquals(new int[]{2, 3, 8, 5}, tiled.shape());
    }

    @Test
    void testNestingDepth2Clone() {
        BatchedDiffTensor bdt = depth2Data();
        IDiffTensor cloned = bdt.clone();
        assertInstanceOf(BatchedDiffTensor.class, cloned);
        assertEquals(2, ((BatchedDiffTensor) cloned).nestingDepth());
        assertArrayEquals(new int[]{2, 3, 4, 5}, cloned.shape());
    }

    @Test
    void testNestingDepth2Copy() {
        BatchedDiffTensor bdt = depth2Data();
        IDiffTensor copied = bdt.copy();
        assertInstanceOf(BatchedDiffTensor.class, copied);
        assertEquals(2, ((BatchedDiffTensor) copied).nestingDepth());
        assertArrayEquals(new int[]{2, 3, 4, 5}, copied.shape());
    }

    // ==================== Depth 3 ====================

    @Test
    void testNestingDepth3() {
        // [2, 3, 4, 5, 6] → user sees [5, 6]
        double[] raw = new double[2 * 3 * 4 * 5 * 6];
        for (int i = 0; i < raw.length; i++) raw[i] = i + 1;
        RereDiffTensor data = new RereDiffTensor(raw, 2, 3, 4, 5, 6);

        BatchedDiffTensor bdt = new BatchedDiffTensor(data, 3);
        assertEquals(3, bdt.nestingDepth());
        assertArrayEquals(new int[]{2, 3, 4, 5, 6}, bdt.shape());

        // Sum over user dim 0 (actual dim 3) → [2, 3, 4, 6]
        IDiffTensor reduced = bdt.sum(0, false);
        assertArrayEquals(new int[]{2, 3, 4, 6}, reduced.shape());

        // Reshape [5, 6] → [2, 15] → full: [2, 3, 4, 2, 15] (total = 2*3*4*2*15 = 720 = 2*3*4*5*6)
        IDiffTensor reshaped = bdt.reshape(2, 15);
        assertArrayEquals(new int[]{2, 3, 4, 2, 15}, reshaped.shape());
    }

    // ==================== Flat model verification ====================

    @Test
    void testNestingUnwrapPreservesFlatModel() {
        BatchedDiffTensor bdt = depth2Data();
        IDiffTensor unwrapped = bdt.unwrap();
        assertFalse(unwrapped instanceof BatchedDiffTensor);
        assertArrayEquals(new int[]{2, 3, 4, 5}, unwrapped.shape());
    }

    @Test
    void testNestingWrapPreservesDepth() {
        BatchedDiffTensor bdt = depth2Data();
        IDiffTensor result = bdt.mul(2.0);
        assertInstanceOf(BatchedDiffTensor.class, result);
        assertEquals(2, ((BatchedDiffTensor) result).nestingDepth());
    }

    // ==================== Nested vmap with actual AD gradients ====================

    @Test
    void testNestedVmapReluSquareGradient() {
        // [2, 3, 4] — depth=2, user sees [4] (1 non-batch dim)
        RereDiffTensor data = new RereDiffTensor(new double[]{
            1, -2, 3, 4,
            5, -6, 7, 8,
            9, 10, -11, 12,
            13, 14, 15, -16,
            17, 18, 19, 20,
            21, 22, 23, 24
        }, 2, 3, 4);
        data.setRequiresGrad(true);

        BatchedDiffTensor bdt = new BatchedDiffTensor(data, 2);

        // relu then square then sum all non-batch dims
        IDiffTensor result = bdt.relu().square().sum();
        result.backward();

        // d/dx of relu(x)^2 summed = 2*x if x>0 else 0
        double[] grad = data.gradData();
        double[] raw = data.toDoubleArray();
        for (int i = 0; i < raw.length; i++) {
            double expected = raw[i] > 0 ? 2.0 * raw[i] : 0.0;
            assertEquals(expected, grad[i], 1e-10);
        }
    }

    @Test
    void testNestedVmapSumThenBackward() {
        // [2, 2, 3] — depth=2, user sees [3]
        RereDiffTensor data = new RereDiffTensor(new double[]{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12
        }, 2, 2, 3);
        data.setRequiresGrad(true);

        BatchedDiffTensor bdt = new BatchedDiffTensor(data, 2);

        // Element-wise mul then reduce
        IDiffTensor result = bdt.mul(2.0).sum();
        result.backward();

        // d/dx of sum(2*x) = 2 for each element
        double[] grad = data.gradData();
        for (double g : grad) {
            assertEquals(2.0, g, 1e-12);
        }
    }

    @Test
    void testNestingDepth2Pad() {
        // [2, 3, 4, 5] — user sees [4, 5], pad with 1 on each side of each user dim
        BatchedDiffTensor bdt = depth2Data();
        IDiffTensor padded = bdt.pad(new int[][]{{1, 1}, {1, 1}}, "constant", 0.0);
        // user dims: 4+2=6, 5+2=7 → full: [2, 3, 6, 7]
        assertArrayEquals(new int[]{2, 3, 6, 7}, padded.shape());
    }

    @Test
    void testNestingDepth2Expand() {
        // expand works for depth>1 when userShape matches existing non-batch dims
        // (no new leading dims — those require spanPad/unsqueeze which is complex)
        BatchedDiffTensor bdt = depth2Data();
        // User expands [4, 5] to [4, 5] — no-op expansion
        IDiffTensor expanded = bdt.expand(4, 5);
        assertArrayEquals(new int[]{2, 3, 4, 5}, expanded.shape());
    }

    @Test
    void testNestingDepth2BroadcastTo() {
        BatchedDiffTensor bdt = depth2Data();
        // Broadcast [4, 5] to [4, 5] — no-op
        IDiffTensor broadcasted = bdt.broadcastTo(4, 5);
        assertArrayEquals(new int[]{2, 3, 4, 5}, broadcasted.shape());
    }
}
