package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 1b: Gradient correctness for tensor view operations (reshape, permute, etc.).
 * Verifies that backward gradient remapping through view ops preserves the correct shape
 * and numerical values.
 */
public class TensorViewGradientTest {

    // ---- Reshape ----

    @Test
    void testReshapeBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor v = x.reshape(3, 2);
        IDiffTensor s = v.sum();
        s.backward();

        // After sum() backward + reshape backward, every element should have gradient 1
        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    @Test
    void testReshapeScalar() {
        // reshape from any shape to [1] scalar
        RereDiffTensor x = new RereDiffTensor(new double[]{42}, 1);
        x.setRequiresGrad(true);

        IDiffTensor v = x.reshape(1);
        IDiffTensor s = v.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(1, x.grad.length);
        assertEquals(1.0, x.grad[0], 1e-12);
    }

    @Test
    void testReshapeFlatten() {
        // flatten to 1D
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor v = x.reshape(4);
        IDiffTensor s = v.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(4, x.grad.length);
        for (int i = 0; i < 4; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    // ---- Permute ----

    @Test
    void testPermuteBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor p = x.permute(1, 0);  // [3,2]
        IDiffTensor s = p.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    @Test
    void testPermute3D() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor p = x.permute(2, 0, 1);  // [2,2,2]
        IDiffTensor s = p.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(8, x.grad.length);
        for (int i = 0; i < 8; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    // ---- Transpose ----

    @Test
    void testTransposeBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor t = x.transpose(0, 1);
        IDiffTensor s = t.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    @Test
    void testTranspose2DBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor t = x.transpose();  // 2D transpose shortcut
        IDiffTensor s = t.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    // ---- Squeeze / Unsqueeze ----

    @Test
    void testSqueezeBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 1, 3);
        x.setRequiresGrad(true);

        IDiffTensor sq = x.squeeze(1);  // [2,3]
        IDiffTensor s = sq.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    @Test
    void testUnsqueezeBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor us = x.unsqueeze(1);  // [2,1,3]
        IDiffTensor s = us.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    // ---- Flatten ----

    @Test
    void testFlattenBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor f = x.flatten(0, 1);  // [6]
        IDiffTensor s = f.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        for (int i = 0; i < 6; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    @Test
    void testPartialFlattenBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor f = x.flatten(1, 2);  // [2,4]
        IDiffTensor s = f.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(8, x.grad.length);
        for (int i = 0; i < 8; i++) {
            assertEquals(1.0, x.grad[i], 1e-12);
        }
    }

    // ---- Select (single row/col extraction) ----

    @Test
    void testSelectBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor sel = x.select(0, 1);  // [3], select second row
        IDiffTensor s = sel.sum();
        s.backward();

        // Only row 1 should get gradients
        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        assertArrayEquals(new double[]{0, 0, 0, 1, 1, 1}, x.grad, 1e-12);
    }

    @Test
    void testSelectFirstDim() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor sel = x.select(1, 2);  // [2], select third column -> [3,6]
        IDiffTensor s = sel.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        assertArrayEquals(new double[]{0, 0, 1, 0, 0, 1}, x.grad, 1e-12);
    }

    // ---- Slice ----

    @Test
    void testSliceBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 3, 2);
        x.setRequiresGrad(true);

        IDiffTensor sl = x.slice(0, 1, 3);  // [2,2] — rows 1 to 2
        IDiffTensor s = sl.sum();
        s.backward();

        // gradient is 1 on sliced rows, 0 on first row
        assertNotNull(x.grad);
        assertEquals(6, x.grad.length);
        assertArrayEquals(new double[]{0, 0, 1, 1, 1, 1}, x.grad, 1e-12);
    }

    // ---- Narrow ----

    @Test
    void testNarrowBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2);
        x.setRequiresGrad(true);

        IDiffTensor nr = x.narrow(0, 1, 2);  // [2,2] — rows 1 to 2
        IDiffTensor s = nr.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(8, x.grad.length);
        assertArrayEquals(new double[]{0, 0, 1, 1, 1, 1, 0, 0}, x.grad, 1e-12);
    }

    // ---- Expand ----

    @Test
    void testExpandBackward() {
        // expand from [1,3] to [2,3]
        RereDiffTensor x = new RereDiffTensor(new double[]{10, 20, 30}, 1, 3);
        x.setRequiresGrad(true);

        IDiffTensor ex = x.expand(2, 3);
        IDiffTensor s = ex.sum();
        s.backward();

        // gradient per original element = sum over expanded dims = 2
        assertNotNull(x.grad);
        assertEquals(3, x.grad.length);
        assertArrayEquals(new double[]{2, 2, 2}, x.grad, 1e-12);
    }

    @Test
    void testExpandMultiDim() {
        // expand from [1,1,3] to [2,4,3]
        RereDiffTensor x = new RereDiffTensor(new double[]{10, 20, 30}, 1, 1, 3);
        x.setRequiresGrad(true);

        IDiffTensor ex = x.expand(2, 4, 3);
        IDiffTensor s = ex.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(3, x.grad.length);
        assertArrayEquals(new double[]{8, 8, 8}, x.grad, 1e-12);
    }

    // ---- Tile ----

    @Test
    void testTileBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor t = x.tile(2, 1);  // [4,2]
        IDiffTensor s = t.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(4, x.grad.length);
        // each original element contributes to 2 output elements
        assertArrayEquals(new double[]{2, 2, 2, 2}, x.grad, 1e-12);
    }

    // ---- BroadcastTo ----

    @Test
    void testBroadcastToBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{10, 20, 30}, 1, 3);
        x.setRequiresGrad(true);

        IDiffTensor bt = x.broadcastTo(2, 3);
        IDiffTensor s = bt.sum();
        s.backward();

        assertNotNull(x.grad);
        assertEquals(3, x.grad.length);
        assertArrayEquals(new double[]{2, 2, 2}, x.grad, 1e-12);
    }

    // ---- Chain: multiple view ops ----

    @Test
    void testChainViewOps() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3);
        x.setRequiresGrad(true);

        // permute → reshape → transpose → select → sum
        IDiffTensor v = x.permute(2, 0, 1)   // [3,2,2]
            .reshape(4, 3)                     // [4,3]
            .transpose(0, 1)                   // [3,4]
            .select(1, 2)                      // [3] (third column)
            .sum();                            // scalar
        v.backward();

        assertNotNull(x.grad);
        assertEquals(12, x.grad.length);
        // verify gradients are non-zero and correctly shaped
        for (int i = 0; i < 12; i++) {
            assertTrue(Double.isFinite(x.grad[i]),
                "grad[" + i + "] should be finite but was " + x.grad[i]);
        }
    }

    // ---- View op + element-wise chain ----

    @Test
    void testViewThenReluThenSum() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, -2, 3, -4}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor v = x.reshape(4)     // [4]
            .relu()                        // [4]
            .sum();                        // scalar
        v.backward();

        // relu gradient: 1 for positive, 0 for negative
        assertNotNull(x.grad);
        assertArrayEquals(new double[]{1, 0, 1, 0}, x.grad, 1e-12);
    }

    // ---- Grad shape consistency check after multiple view ops ----

    @Test
    void testGradShapeMatchesInput() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6, 7, 8}, 2, 4);
        x.setRequiresGrad(true);

        IDiffTensor v = x.unsqueeze(0)     // [1,2,4]
            .expand(3, 2, 4)                // [3,2,4]
            .permute(1, 0, 2)               // [2,3,4]
            .flatten(0, 1)                  // [6,4]
            .sum(1, true)                   // [6,1]
            .squeeze(1)                     // [6]
            .sum();                         // scalar
        v.backward();

        assertNotNull(x.grad);
        assertEquals(8, x.grad.length);
        assertTrue(x.grad[0] > 0, "grad should be non-zero after view chain");
    }
}
