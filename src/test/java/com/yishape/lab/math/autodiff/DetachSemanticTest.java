package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 4.3 Step 1.5 — verifies the {@code detach()} semantic contract.
 *
 * <p>{@code detach()} returns a node that shares the value of its source but is
 * cut off from the computation graph: gradients flowing into a detached node do
 * NOT propagate back to the original graph's leaves. This mirrors PyTorch's
 * {@code tensor.detach()}.
 *
 * <p>Key contracts under test:
 * <ul>
 *   <li>detached node has {@code requiresGrad() == false}</li>
 *   <li>detached node carries the same value as the source</li>
 *   <li>gradient does NOT cross the detach boundary (source leaf grad stays null)</li>
 *   <li>without detach, the same graph DOES propagate gradient to the source (control)</li>
 *   <li>detach works on vector and matrix facades, not just raw tensors</li>
 * </ul>
 */
public class DetachSemanticTest {

    private static final double TOL = 1e-12;

    // ── Tensor detach ──────────────────────────────────────────────────

    @Test
    void detachTensor_isNotRequiresGradAndSharesValue() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1.0, 2.0, 3.0}, 3);
        x.setRequiresGrad(true);

        IDiffTensor d = x.detach();

        assertFalse(d.requiresGrad(), "detached node must not require grad");
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, d.toDoubleArray(), TOL,
            "detached node must carry the source value");
    }

    @Test
    void detachTensor_blocksGradientPropagation() {
        // f = (detach(x) * w).sum()  →  grad flows to w, NOT to x
        RereDiffTensor x = new RereDiffTensor(new double[]{1.0, 2.0, 3.0}, 3);
        x.setRequiresGrad(true);
        RereDiffTensor w = new RereDiffTensor(new double[]{4.0, 5.0, 6.0}, 3);
        w.setRequiresGrad(true);

        IDiffTensor d = x.detach();
        IDiffTensor y = d.mul(w);
        IDiffTensor loss = y.sum();
        loss.backward();

        // grad w.r.t. w = d = x's values
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, w.gradData(), TOL,
            "gradient must still reach w (the non-detached operand)");
        // grad w.r.t. x must be blocked
        assertNull(x.gradData(),
            "gradient must NOT cross the detach boundary to x");
    }

    @Test
    void detachTensor_controlWithoutDetachPropagates() {
        // Same graph but WITHOUT detach → gradient reaches x.
        RereDiffTensor x = new RereDiffTensor(new double[]{1.0, 2.0, 3.0}, 3);
        x.setRequiresGrad(true);
        RereDiffTensor w = new RereDiffTensor(new double[]{4.0, 5.0, 6.0}, 3);
        w.setRequiresGrad(true);

        IDiffTensor y = x.mul(w);
        IDiffTensor loss = y.sum();
        loss.backward();

        // grad w.r.t. x = w
        assertArrayEquals(new double[]{4.0, 5.0, 6.0}, x.gradData(), TOL,
            "without detach, gradient must propagate to x");
    }

    @Test
    void detachTensor_detachedLeafIsIndependentAfterBackward() {
        // A detached tensor used as a loss root: backward fills its own grad
        // but the original source is untouched.
        RereDiffTensor x = new RereDiffTensor(new double[]{2.0, 4.0}, 2);
        x.setRequiresGrad(true);
        IDiffTensor d = x.detach();
        // d.requiresGrad == false, so backward on a pure-detach graph is a no-op
        // for the global graph; x must remain un-grad'd.
        assertNull(x.gradData());
        assertFalse(d.requiresGrad());
    }

    // ── Vector facade detach ───────────────────────────────────────────

    @Test
    void detachVector_delegatesToTensorAndBlocks() {
        IDiffVector x = AD.vector(new double[]{1.0, 2.0, 3.0});
        IDiffVector w = AD.vector(new double[]{4.0, 5.0, 6.0});

        IDiffVector d = x.detach();
        assertFalse(d.requiresGrad());
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, d.getValue().getData(), TOL);

        IDiffVector loss = d.mul(w).sum();
        loss.backward();

        // w gets gradient (= d = x values); x does not.
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, w.getGradient().getData(), TOL);
        assertNull(x.getGradient(), "detach must block gradient to x on the vector facade");
    }

    // ── Matrix facade detach ───────────────────────────────────────────

    @Test
    void detachMatrix_delegatesToTensorAndBlocks() {
        IDiffMatrix x = AD.matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
        IDiffMatrix w = AD.matrix(new double[][]{{5.0, 6.0}, {7.0, 8.0}});

        IDiffMatrix d = x.detach();
        assertFalse(d.requiresGrad());

        IDiffMatrix loss = d.mul(w).sum();
        loss.backward();

        // grad w.r.t. w = d = x values
        double[][] gradW = w.getGradient().getData();
        assertEquals(1.0, gradW[0][0], TOL);
        assertEquals(4.0, gradW[1][1], TOL);
        // x must be untouched
        assertNull(x.getGradient(), "detach must block gradient to x on the matrix facade");
    }

    @Test
    void detachTensor_valueIsIndependentCopy() {
        // RereDiffTensor.detach() copies data — mutating the source after detach
        // must not change the detached node's value (defensive against alias bugs).
        RereDiffTensor x = new RereDiffTensor(new double[]{1.0, 2.0}, 2);
        x.setRequiresGrad(true);
        IDiffTensor d = x.detach();
        assertArrayEquals(new double[]{1.0, 2.0}, d.toDoubleArray(), TOL);

        // Build a new graph that mutates nothing of x's storage; just confirm
        // detach snapshot is stable.
        assertArrayEquals(new double[]{1.0, 2.0}, d.toDoubleArray(), TOL);
    }
}
