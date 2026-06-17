package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 4.3 Step 3.7 — regression guard for {@code symbolicBackwardFn} coverage.
 *
 * <p>{@link AD#grad(IDiffTensor, IDiffTensor...)} (tape-of-tape second-order AD)
 * traverses each node and applies its {@code symbolicBackwardFn}; nodes with a
 * null {@code symbolicBackwardFn} are <b>silently skipped</b> (AD.java:364). A
 * null symbolic backward therefore produces a <em>silent zero</em> in any
 * second-order gradient downstream of that op — the worst possible failure mode
 * (no exception, just a wrong Hessian/HVP).
 *
 * <p>This test pins the coverage contract in <b>both directions</b>:
 * <ul>
 *   <li>COVERED ops <b>must</b> carry a non-null {@code symbolicBackwardFn}.</li>
 *   <li>KNOWN_GAP ops <b>must</b> still carry a null one (until someone
 *       implements + verifies them, at which point they move to COVERED).</li>
 * </ul>
 * Any op that changes status forces a test edit here, making the change visible
 * in review.
 *
 * <p>For the ops whose symbolic backward has a clean closed form, a numerical
 * HVP cross-check (vs central-difference) proves the formula is correct, not
 * merely present.
 */
public class SymbolicBackwardCoverageTest {

    private static final double NUM_TOL = 1e-5;

    // ── Helpers ────────────────────────────────────────────────────────

    private static List<RereDiffTensor> topo(RereDiffTensor root) {
        List<RereDiffTensor> order = new ArrayList<>();
        root.buildTopo(order, new HashSet<>());
        return order;
    }

    /** opTags of intermediate (non-leaf) nodes whose symbolicBackwardFn is null. */
    private static Set<String> nullSymbolicOpTags(RereDiffTensor root) {
        Set<String> gaps = new HashSet<>();
        for (RereDiffTensor v : topo(root)) {
            if (!v.isLeaf() && v.symbolicBackwardFn() == null) {
                gaps.add(v.opTag());
            }
        }
        return gaps;
    }

    // ── COVERED: core differentiable ops must have symbolicBackwardFn ──

    @Test
    void unaryElementwise_areCovered() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0.3, 0.7, 1.1}, 3);
        x.setRequiresGrad(true);
        for (IDiffTensor y : new IDiffTensor[]{
            x.neg(), x.abs(), x.square(), x.sqrt(), x.exp(), x.log(),
            x.sigmoid(), x.tanh(), x.relu(), x.gelu(), x.sin(), x.cos(),
            x.pow(3), x.leakyRelu(0.1), x.elu(1.0), x.selu(), x.silu(),
            x.mish(), x.softplus(1.0), x.hardtanh(-1.0, 1.0)
        }) {
            assertNotNull(((RereDiffTensor) y).symbolicBackwardFn(),
                "unary op must carry symbolicBackwardFn: " + ((RereDiffTensor) y).opTag());
        }
    }

    @Test
    void binaryElementwise_areCovered() {
        RereDiffTensor a = new RereDiffTensor(new double[]{0.5, 1.5, 2.0}, 3);
        RereDiffTensor b = new RereDiffTensor(new double[]{0.2, 0.4, 0.6}, 3);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        for (IDiffTensor y : new IDiffTensor[]{ a.add(b), a.sub(b), a.mul(b), a.div(b) }) {
            assertNotNull(((RereDiffTensor) y).symbolicBackwardFn(),
                "binary op must carry symbolicBackwardFn: " + ((RereDiffTensor) y).opTag());
        }
    }

    @Test
    void reductions_areCovered() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0.3, 0.7, 1.1}, 3);
        x.setRequiresGrad(true);
        assertNotNull(((RereDiffTensor) x.sum()).symbolicBackwardFn(), "sum must be covered");
        assertNotNull(((RereDiffTensor) x.mean(0, false)).symbolicBackwardFn(), "mean(dim) must be covered");
        assertNotNull(((RereDiffTensor) x.sum(0, false)).symbolicBackwardFn(), "sum(dim) must be covered");
    }

    @Test
    void matmul_isCovered() {
        RereDiffTensor a = new RereDiffTensor(new double[]{0.5, 0.1, 0.2, 0.4}, 2, 2);
        RereDiffTensor b = new RereDiffTensor(new double[]{0.3, 0.7, 0.2, 0.8}, 2, 2);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        IDiffTensor y = a.mmul(b);
        assertNotNull(((RereDiffTensor) y).symbolicBackwardFn(), "mmul must be covered");
    }

    @Test
    void viewOps_permuteUnsqueezeFlattenReshapeSqueeze_areCovered() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3);
        x.setRequiresGrad(true);

        assertNotNull(((RereDiffTensor) x.permute(1, 0)).symbolicBackwardFn(), "permute");
        assertNotNull(((RereDiffTensor) x.reshape(3, 2)).symbolicBackwardFn(), "reshape");
        assertNotNull(((RereDiffTensor) x.flatten(0, 1)).symbolicBackwardFn(), "flatten");
        assertNotNull(((RereDiffTensor) x.unsqueeze(0)).symbolicBackwardFn(), "unsqueeze");

        // squeeze: parent [2,3] has no size-1 dim, so build [1,2,3] first.
        RereDiffTensor x1 = new RereDiffTensor(new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 1, 2, 3);
        x1.setRequiresGrad(true);
        IDiffTensor sq = x1.squeeze(0);
        assertNotNull(((RereDiffTensor) sq).symbolicBackwardFn(),
            "squeeze must carry symbolicBackwardFn (fixed 2026-06-17)");
    }

    @Test
    void softmax_logSoftmax_areCovered() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0.3, 0.7, 1.1, 0.2}, 2, 2);
        x.setRequiresGrad(true);
        assertNotNull(((RereDiffTensor) x.softmax(1)).symbolicBackwardFn(), "softmax");
        assertNotNull(((RereDiffTensor) x.logSoftmax(1)).symbolicBackwardFn(), "logSoftmax");
    }

    // ── KNOWN GAPS: ops that legitimately lack symbolicBackwardFn ──────
    //
    // These ops' first-order backward uses an index-scatter that has no
    // differentiable primitive yet, so tape-of-tape second-order AD through
    // them yields a silent zero. Each is documented here so the gap is
    // explicit; when one is implemented + numerically verified, move it to
    // the COVERED tests above and delete its entry here.

    private static final Set<String> KNOWN_GAPS = Set.of(
        "expand",      // backward = sum-reduce over broadcast dims; needs differentiable reduce-by-mask
        "select",      // backward = index scatter; needs differentiable scatter primitive
        "slice",       // backward = index scatter; needs differentiable scatter primitive
        "contiguous",  // identity copy; second-order trivially identity but not wired
        "tile"         // backward = sum-reduce over repeat groups; needs differentiable reduce
    );

    @Test
    void knownGaps_areStillNull() {
        // If this fails, an op gained a symbolicBackwardFn — move it to the
        // COVERED tests and drop it from KNOWN_GAPS.
        RereDiffTensor x = new RereDiffTensor(new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, 2, 3);
        x.setRequiresGrad(true);
        // expand broadcasts size-1 dims, so use a [1,3] parent → [4,3].
        RereDiffTensor x1 = new RereDiffTensor(new double[]{0.1, 0.2, 0.3}, 1, 3);
        x1.setRequiresGrad(true);

        Set<String> seen = new HashSet<>();
        seen.addAll(nullSymbolicOpTags((RereDiffTensor) x1.expand(4, 3)));
        seen.addAll(nullSymbolicOpTags((RereDiffTensor) x.slice(0, 0, 1)));
        seen.addAll(nullSymbolicOpTags((RereDiffTensor) x.select(0, 0)));
        seen.addAll(nullSymbolicOpTags((RereDiffTensor) x.tile(2)));
        // contiguous only creates a node when storage is non-contiguous; skip if absent.

        // Every KNOWN_GAP that actually produced a node must appear in `seen`.
        for (String gap : KNOWN_GAPS) {
            if ("contiguous".equals(gap)) continue; // may be a no-op
            assertTrue(seen.contains(gap),
                "KNOWN_GAP '" + gap + "' no longer has a null symbolicBackwardFn — " +
                    "move it to the COVERED tests");
        }
    }

    // ── Correctness: the newly-fixed squeeze symbolic backward ─────────

    @Test
    void squeeze_symbolicBackward_matchesNumericalHvp() {
        // f = sum(squeeze(x)^2), x shape [1,n]. squeeze→[n], f=Σ y_i^2.
        // grad w.r.t x = 2y (broadcast back through size-1 dim) = 2x.
        // H = 2I, H@v = 2v. Exercises the squeeze symbolic backward end-to-end.
        double[] xData = {0.3, -0.5, 0.8};
        double[] v = {1.0, -2.0, 0.5};

        double[] analytic = tensorHvpSqueeze(xData, v);
        double[] numerical = numericalHvpSqueeze(xData, v);
        assertArrayEquals(numerical, analytic, NUM_TOL,
            () -> "squeeze HVP mismatch:\n  numerical=" + java.util.Arrays.toString(numerical)
                + "\n  analytic  =" + java.util.Arrays.toString(analytic));
    }

    /**
     * H@v for f = sum(squeeze([1,n] x)^2), computed by replicating
     * {@link MixedMode#hvp}'s mechanism at the tensor level (the squeeze op
     * only exists on the tensor API, not the vector facade):
     *   grad = AD.grad(f, x)          // tape-of-tape, uses squeeze symbolicBackwardFn
     *   s   = (grad · v).sum()        // scalar
     *   s.backward()                  // first-order reverse through the tape-of-tape
     *   x.grad                         // = H @ v
     */
    private static double[] tensorHvpSqueeze(double[] xData, double[] v) {
        int n = xData.length;
        RereDiffTensor x = new RereDiffTensor(xData, 1, n);
        x.setRequiresGrad(true);
        IDiffTensor sq = x.squeeze(0);
        IDiffTensor f = sq.pow(2).sum();
        IDiffTensor[] garr = AD.grad(f, x);
        RereDiffTensor g = (RereDiffTensor) garr[0];
        RereDiffTensor vConst = new RereDiffTensor(v, 1, n);   // constant leaf (requiresGrad=false)
        IDiffTensor s = g.mul(vConst).sum();
        x.setGradData(new double[n]);   // zero before accumulating H@v
        s.backward();
        return x.gradData();
    }

    private static double[] numericalHvpSqueeze(double[] x, double[] v) {
        double eps = 1e-6;
        int n = x.length;
        double[] xp = new double[n], xm = new double[n];
        for (int i = 0; i < n; i++) { xp[i] = x[i] + eps * v[i]; xm[i] = x[i] - eps * v[i]; }
        double[] gp = gradSqueeze(xp);
        double[] gm = gradSqueeze(xm);
        double[] hvp = new double[n];
        for (int i = 0; i < n; i++) hvp[i] = (gp[i] - gm[i]) / (2 * eps);
        return hvp;
    }

    /** Tape-of-tape gradient of f = sum(squeeze(x)^2) w.r.t x (shape [1,n]). */
    private static double[] gradSqueeze(double[] x) {
        RereDiffTensor t = new RereDiffTensor(x, 1, x.length);
        t.setRequiresGrad(true);
        IDiffTensor sq = t.squeeze(0);
        IDiffTensor loss = sq.pow(2).sum();
        IDiffTensor[] grads = AD.grad(loss, t);
        return grads[0].toDoubleArray();
    }
}
