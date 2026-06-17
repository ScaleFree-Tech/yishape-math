package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 4.3 Step 2.8 — verifies the {@code backward(boolean retainGraph)} contract.
 *
 * <p>By default {@code backward()} releases the graph edges of intermediate nodes
 * (inputs / backwardFn / symbolicBackwardFn) so the JVM can reclaim them. With
 * {@code retainGraph=true} those edges are preserved, enabling re-running
 * backward and second-order AD (symbolicBackwardFn must survive).
 *
 * <p><b>Test-graph note.</b> The root node is never cleared (it must stay
 * backward-able), and a fused single-hop graph (e.g. {@code pow().sum()} → one
 * {@code powSum} node) lets the root's backwardFn reach the leaf directly. To
 * observe edge clearing we therefore use a <em>multi-hop</em> graph
 * {@code x.exp().sigmoid().sum()} where the {@code exp} node sits between the
 * root and the leaf — clearing it severs the path regardless of whether
 * {@code sigmoid().sum()} fuses.
 */
public class RetainGraphTest {

    private static final double TOL = 1e-12;

    private static List<RereDiffTensor> topo(RereDiffTensor root) {
        List<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.buildTopo(order, visited);
        return order;
    }

    /** Non-root, non-leaf intermediate nodes (the ones edge-clearing targets). */
    private static List<RereDiffTensor> intermediates(RereDiffTensor root) {
        List<RereDiffTensor> out = new ArrayList<>();
        for (RereDiffTensor v : topo(root)) {
            if (!v.isLeaf() && v != root) out.add(v);
        }
        return out;
    }

    @Test
    void retainGraphTrue_allowsSecondBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1.0, 2.0, 3.0}, 3);
        x.setRequiresGrad(true);
        IDiffTensor f = x.pow(2).sum();   // df/dx = 2x

        f.backward(true);
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, x.gradData(), TOL);

        f.backward(true);
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, x.gradData(), TOL,
            "retainGraph=true must allow a second backward to reproduce gradients");
    }

    @Test
    void retainGraphTrue_preservesIntermediateEdges() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0.5, 1.0, 1.5}, 3);
        x.setRequiresGrad(true);
        RereDiffTensor f = (RereDiffTensor) x.exp().sigmoid().sum();

        f.backward(true);

        List<RereDiffTensor> mids = intermediates(f);
        assertFalse(mids.isEmpty(), "test graph must have at least one intermediate");
        for (RereDiffTensor v : mids) {
            assertNotNull(v.inputs(), "retainGraph=true must preserve inputs: " + v.opTag());
            assertNotNull(v.symbolicBackwardFn(),
                "retainGraph=true must preserve symbolicBackwardFn: " + v.opTag());
        }
    }

    @Test
    void retainGraphFalse_clearsIntermediateEdges() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0.5, 1.0, 1.5}, 3);
        x.setRequiresGrad(true);
        RereDiffTensor f = (RereDiffTensor) x.exp().sigmoid().sum();

        f.backward(false);

        List<RereDiffTensor> mids = intermediates(f);
        assertFalse(mids.isEmpty(), "test graph must have at least one intermediate");
        for (RereDiffTensor v : mids) {
            assertNull(v.inputs(), "retainGraph=false must null intermediate inputs: " + v.opTag());
            assertNull(v.symbolicBackwardFn(),
                "retainGraph=false must null symbolicBackwardFn: " + v.opTag());
        }
    }

    @Test
    void retainGraphFalse_seversPathToLeafOnSecondBackward() {
        // Multi-hop: root → (fused sigmoid/sum or sum) → exp → x.
        // After backward(false) the exp node's edges are cleared, so a second
        // backward cannot propagate grad through exp to x.
        RereDiffTensor x = new RereDiffTensor(new double[]{0.5, 1.0, 1.5}, 3);
        x.setRequiresGrad(true);
        RereDiffTensor f = (RereDiffTensor) x.exp().sigmoid().sum();

        f.backward(false);
        double[] g1 = x.gradData().clone();
        // sanity: first backward reached x
        assertTrue(g1[0] > 0.0 && g1[1] > 0.0 && g1[2] > 0.0);

        x.setGradData(new double[]{0.0, 0.0, 0.0});  // clear accumulation
        f.backward(false);
        double[] g2 = x.gradData();
        // grad must NOT have been recomputed (path through exp severed)
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, g2, TOL,
            "after retainGraph=false, second backward must not reach the leaf (intermediate cleared)");
    }

    @Test
    void defaultBackward_clearsLikeRetainFalse() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0.5, 1.0, 1.5}, 3);
        x.setRequiresGrad(true);
        RereDiffTensor f = (RereDiffTensor) x.exp().sigmoid().sum();

        f.backward();  // no-arg → retainGraph=false

        for (RereDiffTensor v : intermediates(f)) {
            assertNull(v.symbolicBackwardFn(),
                "default backward() must clear graph (== retainGraph=false): " + v.opTag());
        }
    }

    @Test
    void retainGraphTrue_preservesSymbolicBackwardFnForSecondOrder() {
        // The whole point for higher-order AD: symbolicBackwardFn survives the
        // first backward so AD.grad() can still traverse the tape-of-tape.
        RereDiffTensor x = new RereDiffTensor(new double[]{0.0, 1.0, 2.0}, 3);
        x.setRequiresGrad(true);
        RereDiffTensor f = (RereDiffTensor) x.exp().sum();

        f.backward(true);

        boolean symbolicSurvives = false;
        for (RereDiffTensor v : topo(f)) {
            if (!v.isLeaf() && v.symbolicBackwardFn() != null) {
                symbolicSurvives = true;
                break;
            }
        }
        assertTrue(symbolicSurvives,
            "retainGraph=true must preserve symbolicBackwardFn for second-order AD");
    }

    @Test
    void retainGraphTrue_worksOnVectorFacade() {
        IDiffVector x = AD.vector(new double[]{1.0, 2.0, 3.0});
        IDiffVector f = x.pow(2).sum();

        f.backward(true);
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, x.getGradient().getData(), TOL);

        f.backward(true);
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, x.getGradient().getData(), TOL,
            "retainGraph must work through the vector facade");
    }
}
