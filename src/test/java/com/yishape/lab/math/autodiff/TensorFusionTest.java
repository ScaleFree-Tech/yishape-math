package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 2b: Pattern fusion correctness — verifies that fused operations
 * (e.g., square().sum() → squareSum) produce the same gradients as
 * the non-fused equivalents.
 */
public class TensorFusionTest {

    @Test
    void testSquareSumFusion() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor fused = x.square().sum();
        fused.backward();
        double[] fusedGrad = x.gradData().clone();

        // Non-fused reference
        RereDiffTensor y = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        y.setRequiresGrad(true);
        IDiffTensor ref = y.square().sum();
        ref.backward();
        double[] refGrad = y.gradData();

        assertArrayEquals(refGrad, fusedGrad, 1e-12);
    }

    @Test
    void testReluSumFusion() {
        RereDiffTensor x = new RereDiffTensor(new double[]{-1, 2, -3, 4, 0, 6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor fused = x.relu().sum();
        fused.backward();
        double[] fusedGrad = x.gradData().clone();

        RereDiffTensor y = new RereDiffTensor(new double[]{-1, 2, -3, 4, 0, 6}, 2, 3);
        y.setRequiresGrad(true);
        y.relu().sum().backward();
        double[] refGrad = y.gradData();

        assertArrayEquals(refGrad, fusedGrad, 1e-12);
    }

    @Test
    void testExpSumFusion() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0, 1, -1, 2}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor fused = x.exp().sum();
        fused.backward();
        double[] fusedGrad = x.gradData().clone();

        RereDiffTensor y = new RereDiffTensor(new double[]{0, 1, -1, 2}, 2, 2);
        y.setRequiresGrad(true);
        y.exp().sum().backward();
        double[] refGrad = y.gradData();

        assertArrayEquals(refGrad, fusedGrad, 1e-12);
    }

    @Test
    void testSigmoidSumFusion() {
        RereDiffTensor x = new RereDiffTensor(new double[]{-1, 0, 1, 2}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor fused = x.sigmoid().sum();
        fused.backward();
        double[] fusedGrad = x.gradData().clone();

        RereDiffTensor y = new RereDiffTensor(new double[]{-1, 0, 1, 2}, 2, 2);
        y.setRequiresGrad(true);
        y.sigmoid().sum().backward();
        double[] refGrad = y.gradData();

        assertArrayEquals(refGrad, fusedGrad, 1e-12);
    }

    @Test
    void testPowSumFusion() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor fused = x.pow(3).sum();
        fused.backward();
        double[] fusedGrad = x.gradData().clone();

        RereDiffTensor y = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        y.setRequiresGrad(true);
        y.pow(3).sum().backward();
        double[] refGrad = y.gradData();

        assertArrayEquals(refGrad, fusedGrad, 1e-12);
    }

    @Test
    void testTanhSumFusion() {
        RereDiffTensor x = new RereDiffTensor(new double[]{-1, 0, 1, 2}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor fused = x.tanh().sum();
        fused.backward();
        double[] fusedGrad = x.gradData().clone();

        RereDiffTensor y = new RereDiffTensor(new double[]{-1, 0, 1, 2}, 2, 2);
        y.setRequiresGrad(true);
        y.tanh().sum().backward();
        double[] refGrad = y.gradData();

        assertArrayEquals(refGrad, fusedGrad, 1e-12);
    }

    @Test
    void testFusionChainWithView() {
        // Verify fusion works across view ops
        RereDiffTensor x = new RereDiffTensor(new double[]{1, -2, 3, -4, 5, -6}, 2, 3);
        x.setRequiresGrad(true);

        // reshape → relu → sum should be reluSum at the end
        IDiffTensor v = x.reshape(3, 2).relu().sum();
        v.backward();

        double[] g = x.gradData();
        assertNotNull(g);
        assertEquals(6, g.length);
        // After relu gradient: positive → 1, negative → 0
        assertArrayEquals(new double[]{1, 0, 1, 0, 1, 0}, g, 1e-12);
    }

    @Test
    void testFusionExportShape() {
        // Fused node should have exportShape set for GPU/HPC
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor s = x.square().sum();
        assertTrue(s instanceof RereDiffTensor);
        RereDiffTensor rs = (RereDiffTensor) s;
        assertArrayEquals(new int[]{2, 2}, rs.exportShape(), "exportShape should match input shape");
        assertEquals("squareSum", rs.opTag());
    }

    @Test
    void testNonFusedSumWhenNotPattern() {
        // add(1) doesn't match pattern → should produce regular sum
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        x.setRequiresGrad(true);

        IDiffTensor s = x.add(1).sum();
        s.backward();

        assertNotNull(x.gradData());
        assertArrayEquals(new double[]{1, 1, 1}, x.gradData(), 1e-12);
    }
}
