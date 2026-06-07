package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.ODEDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
/**
 * Phase 3c: Neural ODE tensor version (ODEDiffTensor) correctness.
 */
public class ODEDiffTensorTest {
    void testForwardShapes() {
        // z0 = [1, 2], integrate from t=0 to t=1 with dt=0.1
        // dynamics = dz/dt = -z (simple exponential decay)
        RereDiffTensor z0 = new RereDiffTensor(new double[]{1, 2}, 2);
        IDiffTensor result = new ODEDiffTensor(
            z -> z.mul(-1.0),  // dz/dt = -z
            z0, 0, 1, 0.1);

        assertEquals(2, result.totalSize());
        double[] val = result.toDoubleArray();
        // Should have decayed toward 0
        assertTrue(val[0] > 0 && val[0] < 1, "Expected decay: " + val[0]);
        assertTrue(val[1] > 0 && val[1] < 2, "Expected decay: " + val[1]);
    }

    @Test
    void testGradientFlows() {
        // loss(z(1)) = sum(z(1)), dz/dt = z (exponential growth)
        // gradient of loss w.r.t. z0 should be exp(1) for each element
        RereDiffTensor z0 = new RereDiffTensor(new double[]{1.0}, 1);
        z0.setRequiresGrad(true);

        IDiffTensor z1 = new ODEDiffTensor(
            z -> z.mul(1.0),  // dz/dt = z
            z0, 0, 1.0, 0.2);

        IDiffTensor loss = z1.sum();
        loss.backward();

        // d(loss)/dz0 = exp(1) ≈ 2.718...
        assertNotNull(z0.gradData());
        assertEquals(Math.E, z0.gradData()[0], 0.3); // RK4 ≈ exact for exponential
    }

    @Test
    void test2DTensor() {
        // 2D initial state: shape [2, 2], dz/dt = -2*z
        RereDiffTensor z0 = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        z0.setRequiresGrad(true);

        IDiffTensor result = new ODEDiffTensor(
            z -> z.mul(-2.0),
            z0, 0, 0.5, 0.1);

        // Forward check
        double[] val = result.toDoubleArray();
        assertEquals(4, val.length);
        // Values should have decayed
        for (double v : val) {
            assertTrue(v > 0 && v <= 4);
        }

        // Gradient check
        result.sum().backward();
        assertNotNull(z0.gradData());
        assertEquals(4, z0.gradData().length);
        // All gradients should be positive (all inputs contribute positively)
        for (double g : z0.gradData()) {
            assertTrue(g > 0);
        }
    }

    @Test
    void testNonlinearDynamics() {
        // dz/dt = -z^2, from z0=2, t=0 to t=1
        // Analytical: z(t) = 1/(t + 1/z0) = 1/(t + 0.5)
        // z(1) = 1/(1 + 0.5) = 2/3 ≈ 0.667
        RereDiffTensor z0 = new RereDiffTensor(new double[]{2.0}, 1);
        z0.setRequiresGrad(true);

        IDiffTensor result = new ODEDiffTensor(
            z -> z.square().mul(-1.0),
            z0, 0, 1.0, 0.05);

        double[] val = result.toDoubleArray();
        assertEquals(0.667, val[0], 0.05);

        // Gradient: d(loss)/dz0 where loss = z(1)
        // From sensitivity: z(1) = 1/(1 + 0.5) where z0=2
        // dz(1)/dz0 = 1/(1+1/z0)^2 * 1/z0^2 = 1/(z0+1)^2
        // For z0=2: dz(1)/dz0 = 1/9 ≈ 0.111
        result.sum().backward();
        assertNotNull(z0.gradData());
        assertEquals(1.0 / 9, z0.gradData()[0], 0.02);
    }

    @Test
    void testBatchedDynamics() {
        // dynamics: dz/dt = z (exponential growth), batched 2 samples
        RereDiffTensor z0 = new RereDiffTensor(new double[]{1, 0.5, 2, 3}, 2, 2);
        z0.setRequiresGrad(true);

        IDiffTensor result = new ODEDiffTensor(
            z -> z.mul(1.0),
            z0, 0, 0.5, 0.1);

        double[] val = result.toDoubleArray();
        assertEquals(4, val.length);
        // All values should have grown
        assertTrue(val[0] > 1);
        assertTrue(val[1] > 0.5);
        assertTrue(val[2] > 2);
        assertTrue(val[3] > 3);

        // Gradient flows back
        result.sum().backward();
        assertNotNull(z0.gradData());
        assertEquals(4, z0.gradData().length);
        for (double g : z0.gradData()) {
            assertTrue(g > 1);
        }
    }
}
