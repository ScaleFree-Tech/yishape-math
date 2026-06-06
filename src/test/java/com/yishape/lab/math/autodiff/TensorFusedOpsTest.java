package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.TensorFusedOps;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 2c: TensorFusedOps correctness.
 */
public class TensorFusedOpsTest {

    @Test
    void testFusedExp() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0, 1, -1, 2}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor fused = new TensorFusedOps(x).exp().done();
        IDiffTensor s = fused.sum();
        s.backward();

        // Reference: separate ops
        RereDiffTensor y = new RereDiffTensor(new double[]{0, 1, -1, 2}, 2, 2);
        y.setRequiresGrad(true);
        y.exp().sum().backward();

        assertArrayEquals(y.grad, x.grad, 1e-12);
    }

    @Test
    void testFusedExpThenRelu() {
        RereDiffTensor x = new RereDiffTensor(new double[]{-1, 0, 1, 2}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor fused = new TensorFusedOps(x).exp().relu().done();
        IDiffTensor s = fused.sum();
        s.backward();

        RereDiffTensor y = new RereDiffTensor(new double[]{-1, 0, 1, 2}, 2, 2);
        y.setRequiresGrad(true);
        y.exp().relu().sum().backward();

        assertArrayEquals(y.grad, x.grad, 1e-12);
    }

    @Test
    void testFusedThroughADFacade() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor fused = AD.fuseTensor(x).square().add(1).done();
        IDiffTensor s = fused.sum();
        s.backward();

        RereDiffTensor y = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        y.setRequiresGrad(true);
        y.square().add(1).sum().backward();

        assertArrayEquals(y.grad, x.grad, 1e-12);
    }

    @Test
    void testFusedIdentity() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        x.setRequiresGrad(true);

        IDiffTensor fused = new TensorFusedOps(x).done();
        IDiffTensor s = fused.sum();
        s.backward();

        assertArrayEquals(new double[]{1, 1, 1}, x.grad, 1e-12);
    }

    @Test
    void testFusedSigmoid() {
        RereDiffTensor x = new RereDiffTensor(new double[]{-2, -1, 0, 1, 2}, 5);
        x.setRequiresGrad(true);

        IDiffTensor fused = new TensorFusedOps(x).sigmoid().done();
        fused.sum().backward();

        RereDiffTensor y = new RereDiffTensor(new double[]{-2, -1, 0, 1, 2}, 5);
        y.setRequiresGrad(true);
        y.sigmoid().sum().backward();

        assertArrayEquals(y.grad, x.grad, 1e-10);
    }

    @Test
    void testFusedChainMulAdd() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        x.setRequiresGrad(true);

        IDiffTensor fused = new TensorFusedOps(x).mul(2).add(1).done();
        fused.sum().backward();

        RereDiffTensor y = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        y.setRequiresGrad(true);
        y.mul(2).add(1).sum().backward();

        assertArrayEquals(y.grad, x.grad, 1e-12);
    }
}
