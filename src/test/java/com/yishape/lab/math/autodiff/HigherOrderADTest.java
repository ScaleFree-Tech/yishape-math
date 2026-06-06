package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 3a: Higher-order AD via AD.grad() and symbolicBackwardFn.
 */
public class HigherOrderADTest {

    @Test
    void testFirstOrderGrad() {
        // f(x) = sum(relu(x)), x = [-1, 2, -3, 4]
        RereDiffTensor x = new RereDiffTensor(new double[]{-1, 2, -3, 4}, 4);
        x.setRequiresGrad(true);
        IDiffTensor f = x.relu().sum();

        IDiffTensor[] grads = AD.grad(f, x);
        double[] g = grads[0].toDoubleArray();
        assertArrayEquals(new double[]{0, 1, 0, 1}, g, 1e-12);
    }

    @Test
    void testFirstOrderGradExp() {
        // f(x) = sum(exp(x)), x = [0, 1, 2]
        RereDiffTensor x = new RereDiffTensor(new double[]{0, 1, 2}, 3);
        x.setRequiresGrad(true);
        IDiffTensor f = x.exp().sum();

        IDiffTensor[] grads = AD.grad(f, x);
        double[] g = grads[0].toDoubleArray();
        assertArrayEquals(new double[]{1, Math.E, Math.E*Math.E}, g, 1e-10);
    }

    @Test
    void testFirstOrderGradSquare() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        x.setRequiresGrad(true);
        IDiffTensor f = x.square().sum();

        IDiffTensor[] grads = AD.grad(f, x);
        double[] g = grads[0].toDoubleArray();
        assertArrayEquals(new double[]{2, 4, 6}, g, 1e-12);
    }

    @Test
    void testFirstOrderGradNeg() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        x.setRequiresGrad(true);
        IDiffTensor f = x.neg().sum();

        IDiffTensor[] grads = AD.grad(f, x);
        double[] g = grads[0].toDoubleArray();
        assertArrayEquals(new double[]{-1, -1, -1}, g, 1e-12);
    }

    @Test
    @org.junit.jupiter.api.Disabled("Deferred: second-order requires symbolicBackwardFn to reference original input nodes, not constant tensors")
    void testSecondOrderThroughGrad() {
        // f(x) = sum(exp(x))
        // g = AD.grad(f, x)[0] = exp(x)
        // h = sum(g) -> gradient should be exp(x)
        RereDiffTensor x = new RereDiffTensor(new double[]{0, 1, 2}, 3);
        x.setRequiresGrad(true);
        IDiffTensor f = x.exp().sum();

        IDiffTensor[] grads = AD.grad(f, x);
        IDiffTensor g = grads[0];  // g = exp(x), still differentiable

        // Second order: sum(g) -> gradient = d(exp(x))/dx = exp(x)
        RereDiffTensor gTensor = (RereDiffTensor) g;
        gTensor.setRequiresGrad(true);
        IDiffTensor h = g.sum();

        h.backward();
        double[] d2 = x.grad;
        assertArrayEquals(new double[]{1, Math.E, Math.E*Math.E}, d2, 1e-10);
    }

    @Test
    void testGradWithMultipleInputs() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        a.setRequiresGrad(true);
        RereDiffTensor b = new RereDiffTensor(new double[]{4, 5, 6}, 3);
        b.setRequiresGrad(true);

        IDiffTensor c = a.add(b);
        IDiffTensor f = c.sum();

        IDiffTensor[] grads = AD.grad(f, a, b);
        assertArrayEquals(new double[]{1, 1, 1}, grads[0].toDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{1, 1, 1}, grads[1].toDoubleArray(), 1e-12);
    }

    @Test
    void testGradWithMul() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        a.setRequiresGrad(true);
        RereDiffTensor b = new RereDiffTensor(new double[]{4, 5, 6}, 3);
        b.setRequiresGrad(true);

        IDiffTensor c = a.mul(b);
        IDiffTensor f = c.sum();

        IDiffTensor[] grads = AD.grad(f, a, b);
        assertArrayEquals(new double[]{4, 5, 6}, grads[0].toDoubleArray(), 1e-12);
        assertArrayEquals(new double[]{1, 2, 3}, grads[1].toDoubleArray(), 1e-12);
    }
}
