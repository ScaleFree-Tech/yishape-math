package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.vjp.VjpResult;
import com.yishape.lab.math.autodiff.MixedMode;
import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.vjp.BatchVjpResult;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;

/**
 * Tests for VJP (Vector-Jacobian Product) and vmap infrastructure.
 */
public class VjpTest {

    private static final double TOL = 1e-10;

    // ---- AD.vjp() basic ----

    @Test
    void testVjpSimple() {
        // f(x) = x^2, J = 2x
        // vjp(g) = 2x * g
        IDiffVector x = AD.vector(3.0, 5.0);
        Function<IDiffVector, IDiffVector> f = v -> v.pow(2);

        VjpResult result = AD.vjp(f, x);
        assertNotNull(result.y());
        assertNotNull(result.vjpFn());

        // J^T @ [1, 1] = [2*3, 2*5] = [6, 10]
        IDiffVector g = AD.vector(1.0, 1.0);
        IDiffVector vjpResult = result.vjpFn().apply(g);
        assertArrayEquals(new double[]{6.0, 10.0}, vjpResult.getValue().getData(), TOL);
    }

    @Test
    void testVjpMultipleApplies() {
        // f(x) = x^2, test reusing VjpFunction with different upstream gradients
        IDiffVector x = AD.vector(2.0, 3.0);
        Function<IDiffVector, IDiffVector> f = v -> v.pow(2);

        VjpResult result = AD.vjp(f, x);

        // First apply: g = [1, 1]
        IDiffVector g1 = AD.vector(1.0, 1.0);
        IDiffVector r1 = result.vjpFn().apply(g1);
        assertArrayEquals(new double[]{4.0, 6.0}, r1.getValue().getData(), TOL);

        // Second apply: g = [0.5, 2.0]
        IDiffVector g2 = AD.vector(0.5, 2.0);
        IDiffVector r2 = result.vjpFn().apply(g2);
        assertArrayEquals(new double[]{2.0, 12.0}, r2.getValue().getData(), TOL);

        // Third apply: g = [0, 0] should give [0, 0]
        IDiffVector g3 = AD.vector(0.0, 0.0);
        IDiffVector r3 = result.vjpFn().apply(g3);
        assertArrayEquals(new double[]{0.0, 0.0}, r3.getValue().getData(), TOL);
    }

    @Test
    void testVjpForwardOutput() {
        // Verify the forward output y = fn(x) is correct
        IDiffVector x = AD.vector(1.0, 2.0, 3.0);
        Function<IDiffVector, IDiffVector> f = v -> v.pow(2).sum(); // scalar output

        VjpResult result = AD.vjp(f, x);
        assertEquals(14.0, result.y().getValue().get(0), TOL);
    }

    @Test
    void testVjpMatchesFiniteDifferences() {
        // f(x) = x^2 (element-wise square)
        // Jacobian J = diag(2*x), VJP with g = [g0, g1, g2] gives [2*x0*g0, 2*x1*g1, 2*x2*g2]
        IDiffVector x = AD.vector(2.0, 3.0, 4.0);
        Function<IDiffVector, IDiffVector> f = v -> v.pow(2);

        // VJP with g = [1, 2, 3] should give [2*2*1, 2*3*2, 2*4*3] = [4, 12, 24]
        VjpResult vjp = AD.vjp(f, x);
        IDiffVector result = vjp.vjpFn().apply(AD.vector(1.0, 2.0, 3.0));
        assertArrayEquals(new double[]{4.0, 12.0, 24.0}, result.getValue().getData(), TOL);

        // Finite difference verification for first component
        double eps = 1e-6;
        double[] xData = {2.0, 3.0, 4.0};

        // df0/dx0 at x: f(x+eps*e0)_0 - f(x-eps*e0)_0 / (2*eps)
        double fwd0 = Math.pow(2.0 + eps, 2);
        double bwd0 = Math.pow(2.0 - eps, 2);
        double fdGrad00 = (fwd0 - bwd0) / (2 * eps);
        assertEquals(4.0, fdGrad00, 1e-5); // 2*2 = 4

        // Verify VJP matches finite diff gradient for scalar-valued function
        IDiffVector xs = AD.vector(1.0, 2.0, 3.0);
        Function<IDiffVector, IDiffVector> fs = v -> v.pow(2).sum(); // scalar

        VjpResult vjps = AD.vjp(fs, xs);
        IDiffVector grad = vjps.vjpFn().apply(AD.vector(1.0));

        // Manual finite differences
        double[] xd = {1.0, 2.0, 3.0};
        for (int i = 0; i < 3; i++) {
            double save = xd[i];
            xd[i] = save + eps;
            double fp = xd[0]*xd[0] + xd[1]*xd[1] + xd[2]*xd[2];
            xd[i] = save - eps;
            double fm = xd[0]*xd[0] + xd[1]*xd[1] + xd[2]*xd[2];
            xd[i] = save;
            double fd = (fp - fm) / (2 * eps);
            assertEquals(fd, grad.getValue().get(i), 1e-5,
                "FD mismatch at index " + i);
        }
    }

    @Test
    void testVjpDotProduct() {
        // h(x, y) = dot(x, y)
        // vjp_x(g) = g * y, vjp_y(g) = g * x
        IDiffVector x = AD.vector(1.0, 2.0, 3.0);
        IDiffVector y = AD.vector(4.0, 5.0, 6.0);

        // Function that takes a single input and computes dot with a fixed vector
        // We'll just use the standard backward for this test
        IDiffVector z = x.dot(y);
        z.backward();
        assertArrayEquals(new double[]{4.0, 5.0, 6.0}, x.getGradient().getData(), TOL);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, y.getGradient().getData(), TOL);
    }

    // ---- AD.batchVjp() ----

    @Test
    void testBatchVjp() {
        IDiffVector x1 = AD.vector(1.0, 2.0);
        IDiffVector x2 = AD.vector(3.0, 4.0);
        IDiffVector x3 = AD.vector(5.0, 6.0);
        List<IDiffVector> xs = List.of(x1, x2, x3);

        Function<IDiffVector, IDiffVector> f = v -> v.pow(2).sum(); // scalar output

        BatchVjpResult batch = AD.batchVjp(f, xs);
        assertEquals(3, batch.batchSize());
        assertEquals(3, batch.ys().length);
        assertEquals(3, batch.vjpFns().length);

        // Forward outputs
        assertEquals(5.0, batch.ys()[0].getValue().get(0), TOL);  // 1^2 + 2^2 = 5
        assertEquals(25.0, batch.ys()[1].getValue().get(0), TOL); // 3^2 + 4^2 = 25
        assertEquals(61.0, batch.ys()[2].getValue().get(0), TOL); // 5^2 + 6^2 = 61
    }

    @Test
    void testBatchVjpSumGradients() {
        IDiffVector x1 = AD.vector(1.0, 2.0);
        IDiffVector x2 = AD.vector(3.0, 4.0);
        List<IDiffVector> xs = List.of(x1, x2);

        // f(x) = x^2, gradient = 2x
        // sum of gradients = 2*(1,2) + 2*(3,4) = (8, 12)
        BatchVjpResult batch = AD.batchVjp(v -> v.pow(2), xs);
        IDiffVector sumGrad = batch.sumGradients(AD.vector(1.0, 1.0));
        assertArrayEquals(new double[]{8.0, 12.0}, sumGrad.getValue().getData(), TOL);
    }

    @Test
    void testBatchVjpMeanGradients() {
        IDiffVector x1 = AD.vector(1.0, 2.0);
        IDiffVector x2 = AD.vector(3.0, 4.0);
        List<IDiffVector> xs = List.of(x1, x2);

        // f(x) = x^2, gradient = 2x
        // mean of gradients = (2*(1,2) + 2*(3,4)) / 2 = (4, 6)
        BatchVjpResult batch = AD.batchVjp(v -> v.pow(2), xs);
        IDiffVector meanGrad = batch.meanGradients(AD.vector(1.0, 1.0));
        assertArrayEquals(new double[]{4.0, 6.0}, meanGrad.getValue().getData(), TOL);
    }

    @Test
    void testBatchVjpEmpty() {
        assertThrows(IllegalArgumentException.class,
            () -> AD.batchVjp(v -> v.pow(2), List.of()));
    }

    // ---- AD.vmap() ----

    @Test
    void testVmap() {
        IDiffVector x1 = AD.vector(1.0, 2.0);
        IDiffVector x2 = AD.vector(3.0, 4.0);
        List<IDiffVector> xs = List.of(x1, x2);

        IDiffVector[] ys = AD.vmap(v -> v.pow(2), xs);
        assertEquals(2, ys.length);
        assertArrayEquals(new double[]{1.0, 4.0}, ys[0].getValue().getData(), TOL);
        assertArrayEquals(new double[]{9.0, 16.0}, ys[1].getValue().getData(), TOL);
    }

    @Test
    void testVmapSum() {
        IDiffVector x1 = AD.vector(1.0, 2.0);
        IDiffVector x2 = AD.vector(3.0, 4.0);
        List<IDiffVector> xs = List.of(x1, x2);

        IDiffVector sum = AD.vmapSum(v -> v.pow(2).sum(), xs);
        // (1+4) + (9+16) = 30
        assertEquals(30.0, sum.getValue().get(0), TOL);
    }

    @Test
    void testVmapMean() {
        IDiffVector x1 = AD.vector(1.0, 2.0);
        IDiffVector x2 = AD.vector(3.0, 4.0);
        List<IDiffVector> xs = List.of(x1, x2);

        IDiffVector mean = AD.vmapMean(v -> v.pow(2).sum(), xs);
        // ((1+4) + (9+16)) / 2 = 15
        assertEquals(15.0, mean.getValue().get(0), TOL);
    }

    @Test
    void testVmapEmpty() {
        assertThrows(IllegalArgumentException.class,
            () -> AD.vmap(v -> v.pow(2), List.of()));
    }

    // ---- MixedMode.vjp() ----

    @Test
    void testMixedModeVjp() {
        IDiffVector x = AD.vector(2.0, 3.0);
        Function<IDiffVector, IDiffVector> f = v -> v.pow(2).sum(); // scalar

        // J @ [1, 1]^T = [2*2, 2*3] = [4, 6]
        double[] result = MixedMode.vjp(f, x, IDoubleVector.of(1.0, 1.0));
        assertArrayEquals(new double[]{4.0, 6.0}, result, TOL);
    }

    // ---- VJP with linear composition ----

    @Test
    void testVjpScalarLinearComposition() {
        // f(x) = 2*x0 + 3*x1 (scalar output)
        // Uses AD operations on the input to preserve the computation graph
        IDiffVector x = AD.vector(1.0, 2.0);

        Function<IDiffVector, IDiffVector> f = v -> {
            IDiffVector x0 = v.slice(0, 1); // [x0]
            IDiffVector x1 = v.slice(1, 2); // [x1]
            return x0.mul(2.0).add(x1.mul(3.0)); // scalar: 2*x0 + 3*x1
        };

        // Gradient: [2, 3]
        VjpResult result = AD.vjp(f, x);
        IDiffVector g = result.vjpFn().apply(AD.vector(1.0));
        assertArrayEquals(new double[]{2.0, 3.0}, g.getValue().getData(), TOL);

        // Different upstream gradient should scale
        IDiffVector g2 = result.vjpFn().apply(AD.vector(5.0));
        assertArrayEquals(new double[]{10.0, 15.0}, g2.getValue().getData(), TOL);
    }

    // ---- Edge cases ----

    @Test
    void testVjpScalarInput() {
        IDiffVector x = AD.vector(5.0);
        Function<IDiffVector, IDiffVector> f = v -> v.pow(3);

        VjpResult result = AD.vjp(f, x);
        // f(x) = x^3, f'(x) = 3x^2 = 75 at x=5
        IDiffVector g = result.vjpFn().apply(AD.vector(1.0));
        assertEquals(75.0, g.getValue().get(0), TOL);
    }

    @Test
    void testVjpBackwardPreservesOriginal() {
        // After vjp, the original variables should still work with normal backward
        IDiffVector x = AD.vector(1.0, 2.0);

        // Use vjp first
        VjpResult result = AD.vjp(v -> v.pow(2), x);
        result.vjpFn().apply(AD.vector(1.0, 1.0));

        // Reset and do normal backward
        x.zeroGradient();
        IDiffVector y = x.pow(2).sum();
        y.backward();
        assertArrayEquals(new double[]{2.0, 4.0}, x.getGradient().getData(), TOL);
    }
}
