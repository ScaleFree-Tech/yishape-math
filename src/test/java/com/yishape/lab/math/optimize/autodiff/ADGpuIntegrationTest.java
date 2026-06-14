package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.optimize.IOptimizer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end integration tests for the GPU→HPC→CPU autodiff fallback chain.
 * <p>
 * Every test exercises the full {@code AD.tryGpuExecute → AD.tryHpcExecute → loss.backward()}
 * chain via the {@code AD.optimize()} entry point or direct {@code tryGpuExecute} call.
 * Tests pass regardless of which tier actually executes — if GPU/HPC are unavailable,
 * the Java backward path is exercised transparently.
 * <p>
 * When the GPU native module IS on the classpath and a GPU is present, the GPU path
 * is exercised, and results are compared against CPU (pure Java) backward to verify
 * cross-tier correctness.
 */
public class ADGpuIntegrationTest {

    private static boolean gpuPresent;
    private static final double TOL = 1e-10;
    private static final double GPU_TOL = 1e-4; // GPU f32 precision

    @BeforeAll
    static void detectEnvironment() {
        // Disable minElements threshold so GPU path is exercised even for tiny test graphs
        System.setProperty("yishape.gpu.minElements", "0");
        gpuPresent = GpuOptionalRuntime.isGpuAvailable();
        System.out.println("GPU available: " + gpuPresent);
    }

    // ==================== Helper ====================

    /**
     * Run the full GPU→HPC→CPU chain on a loss, then return the leaf gradients.
     */
    private double[] runChain(IDiffVector loss, IDiffVector... inputs) {
        boolean executed = AD.tryGpuExecute(loss);
        if (!executed) {
            executed = AD.tryHpcExecute(loss);
        }
        if (!executed) {
            loss.backward();
        }
        // Collect gradients from all inputs
        if (inputs.length == 1) {
            return inputs[0].getGradient().getData();
        }
        // For multi-leaf, concatenate all gradients
        int totalLen = 0;
        double[][] grads = new double[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            grads[i] = inputs[i].getGradient().getData();
            totalLen += grads[i].length;
        }
        double[] all = new double[totalLen];
        int pos = 0;
        for (double[] g : grads) {
            System.arraycopy(g, 0, all, pos, g.length);
            pos += g.length;
        }
        return all;
    }

    /**
     * Compute expected gradient via central finite differences.
     */
    private double[] numericalGradient(java.util.function.ToDoubleFunction<double[]> fn,
                                        double[] x, double eps) {
        double[] grad = new double[x.length];
        double[] xp = x.clone();
        double[] xm = x.clone();
        for (int i = 0; i < x.length; i++) {
            xp[i] = x[i] + eps;
            xm[i] = x[i] - eps;
            grad[i] = (fn.applyAsDouble(xp) - fn.applyAsDouble(xm)) / (2 * eps);
            xp[i] = x[i];
            xm[i] = x[i];
        }
        return grad;
    }

    // ==================== 1. Individual Op Gradients ====================

    @Test
    void testSquareGrad() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        IDiffVector loss = x.pow(2).sum();
        double[] grad = runChain(loss, x);
        assertArrayEquals(new double[]{2, 4, 6}, grad, TOL);
    }

    @Test
    void testExpGrad() {
        IDiffVector x = AD.vector(new double[]{0, 1, 2});
        IDiffVector loss = x.exp().sum();
        double[] grad = runChain(loss, x);
        double tol = gpuPresent ? GPU_TOL : TOL;
        assertEquals(Math.exp(0), grad[0], tol);
        assertEquals(Math.exp(1), grad[1], tol);
        assertEquals(Math.exp(2), grad[2], tol);
    }

    @Test
    void testLogGrad() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        IDiffVector loss = x.log().sum();
        double[] grad = runChain(loss, x);
        double tol = gpuPresent ? GPU_TOL : TOL;
        // d/dx log(x) = 1/x
        assertEquals(1.0, grad[0], tol);
        assertEquals(0.5, grad[1], tol);
        assertEquals(1.0 / 3, grad[2], tol);
    }

    @Test
    void testSigmoidGrad() {
        IDiffVector x = AD.vector(new double[]{-1, 0, 1});
        IDiffVector loss = x.sigmoid().sum();
        double[] grad = runChain(loss, x);
        double tol = gpuPresent ? GPU_TOL : TOL;
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        for (int i = 0; i < 3; i++) {
            double s = 1.0 / (1.0 + Math.exp(-x.getData()[i]));
            assertEquals(s * (1 - s), grad[i], tol);
        }
    }

    @Test
    void testTanhGrad() {
        IDiffVector x = AD.vector(new double[]{-1, 0, 1});
        IDiffVector loss = x.tanh().sum();
        double[] grad = runChain(loss, x);
        double tol = gpuPresent ? GPU_TOL : TOL;
        // d/dx tanh(x) = 1 - tanh(x)^2
        for (int i = 0; i < 3; i++) {
            double t = Math.tanh(x.getData()[i]);
            assertEquals(1 - t * t, grad[i], tol);
        }
    }

    @Test
    void testReluGrad() {
        IDiffVector x = AD.vector(new double[]{-2, -1, 0, 1, 2});
        IDiffVector loss = x.relu().sum();
        double[] grad = runChain(loss, x);
        assertArrayEquals(new double[]{0, 0, 0, 1, 1}, grad, TOL);
    }

    @Test
    void testGeluGrad() {
        IDiffVector x = AD.vector(new double[]{-1, 0, 1});
        IDiffVector loss = x.gelu().sum();
        double[] grad = runChain(loss, x);
        // Verify against numerical gradient
        double[] xData = {-1, 0, 1};
        double[] numGrad = numericalGradient(
            d -> {
                IDiffVector v = AD.vector(d);
                return v.gelu().sum().getValue().getData()[0];
            }, xData, 1e-6);
        // Use GPU_TOL since GPU gelu uses tanh approximation in f32
        double tol = gpuPresent ? GPU_TOL : TOL;
        assertArrayEquals(numGrad, grad, tol);
    }

    // ==================== 2. Binary Op Gradients ====================

    @Test
    void testDotGrad() {
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        IDiffVector loss = a.dot(b);
        double[] gradA = runChain(loss, a, b);
        // d/da (a·b) = b, d/db (a·b) = a
        assertArrayEquals(new double[]{4, 5, 6, 1, 2, 3}, gradA, TOL);
    }

    @Test
    void testMulGrad() {
        IDiffVector a = AD.vector(new double[]{2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5});
        IDiffVector loss = a.mul(b).sum();
        double[] grads = runChain(loss, a, b);
        // d/da = b, d/db = a
        assertArrayEquals(new double[]{4, 5, 2, 3}, grads, TOL);
    }

    @Test
    void testAddSubGrad() {
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        IDiffVector loss = a.add(b).sum();
        double[] grads = runChain(loss, a, b);
        // d/da = 1, d/db = 1
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, grads, TOL);
    }

    @Test
    void testDivGrad() {
        IDiffVector a = AD.vector(new double[]{6, 8, 10});
        IDiffVector b = AD.vector(new double[]{2, 4, 5});
        IDiffVector loss = a.div(b).sum();
        double[] grads = runChain(loss, a, b);
        // grads = [d/da, d/db] = [1/b, -a/b^2]
        // d/da = [1/2, 1/4, 1/5] = [0.5, 0.25, 0.2]
        // d/db = [-6/4, -8/16, -10/25] = [-1.5, -0.5, -0.4]
        // Use GPU_TOL since GPU path uses f32 precision
        double tol = gpuPresent ? GPU_TOL : TOL;
        assertArrayEquals(new double[]{0.5, 0.25, 0.2}, new double[]{grads[0], grads[1], grads[2]}, tol);
        assertArrayEquals(new double[]{-1.5, -0.5, -0.4}, new double[]{grads[3], grads[4], grads[5]}, tol);
    }

    // ==================== 3. Chained Operations ====================

    @Test
    void testChainedOps() {
        // f(x) = sum(relu(x^2 - 1))
        IDiffVector x = AD.vector(new double[]{-2, -1, 0, 1, 2});
        IDiffVector ones = AD.vector(new double[]{1, 1, 1, 1, 1});
        IDiffVector loss = x.pow(2).sub(ones).relu().sum();
        double[] grad = runChain(loss, x);
        // x^2-1: [3, 0, -1, 0, 3], relu: [3, 0, 0, 0, 3]
        // d/dx relu(x^2-1) = (x^2-1>0 ? 2x : 0)
        assertArrayEquals(new double[]{-4, 0, 0, 0, 4}, grad, TOL);
    }

    @Test
    void testSigmoidTanhChain() {
        IDiffVector x = AD.vector(new double[]{0.5, 1.0});
        IDiffVector loss = x.sigmoid().tanh().sum();
        double[] grad = runChain(loss, x);
        // Verify against numerical gradient
        double[] numGrad = numericalGradient(
            d -> {
                IDiffVector v = AD.vector(d);
                return v.sigmoid().tanh().sum().getValue().getData()[0];
            }, x.getData(), 1e-6);
        assertArrayEquals(numGrad, grad, 1e-5);
    }

    // ==================== 4. Logistic Loss ====================

    @Test
    void testLogisticLoss() {
        IDiffVector w = AD.vector(new double[]{0.5, -0.3, 0.1});
        double[] xData = {1.0, 2.0, -1.0};
        IDiffVector x = AD.vector(xData);
        double yTrue = 1.0;

        IDiffVector logit = w.dot(x);
        IDiffVector yVec = AD.vector(yTrue);
        IDiffVector negYLogit = yVec.mul(-1).mul(logit);
        IDiffVector loss = negYLogit.exp().add(1).log();

        double[] grad = runChain(loss, w);

        // Verify against numerical gradient
        double[] wData = {0.5, -0.3, 0.1};
        double[] numGrad = numericalGradient(
            wd -> {
                double logitVal = 0;
                for (int i = 0; i < wd.length; i++) logitVal += wd[i] * xData[i];
                return Math.log(1 + Math.exp(-yTrue * logitVal));
            }, wData, 1e-6);
        assertArrayEquals(numGrad, grad, 1e-5);
    }

    // ==================== 5. GPU vs CPU Comparison ====================

    /**
     * Helper: run GPU graph execution and fallback to CPU, returning the gradient.
     * Returns null if GPU execution succeeded (to indicate GPU path was used),
     * or the CPU gradient if fallback was needed.
     */
    private double[] runWithGpuOrFallback(IDiffVector loss, IDiffVector input) {
        boolean gpuUsed = AD.tryGpuExecute(loss);
        if (gpuUsed && input.getGradient() != null) {
            return input.getGradient().getData();
        }
        // Fallback to CPU backward
        loss.backward();
        return input.getGradient().getData();
    }

    @Test
    void testGpuVsCpuSquareGrad() {
        // Run through GPU (or fallback to CPU)
        IDiffVector x1 = AD.vector(new double[]{1, 2, 3, 4, 5});
        IDiffVector loss1 = x1.pow(2).sum();
        double[] grad1 = runWithGpuOrFallback(loss1, x1);

        // Run through CPU
        IDiffVector x2 = AD.vector(new double[]{1, 2, 3, 4, 5});
        IDiffVector loss2 = x2.pow(2).sum();
        loss2.backward();
        double[] cpuGrad = x2.getGradient().getData();

        assertArrayEquals(cpuGrad, grad1, GPU_TOL);
    }

    @Test
    void testGpuVsCpuExpGrad() {
        IDiffVector x1 = AD.vector(new double[]{0, 0.5, 1, 1.5, 2});
        IDiffVector loss1 = x1.exp().sum();
        double[] grad1 = runWithGpuOrFallback(loss1, x1);

        IDiffVector x2 = AD.vector(new double[]{0, 0.5, 1, 1.5, 2});
        IDiffVector loss2 = x2.exp().sum();
        loss2.backward();
        double[] cpuGrad = x2.getGradient().getData();

        assertArrayEquals(cpuGrad, grad1, GPU_TOL);
    }

    @Test
    void testGpuVsCpuSigmoidGrad() {
        IDiffVector x1 = AD.vector(new double[]{-2, -1, 0, 1, 2});
        IDiffVector loss1 = x1.sigmoid().sum();
        double[] grad1 = runWithGpuOrFallback(loss1, x1);

        IDiffVector x2 = AD.vector(new double[]{-2, -1, 0, 1, 2});
        IDiffVector loss2 = x2.sigmoid().sum();
        loss2.backward();
        double[] cpuGrad = x2.getGradient().getData();

        assertArrayEquals(cpuGrad, grad1, GPU_TOL);
    }

    @Test
    void testGpuVsCpuMulGrad() {
        IDiffVector a1 = AD.vector(new double[]{1, 2, 3});
        IDiffVector b1 = AD.vector(new double[]{4, 5, 6});
        IDiffVector loss1 = a1.mul(b1).sum();
        boolean gpuUsed = AD.tryGpuExecute(loss1);
        if (!gpuUsed || a1.getGradient() == null) {
            loss1.backward();
        }
        double[] gpuGradA = a1.getGradient().getData();
        double[] gpuGradB = b1.getGradient().getData();

        IDiffVector a2 = AD.vector(new double[]{1, 2, 3});
        IDiffVector b2 = AD.vector(new double[]{4, 5, 6});
        IDiffVector loss2 = a2.mul(b2).sum();
        loss2.backward();
        double[] cpuGradA = a2.getGradient().getData();
        double[] cpuGradB = b2.getGradient().getData();

        assertArrayEquals(cpuGradA, gpuGradA, GPU_TOL);
        assertArrayEquals(cpuGradB, gpuGradB, GPU_TOL);
    }

    @Test
    void testGpuVsCpuDotGrad() {
        IDiffVector a1 = AD.vector(new double[]{1, 2, 3});
        IDiffVector b1 = AD.vector(new double[]{4, 5, 6});
        IDiffVector loss1 = a1.dot(b1);
        boolean gpuUsed = AD.tryGpuExecute(loss1);
        if (!gpuUsed || a1.getGradient() == null) {
            loss1.backward();
        }
        double[] gpuGradA = a1.getGradient().getData();
        double[] gpuGradB = b1.getGradient().getData();

        IDiffVector a2 = AD.vector(new double[]{1, 2, 3});
        IDiffVector b2 = AD.vector(new double[]{4, 5, 6});
        IDiffVector loss2 = a2.dot(b2);
        loss2.backward();
        double[] cpuGradA = a2.getGradient().getData();
        double[] cpuGradB = b2.getGradient().getData();

        assertArrayEquals(cpuGradA, gpuGradA, GPU_TOL);
        assertArrayEquals(cpuGradB, gpuGradB, GPU_TOL);
    }

    @Test
    void testGpuVsCpuChainedGrad() {
        IDiffVector x1 = AD.vector(new double[]{-1, 0, 1, 2});
        IDiffVector loss1 = x1.relu().sigmoid().sum();
        double[] grad1 = runWithGpuOrFallback(loss1, x1);

        IDiffVector x2 = AD.vector(new double[]{-1, 0, 1, 2});
        IDiffVector loss2 = x2.relu().sigmoid().sum();
        loss2.backward();
        double[] cpuGrad = x2.getGradient().getData();

        assertArrayEquals(cpuGrad, grad1, GPU_TOL);
    }

    // ==================== 6. Optimization Convergence ====================

    @Test
    void testOptimizeSquareLoss() {
        double[] target = {3.0, 4.0, 5.0};
        IDoubleVector initX = IDoubleVector.of(new double[]{0, 0, 0});

        java.util.function.Function<IDiffVector, IDiffVector> lossBuilder = x -> {
            IDiffVector t = AD.vector(target);
            return x.sub(t).square().sum();
        };

        IOptimizer optimizer = com.yishape.lab.math.optimize.Opts.lbfgs();
        var result = AD.optimize(initX, lossBuilder, optimizer);

        assertNotNull(result);
        double[] xOpt = ((IDoubleVector) result.getOptimalPoint()).getData();
        for (int i = 0; i < target.length; i++) {
            assertEquals(target[i], xOpt[i], 1e-3,
                "x[" + i + "] should converge to " + target[i]);
        }
    }

    @Test
    void testOptimizeMultiDim() {
        // Minimize f(x) = sum(x^4 - 8x^2 + 16)  = sum((x^2 - 4)^2)
        // Minimum at x = [+2, -2] (or [-2, +2] etc.)
        IDoubleVector initX = IDoubleVector.of(new double[]{0.5, -0.5});

        java.util.function.Function<IDiffVector, IDiffVector> lossBuilder = x -> {
            // (x^2 - 4)^2
            IDiffVector four = AD.vector(new double[]{4, 4});
            IDiffVector diff = x.mul(x).sub(four);
            return diff.mul(diff).sum();
        };

        IOptimizer optimizer = com.yishape.lab.math.optimize.Opts.lbfgs();
        var result = AD.optimize(initX, lossBuilder, optimizer);

        assertNotNull(result);
        double[] xOpt = ((IDoubleVector) result.getOptimalPoint()).getData();
        // x[i]^2 should be close to 4
        assertEquals(4.0, xOpt[0] * xOpt[0], 0.1);
        assertEquals(4.0, xOpt[1] * xOpt[1], 0.1);
    }

    // ==================== 7. Fallback Behavior ====================

    @Test
    void testFallbackChainCompletes() {
        // This test always passes — it just verifies the chain doesn't throw
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        IDiffVector loss = x.pow(2).sum();
        double[] grad = runChain(loss, x);
        assertNotNull(grad);
        assertEquals(3, grad.length);
        assertArrayEquals(new double[]{2, 4, 6}, grad, TOL);
    }

    @Test
    void testMultiLeafGraph() {
        IDiffVector a = AD.vector(new double[]{1, 2});
        IDiffVector b = AD.vector(new double[]{3, 4});
        IDiffVector loss = a.mul(b).sum();

        double[] grads = runChain(loss, a, b);
        // d/da = b = [3, 4], d/db = a = [1, 2]
        assertArrayEquals(new double[]{3, 4, 1, 2}, grads, TOL);
    }

    @Test
    void testExpLogRoundtrip() {
        // exp(log(x)) = x, so grad should be 1
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        IDiffVector loss = x.log().exp().sum();
        double[] grad = runChain(loss, x);
        assertArrayEquals(new double[]{1, 1, 1}, grad, TOL);
    }

    @Test
    void testMeanGrad() {
        IDiffVector x = AD.vector(new double[]{2, 4, 6, 8});
        IDiffVector loss = x.mean();
        double[] grad = runChain(loss, x);
        // d/dx mean(x) = 1/n for each element
        assertArrayEquals(new double[]{0.25, 0.25, 0.25, 0.25}, grad, TOL);
    }
}
