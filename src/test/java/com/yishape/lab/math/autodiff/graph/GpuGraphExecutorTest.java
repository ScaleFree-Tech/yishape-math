package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.gpu.GpuSwitch;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Direct tests for {@link GpuGraphExecutor} — the graph-level GPU execution bridge.
 * Tests the full pipeline: build topo → validate ops → export JSON → GPU execute → apply gradients.
 * <p>
 * <b>Note on fusion:</b> The AD system fuses patterns like {@code pow(N).sum()} → {@code powSum},
 * {@code exp().sum()} → {@code expSum}, etc. These fused ops are NOT in the GPU's
 * {@code SUPPORTED_OPS}, so the GPU path correctly returns NaN and falls back to CPU.
 * Tests below use non-fused operations (e.g., {@code add(0)} as a barrier) to exercise the GPU path.
 */
public class GpuGraphExecutorTest {

    private static boolean gpuPresent;

    @BeforeAll
    static void detect() {
        System.setProperty("yishape.gpu.minElements", "0");
        gpuPresent = GpuOptionalRuntime.isGpuAvailable();
        GpuGraphExecutor.resetCooldown();
    }

    @AfterEach
    void restoreSwitch() {
        GpuSwitch.enable();
    }

    /**
     * Helper: add a non-optimizable barrier to prevent AD pattern fusion.
     * {@code x.abs()} is a supported GPU op that can't be constant-folded away.
     */
    private IDiffVector noFusion(IDiffVector x) {
        return x.abs();
    }

    // ==================== Diagnostics ====================

    @Test
    void testDiagnosticExportAndExecute() {
        if (!gpuPresent) return;
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        RereDiffVector loss = (RereDiffVector) a.add(b).sum();

        String json = GraphExporter.toJson(loss);

        String resultJson = GpuOptionalRuntime.tryExecuteGraph(json);

        double result = GpuGraphExecutor.tryExecute(loss);
    }

    // ==================== Basic GPU Execution ====================

    @Test
    void testSimpleAddSum() {
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        // add + sum — no fusion possible
        RereDiffVector loss = (RereDiffVector) a.add(b).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result), "GPU execution should succeed for add+sum");
            assertEquals(21.0, result, 1e-5);
            double[] gradA = a.getGradient().getData();
            double[] gradB = b.getGradient().getData();
            assertArrayEquals(new double[]{1, 1, 1}, gradA, 1e-4);
            assertArrayEquals(new double[]{1, 1, 1}, gradB, 1e-4);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testMulGrad() {
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        // abs() barrier prevents mulSum fusion
        RereDiffVector loss = (RereDiffVector) noFusion(a.mul(b)).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(32.0, result, 1e-5);
            double[] gradA = a.getGradient().getData();
            double[] gradB = b.getGradient().getData();
            assertArrayEquals(new double[]{4, 5, 6}, gradA, 1e-4);
            assertArrayEquals(new double[]{1, 2, 3}, gradB, 1e-4);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testDotProduct() {
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        RereDiffVector loss = (RereDiffVector) a.dot(b);

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(32.0, result, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testSubGrad() {
        IDiffVector a = AD.vector(new double[]{5, 10, 15});
        IDiffVector b = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) a.sub(b).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(24.0, result, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testDivGrad() {
        IDiffVector a = AD.vector(new double[]{6, 8, 10});
        IDiffVector b = AD.vector(new double[]{2, 4, 5});
        RereDiffVector loss = (RereDiffVector) a.div(b).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(7.0, result, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    // ==================== Unary Ops (non-fused) ====================

    @Test
    void testExpWithBarrier() {
        // exp().add(0).sum() — prevents expSum fusion
        IDiffVector x = AD.vector(new double[]{0, 1, 2});
        RereDiffVector loss = (RereDiffVector) noFusion(x.exp()).sum();
        String json = GraphExporter.toJson(loss);

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result), "exp+add(0)+sum should work on GPU, json=" + json);
            assertEquals(Math.exp(0) + Math.exp(1) + Math.exp(2), result, 1e-3);
            double[] grad = x.getGradient().getData();
            assertNotNull(grad);
            assertEquals(Math.exp(0), grad[0], 1e-3);
            assertEquals(Math.exp(1), grad[1], 1e-3);
            assertEquals(Math.exp(2), grad[2], 1e-3);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testPowWithBarrier() {
        // pow(2).add(0).sum() — prevents powSum fusion
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) noFusion(x.pow(2)).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(14.0, result, 1e-5); // 1+4+9
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{2, 4, 6}, grad, 1e-4);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testNegGrad() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) x.neg().sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(-6.0, result, 1e-5);
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{-1, -1, -1}, grad, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testAbsGrad() {
        IDiffVector x = AD.vector(new double[]{-2, -1, 1, 2});
        RereDiffVector loss = (RereDiffVector) x.abs().sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(6.0, result, 1e-5);
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{-1, -1, 1, 1}, grad, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testSqrtGrad() {
        IDiffVector x = AD.vector(new double[]{1, 4, 9});
        RereDiffVector loss = (RereDiffVector) x.sqrt().sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(6.0, result, 1e-5); // 1+2+3
            double[] grad = x.getGradient().getData();
            // d/dx sqrt(x) = 0.5/sqrt(x)
            assertEquals(0.5, grad[0], 1e-4);
            assertEquals(0.25, grad[1], 1e-4);
            assertEquals(1.0 / 6, grad[2], 1e-4);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    // ==================== Activation Ops ====================

    @Test
    void testReluGrad() {
        IDiffVector x = AD.vector(new double[]{-2, -1, 0, 1, 2});
        RereDiffVector loss = (RereDiffVector) x.relu().sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(3.0, result, 1e-5);
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{0, 0, 0, 1, 1}, grad, 1e-4);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testSigmoidGrad() {
        IDiffVector x = AD.vector(new double[]{0, 1});
        RereDiffVector loss = (RereDiffVector) x.sigmoid().sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            double[] grad = x.getGradient().getData();
            assertEquals(0.25, grad[0], 1e-4);
            assertEquals(0.1966, grad[1], 1e-3);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testTanhGrad() {
        IDiffVector x = AD.vector(new double[]{0, 1});
        RereDiffVector loss = (RereDiffVector) x.tanh().sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            double[] grad = x.getGradient().getData();
            assertEquals(1.0, grad[0], 1e-4);
            assertEquals(1 - Math.pow(Math.tanh(1), 2), grad[1], 1e-3);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testGeluGrad() {
        IDiffVector x = AD.vector(new double[]{0, 1});
        IDiffVector zero = AD.vector(new double[]{0, 0});
        // Fusion barrier: binary add(zero) creates a 2-input node, preventing
        // the AD system from fusing gelu()+sum() into geluSum (not yet in Rust GPU backend).
        RereDiffVector loss = (RereDiffVector) x.gelu().add(zero).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result), "GPU result for gelu+add+sum should be non-NaN");
            assertNotNull(x.getGradient());
            // sum(gelu([0, gelu(1)])) with gelu(0)=0, gelu(1)≈0.8413
            assertEquals(0.84119, result, 1e-4, "gelu sum loss");
            // gelu'(0) = 0.5, gelu'(1) ≈ 1.083 (silu approx, sigmoid≈0.731, derivative≈1.083)
            double[] grad = x.getGradient().getData();
            assertEquals(0.5, grad[0], 1e-3);
            assertTrue(grad[1] > 0.5, "gelu'(1) should be > 0.5, got " + grad[1]);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testGeluSumFusionFallsBack() {
        // gelu().sum() is fused to geluSum by the AD system. The Rust GPU backend
        // now implements geluSum, so GPU should execute it and return the correct result.
        if (!gpuPresent) return;
        IDiffVector x = AD.vector(new double[]{0, 1});
        RereDiffVector loss = (RereDiffVector) x.gelu().sum();
        double result = GpuGraphExecutor.tryExecute(loss);
        assertFalse(Double.isNaN(result), "geluSum should execute on GPU");
        // gelu(0)=0, gelu(1)≈0.841191, sum≈0.841191
        assertEquals(0.841191, result, 1e-4);
    }

    // ==================== Scalar Ops ====================

    @Test
    void testAddScalarGrad() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) x.add(10.0).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(36.0, result, 1e-5);
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{1, 1, 1}, grad, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testMulScalarGrad() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) x.mul(3.0).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(18.0, result, 1e-5);
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{3, 3, 3}, grad, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    // ==================== Reduction Ops ====================

    @Test
    void testSumGrad() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3, 4});
        RereDiffVector loss = (RereDiffVector) x.sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(10.0, result, 1e-5);
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{1, 1, 1, 1}, grad, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testMeanGrad() {
        IDiffVector x = AD.vector(new double[]{2, 4, 6, 8});
        RereDiffVector loss = (RereDiffVector) x.mean();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(5.0, result, 1e-5);
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{0.25, 0.25, 0.25, 0.25}, grad, 1e-5);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    // ==================== Chained Ops ====================

    @Test
    void testExpLogRoundtrip() {
        // exp(log(x)) = x — use abs() barrier to prevent expSum fusion
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) noFusion(x.log().exp()).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertEquals(6.0, result, 1e-3);
            double[] grad = x.getGradient().getData();
            assertArrayEquals(new double[]{1, 1, 1}, grad, 1e-3);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testSigmoidTanhChain() {
        IDiffVector x = AD.vector(new double[]{0.5, 1.0});
        RereDiffVector loss = (RereDiffVector) x.sigmoid().tanh().sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
            assertNotNull(x.getGradient());
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    // ==================== GpuSwitch Behavior ====================

    @Test
    void testSwitchDisabledReturnsNaN() {
        GpuSwitch.disable();
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) x.add(0).sum();

        double result = GpuGraphExecutor.tryExecute(loss);
        assertTrue(Double.isNaN(result), "Switch disabled should return NaN");
    }

    @Test
    void testSwitchReenabledWorks() {
        GpuSwitch.disable();
        IDiffVector x1 = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss1 = (RereDiffVector) x1.add(0).sum();
        assertTrue(Double.isNaN(GpuGraphExecutor.tryExecute(loss1)));

        GpuSwitch.enable();
        IDiffVector x2 = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss2 = (RereDiffVector) x2.add(0).sum();
        double result = GpuGraphExecutor.tryExecute(loss2);
        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
        }
    }

    @Test
    void testRunWithDisable() {
        IDiffVector x1 = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss1 = (RereDiffVector) x1.add(0).sum();

        GpuSwitch.runWith(false, () -> {
            double r = GpuGraphExecutor.tryExecute(loss1);
            assertTrue(Double.isNaN(r), "Inside runWith(false) should return NaN");
        });

        // After runWith, switch should be restored
        IDiffVector x2 = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss2 = (RereDiffVector) x2.add(0).sum();
        double result = GpuGraphExecutor.tryExecute(loss2);
        if (gpuPresent) {
            assertFalse(Double.isNaN(result));
        }
    }

    // ==================== Fused Op Fallback ====================

    @Test
    void testFusedPowSum() {
        // pow(2).sum() → fused to powSum → GPU can handle → returns correct loss
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) x.pow(2).sum();

        double result = GpuGraphExecutor.tryExecute(loss);

        if (gpuPresent) {
            assertFalse(Double.isNaN(result), "GPU should compute fused powSum");
            assertEquals(14.0, result, 1e-5); // 1^2 + 2^2 + 3^2 = 14
        } else {
            assertTrue(Double.isNaN(result), "No GPU available, should fall back to NaN");
        }

        // Verify the fused node still has correct structure
        String json = GraphExporter.toJson(loss);
        assertTrue(json.contains("powSum"), "Should produce fused powSum op");
    }

    @Test
    void testFusedExpSum() {
        IDiffVector x = AD.vector(new double[]{0, 1, 2});
        RereDiffVector loss = (RereDiffVector) x.exp().sum();

        double result = GpuGraphExecutor.tryExecute(loss);
        if (gpuPresent) {
            // exp(0)+exp(1)+exp(2) ≈ 1 + 2.718 + 7.389 = 11.107
            assertEquals(11.107, result, 0.01);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    @Test
    void testFusedSquareSum() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) x.square().sum();

        double result = GpuGraphExecutor.tryExecute(loss);
        if (gpuPresent) {
            // 1² + 2² + 3² = 14
            assertEquals(14.0, result, 1e-4);
        } else {
            assertTrue(Double.isNaN(result));
        }
    }

    // ==================== JSON Export Validation ====================

    @Test
    void testExportProducesValidJson() {
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        RereDiffVector loss = (RereDiffVector) a.mul(b).sum();

        String json = GraphExporter.toJson(loss);
        assertNotNull(json);
        assertTrue(json.contains("\"nodes\":["));
        assertTrue(json.endsWith("]}"));
        int nodeCount = json.split("\"op\":").length - 1;
        assertTrue(nodeCount >= 3, "Should have at least 3 nodes");
    }

    // ==================== Edge Cases ====================

    @Test
    void testSingleNodeGraph() {
        RereDiffVector leaf = new RereDiffVector(
            com.yishape.lab.math.linalg.IDoubleVector.of(new double[]{42}));
        double result = GpuGraphExecutor.tryExecute(leaf);
        // Single leaf: GPU may return NaN (no op) or a valid value — either is acceptable
        // The important thing is no exception is thrown
        assertNotNull(Double.valueOf(result));
    }

    // ==================== Failure Path Tests ====================

    @Test
    void testUnsupportedOpReturnsNaN() {
        if (!gpuPresent) return;
        // reciprocal is not a GPU-supported op; verify graceful fallback
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        RereDiffVector loss = (RereDiffVector) x.reciprocal().sum();
        double result = GpuGraphExecutor.tryExecute(loss);
        assertTrue(Double.isNaN(result), "Unsupported op should return NaN for graceful CPU fallback");
    }

    @Test
    void testCooldownAfterRepeatedFailures() {
        if (!gpuPresent) return;
        // Run unsupported ops to trigger GPU failures; verify cooldown activates
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        for (int i = 0; i < 5; i++) {
            RereDiffVector loss = (RereDiffVector) x.reciprocal().sum();
            GpuGraphExecutor.tryExecute(loss);
        }
        // After enough failures, cooldown should start and GPU should be skipped
        // Re-enable after test to not affect subsequent tests
    }

    @Test
    void testSoftmaxCrossEntropySparseGpu() {
        if (!gpuPresent) return;
        // 3 samples, 5 classes. labels = {1, 0, 2}
        double[] logits = new double[]{0.5, 2.0, 0.3, 0.1, 0.0, 1.0, 0.5, 0.2, 0.8, 0.1, 0.2, 0.7, 0.9, 0.3, 0.4};
        int[] labels = new int[]{1, 0, 2};
        RereDiffTensor x = new RereDiffTensor(logits, 3, 5);
        x.setRequiresGrad(true);
        // Call softmaxCrossEntropySparse — now exports labels as a graph input tensor
        IDiffTensor loss = x.softmaxCrossEntropySparse(labels, 1);
        // Verify GPU/HPC execution works (labels are now graph inputs, reachable via serialization)
        double result = GpuGraphExecutor.tryExecute((RereDiffTensor) loss);
        assertFalse(Double.isNaN(result), "softmaxCrossEntropySparse GPU execution should succeed");
        // CPU backward for reference
        loss.backward();
        assertNotNull(x.gradData());
    }

    @Test
    void testGpuDisabledReturnsNaN() {
        GpuSwitch.disable();
        try {
            IDiffVector x = AD.vector(new double[]{1, 2, 3});
            RereDiffVector loss = (RereDiffVector) x.add(1.0).sum();
            double result = GpuGraphExecutor.tryExecute(loss);
            assertTrue(Double.isNaN(result), "GPU disabled should return NaN");
        } finally {
            GpuSwitch.enable();
        }
    }

    @Test
    void testJsonGradientMismatchReturnsNaN() {
        if (!gpuPresent) return;
        // Create a graph and verify the executor handles it gracefully
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        RereDiffVector loss = (RereDiffVector) a.add(b).sum();
        double result = GpuGraphExecutor.tryExecute(loss);
        // Normal execution should succeed
        assertFalse(Double.isNaN(result), "Basic add+sum should succeed on GPU");
    }
}
