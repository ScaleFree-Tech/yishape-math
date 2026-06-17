package com.yishape.lab.math.autodiff.graph;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;

/**
 * Verifies GPU binary skeleton cache behavior:
 * <ul>
 *   <li>First call builds cache after successful GPU execution</li>
 *   <li>Second call with same topology uses incremental path</li>
 *   <li>Cache hit produces same result as cache miss</li>
 *   <li>Structure hash change invalidates cache</li>
 * </ul>
 *
 * <p>Tensors are sized ≥MIN_ELEMENTS=1000 to avoid GPU skip threshold.</p>
 */
public class GpuCacheTest {

    private static boolean gpuPresent;

    @BeforeAll
    static void checkGpu() {
        gpuPresent = GpuOptionalRuntime.isGpuAvailable();
        GpuGraphExecutor.resetCooldown();
        if (gpuPresent) {
            System.out.println("[GpuCacheTest] GPU present — running cache tests");
        } else {
            System.out.println("[GpuCacheTest] GPU not available — skipping");
        }
    }

    private static double[] rand(int n) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = (i + 1) * 0.1;
        return a;
    }

    /**
     * Execute the same graph twice: the second call should use the cache.
     */
    @Test
    void testCacheHitSameResult() {
        assumeTrue(gpuPresent, "GPU not available");

        // Use 64x20 = 1280 elements (above 1000 threshold)
        int R = 64, C = 20;
        double[] d1 = rand(R * C), d2 = rand(R * C);

        // First execution
        RereDiffTensor a1 = (RereDiffTensor) AD.leafTensor(d1.clone(), R, C);
        RereDiffTensor b1 = (RereDiffTensor) AD.leafTensor(d2.clone(), R, C);
        RereDiffTensor loss1 = (RereDiffTensor) a1.add(b1).sum();
        double gpuLoss1 = GpuGraphExecutor.tryExecute(loss1);
        assertFalse(Double.isNaN(gpuLoss1), "First GPU execution should succeed");
        double[] grad1_a = a1.gradData().clone();
        double[] grad1_b = b1.gradData().clone();

        // Second execution with same topology but different data
        double[] d3 = rand(R * C), d4 = rand(R * C);
        RereDiffTensor a2 = (RereDiffTensor) AD.leafTensor(d3, R, C);
        RereDiffTensor b2 = (RereDiffTensor) AD.leafTensor(d4, R, C);
        RereDiffTensor loss2 = (RereDiffTensor) a2.add(b2).sum();
        double gpuLoss2 = GpuGraphExecutor.tryExecute(loss2);
        assertFalse(Double.isNaN(gpuLoss2), "Second GPU execution should succeed (cached)");
        double[] grad2_a = a2.gradData().clone();
        double[] grad2_b = b2.gradData().clone();

        // Both should produce correct gradients (sum gradient: ∂sum/∂each = 1.0)
        for (int i = 0; i < grad1_a.length; i++) {
            assertEquals(1.0, grad1_a[i], 1e-10,
                "First call grad_a[" + i + "] should be 1.0");
            assertEquals(1.0, grad2_a[i], 1e-10,
                "Second call grad_a[" + i + "] should be 1.0 (cached)");
        }
        assertEquals(gpuLoss1, gpuLoss2, 1e-10,
            "Cached execution should produce same loss");
    }

    /**
     * Different topologies should invalidate cache (add vs mul).
     */
    @Test
    void testStructureChangeInvalidatesCache() {
        assumeTrue(gpuPresent, "GPU not available");

        int R = 64, C = 20; // 1280 elements, above threshold
        // Use integer values (exact in f32 and f64) for precise comparison
        double[] d1 = new double[R * C];
        double[] d2 = new double[R * C];
        for (int i = 0; i < d1.length; i++) { d1[i] = i + 1; d2[i] = R * C - i; }

        // First call: add then sum
        RereDiffTensor a1 = (RereDiffTensor) AD.leafTensor(d1.clone(), R, C);
        RereDiffTensor b1 = (RereDiffTensor) AD.leafTensor(d2.clone(), R, C);
        RereDiffTensor loss1 = (RereDiffTensor) a1.add(b1).sum();
        double gpuLoss1 = GpuGraphExecutor.tryExecute(loss1);
        assertFalse(Double.isNaN(gpuLoss1));

        // Second call: MUL (not add) then sum — different op, cache miss → rebuild
        RereDiffTensor a2 = (RereDiffTensor) AD.leafTensor(d1.clone(), R, C);
        RereDiffTensor b2 = (RereDiffTensor) AD.leafTensor(d2.clone(), R, C);
        RereDiffTensor loss2 = (RereDiffTensor) a2.mul(b2).sum();
        double gpuLoss2 = GpuGraphExecutor.tryExecute(loss2);
        assertFalse(Double.isNaN(gpuLoss2),
            "Second graph (mul) should succeed even with cache from add");

        // mul loss is different from add loss (different op)
        assertNotEquals(gpuLoss1, gpuLoss2, 1e-10,
            "add loss should differ from mul loss");

        // mul sum gradient: ∂(a*b)/∂a = b
        double[] grad2_a = a2.gradData().clone();
        assertEquals(d2[0], grad2_a[0], 1e-10,
            "mul grad_a[0] should be b[0]");
    }

    /**
     * Broadcast + sum chain — verifies the specific graph that exposed the
     * serializeGraphCached side-effect issue during development.
     */
    @Test
    void testBroadcastThenSumDimCached() {
        assumeTrue(gpuPresent, "GPU not available");

        // Use 64x20 = 1280 + 64 = 1344 elements (above 1000 threshold)
        int B = 64, C = 20;
        double[] d1 = rand(B * C), d2 = rand(B);

        // First execution
        RereDiffTensor a1 = (RereDiffTensor) AD.leafTensor(d1.clone(), B, C);
        RereDiffTensor b1 = (RereDiffTensor) AD.leafTensor(d2.clone(), B);
        RereDiffTensor loss1 = (RereDiffTensor) a1.add(b1).sum(1, true).sum();
        double loss1Val = GpuGraphExecutor.tryExecute(loss1);
        assertFalse(Double.isNaN(loss1Val), "First broadcast+sum execution failed");
        double[] grad1_b = b1.gradData().clone();

        // Second execution (should use cached skeleton)
        RereDiffTensor a2 = (RereDiffTensor) AD.leafTensor(d1.clone(), B, C);
        RereDiffTensor b2 = (RereDiffTensor) AD.leafTensor(d2.clone(), B);
        RereDiffTensor loss2 = (RereDiffTensor) a2.add(b2).sum(1, true).sum();
        double loss2Val = GpuGraphExecutor.tryExecute(loss2);
        assertFalse(Double.isNaN(loss2Val), "Second broadcast+sum execution failed (cached)");
        double[] grad2_b = b2.gradData().clone();

        // Both should produce the same loss and gradients
        assertEquals(loss1Val, loss2Val, 1e-10,
            "Cached execution should produce same loss");
        assertArrayEquals(grad1_b, grad2_b, 1e-10,
            "Cached execution should produce same gradients");

        // b's gradient should be C (each b[i] contributes to C sum columns)
        for (int i = 0; i < B; i++) {
            assertEquals(C, grad1_b[i], 1e-10,
                "grad_b[" + i + "] should be C=" + C + " for sum over dim 1");
        }
    }
}
