package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for FlashAttentionDispatch: native availability checks, fallback behavior,
 * and API contract validation.
 *
 * <p>The pure Java FlashAttention algorithm is tested in yishape-dl's FlashAttentionTest.
 * This class tests the dispatch layer that bridges Java ↔ HPC native FlashAttention.</p>
 */
class TestFlashAttentionDispatch {

    // ============ Availability checks ============

    @Test
    void testIsNativeAvailableDoesNotThrow() {
        // Should never throw, even when HPC library is absent
        boolean available = FlashAttentionDispatch.isNativeAvailable();
        // May be true or false — just verifying it doesn't crash
        assertDoesNotThrow(FlashAttentionDispatch::isNativeAvailable,
            "isNativeAvailable() should never throw");
    }

    @Test
    void testIsNativeAvailableIdempotent() {
        // Calling twice should give same result
        boolean first = FlashAttentionDispatch.isNativeAvailable();
        boolean second = FlashAttentionDispatch.isNativeAvailable();
        assertEquals(first, second, "isNativeAvailable() should be idempotent");
    }

    // ============ Forward: null on unavailable backend ============

    @Test
    void testFlashForwardReturnsNullWhenUnavailable() {
        // When native backend is absent, flashForward should return null gracefully
        if (!FlashAttentionDispatch.isNativeAvailable()) {
            int seqLen = 4, numHeads = 2, headDim = 8, numKVHeads = 2;
            double[] Q = new double[seqLen * numHeads * headDim];
            double[] K = new double[seqLen * numKVHeads * headDim];
            double[] V = new double[seqLen * numKVHeads * headDim];

            FlashAttentionDispatch.NativeFlashResult result =
                FlashAttentionDispatch.flashForward(Q, K, V, seqLen, numHeads, numKVHeads, headDim, false);

            assertNull(result, "flashForward should return null when native backend unavailable");
        }
    }

    @Test
    void testFlashForwardNeverThrows() {
        // Even with invalid inputs, should gracefully return null rather than throw
        int seqLen = 0, numHeads = 0, headDim = 0, numKVHeads = 0;
        assertDoesNotThrow(() -> {
            FlashAttentionDispatch.NativeFlashResult result = FlashAttentionDispatch.flashForward(
                new double[0], new double[0], new double[0],
                seqLen, numHeads, numKVHeads, headDim, false);
            // Can be null (unavailable) or non-null (HPC handled it) — either is OK
        }, "flashForward should never throw, even with zero-sized inputs");
    }

    // ============ Backward: null on unavailable backend ============

    @Test
    void testFlashBackwardReturnsNullWhenUnavailable() {
        if (!FlashAttentionDispatch.isNativeAvailable()) {
            int seqLen = 4, numHeads = 2, headDim = 8, numKVHeads = 2;
            double[] dO = new double[seqLen * numHeads * headDim];
            double[] Q = new double[seqLen * numHeads * headDim];
            double[] K = new double[seqLen * numKVHeads * headDim];
            double[] V = new double[seqLen * numKVHeads * headDim];
            double[] L = new double[numHeads * seqLen];
            double[] M = new double[numHeads * seqLen];

            double[][] result = FlashAttentionDispatch.flashBackward(
                dO, Q, K, V, L, M, seqLen, numHeads, numKVHeads, headDim, false);

            assertNull(result, "flashBackward should return null when native backend unavailable");
        }
    }

    @Test
    void testFlashBackwardNeverThrows() {
        assertDoesNotThrow(() -> {
            double[][] result = FlashAttentionDispatch.flashBackward(
                new double[0], new double[0], new double[0], new double[0],
                new double[0], new double[0], 0, 0, 0, 0, false);
            // Can be null or non-null — either is OK
        }, "flashBackward should never throw, even with zero-sized inputs");
    }

    // ============ NativeFlashResult record ============

    @Test
    void testNativeFlashResultRecord() {
        double[] O = {1.0, 2.0, 3.0};
        double[] L = {0.5, 0.8};
        double[] M = {1.0, 2.0};

        FlashAttentionDispatch.NativeFlashResult result =
            new FlashAttentionDispatch.NativeFlashResult(O, L, M);

        assertSame(O, result.O());
        assertSame(L, result.L());
        assertSame(M, result.M());
    }

    // ============ GQA compatibility: API surface test ============

    @Test
    void testGqaSignatureAcceptsFewerKVHeads() {
        // GQA: numHeads=4, numKVHeads=2 — should be valid
        if (!FlashAttentionDispatch.isNativeAvailable()) {
            int seqLen = 8, numHeads = 4, numKVHeads = 2, headDim = 16;
            double[] Q = new double[seqLen * numHeads * headDim];
            double[] K = new double[seqLen * numKVHeads * headDim];
            double[] V = new double[seqLen * numKVHeads * headDim];

            FlashAttentionDispatch.NativeFlashResult result =
                FlashAttentionDispatch.flashForward(Q, K, V, seqLen, numHeads, numKVHeads, headDim, false);

            assertNull(result,
                "GQA (numHeads=" + numHeads + ", numKVHeads=" + numKVHeads + ") should return null cleanly");
        }
    }

    // ============ Causal flag test ============

    @Test
    void testCausalSignature() {
        // Just verify causal=true doesn't cause issues at the dispatch layer
        if (!FlashAttentionDispatch.isNativeAvailable()) {
            int seqLen = 4, numHeads = 2, numKVHeads = 2, headDim = 8;
            double[] Q = new double[seqLen * numHeads * headDim];
            double[] K = new double[seqLen * numKVHeads * headDim];
            double[] V = new double[seqLen * numKVHeads * headDim];

            FlashAttentionDispatch.NativeFlashResult result =
                FlashAttentionDispatch.flashForward(Q, K, V, seqLen, numHeads, numKVHeads, headDim, true);

            assertNull(result, "Causal flashForward should return null cleanly when unavailable");
        }
    }
}
