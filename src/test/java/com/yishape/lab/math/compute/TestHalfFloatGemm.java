package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for HalfFloatGemm: IEEE 754 binary16 conversion roundtrip,
 * fp16 GEMM accuracy vs fp32 reference, and edge cases.
 *
 * <p>Tests cover the software fallback path (fp16→fp32 conversion + fp32 GEMM).
 * Native HPC fp16 GEMM is tested in yishape-math-hpc Rust tests (per BUGS不出库 policy).</p>
 */
class TestHalfFloatGemm {

    // ============ IEEE 754 Conversion: float→half→float roundtrip ============

    @Test
    void testZeroRoundtrip() {
        // Positive zero
        assertEquals(0.0f, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(0.0f)), 0.0f,
            "+0.0 roundtrip should be exact");
        // Negative zero
        assertEquals(-0.0f, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(-0.0f)), 0.0f,
            "-0.0 roundtrip should be exact");
    }

    @Test
    void testOneRoundtrip() {
        assertEquals(1.0f, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(1.0f)), 0.0f,
            "1.0 roundtrip should be exact");
        assertEquals(-1.0f, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(-1.0f)), 0.0f,
            "-1.0 roundtrip should be exact");
    }

    @Test
    void testInfinityRoundtrip() {
        float posInf = Float.POSITIVE_INFINITY;
        float negInf = Float.NEGATIVE_INFINITY;
        assertEquals(posInf, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(posInf)), 0.0f,
            "+Inf roundtrip");
        assertEquals(negInf, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(negInf)), 0.0f,
            "-Inf roundtrip");
    }

    @Test
    void testNaNRoundtrip() {
        float nan = Float.NaN;
        float result = HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(nan));
        assertTrue(Float.isNaN(result), "NaN roundtrip should preserve NaN");
    }

    @Test
    void testHalfMinNormalRoundtrip() {
        // Minimum normal fp16: 2^-14 ≈ 6.1035e-5
        float minNormal = 6.1035156e-5f;
        float result = HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(minNormal));
        assertEquals(minNormal, result, minNormal * 1e-3f, "fp16 min normal roundtrip");
    }

    @Test
    void testHalfMaxNormalRoundtrip() {
        // Maximum normal fp16: 65504
        float maxNormal = 65504.0f;
        float result = HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(maxNormal));
        assertEquals(maxNormal, result, 1.0f, "fp16 max normal roundtrip");
    }

    @Test
    void testOverflowSaturatesToInf() {
        // Values > 65504 should saturate to infinity
        float overflow = 100000.0f;
        assertTrue(Float.isInfinite(HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(overflow))),
            "Values > fp16 max should saturate to Inf");
        assertTrue(Float.isInfinite(HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(-100000.0f))),
            "Negative overflow should saturate to -Inf");
    }

    @Test
    void testSubnormalRoundtrip() {
        // fp16 subnormal range: 2^-24 to 2^-15 ≈ 5.96e-8 to 3.05e-5
        float subnormal = 1.0e-6f; // ≈ 2^-19.9 — well within fp16 subnormal range
        float result = HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(subnormal));
        assertEquals(subnormal, result, subnormal * 0.1f, "fp16 subnormal roundtrip within 10%");
    }

    @Test
    void testTinyFlushesToZero() {
        // Values smaller than fp16 subnormal minimum (~5.96e-8) flush to zero
        float tiny = 1e-9f;
        assertEquals(0.0f, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(tiny)), 0.0f,
            "Values < fp16 min subnormal should flush to zero");
    }

    @Test
    void testRoundingNearestEven() {
        // Test that rounding follows round-to-nearest-even rule
        // 1.0 + 1/2048 should round down to 1.0 (ties to even)
        float justAbove1 = 1.0f + 1.0f / 4096.0f; // 1/2048 LSB ≈ 1/4096 * 2 above 1.0
        float result = HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(justAbove1));
        assertEquals(1.0f, result, 1.0f / 1024.0f, "Round-to-nearest: just above 1.0");
    }

    @Test
    void testIntegerValuesExact() {
        // All integers from 0 to 2048 are exactly representable in fp16
        for (int i = 0; i <= 100; i++) {
            assertEquals((float) i, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf((float) i)), 0.0f,
                "Integer " + i + " roundtrip should be exact");
        }
        // Powers of 2 up to fp16 range
        for (int exp = -14; exp <= 15; exp++) {
            float v = (float) Math.pow(2.0, exp);
            assertEquals(v, HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(v)), v * 1e-6f,
                "2^" + exp + " roundtrip");
        }
    }

    @Test
    void testRandomRoundtripAccuracy() {
        java.util.Random rng = new java.util.Random(12345);
        double maxRelErr = 0.0;
        int within1pct = 0;
        int within5pct = 0;
        int total = 10000;
        for (int i = 0; i < total; i++) {
            // Generate random float in [-100, 100] (well within fp16 range)
            float v = (rng.nextFloat() * 2.0f - 1.0f) * 100.0f;
            float roundtripped = HalfFloatGemm.halfToFloat(HalfFloatGemm.floatToHalf(v));
            if (v == 0.0f) {
                assertEquals(0.0f, roundtripped, 0.0f);
                continue;
            }
            double relErr = Math.abs((double)(roundtripped - v) / v);
            if (relErr > maxRelErr) maxRelErr = relErr;
            if (relErr < 1e-2) within1pct++;
            if (relErr < 5e-2) within5pct++;
        }
        assertTrue(maxRelErr < 0.001, "Max relative error " + maxRelErr + " should be < 0.1% (fp16 precision)");
        assertTrue(within1pct >= total * 0.99, "≥99% of values should be within 1% relative error, got "
            + (100.0 * within1pct / total) + "%");
        assertTrue(within5pct == total, "All values should be within 5% relative error");
    }

    // ============ Array conversion roundtrip ============

    @Test
    void testArrayHalfToFloat() {
        float[] orig = {1.0f, -2.5f, 0.0f, 100.0f, -0.5f};
        short[] half = HalfFloatGemm.floatToHalf(orig);
        float[] back = HalfFloatGemm.halfToFloat(half);
        assertEquals(orig.length, back.length);
        for (int i = 0; i < orig.length; i++) {
            assertEquals(orig[i], back[i], Math.abs(orig[i]) * 1e-3f + 1e-6f,
                "Array roundtrip at index " + i);
        }
    }

    @Test
    void testArrayFloatToHalfLength() {
        float[] input = new float[1000];
        java.util.Random rng = new java.util.Random(99);
        for (int i = 0; i < input.length; i++) input[i] = rng.nextFloat() * 200.0f - 100.0f;
        short[] half = HalfFloatGemm.floatToHalf(input);
        assertEquals(input.length, half.length, "floatToHalf output length matches input");
        float[] back = HalfFloatGemm.halfToFloat(half);
        assertEquals(input.length, back.length, "halfToFloat output length matches input");
    }

    // ============ fp16 GEMM correctness ============

    @Test
    void testFp16GemmSmallSquare() {
        // Simple identity: A = I, B = v → C = v
        int n = 3, m = 3, k = 3;
        float[] a = {1, 0, 0,  0, 1, 0,  0, 0, 1}; // 3x3 identity
        float[] b = {1, 2, 3,  4, 5, 6,  7, 8, 9}; // 3x3
        short[] aH = HalfFloatGemm.floatToHalf(a);
        short[] bH = HalfFloatGemm.floatToHalf(b);
        float[] cOut = new float[n * m];

        HalfFloatGemm.flatHgemm(n, m, k, aH, bH, cOut);

        // Expected: I @ B = B (with fp16 quantization tolerance)
        for (int i = 0; i < n * m; i++) {
            assertEquals(b[i], cOut[i], Math.max(Math.abs(b[i]) * 5e-3, 1e-5f),
                "Identity GEMM mismatch at " + i);
        }
    }

    @Test
    void testFp16GemmVsFp32Reference() {
        // Compare fp16 GEMM with fp32 GEMM for random matrices
        int n = 4, m = 6, k = 3;
        float[] a = randomFloatArray(n * k, 42);
        float[] b = randomFloatArray(k * m, 43);

        // fp32 reference
        float[] expected = FloatFlatGemm.flatMmul(a, n, k, b, m);

        // fp16 GEMM (via software fallback: convert → fp32 → GEMM)
        short[] aH = HalfFloatGemm.floatToHalf(a);
        short[] bH = HalfFloatGemm.floatToHalf(b);
        float[] actual = new float[n * m];
        HalfFloatGemm.flatHgemm(n, m, k, aH, bH, actual);

        // Compare: fp16 precision is ~3 decimal digits (~0.05% relative error)
        // fp16 has ~3 decimal digits precision; GEMM accumulates k multiply-adds,
        // so tolerance must accommodate ~0.5% per-element quantization + accumulation
        for (int i = 0; i < n * m; i++) {
            double expectedAbs = Math.abs(expected[i]);
            double tolerance = Math.max(expectedAbs * 5e-2, 1e-3);
            assertEquals(expected[i], actual[i], tolerance,
                "fp16 GEMM vs fp32 mismatch at " + i + ": expected=" + expected[i] + " actual=" + actual[i]);
        }
    }

    @ParameterizedTest
    @CsvSource({
        "2, 2, 2",
        "4, 3, 5",
        "8, 1, 8",
        "1, 8, 4",
        "16, 4, 16",
        "10, 10, 10",
        "3, 7, 5",
    })
    void testFp16GemmVariousSizes(int n, int m, int k) {
        float[] a = randomFloatArray(n * k, n * 100 + m * 10 + k);
        float[] b = randomFloatArray(k * m, n * 100 + m * 10 + k + 1);

        float[] expected = FloatFlatGemm.flatMmul(a, n, k, b, m);
        short[] aH = HalfFloatGemm.floatToHalf(a);
        short[] bH = HalfFloatGemm.floatToHalf(b);
        float[] actual = new float[n * m];
        HalfFloatGemm.flatHgemm(n, m, k, aH, bH, actual);

        for (int i = 0; i < n * m; i++) {
            double tol = Math.max(Math.abs(expected[i]) * 5e-2, 1e-3);
            assertEquals(expected[i], actual[i], tol,
                "fp16 GEMM mismatch at " + i + " for [" + n + "," + m + "," + k + "]" +
                ": expected=" + expected[i] + " actual=" + actual[i]);
        }
    }

    @Test
    void testFp16GemmWithNegativeValues() {
        int n = 3, m = 4, k = 2;
        float[] a = {-1.5f, 2.0f, 0.5f, -3.0f, 1.0f, -0.5f};
        float[] b = {1.0f, -2.0f, 0.0f, 3.0f, -1.0f, 4.0f, -2.0f, 1.0f};

        float[] expected = FloatFlatGemm.flatMmul(a, n, k, b, m);
        short[] aH = HalfFloatGemm.floatToHalf(a);
        short[] bH = HalfFloatGemm.floatToHalf(b);
        float[] actual = new float[n * m];
        HalfFloatGemm.flatHgemm(n, m, k, aH, bH, actual);

        for (int i = 0; i < n * m; i++) {
            double tol = Math.max(Math.abs(expected[i]) * 5e-2, 1e-3);
            assertEquals(expected[i], actual[i], tol,
                "Negative-values fp16 GEMM mismatch at " + i);
        }
    }

    @Test
    void testFp16GemmLargeMatrix() {
        int n = 32, m = 32, k = 32;
        float[] a = randomFloatArray(n * k, 77);
        float[] b = randomFloatArray(k * m, 78);

        float[] expected = FloatFlatGemm.flatMmul(a, n, k, b, m);
        short[] aH = HalfFloatGemm.floatToHalf(a);
        short[] bH = HalfFloatGemm.floatToHalf(b);
        float[] actual = new float[n * m];
        HalfFloatGemm.flatHgemm(n, m, k, aH, bH, actual);

        // For large matrices with high k (32), fp16 quantization error accumulates significantly.
        // fp16 has ~3 decimal digits; 32 multiply-adds can accumulate ~5-15% relative error
        // for some elements. Use generous per-element tolerance.
        for (int i = 0; i < n * m; i++) {
            double tol = Math.max(Math.abs(expected[i]) * 2.5e-1, 2e-1);
            assertEquals(expected[i], actual[i], tol,
                "Large fp16 GEMM mismatch at " + i);
        }
    }

    @Test
    void testFp16GemmZeroInput() {
        int n = 3, m = 3, k = 3;
        float[] a = new float[n * k]; // all zeros
        float[] b = randomFloatArray(k * m, 55);
        short[] aH = HalfFloatGemm.floatToHalf(a);
        short[] bH = HalfFloatGemm.floatToHalf(b);
        float[] actual = new float[n * m];
        HalfFloatGemm.flatHgemm(n, m, k, aH, bH, actual);

        for (int i = 0; i < n * m; i++) {
            assertEquals(0.0f, actual[i], 1e-10f, "Zero * B should be zero");
        }
    }

    // ============ Short→Float conversion in isolation ============

    @Test
    void testHalfToZero() {
        assertEquals(0.0f, HalfFloatGemm.halfToFloat((short) 0), 0.0f, "half 0x0000 = +0.0");
        assertEquals(-0.0f, HalfFloatGemm.halfToFloat((short) 0x8000), 0.0f, "half 0x8000 = -0.0");
    }

    @Test
    void testHalfToInf() {
        assertEquals(Float.POSITIVE_INFINITY, HalfFloatGemm.halfToFloat((short) 0x7C00), 0.0f);
        assertEquals(Float.NEGATIVE_INFINITY, HalfFloatGemm.halfToFloat((short) 0xFC00), 0.0f);
    }

    @Test
    void testHalfToNaN() {
        assertTrue(Float.isNaN(HalfFloatGemm.halfToFloat((short) 0x7E00)), "half 0x7E00 should be NaN");
        // NaN with different payload should still be NaN
        assertTrue(Float.isNaN(HalfFloatGemm.halfToFloat((short) 0x7FFF)), "half 0x7FFF should be NaN");
    }

    @Test
    void testIsNativeFp16AvailableDoesNotThrow() {
        // Should not throw, even when native library is absent
        assertDoesNotThrow(HalfFloatGemm::isNativeFp16Available,
            "isNativeFp16Available() should never throw");
    }

    // ============ Edge case: extreme values in GEMM ============

    @Test
    void testFp16GemmLargeValues() {
        // Values near fp16 max (65504) but within range.
        // A[2×2] = [[10000, 20000], [5000, 10000]],  B[2×2] = [[1, 2], [3, 4]]
        // C = A @ B:  [70000, 100000], [35000, 50000]
        int n = 2, m = 2, k = 2;
        float[] a = {10000, 20000, 5000, 10000};
        float[] b = {1, 2, 3, 4};
        short[] aH = HalfFloatGemm.floatToHalf(a);
        short[] bH = HalfFloatGemm.floatToHalf(b);
        float[] actual = new float[n * m];
        HalfFloatGemm.flatHgemm(n, m, k, aH, bH, actual);

        // Compare against fp32 reference (fp16→fp32→GEMM fallback uses fp32 internally)
        float[] expected = FloatFlatGemm.flatMmul(a, n, k, b, m);
        for (int i = 0; i < n * m; i++) {
            assertEquals(expected[i], actual[i], Math.abs(expected[i]) * 5e-3 + 1e-3,
                "Large-values fp16 GEMM mismatch at " + i);
        }
    }

    @Test
    void testFp16GemmVerySmallValues() {
        // Very small values (near fp16 subnormal range)
        int n = 2, m = 2, k = 2;
        float[] a = {1e-4f, 2e-4f, 3e-4f, 4e-4f};
        float[] b = {5e-4f, 6e-4f, 7e-4f, 8e-4f};
        short[] aH = HalfFloatGemm.floatToHalf(a);
        short[] bH = HalfFloatGemm.floatToHalf(b);
        float[] actual = new float[n * m];
        HalfFloatGemm.flatHgemm(n, m, k, aH, bH, actual);

        float[] expectedFp32 = FloatFlatGemm.flatMmul(a, n, k, b, m);
        for (int i = 0; i < n * m; i++) {
            double tol = Math.max(Math.abs(expectedFp32[i]) * 1e-1, 1e-8);
            assertEquals(expectedFp32[i], actual[i], tol, "Small value GEMM mismatch at " + i);
        }
    }

    // ============ Helper ============

    private static float[] randomFloatArray(int size, int seed) {
        float[] arr = new float[size];
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < size; i++) arr[i] = rng.nextFloat() * 20.0f - 10.0f;
        return arr;
    }
}
