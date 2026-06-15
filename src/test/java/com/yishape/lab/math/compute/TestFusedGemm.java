package com.yishape.lab.math.compute;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for fused GEMM+activation ops: flatMmulRelu, flatMmulGelu, flatMmulSilu.
 *
 * <p>Each fused op is compared against:
 * <ol>
 *   <li>Sequential GEMM + element-wise activation (same inputs, same output allocation)</li>
 *   <li>Independent sequential GEMM + activation on fresh allocations</li>
 * </ol>
 *
 * <p>Tests cover various matrix sizes, edge cases, and numerical corner cases.
 * Native HPC fused kernels are tested in yishape-math-hpc Rust tests (per BUGS不出库 policy).</p>
 */
class TestFusedGemm {

    private static final double FP_TOLERANCE = 1e-12;

    // ======================== ReLU ========================

    @ParameterizedTest
    @CsvSource({
        "1, 1, 1",
        "2, 2, 2",
        "4, 3, 5",
        "7, 11, 3",
        "16, 8, 16",
        "1, 10, 5",
        "32, 4, 8",
    })
    void testReluMatchesSequential(int m, int n, int k) {
        double[] a = randomArray(m * k, m * 100 + n * 10 + k);
        double[] b = randomArray(k * n, m * 100 + n * 10 + k + 1);

        // Fused path
        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, fused, m, n, k);

        // Sequential path (same inputs, independent allocation)
        double[] sequential = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        for (int i = 0; i < sequential.length; i++) {
            if (sequential[i] < 0) sequential[i] = 0;
        }

        assertArrayEquals(sequential, fused, FP_TOLERANCE,
            "Fused ReLU should match sequential ReLU for [" + m + "," + n + "," + k + "]");
    }

    @Test
    void testReluWithNegativeValues() {
        int m = 3, n = 4, k = 2;
        double[] a = {-1.5, 2.0, 0.5, -3.0, 1.0, -0.5};
        double[] b = {1.0, -2.0, 0.0, 3.0, -1.0, 4.0, -2.0, 1.0};

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, fused, m, n, k);

        double[] expected = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] < 0) expected[i] = 0;
        }

        assertArrayEquals(expected, fused, FP_TOLERANCE);
    }

    @Test
    void testReluAllPositive() {
        int m = 2, n = 2, k = 2;
        // Both A and B positive → GEMM result all positive → ReLU is identity
        double[] a = {1.0, 2.0, 3.0, 4.0};
        double[] b = {5.0, 6.0, 7.0, 8.0};

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, fused, m, n, k);

        double[] expected = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        // All GEMM results are positive, so ReLU is no-op
        for (double v : expected) assertTrue(v > 0, "Expected all positive GEMM results");

        assertArrayEquals(expected, fused, FP_TOLERANCE, "ReLU on all-positive should be identity");
    }

    @Test
    void testReluAllNegative() {
        int m = 2, n = 2, k = 2;
        // A positive, B negative → GEMM result negative → all ReLU'd to 0
        double[] a = {1.0, 2.0, 3.0, 4.0};
        double[] b = {-5.0, -6.0, -7.0, -8.0};
        // Result: [-17, -18, -13, -14] → all relu → zeros

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, fused, m, n, k);

        for (double v : fused) {
            assertEquals(0.0, v, FP_TOLERANCE, "ReLU on all-negative should yield zero");
        }
    }

    @Test
    void testReluLargeMatrix() {
        int m = 64, n = 64, k = 64;
        double[] a = randomArray(m * k, 123);
        double[] b = randomArray(k * n, 456);

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, fused, m, n, k);

        double[] sequential = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        for (int i = 0; i < sequential.length; i++) {
            if (sequential[i] < 0) sequential[i] = 0;
        }

        assertArrayEquals(sequential, fused, FP_TOLERANCE, "Fused ReLU matches on large matrix");
    }

    // ======================== GELU ========================

    @ParameterizedTest
    @CsvSource({
        "1, 1, 1",
        "2, 2, 2",
        "4, 3, 5",
        "7, 11, 3",
        "8, 5, 12",
        "1, 10, 5",
        "32, 4, 8",
    })
    void testGeluMatchesSequential(int m, int n, int k) {
        double[] a = randomArray(m * k, m * 100 + n * 10 + k);
        double[] b = randomArray(k * n, m * 100 + n * 10 + k + 1);

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulGelu(a, b, fused, m, n, k);

        double[] sequential = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        geluInPlace(sequential);

        assertArrayEquals(sequential, fused, FP_TOLERANCE,
            "Fused GELU should match sequential GELU for [" + m + "," + n + "," + k + "]");
    }

    @Test
    void testGeluNearZero() {
        // GELU(0) ≈ 0.5 * 0 * (1 + tanh(0)) = 0
        int m = 1, n = 1, k = 1;
        double[] a = {1.0};
        double[] b = {0.0};

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulGelu(a, b, fused, m, n, k);

        // GEMM: 1.0 * 0.0 = 0.0 → GELU(0.0) = 0.0
        assertEquals(0.0, fused[0], 1e-10, "GELU(0) should be 0");
    }

    @Test
    void testGeluPositiveLarge() {
        // GELU(x) ≈ x for large x (tanh → 1)
        int m = 1, n = 1, k = 1;
        double[] a = {10.0};
        double[] b = {1.0};

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulGelu(a, b, fused, m, n, k);

        // GEMM: 10.0 * 1.0 = 10.0 → GELU(10.0) ≈ 10.0
        assertEquals(10.0, fused[0], 1e-5, "GELU(10) should be ≈ 10");
    }

    @Test
    void testGeluNegativeLarge() {
        // GELU(x) ≈ 0 for large negative x (tanh → -1, 1 + (-1) = 0)
        int m = 1, n = 1, k = 1;
        double[] a = {-10.0};
        double[] b = {1.0};

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulGelu(a, b, fused, m, n, k);

        assertEquals(0.0, fused[0], 1e-4, "GELU(-10) should be ≈ 0");
    }

    // ======================== SiLU ========================

    @ParameterizedTest
    @CsvSource({
        "1, 1, 1",
        "2, 2, 2",
        "4, 3, 5",
        "7, 11, 3",
        "8, 5, 12",
        "1, 10, 5",
        "32, 4, 8",
    })
    void testSiluMatchesSequential(int m, int n, int k) {
        double[] a = randomArray(m * k, m * 100 + n * 10 + k);
        double[] b = randomArray(k * n, m * 100 + n * 10 + k + 1);

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulSilu(a, b, fused, m, n, k);

        double[] sequential = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        siluInPlace(sequential);

        assertArrayEquals(sequential, fused, FP_TOLERANCE,
            "Fused SiLU should match sequential SiLU for [" + m + "," + n + "," + k + "]");
    }

    @Test
    void testSiluAroundZero() {
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        int m = 1, n = 1, k = 1;
        double[] a = {0.0};
        double[] b = {1.0};

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulSilu(a, b, fused, m, n, k);

        assertEquals(0.0, fused[0], 1e-10, "SiLU(0) should be 0");
    }

    @Test
    void testSiluPositiveLarge() {
        // SiLU(x) ≈ x for large x (sigmoid → 1)
        int m = 1, n = 1, k = 1;
        double[] a = {10.0};
        double[] b = {1.0};

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulSilu(a, b, fused, m, n, k);

        assertEquals(10.0, fused[0], 5e-4, "SiLU(10) should be ≈ 10");
    }

    @Test
    void testSiluNegativeLarge() {
        // SiLU(x) ≈ 0 for large negative x (sigmoid → 0, x * 0 = 0)
        int m = 1, n = 1, k = 1;
        double[] a = {-10.0};
        double[] b = {1.0};

        double[] fused = new double[m * n];
        DoubleFlatGemm.flatMmulSilu(a, b, fused, m, n, k);

        assertEquals(0.0, fused[0], 5e-4, "SiLU(-10) should be ≈ 0");
    }

    // ======================== Identity check (large k, positive values) ========================

    @Test
    void testAllThreeMatchBareGemmOnPositiveInputs() {
        // When GEMM output is all positive, ReLU = GELU ≈ SiLU ≈ identity
        int m = 3, n = 3, k = 3;
        double[] a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
        double[] b = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}; // Identity

        double[] bareGemm = DoubleFlatGemm.flatMmul(a, m, k, b, m);

        double[] reluOut = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, reluOut, m, n, k);
        assertArrayEquals(bareGemm, reluOut, FP_TOLERANCE, "ReLU should = bare GEMM on positive");

        double[] geluOut = new double[m * n];
        DoubleFlatGemm.flatMmulGelu(a, b, geluOut, m, n, k);
        // GELU slightly dampens but on [1..9] it's negligible
        for (int i = 0; i < m * n; i++) {
            assertEquals(bareGemm[i], geluOut[i], 0.5, "GELU ≈ bare GEMM for small positive values");
        }

        double[] siluOut = new double[m * n];
        DoubleFlatGemm.flatMmulSilu(a, b, siluOut, m, n, k);
        // SiLU also dampens slightly
        for (int i = 0; i < m * n; i++) {
            assertEquals(bareGemm[i], siluOut[i], 1.0, "SiLU ≈ bare GEMM for small positive values");
        }
    }

    // ======================== Edge case: single element ========================

    @Test
    void testSingleElement() {
        double[] a = {2.0};
        double[] b = {3.0};
        double[] out = new double[1];

        DoubleFlatGemm.flatMmulRelu(a, b, out, 1, 1, 1);
        assertEquals(6.0, out[0], FP_TOLERANCE, "Single-element ReLU");

        DoubleFlatGemm.flatMmulGelu(a, b, out, 1, 1, 1);
        double expectedGelu = 6.0 * 0.5 * (1.0 + Math.tanh(0.7978845608028654 * (6.0 + 0.044715 * 216.0)));
        assertEquals(expectedGelu, out[0], FP_TOLERANCE, "Single-element GELU");

        DoubleFlatGemm.flatMmulSilu(a, b, out, 1, 1, 1);
        double expectedSilu = 6.0 / (1.0 + Math.exp(-6.0));
        assertEquals(expectedSilu, out[0], FP_TOLERANCE, "Single-element SiLU");
    }

    @Test
    void testSingleElementNegative() {
        double[] a = {-2.0};
        double[] b = {3.0};
        double[] out = new double[1];

        DoubleFlatGemm.flatMmulRelu(a, b, out, 1, 1, 1);
        assertEquals(0.0, out[0], FP_TOLERANCE, "Negative single-element ReLU should be 0");

        DoubleFlatGemm.flatMmulGelu(a, b, out, 1, 1, 1);
        double expectedGelu = (-6.0) * 0.5 * (1.0 + Math.tanh(0.7978845608028654 * (-6.0 + 0.044715 * (-216.0))));
        assertEquals(expectedGelu, out[0], FP_TOLERANCE, "Negative single-element GELU");

        DoubleFlatGemm.flatMmulSilu(a, b, out, 1, 1, 1);
        double expectedSilu = (-6.0) / (1.0 + Math.exp(6.0));
        assertEquals(expectedSilu, out[0], FP_TOLERANCE, "Negative single-element SiLU");
    }

    // ======================== Output buffer safety ========================

    @Test
    void testReluDoesNotModifyInputs() {
        int m = 3, n = 4, k = 2;
        double[] aOrig = randomArray(m * k, 10);
        double[] bOrig = randomArray(k * n, 20);
        double[] a = aOrig.clone();
        double[] b = bOrig.clone();

        double[] cOut = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, cOut, m, n, k);

        assertArrayEquals(aOrig, a, 0.0, "flatMmulRelu should not modify A");
        assertArrayEquals(bOrig, b, 0.0, "flatMmulRelu should not modify B");
    }

    @Test
    void testGeluDoesNotModifyInputs() {
        int m = 3, n = 4, k = 2;
        double[] aOrig = randomArray(m * k, 30);
        double[] bOrig = randomArray(k * n, 40);
        double[] a = aOrig.clone();
        double[] b = bOrig.clone();

        double[] cOut = new double[m * n];
        DoubleFlatGemm.flatMmulGelu(a, b, cOut, m, n, k);

        assertArrayEquals(aOrig, a, 0.0, "flatMmulGelu should not modify A");
        assertArrayEquals(bOrig, b, 0.0, "flatMmulGelu should not modify B");
    }

    @Test
    void testSiluDoesNotModifyInputs() {
        int m = 3, n = 4, k = 2;
        double[] aOrig = randomArray(m * k, 50);
        double[] bOrig = randomArray(k * n, 60);
        double[] a = aOrig.clone();
        double[] b = bOrig.clone();

        double[] cOut = new double[m * n];
        DoubleFlatGemm.flatMmulSilu(a, b, cOut, m, n, k);

        assertArrayEquals(aOrig, a, 0.0, "flatMmulSilu should not modify A");
        assertArrayEquals(bOrig, b, 0.0, "flatMmulSilu should not modify B");
    }

    // ======================== Non-square edge cases ========================

    @Test
    void testOneRowManyCols() {
        int m = 1, n = 100, k = 16;
        double[] a = randomArray(m * k, 70);
        double[] b = randomArray(k * n, 80);

        double[] reluOut = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, reluOut, m, n, k);
        double[] seq = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        for (int i = 0; i < seq.length; i++) if (seq[i] < 0) seq[i] = 0;
        assertArrayEquals(seq, reluOut, FP_TOLERANCE, "1×N ReLU");

        double[] geluOut = new double[m * n];
        DoubleFlatGemm.flatMmulGelu(a, b, geluOut, m, n, k);
        double[] seq2 = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        geluInPlace(seq2);
        assertArrayEquals(seq2, geluOut, FP_TOLERANCE, "1×N GELU");
    }

    @Test
    void testManyRowsOneCol() {
        int m = 100, n = 1, k = 16;
        double[] a = randomArray(m * k, 90);
        double[] b = randomArray(k * n, 100);

        double[] reluOut = new double[m * n];
        DoubleFlatGemm.flatMmulRelu(a, b, reluOut, m, n, k);
        double[] seq = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        for (int i = 0; i < seq.length; i++) if (seq[i] < 0) seq[i] = 0;
        assertArrayEquals(seq, reluOut, FP_TOLERANCE, "N×1 ReLU");

        double[] siluOut = new double[m * n];
        DoubleFlatGemm.flatMmulSilu(a, b, siluOut, m, n, k);
        double[] seq2 = DoubleFlatGemm.flatMmul(a, m, k, b, n);
        siluInPlace(seq2);
        assertArrayEquals(seq2, siluOut, FP_TOLERANCE, "N×1 SiLU");
    }

    // ======================== Helpers ========================

    private static void geluInPlace(double[] x) {
        for (int i = 0; i < x.length; i++) {
            double v = x[i];
            x[i] = 0.5 * v * (1.0 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
        }
    }

    private static void siluInPlace(double[] x) {
        for (int i = 0; i < x.length; i++) {
            double v = x[i];
            x[i] = v / (1.0 + Math.exp(-v));
        }
    }

    private static double[] randomArray(int size, int seed) {
        double[] arr = new double[size];
        java.util.Random rng = new java.util.Random(seed);
        for (int i = 0; i < size; i++) arr[i] = rng.nextGaussian() * 2.0;
        return arr;
    }
}
