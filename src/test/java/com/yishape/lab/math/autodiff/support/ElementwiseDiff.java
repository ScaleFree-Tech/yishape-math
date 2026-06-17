package com.yishape.lab.math.autodiff.support;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

/**
 * Element-wise comparison of two double arrays with detailed diagnostic reporting.
 *
 * <p>Designed for cross-backend consistency testing: compares results computed
 * by different backends (CPU, HPC, GPU) and produces human-readable reports
 * highlighting the largest discrepancies.
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * ElementwiseDiff diff = ElementwiseDiff.compare(cpuResult, gpuResult,
 *     ToleranceClass.forOp("mmul", true));
 * assertTrue(diff.passes(),
 *     () -> diff.diagnosticReport());
 * }</pre>
 *
 * @since 0.9.0
 */
public final class ElementwiseDiff {

    /** Reference (expected) values. */
    public final double[] ref;

    /** Test (actual) values. */
    public final double[] test;

    /** Per-element absolute difference. */
    public final double[] absDiff;

    /** Per-element relative difference (relative to max(|ref|,|test|)). */
    public final double[] relDiff;

    /** Maximum absolute difference over all elements. */
    public final double maxAbsDiff;

    /** Maximum relative difference over all elements. */
    public final double maxRelDiff;

    /** Index of maximum absolute difference. */
    public final int maxAbsDiffIndex;

    /** Index of maximum relative difference. */
    public final int maxRelDiffIndex;

    /** Tolerance used for comparison. */
    public final ToleranceClass tolerance;

    /** Number of elements passing the tolerance check. */
    public final long passingCount;

    private ElementwiseDiff(double[] ref, double[] test, ToleranceClass tolerance) {
        int n = ref.length;
        this.ref = ref;
        this.test = test;
        this.tolerance = tolerance;
        this.absDiff = new double[n];
        this.relDiff = new double[n];

        double maxAbs = 0, maxRel = 0;
        int maxAbsIdx = 0, maxRelIdx = 0;
        long passing = 0;

        for (int i = 0; i < n; i++) {
            absDiff[i] = Math.abs(ref[i] - test[i]);
            double denom = Math.max(Math.abs(ref[i]), Math.abs(test[i]));
            relDiff[i] = (denom > 1e-300) ? absDiff[i] / denom : absDiff[i];

            if (absDiff[i] > maxAbs) {
                maxAbs = absDiff[i];
                maxAbsIdx = i;
            }
            if (relDiff[i] > maxRel) {
                maxRel = relDiff[i];
                maxRelIdx = i;
            }

            if (tolerance.within(ref[i], test[i])) {
                passing++;
            }
        }

        this.maxAbsDiff = maxAbs;
        this.maxRelDiff = maxRel;
        this.maxAbsDiffIndex = maxAbsIdx;
        this.maxRelDiffIndex = maxRelIdx;
        this.passingCount = passing;
    }

    /**
     * Compare two arrays element-by-element.
     *
     * @param ref       reference (expected) array
     * @param test      test (actual) array
     * @param tolerance tolerance class
     * @return element-wise difference report
     * @throws IllegalArgumentException if arrays have different lengths
     */
    public static ElementwiseDiff compare(double[] ref, double[] test,
                                           ToleranceClass tolerance) {
        if (ref.length != test.length) {
            throw new IllegalArgumentException(
                "Array length mismatch: ref=" + ref.length + " test=" + test.length);
        }
        return new ElementwiseDiff(ref.clone(), test.clone(), tolerance);
    }

    /**
     * Whether all elements are within tolerance.
     */
    public boolean passes() {
        return passingCount == absDiff.length;
    }

    /**
     * Number of elements failing the tolerance check.
     */
    public long failingCount() {
        return absDiff.length - passingCount;
    }

    /**
     * Mean absolute difference.
     */
    public double meanAbsDiff() {
        return Arrays.stream(absDiff).average().orElse(0);
    }

    /**
     * Median absolute difference.
     */
    public double medianAbsDiff() {
        double[] sorted = absDiff.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        if (n % 2 == 0) {
            return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
        }
        return sorted[n / 2];
    }

    /**
     * Returns indices of the {@code k} largest absolute differences.
     */
    public int[] topKIndices(int k) {
        return IntStream.range(0, absDiff.length)
            .boxed()
            .sorted(Comparator.<Integer>comparingDouble(i -> absDiff[i]).reversed())
            .limit(k)
            .mapToInt(Integer::intValue)
            .toArray();
    }

    /**
     * Builds a multi-line diagnostic report suitable for assertion failure messages.
     * Includes statistics, top-20 largest differences, and pass/fail summary.
     */
    public String diagnosticReport() {
        return diagnosticReport(20);
    }

    /**
     * Builds a diagnostic report showing the top {@code topK} largest differences.
     */
    public String diagnosticReport(int topK) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
            "\n═══════════════════ Cross-Backend Consistency Failure ═══════════════════\n"));
        sb.append(String.format("Tolerance class : %s (abs=%.1e, rel=%.1e)\n",
            tolerance.name(), tolerance.absTol, tolerance.relTol));
        sb.append(String.format("Array length    : %d\n", absDiff.length));
        sb.append(String.format("Passing         : %d / %d (%.1f%%)\n",
            passingCount, absDiff.length,
            100.0 * passingCount / absDiff.length));
        sb.append(String.format("Max abs diff    : %.2e at index %d\n",
            maxAbsDiff, maxAbsDiffIndex));
        sb.append(String.format("                  ref[%d]=%.12f  test[%d]=%.12f\n",
            maxAbsDiffIndex, ref[maxAbsDiffIndex],
            maxAbsDiffIndex, test[maxAbsDiffIndex]));
        sb.append(String.format("Max rel diff    : %.2e at index %d\n",
            maxRelDiff, maxRelDiffIndex));
        sb.append(String.format("                  ref[%d]=%.12f  test[%d]=%.12f\n",
            maxRelDiffIndex, ref[maxRelDiffIndex],
            maxRelDiffIndex, test[maxRelDiffIndex]));
        sb.append(String.format("Mean abs diff   : %.2e\n", meanAbsDiff()));
        sb.append(String.format("Median abs diff : %.2e\n", medianAbsDiff()));

        int[] top = topKIndices(topK);
        sb.append(String.format("\nTop-%d largest absolute differences:\n", Math.min(topK, top.length)));
        sb.append(String.format("  %6s  %-16s  %-16s  %-12s  %-12s\n",
            "Index", "Ref", "Test", "AbsDiff", "RelDiff"));
        for (int idx : top) {
            sb.append(String.format("  %6d  %16.8e  %16.8e  %12.2e  %12.2e\n",
                idx, ref[idx], test[idx], absDiff[idx], relDiff[idx]));
        }

        sb.append(String.format(
            "══════════════════════════════════════════════════════════════════════════════\n"));
        return sb.toString();
    }

    /**
     * Returns a single-line summary: {@code "maxAbs=1.2e-5 at idx=42 (42/100 fail)"}.
     */
    public String summary() {
        return String.format("maxAbs=%.2e at idx=%d (%d/%d fail, %.1f%%)",
            maxAbsDiff, maxAbsDiffIndex,
            failingCount(), absDiff.length,
            100.0 * failingCount() / absDiff.length);
    }

    @Override
    public String toString() {
        return summary();
    }
}
