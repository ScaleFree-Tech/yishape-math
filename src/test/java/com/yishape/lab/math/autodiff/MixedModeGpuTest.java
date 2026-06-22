package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.linalg.IDoubleVector;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Phase 4.3 Step 4 — validates the {@code Backend}-aware {@link MixedMode#hvp}
 * / {@link MixedMode#hessian} overloads that route the second-order backward
 * through a native backend with CPU fallback.
 *
 * <p><b>Correctness contract (backend-agnostic).</b> Every {@code prefer != CPU}
 * result is cross-checked against a central-difference numerical HVP. These
 * assertions hold no matter which tier (GPU / HPC / CPU) actually executed: if
 * a native tier executed and produced a wrong H@v, the numerical comparison
 * fails here — the silent-wrong-result failure mode called out in the Phase 4.3
 * risk register. Tests skip (not fail) when no native backend is available.
 *
 * <p><b>Observed GPU behaviour.</b> On the tape-of-tape gradient graphs built by
 * {@link AD#grad}, the GPU executor currently reports a leaf-gradient count
 * mismatch and returns {@code false} (→ HPC → CPU), so the second-order
 * backward runs on HPC or CPU and the result is correct. The GPU-prefer tests
 * therefore assert correctness via the numerical comparison; they do not assert
 * that GPU native executed. HPC second-order is exercised directly and
 * validated correct.
 *
 * <p>The CPU-default path remains covered by {@link HvpCorrectnessTest}; this
 * class only exercises the {@code prefer != CPU} overloads.
 */
public class MixedModeGpuTest {

    private static final double NUM_TOL = 1e-4;  // f32 native vs numerical

    private static boolean gpuPresent;
    private static boolean hpcPresent;

    @BeforeAll
    static void detect() {
        System.setProperty("yishape.gpu.minElements", "0");
        System.setProperty("yishape.hpc.minElements", "0");
        gpuPresent = GpuOptionalRuntime.isGpuAvailable();
        hpcPresent = HpcOptionalRuntime.isNativeRuntimeAvailable();
    }

    // ── Numerical HVP reference ───────────────────────────────────────

    private static double[] gradAt(double[] xData, Function<IDiffVector, IDiffVector> fn) {
        IDiffVector x = AD.vector(xData);
        IDiffVector y = fn.apply(x);
        IDiffVector[] grads = AD.grad(y, x);
        return grads[0].getValue().getData();
    }

    private static double[] numericalHvp(double[] x, double[] v,
                                         Function<IDiffVector, IDiffVector> fn) {
        double eps = 1e-6;
        int n = x.length;
        double[] xp = new double[n], xm = new double[n];
        for (int i = 0; i < n; i++) { xp[i] = x[i] + eps * v[i]; xm[i] = x[i] - eps * v[i]; }
        double[] gp = gradAt(xp, fn);
        double[] gm = gradAt(xm, fn);
        double[] hvp = new double[n];
        for (int i = 0; i < n; i++) hvp[i] = (gp[i] - gm[i]) / (2 * eps);
        return hvp;
    }

    private static void assertHvpMatchesNumerical(String label, double[] x, double[] v,
                                                  Function<IDiffVector, IDiffVector> fn,
                                                  AD.Backend prefer) {
        double[] analytic = MixedMode.hvp(fn, AD.vector(x), IDoubleVector.of(v), prefer);
        double[] numerical = numericalHvp(x, v, fn);
        assertArrayEquals(numerical, analytic, NUM_TOL,
            () -> label + " [" + prefer + "] HVP mismatch:\n  numerical="
                + java.util.Arrays.toString(numerical)
                + "\n  analytic  =" + java.util.Arrays.toString(analytic));
    }

    // ── HPC path (validated correct) ──────────────────────────────────

    @Test
    void hvp_hpcPref_matchesNumerical_sumExp() {
        assumeTrue(hpcPresent, "HPC not available");
        double[] x = {0.3, -0.5, 0.8, 1.1};
        double[] v = {1.0, -2.0, 0.5, 3.0};
        assertHvpMatchesNumerical("sum(exp(x))", x, v, z -> z.exp().sum(), AD.Backend.HPC);
    }

    @Test
    void hvp_hpcPref_matchesNumerical_sumCubic() {
        assumeTrue(hpcPresent, "HPC not available");
        double[] x = {0.7, 1.3, -0.9, 0.4};
        double[] v = {2.0, -1.0, 0.5, 1.5};
        assertHvpMatchesNumerical("sum(x^3)", x, v, z -> z.pow(3).sum(), AD.Backend.HPC);
    }

    @Test
    void hvp_hpcPref_matchesNumerical_dotSelf() {
        assumeTrue(hpcPresent, "HPC not available");
        double[] x = {1.0, -2.0, 3.0};
        double[] v = {0.5, 1.5, -2.0};
        assertHvpMatchesNumerical("x·x", x, v, z -> z.dot(z), AD.Backend.HPC);
    }

    @Test
    void hessian_hpcPref_matchesNumerical() {
        // Full Hessian via n native HVP calls — exercises repeated native
        // second-order backwards on the same leaf within one JVM; must not
        // accumulate residual gradients across columns.
        assumeTrue(hpcPresent, "HPC not available");
        double[] x = {0.2, -0.1, 0.4};
        Function<IDiffVector, IDiffVector> fn = z -> z.exp().sum();
        double[][] H = MixedMode.hessian(fn, AD.vector(x), AD.Backend.HPC).getData();
        // H = diag(exp(x)); each column i = H @ e_i.
        for (int i = 0; i < x.length; i++) {
            double[] ei = new double[x.length]; ei[i] = 1.0;
            double[] numerical = numericalHvp(x, ei, fn);
            for (int j = 0; j < x.length; j++) {
                assertEquals(numerical[j], H[j][i], NUM_TOL,
                    "hessian[HPC] H[" + j + "][" + i + "]");
            }
        }
    }

    // ── GPU-prefer path (backend-agnostic correctness) ───────────────

    @Test
    void hvp_gpuPref_matchesNumerical_sumExp() {
        // Whether GPU executes or falls back to HPC/CPU, the result must
        // match the numerical HVP — this is the silent-wrong-result guard.
        assumeTrue(gpuPresent || hpcPresent, "no native backend available");
        double[] x = {0.3, -0.5, 0.8, 1.1};
        double[] v = {1.0, -2.0, 0.5, 3.0};
        assertHvpMatchesNumerical("sum(exp(x))", x, v, z -> z.exp().sum(), AD.Backend.GPU);
    }

    @Test
    void hvp_gpuPref_matchesNumerical_dotSelf() {
        assumeTrue(gpuPresent || hpcPresent, "no native backend available");
        double[] x = {1.0, -2.0, 3.0};
        double[] v = {0.5, 1.5, -2.0};
        assertHvpMatchesNumerical("x·x", x, v, z -> z.dot(z), AD.Backend.GPU);
    }

    @Test
    void hessian_gpuPref_matchesNumerical() {
        assumeTrue(gpuPresent || hpcPresent, "no native backend available");
        double[] x = {0.2, -0.1, 0.4};
        Function<IDiffVector, IDiffVector> fn = z -> z.exp().sum();
        double[][] H = MixedMode.hessian(fn, AD.vector(x), AD.Backend.GPU).getData();
        for (int i = 0; i < x.length; i++) {
            double[] ei = new double[x.length]; ei[i] = 1.0;
            double[] numerical = numericalHvp(x, ei, fn);
            for (int j = 0; j < x.length; j++) {
                assertEquals(numerical[j], H[j][i], NUM_TOL,
                    "hessian[GPU] H[" + j + "][" + i + "]");
            }
        }
    }

    // ── CPU default unchanged (regression guard) ──────────────────────

    @Test
    void hvp_cpuDefault_stillCorrect() {
        // The default overload must remain pure-CPU and correct (Phase 4.3
        // Step 4 must not perturb existing callers).
        double[] x = {0.3, -0.5, 0.8, 1.1};
        double[] v = {1.0, -2.0, 0.5, 3.0};
        double[] analytic = MixedMode.hvp(z -> z.exp().sum(), AD.vector(x), IDoubleVector.of(v));
        // H@v = exp(x) ⊙ v
        double[] expected = new double[x.length];
        for (int i = 0; i < x.length; i++) expected[i] = Math.exp(x[i]) * v[i];
        assertArrayEquals(expected, analytic, 1e-9, "CPU-default hvp correctness");
    }

    @Test
    void hvp_gpuPref_matchesCpuDefault() {
        // GPU-prefer (via whatever tier executes) must agree with pure-CPU.
        double[] x = {0.3, -0.5, 0.8, 1.1};
        double[] v = {1.0, -2.0, 0.5, 3.0};
        Function<IDiffVector, IDiffVector> fn = z -> z.exp().sum();
        double[] cpu = MixedMode.hvp(fn, AD.vector(x), IDoubleVector.of(v));
        double[] viaGpu = MixedMode.hvp(fn, AD.vector(x), IDoubleVector.of(v), AD.Backend.GPU);
        assertArrayEquals(cpu, viaGpu, NUM_TOL, "CPU-default vs GPU-prefer HVP diverged");
    }
}
