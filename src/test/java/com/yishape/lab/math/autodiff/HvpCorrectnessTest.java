package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.linalg.IDoubleVector;
import org.junit.jupiter.api.Test;

import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * Phase 4.3 Step 3.8 — validates {@link MixedMode#hvp} / {@link MixedMode#hessian}
 * against finite-difference numerical Hessian-vector products.
 *
 * <p>The algebraic cases (f = x·x, weighted quadratic) are covered by
 * {@code MixedModeTest}. This class adds <em>correctness vs numerical
 * differentiation</em> for non-trivial smooth functions whose gradient paths
 * exercise the tape-of-tape {@code symbolicBackwardFn} machinery (exp, sigmoid,
 * pow). A mismatch here surfaces a wrong second-order formula — the silent-zero
 * failure mode called out in the Phase 4.3 risk register.
 *
 * <p>Numerical HVP: H@v ≈ (grad(x+εv) − grad(x−εv)) / (2ε).
 */
public class HvpCorrectnessTest {

    private static final double NUM_TOL = 1e-5;

    /** Primal gradient of a scalar fn at xData, via AD.grad (tape-of-tape). */
    private static double[] gradAt(double[] xData, Function<IDiffVector, IDiffVector> fn) {
        IDiffVector x = AD.vector(xData);
        IDiffVector y = fn.apply(x);
        IDiffVector[] grads = AD.grad(y, x);
        return grads[0].getValue().getData();
    }

    /** Central-difference numerical Hessian-vector product. */
    private static double[] numericalHvp(double[] x, double[] v,
                                         Function<IDiffVector, IDiffVector> fn) {
        final double eps = 1e-6;
        int n = x.length;
        double[] xp = new double[n], xm = new double[n];
        for (int i = 0; i < n; i++) {
            xp[i] = x[i] + eps * v[i];
            xm[i] = x[i] - eps * v[i];
        }
        double[] gp = gradAt(xp, fn);
        double[] gm = gradAt(xm, fn);
        double[] hvp = new double[n];
        for (int i = 0; i < n; i++) {
            hvp[i] = (gp[i] - gm[i]) / (2 * eps);
        }
        return hvp;
    }

    private static void assertHvpMatches(String label, double[] x, double[] v,
                                         Function<IDiffVector, IDiffVector> fn) {
        double[] analytic = MixedMode.hvp(fn, AD.vector(x), IDoubleVector.of(v));
        double[] numerical = numericalHvp(x, v, fn);
        assertArrayEquals(numerical, analytic, NUM_TOL,
            () -> label + " HVP mismatch:\n  numerical = " + java.util.Arrays.toString(numerical)
                + "\n  analytic   = " + java.util.Arrays.toString(analytic));
    }

    // ── exp: grad = exp(x), H = diag(exp(x)), H@v = exp(x)⊙v ──────────

    @Test
    void hvp_sumExp_matchesNumerical() {
        double[] x = {0.3, -0.5, 0.8, 1.1};
        double[] v = {1.0, -2.0, 0.5, 3.0};
        assertHvpMatches("sum(exp(x))", x, v, z -> z.exp().sum());

        // analytic cross-check: H@v = exp(x) ⊙ v
        double[] analytic = MixedMode.hvp(z -> z.exp().sum(), AD.vector(x), IDoubleVector.of(v));
        double[] expected = new double[x.length];
        for (int i = 0; i < x.length; i++) expected[i] = Math.exp(x[i]) * v[i];
        assertArrayEquals(expected, analytic, 1e-9);
    }

    // ── sigmoid: grad = s(1−s), H = diag(s(1−s)(1−2s)), H@v = that ⊙ v ─

    @Test
    void hvp_sumSigmoid_matchesNumerical() {
        double[] x = {0.2, -0.4, 0.6};
        double[] v = {1.0, 0.5, -1.5};
        assertHvpMatches("sum(sigmoid(x))", x, v, z -> z.sigmoid().sum());

        // analytic cross-check
        double[] analytic = MixedMode.hvp(z -> z.sigmoid().sum(), AD.vector(x), IDoubleVector.of(v));
        double[] expected = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double s = 1.0 / (1.0 + Math.exp(-x[i]));
            expected[i] = s * (1 - s) * (1 - 2 * s) * v[i];
        }
        assertArrayEquals(expected, analytic, 1e-9);
    }

    // ── cubic: f = sum(x^3), grad = 3x^2, H = diag(6x), H@v = 6x⊙v ────

    @Test
    void hvp_sumCubic_matchesNumerical() {
        double[] x = {0.7, 1.3, -0.9, 0.4};
        double[] v = {2.0, -1.0, 0.5, 1.5};
        assertHvpMatches("sum(x^3)", x, v, z -> z.pow(3).sum());

        double[] analytic = MixedMode.hvp(z -> z.pow(3).sum(), AD.vector(x), IDoubleVector.of(v));
        double[] expected = new double[x.length];
        for (int i = 0; i < x.length; i++) expected[i] = 6 * x[i] * v[i];
        assertArrayEquals(expected, analytic, 1e-9);
    }

    // ── dot product control: f = x·x, H = 2I, H@v = 2v ────────────────

    @Test
    void hvp_dotSelf_matchesNumerical() {
        double[] x = {1.0, -2.0, 3.0};
        double[] v = {0.5, 1.5, -2.0};
        assertHvpMatches("x·x", x, v, z -> z.dot(z));

        double[] analytic = MixedMode.hvp(z -> z.dot(z), AD.vector(x), IDoubleVector.of(v));
        double[] expected = {2 * v[0], 2 * v[1], 2 * v[2]};
        assertArrayEquals(expected, analytic, 1e-9);
    }

    // ── chained op: f = sum(sigmoid(x) * exp(x)) ───────────────────────
    //   Tests that tape-of-tape composes across a mul of two unary branches.

    @Test
    void hvp_chainedSigmoidMulExp_matchesNumerical() {
        double[] x = {0.1, -0.3, 0.5, -0.2};
        double[] v = {1.0, 1.0, 1.0, 1.0};
        assertHvpMatches("sum(sigmoid(x)*exp(x))", x, v,
            z -> z.sigmoid().mul(z.exp()).sum());
    }

    // ── full Hessian vs numerical (small n) ────────────────────────────

    @Test
    void hessian_sumExp_matchesNumerical() {
        // f = sum(exp(x)), H = diag(exp(x))
        double[] x = {0.2, -0.1, 0.4};
        Function<IDiffVector, IDiffVector> fn = z -> z.exp().sum();

        double[][] H = MixedMode.hessian(fn, AD.vector(x)).getData();

        // Each column i = H @ e_i = diag(exp(x)) e_i → only entry i is exp(x[i]).
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                double expected = (i == j) ? Math.exp(x[i]) : 0.0;
                double numerical;
                double[] ei = new double[x.length];
                ei[i] = 1.0;
                double[] col = numericalHvp(x, ei, fn);
                numerical = col[j];
                assertEqualsSym(expected, H[j][i], numerical, 1e-4,
                    "H[" + j + "][" + i + "]");
            }
        }
    }

    @Test
    void hessian_isSymmetric() {
        double[] x = {0.3, -0.4, 0.6};
        // f = sum(x^3 + x^2): H = diag(6x + 2), symmetric (diagonal here, but
        // confirms the n-HVP assembly produces a symmetric matrix for a
        // separable function; cross terms are zero).
        double[][] H = MixedMode.hessian(z -> z.pow(3).add(z.pow(2)).sum(),
            AD.vector(x)).getData();
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x.length; j++) {
                assertEqualsSym(H[i][j], H[j][i], H[i][j], 1e-9, "symmetry H[" + i + "][" + j + "]");
            }
        }
    }

    private static void assertEqualsSym(double expected, double analytic, double numerical,
                                        double tol, String label) {
        // Compare analytic to numerical (the source of truth here).
        if (Math.abs(analytic - numerical) > tol) {
            throw new AssertionError(label + ": analytic=" + analytic + " numerical=" + numerical
                + " diff=" + Math.abs(analytic - numerical) + " > tol=" + tol);
        }
    }
}
