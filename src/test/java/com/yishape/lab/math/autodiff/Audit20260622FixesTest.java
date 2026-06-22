package com.yishape.lab.math.autodiff;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;

/**
 * Regression tests locking in the fixes for the 2026-06-22 deep audit findings.
 *
 * <p>Each test targets one confirmed bug. If a test fails, the corresponding
 * fix has been reverted or a related code path has regressed.</p>
 */
public class Audit20260622FixesTest {

    // ==================== C1: var/std diff path must match non-diff (÷(n-1)) ====================

    /** Before fix: diff var() used ÷n while non-diff RereDoubleTensor.var() used ÷(n-1). */
    @Test
    public void testVarStdDiffMatchesNonDiff() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0};
        // 2D shape [1,5] so reducing dim 1 yields [1] (avoids 0-dim scalar).
        // Non-diff reference (RereDoubleTensor.var/std — unbiased, ÷(n-1))
        RereDoubleTensor ref = new RereDoubleTensor(data.clone(), 1, 5);
        double refVar = ref.var(1, false).toDoubleArray()[0];
        double refStd = ref.std(1, false).toDoubleArray()[0];

        // Diff path (requiresGrad=true → DiffTensorReduce.var)
        IDiffTensor x = AD.tensor(data, 1, 5);
        double diffVar = x.var(1, false).toDoubleArray()[0];
        double diffStd = x.std(1, false).toDoubleArray()[0];

        assertEquals(refVar, diffVar, 1e-12, "diff var() must match non-diff (÷n-1)");
        assertEquals(refStd, diffStd, 1e-12, "diff std() must match non-diff (÷n-1)");
        // Sanity: unbiased sample variance of {1..5} = 2.5
        assertEquals(2.5, diffVar, 1e-12, "sample variance sanity check");
    }

    // ==================== C4: log backward sign/consistency ====================

    /** Before fix: backward used g/|x| (always positive); correct is g/x (sign preserved). */
    @Test
    public void testLogGradientValue() {
        IDiffVector x = AD.vector(0.5, 1.0, 2.0);
        x.log().sum().backward();
        double[] g = x.getGradient().getData();
        assertEquals(1.0 / 0.5, g[0], 1e-10, "d log(x)/dx = 1/x");
        assertEquals(1.0 / 1.0, g[1], 1e-10);
        assertEquals(1.0 / 2.0, g[2], 1e-10);
    }

    @Test
    public void testLogCheckGradient() {
        IDiffVector x = AD.vector(0.3, 0.7, 1.5, 2.2);
        assertTrue(AD.checkGradient(v -> v.log().sum(), x, 1e-6),
                "log() gradient must pass finite-difference check");
    }

    // ==================== C5: hardtanh/clamp backward boundary ====================

    /** Before fix: backward used strict >/<, giving 0 gradient at x==min/x==max even
     *  though forward passes those values through. */
    @Test
    public void testHardtanhBoundaryGradient() {
        IDiffVector x = AD.vector(-1.0, 0.0, 1.0);
        x.hardtanh(-1.0, 1.0).sum().backward();
        double[] g = x.getGradient().getData();
        assertArrayEquals(new double[]{1.0, 1.0, 1.0}, g, 1e-12,
                "gradient must pass through at x==min and x==max");
    }

    @Test
    public void testClampBoundaryGradient() {
        IDiffVector x = AD.vector(-1.0, 0.0, 1.0);
        x.clamp(-1.0, 1.0).sum().backward();
        double[] g = x.getGradient().getData();
        assertArrayEquals(new double[]{1.0, 1.0, 1.0}, g, 1e-12,
                "clamp gradient must pass through at boundaries");
    }

    // ==================== C6: sinh/cosh backward must clamp input ====================

    /** Before fix: backward used unclamped x → cosh(800)=Infinity exploded the gradient. */
    @Test
    public void testSinhLargeInputFiniteGradient() {
        IDiffVector x = AD.vector(800.0);
        x.sinh().sum().backward();
        double g = x.getGradient().get(0);
        assertTrue(Double.isFinite(g), "sinh backward must be finite for large input, got " + g);
    }

    @Test
    public void testCoshLargeInputFiniteGradient() {
        IDiffVector x = AD.vector(800.0);
        x.cosh().sum().backward();
        double g = x.getGradient().get(0);
        assertTrue(Double.isFinite(g), "cosh backward must be finite for large input, got " + g);
    }

    // ==================== C3: addInPlace must modify receiver + isLeaf guard ====================

    /** Before fix: addInPlace delegated to add() (a no-op on the receiver). */
    @Test
    public void testAddInPlaceModifiesReceiver() {
        IDiffVector x = AD.vector(1.0, 2.0, 3.0);
        IDiffVector other = AD.vector(10.0, 20.0, 30.0);
        IDiffVector ret = x.addInPlace(other);
        assertSame(x, ret, "addInPlace must return this");
        assertArrayEquals(new double[]{11.0, 22.0, 33.0}, x.getValue().getData(), 1e-12,
                "receiver value must be updated in-place");
        assertNull(((RereDiffVector) x).tensor.gradData(),
                "addInPlace must null stale gradient");
    }

    @Test
    public void testAddInPlaceRejectsNonLeaf() {
        IDiffVector x = AD.vector(1.0, 2.0, 3.0);
        IDiffVector y = x.add(AD.vector(0.1, 0.1, 0.1)); // non-leaf
        assertThrows(IllegalStateException.class,
                () -> y.addInPlace(AD.vector(1.0, 1.0, 1.0)),
                "addInPlace on non-leaf must throw");
    }

    // ==================== H1: scalar in-place must null gradient ====================

    @Test
    public void testScalarInPlaceNullsGradient() {
        IDiffVector x = AD.vector(1.0, 2.0, 3.0);
        // seed a gradient, then ensure in-place op clears it
        x.mulInPlace(2.0);
        assertNull(((RereDiffVector) x).tensor.gradData());
        x.addScalarInPlace(5.0);
        assertNull(((RereDiffVector) x).tensor.gradData(),
                "addScalarInPlace must null stale gradient");
        x.subScalarInPlace(1.0);
        assertNull(((RereDiffVector) x).tensor.gradData(),
                "subScalarInPlace must null stale gradient");
        x.multiplyByScalarInPlace(0.5);
        assertNull(((RereDiffVector) x).tensor.gradData(),
                "multiplyByScalarInPlace must null stale gradient");
    }

    // ==================== H3: backward(IDoubleTensor) must validate shape ====================

    @Test
    public void testBackwardRejectsWrongShapeGradient() {
        IDiffTensor x = AD.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor y = x.sum(); // shape [1]
        RereDoubleTensor bad = new RereDoubleTensor(new double[]{1.0, 2.0}, new int[]{2});
        assertThrows(IllegalArgumentException.class, () -> y.backward(bad),
                "backward must reject a gradient whose size differs from the output");
    }

    @Test
    public void testBackwardAcceptsCorrectShapeGradient() {
        IDiffTensor x = AD.tensor(new double[]{1, 2, 3, 4}, 2, 2);
        IDiffTensor y = x.sum(); // shape [1]
        RereDoubleTensor seed = new RereDoubleTensor(new double[]{1.0}, new int[]{1});
        y.backward(seed);
        assertArrayEquals(new double[]{1, 1, 1, 1}, x.grad().toDoubleArray(), 1e-12);
    }

    // ==================== M4: sum() forward uses sumAll (value unchanged) ====================

    @Test
    public void testSumForwardValue() {
        IDiffTensor x = AD.tensor(new double[]{1, 2, 3, 4}, 4);
        IDiffTensor s = x.sum();
        assertEquals(10.0, s.toDoubleArray()[0], 1e-12);
    }
}
