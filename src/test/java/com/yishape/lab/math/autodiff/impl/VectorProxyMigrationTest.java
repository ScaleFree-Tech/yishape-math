package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Gradient correctness for vector ops migrated to the tensor layer (§7c/§7b).
 * Each op now delegates via {@code wrap(tensor.xxx())}; this verifies the
 * tensor-side backward produces gradients matching central differences.
 */
class VectorProxyMigrationTest {

    private static final double TOL = 1e-5;
    private static final double ABS_TOL = 1e-6;

    private static boolean chk(java.util.function.Function<IDiffVector, IDiffVector> loss, double[] x) {
        return AD.checkGradient(loss, AD.vector(x), TOL);
    }

    /**
     * Absolute-error gradient check for ops that legitimately produce zero gradients
     * at certain points (e.g. var/std gradient is zero at the mean, normalize near a
     * zero-gradient element). Relative error is undefined at zero, so assert on the
     * absolute error against central differences instead.
     */
    private static boolean chkAbs(java.util.function.Function<IDiffVector, IDiffVector> loss, double[] x) {
        var r = AD.checkGradientDetailed(loss, AD.vector(x), TOL);
        return r.maxAbsoluteError() < ABS_TOL;
    }

    @Test void arcsin() {
        assertTrue(chk(v -> v.arcsin().sum(), new double[]{-0.9, -0.2, 0.0, 0.4, 0.95}));
    }

    @Test void arccos() {
        assertTrue(chk(v -> v.arccos().sum(), new double[]{-0.8, 0.1, 0.5, 0.9}));
    }

    @Test void arctan() {
        assertTrue(chk(v -> v.arctan().pow(2).sum(), new double[]{-2.0, -0.5, 0.3, 1.7}));
    }

    @Test void sinh() {
        assertTrue(chk(v -> v.sinh().sum(), new double[]{-2.0, -0.5, 0.0, 0.7, 1.5}));
    }

    @Test void cosh() {
        assertTrue(chk(v -> v.cosh().sum(), new double[]{-2.0, -0.5, 0.0, 0.7, 1.5}));
    }

    @Test void remainder() {
        // straight-through estimator: gradient is identity
        assertTrue(chk(v -> v.remainder(1.0).sum(), new double[]{0.3, 1.2, 2.7, -0.4}));
    }

    @Test void cross() {
        // cross with a constant vector; sum to scalar
        var other = Linalg.vector(new double[]{0.0, 1.0, 0.0});
        assertTrue(chk(v -> v.cross(other).sum(), new double[]{1.0, 2.0, 3.0}));
    }

    @Test void mmul() {
        // y = mat @ x, mat constant [3x2], x is the diff vector of length 2
        var mat = Linalg.matrix(new double[][]{{1, 2}, {0, -1}, {3, 1}});
        assertTrue(chk(v -> v.mmul(mat).sum(), new double[]{0.5, -1.5}));
    }

    @Test void tile() {
        assertTrue(chk(v -> v.tile(3).sum(), new double[]{1.0, 2.0, 3.0, 4.0}));
    }

    @Test void cumsum() {
        assertTrue(chk(v -> v.cumsum().pow(2).sum(), new double[]{1.0, 2.0, 3.0, 4.0}));
    }

    @Test void cumprod() {
        assertTrue(chk(v -> v.cumprod().sum(), new double[]{1.0, 2.0, 0.5, 3.0}));
    }

    @Test void min() {
        assertTrue(chk(v -> v.min(), new double[]{3.0, 1.0, 2.0, 1.5, 4.0}));
    }

    @Test void max() {
        assertTrue(chk(v -> v.max(), new double[]{3.0, 1.0, 5.0, 1.5, 4.0}));
    }

    @Test void prod() {
        assertTrue(chk(v -> v.prod(), new double[]{1.0, 2.0, 0.5, 3.0}));
    }

    @Test void norm2() {
        assertTrue(chk(v -> v.norm2(), new double[]{3.0, 1.0, 2.0, 4.0}));
    }

    @Test void norm1() {
        assertTrue(chk(v -> v.norm1(), new double[]{3.0, -1.0, 2.0, -4.0}));
    }

    @Test void ptp() {
        assertTrue(chk(v -> v.ptp(), new double[]{3.0, 1.0, 5.0, 1.5, 4.0}));
    }

    @Test void normalize() {
        assertTrue(chkAbs(v -> v.normalize().sum(), new double[]{3.0, 1.0, 2.0, 4.0}));
    }

    @Test void std() {
        assertTrue(chkAbs(v -> v.std(), new double[]{3.0, 1.0, 2.0, 5.0, 4.0}));
    }

    @Test void stdDdof() {
        assertTrue(chkAbs(v -> v.std(1), new double[]{3.0, 1.0, 2.0, 5.0, 4.0}));
    }

    @Test void var() {
        assertTrue(chkAbs(v -> v.var(), new double[]{3.0, 1.0, 2.0, 5.0, 4.0}));
    }

    @Test void varDdof() {
        assertTrue(chkAbs(v -> v.var(2), new double[]{3.0, 1.0, 2.0, 5.0, 4.0}));
    }

    @Test void reverse() {
        assertTrue(chkAbs(v -> v.reverse().sum(), new double[]{1.0, 2.0, 3.0, 4.0}));
    }

    @Test void repeat() {
        assertTrue(chkAbs(v -> v.repeat(2).sum(), new double[]{1.0, 2.0, 3.0}));
    }

    @Test void sliceStep() {
        assertTrue(chkAbs(v -> v.slice(0, 4, 2).sum(), new double[]{1.0, 2.0, 3.0, 4.0, 5.0}));
    }

    @Test void fancyGet() {
        assertTrue(chkAbs(v -> v.fancyGet(new int[]{3, 1, 0}).sum(), new double[]{1.0, 2.0, 3.0, 4.0}));
    }

    @Test void booleanGet() {
        assertTrue(chkAbs(v -> v.booleanGet(new boolean[]{false, true, false, true}).sum(),
                new double[]{1.0, 2.0, 3.0, 4.0}));
    }

    @Test void diff() {
        assertTrue(chkAbs(v -> v.diff().sum(), new double[]{1.0, 3.0, 2.0, 5.0, 4.0}));
    }

    @Test void sort() {
        assertTrue(chkAbs(v -> v.sort().sum(), new double[]{3.0, 1.0, 4.0, 1.5, 2.0}));
    }

    @Test void whereScalar() {
        assertTrue(chkAbs(v -> v.where(new boolean[]{true, false, true, false}, 0.0, 9.0).sum(),
                new double[]{1.0, 2.0, 3.0, 4.0}));
    }

    @Test void whereVectors() {
        // grad routes to x where condition true; x is the checked variable.
        boolean[] cond = {true, false, true, false};
        assertTrue(chkAbs(v -> v.where(cond, v, AD.vector(new double[]{10, 20, 30, 40})).sum(),
                new double[]{1.0, 2.0, 3.0, 4.0}));
    }

    @Test void concat() {
        var other = Linalg.vector(new double[]{10.0, 20.0});
        assertTrue(chkAbs(v -> v.concat(other).sum(), new double[]{1.0, 2.0, 3.0}));
    }

    @Test void round() {
        // Avoid x.5 rounding boundaries — central differences cross the step there.
        assertTrue(chkAbs(v -> v.round().sum(), new double[]{1.3, 2.71, -0.42, 3.51}));
    }

    @Test void floor() {
        assertTrue(chkAbs(v -> v.floor().sum(), new double[]{1.3, 2.7, -0.4, 3.5}));
    }

    @Test void ceil() {
        assertTrue(chkAbs(v -> v.ceil().sum(), new double[]{1.3, 2.7, -0.4, 3.5}));
    }

    @Test void trunc() {
        assertTrue(chkAbs(v -> v.trunc().sum(), new double[]{1.3, 2.7, -0.4, 3.5}));
    }

    @Test void sign() {
        assertTrue(chkAbs(v -> v.sign().sum(), new double[]{1.3, -2.7, -0.4, 3.5}));
    }

    @Test void dot() {
        var other = Linalg.vector(new double[]{2.0, -1.0, 3.0, 0.5});
        assertTrue(chk(v -> v.dot(other), new double[]{1.0, 2.0, 3.0, 4.0}));
    }
}
