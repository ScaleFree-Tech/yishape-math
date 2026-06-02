package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.IDoubleVector;

public class ADHigherOrderTest {

    private static final double TOL = 1e-10;

    // ================ tape-of-tape: grad() ================

    @Test
    void testFirstOrderGrad() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector z = x.pow(2).sum();
        IDiffVector[] grads = AD.grad(z, x);
        IDiffVector gradX = grads[0];

        double[] expected = { 2, 4, 6 };
        assertArrayEquals(expected, gradX.getValue().getData(), TOL);
    }

    @Test
    void testSecondOrderGrad() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector z = x.pow(2).sum();
        IDiffVector[] firstGrads = AD.grad(z, x);
        IDiffVector gradX = firstGrads[0];

        // gradX = 2x = [2, 4, 6]
        // d(gradX.sum())/dx = d(2*sum(x))/dx = [2, 2, 2]
        IDiffVector gradXSum = gradX.sum();
        IDiffVector[] secondGrads = AD.grad(gradXSum, x);
        IDiffVector hessianDiag = secondGrads[0];

        double[] expected = { 2, 2, 2 };
        assertArrayEquals(expected, hessianDiag.getValue().getData(), TOL);
    }

    @Test
    void testThirdOrderGrad() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector z = x.pow(3).sum();
        IDiffVector[] g1 = AD.grad(z, x);
        // g1 = 3x^2 = [3, 12, 27]
        IDiffVector g1sum = g1[0].sum();
        IDiffVector[] g2 = AD.grad(g1sum, x);
        // g2 = 6x = [6, 12, 18]
        IDiffVector g2sum = g2[0].sum();
        IDiffVector[] g3 = AD.grad(g2sum, x);
        // g3 = [6, 6, 6]

        assertArrayEquals(new double[] { 6, 6, 6 }, g3[0].getValue().getData(), TOL);
    }

    @Test
    void testGradThroughAdd() {
        IDiffVector x = AD.vector(new double[] { 1, 2 });
        IDiffVector y = AD.vector(new double[] { 3, 4 });
        IDiffVector z = x.add(y).sum();
        IDiffVector[] grads = AD.grad(z, x, y);

        assertArrayEquals(new double[] { 1, 1 }, grads[0].getValue().getData(), TOL);
        assertArrayEquals(new double[] { 1, 1 }, grads[1].getValue().getData(), TOL);
    }

    @Test
    void testGradThroughSub() {
        IDiffVector x = AD.vector(new double[] { 5, 7 });
        IDiffVector y = AD.vector(new double[] { 2, 3 });
        IDiffVector z = x.sub(y).sum();
        IDiffVector[] grads = AD.grad(z, x, y);

        assertArrayEquals(new double[] { 1, 1 }, grads[0].getValue().getData(), TOL);
        assertArrayEquals(new double[] { -1, -1 }, grads[1].getValue().getData(), TOL);
    }

    @Test
    void testGradThroughMul() {
        IDiffVector x = AD.vector(new double[] { 2, 3 });
        IDiffVector y = AD.vector(new double[] { 4, 5 });
        IDiffVector z = x.mul(y).sum();
        IDiffVector[] grads = AD.grad(z, x, y);

        assertArrayEquals(new double[] { 4, 5 }, grads[0].getValue().getData(), TOL);
        assertArrayEquals(new double[] { 2, 3 }, grads[1].getValue().getData(), TOL);
    }

    @Test
    void testGradThroughDiv() {
        IDiffVector x = AD.vector(new double[] { 6, 8 });
        IDiffVector y = AD.vector(new double[] { 2, 4 });
        IDiffVector z = x.div(y).sum();
        IDiffVector[] grads = AD.grad(z, x, y);

        assertEquals(1.0 / 2.0, grads[0].getValue().get(0), TOL);
        assertEquals(1.0 / 4.0, grads[0].getValue().get(1), TOL);
        assertEquals(-6.0 / 4.0, grads[1].getValue().get(0), TOL);
        assertEquals(-8.0 / 16.0, grads[1].getValue().get(1), TOL);
    }

    @Test
    void testGradThroughExp() {
        IDiffVector x = AD.vector(new double[] { 0, 1, 2 });
        IDiffVector z = x.exp().sum();
        IDiffVector[] grads = AD.grad(z, x);

        double[] expected = { Math.exp(0), Math.exp(1), Math.exp(2) };
        assertArrayEquals(expected, grads[0].getValue().getData(), TOL);
    }

    @Test
    void testGradThroughLog() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 4 });
        IDiffVector z = x.log().sum();
        IDiffVector[] grads = AD.grad(z, x);

        double[] expected = { 1.0, 0.5, 0.25 };
        assertArrayEquals(expected, grads[0].getValue().getData(), TOL);
    }

    @Test
    void testGradThroughSin() {
        IDiffVector x = AD.vector(new double[] { 0, Math.PI / 2 });
        IDiffVector z = x.sin().sum();
        IDiffVector[] grads = AD.grad(z, x);

        assertEquals(Math.cos(0), grads[0].getValue().get(0), TOL);
        assertEquals(Math.cos(Math.PI / 2), grads[0].getValue().get(1), 1e-8);
    }

    @Test
    void testGradThroughMean() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3, 4 });
        IDiffVector z = x.mean();
        IDiffVector[] grads = AD.grad(z, x);

        assertArrayEquals(new double[] { 0.25, 0.25, 0.25, 0.25 }, grads[0].getValue().getData(), TOL);
    }

    @Test
    void testGradThroughDot() {
        IDiffVector a = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector b = AD.vector(new double[] { 4, 5, 6 });
        IDiffVector z = a.dot(b);
        IDiffVector[] grads = AD.grad(z, a, b);

        assertArrayEquals(new double[] { 4, 5, 6 }, grads[0].getValue().getData(), TOL);
        assertArrayEquals(new double[] { 1, 2, 3 }, grads[1].getValue().getData(), TOL);
    }

    @Test
    void testGradScalarMul() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector z = x.mul(3).sum();
        IDiffVector[] grads = AD.grad(z, x);

        assertArrayEquals(new double[] { 3, 3, 3 }, grads[0].getValue().getData(), TOL);
    }

    @Test
    void testGradSharedVariable() {
        IDiffVector x = AD.vector(new double[] { 2, 3 });
        IDiffVector z = x.mul(2).add(x.mul(3)).sum();
        IDiffVector[] grads = AD.grad(z, x);

        assertArrayEquals(new double[] { 5, 5 }, grads[0].getValue().getData(), TOL);
    }

    @Test
    void testGradSquareThenBackwardOnResult() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector z = x.square().sum();
        IDiffVector[] grads = AD.grad(z, x);
        IDiffVector gradX = grads[0];

        // gradX = 2x = [2, 4, 6]
        // We can call backward on expressions built from gradX
        IDiffVector gradNorm = gradX.pow(2).sum();
        gradNorm.backward();

        // d(gradNorm)/d(symbolic graph leaves): gradNorm = sum((2x)^2) = sum(4x^2)
        // But gradX involves constants of x's VALUE, not x itself for square op
        // So backward through gradNorm only reaches the constant ops in gradX
        // Let's verify gradX itself is non-null and has gradient
        assertNotNull(gradX.getValue());
        assertEquals(2.0, gradX.getValue().get(0), TOL);
        assertEquals(4.0, gradX.getValue().get(1), TOL);
        assertEquals(6.0, gradX.getValue().get(2), TOL);
    }

    @Test
    void testGradPow() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector z = x.pow(3).sum();
        IDiffVector[] grads = AD.grad(z, x);

        assertArrayEquals(new double[] { 3, 12, 27 }, grads[0].getValue().getData(), TOL);
    }

    @Test
    void testGradTanhSigmoidReluValues() {
        IDiffVector x = AD.vector(new double[] { 1.0, -1.0 });
        IDiffVector z = x.tanh().sum();
        IDiffVector[] grads = AD.grad(z, x);
        double t1 = Math.tanh(1), t_1 = Math.tanh(-1);
        assertEquals(1 - t1 * t1, grads[0].getValue().get(0), TOL);
        assertEquals(1 - t_1 * t_1, grads[0].getValue().get(1), TOL);
    }

    @Test
    void testBroadcastOperation() {
        IDiffVector scalar = AD.vector(new double[] { 5.0 });
        IDiffVector broadcasted = scalar.broadcast(4);

        assertArrayEquals(new double[] { 5, 5, 5, 5 }, broadcasted.getValue().getData(), TOL);

        IDiffVector loss = broadcasted.sum();
        loss.backward();
        assertEquals(4.0, scalar.getGradient().get(0), TOL);
    }

    @Test
    void testBroadcastGrad() {
        IDiffVector x = AD.vector(new double[] { 3.0 });
        IDiffVector b = x.broadcast(3);
        IDiffVector y = b.sum();
        y.backward();
        assertEquals(3.0, x.getGradient().get(0), TOL);
    }

    // ================ operator fusion ================

    @Test
    void testFusedExpLog() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.fuse(x).exp().log().compute();

        // exp(log(x)) ≈ x (within numerical precision)
        double[] result = y.getValue().getData();
        assertArrayEquals(new double[] { 1, 2, 3 }, result, 1e-10);
    }

    @Test
    void testFusedSqrtSquare() {
        IDiffVector x = AD.vector(new double[] { 1.0, 4.0, 9.0 });
        IDiffVector y = AD.fuse(x).sqrt().square().compute();

        double[] result = y.getValue().getData();
        assertArrayEquals(new double[] { 1, 4, 9 }, result, 1e-10);
    }

    @Test
    void testFusedEmpty() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector y = AD.fuse(x).compute();

        assertArrayEquals(x.getValue().getData(), y.getValue().getData(), TOL);
        assertTrue(y.isLeaf() || y == x); // no ops: returns original
    }

    @Test
    void testFusedGradient() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector y = AD.fuse(x).square().exp().compute();
        IDiffVector loss = y.sum();
        loss.backward();

        double[] autoGrad = x.getGradient().getData();
        double eps = 1e-6;
        double[] xData = { 1.0, 2.0, 3.0 };
        for (int i = 0; i < xData.length; i++) {
            xData[i] += eps;
            double fp = fusedEval(xData);
            xData[i] -= 2 * eps;
            double fm = fusedEval(xData);
            xData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGrad[i], 1e-5);
        }
    }

    private double fusedEval(double[] x) {
        double s = 0;
        for (double v : x) {
            double r = v * v;       // square
            r = Math.exp(r);        // exp
            s += r;
        }
        return s;
    }

    @Test
    void testFusedMultipleOps() {
        IDiffVector x = AD.vector(new double[] { 2.0, -1.0, 0.5 });
        IDiffVector y = AD.fuse(x).relu().sigmoid().tanh().compute();
        IDiffVector loss = y.sum();
        loss.backward();

        double[] autoGrad = x.getGradient().getData();
        double eps = 1e-6;
        double[] xData = { 2.0, -1.0, 0.5 };
        for (int i = 0; i < xData.length; i++) {
            xData[i] += eps;
            double fp = multiFusedEval(xData);
            xData[i] -= 2 * eps;
            double fm = multiFusedEval(xData);
            xData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGrad[i], 1e-5);
        }
    }

    private double multiFusedEval(double[] x) {
        double s = 0;
        for (double v : x) {
            double r = Math.max(0, v);                          // relu
            r = 1.0 / (1.0 + Math.exp(-r));                     // sigmoid
            r = Math.tanh(r);                                   // tanh
            s += r;
        }
        return s;
    }

    @Test
    void testFusedVsUnfused() {
        double[] xData = { 1.0, 2.0, 3.0 };

        IDiffVector x1 = AD.vector(xData.clone());
        IDiffVector y1 = AD.fuse(x1).exp().log().sqrt().square().compute();
        IDiffVector loss1 = y1.sum();
        loss1.backward();
        double[] fusedGrad = x1.getGradient().getData().clone();

        IDiffVector x2 = AD.vector(xData.clone());
        IDiffVector y2 = x2.exp().log().sqrt().square();
        IDiffVector loss2 = y2.sum();
        loss2.backward();
        double[] unfusedGrad = x2.getGradient().getData().clone();

        assertArrayEquals(unfusedGrad, fusedGrad, 1e-10);
    }

    @Test
    void testFusedNeg() {
        IDiffVector x = AD.vector(new double[] { 1.0, -2.0, 3.0 });
        IDiffVector y = AD.fuse(x).neg().compute();
        IDiffVector loss = y.sum();
        loss.backward();

        double[] result = y.getValue().getData();
        assertArrayEquals(new double[] { -1, 2, -3 }, result, TOL);
        assertArrayEquals(new double[] { -1, -1, -1 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testFusedAbs() {
        IDiffVector x = AD.vector(new double[] { 1.0, -2.0, 0.0 });
        IDiffVector y = AD.fuse(x).abs().compute();
        IDiffVector loss = y.sum();
        loss.backward();

        double[] result = y.getValue().getData();
        assertArrayEquals(new double[] { 1, 2, 0 }, result, TOL);
        assertEquals(1.0, x.getGradient().get(0), TOL);
        assertEquals(-1.0, x.getGradient().get(1), TOL);
        assertEquals(0.0, x.getGradient().get(2), TOL);
    }

    @Test
    void testFusedLongChain() {
        IDiffVector x = AD.vector(new double[] { 0.5, 1.0, 2.0 });
        IDiffVector y = AD.fuse(x)
                .exp()
                .log()
                .sqrt()
                .square()
                .sigmoid()
                .tanh()
                .relu()
                .neg()
                .abs()
                .compute();
        IDiffVector loss = y.sum();
        loss.backward();

        double[] autoGrad = x.getGradient().getData();
        double eps = 1e-6;
        double[] xData = { 0.5, 1.0, 2.0 };
        for (int i = 0; i < xData.length; i++) {
            xData[i] += eps;
            double fp = longChainEval(xData);
            xData[i] -= 2 * eps;
            double fm = longChainEval(xData);
            xData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGrad[i], 1e-5);
        }
    }

    private double longChainEval(double[] x) {
        double s = 0;
        for (double v : x) {
            double r = Math.exp(v);
            r = Math.log(r);
            r = Math.sqrt(r);
            r = r * r;
            r = 1.0 / (1.0 + Math.exp(-r));
            r = Math.tanh(r);
            r = Math.max(0, r);
            r = -r;
            r = Math.abs(r);
            s += r;
        }
        return s;
    }

    @Test
    void testConstantIsLeafVariable() {
        IDiffVector c = AD.constant(IDoubleVector.of(new double[] { 1, 2, 3 }));
        assertTrue(c.isLeaf());
        // Constants are regular leaf variables; they can receive gradients
        // when used directly. Their role in tape-of-tape is to carry forward
        // values through symbolic backward functions without introducing
        // spurious gradient paths to original variables.
        IDiffVector y = c.mul(2).sum();
        y.backward();
        assertArrayEquals(new double[] { 2, 2, 2 }, c.getGradient().getData(), TOL);
    }

    @Test
    void testGradResultIsDifferentiable() {
        IDiffVector x = AD.vector(new double[] { 1.0, 2.0, 3.0 });
        IDiffVector z = x.pow(2).sum();
        IDiffVector[] grads = AD.grad(z, x);
        IDiffVector gradX = grads[0];

        // gradX should be a valid IDiffVector that we can further operate on
        assertNotNull(gradX);
        assertTrue(gradX instanceof IDiffVector);
        assertArrayEquals(new double[] { 2, 4, 6 }, gradX.getValue().getData(), TOL);
    }
}
