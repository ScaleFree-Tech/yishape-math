package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.IDoubleVector;

public class ADTest {

    private static final double TOL = 1e-10;

    @Test
    void testPowerGradient() {
        IDiffVector x = AD.vector(3.0);
        IDiffVector y = x.pow(2);
        y.backward();
        assertEquals(9.0, y.getValue().get(0), TOL);
        assertEquals(6.0, x.getGradient().get(0), TOL);
    }

    @Test
    void testSumOfSquares() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector y = x.pow(2).sum();
        y.backward();
        assertEquals(14.0, y.getValue().get(0), TOL);
        double[] grad = x.getGradient().getData();
        assertArrayEquals(new double[] { 2, 4, 6 }, grad, TOL);
    }

    @Test
    void testExpSum() {
        IDiffVector x = AD.vector(new double[] { 0, 1, 2 });
        IDiffVector y = x.exp().sum();
        y.backward();
        double[] grad = x.getGradient().getData();
        assertEquals(Math.exp(0), grad[0], TOL);
        assertEquals(Math.exp(1), grad[1], TOL);
        assertEquals(Math.exp(2), grad[2], TOL);
    }

    @Test
    void testDotProduct() {
        IDiffVector a = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector b = AD.vector(new double[] { 4, 5, 6 });
        IDiffVector z = a.dot(b);
        z.backward();
        assertEquals(32.0, z.getValue().get(0), TOL);
        double[] gradA = a.getGradient().getData();
        double[] gradB = b.getGradient().getData();
        assertArrayEquals(new double[] { 4, 5, 6 }, gradA, TOL);
        assertArrayEquals(new double[] { 1, 2, 3 }, gradB, TOL);
    }

    @Test
    void testAddVariables() {
        IDiffVector x = AD.vector(new double[] { 1, 2 });
        IDiffVector y = AD.vector(new double[] { 3, 4 });
        IDiffVector z = x.add(y).sum();
        z.backward();
        assertArrayEquals(new double[] { 1, 1 }, x.getGradient().getData(), TOL);
        assertArrayEquals(new double[] { 1, 1 }, y.getGradient().getData(), TOL);
    }

    @Test
    void testSubVariables() {
        IDiffVector x = AD.vector(new double[] { 5, 7 });
        IDiffVector y = AD.vector(new double[] { 2, 3 });
        IDiffVector z = x.sub(y).sum();
        z.backward();
        assertArrayEquals(new double[] { 1, 1 }, x.getGradient().getData(), TOL);
        assertArrayEquals(new double[] { -1, -1 }, y.getGradient().getData(), TOL);
    }

    @Test
    void testMulVariables() {
        IDiffVector x = AD.vector(new double[] { 2, 3 });
        IDiffVector y = AD.vector(new double[] { 4, 5 });
        IDiffVector z = x.mul(y).sum();
        z.backward();
        assertArrayEquals(new double[] { 4, 5 }, x.getGradient().getData(), TOL);
        assertArrayEquals(new double[] { 2, 3 }, y.getGradient().getData(), TOL);
    }

    @Test
    void testDivVariables() {
        IDiffVector x = AD.vector(new double[] { 6, 8 });
        IDiffVector y = AD.vector(new double[] { 2, 4 });
        IDiffVector z = x.div(y).sum();
        z.backward();
        double[] gradX = x.getGradient().getData();
        double[] gradY = y.getGradient().getData();
        assertEquals(1.0 / 2.0, gradX[0], TOL);
        assertEquals(1.0 / 4.0, gradX[1], TOL);
        assertEquals(-6.0 / 4.0, gradY[0], TOL);
        assertEquals(-8.0 / 16.0, gradY[1], TOL);
    }

    @Test
    void testScalarArithmetic() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector y = x.mul(2).add(1).sum();
        y.backward();
        assertEquals(15.0, y.getValue().get(0), TOL);
        assertArrayEquals(new double[] { 2, 2, 2 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testLog() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 4 });
        IDiffVector y = x.log().sum();
        y.backward();
        double[] grad = x.getGradient().getData();
        assertEquals(1.0, grad[0], TOL);
        assertEquals(0.5, grad[1], TOL);
        assertEquals(0.25, grad[2], TOL);
    }

    @Test
    void testNeg() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector y = x.neg().sum();
        y.backward();
        assertArrayEquals(new double[] { -1, -1, -1 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testSigmoid() {
        IDiffVector x = AD.vector(new double[] { 0, 1 });
        IDiffVector y = x.sigmoid().sum();
        y.backward();
        double[] grad = x.getGradient().getData();
        double s0 = 1.0 / (1.0 + Math.exp(0));
        double s1 = 1.0 / (1.0 + Math.exp(-1));
        assertEquals(s0 * (1 - s0), grad[0], TOL);
        assertEquals(s1 * (1 - s1), grad[1], TOL);
    }

    @Test
    void testRelu() {
        IDiffVector x = AD.vector(new double[] { -2, -1, 0, 1, 2 });
        IDiffVector y = x.relu().sum();
        y.backward();
        double[] grad = x.getGradient().getData();
        assertEquals(0.0, grad[0], TOL);
        assertEquals(0.0, grad[1], TOL);
        assertEquals(0.0, grad[2], TOL);
        assertEquals(1.0, grad[3], TOL);
        assertEquals(1.0, grad[4], TOL);
    }

    @Test
    void testTanh() {
        IDiffVector x = AD.vector(new double[] { 0, 1 });
        IDiffVector y = x.tanh().sum();
        y.backward();
        double[] grad = x.getGradient().getData();
        double t0 = Math.tanh(0);
        double t1 = Math.tanh(1);
        assertEquals(1.0 - t0 * t0, grad[0], TOL);
        assertEquals(1.0 - t1 * t1, grad[1], TOL);
    }

    @Test
    void testMean() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3, 4 });
        IDiffVector y = x.mean();
        y.backward();
        assertArrayEquals(new double[] { 0.25, 0.25, 0.25, 0.25 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testSharedVariable() {
        IDiffVector x = AD.vector(new double[] { 2, 3 });
        IDiffVector y = x.mul(2);
        IDiffVector z = x.mul(3);
        IDiffVector w = y.add(z).sum();
        w.backward();
        assertArrayEquals(new double[] { 5, 5 }, x.getGradient().getData(), TOL);
    }

    @Test
    void testIsLeaf() {
        IDiffVector x = AD.vector(new double[] { 1, 2 });
        assertTrue(x.isLeaf());
        IDiffVector y = x.add(1);
        assertTrue(!y.isLeaf());
    }

    @Test
    void testZeroGradient() {
        IDiffVector x = AD.vector(new double[] { 1, 2 });
        IDiffVector y = x.pow(2).sum();
        y.backward();
        assertNotNull(x.getGradient());
        x.zeroGradient();
        assertEquals(null, x.getGradient());
    }

    @Test
    void testNumericalGradientCheck() {
        double[] xData = { 0.5, 1.0, 1.5, 2.0 };
        IDiffVector x = AD.vector(xData);
        IDiffVector loss = x.sin().mul(x.exp()).sum();
        loss.backward();
        double[] autoGrad = x.getGradient().getData();

        double eps = 1e-6;
        for (int i = 0; i < xData.length; i++) {
            xData[i] += eps;
            double fp = numericalEval(xData);
            xData[i] -= 2 * eps;
            double fm = numericalEval(xData);
            xData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGrad[i], 1e-5);
        }
    }

    // ---- Priority 1 optimization tests ----

    @Test
    void testConstantFoldingAddSubZero() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        assertTrue(x.add(0.0) == x, "add(0) should return self");
        assertTrue(x.sub(0.0) == x, "sub(0) should return self");
    }

    @Test
    void testConstantFoldingMulDiv() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        assertTrue(x.mul(1.0) == x, "mul(1) should return self");
        assertTrue(x.div(1.0) == x, "div(1) should return self");
    }

    @Test
    void testConstantFoldingMulZero() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector z = x.mul(0.0);
        assertEquals(0.0, z.getValue().get(0), TOL);
        assertEquals(0.0, z.getValue().get(1), TOL);
        assertEquals(0.0, z.getValue().get(2), TOL);
        // mul(0) returns a constant node, detached from x
        // backward through it should NOT propagate to x
        IDiffVector s = z.sum();
        s.backward();
        // x should have no gradient because z is a detached constant
        assertEquals(null, x.getGradient(), "mul(0) constant should not backprop to x");
    }

    @Test
    void testConstantFoldingPow() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        assertTrue(x.pow(1.0) == x, "pow(1) should return self");
        IDiffVector ones = x.pow(0.0);
        assertEquals(1.0, ones.getValue().get(0), TOL);
        assertEquals(1.0, ones.getValue().get(1), TOL);
        assertEquals(1.0, ones.getValue().get(2), TOL);
    }

    @Test
    void testInPlaceAccGradMultiPath() {
        // Multiple backward paths into same leaf → in-place accGrad
        IDiffVector x = AD.vector(new double[] { 2.0 });
        IDiffVector a = x.mul(3.0);  // d(3x)/dx = 3
        IDiffVector b = x.mul(5.0);  // d(5x)/dx = 5
        IDiffVector c = a.add(b).sum();  // gradient = 3 + 5 = 8
        c.backward();
        assertEquals(8.0, x.getGradient().get(0), TOL);
    }

    @Test
    void testSquareSumFusionGradient() {
        // Fused square().sum() should produce same gradient as explicit path
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector fused = x.square().sum();
        fused.backward();
        double[] grad = x.getGradient().getData();
        // d/dx sum(x²) = 2x
        assertArrayEquals(new double[] { 2, 4, 6 }, grad, TOL);
    }

    @Test
    void testSquareMeanFusionGradient() {
        IDiffVector x = AD.vector(new double[] { 1, 2, 3, 4 });
        int n = 4;
        IDiffVector fused = x.square().mean();
        fused.backward();
        double[] grad = x.getGradient().getData();
        // d/dx mean(x²) = 2x/n
        for (int i = 0; i < n; i++) {
            assertEquals(2.0 * (i + 1) / n, grad[i], TOL);
        }
    }

    @Test
    void testThreadLocalTopoReuse() {
        // Run backward many times — ThreadLocal buffers should be reused
        for (int iter = 0; iter < 100; iter++) {
            IDiffVector x = AD.vector(new double[] { iter + 1, iter + 2, iter + 3 });
            IDiffVector y = x.square().sum();
            y.backward();
        }
        // No assertion needed; verifies no memory leak or corruption across reuse
    }

    @Test
    void testSumBackwardReusesGradientBuffer() {
        // First backward: gradient is null, allocates
        IDiffVector x = AD.vector(new double[] { 1, 2, 3 });
        IDiffVector y = x.square().sum();
        y.backward();
        assertArrayEquals(new double[] { 2, 4, 6 }, x.getGradient().getData(), TOL);

        // Second backward on same x (after zeroGradient): gradient is null again
        x.zeroGradient();
        y = x.mul(2.0).sum();
        y.backward();
        assertEquals(2.0, x.getGradient().get(0), TOL);
        assertEquals(2.0, x.getGradient().get(1), TOL);
        assertEquals(2.0, x.getGradient().get(2), TOL);
    }

    // ---- Gradient verification via numerical check ----

    @Test
    void testFusedSquareSumMatchesNumerical() {
        double[] xData = { 0.5, 1.0, 1.5, 2.0 };
        IDiffVector x = AD.vector(xData);
        IDiffVector loss = x.square().sum();
        loss.backward();
        double[] autoGrad = x.getGradient().getData();

        double eps = 1e-6;
        for (int i = 0; i < xData.length; i++) {
            xData[i] += eps;
            double fp = xData[0]*xData[0] + xData[1]*xData[1] + xData[2]*xData[2] + xData[3]*xData[3];
            xData[i] -= 2 * eps;
            double fm = xData[0]*xData[0] + xData[1]*xData[1] + xData[2]*xData[2] + xData[3]*xData[3];
            xData[i] += eps;
            double numGrad = (fp - fm) / (2 * eps);
            assertEquals(numGrad, autoGrad[i], 1e-5);
        }
    }

    private double numericalEval(double[] x) {
        double s = 0;
        for (double v : x) {
            s += Math.sin(v) * Math.exp(v);
        }
        return s;
    }
}
