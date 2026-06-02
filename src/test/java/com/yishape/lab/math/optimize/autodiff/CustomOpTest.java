package com.yishape.lab.math.optimize.autodiff;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.CustomOp;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.IDoubleVector;

public class CustomOpTest {

    private static final double TOL = 1e-10;

    @Test
    void testSingleInputSquareGradient() {
        CustomOp square = new CustomOp() {
            @Override
            protected ForwardResult forward(IDoubleVector[] inputs) {
                double[] x = inputs[0].getData();
                double[] out = new double[x.length];
                for (int i = 0; i < x.length; i++) out[i] = x[i] * x[i];
                return new ForwardResult(IDoubleVector.of(out), inputs[0].copy());
            }

            @Override
            protected IDoubleVector[] backward(IDoubleVector gradOutput, Object ctx) {
                IDoubleVector x = (IDoubleVector) ctx;
                double[] g = gradOutput.getData();
                double[] xd = x.getData();
                double[] dX = new double[xd.length];
                for (int i = 0; i < dX.length; i++) dX[i] = 2.0 * xd[i] * g[i];
                return new IDoubleVector[]{ IDoubleVector.of(dX) };
            }
        };

        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        IDiffVector y = square.apply(x);
        IDiffVector loss = y.sum();
        loss.backward();

        assertArrayEquals(new double[]{2, 4, 6}, x.getGradient().getData(), TOL);
    }

    @Test
    void testMultiInputAddMul() {
        CustomOp addMul = new CustomOp() {
            @Override
            protected ForwardResult forward(IDoubleVector[] inputs) {
                double[] a = inputs[0].getData();
                double[] b = inputs[1].getData();
                double[] out = new double[a.length];
                for (int i = 0; i < out.length; i++) out[i] = a[i] * b[i];
                return new ForwardResult(IDoubleVector.of(out), null);
            }

            @Override
            protected IDoubleVector[] backward(IDoubleVector gradOutput, Object ctx) {
                // We don't need context since we can recompute from saved values
                // In real code, ctx would hold a, b. Here we just return placeholder gradients.
                // This test verifies both inputs receive gradients.
                double[] g = gradOutput.getData();
                int n = g.length;
                double[] dA = new double[n];
                double[] dB = new double[n];
                java.util.Arrays.fill(dA, 1.0);
                java.util.Arrays.fill(dB, 1.0);
                for (int i = 0; i < n; i++) {
                    dA[i] = g[i];
                    dB[i] = g[i];
                }
                return new IDoubleVector[]{IDoubleVector.of(dA), IDoubleVector.of(dB)};
            }
        };

        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        IDiffVector c = addMul.apply(a, b);
        IDiffVector loss = c.sum();
        loss.backward();

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{1, 1, 1}, a.getGradient().getData(), TOL);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().getData(), TOL);
    }

    @Test
    void testForwardContextAccessedInBackward() {
        // y_i = x_i * scale, where scale is computed in forward and passed via context
        CustomOp op = new CustomOp() {
            @Override
            protected ForwardResult forward(IDoubleVector[] inputs) {
                double[] xd = inputs[0].getData();
                double sum = 0;
                for (double v : xd) sum += v;
                double scale = 1.0 / sum;
                double[] out = new double[xd.length];
                for (int i = 0; i < xd.length; i++) out[i] = xd[i] * scale;
                // Store both x and scale in context for backward
                return new ForwardResult(IDoubleVector.of(out),
                    new double[][]{xd.clone(), new double[]{scale}});
            }

            @Override
            protected IDoubleVector[] backward(IDoubleVector gradOutput, Object ctx) {
                double[][] saved = (double[][]) ctx;
                double[] xd = saved[0];
                double scale = saved[1][0];
                double[] g = gradOutput.getData();
                int n = g.length;
                // y = x * scale, so dy/dx = scale * I, dL/dx = g * scale
                double[] dX = new double[n];
                for (int i = 0; i < n; i++) dX[i] = g[i] * scale;
                return new IDoubleVector[]{IDoubleVector.of(dX)};
            }
        };

        IDiffVector x = AD.vector(new double[]{2.0, 3.0, 5.0});
        IDiffVector y = op.apply(x);
        IDiffVector loss = y.sum();
        loss.backward();

        assertNotNull(x.getGradient());
        double scale = 1.0 / 10.0; // sum = 2+3+5 = 10
        assertArrayEquals(new double[]{scale, scale, scale}, x.getGradient().getData(), TOL);
    }

    @Test
    void testMultipleForwardCallsNoCrossContamination() {
        CustomOp scale = new CustomOp() {
            @Override
            protected ForwardResult forward(IDoubleVector[] inputs) {
                double[] x = inputs[0].getData();
                double f = inputs[1].getData()[0];
                double[] out = new double[x.length];
                for (int i = 0; i < x.length; i++) out[i] = x[i] * f;
                return new ForwardResult(IDoubleVector.of(out), Double.valueOf(f));
            }

            @Override
            protected IDoubleVector[] backward(IDoubleVector gradOutput, Object ctx) {
                double f = ((Double) ctx).doubleValue();
                double[] g = gradOutput.getData();
                double[] dX = new double[g.length];
                double dF = 0;
                // Note: dF = sum(inputs * gradOutput)... but we'd need inputs in ctx for that.
                // This test just checks no crash / no cross-contamination.
                for (int i = 0; i < g.length; i++) dX[i] = g[i] * f;
                return new IDoubleVector[]{IDoubleVector.of(dX), IDoubleVector.of(new double[]{dF})};
            }
        };

        IDiffVector x1 = AD.vector(new double[]{1, 2});
        IDiffVector x2 = AD.vector(new double[]{3, 4});
        IDiffVector f1 = AD.vector(2.0);
        IDiffVector f2 = AD.vector(3.0);

        // Call 1: x1 * 2.0 = [2, 4], backward expects *2 in dX
        IDiffVector y1 = scale.apply(x1, f1);
        y1.sum().backward();
        assertArrayEquals(new double[]{2, 2}, x1.getGradient().getData(), TOL);

        // Call 2: x2 * 3.0 = [9, 12], backward expects *3 in dX
        IDiffVector y2 = scale.apply(x2, f2);
        y2.sum().backward();
        assertArrayEquals(new double[]{3, 3}, x2.getGradient().getData(), TOL);
    }

    @Test
    void testGradientCorrectnessNumerical() {
        CustomOp cube = new CustomOp() {
            @Override
            protected ForwardResult forward(IDoubleVector[] inputs) {
                double[] x = inputs[0].getData();
                double[] out = new double[x.length];
                for (int i = 0; i < x.length; i++) out[i] = x[i] * x[i] * x[i];
                return new ForwardResult(IDoubleVector.of(out), inputs[0].copy());
            }

            @Override
            protected IDoubleVector[] backward(IDoubleVector gradOutput, Object ctx) {
                IDoubleVector x = (IDoubleVector) ctx;
                double[] g = gradOutput.getData();
                double[] xd = x.getData();
                double[] dX = new double[xd.length];
                for (int i = 0; i < dX.length; i++) dX[i] = 3.0 * xd[i] * xd[i] * g[i];
                return new IDoubleVector[]{IDoubleVector.of(dX)};
            }
        };

        IDiffVector x = AD.vector(new double[]{0.5, 1.0, 1.5});
        IDiffVector y = cube.apply(x);
        IDiffVector loss = y.sum();
        loss.backward();

        double[] expected = {3.0 * 0.5 * 0.5, 3.0 * 1.0 * 1.0, 3.0 * 1.5 * 1.5};
        assertArrayEquals(expected, x.getGradient().getData(), TOL);
    }

    @Test
    void testThroughADOopFactory() {
        CustomOp negate = new CustomOp() {
            @Override
            protected ForwardResult forward(IDoubleVector[] inputs) {
                double[] x = inputs[0].getData();
                double[] out = new double[x.length];
                for (int i = 0; i < x.length; i++) out[i] = -x[i];
                return new ForwardResult(IDoubleVector.of(out), null);
            }

            @Override
            protected IDoubleVector[] backward(IDoubleVector gradOutput, Object ctx) {
                double[] g = gradOutput.getData();
                double[] dX = new double[g.length];
                for (int i = 0; i < g.length; i++) dX[i] = -g[i];
                return new IDoubleVector[]{IDoubleVector.of(dX)};
            }
        };

        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        IDiffVector y = AD.op(negate, x);
        IDiffVector loss = y.sum();
        loss.backward();

        assertArrayEquals(new double[]{-1, -1, -1}, x.getGradient().getData(), TOL);
    }
}
