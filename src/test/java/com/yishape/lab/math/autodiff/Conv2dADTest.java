package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Gradient checks for IDiffTensor.conv2d().
 */
public class Conv2dADTest {

    private static final double TOL = 1e-6;

    /**
     * Simple conv2d: N=1, C=1, H=3, W=3, outC=1, kH=2, kW=2, stride=1, pad=0.
     * Use a very small numerical gradient check.
     */
    @Test
    void testConv2dGradientSmall() {
        // Input [1,1,3,3]
        double[] xData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        RereDiffTensor x = new RereDiffTensor(xData, 1, 1, 3, 3);
        x.setRequiresGrad(true);

        // Weight [1,1,2,2] — identity-like kernel
        double[] wData = {1, 0, 0, 1};
        RereDiffTensor w = new RereDiffTensor(wData, 1, 1, 2, 2);
        w.setRequiresGrad(true);

        // Bias [1]
        RereDiffTensor b = new RereDiffTensor(new double[]{0.5}, 1);
        b.setRequiresGrad(true);

        // Forward
        IDiffTensor out = x.conv2d(w, b, 1, 0, 1);
        // Output shape: [1,1,2,2]
        assertArrayEquals(new int[]{1, 1, 2, 2}, out.shape());

        // Sum loss
        IDiffTensor loss = out.sum();
        loss.backward();

        // Check gradients exist (non-zero for at least some elements)
        double[] xGrad = x.gradData();
        double[] wGrad = w.gradData();
        double[] bGrad = b.gradData();

        assertNotNull(xGrad);
        assertNotNull(wGrad);
        assertNotNull(bGrad);

        // Bias gradient = sum of d_output = sum of ones = outH * outW = 4
        assertEquals(4.0, bGrad[0], TOL);

        // Weight gradients should be non-trivial
        double wSum = 0;
        for (double v : wGrad) wSum += v;
        assertTrue(Math.abs(wSum) > 0, "Weight gradients should be non-zero");
    }

    @Test
    void testConv2dNumericalGradient() {
        // Small 1x1x3x3 input, 2x2 kernel, verify numerical gradient via finite diff
        double eps = 1e-6;
        double[] xData = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5};
        RereDiffTensor x = new RereDiffTensor(xData.clone(), 1, 1, 3, 3);
        x.setRequiresGrad(true);

        double[] wData = {0.2, -0.1, 0.3, 0.1};
        RereDiffTensor w = new RereDiffTensor(wData, 1, 1, 2, 2);
        w.setRequiresGrad(true);

        RereDiffTensor b = new RereDiffTensor(new double[]{0.0}, 1);

        // Analytical gradient
        IDiffTensor out = x.conv2d(w, b, 1, 0, 1);
        out.sum().backward();
        double[] analGrad = x.gradData().clone();

        // Numerical gradient for input[0] (first element)
        double orig = xData[0];
        xData[0] = orig + eps;
        RereDiffTensor xPlus = new RereDiffTensor(xData, 1, 1, 3, 3);
        double lossPlus = xPlus.conv2d(w, b, 1, 0, 1).sum().item();

        xData[0] = orig - eps;
        RereDiffTensor xMinus = new RereDiffTensor(xData, 1, 1, 3, 3);
        double lossMinus = xMinus.conv2d(w, b, 1, 0, 1).sum().item();

        double numGrad0 = (lossPlus - lossMinus) / (2.0 * eps);
        assertEquals(numGrad0, analGrad[0], 1e-4, "Numerical gradient mismatch at input[0]");
    }

    @Test
    void testConv2dStride2() {
        // Input [1,1,4,4], kernel [1,1,2,2], stride=2, pad=0 → output [1,1,2,2]
        double[] xData = new double[16];
        for (int i = 0; i < 16; i++) xData[i] = i + 1;
        RereDiffTensor x = new RereDiffTensor(xData, 1, 1, 4, 4);
        x.setRequiresGrad(true);

        double[] wData = {1, 0, 0, 1};
        RereDiffTensor w = new RereDiffTensor(wData, 1, 1, 2, 2);
        w.setRequiresGrad(true);

        IDiffTensor out = x.conv2d(w, null, 2, 0, 1);
        assertArrayEquals(new int[]{1, 1, 2, 2}, out.shape());

        out.sum().backward();

        // Check no NaN gradients
        for (double v : x.gradData()) assertFalse(Double.isNaN(v));
        for (double v : w.gradData()) assertFalse(Double.isNaN(v));
    }

    @Test
    void testConv2dPadding() {
        // Input [1,1,2,2], kernel [1,1,3,3], padding=1, stride=1 → output [1,1,2,2]
        double[] xData = {1, 2, 3, 4};
        RereDiffTensor x = new RereDiffTensor(xData, 1, 1, 2, 2);
        x.setRequiresGrad(true);

        double[] wData = new double[9];
        for (int i = 0; i < 9; i++) wData[i] = 0.1 * (i + 1);
        RereDiffTensor w = new RereDiffTensor(wData, 1, 1, 3, 3);
        w.setRequiresGrad(true);

        RereDiffTensor b = new RereDiffTensor(new double[]{0.2}, 1);
        b.setRequiresGrad(true);

        IDiffTensor out = x.conv2d(w, b, 1, 1, 1);
        assertArrayEquals(new int[]{1, 1, 2, 2}, out.shape());

        out.sum().backward();

        for (double v : x.gradData()) assertFalse(Double.isNaN(v));
        for (double v : w.gradData()) assertFalse(Double.isNaN(v));
        assertFalse(Double.isNaN(b.gradData()[0]));
    }

    @Test
    void testConv2dWeightGradientNumerical() {
        double eps = 1e-6;
        double[] xData = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5};
        RereDiffTensor x = new RereDiffTensor(xData.clone(), 1, 1, 3, 3);

        double[] wData = {0.2, -0.1, 0.3, 0.1};
        RereDiffTensor w = new RereDiffTensor(wData.clone(), 1, 1, 2, 2);
        w.setRequiresGrad(true);

        RereDiffTensor b = new RereDiffTensor(new double[]{0.0}, 1);

        IDiffTensor out = x.conv2d(w, b, 1, 0, 1);
        out.sum().backward();
        double analGradW0 = w.gradData()[0];

        // Numerical: perturb weight[0]
        wData[0] += eps;
        RereDiffTensor wPlus = new RereDiffTensor(wData, 1, 1, 2, 2);
        double lossPlus = x.conv2d(wPlus, b, 1, 0, 1).sum().item();

        wData[0] -= 2.0 * eps;
        RereDiffTensor wMinus = new RereDiffTensor(wData, 1, 1, 2, 2);
        double lossMinus = x.conv2d(wMinus, b, 1, 0, 1).sum().item();

        double numGradW0 = (lossPlus - lossMinus) / (2.0 * eps);
        assertEquals(numGradW0, analGradW0, 1e-4, "Numerical gradient mismatch at weight[0]");
    }
}
