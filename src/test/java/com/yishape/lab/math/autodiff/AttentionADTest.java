package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Gradient checks for IDiffTensor.scaledDotProductAttention().
 */
public class AttentionADTest {

    private static final double TOL = 1e-6;
    private static final double NUM_TOL = 1e-4;

    /**
     * Minimal attention: batch=1, seqQ=2, seqK=2, dk=2, dv=2.
     */
    @Test
    void testAttentionGradientExists() {
        // Q [1,2,2]
        double[] qData = {1, 0, 0, 1};
        RereDiffTensor q = new RereDiffTensor(qData, 1, 2, 2);
        q.setRequiresGrad(true);

        // K [1,2,2]
        double[] kData = {1, 0, 0, 1};
        RereDiffTensor k = new RereDiffTensor(kData, 1, 2, 2);
        k.setRequiresGrad(true);

        // V [1,2,2]
        double[] vData = {1, 2, 3, 4};
        RereDiffTensor v = new RereDiffTensor(vData, 1, 2, 2);
        v.setRequiresGrad(true);

        IDiffTensor out = q.scaledDotProductAttention(k, v, null, 0.0);
        assertArrayEquals(new int[]{1, 2, 2}, out.shape());

        out.sum().backward();

        assertNotNull(q.gradData());
        assertNotNull(k.gradData());
        assertNotNull(v.gradData());

        // All gradients should be non-NaN
        for (double val : q.gradData()) assertFalse(Double.isNaN(val));
        for (double val : k.gradData()) assertFalse(Double.isNaN(val));
        for (double val : v.gradData()) assertFalse(Double.isNaN(val));

        // At least some gradient values should be non-zero
        double qSum = 0, kSum = 0, vSum = 0;
        for (double val : q.gradData()) qSum += Math.abs(val);
        for (double val : k.gradData()) kSum += Math.abs(val);
        for (double val : v.gradData()) vSum += Math.abs(val);
        assertTrue(qSum > 0, "Q gradient should be non-zero");
        assertTrue(kSum > 0, "K gradient should be non-zero");
        assertTrue(vSum > 0, "V gradient should be non-zero");
    }

    /**
     * Verify analytical gradient is non-zero and in correct direction (same sign as numerical).
     */
    @Test
    void testAttentionQNumericalGradient() {
        double eps = 0.01; // large eps for reliable numerical gradient
        double[] qData = {0.5, 1.2, -0.3, 0.8};

        // Analytical gradient
        RereDiffTensor q = new RereDiffTensor(qData.clone(), 1, 2, 2);
        q.setRequiresGrad(true);
        RereDiffTensor k = new RereDiffTensor(new double[]{1.0, -0.5, 0.2, 0.7}, 1, 2, 2);
        RereDiffTensor v = new RereDiffTensor(new double[]{2.0, 1.0, 0.0, -1.0}, 1, 2, 2);
        q.scaledDotProductAttention(k, v, null, 0.0).sum().backward();
        double analGradQ0 = q.gradData()[0];

        // Numerical gradient: recompute forward with fresh tensors each time
        // Use brand-new data arrays each call to avoid any shared reference issues
        double lossPlus = scalarLoss(new double[]{qData[0] + eps, qData[1], qData[2], qData[3]});
        double lossMinus = scalarLoss(new double[]{qData[0] - eps, qData[1], qData[2], qData[3]});

        double numGradQ0 = (lossPlus - lossMinus) / (2.0 * eps);
        double lossDiff = lossPlus - lossMinus;
        System.out.println("Q: lossPlus=" + lossPlus + " lossMinus=" + lossMinus + " diff=" + lossDiff + " numGrad=" + numGradQ0 + " analGrad=" + analGradQ0);
        assertTrue(Math.abs(lossDiff) > 1e-8, "Forward must change with perturbation");
        assertTrue(Math.abs(analGradQ0) > 0.01, "Analytical gradient must be non-zero");
        // Numerical and analytical gradients should agree within finite difference tolerance
        assertEquals(numGradQ0, analGradQ0, 5e-3,
            "Numerical gradient mismatch at Q[0,0,0]");
    }

    /** Helper: compute scalar loss for given Q data. */
    private double scalarLoss(double[] qVals) {
        RereDiffTensor qt = new RereDiffTensor(qVals, 1, 2, 2);
        RereDiffTensor kt = new RereDiffTensor(new double[]{1.0, -0.5, 0.2, 0.7}, 1, 2, 2);
        RereDiffTensor vt = new RereDiffTensor(new double[]{2.0, 1.0, 0.0, -1.0}, 1, 2, 2);
        return qt.scaledDotProductAttention(kt, vt, null, 0.0).sum().item();
    }

    @Test
    void testAttentionVNumericalGradient() {
        double eps = 1e-5;
        double[] qData = {0.5, 1.2, -0.3, 0.8};
        RereDiffTensor q = new RereDiffTensor(qData.clone(), 1, 2, 2);

        double[] kData = {1.0, -0.5, 0.2, 0.7};
        RereDiffTensor k = new RereDiffTensor(kData.clone(), 1, 2, 2);

        double[] vDataOrig = {2.0, 1.0, 0.0, -1.0};
        RereDiffTensor v = new RereDiffTensor(vDataOrig.clone(), 1, 2, 2);
        v.setRequiresGrad(true);

        IDiffTensor out = q.scaledDotProductAttention(k, v, null, 0.0);
        out.sum().backward();
        double analGradV0 = v.gradData()[0];
        assertTrue(Math.abs(analGradV0) > 0.01, "Analytical gradient at V[0] should be non-zero, got " + analGradV0);

        double[] vPlusData = vDataOrig.clone();
        vPlusData[0] += eps;
        RereDiffTensor vPlus = new RereDiffTensor(vPlusData, 1, 2, 2);
        RereDiffTensor qFresh1 = new RereDiffTensor(qData.clone(), 1, 2, 2);
        RereDiffTensor kFresh1 = new RereDiffTensor(kData.clone(), 1, 2, 2);
        double lossPlus = qFresh1.scaledDotProductAttention(kFresh1, vPlus, null, 0.0).sum().item();

        double[] vMinusData = vDataOrig.clone();
        vMinusData[0] -= eps;
        RereDiffTensor vMinus = new RereDiffTensor(vMinusData, 1, 2, 2);
        RereDiffTensor qFresh2 = new RereDiffTensor(qData.clone(), 1, 2, 2);
        RereDiffTensor kFresh2 = new RereDiffTensor(kData.clone(), 1, 2, 2);
        double lossMinus = qFresh2.scaledDotProductAttention(kFresh2, vMinus, null, 0.0).sum().item();

        double numGradV0 = (lossPlus - lossMinus) / (2.0 * eps);
        System.out.println("V: lossPlus=" + lossPlus + " lossMinus=" + lossMinus + " diff=" + (lossPlus - lossMinus) + " numGrad=" + numGradV0 + " analGrad=" + analGradV0);
        assertEquals(numGradV0, analGradV0, 1e-3, "Numerical gradient mismatch at V[0,0,0]");
    }

    @Test
    void testAttentionWithMask() {
        // Causal mask: upper triangular = -inf
        double[] qData = {1, 0, 0, 1};
        RereDiffTensor q = new RereDiffTensor(qData, 1, 2, 2);
        q.setRequiresGrad(true);

        double[] kData = {1, 0, 0, 1};
        RereDiffTensor k = new RereDiffTensor(kData, 1, 2, 2);
        k.setRequiresGrad(true);

        double[] vData = {1, 2, 3, 4};
        RereDiffTensor v = new RereDiffTensor(vData, 1, 2, 2);
        v.setRequiresGrad(true);

        // Causal mask: allow only j <= i
        double[] maskData = {0, -1e9, 0, 0}; // [1,1,2,2] or [2,2] broadcast
        RereDiffTensor mask = new RereDiffTensor(maskData, 1, 1, 2, 2);

        IDiffTensor out = q.scaledDotProductAttention(k, v, mask, 0.0);
        out.sum().backward();

        for (double val : q.gradData()) assertFalse(Double.isNaN(val));
        for (double val : k.gradData()) assertFalse(Double.isNaN(val));
        for (double val : v.gradData()) assertFalse(Double.isNaN(val));
    }

    @Test
    void testAttentionDifferentSeqLengths() {
        // Q: batch=1, seqQ=3, dk=2
        // K: batch=1, seqK=4, dk=2
        // V: batch=1, seqK=4, dv=3
        double[] qData = {1, 0, 0, 1, 0.5, 0.5};
        RereDiffTensor q = new RereDiffTensor(qData, 1, 3, 2);
        q.setRequiresGrad(true);

        double[] kData = {1, 0, 0, 1, 0.5, 0.5, 0, 0};
        RereDiffTensor k = new RereDiffTensor(kData, 1, 4, 2);
        k.setRequiresGrad(true);

        double[] vData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        RereDiffTensor v = new RereDiffTensor(vData, 1, 4, 3);
        v.setRequiresGrad(true);

        IDiffTensor out = q.scaledDotProductAttention(k, v, null, 0.0);
        assertArrayEquals(new int[]{1, 3, 3}, out.shape());

        out.sum().backward();

        for (double val : q.gradData()) assertFalse(Double.isNaN(val));
        for (double val : k.gradData()) assertFalse(Double.isNaN(val));
        for (double val : v.gradData()) assertFalse(Double.isNaN(val));
    }

    @Test
    void testAttentionDropout() {
        double[] qData = {0.5, 1.2, -0.3, 0.8};
        RereDiffTensor q = new RereDiffTensor(qData, 1, 2, 2);
        q.setRequiresGrad(true);

        double[] kData = {1.0, -0.5, 0.2, 0.7};
        RereDiffTensor k = new RereDiffTensor(kData, 1, 2, 2);
        k.setRequiresGrad(true);

        double[] vData = {2.0, 1.0, 0.0, -1.0};
        RereDiffTensor v = new RereDiffTensor(vData, 1, 2, 2);
        v.setRequiresGrad(true);

        // With dropout=0.5, gradients should still be finite
        IDiffTensor out = q.scaledDotProductAttention(k, v, null, 0.5);
        out.sum().backward();

        for (double val : q.gradData()) assertFalse(Double.isNaN(val));
        for (double val : k.gradData()) assertFalse(Double.isNaN(val));
        for (double val : v.gradData()) assertFalse(Double.isNaN(val));
    }
}
