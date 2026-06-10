package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for fused loss ops: bceLoss, focalLoss, diceLoss.
 */
public class FusedLossOpsTest {

    @Test
    public void testBceLossPerfectPrediction() {
        IDiffTensor pred = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.bceLoss(target);
        double lossVal = loss.toDoubleArray()[0];
        assertTrue(lossVal < 1e-5, "Perfect prediction BCE should be near 0, got " + lossVal);
    }

    @Test
    public void testBceLossNonzeroLoss() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.9, 0.9}, 2);
        IDiffTensor target = AD.leafTensor(new double[]{0.0, 0.0}, 2);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.bceLoss(target);
        double lossVal = loss.toDoubleArray()[0];
        double expected = -Math.log(1.0 - 0.9);
        assertEquals(expected, lossVal, 0.01);
    }

    @Test
    public void testBceLossMixedLabels() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.8, 0.2}, 2);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0}, 2);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.bceLoss(target);
        double lossVal = loss.toDoubleArray()[0];
        double expected = (-Math.log(0.8) + -Math.log(0.8)) / 2.0;
        assertEquals(expected, lossVal, 0.01);
    }

    @Test
    public void testBceLossBackward() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.7, 0.3, 0.9, 0.1}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.bceLoss(target);
        loss.backward();
        assertNotNull(pred.grad());
        double[] g = pred.grad().toDoubleArray();
        assertEquals(4, g.length);
        for (int i = 0; i < 4; i++) {
            assertTrue(Double.isFinite(g[i]), "Gradient finite at " + i);
            assertNotEquals(0.0, g[i], 1e-12, "Gradient nonzero at " + i);
        }
    }

    @Test
    public void testBceLossTargetGradient() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.7, 0.3}, 2);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0}, 2);
        target.setRequiresGrad(true);
        IDiffTensor loss = pred.bceLoss(target);
        loss.backward();
        assertNotNull(target.grad(), "Target should receive gradient");
        double[] tg = target.grad().toDoubleArray();
        assertEquals(2, tg.length);
        for (double v : tg) assertTrue(Double.isFinite(v));
    }

    @Test
    public void testBceLossConstantTensor() {
        IDiffTensor pred = AD.constantTensor(new double[]{0.8, 0.2}, 2);
        IDiffTensor target = AD.constantTensor(new double[]{1.0, 0.0}, 2);
        IDiffTensor loss = pred.bceLoss(target);
        assertTrue(loss.toDoubleArray()[0] > 0);
    }

    @Test
    public void testBceLoss2DBatch() {
        IDiffTensor pred = AD.leafTensor(new double[]{
            0.9, 0.1, 0.1,
            0.1, 0.9, 0.1
        }, 2, 3);
        IDiffTensor target = AD.leafTensor(new double[]{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        }, 2, 3);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.bceLoss(target);
        assertTrue(loss.toDoubleArray()[0] < 0.3);
        loss.backward();
        assertNotNull(pred.grad());
    }

    @Test
    public void testBceLossNumericalStability() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.9999, 0.0001, 0.5}, 3);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 0.5}, 3);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.bceLoss(target);
        double lossVal = loss.toDoubleArray()[0];
        assertFalse(Double.isNaN(lossVal));
        assertTrue(Double.isFinite(lossVal));
        loss.backward();
        double[] g = pred.grad().toDoubleArray();
        for (int i = 0; i < g.length; i++) {
            assertFalse(Double.isNaN(g[i]));
            assertTrue(Double.isFinite(g[i]));
        }
    }

    @Test
    public void testBceLossShape() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.7, 0.3, 0.9, 0.1}, 2, 2);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 2, 2);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.bceLoss(target);
        assertArrayEquals(new int[]{1}, loss.shape());
        assertEquals(1, loss.totalSize());
    }

    @Test
    public void testFocalLossPerfectPrediction() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.99, 0.01, 0.99}, 3);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 1.0}, 3);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.focalLoss(target, 0.25, 2.0);
        assertTrue(loss.toDoubleArray()[0] < 0.01);
    }

    @Test
    public void testFocalLossDefaultParams() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.5}, 1);
        IDiffTensor target = AD.leafTensor(new double[]{1.0}, 1);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.focalLoss(target, 0.25, 2.0);
        double expected = 0.25 * Math.pow(0.5, 2.0) * (-Math.log(0.5));
        assertEquals(expected, loss.toDoubleArray()[0], 0.01);
    }

    @Test
    public void testFocalLossGammaZero() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.8}, 1);
        IDiffTensor target = AD.leafTensor(new double[]{1.0}, 1);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.focalLoss(target, 0.5, 0.0);
        assertEquals(0.5 * (-Math.log(0.8)), loss.toDoubleArray()[0], 0.01);
    }

    @Test
    public void testFocalLossAlphaOne() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.5}, 1);
        IDiffTensor target = AD.leafTensor(new double[]{1.0}, 1);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.focalLoss(target, 1.0, 2.0);
        assertEquals(Math.pow(0.5, 2.0) * (-Math.log(0.5)), loss.toDoubleArray()[0], 0.01);
    }

    @Test
    public void testFocalLossBackward() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.7, 0.3, 0.9, 0.1}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.focalLoss(target, 0.25, 2.0);
        loss.backward();
        assertNotNull(pred.grad());
        double[] g = pred.grad().toDoubleArray();
        for (int i = 0; i < 4; i++) assertTrue(Double.isFinite(g[i]));
    }

    @Test
    public void testFocalLossTargetGradient() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.7, 0.3}, 2);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0}, 2);
        target.setRequiresGrad(true);
        IDiffTensor loss = pred.focalLoss(target, 0.25, 2.0);
        loss.backward();
        assertNotNull(target.grad());
        for (double v : target.grad().toDoubleArray()) assertTrue(Double.isFinite(v));
    }

    @Test
    public void testFocalLossHighGamma() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.9}, 1);
        IDiffTensor target = AD.leafTensor(new double[]{1.0}, 1);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.focalLoss(target, 0.25, 5.0);
        double lossVal = loss.toDoubleArray()[0];
        assertFalse(Double.isNaN(lossVal));
        assertTrue(lossVal < 0.01);
    }

    @Test
    public void testFocalLossConstantTensor() {
        IDiffTensor pred = AD.constantTensor(new double[]{0.8, 0.2}, 2);
        IDiffTensor target = AD.constantTensor(new double[]{1.0, 0.0}, 2);
        assertTrue(pred.focalLoss(target, 0.25, 2.0).toDoubleArray()[0] > 0);
    }

    @Test
    public void testFocalLoss2DBatch() {
        IDiffTensor pred = AD.leafTensor(new double[]{
            0.9, 0.1, 0.1,
            0.1, 0.9, 0.1
        }, 2, 3);
        IDiffTensor target = AD.leafTensor(new double[]{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0
        }, 2, 3);
        target.setRequiresGrad(false);
        assertTrue(pred.focalLoss(target, 0.25, 2.0).toDoubleArray()[0] < 0.1);
    }

    @Test
    public void testFocalLossShape() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.5, 0.5, 0.5, 0.5}, 2, 2);
        IDiffTensor target = AD.leafTensor(new double[]{0.0, 1.0, 0.0, 1.0}, 2, 2);
        target.setRequiresGrad(false);
        assertArrayEquals(new int[]{1}, pred.focalLoss(target, 0.25, 2.0).shape());
    }

    @Test
    public void testFocalLossNumericalStability() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.9999, 0.0001, 0.5}, 3);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 0.5}, 3);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.focalLoss(target, 0.25, 3.0);
        assertFalse(Double.isNaN(loss.toDoubleArray()[0]));
        loss.backward();
        for (double v : pred.grad().toDoubleArray()) {
            assertFalse(Double.isNaN(v));
            assertTrue(Double.isFinite(v));
        }
    }

    @Test
    public void testDiceLossPerfectPrediction() {
        IDiffTensor pred = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        target.setRequiresGrad(false);
        assertEquals(0.0, pred.diceLoss(target, 1.0).toDoubleArray()[0], 1e-10);
    }

    @Test
    public void testDiceLossCompleteMismatch() {
        IDiffTensor pred = AD.leafTensor(new double[]{1.0, 1.0, 1.0, 1.0}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{0.0, 0.0, 0.0, 0.0}, 4);
        target.setRequiresGrad(false);
        assertEquals(1.0 - 1.0 / 5.0, pred.diceLoss(target, 1.0).toDoubleArray()[0], 1e-10);
    }

    @Test
    public void testDiceLossHalfOverlap() {
        IDiffTensor pred = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 0.0, 0.0}, 4);
        target.setRequiresGrad(false);
        assertEquals(1.0 - 2.0 / 3.0, pred.diceLoss(target, 0.0).toDoubleArray()[0], 1e-10);
    }

    @Test
    public void testDiceLossSmoothingEffect() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.5, 0.5}, 2);
        IDiffTensor target = AD.leafTensor(new double[]{0.0, 0.0}, 2);
        target.setRequiresGrad(false);
        double lossVal = pred.diceLoss(target, 1.0).toDoubleArray()[0];
        assertTrue(lossVal > 0);
        assertTrue(lossVal < 1.0);
    }

    @Test
    public void testDiceLossBackward() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.8, 0.2, 0.9, 0.1}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        target.setRequiresGrad(false);
        IDiffTensor loss = pred.diceLoss(target, 1.0);
        loss.backward();
        assertNotNull(pred.grad());
        for (double v : pred.grad().toDoubleArray()) assertTrue(Double.isFinite(v));
    }

    @Test
    public void testDiceLossTargetGradient() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.7, 0.3}, 2);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0}, 2);
        target.setRequiresGrad(true);
        IDiffTensor loss = pred.diceLoss(target, 1.0);
        loss.backward();
        assertNotNull(target.grad());
        for (double v : target.grad().toDoubleArray()) assertTrue(Double.isFinite(v));
    }

    @Test
    public void testDiceLossConstantTensor() {
        IDiffTensor pred = AD.constantTensor(new double[]{0.8, 0.2}, 2);
        IDiffTensor target = AD.constantTensor(new double[]{1.0, 0.0}, 2);
        double lossVal = pred.diceLoss(target, 1.0).toDoubleArray()[0];
        assertTrue(lossVal >= 0 && lossVal <= 1.0);
    }

    @Test
    public void testDiceLoss2DBatch() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.9, 0.1, 0.1, 0.9}, 2, 2);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 0.0, 1.0}, 2, 2);
        target.setRequiresGrad(false);
        double lossVal = pred.diceLoss(target, 1.0).toDoubleArray()[0];
        assertTrue(lossVal < 0.2);
        pred.diceLoss(target, 1.0).backward();
        assertNotNull(pred.grad());
    }

    @Test
    public void testDiceLossShape() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.5, 0.5, 0.5, 0.5}, 2, 2);
        IDiffTensor target = AD.leafTensor(new double[]{0.0, 0.0, 1.0, 1.0}, 2, 2);
        target.setRequiresGrad(false);
        assertArrayEquals(new int[]{1}, pred.diceLoss(target, 1.0).shape());
    }

    @Test
    public void testBceAndFocalWithGammaZero() {
        // focalLoss(alpha=0.5, gamma=0) = 0.5 * bceLoss (balanced alpha divides both branches)
        IDiffTensor pred = AD.leafTensor(new double[]{0.7, 0.3, 0.9, 0.1}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{1.0, 0.0, 1.0, 0.0}, 4);
        target.setRequiresGrad(false);
        double bceVal = pred.bceLoss(target).toDoubleArray()[0];
        double focalVal = pred.focalLoss(target, 0.5, 0.0).toDoubleArray()[0];
        assertEquals(bceVal * 0.5, focalVal, 1e-6);
    }

    @Test
    public void testAllLossesScalarOutput() {
        IDiffTensor pred = AD.leafTensor(new double[]{0.5, 0.5, 0.5, 0.5}, 4);
        IDiffTensor target = AD.leafTensor(new double[]{0.0, 1.0, 0.0, 1.0}, 4);
        target.setRequiresGrad(false);
        assertEquals(1, pred.bceLoss(target).totalSize());
        assertEquals(1, pred.focalLoss(target, 0.25, 2.0).totalSize());
        assertEquals(1, pred.diceLoss(target, 1.0).totalSize());
    }

    @Test
    public void testAllLossesGradientFlows() {
        IDiffTensor input = AD.leafTensor(new double[]{1.0, -1.0, 0.5, -0.5}, 1, 4);
        IDiffTensor weight = AD.leafTensor(new double[]{0.1, 0.2, 0.3, 0.4}, 4, 1);
        IDiffTensor bias = AD.leafTensor(new double[]{0.0}, 1);
        IDiffTensor logits = input.mmul(weight).add(bias);
        IDiffTensor probs = logits.sigmoid();
        IDiffTensor target = AD.leafTensor(new double[]{1.0}, 1, 1);
        target.setRequiresGrad(false);
        probs.bceLoss(target).backward();
        assertNotNull(weight.grad());
        assertNotEquals(0.0, weight.grad().toDoubleArray()[0], 1e-12);
    }
}
