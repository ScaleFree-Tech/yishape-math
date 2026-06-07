package com.yishape.lab.math.autodiff;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import com.yishape.lab.math.autodiff.impl.FloatDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.support.AutocastContext;
import com.yishape.lab.math.autodiff.support.GradScaler;

/**
 * Tests for AMP / Mixed Precision Training infrastructure:
 * FloatDiffTensor, GradScaler, AutocastContext.
 */
public class AmpMixedPrecisionTest {

    // ==================== FloatDiffTensor ====================

    @Test
    void testFloatDiffTensorCreation() {
        float[] data = {1.5f, 2.5f, 3.5f, 4.5f};
        FloatDiffTensor ft = new FloatDiffTensor(data, 4);
        assertArrayEquals(new int[]{4}, ft.shape());
        // FP64 value should match float→double conversion
        assertArrayEquals(new double[]{1.5, 2.5, 3.5, 4.5}, ft.toDoubleArray(), 1e-12);
    }

    @Test
    void testFloatDiffTensorGetFloatData() {
        float[] data = {1.0f, 2.0f, 3.0f};
        FloatDiffTensor ft = new FloatDiffTensor(data, 3);
        float[] retrieved = ft.getFloatData();
        assertArrayEquals(data, retrieved, 0.0f);
        // Should be a copy, not the same array (defensive copy)
        assertNotSame(data, retrieved);
    }

    @Test
    void testFloatDiffTensorSyncFloatToDouble() {
        float[] data = {1.0f, 2.0f, 3.0f};
        FloatDiffTensor ft = new FloatDiffTensor(data, 3);

        // Modify the master float data (via setFloatData)
        ft.setFloatData(new float[]{10.0f, 20.0f, 30.0f});
        // Sync: FP32→FP64
        ft.syncFloatToDouble();

        // FP64 value should now reflect the updated master
        assertArrayEquals(new double[]{10.0, 20.0, 30.0}, ft.toDoubleArray(), 1e-12);
    }

    @Test
    void testFloatDiffTensorSyncDoubleToFloat() {
        FloatDiffTensor ft = new FloatDiffTensor(new float[]{1.0f, 2.0f, 3.0f}, 3);

        // Modify the FP64 value directly (simulating optimizer step)
        double[] newVal = ft.toDoubleArray();
        newVal[0] = 100.0;
        newVal[1] = 200.0;
        newVal[2] = 300.0;
        ft.setValue(new com.yishape.lab.math.linalg.tensor.RereDoubleTensor(newVal, 3));

        // Sync: FP64→FP32
        ft.syncDoubleToFloat();

        float[] master = ft.getFloatData();
        assertEquals(100.0f, master[0], 1e-5f);
        assertEquals(200.0f, master[1], 1e-5f);
        assertEquals(300.0f, master[2], 1e-5f);
    }

    @Test
    void testFloatDiffTensorGradientAccumulation() {
        FloatDiffTensor ft = new FloatDiffTensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 4);
        ft.setRequiresGrad(true);

        // Forward: relu (keeps FP32 but gradients are FP64)
        IDiffTensor out = ft.relu().sum();
        out.backward();

        // Gradient should be in FP64 (1.0 for positive inputs)
        double[] grad = ft.gradData();
        assertArrayEquals(new double[]{1.0, 1.0, 1.0, 1.0}, grad, 1e-12);
    }

    // ==================== GradScaler ====================

    @Test
    void testGradScalerDefaultConstruction() {
        GradScaler scaler = new GradScaler();
        assertEquals(65536.0, scaler.getScaleFactor(), 1e-12);
        assertEquals(0, scaler.getStepsSinceUpdate());
    }

    @Test
    void testGradScalerScaleVector() {
        GradScaler scaler = new GradScaler(2.0, 2.0, 0.5, 10);
        RereDiffTensor loss = new RereDiffTensor(new double[]{1.0}, 1);

        // Scale the loss by the scale factor (2.0)
        IDiffTensor scaled = scaler.scale(loss);
        assertEquals(2.0, scaled.item(), 1e-12);
    }

    @Test
    void testGradScalerUnscale() {
        GradScaler scaler = new GradScaler(4.0, 2.0, 0.5, 10);
        RereDiffTensor param = new RereDiffTensor(new double[]{2.0, 4.0, 8.0}, 3);
        param.setRequiresGrad(true);
        // Manually set gradient as if backward happened
        param.setGradData(new double[]{8.0, 16.0, 32.0});

        List<IDiffTensor> params = List.of(param);
        scaler.unscale(params);

        // Gradients divided by scale=4: [2.0, 4.0, 8.0]
        assertArrayEquals(new double[]{2.0, 4.0, 8.0}, param.gradData(), 1e-12);
    }

    @Test
    void testGradScalerHasNanOrInf() {
        GradScaler scaler = new GradScaler();
        RereDiffTensor param = new RereDiffTensor(new double[]{1.0, 2.0}, 2);
        param.setRequiresGrad(true);

        // Normal gradients
        param.setGradData(new double[]{1.0, 2.0});
        assertFalse(scaler.hasNanOrInf(List.of(param)));

        // NaN gradient
        param.setGradData(new double[]{Double.NaN, 2.0});
        assertTrue(scaler.hasNanOrInf(List.of(param)));

        // Infinity gradient
        param.setGradData(new double[]{1.0, Double.POSITIVE_INFINITY});
        assertTrue(scaler.hasNanOrInf(List.of(param)));
    }

    @Test
    void testGradScalerUnscaleAndCheck() {
        GradScaler scaler = new GradScaler(2.0, 2.0, 0.5, 10);
        RereDiffTensor param = new RereDiffTensor(new double[]{1.0, 2.0}, 2);
        param.setRequiresGrad(true);
        param.setGradData(new double[]{4.0, 6.0});

        boolean overflow = scaler.unscaleAndCheck(List.of(param));
        assertFalse(overflow);
        assertArrayEquals(new double[]{2.0, 3.0}, param.gradData(), 1e-12);
    }

    @Test
    void testGradScalerUpdateGrowth() {
        GradScaler scaler = new GradScaler(2.0, 2.0, 0.5, 5);
        // 5 steps without overflow → scale doubles
        for (int i = 0; i < 5; i++) scaler.update(false);
        assertEquals(4.0, scaler.getScaleFactor(), 1e-12);

        // Next step should not grow yet (interval = 5)
        scaler.update(false);
        assertEquals(4.0, scaler.getScaleFactor(), 1e-12);

        // 5 more steps → scale doubles again
        for (int i = 0; i < 5; i++) scaler.update(false);
        assertEquals(8.0, scaler.getScaleFactor(), 1e-12);
    }

    @Test
    void testGradScalerUpdateBackoff() {
        GradScaler scaler = new GradScaler(8.0, 2.0, 0.5, 10);
        // Overflow detected → scale halves
        scaler.update(true);
        assertEquals(4.0, scaler.getScaleFactor(), 1e-12);

        // Step counter resets
        assertEquals(0, scaler.getStepsSinceUpdate());

        // Backoff again
        scaler.update(true);
        assertEquals(2.0, scaler.getScaleFactor(), 1e-12);
    }

    @Test
    void testGradScalerMinMaxScale() {
        GradScaler scaler = new GradScaler(1.0, 2.0, 0.5, 1);

        // Test max scale clamping (2^24)
        scaler = new GradScaler(GradScaler.MAX_SCALE, 2.0, 0.5, 1);
        scaler.update(false);
        assertEquals(GradScaler.MAX_SCALE, scaler.getScaleFactor(), 1e-12);

        // Test min scale clamping (2^-16)
        scaler = new GradScaler(GradScaler.MIN_SCALE, 2.0, 0.5, 1);
        scaler.update(true);
        assertEquals(GradScaler.MIN_SCALE, scaler.getScaleFactor(), 1e-12);
    }

    @Test
    void testGradScalerFullCycle() {
        // Simulate a full AMP training cycle
        GradScaler scaler = new GradScaler(4.0, 2.0, 0.5, 10);
        RereDiffTensor w = new RereDiffTensor(new double[]{1.0, 2.0, 3.0}, 3);
        w.setRequiresGrad(true);

        // Forward: simple computation
        IDiffTensor loss = w.square().sum();

        // Scale loss
        IDiffTensor scaledLoss = scaler.scale(loss);

        // Backward
        scaledLoss.backward();

        // Unscale and check
        boolean overflow = scaler.unscaleAndCheck(List.of(w));
        assertFalse(overflow);

        // Gradient should be original (unscaled): d/dw of w^2 = 2*w, scaled by 4, then unscaled
        // original grad = [2, 4, 6], scaled grad = [8, 16, 24], unscaled = [2, 4, 6]
        assertArrayEquals(new double[]{2.0, 4.0, 6.0}, w.gradData(), 1e-12);

        // Update scaler
        scaler.update(false);
        assertEquals(4.0, scaler.getScaleFactor(), 1e-12); // no change (interval not reached)
    }

    // ==================== AutocastContext ====================

    @Test
    void testAutocastIsActive() {
        assertFalse(AutocastContext.isActive());
        try (AutocastContext ctx = new AutocastContext()) {
            assertTrue(AutocastContext.isActive());
            assertEquals(1, AutocastContext.depth());
        }
        assertFalse(AutocastContext.isActive());
    }

    @Test
    void testAutocastNesting() {
        assertFalse(AutocastContext.isActive());
        try (AutocastContext outer = new AutocastContext()) {
            assertTrue(AutocastContext.isActive());
            assertEquals(1, AutocastContext.depth());

            try (AutocastContext inner = new AutocastContext()) {
                assertTrue(AutocastContext.isActive());
                assertEquals(2, AutocastContext.depth());
            }

            assertEquals(1, AutocastContext.depth());
            assertTrue(AutocastContext.isActive());
        }
        assertFalse(AutocastContext.isActive());
    }

    @Test
    void testADAutocastFactory() {
        assertFalse(AutocastContext.isActive());
        try (AutocastContext ctx = AD.autocast()) {
            assertTrue(AutocastContext.isActive());
        }
        assertFalse(AutocastContext.isActive());
    }

    @Test
    void testAutocastMaybeCast() {
        RereDiffTensor tensor = new RereDiffTensor(new double[]{1.0, 2.0, 3.0}, 3);

        // Without autocast: returns same tensor
        assertSame(tensor, AutocastContext.maybeCast(tensor));

        // With autocast: converts to FloatDiffTensor
        try (AutocastContext ctx = new AutocastContext()) {
            IDiffTensor cast = AutocastContext.maybeCast(tensor);
            assertInstanceOf(FloatDiffTensor.class, cast);
            assertArrayEquals(tensor.toDoubleArray(), cast.toDoubleArray(), 1e-5);
        }
    }

    // ==================== Integration: AMP + GradScaler + Autocast ====================

    @Test
    void testAmpFullWorkflow() {
        // Weight parameter
        FloatDiffTensor w = new FloatDiffTensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, 4);
        w.setRequiresGrad(true);

        GradScaler scaler = new GradScaler(2.0, 2.0, 0.5, 10);

        // Simulate training step
        try (AutocastContext ctx = AD.autocast()) {
            // Forward pass (FP32 weights used, but FP64 gradient accumulation)
            w.syncFloatToDouble();
            IDiffTensor prediction = w.mul(2.0);
            IDiffTensor loss = prediction.sum();

            // Scale and backward
            IDiffTensor scaledLoss = scaler.scale(loss);
            scaledLoss.backward();

            // Unscale gradients
            boolean overflow = scaler.unscaleAndCheck(List.of(w));
            assertFalse(overflow);

            // d/dw of sum(w*2) scaled: original grad=2, scaled=4, unscaled=2
            assertArrayEquals(new double[]{2.0, 2.0, 2.0, 2.0}, w.gradData(), 1e-12);

            // Simulate optimizer step: w -= lr * grad
            // In real code, this would be optimizer.step()
            double[] wData = w.value().toDoubleArray();
            wData[0] -= 0.1 * w.gradData()[0];
            wData[1] -= 0.1 * w.gradData()[1];
            wData[2] -= 0.1 * w.gradData()[2];
            wData[3] -= 0.1 * w.gradData()[3];
            w.setValue(new com.yishape.lab.math.linalg.tensor.RereDoubleTensor(wData, 4));

            // Sync updated FP64 weights back to FP32 master
            w.syncDoubleToFloat();
        }

        scaler.update(false);

        // Verify master weights updated correctly: 1-0.2=0.8, 2-0.2=1.8, etc.
        float[] master = w.getFloatData();
        assertEquals(0.8f, master[0], 1e-5f);
        assertEquals(1.8f, master[1], 1e-5f);
        assertEquals(2.8f, master[2], 1e-5f);
        assertEquals(3.8f, master[3], 1e-5f);
    }

    @Test
    void testGradScalerOverflowProtection() {
        // If overflow is detected, the optimizer step should be skipped
        GradScaler scaler = new GradScaler(8.0, 2.0, 0.5, 10);
        RereDiffTensor w = new RereDiffTensor(new double[]{1.0, 2.0}, 2);
        w.setRequiresGrad(true);
        w.setGradData(new double[]{Double.NaN, 1.0});

        boolean overflow = scaler.unscaleAndCheck(List.of(w));
        assertTrue(overflow);

        // Scale should decrease after overflow
        double prevScale = scaler.getScaleFactor();
        scaler.update(true);
        assertTrue(scaler.getScaleFactor() < prevScale);
    }
}
