package com.yishape.lab.math.autodiff.support;

import java.util.List;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.compute.DoubleFlatGemm;

/**
 * Gradient scaler for Automatic Mixed Precision (AMP) training.
 *
 * <p>Prevents underflow in FP16 gradients by scaling the loss before backward()
 * and unscaling gradients before the optimizer step. Dynamically adjusts the
 * scale factor based on overflow detection.</p>
 *
 * <h3>Scaling algorithm</h3>
 * <ol>
 *   <li>Scale loss: {@code scaledLoss = loss * scaleFactor}</li>
 *   <li>Backward pass produces scaled gradients</li>
 *   <li>Unscale: {@code grad = grad / scaleFactor}</li>
 *   <li>If no NaN/Inf in gradients: grow scale (×growthFactor)</li>
 *   <li>If NaN/Inf found: backoff scale (×backoffFactor), skip optimizer step</li>
 * </ol>
 *
 * <p>Numerical operations in {@link #unscale(List)} use the
 * GPU→HPC→SIMD→SISD fallback chain via {@link DoubleVectorComputer}.</p>
 */
public class GradScaler {

    public static final double MAX_SCALE = 65536.0;
    public static final double MIN_SCALE = 1e-8;

    // In-place scaling uses DoubleFlatGemm.fusedDaxpyInPlace (HPC → SISD chain).

    private double scaleFactor;
    private final double growthFactor;
    private final double backoffFactor;
    private final int growthInterval;
    private int stepCount;

    /**
     * Create a GradScaler with the default initial scale of 2^16.
     */
    public GradScaler() {
        this(65536.0, 2.0, 0.5, 2000);
    }

    /**
     * @param initScale       initial scale factor
     * @param growthFactor    multiplier on successful steps (e.g. 2.0)
     * @param backoffFactor   multiplier on overflow (e.g. 0.5)
     * @param growthInterval  steps between growth attempts
     */
    public GradScaler(double initScale, double growthFactor,
            double backoffFactor, int growthInterval) {
        this.scaleFactor = Math.min(Math.max(initScale, MIN_SCALE), MAX_SCALE);
        this.growthFactor = growthFactor;
        this.backoffFactor = backoffFactor;
        this.growthInterval = growthInterval;
        this.stepCount = 0;
    }

    /** @return the current scale factor */
    public double getScaleFactor() {
        return scaleFactor;
    }

    /** Scale a loss value for the backward pass. */
    public double scale(double loss) {
        return loss * scaleFactor;
    }

    /** Scale a loss tensor. Returns a new scaled tensor. */
    public IDiffTensor scale(IDiffTensor loss) {
        return loss.mul(scaleFactor);
    }

    /** @return steps since last update. */
    public int getStepsSinceUpdate() {
        return stepCount;
    }

    /**
     * Unscale gradients in-place using GPU→HPC→SIMD→SISD dispatch.
     * Multiplies each gradient array by 1/scaleFactor.
     */
    public void unscale(List<IDiffTensor> params) {
        double invScale = 1.0 / scaleFactor;
        for (IDiffTensor p : params) {
            if (!(p instanceof RereDiffTensor rp)) continue;
            double[] g = rp.gradData();
            if (g != null && g.length > 0) {
                // In-place scale via fused Daxpy: y = a*x + b*y with a=0, b=invScale → y = invScale * y.
                // Dispatches HPC (Rust FFM) → SISD, no temporary array allocation.
                DoubleFlatGemm.fusedDaxpyInPlace(0, g, invScale, g);
            }
        }
    }

    /**
     * Check if any gradient contains NaN or Inf.
     * Uses SISD — NaN/Inf detection has no GPU/SIMD equivalent (bit-pattern check,
     * not arithmetic). JIT auto-vectorizes the comparison loop.
     */
    public boolean hasNanOrInf(List<IDiffTensor> params) {
        for (IDiffTensor p : params) {
            if (!(p instanceof RereDiffTensor rp)) continue;
            double[] g = rp.gradData();
            if (g == null) continue;
            for (int i = 0; i < g.length; i++) {
                if (Double.isNaN(g[i]) || Double.isInfinite(g[i])) return true;
            }
        }
        return false;
    }

    /**
     * Unscale and check in one fused pass (avoids iterating gradients twice).
     * Uses SISD — the fused unscale+check semantic has no GPU/SIMD equivalent.
     */
    public boolean unscaleAndCheck(List<IDiffTensor> params) {
        boolean hasOverflow = false;
        double invScale = 1.0 / scaleFactor;
        for (IDiffTensor p : params) {
            if (!(p instanceof RereDiffTensor rp)) continue;
            double[] g = rp.gradData();
            if (g == null) continue;
            for (int i = 0; i < g.length; i++) {
                double v = g[i] * invScale;
                if (Double.isNaN(v) || Double.isInfinite(v)) {
                    hasOverflow = true;
                }
                g[i] = v;
            }
        }
        return hasOverflow;
    }

    /**
     * Update the scale factor based on overflow status.
     * Call after {@link #unscale(List)} or {@link #unscaleAndCheck(List)}.
     *
     * @param hadOverflow true if NaN/Inf was detected in gradients
     */
    public void update(boolean hadOverflow) {
        if (hadOverflow) {
            // Backoff: reduce scale, reset counter
            scaleFactor = Math.max(scaleFactor * backoffFactor, MIN_SCALE);
            stepCount = 0;
        } else {
            stepCount++;
            if (stepCount >= growthInterval) {
                scaleFactor = Math.min(scaleFactor * growthFactor, MAX_SCALE);
                stepCount = 0;
            }
        }
    }

    /** Get the number of steps since last growth/backoff. */
    public int getStepCount() {
        return stepCount;
    }

    /** Set step count (for testing). */
    public void setStepCount(int count) {
        this.stepCount = count;
    }

    /** Set scale factor directly (for testing). Clamped to [MIN_SCALE, MAX_SCALE]. */
    public void setScaleFactor(double factor) {
        this.scaleFactor = Math.min(Math.max(factor, MIN_SCALE), MAX_SCALE);
    }
}
