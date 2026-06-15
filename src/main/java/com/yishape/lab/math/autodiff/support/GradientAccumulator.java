package com.yishape.lab.math.autodiff.support;

import java.util.ArrayList;
import java.util.List;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Gradient accumulator for gradient accumulation across multiple mini-batches.
 *
 * <p>Accumulates gradients over N steps before performing an optimizer step.
 * Useful when GPU memory is limited and the effective batch size must be larger
 * than what fits in memory.
 *
 * <h3>Usage pattern</h3>
 * <ol>
 *   <li>Create accumulator: {@code GradientAccumulator acc = new GradientAccumulator(params, 4);}</li>
 *   <li>Each micro-step: call {@code loss.backward()} then {@code acc.step(loss);} — accumulates gradients</li>
 *   <li>After step triggers: gradients are scaled to mean. Call {@code optimizer.step()} then
 *       {@code acc.zeroGrads();} to use scaled gradients and reset for next accumulation cycle.</li>
 * </ol>
 *
 * <p>Thread-safety: not thread-safe. Use one accumulator per training thread.</p>
 */
public class GradientAccumulator {
    private final List<IDiffTensor> params;
    private final int accumulationSteps;
    private int currentStep;

    /**
     * @param params            model parameters to accumulate gradients for
     * @param accumulationSteps number of steps to accumulate before returning true from {@link #shouldStep()}
     */
    public GradientAccumulator(List<IDiffTensor> params, int accumulationSteps) {
        if (accumulationSteps <= 0) throw new IllegalArgumentException("accumulationSteps must be > 0");
        this.params = new ArrayList<>(params);
        this.accumulationSteps = accumulationSteps;
        this.currentStep = 0;
    }

    /**
     * @return number of accumulation steps configured
     */
    public int getAccumulationSteps() {
        return accumulationSteps;
    }

    /**
     * @return current step count (0 to accumulationSteps-1)
     */
    public int getCurrentStep() {
        return currentStep;
    }

    /**
     * Record one accumulation step. Call backward() on your loss before this.
     * Scales accumulated gradients by 1/accumulationSteps so the effective
     * gradient is the mean over all accumulation steps.
     *
     * @param loss the loss tensor (backward must be called BEFORE this method)
     */
    public void step(IDiffTensor loss) {
        step(loss, false);
    }

    /**
     * Record one accumulation step with optional optimizer step trigger.
     *
     * <p>When accumulation steps complete, scales gradients by 1/N to compute the mean.
     * <b>Gradients are NOT zeroed by this method</b> — the caller must invoke
     * {@link #zeroGrads()} after the optimizer step to prevent gradient leakage into
     * the next accumulation cycle.</p>
     *
     * @param loss      the loss tensor (backward must be called BEFORE this method)
     * @param doStep    if true, also triggers the accumulation cycle regardless of step count
     */
    public void step(IDiffTensor loss, boolean doStep) {
        currentStep++;
        if (doStep || currentStep >= accumulationSteps) {
            // Scale accumulated gradients by 1/N to compute the mean
            double scale = 1.0 / accumulationSteps;
            for (IDiffTensor p : params) {
                if (!(p instanceof RereDiffTensor rp)) continue;
                double[] g = rp.gradData();
                if (g != null && g.length > 0) {
                    // In-place: g[i] *= scale using binaryOperate for acceleration
                    com.yishape.lab.math.compute.DoubleVectorComputer vc =
                        new com.yishape.lab.math.compute.DoubleVectorComputer();
                    double[] scaled = vc.binaryOperate(g, scale,
                        com.yishape.lab.math.compute.ops.BinaryOperation.MULTIPLY);
                    System.arraycopy(scaled, 0, g, 0, g.length);
                }
            }
            currentStep = 0;
        }
    }

    /**
     * @return true if enough steps have been accumulated to perform an optimizer step.
     *         Equivalent to {@code getCurrentStep() >= getAccumulationSteps()}.
     */
    public boolean shouldStep() {
        return currentStep >= accumulationSteps;
    }

    /**
     * Reset accumulation counter (without scaling gradients).
     * Call this if you want to discard accumulated gradients and start fresh.
     */
    public void reset() {
        currentStep = 0;
    }

    /**
     * Zero all accumulated gradients without performing an optimizer step.
     * Useful for cleanup or when an exception occurs during accumulation.
     */
    public void zeroGrads() {
        for (IDiffTensor p : params) {
            RereDiffTensor rp = (RereDiffTensor) p;
            rp.zeroGradient();
        }
        currentStep = 0;
    }
}
