package com.yishape.lab.math.autodiff.impl;

import java.util.Arrays;

/**
 * Mixed-precision differentiable tensor: FP32 primal storage, FP64 gradient accumulation.
 *
 * <h3>Master Weight Pattern</h3>
 * The FP32 {@code floatValue} array is the "master" (high-precision) copy of the weights.
 * The FP64 {@code value} (inherited from {@link RereDiffTensor}) is used for gradient
 * computation. Before the forward pass, call {@link #syncFloatToDouble()} to copy the
 * master FP32 weights into the FP64 value buffer. After the optimizer step, call
 * {@link #syncDoubleToFloat()} to update the master copy.
 *
 * <h3>Memory Savings</h3>
 * Halves parameter memory vs pure FP64 (8→4 bytes per element). Gradient accumulation
 * remains in FP64 for numerical stability.
 */
public class FloatDiffTensor extends RereDiffTensor {

    private static final long serialVersionUID = 1L;
    /** Master weight in FP32 precision. */
    private float[] floatValue;

    public FloatDiffTensor(float[] data, int... shape) {
        super(floatToDouble(data), shape);
        this.floatValue = data.clone();
    }

    /**
     * Widening conversion: float[] -> double[].
     * This is a type representation change (memory layout), not an arithmetic operation.
     * No GPU/HPC/SIMD equivalent exists for float[]→double[] memory widening.
     * JIT auto-vectorizes this loop via x86 VCVTPS2PD / AArch64 FCVTL.
     */
    private static double[] floatToDouble(float[] data) {
        double[] d = new double[data.length];
        for (int i = 0; i < data.length; i++) d[i] = data[i];
        return d;
    }

    /**
     * Narrowing conversion: double[] -> float[].
     * Same as {@link #floatToDouble(float[])}: type conversion, not arithmetic.
     * JIT auto-vectorizes via x86 VCVTPD2PS / AArch64 FCVTN.
     */
    private static float[] doubleToFloat(double[] data) {
        float[] f = new float[data.length];
        for (int i = 0; i < data.length; i++) f[i] = (float) data[i];
        return f;
    }

    // ---- Master weight sync ----

    // C16: ThreadLocal buffer to avoid repeated allocation on every syncFloatToDouble call
    private static final ThreadLocal<double[]> SYNC_BUF = ThreadLocal.withInitial(() -> new double[0]);

    /**
     * Copy master FP32 weights into the FP64 value buffer (for forward pass).
     * Must be called before forward when using master weight optimization.
     * This is a widening type conversion (float→double), not arithmetic — JIT
     * auto-vectorizes the loop. No GPU/HPC/SIMD equivalent exists.
     */
    public void syncFloatToDouble() {
        double[] dv = SYNC_BUF.get();
        if (dv.length != floatValue.length) {
            dv = new double[floatValue.length];
            SYNC_BUF.set(dv);
        }
        for (int i = 0; i < dv.length; i++) dv[i] = floatValue[i];
        setValue(new com.yishape.lab.math.linalg.tensor.RereDoubleTensor(dv, shape()));
    }

    /**
     * Copy FP64 weights back to master FP32 storage (after optimizer step).
     * Must be called after optimizer.step() when using master weight optimization.
     * This is a narrowing type conversion (double→float), not arithmetic — JIT
     * auto-vectorizes the loop. No GPU/HPC/SIMD equivalent exists.
     */
    public void syncDoubleToFloat() {
        double[] dv = value().toDoubleArray();
        for (int i = 0; i < floatValue.length; i++) floatValue[i] = (float) dv[i];
    }

    /** @return a copy of the FP32 master weight array */
    public float[] getFloatData() {
        return floatValue.clone();
    }

    /**
     * Replace the master FP32 data. Does NOT sync to FP64 value — call
     * {@link #syncFloatToDouble()} separately if needed.
     */
    public void setFloatData(float[] data) {
        if (data.length != floatValue.length) {
            throw new IllegalArgumentException(
                "Data length " + data.length + " != current " + floatValue.length);
        }
        this.floatValue = data.clone();
    }
}
