package com.yishape.lab.math.autodiff.support;

import com.yishape.lab.math.autodiff.IDiffTensor;

/**
 * AutoCloseable context manager for Automatic Mixed Precision (AMP).
 *
 * <p>While autocast is active, {@link #maybeCast(IDiffTensor)} returns
 * the tensor unchanged (FP32). Subclasses or extensions may override
 * {@code maybeCast} to downcast to FP16/BF16.</p>
 *
 * <p>Supports nesting — only the outermost context truly exits.</p>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * try (AutocastContext ctx = AD.autocast()) {
 *     // operations here run in autocast mode
 * }
 * }</pre>
 */
public class AutocastContext implements AutoCloseable {

    private static final ThreadLocal<Integer> DEPTH = ThreadLocal.withInitial(() -> 0);

    /** Returns true if currently inside an autocast context. */
    public static boolean isActive() {
        return DEPTH.get() > 0;
    }

    /** Returns the current nesting depth (0 = outside autocast). */
    public static int depth() {
        return DEPTH.get();
    }

    public AutocastContext() {
        DEPTH.set(DEPTH.get() + 1);
    }

    @Override
    public void close() {
        int d = DEPTH.get();
        if (d > 0) DEPTH.set(d - 1);
    }

    /**
     * Optionally cast a tensor to FP16/BF16 for the forward pass.
     * When autocast is active, converts to FloatDiffTensor.
     * Otherwise returns the tensor unchanged.
     */
    public static IDiffTensor maybeCast(IDiffTensor tensor) {
        if (!isActive() || tensor instanceof com.yishape.lab.math.autodiff.impl.FloatDiffTensor)
            return tensor;
        float[] fdata = new float[(int) tensor.totalSize()];
        double[] ddata = tensor.toDoubleArray();
        for (int i = 0; i < fdata.length; i++) fdata[i] = (float) ddata[i];
        return new com.yishape.lab.math.autodiff.impl.FloatDiffTensor(fdata, tensor.shape());
    }
}
