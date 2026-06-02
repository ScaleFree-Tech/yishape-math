package com.yishape.lab.math.compute.gpu;

/**
 * Four-gate GPU LayerNorm dispatch.
 */
public final class GpuLayerNorm {
    private GpuLayerNorm() {}

    public static double[] tryLayerNorm(double[] x, double[] gamma, double[] beta, int outer, int normDim, double eps) {
        if (x == null || gamma == null || beta == null || outer <= 0 || normDim <= 0) return null;
        if ((long) outer * normDim < GpuConfig.layernormMinElements()) return null;
        if (!GpuConfig.allowAttempts()) return null;
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return GpuOptionalRuntime.tryLayerNorm(x, gamma, beta, outer, normDim, (float) eps);
        } catch (Throwable t) {
            return null;
        }
    }

    /** Returns double[][]{dx, dgamma, dbeta} or null on failure. */
    public static double[][] tryLayerNormBackward(double[] x, double[] gamma, double[] grad, int rows, int dim, double eps) {
        if (x == null || gamma == null || grad == null || rows <= 0 || dim <= 0) return null;
        if ((long) rows * dim < GpuConfig.layernormMinElements()) return null;
        if (!GpuConfig.allowAttempts()) return null;
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return GpuOptionalRuntime.tryLayerNormBackward(x, gamma, grad, rows, dim, (float) eps);
        } catch (Throwable t) {
            return null;
        }
    }
}
