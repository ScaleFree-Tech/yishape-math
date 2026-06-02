package com.yishape.lab.math.compute.gpu;

/**
 * Four-gate GPU GroupNorm backward dispatch.
 */
public final class GpuGroupNorm {
    private GpuGroupNorm() {}

    /** Returns double[][]{dx, dgamma, dbeta} or null on failure. */
    public static double[][] tryGroupNormBackward(double[] x, double[] gamma, double[] grad,
                                                   int numGroups, int gch, int hw, double eps) {
        if (x == null || gamma == null || grad == null || numGroups <= 0 || gch <= 0 || hw <= 0) return null;
        if ((long) numGroups * gch * hw < GpuConfig.layernormMinElements()) return null;
        if (!GpuConfig.allowAttempts()) return null;
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return GpuOptionalRuntime.tryGroupNormBackward(x, gamma, grad, numGroups, gch, hw, (float) eps);
        } catch (Throwable t) {
            return null;
        }
    }
}
