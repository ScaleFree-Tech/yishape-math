package com.yishape.lab.math.compute.gpu;

/**
 * Four-gate GPU BatchNorm dispatch.
 */
public final class GpuBatchNorm {
    private GpuBatchNorm() {}

    public static double[] tryBatchNorm(double[] x, double[] mean, double[] var_,
                                         double[] gamma, double[] beta, int n, int c, int hw) {
        if (x == null || mean == null || var_ == null || gamma == null || beta == null) return null;
        if (n <= 0 || c <= 0 || hw <= 0) return null;
        if ((long) n * c * hw < GpuConfig.batchnormMinElements()) return null;
        if (!GpuConfig.allowAttempts()) return null;
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return GpuOptionalRuntime.tryBatchNorm(x, mean, var_, gamma, beta, n, c, hw);
        } catch (Throwable t) {
            return null;
        }
    }
}
