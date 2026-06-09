package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated normalization operations (instanceNorm, rmsNorm).
 */
public final class HpcNorm {

    private HpcNorm() {
    }

    // ==================== instanceNorm ====================

    /**
     * Attempt HPC instanceNorm forward.
     *
     * @param x       [N*C*spatial] input (row-major flat)
     * @param gamma   [C] per-channel scale
     * @param beta    [C] per-channel shift (nullable)
     * @param spatial number of elements per (N,C) slice (H*W for 2D)
     * @param out     [N*C*spatial] output (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryInstanceNormForward(double[] x, double[] gamma, double[] beta,
                                                  int N, int C, int spatial, double eps,
                                                  double[] out) {
        if (x == null || gamma == null || out == null) {
            return false;
        }
        long total = (long) N * C * spatial;
        if (total < HpcConfig.normMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.instanceNormForward(
                x, gamma, beta, N, C, spatial, eps, out);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC instanceNorm backward.
     *
     * @param x          [N*C*spatial] forward input
     * @param gamma      [C] forward gamma
     * @param gradOutput [N*C*spatial] upstream gradient
     * @param dx         [N*C*spatial] (pre-allocated)
     * @param dgamma     [C] (pre-allocated, pre-zeroed)
     * @param dbeta      [C] (pre-allocated, pre-zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryInstanceNormBackward(double[] x, double[] gamma, double[] gradOutput,
                                                   int N, int C, int spatial, double eps,
                                                   double[] dx, double[] dgamma, double[] dbeta) {
        if (x == null || gamma == null || gradOutput == null
                || dx == null || dgamma == null || dbeta == null) {
            return false;
        }
        long total = (long) N * C * spatial;
        if (total < HpcConfig.normMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.instanceNormBackward(
                x, gamma, gradOutput, N, C, spatial, eps, dx, dgamma, dbeta);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    // ==================== rmsNorm ====================

    /**
     * Attempt HPC rmsNorm forward.
     *
     * @param x      [rows*dim] input (row-major flat)
     * @param gamma  [dim] per-element scale
     * @param out    [rows*dim] output (pre-allocated)
     * @param rmsOut [rows] RMS values (pre-allocated, saved for backward)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryRMSNormForward(double[] x, double[] gamma, int rows, int dim,
                                             double eps, double[] out, double[] rmsOut) {
        if (x == null || gamma == null || out == null || rmsOut == null) {
            return false;
        }
        long total = (long) rows * dim;
        if (total < HpcConfig.rmsNormMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.rmsNormForward(
                x, gamma, rows, dim, eps, out, rmsOut);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC rmsNorm backward.
     *
     * @param x          [rows*dim] forward input
     * @param gamma      [dim] forward gamma
     * @param gradOutput [rows*dim] upstream gradient
     * @param rms        [rows] RMS values saved from forward
     * @param dx         [rows*dim] (pre-allocated)
     * @param dgamma     [dim] (pre-allocated, pre-zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryRMSNormBackward(double[] x, double[] gamma, double[] gradOutput,
                                              double[] rms, int rows, int dim, double eps,
                                              double[] dx, double[] dgamma) {
        if (x == null || gamma == null || gradOutput == null || rms == null
                || dx == null || dgamma == null) {
            return false;
        }
        long total = (long) rows * dim;
        if (total < HpcConfig.rmsNormMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.rmsNormBackward(
                x, gamma, gradOutput, rms, rows, dim, eps, dx, dgamma);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }
}
