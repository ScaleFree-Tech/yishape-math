package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated GroupNorm operations.
 * <p>
 * Backend (Rust FFI + Java bridge) exists since v1.6.0; this wrapper
 * provides the standard gate → threshold → availability → call pattern.
 */
public final class HpcGroupNorm {

    private HpcGroupNorm() {
    }

    /**
     * Attempt HPC groupNorm forward.
     *
     * @param x          [numChannels*H*W] input (row-major flat)
     * @param gamma      [numChannels] per-channel scale
     * @param beta       [numChannels] per-channel shift (nullable)
     * @param numChannels number of channels
     * @param numGroups   number of groups
     * @param H           spatial height
     * @param W           spatial width
     * @param eps         epsilon for numerical stability
     * @param out         [numChannels*H*W] output (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryForward(double[] x, double[] gamma, double[] beta,
                                      int numChannels, int numGroups, int H, int W,
                                      double eps, double[] out) {
        if (x == null || gamma == null || out == null) {
            return false;
        }
        long total = (long) numChannels * H * W;
        if (total < HpcConfig.normMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.groupNormForward(
                x, gamma, beta, numChannels, numGroups, H, W, eps, out);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC groupNorm backward.
     *
     * @param x          [numChannels*H*W] forward input
     * @param gamma      [numChannels] forward gamma
     * @param gradOutput [numChannels*H*W] upstream gradient
     * @param numChannels number of channels
     * @param numGroups   number of groups
     * @param H           spatial height
     * @param W           spatial width
     * @param eps         epsilon for numerical stability
     * @param dx          [numChannels*H*W] (pre-allocated)
     * @param dgamma      [numChannels] (pre-allocated, pre-zeroed)
     * @param dbeta       [numChannels] (pre-allocated, pre-zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryBackward(double[] x, double[] gamma, double[] gradOutput,
                                       int numChannels, int numGroups, int H, int W,
                                       double eps, double[] dx, double[] dgamma, double[] dbeta) {
        if (x == null || gamma == null || gradOutput == null
                || dx == null || dgamma == null || dbeta == null) {
            return false;
        }
        long total = (long) numChannels * H * W;
        if (total < HpcConfig.normMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.groupNormBackward(
                x, gamma, gradOutput, numChannels, numGroups, H, W, eps, dx, dgamma, dbeta);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }
}
