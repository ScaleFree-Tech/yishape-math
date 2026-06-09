package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated interpolation operations (bilinear, nearest-neighbor).
 */
public final class HpcInterpolate {

    private HpcInterpolate() {
    }

    // ==================== bilinear ====================

    /**
     * Attempt HPC bilinear interpolation forward (align_corners=false).
     *
     * @param input  [B*C*H*W] (row-major flat)
     * @param output [B*C*outH*outW] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryBilinearForward(double[] input, int B, int C, int H, int W,
                                              int outH, int outW, double[] output) {
        if (input == null || output == null) {
            return false;
        }
        long total = (long) B * C * H * W;
        if (total < HpcConfig.interpolateMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.interpolateBilinearForward(
                input, B, C, H, W, outH, outW, output);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC bilinear interpolation backward.
     *
     * @param gradOutput [B*C*outH*outW] upstream gradient (row-major flat)
     * @param gradInput  [B*C*H*W] (pre-allocated, will be zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryBilinearBackward(double[] gradOutput, int B, int C, int H, int W,
                                               int outH, int outW, double[] gradInput) {
        if (gradOutput == null || gradInput == null) {
            return false;
        }
        long total = (long) B * C * outH * outW;
        if (total < HpcConfig.interpolateMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.interpolateBilinearBackward(
                gradOutput, B, C, H, W, outH, outW, gradInput);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    // ==================== nearest ====================

    /**
     * Attempt HPC nearest-neighbor interpolation forward (align_corners=false, round).
     *
     * @param input  [B*C*H*W] (row-major flat)
     * @param output [B*C*outH*outW] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryNearestForward(double[] input, int B, int C, int H, int W,
                                             int outH, int outW, double[] output) {
        if (input == null || output == null) {
            return false;
        }
        long total = (long) B * C * H * W;
        if (total < HpcConfig.interpolateMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.interpolateNearestForward(
                input, B, C, H, W, outH, outW, output);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC nearest-neighbor interpolation backward.
     *
     * @param gradOutput [B*C*outH*outW] upstream gradient (row-major flat)
     * @param gradInput  [B*C*H*W] (pre-allocated, will be zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryNearestBackward(double[] gradOutput, int B, int C, int H, int W,
                                              int outH, int outW, double[] gradInput) {
        if (gradOutput == null || gradInput == null) {
            return false;
        }
        long total = (long) B * C * outH * outW;
        if (total < HpcConfig.interpolateMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.interpolateNearestBackward(
                gradOutput, B, C, H, W, outH, outW, gradInput);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }
}
