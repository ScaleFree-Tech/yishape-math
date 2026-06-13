package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated pooling operations (maxPool2d, avgPool2d, adaptiveAvgPool2d).
 */
public final class HpcPool {

    private HpcPool() {
    }

    // ==================== maxPool2d ====================

    /**
     * Attempt HPC maxPool2d forward.
     *
     * @param input   [B*C*H*W] (row-major flat)
     * @param argmax  [B*C*outH*outW] int array (pre-allocated, receives flat H*W indices)
     * @param output  [B*C*outH*outW] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryMaxPool2dForward(double[] input, int B, int C, int H, int W,
                                               int kH, int kW, int stride, int pad,
                                               double[] output, int[] argmax) {
        if (input == null || output == null || argmax == null) {
            return false;
        }
        long total = (long) B * C * H * W;
        if (total < HpcConfig.poolMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.maxPool2dForward(
                input, B, C, H, W, kH, kW, stride, pad, output, argmax); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }

    /**
     * Attempt HPC maxPool2d backward.
     *
     * @param gradOutput [B*C*outH*outW] upstream gradient (row-major flat)
     * @param argmax     [B*C*outH*outW] int array from forward
     * @param gradInput  [B*C*H*W] (pre-allocated, will be zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryMaxPool2dBackward(double[] gradOutput, int[] argmax,
                                                int B, int C, int H, int W, int outH, int outW,
                                                double[] gradInput) {
        if (gradOutput == null || argmax == null || gradInput == null) {
            return false;
        }
        long total = (long) B * C * outH * outW;
        if (total < HpcConfig.poolMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.maxPool2dBackward(
                gradOutput, argmax, B, C, H, W, outH, outW, gradInput); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }

    // ==================== avgPool2d ====================

    /**
     * Attempt HPC avgPool2d forward.
     *
     * @param input   [B*C*H*W] (row-major flat)
     * @param output  [B*C*outH*outW] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryAvgPool2dForward(double[] input, int B, int C, int H, int W,
                                               int kH, int kW, int stride, int pad,
                                               double[] output) {
        if (input == null || output == null) {
            return false;
        }
        long total = (long) B * C * H * W;
        if (total < HpcConfig.poolMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.avgPool2dForward(
                input, B, C, H, W, kH, kW, stride, pad, output); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }

    /**
     * Attempt HPC avgPool2d backward.
     *
     * @param gradOutput [B*C*outH*outW] upstream gradient (row-major flat)
     * @param gradInput  [B*C*H*W] (pre-allocated, will be zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryAvgPool2dBackward(double[] gradOutput, int B, int C, int H, int W,
                                                int kH, int kW, int stride, int pad,
                                                int outH, int outW, double[] gradInput) {
        if (gradOutput == null || gradInput == null) {
            return false;
        }
        long total = (long) B * C * outH * outW;
        if (total < HpcConfig.poolMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.avgPool2dBackward(
                gradOutput, B, C, H, W, kH, kW, stride, pad, outH, outW, gradInput); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }

    // ==================== adaptiveAvgPool2d ====================

    /**
     * Attempt HPC adaptiveAvgPool2d forward.
     *
     * @param input   [B*C*H*W] (row-major flat)
     * @param output  [B*C*outH*outW] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryAdaptiveAvgPool2dForward(double[] input, int B, int C, int H, int W,
                                                       int outH, int outW, double[] output) {
        if (input == null || output == null) {
            return false;
        }
        long total = (long) B * C * H * W;
        if (total < HpcConfig.poolMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.adaptiveAvgPool2dForward(
                input, B, C, H, W, outH, outW, output); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }

    /**
     * Attempt HPC adaptiveAvgPool2d backward.
     *
     * @param gradOutput [B*C*outH*outW] upstream gradient (row-major flat)
     * @param gradInput  [B*C*H*W] (pre-allocated, will be zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryAdaptiveAvgPool2dBackward(double[] gradOutput, int B, int C, int H, int W,
                                                        int outH, int outW, double[] gradInput) {
        if (gradOutput == null || gradInput == null) {
            return false;
        }
        long total = (long) B * C * outH * outW;
        if (total < HpcConfig.poolMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try { int rc = com.yishape.lab.math.hpc.YishapeHpc.adaptiveAvgPool2dBackward(
                gradOutput, B, C, H, W, outH, outW, gradInput); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
    }
}
