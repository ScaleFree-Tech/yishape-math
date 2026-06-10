package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated depthwise 1D convolution with causal padding.
 */
public final class HpcDepthwiseConv1d {

    private HpcDepthwiseConv1d() {
    }

    /**
     * Attempt HPC depthwise 1D convolution forward.
     *
     * @param input    [seqLen × channels] (row-major flat)
     * @param weight   [channels × kDim] (row-major flat)
     * @param bias     [channels] (nullable)
     * @param seqLen   sequence length
     * @param channels number of channels
     * @param kDim     kernel size
     * @param output   [seqLen × channels] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryForward(double[] input, double[] weight, double[] bias,
                                      int seqLen, int channels, int kDim, double[] output) {
        if (input == null || weight == null || output == null) {
            return false;
        }
        long total = (long) seqLen * channels;
        if (total < HpcConfig.depthwiseConv1dMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.depthwiseConv1d(
                input, weight, bias, seqLen, channels, kDim, output);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC depthwise 1D convolution backward.
     *
     * @param input      [seqLen × channels] forward input (row-major flat)
     * @param weight     [channels × kDim] forward weight (row-major flat)
     * @param gradOutput [outLen × channels] upstream gradient (row-major flat)
     * @param seqLen     input sequence length
     * @param channels   number of channels
     * @param kDim       kernel size
     * @param stride     stride
     * @param pad        padding
     * @param dInput     [seqLen × channels] (pre-allocated, will be zeroed)
     * @param dWeight    [channels × kDim] (pre-allocated, will be zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryBackward(double[] input, double[] weight,
                                       double[] gradOutput, int seqLen, int channels,
                                       int kDim, int stride, int pad,
                                       double[] dInput, double[] dWeight) {
        if (input == null || weight == null || gradOutput == null
                || dInput == null || dWeight == null) {
            return false;
        }
        long total = (long) seqLen * channels;
        if (total < HpcConfig.depthwiseConv1dMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.depthwiseConv1dBackward(
                input, weight, gradOutput, seqLen, channels, kDim, stride, pad, dInput, dWeight);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }
}
