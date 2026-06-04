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
}
