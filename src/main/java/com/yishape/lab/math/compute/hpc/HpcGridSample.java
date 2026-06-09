package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated grid sample (spatial grid sampling for image warping).
 */
public final class HpcGridSample {

    private HpcGridSample() {
    }

    /**
     * Attempt HPC grid sample forward.
     *
     * @param input       [N * C * H * W] input image (row-major flat)
     * @param grid        [N * outH * outW * 2] sampling grid
     * @param N           batch size
     * @param C           channels
     * @param H           input height
     * @param W           input width
     * @param outH        output height
     * @param outW        output width
     * @param mode        0=bilinear, 1=nearest
     * @param paddingMode 0=zeros, 1=border, 2=reflection
     * @param output      [N * C * outH * outW] (pre-allocated)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryGridSampleForward(double[] input, double[] grid,
                                                int N, int C, int H, int W, int outH, int outW,
                                                int mode, int paddingMode, double[] output) {
        if (input == null || grid == null || output == null) {
            return false;
        }
        long total = (long) N * C * H * W;
        if (total < HpcConfig.gridSampleMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.gridSampleForward(
                input, grid, N, C, H, W, outH, outW, mode, paddingMode, output);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC grid sample backward.
     *
     * @param gradOutput  [N * C * outH * outW] upstream gradient
     * @param input       [N * C * H * W] forward input (saved)
     * @param grid        [N * outH * outW * 2] forward grid (saved)
     * @param gradInput   [N * C * H * W] (pre-allocated, will be zeroed)
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryGridSampleBackward(double[] gradOutput, double[] input, double[] grid,
                                                 int N, int C, int H, int W, int outH, int outW,
                                                 int mode, int paddingMode, double[] gradInput) {
        if (gradOutput == null || input == null || grid == null || gradInput == null) {
            return false;
        }
        long total = (long) N * C * outH * outW;
        if (total < HpcConfig.gridSampleMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        int rc = com.yishape.lab.math.hpc.YishapeHpc.gridSampleBackward(
                gradOutput, input, grid, N, C, H, W, outH, outW, mode, paddingMode, gradInput);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }
}
