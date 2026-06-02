package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated image processing (decode, resize, crop, normalize).
 * Follows the same optional-native pattern as {@link HpcIm2col}.
 */
public final class HpcImage {

    private HpcImage() {}

    /**
     * Attempt HPC image decode: read image bytes into CHW double[] normalized to [0, 1].
     * First call with out=null to get dimensions, then allocate and call again.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryDecode(byte[] data, int len, int[] outW, int[] outH, int[] outC, double[] out) {
        if (data == null || len <= 0 || outW == null || outH == null || outC == null) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) return false;
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) return false;
        int rc = com.yishape.lab.math.hpc.YishapeHpc.imageDecode(data, len, outW, outH, outC, out);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC bilinear resize of CHW double[] image.
     *
     * @return true if HPC succeeded
     */
    public static boolean tryResize(double[] input, int C, int H, int W, int newH, int newW, double[] output) {
        if (input == null || output == null || C <= 0 || H <= 0 || W <= 0 || newH <= 0 || newW <= 0) {
            return false;
        }
        long total = (long) C * H * W;
        if (total < 1024) return false; // too small, skip HPC
        if (!HpcConfig.allowAttempts()) return false;
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) return false;
        int rc = com.yishape.lab.math.hpc.YishapeHpc.imageResize(input, C, H, W, newH, newW, output);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC crop of CHW double[] image.
     *
     * @return true if HPC succeeded
     */
    public static boolean tryCrop(double[] input, int C, int H, int W, int x, int y, int cropW, int cropH, double[] output) {
        if (input == null || output == null || C <= 0 || H <= 0 || W <= 0 || cropW <= 0 || cropH <= 0) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) return false;
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) return false;
        int rc = com.yishape.lab.math.hpc.YishapeHpc.imageCrop(input, C, H, W, x, y, cropW, cropH, output);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }

    /**
     * Attempt HPC normalize of CHW double[] image with per-channel mean/std.
     *
     * @return true if HPC succeeded
     */
    public static boolean tryNormalize(double[] input, int C, int H, int W,
                                        double[] mean, double[] std, double[] output) {
        if (input == null || mean == null || std == null || output == null || C <= 0 || H <= 0 || W <= 0) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) return false;
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) return false;
        int rc = com.yishape.lab.math.hpc.YishapeHpc.imageNormalize(input, C, H, W, mean, std, output);
        return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
    }
}
