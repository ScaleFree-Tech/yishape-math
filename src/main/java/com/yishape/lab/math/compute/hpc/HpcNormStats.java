package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated normalization statistics computation.
 *
 * <p>Provides per-channel and per-sample mean/variance computation for
 * BatchNorm and LayerNorm. Dispatch: GPU → HPC (Rust native) → scalar fallback.</p>
 */
public final class HpcNormStats {
    private HpcNormStats() {}

    // ======================== Batch channel stats ========================

    /**
     * Compute per-channel mean and variance from NCHW-formatted data.
     *
     * <p>Input is a flat double[] with NCHW layout:
     * element at [b, c, h, w] is at index {@code b*C*HW + c*HW + h*W + w}.
     * Within each channel c, spatial elements (HW) are contiguous; batch samples
     * are interleaved.</p>
     *
     * @param nchwData input data in NCHW layout, length B*C*HW
     * @param B        batch size
     * @param C        number of channels
     * @param HW       spatial size (height * width)
     * @param outMean  pre-allocated output array for means, length C
     * @param outVar   pre-allocated output array for variances, length C
     * @return true if computation succeeded (always true — scalar always succeeds)
     */
    public static boolean batchChannelMeanVar(double[] nchwData, int B, int C, int HW,
                                               double[] outMean, double[] outVar) {
        if (nchwData == null || outMean == null || outVar == null) return false;
        if (B <= 0 || C <= 0 || HW <= 0) return false;
        if (outMean.length < C || outVar.length < C) return false;

        // 0. Try GPU reduce (reshape NCHW → [C, B*HW] logical view not needed for GPU)
        // GPU path to be added in follow-up

        // 1. Try HPC native (Rust)
        if (tryHpcBatchChannelMeanVar(nchwData, B, C, HW, outMean, outVar)) {
            return true;
        }

        // 2. Scalar fallback: two-pass per-channel accumulation
        scalarBatchChannelMeanVar(nchwData, B, C, HW, outMean, outVar);
        return true;
    }

    private static boolean tryHpcBatchChannelMeanVar(double[] data, int B, int C, int HW,
                                                      double[] outMean, double[] outVar) {
        if (!HpcConfig.allowAttempts()) return false;
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) return false;
        // OK = 0, avoids static field reference to optional YishapeHpcStatus class
        try {
            int rc = com.yishape.lab.math.hpc.YishapeHpc.batchChannelMeanVar(data, B, C, HW, outMean, outVar);
            return rc == 0;
        } catch (Throwable t) {
            return false;
        }
    }

    static void scalarBatchChannelMeanVar(double[] data, int B, int C, int HW,
                                           double[] outMean, double[] outVar) {
        double N = (double) B * HW;
        // Pass 1: mean
        for (int c = 0; c < C; c++) {
            double sum = 0;
            for (int b = 0; b < B; b++) {
                int base = b * C * HW + c * HW;
                for (int hw = 0; hw < HW; hw++) sum += data[base + hw];
            }
            outMean[c] = sum / N;
        }
        // Pass 2: variance (biased: sum of squared diffs / N)
        for (int c = 0; c < C; c++) {
            double m = outMean[c];
            double sqSum = 0;
            for (int b = 0; b < B; b++) {
                int base = b * C * HW + c * HW;
                for (int hw = 0; hw < HW; hw++) {
                    double diff = data[base + hw] - m;
                    sqSum += diff * diff;
                }
            }
            outVar[c] = sqSum / N;
        }
    }

    // ======================== Per-sample stats ========================

    /**
     * Compute per-sample mean and variance for LayerNorm.
     * Input is [batchSize, features] in row-major order.
     *
     * @param data      input data, length batchSize * features
     * @param batchSize number of samples
     * @param features  feature dimension per sample
     * @param outMean   pre-allocated output array, length batchSize
     * @param outVar    pre-allocated output array, length batchSize
     * @return true if computation succeeded
     */
    public static boolean perSampleMeanVar(double[] data, int batchSize, int features,
                                            double[] outMean, double[] outVar) {
        if (data == null || outMean == null || outVar == null) return false;
        if (batchSize <= 0 || features <= 0) return false;
        if (outMean.length < batchSize || outVar.length < batchSize) return false;

        // 0. Try GPU
        // GPU path to be added in follow-up

        // 1. Try HPC native (Rust)
        if (tryHpcPerSampleMeanVar(data, batchSize, features, outMean, outVar)) {
            return true;
        }

        // 2. Scalar fallback
        scalarPerSampleMeanVar(data, batchSize, features, outMean, outVar);
        return true;
    }

    private static boolean tryHpcPerSampleMeanVar(double[] data, int batchSize, int features,
                                                   double[] outMean, double[] outVar) {
        if (!HpcConfig.allowAttempts()) return false;
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) return false;
        try {
            int rc = com.yishape.lab.math.hpc.YishapeHpc.perSampleMeanVar(data, batchSize, features, outMean, outVar);
            return rc == 0;
        } catch (Throwable t) {
            return false;
        }
    }

    static void scalarPerSampleMeanVar(double[] data, int batchSize, int features,
                                        double[] outMean, double[] outVar) {
        for (int b = 0; b < batchSize; b++) {
            int off = b * features;
            // Pass 1: mean
            double sum = 0;
            for (int f = 0; f < features; f++) sum += data[off + f];
            double mean = sum / features;
            outMean[b] = mean;
            // Pass 2: variance
            double sqSum = 0;
            for (int f = 0; f < features; f++) {
                double diff = data[off + f] - mean;
                sqSum += diff * diff;
            }
            outVar[b] = sqSum / features;
        }
    }

    // ======================== Per-sample normalize ========================

    /**
     * Per-sample normalization: {@code normalized[i] = (xData[i] - mean[i/F]) * invStd[i/F]}
     * where {@code invStd[b] = 1/sqrt(var[b] + eps)}.
     *
     * <p>After calling {@link #perSampleMeanVar}, use this to compute the
     * normalized values and inverse standard deviations in one fused pass.</p>
     *
     * @param xData      input data, length B * F
     * @param B          batch size
     * @param F          features per sample
     * @param mean       per-sample means, length B (from perSampleMeanVar)
     * @param var        per-sample variances, length B (from perSampleMeanVar)
     * @param eps        epsilon for numerical stability
     * @param normalized output normalized values, length B * F
     * @param invStd     output inverse standard deviations, length B
     */
    public static void perSampleNormalize(double[] xData, int B, int F,
                                           double[] mean, double[] var, double eps,
                                           double[] normalized, double[] invStd) {
        if (xData == null || mean == null || var == null || normalized == null || invStd == null) return;
        if (B <= 0 || F <= 0) return;

        // 0. Try HPC native (Rust)
        if (tryHpcPerSampleNormalize(xData, B, F, mean, var, eps, normalized, invStd)) {
            return;
        }

        // 1. Scalar fallback: single pass per sample
        for (int b = 0; b < B; b++) {
            int off = b * F;
            double is = 1.0 / Math.sqrt(var[b] + eps);
            invStd[b] = is;
            double m = mean[b];
            for (int f = 0; f < F; f++) {
                normalized[off + f] = (xData[off + f] - m) * is;
            }
        }
    }

    // ======================== Fused DAXPY ========================

    /**
     * Try HPC-native fused DAXPY: {@code y[i] = a * x[i] + b * y[i]}.
     * @return true if HPC succeeded, false to fall back to scalar
     */
    public static boolean tryFusedDaxpyInPlace(double a, double[] x, double b, double[] y) {
        if (x == null || y == null || x.length != y.length) return false;
        if (x.length < 4096) return false; // too small for FFI overhead
        if (!HpcConfig.allowAttempts()) return false;
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) return false;
        try {
            int rc = com.yishape.lab.math.hpc.YishapeHpc.fusedDaxpyF64(a, x, b, y);
            return rc == 0;
        } catch (Throwable t) {
            return false;
        }
    }

    private static boolean tryHpcPerSampleNormalize(double[] xData, int B, int F,
                                                      double[] mean, double[] var, double eps,
                                                      double[] normalized, double[] invStd) {
        if (!HpcConfig.allowAttempts()) return false;
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) return false;
        try {
            int rc = com.yishape.lab.math.hpc.YishapeHpc.perSampleNormalize(
                xData, B, F, mean, var, eps, normalized, invStd);
            return rc == 0;
        } catch (Throwable t) {
            return false;
        }
    }
}
