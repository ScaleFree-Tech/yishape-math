package com.yishape.lab.math.compute.hpc;

/**
 * HPC-accelerated im2col / col2im / flat dgemm for DL convolution.
 * Follows the same optional-native pattern as {@link HpcGemm}.
 */
public final class HpcIm2col {

    private HpcIm2col() {
    }

    /**
     * Attempt HPC im2col: unfold image patches into column matrix.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryIm2col(double[] input, int C, int H, int W,
            int Kh, int Kw, int stride, int pad, int dilation, double[] out) {
        if (input == null || out == null) {
            return false;
        }
        long total = (long) C * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.im2col(input, C, H, W, Kh, Kw, stride, pad, dilation, out); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC col2im: scatter column gradients back to image space.
     */
    public static boolean tryCol2im(double[] colGrad, int C, int H, int W,
            int Kh, int Kw, int stride, int pad, int dilation, double[] imgGradOut) {
        if (colGrad == null || imgGradOut == null) {
            return false;
        }
        long total = (long) C * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.col2im(colGrad, C, H, W, Kh, Kw, stride, pad, dilation, imgGradOut); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC flat sgemm (f32): C[m×n] = A[m×k] @ B[k×n], all row-major flat arrays.
     *
     * @return true if HPC succeeded
     */
    public static boolean tryFlatSgemm(int m, int n, int k, float[] a, float[] b, float[] cOut) {
        if (a == null || b == null || cOut == null) {
            return false;
        }
        long flops = (long) m * n * k;
        if (flops < HpcConfig.flatGemmMinFlops()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.flatSgemm(m, n, k, a, b, cOut); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC flat dgemm: C[m×n] = A[m×k] @ B[k×n], all row-major flat arrays.
     *
     * @return true if HPC succeeded
     */
    public static boolean tryFlatDgemm(int m, int n, int k, double[] a, double[] b, double[] cOut) {
        if (a == null || b == null || cOut == null) {
            return false;
        }
        long flops = (long) m * n * k;
        if (flops < HpcConfig.flatGemmMinFlops()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.flatDgemm(m, n, k, a, b, cOut); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC batch flat row-major dgemm: for each batch b, C_b[m×n] = A_b[m×k] @ B_b[k×n].
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryFlatDgemmBatch(int batch, int m, int n, int k,
                                             double[] a, double[] b, double[] cOut) {
        if (a == null || b == null || cOut == null || batch <= 0) {
            return false;
        }
        long flops = (long) batch * m * n * k;
        if (flops < HpcConfig.flatGemmMinFlops()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.flatDgemmBatch(batch, m, n, k, a, b, cOut); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC row-major dgemm with transpose flags: C[m×n] = op(A)[...] @ op(B)[...].
     * transp: 0=NN, 1=TN, 2=NT, 3=TT.
     * Eliminates Java-side flatTranspose allocations — transpose happens in Rust.
     *
     * @return true if HPC succeeded
     */
    public static boolean tryFlatDgemmTransp(int m, int n, int k,
                                              double[] a, double[] b, double[] cOut,
                                              int transp) {
        if (a == null || b == null || cOut == null) {
            return false;
        }
        long flops = (long) m * n * k;
        if (flops < HpcConfig.flatGemmMinFlops()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.flatDgemmTransp(m, n, k, a, b, cOut, transp); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC fused conv2d forward: im2col + gemm + bias.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryConv2dForward(
            double[] input, double[] weight, double[] bias,
            int C, int H, int W, int outCh,
            int Kh, int Kw, int stride, int pad, int dilation,
            double[] output) {
        if (input == null || weight == null || output == null) {
            return false;
        }
        long total = (long) C * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.conv2dForward(
                    input, weight, bias, C, H, W, outCh, Kh, Kw, stride, pad, dilation, output); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC fused ConvTranspose2d forward: transposeIm2col + gemm + bias.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryConvTranspose2dForward(
            double[] input, double[] weight, double[] bias,
            int inCh, int H, int W, int outCh,
            int Kh, int Kw, int stride, int pad,
            int outH, int outW, double[] output) {
        if (input == null || weight == null || output == null) {
            return false;
        }
        long total = (long) inCh * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.convTranspose2dForward(
                    input, weight, bias, inCh, H, W, outCh, Kh, Kw, stride, pad, outH, outW, output); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC fused Conv1d forward: im2col1d + gemm + bias.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryConv1dForward(
            double[] input, double[] weight, double[] bias,
            int inCh, int seqLen, int outCh, int K,
            int stride, int pad, int dilation, double[] output) {
        if (input == null || weight == null || output == null) {
            return false;
        }
        long total = (long) inCh * seqLen;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.conv1dForward(
                    input, weight, bias, inCh, seqLen, outCh, K, stride, pad, dilation, output); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC 1D col2im: scatter column gradients back to input space.
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryCol2im1d(
            double[] colGrad, int inCh, int seqLen, int K,
            int stride, int pad, int dilation, double[] imgGrad) {
        if (colGrad == null || imgGrad == null) {
            return false;
        }
        long total = (long) inCh * seqLen;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.col2im1d(
                    colGrad, inCh, seqLen, K, stride, pad, dilation, imgGrad); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC batch im2col: process B samples in a single call.
     */
    public static boolean tryBatchIm2col(double[] input, int B, int C, int H, int W,
            int Kh, int Kw, int stride, int pad, int dilation, double[] out) {
        if (input == null || out == null) {
            return false;
        }
        long total = (long) B * C * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.batchIm2col(input, B, C, H, W, Kh, Kw, stride, pad, dilation, out); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC batch col2im: scatter B samples' column gradients back to image space.
     */
    public static boolean tryBatchCol2im(double[] colGrad, int B, int C, int H, int W,
            int Kh, int Kw, int stride, int pad, int dilation, double[] imgGradOut) {
        if (colGrad == null || imgGradOut == null) {
            return false;
        }
        long total = (long) B * C * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try { int rc = com.yishape.lab.math.hpc.YishapeHpc.batchCol2im(colGrad, B, C, H, W, Kh, Kw, stride, pad, dilation, imgGradOut); return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK; } catch (Throwable t) { return false; }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    // ==================== Fused GEMM+activation availability ====================

    public static boolean isFusedGemmReluAvailable() {
        return com.yishape.lab.math.hpc.YishapeHpc.isFusedGemmReluAvailable();
    }

    public static boolean isFusedGemmGeluAvailable() {
        return com.yishape.lab.math.hpc.YishapeHpc.isFusedGemmGeluAvailable();
    }

    public static boolean isFusedGemmSiluAvailable() {
        return com.yishape.lab.math.hpc.YishapeHpc.isFusedGemmSiluAvailable();
    }

    // ==================== Batched ConvTranspose2d (B folded into column axis) ====================

    /**
     * Attempt HPC batched transpose-im2col: scatter B samples' inputs into a
     * shared column matrix {@code col[inCh*Kh*Kw, B*outH*outW]} (B interleaved
     * along columns, row-major). Input is [B, inCh, H, W] (B-major).
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryBatchTransposeIm2col(double[] input, int B, int inCh, int H, int W,
            int Kh, int Kw, int stride, int pad, int outH, int outW, double[] col) {
        if (input == null || col == null) {
            return false;
        }
        long total = (long) B * inCh * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try {
                int rc = com.yishape.lab.math.hpc.YishapeHpc.transposeIm2colBatch(
                        input, B, inCh, H, W, Kh, Kw, stride, pad, outH, outW, col);
                return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
            } catch (Throwable t) {
                return false;
            }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC fused batched ConvTranspose2d forward (transposeIm2col + GEMM + bias).
     * Output layout is [outCh, B*outH*outW] (outCh-major); caller permutes to
     * [B, outCh, outH, outW]. Input is [B, inCh, H, W] (B-major).
     *
     * @return true if HPC succeeded, false to fall back to Java
     */
    public static boolean tryConvTranspose2dForwardBatch(double[] input, double[] weight, double[] bias,
            int B, int inCh, int H, int W, int outCh,
            int Kh, int Kw, int stride, int pad, int outH, int outW, double[] output) {
        if (input == null || weight == null || output == null) {
            return false;
        }
        long total = (long) B * inCh * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try {
                int rc = com.yishape.lab.math.hpc.YishapeHpc.convTranspose2dForwardBatch(
                        input, weight, bias, B, inCh, H, W, outCh, Kh, Kw, stride, pad, outH, outW, output);
                return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
            } catch (Throwable t) {
                return false;
            }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC batched ConvTranspose2d weight gradient (sums over B).
     * Input [B,inCh,H,W], gradOutput [B,outCh,oH,oW] (B-major).
     */
    public static boolean tryConvTranspose2dDWeightBatch(double[] input, double[] gradOutput, int B,
            int inCh, int outCh, int H, int W,
            int Kh, int Kw, int stride, int pad, int oH, int oW, double[] dWeight) {
        if (input == null || gradOutput == null || dWeight == null) {
            return false;
        }
        long total = (long) B * inCh * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try {
                int rc = com.yishape.lab.math.hpc.YishapeHpc.convTranspose2dDWeightBatch(
                        input, gradOutput, B, inCh, outCh, H, W, Kh, Kw, stride, pad, oH, oW, dWeight);
                return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
            } catch (Throwable t) {
                return false;
            }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }

    /**
     * Attempt HPC batched ConvTranspose2d input gradient.
     * weight [outCh,inCh,Kh,Kw], gradOutput [B,outCh,oH,oW] (B-major), dInput [B,inCh,H,W] (B-major).
     */
    public static boolean tryConvTranspose2dDInputBatch(double[] weight, double[] gradOutput, int B,
            int inCh, int outCh, int H, int W,
            int Kh, int Kw, int stride, int pad, int oH, int oW, double[] dInput) {
        if (weight == null || gradOutput == null || dInput == null) {
            return false;
        }
        long total = (long) B * inCh * H * W;
        if (total < HpcConfig.convMinElements()) {
            return false;
        }
        if (!HpcConfig.allowAttempts()) {
            return false;
        }
        if (!HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            return false;
        }
        try {
            try {
                int rc = com.yishape.lab.math.hpc.YishapeHpc.convTranspose2dDInputBatch(
                        weight, gradOutput, B, inCh, outCh, H, W, Kh, Kw, stride, pad, oH, oW, dInput);
                return rc == com.yishape.lab.math.hpc.YishapeHpcStatus.OK;
            } catch (Throwable t) {
                return false;
            }
        } catch (LinkageError | RuntimeException e) {
            return false;
        }
    }
}
