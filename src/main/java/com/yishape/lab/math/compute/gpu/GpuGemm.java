package com.yishape.lab.math.compute.gpu;

/**
 * Four-gate GPU GEMM: guard → threshold → allow → call.
 * Returns null / false to signal CPU fallback.
 */
public final class GpuGemm {
    private GpuGemm() {}

    public static double[][] tryMatMul(double[][] a, double[][] b) {
        // Gate 1: guard
        if (a == null || b == null || a.length == 0 || b.length == 0) return null;
        int m = a.length, k = a[0].length, n = b[0].length;
        if (k != b.length) return null;

        // Gate 2: threshold
        long flops = (long) m * n * k;
        if (flops < GpuConfig.gemmMinFlops()) return null;

        // Gate 3: allow
        if (!GpuConfig.allowAttempts()) return null;

        // Gate 4: availability
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;

        // Gate 5: call
        try {
            return GpuOptionalRuntime.tryMatMul(a, b);
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * Flat GPU GEMM: C[m×n] = A[m×k] @ B[k×n], all flat row-major double[].
     * Writes result into cOut and returns true on success.
     */
    public static boolean tryFlatMatMul(double[] a, double[] b, double[] cOut, int m, int k, int n) {
        return tryFlatMatMul(a, 0, m, k, b, 0, n, cOut, 0);
    }

    /**
     * GPU gather / embedding lookup: out[i*D+j] = weight[indices[i]*D+j].
     * Writes result into outData and returns true on success.
     */
    public static boolean tryGather(double[] weight, double[] indices, int embeddingDim, double[] outData) {
        // Gate 1: guard
        if (weight == null || indices == null || outData == null) return false;
        int n = indices.length;
        if (n == 0 || embeddingDim <= 0) return false;
        if (outData.length < n * embeddingDim) return false;

        // Gate 2: threshold — gather is memory-bound, use activation threshold
        int nOut = n * embeddingDim;
        if (nOut < GpuConfig.activationMinElements()) return false;

        // Gate 3: allow
        if (!GpuConfig.allowAttempts()) return false;

        // Gate 4: availability
        if (!GpuOptionalRuntime.isGpuAvailable()) return false;

        // Gate 5: call
        try {
            double[] result = GpuOptionalRuntime.tryGather(weight, indices, embeddingDim);
            if (result != null) {
                System.arraycopy(result, 0, outData, 0, nOut);
                return true;
            }
            return false;
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * GPU im2col: unfold image [C,H,W] into column matrix [C*Kh*Kw, outH*outW].
     * Returns the col matrix (flattened row-major) on success, null on failure.
     */
    public static double[] tryIm2col(double[] input, int C, int H, int W,
                                      int outH, int outW, int kH, int kW,
                                      int stride, int padding) {
        // Gate 1: guard
        if (input == null || C <= 0 || H <= 0 || W <= 0 || outH <= 0 || outW <= 0 || kH <= 0 || kW <= 0)
            return null;
        long total = (long) C * kH * kW * outH * outW;
        if (input.length < C * H * W) return null;

        // Gate 2: threshold
        if (total < GpuConfig.im2colMinElements()) return null;

        // Gate 3: allow
        if (!GpuConfig.allowAttempts()) return null;

        // Gate 4: availability
        if (!GpuOptionalRuntime.isGpuAvailable()) return null;

        // Gate 5: call
        try {
            return GpuOptionalRuntime.tryIm2col(input, C, H, W, outH, outW, kH, kW, stride, padding);
        } catch (Throwable t) {
            return null;
        }
    }

    /**
     * GPU GEMM with transpose: C[m×n] = op(A)[m×k] @ op(B)[k×n].
     * transp: 0=NN, 1=TN, 2=NT, 3=TT.
     */
    public static boolean tryFlatMatMulTransp(double[] a, double[] b, double[] cOut, int m, int k, int n, int transp) {
        // Gate 1: guard
        if (a == null || b == null || cOut == null || m <= 0 || n <= 0 || k <= 0) return false;
        int aLen = ((transp & 1) != 0) ? k * m : m * k;
        int bLen = ((transp & 2) != 0) ? n * k : k * n;
        int cLen = m * n;
        if (a.length < aLen || b.length < bLen || cOut.length < cLen) return false;

        // Gate 2: threshold
        long flops = (long) m * n * k;
        if (flops < GpuConfig.gemmMinFlops()) return false;

        // Gate 3: allow
        if (!GpuConfig.allowAttempts()) return false;

        // Gate 4: availability
        if (!GpuOptionalRuntime.isGpuAvailable()) return false;

        // Gate 5: call
        try {
            double[] cTmp = GpuOptionalRuntime.tryFlatMatMulTransp(a, b, m, k, n, transp);
            if (cTmp != null) {
                System.arraycopy(cTmp, 0, cOut, 0, cLen);
                return true;
            }
            return false;
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Offset-aware flat GPU GEMM for batched use.
     * Slices input arrays at the given offsets before dispatching to GPU.
     */
    public static boolean tryFlatMatMul(double[] a, int aOff, int m, int k,
                                         double[] b, int bOff, int n,
                                         double[] cOut, int cOff) {
        // Gate 1: guard
        if (a == null || b == null || cOut == null || m <= 0 || n <= 0 || k <= 0) return false;
        int aLen = m * k, bLen = k * n, cLen = m * n;
        if (a.length < aOff + aLen || b.length < bOff + bLen || cOut.length < cOff + cLen) return false;

        // Gate 2: threshold
        long flops = (long) m * n * k;
        if (flops < GpuConfig.gemmMinFlops()) return false;

        // Gate 3: allow
        if (!GpuConfig.allowAttempts()) return false;

        // Gate 4: availability
        if (!GpuOptionalRuntime.isGpuAvailable()) return false;

        // Gate 5: call — GPU requires contiguous arrays, slice if offset > 0
        try {
            double[] aSlice = (aOff == 0) ? a : java.util.Arrays.copyOfRange(a, aOff, aOff + aLen);
            double[] bSlice = (bOff == 0) ? b : java.util.Arrays.copyOfRange(b, bOff, bOff + bLen);
            double[] cTmp = new double[cLen];
            if (GpuOptionalRuntime.tryFlatMatMul(aSlice, bSlice, cTmp, m, k, n)) {
                System.arraycopy(cTmp, 0, cOut, cOff, cLen);
                return true;
            }
            return false;
        } catch (Throwable t) {
            return false;
        }
    }
}
