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
