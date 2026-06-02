package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.compute.gpu.GpuGemm;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.hpc.HpcIm2col;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Flat GEMM dispatch: HPC native → SIMD (Vector API) → scalar fallback.
 * <p>
 * Follows the same optional-acceleration pattern as {@link com.yishape.lab.math.compute.hpc.HpcIm2col}:
 * detection is reflection-based so this class loads successfully even when
 * {@code jdk.incubator.vector} is absent.
 */
public final class FlatGemm {

    private static final Logger LOG = Logger.getLogger(FlatGemm.class.getName());

    private FlatGemm() {}

    /** Cached MethodHandle for SIMDDoubleComputer.flatMmul(double[],int,int,double[],int) — null if unavailable */
    private static final MethodHandle MH_FLAT_MMUL;
    /** Cached MethodHandle for SIMDDoubleComputer.flatMmulBatched(double[],double[],int,int,int,int) — null if unavailable */
    private static final MethodHandle MH_FLAT_MMUL_BATCHED;
    /** Cached MethodHandle for SIMDDoubleComputer.flatTranspose(double[],int,int) — null if unavailable */
    private static final MethodHandle MH_FLAT_TRANSPOSE;

    static {
        MethodHandle mmul = null;
        MethodHandle batched = null;
        MethodHandle transpose = null;
        if (ComputerConfig.checkIfSIMDSupported()) {
            try {
                Class<?> cls = Class.forName("com.yishape.lab.math.compute.SIMDDoubleComputer");
                MethodHandles.Lookup lookup = MethodHandles.lookup();

                Method m1 = cls.getMethod("flatMmul", double[].class, int.class, int.class, double[].class, int.class);
                mmul = lookup.unreflect(m1);

                Method m2 = cls.getMethod("flatMmulBatched", double[].class, double[].class, int.class, int.class, int.class, int.class);
                batched = lookup.unreflect(m2);

                Method m3 = cls.getMethod("flatTranspose", double[].class, int.class, int.class);
                transpose = lookup.unreflect(m3);
            } catch (Throwable t) {
                // SIMD class loaded but method binding failed — fall through to scalar
                mmul = null;
                batched = null;
                transpose = null;
            }
        }
        MH_FLAT_MMUL = mmul;
        MH_FLAT_MMUL_BATCHED = batched;
        MH_FLAT_TRANSPOSE = transpose;
    }

    /**
     * C[m×n] = A[m×k] @ B[k×n], all row-major flat arrays.
     * Dispatch: GPU → HPC → SIMD → scalar.
     */
    public static double[] flatMmul(double[] a, int m, int k, double[] b, int n) {
        boolean logFine = LOG.isLoggable(Level.FINE);
        double[] c = new double[m * n];
        // 0. Try GPU first (fastest when available)
        if (GpuGemm.tryFlatMatMul(a, b, c, m, k, n)) {
            if (logFine) LOG.fine("flatMmul GPU [" + m + "," + k + "]@[" + k + "," + n + "]  flops=" + ((long) m * n * k));
            return c;
        }
        if (logFine) {
            LOG.fine("flatMmul GPU skip [" + m + "," + k + "]@[" + k + "," + n + "]"
                + " flops=" + ((long) m * n * k)
                + " allow=" + GpuConfig.allowAttempts()
                + " avail=" + GpuOptionalRuntime.isGpuAvailable());
        }
        // 1. Try HPC native
        if (HpcIm2col.tryFlatDgemm(m, n, k, a, b, c)) {
            if (logFine) LOG.fine("flatMmul HPC [" + m + "," + k + "]@[" + k + "," + n + "]");
            return c;
        }
        // 2. Try SIMD via reflection
        if (MH_FLAT_MMUL != null) {
            try {
                double[] simdResult = (double[]) MH_FLAT_MMUL.invoke(a, m, k, b, n);
                if (logFine) LOG.fine("flatMmul SIMD [" + m + "," + k + "]@[" + k + "," + n + "]");
                return simdResult;
            } catch (Throwable t) {
                // SIMD invoke failed — fall through to scalar
            }
        }
        // 3. Scalar fallback
        if (logFine) LOG.fine("flatMmul SCALAR [" + m + "," + k + "]@[" + k + "," + n + "]");
        flatMmulScalar(a, 0, m, k, b, 0, n, c, 0);
        return c;
    }

    /**
     * Offset-aware flat GEMM for batched use.
     * Skips array copy when offset is 0 (common case in batch loops).
     */
    public static double[] flatMmul(double[] a, int aOff, int m, int k,
                                    double[] b, int bOff, int n) {
        if (aOff == 0 && bOff == 0) {
            return flatMmul(a, m, k, b, n);
        }
        double[] aSlice = java.util.Arrays.copyOfRange(a, aOff, aOff + m * k);
        double[] bSlice = java.util.Arrays.copyOfRange(b, bOff, bOff + k * n);
        return flatMmul(aSlice, m, k, bSlice, n);
    }

    /**
     * Batch flat GEMM: for each b, C_b[m×n] = A_b[m×k] @ B_b[k×n].
     * Dispatch: GPU per-batch → HPC batch → SIMD batch → per-batch scalar.
     */
    public static double[] flatMmulBatched(double[] a, double[] b,
                                           int batch, int m, int k, int n) {
        double[] c = new double[batch * m * n];
        int mk = m * k;
        int kn = k * n;
        int mn = m * n;
        // 0. Try GPU per-batch (fastest when available)
        if (GpuConfig.allowAttempts() && GpuOptionalRuntime.isGpuAvailable()) {
            boolean allGpu = true;
            for (int bi = 0; bi < batch; bi++) {
                if (!GpuGemm.tryFlatMatMul(a, bi * mk, m, k, b, bi * kn, n, c, bi * mn)) {
                    allGpu = false;
                    break;
                }
            }
            if (allGpu) return c;
            // GPU failed mid-batch — reset and fall through to next backend
            java.util.Arrays.fill(c, 0.0);
        }
        // 1. Try HPC batch native
        if (HpcIm2col.tryFlatDgemmBatch(batch, m, n, k, a, b, c)) {
            return c;
        }
        // 2. Try SIMD batch via reflection
        if (MH_FLAT_MMUL_BATCHED != null) {
            try {
                return (double[]) MH_FLAT_MMUL_BATCHED.invoke(a, b, batch, m, k, n);
            } catch (Throwable t) {
                // fall through to scalar
            }
        }
        // 3. Scalar fallback: per-batch
        for (int bi = 0; bi < batch; bi++) {
            flatMmulScalar(a, bi * mk, m, k, b, bi * kn, n, c, bi * mn);
        }
        return c;
    }

    /**
     * Cache-blocked transpose of a row-major flat matrix.
     * dst[j*m+i] = src[i*n+j].
     */
    public static double[] flatTranspose(double[] src, int m, int n) {
        // 1. Try SIMD via reflection
        if (MH_FLAT_TRANSPOSE != null) {
            try {
                return (double[]) MH_FLAT_TRANSPOSE.invoke(src, m, n);
            } catch (Throwable t) {
                // fall through to scalar
            }
        }
        // 2. Scalar fallback
        double[] dst = new double[n * m];
        int blockSize = Math.min(64, Math.min(m, n));
        for (int i = 0; i < m; i += blockSize) {
            int iEnd = Math.min(i + blockSize, m);
            for (int j = 0; j < n; j += blockSize) {
                int jEnd = Math.min(j + blockSize, n);
                for (int ii = i; ii < iEnd; ii++) {
                    for (int jj = j; jj < jEnd; jj++) {
                        dst[jj * m + ii] = src[ii * n + jj];
                    }
                }
            }
        }
        return dst;
    }

    /**
     * Scalar flat GEMM — no Vector API dependency.
     * i-k-j loop with zero-skip optimization.
     */
    private static void flatMmulScalar(double[] a, int aOff, int m, int k,
                                       double[] b, int bOff, int n,
                                       double[] c, int cOff) {
        for (int i = 0; i < m; i++) {
            int cRow = cOff + i * n;
            int aRow = aOff + i * k;
            for (int kk = 0; kk < k; kk++) {
                double aik = a[aRow + kk];
                if (aik == 0.0) continue;
                int bRow = bOff + kk * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += aik * b[bRow + j];
                }
            }
        }
    }
}
