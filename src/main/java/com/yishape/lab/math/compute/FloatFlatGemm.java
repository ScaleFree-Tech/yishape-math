package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * 单精度 Flat GEMM 调度：SIMD → scalar fallback.
 * <p>
 * 镜像 {@link FlatGemm}，但操作 {@code float[]} 数组，
 * 专为推理路径设计。
 */
public final class FloatFlatGemm {

    private static final Logger LOG = Logger.getLogger(FloatFlatGemm.class.getName());

    private FloatFlatGemm() {}

    /** Cached MethodHandle for SIMDFloatComputer.flatMmul — null if unavailable */
    private static final MethodHandle MH_FLAT_MMUL;
    /** Cached MethodHandle for SIMDFloatComputer.flatMmulBatched — null if unavailable */
    private static final MethodHandle MH_FLAT_MMUL_BATCHED;
    /** Cached MethodHandle for SIMDFloatComputer.flatTranspose — null if unavailable */
    private static final MethodHandle MH_FLAT_TRANSPOSE;

    static {
        MethodHandle mmul = null;
        MethodHandle batched = null;
        MethodHandle transpose = null;
        if (ComputerConfig.checkIfSIMDSupported()) {
            try {
                Class<?> cls = Class.forName("com.yishape.lab.math.compute.SIMDFloatComputer");
                MethodHandles.Lookup lookup = MethodHandles.lookup();

                Method m1 = cls.getMethod("flatMmul", float[].class, int.class, int.class, float[].class, int.class);
                mmul = lookup.unreflect(m1);

                Method m2 = cls.getMethod("flatMmulBatched", float[].class, float[].class, int.class, int.class, int.class, int.class);
                batched = lookup.unreflect(m2);

                Method m3 = cls.getMethod("flatTranspose", float[].class, int.class, int.class);
                transpose = lookup.unreflect(m3);
            } catch (Throwable t) {
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
     * C[m×n] = A[m×k] @ B[k×n], all row-major flat float[] arrays.
     * Dispatch: GPU → HPC (via double bridge) → SIMD → scalar.
     */
    public static float[] flatMmul(float[] a, int m, int k, float[] b, int n) {
        boolean logFine = LOG.isLoggable(Level.FINE);
        float[] c = new float[m * n];
        // 0. Try GPU f32 directly
        if (GpuConfig.allowAttempts() && GpuOptionalRuntime.isGpuAvailable()) {
            float[] gpuResult = GpuOptionalRuntime.tryFloatFlatMatMul(a, b, m, k, n);
            if (gpuResult != null) {
                if (logFine) LOG.fine("flatMmul GPU f32 [" + m + "," + k + "]@[" + k + "," + n + "]");
                return gpuResult;
            }
        }
        // 1. Try HPC native f32 directly (avoids float↔double conversion)
        if (HpcIm2col.tryFlatSgemm(m, n, k, a, b, c)) {
            if (logFine) LOG.fine("flatMmul HPC f32 [" + m + "," + k + "]@[" + k + "," + n + "]");
            return c;
        }
        // 2. Try HPC native via double bridge (check availability first to avoid wasteful allocations)
        if (HpcConfig.allowAttempts() && HpcOptionalRuntime.isNativeRuntimeAvailable()) {
            double[] aD = toDouble(a);
            double[] bD = toDouble(b);
            double[] cD = new double[m * n];
            if (HpcIm2col.tryFlatDgemm(m, n, k, aD, bD, cD)) {
                toFloatInPlace(cD, c);
                if (logFine) LOG.fine("flatMmul HPC f64 [" + m + "," + k + "]@[" + k + "," + n + "]");
                return c;
            }
        }
        // 2. Try SIMD via reflection
        if (MH_FLAT_MMUL != null) {
            try {
                float[] simdResult = (float[]) MH_FLAT_MMUL.invoke(a, m, k, b, n);
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
     * Float flat GEMM with transpose: C[m×n] = op(A)[m×k] @ op(B)[k×n].
     * transp: 0=NN, 1=TN, 2=NT, 3=TT.
     */
    public static float[] flatMmulTransp(float[] a, int m, int k, float[] b, int n, int transp) {
        boolean logFine = LOG.isLoggable(Level.FINE);
        float[] c = new float[m * n];
        // 0. Try GPU via double bridge (no native f32 transpose path yet)
        if (GpuConfig.allowAttempts() && GpuOptionalRuntime.isGpuAvailable()) {
            double[] da = toDouble(a);
            double[] db = toDouble(b);
            double[] dc = GpuOptionalRuntime.tryFlatMatMulTransp(da, db, m, k, n, transp);
            if (dc != null) {
                toFloatInPlace(dc, c);
                if (logFine) LOG.fine("flatMmulTransp GPU f64 bridge [" + m + "," + k + "]@[" + k + "," + n + "] transp=" + transp);
                return c;
            }
        }
        // 1. If no transpose needed, use full dispatch
        if (transp == 0) return flatMmul(a, m, k, b, n);
        // 2. Transpose inputs as needed, then use fast dispatch
        boolean ta = (transp & 1) != 0;
        boolean tb = (transp & 2) != 0;
        float[] aEff = ta ? flatTranspose(a, k, m) : a;
        int aRows = m, aCols = k;
        float[] bEff = tb ? flatTranspose(b, n, k) : b;
        int bCols = n;
        if (logFine) LOG.fine("flatMmulTransp HPC/SIMD via transpose f32 [" + m + "," + k + "]@[" + k + "," + n + "] transp=" + transp);
        float[] result = flatMmul(aEff, aRows, aCols, bEff, bCols);
        System.arraycopy(result, 0, c, 0, c.length);
        return c;
    }

    private static void flatMmulTranspScalar(float[] a, int m, int k, float[] b, int n, float[] c, int transp) {
        boolean ta = (transp & 1) != 0;
        boolean tb = (transp & 2) != 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0;
                for (int kk = 0; kk < k; kk++) {
                    float av = ta ? a[kk * m + i] : a[i * k + kk];
                    float bv = tb ? b[j * k + kk] : b[kk * n + j];
                    sum += av * bv;
                }
                c[i * n + j] = sum;
            }
        }
    }

    /**
     * Offset-aware flat GEMM for batched use.
     * Skips array copy when offset is 0 (common case in batch loops).
     */
    public static float[] flatMmul(float[] a, int aOff, int m, int k,
                                   float[] b, int bOff, int n) {
        if (aOff == 0 && bOff == 0) {
            return flatMmul(a, m, k, b, n);
        }
        float[] aSlice = java.util.Arrays.copyOfRange(a, aOff, aOff + m * k);
        float[] bSlice = java.util.Arrays.copyOfRange(b, bOff, bOff + k * n);
        return flatMmul(aSlice, m, k, bSlice, n);
    }

    /**
     * C[m×n] = A[m×k] @ B[k×n] writing into a pre-allocated output buffer.
     * <p>Use this to avoid allocation when the caller owns a pooled buffer.</p>
     */
    public static void flatMmul(float[] a, int m, int k, float[] b, int n,
                                 float[] c, int cOff) {
        float[] tmp = flatMmul(a, m, k, b, n);
        System.arraycopy(tmp, 0, c, cOff, m * n);
    }

    /**
     * Batch flat GEMM: for each b, C_b[m×n] = A_b[m×k] @ B_b[k×n].
     * Dispatch: GPU per-batch → SIMD batch → per-batch scalar.
     */
    public static float[] flatMmulBatched(float[] a, float[] b,
                                          int batch, int m, int k, int n) {
        float[] c = new float[batch * m * n];
        int mk = m * k;
        int kn = k * n;
        int mn = m * n;
        // 0. Try GPU per-batch (fastest when available)
        if (GpuConfig.allowAttempts() && GpuOptionalRuntime.isGpuAvailable()) {
            boolean allGpu = true;
            for (int bi = 0; bi < batch; bi++) {
                float[] aSlice = (bi == 0 && mk == a.length) ? a :
                    java.util.Arrays.copyOfRange(a, bi * mk, bi * mk + mk);
                float[] bSlice = (bi == 0 && kn == b.length) ? b :
                    java.util.Arrays.copyOfRange(b, bi * kn, bi * kn + kn);
                float[] gpuResult = GpuOptionalRuntime.tryFloatFlatMatMul(aSlice, bSlice, m, k, n);
                if (gpuResult != null) {
                    System.arraycopy(gpuResult, 0, c, bi * mn, mn);
                } else {
                    allGpu = false;
                    break;
                }
            }
            if (allGpu) return c;
            // GPU failed mid-batch — reset and fall through
            java.util.Arrays.fill(c, 0.0f);
        }
        // 1. Try SIMD batch via reflection
        if (MH_FLAT_MMUL_BATCHED != null) {
            try {
                return (float[]) MH_FLAT_MMUL_BATCHED.invoke(a, b, batch, m, k, n);
            } catch (Throwable t) {
                // fall through to scalar
            }
        }
        // 2. Scalar fallback: per-batch
        for (int bi = 0; bi < batch; bi++) {
            flatMmulScalar(a, bi * mk, m, k, b, bi * kn, n, c, bi * mn);
        }
        return c;
    }

    /**
     * Cache-blocked transpose of a row-major flat matrix.
     * dst[j*m+i] = src[i*n+j].
     */
    public static float[] flatTranspose(float[] src, int m, int n) {
        // 1. Try SIMD via reflection
        if (MH_FLAT_TRANSPOSE != null) {
            try {
                return (float[]) MH_FLAT_TRANSPOSE.invoke(src, m, n);
            } catch (Throwable t) {
                // fall through to scalar
            }
        }
        // 2. Scalar fallback
        float[] dst = new float[n * m];
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
     * Scalar flat GEMM — no SIMD dependency.
     * i-k-j loop with zero-skip optimization.
     */
    private static void flatMmulScalar(float[] a, int aOff, int m, int k,
                                       float[] b, int bOff, int n,
                                       float[] c, int cOff) {
        for (int i = 0; i < m; i++) {
            int cRow = cOff + i * n;
            int aRow = aOff + i * k;
            for (int kk = 0; kk < k; kk++) {
                float aik = a[aRow + kk];
                if (aik == 0.0f) continue;
                int bRow = bOff + kk * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += aik * b[bRow + j];
                }
            }
        }
    }

    /** Convert float[] to double[] for HPC bridge. */
    private static double[] toDouble(float[] f) {
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) d[i] = f[i];
        return d;
    }

    /** Convert double[] to float[] in-place (HPC bridge output). */
    private static void toFloatInPlace(double[] d, float[] f) {
        for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
    }
}
