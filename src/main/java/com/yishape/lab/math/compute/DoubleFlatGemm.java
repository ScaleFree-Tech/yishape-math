package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.compute.gpu.GpuGemm;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.util.YishapeLogger;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;

/**
 * Flat GEMM dispatch: HPC native → SIMD (Vector API) → scalar fallback.
 * <p>
 * Follows the same optional-acceleration pattern as {@link com.yishape.lab.math.compute.hpc.HpcIm2col}:
 * detection is reflection-based so this class loads successfully even when
 * {@code jdk.incubator.vector} is absent.
 */
public final class DoubleFlatGemm {

    private static final YishapeLogger log = YishapeLogger.getLogger(DoubleFlatGemm.class);

    private DoubleFlatGemm() {}

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
        boolean logFine = log.isDebugEnabled();
        double[] c = new double[m * n];
        // 0. Try GPU first (fastest when available)
        if (GpuGemm.tryFlatMatMul(a, b, c, m, k, n)) {
            if (logFine) log.debug("flatMmul GPU [{},{}]@[{},{}]  flops={}", m, k, k, n, (long) m * n * k);
            return c;
        }
        if (logFine) {
            log.debug("flatMmul GPU skip [{},{}]@[{},{}] flops={} allow={} avail={}",
                m, k, k, n, (long) m * n * k, GpuConfig.allowAttempts(), GpuOptionalRuntime.isGpuAvailable());
        }
        // 1. Try HPC native
        if (HpcIm2col.tryFlatDgemm(m, n, k, a, b, c)) {
            if (logFine) log.debug("flatMmul HPC [{},{}]@[{},{}]", m, k, k, n);
            return c;
        }
        // 2. Try SIMD via reflection
        if (MH_FLAT_MMUL != null) {
            try {
                double[] simdResult = (double[]) MH_FLAT_MMUL.invoke(a, m, k, b, n);
                if (logFine) log.debug("flatMmul SIMD [{},{}]@[{},{}]", m, k, k, n);
                return simdResult;
            } catch (Throwable t) {
                // SIMD invoke failed — fall through to scalar
            }
        }
        // 3. Scalar fallback
        if (logFine) log.debug("flatMmul SCALAR [{},{}]@[{},{}]", m, k, k, n);
        flatMmulScalar(a, 0, m, k, b, 0, n, c, 0);
        return c;
    }

    /**
     * Flat GEMM with transpose: C[m×n] = op(A)[m×k] @ op(B)[k×n].
     * transp: 0=NN, 1=TN, 2=NT, 3=TT.
     * Dispatch: GPU → HPC → SIMD → scalar.
     */
    public static double[] flatMmulTransp(double[] a, int m, int k, double[] b, int n, int transp) {
        boolean logFine = log.isDebugEnabled();
        double[] c = new double[m * n];
        // 0. Try GPU first
        if (GpuGemm.tryFlatMatMulTransp(a, b, c, m, k, n, transp)) {
            if (logFine) log.debug("flatMmulTransp GPU [{},{}]@[{},{}] transp={}", m, k, k, n, transp);
            return c;
        }
        // 1. If no transpose needed, use full HPC → SIMD → scalar dispatch
        if (transp == 0) {
            return flatMmul(a, m, k, b, n);
        }
        // 2. Try HPC with transp flags — transpose happens in Rust (auto-vectorized),
        //    eliminating Java-side flatTranspose allocations.
        if (HpcIm2col.tryFlatDgemmTransp(m, n, k, a, b, c, transp)) {
            if (logFine) log.debug("flatMmulTransp HPC transp [{},{}]@[{},{}] transp={}", m, k, k, n, transp);
            return c;
        }
        // 3. Java-side transpose as fallback, then use fast dispatch
        boolean ta = (transp & 1) != 0;
        boolean tb = (transp & 2) != 0;
        double[] aEff = ta ? flatTranspose(a, k, m) : a;
        int aRows = m, aCols = k;
        double[] bEff = tb ? flatTranspose(b, n, k) : b;
        int bCols = n;
        if (logFine) log.debug("flatMmulTransp SIMD/scalar via transpose [{},{}]@[{},{}] transp={}", m, k, k, n, transp);
        double[] result = flatMmul(aEff, aRows, aCols, bEff, bCols);
        System.arraycopy(result, 0, c, 0, c.length);
        return c;
    }

    private static void flatMmulTranspScalar(double[] a, int m, int k, double[] b, int n, double[] c, int transp) {
        boolean ta = (transp & 1) != 0;
        boolean tb = (transp & 2) != 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int kk = 0; kk < k; kk++) {
                    double av = ta ? a[kk * m + i] : a[i * k + kk];
                    double bv = tb ? b[j * k + kk] : b[kk * n + j];
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
     * Flat GEMM writing to pre-allocated {@code out}.
     * out.length must be &gt;= m * n. Caller is responsible for sizing.
     * Dispatch: GPU → HPC → SIMD → scalar.
     */
    public static void flatMmul(double[] a, int m, int k, double[] b, int n, double[] out) {
        boolean logFine = log.isDebugEnabled();
        // 0. Try GPU first
        if (GpuGemm.tryFlatMatMul(a, b, out, m, k, n)) {
            if (logFine) log.debug("flatMmul(out) GPU [{},{}]@[{},{}]", m, k, k, n);
            return;
        }
        // 1. Try HPC native
        if (HpcIm2col.tryFlatDgemm(m, n, k, a, b, out)) {
            if (logFine) log.debug("flatMmul(out) HPC [{},{}]@[{},{}]", m, k, k, n);
            return;
        }
        // 2. Try SIMD via reflection
        if (MH_FLAT_MMUL != null) {
            try {
                double[] simdResult = (double[]) MH_FLAT_MMUL.invoke(a, m, k, b, n);
                System.arraycopy(simdResult, 0, out, 0, m * n);
                if (logFine) log.debug("flatMmul(out) SIMD [{},{}]@[{},{}]", m, k, k, n);
                return;
            } catch (Throwable t) {
                // fall through to scalar
            }
        }
        // 3. Scalar fallback
        if (logFine) log.debug("flatMmul(out) SCALAR [{},{}]@[{},{}]", m, k, k, n);
        flatMmulScalar(a, 0, m, k, b, 0, n, out, 0);
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
     * Cache-blocked transpose writing to pre-allocated {@code dst}.
     * dst.length must be &gt;= m * n. Caller is responsible for sizing.
     * dst[j*m+i] = src[i*n+j].
     */
    public static void flatTranspose(double[] src, int m, int n, double[] dst) {
        // 1. Try SIMD via reflection
        if (MH_FLAT_TRANSPOSE != null) {
            try {
                double[] simdResult = (double[]) MH_FLAT_TRANSPOSE.invoke(src, m, n);
                System.arraycopy(simdResult, 0, dst, 0, m * n);
                return;
            } catch (Throwable t) {
                // fall through to scalar
            }
        }
        // 2. Scalar fallback
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

    // ======================== Element-wise fused ops ========================

    /**
     * Fused in-place DAXPY: {@code y[i] = a * x[i] + b * y[i]}.
     * Used by EMA shadow updates and other linear combinations.
     * Dispatch: HPC (Rust native) → SISD (JIT auto-vectorized scalar).
     * GPU not used: DAXPY is memory-bandwidth-bound; PCIe transfer overhead dominates.
     *
     * @param a coefficient for x
     * @param x source array
     * @param b coefficient for y (the scaling factor)
     * @param y target array (modified in-place)
     */
    public static void fusedDaxpyInPlace(double a, double[] x, double b, double[] y) {
        int len = y.length;
        if (x.length != len) {
            throw new IllegalArgumentException("x.length=" + x.length + " != y.length=" + len);
        }
        // 0. Try HPC native (Rust) — only for arrays large enough to justify FFI overhead
        if (len >= 4096 && com.yishape.lab.math.compute.hpc.HpcNormStats.tryFusedDaxpyInPlace(a, x, b, y)) {
            return;
        }
        // 1. SISD: JIT auto-vectorizes simple counted loops with stable stride
        if (b == 0.0) {
            if (a == 1.0) {
                System.arraycopy(x, 0, y, 0, len);
            } else {
                for (int i = 0; i < len; i++) y[i] = a * x[i];
            }
        } else {
            for (int i = 0; i < len; i++) {
                y[i] = a * x[i] + b * y[i];
            }
        }
    }
}
