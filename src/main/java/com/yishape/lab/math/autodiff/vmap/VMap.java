package com.yishape.lab.math.autodiff.vmap;

import java.io.Serializable;

import com.yishape.lab.math.compute.ComputerConfig;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.util.YishapeLogger;

/**
 * Unified vmap dispatch facade. Detects GPU / HPC / SIMD availability and
 * delegates operations to the fastest available backend.
 *
 * <p>Pattern mirrors {@link com.yishape.lab.math.compute.DoubleVectorComputer}:
 * lazy-singleton instantiation, fast-path caching, and a {@code fetchVMap} method
 * that resolves the best backend per invocation based on data size.
 *
 * <p>Dispatch chain: GPU → HPC → SIMD → SISD.
 *
 * @author lteb2
 */
public class VMap implements IVMap, Serializable {

    private static final YishapeLogger log = YishapeLogger.getLogger(VMap.class);

    /** Shared singleton — use this to access vmap operations. */
    public static final VMap INSTANCE = new VMap();

    // ---- lazy singletons ----
    private static volatile GpuVMap gpu;
    private static volatile HpcVMap hpc;
    private static volatile SimdVMap simd;
    private static volatile SisdVMap sisd;

    // ---- cached availability flags ----
    private static volatile Boolean hpcSupported;
    private static volatile Boolean gpuSupported;

    // ---- cached resolved base (best of HPC/SIMD/SISD) ----
    private static volatile IVMap resolvedBase;

    // ---- the delegate used by this VMap instance ----
    private transient volatile IVMap base;

    static {
        // Lazy detection on first use — no eager checks
    }

    // ---- detection ----

    private static boolean checkIfGpuSupported() {
        if (gpuSupported == null) {
            gpuSupported = ComputerConfig.checkIfGPUSupported();
        }
        return gpuSupported;
    }

    private static boolean checkIfHpcSupported() {
        if (hpcSupported == null) {
            hpcSupported = HpcOptionalRuntime.isNativeRuntimeAvailable();
            if (hpcSupported) {
                log.info("HPC vmap backend available");
            }
        }
        return hpcSupported;
    }

    private static boolean checkIfSimdSupported() {
        return ComputerConfig.checkIfSIMDSupported();
    }

    // ---- dispatch ----

    private IVMap fetchVMap(long totalElements) {
        // Fast path: resolved base cached, GPU not applicable
        IVMap baseImpl = resolvedBase;
        if (baseImpl != null
                && !(ComputerConfig.USE_GPU && checkIfGpuSupported()
                     && totalElements > VMapConfig.gpuVMapThreshold())) {
            return baseImpl;
        }

        // Resolve base: HPC → SIMD → SISD
        IVMap bestBase = resolveBase();
        resolvedBase = bestBase;

        // Wrap with GPU if eligible
        if (ComputerConfig.USE_GPU && checkIfGpuSupported()
                && totalElements > VMapConfig.gpuVMapThreshold()) {
            if (gpu == null) {
                synchronized (VMap.class) {
                    if (gpu == null) {
                        gpu = new GpuVMap(bestBase);
                        log.info("GPU VMap initialized, wrapping {}",
                            bestBase instanceof HpcVMap ? "HPC" :
                            bestBase instanceof SimdVMap ? "SIMD" : "SISD");
                    }
                }
            }
            if (gpu != null) {
                return gpu;
            }
        }

        return bestBase;
    }

    private static IVMap resolveBase() {
        if (checkIfHpcSupported()) {
            if (hpc == null) {
                synchronized (VMap.class) {
                    if (hpc == null) {
                        hpc = new HpcVMap(resolveSimdOrSisd());
                        log.info("HPC VMap initialized");
                    }
                }
            }
            return hpc;
        }
        return resolveSimdOrSisd();
    }

    private static IVMap resolveSimdOrSisd() {
        if (ComputerConfig.USE_SIMD && checkIfSimdSupported()) {
            if (simd == null) {
                synchronized (VMap.class) {
                    if (simd == null) {
                        simd = new SimdVMap(getSisd());
                        log.info("SIMD VMap initialized");
                    }
                }
            }
            return simd;
        }
        return getSisd();
    }

    private static SisdVMap getSisd() {
        if (sisd == null) {
            synchronized (VMap.class) {
                if (sisd == null) {
                    sisd = new SisdVMap();
                }
            }
        }
        return sisd;
    }

    // ---- IVMap methods (delegate via fetchVMap) ----

    @Override
    public double[] stack(double[][] arrays) {
        long total = ((long) arrays.length) * arrays[0].length;
        return fetchVMap(total).stack(arrays);
    }

    @Override
    public double[][] unstack(double[] flat, int n, int d) {
        long total = (long) n * d;
        return fetchVMap(total).unstack(flat, n, d);
    }

    @Override
    public double[] batchReduce(double[] flat, int n, int d, ReduceOperation op) {
        long total = (long) n * d;
        return fetchVMap(total).batchReduce(flat, n, d, op);
    }

    @Override
    public double[] binaryOperate(double[] a, double[] b, BinaryOperation op) {
        return fetchVMap(a.length).binaryOperate(a, b, op);
    }

    @Override
    public double[] unaryOp(double[] a, UniversalOperation op, double param) {
        return fetchVMap(a.length).unaryOp(a, op, param);
    }

    @Override
    public double[] batchMatmul(double[] a, double[] b, int batch, int m, int k, int n) {
        long total = (long) batch * m * k + (long) batch * k * n;
        return fetchVMap(total).batchMatmul(a, b, batch, m, k, n);
    }

    // ---- DL ops ----

    @Override
    public double[] batchSoftmax(double[] flat, int batch, int dim) {
        return fetchVMap((long) batch * dim).batchSoftmax(flat, batch, dim);
    }

    @Override
    public double[] batchLayerNorm(double[] x, double[] gamma, double[] beta, int batch, int dim, float eps) {
        return fetchVMap((long) batch * dim).batchLayerNorm(x, gamma, beta, batch, dim, eps);
    }

    @Override
    public double[] im2col(double[] input, int C, int H, int W, int kH, int kW, int stride, int pad) {
        return fetchVMap((long) C * H * W).im2col(input, C, H, W, kH, kW, stride, pad);
    }

    @Override
    public double[] batchDropout(double[] flat, double p, long seed) {
        return fetchVMap(flat.length).batchDropout(flat, p, seed);
    }
}
