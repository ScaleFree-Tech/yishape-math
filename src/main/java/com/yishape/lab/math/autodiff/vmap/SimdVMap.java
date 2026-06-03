package com.yishape.lab.math.autodiff.vmap;

import java.util.Arrays;

import com.yishape.lab.math.compute.DoubleFlatGemm;
import com.yishape.lab.math.compute.IDoubleVectorComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.util.YishapeLogger;

/**
 * SIMD (Vector API) vmap backend. Reflectively loads {@code SIMDDoubleComputer}
 * so this class compiles even without {@code --add-modules jdk.incubator.vector}.
 * Falls back to {@link SisdVMap} when SIMD is unavailable.
 *
 * <p>DL operations ({@code batchSoftmax}, {@code batchLayerNorm}, {@code im2col},
 * {@code batchDropout}) delegate to the SISD fallback — the multi-pass per-row
 * reduction logic in these ops does not benefit meaningfully from SIMD over scalar.
 *
 * @author lteb2
 */
final class SimdVMap implements IVMap {

    private static final YishapeLogger log = YishapeLogger.getLogger(SimdVMap.class);

    private final SisdVMap sisd;
    private final IDoubleVectorComputer simdComputer;

    SimdVMap(SisdVMap sisdFallback) {
        this.sisd = sisdFallback;
        this.simdComputer = loadSimdComputer();
    }

    private static IDoubleVectorComputer loadSimdComputer() {
        try {
            Class<?> cls = Class.forName("com.yishape.lab.math.compute.SIMDDoubleComputer");
            return (IDoubleVectorComputer) cls.getDeclaredConstructor().newInstance();
        } catch (Throwable t) {
            return null;
        }
    }

    private boolean simdAvailable() {
        return simdComputer != null;
    }

    // ---- stack / unstack (pure memory) ----

    @Override
    public double[] stack(double[][] arrays) {
        return sisd.stack(arrays);
    }

    @Override
    public double[][] unstack(double[] flat, int n, int d) {
        return sisd.unstack(flat, n, d);
    }

    // ---- batchReduce: per-row SIMD reduce, fallback SISD ----

    @Override
    public double[] batchReduce(double[] flat, int n, int d, ReduceOperation op) {
        if (!simdAvailable()) {
            return sisd.batchReduce(flat, n, d, op);
        }
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            int off = i * d;
            double[] row = Arrays.copyOfRange(flat, off, off + d);
            result[i] = simdComputer.reduceOperate(row, op);
        }
        return result;
    }

    // ---- element-wise: delegate to SIMD, fallback SISD ----

    @Override
    public double[] binaryOperate(double[] a, double[] b, BinaryOperation op) {
        if (simdAvailable()) {
            return simdComputer.binaryOperate(a, b, op);
        }
        return sisd.binaryOperate(a, b, op);
    }

    @Override
    public double[] unaryOp(double[] a, UniversalOperation op, double param) {
        if (simdAvailable()) {
            return simdComputer.universalOperate(a, op, param);
        }
        return sisd.unaryOp(a, op, param);
    }

    // ---- batchMatmul via DoubleFlatGemm ----

    @Override
    public double[] batchMatmul(double[] a, double[] b, int batch, int m, int k, int n) {
        return DoubleFlatGemm.flatMmulBatched(a, b, batch, m, k, n);
    }

    // ---- DL ops: delegate to SISD ----

    @Override
    public double[] batchSoftmax(double[] flat, int batch, int dim) {
        return sisd.batchSoftmax(flat, batch, dim);
    }

    @Override
    public double[] batchLayerNorm(double[] x, double[] gamma, double[] beta, int batch, int dim, float eps) {
        return sisd.batchLayerNorm(x, gamma, beta, batch, dim, eps);
    }

    @Override
    public double[] im2col(double[] input, int C, int H, int W, int kH, int kW, int stride, int pad) {
        return sisd.im2col(input, C, H, W, kH, kW, stride, pad);
    }

    @Override
    public double[] batchDropout(double[] flat, double p, long seed) {
        return sisd.batchDropout(flat, p, seed);
    }
}
