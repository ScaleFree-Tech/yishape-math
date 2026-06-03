package com.yishape.lab.math.autodiff.vmap;

import com.yishape.lab.math.compute.DoubleFlatGemm;
import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import static com.yishape.lab.math.compute.ops.BinaryOperation.ADD;
import static com.yishape.lab.math.compute.ops.BinaryOperation.DIVIDE;
import static com.yishape.lab.math.compute.ops.BinaryOperation.MULTIPLY;
import static com.yishape.lab.math.compute.ops.BinaryOperation.SUBTRACT;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import static com.yishape.lab.math.compute.ops.UniversalOperation.ABS;
import static com.yishape.lab.math.compute.ops.UniversalOperation.COS;
import static com.yishape.lab.math.compute.ops.UniversalOperation.EXP;
import static com.yishape.lab.math.compute.ops.UniversalOperation.GELU;
import static com.yishape.lab.math.compute.ops.UniversalOperation.LOG;
import static com.yishape.lab.math.compute.ops.UniversalOperation.RELU;
import static com.yishape.lab.math.compute.ops.UniversalOperation.SIGMOID;
import static com.yishape.lab.math.compute.ops.UniversalOperation.SIN;
import static com.yishape.lab.math.compute.ops.UniversalOperation.SQRT;
import static com.yishape.lab.math.compute.ops.UniversalOperation.TANH;
import com.yishape.lab.util.YishapeLogger;

/**
 * HPC (Rust native) vmap backend. Uses HPC for element-wise ops, batched
 * GEMM, and im2col. No native batched reduction — delegates reduction,
 * softmax, layer norm, and dropout to SIMD or SISD.
 *
 * @author lteb2
 */
final class HpcVMap implements IVMap {

    private static final YishapeLogger log = YishapeLogger.getLogger(HpcVMap.class);

    private final IVMap delegate;

    HpcVMap(IVMap delegate) {
        this.delegate = delegate;
    }

    private boolean hpcAvailable() {
        return HpcConfig.allowAttempts() && HpcOptionalRuntime.isNativeRuntimeAvailable();
    }

    // ---- stack / unstack (pure memory) ----

    @Override
    public double[] stack(double[][] arrays) {
        return delegate.stack(arrays);
    }

    @Override
    public double[][] unstack(double[] flat, int n, int d) {
        return delegate.unstack(flat, n, d);
    }

    // ---- batchReduce: no native HPC batched reduce, delegate ----

    @Override
    public double[] batchReduce(double[] flat, int n, int d, ReduceOperation op) {
        return delegate.batchReduce(flat, n, d, op);
    }

    // ---- element-wise: try HPC first, fallback to delegate ----

    @Override
    public double[] binaryOperate(double[] a, double[] b, BinaryOperation op) {
        if (hpcAvailable() && a.length >= VMapConfig.hpcElementwiseThreshold()) {
            double[] result = tryHpcBinaryOp(a, b, op);
            if (result != null) {
                return result;
            }
        }
        return delegate.binaryOperate(a, b, op);
    }

    @Override
    public double[] unaryOp(double[] a, UniversalOperation op, double param) {
        if (hpcAvailable() && a.length >= VMapConfig.hpcElementwiseThreshold()) {
            double[] result = tryHpcUnaryOp(a, op);
            if (result != null) {
                return result;
            }
        }
        return delegate.unaryOp(a, op, param);
    }

    private static double[] tryHpcBinaryOp(double[] a, double[] b, BinaryOperation op) {
        String name = switch (op) {
            case ADD -> "addF64";
            case SUBTRACT -> "subF64";
            case MULTIPLY -> "mulF64";
            case DIVIDE -> "divF64";
            default -> null;
        };
        if (name == null) return null;
        return HpcOptionalRuntime.tryElementwiseF64(name, a, b);
    }

    private static double[] tryHpcUnaryOp(double[] a, UniversalOperation op) {
        String name = switch (op) {
            case RELU -> "reluF64";
            case SIGMOID -> "sigmoidF64";
            case TANH -> "tanhF64";
            case GELU -> "geluF64";
            case EXP -> "expF64";
            case LOG -> "logF64";
            case ABS -> "absF64";
            case SQRT -> "sqrtF64";
            case SIN -> "sinF64";
            case COS -> "cosF64";
            default -> null;
        };
        if (name == null) return null;
        return HpcOptionalRuntime.tryActivationF64(name, a);
    }

    // ---- batchMatmul via DoubleFlatGemm (internal HPC dispatch) ----

    @Override
    public double[] batchMatmul(double[] a, double[] b, int batch, int m, int k, int n) {
        return DoubleFlatGemm.flatMmulBatched(a, b, batch, m, k, n);
    }

    // ---- DL ops ----

    @Override
    public double[] batchSoftmax(double[] flat, int batch, int dim) {
        return delegate.batchSoftmax(flat, batch, dim);
    }

    @Override
    public double[] batchLayerNorm(double[] x, double[] gamma, double[] beta, int batch, int dim, float eps) {
        return delegate.batchLayerNorm(x, gamma, beta, batch, dim, eps);
    }

    @Override
    public double[] im2col(double[] input, int C, int H, int W, int kH, int kW, int stride, int pad) {
        if (hpcAvailable()) {
            int outH = (H + 2 * pad - kH) / stride + 1;
            int outW = (W + 2 * pad - kW) / stride + 1;
            int rows = C * kH * kW;
            int cols = outH * outW;
            double[] out = new double[rows * cols];
            if (HpcIm2col.tryIm2col(input, C, H, W, kH, kW, stride, pad, out)) {
                return out;
            }
        }
        return delegate.im2col(input, C, H, W, kH, kW, stride, pad);
    }

    @Override
    public double[] batchDropout(double[] flat, double p, long seed) {
        return delegate.batchDropout(flat, p, seed);
    }
}
