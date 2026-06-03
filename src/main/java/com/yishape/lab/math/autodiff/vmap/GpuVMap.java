package com.yishape.lab.math.autodiff.vmap;

import com.yishape.lab.math.compute.DoubleFlatGemm;
import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.gpu.GpuReduce;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import static com.yishape.lab.math.compute.ops.BinaryOperation.ADD;
import static com.yishape.lab.math.compute.ops.BinaryOperation.DIVIDE;
import static com.yishape.lab.math.compute.ops.BinaryOperation.MULTIPLY;
import static com.yishape.lab.math.compute.ops.BinaryOperation.SUBTRACT;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MAX;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MEAN;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MIN;
import static com.yishape.lab.math.compute.ops.ReduceOperation.PROD;
import static com.yishape.lab.math.compute.ops.ReduceOperation.SUM;
import static com.yishape.lab.math.compute.ops.ReduceOperation.VARIANCE;
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
 * GPU vmap backend. Wraps a delegate (HpcVMap, SimdVMap, or SisdVMap).
 * Tries GPU native for all operations; falls through to the delegate on failure.
 *
 * <p>GPU provides native batched reduction ({@code outer/inner} semantics),
 * element-wise ops, GEMM, softmax, layer norm, and im2col.
 *
 * @author lteb2
 */
final class GpuVMap implements IVMap {

    private static final YishapeLogger log = YishapeLogger.getLogger(GpuVMap.class);

    private final IVMap delegate;

    GpuVMap(IVMap delegate) {
        this.delegate = delegate;
    }

    private boolean gpuAvailable() {
        return GpuConfig.allowAttempts() && GpuOptionalRuntime.isGpuAvailable();
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

    // ---- batchReduce: GPU native outer/inner reduce ----

    @Override
    public double[] batchReduce(double[] flat, int n, int d, ReduceOperation op) {
        if (gpuAvailable()) {
            double[] result = tryGpuReduce(flat, n, d, op);
            if (result != null) {
                return result;
            }
            if (log.isDebugEnabled()) {
                log.debug("GPU batched reduce failed for op={}, falling to delegate", op);
            }
        }
        return delegate.batchReduce(flat, n, d, op);
    }

    private static double[] tryGpuReduce(double[] flat, int n, int d,
                                          ReduceOperation op) {
        if (op == ReduceOperation.STANDARD_DEVIATION) {
            double[] var = GpuReduce.tryReduce(GpuReduce.VARIANCE, flat, n, d);
            if (var != null) {
                double[] out = new double[var.length];
                for (int i = 0; i < var.length; i++) {
                    out[i] = Math.sqrt(var[i]);
                }
                return out;
            }
            return null;
        }
        int gpuOp = switch (op) {
            case SUM -> GpuReduce.SUM;
            case MEAN -> GpuReduce.MEAN;
            case MAX -> GpuReduce.MAX;
            case MIN -> GpuReduce.MIN;
            case PROD -> GpuReduce.PROD;
            case VARIANCE -> GpuReduce.VARIANCE;
            default -> -1;
        };
        if (gpuOp == -1) return null;
        return GpuReduce.tryReduce(gpuOp, flat, n, d);
    }

    // ---- element-wise: GPU first, fallback to delegate ----

    @Override
    public double[] binaryOperate(double[] a, double[] b, BinaryOperation op) {
        if (gpuAvailable() && a.length >= GpuConfig.elementwiseMinElements()) {
            double[] result = tryGpuBinaryOp(a, b, op);
            if (result != null) {
                return result;
            }
        }
        return delegate.binaryOperate(a, b, op);
    }

    @Override
    public double[] unaryOp(double[] a, UniversalOperation op, double param) {
        if (gpuAvailable() && a.length >= GpuConfig.activationMinElements()) {
            double[] result = tryGpuActivation(a, op);
            if (result != null) {
                return result;
            }
        }
        return delegate.unaryOp(a, op, param);
    }

    private static double[] tryGpuBinaryOp(double[] a, double[] b, BinaryOperation op) {
        try {
            return switch (op) {
                case ADD -> GpuOptionalRuntime.tryAdd(a, b);
                case SUBTRACT -> GpuOptionalRuntime.trySub(a, b);
                case MULTIPLY -> GpuOptionalRuntime.tryMul(a, b);
                case DIVIDE -> GpuOptionalRuntime.tryDiv(a, b);
                default -> null;
            };
        } catch (Throwable t) {
            return null;
        }
    }

    private static double[] tryGpuActivation(double[] a, UniversalOperation op) {
        try {
            return switch (op) {
                case RELU -> GpuOptionalRuntime.tryRelu(a);
                case GELU -> GpuOptionalRuntime.tryGelu(a);
                case SIGMOID -> GpuOptionalRuntime.trySigmoid(a);
                case TANH -> GpuOptionalRuntime.tryTanh(a);
                case EXP -> GpuOptionalRuntime.tryExp(a);
                case LOG -> GpuOptionalRuntime.tryLog(a);
                case ABS -> GpuOptionalRuntime.tryAbs(a);
                case SQRT -> GpuOptionalRuntime.trySqrt(a);
                case SIN -> GpuOptionalRuntime.trySin(a);
                case COS -> GpuOptionalRuntime.tryCos(a);
                default -> null;
            };
        } catch (Throwable t) {
            return null;
        }
    }

    // ---- batchMatmul via DoubleFlatGemm (internal GPU dispatch) ----

    @Override
    public double[] batchMatmul(double[] a, double[] b, int batch, int m, int k, int n) {
        return DoubleFlatGemm.flatMmulBatched(a, b, batch, m, k, n);
    }

    // ---- DL ops ----

    @Override
    public double[] batchSoftmax(double[] flat, int batch, int dim) {
        if (gpuAvailable()) {
            double[] result = GpuOptionalRuntime.trySoftmax(flat, batch, dim);
            if (result != null) {
                return result;
            }
        }
        return delegate.batchSoftmax(flat, batch, dim);
    }

    @Override
    public double[] batchLayerNorm(double[] x, double[] gamma, double[] beta, int batch, int dim, float eps) {
        if (gpuAvailable()) {
            double[] result = GpuOptionalRuntime.tryLayerNorm(x, gamma, beta, batch, dim, eps);
            if (result != null) {
                return result;
            }
        }
        return delegate.batchLayerNorm(x, gamma, beta, batch, dim, eps);
    }

    @Override
    public double[] im2col(double[] input, int C, int H, int W, int kH, int kW, int stride, int pad) {
        if (gpuAvailable()) {
            int outH = (H + 2 * pad - kH) / stride + 1;
            int outW = (W + 2 * pad - kW) / stride + 1;
            double[] result = GpuOptionalRuntime.tryIm2col(input, C, H, W, outH, outW, kH, kW, stride, pad);
            if (result != null) {
                return result;
            }
        }
        return delegate.im2col(input, C, H, W, kH, kW, stride, pad);
    }

    @Override
    public double[] batchDropout(double[] flat, double p, long seed) {
        return delegate.batchDropout(flat, p, seed);
    }
}
