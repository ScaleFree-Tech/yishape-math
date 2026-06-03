package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.compute.gpu.GpuGemm;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.gpu.GpuReduce;
import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import static com.yishape.lab.math.compute.ops.BinaryOperation.ADD;
import static com.yishape.lab.math.compute.ops.BinaryOperation.DIVIDE;
import static com.yishape.lab.math.compute.ops.BinaryOperation.MULTIPLY;
import static com.yishape.lab.math.compute.ops.BinaryOperation.SUBTRACT;
import com.yishape.lab.math.compute.ops.BinaryReduceOperation;
import com.yishape.lab.math.compute.ops.LogicalCompare;
import com.yishape.lab.math.compute.ops.LogicalOperation;
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

/**
 * GPU-accelerated double vector computer. Wraps the GPU facade classes
 * ({@link GpuGemm}, {@link GpuReduce}, {@link GpuOptionalRuntime}) and
 * delegates to an underlying SIMD/SISD computer for fallback.
 *
 * <p>Design follows the same pattern as the legacy {@code GPUDoubleComputer}:
 * a standalone {@link IDoubleVectorComputer} implementation that tries GPU
 * first and silently falls back to CPU on any failure. The SIMD/SISD
 * computer classes remain completely unaware of GPU existence.
 *
 * <p>Dispatch chain in {@link DoubleVectorComputer}:
 * GPU → SIMD → SISD. When {@link com.yishape.lab.math.compute.gpu.GpuSwitch#disable()}
 * is called, all GPU attempts short-circuit via {@link GpuConfig#allowAttempts()}.
 *
 * @author lteb2
 */
final class GPUDoubleComputer implements IDoubleVectorComputer {

    /** Check if GPU is available, used by {@link ComputerConfig#checkIfGPUSupported()}. */
    public static boolean isGPUAvailable() {
        return GpuOptionalRuntime.isGpuAvailable();
    }

    /** Underlying SIMD or SISD computer used as CPU fallback. */
    private final IDoubleVectorComputer delegate;

    GPUDoubleComputer(IDoubleVectorComputer delegate) {
        this.delegate = delegate;
    }

    // ==================== GPU-accelerated operations ====================

    @Override
    public double[][] mmul(double[][] a, double[][] b) {
        double[][] ob = GpuGemm.tryMatMul(a, b);
        if (ob != null) return ob;
        return delegate.mmul(a, b);
    }

    @Override
    public double[] binaryOperate(double[] x1, double[] x2, BinaryOperation operation) {
        if (x1.length >= GpuConfig.elementwiseMinElements()) {
            double[] gpuResult = tryGpuBinaryOp(x1, x2, operation);
            if (gpuResult != null) return gpuResult;
        }
        if (x1.length >= HpcConfig.activationMinElements()) {
            double[] hpcResult = tryHpcBinaryOp(x1, x2, operation);
            if (hpcResult != null) return hpcResult;
        }
        return delegate.binaryOperate(x1, x2, operation);
    }

    @Override
    public double reduceOperate(double[] x, ReduceOperation operation) {
        if (x.length >= GpuConfig.reduceMinElements()
                && (operation == ReduceOperation.SUM
                    || operation == ReduceOperation.MEAN
                    || operation == ReduceOperation.MAX
                    || operation == ReduceOperation.MIN
                    || operation == ReduceOperation.PROD
                    || operation == ReduceOperation.VARIANCE
                    || operation == ReduceOperation.STANDARD_DEVIATION)) {
            double[] gpuResult = tryGpuReduce(x, operation);
            if (gpuResult != null && gpuResult.length > 0) {
                return gpuResult[0];
            }
        }
        return delegate.reduceOperate(x, operation);
    }

    // ==================== Delegated operations (no GPU path) ====================

    @Override
    public double[] binaryOperate(double[] x1, double x2, BinaryOperation operation) {
        return delegate.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[][] binaryOperate(double[][] x1, double[][] x2, BinaryOperation operation) {
        return delegate.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[][] binaryOperate(double[][] x1, double x2, BinaryOperation operation) {
        return delegate.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[] universalOperate(double[] x, UniversalOperation operation, double additionalParam) {
        if (x.length >= GpuConfig.activationMinElements()) {
            double[] gpuResult = tryGpuActivation(x, operation);
            if (gpuResult != null) return gpuResult;
        }
        if (x.length >= HpcConfig.activationMinElements()) {
            double[] hpcResult = tryHpcActivation(x, operation);
            if (hpcResult != null) return hpcResult;
        }
        return delegate.universalOperate(x, operation, additionalParam);
    }

    @Override
    public double[][] universalOperate(double[][] x, UniversalOperation operation, double additionalParam) {
        if (x.length > 0 && x[0].length >= GpuConfig.activationMinElements()) {
            boolean anyGpu = false;
            double[][] result = new double[x.length][];
            for (int i = 0; i < x.length; i++) {
                double[] gpuRow = tryGpuActivation(x[i], operation);
                if (gpuRow != null) {
                    result[i] = gpuRow;
                    anyGpu = true;
                } else {
                    anyGpu = false;
                    break;
                }
            }
            if (anyGpu) return result;
        }
        if (x.length > 0 && x[0].length >= HpcConfig.activationMinElements()) {
            boolean anyHpc = false;
            double[][] result = new double[x.length][];
            for (int i = 0; i < x.length; i++) {
                double[] hpcRow = tryHpcActivation(x[i], operation);
                if (hpcRow != null) {
                    result[i] = hpcRow;
                    anyHpc = true;
                } else {
                    anyHpc = false;
                    break;
                }
            }
            if (anyHpc) return result;
        }
        return delegate.universalOperate(x, operation, additionalParam);
    }

    @Override
    public double reduceOperate(double[][] x, ReduceOperation operation) {
        return delegate.reduceOperate(x, operation);
    }

    @Override
    public double binaryReduceOperate(double[] x1, double[] x2, BinaryReduceOperation operation) {
        return delegate.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public double binaryReduceOperate(double[][] x1, double[][] x2, BinaryReduceOperation operation) {
        return delegate.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public double[] elementWiseMin(double[] x1, double[] x2) {
        return delegate.elementWiseMin(x1, x2);
    }

    @Override
    public double[][] elementWiseMin(double[][] x1, double[][] x2) {
        return delegate.elementWiseMin(x1, x2);
    }

    @Override
    public double[] elementWiseMax(double[] x1, double[] x2) {
        return delegate.elementWiseMax(x1, x2);
    }

    @Override
    public double[][] elementWiseMax(double[][] x1, double[][] x2) {
        return delegate.elementWiseMax(x1, x2);
    }

    @Override
    public double[] negate(double[] x) {
        return delegate.negate(x);
    }

    @Override
    public double[][] negate(double[][] x) {
        return delegate.negate(x);
    }

    @Override
    public boolean[] logicalCompare(double[] x1, double[] x2, LogicalCompare operation) {
        return delegate.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[] logicalOperate(double[] x1, LogicalOperation operation) {
        return delegate.logicalOperate(x1, operation);
    }

    @Override
    public boolean[] logicalOperate(double[] x1, double[] x2, LogicalOperation operation) {
        return delegate.logicalOperate(x1, x2, operation);
    }

    @Override
    public boolean[][] logicalCompare(double[][] x1, double[][] x2, LogicalCompare operation) {
        return delegate.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[][] logicalOperate(double[][] x1, LogicalOperation operation) {
        return delegate.logicalOperate(x1, operation);
    }

    @Override
    public boolean[][] logicalOperate(double[][] x1, double[][] x2, LogicalOperation operation) {
        return delegate.logicalOperate(x1, x2, operation);
    }

    @Override
    public double[][] transpose(double[][] matrix) {
        return delegate.transpose(matrix);
    }

    @Override
    public double[][] transpose(double[] rowVector) {
        return delegate.transpose(rowVector);
    }

    @Override
    public double[][] outer(double[] a, double[] b) {
        return delegate.outer(a, b);
    }

    @Override
    public double[] sign(double[] array) {
        return delegate.sign(array);
    }

    @Override
    public double[][] sign(double[][] array) {
        return delegate.sign(array);
    }

    @Override
    public double[] diff(double[] array, int stride) {
        return delegate.diff(array, stride);
    }

    // ==================== GPU helper methods ====================

    private static double[] tryGpuBinaryOp(double[] x1, double[] x2, BinaryOperation op) {
        if (!GpuConfig.allowAttempts() || !GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            if (op == BinaryOperation.ADD) return GpuOptionalRuntime.tryAdd(x1, x2);
            if (op == BinaryOperation.SUBTRACT) return GpuOptionalRuntime.trySub(x1, x2);
            if (op == BinaryOperation.MULTIPLY) return GpuOptionalRuntime.tryMul(x1, x2);
            if (op == BinaryOperation.DIVIDE) return GpuOptionalRuntime.tryDiv(x1, x2);
        } catch (Throwable t) {
            // GPU op failed, fall through to delegate
        }
        return null;
    }

    private static double[] tryGpuReduce(double[] x, ReduceOperation op) {
        if (!GpuConfig.allowAttempts() || !GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            int gpuOp;
            if (op == ReduceOperation.SUM) gpuOp = GpuReduce.SUM;
            else if (op == ReduceOperation.MEAN) gpuOp = GpuReduce.MEAN;
            else if (op == ReduceOperation.MAX) gpuOp = GpuReduce.MAX;
            else if (op == ReduceOperation.MIN) gpuOp = GpuReduce.MIN;
            else if (op == ReduceOperation.PROD) gpuOp = GpuReduce.PROD;
            else if (op == ReduceOperation.VARIANCE) gpuOp = GpuReduce.VARIANCE;
            else if (op == ReduceOperation.STANDARD_DEVIATION) {
                // Compute std dev as sqrt(variance) on Java side
                double[] varResult = GpuReduce.tryReduce(GpuReduce.VARIANCE, x, 1, x.length);
                if (varResult != null && varResult.length > 0) {
                    return new double[] { Math.sqrt(varResult[0]) };
                }
                return null;
            }
            else return null;
            return GpuReduce.tryReduce(gpuOp, x, 1, x.length);
        } catch (Throwable t) {
            // GPU reduce failed, fall through to delegate
        }
        return null;
    }

    // ==================== HPC helper methods ====================

    private static double[] tryGpuActivation(double[] x, UniversalOperation op) {
        if (!GpuConfig.allowAttempts() || !GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return switch (op) {
                case RELU -> GpuOptionalRuntime.tryRelu(x);
                case SIGMOID -> GpuOptionalRuntime.trySigmoid(x);
                case TANH -> GpuOptionalRuntime.tryTanh(x);
                case GELU -> GpuOptionalRuntime.tryGelu(x);
                case EXP -> GpuOptionalRuntime.tryExp(x);
                case LOG -> GpuOptionalRuntime.tryLog(x);
                case ABS -> GpuOptionalRuntime.tryAbs(x);
                case SQRT -> GpuOptionalRuntime.trySqrt(x);
                case SIN -> GpuOptionalRuntime.trySin(x);
                case COS -> GpuOptionalRuntime.tryCos(x);
                default -> null;
            };
        } catch (Throwable t) {
            return null;
        }
    }

    private static double[] tryHpcActivation(double[] x, UniversalOperation op) {
        if (!HpcConfig.allowAttempts()) return null;
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
        return HpcOptionalRuntime.tryActivationF64(name, x);
    }

    private static double[] tryHpcBinaryOp(double[] a, double[] b, BinaryOperation op) {
        if (!HpcConfig.allowAttempts()) return null;
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
}
