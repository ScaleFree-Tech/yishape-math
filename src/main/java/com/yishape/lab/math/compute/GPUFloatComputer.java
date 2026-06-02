package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;
import com.yishape.lab.math.compute.gpu.GpuReduce;
import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;

/**
 * GPU-accelerated float vector computer. Calls the f32 GPU API
 * ({@link GpuOptionalRuntime#tryFloatAdd( float[], float[] )}, etc.) directly,
 * avoiding float↔double conversion overhead. Falls back to HPC via
 * {@link FloatFlatGemm} for matmul, then to an underlying SIMD/SISD delegate.
 *
 * <p>Dispatch chain: GPU → HPC (via {@link FloatFlatGemm}) → delegate (SIMD → SISD).
 *
 * @author lteb2
 */
final class GPUFloatComputer implements IFloatVectorComputer {

    /** Check if GPU is available, used by {@link ComputerConfig#checkIfGPUSupported()}. */
    public static boolean isGPUAvailable() {
        return GpuOptionalRuntime.isGpuAvailable();
    }

    /** Underlying SIMD or SISD computer used as CPU fallback. */
    private final IFloatVectorComputer delegate;

    GPUFloatComputer(IFloatVectorComputer delegate) {
        this.delegate = delegate;
    }

    // ==================== GPU-accelerated operations ====================

    @Override
    public float[][] mmul(float[][] a, float[][] b) {
        // 1. Try GPU f32 natively
        float[][] gpuResult = tryGpuMmul(a, b);
        if (gpuResult != null) return gpuResult;
        // 2. Try HPC via FloatFlatGemm double bridge
        float[][] hpcResult = tryHpcMmul(a, b);
        if (hpcResult != null) return hpcResult;
        // 3. Fall back to delegated SIMD/SISD
        return delegate.mmul(a, b);
    }

    @Override
    public float[] binaryOperate(float[] x1, float[] x2, BinaryOperation operation) {
        if (x1.length >= GpuConfig.elementwiseMinElements()) {
            float[] gpuResult = tryGpuBinaryOp(x1, x2, operation);
            if (gpuResult != null) return gpuResult;
        }
        if (x1.length >= HpcConfig.activationMinElements()) {
            float[] hpcResult = tryHpcBinaryOp(x1, x2, operation);
            if (hpcResult != null) return hpcResult;
        }
        return delegate.binaryOperate(x1, x2, operation);
    }

    @Override
    public float reduceOperate(float[] x, ReduceOperation operation) {
        if (x.length >= GpuConfig.reduceMinElements()
                && (operation == ReduceOperation.SUM
                    || operation == ReduceOperation.MEAN
                    || operation == ReduceOperation.MAX
                    || operation == ReduceOperation.MIN
                    || operation == ReduceOperation.PROD
                    || operation == ReduceOperation.VARIANCE
                    || operation == ReduceOperation.STANDARD_DEVIATION)) {
            float[] gpuResult = tryGpuReduce(x, operation);
            if (gpuResult != null && gpuResult.length > 0) {
                return gpuResult[0];
            }
        }
        return delegate.reduceOperate(x, operation);
    }

    // ==================== Delegated operations (no GPU path) ====================

    @Override
    public float[] binaryOperate(float[] x1, float x2, BinaryOperation operation) {
        return delegate.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[][] binaryOperate(float[][] x1, float[][] x2, BinaryOperation operation) {
        return delegate.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[][] binaryOperate(float[][] x1, float x2, BinaryOperation operation) {
        return delegate.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[] universalOperate(float[] x, UniversalOperation operation, float additionalParam) {
        if (x.length >= GpuConfig.elementwiseMinElements()) {
            float[] gpuResult = tryGpuActivation(x, operation);
            if (gpuResult != null) return gpuResult;
        }
        if (x.length >= HpcConfig.activationMinElements()) {
            float[] hpcResult = tryHpcActivation(x, operation);
            if (hpcResult != null) return hpcResult;
        }
        return delegate.universalOperate(x, operation, additionalParam);
    }

    @Override
    public float[][] universalOperate(float[][] x, UniversalOperation operation, float additionalParam) {
        // For 2D arrays, delegate per-row to the 1D GPU/HPC path
        if (x.length > 0 && x[0].length >= GpuConfig.elementwiseMinElements()) {
            boolean anyGpu = false;
            float[][] result = new float[x.length][];
            for (int i = 0; i < x.length; i++) {
                float[] gpuRow = tryGpuActivation(x[i], operation);
                if (gpuRow != null) {
                    result[i] = gpuRow;
                    anyGpu = true;
                } else {
                    break; // If any row fails, fall back entirely to delegate
                }
            }
            if (anyGpu) return result;
        }
        if (x.length > 0 && x[0].length >= HpcConfig.activationMinElements()) {
            boolean anyHpc = false;
            float[][] result = new float[x.length][];
            for (int i = 0; i < x.length; i++) {
                float[] hpcRow = tryHpcActivation(x[i], operation);
                if (hpcRow != null) {
                    result[i] = hpcRow;
                    anyHpc = true;
                } else {
                    break;
                }
            }
            if (anyHpc) return result;
        }
        return delegate.universalOperate(x, operation, additionalParam);
    }

    @Override
    public float reduceOperate(float[][] x, ReduceOperation operation) {
        return delegate.reduceOperate(x, operation);
    }

    @Override
    public float binaryReduceOperate(float[] x1, float[] x2, BinaryReduceOperation operation) {
        return delegate.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public float binaryReduceOperate(float[][] x1, float[][] x2, BinaryReduceOperation operation) {
        return delegate.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public float[] elementWiseMin(float[] x1, float[] x2) {
        return delegate.elementWiseMin(x1, x2);
    }

    @Override
    public float[][] elementWiseMin(float[][] x1, float[][] x2) {
        return delegate.elementWiseMin(x1, x2);
    }

    @Override
    public float[] elementWiseMax(float[] x1, float[] x2) {
        return delegate.elementWiseMax(x1, x2);
    }

    @Override
    public float[][] elementWiseMax(float[][] x1, float[][] x2) {
        return delegate.elementWiseMax(x1, x2);
    }

    @Override
    public float[] negate(float[] x) {
        return delegate.negate(x);
    }

    @Override
    public float[][] negate(float[][] x) {
        return delegate.negate(x);
    }

    @Override
    public boolean[] logicalCompare(float[] x1, float[] x2, LogicalCompare operation) {
        return delegate.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[] logicalOperate(float[] x1, LogicalOperation operation) {
        return delegate.logicalOperate(x1, operation);
    }

    @Override
    public boolean[] logicalOperate(float[] x1, float[] x2, LogicalOperation operation) {
        return delegate.logicalOperate(x1, x2, operation);
    }

    @Override
    public boolean[][] logicalCompare(float[][] x1, float[][] x2, LogicalCompare operation) {
        return delegate.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, LogicalOperation operation) {
        return delegate.logicalOperate(x1, operation);
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, float[][] x2, LogicalOperation operation) {
        return delegate.logicalOperate(x1, x2, operation);
    }

    @Override
    public float[][] transpose(float[][] matrix) {
        return delegate.transpose(matrix);
    }

    @Override
    public float[][] transpose(float[] rowVector) {
        return delegate.transpose(rowVector);
    }

    @Override
    public float[][] outer(float[] a, float[] b) {
        return delegate.outer(a, b);
    }

    @Override
    public float[] sign(float[] array) {
        return delegate.sign(array);
    }

    @Override
    public float[][] sign(float[][] array) {
        return delegate.sign(array);
    }

    @Override
    public float[] diff(float[] array, int stride) {
        return delegate.diff(array, stride);
    }

    // ==================== HPC helper methods ====================

    /**
     * Try float matmul via HPC double bridge (FloatFlatGemm → HpcIm2col.tryFlatDgemm).
     * Returns null if HPC is unavailable or below threshold.
     */
    private static float[][] tryHpcMmul(float[][] a, float[][] b) {
        int m = a.length, k = a[0].length, n = b[0].length;
        if (k != b.length) return null;
        long flops = (long) m * n * k;
        if (flops < HpcConfig.gemmMinFlops()) return null;
        if (!HpcConfig.allowAttempts()) return null;

        float[] aFlat = new float[m * k];
        float[] bFlat = new float[k * n];
        for (int i = 0; i < m; i++) System.arraycopy(a[i], 0, aFlat, i * k, k);
        for (int i = 0; i < k; i++) System.arraycopy(b[i], 0, bFlat, i * n, n);

        float[] cFlat = FloatFlatGemm.flatMmul(aFlat, m, k, bFlat, n);

        float[][] c = new float[m][n];
        for (int i = 0; i < m; i++) System.arraycopy(cFlat, i * n, c[i], 0, n);
        return c;
    }

    // ==================== GPU helper methods (f32 zero-copy) ====================

    private static float[][] tryGpuMmul(float[][] a, float[][] b) {
        if (!GpuConfig.allowAttempts() || !GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return GpuOptionalRuntime.tryFloatMatMul(a, b);
        } catch (Throwable t) {
            return null;
        }
    }

    private static float[] tryGpuBinaryOp(float[] x1, float[] x2, BinaryOperation op) {
        if (!GpuConfig.allowAttempts() || !GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            if (op == BinaryOperation.ADD) return GpuOptionalRuntime.tryFloatAdd(x1, x2);
            if (op == BinaryOperation.SUBTRACT) return GpuOptionalRuntime.tryFloatSub(x1, x2);
            if (op == BinaryOperation.MULTIPLY) return GpuOptionalRuntime.tryFloatMul(x1, x2);
            if (op == BinaryOperation.DIVIDE) return GpuOptionalRuntime.tryFloatDiv(x1, x2);
        } catch (Throwable t) {
            // GPU op failed, fall through to delegate
        }
        return null;
    }

    private static float[] tryGpuReduce(float[] x, ReduceOperation op) {
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
                float[] varResult = GpuOptionalRuntime.tryFloatReduce(GpuReduce.VARIANCE, x, 1, x.length);
                if (varResult != null && varResult.length > 0) {
                    return new float[] { (float) Math.sqrt(varResult[0]) };
                }
                return null;
            }
            else return null;
            return GpuOptionalRuntime.tryFloatReduce(gpuOp, x, 1, x.length);
        } catch (Throwable t) {
            // GPU reduce failed, fall through to delegate
        }
        return null;
    }

    private static float[] tryGpuActivation(float[] x, UniversalOperation op) {
        if (!GpuConfig.allowAttempts() || !GpuOptionalRuntime.isGpuAvailable()) return null;
        try {
            return switch (op) {
                case RELU -> GpuOptionalRuntime.tryFloatRelu(x);
                case SIGMOID -> GpuOptionalRuntime.tryFloatSigmoid(x);
                case TANH -> GpuOptionalRuntime.tryFloatTanh(x);
                case GELU -> GpuOptionalRuntime.tryFloatGelu(x);
                case EXP -> GpuOptionalRuntime.tryFloatExp(x);
                case LOG -> GpuOptionalRuntime.tryFloatLog(x);
                case ABS -> GpuOptionalRuntime.tryFloatAbs(x);
                case SQRT -> GpuOptionalRuntime.tryFloatSqrt(x);
                case SIN -> GpuOptionalRuntime.tryFloatSin(x);
                case COS -> GpuOptionalRuntime.tryFloatCos(x);
                default -> null;
            };
        } catch (Throwable t) {
            return null;
        }
    }

    // ==================== HPC activation/binary helpers ====================

    private static float[] tryHpcActivation(float[] x, UniversalOperation op) {
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
        return HpcOptionalRuntime.tryFloatActivation(name, x);
    }

    private static float[] tryHpcBinaryOp(float[] a, float[] b, BinaryOperation op) {
        if (!HpcConfig.allowAttempts()) return null;
        String name = switch (op) {
            case ADD -> "addF64";
            case SUBTRACT -> "subF64";
            case MULTIPLY -> "mulF64";
            case DIVIDE -> "divF64";
            default -> null;
        };
        if (name == null) return null;
        return HpcOptionalRuntime.tryFloatElementwise(name, a, b);
    }
}
