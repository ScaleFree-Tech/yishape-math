package com.yishape.lab.math.compute.gpu;

import java.lang.reflect.Method;

import com.yishape.lab.util.YishapeLogger;

/**
 * Reflective bridge to the optional GPU module ({@code yishape-math-gpu}).
 * If the JAR is absent, all methods return null / false.
 * Analogous to {@link com.yishape.lab.math.compute.hpc.HpcOptionalRuntime}.
 */
public final class GpuOptionalRuntime {

    private static final YishapeLogger log = YishapeLogger.getLogger(GpuOptionalRuntime.class);
    private static final String GPU_CLASS = "com.yishape.lab.gpu.YishapeGpu";

    private static final Class<?> GPU;
    private static final Method M_IS_AVAILABLE;
    private static final Method M_DEVICE_NAME;
    private static final Method M_MAT_MUL;
    private static final Method M_FLAT_MAT_MUL;
    private static final Method M_ELEMENTWISE;
    private static final Method M_ADD;
    private static final Method M_SUB;
    private static final Method M_MUL;
    private static final Method M_DIV;
    private static final Method M_ACTIVATION;
    private static final Method M_RELU;
    private static final Method M_GELU;
    private static final Method M_SIGMOID;
    private static final Method M_TANH;
    private static final Method M_ELU;
    private static final Method M_LEAKY_RELU;
    private static final Method M_SILU;
    private static final Method M_SOFTPLUS;
    private static final Method M_SELU;
    private static final Method M_HARDTANH;
    private static final Method M_EXP;
    private static final Method M_LOG;
    private static final Method M_ABS;
    private static final Method M_SQRT;
    private static final Method M_SIN;
    private static final Method M_COS;
    private static final Method M_REDUCE;
    private static final Method M_LAYER_NORM;
    private static final Method M_LAYER_NORM_BACKWARD;
    private static final Method M_GROUP_NORM_BACKWARD;
    private static final Method M_BATCH_NORM;
    private static final Method M_SHUTDOWN;
    // Float32 (f32) GPU methods
    private static final Method M_FLOAT_MAT_MUL;
    private static final Method M_FLOAT_ADD;
    private static final Method M_FLOAT_SUB;
    private static final Method M_FLOAT_MUL;
    private static final Method M_FLOAT_DIV;
    private static final Method M_FLOAT_REDUCE;
    private static final Method M_FLOAT_FLAT_MAT_MUL;
    private static final Method M_FLOAT_ACTIVATION;
    private static final Method M_FLOAT_RELU;
    private static final Method M_FLOAT_GELU;
    private static final Method M_FLOAT_SIGMOID;
    private static final Method M_FLOAT_TANH;
    private static final Method M_FLOAT_ELU;
    private static final Method M_FLOAT_LEAKY_RELU;
    private static final Method M_FLOAT_SILU;
    private static final Method M_FLOAT_SOFTPLUS;
    private static final Method M_FLOAT_SELU;
    private static final Method M_FLOAT_HARDTANH;
    private static final Method M_FLOAT_EXP;
    private static final Method M_FLOAT_LOG;
    private static final Method M_FLOAT_ABS;
    private static final Method M_FLOAT_SQRT;
    private static final Method M_FLOAT_SIN;
    private static final Method M_FLOAT_COS;
    private static final Method M_SOFTMAX;
    private static final Method M_LOG_SOFTMAX;
    private static final Method M_NORMALIZE;
    private static final Method M_GATHER;
    private static final Method M_IM2COL;
    private static final Method M_FLAT_MAT_MUL_TRANSP;

    static {
        Class<?> c = null;
        try {
            c = Class.forName(GPU_CLASS);
        } catch (ReflectiveOperationException | LinkageError e) {
            // GPU module not on classpath
        }
        GPU = c;

        if (c != null) {
            M_IS_AVAILABLE = probe(c, "isAvailable");
            M_DEVICE_NAME = probe(c, "deviceName");
            M_MAT_MUL = probe(c, "matMul", double[][].class, double[][].class);
            M_FLAT_MAT_MUL = probe(c, "flatMatMul", double[].class, double[].class, int.class, int.class, int.class);
            M_ELEMENTWISE = probe(c, "elementwise", int.class, double[].class, double[].class, double.class);
            M_ADD = probe(c, "add", double[].class, double[].class);
            M_SUB = probe(c, "sub", double[].class, double[].class);
            M_MUL = probe(c, "mul", double[].class, double[].class);
            M_DIV = probe(c, "div", double[].class, double[].class);
            M_ACTIVATION = probe(c, "activation", int.class, double[].class);
            M_RELU = probe(c, "relu", double[].class);
            M_GELU = probe(c, "gelu", double[].class);
            M_SIGMOID = probe(c, "sigmoid", double[].class);
            M_TANH = probe(c, "tanh", double[].class);
            M_ELU = probe(c, "elu", double[].class);
            M_LEAKY_RELU = probe(c, "leakyRelu", double[].class);
            M_SILU = probe(c, "silu", double[].class);
            M_SOFTPLUS = probe(c, "softplus", double[].class);
            M_SELU = probe(c, "selu", double[].class);
            M_HARDTANH = probe(c, "hardtanh", double[].class);
            M_EXP = probe(c, "exp", double[].class);
            M_LOG = probe(c, "log", double[].class);
            M_ABS = probe(c, "abs", double[].class);
            M_SQRT = probe(c, "sqrt", double[].class);
            M_SIN = probe(c, "sin", double[].class);
            M_COS = probe(c, "cos", double[].class);
            M_REDUCE = probe(c, "reduce", int.class, double[].class, int.class, int.class);
            M_LAYER_NORM = probe(c, "layerNorm", double[].class, double[].class, double[].class, int.class, int.class, float.class);
            M_LAYER_NORM_BACKWARD = probe(c, "layerNormBackward", double[].class, double[].class, double[].class, int.class, int.class, float.class);
            M_GROUP_NORM_BACKWARD = probe(c, "groupNormBackward", double[].class, double[].class, double[].class, int.class, int.class, int.class, float.class);
            M_BATCH_NORM = probe(c, "batchNorm", double[].class, double[].class, double[].class, double[].class, double[].class, int.class, int.class, int.class);
            M_SHUTDOWN = probe(c, "shutdown");
            M_FLOAT_MAT_MUL = probe(c, "floatMatMul", float[][].class, float[][].class);
            M_FLOAT_ADD = probe(c, "floatAdd", float[].class, float[].class);
            M_FLOAT_SUB = probe(c, "floatSub", float[].class, float[].class);
            M_FLOAT_MUL = probe(c, "floatMul", float[].class, float[].class);
            M_FLOAT_DIV = probe(c, "floatDiv", float[].class, float[].class);
            M_FLOAT_REDUCE = probe(c, "floatReduce", int.class, float[].class, int.class, int.class);
            M_FLOAT_FLAT_MAT_MUL = probe(c, "floatFlatMatMul", float[].class, float[].class, int.class, int.class, int.class);
            M_FLOAT_ACTIVATION = probe(c, "floatActivation", int.class, float[].class);
            M_FLOAT_RELU = probe(c, "floatRelu", float[].class);
            M_FLOAT_GELU = probe(c, "floatGelu", float[].class);
            M_FLOAT_SIGMOID = probe(c, "floatSigmoid", float[].class);
            M_FLOAT_TANH = probe(c, "floatTanh", float[].class);
            M_FLOAT_ELU = probe(c, "floatElu", float[].class);
            M_FLOAT_LEAKY_RELU = probe(c, "floatLeakyRelu", float[].class);
            M_FLOAT_SILU = probe(c, "floatSilu", float[].class);
            M_FLOAT_SOFTPLUS = probe(c, "floatSoftplus", float[].class);
            M_FLOAT_SELU = probe(c, "floatSelu", float[].class);
            M_FLOAT_HARDTANH = probe(c, "floatHardtanh", float[].class);
            M_FLOAT_EXP = probe(c, "floatExp", float[].class);
            M_FLOAT_LOG = probe(c, "floatLog", float[].class);
            M_FLOAT_ABS = probe(c, "floatAbs", float[].class);
            M_FLOAT_SQRT = probe(c, "floatSqrt", float[].class);
            M_FLOAT_SIN = probe(c, "floatSin", float[].class);
            M_FLOAT_COS = probe(c, "floatCos", float[].class);
            M_SOFTMAX = probe(c, "softmax", double[].class, int.class, int.class);
            M_LOG_SOFTMAX = probe(c, "logSoftmax", double[].class, int.class, int.class);
            M_NORMALIZE = probe(c, "normalize", double[].class, int.class, int.class, double.class);
            M_GATHER = probe(c, "gather", double[].class, double[].class, int.class);
            M_IM2COL = probe(c, "im2col", double[].class, int.class, int.class, int.class, int.class, int.class, int.class, int.class, int.class, int.class);
            M_FLAT_MAT_MUL_TRANSP = probe(c, "flatMatMulTransp", double[].class, double[].class, int.class, int.class, int.class, int.class);
        } else {
            M_IS_AVAILABLE = null; M_DEVICE_NAME = null; M_MAT_MUL = null; M_FLAT_MAT_MUL = null;
            M_ELEMENTWISE = null; M_ADD = null; M_SUB = null; M_MUL = null; M_DIV = null;
            M_ACTIVATION = null; M_RELU = null; M_GELU = null; M_SIGMOID = null; M_TANH = null;
            M_ELU = null; M_LEAKY_RELU = null; M_SILU = null; M_SOFTPLUS = null; M_SELU = null;
            M_HARDTANH = null; M_EXP = null; M_LOG = null; M_ABS = null; M_SQRT = null;
            M_SIN = null; M_COS = null;
            M_REDUCE = null; M_LAYER_NORM = null; M_LAYER_NORM_BACKWARD = null; M_GROUP_NORM_BACKWARD = null; M_BATCH_NORM = null; M_SHUTDOWN = null;
            M_FLOAT_MAT_MUL = null; M_FLOAT_ADD = null; M_FLOAT_SUB = null;
            M_FLOAT_MUL = null; M_FLOAT_DIV = null; M_FLOAT_REDUCE = null;
            M_FLOAT_FLAT_MAT_MUL = null; M_FLOAT_ACTIVATION = null;
            M_FLOAT_RELU = null; M_FLOAT_GELU = null; M_FLOAT_SIGMOID = null; M_FLOAT_TANH = null;
            M_FLOAT_ELU = null; M_FLOAT_LEAKY_RELU = null; M_FLOAT_SILU = null; M_FLOAT_SOFTPLUS = null;
            M_FLOAT_SELU = null; M_FLOAT_HARDTANH = null; M_FLOAT_EXP = null; M_FLOAT_LOG = null;
            M_FLOAT_ABS = null; M_FLOAT_SQRT = null; M_FLOAT_SIN = null; M_FLOAT_COS = null;
            M_SOFTMAX = null; M_LOG_SOFTMAX = null; M_NORMALIZE = null; M_GATHER = null; M_IM2COL = null; M_FLAT_MAT_MUL_TRANSP = null;
        }
    }

    private GpuOptionalRuntime() {}

    /**
     * 一次检测结果缓存，整个 JVM 运行周期内不重复检测。
     */
    private static volatile Boolean gpuAvailable = null;

    public static boolean isExtensionPresent() { return GPU != null; }

    /**
     * 检测 GPU 是否可用（首次调用检测并缓存，后续直接返回缓存结果）。
     */
    public static boolean isGpuAvailable() {
        if (gpuAvailable != null) {
            return gpuAvailable;
        }
        if (M_IS_AVAILABLE == null) {
            gpuAvailable = false;
            return false;
        }
        try {
            gpuAvailable = (Boolean) M_IS_AVAILABLE.invoke(null);
        } catch (Exception e) {
            gpuAvailable = false;
        }
        if (gpuAvailable) {
            String deviceName = tryDeviceName();
            log.info("GPU detected and available: {}", deviceName != null ? deviceName : "unknown device");
        }
        return gpuAvailable;
    }

    public static String tryDeviceName() {
        if (M_DEVICE_NAME == null) return null;
        try { return (String) M_DEVICE_NAME.invoke(null); }
        catch (Exception e) { return null; }
    }

    public static double[][] tryMatMul(double[][] a, double[][] b) {
        if (M_MAT_MUL == null) return null;
        try {
            Object out = M_MAT_MUL.invoke(null, a, b);
            return (out instanceof double[][]) ? (double[][]) out : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logError("tryMatMul", e);
            return null;
        }
    }

    /**
     * Flat matrix multiply: C[m×n] = A[m×k] @ B[k×n], all flat row-major.
     * Writes result into cOut and returns true on success.
     */
    public static boolean tryFlatMatMul(double[] a, double[] b, double[] cOut, int m, int k, int n) {
        if (M_FLAT_MAT_MUL == null) return false;
        try {
            Object out = M_FLAT_MAT_MUL.invoke(null, a, b, m, k, n);
            if (out instanceof double[] result) {
                System.arraycopy(result, 0, cOut, 0, Math.min(result.length, cOut.length));
                return true;
            }
            return false;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logError("tryFlatMatMul", e);
            return false;
        }
    }

    public static double[] tryAdd(double[] a, double[] b) {
        return tryInvoke(M_ADD, "tryAdd", a, b);
    }

    public static double[] trySub(double[] a, double[] b) {
        return tryInvoke(M_SUB, "trySub", a, b);
    }

    public static double[] tryMul(double[] a, double[] b) {
        return tryInvoke(M_MUL, "tryMul", a, b);
    }

    public static double[] tryDiv(double[] a, double[] b) {
        return tryInvoke(M_DIV, "tryDiv", a, b);
    }

    public static double[] tryRelu(double[] input) {
        return tryInvoke(M_RELU, "tryRelu", input);
    }

    public static double[] tryGelu(double[] input) {
        return tryInvoke(M_GELU, "tryGelu", input);
    }

    public static double[] trySigmoid(double[] input) {
        return tryInvoke(M_SIGMOID, "trySigmoid", input);
    }

    public static double[] tryTanh(double[] input) {
        return tryInvoke(M_TANH, "tryTanh", input);
    }

    public static double[] tryElu(double[] input) {
        return tryInvoke(M_ELU, "tryElu", input);
    }

    public static double[] tryLeakyRelu(double[] input) {
        return tryInvoke(M_LEAKY_RELU, "tryLeakyRelu", input);
    }

    public static double[] trySilu(double[] input) {
        return tryInvoke(M_SILU, "trySilu", input);
    }

    public static double[] trySoftplus(double[] input) {
        return tryInvoke(M_SOFTPLUS, "trySoftplus", input);
    }

    public static double[] trySelu(double[] input) {
        return tryInvoke(M_SELU, "trySelu", input);
    }

    public static double[] tryHardtanh(double[] input) {
        return tryInvoke(M_HARDTANH, "tryHardtanh", input);
    }

    public static double[] tryExp(double[] input) {
        return tryInvoke(M_EXP, "tryExp", input);
    }

    public static double[] tryLog(double[] input) {
        return tryInvoke(M_LOG, "tryLog", input);
    }

    public static double[] tryAbs(double[] input) {
        return tryInvoke(M_ABS, "tryAbs", input);
    }

    public static double[] trySqrt(double[] input) {
        return tryInvoke(M_SQRT, "trySqrt", input);
    }

    public static double[] trySin(double[] input) {
        return tryInvoke(M_SIN, "trySin", input);
    }

    public static double[] tryCos(double[] input) {
        return tryInvoke(M_COS, "tryCos", input);
    }

    public static double[] trySoftmax(double[] input, int rows, int cols) {
        if (M_SOFTMAX == null) return null;
        try {
            Object out = M_SOFTMAX.invoke(null, input, rows, cols);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("trySoftmax", e);
            return null;
        }
    }

    public static double[] tryNormalize(double[] input, int rows, int cols, double p) {
        if (M_NORMALIZE == null) return null;
        try {
            Object out = M_NORMALIZE.invoke(null, input, rows, cols, p);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("tryNormalize", e);
            return null;
        }
    }

    public static double[] tryLogSoftmax(double[] input, int rows, int cols) {
        if (M_LOG_SOFTMAX == null) return null;
        try {
            Object out = M_LOG_SOFTMAX.invoke(null, input, rows, cols);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("tryLogSoftmax", e);
            return null;
        }
    }

    public static double[] tryGather(double[] weight, double[] indices, int embeddingDim) {
        if (M_GATHER == null) return null;
        try {
            Object out = M_GATHER.invoke(null, weight, indices, embeddingDim);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("tryGather", e);
            return null;
        }
    }

    public static double[] tryFlatMatMulTransp(double[] a, double[] b, int m, int k, int n, int transp) {
        if (M_FLAT_MAT_MUL_TRANSP == null) return null;
        try {
            Object out = M_FLAT_MAT_MUL_TRANSP.invoke(null, a, b, m, k, n, transp);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("tryFlatMatMulTransp", e);
            return null;
        }
    }

    public static double[] tryIm2col(double[] input, int C, int H, int W,
                                      int outH, int outW, int kH, int kW,
                                      int stride, int padding) {
        if (M_IM2COL == null) return null;
        try {
            Object out = M_IM2COL.invoke(null, input, C, H, W, outH, outW, kH, kW, stride, padding);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("tryIm2col", e);
            return null;
        }
    }

    public static double[] tryReduce(int op, double[] input, int outer, int inner) {
        if (M_REDUCE == null) return null;
        try {
            Object out = M_REDUCE.invoke(null, op, input, outer, inner);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("tryReduce", e);
            return null;
        }
    }

    public static double[] tryLayerNorm(double[] x, double[] gamma, double[] beta, int outer, int normDim, float eps) {
        if (M_LAYER_NORM == null) return null;
        try {
            Object out = M_LAYER_NORM.invoke(null, x, gamma, beta, outer, normDim, eps);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("tryLayerNorm", e);
            return null;
        }
    }

    /** Returns double[][]{dx, dgamma, dbeta} or null on failure. */
    public static double[][] tryLayerNormBackward(double[] x, double[] gamma, double[] grad, int rows, int dim, float eps) {
        if (M_LAYER_NORM_BACKWARD == null) return null;
        try {
            Object out = M_LAYER_NORM_BACKWARD.invoke(null, x, gamma, grad, rows, dim, eps);
            return (out instanceof double[][]) ? (double[][]) out : null;
        } catch (Exception e) {
            logError("tryLayerNormBackward", e);
            return null;
        }
    }

    /** Returns double[][]{dx, dgamma, dbeta} or null on failure. */
    public static double[][] tryGroupNormBackward(double[] x, double[] gamma, double[] grad,
                                                   int numGroups, int gch, int hw, float eps) {
        if (M_GROUP_NORM_BACKWARD == null) return null;
        try {
            Object out = M_GROUP_NORM_BACKWARD.invoke(null, x, gamma, grad, numGroups, gch, hw, eps);
            return (out instanceof double[][]) ? (double[][]) out : null;
        } catch (Exception e) {
            logError("tryGroupNormBackward", e);
            return null;
        }
    }

    public static double[] tryBatchNorm(double[] x, double[] mean, double[] var_,
                                         double[] gamma, double[] beta, int n, int c, int hw) {
        if (M_BATCH_NORM == null) return null;
        try {
            Object out = M_BATCH_NORM.invoke(null, x, mean, var_, gamma, beta, n, c, hw);
            return (out instanceof double[]) ? (double[]) out : null;
        } catch (Exception e) {
            logError("tryBatchNorm", e);
            return null;
        }
    }

    public static void tryShutdown() {
        if (M_SHUTDOWN == null) return;
        try { M_SHUTDOWN.invoke(null); } catch (Exception e) {
            log.debug("GPU shutdown failed", e);
        }
    }

    public static String tryExecuteGraph(String json) {
        if (GPU == null) return null;
        try {
            // Probe for executeGraph method (may not exist in older GPU module versions)
            Method m = GPU.getMethod("executeGraph", String.class);
            Object out = m.invoke(null, json);
            return (out instanceof String s) ? s : null;
        } catch (NoSuchMethodException e) {
            // GPU module does not support executeGraph yet
            return null;
        } catch (ReflectiveOperationException | LinkageError e) {
            logError("tryExecuteGraph", e);
            return null;
        }
    }

    // ==================== Float32 (f32) methods ====================

    public static float[][] tryFloatMatMul(float[][] a, float[][] b) {
        if (M_FLOAT_MAT_MUL == null) return null;
        try {
            Object out = M_FLOAT_MAT_MUL.invoke(null, a, b);
            return (out instanceof float[][]) ? (float[][]) out : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logError("tryFloatMatMul", e);
            return null;
        }
    }

    public static float[] tryFloatAdd(float[] a, float[] b) {
        return tryFloatInvoke(M_FLOAT_ADD, "tryFloatAdd", a, b);
    }

    public static float[] tryFloatSub(float[] a, float[] b) {
        return tryFloatInvoke(M_FLOAT_SUB, "tryFloatSub", a, b);
    }

    public static float[] tryFloatMul(float[] a, float[] b) {
        return tryFloatInvoke(M_FLOAT_MUL, "tryFloatMul", a, b);
    }

    public static float[] tryFloatDiv(float[] a, float[] b) {
        return tryFloatInvoke(M_FLOAT_DIV, "tryFloatDiv", a, b);
    }

    public static float[] tryFloatReduce(int op, float[] input, int outer, int inner) {
        if (M_FLOAT_REDUCE == null) return null;
        try {
            Object out = M_FLOAT_REDUCE.invoke(null, op, input, outer, inner);
            return (out instanceof float[]) ? (float[]) out : null;
        } catch (Exception e) {
            logError("tryFloatReduce", e);
            return null;
        }
    }

    /**
     * Flat float matrix multiply: C[m×n] = A[m×k] @ B[k×n], all flat row-major f32.
     * Returns result array on success, null on failure.
     */
    public static float[] tryFloatFlatMatMul(float[] a, float[] b, int m, int k, int n) {
        if (M_FLOAT_FLAT_MAT_MUL == null) return null;
        try {
            Object out = M_FLOAT_FLAT_MAT_MUL.invoke(null, a, b, m, k, n);
            return (out instanceof float[]) ? (float[]) out : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logError("tryFloatFlatMatMul", e);
            return null;
        }
    }

    public static float[] tryFloatActivation(int op, float[] input) {
        if (M_FLOAT_ACTIVATION == null) return null;
        try {
            Object out = M_FLOAT_ACTIVATION.invoke(null, op, input);
            return (out instanceof float[]) ? (float[]) out : null;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logError("tryFloatActivation", e);
            return null;
        }
    }

    public static float[] tryFloatRelu(float[] input) {
        return tryFloatInvoke(M_FLOAT_RELU, "tryFloatRelu", input);
    }

    public static float[] tryFloatGelu(float[] input) {
        return tryFloatInvoke(M_FLOAT_GELU, "tryFloatGelu", input);
    }

    public static float[] tryFloatSigmoid(float[] input) {
        return tryFloatInvoke(M_FLOAT_SIGMOID, "tryFloatSigmoid", input);
    }

    public static float[] tryFloatTanh(float[] input) {
        return tryFloatInvoke(M_FLOAT_TANH, "tryFloatTanh", input);
    }

    public static float[] tryFloatElu(float[] input) {
        return tryFloatInvoke(M_FLOAT_ELU, "tryFloatElu", input);
    }

    public static float[] tryFloatLeakyRelu(float[] input) {
        return tryFloatInvoke(M_FLOAT_LEAKY_RELU, "tryFloatLeakyRelu", input);
    }

    public static float[] tryFloatSilu(float[] input) {
        return tryFloatInvoke(M_FLOAT_SILU, "tryFloatSilu", input);
    }

    public static float[] tryFloatSoftplus(float[] input) {
        return tryFloatInvoke(M_FLOAT_SOFTPLUS, "tryFloatSoftplus", input);
    }

    public static float[] tryFloatSelu(float[] input) {
        return tryFloatInvoke(M_FLOAT_SELU, "tryFloatSelu", input);
    }

    public static float[] tryFloatHardtanh(float[] input) {
        return tryFloatInvoke(M_FLOAT_HARDTANH, "tryFloatHardtanh", input);
    }

    public static float[] tryFloatExp(float[] input) {
        return tryFloatInvoke(M_FLOAT_EXP, "tryFloatExp", input);
    }

    public static float[] tryFloatLog(float[] input) {
        return tryFloatInvoke(M_FLOAT_LOG, "tryFloatLog", input);
    }

    public static float[] tryFloatAbs(float[] input) {
        return tryFloatInvoke(M_FLOAT_ABS, "tryFloatAbs", input);
    }

    public static float[] tryFloatSqrt(float[] input) {
        return tryFloatInvoke(M_FLOAT_SQRT, "tryFloatSqrt", input);
    }

    public static float[] tryFloatSin(float[] input) {
        return tryFloatInvoke(M_FLOAT_SIN, "tryFloatSin", input);
    }

    public static float[] tryFloatCos(float[] input) {
        return tryFloatInvoke(M_FLOAT_COS, "tryFloatCos", input);
    }

    @SuppressWarnings("unchecked")
    private static <T> T tryFloatInvoke(Method m, String name, Object... args) {
        if (m == null) return null;
        try {
            Object out = m.invoke(null, args);
            return (T) out;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logError(name, e);
            return null;
        }
    }

    // ==================== Helpers ====================

    @SuppressWarnings("unchecked")
    private static <T> T tryInvoke(Method m, String name, Object... args) {
        if (m == null) return null;
        try {
            Object out = m.invoke(null, args);
            return (T) out;
        } catch (ReflectiveOperationException | LinkageError | ClassCastException e) {
            logError(name, e);
            return null;
        }
    }

    private static Method probe(Class<?> c, String name, Class<?>... paramTypes) {
        try {
            return c.getMethod(name, paramTypes);
        } catch (NoSuchMethodException e) {
            return null;
        }
    }

    private static void logError(String op, Throwable e) {
        log.debug("GPU op failed: {}", op, e);
    }
}
