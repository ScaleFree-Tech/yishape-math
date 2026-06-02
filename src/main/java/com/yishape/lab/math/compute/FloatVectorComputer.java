package com.yishape.lab.math.compute;

import com.yishape.lab.util.YishapeLogger;

import java.io.Serializable;

/**
 * 统一向量操作计算器，根据情况选择合适的计算器
 *
 * @author lteb2
 */
public class FloatVectorComputer implements IFloatVectorComputer,Serializable {

    private static final YishapeLogger log = YishapeLogger.getLogger(FloatVectorComputer.class);


    private static volatile IFloatVectorComputer gpu = null;
    private static volatile IFloatVectorComputer simd = null;
    private static volatile IFloatVectorComputer sisd = null;

    private static volatile Boolean ifSIMDSupported = null;
    private static volatile Boolean ifGPUSupported = null;

    /** Cached base computer (SIMD or SISD) to avoid repeated dispatch after first resolution. */
    private static volatile IFloatVectorComputer resolvedBase = null;

    static {
        // 延迟检测支持，只在需要时才检测
        // ifSIMDSupported = ComputerConfig.checkIfSIMDSupported();
        // ifGPUSupported = ComputerConfig.checkIfGPUSupported();
    }

    /**
     * 检查是否支持SIMD，延迟检测
     *
     * @return
     */
    private static boolean checkIfSIMDSupported() {
        if (ifSIMDSupported == null) {
            // 只在第一次调用时检测SIMD支持
            ifSIMDSupported = ComputerConfig.checkIfSIMDSupported();
        }
        return ifSIMDSupported;
    }

    /**
     * 检查是否支持GPU，延迟检测
     *
     * @return
     */
    private static boolean checkIfGPUSupported() {
        if (ifGPUSupported == null) {
            // 只在第一次调用时检测GPU支持
            ifGPUSupported = ComputerConfig.checkIfGPUSupported();
        }
        return ifGPUSupported;
    }

    /**
     * 基于数据的规模和配置选择合适的计算器
     *
     * <p>Dispatch chain: GPU → SIMD → SISD.
     * GPU layer wraps the chosen SIMD/SISD computer as delegate;
     * when GPU is unavailable or disabled, all GPU attempts short-circuit
     * via {@link com.yishape.lab.math.compute.gpu.GpuConfig#allowAttempts()}.
     *
     * @param size data size for threshold-based dispatch
     * @return appropriate computer instance
     */
    private IFloatVectorComputer fetchComputer(long size) {
        // Fast path: return cached base computer for non-GPU operations
        IFloatVectorComputer base = resolvedBase;
        if (base != null && !(ComputerConfig.USE_GPU && checkIfGPUSupported()
                && size > ComputerConfig.GPU_VECTOR_THRESHOLD)) {
            return base;
        }

        // Slow path: resolve base computer
        IFloatVectorComputer baseComputer = null;
        if (ComputerConfig.USE_SIMD && checkIfSIMDSupported()) {
            if (simd == null) {
                try {
                    Class<?> simdClass = Class.forName("com.yishape.lab.math.compute.SIMDFloatComputer");
                    simd = (IFloatVectorComputer) simdClass.getDeclaredConstructor().newInstance();
                    log.info("SIMD float computer initialized");
                } catch (Throwable t) {
                    log.debug("SIMD unavailable, using SISD: {}", t.toString());
                    ifSIMDSupported = false;
                    simd = null;
                }
            }
            if (simd != null) {
                baseComputer = simd;
            } else {
                if (sisd == null) {
                    sisd = new SISDFloatComputer();
                }
                baseComputer = sisd;
            }
        } else {
            if (sisd == null) {
                sisd = new SISDFloatComputer();
            }
            baseComputer = sisd;
        }
        resolvedBase = baseComputer;

        // Step 2: wrap with GPU if available and enabled
        if (ComputerConfig.USE_GPU && checkIfGPUSupported() && size > ComputerConfig.GPU_VECTOR_THRESHOLD) {
            if (gpu == null) {
                try {
                    Class<?> gpuClass = Class.forName("com.yishape.lab.math.compute.GPUFloatComputer");
                    gpu = (IFloatVectorComputer) gpuClass
                            .getDeclaredConstructor(IFloatVectorComputer.class)
                            .newInstance(baseComputer);
                    log.info("GPU float computer initialized, wrapping {}",
                            simd != null ? "SIMD" : "SISD");
                } catch (Throwable t) {
                    log.debug("GPU float computer unavailable: {}", t.toString());
                    ifGPUSupported = false;
                    gpu = null;
                }
            }
            if (gpu != null) return gpu;
        }

        return baseComputer;
    }

    @Override
    public float[] binaryOperate(float[] x1, float[] x2, BinaryOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[] binaryOperate(float[] x1, float x2, BinaryOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[][] binaryOperate(float[][] x1, float[][] x2, BinaryOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[][] binaryOperate(float[][] x1, float x2, BinaryOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[] universalOperate(float[] x, UniversalOperation operation, float additionalParam) {
        var computer = this.fetchComputer(x.length);
        return computer.universalOperate(x, operation, additionalParam);
    }

    @Override
    public float[][] universalOperate(float[][] x, UniversalOperation operation, float additionalParam) {
        var computer = this.fetchComputer((long) x.length * x[0].length);
        return computer.universalOperate(x, operation, additionalParam);
    }

    @Override
    public float reduceOperate(float[] x, ReduceOperation operation) {
        var computer = this.fetchComputer(x.length);
        return computer.reduceOperate(x, operation);
    }

    @Override
    public float reduceOperate(float[][] x, ReduceOperation operation) {
        var computer = this.fetchComputer((long) x.length * x[0].length);
        return computer.reduceOperate(x, operation);
    }

    @Override
    public float binaryReduceOperate(float[] x1, float[] x2, BinaryReduceOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public float binaryReduceOperate(float[][] x1, float[][] x2, BinaryReduceOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public float[] elementWiseMin(float[] x1, float[] x2) {
        var computer = this.fetchComputer(x1.length);
        return computer.elementWiseMin(x1, x2);
    }

    @Override
    public float[][] elementWiseMin(float[][] x1, float[][] x2) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.elementWiseMin(x1, x2);
    }

    @Override
    public float[] elementWiseMax(float[] x1, float[] x2) {
        var computer = this.fetchComputer(x1.length);
        return computer.elementWiseMax(x1, x2);
    }

    @Override
    public float[][] elementWiseMax(float[][] x1, float[][] x2) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.elementWiseMax(x1, x2);
    }

    @Override
    public float[] negate(float[] x) {
        var computer = this.fetchComputer(x.length);
        return computer.negate(x);
    }

    @Override
    public float[][] negate(float[][] x) {
        var computer = this.fetchComputer((long) x.length * x[0].length);
        return computer.negate(x);
    }

    @Override
    public boolean[] logicalCompare(float[] x1, float[] x2, LogicalCompare operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[] logicalOperate(float[] x1, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.logicalOperate(x1, operation);
    }

    @Override
    public boolean[] logicalOperate(float[] x1, float[] x2, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.logicalOperate(x1, x2, operation);
    }

    @Override
    public float[][] transpose(float[][] matrix) {
        var computer = this.fetchComputer((long) matrix.length * matrix[0].length);
        return computer.transpose(matrix);
    }

    @Override
    public float[][] transpose(float[] rowVector) {
        var computer = this.fetchComputer(rowVector.length);
        return computer.transpose(rowVector);
    }

    @Override
    public float[][] mmul(float[][] a, float[][] b) {
        var computer = this.fetchComputer((long) a.length * a[0].length * b[0].length);
        return computer.mmul(a, b);
    }

    @Override
    public float[][] outer(float[] a, float[] b) {
        var computer = this.fetchComputer(a.length * b.length);
        return computer.outer(a, b);
    }

    @Override
    public float[] sign(float[] array) {
        var computer = this.fetchComputer(array.length);
        return computer.sign(array);
    }

    @Override
    public float[][] sign(float[][] array) {
        var computer = this.fetchComputer((long) array.length * array[0].length);
        return computer.sign(array);
    }

    @Override
    public float[] diff(float[] array, int stride) {
        var computer = this.fetchComputer(array.length);
        return computer.diff(array, stride);
    }

    @Override
    public boolean[][] logicalCompare(float[][] x1, float[][] x2, LogicalCompare operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, LogicalOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.logicalOperate(x1, operation);
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, float[][] x2, LogicalOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.logicalOperate(x1, x2, operation);
    }

}
