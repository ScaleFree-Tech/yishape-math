package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.BinaryReduceOperation;
import com.yishape.lab.math.compute.ops.LogicalCompare;
import com.yishape.lab.math.compute.ops.LogicalOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.util.YishapeLogger;

import java.io.Serializable;

/**
 * 统一向量操作计算器，根据情况选择合适的计算器
 *
 * @author lteb2
 */
public class DoubleVectorComputer implements IDoubleVectorComputer,Serializable {

    private static final YishapeLogger log = YishapeLogger.getLogger(DoubleVectorComputer.class);


    private static volatile IDoubleVectorComputer gpu = null;
    private static volatile IDoubleVectorComputer simd = null;
    private static volatile IDoubleVectorComputer sisd = null;

    private static volatile Boolean ifSIMDSupported = null;
    private static volatile Boolean ifGPUSupported = null;

    /** Cached base computer (SIMD or SISD) to avoid repeated dispatch after first resolution. */
    private static volatile IDoubleVectorComputer resolvedBase = null;

    /** Lock for one-time initialization of computer backends (SIMD/GPU). */
    private static final Object INIT_LOCK = new Object();

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
        Boolean result = ifSIMDSupported;
        if (result == null) {
            synchronized (INIT_LOCK) {
                if (ifSIMDSupported == null) {
                    ifSIMDSupported = ComputerConfig.checkIfSIMDSupported();
                }
                result = ifSIMDSupported;
            }
        }
        return result;
    }

    /**
     * 检查是否支持GPU，延迟检测
     *
     * @return
     */
    private static boolean checkIfGPUSupported() {
        Boolean result = ifGPUSupported;
        if (result == null) {
            synchronized (INIT_LOCK) {
                if (ifGPUSupported == null) {
                    ifGPUSupported = ComputerConfig.checkIfGPUSupported();
                }
                result = ifGPUSupported;
            }
        }
        return result;
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
    private IDoubleVectorComputer fetchComputer(long size) {
        // Fast path: return cached base computer for non-GPU operations
        IDoubleVectorComputer base = resolvedBase;
        if (base != null && !(ComputerConfig.USE_GPU && checkIfGPUSupported()
                && size > ComputerConfig.GPU_VECTOR_THRESHOLD)) {
            return base;
        }

        // Slow path: synchronized one-time initialization of SIMD/GPU backends.
        // Double-checked locking with volatile fields for safe publication.
        synchronized (INIT_LOCK) {
            // Re-check after acquiring lock — another thread may have initialized
            IDoubleVectorComputer baseComputer = resolvedBase;
            if (baseComputer != null && !(ComputerConfig.USE_GPU && checkIfGPUSupported()
                    && size > ComputerConfig.GPU_VECTOR_THRESHOLD)) {
                return baseComputer;
            }

            if (ComputerConfig.USE_SIMD && checkIfSIMDSupported()) {
                if (simd == null) {
                    try {
                        Class<?> simdClass = Class.forName("com.yishape.lab.math.compute.SIMDDoubleComputer");
                        simd = (IDoubleVectorComputer) simdClass.getDeclaredConstructor().newInstance();
                        log.info("SIMD vector computer initialized");
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
                        sisd = new SISDDoubleComputer();
                    }
                    baseComputer = sisd;
                }
            } else {
                if (sisd == null) {
                    sisd = new SISDDoubleComputer();
                }
                baseComputer = sisd;
            }
            resolvedBase = baseComputer;

            // Step 2: wrap with GPU if available and enabled
            if (ComputerConfig.USE_GPU && checkIfGPUSupported() && size > ComputerConfig.GPU_VECTOR_THRESHOLD) {
                if (gpu == null) {
                    try {
                        Class<?> gpuClass = Class.forName("com.yishape.lab.math.compute.GPUDoubleComputer");
                        gpu = (IDoubleVectorComputer) gpuClass
                                .getDeclaredConstructor(IDoubleVectorComputer.class)
                                .newInstance(baseComputer);
                        log.info("GPU vector computer initialized, wrapping {}",
                                simd != null ? "SIMD" : "SISD");
                    } catch (Throwable t) {
                        log.debug("GPU wrapper unavailable, using base computer: {}", t.toString());
                        ifGPUSupported = false;
                        gpu = null;
                    }
                }
                if (gpu != null) {
                    return gpu;
                }
            }

            return baseComputer;
        }
    }

    @Override
    public double[] binaryOperate(double[] x1, double[] x2, BinaryOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[] binaryOperate(double[] x1, double x2, BinaryOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[][] binaryOperate(double[][] x1, double[][] x2, BinaryOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[][] binaryOperate(double[][] x1, double x2, BinaryOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[] universalOperate(double[] x, UniversalOperation operation, double additionalParam) {
        var computer = this.fetchComputer(x.length);
        return computer.universalOperate(x, operation, additionalParam);
    }

    @Override
    public double[][] universalOperate(double[][] x, UniversalOperation operation, double additionalParam) {
        var computer = this.fetchComputer((long) x.length * x[0].length);
        return computer.universalOperate(x, operation, additionalParam);
    }

    @Override
    public double reduceOperate(double[] x, ReduceOperation operation) {
        var computer = this.fetchComputer(x.length);
        return computer.reduceOperate(x, operation);
    }

    @Override
    public double reduceOperate(double[][] x, ReduceOperation operation) {
        var computer = this.fetchComputer((long) x.length * x[0].length);
        return computer.reduceOperate(x, operation);
    }

    @Override
    public double binaryReduceOperate(double[] x1, double[] x2, BinaryReduceOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public double binaryReduceOperate(double[][] x1, double[][] x2, BinaryReduceOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public double[] elementWiseMin(double[] x1, double[] x2) {
        var computer = this.fetchComputer(x1.length);
        return computer.elementWiseMin(x1, x2);
    }

    @Override
    public double[][] elementWiseMin(double[][] x1, double[][] x2) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.elementWiseMin(x1, x2);
    }

    @Override
    public double[] elementWiseMax(double[] x1, double[] x2) {
        var computer = this.fetchComputer(x1.length);
        return computer.elementWiseMax(x1, x2);
    }

    @Override
    public double[][] elementWiseMax(double[][] x1, double[][] x2) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.elementWiseMax(x1, x2);
    }

    @Override
    public double[] negate(double[] x) {
        var computer = this.fetchComputer(x.length);
        return computer.negate(x);
    }

    @Override
    public double[][] negate(double[][] x) {
        var computer = this.fetchComputer((long) x.length * x[0].length);
        return computer.negate(x);
    }

    @Override
    public boolean[] logicalCompare(double[] x1, double[] x2, LogicalCompare operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[] logicalCompare(double[] x, double scalar, LogicalCompare operation) {
        var computer = this.fetchComputer(x.length);
        return computer.logicalCompare(x, scalar, operation);
    }

    @Override
    public boolean[] logicalOperate(double[] x1, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.logicalOperate(x1, operation);
    }

    @Override
    public boolean[] logicalOperate(double[] x, java.util.function.DoublePredicate predicate) {
        var computer = this.fetchComputer(x.length);
        return computer.logicalOperate(x, predicate);
    }

    @Override
    public boolean[][] logicalOperate(double[][] x, java.util.function.DoublePredicate predicate) {
        var computer = this.fetchComputer((long) x.length * x[0].length);
        return computer.logicalOperate(x, predicate);
    }

    @Override
    public boolean[] logicalOperate(double[] x1, double[] x2, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.logicalOperate(x1, x2, operation);
    }

    @Override
    public double[][] transpose(double[][] matrix) {
        var computer = this.fetchComputer((long) matrix.length * matrix[0].length);
        return computer.transpose(matrix);
    }

    @Override
    public double[][] transpose(double[] rowVector) {
        var computer = this.fetchComputer(rowVector.length);
        return computer.transpose(rowVector);
    }

    @Override
    public double[][] mmul(double[][] a, double[][] b) {
        var computer = this.fetchComputer((long) a.length * a[0].length * b[0].length);
        return computer.mmul(a, b);
    }

    @Override
    public double[][] outer(double[] a, double[] b) {
        var computer = this.fetchComputer(a.length * b.length);
        return computer.outer(a, b);
    }

    @Override
    public double[] sign(double[] array) {
        var computer = this.fetchComputer(array.length);
        return computer.sign(array);
    }

    @Override
    public double[] fill(int size, double value) {
        var computer = this.fetchComputer(size);
        return computer.fill(size, value);
    }

    @Override
    public double[][] sign(double[][] array) {
        var computer = this.fetchComputer((long) array.length * array[0].length);
        return computer.sign(array);
    }

    @Override
    public double[] diff(double[] array, int stride) {
        var computer = this.fetchComputer(array.length);
        return computer.diff(array, stride);
    }

    @Override
    public double[] where(boolean[] mask, double[] a, double[] b) {
        var computer = this.fetchComputer(mask.length);
        return computer.where(mask, a, b);
    }

    @Override
    public double[][] where(boolean[][] mask, double[][] a, double[][] b) {
        var computer = this.fetchComputer((long) mask.length * mask[0].length);
        return computer.where(mask, a, b);
    }

    @Override
    public boolean[][] logicalCompare(double[][] x1, double[][] x2, LogicalCompare operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[][] logicalOperate(double[][] x1, LogicalOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.logicalOperate(x1, operation);
    }

    @Override
    public boolean[][] logicalOperate(double[][] x1, double[][] x2, LogicalOperation operation) {
        var computer = this.fetchComputer((long) x1.length * x1[0].length);
        return computer.logicalOperate(x1, x2, operation);
    }

}
