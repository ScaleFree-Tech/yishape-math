package com.yishape.lab.math.compute;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * 统一向量操作计算器，根据情况选择合适的计算器
 *
 * @author lteb2
 */
public class DoubleVectorComputer implements IDoubleVectorComputer,Serializable {

    private static final Logger log = LoggerFactory.getLogger(DoubleVectorComputer.class);


    private static IDoubleVectorComputer gpu = null;
    private static IDoubleVectorComputer simd = null;
    private static IDoubleVectorComputer sisd = null;

    private static Boolean ifSIMDSupported = null; // 使用Boolean对象，null表示未检测
    private static Boolean ifGPUSupported = null; // 使用Boolean对象，null表示未检测

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
     * @param size
     * @return
     */
    private IDoubleVectorComputer fetchComputer(long size) {
        IDoubleVectorComputer computer = null;
        if (ComputerConfig.USE_GPU && checkIfGPUSupported() && size > ComputerConfig.GPU_VECTOR_THRESHOLD) {
            if (gpu == null) {
                try {
                    Class<?> gpuClass = Class.forName("com.yishape.lab.math.compute.GPUDoubleComputer");
                    gpu = (IDoubleVectorComputer) gpuClass.getDeclaredConstructor().newInstance();
                } catch (Exception e) {
                    log.error("exception", e);
                }
            }
            computer = gpu;
        } else if (ComputerConfig.USE_SIMD && checkIfSIMDSupported()) {
            if (simd == null) {
                try {
                    Class<?> simdClass = Class.forName("com.yishape.lab.math.compute.SIMDDoubleComputer");
                    simd = (IDoubleVectorComputer) simdClass.getDeclaredConstructor().newInstance();
                } catch (Throwable t) {
                    log.debug("SIMD unavailable, using SISD: {}", t.toString());
                    ifSIMDSupported = false;
                    simd = null;
                }
            }
            if (simd != null) {
                computer = simd;
            } else {
                if (sisd == null) {
                    sisd = new SISDDoubleComputer();
                }
                computer = sisd;
            }
        } else {
            if (sisd == null) {
                sisd = new SISDDoubleComputer();
            }
            computer = sisd;
        }

        return computer;
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
        var computer = this.fetchComputer(x1.length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[][] binaryOperate(double[][] x1, double x2, BinaryOperation operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public double[] universalOperate(double[] x, UniversalOperation operation, double additionalParam) {
        var computer = this.fetchComputer(x.length);
        return computer.universalOperate(x, operation, additionalParam);
    }

    @Override
    public double[][] universalOperate(double[][] x, UniversalOperation operation, double additionalParam) {
        var computer = this.fetchComputer(x.length * x[0].length);
        return computer.universalOperate(x, operation, additionalParam);
    }

    @Override
    public double reduceOperate(double[] x, ReduceOperation operation) {
        var computer = this.fetchComputer(x.length);
        return computer.reduceOperate(x, operation);
    }

    @Override
    public double reduceOperate(double[][] x, ReduceOperation operation) {
        var computer = this.fetchComputer(x.length * x[0].length);
        return computer.reduceOperate(x, operation);
    }

    @Override
    public double binaryReduceOperate(double[] x1, double[] x2, BinaryReduceOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public double binaryReduceOperate(double[][] x1, double[][] x2, BinaryReduceOperation operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public double[] elementWiseMin(double[] x1, double[] x2) {
        var computer = this.fetchComputer(x1.length);
        return computer.elementWiseMin(x1, x2);
    }

    @Override
    public double[][] elementWiseMin(double[][] x1, double[][] x2) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.elementWiseMin(x1, x2);
    }

    @Override
    public double[] elementWiseMax(double[] x1, double[] x2) {
        var computer = this.fetchComputer(x1.length);
        return computer.elementWiseMax(x1, x2);
    }

    @Override
    public double[][] elementWiseMax(double[][] x1, double[][] x2) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.elementWiseMax(x1, x2);
    }

    @Override
    public double[] negate(double[] x) {
        var computer = this.fetchComputer(x.length);
        return computer.negate(x);
    }

    @Override
    public double[][] negate(double[][] x) {
        var computer = this.fetchComputer(x.length * x[0].length);
        return computer.negate(x);
    }

    @Override
    public boolean[] logicalCompare(double[] x1, double[] x2, LogicalCompare operation) {
        var computer = this.fetchComputer(x1.length * x2.length);
        return computer.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[] logicalOperate(double[] x1, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.logicalOperate(x1, operation);
    }

    @Override
    public boolean[] logicalOperate(double[] x1, double[] x2, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length * x2.length);
        return computer.logicalOperate(x1, x2, operation);
    }

    @Override
    public double[][] transpose(double[][] matrix) {
        var computer = this.fetchComputer(matrix.length * matrix[0].length);
        return computer.transpose(matrix);
    }

    @Override
    public double[][] transpose(double[] rowVector) {
        var computer = this.fetchComputer(rowVector.length);
        return computer.transpose(rowVector);
    }

    @Override
    public double[][] mmul(double[][] a, double[][] b) {
        var computer = this.fetchComputer(a.length * b.length);
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
    public double[][] sign(double[][] array) {
        var computer = this.fetchComputer(array.length * array[0].length);
        return computer.sign(array);
    }

    @Override
    public double[] diff(double[] array, int stride) {
        var computer = this.fetchComputer(array.length);
        return computer.diff(array, stride);
    }

    @Override
    public boolean[][] logicalCompare(double[][] x1, double[][] x2, LogicalCompare operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[][] logicalOperate(double[][] x1, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.logicalOperate(x1, operation);
    }

    @Override
    public boolean[][] logicalOperate(double[][] x1, double[][] x2, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.logicalOperate(x1, x2, operation);
    }

}
