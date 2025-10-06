package com.yishape.lab.math.compute;

/**
 * 统一向量操作计算器，根据情况选择合适的计算器
 *
 * @author lteb2
 */
public class FloatVectorComputer implements IFloatVectorComputer {

    private static IFloatVectorComputer gpu = null;
    private static IFloatVectorComputer simd = null;
    private static IFloatVectorComputer sisd = null;

    private static boolean ifSIMDSupported = false;
    private static boolean ifGPUSupported = false;
    
    static {
        ifSIMDSupported = ComputerConfig.checkIfSIMDSupported();
        ifGPUSupported = ComputerConfig.checkIfGPUSupported();
    }
    
    /**
     * 基于数据的规模和配置选择合适的计算器
     *
     * @param size
     * @return
     */
    private IFloatVectorComputer fetchComputer(long size) {
        IFloatVectorComputer computer = null;
        if (computer == null) {
            if (size > ComputerConfig.GPU_VECTOR_THRESHOLD && ifGPUSupported) {
                if (gpu == null) {
                    gpu = new GPUFloatComputer();
                }
                computer = gpu;
            } else if (ComputerConfig.USE_SIMD &&ifSIMDSupported) {
                if (simd == null) {
                    simd = new SIMDFloatComputer();
                }
                computer = simd;
            } else {
                if (sisd == null) {
                    sisd = new SISDFloatComputer();
                }
                computer = sisd;
            }
        }
        return computer;
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
        var computer = this.fetchComputer(x1.length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[][] binaryOperate(float[][] x1, float x2, BinaryOperation operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.binaryOperate(x1, x2, operation);
    }

    @Override
    public float[] universalOperate(float[] x, UniversalOperation operation, float additionalParam) {
        var computer = this.fetchComputer(x.length);
        return computer.universalOperate(x, operation, additionalParam);
    }

    @Override
    public float[][] universalOperate(float[][] x, UniversalOperation operation, float additionalParam) {
        var computer = this.fetchComputer(x.length * x[0].length);
        return computer.universalOperate(x, operation, additionalParam);
    }

    @Override
    public float reduceOperate(float[] x, ReduceOperation operation) {
        var computer = this.fetchComputer(x.length);
        return computer.reduceOperate(x, operation);
    }

    @Override
    public float reduceOperate(float[][] x, ReduceOperation operation) {
        var computer = this.fetchComputer(x.length * x[0].length);
        return computer.reduceOperate(x, operation);
    }

    @Override
    public float binaryReduceOperate(float[] x1, float[] x2, BinaryReduceOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public float binaryReduceOperate(float[][] x1, float[][] x2, BinaryReduceOperation operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.binaryReduceOperate(x1, x2, operation);
    }

    @Override
    public float[] elementWiseMin(float[] x1, float[] x2) {
        var computer = this.fetchComputer(x1.length);
        return computer.elementWiseMin(x1, x2);
    }

    @Override
    public float[][] elementWiseMin(float[][] x1, float[][] x2) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.elementWiseMin(x1, x2);
    }

    @Override
    public float[] elementWiseMax(float[] x1, float[] x2) {
        var computer = this.fetchComputer(x1.length);
        return computer.elementWiseMax(x1, x2);
    }

    @Override
    public float[][] elementWiseMax(float[][] x1, float[][] x2) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.elementWiseMax(x1, x2);
    }

    @Override
    public float[] negate(float[] x) {
        var computer = this.fetchComputer(x.length);
        return computer.negate(x);
    }

    @Override
    public float[][] negate(float[][] x) {
        var computer = this.fetchComputer(x.length * x[0].length);
        return computer.negate(x);
    }

    @Override
    public boolean[] logicalCompare(float[] x1, float[] x2, LogicalCompare operation) {
        var computer = this.fetchComputer(x1.length * x2.length);
        return computer.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[] logicalOperate(float[] x1, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length);
        return computer.logicalOperate(x1, operation);
    }

    @Override
    public boolean[] logicalOperate(float[] x1, float[] x2, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length * x2.length);
        return computer.logicalOperate(x1, x2, operation);
    }

    @Override
    public float[][] transpose(float[][] matrix) {
        var computer = this.fetchComputer(matrix.length * matrix[0].length);
        return computer.transpose(matrix);
    }

    @Override
    public float[][] transpose(float[] rowVector) {
        var computer = this.fetchComputer(rowVector.length);
        return computer.transpose(rowVector);
    }

    @Override
    public float[][] mmul(float[][] a, float[][] b) {
        var computer = this.fetchComputer(a.length * b.length);
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
        var computer = this.fetchComputer(array.length * array[0].length);
        return computer.sign(array);
    }

    @Override
    public float[] diff(float[] array, int stride) {
        var computer = this.fetchComputer(array.length);
        return computer.diff(array, stride);
    }

    @Override
    public boolean[][] logicalCompare(float[][] x1, float[][] x2, LogicalCompare operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.logicalCompare(x1, x2, operation);
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.logicalOperate(x1, operation);
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, float[][] x2, LogicalOperation operation) {
        var computer = this.fetchComputer(x1.length * x1[0].length);
        return computer.logicalOperate(x1, x2, operation);
    }

}
