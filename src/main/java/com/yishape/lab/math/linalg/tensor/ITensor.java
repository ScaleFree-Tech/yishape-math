package com.yishape.lab.math.linalg.tensor;

import java.util.Arrays;

/**
 * 张量基础接口（轻量级，IDoubleTensor 的子集）.
 * <p>
 * 工厂方法返回 IDoubleTensor 实例，新代码推荐使用 IDoubleTensor API.
 */
public interface ITensor {

    // ==================== 工厂方法 ====================

    static IDoubleTensor tensor(double[] data, int... shape) {
        return new RereDoubleTensor(data.clone(), shape);
    }

    static IDoubleTensor tensor(double[][] data) {
        int rows = data.length;
        int cols = data[0].length;
        double[] flat = new double[rows * cols];
        int idx = 0;
        for (double[] row : data) {
            for (int j = 0; j < cols; j++) {
                flat[idx++] = row[j];
            }
        }
        return new RereDoubleTensor(flat, rows, cols);
    }

    static IDoubleTensor tensor(double[][][] data) {
        int dim0 = data.length;
        int dim1 = data[0].length;
        int dim2 = data[0][0].length;
        double[] flat = new double[dim0 * dim1 * dim2];
        int idx = 0;
        for (double[][] d1 : data) {
            for (double[] d2 : d1) {
                for (double v : d2) {
                    flat[idx++] = v;
                }
            }
        }
        return new RereDoubleTensor(flat, dim0, dim1, dim2);
    }

    static IDoubleTensor ones(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        double[] data = new double[(int) size];
        Arrays.fill(data, 1.0);
        return new RereDoubleTensor(data, shape);
    }

    static IDoubleTensor zeros(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        return new RereDoubleTensor(new double[(int) size], shape);
    }

    static IDoubleTensor rand(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        double[] data = new double[(int) size];
        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }
        return new RereDoubleTensor(data, shape);
    }

    static IDoubleTensor randn(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        double[] data = new double[(int) size];
        for (int i = 0; i < size; i++) {
            data[i] = nextGaussian();
        }
        return new RereDoubleTensor(data, shape);
    }

    static IDoubleTensor eye(int n, int... extraDims) {
        int[] shape = new int[extraDims.length + 2];
        shape[0] = n;
        shape[1] = n;
        System.arraycopy(extraDims, 0, shape, 2, extraDims.length);
        long size = 1;
        for (int d : shape) size *= d;
        double[] data = new double[(int) size];
        int[] strides = TensorShape.computeCStrides(shape);
        int diagStride = strides[0] + strides[1];
        long batchCount = extraDims.length == 0 ? 1 :
            size / (n * n);
        for (long b = 0; b < batchCount; b++) {
            for (int i = 0; i < n; i++) {
                data[(int) (i * diagStride + b)] = 1.0;
            }
        }
        return new RereDoubleTensor(data, shape);
    }

    static IDoubleTensor scalar(double value) {
        return new RereDoubleTensor(new double[]{value}, 1);
    }

    static IDoubleTensor empty(int... shape) {
        long size = 1;
        for (int d : shape) size *= d;
        return new RereDoubleTensor(new double[(int) size], shape);
    }

    /**
     * 从 IDoubleVector + shape 创建.
     */
    static IDoubleTensor fromVector(com.yishape.lab.math.linalg.IDoubleVector vec, int... shape) {
        return new RereDoubleTensor(vec.toDoubleArray(), shape);
    }

    /**
     * 从 flat double[] + strides + offset 创建视图.
     */
    static IDoubleTensor fromStrided(double[] data, int offset, int[] shape, int[] strides) {
        return new RereDoubleTensor(data, offset, new TensorShape(shape, strides));
    }

    static IDoubleTensor full(int[] shape, double value) {
        long size = 1;
        for (int d : shape) size *= d;
        double[] data = new double[(int) size];
        Arrays.fill(data, value);
        return new RereDoubleTensor(data, shape);
    }

    static IDoubleTensor arange(double start, double end, double step) {
        int n = (int) Math.ceil((end - start) / step);
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = start + i * step;
        }
        return new RereDoubleTensor(data, n);
    }

    private static double nextGaussian() {
        double u1 = Math.random();
        double u2 = Math.random();
        return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }


    // ==================== 实例方法 ====================

    int rank();
    int[] shape();
    int dim(int axis);
    long totalSize();
    double get(int... indices);
    ITensor set(double value, int... indices);
    ITensor fill(double value);
    IDoubleTensor copy();
    IDoubleTensor clone();
    IDoubleTensor reshape(int... newShape);
    double[] toDoubleArray();
}
