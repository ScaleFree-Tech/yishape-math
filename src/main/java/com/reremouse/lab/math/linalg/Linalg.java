package com.reremouse.lab.math.linalg;

import java.util.List;

/**
 * 线性代数工厂类 / Linear Algebra Factory Class
 * <p>
 * 本类提供了创建各种矩阵和向量的静态工厂方法，通过委托给IMatrix和IVector的统一接口实现。
 * 除了`of`方法重命名为`matrix`和`vector`方法外，其它方法名保持不变。
 * </p>
 * <p>
 * This class provides static factory methods for creating matrices and vectors,
 * implemented by delegating to the unified IMatrix and IVector interfaces.
 * Except for `of` methods renamed to `matrix` and `vector` methods, all other method names remain unchanged.
 * </p>
 *
 * <h3>架构设计 / Architecture Design:</h3>
 * <p>
 * 委托链：**Linalg** → **IMatrix/IVector** → **具体实现类**<br>
 * Delegation chain: **Linalg** → **IMatrix/IVector** → **Concrete implementations**
 * </p>
 * <ul>
 * <li>**矩阵创建** / **Matrix creation**: {@code Linalg.matrix(...) → IMatrix.of(...) → IFloatMatrix/IDoubleMatrix}</li>
 * <li>**向量创建** / **Vector creation**: {@code Linalg.vector(...) → IVector.of(...) → IFloatVector/IDoubleVector}</li>
 * <li>**类型安全** / **Type safety**: 编译时类型检查和运行时类型安全 / Compile-time type checking and runtime type safety</li>
 * <li>**API统一** / **API consistency**: 提供一致的命名和使用模式 / Provides consistent naming and usage patterns</li>
 * </ul>
 *
 * <h3>使用示例 / Usage Examples:</h3>
 * <pre>
 * {@code
 * // 矩阵创建 / Matrix creation
 * IMatrix<Double> m1 = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
 * IMatrix<Float> m2 = Linalg.matrix(new float[][]{{1f, 2f}, {3f, 4f}});
 * IMatrix<Double> identity = Linalg.eye(3);
 * IMatrix<Double> random = Linalg.rand(3, 3);
 * 
 * // 向量创建 / Vector creation
 * IVector<Double> v1 = Linalg.vector(new double[]{1, 2, 3});
 * IVector<Float> v2 = Linalg.vector(new float[]{1f, 2f, 3f});
 * IVector<Double> range = Linalg.range(0, 10, 2);  // [0, 2, 4, 6, 8]
 * IVector<Double> ones = Linalg.ones(5);
 * }
 * </pre>
 *
 * <h3>优势 / Advantages:</h3>
 * <ul>
 * <li>**集中化** / **Centralized**: 所有工厂方法集中在一个类中 / All factory methods centralized in one class</li>
 * <li>**易发现** / **Discoverable**: IDE自动完成和文档更便于使用 / IDE autocomplete and documentation easier to use</li>
 * <li>**一致性** / **Consistency**: 命名约定和使用模式一致 / Consistent naming conventions and usage patterns</li>
 * <li>**可扩展** / **Extensible**: 易于添加新的工厂方法 / Easy to add new factory methods</li>
 * </ul>
 * 
 * @author lteb2
 * @version 2.0
 * @since 1.0
 * @see IMatrix 矩阵操作接口 / Matrix operations interface
 * @see IVector 向量操作接口 / Vector operations interface
 * @see IFloatMatrix Float类型矩阵实现 / Float type matrix implementation
 * @see IDoubleMatrix Double类型矩阵实现 / Double type matrix implementation
 * @see IFloatVector Float类型向量实现 / Float type vector implementation
 * @see IDoubleVector Double类型向量实现 / Double type vector implementation
 */
public class Linalg {
    
    // ========== 矩阵创建方法 (委托给IMatrix) / Matrix Creation Methods (Delegating to IMatrix) ==========
    
    // 'of' methods renamed to 'matrix' methods, all others keep original names
    // Note: Generic matrix method removed due to type system limitations
    
    public static IMatrix<Double> matrix(double[][] data) { 
        return IMatrix.of(data); 
    }
    
    public static IMatrix<Float> matrix(float[][] data) { 
        return IMatrix.of(data); 
    }
    
    public static IMatrix<Double> matrix(Double[][] data) { 
        return IMatrix.of(data); 
    }
    
    public static IMatrix<Float> matrix(Float[][] data) { 
        return IMatrix.of(data); 
    }
    
    public static IMatrix<Double> matrixFromDoubleList(List<double[]> data) { 
        return IMatrix.ofDoubleList(data); 
    }
    
    public static IMatrix<Float> matrixFromFloatList(List<float[]> data) { 
        return IMatrix.ofFloatList(data); 
    }
    
    // All other methods delegate to IMatrix with same names
    public static IMatrix<Double> rand(int rows, int cols) { 
        return IMatrix.rand(rows, cols); 
    }
    
    public static <T extends Number> IMatrix<T> rand(int rows, int cols, Class<T> type) {
        return IMatrix.rand(rows, cols, type);
    }
    
    public static IMatrix<Double> rand(int rows, int cols, long seed) { 
        return IMatrix.rand(rows, cols, seed); 
    }
    
    public static <T extends Number> IMatrix<T> rand(int rows, int cols, long seed, Class<T> type) {
        return IMatrix.rand(rows, cols, seed, type);
    }
    
    public static IMatrix<Double> randn(int rows, int cols) { 
        return IMatrix.randn(rows, cols); 
    }
    
    public static <T extends Number> IMatrix<T> randn(int rows, int cols, Class<T> type) {
        return IMatrix.randn(rows, cols, type);
    }
    
    public static IMatrix<Double> randn(int rows, int cols, double mean, double std) { 
        return IMatrix.randn(rows, cols, mean, std); 
    }
    
    public static <T extends Number> IMatrix<T> randn(int rows, int cols, T mean, T std, Class<T> type) {
        return IMatrix.randn(rows, cols, mean, std, type);
    }
    
    public static <T extends Number> IMatrix<T> randn(int rows, int cols, T mean, T std) {
        return IMatrix.randn(rows, cols, mean, std);
    }
    
    public static IMatrix<Double> ones(int rows, int cols) { 
        return IMatrix.ones(rows, cols); 
    }
    
    public static <T extends Number> IMatrix<T> ones(int rows, int cols, Class<T> type) {
        return IMatrix.ones(rows, cols, type);
    }
    
    public static IMatrix<Double> zeros(int rows, int cols) { 
        return IMatrix.zeros(rows, cols); 
    }
    
    public static <T extends Number> IMatrix<T> zeros(int rows, int cols, Class<T> type) {
        return IMatrix.zeros(rows, cols, type);
    }
    
    public static IMatrix<Double> eye(int size) { 
        return IMatrix.eye(size); 
    }
    
    public static <T extends Number> IMatrix<T> eye(int size, Class<T> type) {
        return IMatrix.eye(size, type);
    }
    
    public static <T extends Number> IMatrix<T> diag(IVector diagonal, Class<T> type) {
        return IMatrix.diag(diagonal, type);
    }
    
    public static <T extends Number> IMatrix<T> diag(IVector diagonal) { 
        return IMatrix.diag(diagonal); 
    }
    
    public static <T extends Number> IMatrix<T> diag(T[] diagonal) { 
        return IMatrix.diag(diagonal); 
    }
    
    public static <T extends Number> IMatrix<T> diag(T[] diagonal, Class<T> type) {
        return IMatrix.diag(diagonal, type);
    }
    
    public static <T extends Number> IMatrix<T> diag(float[] diagonal) { 
        return IMatrix.diag(diagonal); 
    }
    
    public static <T extends Number> IMatrix<T> diag(double[] diagonal) { 
        return IMatrix.diag(diagonal); 
    }
    
    public static <T extends Number> IMatrix<T> fromArray(T[] data, int rows, int cols) {
        return IMatrix.fromArray(data, rows, cols);
    }
    
    public static <T extends Number> IMatrix<T> fromArray(T[] data, int rows, int cols, Class<T> type) {
        return IMatrix.fromArray(data, rows, cols, type);
    }
    
    public static IMatrix<Double> fromArray(double[] data, int rows, int cols) {
        return IMatrix.fromArray(data, rows, cols);
    }
    
    public static IMatrix<Float> fromArray(float[] data, int rows, int cols) {
        return IMatrix.fromArray(data, rows, cols);
    }
    
    @SafeVarargs 
    public static <T extends Number> IMatrix<T> average(IMatrix<T>... matrices) { 
        if (matrices.length < 2) {
            throw new IllegalArgumentException("至少需要两个矩阵 / At least two matrices required");
        }
        // Split into two arrays for the existing average method
        int half = matrices.length / 2;
        @SuppressWarnings("unchecked")
        IMatrix<T>[] firstHalf = new IMatrix[half];
        @SuppressWarnings("unchecked")
        IMatrix<T>[] secondHalf = new IMatrix[matrices.length - half];
        System.arraycopy(matrices, 0, firstHalf, 0, half);
        System.arraycopy(matrices, half, secondHalf, 0, matrices.length - half);
        @SuppressWarnings("unchecked")
        Class<T> type = (Class<T>) Double.class;
        return IMatrix.average(firstHalf, secondHalf, type); 
    }
    
    public static <T extends Number> IMatrix<T> load(String filename, Class<T> type) { 
        return IMatrix.load(filename, type); 
    }
    
    public static IMatrix<Double> load(String filename) {
        return IMatrix.load(filename);
    }
    
    // ========== 向量创建方法 (委托给IVector) / Vector Creation Methods (Delegating to IVector) ==========
    
    // 'of' methods renamed to 'vector' methods, all others keep original names
    
    public static <T extends Number> IVector<T> vector(double[] data) {
        return IVector.of(data);
    }
    
    public static <T extends Number> IVector<T> vector(Double[] data) {
        return IVector.of(data);
    }
    
    public static <T extends Number> IVector<T> vector(float[] data) {
        return IVector.of(data);
    }
    
    public static <T extends Number> IVector<T> vector(Float[] data) {
        return IVector.of(data);
    }
    
    public static <T extends Number> IVector<T> vector(int[] data) {
        return IVector.of(data);
    }
    
    public static <T extends Number> IVector<T> vector(Integer[] data) {
        return IVector.of(data);
    }
    
    public static IVector<Double> vector(Double value) {
        return IVector.of(value);
    }
    
    public static IVector<Double> vector(Double value1, Double value2) {
        return IVector.of(value1, value2);
    }
    
    public static IVector<Float> vector(Float value) {
        return IVector.of(value);
    }
    
    public static IVector<Float> vector(Float value1, Float value2) {
        return IVector.of(value1, value2);
    }
    
    // Range vector creation methods
    public static <T extends Number> IVector<T> range(int start, int end, int step, Class<T> type) {
        return IVector.range(start, end, step, type);
    }
    
    public static IVector<Double> range(int start, int end, int step) {
        return IVector.range(start, end, step);
    }
    
    public static <T extends Number> IVector<T> range(int start, int end, Class<T> type) {
        return IVector.range(start, end, type);
    }
    
    public static IVector<Double> range(int start, int end) {
        return IVector.range(start, end);
    }
    
    public static <T extends Number> IVector<T> range(int end, Class<T> type) {
        return IVector.range(end, type);
    }
    
    public static IVector<Double> range(int end) {
        return IVector.range(end);
    }
    
    // Special vector creation methods
    public static <T extends Number> IVector<T> ones(int len, Class<T> type) {
        return IVector.ones(len, type);
    }
    
    public static IVector<Double> ones(int len) {
        return IVector.ones(len);
    }
    
    public static <T extends Number> IVector<T> zeros(int len, Class<T> type) {
        return IVector.zeros(len, type);
    }
    
    public static IVector<Double> zeros(int len) {
        return IVector.zeros(len);
    }
    
    // Random vector generation methods
    public static <T extends Number> IVector<T> rand(int length, Class<T> type) {
        return IVector.rand(length, type);
    }
    
    public static IVector<Double> rand(int length) {
        return IVector.rand(length);
    }
    
    public static <T extends Number> IVector<T> randn(int length, T mean, T std, Class<T> type) {
        return IVector.randn(length, mean, std, type);
    }
    
    public static IVector<Double> randn(int length, Double mean, Double std) {
        return IVector.randn(length, mean, std);
    }
    
    // Linear space generation methods
    public static <T extends Number> IVector<T> linspace(T start, T stop, int num, Class<T> type) {
        return IVector.linspace(start, stop, num, type);
    }
    
    public static IVector<Double> linspace(Double start, Double stop, int num) {
        return IVector.linspace(start, stop, num);
    }
    
    public static <T extends Number> IVector<T> logspace(T start, T stop, int num, Class<T> type) {
        return IVector.logspace(start, stop, num, type);
    }
    
    public static IVector<Double> logspace(Double start, Double stop, int num) {
        return IVector.logspace(start, stop, num);
    }
}