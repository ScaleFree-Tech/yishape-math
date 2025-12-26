package com.yishape.lab.math.linalg;

import java.util.List;
import com.yishape.lab.util.Tuple2;

/**
 * 线性代数工厂类 / Linear Algebra Factory Class
 * <p>
 * 本类提供了创建各种矩阵和向量的静态工厂方法，通过委托给IMatrix和IVector的统一接口实现。
 * 除了`of`方法重命名为`matrix`和`vector`方法外，其它方法名保持不变。
 * </p>
 * <p>
 * This class provides static factory methods for creating matrices and vectors,
 * implemented by delegating to the unified IMatrix and IVector interfaces.
 * Except for `of` methods renamed to `matrix` and `vector` methods, all other
 * method names remain unchanged.
 * </p>
 *
 * <h3>架构设计 / Architecture Design:</h3>
 * <p>
 * 委托链：**Linalg** → **IMatrix/IVector** → **具体实现类**<br>
 * Delegation chain: **Linalg** → **IMatrix/IVector** → **Concrete implementations
 **
 * </p>
 * <ul>
 * <li>**矩阵创建** / **Matrix creation**:
 * {@code Linalg.matrix(...) → IMatrix.of(...) → IFloatMatrix/IDoubleMatrix}</li>
 * <li>**向量创建** / **Vector creation**:
 * {@code Linalg.vector(...) → IVector.of(...) → IFloatVector/IDoubleVector}</li>
 * <li>**类型安全** / **Type safety**: 编译时类型检查和运行时类型安全 / Compile-time type checking
 * and runtime type safety</li>
 * <li>**API统一** / **API consistency**: 提供一致的命名和使用模式 / Provides consistent
 * naming and usage patterns</li>
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
 * <li>**集中化** / **Centralized**: 所有工厂方法集中在一个类中 / All factory methods
 * centralized in one class</li>
 * <li>**易发现** / **Discoverable**: IDE自动完成和文档更便于使用 / IDE autocomplete and
 * documentation easier to use</li>
 * <li>**一致性** / **Consistency**: 命名约定和使用模式一致 / Consistent naming conventions
 * and usage patterns</li>
 * <li>**可扩展** / **Extensible**: 易于添加新的工厂方法 / Easy to add new factory
 * methods</li>
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
    /**
     * 创建Double类型矩阵（double数组） / Create Double matrix (double array)
     * <p>
     * 使用给定的double二维数组创建矩阵实例。此方法委托给 {@link IMatrix#of(double[][])} 实现。 Creates a
     * matrix instance with the given double 2D array. This method delegates to
     * {@link IMatrix#of(double[][])}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
     * IMatrix<Double> matrix = Linalg.matrix(data);
     * }</pre>
     * </p>
     *
     * @param data double二维数组，表示矩阵数据 / double 2D array representing matrix data
     * @return 新的矩阵实例（IDoubleMatrix）/ New matrix instance (IDoubleMatrix)
     * @throws IllegalArgumentException 如果数据为null或空数组 / if data is null or empty
     * array
     * @see IMatrix#of(double[][]) 实际实现方法 / Actual implementation method
     */
    public static IMatrix<Double> matrix(double[][] data) {
        return IMatrix.of(data);
    }

    /**
     * 创建Float类型矩阵（float数组） / Create Float matrix (float array)
     * <p>
     * 使用给定的float二维数组创建矩阵实例。此方法委托给 {@link IMatrix#of(float[][])} 实现。 Creates a
     * matrix instance with the given float 2D array. This method delegates to
     * {@link IMatrix#of(float[][])}.
     * </p>
     *
     * @param data float二维数组，表示矩阵数据 / float 2D array representing matrix data
     * @return 新的矩阵实例（IFloatMatrix）/ New matrix instance (IFloatMatrix)
     * @throws IllegalArgumentException 如果数据为null或空数组 / if data is null or empty
     * array
     * @see IMatrix#of(float[][]) 实际实现方法 / Actual implementation method
     */
    public static IMatrix<Float> matrix(float[][] data) {
        return IMatrix.of(data);
    }

    /**
     * 创建Double类型矩阵（Double包装类数组） / Create Double matrix (Double wrapper array)
     * <p>
     * 使用给定的Double包装类二维数组创建矩阵实例。此方法委托给 {@link IMatrix#of(Double[][])} 实现。
     * Creates a matrix instance with the given Double wrapper 2D array. This
     * method delegates to {@link IMatrix#of(Double[][])}.
     * </p>
     *
     * @param data Double二维数组，表示矩阵数据 / Double 2D array representing matrix data
     * @return 新的矩阵实例（IDoubleMatrix）/ New matrix instance (IDoubleMatrix)
     * @throws IllegalArgumentException 如果数据为null或空数组 / if data is null or empty
     * array
     * @see IMatrix#of(Double[][]) 实际实现方法 / Actual implementation method
     */
    public static IMatrix<Double> matrix(Double[][] data) {
        return IMatrix.of(data);
    }

    /**
     * 创建Float类型矩阵（Float包装类数组） / Create Float matrix (Float wrapper array)
     * <p>
     * 使用给定的Float包装类二维数组创建矩阵实例。此方法委托给 {@link IMatrix#of(Float[][])} 实现。 Creates
     * a matrix instance with the given Float wrapper 2D array. This method
     * delegates to {@link IMatrix#of(Float[][])}.
     * </p>
     *
     * @param data Float二维数组，表示矩阵数据 / Float 2D array representing matrix data
     * @return 新的矩阵实例（IFloatMatrix）/ New matrix instance (IFloatMatrix)
     * @throws IllegalArgumentException 如果数据为null或空数组 / if data is null or empty
     * array
     * @see IMatrix#of(Float[][]) 实际实现方法 / Actual implementation method
     */
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
    /**
     * 创建随机矩阵（默认Double类型） / Create random matrix (default Double type)
     * <p>
     * 创建指定大小的随机矩阵，元素值在[0,1)范围内。此方法委托给 {@link IMatrix#rand(int, int)} 实现。
     * Creates a random matrix of specified size with elements in [0,1) range.
     * This method delegates to {@link IMatrix#rand(int, int)}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> randomMatrix = Linalg.rand(3, 4);  // 3x4随机矩阵
     * }</pre>
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     * @see IMatrix#rand(int, int) 实际实现方法 / Actual implementation method
     */
    public static IMatrix<Double> rand(int rows, int cols) {
        return IMatrix.rand(rows, cols);
    }

    /**
     * 创建指定类型的随机矩阵 / Create random matrix of specified type
     * <p>
     * 创建指定大小和类型的随机矩阵，元素值在[0,1)范围内。此方法委托给 {@link IMatrix#rand(int, int, Class)}
     * 实现。 Creates a random matrix of specified size and type with elements in
     * [0,1) range. This method delegates to
     * {@link IMatrix#rand(int, int, Class)}.
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     * @see IMatrix#rand(int, int, Class) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IMatrix<T> rand(int rows, int cols, Class<T> type) {
        return IMatrix.rand(rows, cols, type);
    }

    /**
     * 创建随机矩阵（指定种子，默认Double类型） / Create random matrix (specified seed, default
     * Double type)
     * <p>
     * 创建指定大小的随机矩阵，使用指定的种子值确保结果可重现。此方法委托给 {@link IMatrix#rand(int, int, long)}
     * 实现。 Creates a random matrix of specified size using the specified seed
     * value to ensure reproducible results. This method delegates to
     * {@link IMatrix#rand(int, int, long)}.
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param seed 随机数种子 / Random number seed
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     * @see IMatrix#rand(int, int, long) 实际实现方法 / Actual implementation method
     */
    public static IMatrix<Double> rand(int rows, int cols, long seed) {
        return IMatrix.rand(rows, cols, seed);
    }

    /**
     * 创建指定类型的随机矩阵（指定种子） / Create random matrix of specified type (specified
     * seed)
     * <p>
     * 创建指定大小、类型和种子的随机矩阵，元素值在[0,1)范围内。此方法委托给
     * {@link IMatrix#rand(int, int, long, Class)} 实现。 Creates a random matrix
     * of specified size, type and seed with elements in [0,1) range. This
     * method delegates to {@link IMatrix#rand(int, int, long, Class)}.
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param seed 随机数种子 / Random number seed
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     * @see IMatrix#rand(int, int, long, Class) 实际实现方法 / Actual implementation
     * method
     */
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

    /**
     * 创建全1矩阵（默认Double类型） / Create ones matrix (default Double type)
     * <p>
     * 创建指定大小的全1矩阵，所有元素都为1。此方法委托给 {@link IMatrix#ones(int, int)} 实现。 Creates a
     * matrix of specified size with all elements set to 1. This method
     * delegates to {@link IMatrix#ones(int, int)}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> onesMatrix = Linalg.ones(3, 4);  // 3x4全1矩阵
     * }</pre>
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 全1矩阵 / Ones matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     * @see IMatrix#ones(int, int) 实际实现方法 / Actual implementation method
     */
    public static IMatrix<Double> ones(int rows, int cols) {
        return IMatrix.ones(rows, cols);
    }

    /**
     * 创建指定类型的全1矩阵 / Create ones matrix of specified type
     * <p>
     * 创建指定大小和类型的全1矩阵，所有元素都为1。此方法委托给 {@link IMatrix#ones(int, int, Class)} 实现。
     * Creates a matrix of specified size and type with all elements set to 1.
     * This method delegates to {@link IMatrix#ones(int, int, Class)}.
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 全1矩阵 / Ones matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     * @see IMatrix#ones(int, int, Class) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IMatrix<T> ones(int rows, int cols, Class<T> type) {
        return IMatrix.ones(rows, cols, type);
    }

    /**
     * 创建零矩阵（默认Double类型） / Create zeros matrix (default Double type)
     * <p>
     * 创建指定大小的零矩阵，所有元素都为0。此方法委托给 {@link IMatrix#zeros(int, int)} 实现。 Creates a
     * matrix of specified size with all elements set to 0. This method
     * delegates to {@link IMatrix#zeros(int, int)}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> zerosMatrix = Linalg.zeros(3, 4);  // 3x4零矩阵
     * }</pre>
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 零矩阵 / Zeros matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     * @see IMatrix#zeros(int, int) 实际实现方法 / Actual implementation method
     */
    public static IMatrix<Double> zeros(int rows, int cols) {
        return IMatrix.zeros(rows, cols);
    }

    /**
     * 创建指定类型的零矩阵 / Create zeros matrix of specified type
     * <p>
     * 创建指定大小和类型的零矩阵，所有元素都为0。此方法委托给 {@link IMatrix#zeros(int, int, Class)} 实现。
     * Creates a matrix of specified size and type with all elements set to 0.
     * This method delegates to {@link IMatrix#zeros(int, int, Class)}.
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 零矩阵 / Zeros matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     * @see IMatrix#zeros(int, int, Class) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IMatrix<T> zeros(int rows, int cols, Class<T> type) {
        return IMatrix.zeros(rows, cols, type);
    }

    /**
     * 创建单位矩阵（默认Double类型） / Create identity matrix (default Double type)
     * <p>
     * 创建指定大小的单位矩阵（方阵），主对角线元素为1，其他元素为0。此方法委托给 {@link IMatrix#eye(int)} 实现。
     * Creates an identity matrix of specified size (square matrix) with 1s on
     * the main diagonal and 0s elsewhere. This method delegates to
     * {@link IMatrix#eye(int)}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> identityMatrix = Linalg.eye(3);  // 3x3单位矩阵
     * }</pre>
     * </p>
     *
     * @param size 矩阵大小（行数和列数相同） / Matrix size (rows and columns are the same)
     * @return 单位矩阵 / Identity matrix
     * @throws IllegalArgumentException 如果大小小于等于0 / if size is less than or
     * equal to 0
     * @see IMatrix#eye(int) 实际实现方法 / Actual implementation method
     */
    public static IMatrix<Double> eye(int size) {
        return IMatrix.eye(size);
    }

    /**
     * 创建指定类型的单位矩阵 / Create identity matrix of specified type
     * <p>
     * 创建指定大小和类型的单位矩阵（方阵），主对角线元素为1，其他元素为0。此方法委托给 {@link IMatrix#eye(int, Class)}
     * 实现。 Creates an identity matrix of specified size and type (square matrix)
     * with 1s on the main diagonal and 0s elsewhere. This method delegates to
     * {@link IMatrix#eye(int, Class)}.
     * </p>
     *
     * @param size 矩阵大小（行数和列数相同） / Matrix size (rows and columns are the same)
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 单位矩阵 / Identity matrix
     * @throws IllegalArgumentException 如果大小小于等于0 / if size is less than or
     * equal to 0
     * @see IMatrix#eye(int, Class) 实际实现方法 / Actual implementation method
     */
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
    /**
     * 创建向量（double数组） / Create vector (double array)
     * <p>
     * 使用给定的double数组创建向量实例。此方法委托给 {@link IVector#of(double[])} 实现。 Creates a
     * vector instance with the given double array. This method delegates to
     * {@link IVector#of(double[])}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * double[] data = {1.0, 2.0, 3.0, 4.0};
     * IVector<Double> vector = Linalg.vector(data);
     * }</pre>
     * </p>
     *
     * @param data double数组，表示向量数据 / double array representing vector data
     * @param <T> 数值类型 / Numeric type
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(double[]) 实际实现方法 / Actual implementation method
     */
    public static <T extends Double> IVector<T> vector(double[] data) {
        return IVector.of(data);
    }

    /**
     * 创建向量（Double包装类数组） / Create vector (Double wrapper array)
     * <p>
     * 使用给定的Double包装类数组创建向量实例。此方法委托给 {@link IVector#of(Double[])} 实现。 Creates a
     * vector instance with the given Double wrapper array. This method
     * delegates to {@link IVector#of(Double[])}.
     * </p>
     *
     * @param data Double数组，表示向量数据 / Double array representing vector data
     * @param <T> 数值类型 / Numeric type
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(Double[]) 实际实现方法 / Actual implementation method
     */
    public static <T extends Double> IVector<T> vector(Double[] data) {
        return IVector.of(data);
    }

    /**
     * 创建向量（float数组） / Create vector (float array)
     * <p>
     * 使用给定的float数组创建向量实例。此方法委托给 {@link IVector#of(float[])} 实现。 Creates a
     * vector instance with the given float array. This method delegates to
     * {@link IVector#of(float[])}.
     * </p>
     *
     * @param data float数组，表示向量数据 / float array representing vector data
     * @param <T> 数值类型 / Numeric type
     * @return 新的向量实例（IFloatVector）/ New vector instance (IFloatVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(float[]) 实际实现方法 / Actual implementation method
     */
    public static <T extends Float> IVector<T> vector(float[] data) {
        return IVector.of(data);
    }

    /**
     * 创建向量（Float包装类数组） / Create vector (Float wrapper array)
     * <p>
     * 使用给定的Float包装类数组创建向量实例。此方法委托给 {@link IVector#of(Float[])} 实现。 Creates a
     * vector instance with the given Float wrapper array. This method delegates
     * to {@link IVector#of(Float[])}.
     * </p>
     *
     * @param data Float数组，表示向量数据 / Float array representing vector data
     * @param <T> 数值类型 / Numeric type
     * @return 新的向量实例（IFloatVector）/ New vector instance (IFloatVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(Float[]) 实际实现方法 / Actual implementation method
     */
    public static <T extends Float> IVector<T> vector(Float[] data) {
        return IVector.of(data);
    }

    /**
     * 创建向量（int数组） / Create vector (int array)
     * <p>
     * 使用给定的int数组创建向量实例，自动转换为Double类型。此方法委托给 {@link IVector#of(int[])} 实现。
     * Creates a vector instance with the given int array, automatically
     * converted to Double type. This method delegates to
     * {@link IVector#of(int[])}.
     * </p>
     *
     * @param data int数组，表示向量数据 / int array representing vector data
     * @param <T> 数值类型 / Numeric type
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(int[]) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IVector<T> vector(int[] data) {
        return IVector.of(data);
    }

    /**
     * 创建向量（Integer包装类数组） / Create vector (Integer wrapper array)
     * <p>
     * 使用给定的Integer包装类数组创建向量实例，自动转换为Double类型。此方法委托给
     * {@link IVector#of(Integer[])} 实现。 Creates a vector instance with the
     * given Integer wrapper array, automatically converted to Double type. This
     * method delegates to {@link IVector#of(Integer[])}.
     * </p>
     *
     * @param data Integer数组，表示向量数据 / Integer array representing vector data
     * @param <T> 数值类型 / Numeric type
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(Integer[]) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IVector<T> vector(Integer[] data) {
        return IVector.of(data);
    }

    public static <T extends Number> IVector<T> vector(int value) {
        return IVector.of(new Integer[]{value});
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
    /**
     * 创建范围向量（指定类型和步长） / Create range vector (specified type and step)
     * <p>
     * 创建指定类型的范围向量，从start到end（不包含end），步长为step。此方法委托给
     * {@link IVector#range(int, int, int, Class)} 实现。 Creates a range vector of
     * specified type from start to end (exclusive) with specified step. This
     * method delegates to {@link IVector#range(int, int, int, Class)}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> range = Linalg.range(0, 10, 2, Double.class);  // [0, 2, 4, 6, 8]
     * }</pre>
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @param step 步长 / Step size
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 范围向量 / Range vector
     * @throws IllegalArgumentException 如果step为0或负数 / if step is 0 or negative
     * @see IVector#range(int, int, int, Class) 实际实现方法 / Actual implementation
     * method
     */
    public static <T extends Number> IVector<T> range(int start, int end, int step, Class<T> type) {
        return IVector.range(start, end, step, type);
    }

    /**
     * 创建范围向量（默认Double类型，指定步长） / Create range vector (default Double type,
     * specified step)
     * <p>
     * 创建范围向量，从start到end（不包含end），步长为step，默认Double类型。此方法委托给
     * {@link IVector#range(int, int, int)} 实现。 Creates a range vector from
     * start to end (exclusive) with specified step, default Double type. This
     * method delegates to {@link IVector#range(int, int, int)}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> range = Linalg.range(0, 10, 2);  // [0, 2, 4, 6, 8]
     * }</pre>
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @param step 步长 / Step size
     * @return 范围向量 / Range vector
     * @throws IllegalArgumentException 如果step为0或负数 / if step is 0 or negative
     * @see IVector#range(int, int, int) 实际实现方法 / Actual implementation method
     */
    public static IVector<Double> range(int start, int end, int step) {
        return IVector.range(start, end, step);
    }

    /**
     * 创建范围向量（指定类型，步长为1） / Create range vector (specified type, step size 1)
     * <p>
     * 创建指定类型的范围向量，从start到end（不包含end），步长为1。此方法委托给
     * {@link IVector#range(int, int, Class)} 实现。 Creates a range vector of
     * specified type from start to end (exclusive) with step size 1. This
     * method delegates to {@link IVector#range(int, int, Class)}.
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 范围向量 / Range vector
     * @see IVector#range(int, int, Class) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IVector<T> range(int start, int end, Class<T> type) {
        return IVector.range(start, end, type);
    }

    /**
     * 创建范围向量（默认Double类型，步长为1） / Create range vector (default Double type, step
     * size 1)
     * <p>
     * 创建范围向量，从start到end（不包含end），步长为1，默认Double类型。此方法委托给
     * {@link IVector#range(int, int)} 实现。 Creates a range vector from start to
     * end (exclusive) with step size 1, default Double type. This method
     * delegates to {@link IVector#range(int, int)}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> range = Linalg.range(0, 5);  // [0, 1, 2, 3, 4]
     * }</pre>
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @return 范围向量 / Range vector
     * @see IVector#range(int, int) 实际实现方法 / Actual implementation method
     */
    public static IVector<Double> range(int start, int end) {
        return IVector.range(start, end);
    }

    /**
     * 创建范围向量（指定类型，从0开始） / Create range vector (specified type, starting from 0)
     * <p>
     * 创建指定类型的范围向量，从0到end（不包含end），步长为1。此方法委托给 {@link IVector#range(int, Class)}
     * 实现。 Creates a range vector of specified type from 0 to end (exclusive)
     * with step size 1. This method delegates to
     * {@link IVector#range(int, Class)}.
     * </p>
     *
     * @param end 结束值（不包含） / End value (exclusive)
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 范围向量 / Range vector
     * @see IVector#range(int, Class) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IVector<T> range(int end, Class<T> type) {
        return IVector.range(end, type);
    }

    /**
     * 创建范围向量（默认Double类型，从0开始） / Create range vector (default Double type,
     * starting from 0)
     * <p>
     * 创建范围向量，从0到end（不包含end），步长为1，默认Double类型。此方法委托给 {@link IVector#range(int)}
     * 实现。 Creates a range vector from 0 to end (exclusive) with step size 1,
     * default Double type. This method delegates to {@link IVector#range(int)}.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> range = Linalg.range(5);  // [0, 1, 2, 3, 4]
     * }</pre>
     * </p>
     *
     * @param end 结束值（不包含） / End value (exclusive)
     * @return 范围向量 / Range vector
     * @see IVector#range(int) 实际实现方法 / Actual implementation method
     */
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

    public static <T extends Number> IVector<T> randn(int length, Class<T> type) {
        return IVector.randn(length, type);
    }

    public static IVector<Double> randn(int length) {
        return IVector.randn(length);
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

    /**
     * 创建下三角阵（下三角为指定值）
     *
     * @param m 矩阵大小
     * @param values 下三角元素值（按行优先顺序）
     * @return 下三角矩阵
     */
    public static IMatrix<Double> lowerTriMatrix(int m, double[][] values) {
        return IMatrix.lowerTriMatrix(m, values);
    }

    /**
     * 创建下三角阵
     *
     * @param m 矩阵大小
     * @return 下三角矩阵
     */
    public static IMatrix<Double> lowerTriMatrix(int m) {
        return IMatrix.lowerTriMatrix(m);
    }

    /**
     * 创建上三角阵
     *
     * @param m 矩阵大小
     * @return 上三角矩阵
     */
    public static IMatrix<Double> upperTriMatrix(int m) {
        return IMatrix.upperTriMatrix(m);
    }

    /**
     * 创建单位下三角阵（对角线为1，下三角为指定值）
     *
     * @param m 矩阵大小
     * @param values 下三角元素值（按行优先顺序）
     * @return 单位下三角矩阵
     */
    public static IMatrix<Double> unitLowerTriMatrix(int m, double[][] values) {
        return IMatrix.unitLowerTriMatrix(m, values);
    }

    /**
     * 创建上三角阵（上三角为指定值）
     *
     * @param m 矩阵大小
     * @param values 上三角元素值（按行优先顺序）
     * @return 上三角矩阵
     */
    public static IMatrix<Double> upperTriMatrix(int m, double[][] values) {
        return IMatrix.upperTriMatrix(m, values);
    }

    /**
     * 创建置换矩阵
     *
     * @param size 矩阵大小
     * @param pivot 置换向量
     * @return 置换矩阵
     */
    public static IMatrix<Double> permutationMatrix(int size, int[] pivot) {
        return IMatrix.permutationMatrix(size, pivot);
    }

    /**
     * 最小二乘法求解线性方程组 / Solving linear equations by the least square method
     *
     * @param A 系数矩阵 / Coefficient matrix
     * @param b 常数向量 / Constant vector
     * @return 最小二乘解及残差 / Least squares solution and residual
     */
    public static Tuple2<IVector<Double>, Double> lstsq(IMatrix<Double> A, IVector<Double> b) {
        return A.lstsq(b);
    }
    
    /**
     * 最小二乘法求解线性方程组 / Solving linear equations by the least square method
     *
     * @param A 系数矩阵 / Coefficient matrix
     * @param B 常数矩阵 / Constant vector
     * @return 最小二乘解及残差 / Least squares solution and residual
     */
    public static Tuple2<IMatrix<Double>, Double> lstsq(IMatrix<Double> A, IMatrix<Double> B) {
        return A.lstsq(B);
    }

    /**
     * 以传统方法求解线性方程组
     *
     * @param A
     * @param b
     * @return
     */
    public static IVector<Double> solve(IMatrix<Double> A, IVector<Double> b) {
        return A.solve(b);
    }

    /**
     * 根据行索引数组重新排列矩阵的行
     *
     * @param matrix 原矩阵
     * @param rowIndices 行索引数组
     * @return 重新排列后的矩阵
     */
    public static <T extends Number> IMatrix<T> permuteRows(IMatrix<T> matrix, int[] rowIndices) {
        return matrix.permuteRows(rowIndices);
    }

    /**
     * 根据列索引数组重新排列矩阵的列
     *
     * @param matrix 原矩阵
     * @param colIndices 列索引数组
     * @return 重新排列后的矩阵
     */
    public static <T extends Number> IMatrix<T> permuteColumns(IMatrix<T> matrix, int[] colIndices) {
        return matrix.permuteColumns(colIndices);
    }

    /**
     * 前向替换求解下三角线性系统
     *
     * @param L 下三角矩阵
     * @param B 右端矩阵
     * @return 解矩阵 X
     */
    public static <T extends Number> IMatrix<T> forwardSolve(IMatrix<T> L, IMatrix<T> B) {
        return L.forwardSolve(B);
    }

    /**
     * 后向替换求解上三角线性系统
     *
     * @param U 上三角矩阵
     * @param B 右端矩阵
     * @return 解矩阵 X
     */
    public static <T extends Number> IMatrix<T> backwardSolve(IMatrix<T> U, IMatrix<T> B) {
        return U.backwardSolve(B);
    }

    /**
     * Solve a system of linear equations A &times; X = B.
     *
     * @param A Coefficient matrix
     * @param B Right-hand side matrix
     * @return Solution matrix X
     * @throws IllegalArgumentException if matrices dimensions don't match
     * @throws ArithmeticException if the matrix is singular
     */
    public static <T extends Number> IMatrix<T> solveLinearSystem(IMatrix<T> A, IMatrix<T> B) {
        return A.solveLinearSystem(B);
    }

    /**
     * Solve a system of linear equations A &times; x = b.
     *
     * @param A Coefficient matrix
     * @param b Right-hand side vector
     * @return Solution vector x
     * @throws IllegalArgumentException if dimensions don't match
     * @throws ArithmeticException if the matrix is singular
     */
    public static <T extends Number> IVector<T> solveLinearSystem(IMatrix<T> A, IVector<T> b) {
        return A.solveLinearSystem(b);
    }

    /**
     * Create a bidiagonal matrix with specified diagonal and superdiagonal values.
     *
     * @param n Matrix size (n x n)
     * @param diagonal Diagonal values
     * @param superDiagonal Superdiagonal values
     * @return Bidiagonal matrix
     */
    public static IMatrix<Double> bidiagonalMatrix(int n, double[] diagonal, double[] superDiagonal) {
        double[][] data = new double[n][n];
        
        // Fill diagonal
        for (int i = 0; i < Math.min(n, diagonal.length); i++) {
            data[i][i] = diagonal[i];
        }
        
        // Fill superdiagonal
        for (int i = 0; i < Math.min(n - 1, superDiagonal.length); i++) {
            data[i][i + 1] = superDiagonal[i];
        }
        
        return IMatrix.of(data);
    }

    /**
     * Create a tridiagonal matrix with specified diagonal, subdiagonal, and superdiagonal values.
     *
     * @param n Matrix size (n x n)
     * @param subDiagonal Subdiagonal values
     * @param diagonal Diagonal values
     * @param superDiagonal Superdiagonal values
     * @return Tridiagonal matrix
     */
    public static IMatrix<Double> tridiagonalMatrix(int n, double[] subDiagonal, double[] diagonal, double[] superDiagonal) {
        double[][] data = new double[n][n];
        
        // Fill subdiagonal
        for (int i = 1; i < Math.min(n, subDiagonal.length + 1); i++) {
            data[i][i - 1] = subDiagonal[i - 1];
        }
        
        // Fill diagonal
        for (int i = 0; i < Math.min(n, diagonal.length); i++) {
            data[i][i] = diagonal[i];
        }
        
        // Fill superdiagonal
        for (int i = 0; i < Math.min(n - 1, superDiagonal.length); i++) {
            data[i][i + 1] = superDiagonal[i];
        }
        
        return IMatrix.of(data);
    }

    /**
     * Create a block diagonal matrix from a list of matrices.
     *
     * @param matrices Array of matrices to place on the diagonal
     * @return Block diagonal matrix
     */
    public static IMatrix<Double> blockDiagonalMatrix(IMatrix<Double>[] matrices) {
        // Calculate total dimensions
        int totalRows = 0;
        int totalCols = 0;
        for (IMatrix<Double> matrix : matrices) {
            totalRows += matrix.rows();
            totalCols += matrix.cols();
        }
        
        // Create result matrix
        IMatrix<Double> result = IMatrix.zeros(totalRows, totalCols);
        
        // Fill diagonal blocks
        int rowOffset = 0;
        int colOffset = 0;
        for (IMatrix<Double> matrix : matrices) {
            int rows = matrix.rows();
            int cols = matrix.cols();
            
            // Copy matrix to block diagonal position
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.put(rowOffset + i, colOffset + j, matrix.get(i, j));
                }
            }
            
            rowOffset += rows;
            colOffset += cols;
        }
        
        return result;
    }

}
