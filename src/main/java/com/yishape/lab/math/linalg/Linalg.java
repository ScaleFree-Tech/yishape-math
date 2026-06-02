package com.yishape.lab.math.linalg;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.FloatVectorComputer;
import com.yishape.lab.math.compute.IFloatVectorComputer;
import com.yishape.lab.math.compute.gpu.GpuBatchNorm;
import com.yishape.lab.math.compute.gpu.GpuLayerNorm;
import com.yishape.lab.math.core.Complex;
import com.yishape.lab.math.linalg.complex.IComplexMatrix;
import com.yishape.lab.math.linalg.sparse.ISparseLinearSolver;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.linalg.sparse.ISparsePreconditioner;
import com.yishape.lab.math.linalg.sparse.ISpecialSparseMatrix;
import com.yishape.lab.math.linalg.sparse.SparseILUPreconditioner;
import com.yishape.lab.math.linalg.sparse.impl.SparseConjugateGradientSolver;
import com.yishape.lab.math.linalg.sparse.impl.SparseBICGSTABSolver;
import com.yishape.lab.math.linalg.sparse.impl.SparseGMRESSolver;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
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
    public static IDoubleMatrix matrix(double[][] data) {
        return IDoubleMatrix.of(data);
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
    public static IFloatMatrix matrix(float[][] data) {
        return IFloatMatrix.of(data);
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
    public static IDoubleMatrix matrix(Double[][] data) {
        return IDoubleMatrix.of(data);
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
    public static IFloatMatrix matrix(Float[][] data) {
        return IFloatMatrix.of(data);
    }

    public static IDoubleMatrix matrixFromDoubleList(List<double[]> data) {
        return IMatrix.ofDoubleList(data);
    }

    public static IFloatMatrix matrixFromFloatList(List<float[]> data) {
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
    public static IDoubleMatrix rand(int rows, int cols) {
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
    public static IDoubleMatrix rand(int rows, int cols, long seed) {
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

    public static IDoubleMatrix randn(int rows, int cols) {
        return IMatrix.randn(rows, cols);
    }

    public static <T extends Number> IMatrix<T> randn(int rows, int cols, Class<T> type) {
        return IMatrix.randn(rows, cols, type);
    }

    public static IDoubleMatrix randn(int rows, int cols, double mean, double std) {
        return IMatrix.randn(rows, cols, mean, std);
    }

    public static <T extends Number> IMatrix<T> randn(int rows, int cols, T mean, T std, Class<T> type) {
        return IMatrix.randn(rows, cols, mean, std, type);
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
    public static IDoubleMatrix ones(int rows, int cols) {
        return IDoubleMatrix.ones(rows, cols);
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
    public static IDoubleMatrix zeros(int rows, int cols) {
        return IDoubleMatrix.zeros(rows, cols);
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
    public static IDoubleMatrix eye(int size) {
        return IDoubleMatrix.eye(size);
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

    public static  IFloatMatrix diag(float[] diagonal) {
        return IFloatMatrix.diag(diagonal);
    }

    public static  IDoubleMatrix diag(double[] diagonal) {
        return IDoubleMatrix.diag(diagonal);
    }

    public static <T extends Number> IMatrix<T> fromArray(T[] data, int rows, int cols) {
        return IMatrix.fromArray(data, rows, cols);
    }

    public static <T extends Number> IMatrix<T> fromArray(T[] data, int rows, int cols, Class<T> type) {
        return IMatrix.fromArray(data, rows, cols, type);
    }

    public static IDoubleMatrix fromArray(double[] data, int rows, int cols) {
        return IDoubleMatrix.fromArray(data, rows, cols);
    }

    public static IFloatMatrix fromArray(float[] data, int rows, int cols) {
        return IFloatMatrix.fromArray(data, rows, cols);
    }

    /**
     * Kronecker 积，等价于 {@code a.kron(b)} 与 NumPy {@code numpy.kron(a, b)}。
     */
    public static <T extends Number> IMatrix<T> kron(IMatrix<T> a, IMatrix<T> b) {
        return a.kron(b);
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
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(double[]) 实际实现方法 / Actual implementation method
     */
    public static IDoubleVector vector(double...data) {
        return IDoubleVector.of(data);
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
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(Double[]) 实际实现方法 / Actual implementation method
     */
    public static IDoubleVector vector(Double[] data) {
        return IDoubleVector.of(data);
    }
    
    
    /**
     * 创建向量
     * @param data
     * @return 
     */
    public static IDoubleVector vector(List<Double> data) {
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
     * @return 新的向量实例（IFloatVector）/ New vector instance (IFloatVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(float[]) 实际实现方法 / Actual implementation method
     */
    public static IFloatVector vector(float[] data) {
        return IFloatVector.of(data);
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
     * @return 新的向量实例（IFloatVector）/ New vector instance (IFloatVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see IVector#of(Float[]) 实际实现方法 / Actual implementation method
     */
    public static IFloatVector vector(Float[] data) {
        return IFloatVector.of(data);
    }
    
   /**
     * 从Float List创建向量，因为和Double List同参数签名，必须换个名字
     * @param data Float List
     * @return 
     */
    public static IFloatVector vectorFromFloatList(List<Float> data) {
        return IVector.ofFloatList(data);
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

    public static IDoubleVector vector(double value) {
        return IDoubleVector.of(value);
    }

    public static IDoubleVector vector(double value1, double value2) {
        return IDoubleVector.of(value1, value2);
    }

    public static IFloatVector vector(float value) {
        return IFloatVector.of(value);
    }

    public static IFloatVector vector(float value1, float value2) {
        return IFloatVector.of(value1, value2);
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
    public static IDoubleVector range(int start, int end, int step) {
        return IDoubleVector.range(start, end, step);
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
    public static IDoubleVector range(int start, int end) {
        return IDoubleVector.range(start, end);
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
    public static IDoubleVector range(int end) {
        return IDoubleVector.range(end);
    }

    // Special vector creation methods
    public static <T extends Number> IVector<T> ones(int len, Class<T> type) {
        return IVector.ones(len, type);
    }

    public static IDoubleVector ones(int len) {
        return IDoubleVector.ones(len);
    }

    public static <T extends Number> IVector<T> zeros(int len, Class<T> type) {
        return IVector.zeros(len, type);
    }

    public static IDoubleVector zeros(int len) {
        return IDoubleVector.zeros(len);
    }

    // Random vector generation methods
    public static <T extends Number> IVector<T> rand(int length, Class<T> type) {
        return IVector.rand(length, type);
    }

    public static IDoubleVector rand(int length) {
        return IDoubleVector.rand(length);
    }

    public static <T extends Number> IVector<T> randn(int length, T mean, T std, Class<T> type) {
        return IVector.randn(length, mean, std, type);
    }

    public static IDoubleVector randn(int length, double mean, double std) {
        return IDoubleVector.randn(length, mean, std);
    }

    public static <T extends Number> IVector<T> randn(int length, Class<T> type) {
        return IVector.randn(length, type);
    }

    public static IDoubleVector randn(int length) {
        return IDoubleVector.randn(length);
    }

    // Linear space generation methods
    public static <T extends Number> IVector<T> linspace(T start, T stop, int num, Class<T> type) {
        return IVector.linspace(start, stop, num, type);
    }

    public static IDoubleVector linspace(double start, double stop, int num) {
        return IDoubleVector.linspace(start, stop, num);
    }

    public static <T extends Number> IVector<T> logspace(T start, T stop, int num, Class<T> type) {
        return IVector.logspace(start, stop, num, type);
    }

    public static IDoubleVector logspace(double start, double stop, int num) {
        return IDoubleVector.logspace(start, stop, num);
    }

    /**
     * 创建下三角阵（下三角为指定值）
     *
     * @param m 矩阵大小
     * @param values 下三角元素值（按行优先顺序）
     * @return 下三角矩阵
     */
    public static IDoubleMatrix lowerTriMatrix(int m, double[][] values) {
        return IMatrix.lowerTriMatrix(m, values);
    }

    /**
     * 创建下三角阵
     *
     * @param m 矩阵大小
     * @return 下三角矩阵
     */
    public static IDoubleMatrix lowerTriMatrix(int m) {
        return IMatrix.lowerTriMatrix(m);
    }

    /**
     * 创建上三角阵
     *
     * @param m 矩阵大小
     * @return 上三角矩阵
     */
    public static IDoubleMatrix upperTriMatrix(int m) {
        return IMatrix.upperTriMatrix(m);
    }

    /**
     * 创建单位下三角阵（对角线为1，下三角为指定值）
     *
     * @param m 矩阵大小
     * @param values 下三角元素值（按行优先顺序）
     * @return 单位下三角矩阵
     */
    public static IDoubleMatrix unitLowerTriMatrix(int m, double[][] values) {
        return IMatrix.unitLowerTriMatrix(m, values);
    }

    /**
     * 创建上三角阵（上三角为指定值）
     *
     * @param m 矩阵大小
     * @param values 上三角元素值（按行优先顺序）
     * @return 上三角矩阵
     */
    public static IDoubleMatrix upperTriMatrix(int m, double[][] values) {
        return IMatrix.upperTriMatrix(m, values);
    }

    /**
     * 创建置换矩阵
     *
     * @param size 矩阵大小
     * @param pivot 置换向量
     * @return 置换矩阵
     */
    public static IDoubleMatrix permutationMatrix(int size, int[] pivot) {
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
    public static IDoubleMatrix bidiagonalMatrix(int n, double[] diagonal, double[] superDiagonal) {
        double[][] data = new double[n][n];
        
        // Fill diagonal
        for (int i = 0; i < Math.min(n, diagonal.length); i++) {
            data[i][i] = diagonal[i];
        }
        
        // Fill superdiagonal
        for (int i = 0; i < Math.min(n - 1, superDiagonal.length); i++) {
            data[i][i + 1] = superDiagonal[i];
        }
        
        return IDoubleMatrix.of(data);
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
    public static IDoubleMatrix tridiagonalMatrix(int n, double[] subDiagonal, double[] diagonal, double[] superDiagonal) {
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
        
        return IDoubleMatrix.of(data);
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

    // ==================== Toeplitz 矩阵 / Toeplitz Matrix ====================

    /** 创建对称 Toeplitz 矩阵 (column 为第一列) / Create symmetric Toeplitz matrix */
    public static IMatrix<Double> toeplitz(double[] column) {
        int n = column.length;
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = column[Math.abs(i - j)];
            }
        }
        return IMatrix.of(data);
    }

    /** 创建非对称 Toeplitz 矩阵 / Create non-symmetric Toeplitz matrix */
    public static IMatrix<Double> toeplitz(double[] column, double[] row) {
        int n = column.length;
        int m = row.length;
        if (Math.abs(column[0] - row[0]) > 1e-15) {
            throw new IllegalArgumentException("column[0] must equal row[0] for Toeplitz matrix");
        }
        double[][] data = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                data[i][j] = (i >= j) ? column[i - j] : row[j - i];
            }
        }
        return IMatrix.of(data);
    }

    /** 创建对称 Toeplitz 矩阵 / Create symmetric Toeplitz matrix from vector */
    public static IMatrix<Double> toeplitz(IVector<Double> column) {
        return toeplitz(column.toDoubleArray());
    }

    /** 创建非对称 Toeplitz 矩阵 / Create non-symmetric Toeplitz matrix from vectors */
    public static IMatrix<Double> toeplitz(IVector<Double> column, IVector<Double> row) {
        return toeplitz(column.toDoubleArray(), row.toDoubleArray());
    }

    /**
     * 一维线性插值（对齐 {@code numpy.interp}：区间外取端点）/ 1D linear interpolation
     */
    public static IVector<Double> interp(IVector<Double> xq, IVector<Double> xp, IVector<Double> fp) {
        Objects.requireNonNull(xq, "xq");
        Objects.requireNonNull(xp, "xp");
        Objects.requireNonNull(fp, "fp");
        int n = xp.length();
        if (fp.length() != n) {
            throw new IllegalArgumentException("xp 与 fp 长度须相同 / xp and fp must have same length");
        }
        if (n < 2) {
            throw new IllegalArgumentException("xp 至少长度 2 / xp must have length >= 2");
        }
        for (int i = 1; i < n; i++) {
            if (xp.get(i) < xp.get(i - 1)) {
                throw new IllegalArgumentException("xp 须非降序 / xp must be non-decreasing");
            }
        }
        int nq = xq.length();
        double[] out = new double[nq];
        for (int q = 0; q < nq; q++) {
            double x = xq.get(q);
            if (x <= xp.get(0)) {
                out[q] = fp.get(0);
                continue;
            }
            if (x >= xp.get(n - 1)) {
                out[q] = fp.get(n - 1);
                continue;
            }
            int lo = 0;
            int hi = n - 1;
            while (lo < hi - 1) {
                int mid = (lo + hi) >>> 1;
                if (xp.get(mid) <= x) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            double x0 = xp.get(lo);
            double x1 = xp.get(lo + 1);
            double t = (x - x0) / (x1 - x0);
            out[q] = (1.0 - t) * fp.get(lo) + t * fp.get(lo + 1);
        }
        return vector(out);
    }

    /**
     * 二维网格（默认 {@code indexing='xy'}，与 NumPy {@code meshgrid} 一致）/ Coordinate grids
     */
    public static Tuple2<IMatrix<Double>, IMatrix<Double>> meshgrid(IVector<Double> x, IVector<Double> y) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        int nx = x.length();
        int ny = y.length();
        double[][] xd = new double[ny][nx];
        double[][] yd = new double[ny][nx];
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                xd[i][j] = x.get(j);
                yd[i][j] = y.get(i);
            }
        }
        return new Tuple2<>(matrix(xd), matrix(yd));
    }

    /**
     * 沿轴堆叠矩阵（axis=0 为垂直拼接，axis=1 为水平拼接）/ Stack like {@code numpy.stack} for 2D
     */
    @SafeVarargs
    public static <T extends Number> IMatrix<T> stack(int axis, IMatrix<T>... matrices) {
        if (matrices == null || matrices.length == 0) {
            throw new IllegalArgumentException("至少一个矩阵 / At least one matrix required");
        }
        if (axis == 0) {
            IMatrix<T> acc = matrices[0];
            for (int i = 1; i < matrices.length; i++) {
                acc = acc.vstack(matrices[i]);
            }
            return acc;
        }
        if (axis == 1) {
            IMatrix<T> acc = matrices[0];
            for (int i = 1; i < matrices.length; i++) {
                acc = acc.hstack(matrices[i]);
            }
            return acc;
        }
        throw new IllegalArgumentException("axis 仅支持 0 或 1 / axis must be 0 or 1");
    }

    /**
     * 链式矩阵乘（对齐 {@code numpy.linalg.multi_dot} 从左到右顺序）/ Matrix chain product
     */
    @SafeVarargs
    public static IMatrix<Double> multiDot(IMatrix<Double>... matrices) {
        if (matrices == null || matrices.length == 0) {
            throw new IllegalArgumentException("至少一个矩阵 / At least one matrix required");
        }
        IMatrix<Double> r = matrices[0];
        for (int i = 1; i < matrices.length; i++) {
            r = r.mmul(matrices[i]);
        }
        return r;
    }

    /**
     * 是否已加载 yishape-math-hpc 原生库（{@code yishape_math_rust}）。
     */
    public static boolean isHpcNativeRuntimeLoaded() {
        return com.yishape.lab.math.compute.hpc.HpcOptionalRuntime.isNativeRuntimeAvailable();
    }

    // ==================== 无分配浮点数距离计算 / Zero-allocation Float Distance ====================

    private static final IFloatVectorComputer FC = new FloatVectorComputer();

    /**
     * 两个 float[] 的内积（使用 SIMD/HPC 加速，不产生 IVector 包装开销）。
     */
    public static float dot(float[] a, float[] b) {
        return FC.binaryReduceOperate(a, b, IFloatVectorComputer.BinaryReduceOperation.DOT);
    }

    /**
     * 两个 float[] 的平方欧几里得距离（使用 SIMD/HPC 加速，不产生 IVector 包装开销）。
     */
    public static float squaredDistance(float[] a, float[] b) {
        float aa = dot(a, a);
        float bb = dot(b, b);
        float ab = dot(a, b);
        return aa + bb - 2.0f * ab;
    }

    // ==================== 复数矩阵工厂方法 / Complex Matrix Factory Methods ====================

    /**
     * 从实部和虚部创建复数矩阵 / Create complex matrix from real and imaginary parts
     */
    public static IComplexMatrix complexMatrix(double[][] real, double[][] imag) {
        return IComplexMatrix.fromRealImag(real, imag);
    }

    /**
     * 从 Complex 数组创建复数矩阵 / Create complex matrix from Complex array
     */
    public static IComplexMatrix complexMatrix(Complex[][] data) {
        return IComplexMatrix.fromComplex(data);
    }

    /**
     * 从极坐标创建复数矩阵 / Create complex matrix from polar coordinates
     */
    public static IComplexMatrix complexMatrixFromPolar(double[][] magnitude, double[][] phase) {
        return IComplexMatrix.fromPolar(magnitude, phase);
    }

    /**
     * 创建全零复数矩阵 / Create complex zero matrix
     */
    public static IComplexMatrix complexZeros(int rows, int cols) {
        double[][] real = new double[rows][cols];
        double[][] imag = new double[rows][cols];
        return IComplexMatrix.fromRealImag(real, imag);
    }

    /**
     * 创建全一复数矩阵 / Create complex ones matrix
     */
    public static IComplexMatrix complexOnes(int rows, int cols) {
        double[][] real = new double[rows][cols];
        double[][] imag = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                real[i][j] = 1.0;
            }
        }
        return IComplexMatrix.fromRealImag(real, imag);
    }

    /**
     * 创建复数单位矩阵 / Create complex identity matrix
     */
    public static IComplexMatrix complexEye(int size) {
        double[][] real = new double[size][size];
        double[][] imag = new double[size][size];
        for (int i = 0; i < size; i++) {
            real[i][i] = 1.0;
        }
        return IComplexMatrix.fromRealImag(real, imag);
    }

    /**
     * 创建复数对角矩阵 / Create complex diagonal matrix
     */
    public static IComplexMatrix complexDiag(Complex[] diagonal) {
        int n = diagonal.length;
        double[][] real = new double[n][n];
        double[][] imag = new double[n][n];
        for (int i = 0; i < n; i++) {
            real[i][i] = diagonal[i].real;
            imag[i][i] = diagonal[i].imag;
        }
        return IComplexMatrix.fromRealImag(real, imag);
    }

    // ==================== 复数向量工厂方法 / Complex Vector Factory Methods ====================

    /**
     * 从实部和虚部创建复数向量 / Create complex vector from real and imaginary parts
     */
    public static IComplexMatrix.IComplexVector complexVector(double[] real, double[] imag) {
        return IComplexMatrix.IComplexVector.fromRealImag(real, imag);
    }

    /**
     * 从 Complex 数组创建复数向量 / Create complex vector from Complex array
     */
    public static IComplexMatrix.IComplexVector complexVector(Complex[] data) {
        return IComplexMatrix.IComplexVector.fromComplex(data);
    }

    /**
     * 从极坐标创建复数向量 / Create complex vector from polar coordinates
     */
    public static IComplexMatrix.IComplexVector complexVectorFromPolar(double[] magnitude, double[] phase) {
        return IComplexMatrix.IComplexVector.fromPolar(magnitude, phase);
    }

    /**
     * 创建全零复数向量 / Create complex zero vector
     */
    public static IComplexMatrix.IComplexVector complexZeros(int length) {
        return IComplexMatrix.IComplexVector.fromRealImag(new double[length], new double[length]);
    }

    /**
     * 创建全一复数向量 / Create complex ones vector
     */
    public static IComplexMatrix.IComplexVector complexOnes(int length) {
        double[] real = new double[length];
        Arrays.fill(real, 1.0);
        return IComplexMatrix.IComplexVector.fromRealImag(real, new double[length]);
    }

    // ==================== 稀疏矩阵工厂方法 / Sparse Matrix Factory Methods ====================

    /**
     * 从稠密数组创建稀疏矩阵（自动检测非零元素）/ Create sparse matrix from dense array
     */
    public static ISparseMatrix sparse(double[][] data) {
        return ISparseMatrix.fromDense(data);
    }

    /**
     * 从稠密数组创建稀疏矩阵（指定阈值）/ Create sparse matrix from dense array with tolerance
     */
    public static ISparseMatrix sparse(double[][] data, double tolerance) {
        return ISparseMatrix.fromDense(data, tolerance);
    }

    /**
     * 从 COO 格式创建稀疏矩阵 / Create sparse matrix from COO format
     */
    public static ISparseMatrix sparseFromCOO(int[] rowIdx, int[] colIdx, double[] values, int rows, int cols) {
        return ISparseMatrix.fromCOO(rowIdx, colIdx, values, rows, cols);
    }

    /**
     * 从 CSR 格式创建稀疏矩阵 / Create sparse matrix from CSR format
     */
    public static ISparseMatrix sparseFromCSR(int[] rowPtr, int[] colInd, double[] values, int rows, int cols) {
        return ISparseMatrix.fromCSR(rowPtr, colInd, values, rows, cols);
    }

    /**
     * 从 CSC 格式创建稀疏矩阵 / Create sparse matrix from CSC format
     */
    public static ISparseMatrix sparseFromCSC(int[] rowInd, int[] colPtr, double[] values, int rows, int cols) {
        return ISparseMatrix.fromCSC(rowInd, colPtr, values, rows, cols);
    }

    /**
     * 创建稀疏单位矩阵 / Create sparse identity matrix
     */
    public static ISparseMatrix sparseEye(int size) {
        return ISparseMatrix.eye(size);
    }

    /**
     * 创建稀疏单位矩阵（指定尺寸）/ Create sparse identity matrix with specified dimensions
     */
    public static ISparseMatrix sparseEye(int rows, int cols) {
        return ISparseMatrix.eye(rows, cols);
    }

    /**
     * 创建稀疏对角矩阵 / Create sparse diagonal matrix
     */
    public static ISparseMatrix sparseDiag(double[] values) {
        return ISparseMatrix.diag(values);
    }

    /**
     * 创建稀疏对角矩阵（指定尺寸）/ Create sparse diagonal matrix with specified size
     */
    public static ISparseMatrix sparseDiag(double[] values, int size) {
        return ISparseMatrix.diag(values, size);
    }

    // ==================== 特殊稀疏矩阵工厂方法 / Special Sparse Matrix Factory Methods ====================

    /**
     * 创建对角稀疏矩阵 / Create diagonal sparse matrix
     */
    public static ISparseMatrix diagonalSparse(double[] diagonal) {
        return new ISpecialSparseMatrix.DiagonalSparseMatrix(diagonal);
    }

    /**
     * 创建对角稀疏矩阵（指定尺寸）/ Create diagonal sparse matrix with specified dimensions
     */
    public static ISparseMatrix diagonalSparse(double[] diagonal, int rows, int cols) {
        return new ISpecialSparseMatrix.DiagonalSparseMatrix(diagonal, rows, cols);
    }

    /**
     * 创建三对角稀疏矩阵 / Create tridiagonal sparse matrix
     */
    public static ISparseMatrix tridiagonalSparse(double[] lower, double[] main, double[] upper) {
        return new ISpecialSparseMatrix.TridiagonalSparseMatrix(lower, main, upper);
    }

    /**
     * 创建稀疏单位矩阵（特殊实现）/ Create sparse identity matrix (special implementation)
     */
    public static ISparseMatrix identitySparse(int size) {
        return new ISpecialSparseMatrix.IdentitySparseMatrix(size);
    }

    /**
     * 创建稀疏零矩阵 / Create sparse zero matrix
     */
    public static ISparseMatrix zeroSparse(int rows, int cols) {
        return new ISpecialSparseMatrix.ZeroSparseMatrix(rows, cols);
    }

    // ==================== 稀疏求解器工厂方法 / Sparse Solver Factory Methods ====================

    /** 创建 ILU(0) 预条件子 / Create ILU(0) preconditioner */
    public static ISparsePreconditioner sparseILU() {
        return new SparseILUPreconditioner();
    }

    /** 创建 ILUT 预条件子 / Create ILUT preconditioner with drop tolerance */
    public static ISparsePreconditioner sparseILU(double dropTolerance) {
        return new SparseILUPreconditioner(dropTolerance);
    }

    /** 创建稀疏 CG 求解器 / Create sparse Conjugate Gradient solver */
    public static ISparseLinearSolver sparseSolverCG(double tol, int maxIter, ISparsePreconditioner precond) {
        return new SparseConjugateGradientSolver(tol, maxIter, precond);
    }

    /** 创建稀疏 BiCGStab 求解器 / Create sparse BiCGStab solver */
    public static ISparseLinearSolver sparseSolverBiCGStab(double tol, int maxIter, ISparsePreconditioner precond) {
        return new SparseBICGSTABSolver(tol, maxIter, precond);
    }

    /** 创建稀疏 GMRES 求解器 / Create sparse GMRES solver */
    public static ISparseLinearSolver sparseSolverGMRES(double tol, int maxIter, int restart, ISparsePreconditioner precond) {
        return new SparseGMRESSolver(tol, maxIter, restart, precond);
    }

    // ==================== 张量工厂方法 / Tensor Factory Methods ====================

    /**
     * 从数据创建张量 / Create tensor from data
     */
    public static ITensor tensor(double[] data, int... shape) {
        return ITensor.tensor(data, shape);
    }

    /**
     * 从二维数据创建张量 / Create tensor from 2D data
     */
    public static ITensor tensor(double[][] data) {
        return ITensor.tensor(data);
    }

    /**
     * 从三维数据创建张量 / Create tensor from 3D data
     */
    public static ITensor tensor(double[][][] data) {
        return ITensor.tensor(data);
    }

    /**
     * 创建全一维张量 / Create ones tensor
     */
    public static ITensor ones(int... shape) {
        return ITensor.ones(shape);
    }

    /**
     * 创建全零维张量 / Create zeros tensor
     */
    public static ITensor zeros(int... shape) {
        return ITensor.zeros(shape);
    }

    /**
     * 创建随机张量（均匀分布）/ Create random tensor (uniform distribution)
     */
    public static ITensor rand(int... shape) {
        return ITensor.rand(shape);
    }

    /**
     * 创建随机张量（正态分布）/ Create random tensor (normal distribution)
     */
    public static ITensor randn(int... shape) {
        return ITensor.randn(shape);
    }

    /**
     * 创建多维单位矩阵 / Create multi-dimensional identity tensor
     */
    public static ITensor eye(int n, int... extraDims) {
        return ITensor.eye(n, extraDims);
    }

    // ==================== IDoubleTensor 额外工厂方法 ====================

    /** 标量张量 */
    public static IDoubleTensor scalar(double value) {
        return ITensor.scalar(value);
    }

    /** 空张量 */
    public static IDoubleTensor empty(int... shape) {
        return ITensor.empty(shape);
    }

    /** 填充张量 */
    public static IDoubleTensor full(int[] shape, double value) {
        return ITensor.full(shape, value);
    }

    /** 等差数列 */
    public static IDoubleTensor arange(double start, double end, double step) {
        return ITensor.arange(start, end, step);
    }

    /** 从 IDoubleVector + shape 重建 */
    public static IDoubleTensor tensor(IDoubleVector vec, int... shape) {
        return ITensor.fromVector(vec, shape);
    }

    /** From strided data (low-level view construction) */
    public static IDoubleTensor fromStrided(double[] data, int offset,
                                             int[] shape, int[] strides) {
        return ITensor.fromStrided(data, offset, shape, strides);
    }


    // ==================== IDoubleTensor 工具方法 ====================

    /** 创建模型参数张量 (randn * std) */
    public static IDoubleTensor randn(double std, int... shape) {
        IDoubleTensor t = ITensor.randn(shape);
        return t.mul(std);
    }

    /** 拼接多个张量 */
    public static IDoubleTensor cat(int dim, IDoubleTensor... tensors) {
        if (tensors.length == 0) throw new IllegalArgumentException("cat requires at least 1 tensor");
        return tensors[0].cat(dim, java.util.Arrays.copyOfRange(tensors, 1, tensors.length));
    }

    /** 堆叠多个张量 */
    public static IDoubleTensor stack(int dim, IDoubleTensor... tensors) {
        if (tensors.length == 0) throw new IllegalArgumentException("stack requires at least 1 tensor");
        return tensors[0].stack(dim, java.util.Arrays.copyOfRange(tensors, 1, tensors.length));
    }
}
