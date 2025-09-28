package com.reremouse.lab.math.linalg;

import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * 泛型矩阵操作接口 / Generic Matrix Operations Interface
 * <p>
 * 本接口定义了支持泛型数值类型的矩阵操作，包括基本的数学运算、矩阵变换、数据访问等功能。 提供了创建矩阵的静态工厂方法和各种矩阵运算的抽象方法定义。
 * </p>
 * <p>
 * This interface defines matrix operations supporting generic numeric types,
 * including basic mathematical operations, matrix transformations, data access
 * and other functionalities. Provides static factory methods for creating
 * matrices and abstract method definitions for various matrix operations.
 * </p>
 *
 * <h3>静态工厂方法架构 / Static Factory Method Architecture:</h3>
 * <p>
 * 本接口采用直接委托模式，静态工厂方法直接委托给具体实现类： This interface uses direct delegation pattern,
 * static factory methods directly delegate to concrete implementations:
 * </p>
 * <ul>
 * <li>**用户调用** / **User calls**: {@code IMatrix.of(data)} 或
 * {@code Linalg.matrix(data)}</li>
 * <li>**直接委托** / **Direct delegation**:
 * {@code IMatrix → IFloatMatrix/IDoubleMatrix}</li>
 * <li>**Linalg委托** / **Linalg delegation**:
 * {@code Linalg → IMatrix → IFloatMatrix/IDoubleMatrix}</li>
 * <li>**类型推断** / **Type inference**: 根据输入数据类型自动选择合适的实现 / Automatically selects
 * appropriate implementation based on input data type</li>
 * <li>**支持类型** / **Supported types**: Float, Double（默认）/ Float, Double
 * (default)</li>
 * </ul>
 *
 * <h3>使用示例 / Usage Examples:</h3>
 * <pre>
 * {@code
 * // 通过IMatrix创建 / Create via IMatrix
 * IMatrix<Double> m1 = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
 * IMatrix<Float> m2 = IMatrix.of(new float[][]{{1f, 2f}, {3f, 4f}});
 *
 * // 通过Linalg创建（推荐）/ Create via Linalg (recommended)
 * IMatrix<Double> m3 = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
 * IMatrix<Float> m4 = Linalg.matrix(new float[][]{{1f, 2f}, {3f, 4f}});
 *
 * // 特殊矩阵 / Special matrices
 * IMatrix<Double> ones = IMatrix.ones(3, 3);     // 3x3 全1矩阵 / 3x3 ones matrix
 * IMatrix<Double> zeros = IMatrix.zeros(2, 4);   // 2x4 零矩阵 / 2x4 zeros matrix
 * IMatrix<Double> eye = IMatrix.eye(4);           // 4x4 单位矩阵 / 4x4 identity matrix
 * IMatrix<Double> random = IMatrix.rand(3, 3);   // 3x3 随机矩阵 / 3x3 random matrix
 * }
 * </pre>
 *
 * @param <T> 数值类型，必须继承自Number / Numeric type, must extend Number
 * @author lteb2
 * @version 1.0
 * @since 1.0
 * @see Linalg 推荐使用Linalg类的工厂方法 / Recommended to use Linalg class factory
 * methods
 * @see IFloatMatrix Float类型矩阵的具体实现 / Concrete implementation for Float type
 * matrices
 * @see IDoubleMatrix Double类型矩阵的具体实现 / Concrete implementation for Double type
 * matrices
 */
public interface IMatrix<T extends Number> {

    // ========== 泛型工厂方法支持 / Generic Factory Method Support ==========
    // 注意：这些方法直接委托给具体实现类（IFloatMatrix/IDoubleMatrix）。推荐使用 Linalg 类的工厂方法。
    // Note: These methods directly delegate to concrete implementations (IFloatMatrix/IDoubleMatrix). 
    // It's recommended to use factory methods from the Linalg class.
    /**
     * 从二维double数组创建矩阵 / Create matrix from 2D double array
     * <p>
     * 将二维double数组转换为矩阵对象 Converts a 2D double array to a matrix object
     * </p>
     * 
     * @param data 二维double数组 / 2D double array
     * @param <T> 数值类型 / Numeric type
     * @return 矩阵对象 / Matrix object
     * @throws IllegalArgumentException 如果数组为null或空 / if array is null or empty
     * 
     * @example
     * <pre>
     * double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
     * IMatrix&lt;Double&gt; matrix = IMatrix.of(data);
     * </pre>
     */
    public static <T extends Number> IMatrix<T> of(double[][] data) {
        return (IMatrix<T>) IDoubleMatrix.of(data);
    }

    /**
     * 从二维Double包装类数组创建矩阵 / Create matrix from 2D Double wrapper array
     * <p>
     * 将二维Double包装类数组转换为矩阵对象 Converts a 2D Double wrapper array to a matrix object
     * </p>
     * 
     * @param data 二维Double包装类数组 / 2D Double wrapper array
     * @param <T> 数值类型 / Numeric type
     * @return 矩阵对象 / Matrix object
     * @throws IllegalArgumentException 如果数组为null或空 / if array is null or empty
     * 
     * @example
     * <pre>
     * Double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
     * IMatrix&lt;Double&gt; matrix = IMatrix.of(data);
     * </pre>
     */
    public static <T extends Number> IMatrix<T> of(Double[][] data) {
        return (IMatrix<T>) IDoubleMatrix.of(data);
    }

    /**
     * 从二维float数组创建矩阵 / Create matrix from 2D float array
     * <p>
     * 将二维float数组转换为矩阵对象 Converts a 2D float array to a matrix object
     * </p>
     * 
     * @param data 二维float数组 / 2D float array
     * @param <T> 数值类型 / Numeric type
     * @return 矩阵对象 / Matrix object
     * @throws IllegalArgumentException 如果数组为null或空 / if array is null or empty
     * 
     * @example
     * <pre>
     * float[][] data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
     * IMatrix&lt;Float&gt; matrix = IMatrix.of(data);
     * </pre>
     */
    public static <T extends Number> IMatrix<T> of(float[][] data) {
        return (IMatrix<T>) IFloatMatrix.of(data);
    }

    /**
     * 从二维Float包装类数组创建矩阵 / Create matrix from 2D Float wrapper array
     * <p>
     * 将二维Float包装类数组转换为矩阵对象 Converts a 2D Float wrapper array to a matrix object
     * </p>
     * 
     * @param data 二维Float包装类数组 / 2D Float wrapper array
     * @param <T> 数值类型 / Numeric type
     * @return 矩阵对象 / Matrix object
     * @throws IllegalArgumentException 如果数组为null或空 / if array is null or empty
     * 
     * @example
     * <pre>
     * Float[][] data = {{1.0f, 2.0f}, {3.0f, 4.0f}};
     * IMatrix&lt;Float&gt; matrix = IMatrix.of(data);
     * </pre>
     */
    public static <T extends Number> IMatrix<T> of(Float[][] data) {
        return (IMatrix<T>) IFloatMatrix.of(data);
    }

    /**
     * 矩阵工厂方法（从Vector数组创建） / Matrix factory method (from Vector array)
     * <p>
     * 使用给定的IDoubleVector数组创建矩阵实例，每个Vector代表矩阵的一行 Creates a matrix instance with
     * the given IDoubleVector array, each IDoubleVector representing a row of
     * the matrix
     * </p>
     *
     * @param data 包含IDoubleVector的数组，每个Vector表示矩阵的一行 / Array containing
     * IDoubleVectors, each IDoubleVector representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空数组或行长度不一致 / if data is null,
     * empty array, or row lengths are inconsistent
     */
    /**
     * 从IDoubleVector数组创建矩阵 / Create matrix from IDoubleVector array
     * <p>
     * 将IDoubleVector数组转换为矩阵对象，每个向量作为矩阵的一行 Converts an IDoubleVector array to a matrix object, with each vector as a row
     * </p>
     * 
     * @param data IDoubleVector数组 / IDoubleVector array
     * @param <T> 数值类型 / Numeric type
     * @return 矩阵对象 / Matrix object
     * @throws IllegalArgumentException 如果数组为null或空 / if array is null or empty
     * 
     * @example
     * <pre>
     * IDoubleVector[] vectors = {
     *     IDoubleVector.of(1.0, 2.0),
     *     IDoubleVector.of(3.0, 4.0)
     * };
     * IMatrix&lt;Double&gt; matrix = IMatrix.of(vectors);
     * </pre>
     */
    public static <T extends Number> IMatrix<T> of(IDoubleVector[] data) {
        return (IMatrix<T>) IDoubleMatrix.of(data);
    }

    /**
     * 矩阵工厂方法（从Vector数组创建） / Matrix factory method (from Vector array)
     * <p>
     * 使用给定的IFloatVector数组创建矩阵实例，每个Vector代表矩阵的一行 Creates a matrix instance with
     * the given IFloatVector array, each IFloatVector representing a row of the
     * matrix
     * </p>
     *
     * @param data 包含IFloatVector的数组，每个Vector表示矩阵的一行 / Array containing
     * IFloatVectors, each IFloatVector representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空数组或行长度不一致 / if data is null,
     * empty array, or row lengths are inconsistent
     */
    /**
     * 从IFloatVector数组创建矩阵 / Create matrix from IFloatVector array
     * <p>
     * 将IFloatVector数组转换为矩阵对象，每个向量作为矩阵的一行 Converts an IFloatVector array to a matrix object, with each vector as a row
     * </p>
     * 
     * @param data IFloatVector数组 / IFloatVector array
     * @param <T> 数值类型 / Numeric type
     * @return 矩阵对象 / Matrix object
     * @throws IllegalArgumentException 如果数组为null或空 / if array is null or empty
     * 
     * @example
     * <pre>
     * IFloatVector[] vectors = {
     *     IFloatVector.of(1.0f, 2.0f),
     *     IFloatVector.of(3.0f, 4.0f)
     * };
     * IMatrix&lt;Float&gt; matrix = IMatrix.of(vectors);
     * </pre>
     */
    public static <T extends Number> IMatrix<T> of(IFloatVector[] data) {
        return (IMatrix<T>) IFloatMatrix.of(data);
    }

    /**
     * 矩阵工厂方法（从double数组列表创建） / Matrix factory method (from double array List)
     * <p>
     * 使用给定的double数组列表创建矩阵实例，每个数组代表矩阵的一行 Creates a matrix instance with the
     * given list of double arrays, each array representing a row of the matrix
     * </p>
     *
     * @param data 包含double数组的列表，每个数组表示矩阵的一行 / List containing double arrays,
     * each array representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空列表或行长度不一致 / if data is null,
     * empty list, or row lengths are inconsistent
     */
    public static IMatrix<Double> ofDoubleList(List<double[]> data) {
        return (IMatrix<Double>) IDoubleMatrix.of(data);
    }

    /**
     * 矩阵工厂方法（从float数组列表创建） / Matrix factory method (from float array List)
     * <p>
     * 使用给定的float数组列表创建矩阵实例，每个数组代表矩阵的一行 Creates a matrix instance with the given
     * list of float arrays, each array representing a row of the matrix
     * </p>
     *
     * @param data 包含float数组的列表，每个数组表示矩阵的一行 / List containing float arrays, each
     * array representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空列表或行长度不一致 / if data is null,
     * empty list, or row lengths are inconsistent
     */
    public static IMatrix<Float> ofFloatList(List<float[]> data) {
        return (IMatrix<Float>) IFloatMatrix.of(data);
    }

    /**
     * 创建指定类型的全1矩阵 / Create ones matrix of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 全1矩阵 / Ones matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> ones(int rows, int cols, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.ones(rows, cols);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.ones(rows, cols);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建全1矩阵（默认Double类型） / Create matrix of ones (default Double type)
     * <p>
     * 创建一个指定大小的全1矩阵，默认使用Double类型 Creates a matrix of specified size filled with ones, using Double type by default
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param <T> 数值类型 / Numeric type
     * @return 全1矩阵 / Matrix of ones
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    static <T extends Number> IMatrix<T> ones(int rows, int cols) {
        Class type = Double.class;
        return ones(rows, cols, type);
    }

    /**
     * 创建指定类型的零矩阵 / Create zeros matrix of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 零矩阵 / Zeros matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> zeros(int rows, int cols, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.zeros(rows, cols);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.zeros(rows, cols);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建全0矩阵（默认Double类型） / Create matrix of zeros (default Double type)
     * <p>
     * 创建一个指定大小的全0矩阵，默认使用Double类型 Creates a matrix of specified size filled with zeros, using Double type by default
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param <T> 数值类型 / Numeric type
     * @return 全0矩阵 / Matrix of zeros
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    static <T extends Number> IMatrix<T> zeros(int rows, int cols) {
        Class type = Double.class;
        return zeros(rows, cols, type);
    }

    /**
     * 创建指定类型的单位矩阵 / Create identity matrix of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param size 矩阵大小（行数和列数相同） / Matrix size (rows and columns are the same)
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 单位矩阵 / Identity matrix
     * @throws IllegalArgumentException 如果大小小于等于0 / if size is less than or
     * equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> eye(int size, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.eye(size);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.eye(size);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建单位矩阵（默认Double类型） / Create identity matrix (default Double type)
     * <p>
     * 创建一个指定大小的单位矩阵，默认使用Double类型 Creates an identity matrix of specified size, using Double type by default
     * </p>
     *
     * @param size 矩阵大小（正方形矩阵的边长） / Matrix size (side length of square matrix)
     * @param <T> 数值类型 / Numeric type
     * @return 单位矩阵 / Identity matrix
     * @throws IllegalArgumentException 如果大小小于等于0 / if size is less than or equal to 0
     */
    static <T extends Number> IMatrix<T> eye(int size) {
        Class type = Double.class;
        return eye(size, type);
    }

    /**
     * 创建指定类型的随机矩阵 / Create random matrix of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> rand(int rows, int cols, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.rand(rows, cols);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.rand(rows, cols);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建随机矩阵（默认Double类型） / Create random matrix (default Double type)
     * <p>
     * 创建一个指定大小的随机矩阵，元素值在[0,1)范围内，默认使用Double类型 Creates a random matrix of specified size with elements in [0,1) range, using Double type by default
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param <T> 数值类型 / Numeric type
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    static <T extends Number> IMatrix<T> rand(int rows, int cols) {
        Class type = Double.class;
        return rand(rows, cols, type);
    }

    // ========== 额外泛型工厂方法 / Additional Generic Factory Methods ==========
    /**
     * 从一维数组创建矩阵 / Create matrix from 1D array
     * <p>
     * 将一维数组重塑为指定大小的矩阵 Reshapes a 1D array into a matrix of specified size
     * </p>
     *
     * @param data 一维数组数据 / 1D array data
     * @param rows 目标矩阵行数 / Target matrix rows
     * @param cols 目标矩阵列数 / Target matrix columns
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 重塑后的矩阵 / Reshaped matrix
     * @throws IllegalArgumentException 如果数组长度与目标尺寸不匹配 / if array length doesn't
     * match target dimensions
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> fromArray(T[] data, int rows, int cols, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.fromArray((Float[]) data, rows, cols);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.fromArray((Double[]) data, rows, cols);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

    /**
     * 从一维数组创建矩阵 / Create matrix from 1D array
     * <p>
     * 将一维数组重塑为指定大小的矩阵 Reshapes a 1D array into a matrix of specified size
     * </p>
     * 
     * @param data 一维数组数据 / 1D array data
     * @param rows 目标矩阵行数 / Target matrix rows
     * @param cols 目标矩阵列数 / Target matrix columns
     * @param <T> 数值类型 / Numeric type
     * @return 重塑后的矩阵 / Reshaped matrix
     * @throws IllegalArgumentException 如果数组为null或空，或数组长度与目标尺寸不匹配 / if array is null or empty, or array length doesn't match target dimensions
     * 
     * @example
     * <pre>
     * Double[] data = {1.0, 2.0, 3.0, 4.0};
     * IMatrix&lt;Double&gt; matrix = IMatrix.fromArray(data, 2, 2);
     * </pre>
     */
    static <T extends Number> IMatrix<T> fromArray(T[] data, int rows, int cols) {
        if (data == null || data.length == 0) {
            throw new IllegalArgumentException("数组不能为null或空 / Array cannot be null or empty");
        }
        @SuppressWarnings("unchecked")
        Class<T> type = (Class<T>) data.getClass().getComponentType();
        return fromArray(data, rows, cols, type);
    }

    /**
     * 从原始类型数组创建矩阵 / Create matrix from primitive array
     * <p>
     * 将double原始数组重塑为指定大小的矩阵 Reshapes a double primitive array into a matrix of
     * specified size
     * </p>
     *
     * @param data double原始数组数据 / double primitive array data
     * @param rows 目标矩阵行数 / Target matrix rows
     * @param cols 目标矩阵列数 / Target matrix columns
     * @return 重塑后的矩阵 / Reshaped matrix
     * @throws IllegalArgumentException 如果数组长度与目标尺寸不匹配 / if array length doesn't
     * match target dimensions
     */
    public static IMatrix<Double> fromArray(double[] data, int rows, int cols) {
        return (IMatrix<Double>) IDoubleMatrix.fromArray(data, rows, cols);
    }

    /**
     * 从原始类型数组创建矩阵 / Create matrix from primitive array
     * <p>
     * 将float原始数组重塑为指定大小的矩阵 Reshapes a float primitive array into a matrix of
     * specified size
     * </p>
     *
     * @param data float原始数组数据 / float primitive array data
     * @param rows 目标矩阵行数 / Target matrix rows
     * @param cols 目标矩阵列数 / Target matrix columns
     * @return 重塑后的矩阵 / Reshaped matrix
     * @throws IllegalArgumentException 如果数组长度与目标尺寸不匹配 / if array length doesn't
     * match target dimensions
     */
    public static IMatrix<Float> fromArray(float[] data, int rows, int cols) {
        return (IMatrix<Float>) IFloatMatrix.fromArray(data, rows, cols);
    }

    /**
     * 创建对角矩阵 / Create diagonal matrix
     * <p>
     * 从给定的对角线元素创建对角矩阵 Creates a diagonal matrix from the given diagonal
     * elements
     * </p>
     *
     * @param diagonal 对角线元素数组 / Array of diagonal elements
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 对角矩阵 / Diagonal matrix
     * @throws IllegalArgumentException 如果对角线数组为null或空 / if diagonal array is
     * null or empty
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> diag(T[] diagonal, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.diag((Float[]) diagonal);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.diag((Double[]) diagonal);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

    /**
     * 创建对角矩阵（默认Double类型） / Create diagonal matrix (default Double type)
     * <p>
     * 从给定的对角线元素创建对角矩阵，默认使用Double类型 Creates a diagonal matrix from the given diagonal elements, using Double type by default
     * </p>
     * 
     * @param diagonal 对角线元素数组 / Array of diagonal elements
     * @param <T> 数值类型 / Numeric type
     * @return 对角矩阵 / Diagonal matrix
     * @throws IllegalArgumentException 如果对角线数组为null或空 / if diagonal array is null or empty
     */
    static <T extends Number> IMatrix<T> diag(T[] diagonal) {
        Class type = Double.class;
        return diag(diagonal, type);
    }

    /**
     * 从向量创建对角矩阵 / Create diagonal matrix from vector
     * <p>
     * 从给定的向量创建对角矩阵 Creates a diagonal matrix from the given vector
     * </p>
     * 
     * @param diagonal 对角线元素向量 / Vector of diagonal elements
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 对角矩阵 / Diagonal matrix
     * @throws IllegalArgumentException 如果向量为null或空 / if vector is null or empty
     */
    static <T extends Number> IMatrix<T> diag(IVector<T> diagonal, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.diag(diagonal.toFloatArray());
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.diag(diagonal.toDoubleArray());
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

    /**
     * 从向量创建对角矩阵（默认Double类型） / Create diagonal matrix from vector (default Double type)
     * <p>
     * 从给定的向量创建对角矩阵，默认使用Double类型 Creates a diagonal matrix from the given vector, using Double type by default
     * </p>
     * 
     * @param diagonal 对角线元素向量 / Vector of diagonal elements
     * @param <T> 数值类型 / Numeric type
     * @return 对角矩阵 / Diagonal matrix
     * @throws IllegalArgumentException 如果向量为null或空 / if vector is null or empty
     */
    static <T extends Number> IMatrix<T> diag(IVector<T> diagonal) {
        return diag(diagonal.toDoubleArray());
    }

    /**
     * 从double数组创建对角矩阵 / Create diagonal matrix from double array
     * <p>
     * 从给定的double数组创建对角矩阵 Creates a diagonal matrix from the given double array
     * </p>
     * 
     * @param diagonal 对角线元素数组 / Array of diagonal elements
     * @param <T> 数值类型 / Numeric type
     * @return 对角矩阵 / Diagonal matrix
     * @throws IllegalArgumentException 如果数组为null或空 / if array is null or empty
     */
    static <T extends Number> IMatrix<T> diag(double[] diagonal) {
        Class type = Double.class;
        return diag(RereMathUtil.toClassArray(diagonal), type);
    }

    /**
     * 从float数组创建对角矩阵 / Create diagonal matrix from float array
     * <p>
     * 从给定的float数组创建对角矩阵 Creates a diagonal matrix from the given float array
     * </p>
     * 
     * @param diagonal 对角线元素数组 / Array of diagonal elements
     * @param <T> 数值类型 / Numeric type
     * @return 对角矩阵 / Diagonal matrix
     * @throws IllegalArgumentException 如果数组为null或空 / if array is null or empty
     */
    static <T extends Number> IMatrix<T> diag(float[] diagonal) {
        Class type = Float.class;
        return diag(RereMathUtil.toClassArray(diagonal), type);
    }

    /**
     * 创建随机矩阵（指定种子） / Create random matrix with specified seed
     * <p>
     * 创建一个指定大小的随机矩阵，使用指定的种子值确保结果可重现 Creates a random matrix of specified size
     * using the specified seed value to ensure reproducible results
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
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> rand(int rows, int cols, long seed, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.rand(rows, cols, seed);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.rand(rows, cols, seed);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

    /**
     * 创建随机矩阵（指定种子，默认Double类型） / Create random matrix with specified seed (default Double type)
     * <p>
     * 创建一个指定大小的随机矩阵，使用指定的种子值确保结果可重现，默认使用Double类型 Creates a random matrix of specified size using the specified seed value to ensure reproducible results, using Double type by default
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param seed 随机数种子 / Random number seed
     * @param <T> 数值类型 / Numeric type
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    static <T extends Number> IMatrix<T> rand(int rows, int cols, long seed) {
        Class type = Double.class;
        return rand(rows, cols, seed, type);
    }

    /**
     * 创建随机矩阵（正态分布，指定种子） / Create random matrix with normal distribution and
     * specified seed
     * <p>
     * 创建一个指定大小的正态随机矩阵，使用指定的种子值确保结果可重现 Creates a normal random matrix of
     * specified size using the specified seed value to ensure reproducible
     * results
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param seed 随机数种子 / Random number seed
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 正态随机矩阵 / Normal random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> randn(int rows, int cols, long seed, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.randn(rows, cols, seed);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.randn(rows, cols, seed);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

    /**
     * 创建随机矩阵（正态分布，指定种子，默认Double类型） / Create random matrix with normal distribution and specified seed (default Double type)
     * <p>
     * 创建一个指定大小的正态随机矩阵，使用指定的种子值确保结果可重现，默认使用Double类型 Creates a normal random matrix of specified size using the specified seed value to ensure reproducible results, using Double type by default
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param seed 随机数种子 / Random number seed
     * @param <T> 数值类型 / Numeric type
     * @return 正态随机矩阵 / Normal random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    static <T extends Number> IMatrix<T> randn(int rows, int cols, long seed) {
        Class type = Double.class;
        return randn(rows, cols, seed, type);
    }

    /**
     * 创建随机矩阵（正态分布） / Create random matrix with normal distribution
     * <p>
     * 创建一个指定大小的正态随机矩阵，元素值服从标准正态分布 Creates a normal random matrix of specified
     * size with elements following standard normal distribution
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 正态随机矩阵 / Normal random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> randn(int rows, int cols, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.randn(rows, cols);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.randn(rows, cols);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建随机矩阵（正态分布，默认Double类型） / Create random matrix with normal distribution (default Double type)
     * <p>
     * 创建一个指定大小的正态随机矩阵，元素值服从标准正态分布，默认使用Double类型 Creates a normal random matrix of specified size with elements following standard normal distribution, using Double type by default
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param <T> 数值类型 / Numeric type
     * @return 正态随机矩阵 / Normal random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    static <T extends Number> IMatrix<T> randn(int rows, int cols) {
        Class type = Double.class;
        return randn(rows, cols, type);
    }

    /**
     * 创建随机矩阵（指定正态分布的均值和标准差） / Create random matrix with specified mean and
     * standard deviation
     * <p>
     * 创建一个指定大小的随机矩阵，元素值服从正态分布，具有指定的均值和标准差 Creates a random matrix of specified
     * size with elements following normal distribution with specified mean and
     * standard deviation
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param mean 正态分布的均值 / Mean of normal distribution
     * @param std 正态分布的标准差 / Standard deviation of normal distribution
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0，或标准差小于0 / if rows or
     * columns are less than or equal to 0, or standard deviation is negative
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> randn(int rows, int cols, T mean, T std, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.randn(rows, cols, (Float) mean, (Float) std);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.randn(rows, cols, (Double) mean, (Double) std);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建指定均值和标准差的正态分布随机矩阵 / Create a normal distribution random matrix with specified mean and standard deviation
     * <p>使用指定的均值和标准差生成正态分布的随机矩阵。</p>
     * <p>Generates a random matrix with normal distribution using specified mean and standard deviation.</p>
     * 
     * <p>使用示例 / Usage example:</p>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.randn(3, 3, 0.0, 1.0);
     * }</pre>
     * 
     * @param rows 行数 / Number of rows
     * @param cols 列数 / Number of columns
     * @param mean 均值 / Mean value
     * @param std 标准差 / Standard deviation
     * @param <T> 数值类型 / Numeric type
     * @return 正态分布随机矩阵 / Normal distribution random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0，或均值、标准差为null / If rows or cols <= 0, or mean/std is null
     */
    static <T extends Number> IMatrix<T> randn(int rows, int cols, T mean, T std) {
        if (mean == null || std == null) {
            throw new IllegalArgumentException("均值和标准差不能为null / Mean and std cannot be null");
        }
        @SuppressWarnings("unchecked")
        Class<T> type = (Class<T>) mean.getClass();
        return randn(rows, cols, mean, std, type);
    }

    /**
     * 计算两个矩阵数组的平均值 / Calculate average of two matrix arrays
     * <p>
     * 计算两个相同长度的矩阵数组对应元素的平均值 Calculates the average of corresponding elements in
     * two matrix arrays of the same length
     * </p>
     *
     * @param a 第一个矩阵数组 / First matrix array
     * @param b 第二个矩阵数组 / Second matrix array
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 平均值矩阵 / Average matrix
     * @throws IllegalArgumentException 如果数组长度不匹配 / if array lengths don't match
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> average(IMatrix<T>[] a, IMatrix<T>[] b, Class<T> type) {
        if (type == Float.class) {
            // Convert generic arrays to specific type arrays
            IFloatMatrix[] floatA = new IFloatMatrix[a.length];
            IFloatMatrix[] floatB = new IFloatMatrix[b.length];
            for (int i = 0; i < a.length; i++) {
                floatA[i] = (IFloatMatrix) a[i];
                floatB[i] = (IFloatMatrix) b[i];
            }
            return (IMatrix<T>) IFloatMatrix.average(floatA, floatB);
        } else if (type == Double.class) {
            // Convert generic arrays to specific type arrays
            IDoubleMatrix[] doubleA = new IDoubleMatrix[a.length];
            IDoubleMatrix[] doubleB = new IDoubleMatrix[b.length];
            for (int i = 0; i < a.length; i++) {
                doubleA[i] = (IDoubleMatrix) a[i];
                doubleB[i] = (IDoubleMatrix) b[i];
            }
            return (IMatrix<T>) IDoubleMatrix.average(doubleA, doubleB);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 计算两个矩阵的平均值 / Calculate average of two matrices
     * <p>
     * 计算两个相同维度的矩阵对应元素的平均值 Calculates the average of corresponding elements in
     * two matrices of the same dimensions
     * </p>
     *
     * @param a 第一个矩阵 / First matrix
     * @param b 第二个矩阵 / Second matrix
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 平均值矩阵 / Average matrix
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     * @throws NullPointerException 如果任何参数为null / if any parameter is null
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> average(IMatrix<T> a, IMatrix<T> b, Class<T> type) {
        if (a == null || b == null) {
            throw new NullPointerException("矩阵不能为null / Matrix cannot be null");
        }

        if (a.getRowNum() != b.getRowNum() || a.getColNum() != b.getColNum()) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match: "
                    + "a[" + a.getRowNum() + "x" + a.getColNum() + "] vs "
                    + "b[" + b.getRowNum() + "x" + b.getColNum() + "]");
        }

        if (type == Float.class) {
            // Convert to Float matrices and calculate average
            IFloatMatrix floatA = (IFloatMatrix) a;
            IFloatMatrix floatB = (IFloatMatrix) b;

            int rows = a.getRowNum();
            int cols = a.getColNum();
            IFloatMatrix result = IFloatMatrix.zeros(rows, cols);

            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    float avg = (floatA.get(row, col) + floatB.get(row, col)) / 2.0f;
                    result.put(row, col, avg);
                }
            }

            return (IMatrix<T>) result;
        } else if (type == Double.class) {
            // Convert to Double matrices and calculate average
            IDoubleMatrix doubleA = (IDoubleMatrix) a;
            IDoubleMatrix doubleB = (IDoubleMatrix) b;

            int rows = a.getRowNum();
            int cols = a.getColNum();
            IDoubleMatrix result = IDoubleMatrix.zeros(rows, cols);

            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    double avg = (doubleA.get(row, col) + doubleB.get(row, col)) / 2.0;
                    result.put(row, col, avg);
                }
            }

            return (IMatrix<T>) result;
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 从本地指定位置path加载恢复矩阵 / Load matrix from specified local path
     * <p>
     * 从文件中加载矩阵数据 Loads matrix data from file
     * </p>
     *
     * @param path 文件路径 / File path
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 加载的矩阵 / Loaded matrix
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IMatrix<T> load(String path, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.load(path);
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.load(path);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    static IMatrix<Double> load(String path) {
        return (IMatrix<Double>) IDoubleMatrix.load(path);
    }

    /**
     * 获取行数 / Get row count
     * <p>
     * 返回矩阵的行数 Returns the number of rows in the matrix
     * </p>
     *
     * @return 矩阵行数 / Number of rows in the matrix
     */
    public int getRowNum();

    /**
     * 获取列数 / Get column count
     * <p>
     * 返回矩阵的列数 Returns the number of columns in the matrix
     * </p>
     *
     * @return 矩阵列数 / Number of columns in the matrix
     */
    public int getColNum();

    /**
     * 获取指定位置的元素值 / Get element value at specified position
     * <p>
     * 返回矩阵中指定行列位置的元素值 Returns the element value at the specified row and column
     * position in the matrix
     * </p>
     *
     * @param row 行索引（从0开始） / Row index (0-based)
     * @param col 列索引（从0开始） / Column index (0-based)
     * @return 指定位置的元素值 / Element value at the specified position
     * @throws IndexOutOfBoundsException 如果行列索引超出范围 / if row or column index is
     * out of bounds
     */
    public T get(int row, int col);

    /**
     * 设置指定位置的元素值 / Set element value at specified position
     * <p>
     * 设置矩阵中指定行列位置的元素值 Sets the element value at the specified row and column
     * position in the matrix
     * </p>
     *
     * @param row 行索引（从0开始） / Row index (0-based)
     * @param col 列索引（从0开始） / Column index (0-based)
     * @param value 要设置的值 / Value to set
     * @throws IndexOutOfBoundsException 如果行列索引超出范围 / if row or column index is
     * out of bounds
     */
    public IMatrix put(int row, int col, T value);

    /**
     * 矩阵加法运算 / Matrix addition
     * <p>
     * 对应元素相加，要求两个矩阵维度相同 Element-wise addition, requires both matrices to have
     * the same dimensions
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     */
    public IMatrix<T> add(IMatrix<T> other);

    /**
     * 矩阵减法运算（矩阵） / Matrix subtraction with another matrix
     * <p>
     * 对应元素相减，要求两个矩阵维度相同 Element-wise subtraction, requires both matrices to
     * have the same dimensions
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     * @throws NullPointerException 如果other为null / if other is null
     */
    public IMatrix<T> sub(IMatrix<T> other);

    /**
     * 矩阵减法运算（标量） / Matrix subtraction with scalar
     * <p>
     * 矩阵中的每个元素减去标量值 Subtracts a scalar value from each element in the matrix
     * </p>
     *
     * @param scalar 标量值 / The scalar value
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果标量为null / if scalar is null
     */
    public IMatrix<T> sub(T scalar);

    /**
     * 矩阵标量乘法运算 / Matrix scalar multiplication
     * <p>
     * 矩阵中的每个元素乘以标量值 Multiplies each element in the matrix by a scalar value
     * </p>
     *
     * @param scalar 标量乘数 / The scalar multiplier
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    public IMatrix<T> multiplyScalar(T scalar);

    /**
     * 矩阵除法运算（矩阵） / Matrix division with another matrix
     * <p>
     * 对应元素相除，要求两个矩阵维度相同 Element-wise division, requires both matrices to have
     * the same dimensions
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     * @throws ArithmeticException 如果other矩阵中有零元素 / if other matrix contains
     * zero elements
     * @throws NullPointerException 如果other为null / if other is null
     */
    public IMatrix<T> divide(IMatrix<T> other);

    /**
     * 矩阵标量除法运算 / Matrix scalar division
     * <p>
     * 将矩阵中的每个元素除以指定的标量值 Divides each element of the matrix by the specified scalar value
     * </p>
     * <p>
     * 公式：Cᵢⱼ = Aᵢⱼ / scalar Formula: Cᵢⱼ = Aᵢⱼ / scalar
     * </p>
     * <p>
     * 使用示例 Usage example:
     * <pre>
     * IMatrix&lt;Double&gt; matrix = IMatrix.of(new double[][]{{4.0, 8.0}, {12.0, 16.0}});
     * IMatrix&lt;Double&gt; result = matrix.divideByScalar(2.0);
     * // result: [[2.0, 4.0], [6.0, 8.0]]
     * </pre>
     * </p>
     *
     * @param scalar 除数标量 / The divisor scalar
     * @return 新的矩阵，包含除法运算结果 / A new matrix containing the division result
     * @throws ArithmeticException 如果标量为零 / if scalar is zero
     * @throws NullPointerException 如果scalar为null / if scalar is null
     */
    public IMatrix<T> divideByScalar(T scalar);

    /**
     * 矩阵点积运算 / Matrix dot product
     * <p>
     * 计算两个矩阵的Frobenius内积（对应元素相乘后求和） Calculates the Frobenius inner product of
     * two matrices (sum of element-wise multiplication)
     * </p>
     * <p>
     * 公式：dot = Σᵢⱼ Aᵢⱼ × Bᵢⱼ Formula: dot = Σᵢⱼ Aᵢⱼ × Bᵢⱼ
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 点积结果 / Dot product result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     * @throws NullPointerException 如果other为null / if other is null
     */
    public T dot(IMatrix<T> other);

    /**
     * 矩阵乘法运算 / Matrix multiplication
     * <p>
     * 计算两个矩阵的乘积，要求第一个矩阵的列数等于第二个矩阵的行数 Computes the product of two matrices,
     * requires the number of columns of the first matrix to equal the number of
     * rows of the second matrix
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> A = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
     * IMatrix<Double> B = Linalg.matrix(new double[][]{{5, 6}, {7, 8}});
     * IMatrix<Double> C = A.mmul(B);  // 结果: [[19, 22], [43, 50]]
     * }</pre>
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 矩阵乘法结果 / Matrix multiplication result
     * @throws IllegalArgumentException 如果矩阵维度不匹配进行乘法运算 / if matrix dimensions
     * don't match for multiplication
     * @throws NullPointerException 如果other为null / if other is null
     */
    public IMatrix<T> mmul(IMatrix<T> other);

    /**
     * 矩阵与向量乘法 / Matrix-vector multiplication
     * <p>
     * 计算矩阵与列向量的乘积，结果仍为列向量。要求矩阵的列数等于向量的长度。 Computes the product of a matrix with
     * a column vector, result is still a column vector. Requires matrix column
     * count to equal vector length.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
     * IVector<Double> vector = Linalg.vector(new double[]{5, 6});
     * IVector<Double> result = matrix.mmul(vector);  // 结果: [17, 39]
     * }</pre>
     * </p>
     *
     * @param other 要相乘的向量 / Vector to multiply with
     * @return 矩阵与向量的乘积结果 / Matrix-vector multiplication result
     * @throws IllegalArgumentException 如果矩阵列数与向量长度不匹配 / if matrix column count
     * doesn't match vector length
     * @throws NullPointerException 如果other为null / if other is null
     */
    public IVector<T> mmul(IVector<T> other);

    /**
     * 矩阵复制 / Matrix copy
     * <p>
     * 创建当前矩阵的深拷贝 Creates a deep copy of the current matrix
     * </p>
     *
     * @return 矩阵的副本 / Copy of the matrix
     */
    public IMatrix<T> copy();

    /**
     * 矩阵元素求和 / Matrix element sum
     * <p>
     * 返回矩阵中所有元素的总和 Returns the sum of all elements in the matrix
     * </p>
     *
     * @return 元素总和 / Sum of all elements
     */
    public T sum();

    /**
     * 矩阵元素均值 / Matrix element mean
     * <p>
     * 返回矩阵中所有元素的平均值 Returns the mean of all elements in the matrix
     * </p>
     *
     * @return 元素均值 / Mean of all elements
     */
    public T mean();

    /**
     * 矩阵最大值 / Matrix maximum value
     * <p>
     * 返回矩阵中的最大元素值 Returns the maximum element value in the matrix
     * </p>
     *
     * @return 最大元素值 / Maximum element value
     */
    public T max();

    /**
     * 矩阵最小值 / Matrix minimum value
     * <p>
     * 返回矩阵中的最小元素值 Returns the minimum element value in the matrix
     * </p>
     *
     * @return 最小元素值 / Minimum element value
     */
    public T min();

    // ========== 高级矩阵操作 / Advanced Matrix Operations ==========
    /**
     * 获取矩阵形状 / Get matrix shape
     * <p>
     * 返回矩阵的维度信息，包含行数和列数 Returns the dimension information of the matrix,
     * including row and column counts
     * </p>
     *
     * @return 包含行数和列数的数组 [行数, 列数] / Array containing row and column counts
     * [rows, columns]
     */
    public int[] shape();

    /**
     * 获取矩阵行数 / Get number of rows (alias for getRowNum)
     * <p>
     * 返回矩阵的行数，这是getRowNum()方法的别名 Returns the number of rows in the matrix, this is an alias for getRowNum()
     * </p>
     *
     * @return 矩阵的行数 / Number of rows in the matrix
     */
    public int rows();

    /**
     * 获取矩阵列数 / Get number of columns (alias for getColNum)
     * <p>
     * 返回矩阵的列数，这是getColNum()方法的别名 Returns the number of columns in the matrix, this is an alias for getColNum()
     * </p>
     *
     * @return 矩阵的列数 / Number of columns in the matrix
     */
    public int cols();

    /**
     * 矩阵转置（就地操作） / Matrix transpose (in-place operation)
     * <p>
     * 对当前矩阵进行就地转置操作，修改原矩阵 Performs in-place transpose operation on the current
     * matrix, modifying the original matrix
     * </p>
     * <p>
     * <strong>注意：</strong>此操作会修改原矩阵，请谨慎使用 <strong>Note:</strong> This operation
     * modifies the original matrix, use with caution
     * </p>
     *
     * @return 转置后的矩阵（原矩阵被修改）/ Transposed matrix (original matrix is modified)
     */
    public IMatrix<T> transposeInPlace();

    /**
     * 矩阵转置（创建新对象） / Matrix transpose (create new object)
     * <p>
     * 创建一个新的转置矩阵，不修改原矩阵 Creates a new transposed matrix without modifying the
     * original matrix
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
     * IMatrix<Double> transposed = matrix.transposeNew();  // 结果: [[1, 3], [2, 4]]
     * }</pre>
     * </p>
     *
     * @return 转置后的新矩阵 / New transposed matrix
     */
    public IMatrix<T> transposeNew();

    /**
     * 矩阵转置简写方法 / Matrix transpose shorthand method
     * <p>
     * transpose()方法的简写形式 Shorthand form of transpose() method
     * </p>
     *
     * @return 转置后的矩阵 / Transposed matrix
     * @see #transpose()
     */
    public default IMatrix<T> t() {
        return transpose();
    }

    /**
     * 矩阵转置（产生新对象） / Matrix transpose (create new object)
     * <p>
     * 创建一个新的转置矩阵，不修改原矩阵 Creates a new transposed matrix without modifying the
     * original matrix
     * </p>
     *
     * @return 转置后的矩阵 / Transposed matrix
     */
    public default IMatrix<T> transpose() {
        return this.transposeNew();
    }

    // ========== 矩阵分解方法 / Matrix Decomposition Methods ==========
    /**
     * 特征分解 / Eigenvalue decomposition
     * <p>
     * 计算矩阵的特征值和特征向量，将矩阵A分解为A = VΛV⁻¹的形式，其中V是特征向量矩阵，Λ是对角特征值矩阵 Computes
     * eigenvalues and eigenvectors of the matrix, decomposing matrix A into A =
     * VΛV⁻¹ form, where V is the eigenvector matrix and Λ is the diagonal
     * eigenvalue matrix
     * </p>
     * <p>
     * 特征值按大小排列（从大到小），特征向量矩阵的列与特征值顺序对应 Eigenvalues are ordered by magnitude
     * (largest to smallest), eigenvector matrix columns correspond to
     * eigenvalue order
     * </p>
     *
     * @return 包含特征值和特征向量的元组，第一个元素是特征值向量，第二个元素是特征向量矩阵（列为特征向量） Tuple containing
     * eigenvalues and eigenvectors, first element is eigenvalue vector, second
     * element is eigenvector matrix (columns are eigenvectors)
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     * @throws ArithmeticException 如果矩阵无法进行特征分解 / if matrix cannot be
     * eigendecomposed
     */
    public Tuple2<IVector<T>, IMatrix<T>> eigen();

    /**
     * 奇异值分解 / Singular Value Decomposition (SVD)
     * <p>
     * 将矩阵A分解为A = UΣV^T的形式，其中U和V是正交矩阵，Σ是对角奇异值矩阵 Decomposes matrix A into A =
     * UΣV^T form, where U and V are orthogonal matrices and Σ is the diagonal
     * singular values matrix
     * </p>
     * <p>
     * 奇异值按降序排列，U的列是左奇异向量，V^T的行是右奇异向量 Singular values are ordered in descending
     * order, columns of U are left singular vectors, rows of V^T are right
     * singular vectors
     * </p>
     *
     * @return 包含U矩阵、奇异值向量和V^T矩阵的三元组 Tuple containing U matrix, singular values
     * vector, and V^T matrix
     * @throws ArithmeticException 如果矩阵无法进行奇异值分解 / if matrix cannot be SVD
     * decomposed
     */
    public Tuple3<IMatrix<T>, IVector<T>, IMatrix<T>> svd();

    /**
     * QR分解 / QR decomposition
     * <p>
     * 将矩阵A分解为A = QR的形式，其中Q是正交矩阵，R是上三角矩阵 Decomposes matrix A into A = QR form,
     * where Q is an orthogonal matrix and R is an upper triangular matrix
     * </p>
     * <p>
     * QR分解常用于求解线性方程组和最小二乘问题 QR decomposition is commonly used for solving
     * linear systems and least squares problems
     * </p>
     *
     * @return 包含Q矩阵和R矩阵的元组 Tuple containing Q matrix and R matrix
     * @throws ArithmeticException 如果矩阵无法进行QR分解 / if matrix cannot be QR
     * decomposed
     */
    public Tuple2<IMatrix<T>, IMatrix<T>> qr();

    /**
     * 求解矩阵的逆 / Matrix inverse
     * <p>
     * 计算方阵的逆矩阵，使得A × A⁻¹ = I（单位矩阵） Computes the inverse of a square matrix such
     * that A × A⁻¹ = I (identity matrix)
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> A = Linalg.matrix(new double[][]{{2, 1}, {1, 1}});
     * IMatrix<Double> A_inv = A.inv();  // 结果: [[1, -1], [-1, 2]]
     * IMatrix<Double> identity = A.mmul(A_inv);  // 结果: [[1, 0], [0, 1]]
     * }</pre>
     * </p>
     *
     * @return 逆矩阵 / Inverse matrix
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     * @throws ArithmeticException 如果矩阵是奇异的（不可逆） / if matrix is singular
     * (non-invertible)
     */
    public IMatrix<T> inv();

    /**
     * 求解矩阵的伪逆 / Matrix pseudo-inverse
     * <p>
     * 计算矩阵的Moore-Penrose伪逆，适用于非方阵或奇异矩阵 Computes the Moore-Penrose
     * pseudo-inverse of the matrix, suitable for non-square or singular
     * matrices
     * </p>
     * <p>
     * 伪逆满足以下性质：Pseudo-inverse satisfies the following properties:
     * <ul>
     * <li>A A⁺ A = A</li>
     * <li>A⁺ A A⁺ = A⁺</li>
     * <li>(A A⁺)ᵀ = A A⁺</li>
     * <li>(A⁺ A)ᵀ = A⁺ A</li>
     * </ul>
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
     * IMatrix<Double> pseudoInverse = matrix.pinv();
     * }</pre>
     * </p>
     *
     * @return 伪逆矩阵 / Pseudo-inverse matrix
     * @throws ArithmeticException 如果矩阵无法进行伪逆计算 / if matrix cannot be
     * pseudo-inverted
     */
    public IMatrix<T> pinv();

    // ========== 行列操作方法 / Row and Column Operations ==========
    /**
     * 计算行求和 / Calculate row sums
     * <p>
     * 计算矩阵每一行的元素和，返回一个向量，其中每个元素对应原矩阵一行的和 Calculates the sum of elements in each
     * row of the matrix, returns a vector where each element corresponds to the
     * sum of a row in the original matrix
     * </p>
     *
     * @return 包含每行和的向量，长度为矩阵行数 / Vector containing row sums, length equals
     * matrix row count
     */
    public IVector<T> rowSums();

    /**
     * 计算行均值 / Calculate row means
     * <p>
     * 计算矩阵每一行的元素均值，返回一个向量，其中每个元素对应原矩阵一行的均值 Calculates the mean of elements in
     * each row of the matrix, returns a vector where each element corresponds
     * to the mean of a row in the original matrix
     * </p>
     *
     * @return 包含每行均值的向量，长度为矩阵行数 / Vector containing row means, length equals
     * matrix row count
     */
    public IVector<T> rowMeans();

    /**
     * 计算列求和 / Calculate column sums
     * <p>
     * 计算矩阵每一列的元素和，返回一个向量，其中每个元素对应原矩阵一列的和 Calculates the sum of elements in each
     * column of the matrix, returns a vector where each element corresponds to
     * the sum of a column in the original matrix
     * </p>
     *
     * @return 包含每列和的向量，长度为矩阵列数 / Vector containing column sums, length equals
     * matrix column count
     */
    public IVector<T> colSums();

    /**
     * 计算列均值 / Calculate column means
     * <p>
     * 计算矩阵每一列的元素均值，返回一个向量，其中每个元素对应原矩阵一列的均值 Calculates the mean of elements in
     * each column of the matrix, returns a vector where each element
     * corresponds to the mean of a column in the original matrix
     * </p>
     *
     * @return 包含每列均值的向量，长度为矩阵列数 / Vector containing column means, length equals
     * matrix column count
     */
    public IVector<T> colMeans();

    /**
     * 获取指定列向量 / Get specified column vector
     * <p>
     * 获取矩阵中指定索引的列，返回一个向量包含该列的所有元素 Gets the column at the specified index from
     * the matrix, returns a vector containing all elements of that column
     * </p>
     *
     * @param i 列索引（从0开始）/ Column index (0-based)
     * @return 包含指定列元素的向量，长度为矩阵行数 / Vector containing column elements, length
     * equals matrix row count
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
    public IVector<T> getColumn(int i);

    /**
     * 获取指定列作为列向量矩阵 / Get specified column as column vector matrix
     * <p>
     * 获取矩阵中指定索引的列，返回一个n×1的矩阵（列向量） Gets the column at the specified index from
     * the matrix, returns an n×1 matrix (column vector)
     * </p>
     *
     * @param i 列索引（从0开始）/ Column index (0-based)
     * @return n×1的矩阵，包含指定列元素 / n×1 matrix containing column elements
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
    public IMatrix<T> getColumnAsCloumnVector(int i);

    /**
     * 获取指定行向量 / Get specified row vector
     * <p>
     * 获取矩阵中指定索引的行，返回一个向量包含该行的所有元素 Gets the row at the specified index from the
     * matrix, returns a vector containing all elements of that row
     * </p>
     *
     * @param i 行索引（从0开始）/ Row index (0-based)
     * @return 包含指定行元素的向量，长度为矩阵列数 / Vector containing row elements, length
     * equals matrix column count
     * @throws IndexOutOfBoundsException 如果行索引超出范围 / if row index is out of
     * bounds
     */
    public IVector<T> getRow(int i);

    /**
     * 获取指定列矩阵 / Get specified column as matrix
     * <p>
     * 获取矩阵中指定索引的列，返回一个n×1的矩阵（列向量） Gets the column at the specified index from
     * the matrix, returns an n×1 matrix (column vector)
     * </p>
     *
     * @param colIndex 列索引（从0开始）/ Column index (0-based)
     * @return n×1的矩阵，包含指定列元素 / n×1 matrix containing column elements
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
    public IMatrix<T> getColumnMatrix(int colIndex);

    /**
     * 设置指定列（矩阵形式） / Set specified column (matrix form)
     * <p>
     * 将指定列设置为给定的列矩阵 Sets the specified column to the given column matrix
     * </p>
     *
     * @param colIndex 列索引（从0开始）/ Column index (0-based)
     * @param column 要设置的列矩阵 / Column matrix to set
     * @return 修改后的矩阵（就地操作） / Modified matrix (in-place operation)
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     * @throws IllegalArgumentException 如果列矩阵尺寸不匹配 / if column matrix dimensions
     * don't match
     */
    public IMatrix putColumn(int colIndex, IMatrix<T> column);

    /**
     * 设置指定列（向量形式） / Set specified column (vector form)
     * <p>
     * 将指定列设置为给定的向量 Sets the specified column to the given vector
     * </p>
     *
     * @param colIndex 列索引（从0开始）/ Column index (0-based)
     * @param column 要设置的列向量 / Column vector to set
     * @return 修改后的矩阵（就地操作） / Modified matrix (in-place operation)
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector length doesn't
     * match
     */
    public IMatrix setColumn(int colIndex, IVector<T> column);

    /**
     * 设置指定行（向量形式） / Set specified row (vector form)
     * <p>
     * 将指定行设置为给定的向量 Sets the specified row to the given vector
     * </p>
     *
     * @param rowIndex 行索引（从0开始）/ Row index (0-based)
     * @param row 要设置的行向量 / Row vector to set
     * @return 修改后的矩阵（就地操作） / Modified matrix (in-place operation)
     * @throws IndexOutOfBoundsException 如果行索引超出范围 / if row index is out of
     * bounds
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector length doesn't
     * match
     */
    public IMatrix setRow(int rowIndex, IVector<T> row);

    /**
     * 获取多个指定列 / Get multiple specified columns
     * <p>
     * 根据列索引数组获取多个列，返回列向量数组 Gets multiple columns based on column index array,
     * returns column vector array
     * </p>
     *
     * @param indices 列索引数组 / Array of column indices
     * @return 列向量数组 / Array of column vectors
     * @throws IndexOutOfBoundsException 如果任何列索引超出范围 / if any column index is
     * out of bounds
     * @throws IllegalArgumentException 如果indices为null或空 / if indices is null or
     * empty
     */
    public IVector<T>[] getColumns(int[] indices);

    // ========== 元素操作 / Element Operations ==========
    /**
     * 矩阵平方根运算 / Matrix square root operation
     * <p>
     * 对矩阵中每个元素进行平方根运算（√x） Performs square root operation (√x) on each element
     * in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含平方根运算结果 / New matrix object containing square root
     * operation results
     * @throws ArithmeticException 如果任何元素为负数 / if any element is negative
     */
    public IMatrix<T> sqrt();

    /**
     * 矩阵幂运算 / Matrix power operation
     * <p>
     * 对矩阵中每个元素进行幂运算（x^power） Performs power operation (x^power) on each element
     * in the matrix
     * </p>
     *
     * @param power 幂指数 / Power exponent
     * @return 新的矩阵对象，包含幂运算结果 / New matrix object containing power operation
     * results
     * @throws IllegalArgumentException 如果幂指数为null / if power is null
     * @throws ArithmeticException 如果底数为负数且幂指数不是整数 / if base is negative and
     * power is not an integer
     */
    public IMatrix<T> pow(T power);

    /**
     * 矩阵指数运算 / Matrix exponential operation
     * <p>
     * 对矩阵中每个元素进行指数运算（e^x） Performs exponential operation (e^x) on each element
     * in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含指数运算结果 / New matrix object containing exponential
     * operation results
     */
    public IMatrix<T> exp();

    /**
     * 矩阵自然对数运算 / Matrix natural logarithm operation
     * <p>
     * 对矩阵中每个元素进行自然对数运算（ln(x)） Performs natural logarithm operation (ln(x)) on
     * each element in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含自然对数运算结果 / New matrix object containing natural
     * logarithm operation results
     * @throws ArithmeticException 如果任何元素小于等于0 / if any element is less than or
     * equal to 0
     */
    public IMatrix<T> log();

    /**
     * 计算Frobenius范数 / Compute Frobenius norm
     * <p>
     * 计算矩阵的Frobenius范数，即所有元素平方和的平方根 Calculates the Frobenius norm of the
     * matrix, which is the square root of the sum of squares of all elements
     * </p>
     * <p>
     * 公式：||A||_F = √(Σᵢⱼ |aᵢⱼ|²) Formula: ||A||_F = √(Σᵢⱼ |aᵢⱼ|²)
     * </p>
     *
     * @return Frobenius范数 / Frobenius norm
     */
    public T frobeniusNorm();

    public default T frobenius() {
        return this.frobeniusNorm();
    }

    public default T norm() {
        return this.frobeniusNorm();
    }

    // 快速检查方法
    public default boolean isSquare() {
        return this.cols() == this.rows();
    }

    /**
     * 检查矩阵是否为对称矩阵 / Check if the matrix is symmetric
     * <p>
     * 对称矩阵满足 A = A^T，即矩阵等于其转置 A symmetric matrix satisfies A = A^T, i.e., the matrix equals its transpose
     * </p>
     * <p>
     * 注意：只有方阵才可能是对称的 Note: Only square matrices can be symmetric
     * </p>
     *
     * @return 如果矩阵是对称的则返回true / true if the matrix is symmetric
     * @throws IllegalStateException 如果矩阵不是方阵 / if the matrix is not square
     */
    public boolean isSymmetric();

    /**
     * 检查矩阵是否为正定矩阵 / Check if the matrix is positive definite
     * <p>
     * 正定矩阵是对称矩阵，且所有特征值都为正数 A positive definite matrix is symmetric and all eigenvalues are positive
     * </p>
     * <p>
     * 等价条件：对于任意非零向量x，都有x^T * A * x > 0 Equivalent condition: for any non-zero vector x, x^T * A * x > 0
     * </p>
     *
     * @return 如果矩阵是正定的则返回true / true if the matrix is positive definite
     * @throws IllegalStateException 如果矩阵不是方阵 / if the matrix is not square
     */
    public boolean isPositiveDefinite();

    // 便利转换方法
    /**
     * 矩阵展平为向量 / Flatten matrix to vector
     * <p>
     * 将矩阵按行优先顺序展平为一维向量 Flattens the matrix to a 1D vector in row-major order
     * </p>
     * <p>
     * 对于m×n矩阵，结果向量长度为m*n For an m×n matrix, the resulting vector has length m*n
     * </p>
     * <p>
     * 使用示例 Usage example:
     * <pre>
     * IMatrix&lt;Double&gt; matrix = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
     * IVector&lt;Double&gt; vector = matrix.flatten();
     * // vector: [1, 2, 3, 4]
     * </pre>
     * </p>
     *
     * @return 展平后的向量 / Flattened vector
     */
    public IVector<T> flatten();

    /**
     * 计算Frobenius距离 / Compute Frobenius distance
     */
    public T frobeniusDistance(IMatrix<T> other);

    /**
     * 矩阵按行归一化 / Row-wise normalization
     */
    public IMatrix<T> normalizeRows();

    /**
     * 矩阵按列归一化 / Column-wise normalization
     */
    public IMatrix<T> normalizeColumns();

    /**
     * 整体归一化
     *
     * @return
     */
    public IMatrix<T> normalize();

    /**
     * 矩阵数据中心化 / Matrix data centering
     */
    public IMatrix<T> center();

    /**
     * 计算协方差矩阵 / Compute covariance matrix
     */
    public IMatrix<T> covariance();

    /**
     * 计算协方差矩阵 / Compute covariance matrix
     */
    public IMatrix<T> cov();

    /**
     * 从已中心化数据计算协方差矩阵 / Compute covariance matrix from centered data
     */
    public IMatrix<T> covarianceFromCentered();

    /**
     * 获取矩阵对角线元素 / Get matrix diagonal elements
     */
    public IVector<T> diag();

    // ========== 矩阵连接和变换操作 / Matrix Concatenation and Transformation Operations ==========
    /**
     * 矩阵连接（水平方向） / Matrix concatenation (horizontal)
     * <p>
     * 将两个矩阵在水平方向上连接，要求行数相同 Concatenates two matrices horizontally, requires
     * same number of rows
     * </p>
     *
     * @param other 要连接的另一个矩阵 / The other matrix to concatenate
     * @return 连接后的矩阵 / Concatenated matrix
     * @throws IllegalArgumentException 如果矩阵行数不匹配 / if matrix row counts don't
     * match
     */
    public IMatrix<T> hstack(IMatrix<T> other);

    /**
     * 矩阵连接（垂直方向） / Matrix concatenation (vertical)
     * <p>
     * 将两个矩阵在垂直方向上连接，要求列数相同 Concatenates two matrices vertically, requires same
     * number of columns
     * </p>
     *
     * @param other 要连接的另一个矩阵 / The other matrix to concatenate
     * @return 连接后的矩阵 / Concatenated matrix
     * @throws IllegalArgumentException 如果矩阵列数不匹配 / if matrix column counts
     * don't match
     */
    public IMatrix<T> vstack(IMatrix<T> other);

    /**
     * 矩阵分割（水平方向） / Matrix splitting (horizontal)
     * <p>
     * 将矩阵在水平方向上分割为多个子矩阵 Splits the matrix horizontally into multiple
     * sub-matrices
     * </p>
     *
     * @param indices 分割点的列索引数组 / Array of column indices for split points
     * @return 分割后的子矩阵数组 / Array of split sub-matrices
     * @throws IllegalArgumentException 如果分割索引无效 / if split indices are invalid
     */
    public IMatrix<T>[] hsplit(int[] indices);

    /**
     * 矩阵分割（垂直方向） / Matrix splitting (vertical)
     * <p>
     * 将矩阵在垂直方向上分割为多个子矩阵 Splits the matrix vertically into multiple sub-matrices
     * </p>
     *
     * @param indices 分割点的行索引数组 / Array of row indices for split points
     * @return 分割后的子矩阵数组 / Array of split sub-matrices
     * @throws IllegalArgumentException 如果分割索引无效 / if split indices are invalid
     */
    public IMatrix<T>[] vsplit(int[] indices);

    /**
     * 矩阵重塑 / Matrix reshape
     * <p>
     * 将矩阵重塑为新的维度，保持元素总数不变 Reshapes the matrix to new dimensions while keeping
     * the total number of elements unchanged
     * </p>
     *
     * @param rows 新的行数 / New number of rows
     * @param cols 新的列数 / New number of columns
     * @return 重塑后的矩阵 / Reshaped matrix
     * @throws IllegalArgumentException 如果新维度与元素总数不匹配 / if new dimensions don't
     * match total element count
     */
    public IMatrix<T> reshape(int rows, int cols);

    // ========== 数学函数 / Mathematical Functions ==========
    /**
     * 矩阵绝对值 / Matrix absolute value
     * <p>
     * 对矩阵中每个元素进行绝对值运算 Performs absolute value operation on each element in the
     * matrix
     * </p>
     *
     * @return 新的矩阵对象，包含绝对值运算结果 / New matrix object containing absolute value
     * operation results
     */
    public IMatrix<T> abs();

    /**
     * 矩阵符号函数 / Matrix sign function
     * <p>
     * 对矩阵中每个元素进行符号函数运算（-1, 0, 1） Performs sign function operation on each
     * element in the matrix (-1, 0, 1)
     * </p>
     *
     * @return 新的矩阵对象，包含符号函数运算结果 / New matrix object containing sign function
     * operation results
     */
    public IMatrix<T> sign();

    /**
     * 矩阵正弦函数 / Matrix sine function
     * <p>
     * 对矩阵中每个元素进行正弦函数运算 Performs sine function operation on each element in the
     * matrix
     * </p>
     *
     * @return 新的矩阵对象，包含正弦函数运算结果 / New matrix object containing sine function
     * operation results
     */
    public IMatrix<T> sin();

    /**
     * 矩阵余弦函数 / Matrix cosine function
     * <p>
     * 对矩阵中每个元素进行余弦函数运算 Performs cosine function operation on each element in
     * the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含余弦函数运算结果 / New matrix object containing cosine function
     * operation results
     */
    public IMatrix<T> cos();

    /**
     * 矩阵正切函数 / Matrix tangent function
     * <p>
     * 对矩阵中每个元素进行正切函数运算 Performs tangent function operation on each element in
     * the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含正切函数运算结果 / New matrix object containing tangent function
     * operation results
     */
    public IMatrix<T> tan();

    /**
     * 矩阵双曲正弦函数 / Matrix hyperbolic sine function
     * <p>
     * 对矩阵中每个元素进行双曲正弦函数运算 Performs hyperbolic sine function operation on each
     * element in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含双曲正弦函数运算结果 / New matrix object containing hyperbolic
     * sine function operation results
     */
    public IMatrix<T> sinh();

    /**
     * 矩阵双曲余弦函数 / Matrix hyperbolic cosine function
     * <p>
     * 对矩阵中每个元素进行双曲余弦函数运算 Performs hyperbolic cosine function operation on each
     * element in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含双曲余弦函数运算结果 / New matrix object containing hyperbolic
     * cosine function operation results
     */
    public IMatrix<T> cosh();

    /**
     * 矩阵双曲正切函数 / Matrix hyperbolic tangent function
     * <p>
     * 对矩阵中每个元素进行双曲正切函数运算 Performs hyperbolic tangent function operation on each
     * element in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含双曲正切函数运算结果 / New matrix object containing hyperbolic
     * tangent function operation results
     */
    public IMatrix<T> tanh();

    // ========== 统计操作 / Statistical Operations ==========
    /**
     * 矩阵元素标准差 / Matrix element standard deviation
     * <p>
     * 返回矩阵中所有元素的标准差 Returns the standard deviation of all elements in the
     * matrix
     * </p>
     *
     * @return 元素标准差 / Standard deviation of all elements
     */
    public T std();

    /**
     * 矩阵元素方差 / Matrix element variance
     * <p>
     * 返回矩阵中所有元素的方差 Returns the variance of all elements in the matrix
     * </p>
     *
     * @return 元素方差 / Variance of all elements
     */
    public T var();

    /**
     * 计算矩阵行列式 / Compute matrix determinant
     * <p>
     * 计算方阵的行列式值，只适用于方阵 Computes the determinant of a square matrix, only
     * applicable to square matrices
     * </p>
     *
     * @return 行列式值 / Determinant value
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    public T det();

    /**
     * 计算矩阵迹 / Compute matrix trace
     * <p>
     * 计算方阵的迹（对角线元素之和），只适用于方阵 Computes the trace of a square matrix (sum of
     * diagonal elements), only applicable to square matrices
     * </p>
     *
     * @return 矩阵迹 / Matrix trace
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    public T trace();

    /**
     * 计算矩阵条件数 / Compute matrix condition number
     * <p>
     * 计算矩阵的条件数，衡量矩阵的数值稳定性 Computes the condition number of the matrix,
     * measuring numerical stability
     * </p>
     *
     * @return 条件数 / Condition number
     */
    public T cond();

    /**
     * 计算矩阵秩 / Compute matrix rank
     * <p>
     * 计算矩阵的秩（线性无关行或列的最大数目） Computes the rank of the matrix (maximum number of
     * linearly independent rows or columns)
     * </p>
     *
     * @return 矩阵秩 / Matrix rank
     */
    public int rank();

    // ========== 矩阵分解方法 / Matrix Decomposition Methods ==========
    /**
     * LU分解 / LU decomposition
     * <p>
     * 计算矩阵的LU分解，将矩阵A分解为A = LU的形式， 其中L是下三角矩阵，U是上三角矩阵。 Computes the LU
     * decomposition of the matrix, decomposing matrix A into A = LU form, where
     * L is a lower triangular matrix and U is an upper triangular matrix.
     * </p>
     *
     * @return 返回L和U矩阵 / Returns L and U matrices
     */
    public Tuple2<IMatrix<T>, IMatrix<T>> lu();

    /**
     * Cholesky分解 / Cholesky decomposition
     * <p>
     * 计算正定矩阵的Cholesky分解，将矩阵A分解为A = LL^T的形式， 其中L是下三角矩阵。 Computes the Cholesky
     * decomposition of a positive definite matrix, decomposing matrix A into A
     * = LL^T form, where L is a lower triangular matrix.
     * </p>
     *
     * @return Cholesky分解的下三角矩阵L / Lower triangular matrix L from Cholesky
     * decomposition
     * @throws IllegalArgumentException 如果矩阵不是正定矩阵 / if matrix is not positive
     * definite
     */
    public IMatrix<T> cholesky();

    // ========== 线性方程组求解 / Linear System Solving ==========
    /**
     * 求解线性方程组 / Solve linear system
     * <p>
     * 求解线性方程组 Ax = b，其中A是当前矩阵，b是右端向量 Solves the linear system Ax = b, where A
     * is the current matrix and b is the right-hand side vector
     * </p>
     *
     * @param b 右端向量 / Right-hand side vector
     * @return 解向量 x / Solution vector x
     * @throws IllegalArgumentException 如果矩阵和向量维度不匹配 / if matrix and vector
     * dimensions don't match
     * @throws ArithmeticException 如果矩阵不可逆 / if matrix is not invertible
     */
    public IVector<T> solve(IVector<T> b);

    /**
     * 求解线性方程组（矩阵形式） / Solve linear system (matrix form)
     * <p>
     * 求解线性方程组 AX = B，其中A是当前矩阵，B是右端矩阵 Solves the linear system AX = B, where A
     * is the current matrix and B is the right-hand side matrix
     * </p>
     *
     * @param B 右端矩阵 / Right-hand side matrix
     * @return 解矩阵 X / Solution matrix X
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     * @throws ArithmeticException 如果矩阵不可逆 / if matrix is not invertible
     */
    public IMatrix<T> solve(IMatrix<T> B);

    // ========== 高级索引操作 / Advanced Indexing Operations ==========
    /**
     * 矩阵切片操作 / Matrix slice operation
     * <p>
     * 根据行和列的切片表达式对矩阵进行切片操作，支持负数索引 Performs matrix slicing based on row and
     * column slice expressions, supports negative indexing
     * </p>
     *
     * @param rowSlice 行切片表达式，如 "1:3", ":-1", "::2" / Row slice expression, e.g.
     * "1:3", ":-1", "::2"
     * @param colSlice 列切片表达式，如 "0:2", ":-1", "::2" / Column slice expression,
     * e.g. "0:2", ":-1", "::2"
     * @return 切片后的矩阵 / Sliced matrix
     * @throws IllegalArgumentException 如果切片表达式无效 / if slice expressions are
     * invalid
     */
    public IMatrix<T> slice(String rowSlice, String colSlice);

    public default IMatrix<T> slice(int rowStart, int rowEnd, int colStart, int colEnd) {
        return this.subMatrix(rowStart, rowEnd, colStart, colEnd);
    }

    public default IMatrix<T> sliceRows(int rowStart, int rowEnd) {
        return this.subMatrix(rowStart, rowEnd, 0, this.cols());
    }

    public default IMatrix<T> slice(int colStart, int colEnd) {
        return this.subMatrix(0, this.rows(), colStart, colEnd);
    }

    /**
     * 矩阵行切片操作 / Matrix row slice operation
     * <p>
     * 根据行切片表达式对矩阵进行行切片操作，支持负数索引 Performs row slicing based on row slice
     * expression, supports negative indexing
     * </p>
     *
     * @param rowSlice 行切片表达式，如 "1:3", ":-1", "::2" / Row slice expression, e.g.
     * "1:3", ":-1", "::2"
     * @return 切片后的矩阵 / Sliced matrix
     * @throws IllegalArgumentException 如果切片表达式无效 / if slice expression is
     * invalid
     */
    public IMatrix<T> sliceRows(String rowSlice);

    /**
     * 矩阵列切片操作 / Matrix column slice operation
     * <p>
     * 根据列切片表达式对矩阵进行列切片操作，支持负数索引 Performs column slicing based on column slice
     * expression, supports negative indexing
     * </p>
     *
     * @param colSlice 列切片表达式，如 "0:2", ":-1", "::2" / Column slice expression,
     * e.g. "0:2", ":-1", "::2"
     * @return 切片后的矩阵 / Sliced matrix
     * @throws IllegalArgumentException 如果切片表达式无效 / if slice expression is
     * invalid
     */
    public IMatrix<T> sliceColumns(String colSlice);

    /**
     * 花式索引获取矩阵元素 / Fancy indexing for matrix elements
     * <p>
     * 根据行和列索引数组获取对应位置的元素组成新矩阵，支持负数索引 Gets elements at specified row and column
     * positions to form a new matrix, supports negative indexing
     * </p>
     *
     * @param rowIndices 行索引数组 / Array of row indices
     * @param colIndices 列索引数组 / Array of column indices
     * @return 新的矩阵对象，包含指定位置的元素 / New matrix object containing elements at
     * specified positions
     * @throws IndexOutOfBoundsException 如果任何索引超出范围 / if any index is out of
     * bounds
     */
    public IMatrix<T> fancyGet(int[] rowIndices, int[] colIndices);

    // ========== 子矩阵操作 / Submatrix Operations ==========
    /**
     * 获取子矩阵 / Get submatrix
     * <p>
     * 获取指定行列范围的子矩阵 Gets a submatrix within the specified row and column range
     * </p>
     *
     * @param startRow 起始行索引 / Start row index
     * @param endRow 结束行索引（不包含） / End row index (exclusive)
     * @param startCol 起始列索引 / Start column index
     * @param endCol 结束列索引（不包含） / End column index (exclusive)
     * @return 子矩阵 / Submatrix
     * @throws IndexOutOfBoundsException 如果索引超出范围 / if indices are out of bounds
     */
    public IMatrix<T> subMatrix(int startRow, int endRow, int startCol, int endCol);

    /**
     * 设置子矩阵 / Set submatrix
     * <p>
     * 将指定子矩阵设置到当前矩阵的指定位置 Sets the specified submatrix to the current matrix at
     * the specified position
     * </p>
     *
     * @param startRow 起始行索引 / Start row index
     * @param endRow 结束行索引（不包含） / End row index (exclusive)
     * @param startCol 起始列索引 / Start column index
     * @param endCol 结束列索引（不包含） / End column index (exclusive)
     * @param subMatrix 要设置的子矩阵 / Submatrix to set
     * @throws IndexOutOfBoundsException 如果索引超出范围 / if indices are out of bounds
     * @throws IllegalArgumentException 如果子矩阵尺寸不匹配 / if submatrix dimensions
     * don't match
     */
    public IMatrix setSubMatrix(int startRow, int endRow, int startCol, int endCol, IMatrix<T> subMatrix);

    /**
     * 设置指定位置的元素值 (alias for put) / Set element value at specified position
     * <p>
     * 这是put方法的别名，提供更直观的命名 This is an alias for the put method, providing more
     * intuitive naming
     * </p>
     */
    public IMatrix set(int row, int col, T value);

    /**
     * QR算法特征分解的辅助方法 / QR algorithm eigendecomposition auxiliary method
     * <p>
     * 使用QR算法进行特征分解的辅助方法，专门用于Float类型的特征分解辅助方法 Auxiliary method for
     * eigendecomposition using QR algorithm, specialized auxiliary method for
     * Float type eigendecomposition
     * </p>
     * <p>
     * QR算法是一种迭代方法，通过重复应用QR分解来逼近特征值 QR algorithm is an iterative method that
     * approximates eigenvalues by repeatedly applying QR decomposition
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Float> matrix = Linalg.matrix(new float[][]{{4, 1}, {1, 3}});
     * Tuple2<IVector<Float>, IMatrix<Float>> result = matrix.qrEigenDecomposition();
     * IVector<Float> eigenvalues = result.getFirst();
     * IMatrix<Float> eigenvectors = result.getSecond();
     * }</pre>
     * </p>
     *
     * @return 包含特征值和特征向量的元组，第一个元素是特征值向量，第二个元素是特征向量矩阵 Tuple containing
     * eigenvalues and eigenvectors, first element is eigenvalue vector, second
     * element is eigenvector matrix
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     * @throws ArithmeticException 如果矩阵无法进行特征分解 / if matrix cannot be
     * eigendecomposed
     */
    public Tuple2<IVector<T>, IMatrix<T>> qrEigenDecomposition();

    /**
     * 转换为双精度二维数组 / Convert to double 2D array
     * <p>
     * 将矩阵转换为double类型的二维数组，便于与其他库或系统集成 Converts the matrix to a double 2D array
     * for easy integration with other libraries or systems
     * </p>
     * <p>
     * 转换后的数组结构：result[row][col] = matrix.get(row, col) Converted array
     * structure: result[row][col] = matrix.get(row, col)
     * </p>
     *
     * @return 双精度二维数组，行数为矩阵行数，列数为矩阵列数 / Double 2D array with rows equal to
     * matrix rows and columns equal to matrix columns
     * @throws UnsupportedOperationException 如果矩阵包含非数值类型或无法转换 / if matrix
     * contains non-numeric types or cannot be converted
     */
    public double[][] toDoubleArray();

    /**
     * 转换为单精度二维数组 / Convert to float 2D array
     * <p>
     * 将矩阵转换为float类型的二维数组，便于与其他库或系统集成 Converts the matrix to a float 2D array
     * for easy integration with other libraries or systems
     * </p>
     * <p>
     * 转换后的数组结构：result[row][col] = matrix.get(row, col) Converted array
     * structure: result[row][col] = matrix.get(row, col)
     * </p>
     * <p>
     * 注意：如果原矩阵是double类型，转换时可能会有精度损失 Note: If the original matrix is double
     * type, there may be precision loss during conversion
     * </p>
     *
     * @return 单精度二维数组，行数为矩阵行数，列数为矩阵列数 / Float 2D array with rows equal to
     * matrix rows and columns equal to matrix columns
     * @throws UnsupportedOperationException 如果矩阵包含非数值类型或无法转换 / if matrix
     * contains non-numeric types or cannot be converted
     */
    public float[][] toFloatArray();

    /**
     * 转换为整数二维数组 / Convert to integer 2D array
     * <p>
     * 将矩阵转换为int类型的二维数组，便于与其他库或系统集成 Converts the matrix to an int 2D array for
     * easy integration with other libraries or systems
     * </p>
     * <p>
     * 转换后的数组结构：result[row][col] = matrix.get(row, col) Converted array
     * structure: result[row][col] = matrix.get(row, col)
     * </p>
     * <p>
     * 注意：如果原矩阵是浮点类型，转换时会进行截断操作 Note: If the original matrix is floating-point
     * type, truncation will occur during conversion
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.of(new double[][]{{1.7, 2.3}, {3.9, 4.1}});
     * int[][] intArray = matrix.toIntArray();
     * // 结果: [[1, 2], [3, 4]] (浮点数被截断)
     * }</pre>
     * </p>
     *
     * @return 整数二维数组，行数为矩阵行数，列数为矩阵列数 / Integer 2D array with rows equal to
     * matrix rows and columns equal to matrix columns
     * @throws UnsupportedOperationException 如果矩阵包含非数值类型或无法转换 / if matrix
     * contains non-numeric types or cannot be converted
     */
    public int[][] toIntArray();

    /**
     * 矩阵元素映射操作 / Matrix element mapping operation
     * <p>
     * 对矩阵中的每个元素应用指定的函数，返回同形状的新矩阵 Applies the specified function to each element
     * in the matrix, returns a new matrix of the same shape
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1, 2}, {3, 4}});
     * IMatrix<Double> squared = matrix.apply(x -> x * x);  // 结果: [[1, 4], [9, 16]]
     * IMatrix<Double> doubled = matrix.apply(x -> x * 2);  // 结果: [[2, 4], [6, 8]]
     * }</pre>
     * </p>
     *
     * @param fun 要应用的函数，接受一个元素并返回转换后的元素 / Function to apply, takes an element
     * and returns transformed element
     * @return 新的矩阵对象，包含应用函数后的结果 / New matrix object containing results after
     * applying the function
     * @throws IllegalArgumentException 如果fun为null / if fun is null
     */
    public IMatrix<T> apply(Function<T, T> fun);

    /**
     * 矩阵元素映射操作（别名） / Matrix element mapping operation (alias)
     * <p>
     * apply函数的别名，提供更直观的命名 Alias for apply function, providing more intuitive
     * naming
     * </p>
     * <p>
     * 对矩阵中的每个元素应用指定的函数，返回同形状的新矩阵 Applies the specified function to each element
     * in the matrix, returns a new matrix of the same shape
     * </p>
     *
     * @param fun 要应用的函数，接受一个元素并返回转换后的元素 / Function to apply, takes an element
     * and returns transformed element
     * @return 新的矩阵对象，包含应用函数后的结果 / New matrix object containing results after
     * applying the function
     * @throws IllegalArgumentException 如果fun为null / if fun is null
     * @see #apply(Function) 实际实现方法 / Actual implementation method
     */
    public default IMatrix<T> map(Function<T, T> fun) {
        return this.apply(fun);
    }

    /**
     * 将矩阵数据保存到本地指定路径 / Save matrix data to specified local path
     * <p>
     * 将矩阵数据以特定格式保存到本地文件系统，便于数据持久化和后续加载 Saves matrix data in a specific format to the local file system for data persistence and subsequent loading
     * </p>
     * <p>
     * 支持的文件格式通常包括CSV、二进制格式等，具体格式由实现决定 Supported file formats typically include CSV, binary formats, etc., with specific format determined by implementation
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
     * matrix.save("data/matrix.csv");  // 保存为CSV格式
     * matrix.save("data/matrix.bin");  // 保存为二进制格式
     * }</pre>
     * </p>
     *
     * @param path 保存路径，包括文件名和扩展名 / Save path including filename and extension
     * @throws IllegalArgumentException 如果路径为null或无效 / if path is null or invalid
     * @throws IOException 如果文件写入失败 / if file writing fails
     * @throws SecurityException 如果没有写入权限 / if write permission is denied
     */
    public void save(String path);

    /**
     * 计算矩阵的L1范数 / Compute L1 norm of the matrix
     * <p>
     * L1范数是矩阵所有元素绝对值的和，也称为曼哈顿范数 The L1 norm is the sum of absolute values of all matrix elements, also known as Manhattan norm
     * </p>
     * <p>
     * 公式：||A||₁ = Σᵢⱼ |aᵢⱼ| Formula: ||A||₁ = Σᵢⱼ |aᵢⱼ|
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.of(new double[][]{{-1, 2}, {3, -4}});
     * Double l1Norm = matrix.norm1();  // 结果: 10.0 (1+2+3+4)
     * }</pre>
     * </p>
     *
     * @return L1范数值 / L1 norm value
     */
    public default T norm1() {
        return this.abs().sum();
    }

    /**
     * 设置矩阵的对角线元素 / Set the diagonal elements of the matrix
     * <p>
     * 将矩阵的主对角线元素设置为给定向量的元素 Sets the main diagonal elements of the matrix to the elements of the given vector
     * </p>
     * <p>
     * 对于m×n矩阵，对角线长度为min(m,n) For an m×n matrix, the diagonal length is min(m,n)
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.zeros(3, 3);
     * IVector<Double> diagonal = IVector.of(new double[]{1, 2, 3});
     * matrix.setDiag(diagonal);
     * // 结果矩阵: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
     * }</pre>
     * </p>
     *
     * @param diagonal 对角线向量，长度应不超过min(rows, cols) / Diagonal vector, length should not exceed min(rows, cols)
     * @return 修改后的矩阵（就地操作） / Modified matrix (in-place operation)
     * @throws IllegalArgumentException 如果diagonal为null或长度超过对角线长度 / if diagonal is null or length exceeds diagonal length
     */
    public IMatrix<T> setDiag(IVector<T> diagonal);

    /**
     * 在矩阵的列上与向量进行广播运算 / Broadcast operation between matrix columns and vector
     * <p>
     * 将给定向量与矩阵的每一列进行指定的运算操作 Performs specified operation between the given vector and each column of the matrix
     * </p>
     * <p>
     * 广播规则：向量长度必须等于矩阵的行数 Broadcasting rule: vector length must equal matrix row count
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
     * IVector<Double> vector = IVector.of(new double[]{10, 20});
     * IMatrix<Double> result = matrix.broadcastColumn(vector, (col, vec) -> col.add(vec));
     * // 结果: [[11, 12], [23, 24]]
     * }</pre>
     * </p>
     *
     * @param colVector 用于广播的向量，长度必须等于矩阵行数 / Vector for broadcasting, length must equal matrix row count
     * @param fun 双参数函数，定义列向量与广播向量的运算规则 / Binary function defining operation between column vector and broadcast vector
     * @return 广播运算后的新矩阵 / New matrix after broadcast operation
     * @throws IllegalArgumentException 如果向量长度与矩阵行数不匹配或fun为null / if vector length doesn't match matrix row count or fun is null
     */
    public IMatrix<T> broadcastColumn(IVector<T> colVector, BiFunction<IVector<T>, IVector<T>, IVector<T>> fun);

    /**
     * 在矩阵的行上与向量进行广播运算 / Broadcast operation between matrix rows and vector
     * <p>
     * 将给定向量与矩阵的每一行进行指定的运算操作 Performs specified operation between the given vector and each row of the matrix
     * </p>
     * <p>
     * 广播规则：向量长度必须等于矩阵的列数 Broadcasting rule: vector length must equal matrix column count
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
     * IVector<Double> vector = IVector.of(new double[]{10, 20});
     * IMatrix<Double> result = matrix.broadcastRow(vector, (row, vec) -> row.add(vec));
     * // 结果: [[11, 22], [13, 24]]
     * }</pre>
     * </p>
     *
     * @param colVector 用于广播的向量，长度必须等于矩阵列数 / Vector for broadcasting, length must equal matrix column count
     * @param fun 双参数函数，定义行向量与广播向量的运算规则 / Binary function defining operation between row vector and broadcast vector
     * @return 广播运算后的新矩阵 / New matrix after broadcast operation
     * @throws IllegalArgumentException 如果向量长度与矩阵列数不匹配或fun为null / if vector length doesn't match matrix column count or fun is null
     */
    public IMatrix<T> broadcastRow(IVector<T> colVector, BiFunction<IVector<T>, IVector<T>, IVector<T>> fun);

    /**
     * 矩阵列向量广播加法 / Matrix column vector broadcast addition
     * <p>
     * 将向量与矩阵的每一列进行加法运算 Performs addition between the vector and each column of the matrix
     * </p>
     *
     * @param colVector 用于加法的向量 / Vector for addition
     * @return 加法运算后的新矩阵 / New matrix after addition
     */
    default IMatrix<T> broadcastAddColumn(IVector<T> colVector) {
        return this.broadcastColumn(colVector, (v1, v2) -> {
            return v1.add(v2);
        });
    }
    
    /**
     * 矩阵行向量广播加法 / Matrix row vector broadcast addition
     * <p>
     * 将向量与矩阵的每一行进行加法运算 Performs addition between the vector and each row of the matrix
     * </p>
     *
     * @param colVector 用于加法的向量 / Vector for addition
     * @return 加法运算后的新矩阵 / New matrix after addition
     */
    default IMatrix<T> broadcastAddRow(IVector<T> colVector) {
        return this.broadcastColumn(colVector, (v1, v2) -> {
            return v1.add(v2);
        });
    }
    
    /**
     * 矩阵列向量广播减法 / Matrix column vector broadcast subtraction
     * <p>
     * 将矩阵的每一列减去向量 Subtracts the vector from each column of the matrix
     * </p>
     *
     * @param colVector 用于减法的向量 / Vector for subtraction
     * @return 减法运算后的新矩阵 / New matrix after subtraction
     */
    default IMatrix<T> broadcastSubColumn(IVector<T> colVector) {
        return this.broadcastColumn(colVector, (v1, v2) -> {
            return v1.sub(v2);
        });
    }
    
    /**
     * 矩阵行向量广播减法 / Matrix row vector broadcast subtraction
     * <p>
     * 将矩阵的每一行减去向量 Subtracts the vector from each row of the matrix
     * </p>
     *
     * @param colVector 用于减法的向量 / Vector for subtraction
     * @return 减法运算后的新矩阵 / New matrix after subtraction
     */
    default IMatrix<T> broadcastSubRow(IVector<T> colVector) {
        return this.broadcastColumn(colVector, (v1, v2) -> {
            return v1.sub(v2);
        });
    }
    
    /**
     * 矩阵列向量广播乘法（默认） / Matrix column vector broadcast multiplication (default)
     * <p>
     * 将向量与矩阵的每一列进行逐元素乘法运算 Performs element-wise multiplication between the vector and each column of the matrix
     * </p>
     *
     * @param colVector 用于乘法的向量 / Vector for multiplication
     * @return 乘法运算后的新矩阵 / New matrix after multiplication
     */
    default IMatrix<T> broadcastColumn(IVector<T> colVector) {
        return this.broadcastColumn(colVector, (v1, v2) -> {
            return v1.multiply(v2);
        });
    }
    
    /**
     * 矩阵列向量广播乘法 / Matrix column vector broadcast multiplication
     * <p>
     * 将向量与矩阵的每一列进行逐元素乘法运算 Performs element-wise multiplication between the vector and each column of the matrix
     * </p>
     *
     * @param colVector 用于乘法的向量 / Vector for multiplication
     * @return 乘法运算后的新矩阵 / New matrix after multiplication
     */
    default IMatrix<T> broadcastMultipyColumn(IVector<T> colVector) {
        return this.broadcastColumn(colVector, (v1, v2) -> {
            return v1.multiply(v2);
        });
    }
    
    /**
     * 矩阵列向量广播除法 / Matrix column vector broadcast division
     * <p>
     * 将矩阵的每一列除以向量 Divides each column of the matrix by the vector
     * </p>
     *
     * @param colVector 用于除法的向量 / Vector for division
     * @return 除法运算后的新矩阵 / New matrix after division
     */
    default IMatrix<T> broadcastDivideColumn(IVector<T> colVector) {
        return this.broadcastColumn(colVector, (v1, v2) -> {
            return v1.divide(v2);
        });
    }
    
    /**
     * 矩阵行向量广播除法 / Matrix row vector broadcast division
     * <p>
     * 将矩阵的每一行除以向量 Divides each row of the matrix by the vector
     * </p>
     *
     * @param colVector 用于除法的向量 / Vector for division
     * @return 除法运算后的新矩阵 / New matrix after division
     */
    default IMatrix<T> broadcastDivideRow(IVector<T> colVector) {
        return this.broadcastColumn(colVector, (v1, v2) -> {
            return v1.divide(v2);
        });
    }
    
    /**
     * 逐元素乘法 / Element-wise multiplication
     * <p>对两个矩阵进行逐元素乘法运算，返回新的矩阵。</p>
     * <p>Performs element-wise multiplication of two matrices and returns a new matrix.</p>
     * 
     * <p>使用示例 / Usage example:</p>
     * <pre>{@code
     * IMatrix<Double> a = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
     * IMatrix<Double> b = IMatrix.of(new double[][]{{2, 3}, {4, 5}});
     * IMatrix<Double> result = a.multiply(b); // {{2, 6}, {12, 20}}
     * }</pre>
     * 
     * @param other 另一个矩阵 / The other matrix
     * @return 逐元素乘法的结果矩阵 / The result matrix of element-wise multiplication
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / If matrix dimensions don't match
     */
    public IMatrix<T> multiply(IMatrix<T> other);
    
    
    /**
     * 矩阵与向量的逐元素相等比较 / Element-wise equality comparison between matrix and vector
     * <p>将矩阵的每个元素与向量进行相等比较，返回布尔矩阵。</p>
     * <p>Compares each element of the matrix with the vector for equality and returns a boolean matrix.</p>
     * 
     * <p>使用示例 / Usage example:</p>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
     * IVector<Double> vector = IVector.of(new double[]{2, 3});
     * boolean[][] result = matrix.equals(vector);
     * }</pre>
     * 
     * @param other 比较的向量 / The vector to compare with
     * @return 相等比较的布尔矩阵 / Boolean matrix of equality comparison results
     */
    public boolean[][] equals(IVector<T> other);

    /**
     * 矩阵与向量的逐元素小于比较 / Element-wise less-than comparison between matrix and vector
     * <p>将矩阵的每个元素与向量进行小于比较，返回布尔矩阵。</p>
     * <p>Compares each element of the matrix with the vector for less-than and returns a boolean matrix.</p>
     * 
     * <p>使用示例 / Usage example:</p>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
     * IVector<Double> vector = IVector.of(new double[]{2, 3});
     * boolean[][] result = matrix.lessThan(vector);
     * }</pre>
     * 
     * @param other 比较的向量 / The vector to compare with
     * @return 小于比较的布尔矩阵 / Boolean matrix of less-than comparison results
     */
    public boolean[][] lessThan(IVector<T> other);

    /**
     * 矩阵与向量的逐元素大于比较 / Element-wise greater-than comparison between matrix and vector
     * <p>将矩阵的每个元素与向量进行大于比较，返回布尔矩阵。</p>
     * <p>Compares each element of the matrix with the vector for greater-than and returns a boolean matrix.</p>
     * 
     * <p>使用示例 / Usage example:</p>
     * <pre>{@code
     * IMatrix<Double> matrix = IMatrix.of(new double[][]{{1, 2}, {3, 4}});
     * IVector<Double> vector = IVector.of(new double[]{2, 3});
     * boolean[][] result = matrix.greaterThan(vector);
     * }</pre>
     * 
     * @param other 比较的向量 / The vector to compare with
     * @return 大于比较的布尔矩阵 / Boolean matrix of greater-than comparison results
     */
    public boolean[][] greaterThan(IVector<T> other);
    
    
    /**
     * 创建下三角阵
     * @param m 矩阵大小
     * @return 下三角矩阵
     */
    public static IMatrix<Double> lowerTriMatrix(int m){
        // 创建一个m×m的零矩阵
        double[][] data = new double[m][m];
        
        // 填充下三角部分（包括对角线）
        for (int i = 0; i < m; i++) {
            for (int j = 0; j <= i; j++) {
                data[i][j] = 1.0;
            }
        }
        
        return IMatrix.of(data);
    }
    
    
    /**
     * 创建上三角阵
     * @param m 矩阵大小
     * @return 上三角矩阵
     */
    public static IMatrix<Double> upperTriMatrix(int m){
        // 创建一个m×m的零矩阵
        double[][] data = new double[m][m];
        
        // 填充上三角部分（包括对角线）
        for (int i = 0; i < m; i++) {
            for (int j = i; j < m; j++) {
                data[i][j] = 1.0;
            }
        }
        
        return IMatrix.of(data);
    }
    
    /**
     * 将矩阵无条件转换为Float类型
     * @return 
     */
    public default IMatrix<Float> toFloatMatrix(){
        int rows = this.getRowNum();
        int cols = this.getColNum();
        
        // Create a new float array with the same dimensions
        float[][] floatData = new float[rows][cols];
        
        // Copy data from the current matrix, converting each element to float
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                floatData[i][j] = this.get(i, j).floatValue();
            }
        }
        
        // Create and return a new Float matrix
        return IMatrix.of(floatData);
    }
    
    /**
     * 将矩阵无条件转换为Double类型
     * @return 
     */
    public default IMatrix<Double> toDoubleMatrix(){
        int rows = this.getRowNum();
        int cols = this.getColNum();
        
        // Create a new double array with the same dimensions
        double[][] doubleData = new double[rows][cols];
        
        // Copy data from the current matrix, converting each element to double
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                doubleData[i][j] = this.get(i, j).doubleValue();
            }
        }
        
        // Create and return a new Double matrix
        return IMatrix.of(doubleData);
    }
    
    
}
