package com.reremouse.lab.math.linalg;

import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import java.util.List;

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
    public static <T extends Number> IMatrix<T> of(double[][] data) {
        return (IMatrix<T>) IDoubleMatrix.of(data);
    }

    public static <T extends Number> IMatrix<T> of(Double[][] data) {
        return (IMatrix<T>) IDoubleMatrix.of(data);
    }

    public static <T extends Number> IMatrix<T> of(float[][] data) {
        return (IMatrix<T>) IFloatMatrix.of(data);
    }

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
            throw new UnsupportedOperationException("Double类型尚未实现 / Double type not implemented yet");
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

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
            throw new UnsupportedOperationException("Double类型尚未实现 / Double type not implemented yet");
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

    static <T extends Number> IMatrix<T> diag(T[] diagonal) {
        Class type = Double.class;
        return diag(diagonal, type);
    }

    static <T extends Number> IMatrix<T> diag(IVector<T> diagonal, Class<T> type) {
        if (type == Float.class) {
            return (IMatrix<T>) IFloatMatrix.diag(diagonal.toFloatArray());
        } else if (type == Double.class) {
            return (IMatrix<T>) IDoubleMatrix.diag(diagonal.toDoubleArray());
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

    static <T extends Number> IMatrix<T> diag(IVector<T> diagonal) {
        return diag(diagonal.toDoubleArray());
    }

    static <T extends Number> IMatrix<T> diag(double[] diagonal) {
        Class type = Double.class;
        return diag(RereMathUtil.toClassArray(diagonal), type);
    }

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
            throw new UnsupportedOperationException("Double类型尚未实现 / Double type not implemented yet");
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

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
            throw new UnsupportedOperationException("Double类型尚未实现 / Double type not implemented yet");
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName());
        }
    }

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
     */
    public IMatrix<T> sub(IMatrix<T> other);

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
    public IMatrix<T> mmul(T scalar);

    public IMatrix<T> divide(IMatrix<T> other);

    public T dot(IMatrix<T> other);

    /**
     * 矩阵乘法运算 / Matrix multiplication
     * <p>
     * 计算两个矩阵的乘积，要求第一个矩阵的列数等于第二个矩阵的行数 Computes the product of two matrices,
     * requires the number of columns of the first matrix to equal the number of
     * rows of the second matrix
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 矩阵乘法结果 / Matrix multiplication result
     * @throws IllegalArgumentException 如果矩阵维度不匹配进行乘法运算 / if matrix dimensions
     * don't match for multiplication
     */
    public IMatrix<T> mmul(IMatrix<T> other);

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
     *
     * @return
     */
    public int rows();

    /**
     * 获取矩阵列数 / Get number of columns (alias for getColNum)
     *
     * @return
     */
    public int cols();

    public IMatrix<T> transposeInPlace();

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
     * 特征分解，返回的特征值按大小排列，返回的矩阵的列为各个特征向量，与特征值的顺序对应
     *
     * @return 返回特征向量和特征值，其中返回的矩阵的列为各个特征向量，返回的向量中包含所有特征值
     */
    public Tuple2<IVector<T>, IMatrix<T>> eigen();

    /**
     * 奇异值分解
     *
     * @return 返回U、奇异值向量、V^T / Returns U, singular values vector, and V^T
     */
    public Tuple3<IMatrix<T>, IVector<T>, IMatrix<T>> svd();

    /**
     * QR分解
     *
     * @return 返回Q和R矩阵 / Returns Q and R matrices
     */
    public Tuple2<IMatrix<T>, IMatrix<T>> qr();

    /**
     * 求解矩阵的逆 / Matrix inverse
     *
     * @return 逆矩阵 / Inverse matrix
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     * @throws ArithmeticException 如果矩阵是奇异的（不可逆） / if matrix is singular
     * (non-invertible)
     */
    public IMatrix<T> inv();

    /**
     * 求解矩阵的伪逆 / Matrix pseudo-inverse
     *
     * @return 伪逆矩阵 / Pseudo-inverse matrix
     */
    public IMatrix<T> pinv();

    // ========== 行列操作方法 / Row and Column Operations ==========
    /**
     * 计算行求和 / Calculate row sums
     *
     * @return
     */
    public IVector<T> rowSums();

    /**
     * 计算行均值 / Calculate row means
     */
    public IVector<T> rowMeans();

    /**
     * 计算列求和 / Calculate column sums
     */
    public IVector<T> colSums();

    /**
     * 计算列均值 / Calculate column means
     */
    public IVector<T> colMeans();

    /**
     * 获取指定列向量 / Get specified column vector
     */
    public IVector<T> getColumn(int i);

    public IMatrix<T> getColumnAsCloumnVector(int i);

    /**
     * 获取指定行向量 / Get specified row vector
     */
    public IVector<T> getRow(int i);

    /**
     * 获取指定列矩阵 / Get specified column as matrix
     */
    public IMatrix<T> getColumnMatrix(int colIndex);

    /**
     * 设置指定列 / Set specified column
     */
    public IMatrix putColumn(int colIndex, IMatrix<T> column);
    
    
    public IMatrix setColumn(int colIndex, IVector<T> column);

    public IMatrix setRow(int rowIndex, IVector<T> column);

    
    
    /**
     * 获取多个指定列 / Get multiple specified columns
     */
    public IVector<T>[] getColumns(int[] indices);

    // ========== 元素操作 / Element Operations ==========
    /**
     * 矩阵平方根运算 / Matrix square root operation
     */
    public IMatrix<T> sqrt();

    /**
     * 矩阵幂运算 / Matrix power operation
     */
    public IMatrix<T> pow(T power);

    /**
     * 矩阵指数运算 / Matrix exponential operation
     */
    public IMatrix<T> exp();

    /**
     * 矩阵自然对数运算 / Matrix natural logarithm operation
     */
    public IMatrix<T> log();

    /**
     * 计算Frobenius范数 / Compute Frobenius norm
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

    public boolean isSymmetric();

    public boolean isPositiveDefinite();

    // 便利转换方法
    public IVector<T> flatten();         // 展平为向量

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
     * 专门用于Float类型的QR算法特征分解辅助方法 Specialized QR algorithm eigendecomposition
     * auxiliary method for Float type
     * </p>
     *
     * @return 特征值和特征向量 / Eigenvalues and eigenvectors
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

    public int[][] toIntArray();
}
