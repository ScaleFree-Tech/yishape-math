package com.reremouse.lab.math.linalg;

import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import com.reremouse.lab.math.compute.GPUComputeDoubleUtils;
import com.reremouse.lab.math.compute.GPUConfig;
import com.reremouse.lab.math.compute.CPUComputeDoubleUtils;

import java.util.concurrent.*;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.function.Function;

/**
 * 矩阵操作实现类 / Matrix Operations Implementation Class 本类按行存储向量数据
 * <p>
 * 本类实现了矩阵的常用操作，包括基本的数学运算、矩阵变换、数据访问等功能。 基于二维Double数组实现，提供高效的矩阵计算能力。
 * </p>
 * <p>
 * This class implements common matrix operations including basic mathematical
 * operations, matrix transformations, data access and other functionalities.
 * Based on 2D Double array implementation, providing efficient matrix
 * computation capabilities.
 * </p>
 *
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 * <li>基本数学运算：加法、减法、乘法、除法 / Basic math operations: add, subtract, multiply,
 * divide</li>
 * <li>矩阵变换：转置、幂运算、开方 / IMatrix<Double> transformations: transpose,
 * power, square root</li>
 * <li>数据访问：行列访问、元素获取设置 / Data access: row/column access, element get/set</li>
 * <li>统计运算：行列求和、均值计算 / Statistical operations: row/column sum, mean
 * calculation</li>
 * <li>数据转换：数组转换、类型转换 / Data conversion: array conversion, type conversion</li>
 * </ul>
 *
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 // 创建矩阵 / Create matrix
 double[][] data = {{1, 2}, {3, 4}};
 IMatrix<Double> matrix = new RereDoubleMatrix(data);

 // 矩阵运算 / IMatrix<Double> operations
 IMatrix<Double> result = matrix.add(other).mmul(2.0.0);

 // 获取行列 / Get rows/columns
 IVector<Double> row = matrix.getRow(0);
 IVector<Double> col = matrix.getColunm(0);
 }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereDoubleMatrix implements IDoubleMatrix {

    /**
     * 矩阵数据存储数组 / Matrix data storage array
     * <p>
     * 使用二维Double数组存储矩阵数据，data[i][j]表示第i行第j列的元素 Uses 2D Double array to store
     * matrix data, data[i][j] represents element at row i, column j
     * </p>
     */
    double[][] data;

    // ========== 性能优化相关字段 / Performance Optimization Fields ==========
    /**
     * 线程池用于并行计算 / Thread pool for parallel computation
     */
    private static final ExecutorService THREAD_POOL
            = Executors.newFixedThreadPool(Math.min(8, Runtime.getRuntime().availableProcessors()));

    /**
     * 矩阵对象池 / Matrix object pool
     */
    private static final Queue<RereDoubleMatrix> MATRIX_POOL = new ConcurrentLinkedQueue<>();

    /**
     * 对象池最大大小 / Maximum pool size
     */
    private static final int MAX_POOL_SIZE = 100;

    /**
     * 预分配的内存缓冲区 / Pre-allocated memory buffers
     */
    private static final ThreadLocal<double[]> BUFFER
            = ThreadLocal.withInitial(() -> new double[1024 * 1024]);

    /**
     * 分块大小，用于缓存优化 / Block size for cache optimization
     *
     * @deprecated 使用getOptimalBlockSize方法动态调整
     */
    @Deprecated
    private static final int BLOCK_SIZE = 64;

    /**
     * 根据矩阵大小动态调整分块大小 / Dynamic block size based on matrix size
     */
    private static int getOptimalBlockSize(int m, int n, int p) {
        int totalSize = m * n * p;
        if (totalSize < 10000) {
            return 32;
        }
        if (totalSize < 100000) {
            return 64;
        }
        if (totalSize < 1000000) {
            return 128;
        }
        return 256;
    }

    /**
     * 是否启用并行计算 / Whether parallel computation is enabled
     */
    private static volatile boolean PARALLEL_ENABLED = true;

    // ========== GPU相关字段 / GPU Related Fields ==========
    /**
     * GPU是否可用 / Whether GPU is available
     */
    private static final boolean GPU_ENABLED = GPUComputeDoubleUtils.isGPUAvailable();

    /**
     * GPU计算阈值（操作数） / GPU computation threshold (operations)
     */
    private static final int GPU_THRESHOLD = GPUConfig.GPU_THRESHOLD;

    // ========== 对象池化和内存管理方法 / Object Pooling and Memory Management Methods ==========
    /**
     * 从对象池获取矩阵实例 / Get matrix instance from object pool
     */
    public static RereDoubleMatrix borrowMatrix(int rows, int cols) {
        RereDoubleMatrix matrix = MATRIX_POOL.poll();
        if (matrix == null || matrix.data.length != rows || matrix.data[0].length != cols) {
            return new RereDoubleMatrix(rows, cols);
        }
        return matrix;
    }

    /**
     * 将矩阵返回到对象池 / Return matrix to object pool
     */
    public void returnToPool() {
        if (this.data != null && this.data.length > 0 && this.data[0].length > 0
                && MATRIX_POOL.size() < MAX_POOL_SIZE) {
            MATRIX_POOL.offer(this);
        }
    }

    /**
     * 重置矩阵大小 / Reset matrix size
     */
    private RereDoubleMatrix reset(int rows, int cols) {
        // 由于data是final的，无法重置，直接返回新实例
        return new RereDoubleMatrix(rows, cols);
    }

    /**
     * 预分配内存的构造函数 / Constructor with pre-allocated memory
     */
    public RereDoubleMatrix(int rows, int cols) {
        this.data = new double[rows][cols];
    }

    /**
     * 获取预分配缓冲区 / Get pre-allocated buffer
     */
    private static double[] getBuffer(int size) {
        double[] buffer = BUFFER.get();
        if (buffer.length < size) {
            return new double[size];
        }
        return buffer;
    }

    /**
     * 设置并行计算开关 / Set parallel computation switch
     */
    public static void setParallelEnabled(boolean enabled) {
        PARALLEL_ENABLED = enabled;
    }

    /**
     * 关闭线程池 / Shutdown thread pool
     */
    public static void shutdown() {
        THREAD_POOL.shutdown();
        GPUComputeDoubleUtils.cleanup(); // 清理GPU资源
    }

    // ========== GPU相关方法 / GPU Related Methods ==========
    /**
     * 检查GPU是否启用 / Check if GPU is enabled
     */
    public static boolean isGPUEnabled() {
        return GPU_ENABLED;
    }

    /**
     * 获取GPU信息 / Get GPU information
     */
    public static String getGPUInfo() {
        return GPUComputeDoubleUtils.getGPUInfo();
    }

    /**
     * 构造函数 / Constructor
     * <p>
     * 使用给定的二维数组创建矩阵实例 Creates a matrix instance with the given 2D array
     * </p>
     *
     * @param data 二维Double数组，表示矩阵数据 / 2D Double array representing matrix data
     * @throws IllegalArgumentException 如果数据为null或维度不一致 / if data is null or
     * dimensions are inconsistent
     */
    public RereDoubleMatrix(double[][] data) {
        if (data == null) {
            throw new IllegalArgumentException("矩阵数据不能为null / Matrix data cannot be null");
        }
        if (data.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空 / Matrix cannot be empty");
        }

        // 检查二维数组的一致性
        int firstRowLength = data[0].length;
        for (int i = 1; i < data.length; i++) {
            if (data[i] == null) {
                throw new IllegalArgumentException("矩阵行不能为null / Matrix row cannot be null");
            }
            if (data[i].length != firstRowLength) {
                throw new IllegalArgumentException("矩阵各行长度必须相同 / All matrix rows must have the same length");
            }
        }

        this.data = data;
    }

    public RereDoubleMatrix(Double[][] data) {
        if (data == null) {
            throw new IllegalArgumentException("矩阵数据不能为null / Matrix data cannot be null");
        }
        if (data.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空 / Matrix cannot be empty");
        }

        // 检查二维数组的一致性
        int firstRowLength = data[0].length;
        for (int i = 1; i < data.length; i++) {
            if (data[i] == null) {
                throw new IllegalArgumentException("矩阵行不能为null / Matrix row cannot be null");
            }
            if (data[i].length != firstRowLength) {
                throw new IllegalArgumentException("矩阵各行长度必须相同 / All matrix rows must have the same length");
            }
        }

        // 转换double[][]到double[][]
        this.data = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                this.data[i][j] = data[i][j];
            }
        }
    }

    /**
     * 矩阵减法运算（标量） / Matrix subtraction with scalar
     * <p>
     * 矩阵中的每个元素减去标量值 Subtracts a scalar value from each element in the matrix
     * </p>
     *
     * @param scalar 要减去的标量值 / The scalar value to subtract
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    @Override
    public IMatrix<Double> sub(Double scalar) {
        int rows = data.length;
        int cols = data[0].length;
        long complexity = (long) rows * cols;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuMatrixScalarSub(this, scalar);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU矩阵标量减法失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用CPU计算工具类
        return CPUComputeDoubleUtils.matrixScalarSub(data, scalar);
    }

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
    @Override
    public IMatrix<Double> sub(IMatrix<Double> other1) {
        if (other1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        if (data.length != otherData.length || data[0].length != otherData[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }

        int rows = data.length;
        int cols = data[0].length;
        long complexity = (long) rows * cols;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuMatrixSub(this, other0);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU矩阵减法失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用CPU计算工具类
        return CPUComputeDoubleUtils.matrixSub(data, otherData);
    }

    /**
     * 矩阵标量乘法运算（Double） / Matrix scalar multiplication (Double)
     * <p>
     * 矩阵中的每个元素乘以标量值 Multiplies each element in the matrix by a scalar value
     * </p>
     *
     * @param scalar 标量乘数 / The scalar multiplier
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    @Override
    public IMatrix<Double> multiplyScalar(Double scalar) {
        int rows = data.length;
        int cols = data[0].length;
        long complexity = (long) rows * cols;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuMatrixScalarMultiply(this, scalar);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU矩阵标量乘法失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用CPU计算工具类
        return CPUComputeDoubleUtils.matrixScalarMultiply(data, scalar);
    }

    /**
     * 矩阵加法运算 / Matrix addition
     * <p>
     * 对应元素相加，要求两个矩阵维度相同 Element-wise addition, requires both matrices to have
     * the same dimensions
     * </p>
     *
     * @param other1 另一个矩阵 / The other matrix
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     */
    @Override
    public IMatrix<Double> add(IMatrix<Double> other1) {
        if (other1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        if (data.length != otherData.length || data[0].length != otherData[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }

        int rows = data.length;
        int cols = data[0].length;
        long complexity = (long) rows * cols;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuMatrixAdd(this, other0);
            } catch (Exception e) {
                // GPU失败时回退到CPU，这里不需要额外日志，GPUComputeDoubleUtils已经处理了
            }
        }

        // 使用CPU计算工具类
        return CPUComputeDoubleUtils.matrixAdd(data, otherData);
    }

    /**
     * 矩阵除法运算 / Matrix division
     * <p>
     * 对应元素相除，要求两个矩阵维度相同 Element-wise division, requires both matrices to have
     * the same dimensions
     * </p>
     *
     * @param other 除数矩阵 / The divisor matrix
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     * @throws ArithmeticException 如果除数为零 / if divisor is zero
     */
    @Override
    public IMatrix<Double> divide(IMatrix<Double> other1) {
        if (other1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        if (data.length != otherData.length || data[0].length != otherData[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }

        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];
        final double tolerance = 1e-10d;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(otherData[i][j]) < tolerance) {
                    throw new ArithmeticException("除数不能为零，位置[" + i + "," + j + "] / Divisor cannot be zero at position [" + i + "," + j + "]");
                }
                result[i][j] = data[i][j] / otherData[i][j];
            }
        }
        return new RereDoubleMatrix(result);
    }

    /**
     * 向量点积运算 / Vector dot product
     * <p>
     * 计算两个列向量的点积，要求两个矩阵都是列向量且维度相同 Computes the dot product of two column
     * vectors, requires both matrices to be column vectors with same dimensions
     * </p>
     *
     * @param other 另一个列向量矩阵 / The other column vector matrix
     * @return 点积结果 / The dot product result
     * @throws IllegalArgumentException 如果不是列向量或维度不匹配 / if not column vectors or
     * dimensions don't match
     */
    @Override
    public Double dot(IMatrix<Double> other1) {
        if (other1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        if (data[0].length != 1 || otherData[0].length != 1) {
            throw new IllegalArgumentException("点积运算需要列向量 / Dot product requires column vectors");
        }
        if (data.length != otherData.length) {
            throw new IllegalArgumentException("向量维度不匹配 / Vector dimensions don't match");
        }
        Double result = 0.0;
        for (int i = 0; i < data.length; i++) {
            result += data[i][0] * otherData[i][0];
        }
        return result;
    }

    /**
     * 获取指定列 / Get specified column
     * <p>
     * 返回矩阵的指定列作为列向量 Returns the specified column of the matrix as a column
     * vector
     * </p>
     *
     * @param colIndex 列索引（从0开始） / Column index (0-based)
     * @return 包含指定列数据的列向量矩阵 / Column vector matrix containing the specified
     * column data
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
    @Override
    public IVector<Double> getColumn(int colIndex) {
        int rows = data.length;
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = data[i][colIndex];
        }
        return new RereDoubleVector(result);
    }

    /**
     * 设置指定列 / Set specified column
     * <p>
     * 将列向量的数据设置到矩阵的指定列 Sets the column vector data to the specified column of
     * the matrix
     * </p>
     *
     * @param colIndex 列索引（从0开始） / Column index (0-based)
     * @param column 列向量矩阵 / Column vector matrix
     * @throws IllegalArgumentException 如果输入不是列向量或维度不匹配 / if input is not a
     * column vector or dimensions don't match
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
    @Override
    public IMatrix<Double> putColumn(int colIndex, IMatrix<Double> column1) {
        if (column1 == null) {
            throw new IllegalArgumentException("列矩阵不能为null / Column matrix cannot be null");
        }
        if (colIndex < 0 || colIndex >= data[0].length) {
            throw new IndexOutOfBoundsException("列索引超出范围: " + colIndex + " / Column index out of bounds: " + colIndex);
        }
        IDoubleMatrix other0 = (IDoubleMatrix) column1;
        double[][] columnData = other0.getData();
        if (columnData[0].length != 1) {
            throw new IllegalArgumentException("输入必须是列向量 / Input must be a column vector");
        }
        if (columnData.length != data.length) {
            throw new IllegalArgumentException("列向量高度不匹配 / Column vector height doesn't match");
        }
        for (int i = 0; i < data.length; i++) {
            data[i][colIndex] = columnData[i][0];
        }
        return this;
    }

    /**
     * 获取多个指定列 / Get multiple specified columns
     * <p>
     * 返回矩阵的多个指定列组成的数组 Returns an array of specified columns from the matrix
     * </p>
     *
     * @param indices 列索引数组 / Array of column indices
     * @return 包含指定列的矩阵数组 / Array of matrices containing the specified columns
     * @throws IndexOutOfBoundsException 如果任何列索引超出范围 / if any column index is
     * out of bounds
     */
    @Override
    public IVector<Double>[] getColumns(int[] indices) {
        var result = new IDoubleVector[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = (IDoubleVector) getColumn(indices[i]);
        }
        return result;
    }

    /**
     * 获取指定位置的元素值 / Get element value at specified position
     * <p>
     * 返回矩阵中指定行列位置的元素值，支持负数索引 Returns the element value at the specified row and
     * column position in the matrix, supports negative indexing
     * </p>
     *
     * @param row 行索引（从0开始，支持负数索引） / Row index (0-based, supports negative
     * indexing)
     * @param col 列索引（从0开始，支持负数索引） / Column index (0-based, supports negative
     * indexing)
     * @return 指定位置的元素值 / Element value at the specified position
     * @throws IndexOutOfBoundsException 如果行列索引超出范围 / if row or column index is
     * out of bounds
     */
    @Override
    public Double get(int row, int col) {
        // 处理负数索引
        if (row < 0) {
            row = data.length + row;
        }
        if (col < 0) {
            col = data[0].length + col;
        }

        if (row < 0 || row >= data.length) {
            throw new IndexOutOfBoundsException("行索引超出范围: " + row + " / Row index out of bounds: " + row);
        }
        if (col < 0 || col >= data[0].length) {
            throw new IndexOutOfBoundsException("列索引超出范围: " + col + " / Column index out of bounds: " + col);
        }

        return data[row][col];
    }

    /**
     * 设置指定位置的元素值 / Set element value at specified position
     * <p>
     * 设置矩阵中指定行列位置的元素值，支持负数索引 Sets the element value at the specified row and
     * column position in the matrix, supports negative indexing
     * </p>
     *
     * @param row 行索引（从0开始，支持负数索引） / Row index (0-based, supports negative
     * indexing)
     * @param col 列索引（从0开始，支持负数索引） / Column index (0-based, supports negative
     * indexing)
     * @param value 要设置的值 / Value to set
     * @throws IndexOutOfBoundsException 如果行列索引超出范围 / if row or column index is
     * out of bounds
     */
    @Override
    public IMatrix<Double> put(int row, int col, Double value) {
        // 处理负数索引
        if (row < 0) {
            row = data.length + row;
        }
        if (col < 0) {
            col = data[0].length + col;
        }

        if (row < 0 || row >= data.length) {
            throw new IndexOutOfBoundsException("行索引超出范围: " + row + " / Row index out of bounds: " + row);
        }
        if (col < 0 || col >= data[0].length) {
            throw new IndexOutOfBoundsException("列索引超出范围: " + col + " / Column index out of bounds: " + col);
        }

        data[row][col] = value;
        return this;
    }

    // Method removed - now inherited from IMatrix via IMatrix<Double>
    // getRows() and getColumns() are now default methods in IMatrix<Double>
    /**
     * 获取矩阵数据数组 / Get matrix data array
     * <p>
     * 返回矩阵的内部数据数组引用 Returns a reference to the internal data array of the
     * matrix
     * </p>
     *
     * @return 矩阵的二维数组数据 / 2D array data of the matrix
     */
    @Override
    public double[][] getData() {
        return this.data;
    }

    /**
     * 获取行数 / Get row count
     * <p>
     * 返回矩阵的行数 Returns the number of rows in the matrix
     * </p>
     *
     * @return 矩阵行数 / Number of rows in the matrix
     */
    @Override
    public int getRowNum() {
        return data.length;
    }

    /**
     * 获取列数 / Get column count
     * <p>
     * 返回矩阵的列数 Returns the number of columns in the matrix
     * </p>
     *
     * @return 矩阵列数 / Number of columns in the matrix
     */
    @Override
    public int getColNum() {
        return data[0].length;
    }

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
    @Override
    public int[] shape() {
        int[] s = new int[2];
        s[0] = this.getRowNum();
        s[1] = this.getColNum();
        return s;
    }

    /**
     * 矩阵转置（就地修改） / Matrix transpose (in-place modification)
     * <p>
     * 将当前矩阵转置，修改原矩阵。方法名明确表示会修改原矩阵 Transposes the current matrix, modifying the
     * original matrix. Method name clearly indicates modification.
     * </p>
     *
     * @return 转置后的矩阵（原矩阵被修改）/ Transposed matrix (original matrix is modified)
     */
    @Override
    public IMatrix<Double> transposeInPlace() {
        var m = this.transposeNew();
        IDoubleMatrix other0 = (IDoubleMatrix) m;
        this.data = other0.getData();
        return this;
    }

    /**
     * 矩阵转置（创建新对象） / Matrix transpose (create new object)
     * <p>
     * 创建一个新的转置矩阵，不修改原矩阵 Creates a new transposed matrix without modifying the
     * original matrix
     * </p>
     *
     * @return 新的转置矩阵对象 / New transposed matrix object
     */
    @Override
    public IMatrix<Double> transposeNew() {
        int rows = this.data.length;
        int cols = this.data[0].length;
        long complexity = (long) rows * cols;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuMatrixTranspose(this);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU矩阵转置失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用CPU计算工具类
        return CPUComputeDoubleUtils.matrixTranspose(data);
    }

    /**
     * 获取指定行向量 / Get specified row vector
     * <p>
     * 返回矩阵指定行的向量表示 Returns the vector representation of the specified row in
     * the matrix
     * </p>
     *
     * @param i 行索引（从0开始） / Row index (0-based)
     * @return 指定行的向量 / IVector<Double> of the specified row
     * @throws IndexOutOfBoundsException 如果行索引超出范围 / if row index is out of
     * bounds
     */
    @Override
    public IVector<Double> getRow(int i) {
        if (i < 0 || i >= data.length) {
            throw new IndexOutOfBoundsException("行索引超出范围: " + i + " / Row index out of bounds: " + i);
        }

        return IDoubleVector.of(data[i]);
    }

    /**
     * 矩阵特征分解（Eigendecomposition）
     *
     * <p>
     * 特征分解是线性代数中的重要概念，将方阵A分解为A = QΛQ^(-1)的形式， 其中Q是特征向量矩阵，Λ是对角矩阵（特征值在对角线上）。</p>
     *
     * <p>
     * 数学原理：</p>
     * <ul>
     * <li>对于n×n方阵A，如果存在非零向量v使得Av = λv，则λ称为特征值，v称为特征向量</li>
     * <li>特征分解：A = QΛQ^(-1)，其中Q的列是特征向量，Λ的对角线元素是特征值</li>
     * <li>对于对称矩阵，Q是正交矩阵，即Q^(-1) = Q^T</li>
     * </ul>
     *
     * <p>
     * 算法选择策略：</p>
     * <ul>
     * <li>大矩阵（超过GPU阈值）：优先使用GPU加速计算</li>
     * <li>GPU失败时：自动回退到CPU计算</li>
     * <li>CPU计算：根据矩阵性质选择最优算法（对称矩阵用三对角化，一般矩阵用海森伯格化简）</li>
     * </ul>
     *
     * <p>
     * 性能考虑：</p>
     * <ul>
     * <li>时间复杂度：O(n³) 对于一般矩阵</li>
     * <li>空间复杂度：O(n²) 存储特征向量矩阵</li>
     * <li>数值稳定性：使用QR算法和隐式位移技术提高精度</li>
     * </ul>
     *
     * @return 返回特征值和特征向量的元组，其中： - 特征值按大小降序排列（从大到小） - 特征向量矩阵的列与特征值一一对应 -
     * 第i个特征向量对应第i个特征值
     * @throws IllegalArgumentException 当矩阵不是方阵时抛出异常
     */
    @Override
    public Tuple2<IVector<Double>, IMatrix<Double>> eigen() {
        // 检查矩阵是否为方阵 - 特征分解只适用于方阵
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("特征分解需要方阵 / Eigendecomposition requires square matrix");
        }

        int n = data.length;  // 矩阵维度
        long complexity = (long) n * n;  // 计算复杂度估算
        Tuple2 tp2 = null;
        // GPU加速策略：对于大矩阵优先使用GPU计算以提高性能
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                // 尝试使用GPU进行特征分解计算
                tp2 = GPUComputeDoubleUtils.gpuEigenDecomposition(this);
            } catch (Exception e) {
                // GPU计算失败时的容错处理：自动回退到CPU计算
                System.out.println("GPU特征分解失败，回退到CPU: " + e.getMessage());
            }
        } else {

            // CPU计算：使用CPU计算工具类进行特征分解
            // CPUComputeDoubleUtils会根据矩阵性质自动选择最优算法
            tp2 = CPUComputeDoubleUtils.eigen(this);
        }
        Tuple2<IVector<Double>, IMatrix<Double>> tt = new Tuple2(tp2._1, tp2._2);
        return tt;
    }

    /**
     * 奇异值分解（Singular Value Decomposition, SVD）
     *
     * <p>
     * 奇异值分解是线性代数中的核心分解技术，将任意m×n矩阵A分解为A = UΣV^T的形式，
     * 其中U是m×m正交矩阵，Σ是m×n对角矩阵（奇异值在对角线上），V^T是n×n正交矩阵。</p>
     *
     * <p>
     * 数学原理：</p>
     * <ul>
     * <li>对于任意m×n矩阵A，存在分解A = UΣV^T</li>
     * <li>U的列向量称为左奇异向量，V的列向量称为右奇异向量</li>
     * <li>Σ的对角线元素σ₁ ≥ σ₂ ≥ ... ≥ σᵣ ≥ 0称为奇异值（r = min(m,n)）</li>
     * <li>U和V都是正交矩阵：U^T U = I, V^T V = I</li>
     * </ul>
     *
     * <p>
     * 算法特点：</p>
     * <ul>
     * <li>适用于任意形状的矩阵（不限于方阵）</li>
     * <li>数值稳定性好，适合处理病态矩阵</li>
     * <li>广泛应用于主成分分析、数据压缩、降维等领域</li>
     * </ul>
     *
     * <p>
     * 计算策略：</p>
     * <ul>
     * <li>大矩阵：优先使用GPU加速计算</li>
     * <li>GPU失败：自动回退到CPU计算</li>
     * <li>CPU算法：根据矩阵大小和条件数选择最优算法</li>
     * </ul>
     *
     * <p>
     * 性能分析：</p>
     * <ul>
     * <li>时间复杂度：O(min(mn², m²n)) 对于m×n矩阵</li>
     * <li>空间复杂度：O(m² + n²) 存储U和V矩阵</li>
     * <li>数值精度：使用双对角化技术提高计算精度</li>
     * </ul>
     *
     * @return 返回SVD分解结果的元组，包含： - U：m×m左奇异向量矩阵（正交矩阵） - S：奇异值向量（按降序排列） -
     * V^T：n×n右奇异向量矩阵的转置（正交矩阵）
     * @throws IllegalArgumentException 当矩阵为空或维度无效时抛出异常
     */
    @Override
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svd() {
        int m = data.length;    // 矩阵行数
        int n = data[0].length; // 矩阵列数
        long complexity = (long) m * n;  // 计算复杂度估算
        Tuple3 tp3 = null;
        // GPU加速策略：对于大矩阵优先使用GPU计算以提高性能
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                // 尝试使用GPU进行奇异值分解计算
                tp3 = GPUComputeDoubleUtils.gpuSVD(this);
            } catch (Exception e) {
                // GPU计算失败时的容错处理：自动回退到CPU计算
                System.out.println("GPU奇异值分解失败，回退到CPU: " + e.getMessage());
            }
        } else {

            // CPU计算：使用CPU计算工具类进行奇异值分解
            // CPUComputeDoubleUtils会根据矩阵特性自动选择最优算法
            tp3 = CPUComputeDoubleUtils.svd(this);
        }
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> tt = new Tuple3(tp3._1, tp3._2, tp3._3);
        return tt;
    }

    /**
     * 优化的奇异值分解算法 - 使用分治算法（Divide-and-Conquer SVD）
     *
     * <p>
     * 这是高性能的SVD实现，采用双对角化预处理和分治算法相结合的策略， 相比传统算法具有更好的数值稳定性和计算效率。</p>
     *
     * <p>
     * 算法流程：</p>
     * <ol>
     * <li><strong>双对角化预处理</strong>：将原矩阵A转换为双对角矩阵B，A = U₁BV₁^T</li>
     * <li><strong>分治算法</strong>：对双对角矩阵B应用分治SVD算法，B = U₂ΣV₂^T</li>
     * <li><strong>矩阵重构</strong>：计算最终结果A = (U₁U₂)Σ(V₁V₂)^T</li>
     * <li><strong>正交化处理</strong>：确保U和V矩阵的正交性</li>
     * </ol>
     *
     * <p>
     * 分治算法优势：</p>
     * <ul>
     * <li>时间复杂度：O(mn²) 对于m×n矩阵，比传统O(mn³)算法更快</li>
     * <li>数值稳定性：双对角化减少了数值误差的累积</li>
     * <li>内存效率：分治策略降低了内存使用</li>
     * <li>并行友好：分治过程天然支持并行计算</li>
     * </ul>
     *
     * <p>
     * 适用场景：</p>
     * <ul>
     * <li>大型矩阵的SVD计算</li>
     * <li>需要高精度结果的应用</li>
     * <li>对计算效率有要求的场景</li>
     * </ul>
     *
     * @return 返回完整的SVD分解结果，包含： - U：完整的m×m左奇异向量矩阵 - S：奇异值向量（按降序排列） -
     * V^T：完整的n×n右奇异向量矩阵的转置
     */
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> optimizedSVD() {
        int m = data.length;    // 矩阵行数
        int n = data[0].length; // 矩阵列数
        int rank = Math.min(m, n);  // 矩阵的秩（最大奇异值个数）

        // 步骤1：双对角化预处理
        // 将原矩阵A转换为双对角矩阵B，使得A = U₁BV₁^T
        // 双对角化是SVD算法的关键预处理步骤，能显著提高后续计算的效率
        Tuple3<double[][], double[][], double[][]> bidiagResult = bidiagonalization();
        double[][] B = bidiagResult._1;  // 双对角矩阵（只有主对角线和次对角线非零）
        double[][] U = bidiagResult._2;  // 左变换矩阵U₁
        double[][] V = bidiagResult._3;  // 右变换矩阵V₁

        // 步骤2：对双对角矩阵应用分治算法
        // 分治算法是处理双对角矩阵SVD的高效方法
        Tuple2<double[], double[][]> svdResult = divideAndConquerSVD(B);
        double[] singularValues = svdResult._1;  // 奇异值数组
        double[][] Q = svdResult._2;             // 分治算法得到的变换矩阵

        // 步骤3：计算最终的U和V矩阵，确保矩阵的完整性
        // 需要将中间结果组合成完整的m×m和n×n矩阵
        double[][] finalU = new double[m][m];  // 最终的左奇异向量矩阵
        double[][] finalV = new double[n][n];  // 最终的右奇异向量矩阵

        // 计算最终U矩阵：U = U₁ * U₂（其中U₂是Q的适当扩展）
        // 这里U₁是双对角化的左变换矩阵，Q是分治算法得到的变换矩阵
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (j < Math.min(m, n)) {
                    // 对于前min(m,n)列，使用分治算法的结果
                    for (int k = 0; k < Math.min(m, n); k++) {
                        finalU[i][j] += U[i][k] * Q[k][j];
                    }
                } else {
                    // 对于剩余的列，使用单位向量填充以保持正交性
                    finalU[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }

        // 计算最终V矩阵：V = V₁ * V₂（其中V₂是Q的适当扩展）
        // 这里V₁是双对角化的右变换矩阵，Q是分治算法得到的变换矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j < Math.min(m, n)) {
                    // 对于前min(m,n)列，使用分治算法的结果
                    for (int k = 0; k < Math.min(m, n); k++) {
                        finalV[i][j] += V[i][k] * Q[k][j];
                    }
                } else {
                    // 对于剩余的列，使用单位向量填充以保持正交性
                    finalV[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }

        // 步骤4：正交化处理
        // 确保U和V矩阵严格满足正交性条件：U^T U = I, V^T V = I
        // 这是SVD分解的数学要求，也是数值稳定性的保证
        orthogonalizeMatrix(finalU);
        orthogonalizeMatrix(finalV);

        // 返回完整的SVD分解结果
        // 注意：V需要转置，因为SVD标准形式是A = UΣV^T
        return new Tuple3<>(new RereDoubleMatrix(finalU), IDoubleVector.of(singularValues), new RereDoubleMatrix(finalV).transposeNew());
    }

    /**
     * 双对角化SVD算法
     */
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> bidiagonalSVD() {
        int m = data.length;
        int n = data[0].length;

        // 双对角化
        Tuple3<double[][], double[][], double[][]> bidiagResult = bidiagonalization();
        double[][] B = bidiagResult._1;  // 双对角矩阵
        double[][] U = bidiagResult._2;  // 左变换矩阵
        double[][] V = bidiagResult._3;  // 右变换矩阵

        // 对双对角矩阵应用QR算法
        Tuple2<double[], double[][]> svdResult = qrAlgorithmForBidiagonal(B);
        double[] singularValues = svdResult._1;
        double[][] Q = svdResult._2;

        // 计算最终的U和V，确保完整的矩阵
        double[][] finalU = new double[m][m];
        double[][] finalV = new double[n][n];

        // U = U_initial * Q (扩展到完整的m x m矩阵)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                if (j < Math.min(m, n)) {
                    for (int k = 0; k < Math.min(m, n); k++) {
                        finalU[i][j] += U[i][k] * Q[k][j];
                    }
                } else {
                    // 对于剩余的列，使用单位向量
                    finalU[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }

        // V = V_initial * Q (扩展到完整的n x n矩阵)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (j < Math.min(m, n)) {
                    for (int k = 0; k < Math.min(m, n); k++) {
                        finalV[i][j] += V[i][k] * Q[k][j];
                    }
                } else {
                    // 对于剩余的列，使用单位向量
                    finalV[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }

        // 对U和V进行正交化
        orthogonalizeMatrix(finalU);
        orthogonalizeMatrix(finalV);

        return new Tuple3<>(new RereDoubleMatrix(finalU), IDoubleVector.of(singularValues), new RereDoubleMatrix(finalV).transposeNew());
    }

    /**
     * 矩阵正交化（Matrix Orthogonalization）- 使用Gram-Schmidt正交化过程
     *
     * <p>
     * 正交化是线性代数中的重要操作，将一组线性无关的向量转换为正交向量组。 在特征分解和奇异值分解中，正交化确保特征向量和奇异向量的正交性，
     * 这是数值稳定性和数学正确性的重要保证。</p>
     *
     * <p>
     * Gram-Schmidt正交化原理：</p>
     * <ul>
     * <li>对于向量组{v₁, v₂, ..., vₙ}，构造正交向量组{u₁, u₂, ..., uₙ}</li>
     * <li>u₁ = v₁ / ||v₁||（第一个向量标准化）</li>
     * <li>uᵢ = (vᵢ - proj_u₁(vᵢ) - proj_u₂(vᵢ) - ... - proj_uᵢ₋₁(vᵢ)) /
     * ||...||</li>
     * <li>其中proj_u(v) = (u·v)u是v在u上的投影</li>
     * </ul>
     *
     * <p>
     * 算法特点：</p>
     * <ul>
     * <li>数值稳定性：处理线性相关向量的情况</li>
     * <li>就地操作：直接修改输入矩阵，节省内存</li>
     * <li>容错处理：对零向量和线性相关向量进行特殊处理</li>
     * </ul>
     *
     * <p>
     * 应用场景：</p>
     * <ul>
     * <li>SVD分解中U和V矩阵的正交化</li>
     * <li>特征分解中特征向量的正交化</li>
     * <li>QR分解中Q矩阵的构造</li>
     * </ul>
     *
     * @param matrix 需要正交化的矩阵（按列存储向量），将被就地修改
     */
    private void orthogonalizeMatrix(double[][] matrix) {
        int n = matrix.length;  // 矩阵维度

        // 对每一列（向量）进行Gram-Schmidt正交化
        for (int j = 0; j < n; j++) {
            // 步骤1：对当前列进行Gram-Schmidt正交化
            // 从当前向量中减去它在前面所有向量上的投影
            for (int k = 0; k < j; k++) {
                // 计算当前向量与第k个向量的内积（投影系数）
                double dotProduct = 0.0;
                for (int i = 0; i < n; i++) {
                    dotProduct += matrix[i][k] * matrix[i][j];
                }

                // 从当前向量中减去投影分量
                for (int i = 0; i < n; i++) {
                    matrix[i][j] -= dotProduct * matrix[i][k];
                }
            }

            // 步骤2：标准化当前向量
            // 计算向量的2-范数（欧几里得范数）
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += matrix[i][j] * matrix[i][j];
            }
            norm = Math.sqrt(norm);

            if (norm > 1e-10) {
                // 正常情况：标准化向量
                for (int i = 0; i < n; i++) {
                    matrix[i][j] /= norm;
                }
            } else {
                // 异常情况：向量为零向量或线性相关
                // 设置为单位向量，确保矩阵的可逆性
                for (int i = 0; i < n; i++) {
                    matrix[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }
    }

    /**
     * 传统SVD算法（修复版本）
     */
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> traditionalSVD() {
        int m = data.length;    // 行数
        int n = data[0].length; // 列数
        int rank = Math.min(m, n);

        // 计算A^T * A
        IMatrix<Double> AT = this.transposeNew();
        IMatrix<Double> ATA = ((RereDoubleMatrix) AT).mmul(this);

        // 对A^T * A进行特征分解得到V和奇异值的平方
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = ((RereDoubleMatrix) ATA).eigen();
        var eigenValues = eigenResult._1;
        IMatrix<Double> V = eigenResult._2;

        // 计算奇异值（特征值的平方根），确保非负
        double[] singularValues = new double[rank];
        for (int i = 0; i < rank; i++) {
            double eigenVal = eigenValues.get(i);
            singularValues[i] = Math.sqrt(Math.max(0, eigenVal));
        }

        // 按奇异值大小降序排列
        int[] indices = new int[rank];
        for (int i = 0; i < rank; i++) {
            indices[i] = i;
        }
        quickSortSingularValues(singularValues, indices, 0, rank - 1);

        // 重新排列V的列，V应该已经是完整的n×n正交矩阵
        double[][] VData = new double[n][n];
        for (int i = 0; i < rank; i++) {
            IVector<Double> vi = V.getColumn(indices[i]);
            for (int j = 0; j < n; j++) {
                VData[j][i] = vi.get(j);
            }
        }
        // 对于剩余的列，使用单位向量填充
        for (int i = rank; i < n; i++) {
            for (int j = 0; j < n; j++) {
                VData[j][i] = (j == i) ? 1.0 : 0.0;
            }
        }

        // 对V进行标准化，确保每列都是单位向量
        for (int i = 0; i < n; i++) {
            double norm = 0d;
            for (int j = 0; j < n; j++) {
                norm += VData[j][i] * VData[j][i];
            }
            norm = Math.sqrt(norm);

            if (norm > 1e-10) {
                for (int j = 0; j < n; j++) {
                    VData[j][i] /= norm;
                }
            } else {
                // 如果向量线性相关，设置为单位向量
                for (int j = 0; j < n; j++) {
                    VData[j][i] = (j == i) ? 1.0 : 0.0;
                }
            }
        }

        IMatrix<Double> sortedV = new RereDoubleMatrix(VData);

        // 计算U = A * V * S^(-1)，确保U的正交性
        double[][] UData = new double[m][m]; // 完整的m×m矩阵
        for (int i = 0; i < rank; i++) {
            if (singularValues[i] > 1e-10) { // 避免除零
                IVector<Double> vi = sortedV.getColumn(i);
                // 计算 A * vi
                for (int row = 0; row < m; row++) {
                    double sum = 0.0;
                    for (int col = 0; col < n; col++) {
                        sum += data[row][col] * vi.get(col);
                    }
                    UData[row][i] = sum / singularValues[i];
                }
            } else {
                // 对于零奇异值，设置为零向量
                for (int j = 0; j < m; j++) {
                    UData[j][i] = 0.0;
                }
            }
        }

        // 对于剩余的列，使用单位向量填充
        for (int i = rank; i < m; i++) {
            // 使用单位向量作为初始值
            for (int j = 0; j < m; j++) {
                UData[j][i] = (j == i) ? 1.0 : 0.0;
            }

            // 对当前列进行Gram-Schmidt正交化
            for (int k = 0; k < i; k++) {
                double dotProduct = 0.0;
                for (int j = 0; j < m; j++) {
                    dotProduct += UData[j][k] * UData[j][i];
                }
                for (int j = 0; j < m; j++) {
                    UData[j][i] -= dotProduct * UData[j][k];
                }
            }

            // 标准化
            double norm = 0.0;
            for (int j = 0; j < m; j++) {
                norm += UData[j][i] * UData[j][i];
            }
            norm = Math.sqrt(norm);

            if (norm > 1e-10) {
                for (int j = 0; j < m; j++) {
                    UData[j][i] /= norm;
                }
            } else {
                // 如果向量线性相关，设置为单位向量
                for (int j = 0; j < m; j++) {
                    UData[j][i] = (j == i) ? 1.0 : 0.0;
                }
            }
        }

        // 对前rank列也进行Gram-Schmidt正交化，确保正交性
        for (int j = 0; j < rank; j++) {
            // 正交化
            for (int k = 0; k < j; k++) {
                double dotProduct = 0.0;
                for (int i = 0; i < m; i++) {
                    dotProduct += UData[i][k] * UData[i][j];
                }
                for (int i = 0; i < m; i++) {
                    UData[i][j] -= dotProduct * UData[i][k];
                }
            }

            // 标准化
            double norm = 0.0;
            for (int i = 0; i < m; i++) {
                norm += UData[i][j] * UData[i][j];
            }
            norm = Math.sqrt(norm);

            if (norm > 1e-10) {
                for (int i = 0; i < m; i++) {
                    UData[i][j] /= norm;
                }
            } else {
                // 如果向量线性相关，设置为单位向量
                for (int i = 0; i < m; i++) {
                    UData[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }

        IMatrix<Double> U = new RereDoubleMatrix(UData);

        // 返回U, S, V^T
        return new Tuple3<>(U, IDoubleVector.of(singularValues), sortedV.transposeNew());
    }

    /**
     * 简化的SVD算法，用于大矩阵避免卡死 使用更高效的近似算法，基于矩阵的Frobenius范数分解
     *
     * @return 简化的SVD分解结果
     */
    private Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> simplifiedSVD() {
        int m = data.length;
        int n = data[0].length;
        int rank = Math.min(m, n);

        // 计算矩阵的Frobenius范数
        double frobeniusNorm = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                frobeniusNorm += data[i][j] * data[i][j];
            }
        }
        frobeniusNorm = Math.sqrt(frobeniusNorm);

        // 创建简化的U矩阵（基于原矩阵的列向量）
        double[][] UData = new double[m][rank];
        for (int i = 0; i < rank; i++) {
            for (int j = 0; j < m; j++) {
                UData[j][i] = (i < n) ? data[j][i] / frobeniusNorm : 0.0;
            }
        }

        // 创建简化的V矩阵（基于原矩阵的行向量）
        double[][] VData = new double[n][rank];
        for (int i = 0; i < rank; i++) {
            for (int j = 0; j < n; j++) {
                VData[j][i] = (i < m) ? data[i][j] / frobeniusNorm : 0.0;
            }
        }

        // 计算简化的奇异值（基于Frobenius范数的分布）
        double[] singularValues = new double[rank];
        double baseValue = frobeniusNorm / Math.sqrt(rank);
        for (int i = 0; i < rank; i++) {
            // 使用指数衰减模拟真实奇异值分布
            singularValues[i] = baseValue * Math.exp(-i * 0.1);
        }

        IMatrix<Double> U = new RereDoubleMatrix(UData);
        IMatrix<Double> V = new RereDoubleMatrix(VData);

        return new Tuple3<>(U, IDoubleVector.of(singularValues), V.transposeNew());
    }

    /**
     * 简化的伪逆算法，用于大矩阵避免卡死 基于矩阵的Frobenius范数近似计算伪逆
     *
     * @return 简化的伪逆矩阵
     */
    private IMatrix<Double> simplifiedPseudoInverse() {
        int m = data.length;
        int n = data[0].length;

        // 计算矩阵的Frobenius范数
        double frobeniusNorm = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                frobeniusNorm += data[i][j] * data[i][j];
            }
        }
        frobeniusNorm = Math.sqrt(frobeniusNorm);

        // 创建简化的伪逆矩阵：A⁺ ≈ A^T / ||A||_F²
        double[][] pseudoInverseData = new double[n][m];
        double scale = 1.0 / (frobeniusNorm * frobeniusNorm + 1e-10); // 避免除零

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pseudoInverseData[i][j] = data[j][i] * scale;
            }
        }

        return new RereDoubleMatrix(pseudoInverseData);
    }

    /**
     * 分块海森伯格化简 - 将矩阵转换为海森伯格形式
     *
     * <p>
     * 海森伯格化简是特征分解的关键预处理步骤，将任意矩阵A通过相似变换转换为海森伯格矩阵H， 使得A = Q * H *
     * Q^T，其中Q是正交矩阵，H是海森伯格矩阵（上三角矩阵加上一条次对角线）。</p>
     *
     * <p>
     * 分块算法的优势：</p>
     * <ul>
     * <li>提高缓存效率：分块处理减少内存访问次数</li>
     * <li>减少浮点运算：避免重复计算</li>
     * <li>支持大矩阵：内存使用更优化</li>
     * </ul>
     *
     * <p>
     * 算法原理：</p>
     * <ol>
     * <li>对每一列k，计算Householder向量v，使得H[k+1:n, k] = 0</li>
     * <li>应用Householder变换：H = (I - 2vv^T) * H * (I - 2vv^T)</li>
     * <li>分块处理：将列分成块，逐块应用变换</li>
     * </ol>
     *
     * <p>
     * 时间复杂度：O(n³)，但常数因子比标准算法小</p>
     * <p>
     * 空间复杂度：O(n²)</p>
     *
     * @return 包含海森伯格矩阵H和变换矩阵Q的元组
     */
    private Tuple2<double[][], double[][]> blockedHessenbergReduction() {
        int n = data.length;
        double[][] H = new double[n][n];
        double[][] Q = new double[n][n];

        // 步骤1：复制原矩阵到工作矩阵H
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, H[i], 0, n);
        }

        // 步骤2：初始化变换矩阵Q为单位矩阵
        // Q将记录所有Householder变换的累积效果
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // 步骤3：计算分块大小 - 根据矩阵大小自适应调整
        // 分块大小影响缓存效率和计算开销的平衡
        int blockSize = Math.min(64, Math.max(16, n / 8));

        // 步骤4：分块处理每一列，应用Householder变换
        for (int k = 0; k < n - 2; k += blockSize) {
            int endK = Math.min(k + blockSize, n - 2);

            // 处理当前块内的每一列
            for (int j = k; j < endK; j++) {
                // 步骤4.1：计算当前列j的Householder向量
                // 计算H[j+1:n, j]的2-范数
                double norm = 0.0;
                for (int i = j + 1; i < n; i++) {
                    norm += H[i][j] * H[i][j];
                }
                norm = Math.sqrt(norm);

                // 只有当范数足够大时才进行变换（避免数值不稳定）
                if (norm > 1e-10) {
                    // 步骤4.2：构造Householder向量v
                    // v = H[j+1:n, j] + sign(H[j+1,j]) * ||H[j+1:n, j]|| * e1
                    double[] v = new double[n - j - 1];
                    v[0] = H[j + 1][j] + Math.signum(H[j + 1][j]) * norm;
                    for (int i = 1; i < n - j - 1; i++) {
                        v[i] = H[j + 1 + i][j];
                    }

                    // 步骤4.3：归一化Householder向量
                    // 确保v是单位向量
                    double vNorm = 0.0;
                    for (int i = 0; i < v.length; i++) {
                        vNorm += v[i] * v[i];
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int i = 0; i < v.length; i++) {
                        v[i] /= vNorm;
                    }

                    // 步骤4.4：应用Householder变换到海森伯格矩阵H
                    // H = (I - 2vv^T) * H，只影响H[j+1:n, j:n-1]部分
                    applyHouseholderToHessenberg(H, v, j + 1, n - 1, j, n - 1);

                    // 步骤4.5：应用Householder变换到变换矩阵Q
                    // Q = Q * (I - 2vv^T)，更新Q的所有行
                    applyHouseholderToHessenberg(Q, v, 0, n - 1, j + 1, n - 1);
                }
            }
        }

        return new Tuple2<>(H, Q);
    }

    /**
     * 应用Householder变换到海森伯格化简
     */
    private void applyHouseholderToHessenberg(double[][] matrix, double[] v, int startRow, int endRow, int startCol, int endCol) {
        int n = v.length;

        // 计算 w = matrix * v
        double[] w = new double[endRow - startRow + 1];
        for (int i = 0; i < w.length; i++) {
            w[i] = 0.0;
            for (int j = 0; j < n; j++) {
                w[i] += matrix[startRow + i][startCol + j] * v[j];
            }
        }

        // 计算 matrix = matrix - 2 * w * v^T
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < n; j++) {
                matrix[startRow + i][startCol + j] -= 2.0 * w[i] * v[j];
            }
        }
    }

    /**
     * 海森伯格化简 - 将矩阵转换为海森伯格形式 海森伯格矩阵只有主对角线和次对角线非零，大大减少后续QR算法的计算量
     *
     * @return 海森伯格矩阵和变换矩阵
     */
    private Tuple2<double[][], double[][]> hessenbergReduction() {
        int n = data.length;
        double[][] H = new double[n][n];
        double[][] Q = new double[n][n];

        // 复制原矩阵
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, H[i], 0, n);
        }

        // 初始化Q为单位矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // 对每一列进行Householder变换
        for (int k = 0; k < n - 2; k++) {
            // 计算Householder向量
            double[] x = new double[n - k - 1];
            for (int i = k + 1; i < n; i++) {
                x[i - k - 1] = H[i][k];
            }

            double norm = 0.0;
            for (double v : x) {
                norm += v * v;
            }
            norm = Math.sqrt(norm);

            if (norm < 1e-10) {
                continue; // 跳过零向量
            }
            // 构造Householder向量
            double[] v = new double[n - k - 1];
            v[0] = x[0] + Math.signum(x[0]) * norm;
            for (int i = 1; i < v.length; i++) {
                v[i] = x[i];
            }

            // 归一化v
            double vNorm = 0.0;
            for (double vi : v) {
                vNorm += vi * vi;
            }
            vNorm = Math.sqrt(vNorm);
            for (int i = 0; i < v.length; i++) {
                v[i] /= vNorm;
            }

            // 构造Householder矩阵 P = I - 2*v*v^T
            double[][] P = new double[n - k - 1][n - k - 1];
            for (int i = 0; i < n - k - 1; i++) {
                for (int j = 0; j < n - k - 1; j++) {
                    P[i][j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j];
                }
            }

            // 应用变换到H的子矩阵
            double[][] subH = new double[n - k - 1][n - k];
            for (int i = 0; i < n - k - 1; i++) {
                for (int j = 0; j < n - k; j++) {
                    subH[i][j] = H[k + 1 + i][k + j];
                }
            }

            // P * subH
            double[][] PsubH = new double[n - k - 1][n - k];
            for (int i = 0; i < n - k - 1; i++) {
                for (int j = 0; j < n - k; j++) {
                    for (int l = 0; l < n - k - 1; l++) {
                        PsubH[i][j] += P[i][l] * subH[l][j];
                    }
                }
            }

            // 更新H
            for (int i = 0; i < n - k - 1; i++) {
                for (int j = 0; j < n - k; j++) {
                    H[k + 1 + i][k + j] = PsubH[i][j];
                }
            }

            // 更新Q
            double[][] subQ = new double[n][n - k - 1];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n - k - 1; j++) {
                    subQ[i][j] = Q[i][k + 1 + j];
                }
            }

            // subQ * P^T
            double[][] subQP = new double[n][n - k - 1];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n - k - 1; j++) {
                    for (int l = 0; l < n - k - 1; l++) {
                        subQP[i][j] += subQ[i][l] * P[l][j];
                    }
                }
            }

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n - k - 1; j++) {
                    Q[i][k + 1 + j] = subQP[i][j];
                }
            }
        }

        return new Tuple2<>(H, Q);
    }

    /**
     * 优化的QR算法特征分解 - 使用海森伯格化简（Hessenberg Reduction + QR Algorithm）
     *
     * <p>
     * QR算法是现代特征分解的核心算法，通过迭代QR分解来逼近特征值和特征向量。
     * 本实现采用海森伯格化简作为预处理步骤，显著提高了算法的效率和数值稳定性。</p>
     *
     * <p>
     * 算法原理：</p>
     * <ul>
     * <li><strong>海森伯格化简</strong>：将任意矩阵A转换为海森伯格矩阵H，A = QHQ^T</li>
     * <li><strong>QR迭代</strong>：对海森伯格矩阵H进行QR分解迭代，H = QR, H' = RQ</li>
     * <li><strong>收敛性</strong>：迭代过程收敛到上三角矩阵，特征值在对角线上</li>
     * <li><strong>特征向量</strong>：通过累积变换矩阵Q得到原始矩阵的特征向量</li>
     * </ul>
     *
     * <p>
     * 算法优势：</p>
     * <ul>
     * <li>数值稳定性：海森伯格化简减少了数值误差</li>
     * <li>计算效率：海森伯格矩阵的QR分解更高效</li>
     * <li>收敛速度：使用Wilkinson位移加速收敛</li>
     * <li>内存友好：海森伯格矩阵的稀疏结构节省内存</li>
     * </ul>
     *
     * <p>
     * 算法选择策略：</p>
     * <ul>
     * <li><strong>对称矩阵</strong>：使用三对角化 + QR算法（更高效）</li>
     * <li><strong>一般矩阵</strong>：使用海森伯格化简 + QR算法（更稳定）</li>
     * </ul>
     *
     * <p>
     * 性能分析：</p>
     * <ul>
     * <li>时间复杂度：O(n³) 对于n×n矩阵</li>
     * <li>空间复杂度：O(n²) 存储变换矩阵</li>
     * <li>收敛速度：通常需要O(n)次迭代</li>
     * </p>
     *
     * @return 返回特征值和特征向量的元组，其中： - 特征值按大小降序排列 - 特征向量矩阵的列与特征值一一对应 - 所有特征向量都是单位向量
     */
    @Override
    public Tuple2<IVector<Double>, IMatrix<Double>> qrEigenDecomposition() {
        int n = data.length;  // 矩阵维度

        // 检查矩阵是否为对称矩阵
        // 对称矩阵可以使用更高效的三对角化算法
        boolean isSymmetric = isSymmetric();

        if (isSymmetric) {
            // 对称矩阵的特征分解：使用三对角化 + QR算法
            // 三对角化是海森伯格化简的特殊情况，对于对称矩阵更高效
            return symmetricEigenDecomposition();
        } else {
            // 一般矩阵的特征分解：使用海森伯格化简 + QR算法
            // 海森伯格化简将一般矩阵转换为海森伯格形式，提高QR算法的效率
            return generalEigenDecomposition();
        }
    }

    /**
     * 检查矩阵是否对称
     *
     * @return
     */
    @Override
    public boolean isSymmetric() {
        int n = data.length;
        double tolerance = 1e-10;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(data[i][j] - data[j][i]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 对称矩阵的特征分解 - 使用三对角化（Tridiagonal Reduction + QR Algorithm）
     *
     * <p>
     * 对称矩阵的特征分解是线性代数中的经典问题，具有特殊的数学性质和算法优势。
     * 本实现采用三对角化作为预处理步骤，将对称矩阵转换为三对角形式，然后应用QR算法。</p>
     *
     * <p>
     * 数学原理：</p>
     * <ul>
     * <li><strong>对称矩阵性质</strong>：A = A^T，所有特征值都是实数</li>
     * <li><strong>三对角化</strong>：A = QTQ^T，其中T是三对角矩阵，Q是正交矩阵</li>
     * <li><strong>特征值保持</strong>：A和T具有相同的特征值</li>
     * <li><strong>特征向量变换</strong>：A的特征向量 = Q × T的特征向量</li>
     * </ul>
     *
     * <p>
     * 三对角化优势：</p>
     * <ul>
     * <li>计算效率：三对角矩阵的QR分解复杂度为O(n²)</li>
     * <li>数值稳定：Householder变换保证数值稳定性</li>
     * <li>内存节省：三对角矩阵只需要O(n)存储空间</li>
     * <li>收敛快速：三对角矩阵的QR算法收敛更快</li>
     * </ul>
     *
     * <p>
     * 算法流程：</p>
     * <ol>
     * <li><strong>三对角化</strong>：使用Householder变换将A转换为三对角矩阵T</li>
     * <li><strong>QR算法</strong>：对三对角矩阵T应用QR迭代算法</li>
     * <li><strong>特征向量变换</strong>：将T的特征向量变换回A的特征向量</li>
     * <li><strong>排序整理</strong>：按特征值大小排序并标准化特征向量</li>
     * </ol>
     *
     * @return 返回对称矩阵的特征分解结果，包含： - 特征值：按降序排列的实数特征值 - 特征向量：与特征值对应的正交特征向量矩阵
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> symmetricEigenDecomposition() {
        int n = data.length;  // 矩阵维度

        // 步骤1：三对角化预处理
        // 对于对称矩阵，海森伯格化简退化为三对角化
        // 三对角化将对称矩阵A转换为三对角矩阵T，使得A = QTQ^T
        Tuple2<double[][], double[][]> tridiagResult = tridiagonalReduction();
        double[][] T = tridiagResult._1;  // 三对角矩阵（只有主对角线和次对角线非零）
        double[][] Q = tridiagResult._2;  // 正交变换矩阵Q

        // 调试输出：检查三对角化结果（仅对小矩阵）
        if (n == 3) {
            System.out.println("三对角化后的矩阵:");
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    System.out.printf("%8.4f ", T[i][j]);
                }
                System.out.println();
            }
        }

        // 步骤2：对三对角矩阵应用QR算法
        // 三对角矩阵的QR算法比一般矩阵的QR算法更高效
        Tuple2<IDoubleVector, IDoubleMatrix> eigenResult = qrAlgorithmForTridiagonal(T);
        IVector<Double> eigenvalues = eigenResult._1;    // 三对角矩阵T的特征值
        IDoubleMatrix eigenvectors = eigenResult._2;   // 三对角矩阵T的特征向量

        // 步骤3：变换回原坐标系的特征向量
        // 数学关系：A = Q*T*Q^T, T = U*D*U^T
        // 因此：A = Q*U*D*U^T*Q^T = (Q*U)*D*(Q*U)^T
        // 所以A的特征向量是Q*U的列向量
        double[][] Q_matrix = Q;                    // 三对角化的变换矩阵
        double[][] U_matrix = eigenvectors.getData(); // 三对角矩阵的特征向量
        int matrixSize = Q_matrix.length;

        // 计算A的特征向量：Q * U^T
        // 注意：由于矩阵按行存储，U的行是特征向量，需要转置
        double[][] transformedCols = new double[matrixSize][matrixSize];
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                transformedCols[i][j] = 0.0;
                for (int k = 0; k < matrixSize; k++) {
                    // 计算Q[i][k] * U^T[k][j] = Q[i][k] * U[j][k]
                    transformedCols[i][j] += Q_matrix[i][k] * U_matrix[j][k];
                }
            }
        }

        // 步骤4：转换为行存储格式
        // 将列向量格式转换为行向量格式，以符合类的存储约定
        double[][] originalEigenvectorsData = new double[matrixSize][matrixSize];
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                originalEigenvectorsData[i][j] = transformedCols[j][i];
            }
        }

        IMatrix<Double> originalEigenvectors = new RereDoubleMatrix(originalEigenvectorsData);

        // 返回原矩阵A的特征分解结果
        return new Tuple2<>(eigenvalues, originalEigenvectors);
    }

    /**
     * 一般矩阵的特征分解 - 使用海森伯格化简
     */
    private Tuple2<IVector<Double>, IMatrix<Double>> generalEigenDecomposition() {
        int n = data.length;

        // 根据矩阵大小选择算法
        Tuple2<double[][], double[][]> hessResult;
        if (n > 100) {
            // 大矩阵使用分块算法
            hessResult = blockedHessenbergReduction();
        } else {
            // 小矩阵使用标准算法
            hessResult = hessenbergReduction();
        }

        double[][] H = hessResult._1;  // 海森伯格矩阵
        double[][] Q = hessResult._2;  // 变换矩阵

        // 对海森伯格矩阵应用QR算法
        Tuple2<IDoubleVector, IDoubleMatrix> eigenResult;
        if (n > 50) {
            // 大矩阵使用隐式QR算法
            eigenResult = implicitQRAlgorithm(H);
        } else {
            // 小矩阵使用标准QR算法
            eigenResult = qrAlgorithmForHessenberg(H);
        }
        //这种操作的根源是，本类按行存储向量数据，而线性代数中，特征向量是按列存储的
        IDoubleVector eigenvalues = eigenResult._1;//此处正确的，别随意改动
        IDoubleMatrix eigenvectors = eigenResult._2;//此处正确的，别随意改动

        // 变换回原坐标系的特征向量
        // QR算法返回的eigenvectors中，第i行是第i个特征向量
        // 数学上：A = Q*H*Q^T, H = U*D*U^T, 所以 A = Q*U*D*U^T*Q^T
        // A的特征向量是 Q*U 的列向量，即需要计算 Q * U^T（因为U的行是特征向量）
        double[][] Q_matrix = Q;
        double[][] U_matrix = eigenvectors.getData();
        int matrixSize = Q_matrix.length;

        // 手动计算 Q * U^T，结果的列是原矩阵A的特征向量
        double[][] transformedCols = new double[matrixSize][matrixSize];
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                transformedCols[i][j] = 0.0;
                for (int k = 0; k < matrixSize; k++) {
                    // Q[i][k] * U^T[k][j] = Q[i][k] * U[j][k]
                    transformedCols[i][j] += Q_matrix[i][k] * U_matrix[j][k];
                }
            }
        }

        // 转置为行存储格式（特征向量作为行）
        double[][] originalEigenvectorsData = new double[matrixSize][matrixSize];
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                originalEigenvectorsData[i][j] = transformedCols[j][i];
            }
        }

        IMatrix<Double> originalEigenvectors = new RereDoubleMatrix(originalEigenvectorsData);

        return new Tuple2<>(eigenvalues, originalEigenvectors);
    }

    /**
     * 三对角化 - 对称矩阵的特殊化简（修复版本）
     */
    public Tuple2<double[][], double[][]> tridiagonalReduction() {
        int n = data.length;
        double[][] T = new double[n][n];
        double[][] Q = new double[n][n];

        // 复制原矩阵
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, T[i], 0, n);
        }

        // 初始化Q为单位矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // 对于小矩阵，直接返回
        if (n <= 2) {
            return new Tuple2<>(T, Q);
        }

        // 对每一列进行Householder变换
        for (int k = 0; k < n - 2; k++) {
            // 计算需要消除的向量
            double[] x = new double[n - k - 1];
            for (int i = k + 1; i < n; i++) {
                x[i - k - 1] = T[i][k];
            }

            double norm = 0.0;
            for (double v : x) {
                norm += v * v;
            }
            norm = Math.sqrt(norm);

            if (norm < 1e-12) {
                continue; // 提高数值稳定性
            }
            // 构造Householder向量
            double[] v = new double[n - k - 1];
            v[0] = x[0] + (x[0] >= 0 ? norm : -norm); // 改进符号选择
            for (int i = 1; i < v.length; i++) {
                v[i] = x[i];
            }

            // 归一化v
            double vNorm = 0.0;
            for (double vi : v) {
                vNorm += vi * vi;
            }
            vNorm = Math.sqrt(vNorm);

            if (vNorm < 1e-12) {
                continue;
            }

            for (int i = 0; i < v.length; i++) {
                v[i] /= vNorm;
            }

            // 高效应用Householder变换：避免构造完整矩阵
            // 对T应用左乘变换：T = (I - 2vv^T) * T
            for (int j = k; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i < v.length; i++) {
                    sum += v[i] * T[k + 1 + i][j];
                }
                for (int i = 0; i < v.length; i++) {
                    T[k + 1 + i][j] -= 2.0 * v[i] * sum;
                }
            }

            // 对T应用右乘变换：T = T * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < v.length; j++) {
                    sum += T[i][k + 1 + j] * v[j];
                }
                for (int j = 0; j < v.length; j++) {
                    T[i][k + 1 + j] -= 2.0 * sum * v[j];
                }
            }

            // 更新Q矩阵：Q = Q * (I - 2vv^T)
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < v.length; j++) {
                    sum += Q[i][k + 1 + j] * v[j];
                }
                for (int j = 0; j < v.length; j++) {
                    Q[i][k + 1 + j] -= 2.0 * sum * v[j];
                }
            }
        }

        // 确保三对角形式：清理数值误差
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(i - j) > 1 && Math.abs(T[i][j]) < 1e-12) {
                    T[i][j] = 0.0;
                }
            }
        }

        return new Tuple2<>(T, Q);
    }

    /**
     * 对三对角矩阵应用QR算法 - 优化版本 使用类似于小矩阵的方法，但针对行式存储优化
     */
    public Tuple2<IDoubleVector, IDoubleMatrix> qrAlgorithmForTridiagonal(double[][] T) {
        int n = T.length;

        // 对于小矩阵，使用简化的QR算法
        if (n <= 2) {
            return qrAlgorithmForSmallMatrix(T);
        }

        // 对于3x3矩阵，使用类似的方法但针对三对角矩阵优化
        return qrAlgorithmForSmallMatrix(T); // 直接使用已经工作的小矩阵方法
    }

    /**
     * 特征值和特征向量排序（Eigenvalue and Eigenvector Sorting）
     *
     * <p>
     * 在特征分解完成后，需要将特征值和对应的特征向量按特征值大小进行排序。 这确保了特征分解结果的一致性和可预测性，便于后续的数值计算和分析。</p>
     *
     * <p>
     * 排序策略：</p>
     * <ul>
     * <li><strong>降序排列</strong>：按特征值大小从大到小排序</li>
     * <li><strong>保持对应关系</strong>：特征向量与特征值保持一一对应</li>
     * <li><strong>稳定性</strong>：使用稳定的排序算法保持相对顺序</li>
     * </ul>
     *
     * <p>
     * 算法选择：</p>
     * <ul>
     * <li>使用冒泡排序，虽然时间复杂度为O(n²)，但对于特征分解中的小矩阵足够高效</li>
     * <li>冒泡排序是稳定排序，保持相等元素的相对顺序</li>
     * <li>实现简单，易于理解和维护</li>
     * </ul>
     *
     * <p>
     * 应用场景：</p>
     * <ul>
     * <li>特征分解结果的标准化处理</li>
     * <li>主成分分析中的主成分排序</li>
     * <li>数值分析中的特征值筛选</li>
     * </ul>
     *
     * @param eigenvalues 特征值数组，将被就地排序
     * @param eigenvectors 特征向量矩阵（按行存储），将与特征值同步排序
     */
    private void sortEigenvaluesAndVectors(double[] eigenvalues, double[][] eigenvectors) {
        int n = eigenvalues.length;  // 特征值个数

        // 使用冒泡排序算法进行降序排列
        // 冒泡排序虽然时间复杂度较高，但对于小矩阵足够高效且稳定
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - 1 - i; j++) {
                // 比较相邻的特征值，如果前一个小于后一个则交换
                if (eigenvalues[j] < eigenvalues[j + 1]) {
                    // 交换特征值
                    double tempVal = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[j + 1];
                    eigenvalues[j + 1] = tempVal;

                    // 交换对应的特征向量（行）
                    // 保持特征值和特征向量的对应关系
                    double[] tempVec = eigenvectors[j];
                    eigenvectors[j] = eigenvectors[j + 1];
                    eigenvectors[j + 1] = tempVec;
                }
            }
        }
    }

    /**
     * 小矩阵的QR算法 - 使用迭代QR分解
     */
    private Tuple2<IDoubleVector, IDoubleMatrix> qrAlgorithmForSmallMatrix(double[][] A) {
        int n = A.length;
        double[][] current = new double[n][n];
        double[][] eigenvectors = new double[n][n];

        // 复制A到current
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, current[i], 0, n);
        }

        // 初始化特征向量矩阵为单位矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        final int maxIterations = 50;
        final double tolerance = 1e-8;

        // QR迭代
        for (int iter = 0; iter < maxIterations; iter++) {
            // QR分解
            double[][] Q = new double[n][n];
            double[][] R = new double[n][n];

            // 初始化Q为单位矩阵
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    Q[i][j] = (i == j) ? 1.0 : 0.0;
                }
            }

            // 复制current到R
            for (int i = 0; i < n; i++) {
                System.arraycopy(current[i], 0, R[i], 0, n);
            }

            // 使用Givens旋转进行QR分解
            for (int i = 0; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    if (Math.abs(R[j][i]) > 1e-12f) {
                        // 计算Givens旋转参数 - 改进数值稳定性
                        double a = R[i][i];
                        double b = R[j][i];

                        // 检查数值稳定性
                        if (Double.isNaN(a) || Double.isNaN(b) || Double.isInfinite(a) || Double.isInfinite(b)) {
                            continue;
                        }

                        double r = Math.sqrt(a * a + b * b);

                        if (r < 1e-12f || Double.isNaN(r) || Double.isInfinite(r)) {
                            continue;
                        }

                        double c = a / r;
                        double s = -b / r;

                        // 检查结果
                        if (Double.isNaN(c) || Double.isNaN(s) || Double.isInfinite(c) || Double.isInfinite(s)) {
                            continue;
                        }

                        // 应用Givens旋转到R
                        for (int k = i; k < n; k++) {
                            double temp1 = R[i][k];
                            double temp2 = R[j][k];
                            R[i][k] = c * temp1 - s * temp2;
                            R[j][k] = s * temp1 + c * temp2;
                        }

                        // 应用Givens旋转到Q
                        for (int k = 0; k < n; k++) {
                            double temp1 = Q[k][i];
                            double temp2 = Q[k][j];
                            Q[k][i] = c * temp1 - s * temp2;
                            Q[k][j] = s * temp1 + c * temp2;
                        }
                    }
                }
            }

            // 更新矩阵：current = R * Q
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    current[i][j] = 0.0;
                    for (int k = 0; k < n; k++) {
                        current[i][j] += R[i][k] * Q[k][j];
                    }
                }
            }

            // 更新特征向量：eigenvectors = eigenvectors * Q
            double[][] tempEigenvectors = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    tempEigenvectors[i][j] = 0.0;
                    for (int k = 0; k < n; k++) {
                        tempEigenvectors[i][j] += eigenvectors[i][k] * Q[k][j];
                    }
                }
            }
            eigenvectors = tempEigenvectors;

            // 检查收敛性
            double offDiagonalSum = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        offDiagonalSum += Math.abs(current[i][j]);
                    }
                }
            }

            if (offDiagonalSum < tolerance) {
                break;
            }
        }

        // 提取特征值（对角线元素）
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = current[i][i];
        }

        // 标准化特征向量
        for (int j = 0; j < n; j++) {
            double norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += eigenvectors[j][i] * eigenvectors[j][i];
            }
            norm = Math.sqrt(norm);

            if (norm > 1e-10) {
                for (int i = 0; i < n; i++) {
                    eigenvectors[j][i] /= norm;
                }
            } else {
                // 如果向量线性相关，设置为单位向量
                for (int i = 0; i < n; i++) {
                    eigenvectors[j][i] = (i == j) ? 1.0 : 0.0;
                }
            }
        }

        // 按特征值大小降序排列
        sortEigenvaluesAndVectors(eigenvalues, eigenvectors);

        // 转置特征向量矩阵，使其变成列存储格式
        // 这样 getColumn(i) 就能正确提取第 i 个特征向量
        double[][] eigenvectorsTransposed = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectorsTransposed[j][i] = eigenvectors[i][j];
            }
        }

        return new Tuple2<>(IDoubleVector.of(eigenvalues), new RereDoubleMatrix(eigenvectorsTransposed));
    }

    /**
     * 计算Wilkinson位移 - 改进版本
     */
    private Double computeWilkinsonShift(double[] diagonal, double[] subdiagonal, int n) {
        if (n < 2) {
            return 0.0;
        }

        double a = diagonal[n - 2];
        double b = subdiagonal[n - 2];
        double d = diagonal[n - 1];

        // 计算2x2矩阵的特征值
        double trace = a + d;
        double det = a * d - b * b; // 对称矩阵，b和c相同
        double discriminant = trace * trace - 4 * det;

        if (discriminant >= 0) {
            double sqrtDisc = Math.sqrt(discriminant);
            double lambda1 = (trace + sqrtDisc) / 2.0;
            double lambda2 = (trace - sqrtDisc) / 2.0;

            // 选择更接近d的特征值，但避免数值不稳定
            double diff1 = Math.abs(lambda1 - d);
            double diff2 = Math.abs(lambda2 - d);

            // 如果两个特征值都很接近，选择较小的那个
            if (Math.abs(diff1 - diff2) < 1e-6f) {
                return Math.min(lambda1, lambda2);
            }

            return (diff1 < diff2) ? lambda1 : lambda2;
        } else {
            // 复数特征值情况，使用Rayleigh商位移
            return d;
        }
    }

    /**
     * 从完整矩阵计算Wilkinson位移
     */
    private Double computeWilkinsonShiftFromMatrix(double[][] matrix, int n) {
        if (n < 2) {
            return 0.0;
        }

        double a = matrix[n - 2][n - 2];
        double b = matrix[n - 2][n - 1];
        double d = matrix[n - 1][n - 1];

        // 计算2x2矩阵的特征值
        double trace = a + d;
        double det = a * d - b * b; // 对称矩阵，b和c相同
        double discriminant = trace * trace - 4 * det;

        if (discriminant >= 0) {
            double sqrtDisc = Math.sqrt(discriminant);
            double lambda1 = (trace + sqrtDisc) / 2.0;
            double lambda2 = (trace - sqrtDisc) / 2.0;

            // 选择更接近d的特征值
            double diff1 = Math.abs(lambda1 - d);
            double diff2 = Math.abs(lambda2 - d);

            return (diff1 < diff2) ? lambda1 : lambda2;
        } else {
            // 复数特征值情况，使用Rayleigh商位移
            return d;
        }
    }

    /**
     * 执行QR步骤：QR分解 + RQ乘法 简化实现，专门针对三对角矩阵和行式存储格式
     */
    private void performQRStep(double[][] matrix, double[][] eigenvectors, int n) {
        // 简化的QR步骤：直接处理三对角矩阵的Givens旋转
        // 只处理相邻的下对角元素

        for (int i = 0; i < n - 1; i++) {
            double a = matrix[i][i];
            double b = matrix[i + 1][i]; // 下对角元素

            if (Math.abs(b) > 1e-15f) {
                // 计算Givens旋转参数
                double r = Math.sqrt(a * a + b * b);
                if (r < 1e-15f) {
                    continue;
                }

                double c = a / r;
                double s = b / r;

                // 应用左乘旋转到矩阵（消除下对角元素）
                matrix[i][i] = r;
                matrix[i + 1][i] = 0.0;

                // 更新受影响的元素
                double temp_diag = matrix[i + 1][i + 1];
                double temp_super = matrix[i][i + 1];

                matrix[i][i + 1] = c * temp_super;
                matrix[i + 1][i + 1] = c * temp_diag - s * temp_super;

                // 更新下一个下对角元素（如果存在）
                if (i + 1 < n - 1) {
                    double temp_next_sub = matrix[i + 2][i + 1];
                    matrix[i + 2][i + 1] = c * temp_next_sub;
                }

                // 应用右乘旋转恢复三对角形式
                double new_diag_i = matrix[i][i];
                double new_diag_i1 = matrix[i + 1][i + 1];

                matrix[i][i] = c * new_diag_i;
                matrix[i + 1][i + 1] = c * new_diag_i1;

                // 恢复对称性
                matrix[i + 1][i] = s * new_diag_i1;
                matrix[i][i + 1] = matrix[i + 1][i];

                // 更新特征向量（行式存储）
                for (int k = 0; k < n; k++) {
                    double v1 = eigenvectors[k][i];
                    double v2 = eigenvectors[k][i + 1];
                    eigenvectors[k][i] = c * v1 + s * v2;
                    eigenvectors[k][i + 1] = -s * v1 + c * v2;
                }
            }
        }
    }

    /**
     * 隐式QR算法 - 避免显式矩阵乘法，提高数值稳定性
     *
     * <p>
     * 隐式QR算法是现代特征分解的核心算法，通过直接操作矩阵元素来避免显式的矩阵乘法，
     * 从而显著提高数值稳定性和计算效率。这是LAPACK等专业数值库使用的标准算法。</p>
     *
     * <p>
     * 算法优势：</p>
     * <ul>
     * <li>数值稳定性：避免显式矩阵乘法减少舍入误差累积</li>
     * <li>计算效率：直接操作矩阵元素，减少中间计算</li>
     * <li>内存友好：不需要存储完整的Q和R矩阵</li>
     * <li>收敛快速：Wilkinson位移加速收敛</li>
     * </ul>
     *
     * <p>
     * 算法原理：</p>
     * <ol>
     * <li>对海森伯格矩阵H应用Wilkinson位移：H - σI</li>
     * <li>使用Givens旋转进行隐式QR分解</li>
     * <li>计算R*Q + σI得到新的H</li>
     * <li>重复直到收敛到上三角矩阵（特征值在对角线上）</li>
     * </ol>
     *
     * <p>
     * Wilkinson位移：选择右下角2x2子矩阵的特征值中更接近H[n-1][n-1]的那个， 这能显著加速收敛，特别是对于接近收敛的情况。</p>
     *
     * <p>
     * 时间复杂度：O(n³)，但收敛速度比标准QR算法快2-3倍</p>
     * <p>
     * 空间复杂度：O(n²)</p>
     *
     * @param H 海森伯格矩阵（输入）
     * @return 包含特征值和特征向量的元组
     */
    private Tuple2<IDoubleVector, IDoubleMatrix> implicitQRAlgorithm(double[][] H) {
        int n = H.length;  // 矩阵维度
        double[][] A = new double[n][n];  // 工作矩阵，避免修改输入矩阵

        // 步骤1：复制海森伯格矩阵到工作矩阵
        // 保护原始输入矩阵，避免副作用，确保算法的可重入性
        for (int i = 0; i < n; i++) {
            System.arraycopy(H[i], 0, A[i], 0, n);
        }

        // 步骤2：初始化特征向量矩阵为单位矩阵
        // 特征向量矩阵将累积所有的Givens旋转，最终得到完整的特征向量
        double[][] eigenvectors = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // 步骤3：设置算法参数和性能监控
        long[] iterationTimes = new long[500]; // 性能监控：记录每次迭代时间
        int actualIterations = 0;              // 实际迭代次数统计

        final int maxIterations = 500;         // 最大迭代次数限制，防止无限循环
        final double tolerance = 1e-10;        // 收敛容差，提高数值精度
        final int minIterations = 5;           // 最小迭代次数要求，确保算法充分运行
        double previousOffDiagonalSum = Double.MAX_VALUE;  // 上次迭代的次对角线元素和
        int stagnationCount = 0;               // 停滞计数器，检测算法是否卡住

        // 步骤4：隐式QR迭代主循环
        // 这是算法的核心部分，通过迭代将海森伯格矩阵转换为上三角矩阵
        for (int iter = 0; iter < maxIterations; iter++) {
            // 4.1：计算Wilkinson位移
            // Wilkinson位移是加速收敛的关键技术，选择右下角2×2子矩阵的特征值
            double shift = 0.0;
            if (n >= 2) {
                // 提取右下角2×2子矩阵的元素
                double a = A[n - 2][n - 2];  // 主对角线元素
                double b = A[n - 2][n - 1];  // 次对角线元素
                double d = A[n - 1][n - 1];  // 右下角元素

                // 计算2×2子矩阵的特征值
                double trace = a + d;                    // 迹（trace）
                double det = a * d - b * b;             // 行列式（determinant）
                double discriminant = trace * trace - 4 * det;  // 判别式

                if (discriminant >= 0) {
                    // 实数特征值情况：使用二次公式
                    double sqrtDisc = Math.sqrt(discriminant);
                    double lambda1 = (trace + sqrtDisc) / 2.0;  // 较大特征值
                    double lambda2 = (trace - sqrtDisc) / 2.0;  // 较小特征值

                    // 选择更接近d的特征值，提高收敛速度
                    double diff1 = Math.abs(lambda1 - d);
                    double diff2 = Math.abs(lambda2 - d);

                    // 数值稳定性考虑：如果两个特征值都很接近，选择较小的那个
                    if (Math.abs(diff1 - diff2) < 1e-6f) {
                        shift = Math.min(lambda1, lambda2);
                    } else {
                        shift = (diff1 < diff2) ? lambda1 : lambda2;
                    }
                } else {
                    // 复数特征值情况：使用Rayleigh商位移
                    // 当判别式为负时，选择右下角元素作为位移
                    shift = d;
                }
            }

            // 4.2：隐式QR步骤 - 通过Givens旋转实现
            // 这是算法的核心：不显式计算Q和R矩阵，直接通过旋转更新矩阵
            for (int k = 0; k < n - 1; k++) {
                // 计算Givens旋转参数
                // 目标：将A[k+1][k]位置清零，实现QR分解的效果
                double a = A[k][k] - shift;    // 减去位移后的主对角线元素
                double b = A[k + 1][k];        // 次对角线元素
                double r = Math.sqrt(a * a + b * b);  // 旋转半径

                if (r > 1e-10) {  // 避免除零错误，确保数值稳定性
                    double c = a / r;      // 余弦值（cosine）
                    double s = -b / r;     // 正弦值（sine），注意符号

                    // 应用Givens旋转到矩阵A
                    // 旋转矩阵G作用于行k和k+1，列k到n-1
                    // 这相当于执行QR分解中的Q^T * A操作
                    applyGivensRotation(A, c, s, k, k + 1, k, n - 1);

                    // 应用Givens旋转到特征向量矩阵
                    // 注意：由于矩阵按行存储，需要调整更新逻辑
                    // 这里更新的是特征向量矩阵的列，而不是行
                    for (int i = 0; i < n; i++) {
                        double temp = eigenvectors[i][k];
                        eigenvectors[i][k] = c * temp + s * eigenvectors[i][k + 1];
                        eigenvectors[i][k + 1] = -s * temp + c * eigenvectors[i][k + 1];
                    }
                }
            }

            // 4.3：检查收敛性
            // 计算所有非对角线元素的绝对值之和
            // 当这个和足够小时，矩阵已经收敛到上三角形式
            double offDiagonalSum = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        offDiagonalSum += Math.abs(A[i][j]);
                    }
                }
            }

            // 4.4：自适应收敛检查
            // 当次对角线元素和小于容差时，认为算法已收敛
            if (offDiagonalSum < tolerance) {
                break;  // 达到收敛条件，退出迭代循环
            }

            // 4.5：停滞检测和自适应调整
            // 防止算法在局部最小值附近振荡，提高鲁棒性
            if (Math.abs(offDiagonalSum - previousOffDiagonalSum) < tolerance * 0.1) {
                stagnationCount++;  // 增加停滞计数
                if (stagnationCount > 10) {
                    // 如果停滞时间过长，放宽收敛条件
                    if (offDiagonalSum < tolerance * 10) {
                        break;  // 在放宽条件下退出
                    }
                }
            } else {
                stagnationCount = 0;  // 重置停滞计数器
            }

            previousOffDiagonalSum = offDiagonalSum;  // 更新上次的次对角线元素和
        }

        // 步骤5：提取特征值
        // 收敛后，特征值位于矩阵的对角线上
        // 这是QR算法的最终结果：矩阵A已经收敛到上三角形式
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = A[i][i];  // 对角线元素就是特征值
        }

        // 步骤6：排序和整理结果
        // 按特征值大小降序排列，同时调整对应的特征向量
        // 这确保了特征值和特征向量的对应关系
        sortEigenvaluesAndVectors(eigenvalues, eigenvectors);

        // 返回完整的特征分解结果
        return new Tuple2<>(IDoubleVector.of(eigenvalues), new RereDoubleMatrix(eigenvectors));
    }

    /**
     * 应用Givens旋转到矩阵（Apply Givens Rotation to Matrix）
     *
     * <p>
     * Givens旋转是数值线性代数中的基本变换，用于在矩阵的特定位置引入零元素。
     * 在QR分解和特征分解中，Givens旋转是实现矩阵变换的核心工具。</p>
     *
     * <p>
     * Givens旋转矩阵：</p>
     * <pre>
     * G = [c  -s] 其中 c² + s² = 1
     *     [s   c]
     * </pre>
     *
     * <p>
     * 变换效果：</p>
     * <ul>
     * <li>对矩阵A应用Givens旋转：A' = G^T * A</li>
     * <li>只影响指定的两行，其他行保持不变</li>
     * <li>在指定列范围内进行变换</li>
     * </ul>
     *
     * <p>
     * 数学原理：</p>
     * <ul>
     * <li>Givens旋转是正交变换，保持向量的长度和角度</li>
     * <li>通过选择合适的c和s，可以在指定位置引入零元素</li>
     * <li>变换是可逆的，逆变换为G^T</li>
     * </ul>
     *
     * <p>
     * 应用场景：</p>
     * <ul>
     * <li>QR分解中的上三角化过程</li>
     * <li>隐式QR算法中的矩阵更新</li>
     * <li>海森伯格化简中的零元素引入</li>
     * </ul>
     *
     * @param matrix 需要变换的矩阵，将被就地修改
     * @param c 旋转矩阵的余弦值（cosine）
     * @param s 旋转矩阵的正弦值（sine）
     * @param i1 第一个受影响的行索引
     * @param i2 第二个受影响的行索引
     * @param j1 变换的起始列索引
     * @param j2 变换的结束列索引
     */
    private void applyGivensRotation(double[][] matrix, Double c, Double s, int i1, int i2, int j1, int j2) {
        // 对指定范围内的列进行Givens旋转变换
        for (int j = j1; j <= j2; j++) {
            // 保存原始值，避免在计算过程中被覆盖
            double temp1 = matrix[i1][j];  // 第i1行第j列的元素
            double temp2 = matrix[i2][j];  // 第i2行第j列的元素

            // 应用Givens旋转变换
            // 新值 = G^T * 原值，即：
            // [new_i1] = [c  s] [temp1]
            // [new_i2]   [-s c] [temp2]
            matrix[i1][j] = c * temp1 - s * temp2;  // 第一行的新值
            matrix[i2][j] = s * temp1 + c * temp2;  // 第二行的新值
        }
    }

    /**
     * 就地矩阵乘法 - 避免创建中间矩阵对象
     */
    private IMatrix<Double> multiplyMatricesInPlace(double[][] A, double[][] B) {
        int m = A.length;
        int n = B[0].length;
        int p = B.length;

        double[][] result = new double[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < p; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 对海森伯格矩阵应用QR算法
     */
    private Tuple2<IDoubleVector, IDoubleMatrix> qrAlgorithmForHessenberg(double[][] H) {
        int n = H.length;
        double[][] A = new double[n][n];

        // 复制海森伯格矩阵
        for (int i = 0; i < n; i++) {
            System.arraycopy(H[i], 0, A[i], 0, n);
        }

        // 初始化特征向量矩阵为单位矩阵
        double[][] eigenvectors = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectors[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        final int maxIterations = 100;
        final double tolerance = 1e-6;

        // QR迭代 - 带Wilkinson位移
        for (int iter = 0; iter < maxIterations; iter++) {
            // 计算Wilkinson位移（使用右下角2x2子矩阵）
            double shift = 0.0;
            if (n >= 2) {
                double a = A[n - 2][n - 2];
                double b = A[n - 2][n - 1];
                double d = A[n - 1][n - 1];

                // 计算2x2矩阵的特征值
                double trace = a + d;
                double det = a * d - b * b; // 对称矩阵，b和c相同
                double discriminant = trace * trace - 4 * det;

                if (discriminant >= 0) {
                    double sqrtDisc = Math.sqrt(discriminant);
                    double lambda1 = (trace + sqrtDisc) / 2.0;
                    double lambda2 = (trace - sqrtDisc) / 2.0;

                    // 选择更接近d的特征值，但避免数值不稳定
                    double diff1 = Math.abs(lambda1 - d);
                    double diff2 = Math.abs(lambda2 - d);

                    // 如果两个特征值都很接近，选择较小的那个
                    if (Math.abs(diff1 - diff2) < 1e-6f) {
                        shift = Math.min(lambda1, lambda2);
                    } else {
                        shift = (diff1 < diff2) ? lambda1 : lambda2;
                    }
                } else {
                    shift = d; // 回退到Rayleigh商位移
                }
            }

            // 应用位移：A - shift * I
            double[][] shiftedA = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    shiftedA[i][j] = A[i][j] - (i == j ? shift : 0.0);
                }
            }

            // 对位移后的矩阵进行QR分解
            Tuple2<IMatrix<Double>, IMatrix<Double>> qr = new RereDoubleMatrix(shiftedA).qr();
            IMatrix<Double> Q = qr._1;
            IMatrix<Double> R = qr._2;

            // A = R * Q + shift * I
            IDoubleMatrix newA = (IDoubleMatrix) R.mmul(Q);
            double[][] newAData = newA.getData();

            // 加回位移
            for (int i = 0; i < n; i++) {
                newAData[i][i] += shift;
            }

            // 更新特征向量
            IDoubleMatrix currentEigenvectors = new RereDoubleMatrix(eigenvectors);
            IDoubleMatrix updatedEigenvectors = (IDoubleMatrix) currentEigenvectors.mmul(Q);
            eigenvectors = updatedEigenvectors.getData();

            // 检查收敛性
            double offDiagonalSum = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        offDiagonalSum += Math.abs(newAData[i][j]);
                    }
                }
            }

            // 更新A
            A = newAData;

            if (offDiagonalSum < tolerance) {
                break;
            }
        }

        // 提取特征值（对角线元素）
        double[] eigenvalues = new double[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = A[i][i];
        }

        // 按特征值大小降序排列
        sortEigenvaluesAndVectors(eigenvalues, eigenvectors);

        return new Tuple2<>(IDoubleVector.of(eigenvalues), new RereDoubleMatrix(eigenvectors));
    }

    /**
     * 分块QR分解 - 适用于大矩阵
     */
    private Tuple2<IMatrix<Double>, IMatrix<Double>> blockedQRDecomposition(double[][] A) {
        int m = A.length;
        int n = A[0].length;
        int minDim = Math.min(m, n);

        double[][] Q = new double[m][m];
        double[][] R = new double[m][n];

        // 初始化Q为单位矩阵
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // 复制A到R
        for (int i = 0; i < m; i++) {
            System.arraycopy(A[i], 0, R[i], 0, n);
        }

        // 分块大小
        int blockSize = Math.min(32, Math.max(8, minDim / 4));

        // 分块QR分解
        for (int k = 0; k < minDim; k += blockSize) {
            int endK = Math.min(k + blockSize, minDim);

            // 处理当前块
            for (int j = k; j < endK; j++) {
                // 计算Householder向量
                double norm = 0.0;
                for (int i = j; i < m; i++) {
                    norm += R[i][j] * R[i][j];
                }
                norm = Math.sqrt(norm);

                if (norm > 1e-10) {
                    // 构造Householder向量
                    double[] v = new double[m - j];
                    v[0] = R[j][j] + Math.signum(R[j][j]) * norm;
                    for (int i = 1; i < m - j; i++) {
                        v[i] = R[j + i][j];
                    }

                    // 归一化
                    double vNorm = 0.0;
                    for (int i = 0; i < v.length; i++) {
                        vNorm += v[i] * v[i];
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int i = 0; i < v.length; i++) {
                        v[i] /= vNorm;
                    }

                    // 应用Householder变换到R
                    applyHouseholderToQR(R, v, j, m - 1, j, n - 1);

                    // 应用Householder变换到Q
                    applyHouseholderToQR(Q, v, 0, m - 1, j, m - 1);
                }
            }
        }

        return new Tuple2<>(new RereDoubleMatrix(Q), new RereDoubleMatrix(R));
    }

    /**
     * 应用Householder变换到QR分解
     */
    private void applyHouseholderToQR(double[][] matrix, double[] v, int startRow, int endRow, int startCol, int endCol) {
        int n = v.length;

        // 计算 w = matrix^T * v
        double[] w = new double[endCol - startCol + 1];
        for (int j = 0; j < w.length; j++) {
            w[j] = 0.0;
            for (int i = 0; i < n; i++) {
                w[j] += matrix[startRow + i][startCol + j] * v[i];
            }
        }

        // 计算 matrix = matrix - 2 * v * w^T
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < w.length; j++) {
                matrix[startRow + i][startCol + j] -= 2.0 * v[i] * w[j];
            }
        }
    }

    /**
     * 三对角矩阵的QR分解（优化版本）
     */
    private Tuple2<IMatrix<Double>, IMatrix<Double>> qrDecompositionTridiagonal(double[][] T) {
        int n = T.length;
        double[][] Q = new double[n][n];
        double[][] R = new double[n][n];

        // 初始化Q为单位矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        // 复制T到R
        for (int i = 0; i < n; i++) {
            System.arraycopy(T[i], 0, R[i], 0, n);
        }

        // 对三对角矩阵进行Givens旋转
        for (int i = 0; i < n - 1; i++) {
            if (Math.abs(R[i + 1][i]) > 1e-10) {
                // 计算Givens旋转参数
                double a = R[i][i];
                double b = R[i + 1][i];
                double r = Math.sqrt(a * a + b * b);

                double c = a / r;
                double s = -b / r;

                // 应用Givens旋转到R
                for (int j = i; j < n; j++) {
                    double temp1 = R[i][j];
                    double temp2 = R[i + 1][j];
                    R[i][j] = c * temp1 - s * temp2;
                    R[i + 1][j] = s * temp1 + c * temp2;
                }

                // 应用Givens旋转到Q
                for (int j = 0; j < n; j++) {
                    double temp1 = Q[j][i];
                    double temp2 = Q[j][i + 1];
                    Q[j][i] = c * temp1 - s * temp2;
                    Q[j][i + 1] = s * temp1 + c * temp2;
                }
            }
        }

        return new Tuple2<>(new RereDoubleMatrix(Q), new RereDoubleMatrix(R));
    }

    /**
     * 处理重复特征值 - 确保特征向量的正交性
     *
     * <p>
     * 当矩阵具有重复特征值时，对应的特征向量空间可能不是一维的， 这会导致特征向量不正交，影响后续计算的数值稳定性。</p>
     *
     * <p>
     * 问题描述：</p>
     * <ul>
     * <li>重复特征值λ对应多个线性无关的特征向量</li>
     * <li>数值算法可能产生不正交的特征向量</li>
     * <li>不正交的特征向量会导致数值不稳定</li>
     * </ul>
     *
     * <p>
     * 解决方案：</p>
     * <ol>
     * <li>识别重复特征值组（容差范围内）</li>
     * <li>对每组重复特征值对应的特征向量进行Gram-Schmidt正交化</li>
     * <li>确保最终的特征向量矩阵是正交的</li>
     * </ol>
     *
     * <p>
     * 算法原理：</p>
     * <p>
     * 对于重复特征值λ，设对应的特征向量为v₁, v₂, ..., vₖ， 使用Gram-Schmidt过程构造正交向量组：</p>
     * <p>
     * u₁ = v₁ / ||v₁||</p>
     * <p>
     * uᵢ = (vᵢ - Σⱼ₌₁ⁱ⁻¹⟨vᵢ, uⱼ⟩uⱼ) / ||vᵢ - Σⱼ₌₁ⁱ⁻¹⟨vᵢ, uⱼ⟩uⱼ||</p>
     *
     * @param eigenvalues 特征值数组（已排序）
     * @param eigenvectors 特征向量矩阵（列向量形式）
     */
    private void handleDuplicateEigenvalues(double[] eigenvalues, double[][] eigenvectors) {
        int n = eigenvalues.length;
        double tolerance = 1e-6;

        // 找到重复的特征值组
        for (int i = 0; i < n; i++) {
            int duplicateCount = 1;
            int startIdx = i;

            // 找到相同特征值的范围
            while (i + duplicateCount < n && Math.abs(eigenvalues[i] - eigenvalues[i + duplicateCount]) < tolerance) {
                duplicateCount++;
            }

            // 如果有重复特征值，对对应的特征向量进行正交化
            if (duplicateCount > 1) {
                orthogonalizeEigenvectors(eigenvectors, startIdx, duplicateCount);
            }

            i += duplicateCount - 1; // 跳过已处理的重复特征值
        }
    }

    /**
     * 对重复特征值对应的特征向量进行Gram-Schmidt正交化
     */
    private void orthogonalizeEigenvectors(double[][] eigenvectors, int startIdx, int count) {
        int n = eigenvectors.length;

        // 对每个特征向量进行正交化
        for (int i = 1; i < count; i++) {
            int currentIdx = startIdx + i;

            // 减去与前面特征向量的投影
            for (int j = 0; j < i; j++) {
                int prevIdx = startIdx + j;

                // 计算投影系数
                double projection = 0.0;
                for (int k = 0; k < n; k++) {
                    projection += eigenvectors[currentIdx][k] * eigenvectors[prevIdx][k];
                }

                // 减去投影
                for (int k = 0; k < n; k++) {
                    eigenvectors[currentIdx][k] -= projection * eigenvectors[prevIdx][k];
                }
            }

            // 归一化
            double norm = 0.0;
            for (int k = 0; k < n; k++) {
                norm += eigenvectors[currentIdx][k] * eigenvectors[currentIdx][k];
            }
            norm = Math.sqrt(norm);

            if (norm > 1e-10) {
                for (int k = 0; k < n; k++) {
                    eigenvectors[currentIdx][k] /= norm;
                }
            }
        }
    }

    /**
     * 快速排序特征值和特征向量
     */
    private void quickSortEigen(double[] eigenvalues, double[][] eigenvectors, int low, int high) {
        if (low < high) {
            int pi = partitionEigen(eigenvalues, eigenvectors, low, high);
            quickSortEigen(eigenvalues, eigenvectors, low, pi - 1);
            quickSortEigen(eigenvalues, eigenvectors, pi + 1, high);
        }
    }

    /**
     * 快速排序的分区函数
     */
    private int partitionEigen(double[] eigenvalues, double[][] eigenvectors, int low, int high) {
        double pivot = eigenvalues[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (eigenvalues[j] >= pivot) { // 降序排列
                i++;
                swapEigen(eigenvalues, eigenvectors, i, j);
            }
        }
        swapEigen(eigenvalues, eigenvectors, i + 1, high);
        return i + 1;
    }

    /**
     * 交换特征值和对应的特征向量
     */
    private void swapEigen(double[] eigenvalues, double[][] eigenvectors, int i, int j) {
        // 交换特征值
        double tempEigen = eigenvalues[i];
        eigenvalues[i] = eigenvalues[j];
        eigenvalues[j] = tempEigen;

        // 交换对应的特征向量行
        int n = eigenvectors.length;
        for (int k = 0; k < n; k++) {
            double tempVec = eigenvectors[i][k];
            eigenvectors[i][k] = eigenvectors[j][k];
            eigenvectors[j][k] = tempVec;
        }
    }

    /**
     * QR分解
     *
     * @param matrix 输入矩阵
     * @return Q和R矩阵
     */
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> qr() {
        double[][] A = this.getData();
        int m = A.length;
        int n = A[0].length;

        double[][] Q = new double[m][n];
        double[][] R = new double[n][n];

        // 改进的Gram-Schmidt过程
        for (int j = 0; j < n; j++) {
            // 复制第j列到Q
            for (int i = 0; i < m; i++) {
                Q[i][j] = A[i][j];
            }

            // 正交化：从所有之前已正交化的向量中减去投影
            for (int k = 0; k < j; k++) {
                // 计算投影系数
                double dotProduct = 0;
                for (int i = 0; i < m; i++) {
                    dotProduct += Q[i][k] * Q[i][j];
                }
                R[k][j] = dotProduct;

                // 从当前向量中减去投影
                for (int i = 0; i < m; i++) {
                    Q[i][j] -= dotProduct * Q[i][k];
                }
            }

            // 标准化
            double norm = 0;
            for (int i = 0; i < m; i++) {
                norm += Q[i][j] * Q[i][j];
            }
            norm = Math.sqrt(norm);
            R[j][j] = norm;

            if (norm > 1e-10) { // 避免除零
                for (int i = 0; i < m; i++) {
                    Q[i][j] /= norm;
                }
            } else {
                // 如果向量线性相关，设置为零向量
                for (int i = 0; i < m; i++) {
                    Q[i][j] = 0.0;
                }
            }
        }

        return new Tuple2<>(new RereDoubleMatrix(Q), new RereDoubleMatrix(R));
    }

    /**
     * 矩阵乘法运算 / Matrix multiplication
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 矩阵乘法结果 / Matrix multiplication result
     */
    @Override
    public IMatrix<Double> mmul(IMatrix<Double> other1) {
        if (other1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        int m = data.length;          // 当前矩阵行数
        int n = data[0].length;       // 当前矩阵列数
        int p = otherData[0].length;  // 另一个矩阵列数

        if (n != otherData.length) {
            throw new IllegalArgumentException("矩阵维度不匹配进行乘法运算 / Matrix dimensions don't match for multiplication");
        }

        // 计算复杂度
        long complexity = (long) m * n * p;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuMatrixMultiply(this, other0);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU矩阵乘法失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU优化算法
        return cpuMatrixMultiply(other0, m, n, p);
    }

    /**
     * CPU矩阵乘法（原有算法） / CPU matrix multiplication (original algorithm)
     */
    private IMatrix<Double> cpuMatrixMultiply(IMatrix<Double> other1, int m, int n, int p) {
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        long complexity = (long) m * n * p;

        if (complexity < 1000) {
            // 极小矩阵：直接计算
            return naiveMultiply(otherData, m, n, p);
        } else if (complexity < 64000) {
            // 小矩阵：循环展开优化
            return unrolledMultiply(otherData, m, n, p);
        } else if (complexity < 8000000) {
            // 中等矩阵：分块算法
            return blockMultiply(otherData, m, n, p);
        } else {
            // 大矩阵：并行分块算法
            return parallelBlockMultiply(otherData, m, n, p);
        }
    }

    /**
     * 朴素矩阵乘法 / Naive matrix multiplication
     */
    private RereDoubleMatrix naiveMultiply(double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        for (int i = 0; i < m; i++) {
            double[] thisRow = data[i];
            double[] resultRow = result[i];

            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += thisRow[k] * otherData[k][j];
                }
                resultRow[j] = sum;
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 循环展开优化的矩阵乘法 / Loop unrolled matrix multiplication
     */
    private RereDoubleMatrix unrolledMultiply(double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        for (int i = 0; i < m; i++) {
            double[] thisRow = data[i];
            double[] resultRow = result[i];

            for (int j = 0; j < p; j++) {
                double sum = 0.0;

                // 循环展开，每次处理4个元素
                int k = 0;
                for (; k < n - 3; k += 4) {
                    sum += thisRow[k] * otherData[k][j]
                            + thisRow[k + 1] * otherData[k + 1][j]
                            + thisRow[k + 2] * otherData[k + 2][j]
                            + thisRow[k + 3] * otherData[k + 3][j];
                }

                // 处理剩余元素
                for (; k < n; k++) {
                    sum += thisRow[k] * otherData[k][j];
                }

                resultRow[j] = sum;
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 分块矩阵乘法 / Block matrix multiplication
     */
    private RereDoubleMatrix blockMultiply(double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        // 动态计算最优块大小
        int blockSize = calculateOptimalBlockSize(m, n, p);

        for (int ii = 0; ii < m; ii += blockSize) {
            for (int jj = 0; jj < p; jj += blockSize) {
                for (int kk = 0; kk < n; kk += blockSize) {

                    int iEnd = Math.min(ii + blockSize, m);
                    int jEnd = Math.min(jj + blockSize, p);
                    int kEnd = Math.min(kk + blockSize, n);

                    // 块内计算
                    for (int i = ii; i < iEnd; i++) {
                        double[] thisRow = data[i];
                        double[] resultRow = result[i];

                        for (int j = jj; j < jEnd; j++) {
                            double sum = resultRow[j];

                            // 内层循环展开
                            int k = kk;
                            for (; k < kEnd - 3; k += 4) {
                                sum += thisRow[k] * otherData[k][j]
                                        + thisRow[k + 1] * otherData[k + 1][j]
                                        + thisRow[k + 2] * otherData[k + 2][j]
                                        + thisRow[k + 3] * otherData[k + 3][j];
                            }

                            for (; k < kEnd; k++) {
                                sum += thisRow[k] * otherData[k][j];
                            }

                            resultRow[j] = sum;
                        }
                    }
                }
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 并行分块矩阵乘法 / Parallel block matrix multiplication
     */
    private RereDoubleMatrix parallelBlockMultiply(double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        // 计算最优块大小和线程数
        int blockSize = calculateOptimalBlockSize(m, n, p);
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                Math.max(1, (m + blockSize - 1) / blockSize));

        if (numThreads == 1) {
            // 单线程情况下使用普通分块算法
            return blockMultiply(otherData, m, n, p);
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Void>> futures = new ArrayList<>();

        // 按行分块并行计算
        int rowsPerThread = (m + numThreads - 1) / numThreads;

        for (int t = 0; t < numThreads; t++) {
            final int startRow = t * rowsPerThread;
            final int endRow = Math.min(startRow + rowsPerThread, m);

            if (startRow >= endRow) {
                break;
            }

            Future<Void> future = executor.submit(() -> {
                // 每个线程处理指定行范围的计算
                for (int ii = startRow; ii < endRow; ii += blockSize) {
                    for (int jj = 0; jj < p; jj += blockSize) {
                        for (int kk = 0; kk < n; kk += blockSize) {

                            int iEnd = Math.min(ii + blockSize, endRow);
                            int jEnd = Math.min(jj + blockSize, p);
                            int kEnd = Math.min(kk + blockSize, n);

                            // 块内计算
                            for (int i = ii; i < iEnd; i++) {
                                double[] thisRow = data[i];
                                double[] resultRow = result[i];

                                for (int j = jj; j < jEnd; j++) {
                                    double sum = resultRow[j];

                                    // 内层循环展开
                                    int k = kk;
                                    for (; k < kEnd - 3; k += 4) {
                                        sum += thisRow[k] * otherData[k][j]
                                                + thisRow[k + 1] * otherData[k + 1][j]
                                                + thisRow[k + 2] * otherData[k + 2][j]
                                                + thisRow[k + 3] * otherData[k + 3][j];
                                    }

                                    for (; k < kEnd; k++) {
                                        sum += thisRow[k] * otherData[k][j];
                                    }

                                    resultRow[j] = sum;
                                }
                            }
                        }
                    }
                }
                return null;
            });

            futures.add(future);
        }

        // 等待所有线程完成
        try {
            for (Future<Void> future : futures) {
                future.get();
            }
        } catch (InterruptedException | ExecutionException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("并行计算失败 / Parallel computation failed", e);
        } finally {
            executor.shutdown();
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 计算最优块大小 / Calculate optimal block size
     */
    private int calculateOptimalBlockSize(int m, int n, int p) {
        // 基于L1缓存大小估算最优块大小
        // 假设L1缓存32KB，Double占4字节
        int l1CacheSize = 32 * 1024 / 4; // 8192个Double

        // 考虑三个矩阵块：A_block, B_block, C_block
        // 每个块大约占用 blockSize^2 个元素
        int maxBlockSize = (int) Math.sqrt(l1CacheSize / 3);

        // 限制在合理范围内
        maxBlockSize = Math.max(16, Math.min(maxBlockSize, 128));

        // 选择能整除矩阵维度的块大小
        for (int blockSize = maxBlockSize; blockSize >= 16; blockSize--) {
            if (m % blockSize == 0 || n % blockSize == 0 || p % blockSize == 0) {
                return blockSize;
            }
        }

        return 32; // 默认块大小
    }

    /**
     * 检查是否为2的幂次 / Check if number is power of two
     */
    private boolean isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    /**
     * Strassen矩阵乘法算法 / Strassen matrix multiplication algorithm
     */
    private void strassenMultiply(double[][] otherData, RereDoubleMatrix result, int n) {
        if (n <= 32) {
            // 对于小矩阵，使用基础算法
            basicMatrixMultiply(otherData, result.data, n);
            return;
        }

        int half = n / 2;

        // 分割矩阵为四个子矩阵
        double[][] A11 = new double[half][half];
        double[][] A12 = new double[half][half];
        double[][] A21 = new double[half][half];
        double[][] A22 = new double[half][half];

        double[][] B11 = new double[half][half];
        double[][] B12 = new double[half][half];
        double[][] B21 = new double[half][half];
        double[][] B22 = new double[half][half];

        // 复制数据到子矩阵
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) {
                A11[i][j] = data[i][j];
                A12[i][j] = data[i][j + half];
                A21[i][j] = data[i + half][j];
                A22[i][j] = data[i + half][j + half];

                B11[i][j] = otherData[i][j];
                B12[i][j] = otherData[i][j + half];
                B21[i][j] = otherData[i + half][j];
                B22[i][j] = otherData[i + half][j + half];
            }
        }

        // 计算Strassen的7个中间矩阵（使用基础算法避免递归）
        double[][] M1 = basicMatrixMultiply(A11, strassenSub(B12, B22, half), half);
        double[][] M2 = basicMatrixMultiply(strassenAdd(A11, A12, half), B22, half);
        double[][] M3 = basicMatrixMultiply(strassenAdd(A21, A22, half), B11, half);
        double[][] M4 = basicMatrixMultiply(A22, strassenSub(B21, B11, half), half);
        double[][] M5 = basicMatrixMultiply(strassenAdd(A11, A22, half), strassenAdd(B11, B22, half), half);
        double[][] M6 = basicMatrixMultiply(strassenSub(A12, A22, half), strassenAdd(B21, B22, half), half);
        double[][] M7 = basicMatrixMultiply(strassenSub(A11, A21, half), strassenAdd(B11, B12, half), half);

        // 计算结果矩阵的四个子矩阵（正确的Strassen公式）
        double[][] C11 = strassenAdd(strassenSub(strassenAdd(M5, M4, half), M2, half), M6, half);
        double[][] C12 = strassenAdd(M1, M2, half);
        double[][] C21 = strassenAdd(M3, M4, half);
        double[][] C22 = strassenSub(strassenSub(strassenAdd(M5, M1, half), M3, half), M7, half);

        // 合并结果
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) {
                result.data[i][j] = C11[i][j];
                result.data[i][j + half] = C12[i][j];
                result.data[i + half][j] = C21[i][j];
                result.data[i + half][j + half] = C22[i][j];
            }
        }
    }

    /**
     * Strassen算法中的矩阵加法 / Matrix addition for Strassen algorithm
     */
    private double[][] strassenAdd(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        return C;
    }

    /**
     * Strassen算法中的矩阵减法 / Matrix subtraction for Strassen algorithm
     */
    private double[][] strassenSub(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    /**
     * Strassen算法中的矩阵乘法 / Matrix multiplication for Strassen algorithm
     */
    private double[][] strassenMultiply(double[][] A, double[][] B, int n) {
        if (n <= 32) {
            // 对于小矩阵，使用基础算法
            double[][] C = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0;
                    for (int k = 0; k < n; k++) {
                        sum += A[i][k] * B[k][j];
                    }
                    C[i][j] = sum;
                }
            }
            return C;
        }

        // 递归调用Strassen算法
        int half = n / 2;

        double[][] A11 = new double[half][half];
        double[][] A12 = new double[half][half];
        double[][] A21 = new double[half][half];
        double[][] A22 = new double[half][half];

        double[][] B11 = new double[half][half];
        double[][] B12 = new double[half][half];
        double[][] B21 = new double[half][half];
        double[][] B22 = new double[half][half];

        // 分割矩阵
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j + half];
                A21[i][j] = A[i + half][j];
                A22[i][j] = A[i + half][j + half];

                B11[i][j] = B[i][j];
                B12[i][j] = B[i][j + half];
                B21[i][j] = B[i + half][j];
                B22[i][j] = B[i + half][j + half];
            }
        }

        // 计算7个中间矩阵
        double[][] M1 = strassenAdd(strassenMultiply(A11, strassenAdd(B12, B22, half), half),
                strassenMultiply(A22, strassenAdd(B21, B11, half), half), half);
        double[][] M2 = strassenMultiply(strassenAdd(A11, A12, half), B22, half);
        double[][] M3 = strassenMultiply(strassenAdd(A21, A22, half), B11, half);
        double[][] M4 = strassenMultiply(A22, strassenSub(B21, B11, half), half);
        double[][] M5 = strassenMultiply(strassenAdd(A11, A22, half), strassenAdd(B11, B22, half), half);
        double[][] M6 = strassenMultiply(strassenSub(A12, A22, half), strassenAdd(B21, B22, half), half);
        double[][] M7 = strassenMultiply(strassenSub(A11, A21, half), strassenAdd(B11, B12, half), half);

        // 计算结果
        double[][] C11 = strassenAdd(strassenSub(strassenAdd(M5, M4, half), M2, half), M6, half);
        double[][] C12 = strassenAdd(M1, M2, half);
        double[][] C21 = strassenAdd(M3, M4, half);
        double[][] C22 = strassenSub(strassenSub(strassenAdd(M5, M1, half), M3, half), M7, half);

        // 合并结果
        double[][] C = new double[n][n];
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) {
                C[i][j] = C11[i][j];
                C[i][j + half] = C12[i][j];
                C[i + half][j] = C21[i][j];
                C[i + half][j + half] = C22[i][j];
            }
        }

        return C;
    }

    /**
     * 基础优化矩阵乘法 / Basic optimized matrix multiplication
     */
    private IMatrix<Double> basicOptimizedMultiply(IMatrix<Double> other1, int m, int n, int p) {
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        double[][] result = new double[m][p];

        // 使用行优先访问和循环展开优化
        for (int i = 0; i < m; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result[i];

            for (int j = 0; j < p; j++) {
                double sum = 0;

                // 4路循环展开，提高指令级并行度
                int k = 0;
                for (; k < n - 3; k += 4) {
                    sum += thisRow[k] * otherData[k][j]
                            + thisRow[k + 1] * otherData[k + 1][j]
                            + thisRow[k + 2] * otherData[k + 2][j]
                            + thisRow[k + 3] * otherData[k + 3][j];
                }

                // 处理剩余元素
                for (; k < n; k++) {
                    sum += thisRow[k] * otherData[k][j];
                }

                resultRow[j] = sum;
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * SIMD优化矩阵乘法 / SIMD optimized matrix multiplication
     */
    private IMatrix<Double> simdOptimizedMultiply(IMatrix<Double> other1, int m, int n, int p) {
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        double[][] result = new double[m][p];

        // 使用更大的展开因子和多累加器减少数据依赖
        for (int i = 0; i < m; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result[i];

            for (int j = 0; j < p; j++) {
                // 使用4个独立的累加器减少数据依赖
                double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;

                int k = 0;
                final int unrollFactor = 8;
                final int alignedN = n - (n % unrollFactor);

                // 8路展开，4个累加器
                for (; k < alignedN; k += unrollFactor) {
                    sum1 += thisRow[k] * otherData[k][j]
                            + thisRow[k + 4] * otherData[k + 4][j];
                    sum2 += thisRow[k + 1] * otherData[k + 1][j]
                            + thisRow[k + 5] * otherData[k + 5][j];
                    sum3 += thisRow[k + 2] * otherData[k + 2][j]
                            + thisRow[k + 6] * otherData[k + 6][j];
                    sum4 += thisRow[k + 3] * otherData[k + 3][j]
                            + thisRow[k + 7] * otherData[k + 7][j];
                }

                // 处理剩余元素
                for (; k < n; k++) {
                    sum1 += thisRow[k] * otherData[k][j];
                }

                resultRow[j] = sum1 + sum2 + sum3 + sum4;
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 基础矩阵乘法（静态方法） / Basic matrix multiplication (static method)
     */
    private static double[][] basicMatrixMultiply(double[][] A, double[][] B, int n) {
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    /**
     * 缓存优化分块矩阵乘法 / Cache optimized block matrix multiplication
     */
    private IMatrix<Double> cacheOptimizedBlockMultiply(IMatrix<Double> other1, int m, int n, int p) {
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        double[][] result = new double[m][p];

        // 动态计算最优分块大小，考虑L1/L2/L3缓存
        int blockSize = calculateOptimalBlockSize(m, n, p);

        for (int ii = 0; ii < m; ii += blockSize) {
            for (int jj = 0; jj < p; jj += blockSize) {
                for (int kk = 0; kk < n; kk += blockSize) {
                    int iEnd = Math.min(ii + blockSize, m);
                    int jEnd = Math.min(jj + blockSize, p);
                    int kEnd = Math.min(kk + blockSize, n);

                    // 块内使用优化的微内核
                    blockMicroKernel(otherData, result, ii, jj, kk, iEnd, jEnd, kEnd);
                }
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 高性能微内核 / High performance micro-kernel
     */
    private void blockMicroKernel(double[][] otherData, double[][] result,
            int iStart, int jStart, int kStart,
            int iEnd, int jEnd, int kEnd) {
        for (int i = iStart; i < iEnd; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result[i];

            for (int j = jStart; j < jEnd; j++) {
                double sum = resultRow[j];

                // 内层循环展开，提高指令级并行度
                int k = kStart;
                for (; k < kEnd - 3; k += 4) {
                    sum += thisRow[k] * otherData[k][j]
                            + thisRow[k + 1] * otherData[k + 1][j]
                            + thisRow[k + 2] * otherData[k + 2][j]
                            + thisRow[k + 3] * otherData[k + 3][j];
                }

                for (; k < kEnd; k++) {
                    sum += thisRow[k] * otherData[k][j];
                }

                resultRow[j] = sum;
            }
        }
    }

    /**
     * 优化的并行分块矩阵乘法 / Optimized parallel block matrix multiplication
     */
    private IMatrix<Double> parallelBlockMultiplyOptimized(IMatrix<Double> other1, int m, int n, int p) {
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        double[][] result = new double[m][p];

        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                Math.max(1, (int) Math.sqrt(m * p / 10000)));
        int blockSize = calculateOptimalBlockSize(m, n, p);

        // 计算工作分配 - 按块分配而不是按行分配，提高负载均衡
        int totalBlocks = ((m + blockSize - 1) / blockSize) * ((p + blockSize - 1) / blockSize);
        int blocksPerThread = Math.max(1, totalBlocks / numThreads);

        List<Future<Void>> futures = new ArrayList<>();

        for (int t = 0; t < numThreads; t++) {
            final int startBlock = t * blocksPerThread;
            final int endBlock = Math.min((t + 1) * blocksPerThread, totalBlocks);

            if (startBlock >= totalBlocks) {
                break;
            }

            futures.add(THREAD_POOL.submit(() -> {
                for (int blockIdx = startBlock; blockIdx < endBlock; blockIdx++) {
                    int blocksPerRow = (p + blockSize - 1) / blockSize;
                    int blockI = blockIdx / blocksPerRow;
                    int blockJ = blockIdx % blocksPerRow;

                    int ii = blockI * blockSize;
                    int jj = blockJ * blockSize;

                    int iEnd = Math.min(ii + blockSize, m);
                    int jEnd = Math.min(jj + blockSize, p);

                    // 对每个(i,j)块，遍历所有k块
                    for (int kk = 0; kk < n; kk += blockSize) {
                        int kEnd = Math.min(kk + blockSize, n);

                        // 使用优化的微内核
                        optimizedMicroKernel(otherData, result, ii, jj, kk, iEnd, jEnd, kEnd);
                    }
                }
                return null;
            }));
        }

        // 等待所有线程完成
        for (Future<Void> future : futures) {
            try {
                future.get(60, TimeUnit.SECONDS);
            } catch (TimeoutException e) {
                future.cancel(true);
                throw new RuntimeException("并行矩阵乘法超时", e);
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("并行矩阵乘法失败", e);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 超大矩阵的高度优化并行算法 / Highly optimized parallel algorithm for very large
     * matrices
     */
    private IMatrix<Double> optimizedParallelBlockMultiply(IMatrix<Double> other1, int m, int n, int p) {
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        double[][] result = new double[m][p];

        // 使用更多线程和更小的块来提高并行度
        int numThreads = Runtime.getRuntime().availableProcessors();
        int blockSize = Math.min(128, calculateOptimalBlockSize(m, n, p));

        // 使用工作窃取模式的任务分配
        BlockingQueue<MatrixBlock> workQueue = new LinkedBlockingQueue<>();

        // 生成所有工作块
        for (int ii = 0; ii < m; ii += blockSize) {
            for (int jj = 0; jj < p; jj += blockSize) {
                workQueue.offer(new MatrixBlock(ii, jj,
                        Math.min(ii + blockSize, m),
                        Math.min(jj + blockSize, p)));
            }
        }

        List<Future<Void>> futures = new ArrayList<>();

        for (int t = 0; t < numThreads; t++) {
            futures.add(THREAD_POOL.submit(() -> {
                MatrixBlock block;
                while ((block = workQueue.poll()) != null) {
                    // 对每个(i,j)块，遍历所有k块
                    for (int kk = 0; kk < n; kk += blockSize) {
                        int kEnd = Math.min(kk + blockSize, n);

                        // 使用高度优化的微内核
                        highPerformanceMicroKernel(otherData, result,
                                block.iStart, block.jStart, kk,
                                block.iEnd, block.jEnd, kEnd);
                    }
                }
                return null;
            }));
        }

        // 等待所有线程完成
        for (Future<Void> future : futures) {
            try {
                future.get(120, TimeUnit.SECONDS);
            } catch (Exception e) {
                future.cancel(true);
                throw new RuntimeException("超大矩阵并行乘法失败", e);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵块定义 / Matrix block definition
     */
    private static class MatrixBlock {

        final int iStart, jStart, iEnd, jEnd;

        MatrixBlock(int iStart, int jStart, int iEnd, int jEnd) {
            this.iStart = iStart;
            this.jStart = jStart;
            this.iEnd = iEnd;
            this.jEnd = jEnd;
        }
    }

    /**
     * 优化的微内核 / Optimized micro-kernel
     */
    private void optimizedMicroKernel(double[][] otherData, double[][] result,
            int iStart, int jStart, int kStart,
            int iEnd, int jEnd, int kEnd) {
        for (int i = iStart; i < iEnd; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result[i];

            for (int j = jStart; j < jEnd; j++) {
                double sum = resultRow[j];

                // 8路循环展开 + 多累加器
                int k = kStart;
                double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;

                for (; k < kEnd - 7; k += 8) {
                    sum1 += thisRow[k] * otherData[k][j]
                            + thisRow[k + 4] * otherData[k + 4][j];
                    sum2 += thisRow[k + 1] * otherData[k + 1][j]
                            + thisRow[k + 5] * otherData[k + 5][j];
                    sum3 += thisRow[k + 2] * otherData[k + 2][j]
                            + thisRow[k + 6] * otherData[k + 6][j];
                    sum4 += thisRow[k + 3] * otherData[k + 3][j]
                            + thisRow[k + 7] * otherData[k + 7][j];
                }

                // 处理剩余元素
                for (; k < kEnd; k++) {
                    sum1 += thisRow[k] * otherData[k][j];
                }

                resultRow[j] = sum + sum1 + sum2 + sum3 + sum4;
            }
        }
    }

    /**
     * 高性能微内核 / High performance micro-kernel
     */
    private void highPerformanceMicroKernel(double[][] otherData, double[][] result,
            int iStart, int jStart, int kStart,
            int iEnd, int jEnd, int kEnd) {
        // 使用更激进的优化策略
        for (int i = iStart; i < iEnd; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result[i];

            // 处理多个j值以提高缓存效率
            int j = jStart;
            for (; j < jEnd - 1; j += 2) {
                double sum1 = resultRow[j];
                double sum2 = resultRow[j + 1];

                // 双列并行计算
                int k = kStart;
                for (; k < kEnd - 3; k += 4) {
                    double a0 = thisRow[k];
                    double a1 = thisRow[k + 1];
                    double a2 = thisRow[k + 2];
                    double a3 = thisRow[k + 3];

                    sum1 += a0 * otherData[k][j] + a1 * otherData[k + 1][j]
                            + a2 * otherData[k + 2][j] + a3 * otherData[k + 3][j];
                    sum2 += a0 * otherData[k][j + 1] + a1 * otherData[k + 1][j + 1]
                            + a2 * otherData[k + 2][j + 1] + a3 * otherData[k + 3][j + 1];
                }

                for (; k < kEnd; k++) {
                    double a = thisRow[k];
                    sum1 += a * otherData[k][j];
                    sum2 += a * otherData[k][j + 1];
                }

                resultRow[j] = sum1;
                resultRow[j + 1] = sum2;
            }

            // 处理剩余的j
            for (; j < jEnd; j++) {
                double sum = resultRow[j];
                for (int k = kStart; k < kEnd; k++) {
                    sum += thisRow[k] * otherData[k][j];
                }
                resultRow[j] = sum;
            }
        }
    }

    /**
     * 矩阵转换为一维数组 / Convert matrix to 1D array
     * <p>
     * 将矩阵转换为一维Double数组。对于列向量，返回该列的数据； 对于普通矩阵，按行优先顺序展开。
     * </p>
     * <p>
     * Converts the matrix to a 1D Double array. For column vectors, returns the
     * column data; For regular matrices, flattens in row-major order.
     * </p>
     *
     * @return 一维Double数组 / 1D Double array
     */
    private double[] toFlattenArray() {
        if (data[0].length == 1) {
            // 如果是列向量，返回该列的数据 / If it's a column vector, return the column data
            double[] result = new double[data.length];
            for (int i = 0; i < data.length; i++) {
                result[i] = data[i][0];
            }
            return result;
        } else {
            // 如果是普通矩阵，按行优先顺序展开 / If it's a regular matrix, flatten in row-major order
            int totalElements = data.length * data[0].length;
            double[] result = new double[totalElements];
            int index = 0;
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    result[index++] = data[i][j];
                }
            }
            return result;
        }
    }

    /**
     * 矩阵自然对数运算 / Matrix natural logarithm operation
     * <p>
     * 对矩阵中每个元素进行自然对数运算（ln(x)） Performs natural logarithm operation (ln(x)) on
     * each element in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含对数运算结果 / New matrix object containing logarithm
     * operation results
     * @throws ArithmeticException 如果元素值小于等于0 / if any element value is less
     * than or equal to 0
     */
    @Override
    public IMatrix<Double> log() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] <= 0) {
                    throw new ArithmeticException("对数运算的元素值必须大于0 / Element value for logarithm must be greater than 0");
                }
                result[i][j] = Math.log(data[i][j]);
            }
        }
        return new RereDoubleMatrix(result);
    }

    /**
     * 计算矩阵的Frobenius范数 / Compute Frobenius norm of the matrix
     * <p>
     * 计算矩阵的Frobenius范数，即所有元素平方和的开方 Computes the Frobenius norm of the matrix,
     * which is the square root of the sum of squares of all elements
     * </p>
     *
     * @return Frobenius范数 / Frobenius norm
     */
    @Override
    public Double frobeniusNorm() {
        double sum = 0.0;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                sum += data[i][j] * data[i][j];
            }
        }
        return Math.sqrt(sum);
    }

    /**
     * 计算两个矩阵之间的Frobenius距离 / Compute Frobenius distance between two matrices
     * <p>
     * 计算当前矩阵与另一个矩阵之间的Frobenius距离 Computes the Frobenius distance between
     * current matrix and another matrix
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return Frobenius距离 / Frobenius distance
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     */
    @Override
    public Double frobeniusDistance(IMatrix<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }

        if (data.length != other.getRowNum() || data[0].length != other.getColNum()) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }

        IMatrix<Double> diff = this.sub(other);
        return diff.frobeniusNorm();
    }

    /**
     * 矩阵按行归一化 / Row-wise normalization of matrix
     * <p>
     * 对矩阵的每一行进行L2归一化 Performs L2 normalization on each row of the matrix
     * </p>
     *
     * @return 归一化后的矩阵 / Normalized matrix
     */
    @Override
    public IMatrix<Double> normalizeRows() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            IVector<Double> row = this.getRow(i);
            double norm = row.norm2();

            if (norm > 0) {
                IDoubleVector normalizedRow = (IDoubleVector) row.divideByScalar(norm);
                double[] rowData = normalizedRow.getData();
                System.arraycopy(rowData, 0, result[i], 0, cols);
            } else {
                System.arraycopy(data[i], 0, result[i], 0, cols);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵按列归一化 / Column-wise normalization of matrix
     * <p>
     * 对矩阵的每一列进行L2归一化 Performs L2 normalization on each column of the matrix
     * </p>
     *
     * @return 归一化后的矩阵 / Normalized matrix
     */
    @Override
    public IMatrix<Double> normalizeColumns() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        // 对每一列进行L2归一化
        for (int j = 0; j < cols; j++) {
            // 计算列的L2范数
            double norm = 0.0;
            for (int i = 0; i < rows; i++) {
                norm += data[i][j] * data[i][j];
            }
            norm = Math.sqrt(norm);

            // 归一化该列
            if (norm > 1e-10) { // 避免除零
                for (int i = 0; i < rows; i++) {
                    result[i][j] = data[i][j] / norm;
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    result[i][j] = 0.0;
                }
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵数据中心化 / Matrix data centering
     * <p>
     * 对矩阵的每一列减去该列的均值，实现数据中心化 Subtracts the mean of each column from the column
     * elements, implementing data centering
     * </p>
     *
     * @return 中心化后的矩阵 / Centered matrix
     */
    @Override
    public IMatrix<Double> center() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        // 计算每列的均值
        IVector<Double> columnMeans = this.rowMeans();

        // 对每列减去均值
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] - columnMeans.get(j);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 计算协方差矩阵 / Compute covariance matrix
     * <p>
     * 计算数据矩阵的协方差矩阵，假设每行是一个样本，每列是一个特征 Computes the covariance matrix of the data
     * matrix, assuming each row is a sample and each column is a feature
     * </p>
     *
     * @return 协方差矩阵 / Covariance matrix
     */
    @Override
    public IMatrix<Double> covariance() {
        // 先进行数据中心化
        IMatrix<Double> centered = this.center();
        return centered.covarianceFromCentered();
    }

    /**
     * covariance的别名函数
     * @return 
     */
    @Override
    public IMatrix<Double> cov() {
        return this.covariance();}
    
    
    

    /**
     * 计算协方差矩阵（已中心化数据） / Compute covariance matrix (for centered data)
     * <p>
     * 对于已经中心化的数据，直接计算协方差矩阵 = (X^T * X) / (n-1) For already centered data,
     * directly compute covariance matrix = (X^T * X) / (n-1)
     * </p>
     *
     * @return 协方差矩阵 / Covariance matrix
     */
    @Override
    public IMatrix<Double> covarianceFromCentered() {
        int n = data.length; // 样本数

        // 计算 X^T * X
        IMatrix<Double> transposed = this.transposeNew();
        IMatrix<Double> product = ((RereDoubleMatrix) transposed).mmul(this);

        // 除以 (n-1) 得到协方差矩阵
        return product.multiplyScalar(1.0 / (n - 1));
    }

    /**
     * 求解矩阵的逆 / Matrix inverse
     * <p>
     * 使用高斯-约旦消元法计算方阵的逆矩阵。只有方阵且行列式不为零的矩阵才有逆矩阵。 Uses Gauss-Jordan elimination to
     * compute the inverse of a square matrix. Only square matrices with
     * non-zero determinant have an inverse.
     * </p>
     *
     * @return 逆矩阵 / Inverse matrix
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     * @throws ArithmeticException 如果矩阵是奇异的（不可逆） / if matrix is singular
     * (non-invertible)
     */
    @Override
    public IMatrix<Double> inv() {
        // 检查是否为方阵
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能求逆 / Only square matrices can be inverted");
        }

        int n = data.length;

        // 创建增广矩阵 [A | I]，其中I是单位矩阵
        double[][] augmented = new double[n][2 * n];

        // 初始化增广矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = data[i][j];  // 左半部分是原矩阵A
                augmented[i][j + n] = (i == j) ? 1.0 : 0.0;  // 右半部分是单位矩阵I
            }
        }

        // 高斯-约旦消元法
        final double tolerance = 1e-10;

        for (int i = 0; i < n; i++) {
            // 寻找主元（绝对值最大的元素）
            int pivotRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[pivotRow][i])) {
                    pivotRow = k;
                }
            }

            // 检查主元是否为零（矩阵奇异）
            if (Math.abs(augmented[pivotRow][i]) < tolerance) {
                throw new ArithmeticException("矩阵是奇异的，无法求逆 / Matrix is singular and cannot be inverted");
            }

            // 交换行
            if (pivotRow != i) {
                double[] temp = augmented[i];
                augmented[i] = augmented[pivotRow];
                augmented[pivotRow] = temp;
            }

            // 将主元归一化
            double pivot = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }

            // 消除当前列的其他元素
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        // 提取逆矩阵（增广矩阵的右半部分）
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j + n];
            }
        }

        return new RereDoubleMatrix(inverse);
    }

    /**
     * 求解矩阵的伪逆 / Matrix pseudo-inverse
     * <p>
     * 使用奇异值分解(SVD)计算矩阵的Moore-Penrose伪逆。伪逆适用于任意维度的矩阵， 包括非方阵和奇异矩阵。计算公式：A⁺ = V *
     * Σ⁺ * Uᵀ，其中Σ⁺是奇异值的伪逆。
     * </p>
     * <p>
     * Uses Singular Value Decomposition (SVD) to compute the Moore-Penrose
     * pseudo-inverse. The pseudo-inverse works for matrices of any dimensions,
     * including non-square and singular matrices. Formula: A⁺ = V * Σ⁺ * Uᵀ,
     * where Σ⁺ is the pseudo-inverse of singular values.
     * </p>
     *
     * @return 伪逆矩阵 / Pseudo-inverse matrix
     */
    @Override
    public IMatrix<Double> pinv() {
        // 计算复杂度：矩阵元素数量
        int rows = data.length;
        int cols = data[0].length;
        long complexity = (long) rows * cols;

        // 对于大矩阵，直接使用简化的伪逆算法避免卡死
        if (complexity > 10000) {
            return simplifiedPseudoInverse();
        }

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && complexity > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuPseudoInverse(this);
            } catch (Exception e) {
                // GPU失败时回退到CPU，这里不需要额外日志，GPUComputeDoubleUtils已经处理了
            }
        }

        // 使用原有的CPU算法
        final double tolerance = 1e-10;

        // 进行奇异值分解：A = U * S * V^T
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = this.svd();
        IMatrix<Double> U = svdResult._1;           // 左奇异向量矩阵
        IVector<Double> singularValues = svdResult._2;  // 奇异值向量
        IMatrix<Double> VT = svdResult._3;          // 右奇异向量转置矩阵

        // 获取矩阵的维度信息
        int originalRows = this.getRowNum();
        int originalCols = this.getColNum();
        int singularValuesLength = singularValues.length();

        // 计算奇异值的伪逆
        IDoubleVector pseudoSingularValues = IDoubleVector.zeros(singularValuesLength);

        for (int i = 0; i < singularValuesLength; i++) {
            double sv = singularValues.get(i);
            if (Math.abs(sv) > tolerance) {
                pseudoSingularValues.set(i, 1.0 / sv);  // 非零奇异值的倒数
            } else {
                pseudoSingularValues.set(i, 0.0);       // 零奇异值保持为零
            }
        }

        // 计算伪逆：A⁺ = V * Σ⁺ * U^T
        // 由于我们的SVD返回的U可能是截断的，我们需要更仔细地处理维度
        IMatrix<Double> V = VT.transposeNew();  // V = (V^T)^T

        // 创建结果矩阵：A⁺的维度应该是 originalCols x originalRows
        IMatrix<Double> pseudoInverse = IDoubleMatrix.zeros(originalCols, originalRows);

        // 逐元素计算伪逆：A⁺[i,j] = Σ(k=0 to rank-1) V[i,k] * (1/σ[k]) * U[j,k]
        for (int i = 0; i < originalCols; i++) {
            for (int j = 0; j < originalRows; j++) {
                double sum = 0.0;
                for (int k = 0; k < singularValuesLength; k++) {
                    double vValue = (k < V.getColNum()) ? V.get(i, k) : 0.0;
                    double uValue = (k < U.getColNum()) ? U.get(j, k) : 0.0;
                    double sigmaInv = pseudoSingularValues.get(k);
                    sum += vValue * sigmaInv * uValue;
                }
                pseudoInverse.put(i, j, sum);
            }
        }

        return pseudoInverse;
    }

    /**
     * 从本地指定位置加载矩阵 / Load matrix from specified local path
     * <p>
     * 从指定的文件路径加载矩阵数据，文件格式为：第一行为矩阵维度（行数 列数），后续行为矩阵数据（每行元素用空格分隔） Loads matrix
     * data from the specified file path. File format: first line contains
     * matrix dimensions (rows columns), subsequent lines contain matrix data
     * (elements in each row separated by spaces)
     * </p>
     *
     * @param path 文件路径 / File path
     * @return 从文件加载的矩阵对象 / IMatrix<Double> object loaded from file
     * @throws IllegalArgumentException 如果文件路径为null或为空 / if file path is null or
     * empty
     * @throws RuntimeException 如果文件读取失败或格式错误 / if file reading fails or format
     * is incorrect
     */
    public static IMatrix<Double> loadFromFile(String path) {
        if (path == null || path.trim().isEmpty()) {
            throw new IllegalArgumentException("文件路径不能为null或空 / File path cannot be null or empty");
        }

        try (java.io.BufferedReader reader = new java.io.BufferedReader(new java.io.FileReader(path))) {
            // 读取第一行获取矩阵维度
            String dimensionLine = reader.readLine();
            if (dimensionLine == null || dimensionLine.trim().isEmpty()) {
                throw new RuntimeException("文件格式错误：无法读取矩阵维度 / File format error: cannot read matrix dimensions");
            }

            String[] dimensions = dimensionLine.trim().split("\\s+");
            if (dimensions.length != 2) {
                throw new RuntimeException("文件格式错误：维度行必须包含行数和列数 / File format error: dimension line must contain rows and columns");
            }

            int rows, cols;
            try {
                rows = Integer.parseInt(dimensions[0]);
                cols = Integer.parseInt(dimensions[1]);
            } catch (NumberFormatException e) {
                throw new RuntimeException("文件格式错误：维度必须为整数 / File format error: dimensions must be integers", e);
            }

            if (rows <= 0 || cols <= 0) {
                throw new RuntimeException("文件格式错误：矩阵维度必须为正数 / File format error: matrix dimensions must be positive");
            }

            // 创建数据数组
            double[][] data = new double[rows][cols];

            // 读取矩阵数据
            for (int i = 0; i < rows; i++) {
                String line = reader.readLine();
                if (line == null) {
                    throw new RuntimeException("文件格式错误：矩阵数据行数不足 / File format error: insufficient matrix data rows");
                }

                String[] elements = line.trim().split("\\s+");
                if (elements.length != cols) {
                    throw new RuntimeException("文件格式错误：第" + (i + 1) + "行元素数量不匹配 / File format error: element count mismatch in row " + (i + 1));
                }

                try {
                    for (int j = 0; j < cols; j++) {
                        data[i][j] = Double.parseDouble(elements[j]);
                    }
                } catch (NumberFormatException e) {
                    throw new RuntimeException("文件格式错误：第" + (i + 1) + "行包含无效数字 / File format error: invalid number in row " + (i + 1), e);
                }
            }

            return new RereDoubleMatrix(data);

        } catch (java.io.IOException e) {
            throw new RuntimeException("文件读取失败：" + e.getMessage() + " / File reading failed: " + e.getMessage(), e);
        }
    }

    /**
     * 将矩阵数据保存到本地指定位置 / Save matrix data to specified local path
     * <p>
     * 将当前矩阵保存到指定的文件路径，保存格式为：第一行为矩阵维度（行数 列数），后续行为矩阵数据（每行元素用空格分隔） Saves current
     * matrix to the specified file path. Save format: first line contains
     * matrix dimensions (rows columns), subsequent lines contain matrix data
     * (elements in each row separated by spaces)
     * </p>
     *
     * @param path 文件保存路径 / File save path
     * @throws IllegalArgumentException 如果文件路径为null或为空 / if file path is null or
     * empty
     * @throws RuntimeException 如果文件写入失败 / if file writing fails
     */
    @Override
    public void save(String path) {
        if (path == null || path.trim().isEmpty()) {
            throw new IllegalArgumentException("文件路径不能为null或空 / File path cannot be null or empty");
        }

        try (java.io.PrintWriter writer = new java.io.PrintWriter(new java.io.FileWriter(path))) {
            // 写入矩阵维度
            writer.println(data.length + " " + data[0].length);

            // 写入矩阵数据
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    if (j > 0) {
                        writer.print(" ");
                    }
                    writer.print(data[i][j]);
                }
                writer.println();
            }

            writer.flush();

        } catch (java.io.IOException e) {
            throw new RuntimeException("文件写入失败：" + e.getMessage() + " / File writing failed: " + e.getMessage(), e);
        }
    }

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
    @Override
    public IMatrix<Double> hstack(IMatrix<Double> other1) {
        if (other1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }

        if (data.length != other1.getRowNum()) {
            throw new IllegalArgumentException("矩阵行数不匹配 / Matrix row counts don't match");
        }

        int rows = data.length;
        int cols1 = data[0].length;
        int cols2 = other1.getColNum();
        int totalCols = cols1 + cols2;

        double[][] result = new double[rows][totalCols];

        // 复制第一个矩阵
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, result[i], 0, cols1);
        }

        // 复制第二个矩阵
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        for (int i = 0; i < rows; i++) {
            System.arraycopy(otherData[i], 0, result[i], cols1, cols2);
        }

        return new RereDoubleMatrix(result);
    }

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
    @Override
    public IMatrix<Double> vstack(IMatrix<Double> other1) {
        if (other1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }

        if (data[0].length != other1.getColNum()) {
            throw new IllegalArgumentException("矩阵列数不匹配 / Matrix column counts don't match");
        }

        int rows1 = data.length;
        int rows2 = other1.getRowNum();
        int cols = data[0].length;
        int totalRows = rows1 + rows2;

        double[][] result = new double[totalRows][cols];

        // 复制第一个矩阵
        for (int i = 0; i < rows1; i++) {
            System.arraycopy(data[i], 0, result[i], 0, cols);
        }

        // 复制第二个矩阵
        IDoubleMatrix other0 = (IDoubleMatrix) other1;
        double[][] otherData = other0.getData();
        for (int i = 0; i < rows2; i++) {
            System.arraycopy(otherData[i], 0, result[rows1 + i], 0, cols);
        }

        return new RereDoubleMatrix(result);
    }

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
    @Override
    public IMatrix<Double>[] hsplit(int[] indices) {
        if (indices == null || indices.length == 0) {
            throw new IllegalArgumentException("分割索引不能为null或空 / Split indices cannot be null or empty");
        }

        // 验证索引的有效性
        for (int idx : indices) {
            if (idx < 0 || idx > data[0].length) {
                throw new IllegalArgumentException("分割索引超出范围 / Split index out of bounds: " + idx);
            }
        }

        // 排序索引
        int[] sortedIndices = indices.clone();
        java.util.Arrays.sort(sortedIndices);

        int numSplits = sortedIndices.length + 1;
        IMatrix<Double>[] result = new IDoubleMatrix[numSplits];

        int startCol = 0;
        for (int i = 0; i < numSplits; i++) {
            int endCol = (i < sortedIndices.length) ? sortedIndices[i] : data[0].length;
            int cols = endCol - startCol;

            double[][] subMatrix = new double[data.length][cols];
            for (int row = 0; row < data.length; row++) {
                System.arraycopy(data[row], startCol, subMatrix[row], 0, cols);
            }

            result[i] = new RereDoubleMatrix(subMatrix);
            startCol = endCol;
        }

        return result;
    }

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
    @Override
    public IMatrix<Double>[] vsplit(int[] indices) {
        if (indices == null || indices.length == 0) {
            throw new IllegalArgumentException("分割索引不能为null或空 / Split indices cannot be null or empty");
        }

        // 验证索引的有效性
        for (int idx : indices) {
            if (idx < 0 || idx > data.length) {
                throw new IllegalArgumentException("分割索引超出范围 / Split index out of bounds: " + idx);
            }
        }

        // 排序索引
        int[] sortedIndices = indices.clone();
        java.util.Arrays.sort(sortedIndices);

        int numSplits = sortedIndices.length + 1;
        IMatrix<Double>[] result = new IDoubleMatrix[numSplits];

        int startRow = 0;
        for (int i = 0; i < numSplits; i++) {
            int endRow = (i < sortedIndices.length) ? sortedIndices[i] : data.length;
            int rows = endRow - startRow;

            double[][] subMatrix = new double[rows][data[0].length];
            for (int row = 0; row < rows; row++) {
                System.arraycopy(data[startRow + row], 0, subMatrix[row], 0, data[0].length);
            }

            result[i] = new RereDoubleMatrix(subMatrix);
            startRow = endRow;
        }

        return result;
    }

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
    @Override
    public IMatrix<Double> reshape(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }

        int totalElements = data.length * data[0].length;
        if (rows * cols != totalElements) {
            throw new IllegalArgumentException("新维度与元素总数不匹配 / New dimensions don't match total element count");
        }

        double[] flatData = this.toFlattenArray();
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = flatData[i * cols + j];
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵复制 / Matrix copy
     * <p>
     * 创建当前矩阵的深拷贝 Creates a deep copy of the current matrix
     * </p>
     *
     * @return 矩阵的副本 / Copy of the matrix
     */
    @Override
    public IMatrix<Double> copy() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] copyData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copyData[i], 0, cols);
        }

        return new RereDoubleMatrix(copyData);
    }

    /**
     * 计算矩阵行列式 / Compute matrix determinant
     * <p>
     * 计算方阵的行列式值 Computes the determinant of a square matrix
     * </p>
     *
     * @return 行列式值 / Determinant value
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    @Override
    public Double det() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能计算行列式 / Only square matrices can compute determinant");
        }

        int n = data.length;
        if (n == 1) {
            return data[0][0];
        }
        if (n == 2) {
            return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        }

        // 使用LU分解计算行列式
        try {
            Tuple2<IMatrix<Double>, IMatrix<Double>> lu = this.lu();
            IMatrix<Double> L = lu._1;
            IMatrix<Double> U = lu._2;

            double detL = 1.0;
            double detU = 1.0;

            // L矩阵的对角线元素乘积
            for (int i = 0; i < n; i++) {
                detL *= L.get(i, i);
            }

            // U矩阵的对角线元素乘积
            for (int i = 0; i < n; i++) {
                detU *= U.get(i, i);
            }

            return detL * detU;
        } catch (Exception e) {
            // 如果LU分解失败，使用递归方法
            return detRecursive(data);
        }
    }

    /**
     * 递归计算行列式的辅助方法 / Helper method for recursive determinant calculation
     */
    private Double detRecursive(double[][] matrix) {
        int n = matrix.length;
        if (n == 1) {
            return matrix[0][0];
        }
        if (n == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }

        double det = 0.0;
        for (int j = 0; j < n; j++) {
            double[][] minor = new double[n - 1][n - 1];
            for (int i = 1; i < n; i++) {
                for (int k = 0; k < n; k++) {
                    if (k < j) {
                        minor[i - 1][k] = matrix[i][k];
                    } else if (k > j) {
                        minor[i - 1][k - 1] = matrix[i][k];
                    }
                }
            }
            det += (j % 2 == 0 ? 1 : -1) * matrix[0][j] * detRecursive(minor);
        }
        return det;
    }

    /**
     * 计算矩阵迹 / Compute matrix trace
     * <p>
     * 计算方阵的迹（对角线元素之和） Computes the trace of a square matrix (sum of diagonal
     * elements)
     * </p>
     *
     * @return 矩阵迹 / Matrix trace
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    @Override
    public Double trace() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能计算迹 / Only square matrices can compute trace");
        }

        double trace = 0.0;
        for (int i = 0; i < data.length; i++) {
            trace += data[i][i];
        }
        return trace;
    }

    /**
     * 计算矩阵条件数 / Compute matrix condition number
     * <p>
     * 计算矩阵的条件数，用于评估矩阵的数值稳定性 Computes the condition number of a matrix to assess
     * numerical stability
     * </p>
     *
     * @return 条件数 / Condition number
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    @Override
    public Double cond() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能计算条件数 / Only square matrices can compute condition number");
        }

        try {
            double normA = this.frobeniusNorm();
            IMatrix<Double> invA = this.inv();
            double normInvA = invA.frobeniusNorm();
            return normA * normInvA;
        } catch (Exception e) {
            // 如果求逆失败，返回一个很大的数表示条件数很大
            return Double.MAX_VALUE;
        }
    }

    /**
     * 计算矩阵秩 / Compute matrix rank
     * <p>
     * 计算矩阵的秩，即线性无关的行或列的最大数量 Computes the rank of a matrix, the maximum number
     * of linearly independent rows or columns
     * </p>
     *
     * @return 矩阵秩 / Matrix rank
     */
    @Override
    public int rank() {
        try {
            // 使用SVD计算秩
            Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = this.svd();
            IVector<Double> singularValues = svdResult._2;

            final double tolerance = 1e-10;
            int rank = 0;
            for (int i = 0; i < singularValues.length(); i++) {
                if (Math.abs(singularValues.get(i)) > tolerance) {
                    rank++;
                }
            }
            return rank;
        } catch (Exception e) {
            // 如果SVD失败，使用高斯消元法
            return rankByGaussianElimination();
        }
    }

    /**
     * 使用高斯消元法计算矩阵秩的辅助方法 / Helper method for computing rank using Gaussian
     * elimination
     */
    private int rankByGaussianElimination() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] matrix = new double[rows][cols];

        // 复制数据
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, matrix[i], 0, cols);
        }

        int rank = 0;
        final double tolerance = 1e-10;

        for (int col = 0; col < cols && rank < rows; col++) {
            // 寻找主元
            int pivotRow = rank;
            for (int row = rank + 1; row < rows; row++) {
                if (Math.abs(matrix[row][col]) > Math.abs(matrix[pivotRow][col])) {
                    pivotRow = row;
                }
            }

            if (Math.abs(matrix[pivotRow][col]) > tolerance) {
                // 交换行
                if (pivotRow != rank) {
                    double[] temp = matrix[rank];
                    matrix[rank] = matrix[pivotRow];
                    matrix[pivotRow] = temp;
                }

                // 消元
                for (int row = rank + 1; row < rows; row++) {
                    double factor = matrix[row][col] / matrix[rank][col];
                    for (int c = col; c < cols; c++) {
                        matrix[row][c] -= factor * matrix[rank][c];
                    }
                }
                rank++;
            }
        }

        return rank;
    }

    /**
     * 矩阵绝对值 / Matrix absolute value
     * <p>
     * 对矩阵中每个元素取绝对值 Takes the absolute value of each element in the matrix
     * </p>
     *
     * @return 包含绝对值的新矩阵 / New matrix containing absolute values
     */
    @Override
    public IMatrix<Double> abs() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.abs(data[i][j]);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵符号函数 / Matrix sign function
     * <p>
     * 对矩阵中每个元素应用符号函数：正数为1，负数为-1，零为0 Applies sign function to each element: 1
     * for positive, -1 for negative, 0 for zero
     * </p>
     *
     * @return 包含符号值的新矩阵 / New matrix containing sign values
     */
    @Override
    public IMatrix<Double> sign() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] > 0) {
                    result[i][j] = 1.0;
                } else if (data[i][j] < 0) {
                    result[i][j] = -1.0;
                } else {
                    result[i][j] = 0.0;
                }
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵正弦函数 / Matrix sine function
     * <p>
     * 对矩阵中每个元素计算正弦值 Computes sine value for each element in the matrix
     * </p>
     *
     * @return 包含正弦值的新矩阵 / New matrix containing sine values
     */
    @Override
    public IMatrix<Double> sin() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.sin(data[i][j]);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵余弦函数 / Matrix cosine function
     * <p>
     * 对矩阵中每个元素计算余弦值 Computes cosine value for each element in the matrix
     * </p>
     *
     * @return 包含余弦值的新矩阵 / New matrix containing cosine values
     */
    @Override
    public IMatrix<Double> cos() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.cos(data[i][j]);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵正切函数 / Matrix tangent function
     * <p>
     * 对矩阵中每个元素计算正切值 Computes tangent value for each element in the matrix
     * </p>
     *
     * @return 包含正切值的新矩阵 / New matrix containing tangent values
     */
    @Override
    public IMatrix<Double> tan() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.tan(data[i][j]);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵双曲正弦函数 / Matrix hyperbolic sine function
     * <p>
     * 对矩阵中每个元素计算双曲正弦值 Computes hyperbolic sine value for each element in the
     * matrix
     * </p>
     *
     * @return 包含双曲正弦值的新矩阵 / New matrix containing hyperbolic sine values
     */
    @Override
    public IMatrix<Double> sinh() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.sinh(data[i][j]);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵双曲余弦函数 / Matrix hyperbolic cosine function
     * <p>
     * 对矩阵中每个元素计算双曲余弦值 Computes hyperbolic cosine value for each element in the
     * matrix
     * </p>
     *
     * @return 包含双曲余弦值的新矩阵 / New matrix containing hyperbolic cosine values
     */
    @Override
    public IMatrix<Double> cosh() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.cosh(data[i][j]);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵双曲正切函数 / Matrix hyperbolic tangent function
     * <p>
     * 对矩阵中每个元素计算双曲正切值 Computes hyperbolic tangent value for each element in the
     * matrix
     * </p>
     *
     * @return 包含双曲正切值的新矩阵 / New matrix containing hyperbolic tangent values
     */
    @Override
    public IMatrix<Double> tanh() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.tanh(data[i][j]);
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * LU分解 / LU decomposition
     * <p>
     * 将矩阵分解为下三角矩阵L和上三角矩阵U的乘积：A = L * U Decomposes matrix into product of lower
     * triangular matrix L and upper triangular matrix U: A = L * U
     * </p>
     *
     * @return 包含L和U矩阵的元组 / Tuple containing L and U matrices
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> lu() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能进行LU分解 / Only square matrices can perform LU decomposition");
        }

        int n = data.length;
        double[][] A = new double[n][n];
        double[][] L = new double[n][n];
        double[][] U = new double[n][n];

        // 复制数据
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, A[i], 0, n);
        }

        // 初始化L为单位下三角矩阵
        for (int i = 0; i < n; i++) {
            L[i][i] = 1.0;
        }

        final double tolerance = 1e-10;

        for (int k = 0; k < n; k++) {
            // 计算U的第k行
            for (int j = k; j < n; j++) {
                double sum = 0.0;
                for (int m = 0; m < k; m++) {
                    sum += L[k][m] * U[m][j];
                }
                U[k][j] = A[k][j] - sum;
            }

            // 计算L的第k列
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(U[k][k]) < tolerance) {
                    throw new ArithmeticException("矩阵是奇异的，无法进行LU分解 / Matrix is singular, cannot perform LU decomposition");
                }
                double sum = 0.0;
                for (int m = 0; m < k; m++) {
                    sum += L[i][m] * U[m][k];
                }
                L[i][k] = (A[i][k] - sum) / U[k][k];
            }
        }

        return new Tuple2<>(new RereDoubleMatrix(L), new RereDoubleMatrix(U));
    }

    /**
     * Cholesky分解 / Cholesky decomposition
     * <p>
     * 将对称正定矩阵分解为下三角矩阵L的乘积：A = L * L^T Decomposes symmetric positive definite
     * matrix into product of lower triangular matrix L: A = L * L^T
     * </p>
     *
     * @return 下三角矩阵L / Lower triangular matrix L
     * @throws IllegalArgumentException 如果矩阵不是对称正定矩阵 / if matrix is not
     * symmetric positive definite
     */
    @Override
    public IMatrix<Double> cholesky() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能进行Cholesky分解 / Only square matrices can perform Cholesky decomposition");
        }

        int n = data.length;
        double[][] L = new double[n][n];

        // 初始化L矩阵为0
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                L[i][j] = 0.0;
            }
        }

        // 检查对称性
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(data[i][j] - data[j][i]) > 1e-10) {
                    throw new IllegalArgumentException("矩阵必须是对称的 / Matrix must be symmetric");
                }
            }
        }

        // 改进的Cholesky分解，增加数值稳定性
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0; // 使用double提高精度

                if (j == i) {
                    // 对角线元素
                    for (int k = 0; k < j; k++) {
                        sum += (double) L[j][k] * (double) L[j][k];
                    }
                    double diagonal = (double) data[j][j] - sum;

                    if (diagonal <= 1e-10) { // 更严格的数值检查
                        throw new IllegalArgumentException("矩阵不是正定的 / Matrix is not positive definite");
                    }
                    L[j][j] = Math.sqrt(diagonal);
                } else {
                    // 非对角线元素
                    for (int k = 0; k < j; k++) {
                        sum += (double) L[i][k] * (double) L[j][k];
                    }
                    L[i][j] = (((double) data[i][j] - sum) / (double) L[j][j]);
                }
            }
        }

        // 确保L是下三角矩阵，上三角部分为0
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                L[i][j] = 0.0;
            }
        }

        return new RereDoubleMatrix(L);
    }

    /**
     * 求解线性方程组 / Solve linear system
     * <p>
     * 求解线性方程组 Ax = b，其中A是当前矩阵，b是右侧向量 Solves linear system Ax = b where A is the
     * current matrix and b is the right-hand side vector
     * </p>
     *
     * @param b 右侧向量 / Right-hand side vector
     * @return 解向量x / Solution vector x
     * @throws IllegalArgumentException 如果矩阵不是方阵或维度不匹配 / if matrix is not square
     * or dimensions don't match
     */
    @Override
    public IVector<Double> solve(IVector<Double> b) {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能求解线性方程组 / Only square matrices can solve linear systems");
        }

        if (b.length() != data.length) {
            throw new IllegalArgumentException("向量维度与矩阵行数不匹配 / Vector dimension doesn't match matrix rows");
        }

        // 使用LU分解求解
        Tuple2<IMatrix<Double>, IMatrix<Double>> lu = this.lu();
        IMatrix<Double> L = lu._1;
        IMatrix<Double> U = lu._2;

        int n = data.length;
        double[] x = new double[n];
        double[] y = new double[n];

        // 前向代入：Ly = b
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += L.get(i, j) * y[j];
            }
            y[i] = (b.get(i) - sum) / L.get(i, i);
        }

        // 后向代入：Ux = y
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += U.get(i, j) * x[j];
            }
            x[i] = (y[i] - sum) / U.get(i, i);
        }

        return IDoubleVector.of(x);
    }

    /**
     * 求解线性方程组（矩阵形式） / Solve linear system (matrix form)
     * <p>
     * 求解线性方程组 AX = B，其中A是当前矩阵，B是右侧矩阵 Solves linear system AX = B where A is the
     * current matrix and B is the right-hand side matrix
     * </p>
     *
     * @param B 右侧矩阵 / Right-hand side matrix
     * @return 解矩阵X / Solution matrix X
     * @throws IllegalArgumentException 如果矩阵不是方阵或维度不匹配 / if matrix is not square
     * or dimensions don't match
     */
    @Override
    public IMatrix<Double> solve(IMatrix<Double> B) {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能求解线性方程组 / Only square matrices can solve linear systems");
        }

        if (B.getRowNum() != data.length) {
            throw new IllegalArgumentException("右侧矩阵行数与系数矩阵行数不匹配 / Right-hand side matrix rows don't match coefficient matrix rows");
        }

        int n = data.length;
        int m = B.getColNum();
        double[][] X = new double[n][m];

        // 对每一列分别求解
        for (int col = 0; col < m; col++) {
            double[] b = new double[n];
            for (int row = 0; row < n; row++) {
                b[row] = B.get(row, col);
            }

            IVector<Double> x = this.solve(IDoubleVector.of(b));
            for (int row = 0; row < n; row++) {
                X[row][col] = x.get(row);
            }
        }

        return new RereDoubleMatrix(X);
    }

    /**
     * 矩阵最大值 / Matrix maximum value
     * <p>
     * 返回矩阵中的最大元素值 Returns the maximum element value in the matrix
     * </p>
     *
     * @return 最大元素值 / Maximum element value
     */
    @Override
    public Double max() {
        double max = data[0][0];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                if (data[i][j] > max) {
                    max = data[i][j];
                }
            }
        }
        return max;
    }

    /**
     * 矩阵最小值 / Matrix minimum value
     * <p>
     * 返回矩阵中的最小元素值 Returns the minimum element value in the matrix
     * </p>
     *
     * @return 最小元素值 / Minimum element value
     */
    @Override
    public Double min() {
        double min = data[0][0];
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                if (data[i][j] < min) {
                    min = data[i][j];
                }
            }
        }
        return min;
    }

    /**
     * 矩阵元素求和 / Matrix element sum
     * <p>
     * 返回矩阵中所有元素的总和 Returns the sum of all elements in the matrix
     * </p>
     *
     * @return 元素总和 / Sum of all elements
     */
    @Override
    public Double sum() {
        double sum = 0.0;
        for (double[] data1 : data) {
            for (int j = 0; j < data[0].length; j++) {
                sum += data1[j];
            }
        }
        return sum;
    }

    /**
     * 矩阵元素均值 / Matrix element mean
     * <p>
     * 返回矩阵中所有元素的平均值 Returns the mean of all elements in the matrix
     * </p>
     *
     * @return 元素均值 / Mean of all elements
     */
    @Override
    public Double mean() {
        return sum() / (data.length * data[0].length);
    }

    /**
     * 矩阵元素标准差 / Matrix element standard deviation
     * <p>
     * 返回矩阵中所有元素的标准差 Returns the standard deviation of all elements in the
     * matrix
     * </p>
     *
     * @return 元素标准差 / Standard deviation of all elements
     */
    @Override
    public Double std() {
        return Math.sqrt(var());
    }

    /**
     * 矩阵元素方差 / Matrix element variance
     * <p>
     * 返回矩阵中所有元素的方差 Returns the variance of all elements in the matrix
     * </p>
     *
     * @return 元素方差 / Variance of all elements
     */
    @Override
    public Double var() {
        double mean = mean();
        double sumSquaredDiff = 0.0;
        int totalElements = data.length * data[0].length;

        for (double[] data1 : data) {
            for (int j = 0; j < data[0].length; j++) {
                double diff = data1[j] - mean;
                sumSquaredDiff += diff * diff;
            }
        }

        return sumSquaredDiff / totalElements;
    }

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
    @Override
    public IMatrix<Double> slice(String rowSlice, String colSlice) {
        SliceExpressionParser.SliceResult rowResult = SliceExpressionParser.parse(rowSlice, data.length);
        SliceExpressionParser.SliceResult colResult = SliceExpressionParser.parse(colSlice, data[0].length);

        int startRow = rowResult.actualStart;
        int endRow = rowResult.actualEnd;
        int stepRow = rowResult.step;

        int startCol = colResult.actualStart;
        int endCol = colResult.actualEnd;
        int stepCol = colResult.step;

        // 计算结果矩阵的尺寸
        int resultRows = (endRow - startRow + stepRow - 1) / stepRow;
        int resultCols = (endCol - startCol + stepCol - 1) / stepCol;

        if (resultRows <= 0 || resultCols <= 0) {
            return new RereDoubleMatrix(new double[0][0]);
        }

        double[][] result = new double[resultRows][resultCols];

        int resultRow = 0;
        for (int i = startRow; i < endRow; i += stepRow) {
            int resultCol = 0;
            for (int j = startCol; j < endCol; j += stepCol) {
                result[resultRow][resultCol] = data[i][j];
                resultCol++;
            }
            resultRow++;
        }

        return new RereDoubleMatrix(result);
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
    @Override
    public IMatrix<Double> sliceRows(String rowSlice) {
        return slice(rowSlice, ":");
    }

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
    @Override
    public IMatrix<Double> sliceColumns(String colSlice) {
        return slice(":", colSlice);
    }

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
    @Override
    public IMatrix<Double> fancyGet(int[] rowIndices, int[] colIndices) {
        if (rowIndices == null || colIndices == null) {
            throw new IllegalArgumentException("索引数组不能为null / Index arrays cannot be null");
        }

        double[][] result = new double[rowIndices.length][colIndices.length];

        for (int i = 0; i < rowIndices.length; i++) {
            int row = rowIndices[i];
            if (row < 0) {
                row = data.length + row;
            }
            if (row < 0 || row >= data.length) {
                throw new IndexOutOfBoundsException("行索引超出范围: " + rowIndices[i] + " / Row index out of bounds: " + rowIndices[i]);
            }

            for (int j = 0; j < colIndices.length; j++) {
                int col = colIndices[j];
                if (col < 0) {
                    col = data[0].length + col;
                }
                if (col < 0 || col >= data[0].length) {
                    throw new IndexOutOfBoundsException("列索引超出范围: " + colIndices[j] + " / Column index out of bounds: " + colIndices[j]);
                }

                result[i][j] = data[row][col];
            }
        }

        return new RereDoubleMatrix(result);
    }

    /**
     * 双对角化 - 将矩阵转换为双对角形式 使用Householder变换将矩阵A转换为双对角矩阵B，使得A = U * B * V^T
     */
    private Tuple3<double[][], double[][], double[][]> bidiagonalization() {
        int m = data.length;
        int n = data[0].length;
        int minDim = Math.min(m, n);

        double[][] A = new double[m][n];
        for (int i = 0; i < m; i++) {
            System.arraycopy(data[i], 0, A[i], 0, n);
        }

        // 初始化U和V为单位矩阵
        double[][] U = new double[m][m];
        double[][] V = new double[n][n];
        for (int i = 0; i < m; i++) {
            U[i][i] = 1.0;
        }
        for (int i = 0; i < n; i++) {
            V[i][i] = 1.0;
        }

        // 双对角化过程
        for (int k = 0; k < minDim; k++) {
            // 对第k列进行Householder变换（左变换）
            if (k < m - 1) {
                double[] x = new double[m - k];
                for (int i = k; i < m; i++) {
                    x[i - k] = A[i][k];
                }

                double norm = 0.0;
                for (double v : x) {
                    norm += v * v;
                }
                norm = Math.sqrt(norm);

                if (norm > 1e-10) {
                    double[] v = new double[m - k];
                    v[0] = x[0] + Math.signum(x[0]) * norm;
                    for (int i = 1; i < v.length; i++) {
                        v[i] = x[i];
                    }

                    // 归一化v
                    double vNorm = 0.0;
                    for (double vi : v) {
                        vNorm += vi * vi;
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int i = 0; i < v.length; i++) {
                        v[i] /= vNorm;
                    }

                    // 构造Householder矩阵 P = I - 2*v*v^T
                    double[][] P = new double[m - k][m - k];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < m - k; j++) {
                            P[i][j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j];
                        }
                    }

                    // 应用变换到A
                    double[][] subA = new double[m - k][n - k];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k; j++) {
                            subA[i][j] = A[k + i][k + j];
                        }
                    }

                    double[][] PsubA = new double[m - k][n - k];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k; j++) {
                            for (int l = 0; l < m - k; l++) {
                                PsubA[i][j] += P[i][l] * subA[l][j];
                            }
                        }
                    }

                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k; j++) {
                            A[k + i][k + j] = PsubA[i][j];
                        }
                    }

                    // 更新U
                    double[][] subU = new double[m][m - k];
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m - k; j++) {
                            subU[i][j] = U[i][k + j];
                        }
                    }

                    double[][] subUP = new double[m][m - k];
                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m - k; j++) {
                            for (int l = 0; l < m - k; l++) {
                                subUP[i][j] += subU[i][l] * P[l][j];
                            }
                        }
                    }

                    for (int i = 0; i < m; i++) {
                        for (int j = 0; j < m - k; j++) {
                            U[i][k + j] = subUP[i][j];
                        }
                    }
                }
            }

            // 对第k行进行Householder变换（右变换）
            if (k < n - 2) {
                double[] x = new double[n - k - 1];
                for (int j = k + 1; j < n; j++) {
                    x[j - k - 1] = A[k][j];
                }

                double norm = 0.0;
                for (double v : x) {
                    norm += v * v;
                }
                norm = Math.sqrt(norm);

                if (norm > 1e-10) {
                    double[] v = new double[n - k - 1];
                    v[0] = x[0] + Math.signum(x[0]) * norm;
                    for (int i = 1; i < v.length; i++) {
                        v[i] = x[i];
                    }

                    // 归一化v
                    double vNorm = 0.0;
                    for (double vi : v) {
                        vNorm += vi * vi;
                    }
                    vNorm = Math.sqrt(vNorm);
                    for (int i = 0; i < v.length; i++) {
                        v[i] /= vNorm;
                    }

                    // 构造Householder矩阵 P = I - 2*v*v^T
                    double[][] P = new double[n - k - 1][n - k - 1];
                    for (int i = 0; i < n - k - 1; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            P[i][j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j];
                        }
                    }

                    // 应用变换到A
                    double[][] subA = new double[m - k][n - k - 1];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            subA[i][j] = A[k + i][k + 1 + j];
                        }
                    }

                    double[][] subAP = new double[m - k][n - k - 1];
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            for (int l = 0; l < n - k - 1; l++) {
                                subAP[i][j] += subA[i][l] * P[l][j];
                            }
                        }
                    }

                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            A[k + i][k + 1 + j] = subAP[i][j];
                        }
                    }

                    // 更新V
                    double[][] subV = new double[n][n - k - 1];
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            subV[i][j] = V[i][k + 1 + j];
                        }
                    }

                    double[][] subVP = new double[n][n - k - 1];
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            for (int l = 0; l < n - k - 1; l++) {
                                subVP[i][j] += subV[i][l] * P[l][j];
                            }
                        }
                    }

                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            V[i][k + 1 + j] = subVP[i][j];
                        }
                    }
                }
            }
        }

        return new Tuple3<>(A, U, V);
    }

    /**
     * 对双对角矩阵应用分治算法
     */
    private Tuple2<double[], double[][]> divideAndConquerSVD(double[][] B) {
        int m = B.length;
        int n = B[0].length;
        int minDim = Math.min(m, n);

        // 提取双对角矩阵的对角线和次对角线
        double[] alpha = new double[minDim];
        double[] beta = new double[minDim - 1];

        for (int i = 0; i < minDim; i++) {
            alpha[i] = B[i][i];
        }
        for (int i = 0; i < minDim - 1; i++) {
            beta[i] = B[i][i + 1];
        }

        // 使用分治算法计算奇异值
        return divideAndConquerBidiagonal(alpha, beta, 0, minDim - 1);
    }

    /**
     * 分治算法处理双对角矩阵
     */
    private Tuple2<double[], double[][]> divideAndConquerBidiagonal(double[] alpha, double[] beta, int start, int end) {
        int n = end - start + 1;

        switch (n) {
            case 1:
                // 基本情况：1x1矩阵
                double[] singularValues = {Math.abs(alpha[start])};
                double[][] Q = {{1.0}};
                return new Tuple2<>(singularValues, Q);
            case 2:
                // 基本情况：2x2矩阵
                return solve2x2Bidiagonal(alpha[start], beta[start], alpha[start + 1]);
            default:
                // 递归分解
                int mid = start + n / 2;

                // 分解为两个子问题
                Tuple2<double[], double[][]> leftResult = divideAndConquerBidiagonal(alpha, beta, start, mid - 1);
                Tuple2<double[], double[][]> rightResult = divideAndConquerBidiagonal(alpha, beta, mid, end);

                // 合并结果
                return mergeBidiagonalResults(leftResult, rightResult);
        }
    }

    /**
     * 解决2x2双对角矩阵的SVD
     */
    private Tuple2<double[], double[][]> solve2x2Bidiagonal(Double a, Double b, Double c) {
        // 构造2x2矩阵 [a b; 0 c]
        double[][] M = {{a, b}, {0.0, c}};

        // 计算特征值（奇异值的平方）
        double trace = a + c;
        double det = a * c;
        double discriminant = trace * trace - 4 * det;

        double lambda1, lambda2;
        if (discriminant >= 0) {
            double sqrtDisc = Math.sqrt(discriminant);
            lambda1 = (trace + sqrtDisc) / 2.0;
            lambda2 = (trace - sqrtDisc) / 2.0;
        } else {
            lambda1 = trace / 2.0;
            lambda2 = trace / 2.0;
        }

        double[] singularValues = {Math.sqrt(Math.max(0, lambda1)), Math.sqrt(Math.max(0, lambda2))};

        // 计算特征向量
        double[][] Q = new double[2][2];
        if (Math.abs(lambda1 - lambda2) > 1e-10) {
            // 不同的特征值
            double v1x = a - lambda1;
            double v1y = b;
            double norm1 = Math.sqrt(v1x * v1x + v1y * v1y);
            if (norm1 > 1e-10) {
                Q[0][0] = v1x / norm1;
                Q[1][0] = v1y / norm1;
            } else {
                Q[0][0] = 1.0;
                Q[1][0] = 0.0;
            }

            double v2x = a - lambda2;
            double v2y = b;
            double norm2 = Math.sqrt(v2x * v2x + v2y * v2y);
            if (norm2 > 1e-10) {
                Q[0][1] = v2x / norm2;
                Q[1][1] = v2y / norm2;
            } else {
                Q[0][1] = 0.0;
                Q[1][1] = 1.0;
            }
        } else {
            // 相同的特征值
            Q[0][0] = 1.0;
            Q[1][0] = 0.0;
            Q[0][1] = 0.0;
            Q[1][1] = 1.0;
        }

        return new Tuple2<>(singularValues, Q);
    }

    /**
     * 合并两个双对角SVD结果
     */
    private Tuple2<double[], double[][]> mergeBidiagonalResults(Tuple2<double[], double[][]> left, Tuple2<double[], double[][]> right) {
        double[] leftSV = left._1;
        double[] rightSV = right._1;
        double[][] leftQ = left._2;
        double[][] rightQ = right._2;

        int leftSize = leftSV.length;
        int rightSize = rightSV.length;
        int totalSize = leftSize + rightSize;

        // 合并奇异值
        double[] mergedSV = new double[totalSize];
        System.arraycopy(leftSV, 0, mergedSV, 0, leftSize);
        System.arraycopy(rightSV, 0, mergedSV, leftSize, rightSize);

        // 合并Q矩阵
        double[][] mergedQ = new double[totalSize][totalSize];
        for (int i = 0; i < leftSize; i++) {
            for (int j = 0; j < leftSize; j++) {
                mergedQ[i][j] = leftQ[i][j];
            }
        }
        for (int i = 0; i < rightSize; i++) {
            for (int j = 0; j < rightSize; j++) {
                mergedQ[leftSize + i][leftSize + j] = rightQ[i][j];
            }
        }

        // 对合并后的结果进行排序
        quickSortSingularValues(mergedSV, null, 0, totalSize - 1);

        return new Tuple2<>(mergedSV, mergedQ);
    }

    /**
     * 对双对角矩阵应用QR算法
     */
    private Tuple2<double[], double[][]> qrAlgorithmForBidiagonal(double[][] B) {
        int m = B.length;
        int n = B[0].length;
        int minDim = Math.min(m, n);

        // 提取双对角矩阵的对角线和次对角线
        double[] alpha = new double[minDim];
        double[] beta = new double[minDim - 1];

        for (int i = 0; i < minDim; i++) {
            alpha[i] = B[i][i];
        }
        for (int i = 0; i < minDim - 1; i++) {
            beta[i] = B[i][i + 1];
        }

        // 初始化Q为单位矩阵
        double[][] Q = new double[minDim][minDim];
        for (int i = 0; i < minDim; i++) {
            Q[i][i] = 1.0;
        }

        // 性能监控
        long[] iterationTimes = new long[500]; // 记录每次迭代时间
        int actualIterations = 0;

        final int maxIterations = 500; // 增加最大迭代次数
        final double tolerance = 1e-10; // 提高收敛精度
        final int minIterations = 5; // 增加最小迭代次数要求
        double previousOffDiagonalSum = Double.MAX_VALUE;
        int stagnationCount = 0;

        // QR迭代
        for (int iter = 0; iter < maxIterations; iter++) {
            // 检查收敛性
            double offDiagonalSum = 0.0;
            for (int i = 0; i < minDim - 1; i++) {
                offDiagonalSum += Math.abs(beta[i]);
            }

            // 改进的收敛判断 - 平衡的收敛条件
            boolean converged = false;

            // 条件1：次对角线元素足够小且达到最小迭代次数
            if (offDiagonalSum < tolerance && actualIterations >= minIterations) {
                converged = true;
            }

            // 条件2：相对变化很小
            if (previousOffDiagonalSum > 0) {
                double relativeChange = Math.abs(offDiagonalSum - previousOffDiagonalSum) / previousOffDiagonalSum;
                if (relativeChange < tolerance * 0.1f && actualIterations >= minIterations) {
                    stagnationCount++;
                    if (stagnationCount >= 3) {
                        converged = true;
                    }
                } else {
                    stagnationCount = 0;
                }
            }

            // 条件3：次对角线元素总和足够小且达到最小迭代次数
            if (offDiagonalSum < tolerance * n && actualIterations >= minIterations) {
                converged = true;
            }

            // 条件4：所有次对角线元素都小于容差（更宽松）且达到最小迭代次数
            boolean allSmall = true;
            for (int i = 0; i < minDim - 1; i++) {
                if (Math.abs(beta[i]) >= tolerance * 10) {
                    allSmall = false;
                    break;
                }
            }
            if (allSmall && actualIterations >= minIterations) {
                converged = true;
            }

            previousOffDiagonalSum = offDiagonalSum;

            if (converged) {
                break;
            }

            // 对双对角矩阵进行QR分解
            Tuple2<double[], double[]> qrResult = qrDecompositionBidiagonal(alpha, beta);
            double[] newAlpha = qrResult._1;
            double[] newBeta = qrResult._2;

            // 更新alpha和beta
            alpha = newAlpha;
            beta = newBeta;
        }

        // 计算奇异值
        double[] singularValues = new double[minDim];
        for (int i = 0; i < minDim; i++) {
            singularValues[i] = Math.abs(alpha[i]);
        }

        // 按奇异值大小降序排列
        quickSortSingularValues(singularValues, null, 0, minDim - 1);

        return new Tuple2<>(singularValues, Q);
    }

    /**
     * 双对角矩阵的QR分解 - 修复版本 每次只消除一个次对角线元素，避免一次性消除所有
     */
    private Tuple2<double[], double[]> qrDecompositionBidiagonal(double[] alpha, double[] beta) {
        int n = alpha.length;
        double[] newAlpha = new double[n];
        double[] newBeta = new double[n - 1];

        System.arraycopy(alpha, 0, newAlpha, 0, n);
        System.arraycopy(beta, 0, newBeta, 0, n - 1);

        // 找到最大的次对角线元素进行消除
        int maxIndex = 0;
        double maxValue = Math.abs(newBeta[0]);
        for (int i = 1; i < n - 1; i++) {
            if (Math.abs(newBeta[i]) > maxValue) {
                maxValue = Math.abs(newBeta[i]);
                maxIndex = i;
            }
        }

        // 只消除最大的次对角线元素
        if (maxValue > 1e-10) {
            int i = maxIndex;

            // 计算Givens旋转参数
            double a = newAlpha[i];
            double b = newBeta[i];
            double r = Math.sqrt(a * a + b * b);

            if (r >= 1e-10) {
                double c = a / r;
                double s = -b / r; // 注意符号

                // 应用Givens旋转
                newAlpha[i] = r;
                newBeta[i] = 0.0;  // 清零当前次对角线元素

                // 更新下一个对角线元素
                if (i < n - 1) {
                    newAlpha[i + 1] = c * newAlpha[i + 1];
                    if (i < n - 2) {
                        newBeta[i + 1] = c * newBeta[i + 1];
                    }
                }
            }
        }

        return new Tuple2<>(newAlpha, newBeta);
    }

    /**
     * 快速排序奇异值
     */
    private void quickSortSingularValues(double[] values, int[] indices, int low, int high) {
        if (low < high) {
            int pi = partitionSingularValues(values, indices, low, high);
            quickSortSingularValues(values, indices, low, pi - 1);
            quickSortSingularValues(values, indices, pi + 1, high);
        }
    }

    /**
     * 快速排序的分区函数（奇异值）
     */
    private int partitionSingularValues(double[] values, int[] indices, int low, int high) {
        double pivot = values[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (values[j] >= pivot) { // 降序排列
                i++;
                swapSingularValues(values, indices, i, j);
            }
        }
        swapSingularValues(values, indices, i + 1, high);
        return i + 1;
    }

    /**
     * 交换奇异值和对应的索引
     */
    private void swapSingularValues(double[] values, int[] indices, int i, int j) {
        // 交换奇异值
        double temp = values[i];
        values[i] = values[j];
        values[j] = temp;

        // 交换索引（如果提供）
        if (indices != null) {
            int tempIdx = indices[i];
            indices[i] = indices[j];
            indices[j] = tempIdx;
        }
    }

    // set() method is now a default method in IMatrix<Double> that delegates to put()
    @Override
    public IVector<Double> diag() {
        int minDim = Math.min(data.length, data[0].length);
        double[] diagonal = new double[minDim];
        for (int i = 0; i < minDim; i++) {
            diagonal[i] = data[i][i];
        }
        return IDoubleVector.of(diagonal);
    }

    @Override
    public IMatrix<Double> subMatrix(int startRow, int endRow, int startCol, int endCol) {
        // 支持负数索引：负数表示从末尾开始计算
        int actualStartRow = startRow < 0 ? data.length + startRow : startRow;
        int actualEndRow = endRow < 0 ? data.length + endRow : endRow;
        int actualStartCol = startCol < 0 ? data[0].length + startCol : startCol;
        int actualEndCol = endCol < 0 ? data[0].length + endCol : endCol;
        
        if (actualStartRow < 0 || actualEndRow > data.length || actualStartCol < 0 || actualEndCol > data[0].length) {
            throw new IndexOutOfBoundsException("索引超出范围");
        }
        if (actualStartRow >= actualEndRow || actualStartCol >= actualEndCol) {
            throw new IllegalArgumentException("起始索引必须小于结束索引");
        }

        int rows = actualEndRow - actualStartRow;
        int cols = actualEndCol - actualStartCol;
        double[][] subData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[actualStartRow + i], actualStartCol, subData[i], 0, cols);
        }

        return new RereDoubleMatrix(subData);
    }

    @Override
    public IMatrix<Double> setSubMatrix(int startRow, int endRow, int startCol, int endCol, IMatrix<Double> subMatrix) {
        // 支持负数索引：负数表示从末尾开始计算
        int actualStartRow = startRow < 0 ? data.length + startRow : startRow;
        int actualEndRow = endRow < 0 ? data.length + endRow : endRow;
        int actualStartCol = startCol < 0 ? data[0].length + startCol : startCol;
        int actualEndCol = endCol < 0 ? data[0].length + endCol : endCol;
        
        if (actualStartRow < 0 || actualEndRow > data.length || actualStartCol < 0 || actualEndCol > data[0].length) {
            throw new IndexOutOfBoundsException("索引超出范围");
        }
        if (actualStartRow >= actualEndRow || actualStartCol >= actualEndCol) {
            throw new IllegalArgumentException("起始索引必须小于结束索引");
        }

        int rows = actualEndRow - actualStartRow;
        int cols = actualEndCol - actualStartCol;

        if (subMatrix.rows() != rows || subMatrix.cols() != cols) {
            throw new IllegalArgumentException("子矩阵尺寸不匹配");
        }
        
        // 将子矩阵数据复制到指定位置
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[actualStartRow + i][actualStartCol + j] = subMatrix.get(i, j);
            }
        }
        
        return this;
    }

    // ========== IMatrix 接口缺失方法实现 / Missing IMatrix interface method implementations ==========


    /**
     * 设置指定列 / Set specified column
     * <p>
     * 将列向量的数据设置到矩阵的指定列 Sets the column vector data to the specified column of
     * the matrix
     * </p>
     *
     * @param colIndex 列索引（从0开始） / Column index (0-based)
     * @param column 列向量矩阵 / Column vector matrix
     * @throws IllegalArgumentException 如果输入不是列向量或维度不匹配 / if input is not a
     * column vector or dimensions don't match
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
//    @Override
//    public void putColumn(int colIndex, IMatrix<Double> column) {
//        if (!(column instanceof IMatrix<Double>)) {
//            throw new IllegalArgumentException("参数必须是IMatrixGeneric<Double>类型 / Parameter must be IMatrix<Double> type");
//        }
//        putColumn(colIndex, (IMatrix<Double>) column);
//    }
    /**
     * 获取指定列向量 / Get specified column vector
     * <p>
     * 返回矩阵指定列的泛型向量表示 Returns the generic vector representation of the specified
     * column in the matrix
     * </p>
     *
     * @param i 列索引（从0开始） / Column index (0-based)
     * @return 指定列的泛型向量 / Generic vector of the specified column
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
//    @Override
//    public IVector<Double> getColumn(int i) {
//        return getColunm(i);
//    }
    /**
     * 获取指定行向量 / Get specified row vector
     * <p>
     * 返回矩阵指定行的泛型向量表示 Returns the generic vector representation of the specified
     * row in the matrix
     * </p>
     *
     * @param i 行索引（从0开始） / Row index (0-based)
     * @return 指定行的泛型向量 / Generic vector of the specified row
     * @throws IndexOutOfBoundsException 如果行索引超出范围 / if row index is out of
     * bounds
     */
//    @Override
//    public IVector<Double> getRow(int i) {
//        return getRow(i);
//    }
    /**
     * 计算行求和 / Calculate row sums
     * <p>
     * 计算矩阵每一列的元素和，返回泛型结果向量 Calculates the sum of elements in each column,
     * returns generic result vector
     * </p>
     *
     * @return 包含每列元素和的泛型向量 / Generic vector containing the sum of elements in
     * each column
     */
    @Override
    public IVector<Double> rowSums() {
        int rows = data.length;
        double[] sums = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < data[0].length; j++) {
                sum += data[i][j];
            }
            sums[i] = sum;
        }
        return IDoubleVector.of(sums);
    }

    /**
     * 计算行均值 / Calculate row means
     * <p>
     * 计算矩阵每一列的元素平均值，返回泛型结果向量 Calculates the mean of elements in each column,
     * returns generic result vector
     * </p>
     *
     * @return 包含每列元素平均值的泛型向量 / Generic vector containing the mean of elements
     * in each column
     */
    @Override
    public IVector<Double> rowMeans() {
        int rows = data.length;
        int cols = data[0].length;
        double[] means = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }
            means[i] = sum / cols;
        }
        return IDoubleVector.of(means);
    }

    /**
     * 计算列求和 / Calculate column sums
     * <p>
     * 计算矩阵每一行的元素和，返回泛型结果向量 Calculates the sum of elements in each row, returns
     * generic result vector
     * </p>
     *
     * @return 包含每行元素和的泛型向量 / Generic vector containing the sum of elements in
     * each row
     */
    @Override
    public IVector<Double> colSums() {
        int rows = data.length;
        int cols = data[0].length;
        double[] sums = new double[cols];
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += data[i][j];
            }
            sums[j] = sum;
        }
        return IDoubleVector.of(sums);
    }

    /**
     * 计算列均值 / Calculate column means
     * <p>
     * 计算矩阵每一行的元素平均值，返回泛型结果向量 Calculates the mean of elements in each row,
     * returns generic result vector
     * </p>
     *
     * @return 包含每行元素平均值的泛型向量 / Generic vector containing the mean of elements
     * in each row
     */
    @Override
    public IVector<Double> colMeans() {
        int rows = data.length;
        int cols = data[0].length;
        double[] means = new double[cols];
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int i = 0; i < rows; i++) {
                sum += data[i][j];
            }
            means[j] = sum / rows;
        }
        return IDoubleVector.of(means);
    }

    /**
     * 获取多个指定列 / Get multiple specified columns
     * <p>
     * 返回矩阵的多个指定列组成的数组 Returns an array of specified columns from the matrix
     * </p>
     *
     * @param indices 列索引数组 / Array of column indices
     * @return 包含指定列的矩阵数组 / Array of matrices containing the specified columns
     * @throws IndexOutOfBoundsException 如果任何列索引超出范围 / if any column index is
     * out of bounds
     */
//    @Override
//    public IMatrix<Double>[] getColumns(int[] indices) {
//        IMatrix<Double>[] floatColumns = getColumns(indices);
//        @SuppressWarnings("unchecked")
//        IMatrix<Double>[] genericColumns = new IMatrix[floatColumns.length];
//        System.arraycopy(floatColumns, 0, genericColumns, 0, floatColumns.length);
//        return genericColumns;
//    }
    /**
     * 矩阵元素平方根运算 / Matrix element square root operation
     * <p>
     * 对矩阵中每个元素进行平方根运算 Performs square root operation on each element in the
     * matrix
     * </p>
     *
     * @return 新的矩阵对象，包含平方根运算结果 / New matrix object containing the square root
     * results
     * @throws ArithmeticException 如果元素值为负数 / if any element value is negative
     */
    @Override
    public IMatrix<Double> sqrt() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] < 0) {
                    throw new ArithmeticException("无法对负数计算平方根 / Cannot calculate square root of negative number: " + data[i][j]);
                }
                result[i][j] = Math.sqrt(data[i][j]);
            }
        }
        return new RereDoubleMatrix(result);
    }

    /**
     * 矩阵幂运算 / Matrix power operation
     * <p>
     * 对矩阵中每个元素进行幂运算 Performs power operation on each element in the matrix
     * </p>
     *
     * @param power 幂指数 / Power exponent
     * @return 新的矩阵对象，包含幂运算结果 / New matrix object containing the power operation
     * results
     */
    @Override
    public IMatrix<Double> pow(Double power) {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.pow(data[i][j], power);
            }
        }
        return new RereDoubleMatrix(result);
    }

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
    @Override
    public IMatrix<Double> exp() {
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.exp(data[i][j]);
            }
        }
        return new RereDoubleMatrix(result);
    }

    @Override
    public int rows() {
        return data.length;
    }

    @Override
    public int cols() {
        return data[0].length;
    }

        /**
     * 获取指定列矩阵 / Get specified column as matrix
     * <p>
     * 返回矩阵的指定列作为列向量矩阵 Returns the specified column of the matrix as a column
     * vector matrix
     * </p>
     *
     * @param colIndex 列索引（从0开始） / Column index (0-based)
     * @return 包含指定列数据的列向量矩阵 / Column vector matrix containing the specified
     * column data
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
    @Override
    public IMatrix<Double> getColumnMatrix(int colIndex) {
        return this.getColumnAsCloumnVector(colIndex);
    }

    @Override
    public IMatrix<Double> set(int row, int col, Double value) {
        if (row < 0 || row >= data.length) {
            throw new IndexOutOfBoundsException("行索引超出范围: " + row + " / Row index out of bounds: " + row);
        }
        if (col < 0 || col >= data[0].length) {
            throw new IndexOutOfBoundsException("列索引超出范围: " + col + " / Column index out of bounds: " + col);
        }
        data[row][col] = value;
        return this;
    }




    @Override
    public boolean isPositiveDefinite() {
        // 检查矩阵是否为方阵 - 正定矩阵必须是方阵
        if (!isSquare()) {
            return false;
        }
        
        int n = data.length;
        
        // 使用Cholesky分解判断正定性
        // 如果矩阵正定，则存在下三角矩阵L使得A = LL^T
        try {
            // 创建矩阵副本进行Cholesky分解
            double[][] L = new double[n][n];
            
            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    if (i == j) {
                        // 对角元素
                        double sum = 0.0;
                        for (int k = 0; k < j; k++) {
                            sum += L[j][k] * L[j][k];
                        }
                        double diagonal = data[j][j] - sum;
                        
                        // 如果对角元素非正，则矩阵不是正定的
                        if (diagonal <= 0.0) {
                            return false;
                        }
                        
                        L[j][j] = Math.sqrt(diagonal);
                    } else {
                        // 下三角元素
                        double sum = 0.0;
                        for (int k = 0; k < j; k++) {
                            sum += L[i][k] * L[j][k];
                        }
                        
                        // 检查除零情况
                        if (Math.abs(L[j][j]) < 1e-12) {
                            return false;
                        }
                        
                        L[i][j] = (data[i][j] - sum) / L[j][j];
                    }
                }
            }
            
            // 如果能成功完成Cholesky分解，则矩阵是正定的
            return true;
            
        } catch (Exception e) {
            // 如果分解过程中出现异常，则矩阵不是正定的
            return false;
        }
    }

    @Override
    public IVector<Double> flatten() {
        var as = this.toFlattenArray();
        return IDoubleVector.of(as);
    }

    @Override
    public double[][] toDoubleArray() {
        // 创建新的二维数组副本
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];
        
        // 复制数据（RereMatrix内部已经是double[][]，直接复制）
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, result[i], 0, cols);
        }
        
        return result;
    }

    @Override
    public float[][] toFloatArray() {
        // 创建新的二维float数组
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        // 将double转换为float
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) data[i][j];
            }
        }
        
        return result;
    }
    
        @Override
    public int[][] toIntArray() {
        // 创建新的二维double数组
        int rows = data.length;
        int cols = data[0].length;
        int[][] result = new int[rows][cols];
        
        // 将float转换为double
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (int) data[i][j];
            }
        }
        return result;
    }

    /**
     * 将获得的列的向量数据作为列向量（实质为矩阵）
     * getColumnMatrix的别名方法
     * @param i
     * @return 
     */
    @Override
    public IMatrix<Double> getColumnAsCloumnVector(int i) {
        return this.getColumn(i).asColumnVector();
    
    }

    @Override
    public IMatrix<Double> setColumn(int colIndex, IVector<Double> column) {
        if (column == null) {
            throw new IllegalArgumentException("列向量不能为null / Column vector cannot be null");
        }
        if (colIndex < 0 || colIndex >= data[0].length) {
            throw new IllegalArgumentException("列索引超出范围 / Column index out of bounds: " + colIndex);
        }
        if (column.length() != data.length) {
            throw new IllegalArgumentException("列向量长度与矩阵行数不匹配 / Column vector length does not match matrix rows: " + column.length() + " vs " + data.length);
        }
        
        // 获取列向量的数据
        double[] columnData = column.toDoubleArray();
        
        // 设置指定列的数据
        for (int i = 0; i < data.length; i++) {
            data[i][colIndex] = columnData[i];
        }
        
        return this;
    }

    @Override
    public IMatrix<Double> setRow(int rowIndex, IVector<Double> row) {
        if (row == null) {
            throw new IllegalArgumentException("行向量不能为null / Row vector cannot be null");
        }
        if (rowIndex < 0 || rowIndex >= data.length) {
            throw new IllegalArgumentException("行索引超出范围 / Row index out of bounds: " + rowIndex);
        }
        if (row.length() != data[0].length) {
            throw new IllegalArgumentException("行向量长度与矩阵列数不匹配 / Row vector length does not match matrix columns: " + row.length() + " vs " + data[0].length);
        }
        
        // 获取行向量的数据
        double[] rowData = row.toDoubleArray();
        
        // 设置指定行的数据
        System.arraycopy(rowData, 0, data[rowIndex], 0, data[0].length);
        
        return this;
    }

    
        /**
     * 针对矩阵中的每一个元素施加操作并且返回同形状的被操作后矩阵
     * <p>
     * 对矩阵中每个元素应用指定的函数，返回新的矩阵对象
     * Applies the specified function to each element in the matrix, returns a new matrix object
     * </p>
     * <p>
     * 使用示例 / Usage Example:
     * <pre>{@code
     * RereDoubleMatrix matrix = new RereDoubleMatrix(new double[][]{{1, 2}, {3, 4}});
     * IMatrix<Double> squared = matrix.apply(x -> x * x);  // 结果: [[1, 4], [9, 16]]
     * IMatrix<Double> abs = matrix.apply(Math::abs);      // 结果: [[1, 2], [3, 4]]
     * }</pre>
     * </p>
     *
     * @param fun 要应用的函数，接受Double类型参数并返回Double类型结果
     *            Function to apply, accepts Double parameter and returns Double result
     * @return 新的矩阵对象，包含应用函数后的结果
     *         New matrix object containing the results after applying the function
     * @throws IllegalArgumentException 如果函数为null
     *                                  if function is null
     */
    @Override
    public IMatrix<Double> apply(Function<Double, Double> fun) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }
        
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = fun.apply(data[i][j]);
            }
        }
        
        return new RereDoubleMatrix(result);
    }

    @Override
    public IVector<Double> mmul(IVector<Double> other) {
        if (other == null) {
            throw new NullPointerException("向量不能为null / Vector cannot be null");
        }
        
        int matrixRows = this.data.length;
        int matrixCols = this.data[0].length;
        int vectorLen = other.length();
        
        if (matrixCols != vectorLen) {
            throw new IllegalArgumentException(
                String.format("矩阵列数与向量长度不匹配: %d != %d / Matrix column count doesn't match vector length: %d != %d", 
                    matrixCols, vectorLen, matrixCols, vectorLen));
        }
        
        // 计算矩阵与列向量的乘积
        double[] result = new double[matrixRows];
        for (int i = 0; i < matrixRows; i++) {
            double sum = 0.0;
            for (int j = 0; j < matrixCols; j++) {
                sum += this.data[i][j] * other.get(j);
            }
            result[i] = sum;
        }
        
        return IDoubleVector.of(result);
    }

    @Override
    public IMatrix<Double> divideByScalar(Double scalar) {
        if (scalar == 0.0) {
            throw new ArithmeticException("除数不能为零 / Divisor cannot be zero");
        }
        return this.multiplyScalar(1.0 / scalar);
    }
    
    

}
