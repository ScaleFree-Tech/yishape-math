package com.yishape.lab.math.linalg;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.compute.IDoubleVectorComputer;
import com.yishape.lab.math.linalg.solver.ConditionNumberSolver;
import com.yishape.lab.math.linalg.solver.DeterminantSolver;
import com.yishape.lab.math.linalg.solver.LinearSystemSolver;
import com.yishape.lab.math.linalg.solver.MatrixInversionSolver;
import com.yishape.lab.math.linalg.solver.RankSolver;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.util.RerePrecision;
import java.io.Serializable;

import java.util.concurrent.*;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.DoubleBinaryOperator;
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
 * <li>矩阵变换：转置、幂运算、开方 / IMatrix<Double> transformations: transpose, power,
 * square root</li>
 * <li>数据访问：行列访问、元素获取设置 / Data access: row/column access, element get/set</li>
 * <li>统计运算：行列求和、均值计算 / Statistical operations: row/column sum, mean
 * calculation</li>
 * <li>数据转换：数组转换、类型转换 / Data conversion: array conversion, type conversion</li>
 * </ul>
 *
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 * // 创建矩阵 / Create matrix
 * double[][] data = {{1, 2}, {3, 4}};
 * IMatrix<Double> matrix = new RereDoubleMatrix(data);
 *
 * // 矩阵运算 / IMatrix<Double> operations
 * IMatrix<Double> result = matrix.add(other).mmul(2.0.0);
 *
 * // 获取行列 / Get rows/columns
 * IVector<Double> row = matrix.getRow(0);
 * IVector<Double> col = matrix.getColunm(0);
 * }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereDoubleMatrix implements IDoubleMatrix ,Serializable{

    private static final Logger log = LoggerFactory.getLogger(RereDoubleMatrix.class);


    /**
     * 矩阵数据存储数组 / Matrix data storage array
     * <p>
     * 使用二维Double数组存储矩阵数据，data[i][j]表示第i行第j列的元素 Uses 2D Double array to store
     * matrix data, data[i][j] represents element at row i, column j
     * </p>
     */
    double[][] data;
    
    /**
     * 对于空矩阵（行数为0），存储列数信息 / For empty matrices (0 rows), store column count information
     * 当data.length == 0时，这个字段存储预期的列数
     */
    int emptyMatrixCols = 0;

    private static final IDoubleVectorComputer computer = new DoubleVectorComputer();

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
     * 是否启用并行计算 / Whether parallel computation is enabled
     */
    private static volatile boolean PARALLEL_ENABLED = true;



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
            // 允许空矩阵
            this.data = data;
            return;
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
        var res = this.computer.binaryOperate(data, scalar, IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        return IDoubleMatrix.of(res);
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
    public IMatrix<Double> sub(IMatrix<Double> other) {
        var res = this.computer.binaryOperate(data, other.toDoubleArray(), IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.binaryOperate(data, scalar, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        return IDoubleMatrix.of(res);
    }

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
    @Override
    public IMatrix<Double> add(IMatrix<Double> other) {
        var res = this.computer.binaryOperate(data, other.toDoubleArray(), IDoubleVectorComputer.BinaryOperation.ADD);
        return IDoubleMatrix.of(res);
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
    public IMatrix<Double> divide(IMatrix<Double> other) {
        var res = this.computer.binaryOperate(data, other.toDoubleArray(), IDoubleVectorComputer.BinaryOperation.DIVIDE);
        return IDoubleMatrix.of(res);
    }

    /**
     * Frobenius 内积 / Frobenius inner product
     * <p>
     * 同形矩阵对应元素相乘后求和 / Sum of element-wise products for same-shaped matrices
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return Frobenius 内积 / Frobenius inner product
     * @throws IllegalArgumentException 如果维度不匹配 / if dimensions don't match
     */
    @Override
    public Double frobeniusInnerProduct(IMatrix<Double> other) {
        var res = this.computer.binaryReduceOperate(data, other.toDoubleArray(), IDoubleVectorComputer.BinaryReduceOperation.DOT);
        return res;
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
     * @param column1 列向量矩阵 / Column vector matrix
     * @throws IllegalArgumentException 如果输入不是列向量或维度不匹配 / if input is not a
     * column vector or dimensions don't match
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
    @Override
    public void putColumn(int colIndex, IMatrix<Double> column1) {
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
    public void put(int row, int col, Double value) {
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
        if (data.length == 0) {
            log.debug("DEBUG: getColNum() called on empty matrix, returning: " + emptyMatrixCols);
            return emptyMatrixCols; // 对于空矩阵，返回存储的列数
        }
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
        var res = this.computer.transpose(data);
        return IDoubleMatrix.of(res);
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
     * @return 返回特征值和特征向量的元组，其中： - 特征值按大小降序排列（从大到小） - 特征向量矩阵的列与特征值一一对应 -
     * 第i个特征向量对应第i个特征值
     * @throws IllegalArgumentException 当矩阵不是方阵时抛出异常
     */
    @Override
    public Tuple2<IVector<Double>, IMatrix<Double>> eigen() {
        return Decomps.createEigen().decompose(this);
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
        return Decomps.createSVD().decompose(this);
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
                if (!RerePrecision.equals(data[i][j], data[j][i], tolerance)) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * QR分解
     *
     * @return Q和R矩阵
     */
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> qr() {
        // 使用新的QR分解实现类
        return Decomps.createQR().decompose(this);
    }

    /**
     * 矩阵乘法运算 / Matrix multiplication
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 矩阵乘法结果 / Matrix multiplication result
     */
    @Override
    public IMatrix<Double> mmul(IMatrix<Double> other) {
        var res = this.computer.mmul(data, other.toDoubleArray());
        return IDoubleMatrix.of(res);
    }

    @Override
    public IMatrix<Double> kron(IMatrix<Double> other) {
        if (other == null) {
            throw new NullPointerException("other不能为null / other cannot be null");
        }
        int m = getRowNum();
        int n = getColNum();
        int p = other.getRowNum();
        int q = other.getColNum();
        double[][] a = this.toDoubleArray();
        double[][] b = other.toDoubleArray();
        double[][] res = new double[m * p][n * q];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double aij = a[i][j];
                for (int k = 0; k < p; k++) {
                    for (int l = 0; l < q; l++) {
                        res[i * p + k][j * q + l] = aij * b[k][l];
                    }
                }
            }
        }
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.LOG, 0);
        return IDoubleMatrix.of(res);
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
        var square = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.POW, 2.0);
        double sum = this.computer.reduceOperate(square, IDoubleVectorComputer.ReduceOperation.SUM);
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

        // 使用SIMD对每一行进行L2归一化
        for (int i = 0; i < rows; i++) {
            // 计算行的L2范数
            // 先计算每个元素的平方
            double[] squared = this.computer.universalOperate(data[i], IDoubleVectorComputer.UniversalOperation.POW, 2.0);
            
            // 计算平方和
            double sumOfSquares = this.computer.reduceOperate(squared, IDoubleVectorComputer.ReduceOperation.SUM);
            
            // 计算L2范数
            double norm = Math.sqrt(sumOfSquares);

            if (norm > 0) {
                // 使用SIMD归一化该行
                result[i] = this.computer.binaryOperate(data[i], norm, IDoubleVectorComputer.BinaryOperation.DIVIDE);
            } else {
                System.arraycopy(data[i], 0, result[i], 0, cols);
            }
        }

        return IDoubleMatrix.of(result);
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

        // 使用SIMD对每一列进行L2归一化
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            double[] column = new double[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            
            // 计算列的L2范数
            // 先计算每个元素的平方
            double[] squared = this.computer.universalOperate(column, IDoubleVectorComputer.UniversalOperation.POW, 2.0);
            
            // 计算平方和
            double sumOfSquares = this.computer.reduceOperate(squared, IDoubleVectorComputer.ReduceOperation.SUM);
            
            // 计算L2范数
            double norm = Math.sqrt(sumOfSquares);

            // 归一化该列
            if (norm > 1e-10) { // 避免除零
                double[] normalizedColumn = this.computer.binaryOperate(column, norm, IDoubleVectorComputer.BinaryOperation.DIVIDE);
                for (int i = 0; i < rows; i++) {
                    result[i][j] = normalizedColumn[i];
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    result[i][j] = 0.0;
                }
            }
        }

        return IDoubleMatrix.of(result);
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

        // 使用SIMD计算每列的均值
        double[] columnMeans = new double[cols];
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            double[] column = new double[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            
            // 计算列均值
            columnMeans[j] = this.computer.reduceOperate(column, IDoubleVectorComputer.ReduceOperation.MEAN);
        }

        // 使用SIMD对每列减去均值
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            double[] column = new double[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            
            // 使用SIMD减去均值
            double[] centeredColumn = this.computer.binaryOperate(column, columnMeans[j], IDoubleVectorComputer.BinaryOperation.SUBTRACT);
            
            // 将结果放回矩阵
            for (int i = 0; i < rows; i++) {
                result[i][j] = centeredColumn[i];
            }
        }

        return IDoubleMatrix.of(result);
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
     *
     * @return
     */
    @Override
    public IMatrix<Double> cov() {
        return this.covariance();
    }

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
        return MatrixInversionSolver.invert(this);
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
        return MatrixInversionSolver.pseudoInverse(this);
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

            return IDoubleMatrix.of(data);

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
     * @param other1 要连接的另一个矩阵 / The other matrix to concatenate
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

        return IDoubleMatrix.of(result);
    }

    /**
     * 矩阵连接（垂直方向） / Matrix concatenation (vertical)
     * <p>
     * 将两个矩阵在垂直方向上连接，要求列数相同 Concatenates two matrices vertically, requires same
     * number of columns
     * </p>
     *
     * @param other1 要连接的另一个矩阵 / The other matrix to concatenate
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

        return IDoubleMatrix.of(result);
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

            result[i] = IDoubleMatrix.of(subMatrix);
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

            result[i] = IDoubleMatrix.of(subMatrix);
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

        return IDoubleMatrix.of(result);
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
        int cols = rows > 0 ? data[0].length : emptyMatrixCols;
        double[][] copyData = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copyData[i], 0, cols);
        }

        RereDoubleMatrix out = new RereDoubleMatrix(copyData);
        if (rows == 0) {
            out.emptyMatrixCols = cols;
        }
        return out;
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
        return DeterminantSolver.compute(this);
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
        return ConditionNumberSolver.compute(this);
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
        return RankSolver.compute(this);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.ABS, 0);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.sign(data);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SIN, 0);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.COS, 0);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.TAN, 0);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SINH, 0);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.COSH, 0);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.TANH, 0);
        return IDoubleMatrix.of(res);
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
        // 使用新的LU分解实现类
        return Decomps.createLU().decompose(this);
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
        // 使用新的Cholesky分解实现类
        return Decomps.createCholesky().decompose(this);
    }

    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> schur() {
        return Decomps.createSchur().decompose(this);
    }

    @Override
    public Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> biDiag() {
        return Decomps.createBidiagonal().decompose(this);
    }

    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> triDiag() {
        return Decomps.createTridiagonal().decompose(this);
    }

    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> hessenberg() {
        return Decomps.createHessenberg().decompose(this);
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
        return LinearSystemSolver.solve(this, b);
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
        return LinearSystemSolver.solve(this, B);
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
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.MAX);
        return res;
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
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.MIN);
        return res;
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
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.SUM);
        return res;
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
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.MEAN);
        return res;
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
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.STANDARD_DEVIATION);
        return res;
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
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.VARIANCE);
        return res;
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
        int resultRows = calculateSliceSize(startRow, endRow, stepRow);
        int resultCols = calculateSliceSize(startCol, endCol, stepCol);

        if (resultRows <= 0 || resultCols <= 0) {
            // 返回空矩阵而不是抛出异常
            return IDoubleMatrix.of(new double[0][0]);
        }

        double[][] result = new double[resultRows][resultCols];

        int resultRow = 0;
        if (stepRow > 0) {
            for (int i = startRow; i < endRow; i += stepRow) {
                int resultCol = 0;
                if (stepCol > 0) {
                    for (int j = startCol; j < endCol; j += stepCol) {
                        result[resultRow][resultCol] = data[i][j];
                        resultCol++;
                    }
                } else {
                    for (int j = startCol; j > endCol; j += stepCol) {
                        result[resultRow][resultCol] = data[i][j];
                        resultCol++;
                    }
                }
                resultRow++;
            }
        } else {
            for (int i = startRow; i > endRow; i += stepRow) {
                int resultCol = 0;
                if (stepCol > 0) {
                    for (int j = startCol; j < endCol; j += stepCol) {
                        result[resultRow][resultCol] = data[i][j];
                        resultCol++;
                    }
                } else {
                    for (int j = startCol; j > endCol; j += stepCol) {
                        result[resultRow][resultCol] = data[i][j];
                        resultCol++;
                    }
                }
                resultRow++;
            }
        }

        return IDoubleMatrix.of(result);
    }

    /**
     * 计算切片结果的尺寸
     * Calculate the size of slice result
     */
    private int calculateSliceSize(int start, int end, int step) {
        if (step > 0) {
            return Math.max(0, (end - start + step - 1) / step);
        } else {
            // 对于负数步长，我们需要特殊处理
            // 当end为-1时，表示到开头
            int absStep = Math.abs(step);
            if (start < end) {
                return 0; // 如果start < end且step为负，没有元素
            }
            if (end == -1) {
                // 表示到开头，元素个数 = (start - 0) / abs(step) + 1
                return Math.max(0, start / absStep + 1);
            } else {
                // 一般情况：元素个数 = (start - end - 1) / abs(step) + 1
                return Math.max(0, (start - end - 1) / absStep + 1);
            }
        }
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

        return IDoubleMatrix.of(result);
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

        return IDoubleMatrix.of(subData);
    }

    @Override
    public void setSubMatrix(int startRow, int endRow, int startCol, int endCol, IMatrix<Double> subMatrix) {
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
    }

    // ========== IMatrix 接口缺失方法实现 / Missing IMatrix interface method implementations ==========
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
        
        // 使用SIMD计算每行的和
        for (int i = 0; i < rows; i++) {
            sums[i] = this.computer.reduceOperate(data[i], IDoubleVectorComputer.ReduceOperation.SUM);
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
        
        // 使用SIMD计算每行的均值
        for (int i = 0; i < rows; i++) {
            means[i] = this.computer.reduceOperate(data[i], IDoubleVectorComputer.ReduceOperation.MEAN);
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
        
        // 使用SIMD计算每列的和
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            double[] column = new double[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            // 计算列和
            sums[j] = this.computer.reduceOperate(column, IDoubleVectorComputer.ReduceOperation.SUM);
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
        
        // 使用SIMD计算每列的均值
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            double[] column = new double[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            
            // 计算列均值
            means[j] = this.computer.reduceOperate(column, IDoubleVectorComputer.ReduceOperation.MEAN);
        }
        
        return IDoubleVector.of(means);
    }

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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SQRT, 0);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.POW, power);
        return IDoubleMatrix.of(res);
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
        var res = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.EXP, 0);
        return IDoubleMatrix.of(res);
    }

    @Override
    public int rows() {
        return data.length;
    }

    @Override
    public int cols() {
        return this.getColNum();
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
                        if (RerePrecision.equalsZero(diagonal, 1e-12) || diagonal < 0.0) {
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
                        if (RerePrecision.equalsZero(L[j][j], 1e-12)) {
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
        return data;
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
     * 将获得的列的向量数据作为列向量（实质为矩阵） getColumnMatrix的别名方法
     *
     * @param i
     * @return
     */
    @Override
    public IMatrix<Double> getColumnAsCloumnVector(int i) {
        return this.getColumn(i).asColumnVector();

    }

    @Override
    public void setColumn(int colIndex, IVector<Double> column) {
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
    }

    @Override
    public void setRow(int rowIndex, IVector<Double> row) {
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
    }

    /**
     * 针对矩阵中的每一个元素施加操作并且返回同形状的被操作后矩阵
     * <p>
     * 对矩阵中每个元素应用指定的函数，返回新的矩阵对象 Applies the specified function to each element
     * in the matrix, returns a new matrix object
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
     * @param fun 要应用的函数，接受Double类型参数并返回Double类型结果 Function to apply, accepts
     * Double parameter and returns Double result
     * @return 新的矩阵对象，包含应用函数后的结果 New matrix object containing the results after
     * applying the function
     * @throws IllegalArgumentException 如果函数为null if function is null
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

        return IDoubleMatrix.of(result);
    }

    @Override
    public IVector<Double> mmul(IVector<Double> other) {
        var m2 = other.asColumnMatrix();
        var res = this.computer.mmul(data, m2.toDoubleArray());
        // Extract the column vector from the result matrix
        double[] resultVector = new double[res.length];
        for (int i = 0; i < res.length; i++) {
            resultVector[i] = res[i][0];
        }
        return IDoubleVector.of(resultVector);
    }

    @Override
    public IMatrix<Double> divideByScalar(Double scalar) {
        if (RerePrecision.equalsZero(scalar, 1e-12)) {
            throw new ArithmeticException("除数不能为零 / Divisor cannot be zero");
        }
        var res = this.computer.binaryOperate(data, scalar, IDoubleVectorComputer.BinaryOperation.DIVIDE);
        return IDoubleMatrix.of(res);
    }


    @Override
    public IMatrix<Double> normalize() {
        int rows = this.data.length;
        int cols = this.data[0].length;
        double[][] result = new double[rows][cols];

        // 使用SIMD计算每行的L2范数
        for (int i = 0; i < rows; i++) {
            // 计算每个元素的平方
            double[] squared = this.computer.universalOperate(this.data[i], IDoubleVectorComputer.UniversalOperation.POW, 2.0);
            
            // 计算平方和
            double sumOfSquares = this.computer.reduceOperate(squared, IDoubleVectorComputer.ReduceOperation.SUM);
            
            // 计算L2范数
            double norm = Math.sqrt(sumOfSquares);

            if (RerePrecision.equalsZero(norm, 1e-12)) {
                throw new ArithmeticException("第" + i + "行的L2范数为零，无法归一化 / Row " + i + " L2 norm is zero, cannot normalize");
            }

            // 使用SIMD归一化每行
            result[i] = this.computer.binaryOperate(this.data[i], norm, IDoubleVectorComputer.BinaryOperation.DIVIDE);
        }

        return IDoubleMatrix.of(result);
    }

    @Override
    public void setDiag(IVector<Double> diagonal) {
        if (diagonal == null) {
            throw new IllegalArgumentException("对角线向量不能为null / Diagonal vector cannot be null");
        }

        int rows = this.data.length;
        int cols = this.data[0].length;
        int diagLen = Math.min(rows, cols);

        if (diagonal.length() != diagLen) {
            throw new IllegalArgumentException("对角线向量长度必须等于矩阵的最小维度: " + diagonal.length() + " != " + diagLen + " / Diagonal vector length must equal matrix minimum dimension: " + diagonal.length() + " != " + diagLen);
        }

        // 设置对角线元素
        for (int i = 0; i < diagLen; i++) {
            data[i][i] = diagonal.get(i);
        }
    }

    @Override
    public IMatrix<Double> broadcastColumn(IVector<Double> colVector, BiFunction<IVector<Double>, IVector<Double>, IVector<Double>> fun) {
        if (colVector == null) {
            throw new IllegalArgumentException("列向量不能为null / Column vector cannot be null");
        }
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }

        int rows = this.data.length;
        int cols = this.data[0].length;

        if (colVector.length() != rows) {
            throw new IllegalArgumentException("列向量长度必须等于矩阵行数: " + colVector.length() + " != " + rows + " / Column vector length must equal matrix row count: " + colVector.length() + " != " + rows);
        }

        double[][] result = new double[rows][cols];

        for (int j = 0; j < cols; j++) {
            // 获取当前列
            double[] currentCol = new double[rows];
            for (int i = 0; i < rows; i++) {
                currentCol[i] = this.data[i][j];
            }
            IDoubleVector currentColVector = IDoubleVector.of(currentCol);

            // 应用函数
            IVector<Double> resultCol = fun.apply(currentColVector, colVector);

            // 将结果放回矩阵
            for (int i = 0; i < rows; i++) {
                result[i][j] = resultCol.get(i);
            }
        }

        return IDoubleMatrix.of(result);
    }

    @Override
    public IMatrix<Double> broadcastRow(IVector<Double> rowVector, BiFunction<IVector<Double>, IVector<Double>, IVector<Double>> fun) {
        if (rowVector == null) {
            throw new IllegalArgumentException("行向量不能为null / Row vector cannot be null");
        }
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }

        int rows = this.data.length;
        int cols = this.data[0].length;

        if (rowVector.length() != cols) {
            throw new IllegalArgumentException("行向量长度必须等于矩阵列数: " + rowVector.length() + " != " + cols + " / Row vector length must equal matrix column count: " + rowVector.length() + " != " + cols);
        }

        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            // 获取当前行
            IDoubleVector currentRowVector = IDoubleVector.of(this.data[i]);

            // 应用函数
            IVector<Double> resultRow = fun.apply(currentRowVector, rowVector);

            // 将结果放回矩阵
            for (int j = 0; j < cols; j++) {
                result[i][j] = resultRow.get(j);
            }
        }

        return IDoubleMatrix.of(result);
    }

    @Override
    public IMatrix<Double> multiply(IMatrix<Double> other) {
        var res = this.computer.binaryOperate(data, other.toDoubleArray(), IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        return IDoubleMatrix.of(res);
    }

    @Override
    public boolean[][] eq(IMatrix<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }

        int rows = data.length;
        int cols = data[0].length;

        if (other.cols() != cols||rows!=other.rows()) {
            throw new IllegalArgumentException("矩阵形状不一致");
        }

        return this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.EQUALS);
    }

    @Override
    public boolean[][] lt(IMatrix<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }

        int rows = data.length;
        int cols = data[0].length;

        if (other.cols() != cols||rows!=other.rows()) {
            throw new IllegalArgumentException("矩阵形状不一致");
        }

        return this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.LESS_THAN);
    }

    @Override
    public boolean[][] gt(IMatrix<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        int rows = data.length;
        int cols = data[0].length;

        if (other.cols() != cols||rows!=other.rows()) {
            throw new IllegalArgumentException("矩阵形状不一致");
        }
        return this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.GREATER_THAN);
    }
    
        @Override
    public boolean[][] ge(IMatrix<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        int rows = data.length;
        int cols = data[0].length;

        if (other.cols() != cols||rows!=other.rows()) {
            throw new IllegalArgumentException("矩阵形状不一致");
        }
        return this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.GREATER_THAN_OR_EQUALS);
    }
    
        @Override
    public boolean[][] le(IMatrix<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        int rows = data.length;
        int cols = data[0].length;

        if (other.cols() != cols||rows!=other.rows()) {
            throw new IllegalArgumentException("矩阵形状不一致");
        }
        return this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.LESS_THAN_OR_EQUALS);
    }

    /**
     * 前向替换求解下三角线性系统 / Forward substitution for lower triangular linear system
     * <p>
     * 求解下三角线性系统 LX = B，其中L是当前下三角矩阵，B是右端矩阵。 Solves the lower triangular linear
     * system LX = B, where L is the current lower triangular matrix and B is
     * the right-hand side matrix.
     * </p>
     *
     * @param B 右端矩阵 / Right-hand side matrix
     * @return 解矩阵 X / Solution matrix X
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     * @throws ArithmeticException 如果矩阵对角线元素为零 / if diagonal elements are zero
     */
    @Override
    public IMatrix<Double> forwardSolve(IMatrix<Double> B) {
        if (B == null) {
            throw new IllegalArgumentException("Right-hand side matrix cannot be null");
        }

        int n = this.data.length;
        int nrhs = B.cols();

        if (B.rows() != n) {
            throw new IllegalArgumentException("Matrix dimensions don't match: L is " + n + "x" + n
                    + ", B is " + B.rows() + "x" + nrhs);
        }

        // Copy B to X to avoid modifying the original matrix
        double[][] X = new double[n][nrhs];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                X[i][j] = B.get(i, j).doubleValue();
            }
        }

        // Forward substitution: solve LX = B
        for (int col = 0; col < nrhs; col++) {
            for (int i = 0; i < n; i++) {
                // Subtract L[i][j] * X[j][col] for j < i
                for (int j = 0; j < i; j++) {
                    X[i][col] -= this.data[i][j] * X[j][col];
                }

                // Check for zero diagonal element
                if (Math.abs(this.data[i][i]) < 1e-12) {
                    throw new ArithmeticException("Matrix is singular, diagonal element is zero at index " + i);
                }

                // Divide by diagonal element
                X[i][col] /= this.data[i][i];
            }
        }

        return IDoubleMatrix.of(X);
    }

    /**
     * 后向替换求解上三角线性系统 / Backward substitution for upper triangular linear system
     * <p>
     * 求解上三角线性系统 UX = B，其中U是当前上三角矩阵，B是右端矩阵。 Solves the upper triangular linear
     * system UX = B, where U is the current upper triangular matrix and B is
     * the right-hand side matrix.
     * </p>
     *
     * @param B 右端矩阵 / Right-hand side matrix
     * @return 解矩阵 X / Solution matrix X
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't
     * match
     * @throws ArithmeticException 如果矩阵对角线元素为零 / if diagonal elements are zero
     */
    @Override
    public IMatrix<Double> backwardSolve(IMatrix<Double> B) {
        if (B == null) {
            throw new IllegalArgumentException("Right-hand side matrix cannot be null");
        }

        int n = this.data.length;
        int nrhs = B.cols();

        if (B.rows() != n) {
            throw new IllegalArgumentException("Matrix dimensions don't match: U is " + n + "x" + n
                    + ", B is " + B.rows() + "x" + nrhs);
        }

        // Copy B to X to avoid modifying the original matrix
        double[][] X = new double[n][nrhs];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                X[i][j] = B.get(i, j).doubleValue();
            }
        }

        // Backward substitution: solve UX = B
        for (int col = 0; col < nrhs; col++) {
            for (int i = n - 1; i >= 0; i--) {
                // Subtract U[i][j] * X[j][col] for j > i
                for (int j = i + 1; j < n; j++) {
                    X[i][col] -= this.data[i][j] * X[j][col];
                }

                // Check for zero diagonal element
                if (Math.abs(this.data[i][i]) < 1e-12) {
                    throw new ArithmeticException("Matrix is singular, diagonal element is zero at index " + i);
                }

                // Divide by diagonal element
                X[i][col] /= this.data[i][i];
            }
        }

        return IDoubleMatrix.of(X);
    }

    @Override
    public String toString() {
        int rows = data.length;
        if (rows == 0) {
            return "[]";
        }
        int cols = data[0].length;

        // Format all elements to 2 decimal places
        String[][] formatted = new String[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = data[i][j];
                // 如果四舍五入到2位小数后为0.00，不显示负号
                double rounded = Math.round(value * 100.0) / 100.0;
                if (rounded == 0.0) {
                    formatted[i][j] = String.format("%.2f", 0.0);
                } else {
                    formatted[i][j] = String.format("%.2f", value);
                }
            }
        }

        // Calculate max width for each column
        int[] colWidths = new int[cols];
        for (int j = 0; j < cols; j++) {
            int maxWidth = 0;
            for (int i = 0; i < rows; i++) {
                maxWidth = Math.max(maxWidth, formatted[i][j].length());
            }
            colWidths[j] = maxWidth;
        }

        // Build the formatted string
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (j > 0) {
                    sb.append(" ");
                }
                String s = formatted[i][j];
                int padding = colWidths[j] - s.length();
                for (int p = 0; p < padding; p++) {
                    sb.append(" ");
                }
                sb.append(s);
            }
            if (i < rows - 1) {
                sb.append("\n");
            }
        }
        return sb.toString();
    }

    // ========== 二维广播 ==========

    /**
     * 两矩阵形状广播后的行、列数。
     */
    public static int[] broadcastShape(IMatrix<Double> a, IMatrix<Double> b) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        return broadcastShape(a.rows(), a.cols(), b.rows(), b.cols());
    }

    /**
     * 计算两矩阵形状广播后的行数、列数。
     */
    public static int[] broadcastShape(int rowsA, int colsA, int rowsB, int colsB) {
        int r = broadcastDim(rowsA, rowsB);
        int c = broadcastDim(colsA, colsB);
        return new int[]{r, c};
    }

    private static int broadcastDim(int x, int y) {
        if (x == y) {
            return x;
        }
        if (x == 1) {
            return y;
        }
        if (y == 1) {
            return x;
        }
        throw new IllegalArgumentException(
                "无法广播形状: (" + x + " vs " + y + ") / Cannot broadcast dimensions");
    }

    /**
     * 将二维数据广播到目标形状（仅允许长度为 1 的维被拉伸）。
     */
    public static double[][] broadcastTo(double[][] data, int targetRows, int targetCols) {
        Objects.requireNonNull(data, "data");
        if (data.length == 0 || data[0].length == 0) {
            throw new IllegalArgumentException("空矩阵无法广播 / Empty matrix cannot broadcast");
        }
        int r = data.length;
        int c = data[0].length;
        int br = broadcastDim(r, targetRows);
        int bc = broadcastDim(c, targetCols);
        if (br != targetRows || bc != targetCols) {
            throw new IllegalArgumentException(
                    "目标形状与广播规则不一致 / Target shape incompatible with broadcast");
        }
        if (r == targetRows && c == targetCols) {
            return copy2dForBroadcast(data);
        }
        double[][] out = new double[targetRows][targetCols];
        for (int i = 0; i < targetRows; i++) {
            int si = (r == 1) ? 0 : i;
            for (int j = 0; j < targetCols; j++) {
                int sj = (c == 1) ? 0 : j;
                out[i][j] = data[si][sj];
            }
        }
        return out;
    }

    /**
     * 将矩阵广播到目标形状。
     */
    public static IDoubleMatrix broadcastTo(IDoubleMatrix data, int targetRows, int targetCols) {
        Objects.requireNonNull(data, "data");
        return IDoubleMatrix.of(broadcastTo(data.toDoubleArray(), targetRows, targetCols));
    }

    /**
     * 对两矩阵先广播到公共形状，再逐元素应用 {@code op}。
     */
    public static double[][] broadcastElementWise(
            double[][] a,
            double[][] b,
            DoubleBinaryOperator op) {
        Objects.requireNonNull(op, "op");
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        if (a.length == 0 || b.length == 0) {
            throw new IllegalArgumentException("空矩阵 / Empty matrix");
        }
        int ra = a.length;
        int ca = a[0].length;
        int rb = b.length;
        int cb = b[0].length;
        int[] sh = broadcastShape(ra, ca, rb, cb);
        double[][] ba = broadcastTo(a, sh[0], sh[1]);
        double[][] bb = broadcastTo(b, sh[0], sh[1]);
        double[][] out = new double[sh[0]][sh[1]];
        for (int i = 0; i < sh[0]; i++) {
            for (int j = 0; j < sh[1]; j++) {
                out[i][j] = op.applyAsDouble(ba[i][j], bb[i][j]);
            }
        }
        return out;
    }

    /**
     * 对两双精度矩阵逐元素运算（广播），返回新矩阵。
     */
    public static IDoubleMatrix broadcastElementWise(
            IMatrix<Double> a,
            IMatrix<Double> b,
            DoubleBinaryOperator op) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        return IDoubleMatrix.of(broadcastElementWise(a.toDoubleArray(), b.toDoubleArray(), op));
    }

    private static double[][] copy2dForBroadcast(double[][] data) {
        double[][] c = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            c[i] = data[i].clone();
        }
        return c;
    }

}
