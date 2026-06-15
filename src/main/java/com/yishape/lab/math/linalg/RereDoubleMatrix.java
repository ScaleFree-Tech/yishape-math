package com.yishape.lab.math.linalg;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.hpc.HpcLapackDecomps;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.compute.IDoubleVectorComputer;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.BinaryReduceOperation;
import com.yishape.lab.math.compute.ops.LogicalCompare;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.solver.ConditionNumberSolver;
import com.yishape.lab.math.linalg.solver.DeterminantSolver;
import com.yishape.lab.math.linalg.solver.LinearSystemSolver;
import com.yishape.lab.math.linalg.solver.MatrixInversionSolver;
import com.yishape.lab.math.linalg.solver.RankSolver;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.NonPositiveDefiniteMatrixException;
import com.yishape.lab.math.linalg.decomposition.NonSymmetricMatrixException;
import com.yishape.lab.math.util.RerePrecision;
import java.io.Serializable;

import java.util.concurrent.ForkJoinPool;
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

    private static final YishapeLogger log = YishapeLogger.getLogger(RereDoubleMatrix.class);


    /**
     * 矩阵数据存储数组 / Matrix data storage array
     * <p>
     * 使用二维Double数组存储矩阵数据，data[i][j]表示第i行第j列的元素 Uses 2D Double array to store
     * matrix data, data[i][j] represents element at row i, column j
     * </p>
     */
    private double[][] data;
    
    /**
     * 对于空矩阵（行数为0），存储列数信息 / For empty matrices (0 rows), store column count information
     * 当data.length == 0时，这个字段存储预期的列数
     */
    int emptyMatrixCols = 0;

    /** 标记矩阵是否已在对象池中，防止重复入池 */

    private static final IDoubleVectorComputer computer = new DoubleVectorComputer();

    /**
     * 避免同类矩阵间运算时的防御性拷贝。若 other 也是 RereDoubleMatrix，直接返回其内部 data；
     * 否则回退到 toDoubleArray() 做拷贝。
     */
    private static double[][] otherData(IMatrix<Double> other) {
        return (other instanceof RereDoubleMatrix) ? ((RereDoubleMatrix) other).data : other.toDoubleArray();
    }

    // ========== 性能优化相关字段 / Performance Optimization Fields ==========
    /**
     * 分解用 ForkJoinPool，线程数由系统属性 {@code yishape.decomposition.threads} 控制，默认 min(8, cores)。
     * 用于 SVD 分治、Eigen 收缩后独立子块等场景。
     */
    private static final ForkJoinPool DECOMPOSITION_POOL = createDecompositionPool();

    private static ForkJoinPool createDecompositionPool() {
        int threads = Integer.getInteger("yishape.decomposition.threads",
                Math.min(8, Runtime.getRuntime().availableProcessors()));
        return new ForkJoinPool(threads, ForkJoinPool.defaultForkJoinWorkerThreadFactory,
                (t, e) -> log.warn("Uncaught exception in decomposition pool", e), false);
    }

    /** 并行化阈值：子问题维度低于此不 fork */
    public static final int PARALLELISM_THRESHOLD = 256;

    /**
     * 是否启用并行计算 / Whether parallel computation is enabled
     */
    private static volatile boolean PARALLEL_ENABLED = true;

    /**
     * 预分配内存的构造函数 / Constructor with pre-allocated memory
     */
    public RereDoubleMatrix(int rows, int cols) {
        this.data = new double[rows][cols];
        if (rows == 0) {
            this.emptyMatrixCols = cols;
        }
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
        DECOMPOSITION_POOL.shutdown();
    }

    /**
     * 获取共享线程池（用于 Strassen 等并行计算） / Get shared thread pool (for parallel computation like Strassen)
     */
    public static ForkJoinPool getThreadPool() {
        return DECOMPOSITION_POOL;
    }

    /**
     * 获取分解专用 ForkJoinPool / Get decomposition-dedicated ForkJoinPool
     */
    public static ForkJoinPool getDecompositionPool() {
        return DECOMPOSITION_POOL;
    }

    /**
     * 返回并行化阈值 / Return the parallelism threshold
     */
    public static int getParallelismThreshold() {
        return PARALLELISM_THRESHOLD;
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
    public  IDoubleMatrix sub(double scalar) {
        var res = this.computer.binaryOperate(data, scalar, BinaryOperation.SUBTRACT);
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
    public  IDoubleMatrix sub(IMatrix<Double> other) {
        var res = this.computer.binaryOperate(data, otherData(other), BinaryOperation.SUBTRACT);
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
    public  IDoubleMatrix multiplyByScalar(double scalar) {
        var res = this.computer.binaryOperate(data, scalar, BinaryOperation.MULTIPLY);
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
    public  IDoubleMatrix add(IMatrix<Double> other) {
        var res = this.computer.binaryOperate(data, otherData(other), BinaryOperation.ADD);
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
    public  IDoubleMatrix divide(IMatrix<Double> other) {
        var res = this.computer.binaryOperate(data, otherData(other), BinaryOperation.DIVIDE);
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
    public double frobeniusInnerProduct(IMatrix<Double> other) {
        return this.computer.binaryReduceOperate(data, otherData(other), BinaryReduceOperation.DOT);
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
    public IDoubleVector getColumn(int colIndex) {
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
        if (data.length == 0) {
            throw new IllegalStateException("Cannot put column into an empty matrix");
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
    public double get(int row, int col) {
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
    public void put(int row, int col, double value) {
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
            log.debug("DEBUG: getColNum() called on empty matrix, returning: {}", emptyMatrixCols);
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
    public  IDoubleMatrix transposeInPlace() {
        int r = data.length;
        int c = r > 0 ? data[0].length : 0;
        if (r == c) {
            for (int i = 0; i < r; i++) {
                for (int j = i + 1; j < c; j++) {
                    double tmp = data[i][j];
                    data[i][j] = data[j][i];
                    data[j][i] = tmp;
                }
            }
        } else {
            this.data = this.computer.transpose(data);
        }
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
    public  IDoubleMatrix transposeNew() {
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

        return IDoubleVector.of(data[i].clone());
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
        // 对称特征值 → HPC 对称路径
        Tuple2<IVector<Double>, IMatrix<Double>> ob = HpcLapackDecomps.tryEigenSymmetric(this);
        if (ob != null) {
            return ob;
        }
        // 非对称且特征值均为实数 → HPC 非对称路径
        ob = HpcLapackDecomps.tryEigenNonsymmetric(this);
        if (ob != null) {
            return ob;
        }
        return Decomps.createEigen().decompose(this);
    }

    /**
     * 奇异值分解（Singular Value Decomposition, SVD）
     *
     * <p>
     * 将 m×n 矩阵 A 分解为 A = U Σ V^T：Σ 的非负对角元为奇异值（本实现中 Σ 以长度为 k=min(m,n) 的向量返回），
     * U 的列、V 的列为左/右奇异向量。实现采用瘦型左因子：U 为 m×k 且列正交（U^T U = I_k），
     * 而非满 m×m。</p>
     *
     * <p>
     * 数学原理：</p>
     * <ul>
     * <li>对于任意 m×n 矩阵 A，存在上述分解；r = min(m,n)</li>
     * <li>U：m×r，列正交；本库返回的 V^T：n×n（与 {@link com.yishape.lab.math.linalg.decomposition.ISVDDecomposition} 一致）</li>
     * <li>奇异值 σ₁ ≥ σ₂ ≥ … ≥ σᵣ ≥ 0，在 {@link com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2} 中经后处理保证降序与非负</li>
     * <li>V 为正交矩阵时 V^T V = I_n</li>
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
     * @return 元组 (U, S, V^T)：U 为 m×min(m,n)，S 为奇异值向量长度 min(m,n)，V^T 为 n×n
     * @throws IllegalArgumentException 当矩阵为空（行数或列数为 0）时抛出异常
     */
    @Override
    public Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svd() {
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> ob = HpcLapackDecomps.trySvd(this);
        if (ob != null) {
            return ob;
        }
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
            for (int j = i + 1; j < n; j++) {
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
        Tuple2<IMatrix<Double>, IMatrix<Double>> ob = HpcLapackDecomps.tryQr(this);
        if (ob != null) {
            return ob;
        }
        return Decomps.createQR().decompose(this);
    }

    /**
     * 矩阵乘法运算 / Matrix multiplication
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 矩阵乘法结果 / Matrix multiplication result
     */
    @Override
    public  IDoubleMatrix mmul(IMatrix<Double> other) {
        var res = this.computer.mmul(data, otherData(other));
        return IDoubleMatrix.of(res);
    }

    @Override
    public  IDoubleMatrix kron(IMatrix<Double> other) {
        /**
     * 矩阵Kronecker积 / Matrix Kronecker product
     * <p>
     * 计算当前矩阵与另一个矩阵的Kronecker积（张量积）。对于m×n矩阵A和p×q矩阵B，
     * 结果为mp×nq矩阵，每个元素为A[i,j] * B。
     * </p>
     * <p>
     * Computes the Kronecker product (tensor product) of the current matrix with another matrix.
     * For m×n matrix A and p×q matrix B, the result is an mp×nq matrix where each element is A[i,j] * B.
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return Kronecker积结果矩阵 / Kronecker product result matrix
     * @throws NullPointerException 如果other为null / if other is null
     */
        if (other == null) {
            throw new NullPointerException("other不能为null / other cannot be null");
        }
        int m = getRowNum();
        int n = getColNum();
        int p = other.getRowNum();
        int q = other.getColNum();
        double[][] a = this.data;
        double[][] b = otherData(other);
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
    public  IDoubleMatrix log() {
        var res = this.computer.universalOperate(data, UniversalOperation.LOG, 0);
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
    public double frobeniusNorm() {
        var square = this.computer.universalOperate(data, UniversalOperation.POW, 2.0);
        double sum = this.computer.reduceOperate(square, ReduceOperation.SUM);
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
    public double frobeniusDistance(IMatrix<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }

        if (data.length == 0) {
            return 0.0;
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
    public  IDoubleMatrix normalizeRows() {
        int rows = data.length;
        if (rows == 0) return IDoubleMatrix.of(new double[0][0]);
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        // 使用SIMD对每一行进行L2归一化
        for (int i = 0; i < rows; i++) {
            // 计算行的L2范数
            // 先计算每个元素的平方
            double[] squared = this.computer.universalOperate(data[i], UniversalOperation.POW, 2.0);
            
            // 计算平方和
            double sumOfSquares = this.computer.reduceOperate(squared, ReduceOperation.SUM);
            
            // 计算L2范数
            double norm = Math.sqrt(sumOfSquares);

            if (norm > 0) {
                // 使用SIMD归一化该行
                result[i] = this.computer.binaryOperate(data[i], norm, BinaryOperation.DIVIDE);
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
    public  IDoubleMatrix normalizeColumns() {
        int rows = data.length;
        if (rows == 0) return IDoubleMatrix.of(new double[0][0]);
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
            double[] squared = this.computer.universalOperate(column, UniversalOperation.POW, 2.0);
            
            // 计算平方和
            double sumOfSquares = this.computer.reduceOperate(squared, ReduceOperation.SUM);
            
            // 计算L2范数
            double norm = Math.sqrt(sumOfSquares);

            // 归一化该列
            if (norm > 1e-10) { // 避免除零
                double[] normalizedColumn = this.computer.binaryOperate(column, norm, BinaryOperation.DIVIDE);
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
    public  IDoubleMatrix center() {
        int rows = data.length;
        if (rows == 0) return IDoubleMatrix.of(new double[0][0]);
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        // 使用SIMD计算每列的均值，并减去均值（复用单个列缓冲区）
        double[] columnMeans = new double[cols];
        double[] column = new double[rows]; // reuse buffer
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            columnMeans[j] = this.computer.reduceOperate(column, ReduceOperation.MEAN);
        }

        // 使用SIMD对每列减去均值
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            double[] centeredColumn = this.computer.binaryOperate(column, columnMeans[j], BinaryOperation.SUBTRACT);
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
    public  IDoubleMatrix covariance() {
        // 先进行数据中心化
        IDoubleMatrix centered = this.center();
        return centered.covarianceFromCentered();
    }

    /**
     * covariance的别名函数
     *
     * @return
     */
    @Override
    public  IDoubleMatrix cov() {
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
    public  IDoubleMatrix covarianceFromCentered() {
        int n = data.length; // 样本数
        if (n < 2) {
            throw new IllegalArgumentException(
                    "covarianceFromCentered 至少需要 2 个样本（行数≥2），当前 n=" + n
                            + " / At least 2 samples (rows) required, got n=" + n);
        }

        // 计算 X^T * X
        IDoubleMatrix transposed = this.transposeNew();
        IDoubleMatrix product = ((RereDoubleMatrix) transposed).mmul(this);

        // 除以 (n-1) 得到协方差矩阵
        return product.multiplyByScalar(1.0 / (n - 1));
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
    public  IDoubleMatrix inv() {
        IDoubleMatrix ob = (IDoubleMatrix)HpcLapackDecomps.tryInverse(this);
        if (ob != null) {
            return ob;
        }
        return (IDoubleMatrix)MatrixInversionSolver.invert(this);
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
    public  IDoubleMatrix pinv() {
        return (IDoubleMatrix)MatrixInversionSolver.pseudoInverse(this);
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
    public  IDoubleMatrix hstack(IMatrix<Double> other1) {
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
    public  IDoubleMatrix vstack(IMatrix<Double> other1) {
        if (other1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }

        if (data.length == 0) {
            return (IDoubleMatrix) other1;
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
    public  IDoubleMatrix[] hsplit(int[] indices) {
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
        IDoubleMatrix[] result = new IDoubleMatrix[numSplits];

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
    public  IDoubleMatrix[] vsplit(int[] indices) {
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
        IDoubleMatrix[] result = new IDoubleMatrix[numSplits];

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
    public  IDoubleMatrix reshape(int rows, int cols) {
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
    public  IDoubleMatrix copy() {
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
    public double det() {
        Double hpc = HpcLapackDecomps.tryDet(this);
        if (hpc != null) return hpc;
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
    public double trace() {
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
    public double cond() {
        Double hpc = HpcLapackDecomps.tryCond(this);
        if (hpc != null) return hpc;
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
        Integer hpc = HpcLapackDecomps.tryRank(this);
        if (hpc != null) return hpc;
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
    public  IDoubleMatrix abs() {
        var res = this.computer.universalOperate(data, UniversalOperation.ABS, 0);
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
    public  IDoubleMatrix sign() {
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
    public  IDoubleMatrix sin() {
        var res = this.computer.universalOperate(data, UniversalOperation.SIN, 0);
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
    public  IDoubleMatrix cos() {
        var res = this.computer.universalOperate(data, UniversalOperation.COS, 0);
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
    public  IDoubleMatrix tan() {
        var res = this.computer.universalOperate(data, UniversalOperation.TAN, 0);
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
    public  IDoubleMatrix sinh() {
        var res = this.computer.universalOperate(data, UniversalOperation.SINH, 0);
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
    public  IDoubleMatrix cosh() {
        var res = this.computer.universalOperate(data, UniversalOperation.COSH, 0);
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
    public  IDoubleMatrix tanh() {
        var res = this.computer.universalOperate(data, UniversalOperation.TANH, 0);
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
        Tuple2<IMatrix<Double>, IMatrix<Double>> ob = HpcLapackDecomps.tryLu(this);
        if (ob != null) {
            return ob;
        }
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
    public  IDoubleMatrix cholesky() {
        try {
            IDoubleMatrix ob = (IDoubleMatrix)HpcLapackDecomps.tryCholeskyL(this);
            if (ob != null) {
                return ob;
            }
        } catch (NonSymmetricMatrixException | NonPositiveDefiniteMatrixException e) {
            throw e;
        }
        return (IDoubleMatrix)Decomps.createCholesky().decompose(this);
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
        if (b == null) {
            throw new IllegalArgumentException("Input vector cannot be null");
        }
        if (data.length != b.size()) {
            throw new IllegalArgumentException(
                    "Matrix row dimension mismatch: A has " + data.length
                            + " rows but b has " + b.size() + " elements");
        }
        double[] bd = RereDoubleVector.otherData(b);
        double[][] col = new double[bd.length][1];
        for (int i = 0; i < bd.length; i++) {
            col[i][0] = bd[i];
        }
        return solve(Linalg.matrix(col)).getColumn(0);
    }

    /**
     * 求解线性方程组 {@code AX = B}（矩阵右端）。
     * <p>策略同 {@link LinearSystemSolver}；在一般方阵 LU 步骤之前可经注入尝试 yishape-math-hpc
     * {@code solveSquare}（仅 {@link RereDoubleMatrix} 层接入 HPC）。</p>
     *
     * @param B 右侧矩阵
     * @return 解矩阵 {@code X}
     */
    @Override
    public  IDoubleMatrix solve(IMatrix<Double> B) {
        // v0.5.0：多 RHS 优先走 HPC 批量路径
        if (B.cols() > 1) {
            IDoubleMatrix ob = (IDoubleMatrix)HpcLapackDecomps.trySolveMultiRhs(this, B);
            if (ob != null) {
                return ob;
            }
        }
        return (IDoubleMatrix)LinearSystemSolver.solve(this, B, HpcLapackDecomps::trySolveSquareRhs);
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
    public double max() {
        return this.computer.reduceOperate(data, ReduceOperation.MAX);
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
    public double min() {
        return this.computer.reduceOperate(data, ReduceOperation.MIN);
    }

    /**
     * 矩阵元素求和 / Matrix element sum
     * <p>
     * 返回矩阵中所有元素的总和（1×1 矩阵）Returns the sum of all elements in the matrix (1×1 matrix)
     * </p>
     *
     * @return 1×1 矩阵，包含元素总和 / 1×1 matrix containing the sum of all elements
     */
    @Override
    public IDoubleMatrix sum() {
        double result = this.computer.reduceOperate(data, ReduceOperation.SUM);
        return IDoubleMatrix.of(new double[][]{{result}});
    }

    /**
     * 矩阵元素均值 / Matrix element mean
     * <p>
     * 返回矩阵中所有元素的平均值（1×1 矩阵）Returns the mean of all elements in the matrix (1×1 matrix)
     * </p>
     *
     * @return 1×1 矩阵，包含元素均值 / 1×1 matrix containing the mean of all elements
     */
    @Override
    public IDoubleMatrix mean() {
        double result = this.computer.reduceOperate(data, ReduceOperation.MEAN);
        return IDoubleMatrix.of(new double[][]{{result}});
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
    public double std() {
        return this.computer.reduceOperate(data, ReduceOperation.STANDARD_DEVIATION);
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
    public double var() {
        return this.computer.reduceOperate(data, ReduceOperation.VARIANCE);
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
    public  IDoubleMatrix slice(String rowSlice, String colSlice) {
        if (data.length == 0 || (data.length > 0 && data[0].length == 0)) {
            return IDoubleMatrix.of(new double[0][0]);
        }

        IndexExpressionParser.SliceResult rowResult =
            IndexExpressionParser.parse(rowSlice, data.length);
        IndexExpressionParser.SliceResult colResult =
            IndexExpressionParser.parse(colSlice, data[0].length);

        int resultRows = IndexExpressionParser.calculateSliceSizeLegacy(
            rowResult.actualStart, rowResult.actualEnd, rowResult.step);
        int resultCols = IndexExpressionParser.calculateSliceSizeLegacy(
            colResult.actualStart, colResult.actualEnd, colResult.step);

        if (resultRows <= 0 || resultCols <= 0) {
            if (resultRows <= 0 && resultCols > 0) {
                return new RereDoubleMatrix(0, resultCols);
            } else if (resultCols <= 0 && resultRows > 0) {
                return new RereDoubleMatrix(resultRows, 0);
            } else {
                return IDoubleMatrix.of(new double[0][0]);
            }
        }

        int[] rowIndices = IndexExpressionParser.generateIndices(rowResult);
        int[] colIndices = IndexExpressionParser.generateIndices(colResult);

        double[][] result = new double[resultRows][resultCols];
        for (int i = 0; i < rowIndices.length; i++) {
            for (int j = 0; j < colIndices.length; j++) {
                result[i][j] = data[rowIndices[i]][colIndices[j]];
            }
        }

        return IDoubleMatrix.of(result);
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
    public  IDoubleMatrix sliceRows(String rowSlice) {
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
    public  IDoubleMatrix sliceColumns(String colSlice) {
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
    public  IDoubleMatrix fancyGet(int[] rowIndices, int[] colIndices) {
        if (rowIndices == null || colIndices == null) {
            throw new IllegalArgumentException("索引数组不能为null / Index arrays cannot be null");
        }

        IndexExpressionParser.FancyIndexResult resolvedRows =
            IndexExpressionParser.resolveFancyIndex(rowIndices, data.length);
        IndexExpressionParser.FancyIndexResult resolvedCols =
            IndexExpressionParser.resolveFancyIndex(colIndices, data[0].length);

        double[][] result = new double[resolvedRows.indices.length][resolvedCols.indices.length];
        for (int i = 0; i < resolvedRows.indices.length; i++) {
            for (int j = 0; j < resolvedCols.indices.length; j++) {
                result[i][j] = data[resolvedRows.indices[i]][resolvedCols.indices[j]];
            }
        }
        return IDoubleMatrix.of(result);
    }

    @Override
    public  IDoubleMatrix booleanGet(boolean[] rowMask) {
        if (rowMask == null) {
            throw new IllegalArgumentException("布尔索引数组不能为null / Boolean index array cannot be null");
        }
        if (rowMask.length != data.length) {
            throw new IllegalArgumentException("布尔索引数组长度与矩阵行数不匹配: " + rowMask.length + " != " + data.length
                + " / Boolean index array length doesn't match matrix row count");
        }
        IndexExpressionParser.BooleanIndexResult resolved =
            IndexExpressionParser.resolveBooleanIndex(rowMask);
        int cols = data.length > 0 ? data[0].length : 0;
        double[][] result = new double[resolved.count][cols];
        for (int i = 0; i < resolved.trueIndices.length; i++) {
            int row = resolved.trueIndices[i];
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[row][j];
            }
        }
        return IDoubleMatrix.of(result);
    }

    @Override
    public  IDoubleMatrix booleanGet(boolean[] rowMask, boolean[] colMask) {
        if (rowMask == null || colMask == null) {
            throw new IllegalArgumentException("布尔索引数组不能为null / Boolean index arrays cannot be null");
        }
        if (rowMask.length != data.length) {
            throw new IllegalArgumentException("行布尔索引长度与矩阵行数不匹配: " + rowMask.length + " != " + data.length);
        }
        int cols = data.length > 0 ? data[0].length : 0;
        if (colMask.length != cols) {
            throw new IllegalArgumentException("列布尔索引长度与矩阵列数不匹配: " + colMask.length + " != " + cols);
        }
        IndexExpressionParser.BooleanIndexResult resolvedRows =
            IndexExpressionParser.resolveBooleanIndex(rowMask);
        IndexExpressionParser.BooleanIndexResult resolvedCols =
            IndexExpressionParser.resolveBooleanIndex(colMask);
        double[][] result = new double[resolvedRows.count][resolvedCols.count];
        for (int i = 0; i < resolvedRows.trueIndices.length; i++) {
            for (int j = 0; j < resolvedCols.trueIndices.length; j++) {
                result[i][j] = data[resolvedRows.trueIndices[i]][resolvedCols.trueIndices[j]];
            }
        }
        return IDoubleMatrix.of(result);
    }

    @Override
    public void fancySet(int[] rowIndices, int[] colIndices, Double[] values) {
        if (rowIndices == null || colIndices == null || values == null) {
            throw new IllegalArgumentException("索引数组和值数组不能为null / Index arrays and values array cannot be null");
        }
        int expectedLen = rowIndices.length * colIndices.length;
        if (values.length != expectedLen) {
            throw new IllegalArgumentException("值数组长度(" + values.length + ")须等于行索引数(" + rowIndices.length + ")乘以列索引数(" + colIndices.length + ") / Values length must equal rowIndices.length * colIndices.length");
        }
        IndexExpressionParser.FancyIndexResult resolvedRows =
            IndexExpressionParser.resolveFancyIndex(rowIndices, data.length);
        IndexExpressionParser.FancyIndexResult resolvedCols =
            IndexExpressionParser.resolveFancyIndex(colIndices, data[0].length);
        int idx = 0;
        for (int ri : resolvedRows.indices) {
            for (int ci : resolvedCols.indices) {
                data[ri][ci] = values[idx++];
            }
        }
    }

    @Override
    public void fancySetScalar(int[] rowIndices, int[] colIndices, Double value) {
        if (rowIndices == null || colIndices == null) {
            throw new IllegalArgumentException("索引数组不能为null / Index arrays cannot be null");
        }
        IndexExpressionParser.FancyIndexResult resolvedRows =
            IndexExpressionParser.resolveFancyIndex(rowIndices, data.length);
        IndexExpressionParser.FancyIndexResult resolvedCols =
            IndexExpressionParser.resolveFancyIndex(colIndices, data[0].length);
        for (int ri : resolvedRows.indices) {
            for (int ci : resolvedCols.indices) {
                data[ri][ci] = value;
            }
        }
    }

    // set() method is now a default method in IMatrix<Double> that delegates to put()
    /**
     * 获取矩阵对角线元素 / Get matrix diagonal elements
     * <p>
     * 返回矩阵主对角线上的元素作为向量。对于n×m矩阵，返回min(n,m)个元素。
     * </p>
     * <p>
     * Returns the elements on the main diagonal of the matrix as a vector.
     * For an n×m matrix, returns min(n,m) elements.
     * </p>
     *
     * @return 对角线元素向量 / Diagonal elements vector
     */
    @Override
    public IVector<Double> diag() {
        int minDim = Math.min(data.length, data[0].length);
        double[] diagonal = new double[minDim];
        for (int i = 0; i < minDim; i++) {
            diagonal[i] = data[i][i];
        }
        return IDoubleVector.of(diagonal);
    }

    /**
     * 获取矩阵子矩阵 / Get matrix submatrix
     * <p>
     * 提取矩阵指定范围的子矩阵，支持负数索引（负数表示从末尾开始计算）。
     * </p>
     * <p>
     * Extracts a submatrix from the specified range of the matrix, supporting negative indexing
     * (negative numbers indicate counting from the end).
     * </p>
     *
     * @param startRow 起始行索引（包含）/ Start row index (inclusive)
     * @param endRow 结束行索引（不包含）/ End row index (exclusive)
     * @param startCol 起始列索引（包含）/ Start column index (inclusive)
     * @param endCol 结束列索引（不包含）/ End column index (exclusive)
     * @return 子矩阵 / Submatrix
     * @throws IndexOutOfBoundsException 如果索引超出范围 / if index is out of bounds
     * @throws IllegalArgumentException 如果起始索引大于等于结束索引 / if start index is greater than or equal to end index
     */
    @Override
    public  IDoubleMatrix subMatrix(int startRow, int endRow, int startCol, int endCol) {
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

    /**
     * 设置矩阵子矩阵 / Set matrix submatrix
     * <p>
     * 将指定子矩阵设置到矩阵的指定位置，支持负数索引（负数表示从末尾开始计算）。
     * </p>
     * <p>
     * Sets a submatrix at the specified position of the matrix, supporting negative indexing
     * (negative numbers indicate counting from the end).
     * </p>
     *
     * @param startRow 起始行索引（包含）/ Start row index (inclusive)
     * @param endRow 结束行索引（不包含）/ End row index (exclusive)
     * @param startCol 起始列索引（包含）/ Start column index (inclusive)
     * @param endCol 结束列索引（不包含）/ End column index (exclusive)
     * @param subMatrix 要设置的子矩阵 / Submatrix to set
     * @throws IndexOutOfBoundsException 如果索引超出范围 / if index is out of bounds
     * @throws IllegalArgumentException 如果子矩阵尺寸不匹配 / if submatrix dimensions don't match
     */
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
        for (int i = 0; i < rows; i++) {
            sums[i] = this.computer.reduceOperate(data[i], ReduceOperation.SUM);
        }
        return IDoubleVector.of(sums);
    }

    @Override
    public IVector<Double> rowMeans() {
        int rows = data.length;
        int cols = data[0].length;
        double[] means = new double[rows];
        for (int i = 0; i < rows; i++) {
            means[i] = this.computer.reduceOperate(data[i], ReduceOperation.MEAN);
        }
        return IDoubleVector.of(means);
    }

    @Override
    public IVector<Double> colSums() {
        int rows = data.length;
        if (rows == 0) return IDoubleVector.of(new double[0]);
        int cols = data[0].length;
        double[] sums = new double[cols];
        double[] column = new double[rows]; // reuse buffer across columns
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            sums[j] = this.computer.reduceOperate(column, ReduceOperation.SUM);
        }
        return IDoubleVector.of(sums);
    }

    @Override
    public IVector<Double> colMeans() {
        int rows = data.length;
        if (rows == 0) return IDoubleVector.of(new double[0]);
        int cols = data[0].length;
        double[] means = new double[cols];
        double[] column = new double[rows]; // reuse buffer across columns
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            means[j] = this.computer.reduceOperate(column, ReduceOperation.MEAN);
        }
        return IDoubleVector.of(means);
    }

    @Override
    public IVector<Double> min(int axis) {
        if (axis == 0) {
            return colMins();
        } else if (axis == 1) {
            return rowMins();
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public IVector<Double> max(int axis) {
        if (axis == 0) {
            return colMaxs();
        } else if (axis == 1) {
            return rowMaxs();
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public IVector<Double> sum(int axis) {
        if (axis == 0) {
            return colSums();
        } else if (axis == 1) {
            return rowSums();
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public IVector<Double> mean(int axis) {
        if (axis == 0) {
            return colMeans();
        } else if (axis == 1) {
            return rowMeans();
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public IVector<Double> rowMins() {
        return apply(IVector::minValue, 1);
    }

    @Override
    public IVector<Double> rowMaxs() {
        return apply(IVector::maxValue, 1);
    }

    @Override
    public IVector<Double> colMins() {
        return apply(IVector::minValue, 0);
    }

    @Override
    public IVector<Double> colMaxs() {
        return apply(IVector::maxValue, 0);
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
    public  IDoubleMatrix sqrt() {
        var res = this.computer.universalOperate(data, UniversalOperation.SQRT, 0);
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
    public  IDoubleMatrix pow(Double power) {
        var res = this.computer.universalOperate(data, UniversalOperation.POW, power);
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
    public  IDoubleMatrix exp() {
        var res = this.computer.universalOperate(data, UniversalOperation.EXP, 0);
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
    public  IDoubleMatrix getColumnMatrix(int colIndex) {
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

    /**
     * 将矩阵展平为向量 / Flatten matrix to vector
     * <p>
     * 将矩阵转换为一维向量。对于列向量，返回该列的数据；对于普通矩阵，按行优先顺序展开。
     * </p>
     * <p>
     * Converts the matrix to a 1D vector. For column vectors, returns the column data;
     * for regular matrices, flattens in row-major order.
     * </p>
     *
     * @return 展平后的向量 / Flattened vector
     */
    @Override
    public IVector<Double> flatten() {
        var as = this.toFlattenArray();
        return IDoubleVector.of(as);
    }

    @Override
    public double[][] toDoubleArray() {
        int rows = data.length;
        if (rows == 0) {
            return new double[0][];
        }
        int cols = data[0].length;
        double[][] copy = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, cols);
        }
        return copy;
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
    public  IDoubleMatrix getColumnAsCloumnVector(int i) {
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
        double[] columnData = RereDoubleVector.otherData(column);

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
        double[] rowData = RereDoubleVector.otherData(row);

        // 设置指定行的数据
        System.arraycopy(rowData, 0, data[rowIndex], 0, data[0].length);
    }

    @Override
    public IVector<Double> apply(Function<IVector<Double>, Double> fun, int axis) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }
        if (axis == 0) {
            // JIT EA 可在此处内联并标量替换 colData 数组
            int rows = data.length;
            int cols = data[0].length;
            double[] result = new double[cols];
            for (int j = 0; j < cols; j++) {
                double[] colData = new double[rows];
                for (int i = 0; i < rows; i++) {
                    colData[i] = data[i][j];
                }
                result[j] = fun.apply(IDoubleVector.of(colData));
            }
            return IDoubleVector.of(result);
        } else if (axis == 1) {
            // axis=1 时直接引用 data[i]，避免 getRow() 的间接调用
            double[] result = new double[rows()];
            for (int i = 0; i < rows(); i++) {
                result[i] = fun.apply(IDoubleVector.of(data[i]));
            }
            return IDoubleVector.of(result);
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public  IDoubleMatrix applyMap(Function<Double, Double> fun) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }

        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            // JIT EA 可在简单 lambda 下将 srcRow/dstRow 标量替换为寄存器变量
            double[] srcRow = data[i];
            double[] dstRow = result[i];
            for (int j = 0; j < cols; j++) {
                // fun.apply 会触发 double→Double 装箱，但 JIT 会对 x->x*2 等简单 lambda
                // 做逃逸分析并内联，随后将其标量替换，使装箱对象不复存在
                dstRow[j] = fun.apply(srcRow[j]);
            }
        }

        // 直接构造而非工厂方法：绕过类型推断，零开销
        return new RereDoubleMatrix(result);
    }

    @Override
    public double normL1() {
        int rows = data.length;
        int cols = data[0].length;
        double maxSum = 0;
        for (int j = 0; j < cols; j++) {
            double colSum = 0;
            for (int i = 0; i < rows; i++) {
                colSum += Math.abs(data[i][j]);
            }
            if (colSum > maxSum) maxSum = colSum;
        }
        return maxSum;
    }

    @Override
    public double normInf() {
        int rows = data.length;
        int cols = data[0].length;
        double maxSum = 0;
        for (int i = 0; i < rows; i++) {
            double rowSum = 0;
            for (int j = 0; j < cols; j++) {
                rowSum += Math.abs(data[i][j]);
            }
            if (rowSum > maxSum) maxSum = rowSum;
        }
        return maxSum;
    }

    @Override
    public IVector<Double> mmul(IVector<Double> other) {
        var m2 = other.asColumnMatrix();
        var res = this.computer.mmul(data, otherData(m2));
        // Extract the column vector from the result matrix
        double[] resultVector = new double[res.length];
        for (int i = 0; i < res.length; i++) {
            resultVector[i] = res[i][0];
        }
        return IDoubleVector.of(resultVector);
    }

    @Override
    public  IDoubleMatrix divideByScalar(double scalar) {
        if (RerePrecision.equalsZero(scalar, 1e-12)) {
            throw new ArithmeticException("除数不能为零 / Divisor cannot be zero");
        }
        var res = this.computer.binaryOperate(data, scalar, BinaryOperation.DIVIDE);
        return IDoubleMatrix.of(res);
    }


    @Override
    public  IDoubleMatrix normalize() {
        int rows = this.data.length;
        int cols = this.data[0].length;
        double[][] result = new double[rows][cols];

        // 使用SIMD计算每行的L2范数
        for (int i = 0; i < rows; i++) {
            // 计算每个元素的平方
            double[] squared = this.computer.universalOperate(this.data[i], UniversalOperation.POW, 2.0);
            
            // 计算平方和
            double sumOfSquares = this.computer.reduceOperate(squared, ReduceOperation.SUM);
            
            // 计算L2范数
            double norm = Math.sqrt(sumOfSquares);

            if (RerePrecision.equalsZero(norm, 1e-12)) {
                throw new ArithmeticException("第" + i + "行的L2范数为零，无法归一化 / Row " + i + " L2 norm is zero, cannot normalize");
            }

            // 使用SIMD归一化每行
            result[i] = this.computer.binaryOperate(this.data[i], norm, BinaryOperation.DIVIDE);
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
    public  IDoubleMatrix broadcastColumn(IVector<Double> colVector, BiFunction<IVector<Double>, IVector<Double>, IVector<Double>> fun) {
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
    public  IDoubleMatrix broadcastRow(IVector<Double> rowVector, BiFunction<IVector<Double>, IVector<Double>, IVector<Double>> fun) {
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
    public  IDoubleMatrix multiply(IMatrix<Double> other) {
        var res = this.computer.binaryOperate(data, otherData(other), BinaryOperation.MULTIPLY);
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

        return this.computer.logicalCompare(data, otherData(other), LogicalCompare.EQUALS);
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

        return this.computer.logicalCompare(data, otherData(other), LogicalCompare.LESS_THAN);
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
        return this.computer.logicalCompare(data, otherData(other), LogicalCompare.GREATER_THAN);
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
        return this.computer.logicalCompare(data, otherData(other), LogicalCompare.GREATER_THAN_OR_EQUALS);
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
        return this.computer.logicalCompare(data, otherData(other), LogicalCompare.LESS_THAN_OR_EQUALS);
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
    public  IDoubleMatrix forwardSolve(IMatrix<Double> B) {
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
                X[i][j] = B.get(i, j);
            }
        }

        // Forward substitution: solve LX = B
        //
        // LOOP ORDERING (fixed 2026-05-15): The innermost loop iterates over columns
        // (stride-1) rather than the previous row-innermost ordering (stride-N).
        // For n=500 with B=eye(500), the old ordering accessed X[j][col] with 4KB stride
        // per iteration, causing L1 cache misses on every access. The new ordering
        // makes both X[i][col] and X[j][col] stride-1, enabling full cache line
        // utilization and JIT auto-vectorization. Do NOT revert to row-innermost
        // ordering without benchmarking on matrices n >= 200.
        //
        // The zero check on lij skips structural zeros in triangular matrices
        // (e.g. lowerTriMatrix where upper triangle is all zeros).
        //
        // Reciprocal multiplication (1.0/diag) replaces division; diverges by <= 1 ULP.
        for (int i = 0; i < n; i++) {
            // Subtract L[i][j] * row_j from row_i for all j < i
            for (int j = 0; j < i; j++) {
                double lij = this.data[i][j];
                if (lij == 0.0) continue;
                for (int col = 0; col < nrhs; col++) {
                    X[i][col] -= lij * X[j][col];
                }
            }

            // Check for zero diagonal element
            double diag = this.data[i][i];
            if (Math.abs(diag) < 1e-12) {
                throw new ArithmeticException("Matrix is singular, diagonal element is zero at index " + i);
            }

            // Divide row i by diagonal element
            double invDiag = 1.0 / diag;
            for (int col = 0; col < nrhs; col++) {
                X[i][col] *= invDiag;
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
    public  IDoubleMatrix backwardSolve(IMatrix<Double> B) {
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
                X[i][j] = B.get(i, j);
            }
        }

        // Backward substitution: solve UX = B
        //
        // LOOP ORDERING: same cache optimization as forwardSolve above (fixed 2026-05-15).
        // Innermost loop over columns for stride-1 access. See forwardSolve for rationale.
        for (int i = n - 1; i >= 0; i--) {
            // Subtract U[i][j] * row_j from row_i for all j > i
            for (int j = i + 1; j < n; j++) {
                double uij = this.data[i][j];
                if (uij == 0.0) continue;
                for (int col = 0; col < nrhs; col++) {
                    X[i][col] -= uij * X[j][col];
                }
            }

            // Check for zero diagonal element
            double diag = this.data[i][i];
            if (Math.abs(diag) < 1e-12) {
                throw new ArithmeticException("Matrix is singular, diagonal element is zero at index " + i);
            }

            // Divide row i by diagonal element
            double invDiag = 1.0 / diag;
            for (int col = 0; col < nrhs; col++) {
                X[i][col] *= invDiag;
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
        return IDoubleMatrix.of(broadcastTo(otherData(data), targetRows, targetCols));
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
        return IDoubleMatrix.of(broadcastElementWise(otherData(a), otherData(b), op));
    }

    private static double[][] copy2dForBroadcast(double[][] data) {
        double[][] c = new double[data.length][];
        for (int i = 0; i < data.length; i++) {
            c[i] = data[i].clone();
        }
        return c;
    }

}
