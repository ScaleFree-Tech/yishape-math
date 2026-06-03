package com.yishape.lab.math.linalg;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.compute.FloatVectorComputer;
import com.yishape.lab.math.compute.IFloatVectorComputer;
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
import com.yishape.lab.math.util.RerePrecision;
import java.io.Serializable;

import java.util.concurrent.*;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * 矩阵操作实现类 / Matrix Operations Implementation Class 本类按行存储向量数据
 * <p>
 * 本类实现了矩阵的常用操作，包括基本的数学运算、矩阵变换、数据访问等功能。 基于二维Float数组实现，提供高效的矩阵计算能力。
 * </p>
 * <p>
 * This class implements common matrix operations including basic mathematical
 * operations, matrix transformations, data access and other functionalities.
 * Based on 2D Float array implementation, providing efficient matrix
 * computation capabilities.
 * </p>
 *
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 * <li>基本数学运算：加法、减法、乘法、除法 / Basic math operations: add, subtract, multiply,
 * divide</li>
 * <li>矩阵变换：转置、幂运算、开方 / IMatrix<Float> transformations: transpose, power,
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
 * float[][] data = {{1, 2}, {3, 4}};
 * IMatrix<Float> matrix = new RereFloatMatrix(data);
 *
 * // 矩阵运算 / IMatrix<Float> operations
 * IMatrix<Float> result = matrix.add(other).mmul(2.0.0);
 *
 * // 获取行列 / Get rows/columns
 * IVector<Float> row = matrix.getRow(0);
 * IVector<Float> col = matrix.getColunm(0);
 * }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereFloatMatrix implements IFloatMatrix,Serializable {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereFloatMatrix.class);


    /**
     * 矩阵数据存储数组 / Matrix data storage array
     * <p>
     * 使用二维Float数组存储矩阵数据，data[i][j]表示第i行第j列的元素 Uses 2D Float array to store
     * matrix data, data[i][j] represents element at row i, column j
     * </p>
     */
    float[][] data;
    
    /**
     * 对于空矩阵（行数为0），存储列数信息 / For empty matrices (0 rows), store column count information
     * 当data.length == 0时，这个字段存储预期的列数
     */
    int emptyMatrixCols = 0;

    private static final IFloatVectorComputer computer = new FloatVectorComputer();

    /**
     * 避免同类矩阵间运算时的防御性拷贝。若 other 也是 RereFloatMatrix，直接返回其内部 data；
     * 否则回退到 toFloatArray() 做拷贝。
     */
    private static float[][] otherData(IMatrix<Float> other) {
        return (other instanceof RereFloatMatrix) ? ((RereFloatMatrix) other).data : other.toFloatArray();
    }

    // ========== 性能优化相关字段 / Performance Optimization Fields ==========
    /**
     * @deprecated 委托到 {@link RereDoubleMatrix#getThreadPool()} 共享池
     */
    @Deprecated
    private static final ExecutorService THREAD_POOL = RereDoubleMatrix.getThreadPool();

    /**
     * 矩阵对象池 / Matrix object pool
     */
    private static final Queue<RereFloatMatrix> MATRIX_POOL = new ConcurrentLinkedQueue<>();

    /**
     * 对象池最大大小 / Maximum pool size
     */
    private static final int MAX_POOL_SIZE = 100;

    /**
     * 是否启用并行计算 / Whether parallel computation is enabled
     */
    private static volatile boolean PARALLEL_ENABLED = true;

    // ========== 对象池化和内存管理方法 / Object Pooling and Memory Management Methods ==========
    /**
     * 从对象池获取矩阵实例 / Get matrix instance from object pool
     */
    public static RereFloatMatrix borrowMatrix(int rows, int cols) {
        RereFloatMatrix matrix = MATRIX_POOL.poll();
        if (matrix == null || matrix.data.length != rows || matrix.data[0].length != cols) {
            return new RereFloatMatrix(rows, cols);
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
    public RereFloatMatrix(int rows, int cols) {
        this.data = new float[rows][cols];
        if (rows == 0) {
            this.emptyMatrixCols = cols;
        }
    }

    /**
     * 获取预分配缓冲区 / Get pre-allocated buffer
     */
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
     * @param data 二维Float数组，表示矩阵数据 / 2D Float array representing matrix data
     * @throws IllegalArgumentException 如果数据为null或维度不一致 / if data is null or
     * dimensions are inconsistent
     */
    public RereFloatMatrix(float[][] data) {
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

    public RereFloatMatrix(Float[][] data) {
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

        // 转换float[][]到float[][]
        this.data = new float[data.length][data[0].length];
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
    public  IFloatMatrix sub(double scalar) {
        var res = this.computer.binaryOperate(data, (float)scalar, BinaryOperation.SUBTRACT);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix sub(IMatrix<Float> other) {
        var res = this.computer.binaryOperate(data, otherData(other), BinaryOperation.SUBTRACT);
        return IFloatMatrix.of(res);
    }

    /**
     * 矩阵标量乘法运算（Float） / Matrix scalar multiplication (Float)
     * <p>
     * 矩阵中的每个元素乘以标量值 Multiplies each element in the matrix by a scalar value
     * </p>
     *
     * @param scalar 标量乘数 / The scalar multiplier
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    @Override
    public  IFloatMatrix multiplyByScalar(double scalar) {
        var res = this.computer.binaryOperate(data, (float)scalar, BinaryOperation.MULTIPLY);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix add(IMatrix<Float> other) {
        var res = this.computer.binaryOperate(data, otherData(other), BinaryOperation.ADD);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix divide(IMatrix<Float> other) {
        var res = this.computer.binaryOperate(data, otherData(other), BinaryOperation.DIVIDE);
        return IFloatMatrix.of(res);
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
    public double frobeniusInnerProduct(IMatrix<Float> other) {
        var res = this.computer.binaryReduceOperate(data, otherData(other), BinaryReduceOperation.DOT);
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
    public IFloatVector getColumn(int colIndex) {
        int rows = data.length;
        float[] result = new float[rows];
        for (int i = 0; i < rows; i++) {
            result[i] = data[i][colIndex];
        }
        return new RereFloatVector(result);
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
    public void putColumn(int colIndex, IMatrix<Float> column1) {
        if (column1 == null) {
            throw new IllegalArgumentException("列矩阵不能为null / Column matrix cannot be null");
        }
        if (colIndex < 0 || colIndex >= data[0].length) {
            throw new IndexOutOfBoundsException("列索引超出范围: " + colIndex + " / Column index out of bounds: " + colIndex);
        }
        IFloatMatrix other0 = (IFloatMatrix) column1;
        float[][] columnData = other0.getData();
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
    public IVector<Float>[] getColumns(int[] indices) {
        var result = new IFloatVector[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = (IFloatVector) getColumn(indices[i]);
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

        data[row][col] = (float)value;
    }

    // Method removed - now inherited from IMatrix via IMatrix<Float>
    // getRows() and getColumns() are now default methods in IMatrix<Float>
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
    public float[][] getData() {
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
    public  IFloatMatrix transposeInPlace() {
        int r = data.length;
        int c = r > 0 ? data[0].length : 0;
        if (r == c) {
            for (int i = 0; i < r; i++) {
                for (int j = i + 1; j < c; j++) {
                    float tmp = data[i][j];
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
    public  IFloatMatrix transposeNew() {
        var res = this.computer.transpose(data);
        return IFloatMatrix.of(res);
    }

    /**
     * 获取指定行向量 / Get specified row vector
     * <p>
     * 返回矩阵指定行的向量表示 Returns the vector representation of the specified row in
     * the matrix
     * </p>
     *
     * @param i 行索引（从0开始） / Row index (0-based)
     * @return 指定行的向量 / IVector<Float> of the specified row
     * @throws IndexOutOfBoundsException 如果行索引超出范围 / if row index is out of
     * bounds
     */
    @Override
    public IVector<Float> getRow(int i) {
        if (i < 0 || i >= data.length) {
            throw new IndexOutOfBoundsException("行索引超出范围: " + i + " / Row index out of bounds: " + i);
        }

        return IFloatVector.of(data[i]);
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
    public Tuple2<IVector<Float>, IMatrix<Float>> eigen() {
        var tp = this.toDoubleMatrix().eigen();
        return new Tuple2(tp._1.toFloatVector(), tp._2.toFloatMatrix());
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
     * <li>奇异值 σ₁ ≥ σ₂ ≥ … ≥ σᵣ ≥ 0</li>
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
    public Tuple3<IMatrix<Float>, IVector<Float>, IMatrix<Float>> svd() {
        var tp = this.toDoubleMatrix().svd();
        return new Tuple3(tp._1.toFloatMatrix(), tp._2.toFloatVector(), tp._3.toFloatMatrix());
    }

    /**
     * 检查矩阵是否对称
     *
     * @return
     */
    @Override
    public boolean isSymmetric() {
        int n = data.length;
        float tolerance = 1e-10f;

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
    public Tuple2<IMatrix<Float>, IMatrix<Float>> qr() {
        // 使用新的QR分解实现类
        var tp = Decomps.createQR().decompose(this.toDoubleMatrix());
        return new Tuple2(tp._1.toFloatMatrix(),tp._2.toFloatMatrix());
    }

    /**
     * 矩阵乘法运算 / Matrix multiplication
     *
     * @param other 另一个矩阵 / The other matrix
     * @return 矩阵乘法结果 / Matrix multiplication result
     */
    @Override
    public  IFloatMatrix mmul(IMatrix<Float> other) {
        var res = this.computer.mmul(data, otherData(other));
        return IFloatMatrix.of(res);
    }

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
    @Override
    public  IFloatMatrix kron(IMatrix<Float> other) {
        if (other == null) {
            throw new NullPointerException("other不能为null / other cannot be null");
        }
        int m = getRowNum();
        int n = getColNum();
        int p = other.getRowNum();
        int q = other.getColNum();
        float[][] a = this.data;
        float[][] b = otherData(other);
        float[][] res = new float[m * p][n * q];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float aij = a[i][j];
                for (int k = 0; k < p; k++) {
                    for (int l = 0; l < q; l++) {
                        res[i * p + k][j * q + l] = aij * b[k][l];
                    }
                }
            }
        }
        return IFloatMatrix.of(res);
    }

    /**
     * 矩阵转换为一维数组 / Convert matrix to 1D array
     * <p>
     * 将矩阵转换为一维Float数组。对于列向量，返回该列的数据； 对于普通矩阵，按行优先顺序展开。
     * </p>
     * <p>
     * Converts the matrix to a 1D Float array. For column vectors, returns the
     * column data; For regular matrices, flattens in row-major order.
     * </p>
     *
     * @return 一维Float数组 / 1D Float array
     */
    private float[] toFlattenArray() {
        if (data[0].length == 1) {
            // 如果是列向量，返回该列的数据 / If it's a column vector, return the column data
            float[] result = new float[data.length];
            for (int i = 0; i < data.length; i++) {
                result[i] = data[i][0];
            }
            return result;
        } else {
            // 如果是普通矩阵，按行优先顺序展开 / If it's a regular matrix, flatten in row-major order
            int totalElements = data.length * data[0].length;
            float[] result = new float[totalElements];
            int index = 0;
            for (float[] data1 : data) {
                for (int j = 0; j < data[0].length; j++) {
                    result[index++] = data1[j];
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
    public  IFloatMatrix log() {
        var res = this.computer.universalOperate(data, UniversalOperation.LOG, 0);
        return IFloatMatrix.of(res);
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
        var square = this.computer.universalOperate(data, UniversalOperation.POW, 2.0f);
        float sum = this.computer.reduceOperate(square, ReduceOperation.SUM);
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
    public double frobeniusDistance(IMatrix<Float> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }

        if (data.length != other.getRowNum() || data[0].length != other.getColNum()) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }

        IMatrix<Float> diff = this.sub(other);
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
    public  IFloatMatrix normalizeRows() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];

        // 使用SIMD对每一行进行L2归一化
        for (int i = 0; i < rows; i++) {
            // 计算行的L2范数
            // 先计算每个元素的平方
            float[] squared = this.computer.universalOperate(data[i], UniversalOperation.POW, 2.0f);
            
            // 计算平方和
            float sumOfSquares = this.computer.reduceOperate(squared, ReduceOperation.SUM);
            
            // 计算L2范数
            float norm = (float)Math.sqrt(sumOfSquares);

            if (norm > 0) {
                // 使用SIMD归一化该行
                result[i] = this.computer.binaryOperate(data[i], norm, BinaryOperation.DIVIDE);
            } else {
                System.arraycopy(data[i], 0, result[i], 0, cols);
            }
        }

        return IFloatMatrix.of(result);
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
    public  IFloatMatrix normalizeColumns() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];

        // 使用SIMD对每一列进行L2归一化
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            float[] column = new float[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            
            // 计算列的L2范数
            // 先计算每个元素的平方
            float[] squared = this.computer.universalOperate(column, UniversalOperation.POW, 2.0f);
            
            // 计算平方和
            float sumOfSquares = this.computer.reduceOperate(squared, ReduceOperation.SUM);
            
            // 计算L2范数
            float norm = (float)Math.sqrt(sumOfSquares);

            // 归一化该列
            if (norm > 1e-10) { // 避免除零
                float[] normalizedColumn = this.computer.binaryOperate(column, norm, BinaryOperation.DIVIDE);
                for (int i = 0; i < rows; i++) {
                    result[i][j] = normalizedColumn[i];
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    result[i][j] = 0.0f;
                }
            }
        }

        return IFloatMatrix.of(result);
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
    public  IFloatMatrix center() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];

        // 使用SIMD计算每列的均值
        float[] columnMeans = new float[cols];
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            float[] column = new float[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            
            // 计算列均值
            columnMeans[j] = this.computer.reduceOperate(column, ReduceOperation.MEAN);
        }

        // 使用SIMD对每列减去均值
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            float[] column = new float[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            
            // 使用SIMD减去均值
            float[] centeredColumn = this.computer.binaryOperate(column, columnMeans[j], BinaryOperation.SUBTRACT);
            
            // 将结果放回矩阵
            for (int i = 0; i < rows; i++) {
                result[i][j] = centeredColumn[i];
            }
        }

        return IFloatMatrix.of(result);
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
    public  IFloatMatrix covariance() {
        // 先进行数据中心化
        IFloatMatrix centered = this.center();
        return centered.covarianceFromCentered();
    }

    /**
     * covariance的别名函数
     *
     * @return
     */
    @Override
    public  IFloatMatrix cov() {
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
    public  IFloatMatrix covarianceFromCentered() {
        int n = data.length; // 样本数
        if (n < 2) {
            throw new IllegalArgumentException(
                    "covarianceFromCentered 至少需要 2 个样本（行数≥2），当前 n=" + n
                            + " / At least 2 samples (rows) required, got n=" + n);
        }

        // 计算 X^T * X
        IFloatMatrix transposed = this.transposeNew();
        IFloatMatrix product = ((RereFloatMatrix) transposed).mmul(this);

        // 除以 (n-1) 得到协方差矩阵
        return product.multiplyByScalar(1.0f / (n - 1));
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
    public  IFloatMatrix inv() {
        return MatrixInversionSolver.invert(this.toDoubleMatrix()).toFloatMatrix();
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
    public  IFloatMatrix pinv() {
        return MatrixInversionSolver.pseudoInverse(this.toDoubleMatrix()).toFloatMatrix();
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
     * @return 从文件加载的矩阵对象 / IMatrix<Float> object loaded from file
     * @throws IllegalArgumentException 如果文件路径为null或为空 / if file path is null or
     * empty
     * @throws RuntimeException 如果文件读取失败或格式错误 / if file reading fails or format
     * is incorrect
     */
    public static IMatrix<Float> loadFromFile(String path) {
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
            float[][] data = new float[rows][cols];

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
                        data[i][j] = Float.parseFloat(elements[j]);
                    }
                } catch (NumberFormatException e) {
                    throw new RuntimeException("文件格式错误：第" + (i + 1) + "行包含无效数字 / File format error: invalid number in row " + (i + 1), e);
                }
            }

            return IFloatMatrix.of(data);

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
    public  IFloatMatrix hstack(IMatrix<Float> other1) {
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

        float[][] result = new float[rows][totalCols];

        // 复制第一个矩阵
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, result[i], 0, cols1);
        }

        // 复制第二个矩阵
        IFloatMatrix other0 = (IFloatMatrix) other1;
        float[][] otherData = other0.getData();
        for (int i = 0; i < rows; i++) {
            System.arraycopy(otherData[i], 0, result[i], cols1, cols2);
        }

        return IFloatMatrix.of(result);
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
    public  IFloatMatrix vstack(IMatrix<Float> other1) {
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

        float[][] result = new float[totalRows][cols];

        // 复制第一个矩阵
        for (int i = 0; i < rows1; i++) {
            System.arraycopy(data[i], 0, result[i], 0, cols);
        }

        // 复制第二个矩阵
        IFloatMatrix other0 = (IFloatMatrix) other1;
        float[][] otherData = other0.getData();
        for (int i = 0; i < rows2; i++) {
            System.arraycopy(otherData[i], 0, result[rows1 + i], 0, cols);
        }

        return IFloatMatrix.of(result);
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
    public  IFloatMatrix[] hsplit(int[] indices) {
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
        IFloatMatrix[] result = new IFloatMatrix[numSplits];

        int startCol = 0;
        for (int i = 0; i < numSplits; i++) {
            int endCol = (i < sortedIndices.length) ? sortedIndices[i] : data[0].length;
            int cols = endCol - startCol;

            float[][] subMatrix = new float[data.length][cols];
            for (int row = 0; row < data.length; row++) {
                System.arraycopy(data[row], startCol, subMatrix[row], 0, cols);
            }

            result[i] = IFloatMatrix.of(subMatrix);
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
    public  IFloatMatrix[] vsplit(int[] indices) {
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
        IFloatMatrix[] result = new IFloatMatrix[numSplits];

        int startRow = 0;
        for (int i = 0; i < numSplits; i++) {
            int endRow = (i < sortedIndices.length) ? sortedIndices[i] : data.length;
            int rows = endRow - startRow;

            float[][] subMatrix = new float[rows][data[0].length];
            for (int row = 0; row < rows; row++) {
                System.arraycopy(data[startRow + row], 0, subMatrix[row], 0, data[0].length);
            }

            result[i] = IFloatMatrix.of(subMatrix);
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
    public  IFloatMatrix reshape(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }

        int totalElements = data.length * data[0].length;
        if (rows * cols != totalElements) {
            throw new IllegalArgumentException("新维度与元素总数不匹配 / New dimensions don't match total element count");
        }

        float[] flatData = this.toFlattenArray();
        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = flatData[i * cols + j];
            }
        }

        return IFloatMatrix.of(result);
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
    public  IFloatMatrix copy() {
        int rows = data.length;
        int cols = rows > 0 ? data[0].length : emptyMatrixCols;
        float[][] copyData = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copyData[i], 0, cols);
        }

        RereFloatMatrix out = new RereFloatMatrix(copyData);
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
        return DeterminantSolver.compute(this.toDoubleMatrix());
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
        return ConditionNumberSolver.compute(this.toDoubleMatrix());
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
        return RankSolver.compute(this.toDoubleMatrix());
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
    public  IFloatMatrix abs() {
        var res = this.computer.universalOperate(data, UniversalOperation.ABS, 0);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix sign() {
        var res = this.computer.sign(data);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix sin() {
        var res = this.computer.universalOperate(data, UniversalOperation.SIN, 0);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix cos() {
        var res = this.computer.universalOperate(data, UniversalOperation.COS, 0);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix tan() {
        var res = this.computer.universalOperate(data, UniversalOperation.TAN, 0);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix sinh() {
        var res = this.computer.universalOperate(data, UniversalOperation.SINH, 0);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix cosh() {
        var res = this.computer.universalOperate(data, UniversalOperation.COSH, 0);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix tanh() {
        var res = this.computer.universalOperate(data, UniversalOperation.TANH, 0);
        return IFloatMatrix.of(res);
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
    public Tuple2<IMatrix<Float>, IMatrix<Float>> lu() {
        // 使用新的LU分解实现类
        var tp = Decomps.createLU().decompose(this.toDoubleMatrix());
        return new Tuple2(tp._1.toFloatMatrix(),tp._2.toFloatMatrix());
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
    public  IFloatMatrix cholesky() {
        // 使用新的Cholesky分解实现类
        return Decomps.createCholesky().decompose(this.toDoubleMatrix()).toFloatMatrix();
    }

    @Override
    public Tuple2<IMatrix<Float>, IMatrix<Float>> schur() {
        var tp = Decomps.createSchur().decompose(this.toDoubleMatrix());
        return new Tuple2(tp._1.toFloatMatrix(),tp._2.toFloatMatrix());
    }

    @Override
    public Tuple3<IMatrix<Float>, IMatrix<Float>, IMatrix<Float>> biDiag() {
        var tp = Decomps.createBidiagonal().decompose(this.toDoubleMatrix());
        return new Tuple3(tp._1.toFloatMatrix(),tp._2.toFloatMatrix(),tp._3.toFloatMatrix());
    }

    @Override
    public Tuple2<IMatrix<Float>, IMatrix<Float>> triDiag() {
        var tp = Decomps.createTridiagonal().decompose(this.toDoubleMatrix());
        return new Tuple2(tp._1.toFloatMatrix(),tp._2.toFloatMatrix());
    }

    @Override
    public Tuple2<IMatrix<Float>, IMatrix<Float>> hessenberg() {
        var tp = Decomps.createHessenberg().decompose(this.toDoubleMatrix());
        return new Tuple2(tp._1.toFloatMatrix(),tp._2.toFloatMatrix());
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
    public IVector<Float> solve(IVector<Float> b) {
        return LinearSystemSolver.solve(this.toDoubleMatrix(), b.toDoubleVector()).toFloatVector();
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
    public  IFloatMatrix solve(IMatrix<Float> B) {
        return LinearSystemSolver.solve(this.toDoubleMatrix(), B.toDoubleMatrix()).toFloatMatrix();
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
    public IFloatMatrix sum() {
        float result = this.computer.reduceOperate(data, ReduceOperation.SUM);
        return IFloatMatrix.of(new float[][]{{result}});
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
    public IFloatMatrix mean() {
        float result = this.computer.reduceOperate(data, ReduceOperation.MEAN);
        return IFloatMatrix.of(new float[][]{{result}});
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
    public  IFloatMatrix slice(String rowSlice, String colSlice) {
        if (data.length == 0 || (data.length > 0 && data[0].length == 0)) {
            return IFloatMatrix.of(new float[0][0]);
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
                return new RereFloatMatrix(0, resultCols);
            } else if (resultCols <= 0 && resultRows > 0) {
                return new RereFloatMatrix(resultRows, 0);
            } else {
                return IFloatMatrix.of(new float[0][0]);
            }
        }

        int[] rowIndices = IndexExpressionParser.generateIndices(rowResult);
        int[] colIndices = IndexExpressionParser.generateIndices(colResult);

        float[][] result = new float[resultRows][resultCols];
        for (int i = 0; i < rowIndices.length; i++) {
            for (int j = 0; j < colIndices.length; j++) {
                result[i][j] = data[rowIndices[i]][colIndices[j]];
            }
        }

        return IFloatMatrix.of(result);
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
    public  IFloatMatrix sliceRows(String rowSlice) {
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
    public  IFloatMatrix sliceColumns(String colSlice) {
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
    public  IFloatMatrix fancyGet(int[] rowIndices, int[] colIndices) {
        if (rowIndices == null || colIndices == null) {
            throw new IllegalArgumentException("索引数组不能为null / Index arrays cannot be null");
        }

        IndexExpressionParser.FancyIndexResult resolvedRows =
            IndexExpressionParser.resolveFancyIndex(rowIndices, data.length);
        IndexExpressionParser.FancyIndexResult resolvedCols =
            IndexExpressionParser.resolveFancyIndex(colIndices, data[0].length);

        float[][] result = new float[resolvedRows.indices.length][resolvedCols.indices.length];
        for (int i = 0; i < resolvedRows.indices.length; i++) {
            for (int j = 0; j < resolvedCols.indices.length; j++) {
                result[i][j] = data[resolvedRows.indices[i]][resolvedCols.indices[j]];
            }
        }
        return IFloatMatrix.of(result);
    }

    @Override
    public  IFloatMatrix booleanGet(boolean[] rowMask) {
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
        float[][] result = new float[resolved.count][cols];
        for (int i = 0; i < resolved.trueIndices.length; i++) {
            int row = resolved.trueIndices[i];
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[row][j];
            }
        }
        return IFloatMatrix.of(result);
    }

    @Override
    public  IFloatMatrix booleanGet(boolean[] rowMask, boolean[] colMask) {
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
        float[][] result = new float[resolvedRows.count][resolvedCols.count];
        for (int i = 0; i < resolvedRows.trueIndices.length; i++) {
            for (int j = 0; j < resolvedCols.trueIndices.length; j++) {
                result[i][j] = data[resolvedRows.trueIndices[i]][resolvedCols.trueIndices[j]];
            }
        }
        return IFloatMatrix.of(result);
    }

    @Override
    public void fancySet(int[] rowIndices, int[] colIndices, Float[] values) {
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
    public void fancySetScalar(int[] rowIndices, int[] colIndices, Float value) {
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

    // set() method is now a default method in IMatrix<Float> that delegates to put()
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
    public IVector<Float> diag() {
        int minDim = Math.min(data.length, data[0].length);
        float[] diagonal = new float[minDim];
        for (int i = 0; i < minDim; i++) {
            diagonal[i] = data[i][i];
        }
        return IFloatVector.of(diagonal);
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
    public  IFloatMatrix subMatrix(int startRow, int endRow, int startCol, int endCol) {
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
        float[][] subData = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[actualStartRow + i], actualStartCol, subData[i], 0, cols);
        }

        return IFloatMatrix.of(subData);
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
    public void setSubMatrix(int startRow, int endRow, int startCol, int endCol, IMatrix<Float> subMatrix) {
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
                data[actualStartRow + i][actualStartCol + j] = (float)subMatrix.get(i, j);
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
    public IVector<Float> rowSums() {
        int rows = data.length;
        float[] sums = new float[rows];
        
        // 使用SIMD计算每行的和
        for (int i = 0; i < rows; i++) {
            sums[i] = this.computer.reduceOperate(data[i], ReduceOperation.SUM);
        }
        
        return IFloatVector.of(sums);
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
    public IVector<Float> rowMeans() {
        int rows = data.length;
        int cols = data[0].length;
        float[] means = new float[rows];
        
        // 使用SIMD计算每行的均值
        for (int i = 0; i < rows; i++) {
            means[i] = this.computer.reduceOperate(data[i], ReduceOperation.MEAN);
        }
        
        return IFloatVector.of(means);
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
    public IVector<Float> colSums() {
        int rows = data.length;
        int cols = data[0].length;
        float[] sums = new float[cols];
        
        // 使用SIMD计算每列的和
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            float[] column = new float[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            // 计算列和
            sums[j] = this.computer.reduceOperate(column, ReduceOperation.SUM);
        }
        
        return IFloatVector.of(sums);
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
    public IVector<Float> colMeans() {
        int rows = data.length;
        int cols = data[0].length;
        float[] means = new float[cols];
        
        // 使用SIMD计算每列的均值
        for (int j = 0; j < cols; j++) {
            // 获取列向量
            float[] column = new float[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            
            // 计算列均值
            means[j] = this.computer.reduceOperate(column, ReduceOperation.MEAN);
        }
        
        return IFloatVector.of(means);
    }

    @Override
    public IVector<Float> min(int axis) {
        if (axis == 0) {
            return colMins();
        } else if (axis == 1) {
            return rowMins();
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public IVector<Float> max(int axis) {
        if (axis == 0) {
            return colMaxs();
        } else if (axis == 1) {
            return rowMaxs();
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public IVector<Float> sum(int axis) {
        if (axis == 0) {
            return colSums();
        } else if (axis == 1) {
            return rowSums();
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public IVector<Float> mean(int axis) {
        if (axis == 0) {
            return colMeans();
        } else if (axis == 1) {
            return rowMeans();
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public IVector<Float> rowMins() {
        int rows = data.length;
        float[] mins = new float[rows];
        for (int i = 0; i < rows; i++) {
            mins[i] = this.computer.reduceOperate(data[i], ReduceOperation.MIN);
        }
        return IFloatVector.of(mins);
    }

    @Override
    public IVector<Float> rowMaxs() {
        int rows = data.length;
        float[] maxs = new float[rows];
        for (int i = 0; i < rows; i++) {
            maxs[i] = this.computer.reduceOperate(data[i], ReduceOperation.MAX);
        }
        return IFloatVector.of(maxs);
    }

    @Override
    public IVector<Float> colMins() {
        int rows = data.length;
        int cols = data[0].length;
        float[] mins = new float[cols];
        for (int j = 0; j < cols; j++) {
            float[] column = new float[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            mins[j] = this.computer.reduceOperate(column, ReduceOperation.MIN);
        }
        return IFloatVector.of(mins);
    }

    @Override
    public IVector<Float> colMaxs() {
        int rows = data.length;
        int cols = data[0].length;
        float[] maxs = new float[cols];
        for (int j = 0; j < cols; j++) {
            float[] column = new float[rows];
            for (int i = 0; i < rows; i++) {
                column[i] = data[i][j];
            }
            maxs[j] = this.computer.reduceOperate(column, ReduceOperation.MAX);
        }
        return IFloatVector.of(maxs);
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
    public  IFloatMatrix sqrt() {
        var res = this.computer.universalOperate(data, UniversalOperation.SQRT, 0);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix pow(Float power) {
        var res = this.computer.universalOperate(data, UniversalOperation.POW, power);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix exp() {
        var res = this.computer.universalOperate(data, UniversalOperation.EXP, 0);
        return IFloatMatrix.of(res);
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
    public  IFloatMatrix getColumnMatrix(int colIndex) {
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
            float[][] L = new float[n][n];

            for (int i = 0; i < n; i++) {
                for (int j = 0; j <= i; j++) {
                    if (i == j) {
                        // 对角元素
                        float sum = 0.0f;
                        for (int k = 0; k < j; k++) {
                            sum += L[j][k] * L[j][k];
                        }
                        float diagonal = data[j][j] - sum;

                        // 如果对角元素非正，则矩阵不是正定的
                        if (RerePrecision.equalsZero(diagonal, 1e-12) || diagonal < 0.0) {
                            return false;
                        }

                        L[j][j] = (float)Math.sqrt(diagonal);
                    } else {
                        // 下三角元素
                        float sum = 0.0f;
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
    public IVector<Float> flatten() {
        var as = this.toFlattenArray();
        return IFloatVector.of(as);
    }

    @Override
    public float[][] toFloatArray() {
        int rows = data.length;
        if (rows == 0) {
            return new float[0][];
        }
        int cols = data[0].length;
        float[][] copy = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copy[i], 0, cols);
        }
        return copy;
    }

    @Override
    public double[][] toDoubleArray() {
        // 创建新的二维float数组
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];

        // 将float转换为float
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) data[i][j];
            }
        }

        return result;
    }

    @Override
    public int[][] toIntArray() {
        // 创建新的二维float数组
        int rows = data.length;
        int cols = data[0].length;
        int[][] result = new int[rows][cols];

        // 将float转换为float
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
    public  IFloatMatrix getColumnAsCloumnVector(int i) {
        return (IFloatMatrix)this.getColumn(i).asColumnVector();

    }

    @Override
    public void setColumn(int colIndex, IVector<Float> column) {
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
        float[] columnData = RereFloatVector.otherData(column);

        // 设置指定列的数据
        for (int i = 0; i < data.length; i++) {
            data[i][colIndex] = columnData[i];
        }
    }

    @Override
    public void setRow(int rowIndex, IVector<Float> row) {
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
        float[] rowData = RereFloatVector.otherData(row);

        // 设置指定行的数据
        System.arraycopy(rowData, 0, data[rowIndex], 0, data[0].length);
    }

    @Override
    public IVector<Float> apply(Function<IVector<Float>, Float> fun, int axis) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }
        if (axis == 0) {
            // JIT EA 可在此处内联并标量替换 colData 数组
            int rows = data.length;
            int cols = data[0].length;
            float[] result = new float[cols];
            for (int j = 0; j < cols; j++) {
                float[] colData = new float[rows];
                for (int i = 0; i < rows; i++) {
                    colData[i] = data[i][j];
                }
                result[j] = fun.apply(IFloatVector.of(colData));
            }
            return IFloatVector.of(result);
        } else if (axis == 1) {
            // axis=1 时直接引用 data[i]，避免 getRow() 的间接调用
            float[] result = new float[rows()];
            for (int i = 0; i < rows(); i++) {
                result[i] = fun.apply(IFloatVector.of(data[i]));
            }
            return IFloatVector.of(result);
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    @Override
    public  IFloatMatrix applyMap(Function<Float, Float> fun) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }

        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            // JIT EA 可在简单 lambda 下将 srcRow/dstRow 标量替换为寄存器变量
            float[] srcRow = data[i];
            float[] dstRow = result[i];
            for (int j = 0; j < cols; j++) {
                // fun.apply 会触发 float→Float 装箱，但 JIT 会对 x->x*2 等简单 lambda
                // 做逃逸分析并内联，随后将其标量替换，使装箱对象不复存在
                dstRow[j] = fun.apply(srcRow[j]);
            }
        }

        // 直接构造而非工厂方法：绕过类型推断，零开销
        return new RereFloatMatrix(result);
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
    public IVector<Float> mmul(IVector<Float> other) {
        var m2 = other.asColumnMatrix();
        var res = this.computer.mmul(data, otherData(m2));
        // Extract the column vector from the result matrix
        float[] resultVector = new float[res.length];
        for (int i = 0; i < res.length; i++) {
            resultVector[i] = res[i][0];
        }
        return IFloatVector.of(resultVector);
    }

    @Override
    public  IFloatMatrix divideByScalar(double scalar) {
        if (RerePrecision.equalsZero(scalar, 1e-12)) {
            throw new ArithmeticException("除数不能为零 / Divisor cannot be zero");
        }
        var res = this.computer.binaryOperate(data, (float)scalar, BinaryOperation.DIVIDE);
        return IFloatMatrix.of(res);
    }


    @Override
    public  IFloatMatrix normalize() {
        int rows = this.data.length;
        int cols = this.data[0].length;
        float[][] result = new float[rows][cols];

        // 使用SIMD计算每行的L2范数
        for (int i = 0; i < rows; i++) {
            // 计算每个元素的平方
            float[] squared = this.computer.universalOperate(this.data[i], UniversalOperation.POW, 2.0f);
            
            // 计算平方和
            float sumOfSquares = this.computer.reduceOperate(squared, ReduceOperation.SUM);
            
            // 计算L2范数
            float norm = (float)Math.sqrt(sumOfSquares);

            if (RerePrecision.equalsZero(norm, 1e-12)) {
                throw new ArithmeticException("第" + i + "行的L2范数为零，无法归一化 / Row " + i + " L2 norm is zero, cannot normalize");
            }

            // 使用SIMD归一化每行
            result[i] = this.computer.binaryOperate(this.data[i], norm, BinaryOperation.DIVIDE);
        }

        return IFloatMatrix.of(result);
    }

    @Override
    public void setDiag(IVector<Float> diagonal) {
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
            data[i][i] = (float)diagonal.get(i);
        }
    }

    @Override
    public  IFloatMatrix broadcastColumn(IVector<Float> colVector, BiFunction<IVector<Float>, IVector<Float>, IVector<Float>> fun) {
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

        float[][] result = new float[rows][cols];

        for (int j = 0; j < cols; j++) {
            // 获取当前列
            float[] currentCol = new float[rows];
            for (int i = 0; i < rows; i++) {
                currentCol[i] = this.data[i][j];
            }
            IFloatVector currentColVector = IFloatVector.of(currentCol);

            // 应用函数
            IVector<Float> resultCol = fun.apply(currentColVector, colVector);

            // 将结果放回矩阵
            for (int i = 0; i < rows; i++) {
                result[i][j] = (float)resultCol.get(i);
            }
        }

        return IFloatMatrix.of(result);
    }

    @Override
    public  IFloatMatrix broadcastRow(IVector<Float> rowVector, BiFunction<IVector<Float>, IVector<Float>, IVector<Float>> fun) {
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

        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            // 获取当前行
            IFloatVector currentRowVector = IFloatVector.of(this.data[i]);

            // 应用函数
            IVector<Float> resultRow = fun.apply(currentRowVector, rowVector);

            // 将结果放回矩阵
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float)resultRow.get(j);
            }
        }

        return IFloatMatrix.of(result);
    }

    @Override
    public  IFloatMatrix multiply(IMatrix<Float> other) {
        var res = this.computer.binaryOperate(data, otherData(other), BinaryOperation.MULTIPLY);
        return IFloatMatrix.of(res);
    }

    @Override
    public boolean[][] eq(IMatrix<Float> other) {
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
    public boolean[][] lt(IMatrix<Float> other) {
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
    public boolean[][] gt(IMatrix<Float> other) {
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
    public boolean[][] ge(IMatrix<Float> other) {
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
    public boolean[][] le(IMatrix<Float> other) {
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
    public  IFloatMatrix forwardSolve(IMatrix<Float> B) {
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
        float[][] X = new float[n][nrhs];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                X[i][j] = (float)B.get(i, j);
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

        return IFloatMatrix.of(X);
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
    public  IFloatMatrix backwardSolve(IMatrix<Float> B) {
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
        float[][] X = new float[n][nrhs];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                X[i][j] = (float)B.get(i, j);
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

        return IFloatMatrix.of(X);
    }


}
