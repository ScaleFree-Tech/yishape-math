package com.yishape.lab.math.linalg;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.random.RngFactory;
import com.yishape.lab.math.random.RngProvider;

import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;

/**
 * Double类型矩阵操作接口 / Double Matrix Operations Interface
 * <p>
 * 本接口定义了Double类型矩阵的常用操作，包括基本的数学运算、矩阵变换、数据访问等功能。
 * 这是IDoubleMatrixGeneric<Double>的别名，保持向后兼容性。
 * </p>
 * <p>
 * This interface defines common matrix operations for Double type matrices,
 * including basic mathematical operations, matrix transformations, data access
 * and other functionalities. This is an alias for IDoubleMatrixGeneric<Double> to
 * maintain backward compatibility.
 * </p>
 * <p>
 * <strong>数组导出：</strong>{@link IMatrix#toDoubleArray()} 在 {@link RereDoubleMatrix} 等实现中返回行主序数据的<strong>防御性拷贝</strong>。需要与内部共享且可写的后备数组请使用 {@link #getData()}。
 * </p>
 *
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 * <li>基本数学运算：加法、减法、乘法、除法 / Basic math operations: add, subtract, multiply,
 * divide</li>
 * <li>矩阵变换：转置、幂运算、开方 / IDoubleMatrix transformations: transpose, power, square
 root</li>
 * <li>数据访问：行列访问、元素获取设置 / Data access: row/column access, element get/set</li>
 * <li>统计运算：行列求和、均值计算 / Statistical operations: row/column sum, mean
 * calculation</li>
 * <li>工厂方法：创建随机矩阵、单位矩阵、零矩阵 / Factory methods: create random, ones, zeros
 * matrices</li>
 * <li>数据转换：数组转换、类型转换 / Data conversion: array conversion, type conversion</li>
 * </ul>
 *
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 // 创建矩阵 / Create matrices
 IDoubleMatrix matrix1 = IDoubleMatrix.ones(3, 3);
 IDoubleMatrix matrix2 = IDoubleMatrix.rand(3, 3);
 double[][] data = {{1, 2}, {3, 4}};
 IDoubleMatrix matrix3 = IDoubleMatrix.of(data);

 // 矩阵运算 / IDoubleMatrix operations
 IDoubleMatrix result = matrix1.add(matrix2).mmul(2.0f);

 // 获取行列 / Get rows/columns
 IDoubleVector row = matrix3.getRow(0);
 IDoubleVector col = matrix3.getColunm(0);

 // 矩阵转置 / IDoubleMatrix transpose
 IDoubleMatrix transposed = matrix3.transpose();
 }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IDoubleMatrix extends IMatrix<Double> {

    /**
     * 矩阵工厂方法 / Matrix factory method
     * <p>
     * 使用给定的二维数组创建矩阵实例 Creates a matrix instance with the given 2D array
     * </p>
     *
     * @param data 二维Double数组，表示矩阵数据 / 2D Double array representing matrix data
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null或维度不一致 / if data is null or
     * dimensions are inconsistent
     */
    public static IDoubleMatrix of(double[][] data) {
        return new RereDoubleMatrix(data);
    }

    public static IDoubleMatrix of(Double[][] data) {
        return new RereDoubleMatrix(data);
    }

    /**
     * 矩阵工厂方法（从List创建） / Matrix factory method (from List)
     * <p>
     * 使用给定的Double数组列表创建矩阵实例，每个数组代表矩阵的一行 Creates a matrix instance with the given
     * list of Double arrays, each array representing a row of the matrix
     * </p>
     *
     * @param data 包含Double数组的列表，每个数组表示矩阵的一行 / List containing Double arrays, each
     * array representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空列表或行长度不一致 / if data is null,
     * empty list, or row lengths are inconsistent
     */
    public static IDoubleMatrix of(List<double[]> data) {
        if (data == null) {
            throw new IllegalArgumentException("数据列表不能为null / Data list cannot be null");
        }

        if (data.isEmpty()) {
            throw new IllegalArgumentException("数据列表不能为空 / Data list cannot be empty");
        }

        int rows = data.size();
        int cols = data.get(0).length;

        // 检查所有行的长度是否一致 / Check if all rows have consistent length
        for (int i = 0; i < rows; i++) {
            if (data.get(i) == null) {
                throw new IllegalArgumentException("第" + i + "行数据不能为null / Row " + i + " data cannot be null");
            }
            if (data.get(i).length != cols) {
                throw new IllegalArgumentException("所有行的长度必须一致 / All rows must have consistent length");
            }
        }

        // 转换为二维数组 / Convert to 2D array
        double[][] array = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data.get(i), 0, array[i], 0, cols);
        }

        return new RereDoubleMatrix(array);
    }

    /**
     * 矩阵工厂方法（从Vector数组创建） / Matrix factory method (from Vector array)
     * <p>
 使用给定的Vector数组创建矩阵实例，每个Vector代表矩阵的一行 Creates a matrix instance with the
 given IDoubleVector array, each IDoubleVector representing a row of the
 matrix
 </p>
     *
     * @param data 包含Vector的数组，每个Vector表示矩阵的一行 / Array containing Vectors, each
 IDoubleVector representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空数组或行长度不一致 / if data is null,
     * empty array, or row lengths are inconsistent
     */
    public static IDoubleMatrix of(IDoubleVector[] data) {
        if (data == null) {
            throw new IllegalArgumentException("数据数组不能为null / Data array cannot be null");
        }

        if (data.length == 0) {
            throw new IllegalArgumentException("数据数组不能为空 / Data array cannot be empty");
        }

        int rows = data.length;
        int cols = data[0].getData().length;

        // 检查所有行的长度是否一致 / Check if all rows have consistent length
        for (int i = 0; i < rows; i++) {
            if (data[i] == null) {
                throw new IllegalArgumentException("第" + i + "行Vector不能为null / Row " + i + " Vector cannot be null");
            }
            if (data[i].getData().length != cols) {
                throw new IllegalArgumentException("所有Vector的长度必须一致 / All Vectors must have consistent length");
            }
        }

        // 转换为二维数组 / Convert to 2D array
        double[][] array = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            double[] rowData = data[i].getData();
            System.arraycopy(rowData, 0, array[i], 0, cols);
        }

        return new RereDoubleMatrix(array);
    }

    public static IDoubleMatrix fromArray(double[] data, int rows, int cols) {
        if (data == null) {
            throw new IllegalArgumentException("数据数组不能为null / Data array cannot be null");
        }
        if (data.length != rows * cols) {
            throw new IllegalArgumentException("数组长度必须等于行数×列数 / Array length must equal rows × columns");
        }
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }

        double[][] matrixData = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixData[i][j] = data[i * cols + j];
            }
        }
        return new RereDoubleMatrix(matrixData);
    }

    public static IDoubleMatrix fromArray(Double[] data, int rows, int cols) {
        return fromArray(RereMathUtil.toPrimitive(data), rows, cols);
    }

    /**
     * 创建随机矩阵 / Create random matrix
     * <p>
     * 创建一个指定大小的随机矩阵，元素值服从均匀分布 Creates a random matrix of specified size with
     * elements following uniform distribution
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    public static IDoubleMatrix rand(int rows, int cols) {
        return rand(rows, cols, RngFactory.createDefault());
    }

    public static IDoubleMatrix rand(int rows, int cols, long seed) {
        return rand(rows, cols, RngFactory.createSeeded(seed));
    }

    public static IDoubleMatrix rand(int rows, int cols, RngProvider rng) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rng.nextDouble();
            }
        }
        return new RereDoubleMatrix(data);
    }

    public static IDoubleMatrix randn(int rows, int cols) {
        return randn(rows, cols, RngFactory.createDefault());
    }

    public static IDoubleMatrix randn(int rows, int cols, long seed) {
        return randn(rows, cols, RngFactory.createSeeded(seed));
    }

    public static IDoubleMatrix randn(int rows, int cols, RngProvider rng) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rng.nextGaussian();
            }
        }
        return new RereDoubleMatrix(data);
    }

    public static IDoubleMatrix randn(int rows, int cols, Double mean, Double std) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        if (std < 0) {
            throw new IllegalArgumentException("标准差不能为负数 / Standard deviation cannot be negative");
        }
        RngProvider rng = RngFactory.createDefault();
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = mean + std * rng.nextGaussian();
            }
        }
        return new RereDoubleMatrix(data);
    }

    public static IDoubleMatrix randn(int rows, int cols, Double mean, Double std, RngProvider rng) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        if (std < 0) {
            throw new IllegalArgumentException("标准差不能为负数 / Standard deviation cannot be negative");
        }
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = mean + std * rng.nextGaussian();
            }
        }
        return new RereDoubleMatrix(data);
    }

    /**
     * 创建全1矩阵 / Create ones matrix
     * <p>
     * 创建一个指定大小的矩阵，所有元素都为1 Creates a matrix of specified size with all elements
     * set to 1
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 全1矩阵 / Ones matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    public static IDoubleMatrix ones(int rows, int cols) {
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = 1.0;
            }
        }
        return new RereDoubleMatrix(data);
    }

    /**
     * 创建零矩阵 / Create zeros matrix
     * <p>
     * 创建一个指定大小的矩阵，所有元素都为0 Creates a matrix of specified size with all elements
     * set to 0
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 零矩阵 / Zeros matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    public static IDoubleMatrix zeros(int rows, int cols) {
        double[][] data = new double[rows][cols];
        // Java数组默认初始化为0，所以不需要显式设置 / Java arrays are initialized to 0 by default
        return new RereDoubleMatrix(data);
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
     * @return 平均值矩阵 / Average matrix
     * @throws IllegalArgumentException 如果数组长度不匹配 / if array lengths don't match
     */
    public static IDoubleMatrix average(IDoubleMatrix[] a, IDoubleMatrix[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("数组长度不匹配 / Array lengths don't match");
        }

        // 假设所有矩阵都是列向量，我们需要将它们合并成一个矩阵
        // Assuming all matrices are column vectors, we need to combine them into one matrix
        int rows = a[0].getRowNum();
        int cols = a.length;

        IDoubleMatrix matrixA = IDoubleMatrix.zeros(rows, cols);
        IDoubleMatrix matrixB = IDoubleMatrix.zeros(rows, cols);

        // 将列向量组合成矩阵 / Combine column vectors into matrices
        for (int i = 0; i < cols; i++) {
            matrixA.putColumn(i, a[i]);
            matrixB.putColumn(i, b[i]);
        }

        // 计算平均值 / Calculate average
        IDoubleMatrix result = IDoubleMatrix.zeros(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.put(row, col, (matrixA.get(row, col) + matrixB.get(row, col)) / 2);
            }
        }

        return result;
    }

    // ========== IDoubleMatrix 特有方法 / IDoubleMatrix-specific Methods ==========
    // Note: 大多数方法现在在 IMatrix 中定义，这里保留类型特化的方法
    // Note: Most methods are now defined in IMatrix, keeping type-specific methods here


    /**
     * 从本地指定位置path加载恢复矩阵 / Load matrix from specified local path
     */
    public static IMatrix load(String path) {
        return RereDoubleMatrix.loadFromFile(path);
    }

    /**
     * 创建单位矩阵 / Create identity matrix
     * <p>
     * 创建一个指定大小的单位矩阵（对角线上元素为1，其他元素为0） Creates an identity matrix of specified
     * size (diagonal elements are 1, others are 0)
     * </p>
     *
     * @param size 矩阵大小（行数和列数相同） / Matrix size (rows and columns are the same)
     * @return 单位矩阵 / Identity matrix
     * @throws IllegalArgumentException 如果大小小于等于0 / if size is less than or
     * equal to 0
     */
    public static IDoubleMatrix eye(int size) {
        return eye(size, size);
    }

    /**
     * 创建单位矩阵 / Create identity matrix
     * <p>
     * 创建一个指定大小的单位矩阵（对角线上元素为1，其他元素为0） Creates an identity matrix of specified
     * size (diagonal elements are 1, others are 0)
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 单位矩阵 / Identity matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    public static IDoubleMatrix eye(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }

        double[][] data = new double[rows][cols];
        int minDim = Math.min(rows, cols);
        for (int i = 0; i < minDim; i++) {
            data[i][i] = 1.0;
        }
        return new RereDoubleMatrix(data);
    }

    /**
     * 创建对角矩阵 / Create diagonal matrix
     * <p>
     * 从给定的对角线元素创建对角矩阵 Creates a diagonal matrix from the given diagonal
     * elements
     * </p>
     *
     * @param diagonal 对角线元素数组 / Array of diagonal elements
     * @return 对角矩阵 / Diagonal matrix
     * @throws IllegalArgumentException 如果对角线数组为null或空 / if diagonal array is
     * null or empty
     */
    public static IDoubleMatrix diag(double[] diagonal) {
        if (diagonal == null || diagonal.length == 0) {
            throw new IllegalArgumentException("对角线数组不能为null或空 / Diagonal array cannot be null or empty");
        }

        int size = diagonal.length;
        double[][] data = new double[size][size];
        for (int i = 0; i < size; i++) {
            data[i][i] = diagonal[i];
        }
        return new RereDoubleMatrix(data);
    }

    public static IDoubleMatrix diag(Double[] diagonal) {
        return diag(RereMathUtil.toPrimitive(diagonal));
    }

    public static IDoubleMatrix diag(IDoubleVector diagonal) {
        return diag(diagonal.getData());
    }

    /** 两矩阵广播后的形状；见 {@link IMatrix#broadcastShape(IMatrix, IMatrix)}。 */
    static int[] broadcastShape(IMatrix<Double> a, IMatrix<Double> b) {
        return IMatrix.broadcastShape(a, b);
    }

    static int[] broadcastShape(int rowsA, int colsA, int rowsB, int colsB) {
        return IMatrix.broadcastShape(rowsA, colsA, rowsB, colsB);
    }

    static double[][] broadcastTo(double[][] data, int targetRows, int targetCols) {
        return IMatrix.broadcastTo(data, targetRows, targetCols);
    }

    static IDoubleMatrix broadcastTo(IDoubleMatrix data, int targetRows, int targetCols) {
        return IMatrix.broadcastTo(data, targetRows, targetCols);
    }

    static double[][] broadcastElementWise(double[][] a, double[][] b, DoubleBinaryOperator op) {
        return IMatrix.broadcastElementWise(a, b, op);
    }

    static IDoubleMatrix broadcastElementWise(IMatrix<Double> a, IMatrix<Double> b, DoubleBinaryOperator op) {
        return IMatrix.broadcastElementWise(a, b, op);
    }

    // Note: copy(), max(), min(), sum(), mean() are inherited from IMatrix<Double>
    // Note: All matrix analysis, mathematical functions, statistical operations,
    // decomposition methods, and linear system solving are now inherited from IMatrix<Double>
    // This includes: det(), trace(), cond(), rank(), abs(), sign(), sin(), cos(), tan(),
    // sinh(), cosh(), tanh(), std(), var(), lu(), cholesky(), solve(), etc.
    // Note: subMatrix() and setSubMatrix() are now inherited from IMatrix<Double>
    // Note: set() method is now inherited from IMatrix<Double> as default method
    // 工厂方法将在具体实现类中定义
    // ========== 抽象方法定义 / Abstract Method Definitions ==========
    /**
     * 获取矩阵数据数组 / Get matrix data array
     * <p>
     * 返回矩阵的内部数据数组引用 Returns a reference to the internal data array of the
     * matrix
     * </p>
     *
     * @return 矩阵的二维数组数据 / 2D array data of the matrix
     */
    public double[][] getData();

    // ========== 协变返回类型覆盖 / Covariant Return Type Overrides ==========
    @Override
    public IDoubleMatrix add(IMatrix<Double> other);
    @Override
    public IDoubleMatrix sub(IMatrix<Double> other);
    @Override
    public IDoubleMatrix sub(double scalar);
    @Override
    public IDoubleMatrix multiplyByScalar(double scalar);
    @Override
    public IDoubleMatrix divide(IMatrix<Double> other);
    @Override
    public IDoubleMatrix divideByScalar(double scalar);
    @Override
    public IDoubleMatrix mmul(IMatrix<Double> other);
    @Override
    public IDoubleMatrix copy();
    @Override
    public IDoubleMatrix transposeInPlace();
    @Override
    public IDoubleMatrix transposeNew();
    @Override
    public IDoubleMatrix inv();
    @Override
    public IDoubleMatrix pinv();
    @Override
    public IDoubleMatrix getColumnAsCloumnVector(int i);
    @Override
    public IDoubleMatrix getColumnMatrix(int colIndex);
    @Override
    public IDoubleMatrix sqrt();
    @Override
    public IDoubleMatrix pow(Double power);
    @Override
    public IDoubleMatrix exp();
    @Override
    public IDoubleMatrix log();
    @Override
    public IDoubleMatrix normalizeRows();
    @Override
    public IDoubleMatrix normalizeColumns();
    @Override
    public IDoubleMatrix normalize();
    @Override
    public IDoubleMatrix center();
    @Override
    public IDoubleMatrix covariance();
    @Override
    public IDoubleMatrix cov();
    @Override
    public IDoubleMatrix covarianceFromCentered();
    @Override
    public IDoubleMatrix hstack(IMatrix<Double> other);
    @Override
    public IDoubleMatrix vstack(IMatrix<Double> other);
    @Override
    public IDoubleMatrix reshape(int rows, int cols);
    @Override
    public IDoubleMatrix abs();
    @Override
    public IDoubleMatrix sign();
    @Override
    public IDoubleMatrix sin();
    @Override
    public IDoubleMatrix cos();
    @Override
    public IDoubleMatrix tan();
    @Override
    public IDoubleMatrix sinh();
    @Override
    public IDoubleMatrix cosh();
    @Override
    public IDoubleMatrix tanh();
    @Override
    public IDoubleMatrix cholesky();
    @Override
    public IDoubleMatrix solve(IMatrix<Double> B);
    @Override
    public IDoubleMatrix slice(String rowSlice, String colSlice);
    @Override
    public IDoubleMatrix sliceRows(String rowSlice);
    @Override
    public IDoubleMatrix sliceColumns(String colSlice);
    @Override
    public IDoubleMatrix fancyGet(int[] rowIndices, int[] colIndices);
    @Override
    public IDoubleMatrix booleanGet(boolean[] rowMask);
    @Override
    public IDoubleMatrix booleanGet(boolean[] rowMask, boolean[] colMask);
    @Override
    public IDoubleMatrix subMatrix(int startRow, int endRow, int startCol, int endCol);
    @Override
    public IDoubleMatrix applyMap(java.util.function.Function<Double, Double> fun);
    @Override
    public IDoubleMatrix broadcastColumn(IVector<Double> colVector, java.util.function.BiFunction<IVector<Double>, IVector<Double>, IVector<Double>> fun);
    @Override
    public IDoubleMatrix broadcastRow(IVector<Double> rowVector, java.util.function.BiFunction<IVector<Double>, IVector<Double>, IVector<Double>> fun);
    @Override
    public IDoubleMatrix multiply(IMatrix<Double> other);
    @Override
    public IDoubleMatrix sum();
    @Override
    public IDoubleMatrix mean();

    default IDoubleMatrix multiply(ISparseMatrix sparse) {
        return (IDoubleMatrix) sparse.multiplyDenseFromLeft((IMatrix<Double>) this);
    }

    // ========== 无装箱原语重载 / Boxing-free primitive overloads ==========

    /**
     * 对每个元素应用给定运算符（无装箱）。与 {@link IMatrix#applyMap(Function)} 不同，此方法使用
     * {@code double → double} 原语操作，避免 Double 装箱/拆箱开销。
     *
     * @param fun 要应用的双精度运算符
     * @return 新矩阵
     */
    default IMatrix<Double> applyMap(DoubleUnaryOperator fun) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }
        double[][] data = getData();
        int rows = data.length;
        int cols = data[0].length;
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = fun.applyAsDouble(data[i][j]);
            }
        }
        return IDoubleMatrix.of(result);
    }

    /**
     * 对每行/列应用聚合函数（无装箱）。与 {@link IMatrix#apply(Function, int)} 不同，此方法使用
     * {@code double[] → double} 原语操作，避免 IVector/Double 装箱开销。
     * <p>
     * <strong>使用示例：</strong>
     * <pre>{@code
     * IDoubleMatrix m = IDoubleMatrix.rand(3, 4);
     * IVector<Double> rowSums = m.reduce(row -> {
     *     double s = 0;
     *     for (double v : row) s += v;
     *     return s;
     * }, 1);
     * }</pre>
     *
     * @param fun 要应用的函数，接受 double[] 并返回标量 double
     * @param axis 轴: 0=列, 1=行
     * @return 聚合结果向量
     */
    default IVector<Double> reduce(ToDoubleFunction<double[]> fun, int axis) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }
        double[][] data = getData();
        if (axis == 0) {
            int rows = data.length;
            int cols = data[0].length;
            double[] result = new double[cols];
            for (int j = 0; j < cols; j++) {
                double[] colData = new double[rows];
                for (int i = 0; i < rows; i++) {
                    colData[i] = data[i][j];
                }
                result[j] = fun.applyAsDouble(colData);
            }
            return IDoubleVector.of(result);
        } else if (axis == 1) {
            double[] result = new double[data.length];
            for (int i = 0; i < data.length; i++) {
                result[i] = fun.applyAsDouble(data[i]);
            }
            return IDoubleVector.of(result);
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    /**
     * 按列应用聚合函数（无装箱，等同于 {@code reduce(fun, 0)}）。
     *
     * @param fun 要应用的函数
     * @return 聚合结果向量
     */
    default IVector<Double> reduce(ToDoubleFunction<double[]> fun) {
        return reduce(fun, 0);
    }

    /**
     * {@inheritDoc}
     * @return {@link IDoubleMatrix} 实例
     */
    @Override
    default IDoubleMatrix addColumn(IVector<Double> colVector) {
        requireDoubleVector(colVector, "列向量 / column vector");
        return (IDoubleMatrix) IMatrix.super.addColumn(colVector);
    }

    /**
     * 在矩阵右侧追加一列（{@link IDoubleVector} 便捷重载）。
     */
    default IDoubleMatrix addColumn(IDoubleVector colVector) {
        return addColumn((IVector<Double>) colVector);
    }

    /**
     * {@inheritDoc}
     * @return {@link IDoubleMatrix} 实例
     */
    @Override
    default IDoubleMatrix addRow(IVector<Double> rowVector) {
        requireDoubleVector(rowVector, "行向量 / row vector");
        return (IDoubleMatrix) IMatrix.super.addRow(rowVector);
    }

    /**
     * 在矩阵底部追加一行（{@link IDoubleVector} 便捷重载）。
     */
    default IDoubleMatrix addRow(IDoubleVector rowVector) {
        return addRow((IVector<Double>) rowVector);
    }

    private static void requireDoubleVector(IVector<?> vector, String label) {
        if (vector instanceof IFloatVector) {
            throw new IllegalArgumentException(
                    label + " 必须为 Double 类型，不能传入 Float 向量 / "
                            + label + " must be Double type, Float vector is not allowed");
        }
    }
}
