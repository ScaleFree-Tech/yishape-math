package com.yishape.lab.math.linalg;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.random.RngFactory;
import com.yishape.lab.math.random.RngProvider;

import java.util.List;
import java.util.Objects;
import java.util.function.DoubleBinaryOperator;

/**
 * Float类型矩阵操作接口 / Float Matrix Operations Interface
 * <p>
 * 本接口定义了Float类型矩阵的常用操作，包括基本的数学运算、矩阵变换、数据访问等功能。
 * 这是IFloatMatrixGeneric<Float>的别名，保持向后兼容性。
 * </p>
 * <p>
 * This interface defines common matrix operations for Float type matrices,
 * including basic mathematical operations, matrix transformations, data access
 * and other functionalities. This is an alias for IFloatMatrixGeneric<Float> to
 * maintain backward compatibility.
 * </p>
 * <p>
 * <strong>数组导出：</strong>{@link IMatrix#toFloatArray()} 在
 * {@link RereFloatMatrix}
 * 等实现中返回行主序数据的<strong>防御性拷贝</strong>（{@link IMatrix#toDoubleArray()}
 * 亦为新分配）。需要与内部共享且可写的后备数组请使用 {@link #getData()}。
 * </p>
 *
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 * <li>基本数学运算：加法、减法、乘法、除法 / Basic math operations: add, subtract, multiply,
 * divide</li>
 * <li>矩阵变换：转置、幂运算、开方 / IFloatMatrix transformations: transpose, power, square
 * root</li>
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
 * // 创建矩阵 / Create matrices
 * IFloatMatrix matrix1 = IFloatMatrix.ones(3, 3);
 * IFloatMatrix matrix2 = IFloatMatrix.rand(3, 3);
 * float[][] data = {{1, 2}, {3, 4}};
 * IFloatMatrix matrix3 = IFloatMatrix.of(data);
 *
 * // 矩阵运算 / IFloatMatrix operations
 * IFloatMatrix result = matrix1.add(matrix2).mmul(2.0f);
 *
 * // 获取行列 / Get rows/columns
 * IFloatVector row = matrix3.getRow(0);
 * IFloatVector col = matrix3.getColunm(0);
 *
 * // 矩阵转置 / IFloatMatrix transpose
 * IFloatMatrix transposed = matrix3.transpose();
 * }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IFloatMatrix extends IMatrix<Float> {

    /**
     * 矩阵工厂方法 / Matrix factory method
     * <p>
     * 使用给定的二维数组创建矩阵实例 Creates a matrix instance with the given 2D array
     * </p>
     *
     * @param data 二维Float数组，表示矩阵数据 / 2D Float array representing matrix data
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null或维度不一致 / if data is null or
     * dimensions are inconsistent
     */
    public static IFloatMatrix of(float[][] data) {
        return new RereFloatMatrix(data);
    }

    public static IFloatMatrix of(Float[][] data) {
        return new RereFloatMatrix(data);
    }

    /**
     * 矩阵工厂方法（从List创建） / Matrix factory method (from List)
     * <p>
     * 使用给定的Float数组列表创建矩阵实例，每个数组代表矩阵的一行 Creates a matrix instance with the given
     * list of Float arrays, each array representing a row of the matrix
     * </p>
     *
     * @param data 包含Float数组的列表，每个数组表示矩阵的一行 / List containing Float arrays, each
     * array representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空列表或行长度不一致 / if data is null,
     * empty list, or row lengths are inconsistent
     */
    public static IFloatMatrix of(List<float[]> data) {
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
        float[][] array = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data.get(i), 0, array[i], 0, cols);
        }

        return new RereFloatMatrix(array);
    }

    /**
     * 矩阵工厂方法（从Vector数组创建） / Matrix factory method (from Vector array)
     * <p>
     * 使用给定的Vector数组创建矩阵实例，每个Vector代表矩阵的一行 Creates a matrix instance with the
     * given IFloatVector array, each IFloatVector representing a row of the
     * matrix
     * </p>
     *
     * @param data 包含Vector的数组，每个Vector表示矩阵的一行 / Array containing Vectors, each
     * IFloatVector representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空数组或行长度不一致 / if data is null,
     * empty array, or row lengths are inconsistent
     */
    public static IFloatMatrix of(IFloatVector[] data) {
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
        float[][] array = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            float[] rowData = data[i].getData();
            System.arraycopy(rowData, 0, array[i], 0, cols);
        }

        return new RereFloatMatrix(array);
    }

    public static IFloatMatrix fromArray(float[] data, int rows, int cols) {
        if (data == null) {
            throw new IllegalArgumentException("数据数组不能为null / Data array cannot be null");
        }
        if (data.length != rows * cols) {
            throw new IllegalArgumentException("数组长度必须等于行数×列数 / Array length must equal rows × columns");
        }
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }

        float[][] matrixData = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixData[i][j] = data[i * cols + j];
            }
        }
        return new RereFloatMatrix(matrixData);
    }

    public static IFloatMatrix fromArray(Float[] data, int rows, int cols) {
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
    public static IFloatMatrix rand(int rows, int cols) {
        return rand(rows, cols, RngFactory.createDefault());
    }

    public static IFloatMatrix rand(int rows, int cols, long seed) {
        return rand(rows, cols, RngFactory.createSeeded(seed));
    }

    public static IFloatMatrix rand(int rows, int cols, RngProvider rng) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rng.nextFloat();
            }
        }
        return new RereFloatMatrix(data);
    }

    public static IFloatMatrix randn(int rows, int cols) {
        return randn(rows, cols, RngFactory.createDefault());
    }

    public static IFloatMatrix randn(int rows, int cols, long seed) {
        return randn(rows, cols, RngFactory.createSeeded(seed));
    }

    public static IFloatMatrix randn(int rows, int cols, RngProvider rng) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (float) rng.nextGaussian();
            }
        }
        return new RereFloatMatrix(data);
    }

    public static IFloatMatrix randn(int rows, int cols, Float mean, Float std) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        if (std < 0) {
            throw new IllegalArgumentException("标准差不能为负数 / Standard deviation cannot be negative");
        }
        RngProvider rng = RngFactory.createDefault();
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = mean + std * (float) rng.nextGaussian();
            }
        }
        return new RereFloatMatrix(data);
    }

    public static IFloatMatrix randn(int rows, int cols, Float mean, Float std, RngProvider rng) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        if (std < 0) {
            throw new IllegalArgumentException("标准差不能为负数 / Standard deviation cannot be negative");
        }
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = mean + std * (float) rng.nextGaussian();
            }
        }
        return new RereFloatMatrix(data);
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
    public static IFloatMatrix ones(int rows, int cols) {
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = 1.0f;
            }
        }
        return new RereFloatMatrix(data);
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
    public static IFloatMatrix zeros(int rows, int cols) {
        float[][] data = new float[rows][cols];
        // Java数组默认初始化为0，所以不需要显式设置 / Java arrays are initialized to 0 by default
        return new RereFloatMatrix(data);
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
    public static IFloatMatrix average(IFloatMatrix[] a, IFloatMatrix[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("数组长度不匹配 / Array lengths don't match");
        }

        // 假设所有矩阵都是列向量，我们需要将它们合并成一个矩阵
        // Assuming all matrices are column vectors, we need to combine them into one matrix
        int rows = a[0].getRowNum();
        int cols = a.length;

        IFloatMatrix matrixA = IFloatMatrix.zeros(rows, cols);
        IFloatMatrix matrixB = IFloatMatrix.zeros(rows, cols);

        // 将列向量组合成矩阵 / Combine column vectors into matrices
        for (int i = 0; i < cols; i++) {
            matrixA.putColumn(i, a[i]);
            matrixB.putColumn(i, b[i]);
        }

        // 计算平均值 / Calculate average
        IFloatMatrix result = IFloatMatrix.zeros(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.put(row, col, (matrixA.get(row, col) + matrixB.get(row, col)) / 2);
            }
        }

        return result;
    }

    /**
     * 从本地指定位置path加载恢复矩阵 / Load matrix from specified local path
     */
    public static IMatrix load(String path) {
        return RereFloatMatrix.loadFromFile(path);
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
    public static IFloatMatrix eye(int size) {
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
    public static IFloatMatrix eye(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }

        float[][] data = new float[rows][cols];
        int minDim = Math.min(rows, cols);
        for (int i = 0; i < minDim; i++) {
            data[i][i] = 1.0f;
        }
        return new RereFloatMatrix(data);
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
    public static IFloatMatrix diag(float[] diagonal) {
        if (diagonal == null || diagonal.length == 0) {
            throw new IllegalArgumentException("对角线数组不能为null或空 / Diagonal array cannot be null or empty");
        }

        int size = diagonal.length;
        float[][] data = new float[size][size];
        for (int i = 0; i < size; i++) {
            data[i][i] = diagonal[i];
        }
        return new RereFloatMatrix(data);
    }

    public static IFloatMatrix diag(Float[] diagonal) {
        return diag(RereMathUtil.toPrimitive(diagonal));
    }

    public static IFloatMatrix diag(IFloatVector diagonal) {
        return diag(diagonal.getData());
    }

    // ========== 二维广播（形状见 {@link IMatrix}；`float[][]` 转 double 运算再转回 float）==========
    static int[] broadcastShape(IMatrix<Float> a, IMatrix<Float> b) {
        return IMatrix.broadcastShape(a, b);
    }

    static int[] broadcastShape(int rowsA, int colsA, int rowsB, int colsB) {
        return IMatrix.broadcastShape(rowsA, colsA, rowsB, colsB);
    }

    static float[][] broadcastTo(float[][] data, int targetRows, int targetCols) {
        return double2dToFloat(RereDoubleMatrix.broadcastTo(float2dToDouble(data), targetRows, targetCols));
    }

    static IFloatMatrix broadcastTo(IFloatMatrix data, int targetRows, int targetCols) {
        Objects.requireNonNull(data, "data");
        return IFloatMatrix.of(broadcastTo(data.getData(), targetRows, targetCols));
    }

    static float[][] broadcastElementWise(float[][] a, float[][] b, DoubleBinaryOperator op) {
        return double2dToFloat(RereDoubleMatrix.broadcastElementWise(float2dToDouble(a), float2dToDouble(b), op));
    }

    static IFloatMatrix broadcastElementWise(IMatrix<Float> a, IMatrix<Float> b, DoubleBinaryOperator op) {
        Objects.requireNonNull(a, "a");
        Objects.requireNonNull(b, "b");
        return IFloatMatrix.of(broadcastElementWise(a.toFloatArray(), b.toFloatArray(), op));
    }

    private static double[][] float2dToDouble(float[][] a) {
        Objects.requireNonNull(a, "a");
        double[][] d = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            d[i] = RereMathUtil.floatToDouble(a[i]);
        }
        return d;
    }

    private static float[][] double2dToFloat(double[][] a) {
        Objects.requireNonNull(a, "a");
        float[][] f = new float[a.length][];
        for (int i = 0; i < a.length; i++) {
            f[i] = RereMathUtil.doubleToFloat(a[i]);
        }
        return f;
    }

    // Note: copy(), max(), min(), sum(), mean() are inherited from IMatrix<Float>
    // Note: All matrix analysis, mathematical functions, statistical operations,
    // decomposition methods, and linear system solving are now inherited from IMatrix<Float>
    // This includes: det(), trace(), cond(), rank(), abs(), sign(), sin(), cos(), tan(),
    // sinh(), cosh(), tanh(), std(), var(), lu(), cholesky(), solve(), etc.
    // Note: subMatrix() and setSubMatrix() are now inherited from IMatrix<Float>
    // Note: set() method is now inherited from IMatrix<Float> as default method
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
    public float[][] getData();

    // ========== 协变返回类型覆盖 / Covariant Return Type Overrides ==========
    @Override
    public IFloatMatrix add(IMatrix<Float> other);

    @Override
    public IFloatMatrix sub(IMatrix<Float> other);

    @Override
    public IFloatMatrix sub(double scalar);

    @Override
    public IFloatMatrix multiplyByScalar(double scalar);

    @Override
    public IFloatMatrix divide(IMatrix<Float> other);

    @Override
    public IFloatMatrix divideByScalar(double scalar);

    @Override
    public IFloatMatrix mmul(IMatrix<Float> other);

    @Override
    public IFloatMatrix copy();

    @Override
    public IFloatMatrix transposeInPlace();

    @Override
    public IFloatMatrix transposeNew();

    @Override
    public IFloatMatrix inv();

    @Override
    public IFloatMatrix pinv();

    @Override
    public IFloatMatrix getColumnAsCloumnVector(int i);

    @Override
    public IFloatMatrix getColumnMatrix(int colIndex);

    @Override
    public IFloatMatrix sqrt();

    @Override
    public IFloatMatrix pow(Float power);

    @Override
    public IFloatMatrix exp();

    @Override
    public IFloatMatrix log();

    @Override
    public IFloatMatrix normalizeRows();

    @Override
    public IFloatMatrix normalizeColumns();

    @Override
    public IFloatMatrix normalize();

    @Override
    public IFloatMatrix center();

    @Override
    public IFloatMatrix covariance();

    @Override
    public IFloatMatrix cov();

    @Override
    public IFloatMatrix covarianceFromCentered();

    @Override
    public IFloatMatrix hstack(IMatrix<Float> other);

    @Override
    public IFloatMatrix vstack(IMatrix<Float> other);

    @Override
    public IFloatMatrix reshape(int rows, int cols);

    @Override
    public IFloatMatrix abs();

    @Override
    public IFloatMatrix sign();

    @Override
    public IFloatMatrix sin();

    @Override
    public IFloatMatrix cos();

    @Override
    public IFloatMatrix tan();

    @Override
    public IFloatMatrix sinh();

    @Override
    public IFloatMatrix cosh();

    @Override
    public IFloatMatrix tanh();

    @Override
    public IFloatMatrix cholesky();

    @Override
    public IFloatMatrix solve(IMatrix<Float> B);

    @Override
    public IFloatMatrix slice(String rowSlice, String colSlice);

    @Override
    public IFloatMatrix sliceRows(String rowSlice);

    @Override
    public IFloatMatrix sliceColumns(String colSlice);

    @Override
    public IFloatMatrix fancyGet(int[] rowIndices, int[] colIndices);

    @Override
    public IFloatMatrix booleanGet(boolean[] rowMask);

    @Override
    public IFloatMatrix booleanGet(boolean[] rowMask, boolean[] colMask);

    @Override
    public IFloatMatrix subMatrix(int startRow, int endRow, int startCol, int endCol);

    @Override
    public IFloatMatrix applyMap(java.util.function.Function<Float, Float> fun);

    @Override
    public IFloatMatrix broadcastColumn(IVector<Float> colVector, java.util.function.BiFunction<IVector<Float>, IVector<Float>, IVector<Float>> fun);

    @Override
    public IFloatMatrix broadcastRow(IVector<Float> rowVector, java.util.function.BiFunction<IVector<Float>, IVector<Float>, IVector<Float>> fun);

    @Override
    public IFloatMatrix multiply(IMatrix<Float> other);
    @Override
    public IFloatMatrix sum();
    @Override
    public IFloatMatrix mean();

    // ========== 无装箱原语重载 / Boxing-free primitive overloads ==========
    /**
     * 对每个元素应用给定运算符（无装箱）。与 {@link IMatrix#applyMap(Function)} 不同，此方法使用
     * {@code float → float} 原语操作，避免 Float 装箱/拆箱开销。
     *
     * @param fun 要应用的单精度运算符
     * @return 新矩阵
     */
    default IMatrix<Float> applyMap(FloatUnaryOperator fun) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }
        float[][] data = getData();
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = fun.applyAsFloat(data[i][j]);
            }
        }
        return IFloatMatrix.of(result);
    }

    /**
     * 对每行/列应用聚合函数（无装箱）。与 {@link IMatrix#apply(Function, int)} 不同，此方法使用
     * {@code float[] → float} 原语操作，避免 IVector/Float 装箱开销。
     * <p>
     * <strong>使用示例：</strong>
     * <pre>{@code
     * IFloatMatrix m = IFloatMatrix.rand(3, 4);
     * IVector<Float> rowSums = m.reduce(row -> {
     *     float s = 0;
     *     for (float v : row) s += v;
     *     return s;
     * }, 1);
     * }</pre>
     *
     * @param fun 要应用的函数，接受 float[] 并返回标量 float
     * @param axis 轴: 0=列, 1=行
     * @return 聚合结果向量
     */
    default IVector<Float> reduce(ToFloatFunction fun, int axis) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }
        float[][] data = getData();
        if (axis == 0) {
            int rows = data.length;
            int cols = data[0].length;
            float[] result = new float[cols];
            for (int j = 0; j < cols; j++) {
                float[] colData = new float[rows];
                for (int i = 0; i < rows; i++) {
                    colData[i] = data[i][j];
                }
                result[j] = fun.applyAsFloat(colData);
            }
            return IFloatVector.of(result);
        } else if (axis == 1) {
            float[] result = new float[data.length];
            for (int i = 0; i < data.length; i++) {
                result[i] = fun.applyAsFloat(data[i]);
            }
            return IFloatVector.of(result);
        }
        throw new IllegalArgumentException("axis must be 0 (column) or 1 (row)");
    }

    /**
     * 按列应用聚合函数（无装箱，等同于 {@code reduce(fun, 0)}）。
     *
     * @param fun 要应用的函数
     * @return 聚合结果向量
     */
    default IVector<Float> reduce(ToFloatFunction fun) {
        return reduce(fun, 0);
    }

    /**
     * {@inheritDoc}
     * @return {@link IFloatMatrix} 实例
     */
    @Override
    default IFloatMatrix addColumn(IVector<Float> colVector) {
        requireFloatVector(colVector, "列向量 / column vector");
        return (IFloatMatrix) IMatrix.super.addColumn(colVector);
    }

    /**
     * 在矩阵右侧追加一列（{@link IFloatVector} 便捷重载）。
     */
    default IFloatMatrix addColumn(IFloatVector colVector) {
        return addColumn((IVector<Float>) colVector);
    }

    /**
     * {@inheritDoc}
     * @return {@link IFloatMatrix} 实例
     */
    @Override
    default IFloatMatrix addRow(IVector<Float> rowVector) {
        requireFloatVector(rowVector, "行向量 / row vector");
        return (IFloatMatrix) IMatrix.super.addRow(rowVector);
    }

    /**
     * 在矩阵底部追加一行（{@link IFloatVector} 便捷重载）。
     */
    default IFloatMatrix addRow(IFloatVector rowVector) {
        return addRow((IVector<Float>) rowVector);
    }

    private static void requireFloatVector(IVector<?> vector, String label) {
        if (vector instanceof IDoubleVector) {
            throw new IllegalArgumentException(
                    label + " 必须为 Float 类型，不能传入 Double 向量 / "
                            + label + " must be Float type, Double vector is not allowed");
        }
    }

}
