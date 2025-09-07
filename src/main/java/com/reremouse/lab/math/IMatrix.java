package com.reremouse.lab.math;

import java.util.Random;

import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;
import java.util.List;

/**
 * 矩阵操作接口 / Matrix Operations Interface
 * <p>
 * 本接口定义了矩阵的常用操作，包括基本的数学运算、矩阵变换、数据访问等功能。 提供了创建矩阵的静态工厂方法和各种矩阵运算的抽象方法定义。
 * </p>
 * <p>
 * This interface defines common matrix operations including basic mathematical
 * operations, matrix transformations, data access and other functionalities.
 * Provides static factory methods for creating matrices and abstract method
 * definitions for various matrix operations.
 * </p>
 *
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 * <li>基本数学运算：加法、减法、乘法、除法 / Basic math operations: add, subtract, multiply,
 * divide</li>
 * <li>矩阵变换：转置、幂运算、开方 / IMatrix transformations: transpose, power, square
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
 // 创建矩阵 / Create matrices
 IMatrix matrix1 = IMatrix.ones(3, 3);
 IMatrix matrix2 = IMatrix.rand(3, 3);
 float[][] data = {{1, 2}, {3, 4}};
 IMatrix matrix3 = IMatrix.of(data);

 // 矩阵运算 / IMatrix operations
 IMatrix result = matrix1.add(matrix2).mmul(2.0f);

 // 获取行列 / Get rows/columns
 IVector row = matrix3.getRow(0);
 IVector col = matrix3.getColunm(0);

 // 矩阵转置 / IMatrix transpose
 IMatrix transposed = matrix3.transpose();
 }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IMatrix {

    /**
     * 矩阵工厂方法 / Matrix factory method
     * <p>
     * 使用给定的二维数组创建矩阵实例 Creates a matrix instance with the given 2D array
     * </p>
     *
     * @param data 二维float数组，表示矩阵数据 / 2D float array representing matrix data
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null或维度不一致 / if data is null or
     * dimensions are inconsistent
     */
    public static IMatrix of(float[][] data) {
        return new RereMatrix(data);
    }
    
    /**
     * 矩阵工厂方法（从List创建） / Matrix factory method (from List)
     * <p>
     * 使用给定的float数组列表创建矩阵实例，每个数组代表矩阵的一行
     * Creates a matrix instance with the given list of float arrays, each array representing a row of the matrix
     * </p>
     *
     * @param data 包含float数组的列表，每个数组表示矩阵的一行 / List containing float arrays, each array representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空列表或行长度不一致 / if data is null, empty list, or row lengths are inconsistent
     */
    public static IMatrix of(List<float[]> data) {
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
        
        return new RereMatrix(array);
    }
    
    /**
     * 矩阵工厂方法（从Vector数组创建） / Matrix factory method (from Vector array)
     * <p>
 使用给定的Vector数组创建矩阵实例，每个Vector代表矩阵的一行
 Creates a matrix instance with the given IVector array, each IVector representing a row of the matrix
 </p>
     *
     * @param data 包含Vector的数组，每个Vector表示矩阵的一行 / Array containing Vectors, each IVector representing a row of matrix
     * @return 新的矩阵实例 / New matrix instance
     * @throws IllegalArgumentException 如果数据为null、空数组或行长度不一致 / if data is null, empty array, or row lengths are inconsistent
     */
    public static IMatrix of(IVector[] data){
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
        
        return new RereMatrix(array);
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
    public static IMatrix rand(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        
        Random random = new Random();
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (float) random.nextFloat(); // 生成均匀分布随机数 / Generate uniform distribution random numbers
            }
        }
        return new RereMatrix(data);
    }
    
    /**
     * 创建随机矩阵 / Create random matrix
     * <p>
     * 创建一个指定大小的随机矩阵，元素值服从标准正态分布 Creates a random matrix of specified size with
     * elements following standard normal distribution
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are
     * less than or equal to 0
     */
    public static IMatrix randn(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        
        Random random = new Random();
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (float) random.nextGaussian(); // 生成标准正态分布随机数 / Generate standard normal distribution random numbers
            }
        }
        return new RereMatrix(data);
    }
    
    /**
     * 创建随机矩阵（指定正态分布的均值和标准差） / Create random matrix with specified mean and standard deviation
     * <p>
     * 创建一个指定大小的随机矩阵，元素值服从正态分布，具有指定的均值和标准差
     * Creates a random matrix of specified size with elements following normal distribution with specified mean and standard deviation
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param mean 正态分布的均值 / Mean of normal distribution
     * @param std 正态分布的标准差 / Standard deviation of normal distribution
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0，或标准差小于0 / if rows or columns are less than or equal to 0, or standard deviation is negative
     */
    public static IMatrix randn(int rows, int cols, float mean, float std) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        if (std < 0) {
            throw new IllegalArgumentException("标准差不能为负数 / Standard deviation cannot be negative");
        }
        
        Random random = new Random();
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // 生成标准正态分布随机数，然后按公式 X = mean + std * Z 转换为指定均值和标准差的正态分布
                // Generate standard normal distribution random number, then convert to normal distribution with specified mean and std using formula X = mean + std * Z
                data[i][j] = mean + std * (float) random.nextGaussian();
            }
        }
        return new RereMatrix(data);
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
    public static IMatrix ones(int rows, int cols) {
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = 1.0f;
            }
        }
        return new RereMatrix(data);
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
    public static IMatrix zeros(int rows, int cols) {
        float[][] data = new float[rows][cols];
        // Java数组默认初始化为0，所以不需要显式设置 / Java arrays are initialized to 0 by default
        return new RereMatrix(data);
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
    public static IMatrix average(IMatrix[] a, IMatrix[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("数组长度不匹配 / Array lengths don't match");
        }

        // 假设所有矩阵都是列向量，我们需要将它们合并成一个矩阵
        // Assuming all matrices are column vectors, we need to combine them into one matrix
        int rows = a[0].getRowNum();
        int cols = a.length;

        IMatrix matrixA = IMatrix.zeros(rows, cols);
        IMatrix matrixB = IMatrix.zeros(rows, cols);

        // 将列向量组合成矩阵 / Combine column vectors into matrices
        for (int i = 0; i < cols; i++) {
            matrixA.putColumn(i, a[i]);
            matrixB.putColumn(i, b[i]);
        }

        // 计算平均值 / Calculate average
        IMatrix result = IMatrix.zeros(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.put(row, col, (matrixA.get(row, col) + matrixB.get(row, col)) / 2);
            }
        }

        return result;
    }

    /**
     * 矩阵开方运算 / Matrix square root operation
     * <p>
     * 对矩阵中每个元素进行开方运算 Performs square root operation on each element in the
     * matrix
     * </p>
     *
     * @return 新的矩阵对象，包含开方运算结果 / New matrix object containing the square root
     * results
     * @throws ArithmeticException 如果元素值为负数 / if any element value is negative
     */
    public IMatrix sqrt();

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
    public IMatrix pow(float power);

    /**
     * 计算行求和 / Calculate row sums
     * <p>
     * 计算矩阵每一列的元素和，返回结果向量 Calculates the sum of elements in each column, returns
     * result vector
     * </p>
     *
     * @return 包含每列元素和的向量 / IVector containing the sum of elements in each column
     */
    public IVector rowSums();

    /**
     * 计算行均值 / Calculate row means
     * <p>
     * 计算矩阵每一列的元素平均值，返回结果向量 Calculates the mean of elements in each column,
     * returns result vector
     * </p>
     *
     * @return 包含每列元素平均值的向量 / IVector containing the mean of elements in each
 column
     */
    public IVector rowMeans();

    /**
     * 计算列求和 / Calculate column sums
     * <p>
     * 计算矩阵每一行的元素和，返回结果向量 Calculates the sum of elements in each row, returns
     * result vector
     * </p>
     *
     * @return 包含每行元素和的向量 / IVector containing the sum of elements in each row
     */
    public IVector colSums();

    /**
     * 计算列均值 / Calculate column means
     * <p>
     * 计算矩阵每一行的元素平均值，返回结果向量 Calculates the mean of elements in each row, returns
     * result vector
     * </p>
     *
     * @return 包含每行元素平均值的向量 / IVector containing the mean of elements in each row
     */
    public IVector colMeans();

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
     * 获取指定列向量 / Get specified column vector
     * <p>
     * 返回矩阵指定列的向量表示 Returns the vector representation of the specified column in
     * the matrix
     * </p>
     *
     * @param i 列索引（从0开始） / Column index (0-based)
     * @return 指定列的向量 / IVector of the specified column
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of
     * bounds
     */
    public IVector getColunm(int i);

    /**
     * 获取指定行向量 / Get specified row vector
     * <p>
     * 返回矩阵指定行的向量表示 Returns the vector representation of the specified row in
     * the matrix
     * </p>
     *
     * @param i 行索引（从0开始） / Row index (0-based)
     * @return 指定行的向量 / IVector of the specified row
     * @throws IndexOutOfBoundsException 如果行索引超出范围 / if row index is out of
     * bounds
     */
    public IVector getRow(int i);

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
     * 矩阵转置（就地操作） / Matrix transpose (in-place operation)
     * <p>
     * 对当前矩阵执行转置操作，修改原矩阵数据 Performs transpose operation on the current matrix,
     * modifying the original matrix data
     * </p>
     *
     * @return 转置后的矩阵（指向当前对象） / Transposed matrix (pointing to current object)
     */
    public IMatrix transpose();

    /**
     * 矩阵转置（创建新对象） / Matrix transpose (create new object)
     * <p>
     * 创建一个新的转置矩阵，不修改原矩阵 Creates a new transposed matrix without modifying the
     * original matrix
     * </p>
     *
     * @return 新的转置矩阵对象 / New transposed matrix object
     */
    public IMatrix transposeNew();

    /**
     * 矩阵转置简写方法 / Matrix transpose shorthand method
     * <p>
     * transpose()方法的简写形式 Shorthand form of transpose() method
     * </p>
     *
     * @return 转置后的矩阵 / Transposed matrix
     * @see #transpose()
     */
    public IMatrix t();

    /**
     * 矩阵减法运算（标量） / Matrix subtraction with scalar
     * <p>
     * 矩阵中的每个元素减去标量值 Subtracts a scalar value from each element in the matrix
     * </p>
     *
     * @param scalar 要减去的标量值 / The scalar value to subtract
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    public IMatrix sub(float scalar);

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
    public IMatrix sub(IMatrix other);

    /**
     * 矩阵标量乘法运算（float） / Matrix scalar multiplication (float)
     * <p>
     * 矩阵中的每个元素乘以标量值 Multiplies each element in the matrix by a scalar value
     * </p>
     *
     * @param scalar 标量乘数 / The scalar multiplier
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    public IMatrix mmul(float scalar);

    /**
     * 矩阵标量乘法运算（double） / Matrix scalar multiplication (double)
     * <p>
     * 矩阵中的每个元素乘以标量值，double类型会被转换为float Multiplies each element in the matrix by
     * a scalar value, double is converted to float
     * </p>
     *
     * @param scalar 标量乘数（double类型） / The scalar multiplier (double type)
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    public IMatrix mmul(double scalar);
    
    /**
     * 矩阵乘法运算 / Matrix multiplication
     * <p>
     * 计算两个矩阵的乘积，要求第一个矩阵的列数等于第二个矩阵的行数
     * Computes the product of two matrices, requires the number of columns of the first matrix 
     * to equal the number of rows of the second matrix
     * </p>
     * 
     * @param other 另一个矩阵 / The other matrix
     * @return 矩阵乘法结果 / IMatrix multiplication result  
     * @throws IllegalArgumentException 如果矩阵维度不匹配进行乘法运算 / if matrix dimensions don't match for multiplication
     */
    public IMatrix mmul(IMatrix other);

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
    public IMatrix add(IMatrix other);

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
    public IMatrix divide(IMatrix other);

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
    public float dot(IMatrix other);

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
    public IMatrix getColumn(int colIndex);

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
    public void putColumn(int colIndex, IMatrix column);

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
    public IMatrix[] getColumns(int[] indices);

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
    public float get(int row, int col);

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
    public void put(int row, int col, float value);

    /**
     * 获取矩阵行数 / Get number of rows
     * <p>
     * 返回矩阵的行数 Returns the number of rows in the matrix
     * </p>
     *
     * @return 矩阵行数 / Number of rows in the matrix
     */
    public int getRows();

    /**
     * 获取矩阵列数 / Get number of columns
     * <p>
     * 返回矩阵的列数 Returns the number of columns in the matrix
     * </p>
     *
     * @return 矩阵列数 / Number of columns in the matrix
     */
    public int getColumns();

    /**
     * 特征分解，返回的特征值按大小排列，返回的矩阵的列为各个特征向量，与特征值的顺序对应
     *
     * @return 返回特征向量和特征值，其中返回的矩阵的列为各个特征向量，返回的向量中包含所有特征值
     */
    public Tuple2<IVector, IMatrix> eigen();
    
    /**
     * QR算法特征分解的辅助方法
     * 
     * @return 特征值和特征向量
     */
    public Tuple2<IVector, IMatrix> qrEigenDecomposition();

    /**
     * 奇异值分解
     *
     * @return
     */
    public Tuple3<IMatrix, IVector, IMatrix> svd();
    
    
    /**
     * QR分解
     * 
     * @return Q和R矩阵
     */
    public Tuple2<IMatrix, IMatrix> qr();

    /**
     * 矩阵转换为一维数组 / Convert matrix to 1D array
     * <p>
     * 将矩阵转换为一维float数组。对于列向量，返回该列的数据； 对于普通矩阵，按行优先顺序展开。
     * </p>
     * <p>
     * Converts the matrix to a 1D float array. For column vectors, returns the
     * column data; For regular matrices, flattens in row-major order.
     * </p>
     *
     * @return 一维float数组 / 1D float array
     */
    public float[] toArray();

    /**
     * 矩阵转换为double数组 / Convert matrix to double array
     * <p>
     * 将矩阵转换为一维double数组，用于与其他库的兼容性 Converts the matrix to a 1D double array for
     * compatibility with other libraries
     * </p>
     *
     * @return 一维double数组 / 1D double array
     * @see #toArray()
     */
    public double[] toDoubleArray();

    /**
     * 矩阵指数运算 / Matrix exponential operation
     * <p>
     * 对矩阵中每个元素进行指数运算（e^x） Performs exponential operation (e^x) on each element in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含指数运算结果 / New matrix object containing exponential operation results
     */
    public IMatrix exp();

    /**
     * 矩阵自然对数运算 / Matrix natural logarithm operation
     * <p>
     * 对矩阵中每个元素进行自然对数运算（ln(x)） Performs natural logarithm operation (ln(x)) on each element in the matrix
     * </p>
     *
     * @return 新的矩阵对象，包含对数运算结果 / New matrix object containing logarithm operation results
     * @throws ArithmeticException 如果元素值小于等于0 / if any element value is less than or equal to 0
     */
    public IMatrix log();

    /**
     * 计算矩阵的Frobenius范数 / Compute Frobenius norm of the matrix
     * <p>
     * 计算矩阵的Frobenius范数，即所有元素平方和的开方
     * Computes the Frobenius norm of the matrix, which is the square root of the sum of squares of all elements
     * </p>
     *
     * @return Frobenius范数 / Frobenius norm
     */
    public float frobeniusNorm();

    /**
     * 计算两个矩阵之间的Frobenius距离 / Compute Frobenius distance between two matrices
     * <p>
     * 计算当前矩阵与另一个矩阵之间的Frobenius距离
     * Computes the Frobenius distance between current matrix and another matrix
     * </p>
     *
     * @param other 另一个矩阵 / The other matrix
     * @return Frobenius距离 / Frobenius distance
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't match
     */
    public float frobeniusDistance(IMatrix other);

    /**
     * 矩阵按行归一化 / Row-wise normalization of matrix
     * <p>
     * 对矩阵的每一行进行L2归一化
     * Performs L2 normalization on each row of the matrix
     * </p>
     *
     * @return 归一化后的矩阵 / Normalized matrix
     */
    public IMatrix normalizeRows();

    /**
     * 矩阵按列归一化 / Column-wise normalization of matrix
     * <p>
     * 对矩阵的每一列进行L2归一化
     * Performs L2 normalization on each column of the matrix
     * </p>
     *
     * @return 归一化后的矩阵 / Normalized matrix
     */
    public IMatrix normalizeColumns();

    /**
     * 矩阵数据中心化 / Matrix data centering
     * <p>
     * 对矩阵的每一列减去该列的均值，实现数据中心化
     * Subtracts the mean of each column from the column elements, implementing data centering
     * </p>
     *
     * @return 中心化后的矩阵 / Centered matrix
     */
    public IMatrix center();

    /**
     * 计算协方差矩阵 / Compute covariance matrix
     * <p>
     * 计算数据矩阵的协方差矩阵，假设每行是一个样本，每列是一个特征
     * Computes the covariance matrix of the data matrix, assuming each row is a sample and each column is a feature
     * </p>
     *
     * @return 协方差矩阵 / Covariance matrix
     */
    public IMatrix covariance();

    /**
     * 计算协方差矩阵（已中心化数据） / Compute covariance matrix (for centered data)
     * <p>
     * 对于已经中心化的数据，直接计算协方差矩阵 = (X^T * X) / (n-1)
     * For already centered data, directly compute covariance matrix = (X^T * X) / (n-1)
     * </p>
     *
     * @return 协方差矩阵 / Covariance matrix
     */
    public IMatrix covarianceFromCentered();
    
    /**
     * 求解矩阵的逆 / Matrix inverse
     * <p>
     * 使用高斯-约旦消元法计算方阵的逆矩阵。只有方阵且行列式不为零的矩阵才有逆矩阵。
     * Uses Gauss-Jordan elimination to compute the inverse of a square matrix. 
     * Only square matrices with non-zero determinant have an inverse.
     * </p>
     * 
     * @return 逆矩阵 / Inverse matrix
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     * @throws ArithmeticException 如果矩阵是奇异的（不可逆） / if matrix is singular (non-invertible)
     */
    public IMatrix inv();
    
    /**
     * 求解矩阵的伪逆 / Matrix pseudo-inverse
     * <p>
     * 使用奇异值分解(SVD)计算矩阵的Moore-Penrose伪逆。伪逆适用于任意维度的矩阵，
     * 包括非方阵和奇异矩阵。计算公式：A⁺ = V * Σ⁺ * Uᵀ，其中Σ⁺是奇异值的伪逆。
     * </p>
     * <p>
     * Uses Singular Value Decomposition (SVD) to compute the Moore-Penrose pseudo-inverse.
     * The pseudo-inverse works for matrices of any dimensions, including non-square and singular matrices.
     * Formula: A⁺ = V * Σ⁺ * Uᵀ, where Σ⁺ is the pseudo-inverse of singular values.
     * </p>
     * 
     * @return 伪逆矩阵 / Pseudo-inverse matrix
     */
    public IMatrix pinv();
    
    
    /**
     * 从本地指定位置path加载恢复矩阵 / Load matrix from specified local path
     * <p>
     * 从指定的文件路径加载矩阵数据，文件格式为：第一行为矩阵维度（行数 列数），后续行为矩阵数据（每行元素用空格分隔）
     * Loads matrix data from the specified file path. File format: first line contains matrix dimensions (rows columns), 
     * subsequent lines contain matrix data (elements in each row separated by spaces)
     * </p>
     * 
     * @param path 文件路径 / File path
     * @return 从文件加载的矩阵对象 / IMatrix object loaded from file
     * @throws IllegalArgumentException 如果文件路径为null或为空 / if file path is null or empty
     * @throws RuntimeException 如果文件读取失败或格式错误 / if file reading fails or format is incorrect
     */
    public static IMatrix load(String path) {
        return RereMatrix.loadFromFile(path);
    }
    
    /**
     * 将矩阵数据保存在本地指定位置 / Save matrix data to specified local path
     * <p>
     * 将当前矩阵保存到指定的文件路径，保存格式为：第一行为矩阵维度（行数 列数），后续行为矩阵数据（每行元素用空格分隔）
     * Saves current matrix to the specified file path. Save format: first line contains matrix dimensions (rows columns),
     * subsequent lines contain matrix data (elements in each row separated by spaces)
     * </p>
     * 
     * @param path 文件保存路径 / File save path
     * @throws IllegalArgumentException 如果文件路径为null或为空 / if file path is null or empty
     * @throws RuntimeException 如果文件写入失败 / if file writing fails
     */
    public void save(String path);

    /**
     * 创建单位矩阵 / Create identity matrix
     * <p>
     * 创建一个指定大小的单位矩阵（对角线上元素为1，其他元素为0）
     * Creates an identity matrix of specified size (diagonal elements are 1, others are 0)
     * </p>
     *
     * @param size 矩阵大小（行数和列数相同） / Matrix size (rows and columns are the same)
     * @return 单位矩阵 / Identity matrix
     * @throws IllegalArgumentException 如果大小小于等于0 / if size is less than or equal to 0
     */
    public static IMatrix eye(int size) {
        return eye(size, size);
    }

    /**
     * 创建单位矩阵 / Create identity matrix
     * <p>
     * 创建一个指定大小的单位矩阵（对角线上元素为1，其他元素为0）
     * Creates an identity matrix of specified size (diagonal elements are 1, others are 0)
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @return 单位矩阵 / Identity matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    public static IMatrix eye(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        
        float[][] data = new float[rows][cols];
        int minDim = Math.min(rows, cols);
        for (int i = 0; i < minDim; i++) {
            data[i][i] = 1.0f;
        }
        return new RereMatrix(data);
    }

    /**
     * 创建对角矩阵 / Create diagonal matrix
     * <p>
     * 从给定的对角线元素创建对角矩阵
     * Creates a diagonal matrix from the given diagonal elements
     * </p>
     *
     * @param diagonal 对角线元素数组 / Array of diagonal elements
     * @return 对角矩阵 / Diagonal matrix
     * @throws IllegalArgumentException 如果对角线数组为null或空 / if diagonal array is null or empty
     */
    public static IMatrix diag(float[] diagonal) {
        if (diagonal == null || diagonal.length == 0) {
            throw new IllegalArgumentException("对角线数组不能为null或空 / Diagonal array cannot be null or empty");
        }
        
        int size = diagonal.length;
        float[][] data = new float[size][size];
        for (int i = 0; i < size; i++) {
            data[i][i] = diagonal[i];
        }
        return new RereMatrix(data);
    }

    /**
     * 创建随机矩阵（指定种子） / Create random matrix with specified seed
     * <p>
     * 创建一个指定大小的随机矩阵（均匀分布），使用指定的种子值确保结果可重现
     * Creates a random matrix of specified size using the specified seed value to ensure reproducible results
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param seed 随机数种子 / Random number seed
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    public static IMatrix rand(int rows, int cols, long seed) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        
        Random random = new Random(seed);
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (float) random.nextFloat();
            }
        }
        return new RereMatrix(data);
    }
    
    /**
     * 创建随机矩阵（指定种子） / Create random matrix with specified seed
     * <p>
     * 创建一个指定大小的随机矩阵（正态分布），使用指定的种子值确保结果可重现
     * Creates a random matrix of specified size using the specified seed value to ensure reproducible results
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows
     * @param cols 矩阵列数 / Number of columns
     * @param seed 随机数种子 / Random number seed
     * @return 随机矩阵 / Random matrix
     * @throws IllegalArgumentException 如果行数或列数小于等于0 / if rows or columns are less than or equal to 0
     */
    public static IMatrix randn(int rows, int cols, long seed) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        
        Random random = new Random(seed);
        float[][] data = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = (float) random.nextGaussian();
            }
        }
        return new RereMatrix(data);
    }

    /**
     * 从一维数组创建矩阵 / Create matrix from 1D array
     * <p>
     * 将一维数组重塑为指定大小的矩阵
     * Reshapes a 1D array into a matrix of specified size
     * </p>
     *
     * @param data 一维数组数据 / 1D array data
     * @param rows 目标矩阵行数 / Target matrix rows
     * @param cols 目标矩阵列数 / Target matrix columns
     * @return 重塑后的矩阵 / Reshaped matrix
     * @throws IllegalArgumentException 如果数组长度与目标尺寸不匹配 / if array length doesn't match target dimensions
     */
    public static IMatrix fromArray(float[] data, int rows, int cols) {
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
        return new RereMatrix(matrixData);
    }

    /**
     * 矩阵连接（水平方向） / Matrix concatenation (horizontal)
     * <p>
     * 将两个矩阵在水平方向上连接，要求行数相同
     * Concatenates two matrices horizontally, requires same number of rows
     * </p>
     *
     * @param other 要连接的另一个矩阵 / The other matrix to concatenate
     * @return 连接后的矩阵 / Concatenated matrix
     * @throws IllegalArgumentException 如果矩阵行数不匹配 / if matrix row counts don't match
     */
    public IMatrix hstack(IMatrix other);

    /**
     * 矩阵连接（垂直方向） / Matrix concatenation (vertical)
     * <p>
     * 将两个矩阵在垂直方向上连接，要求列数相同
     * Concatenates two matrices vertically, requires same number of columns
     * </p>
     *
     * @param other 要连接的另一个矩阵 / The other matrix to concatenate
     * @return 连接后的矩阵 / Concatenated matrix
     * @throws IllegalArgumentException 如果矩阵列数不匹配 / if matrix column counts don't match
     */
    public IMatrix vstack(IMatrix other);

    /**
     * 矩阵分割（水平方向） / Matrix splitting (horizontal)
     * <p>
     * 将矩阵在水平方向上分割为多个子矩阵
     * Splits the matrix horizontally into multiple sub-matrices
     * </p>
     *
     * @param indices 分割点的列索引数组 / Array of column indices for split points
     * @return 分割后的子矩阵数组 / Array of split sub-matrices
     * @throws IllegalArgumentException 如果分割索引无效 / if split indices are invalid
     */
    public IMatrix[] hsplit(int[] indices);

    /**
     * 矩阵分割（垂直方向） / Matrix splitting (vertical)
     * <p>
     * 将矩阵在垂直方向上分割为多个子矩阵
     * Splits the matrix vertically into multiple sub-matrices
     * </p>
     *
     * @param indices 分割点的行索引数组 / Array of row indices for split points
     * @return 分割后的子矩阵数组 / Array of split sub-matrices
     * @throws IllegalArgumentException 如果分割索引无效 / if split indices are invalid
     */
    public IMatrix[] vsplit(int[] indices);

    /**
     * 矩阵重塑 / Matrix reshape
     * <p>
     * 将矩阵重塑为新的维度，保持元素总数不变
     * Reshapes the matrix to new dimensions while keeping the total number of elements unchanged
     * </p>
     *
     * @param rows 新的行数 / New number of rows
     * @param cols 新的列数 / New number of columns
     * @return 重塑后的矩阵 / Reshaped matrix
     * @throws IllegalArgumentException 如果新维度与元素总数不匹配 / if new dimensions don't match total element count
     */
    public IMatrix reshape(int rows, int cols);

    /**
     * 矩阵复制 / Matrix copy
     * <p>
     * 创建当前矩阵的深拷贝
     * Creates a deep copy of the current matrix
     * </p>
     *
     * @return 矩阵的副本 / Copy of the matrix
     */
    public IMatrix copy();

    /**
     * 计算矩阵行列式 / Compute matrix determinant
     * <p>
     * 计算方阵的行列式值
     * Computes the determinant of a square matrix
     * </p>
     *
     * @return 行列式值 / Determinant value
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    public float det();

    /**
     * 计算矩阵迹 / Compute matrix trace
     * <p>
     * 计算方阵的迹（对角线元素之和）
     * Computes the trace of a square matrix (sum of diagonal elements)
     * </p>
     *
     * @return 矩阵迹 / Matrix trace
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    public float trace();

    /**
     * 计算矩阵条件数 / Compute matrix condition number
     * <p>
     * 计算矩阵的条件数，用于评估矩阵的数值稳定性
     * Computes the condition number of a matrix to assess numerical stability
     * </p>
     *
     * @return 条件数 / Condition number
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    public float cond();

    /**
     * 计算矩阵秩 / Compute matrix rank
     * <p>
     * 计算矩阵的秩，即线性无关的行或列的最大数量
     * Computes the rank of a matrix, the maximum number of linearly independent rows or columns
     * </p>
     *
     * @return 矩阵秩 / Matrix rank
     */
    public int rank();

    /**
     * 矩阵绝对值 / Matrix absolute value
     * <p>
     * 对矩阵中每个元素取绝对值
     * Takes the absolute value of each element in the matrix
     * </p>
     *
     * @return 包含绝对值的新矩阵 / New matrix containing absolute values
     */
    public IMatrix abs();

    /**
     * 矩阵符号函数 / Matrix sign function
     * <p>
     * 对矩阵中每个元素应用符号函数：正数为1，负数为-1，零为0
     * Applies sign function to each element: 1 for positive, -1 for negative, 0 for zero
     * </p>
     *
     * @return 包含符号值的新矩阵 / New matrix containing sign values
     */
    public IMatrix sign();

    /**
     * 矩阵正弦函数 / Matrix sine function
     * <p>
     * 对矩阵中每个元素计算正弦值
     * Computes sine value for each element in the matrix
     * </p>
     *
     * @return 包含正弦值的新矩阵 / New matrix containing sine values
     */
    public IMatrix sin();

    /**
     * 矩阵余弦函数 / Matrix cosine function
     * <p>
     * 对矩阵中每个元素计算余弦值
     * Computes cosine value for each element in the matrix
     * </p>
     *
     * @return 包含余弦值的新矩阵 / New matrix containing cosine values
     */
    public IMatrix cos();

    /**
     * 矩阵正切函数 / Matrix tangent function
     * <p>
     * 对矩阵中每个元素计算正切值
     * Computes tangent value for each element in the matrix
     * </p>
     *
     * @return 包含正切值的新矩阵 / New matrix containing tangent values
     */
    public IMatrix tan();

    /**
     * 矩阵双曲正弦函数 / Matrix hyperbolic sine function
     * <p>
     * 对矩阵中每个元素计算双曲正弦值
     * Computes hyperbolic sine value for each element in the matrix
     * </p>
     *
     * @return 包含双曲正弦值的新矩阵 / New matrix containing hyperbolic sine values
     */
    public IMatrix sinh();

    /**
     * 矩阵双曲余弦函数 / Matrix hyperbolic cosine function
     * <p>
     * 对矩阵中每个元素计算双曲余弦值
     * Computes hyperbolic cosine value for each element in the matrix
     * </p>
     *
     * @return 包含双曲余弦值的新矩阵 / New matrix containing hyperbolic cosine values
     */
    public IMatrix cosh();

    /**
     * 矩阵双曲正切函数 / Matrix hyperbolic tangent function
     * <p>
     * 对矩阵中每个元素计算双曲正切值
     * Computes hyperbolic tangent value for each element in the matrix
     * </p>
     *
     * @return 包含双曲正切值的新矩阵 / New matrix containing hyperbolic tangent values
     */
    public IMatrix tanh();

    /**
     * LU分解 / LU decomposition
     * <p>
     * 将矩阵分解为下三角矩阵L和上三角矩阵U的乘积：A = L * U
     * Decomposes matrix into product of lower triangular matrix L and upper triangular matrix U: A = L * U
     * </p>
     *
     * @return 包含L和U矩阵的元组 / Tuple containing L and U matrices
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    public Tuple2<IMatrix, IMatrix> lu();

    /**
     * Cholesky分解 / Cholesky decomposition
     * <p>
     * 将对称正定矩阵分解为下三角矩阵L的乘积：A = L * L^T
     * Decomposes symmetric positive definite matrix into product of lower triangular matrix L: A = L * L^T
     * </p>
     *
     * @return 下三角矩阵L / Lower triangular matrix L
     * @throws IllegalArgumentException 如果矩阵不是对称正定矩阵 / if matrix is not symmetric positive definite
     */
    public IMatrix cholesky();

    /**
     * 求解线性方程组 / Solve linear system
     * <p>
     * 求解线性方程组 Ax = b，其中A是当前矩阵，b是右侧向量
     * Solves linear system Ax = b where A is the current matrix and b is the right-hand side vector
     * </p>
     *
     * @param b 右侧向量 / Right-hand side vector
     * @return 解向量x / Solution vector x
     * @throws IllegalArgumentException 如果矩阵不是方阵或维度不匹配 / if matrix is not square or dimensions don't match
     */
    public IVector solve(IVector b);

    /**
     * 求解线性方程组（矩阵形式） / Solve linear system (matrix form)
     * <p>
     * 求解线性方程组 AX = B，其中A是当前矩阵，B是右侧矩阵
     * Solves linear system AX = B where A is the current matrix and B is the right-hand side matrix
     * </p>
     *
     * @param B 右侧矩阵 / Right-hand side matrix
     * @return 解矩阵X / Solution matrix X
     * @throws IllegalArgumentException 如果矩阵不是方阵或维度不匹配 / if matrix is not square or dimensions don't match
     */
    public IMatrix solve(IMatrix B);

    /**
     * 矩阵最大值 / Matrix maximum value
     * <p>
     * 返回矩阵中的最大元素值
     * Returns the maximum element value in the matrix
     * </p>
     *
     * @return 最大元素值 / Maximum element value
     */
    public float max();

    /**
     * 矩阵最小值 / Matrix minimum value
     * <p>
     * 返回矩阵中的最小元素值
     * Returns the minimum element value in the matrix
     * </p>
     *
     * @return 最小元素值 / Minimum element value
     */
    public float min();

    /**
     * 矩阵元素求和 / Matrix element sum
     * <p>
     * 返回矩阵中所有元素的总和
     * Returns the sum of all elements in the matrix
     * </p>
     *
     * @return 元素总和 / Sum of all elements
     */
    public float sum();

    /**
     * 矩阵元素均值 / Matrix element mean
     * <p>
     * 返回矩阵中所有元素的平均值
     * Returns the mean of all elements in the matrix
     * </p>
     *
     * @return 元素均值 / Mean of all elements
     */
    public float mean();

    /**
     * 矩阵元素标准差 / Matrix element standard deviation
     * <p>
     * 返回矩阵中所有元素的标准差
     * Returns the standard deviation of all elements in the matrix
     * </p>
     *
     * @return 元素标准差 / Standard deviation of all elements
     */
    public float std();

    /**
     * 矩阵元素方差 / Matrix element variance
     * <p>
     * 返回矩阵中所有元素的方差
     * Returns the variance of all elements in the matrix
     * </p>
     *
     * @return 元素方差 / Variance of all elements
     */
    public float var();

}
