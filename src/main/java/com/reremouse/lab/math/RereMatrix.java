package com.reremouse.lab.math;

import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.util.Tuple3;


/**
 * 矩阵操作实现类 / Matrix Operations Implementation Class
 * <p>
 * 本类实现了矩阵的常用操作，包括基本的数学运算、矩阵变换、数据访问等功能。
 * 基于二维float数组实现，提供高效的矩阵计算能力。
 * </p>
 * <p>
 * This class implements common matrix operations including basic mathematical operations,
 * matrix transformations, data access and other functionalities.
 * Based on 2D float array implementation, providing efficient matrix computation capabilities.
 * </p>
 * 
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 *   <li>基本数学运算：加法、减法、乘法、除法 / Basic math operations: add, subtract, multiply, divide</li>
 *   <li>矩阵变换：转置、幂运算、开方 / IMatrix transformations: transpose, power, square root</li>
 *   <li>数据访问：行列访问、元素获取设置 / Data access: row/column access, element get/set</li>
 *   <li>统计运算：行列求和、均值计算 / Statistical operations: row/column sum, mean calculation</li>
 *   <li>数据转换：数组转换、类型转换 / Data conversion: array conversion, type conversion</li>
 * </ul>
 * 
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 // 创建矩阵 / Create matrix
 float[][] data = {{1, 2}, {3, 4}};
 IMatrix matrix = new RereMatrix(data);
 
 // 矩阵运算 / IMatrix operations
 IMatrix result = matrix.add(other).mmul(2.0f);
 
 // 获取行列 / Get rows/columns
 IVector row = matrix.getRow(0);
 IVector col = matrix.getColunm(0);
 }
 * </pre>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereMatrix implements IMatrix {

    /**
     * 矩阵数据存储数组 / Matrix data storage array
     * <p>
     * 使用二维float数组存储矩阵数据，data[i][j]表示第i行第j列的元素
     * Uses 2D float array to store matrix data, data[i][j] represents element at row i, column j
     * </p>
     */
    float[][] data;

    /**
     * 构造函数 / Constructor
     * <p>
     * 使用给定的二维数组创建矩阵实例
     * Creates a matrix instance with the given 2D array
     * </p>
     * 
     * @param data 二维float数组，表示矩阵数据 / 2D float array representing matrix data
     * @throws IllegalArgumentException 如果数据为null或维度不一致 / if data is null or dimensions are inconsistent
     */
    public RereMatrix(float[][] data) {
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
    
    /**
     * 矩阵减法运算（标量） / Matrix subtraction with scalar
     * <p>
     * 矩阵中的每个元素减去标量值
     * Subtracts a scalar value from each element in the matrix
     * </p>
     * 
     * @param scalar 要减去的标量值 / The scalar value to subtract
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    @Override
    public IMatrix sub(float scalar) {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] - scalar;
            }
        }
        return new RereMatrix(result);
    }
    
    /**
     * 矩阵减法运算（矩阵） / Matrix subtraction with another matrix
     * <p>
     * 对应元素相减，要求两个矩阵维度相同
     * Element-wise subtraction, requires both matrices to have the same dimensions
     * </p>
     * 
     * @param other 另一个矩阵 / The other matrix
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't match
     */
    @Override
    public IMatrix sub(IMatrix other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        
        float[][] otherData = other.getData();
        if (data.length != otherData.length || data[0].length != otherData[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] - otherData[i][j];
            }
        }
        return new RereMatrix(result);
    }
    
    /**
     * 矩阵标量乘法运算（float） / Matrix scalar multiplication (float)
     * <p>
     * 矩阵中的每个元素乘以标量值
     * Multiplies each element in the matrix by a scalar value
     * </p>
     * 
     * @param scalar 标量乘数 / The scalar multiplier
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    @Override
    public IMatrix mmul(float scalar) {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] * scalar;
            }
        }
        return new RereMatrix(result);
    }
    
    /**
     * 矩阵标量乘法运算（double） / Matrix scalar multiplication (double)
     * <p>
     * 矩阵中的每个元素乘以标量值，double类型会被转换为float
     * Multiplies each element in the matrix by a scalar value, double is converted to float
     * </p>
     * 
     * @param scalar 标量乘数（double类型） / The scalar multiplier (double type)
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     */
    @Override
    public IMatrix mmul(double scalar) {
        return mmul((float) scalar);
    }
    
    /**
     * 矩阵加法运算 / Matrix addition
     * <p>
     * 对应元素相加，要求两个矩阵维度相同
     * Element-wise addition, requires both matrices to have the same dimensions
     * </p>
     * 
     * @param other 另一个矩阵 / The other matrix
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't match
     */
    @Override
    public IMatrix add(IMatrix other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        
        float[][] otherData = other.getData();
        if (data.length != otherData.length || data[0].length != otherData[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] + otherData[i][j];
            }
        }
        return new RereMatrix(result);
    }
    
    /**
     * 矩阵除法运算 / Matrix division
     * <p>
     * 对应元素相除，要求两个矩阵维度相同
     * Element-wise division, requires both matrices to have the same dimensions
     * </p>
     * 
     * @param other 除数矩阵 / The divisor matrix
     * @return 新的矩阵对象，包含运算结果 / New matrix object containing the result
     * @throws IllegalArgumentException 如果矩阵维度不匹配 / if matrix dimensions don't match
     * @throws ArithmeticException 如果除数为零 / if divisor is zero
     */
    @Override
    public IMatrix divide(IMatrix other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        
        float[][] otherData = other.getData();
        if (data.length != otherData.length || data[0].length != otherData[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }
        
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        final float tolerance = 1e-10f;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (Math.abs(otherData[i][j]) < tolerance) {
                    throw new ArithmeticException("除数不能为零，位置[" + i + "," + j + "] / Divisor cannot be zero at position [" + i + "," + j + "]");
                }
                result[i][j] = data[i][j] / otherData[i][j];
            }
        }
        return new RereMatrix(result);
    }
    
    /**
     * 向量点积运算 / Vector dot product
     * <p>
     * 计算两个列向量的点积，要求两个矩阵都是列向量且维度相同
     * Computes the dot product of two column vectors, requires both matrices to be column vectors with same dimensions
     * </p>
     * 
     * @param other 另一个列向量矩阵 / The other column vector matrix
     * @return 点积结果 / The dot product result
     * @throws IllegalArgumentException 如果不是列向量或维度不匹配 / if not column vectors or dimensions don't match
     */
    @Override
    public float dot(IMatrix other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        
        float[][] otherData = other.getData();
        if (data[0].length != 1 || otherData[0].length != 1) {
            throw new IllegalArgumentException("点积运算需要列向量 / Dot product requires column vectors");
        }
        if (data.length != otherData.length) {
            throw new IllegalArgumentException("向量维度不匹配 / Vector dimensions don't match");
        }
        float result = 0.0f;
        for (int i = 0; i < data.length; i++) {
            result += data[i][0] * otherData[i][0];
        }
        return result;
    }
    
    /**
     * 获取指定列 / Get specified column
     * <p>
     * 返回矩阵的指定列作为列向量
     * Returns the specified column of the matrix as a column vector
     * </p>
     * 
     * @param colIndex 列索引（从0开始） / Column index (0-based)
     * @return 包含指定列数据的列向量矩阵 / Column vector matrix containing the specified column data
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of bounds
     */
    @Override
    public IMatrix getColumn(int colIndex) {
        int rows = data.length;
        float[][] result = new float[rows][1];
        for (int i = 0; i < rows; i++) {
            result[i][0] = data[i][colIndex];
        }
        return new RereMatrix(result);
    }
    
    /**
     * 设置指定列 / Set specified column
     * <p>
     * 将列向量的数据设置到矩阵的指定列
     * Sets the column vector data to the specified column of the matrix
     * </p>
     * 
     * @param colIndex 列索引（从0开始） / Column index (0-based)
     * @param column 列向量矩阵 / Column vector matrix
     * @throws IllegalArgumentException 如果输入不是列向量或维度不匹配 / if input is not a column vector or dimensions don't match
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of bounds
     */
    @Override
    public void putColumn(int colIndex, IMatrix column) {
        if (column == null) {
            throw new IllegalArgumentException("列矩阵不能为null / Column matrix cannot be null");
        }
        if (colIndex < 0 || colIndex >= data[0].length) {
            throw new IndexOutOfBoundsException("列索引超出范围: " + colIndex + " / Column index out of bounds: " + colIndex);
        }
        
        float[][] columnData = column.getData();
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
     * 返回矩阵的多个指定列组成的数组
     * Returns an array of specified columns from the matrix
     * </p>
     * 
     * @param indices 列索引数组 / Array of column indices
     * @return 包含指定列的矩阵数组 / Array of matrices containing the specified columns
     * @throws IndexOutOfBoundsException 如果任何列索引超出范围 / if any column index is out of bounds
     */
    @Override
    public IMatrix[] getColumns(int[] indices) {
        IMatrix[] result = new IMatrix[indices.length];
        for (int i = 0; i < indices.length; i++) {
            result[i] = getColumn(indices[i]);
        }
        return result;
    }
    
    /**
     * 获取指定位置的元素值 / Get element value at specified position
     * <p>
     * 返回矩阵中指定行列位置的元素值
     * Returns the element value at the specified row and column position in the matrix
     * </p>
     * 
     * @param row 行索引（从0开始） / Row index (0-based)
     * @param col 列索引（从0开始） / Column index (0-based)
     * @return 指定位置的元素值 / Element value at the specified position
     * @throws IndexOutOfBoundsException 如果行列索引超出范围 / if row or column index is out of bounds
     */
    @Override
    public float get(int row, int col) {
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
     * 设置矩阵中指定行列位置的元素值
     * Sets the element value at the specified row and column position in the matrix
     * </p>
     * 
     * @param row 行索引（从0开始） / Row index (0-based)
     * @param col 列索引（从0开始） / Column index (0-based)
     * @param value 要设置的值 / Value to set
     * @throws IndexOutOfBoundsException 如果行列索引超出范围 / if row or column index is out of bounds
     */
    @Override
    public void put(int row, int col, float value) {
        if (row < 0 || row >= data.length) {
            throw new IndexOutOfBoundsException("行索引超出范围: " + row + " / Row index out of bounds: " + row);
        }
        if (col < 0 || col >= data[0].length) {
            throw new IndexOutOfBoundsException("列索引超出范围: " + col + " / Column index out of bounds: " + col);
        }
        
        data[row][col] = value;
    }
    
    /**
     * 获取矩阵行数 / Get number of rows
     * <p>
     * 返回矩阵的行数
     * Returns the number of rows in the matrix
     * </p>
     * 
     * @return 矩阵行数 / Number of rows in the matrix
     */
    @Override
    public int getRows() {
        return data.length;
    }
    
    /**
     * 获取矩阵列数 / Get number of columns
     * <p>
     * 返回矩阵的列数
     * Returns the number of columns in the matrix
     * </p>
     * 
     * @return 矩阵列数 / Number of columns in the matrix
     */
    @Override
    public int getColumns() {
        return data[0].length;
    }

    /**
     * 计算行求和 / Calculate row sums
     * <p>
     * 计算矩阵每一列的元素和，返回结果向量
     * Calculates the sum of elements in each column, returns result vector
     * </p>
     * 
     * @return 包含每列元素和的向量 / IVector containing the sum of elements in each column
     */
    @Override
    public IVector rowSums() {
        float[] v = new float[data[0].length];

        for (int col = 0; col < data[0].length; col++) {
            float s = 0;
            for (float[] data1 : data) {
                s += data1[col];
            }
            v[col] = s;
        }
        return IVector.of(v);
    }

    /**
     * 计算行均值 / Calculate row means
     * <p>
     * 计算矩阵每一列的元素平均值，返回结果向量
     * Calculates the mean of elements in each column, returns result vector
     * </p>
     * 
     * @return 包含每列元素平均值的向量 / IVector containing the mean of elements in each column
     */
    @Override
    public IVector rowMeans() {
        var len = data.length;
        IVector vs = this.rowSums();
        return vs.divideByScalar((float) len);
    }

    /**
     * 计算列求和 / Calculate column sums
     * <p>
     * 计算矩阵每一行的元素和，返回结果向量
     * Calculates the sum of elements in each row, returns result vector
     * </p>
     * 
     * @return 包含每行元素和的向量 / IVector containing the sum of elements in each row
     */
    @Override
    public IVector colSums() {
        float[] v = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            v[i] = IVector.of(data[i]).sum();
        }
        return IVector.of(v);
    }

    /**
     * 计算列均值 / Calculate column means
     * <p>
     * 计算矩阵每一行的元素平均值，返回结果向量
     * Calculates the mean of elements in each row, returns result vector
     * </p>
     * 
     * @return 包含每行元素平均值的向量 / IVector containing the mean of elements in each row
     */
    @Override
    public IVector colMeans() {
        var len = data[0].length;
        IVector vs = this.colSums();
        return vs.divideByScalar((float) len);
    }

    /**
     * 获取矩阵数据数组 / Get matrix data array
     * <p>
     * 返回矩阵的内部数据数组引用
     * Returns a reference to the internal data array of the matrix
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
     * 返回矩阵的行数
     * Returns the number of rows in the matrix
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
     * 返回矩阵的列数
     * Returns the number of columns in the matrix
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
     * 返回矩阵的维度信息，包含行数和列数
     * Returns the dimension information of the matrix, including row and column counts
     * </p>
     * 
     * @return 包含行数和列数的数组 [行数, 列数] / Array containing row and column counts [rows, columns]
     */
    @Override
    public int[] shape() {
        int[] s = new int[2];
        s[0] = this.getRowNum();
        s[1] = this.getColNum();
        return s;
    }

    /**
     * 矩阵转置（就地操作） / Matrix transpose (in-place operation)
     * <p>
     * 对当前矩阵执行转置操作，修改原矩阵数据
     * Performs transpose operation on the current matrix, modifying the original matrix data
     * </p>
     * 
     * @return 转置后的矩阵（指向当前对象） / Transposed matrix (pointing to current object)
     */
    @Override
    public IMatrix transpose() {
        var m = this.transposeNew();
        this.data = m.getData();
        return this;
    }

    /**
     * 矩阵转置（创建新对象） / Matrix transpose (create new object)
     * <p>
     * 创建一个新的转置矩阵，不修改原矩阵
     * Creates a new transposed matrix without modifying the original matrix
     * </p>
     * 
     * @return 新的转置矩阵对象 / New transposed matrix object
     */
    @Override
    public IMatrix transposeNew() {
        int rows = this.data.length;
        int cols = this.data[0].length;
        float[][] m = new float[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[j][i] = data[i][j];
            }
        }
        return IMatrix.of(m);
    }

    /**
     * 矩阵转置简写方法 / Matrix transpose shorthand method
     * <p>
     * transpose()方法的简写形式
     * Shorthand form of transpose() method
     * </p>
     * 
     * @return 转置后的矩阵 / Transposed matrix
     * @see #transpose()
     */
    @Override
    public IMatrix t() {
        return this.transpose();
    }

    /**
     * 获取指定列向量 / Get specified column vector
     * <p>
     * 返回矩阵指定列的向量表示
     * Returns the vector representation of the specified column in the matrix
     * </p>
     * 
     * @param i 列索引（从0开始） / Column index (0-based)
     * @return 指定列的向量 / IVector of the specified column
     * @throws IndexOutOfBoundsException 如果列索引超出范围 / if column index is out of bounds
     */
    @Override
    public IVector getColunm(int i) {
        if (i < 0 || i >= data[0].length) {
            throw new IndexOutOfBoundsException("列索引超出范围: " + i + " / Column index out of bounds: " + i);
        }
        
        int len = this.getRowNum();
        float v[] = new float[len];
        for (int j = 0; j < len; j++) {
            v[j] = data[j][i];
        }
        return IVector.of(v);
    }

    /**
     * 获取指定行向量 / Get specified row vector
     * <p>
     * 返回矩阵指定行的向量表示
     * Returns the vector representation of the specified row in the matrix
     * </p>
     * 
     * @param i 行索引（从0开始） / Row index (0-based)
     * @return 指定行的向量 / IVector of the specified row
     * @throws IndexOutOfBoundsException 如果行索引超出范围 / if row index is out of bounds
     */
    @Override
    public IVector getRow(int i) {
        if (i < 0 || i >= data.length) {
            throw new IndexOutOfBoundsException("行索引超出范围: " + i + " / Row index out of bounds: " + i);
        }
        
        return IVector.of(data[i]);
    }
    
    /**
     * 矩阵开方运算 / Matrix square root operation
     * <p>
     * 对矩阵中每个元素进行开方运算
     * Performs square root operation on each element in the matrix
     * </p>
     * 
     * @return 新的矩阵对象，包含开方运算结果 / New matrix object containing the square root results
     * @throws ArithmeticException 如果元素值为负数 / if any element value is negative
     */
    @Override
    public IMatrix sqrt() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.sqrt(data[i][j]);
            }
        }
        return new RereMatrix(result);
    }
    
    /**
     * 矩阵幂运算 / Matrix power operation
     * <p>
     * 对矩阵中每个元素进行幂运算
     * Performs power operation on each element in the matrix
     * </p>
     * 
     * @param power 幂指数 / Power exponent
     * @return 新的矩阵对象，包含幂运算结果 / New matrix object containing the power operation results
     */
    @Override
    public IMatrix pow(float power) {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.pow(data[i][j], power);
            }
        }
        return new RereMatrix(result);
    }

    /**
     * 特征分解，返回的特征值按大小排列，返回的矩阵的列为各个特征向量，与特征值的顺序对应
     *
     * @return 返回特征值和特征向量，其中返回的向量中包含所有特征值，返回的矩阵的列为各个特征向量
     */
    @Override
    public Tuple2<IVector,IMatrix> eigen() {
        // 检查是否为方阵
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("特征分解需要方阵 / Eigendecomposition requires square matrix");
        }
        
        int n = data.length;
        float[][] A = new float[n][n];
        // 复制矩阵数据
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, A[i], 0, n);
        }
        
        // 使用QR算法计算特征值和特征向量
        return new RereMatrix(A).qrEigenDecomposition();
    }

    
    /**
     * 奇异值分解
     *
     * @return 返回U, S, V^T的分解，其中U和V^T是正交矩阵，S是奇异值向量
     */
    @Override
    public Tuple3<IMatrix, IVector, IMatrix> svd() {
        int m = data.length;    // 行数
        int n = data[0].length; // 列数
        
        // 计算A^T * A
        IMatrix AT = this.transposeNew();
        IMatrix ATA = ((RereMatrix)AT).mmul(this);
        
        // 对A^T * A进行特征分解得到V和奇异值的平方
        Tuple2<IVector, IMatrix> eigenResult = ((RereMatrix)ATA).eigen();
        IVector eigenValues = eigenResult._1;
        IMatrix V = eigenResult._2;
        
        // 计算奇异值（特征值的平方根）
        float[] singularValues = new float[Math.min(m, n)];
        for (int i = 0; i < singularValues.length; i++) {
            singularValues[i] = (float) Math.sqrt(Math.max(0, eigenValues.get(i)));
        }
        
        // 按奇异值大小降序排列
        int[] indices = new int[singularValues.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        
        // 简单的冒泡排序
        for (int i = 0; i < singularValues.length - 1; i++) {
            for (int j = 0; j < singularValues.length - 1 - i; j++) {
                if (singularValues[j] < singularValues[j + 1]) {
                    // 交换奇异值
                    float temp = singularValues[j];
                    singularValues[j] = singularValues[j + 1];
                    singularValues[j + 1] = temp;
                    
                    // 交换索引
                    int tempIdx = indices[j];
                    indices[j] = indices[j + 1];
                    indices[j + 1] = tempIdx;
                }
            }
        }
        
        // 重新排列V的列
        float[][] VData = new float[n][n];
        for (int i = 0; i < n; i++) {
            IVector vi = V.getColunm(indices[i]);
            for (int j = 0; j < n; j++) {
                VData[j][i] = vi.get(j);
            }
        }
        IMatrix sortedV = new RereMatrix(VData);
        
        // 计算U = A * V * S^(-1)
        float[][] UData = new float[m][Math.min(m, n)];
        for (int i = 0; i < Math.min(m, n); i++) {
                         if (singularValues[i] > 1e-10) { // 避免除零
                 IVector vi = sortedV.getColunm(i);
                 // 将向量转换为列矩阵
                 float[][] colData = new float[vi.length()][1];
                 for (int k = 0; k < vi.length(); k++) {
                     colData[k][0] = vi.get(k);
                 }
                 IMatrix viMatrix = IMatrix.of(colData);
                 IMatrix Avi = ((RereMatrix)this).mmul(viMatrix);
                 for (int j = 0; j < m; j++) {
                     UData[j][i] = Avi.get(j, 0) / singularValues[i];
                 }
             }
        }
        IMatrix U = new RereMatrix(UData);
        
        // 返回U, S, V^T
        return new Tuple3<>(U, IVector.of(singularValues), sortedV.transposeNew());
    }
    
    /**
     * QR算法特征分解的辅助方法
     * 
     * @return 特征值和特征向量
     */
    @Override
    public Tuple2<IVector, IMatrix> qrEigenDecomposition() {
        float[][] A=this.data;
        int n = A.length;
        
        final int maxIterations = 100;
        final float tolerance = 1e-6f;
        
        // 初始化特征向量矩阵为单位矩阵
        float[][] eigenvectors = new float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                eigenvectors[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        
        // QR迭代
        for (int iter = 0; iter < maxIterations; iter++) {
            // QR分解
            Tuple2<IMatrix, IMatrix> qr = new RereMatrix(A).qr();
            IMatrix Q = qr._1;
            IMatrix R = qr._2;
            
            // A = R * Q
            IMatrix newA = ((RereMatrix)R).mmul(Q);
            float[][] newAData = newA.getData();
            
            // 更新特征向量
            IMatrix currentEigenvectors = new RereMatrix(eigenvectors);
            IMatrix updatedEigenvectors = ((RereMatrix)currentEigenvectors).mmul(Q);
            eigenvectors = updatedEigenvectors.getData();
            
            // 检查收敛性
            float offDiagonalSum = 0;
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
        float[] eigenvalues = new float[n];
        for (int i = 0; i < n; i++) {
            eigenvalues[i] = A[i][i];
        }
        
        // 按特征值大小降序排列
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - 1 - i; j++) {
                if (eigenvalues[j] < eigenvalues[j + 1]) {
                    // 交换特征值
                    float temp = eigenvalues[j];
                    eigenvalues[j] = eigenvalues[j + 1];
                    eigenvalues[j + 1] = temp;
                    
                    // 交换对应的特征向量列
                    for (int k = 0; k < n; k++) {
                        float tempVec = eigenvectors[k][j];
                        eigenvectors[k][j] = eigenvectors[k][j + 1];
                        eigenvectors[k][j + 1] = tempVec;
                    }
                }
            }
        }
        
        return new Tuple2<>(IVector.of(eigenvalues), new RereMatrix(eigenvectors));
    }
    
    /**
     * QR分解
     * 
     * @param matrix 输入矩阵
     * @return Q和R矩阵
     */
    @Override
    public Tuple2<IMatrix, IMatrix> qr() {
        float[][] A = this.getData();
        int m = A.length;
        int n = A[0].length;
        
        float[][] Q = new float[m][n];
        float[][] R = new float[n][n];
        
        // Gram-Schmidt过程
        for (int j = 0; j < n; j++) {
            // 复制第j列到Q
            for (int i = 0; i < m; i++) {
                Q[i][j] = A[i][j];
            }
            
            // 正交化
            for (int k = 0; k < j; k++) {
                float dotProduct = 0;
                for (int i = 0; i < m; i++) {
                    dotProduct += Q[i][k] * A[i][j];
                }
                R[k][j] = dotProduct;
                
                for (int i = 0; i < m; i++) {
                    Q[i][j] -= dotProduct * Q[i][k];
                }
            }
            
            // 标准化
            float norm = 0;
            for (int i = 0; i < m; i++) {
                norm += Q[i][j] * Q[i][j];
            }
            norm = (float) Math.sqrt(norm);
            R[j][j] = norm;
            
            if (norm > 1e-10) { // 避免除零
                for (int i = 0; i < m; i++) {
                    Q[i][j] /= norm;
                }
            }
        }
        
        return new Tuple2<>(new RereMatrix(Q), new RereMatrix(R));
    }
    
    /**
     * 矩阵乘法运算 / IMatrix multiplication
     * 
     * @param other 另一个矩阵 / The other matrix
     * @return 矩阵乘法结果 / IMatrix multiplication result  
     */
    @Override
    public IMatrix mmul(IMatrix other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        
        float[][] otherData = other.getData();
        int m = data.length;          // 当前矩阵行数
        int n = data[0].length;       // 当前矩阵列数
        int p = otherData[0].length;  // 另一个矩阵列数
        
        if (n != otherData.length) {
            throw new IllegalArgumentException("矩阵维度不匹配进行乘法运算 / Matrix dimensions don't match for multiplication");
        }
        
        float[][] result = new float[m][p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                float sum = 0;
                for (int k = 0; k < n; k++) {
                    sum += data[i][k] * otherData[k][j];
                }
                result[i][j] = sum;
            }
        }
        
        return new RereMatrix(result);
    }
    
    
    
    

    /**
     * 矩阵转换为一维数组 / Convert matrix to 1D array
     * <p>
     * 将矩阵转换为一维float数组。对于列向量，返回该列的数据；
     * 对于普通矩阵，按行优先顺序展开。
     * </p>
     * <p>
     * Converts the matrix to a 1D float array. For column vectors, returns the column data;
     * For regular matrices, flattens in row-major order.
     * </p>
     * 
     * @return 一维float数组 / 1D float array
     */
    @Override
    public float[] toArray() {
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
            for (int i = 0; i < data.length; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    result[index++] = data[i][j];
                }
            }
            return result;
        }
    }
    
    /**
     * 矩阵转换为double数组 / Convert matrix to double array
     * <p>
     * 将矩阵转换为一维double数组，用于与其他库的兼容性
     * Converts the matrix to a 1D double array for compatibility with other libraries
     * </p>
     * 
     * @return 一维double数组 / 1D double array
     * @see #toArray()
     */
    @Override
    public double[] toDoubleArray() {
        float[] floatArray = toArray();
        double[] result = new double[floatArray.length];
        for (int i = 0; i < floatArray.length; i++) {
            result[i] = floatArray[i];
        }
        return result;
    }

    /**
     * 矩阵指数运算 / Matrix exponential operation
     * <p>
     * 对矩阵中每个元素进行指数运算（e^x）
     * Performs exponential operation (e^x) on each element in the matrix
     * </p>
     * 
     * @return 新的矩阵对象，包含指数运算结果 / New matrix object containing exponential operation results
     */
    @Override
    public IMatrix exp() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.exp(data[i][j]);
            }
        }
        return new RereMatrix(result);
    }

    /**
     * 矩阵自然对数运算 / Matrix natural logarithm operation
     * <p>
     * 对矩阵中每个元素进行自然对数运算（ln(x)）
     * Performs natural logarithm operation (ln(x)) on each element in the matrix
     * </p>
     * 
     * @return 新的矩阵对象，包含对数运算结果 / New matrix object containing logarithm operation results
     * @throws ArithmeticException 如果元素值小于等于0 / if any element value is less than or equal to 0
     */
    @Override
    public IMatrix log() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] <= 0) {
                    throw new ArithmeticException("对数运算的元素值必须大于0 / Element value for logarithm must be greater than 0");
                }
                result[i][j] = (float) Math.log(data[i][j]);
            }
        }
        return new RereMatrix(result);
    }

    /**
     * 计算矩阵的Frobenius范数 / Compute Frobenius norm of the matrix
     * <p>
     * 计算矩阵的Frobenius范数，即所有元素平方和的开方
     * Computes the Frobenius norm of the matrix, which is the square root of the sum of squares of all elements
     * </p>
     *
     * @return Frobenius范数 / Frobenius norm
     */
    @Override
    public float frobeniusNorm() {
        float sum = 0.0f;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                sum += data[i][j] * data[i][j];
            }
        }
        return (float) Math.sqrt(sum);
    }

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
    @Override
    public float frobeniusDistance(IMatrix other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        
        if (data.length != other.getRowNum() || data[0].length != other.getColNum()) {
            throw new IllegalArgumentException("矩阵维度不匹配 / Matrix dimensions don't match");
        }
        
        IMatrix diff = this.sub(other);
        return diff.frobeniusNorm();
    }

    /**
     * 矩阵按行归一化 / Row-wise normalization of matrix
     * <p>
     * 对矩阵的每一行进行L2归一化
     * Performs L2 normalization on each row of the matrix
     * </p>
     *
     * @return 归一化后的矩阵 / Normalized matrix
     */
    @Override
    public IMatrix normalizeRows() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            IVector row = this.getRow(i);
            float norm = row.norm2();
            
            if (norm > 0) {
                IVector normalizedRow = row.divideByScalar(norm);
                float[] rowData = normalizedRow.getData();
                System.arraycopy(rowData, 0, result[i], 0, cols);
            } else {
                System.arraycopy(data[i], 0, result[i], 0, cols);
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵按列归一化 / Column-wise normalization of matrix
     * <p>
     * 对矩阵的每一列进行L2归一化
     * Performs L2 normalization on each column of the matrix
     * </p>
     *
     * @return 归一化后的矩阵 / Normalized matrix
     */
    @Override
    public IMatrix normalizeColumns() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        // 对每一列进行L2归一化
        for (int j = 0; j < cols; j++) {
            // 计算列的L2范数
            float norm = 0.0f;
            for (int i = 0; i < rows; i++) {
                norm += data[i][j] * data[i][j];
            }
            norm = (float) Math.sqrt(norm);
            
            // 归一化该列
            if (norm > 1e-10) { // 避免除零
                for (int i = 0; i < rows; i++) {
                    result[i][j] = data[i][j] / norm;
                }
            } else {
                for (int i = 0; i < rows; i++) {
                    result[i][j] = 0.0f;
                }
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵数据中心化 / Matrix data centering
     * <p>
     * 对矩阵的每一列减去该列的均值，实现数据中心化
     * Subtracts the mean of each column from the column elements, implementing data centering
     * </p>
     *
     * @return 中心化后的矩阵 / Centered matrix
     */
    @Override
    public IMatrix center() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        // 计算每列的均值
        IVector columnMeans = this.rowMeans();
        
        // 对每列减去均值
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = data[i][j] - columnMeans.get(j);
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 计算协方差矩阵 / Compute covariance matrix
     * <p>
     * 计算数据矩阵的协方差矩阵，假设每行是一个样本，每列是一个特征
     * Computes the covariance matrix of the data matrix, assuming each row is a sample and each column is a feature
     * </p>
     *
     * @return 协方差矩阵 / Covariance matrix
     */
    @Override
    public IMatrix covariance() {
        // 先进行数据中心化
        IMatrix centered = this.center();
        return centered.covarianceFromCentered();
    }

    /**
     * 计算协方差矩阵（已中心化数据） / Compute covariance matrix (for centered data)
     * <p>
     * 对于已经中心化的数据，直接计算协方差矩阵 = (X^T * X) / (n-1)
     * For already centered data, directly compute covariance matrix = (X^T * X) / (n-1)
     * </p>
     *
     * @return 协方差矩阵 / Covariance matrix
     */
    @Override
    public IMatrix covarianceFromCentered() {
        int n = data.length; // 样本数
        
        // 计算 X^T * X
        IMatrix transposed = this.transposeNew();
        IMatrix product = ((RereMatrix) transposed).mmul(this);
        
        // 除以 (n-1) 得到协方差矩阵
        return product.mmul(1.0f / (n - 1));
    }

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
    @Override
    public IMatrix inv() {
        // 检查是否为方阵
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能求逆 / Only square matrices can be inverted");
        }
        
        int n = data.length;
        
        // 创建增广矩阵 [A | I]，其中I是单位矩阵
        float[][] augmented = new float[n][2 * n];
        
        // 初始化增广矩阵
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = data[i][j];  // 左半部分是原矩阵A
                augmented[i][j + n] = (i == j) ? 1.0f : 0.0f;  // 右半部分是单位矩阵I
            }
        }
        
        // 高斯-约旦消元法
        final float tolerance = 1e-10f;
        
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
                float[] temp = augmented[i];
                augmented[i] = augmented[pivotRow];
                augmented[pivotRow] = temp;
            }
            
            // 将主元归一化
            float pivot = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }
            
            // 消除当前列的其他元素
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    float factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // 提取逆矩阵（增广矩阵的右半部分）
        float[][] inverse = new float[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j + n];
            }
        }
        
        return new RereMatrix(inverse);
    }

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
    @Override
    public IMatrix pinv() {
        final float tolerance = 1e-10f;
        
        // 进行奇异值分解：A = U * S * V^T
        Tuple3<IMatrix, IVector, IMatrix> svdResult = this.svd();
        IMatrix U = svdResult._1;           // 左奇异向量矩阵
        IVector singularValues = svdResult._2;  // 奇异值向量
        IMatrix VT = svdResult._3;          // 右奇异向量转置矩阵
        
        // 获取矩阵的维度信息
        int originalRows = this.getRowNum();
        int originalCols = this.getColNum();
        int singularValuesLength = singularValues.length();
        
        // 计算奇异值的伪逆
        IVector pseudoSingularValues = IVector.zeros(singularValuesLength);
        
        for (int i = 0; i < singularValuesLength; i++) {
            float sv = singularValues.get(i);
            if (Math.abs(sv) > tolerance) {
                pseudoSingularValues.set(i, 1.0f / sv);  // 非零奇异值的倒数
            } else {
                pseudoSingularValues.set(i, 0.0f);       // 零奇异值保持为零
            }
        }
        
        // 计算伪逆：A⁺ = V * Σ⁺ * U^T
        // 由于我们的SVD返回的U可能是截断的，我们需要更仔细地处理维度
        IMatrix V = VT.transposeNew();  // V = (V^T)^T
        
        // 创建结果矩阵：A⁺的维度应该是 originalCols x originalRows
        IMatrix pseudoInverse = IMatrix.zeros(originalCols, originalRows);
        
        // 逐元素计算伪逆：A⁺[i,j] = Σ(k=0 to rank-1) V[i,k] * (1/σ[k]) * U[j,k]
        for (int i = 0; i < originalCols; i++) {
            for (int j = 0; j < originalRows; j++) {
                float sum = 0.0f;
                for (int k = 0; k < singularValuesLength; k++) {
                    float vValue = (k < V.getColNum()) ? V.get(i, k) : 0.0f;
                    float uValue = (k < U.getColNum()) ? U.get(j, k) : 0.0f;
                    float sigmaInv = pseudoSingularValues.get(k);
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
    public static IMatrix loadFromFile(String path) {
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
            
            return new RereMatrix(data);
            
        } catch (java.io.IOException e) {
            throw new RuntimeException("文件读取失败：" + e.getMessage() + " / File reading failed: " + e.getMessage(), e);
        }
    }
    
    /**
     * 将矩阵数据保存到本地指定位置 / Save matrix data to specified local path
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
     * 将两个矩阵在水平方向上连接，要求行数相同
     * Concatenates two matrices horizontally, requires same number of rows
     * </p>
     *
     * @param other 要连接的另一个矩阵 / The other matrix to concatenate
     * @return 连接后的矩阵 / Concatenated matrix
     * @throws IllegalArgumentException 如果矩阵行数不匹配 / if matrix row counts don't match
     */
    @Override
    public IMatrix hstack(IMatrix other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        
        if (data.length != other.getRowNum()) {
            throw new IllegalArgumentException("矩阵行数不匹配 / Matrix row counts don't match");
        }
        
        int rows = data.length;
        int cols1 = data[0].length;
        int cols2 = other.getColNum();
        int totalCols = cols1 + cols2;
        
        float[][] result = new float[rows][totalCols];
        
        // 复制第一个矩阵
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, result[i], 0, cols1);
        }
        
        // 复制第二个矩阵
        float[][] otherData = other.getData();
        for (int i = 0; i < rows; i++) {
            System.arraycopy(otherData[i], 0, result[i], cols1, cols2);
        }
        
        return new RereMatrix(result);
    }

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
    @Override
    public IMatrix vstack(IMatrix other) {
        if (other == null) {
            throw new IllegalArgumentException("输入矩阵不能为null / Input matrix cannot be null");
        }
        
        if (data[0].length != other.getColNum()) {
            throw new IllegalArgumentException("矩阵列数不匹配 / Matrix column counts don't match");
        }
        
        int rows1 = data.length;
        int rows2 = other.getRowNum();
        int cols = data[0].length;
        int totalRows = rows1 + rows2;
        
        float[][] result = new float[totalRows][cols];
        
        // 复制第一个矩阵
        for (int i = 0; i < rows1; i++) {
            System.arraycopy(data[i], 0, result[i], 0, cols);
        }
        
        // 复制第二个矩阵
        float[][] otherData = other.getData();
        for (int i = 0; i < rows2; i++) {
            System.arraycopy(otherData[i], 0, result[rows1 + i], 0, cols);
        }
        
        return new RereMatrix(result);
    }

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
    @Override
    public IMatrix[] hsplit(int[] indices) {
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
        IMatrix[] result = new IMatrix[numSplits];
        
        int startCol = 0;
        for (int i = 0; i < numSplits; i++) {
            int endCol = (i < sortedIndices.length) ? sortedIndices[i] : data[0].length;
            int cols = endCol - startCol;
            
            float[][] subMatrix = new float[data.length][cols];
            for (int row = 0; row < data.length; row++) {
                System.arraycopy(data[row], startCol, subMatrix[row], 0, cols);
            }
            
            result[i] = new RereMatrix(subMatrix);
            startCol = endCol;
        }
        
        return result;
    }

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
    @Override
    public IMatrix[] vsplit(int[] indices) {
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
        IMatrix[] result = new IMatrix[numSplits];
        
        int startRow = 0;
        for (int i = 0; i < numSplits; i++) {
            int endRow = (i < sortedIndices.length) ? sortedIndices[i] : data.length;
            int rows = endRow - startRow;
            
            float[][] subMatrix = new float[rows][data[0].length];
            for (int row = 0; row < rows; row++) {
                System.arraycopy(data[startRow + row], 0, subMatrix[row], 0, data[0].length);
            }
            
            result[i] = new RereMatrix(subMatrix);
            startRow = endRow;
        }
        
        return result;
    }

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
    @Override
    public IMatrix reshape(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0 / Rows and columns must be greater than 0");
        }
        
        int totalElements = data.length * data[0].length;
        if (rows * cols != totalElements) {
            throw new IllegalArgumentException("新维度与元素总数不匹配 / New dimensions don't match total element count");
        }
        
        float[] flatData = toArray();
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = flatData[i * cols + j];
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵复制 / Matrix copy
     * <p>
     * 创建当前矩阵的深拷贝
     * Creates a deep copy of the current matrix
     * </p>
     *
     * @return 矩阵的副本 / Copy of the matrix
     */
    @Override
    public IMatrix copy() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] copyData = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, copyData[i], 0, cols);
        }
        
        return new RereMatrix(copyData);
    }

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
    @Override
    public float det() {
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
            Tuple2<IMatrix, IMatrix> lu = this.lu();
            IMatrix L = lu._1;
            IMatrix U = lu._2;
            
            float detL = 1.0f;
            float detU = 1.0f;
            
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
    private float detRecursive(float[][] matrix) {
        int n = matrix.length;
        if (n == 1) {
            return matrix[0][0];
        }
        if (n == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }
        
        float det = 0.0f;
        for (int j = 0; j < n; j++) {
            float[][] minor = new float[n - 1][n - 1];
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
     * 计算方阵的迹（对角线元素之和）
     * Computes the trace of a square matrix (sum of diagonal elements)
     * </p>
     *
     * @return 矩阵迹 / Matrix trace
     * @throws IllegalArgumentException 如果矩阵不是方阵 / if matrix is not square
     */
    @Override
    public float trace() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能计算迹 / Only square matrices can compute trace");
        }
        
        float trace = 0.0f;
        for (int i = 0; i < data.length; i++) {
            trace += data[i][i];
        }
        return trace;
    }

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
    @Override
    public float cond() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能计算条件数 / Only square matrices can compute condition number");
        }
        
        try {
            float normA = this.frobeniusNorm();
            IMatrix invA = this.inv();
            float normInvA = invA.frobeniusNorm();
            return normA * normInvA;
        } catch (Exception e) {
            // 如果求逆失败，返回一个很大的数表示条件数很大
            return Float.MAX_VALUE;
        }
    }

    /**
     * 计算矩阵秩 / Compute matrix rank
     * <p>
     * 计算矩阵的秩，即线性无关的行或列的最大数量
     * Computes the rank of a matrix, the maximum number of linearly independent rows or columns
     * </p>
     *
     * @return 矩阵秩 / Matrix rank
     */
    @Override
    public int rank() {
        try {
            // 使用SVD计算秩
            Tuple3<IMatrix, IVector, IMatrix> svdResult = this.svd();
            IVector singularValues = svdResult._2;
            
            final float tolerance = 1e-10f;
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
     * 使用高斯消元法计算矩阵秩的辅助方法 / Helper method for computing rank using Gaussian elimination
     */
    private int rankByGaussianElimination() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] matrix = new float[rows][cols];
        
        // 复制数据
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, matrix[i], 0, cols);
        }
        
        int rank = 0;
        final float tolerance = 1e-10f;
        
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
                    float[] temp = matrix[rank];
                    matrix[rank] = matrix[pivotRow];
                    matrix[pivotRow] = temp;
                }
                
                // 消元
                for (int row = rank + 1; row < rows; row++) {
                    float factor = matrix[row][col] / matrix[rank][col];
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
     * 对矩阵中每个元素取绝对值
     * Takes the absolute value of each element in the matrix
     * </p>
     *
     * @return 包含绝对值的新矩阵 / New matrix containing absolute values
     */
    @Override
    public IMatrix abs() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = Math.abs(data[i][j]);
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵符号函数 / Matrix sign function
     * <p>
     * 对矩阵中每个元素应用符号函数：正数为1，负数为-1，零为0
     * Applies sign function to each element: 1 for positive, -1 for negative, 0 for zero
     * </p>
     *
     * @return 包含符号值的新矩阵 / New matrix containing sign values
     */
    @Override
    public IMatrix sign() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (data[i][j] > 0) {
                    result[i][j] = 1.0f;
                } else if (data[i][j] < 0) {
                    result[i][j] = -1.0f;
                } else {
                    result[i][j] = 0.0f;
                }
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵正弦函数 / Matrix sine function
     * <p>
     * 对矩阵中每个元素计算正弦值
     * Computes sine value for each element in the matrix
     * </p>
     *
     * @return 包含正弦值的新矩阵 / New matrix containing sine values
     */
    @Override
    public IMatrix sin() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.sin(data[i][j]);
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵余弦函数 / Matrix cosine function
     * <p>
     * 对矩阵中每个元素计算余弦值
     * Computes cosine value for each element in the matrix
     * </p>
     *
     * @return 包含余弦值的新矩阵 / New matrix containing cosine values
     */
    @Override
    public IMatrix cos() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.cos(data[i][j]);
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵正切函数 / Matrix tangent function
     * <p>
     * 对矩阵中每个元素计算正切值
     * Computes tangent value for each element in the matrix
     * </p>
     *
     * @return 包含正切值的新矩阵 / New matrix containing tangent values
     */
    @Override
    public IMatrix tan() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.tan(data[i][j]);
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵双曲正弦函数 / Matrix hyperbolic sine function
     * <p>
     * 对矩阵中每个元素计算双曲正弦值
     * Computes hyperbolic sine value for each element in the matrix
     * </p>
     *
     * @return 包含双曲正弦值的新矩阵 / New matrix containing hyperbolic sine values
     */
    @Override
    public IMatrix sinh() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.sinh(data[i][j]);
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵双曲余弦函数 / Matrix hyperbolic cosine function
     * <p>
     * 对矩阵中每个元素计算双曲余弦值
     * Computes hyperbolic cosine value for each element in the matrix
     * </p>
     *
     * @return 包含双曲余弦值的新矩阵 / New matrix containing hyperbolic cosine values
     */
    @Override
    public IMatrix cosh() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.cosh(data[i][j]);
            }
        }
        
        return new RereMatrix(result);
    }

    /**
     * 矩阵双曲正切函数 / Matrix hyperbolic tangent function
     * <p>
     * 对矩阵中每个元素计算双曲正切值
     * Computes hyperbolic tangent value for each element in the matrix
     * </p>
     *
     * @return 包含双曲正切值的新矩阵 / New matrix containing hyperbolic tangent values
     */
    @Override
    public IMatrix tanh() {
        int rows = data.length;
        int cols = data[0].length;
        float[][] result = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = (float) Math.tanh(data[i][j]);
            }
        }
        
        return new RereMatrix(result);
    }

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
    @Override
    public Tuple2<IMatrix, IMatrix> lu() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能进行LU分解 / Only square matrices can perform LU decomposition");
        }
        
        int n = data.length;
        float[][] A = new float[n][n];
        float[][] L = new float[n][n];
        float[][] U = new float[n][n];
        
        // 复制数据
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, A[i], 0, n);
        }
        
        // 初始化L为单位下三角矩阵
        for (int i = 0; i < n; i++) {
            L[i][i] = 1.0f;
        }
        
        final float tolerance = 1e-10f;
        
        for (int k = 0; k < n; k++) {
            // 计算U的第k行
            for (int j = k; j < n; j++) {
                float sum = 0.0f;
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
                float sum = 0.0f;
                for (int m = 0; m < k; m++) {
                    sum += L[i][m] * U[m][k];
                }
                L[i][k] = (A[i][k] - sum) / U[k][k];
            }
        }
        
        return new Tuple2<>(new RereMatrix(L), new RereMatrix(U));
    }

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
    @Override
    public IMatrix cholesky() {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能进行Cholesky分解 / Only square matrices can perform Cholesky decomposition");
        }
        
        int n = data.length;
        float[][] L = new float[n][n];
        
        // 检查对称性
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (Math.abs(data[i][j] - data[j][i]) > 1e-10f) {
                    throw new IllegalArgumentException("矩阵必须是对称的 / Matrix must be symmetric");
                }
            }
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0.0f;
                
                if (j == i) {
                    for (int k = 0; k < j; k++) {
                        sum += L[j][k] * L[j][k];
                    }
                    L[j][j] = (float) Math.sqrt(data[j][j] - sum);
                    
                    if (L[j][j] <= 0) {
                        throw new IllegalArgumentException("矩阵不是正定的 / Matrix is not positive definite");
                    }
                } else {
                    for (int k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (data[i][j] - sum) / L[j][j];
                }
            }
        }
        
        return new RereMatrix(L);
    }

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
    @Override
    public IVector solve(IVector b) {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能求解线性方程组 / Only square matrices can solve linear systems");
        }
        
        if (b.length() != data.length) {
            throw new IllegalArgumentException("向量维度与矩阵行数不匹配 / Vector dimension doesn't match matrix rows");
        }
        
        // 使用LU分解求解
        Tuple2<IMatrix, IMatrix> lu = this.lu();
        IMatrix L = lu._1;
        IMatrix U = lu._2;
        
        int n = data.length;
        float[] x = new float[n];
        float[] y = new float[n];
        
        // 前向代入：Ly = b
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < i; j++) {
                sum += L.get(i, j) * y[j];
            }
            y[i] = (b.get(i) - sum) / L.get(i, i);
        }
        
        // 后向代入：Ux = y
        for (int i = n - 1; i >= 0; i--) {
            float sum = 0.0f;
            for (int j = i + 1; j < n; j++) {
                sum += U.get(i, j) * x[j];
            }
            x[i] = (y[i] - sum) / U.get(i, i);
        }
        
        return IVector.of(x);
    }

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
    @Override
    public IMatrix solve(IMatrix B) {
        if (data.length != data[0].length) {
            throw new IllegalArgumentException("只有方阵才能求解线性方程组 / Only square matrices can solve linear systems");
        }
        
        if (B.getRowNum() != data.length) {
            throw new IllegalArgumentException("右侧矩阵行数与系数矩阵行数不匹配 / Right-hand side matrix rows don't match coefficient matrix rows");
        }
        
        int n = data.length;
        int m = B.getColNum();
        float[][] X = new float[n][m];
        
        // 对每一列分别求解
        for (int col = 0; col < m; col++) {
            float[] b = new float[n];
            for (int row = 0; row < n; row++) {
                b[row] = B.get(row, col);
            }
            
            IVector x = this.solve(IVector.of(b));
            for (int row = 0; row < n; row++) {
                X[row][col] = x.get(row);
            }
        }
        
        return new RereMatrix(X);
    }

    /**
     * 矩阵最大值 / Matrix maximum value
     * <p>
     * 返回矩阵中的最大元素值
     * Returns the maximum element value in the matrix
     * </p>
     *
     * @return 最大元素值 / Maximum element value
     */
    @Override
    public float max() {
        float max = data[0][0];
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
     * 返回矩阵中的最小元素值
     * Returns the minimum element value in the matrix
     * </p>
     *
     * @return 最小元素值 / Minimum element value
     */
    @Override
    public float min() {
        float min = data[0][0];
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
     * 返回矩阵中所有元素的总和
     * Returns the sum of all elements in the matrix
     * </p>
     *
     * @return 元素总和 / Sum of all elements
     */
    @Override
    public float sum() {
        float sum = 0.0f;
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                sum += data[i][j];
            }
        }
        return sum;
    }

    /**
     * 矩阵元素均值 / Matrix element mean
     * <p>
     * 返回矩阵中所有元素的平均值
     * Returns the mean of all elements in the matrix
     * </p>
     *
     * @return 元素均值 / Mean of all elements
     */
    @Override
    public float mean() {
        return sum() / (data.length * data[0].length);
    }

    /**
     * 矩阵元素标准差 / Matrix element standard deviation
     * <p>
     * 返回矩阵中所有元素的标准差
     * Returns the standard deviation of all elements in the matrix
     * </p>
     *
     * @return 元素标准差 / Standard deviation of all elements
     */
    @Override
    public float std() {
        return (float) Math.sqrt(var());
    }

    /**
     * 矩阵元素方差 / Matrix element variance
     * <p>
     * 返回矩阵中所有元素的方差
     * Returns the variance of all elements in the matrix
     * </p>
     *
     * @return 元素方差 / Variance of all elements
     */
    @Override
    public float var() {
        float mean = mean();
        float sumSquaredDiff = 0.0f;
        int totalElements = data.length * data[0].length;
        
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                float diff = data[i][j] - mean;
                sumSquaredDiff += diff * diff;
            }
        }
        
        return sumSquaredDiff / totalElements;
    }

}
