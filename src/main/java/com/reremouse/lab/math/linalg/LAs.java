package com.reremouse.lab.math.linalg;

import java.util.Random;
import java.util.List;
import java.util.ArrayList;
import com.reremouse.lab.math.RereMathUtil;

/**
 * 线性代数工厂类
 * @author lteb2
 */
public class LAs {
    
    // ========== 矩阵静态方法 / Matrix Static Methods ==========
    
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
    public static IMatrix matrix(float[][] data) {
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
    public static IMatrix matrix(List<float[]> data) {
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
    public static IMatrix matrix(IVector[] data){
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

        IMatrix matrixA = LAs.zeros(rows, cols);
        IMatrix matrixB = LAs.zeros(rows, cols);

        // 将列向量组合成矩阵 / Combine column vectors into matrices
        for (int i = 0; i < cols; i++) {
            matrixA.putColumn(i, a[i]);
            matrixB.putColumn(i, b[i]);
        }

        // 计算平均值 / Calculate average
        IMatrix result = LAs.zeros(rows, cols);
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.put(row, col, (matrixA.get(row, col) + matrixB.get(row, col)) / 2);
            }
        }

        return result;
    }

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

    // ========== 向量静态方法 / Vector Static Methods ==========
    
    /**
     * 向量工厂方法（float数组） / Vector factory method (float array)
     * <p>
     * 使用给定的float数组创建向量实例 Creates a vector instance with the given float array
     * </p>
     *
     * @param data float数组，表示向量数据 / float array representing vector data
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector vec(float[] data) {
        return new RereVector(data);
    }

    /**
     * 向量工厂方法（Float包装类数组） / Vector factory method (Float wrapper array)
     * <p>
     * 使用给定的Float包装类数组创建向量实例 Creates a vector instance with the given Float
     * wrapper array
     * </p>
     *
     * @param data Float包装类数组 / Float wrapper array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector vec(Float[] data) {
        return vec(RereMathUtil.toPrimitive(data));
    }

    /**
     * 向量工厂方法（double数组） / Vector factory method (double array)
     * <p>
     * 使用给定的double数组创建向量实例，自动转换为float Creates a vector instance with the given
     * double array, automatically converted to float
     * </p>
     *
     * @param data double数组 / double array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector vec(double[] data) {
        return vec(RereMathUtil.doubleToFloat(data));
    }

    /**
     * 向量工厂方法（Double包装类数组） / Vector factory method (Double wrapper array)
     * <p>
     * 使用给定的Double包装类数组创建向量实例 Creates a vector instance with the given Double
     * wrapper array
     * </p>
     *
     * @param data Double包装类数组 / Double wrapper array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector vec(Double[] data) {
        return vec(RereMathUtil.toPrimitive(data));
    }

    /**
     * 向量工厂方法（int数组） / Vector factory method (int array)
     * <p>
     * 使用给定的int数组创建向量实例，自动转换为float Creates a vector instance with the given int
     * array, automatically converted to float
     * </p>
     *
     * @param data int数组 / int array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector vec(int[] data) {
        return vec(RereMathUtil.intToFloat(data));
    }

    /**
     * 向量工厂方法（Integer包装类数组） / Vector factory method (Integer wrapper array)
     * <p>
     * 使用给定的Integer包装类数组创建向量实例 Creates a vector instance with the given Integer
     * wrapper array
     * </p>
     *
     * @param data Integer包装类数组 / Integer wrapper array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector vec(Integer[] data) {
        return vec(RereMathUtil.toPrimitive(data));
    }

    /**
     * 创建范围向量（带步长） / Create range vector (with step)
     * <p>
     * 创建一个从start到end（不包含end）、步长为step的向量 Creates a vector from start to end
     * (exclusive) with specified step
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @param step 步长 / Step size
     * @return 范围向量 / Range vector
     * @throws IllegalArgumentException 如果step为0或负数 / if step is 0 or negative
     */
    public static IVector range(int start, int end, int step) {
        List<Float> ls = new ArrayList<>();
        int p = start;
        while (p < end) {
            ls.add((float) p);
            p += step;
        }
        var as = ls.toArray(Float[]::new);
        return IVector.of(RereMathUtil.toPrimitive(as));
    }

    /**
     * 创建范围向量（步长为1） / Create range vector (step size 1)
     * <p>
     * 创建一个从start到end（不包含end）、步长为1的向量 Creates a vector from start to end
     * (exclusive) with step size 1
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @return 范围向量 / Range vector
     */
    public static IVector range(int start, int end) {
        return range(start, end, 1);
    }

    /**
     * 创建范围向量（从0开始） / Create range vector (starting from 0)
     * <p>
     * 创建一个从0到end（不包含end）、步长为1的向量 Creates a vector from 0 to end (exclusive)
     * with step size 1
     * </p>
     *
     * @param end 结束值（不包含） / End value (exclusive)
     * @return 范围向量 / Range vector
     */
    public static IVector range(int end) {
        return range(0, end, 1);
    }

    /**
     * 创建全1向量 / Create ones vector
     * <p>
     * 创建一个指定长度的向量，所有元素都为1 Creates a vector of specified length with all
     * elements set to 1
     * </p>
     *
     * @param len 向量长度 / IVector length
     * @return 全1向量 / Ones vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IVector ones(int len) {
        float[] v = new float[len];
        for (int i = 0; i < len; i++) {
            v[i] = 1;
        }
        return IVector.of(v);
    }

    /**
     * 创建零向量 / Create zeros vector
     * <p>
     * 创建一个指定长度的向量，所有元素都为0 Creates a vector of specified length with all
     * elements set to 0
     * </p>
     *
     * @param len 向量长度 / IVector length
     * @return 零向量 / Zeros vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IVector zeros(int len) {
        float[] v = new float[len];
        // Java数组默认初始化为0，所以不需要显式设置 / Java arrays are initialized to 0 by default
        return IVector.of(v);
    }

    /**
     * 创建随机向量 / Create random vector
     * <p>
     * 创建指定长度的随机向量，元素值在[0,1)范围内 Creates a random vector of specified length with
     * elements in [0,1) range
     * </p>
     *
     * @param length 向量长度 / IVector length
     * @return 随机向量 / Random vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IVector rand(int length) {
        if (length <= 0) {
            throw new IllegalArgumentException("向量长度必须大于0 / IVector length must be greater than 0");
        }
        float[] v = new float[length];
        for (int i = 0; i < length; i++) {
            v[i] = (float) Math.random();
        }
        return IVector.of(v);
    }

    /**
     * 创建正态分布随机向量 / Create normal distribution random vector
     * <p>
     * 创建指定长度的正态分布随机向量 Creates a normal distribution random vector of specified
     * length
     * </p>
     *
     * @param length 向量长度 / IVector length
     * @param mean 均值 / Mean
     * @param std 标准差 / Standard deviation
     * @return 正态分布随机向量 / Normal distribution random vector
     * @throws IllegalArgumentException 如果长度小于等于0或标准差小于等于0 / if length is less
     * than or equal to 0 or std is less than or equal to 0
     */
    public static IVector randn(int length, float mean, float std) {
        if (length <= 0) {
            throw new IllegalArgumentException("向量长度必须大于0 / IVector length must be greater than 0");
        }
        if (std <= 0) {
            throw new IllegalArgumentException("标准差必须大于0 / Standard deviation must be greater than 0");
        }
        float[] v = new float[length];
        for (int i = 0; i < length; i++) {
            // Box-Muller变换生成正态分布随机数 / Box-Muller transform for normal distribution
            double u1 = Math.random();
            double u2 = Math.random();
            double z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            v[i] = (float) (mean + std * z0);
        }
        return IVector.of(v);
    }

    /**
     * 创建线性空间向量 / Create linear space vector
     * <p>
     * 创建指定数量的等间距数值向量 Creates a vector with specified number of equally spaced
     * values
     * </p>
     *
     * @param start 起始值 / Start value
     * @param stop 结束值 / Stop value
     * @param num 元素数量 / Number of elements
     * @return 线性空间向量 / Linear space vector
     * @throws IllegalArgumentException 如果元素数量小于2 / if number of elements is
     * less than 2
     */
    public static IVector linspace(float start, float stop, int num) {
        if (num < 2) {
            throw new IllegalArgumentException("元素数量必须大于等于2 / Number of elements must be greater than or equal to 2");
        }
        float[] v = new float[num];
        float step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            v[i] = start + i * step;
        }
        return IVector.of(v);
    }

    /**
     * 创建对数空间向量 / Create logarithmic space vector
     * <p>
     * 创建指定数量的对数等间距数值向量 Creates a vector with specified number of
     * logarithmically equally spaced values
     * </p>
     *
     * @param start 起始值（10^start） / Start value (10^start)
     * @param stop 结束值（10^stop） / Stop value (10^stop)
     * @param num 元素数量 / Number of elements
     * @return 对数空间向量 / Logarithmic space vector
     * @throws IllegalArgumentException 如果元素数量小于2 / if number of elements is
     * less than 2
     */
    public static IVector logspace(float start, float stop, int num) {
        if (num < 2) {
            throw new IllegalArgumentException("元素数量必须大于等于2 / Number of elements must be greater than or equal to 2");
        }
        float[] v = new float[num];
        float step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            v[i] = (float) Math.pow(10, start + i * step);
        }
        return IVector.of(v);
    }
}
