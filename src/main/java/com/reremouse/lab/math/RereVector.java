package com.reremouse.lab.math;

import com.reremouse.lab.util.StringUtils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * 向量操作实现类 / Vector Operations Implementation Class
 * <p>
 * 本类实现了向量的常用操作，包括基本的数学运算、统计运算、切片索引、通用函数等功能。
 * 基于一维float数组实现，提供高效的向量计算能力。
 * </p>
 * <p>
 * This class implements common vector operations including basic mathematical operations,
 * statistical operations, slicing and indexing, universal functions and other functionalities.
 * Based on 1D float array implementation, providing efficient vector computation capabilities.
 * </p>
 * 
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 *   <li>基本数学运算：加法、减法、乘法、内积 / Basic math operations: add, subtract, multiply, inner product</li>
 *   <li>标量运算：与标量的四则运算 / Scalar operations: arithmetic operations with scalars</li>
 *   <li>统计运算：求和、均值、方差、标准差、最值 / Statistical operations: sum, mean, variance, standard deviation, min/max</li>
 *   <li>通用函数：开方、平方、指数、对数、幂运算 / Universal functions: sqrt, square, exp, log, power</li>
 *   <li>切片索引：范围切片、花式索引、布尔索引 / Slicing and indexing: range slicing, fancy indexing, boolean indexing</li>
 *   <li>比较运算：元素级比较操作 / Comparison operations: element-wise comparison</li>
 *   <li>数据转换：类型转换、数据获取 / Data conversion: type conversion, data access</li>
 *   <li>线性代数：范数计算、排序、反转 / Linear algebra: norm calculation, sorting, reversing</li>
 * </ul>
 * 
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 // 创建向量 / Create vector
 float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
 IVector vector = new RereVector(data);
 
 // 基本运算 / Basic operations
 IVector doubled = vector.multiplyScala(2.0f);
 float norm = vector.norm2();
 
 // 统计运算 / Statistical operations
 float mean = vector.mean();
 float std = vector.std();
 
 // 切片操作 / Slicing operations
 IVector slice = vector.slice(1, 3);
 IVector squared = vector.squre();
 }
 * </pre>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereVector implements IVector {

    /**
     * 向量数据存储数组 / Vector data storage array
     * <p>
     * 使用一维float数组存储向量数据，data[i]表示第i个元素
     * Uses 1D float array to store vector data, data[i] represents the i-th element
     * </p>
     */
    private final float data[];

    /**
     * 构造函数 / Constructor
     * <p>
     * 使用给定的float数组创建向量实例
     * Creates a vector instance with the given float array
     * </p>
     * 
     * @param data 一维float数组，表示向量数据 / 1D float array representing vector data
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public RereVector(float[] data) {
        if (data == null) {
            throw new IllegalArgumentException("向量数据不能为null / Vector data cannot be null");
        }
        this.data = data;
    }

    /**
     * 向量减法运算 / Vector subtraction
     * <p>
     * 对应元素相减，要求两个向量长度相同
     * Element-wise subtraction, requires both vectors to have the same length
     * </p>
     * 
     * @param vec 另一个向量 / The other vector
     * @return 新的向量对象，包含减法结果 / New vector object containing subtraction result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public IVector sub(IVector vec) {
        if (vec == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != vec.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + vec.length() + 
                " / Vector lengths don't match: " + this.data.length + " != " + vec.length());
        }
        
        int len = this.data.length;
        float[] v = new float[len];
        for (int i = 0; i < len; i++) {
            v[i] = this.data[i] - vec.getData()[i];
        }
        var vv = IVector.of(v);
        return vv;
    }

    /**
     * 向量加法运算 / Vector addition
     * <p>
     * 对应元素相加，要求两个向量长度相同
     * Element-wise addition, requires both vectors to have the same length
     * </p>
     * 
     * @param vec 另一个向量 / The other vector
     * @return 新的向量对象，包含加法结果 / New vector object containing addition result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public IVector add(IVector vec) {
        if (vec == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != vec.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + vec.length() + 
                " / Vector lengths don't match: " + this.data.length + " != " + vec.length());
        }
        
        int len = this.data.length;
        float[] v = new float[len];
        for (int i = 0; i < len; i++) {
            v[i] = this.data[i] + vec.getData()[i];
        }
        var vv = IVector.of(v);
        return vv;
    }

    /**
     * 向量内积运算 / Vector inner product
     * <p>
     * 计算两个向量的内积（点积），要求两个向量长度相同
     * Computes the inner product (dot product) of two vectors, requires both vectors to have the same length
     * </p>
     * 
     * @param vec 另一个向量 / The other vector
     * @return 内积结果 / Inner product result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public float innerProduct(IVector vec) {
        if (vec == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != vec.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + vec.length() + 
                " / Vector lengths don't match: " + this.data.length + " != " + vec.length());
        }
        
        float p = 0;
        for (int i = 0; i < data.length; i++) {
            p += data[i] * vec.getData()[i];
        }
        return p;
    }

    /**
     * 向量2范数 / Vector L2 norm
     * <p>
     * 计算向量的2范数（欧几里得范数），即所有元素平方和的开方
     * Calculates the L2 norm (Euclidean norm) of the vector, which is the square root of the sum of squares
     * </p>
     * 
     * @return 2范数 / L2 norm
     */
    @Override
    public float norm2() {
        int len = this.data.length;
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += Math.pow(this.data[i], 2);
        }
        sum = (float) Math.sqrt(sum);
        return sum;
    }

    /**
     * 向量1范数 / Vector L1 norm
     * <p>
     * 计算向量的1范数（曼哈顿范数），即所有元素绝对值的和
     * Calculates the L1 norm (Manhattan norm) of the vector, which is the sum of absolute values
     * </p>
     * 
     * @return 1范数 / L1 norm
     */
    @Override
    public float norm1() {
        int len = this.data.length;
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += Math.abs(this.data[i]);
        }
        return sum;
    }

    /**
     * 向量除标量 / Vector divide by scalar
     * <p>
     * 向量中每个元素除以标量值
     * Divides each element in the vector by a scalar value
     * </p>
     * 
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含除法结果 / New vector object containing division result
     * @throws ArithmeticException 如果标量为零 / if scalar is zero
     */
    @Override
    public IVector divideByScalar(float p) {
        if (p == 0.0f) {
            throw new ArithmeticException("除数不能为零 / Divisor cannot be zero");
        }
        
        float[] v = new float[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] / p;
        }
        return IVector.of(v);
    }

    /**
     * 获取向量数据数组 / Get vector data array
     * <p>
     * 返回向量的内部数据数组引用
     * Returns a reference to the internal data array of the vector
     * </p>
     * 
     * @return 向量的数据数组 / Data array of the vector
     */
    @Override
    public float[] getData() {
        return this.data;
    }

    /**
     * 向量乘法运算（元素级） / Vector multiplication (element-wise)
     * <p>
     * 对应元素相乘，要求两个向量长度相同
     * Element-wise multiplication, requires both vectors to have the same length
     * </p>
     * 
     * @param vec 另一个向量 / The other vector
     * @return 新的向量对象，包含乘法结果 / New vector object containing multiplication result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public IVector multiply(IVector vec) {
        if (vec == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != vec.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + vec.length() + 
                " / Vector lengths don't match: " + this.data.length + " != " + vec.length());
        }
        
        float[] v = new float[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] * vec.getData()[i];
        }
        return IVector.of(v);
    }

    /**
     * 向量与矩阵的点积 / Vector-matrix dot product
     * <p>
     * 计算向量与矩阵的点积，向量长度必须与矩阵行数相等
     * Computes the dot product of vector and matrix, vector length must equal matrix row count
     * </p>
     * 
     * @param m 矩阵 / IMatrix
     * @return 结果矩阵 / Result matrix
     * @throws IllegalArgumentException 如果向量长度与矩阵行数不匹配 / if vector length doesn't match matrix row count
     */
    @Override
    public IMatrix dot(IMatrix m) {
        float[][] mm = new float[m.getRowNum()][m.getColNum()];
        for (int i = 0; i < data.length; i++) {
            float w = data[i];
            float[] v = m.getData()[i];
            for (int j = 0; j < v.length; j++) {
                v[j] = w * v[j];
            }
            mm[i] = v;
        }
        return IMatrix.of(mm);
    }

    /**
     * 向量减标量 / Vector minus scalar
     * <p>
     * 向量中每个元素减去标量值
     * Subtracts a scalar value from each element in the vector
     * </p>
     * 
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含减法结果 / New vector object containing subtraction result
     */
    @Override
    public IVector minusScalar(float p) {
        float[] v = new float[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] - p;
        }
        return IVector.of(v);
    }

    /**
     * 向量加标量 / Vector plus scalar
     * <p>
     * 向量中每个元素加上标量值
     * Adds a scalar value to each element in the vector
     * </p>
     * 
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含加法结果 / New vector object containing addition result
     */
    @Override
    public IVector addScalar(float p) {
        float[] v = new float[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] + p;
        }
        return IVector.of(v);
    }

    /**
     * 向量乘标量 / Vector multiply scalar
     * <p>
     * 向量中每个元素乘以标量值
     * Multiplies each element in the vector by a scalar value
     * </p>
     * 
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含乘法结果 / New vector object containing multiplication result
     */
    @Override
    public IVector multiplyScalar(float p) {
        float[] v = new float[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] * p;
        }
        return IVector.of(v);
    }

    /**
     * 向量元素求和 / Vector sum
     * <p>
     * 计算向量中所有元素的和
     * Calculates the sum of all elements in the vector
     * </p>
     * 
     * @return 元素和 / Sum of elements
     */
    @Override
    public float sum() {
        int len = this.data.length;
        float sum = 0;
        for (int i = 0; i < len; i++) {
            sum += this.data[i];
        }
        return sum;
    }

    /**
     * 向量最小值 / Vector minimum
     * <p>
     * 找到向量中的最小元素值
     * Finds the minimum element value in the vector
     * </p>
     * 
     * @return 最小值 / Minimum value
     */
    @Override
    public float min() {
        int len = this.data.length;
        float min = Float.MAX_VALUE;
        for (int i = 0; i < len; i++) {
            if (data[i] < min) {
                min = data[i];
            }
        }
        return min;
    }

    /**
     * 向量最大值 / Vector maximum
     * <p>
     * 找到向量中的最大元素值
     * Finds the maximum element value in the vector
     * </p>
     * 
     * @return 最大值 / Maximum value
     */
    @Override
    public float max() {
        int len = this.data.length;
        float max = Float.MIN_VALUE;
        for (int i = 0; i < len; i++) {
            if (data[i] > max) {
                max = data[i];
            }
        }
        return max;
    }

    /**
     * 最小值索引 / Index of minimum value
     * <p>
     * 返回向量中最小元素的索引位置
     * Returns the index position of the minimum element in the vector
     * </p>
     * 
     * @return 最小值的索引 / Index of minimum value
     */
    @Override
    public int argMin() {
        int len = this.data.length;
        float min = Float.MAX_VALUE;
        int ind = -1;
        for (int i = 0; i < len; i++) {
            if (data[i] < min) {
                min = data[i];
                ind = i;
            }
        }
        return ind;
    }

    /**
     * 最大值索引 / Index of maximum value
     * <p>
     * 返回向量中最大元素的索引位置
     * Returns the index position of the maximum element in the vector
     * </p>
     * 
     * @return 最大值的索引 / Index of maximum value
     */
    @Override
    public int argMax() {
        int len = this.data.length;
        float max = Float.MIN_VALUE;
        int ind = -1;
        for (int i = 0; i < len; i++) {
            if (data[i] > max) {
                max = data[i];
                ind = i;
            }
        }
        return ind;
    }

    /**
     * 向量均值 / Vector mean
     * <p>
     * 计算向量中所有元素的平均值
     * Calculates the mean of all elements in the vector
     * </p>
     * 
     * @return 平均值 / Mean value
     */
    @Override
    public float mean() {
        return this.sum() / (float) this.length();
    }

    /**
     * 向量标准差 / Vector standard deviation
     * <p>
     * 计算向量的标准差（除以N）
     * Calculates the standard deviation of the vector (divided by N)
     * </p>
     * 
     * @return 标准差 / Standard deviation
     */
    @Override
    public float std() {
        return (float) Math.sqrt(this.var());
    }

    /**
     * 向量标准差（自由度修正） / Vector standard deviation (degrees of freedom correction)
     * <p>
     * 计算向量的标准差，使用自由度修正（除以N-ddof）
     * Calculates the standard deviation of the vector with degrees of freedom correction (divided by N-ddof)
     * </p>
     * 
     * @param ddof 自由度修正值 / Degrees of freedom correction
     * @return 标准差 / Standard deviation
     */
    @Override
    public float std(int ddof) {
        return (float) Math.sqrt(this.var(ddof));
    }

    /**
     * 向量方差 / Vector variance
     * <p>
     * 计算向量的方差（除以N）
     * Calculates the variance of the vector (divided by N)
     * </p>
     * 
     * @return 方差 / Variance
     */
    @Override
    public float var() {
        return this.minusScalar(this.mean()).pow(2).sum() / (float) this.length();
    }

    /**
     * 向量方差（自由度修正） / Vector variance (degrees of freedom correction)
     * <p>
     * 计算向量的方差，使用自由度修正（除以N-ddof）
     * Calculates the variance of the vector with degrees of freedom correction (divided by N-ddof)
     * </p>
     * 
     * @param ddof 自由度修正值 / Degrees of freedom correction
     * @return 方差 / Variance
     */
    @Override
    public float var(int ddof) {
        return this.minusScalar(this.mean()).pow(2).sum() / (float) (this.length() - ddof);
    }

    /**
     * 获取向量长度 / Get vector length
     * <p>
     * 返回向量的长度（元素个数）
     * Returns the length (number of elements) of the vector
     * </p>
     * 
     * @return 向量长度 / IVector length
     */
    @Override
    public int length() {
        return data.length;
    }

    /**
     * 获取指定位置的元素 / Get element at specified position
     * <p>
     * 返回向量中指定位置的元素值
     * Returns the element value at the specified position in the vector
     * </p>
     * 
     * @param position 位置索引（从0开始） / Position index (0-based)
     * @return 指定位置的元素值 / Element value at the specified position
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position index is out of bounds
     */
    @Override
    public float get(int position) {
        return data[position];
    }

    /**
     * 向量切片（到指定结束位置） / Vector slice (to specified end position)
     * <p>
     * 返回从开始到指定结束位置的向量切片
     * Returns a vector slice from start to the specified end position
     * </p>
     * 
     * @param end 结束位置（不包含） / End position (exclusive)
     * @return 切片向量 / Sliced vector
     * @throws IndexOutOfBoundsException 如果结束位置超出范围 / if end position is out of bounds
     */
    @Override
    public IVector slice(int end) {
        return this.slice(0, end, 1);
    }

    /**
     * 向量切片（指定开始和结束位置） / Vector slice (specified start and end positions)
     * <p>
     * 返回从指定开始位置到结束位置的向量切片
     * Returns a vector slice from specified start position to end position
     * </p>
     * 
     * @param start 开始位置 / Start position
     * @param end 结束位置（不包含） / End position (exclusive)
     * @return 切片向量 / Sliced vector
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are out of bounds
     */
    @Override
    public IVector slice(int start, int end) {
        return this.slice(start, end, 1);
    }

    /**
     * 向量切片（指定开始、结束位置和步长） / Vector slice (specified start, end positions and step)
     * <p>
     * 返回从指定开始位置到结束位置、指定步长的向量切片
     * Returns a vector slice from specified start position to end position with specified step
     * </p>
     * 
     * @param start 开始位置 / Start position
     * @param end 结束位置（不包含） / End position (exclusive)
     * @param step 步长 / Step size
     * @return 切片向量 / Sliced vector
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are out of bounds
     * @throws IllegalArgumentException 如果步长小于等于0 / if step is less than or equal to 0
     */
    @Override
    public IVector slice(int start, int end, int step) {
        int[] inds = IVector.range(start, end, step).asIntArray();
        float[] v = new float[inds.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[inds[i]];
        }
        return IVector.of(v);
    }

    /**
     * 花式索引 / Fancy indexing
     * <p>
     * 根据给定的位置数组获取对应位置的元素组成新向量
     * Gets elements at specified positions to form a new vector
     * </p>
     * 
     * @param positions 位置索引数组 / Array of position indices
     * @return 新的向量对象，包含指定位置的元素 / New vector object containing elements at specified positions
     * @throws IndexOutOfBoundsException 如果任何位置索引超出范围 / if any position index is out of bounds
     */
    @Override
    public IVector fancyGet(int[] positions) {
        float[] v = new float[positions.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[positions[i]];
        }
        return IVector.of(v);
    }

    /**
     * 布尔索引 / Boolean indexing
     * <p>
     * 根据布尔数组获取对应位置为true的元素组成新向量
     * Gets elements where the corresponding boolean index is true to form a new vector
     * </p>
     * 
     * @param booleanIndex 布尔索引数组 / Boolean index array
     * @return 新的向量对象，包含布尔索引为true位置的元素 / New vector object containing elements where boolean index is true
     * @throws IllegalArgumentException 如果布尔数组长度与向量长度不匹配 / if boolean array length doesn't match vector length
     */
    @Override
    public IVector booleanGet(boolean[] booleanIndex) {
        if (booleanIndex == null) {
            throw new IllegalArgumentException("布尔索引数组不能为null / Boolean index array cannot be null");
        }
        if (booleanIndex.length != this.data.length) {
            throw new IllegalArgumentException("布尔索引数组长度与向量长度不匹配: " + booleanIndex.length + " != " + this.data.length + 
                " / Boolean index array length doesn't match vector length: " + booleanIndex.length + " != " + this.data.length);
        }
        
        List<Float> ls = new ArrayList();
        for (int i = 0; i < booleanIndex.length; i++) {
            if (booleanIndex[i]) {
                ls.add(data[i]);
            }
        }
        Float[] v = ls.toArray(Float[]::new);
        return IVector.of(v);
    }

    /**
     * 设置指定位置的元素 / Set element at specified position
     * <p>
     * 设置向量中指定位置的元素值
     * Sets the element value at the specified position in the vector
     * </p>
     * 
     * @param position 位置索引（从0开始） / Position index (0-based)
     * @param value 要设置的值 / Value to set
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position index is out of bounds
     */
    @Override
    public IVector set(int position, float value) {
        data[position] = value;
        return this;
    }

    /**
     * 范围设置值（带步长） / Range set values (with step)
     * <p>
     * 设置指定范围内、指定步长位置的元素值
     * Sets element values at positions within specified range with specified step
     * </p>
     * 
     * @param start 开始位置 / Start position
     * @param end 结束位置（不包含） / End position (exclusive)
     * @param step 步长 / Step size
     * @param values 要设置的值数组 / Array of values to set
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are out of bounds
     * @throws IllegalArgumentException 如果值数组长度不匹配 / if values array length doesn't match
     */
    @Override
    public IVector setFromTo(int start, int end, int step, float[] values) {
        int[] inds = IVector.range(start, end, step).asIntArray();
        for (int i = 0; i < inds.length; i++) {
            data[inds[i]] = values[i];
        }
        return this;
    }

    /**
     * 范围设置值 / Range set values
     * <p>
     * 设置指定范围内的元素值（步长为1）
     * Sets element values within specified range (step size 1)
     * </p>
     * 
     * @param start 开始位置 / Start position
     * @param end 结束位置（不包含） / End position (exclusive)
     * @param values 要设置的值数组 / Array of values to set
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are out of bounds
     * @throws IllegalArgumentException 如果值数组长度不匹配 / if values array length doesn't match
     */
    @Override
    public IVector setFromTo(int start, int end, float[] values) {
        return this.setFromTo(start, end, 1, values);
    }

    /**
     * 向量相等比较 / Vector equality comparison
     * <p>
     * 对应元素进行相等比较，返回布尔数组
     * Element-wise equality comparison, returns boolean array
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示对应位置元素是否相等 / Boolean array indicating whether corresponding elements are equal
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public boolean[] equals(IVector other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.length() != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.length() + " != " + other.length() + 
                " / Vector lengths don't match: " + this.length() + " != " + other.length());
        }
        
        boolean v[] = new boolean[this.length()];
        for (int i = 0; i < this.length(); i++) {
            v[i] = (this.get(i) == other.get(i));
        }
        return v;
    }

    /**
     * 向量小于比较 / Vector less-than comparison
     * <p>
     * 对应元素进行小于比较，返回布尔数组
     * Element-wise less-than comparison, returns boolean array
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示对应位置元素是否小于 / Boolean array indicating whether corresponding elements are less than
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public boolean[] lessThan(IVector other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.length() != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.length() + " != " + other.length() + 
                " / Vector lengths don't match: " + this.length() + " != " + other.length());
        }
        
        boolean v[] = new boolean[this.length()];
        for (int i = 0; i < this.length(); i++) {
            v[i] = (this.get(i) < other.get(i));
        }
        return v;
    }

    /**
     * 向量大于比较 / Vector greater-than comparison
     * <p>
     * 对应元素进行大于比较，返回布尔数组
     * Element-wise greater-than comparison, returns boolean array
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示对应位置元素是否大于 / Boolean array indicating whether corresponding elements are greater than
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public boolean[] greaterThan(IVector other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.length() != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.length() + " != " + other.length() + 
                " / Vector lengths don't match: " + this.length() + " != " + other.length());
        }
        
        boolean v[] = new boolean[this.length()];
        for (int i = 0; i < this.length(); i++) {
            v[i] = (this.get(i) > other.get(i));
        }
        return v;
    }

    /**
     * 向量元素乘积 / Vector product
     * <p>
     * 计算向量中所有元素的乘积
     * Calculates the product of all elements in the vector
     * </p>
     * 
     * @return 元素乘积 / Product of elements
     */
    @Override
    public float prod() {
        float p = 1;
        for (var e : data) {
            p *= e;
        }
        return p;
    }

    /**
     * 向量裁剪 / Vector clipping
     * <p>
     * 将向量中的元素值限制在指定范围内
     * Clips vector elements to the specified range
     * </p>
     * 
     * @param lower 下界 / Lower bound
     * @param upper 上界 / Upper bound
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     * @throws IllegalArgumentException 如果lower > upper / if lower > upper
     */
    @Override
    public IVector clip(float lower, float upper) {
        if (lower > upper) {
            throw new IllegalArgumentException("下界不能大于上界: " + lower + " > " + upper + 
                " / Lower bound cannot be greater than upper bound: " + lower + " > " + upper);
        }
        
        for (int i = 0; i < this.length(); i++) {
            if (data[i] > upper) {
                data[i] = upper;
            } else if (data[i] < lower) {
                data[i] = lower;
            }
        }
        return this;
    }

    /**
     * 向量峰峰值 / Vector peak-to-peak value
     * <p>
     * 计算向量的峰峰值（最大值减最小值）
     * Calculates the peak-to-peak value of the vector (max - min)
     * </p>
     * 
     * @return 峰峰值 / Peak-to-peak value
     */
    @Override
    public float ptp() {
        return this.max() - this.min();
    }

    /**
     * 向量绝对值 / Vector absolute value
     * <p>
     * 对向量中每个元素取绝对值
     * Takes the absolute value of each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含绝对值结果 / New vector object containing absolute value results
     */
    @Override
    public IVector abs() {
        var data2 = new float[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = Math.abs(data[i]);
        }
        return IVector.of(data2);
    }

    /**
     * 向量填充 / Vector fill
     * <p>
     * 将向量中所有元素设置为指定值
     * Sets all elements in the vector to the specified value
     * </p>
     * 
     * @param value 填充值 / Fill value
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     */
    @Override
    public IVector fill(float value) {
        for (int i = 0; i < this.length(); i++) {
            data[i] = value;
        }
        return this;

    }

    /**
     * 向量排序 / Vector sorting
     * <p>
     * 对向量中的元素进行升序排序
     * Sorts the elements in the vector in ascending order
     * </p>
     * 
     * @return 排序后的向量（就地操作） / Sorted vector (in-place operation)
     */
    @Override
    public IVector sort() {
        Arrays.sort(data);
        return this;
    }

    /**
     * 向量反转 / Vector reverse
     * <p>
     * 反转向量中元素的顺序
     * Reverses the order of elements in the vector
     * </p>
     * 
     * @return 反转后的向量（就地操作） / Reversed vector (in-place operation)
     */
    @Override
    public IVector reverse() {
        for (int i = 0; i < data.length / 2; i++) {
            float temp = data[i];
            data[i] = data[data.length - i - 1];
            data[data.length - i - 1] = temp;
        }
        return this;
    }

    /**
     * 向量复制 / Vector copy
     * <p>
     * 创建向量的深拷贝
     * Creates a deep copy of the vector
     * </p>
     * 
     * @return 新的向量对象，与原向量内容相同 / New vector object with the same content as the original
     */
    @Override
    public IVector copy() {
        float[] v = new float[this.length()];
        System.arraycopy(data, 0, v, 0, this.length());
        return IVector.of(v);
    }

    /**
     * 向量开方 / Vector square root
     * <p>
     * 对向量中每个元素进行开方运算
     * Performs square root operation on each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含开方结果 / New vector object containing square root results
     * @throws ArithmeticException 如果元素值为负数 / if any element value is negative
     */
    @Override
    public IVector sqrt() {
        var data2 = new float[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = (float) Math.sqrt(data[i]);
        }
        return IVector.of(data2);
    }

    /**
     * 向量平方 / Vector square
     * <p>
     * 对向量中每个元素进行平方运算
     * Performs square operation on each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含平方结果 / New vector object containing square results
     */
    @Override
    public IVector squre() {
        var data2 = new float[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = (float) Math.pow(data[i], 2);
        }
        return IVector.of(data2);

    }

    /**
     * 向量指数运算 / Vector exponential
     * <p>
     * 对向量中每个元素进行指数运算（e^x）
     * Performs exponential operation (e^x) on each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含指数运算结果 / New vector object containing exponential results
     */
    @Override
    public IVector exp() {
        var data2 = new float[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = (float) Math.exp(data[i]);
        }
        return IVector.of(data2);
    }

    /**
     * 向量自然对数 / Vector natural logarithm
     * <p>
     * 对向量中每个元素进行自然对数运算（ln(x)）
     * Performs natural logarithm operation (ln(x)) on each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含对数运算结果 / New vector object containing logarithm results
     * @throws ArithmeticException 如果元素值小于等于0 / if any element value is less than or equal to 0
     */
    @Override
    public IVector log() {
        var data2 = new float[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = (float) Math.log(data[i]);
        }
        return IVector.of(data2);
    }

    /**
     * 向量以10为底的对数 / Vector base-10 logarithm
     * <p>
     * 对向量中每个元素进行以10为底的对数运算（log10(x)）
     * Performs base-10 logarithm operation (log10(x)) on each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含对数运算结果 / New vector object containing logarithm results
     * @throws ArithmeticException 如果元素值小于等于0 / if any element value is less than or equal to 0
     */
    @Override
    public IVector log10() {
        var data2 = new float[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = (float) Math.log10(data[i]);
        }
        return IVector.of(data2);
    }

    /**
     * 向量幂运算 / Vector power operation
     * <p>
     * 对向量中每个元素进行幂运算（x^m）
     * Performs power operation (x^m) on each element in the vector
     * </p>
     * 
     * @param m 幂指数 / Power exponent
     * @return 新的向量对象，包含幂运算结果 / New vector object containing power operation results
     */
    @Override
    public IVector pow(float m) {
        var data2 = new float[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = (float) Math.pow(data[i], m);
        }
        return IVector.of(data2);
    }

    /**
     * 向量取余运算 / Vector remainder operation
     * <p>
     * 对向量中每个元素进行取余运算
     * Performs remainder operation on each element in the vector
     * </p>
     * 
     * @param value 除数 / Divisor
     * @return 新的向量对象，包含取余运算结果 / New vector object containing remainder results
     * @throws ArithmeticException 如果除数为零 / if divisor is zero
     */
    @Override
    public IVector remainder(float value) {
        if (value == 0.0f) {
            throw new ArithmeticException("除数不能为零 / Divisor cannot be zero");
        }
        
        var data2 = new float[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = data[i] % value;
        }
        return IVector.of(data2);
    }

    /**
     * 转换为整数数组 / Convert to integer array
     * <p>
     * 将向量转换为整数数组
     * Converts the vector to an integer array
     * </p>
     * 
     * @return 整数数组 / Integer array
     */
    @Override
    public int[] asIntArray() {
        return RereMathUtil.floatToInt(data);
    }

    /**
     * 转换为双精度数组 / Convert to double array
     * <p>
     * 将向量转换为双精度数组
     * Converts the vector to a double array
     * </p>
     * 
     * @return 双精度数组 / Double array
     */
    @Override
    public double[] asDoubleArray() {
        return RereMathUtil.floatToDouble(data);
    }

    /**
     * 计算与另一个向量的欧几里得距离 / Compute Euclidean distance to another vector
     * <p>
     * 计算当前向量与另一个向量之间的欧几里得距离
     * Computes the Euclidean distance between current vector and another vector
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 欧几里得距离 / Euclidean distance
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public float euclideanDistance(IVector other) {
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配 / Vector lengths don't match");
        }
        
        IVector diff = this.sub(other);
        return (float) Math.sqrt(diff.innerProduct(diff));
    }

    /**
     * 计算与另一个向量的曼哈顿距离 / Compute Manhattan distance to another vector
     * <p>
     * 计算当前向量与另一个向量之间的曼哈顿距离（L1距离）
     * Computes the Manhattan distance (L1 distance) between current vector and another vector
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 曼哈顿距离 / Manhattan distance
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    @Override
    public float manhattanDistance(IVector other) {
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配 / Vector lengths don't match");
        }
        
        IVector diff = this.sub(other);
        return diff.abs().sum();
    }

    /**
     * 计算与另一个向量的余弦相似度 / Compute cosine similarity to another vector
     * <p>
     * 计算当前向量与另一个向量之间的余弦相似度
     * Computes the cosine similarity between current vector and another vector
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 余弦相似度 / Cosine similarity
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     * @throws ArithmeticException 如果向量长度为零 / if vector norm is zero
     */
    @Override
    public float cosineSimilarity(IVector other) {
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配 / Vector lengths don't match");
        }
        
        float dotProduct = this.innerProduct(other);
        float norm1 = this.norm2();
        float norm2 = other.norm2();
        
        if (norm1 == 0.0f || norm2 == 0.0f) {
            throw new ArithmeticException("向量长度为零 / Vector norm is zero");
        }
        
        return dotProduct / (norm1 * norm2);
    }

    @Override
    public String toString() {
        var ls = IntStream.range(0, this.data.length)
        .mapToObj(i -> String.format("Value: %.6f", this.data[i]))
        .toArray(String[]::new);
        return StringUtils.join(ls,", ");
    }
    
    // ========== 三角函数操作实现 / Trigonometric Functions Implementation ==========
    
    @Override
    public IVector sin() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.sin(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector cos() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.cos(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector tan() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.tan(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector arcsin() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            if (this.data[i] < -1.0f || this.data[i] > 1.0f) {
                throw new ArithmeticException("反正弦函数输入值超出范围[-1,1]: " + this.data[i] + 
                    " / Arcsine input value outside range [-1,1]: " + this.data[i]);
            }
            result[i] = (float) Math.asin(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector arccos() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            if (this.data[i] < -1.0f || this.data[i] > 1.0f) {
                throw new ArithmeticException("反余弦函数输入值超出范围[-1,1]: " + this.data[i] + 
                    " / Arccosine input value outside range [-1,1]: " + this.data[i]);
            }
            result[i] = (float) Math.acos(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector arctan() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.atan(this.data[i]);
        }
        return IVector.of(result);
    }

    // ========== 双曲函数实现 / Hyperbolic Functions Implementation ==========
    
    @Override
    public IVector sinh() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.sinh(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector cosh() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.cosh(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector tanh() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.tanh(this.data[i]);
        }
        return IVector.of(result);
    }

    // ========== 舍入函数实现 / Rounding Functions Implementation ==========
    
    @Override
    public IVector round() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = Math.round(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector floor() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.floor(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector ceil() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.ceil(this.data[i]);
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector trunc() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (float) Math.rint(this.data[i]);
        }
        return IVector.of(result);
    }

    // ========== 逻辑运算实现 / Logical Operations Implementation ==========
    
    @Override
    public IVector logicalAnd(IVector other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + other.length() + 
                " / Vector lengths don't match: " + this.data.length + " != " + other.length());
        }
        
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (this.data[i] != 0.0f && other.get(i) != 0.0f) ? 1.0f : 0.0f;
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector logicalOr(IVector other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + other.length() + 
                " / Vector lengths don't match: " + this.data.length + " != " + other.length());
        }
        
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (this.data[i] != 0.0f || other.get(i) != 0.0f) ? 1.0f : 0.0f;
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector logicalNot() {
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (this.data[i] == 0.0f) ? 1.0f : 0.0f;
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector logicalXor(IVector other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + other.length() + 
                " / Vector lengths don't match: " + this.data.length + " != " + other.length());
        }
        
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            boolean a = this.data[i] != 0.0f;
            boolean b = other.get(i) != 0.0f;
            result[i] = (a != b) ? 1.0f : 0.0f;
        }
        return IVector.of(result);
    }

    // ========== 累积操作实现 / Cumulative Operations Implementation ==========
    
    @Override
    public IVector cumsum() {
        float[] result = new float[this.data.length];
        float sum = 0.0f;
        for (int i = 0; i < this.data.length; i++) {
            sum += this.data[i];
            result[i] = sum;
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector cumprod() {
        float[] result = new float[this.data.length];
        float product = 1.0f;
        for (int i = 0; i < this.data.length; i++) {
            product *= this.data[i];
            result[i] = product;
        }
        return IVector.of(result);
    }

    // ========== 差分操作实现 / Difference Operations Implementation ==========
    
    @Override
    public IVector diff() {
        if (this.data.length < 2) {
            return IVector.of(new float[0]);
        }
        
        float[] result = new float[this.data.length - 1];
        for (int i = 0; i < result.length; i++) {
            result[i] = this.data[i + 1] - this.data[i];
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector diff(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("差分阶数必须大于等于1: " + n + " / Difference order must be greater than or equal to 1: " + n);
        }
        if (n >= this.data.length) {
            throw new IllegalArgumentException("差分阶数必须小于向量长度: " + n + " >= " + this.data.length + 
                " / Difference order must be less than vector length: " + n + " >= " + this.data.length);
        }
        
        IVector current = this.copy();
        for (int i = 0; i < n; i++) {
            current = current.diff();
        }
        return current;
    }

    // ========== 条件操作实现 / Conditional Operations Implementation ==========
    
    @Override
    public IVector where(boolean[] condition, float x, float y) {
        if (condition == null) {
            throw new IllegalArgumentException("条件数组不能为null / Condition array cannot be null");
        }
        if (condition.length != this.data.length) {
            throw new IllegalArgumentException("条件数组长度与向量长度不匹配: " + condition.length + " != " + this.data.length + 
                " / Condition array length doesn't match vector length: " + condition.length + " != " + this.data.length);
        }
        
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = condition[i] ? x : y;
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector where(boolean[] condition, IVector x, IVector y) {
        if (condition == null) {
            throw new IllegalArgumentException("条件数组不能为null / Condition array cannot be null");
        }
        if (x == null || y == null) {
            throw new IllegalArgumentException("值向量不能为null / Value vectors cannot be null");
        }
        if (condition.length != this.data.length || x.length() != this.data.length || y.length() != this.data.length) {
            throw new IllegalArgumentException("向量长度不匹配 / Vector lengths don't match");
        }
        
        float[] result = new float[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = condition[i] ? x.get(i) : y.get(i);
        }
        return IVector.of(result);
    }

    // ========== 重复和连接操作实现 / Repeat and Concatenation Operations Implementation ==========
    
    @Override
    public IVector repeat(int repeats) {
        if (repeats < 1) {
            throw new IllegalArgumentException("重复次数必须大于等于1: " + repeats + " / Repeat count must be greater than or equal to 1: " + repeats);
        }
        
        float[] result = new float[this.data.length * repeats];
        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < repeats; j++) {
                result[i * repeats + j] = this.data[i];
            }
        }
        return IVector.of(result);
    }
    
    @Override
    public IVector tile(int reps) {
        if (reps < 1) {
            throw new IllegalArgumentException("重复次数必须大于等于1: " + reps + " / Repeat count must be greater than or equal to 1: " + reps);
        }
        
        float[] result = new float[this.data.length * reps];
        for (int i = 0; i < reps; i++) {
            System.arraycopy(this.data, 0, result, i * this.data.length, this.data.length);
        }
        return IVector.of(result);
    }

    // ========== 统计扩展操作实现 / Extended Statistical Operations Implementation ==========
    
    @Override
    public float median() {
        if (this.data.length == 0) {
            throw new ArithmeticException("空向量无法计算中位数 / Cannot compute median for empty vector");
        }
        
        float[] sorted = this.data.clone();
        Arrays.sort(sorted);
        
        if (sorted.length % 2 == 0) {
            // 偶数长度，取中间两个数的平均值 / Even length, take average of two middle numbers
            return (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2.0f;
        } else {
            // 奇数长度，取中间数 / Odd length, take middle number
            return sorted[sorted.length / 2];
        }
    }
    
    @Override
    public float percentile(float q) {
        if (q < 0.0f || q > 100.0f) {
            throw new IllegalArgumentException("百分位数必须在[0,100]范围内: " + q + " / Percentile must be in range [0,100]: " + q);
        }
        if (this.data.length == 0) {
            throw new ArithmeticException("空向量无法计算百分位数 / Cannot compute percentile for empty vector");
        }
        
        float[] sorted = this.data.clone();
        Arrays.sort(sorted);
        
        if (q == 0.0f) return sorted[0];
        if (q == 100.0f) return sorted[sorted.length - 1];
        
        float index = (q / 100.0f) * (sorted.length - 1);
        int lowerIndex = (int) Math.floor(index);
        int upperIndex = (int) Math.ceil(index);
        
        if (lowerIndex == upperIndex) {
            return sorted[lowerIndex];
        }
        
        float weight = index - lowerIndex;
        return sorted[lowerIndex] * (1.0f - weight) + sorted[upperIndex] * weight;
    }
    
    @Override
    public float mode() {
        if (this.data.length == 0) {
            throw new ArithmeticException("空向量无法计算众数 / Cannot compute mode for empty vector");
        }
        
        // 使用HashMap统计频率 / Use HashMap to count frequency
        java.util.Map<Float, Integer> frequency = new java.util.HashMap<>();
        for (float value : this.data) {
            frequency.put(value, frequency.getOrDefault(value, 0) + 1);
        }
        
        float mode = this.data[0];
        int maxFreq = 1;
        
        for (java.util.Map.Entry<Float, Integer> entry : frequency.entrySet()) {
            if (entry.getValue() > maxFreq) {
                maxFreq = entry.getValue();
                mode = entry.getKey();
            }
        }
        
        return mode;
    }

    // ========== 线性代数扩展操作实现 / Extended Linear Algebra Operations Implementation ==========
    
    @Override
    public float norm(float p) {
        if (p < 1.0f) {
            throw new IllegalArgumentException("范数阶数必须大于等于1: " + p + " / Norm order must be greater than or equal to 1: " + p);
        }
        
        if (p == 1.0f) return this.norm1();
        if (p == 2.0f) return this.norm2();
        if (Float.isInfinite(p)) return this.normInf();
        
        float sum = 0.0f;
        for (int i = 0; i < this.data.length; i++) {
            sum += Math.pow(Math.abs(this.data[i]), p);
        }
        return (float) Math.pow(sum, 1.0 / p);
    }
    
    @Override
    public float normInf() {
        float maxAbs = 0.0f;
        for (int i = 0; i < this.data.length; i++) {
            float abs = Math.abs(this.data[i]);
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }
        return maxAbs;
    }
    
    @Override
    public IVector normalize() {
        float norm = this.norm2();
        if (norm == 0.0f) {
            throw new ArithmeticException("向量L2范数为零，无法归一化 / Vector L2 norm is zero, cannot normalize");
        }
        return this.divideByScalar(norm);
    }
    
}
