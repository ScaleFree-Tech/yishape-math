package com.reremouse.lab.math;

import java.util.ArrayList;
import java.util.List;

/**
 * 向量操作接口 / Vector Operations Interface
 * <p>
 * 本接口定义了向量的常用操作，包括基本的数学运算、统计运算、切片索引、通用函数等功能。
 * 提供了创建向量的静态工厂方法和各种向量运算的抽象方法定义。
 * </p>
 * <p>
 * This interface defines common vector operations including basic mathematical operations,
 * statistical operations, slicing and indexing, universal functions and other functionalities.
 * Provides static factory methods for creating vectors and abstract method definitions for various vector operations.
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
 *   <li>工厂方法：创建范围向量、全1向量、零向量 / Factory methods: create range, ones, zeros vectors</li>
 * </ul>
 * 
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 // 创建向量 / Create vectors
 IVector v1 = IVector.of(new float[]{1, 2, 3, 4});
 IVector v2 = IVector.range(10);
 IVector v3 = IVector.ones(5);
 
 // 基本运算 / Basic operations
 IVector sum = v1.add(v2.slice(4));
 float dotProduct = v1.innerProduct(v2.slice(4));
 
 // 统计运算 / Statistical operations
 float mean = v1.mean();
 float std = v1.std();
 
 // 通用函数 / Universal functions
 IVector squared = v1.squre();
 IVector normalized = v1.divideByScala(v1.norm2());
 
 // 切片和索引 / Slicing and indexing
 IVector slice = v2.slice(2, 8, 2);
 IVector fancy = v1.fancyGet(new int[]{0, 2, 3});
 }
 * </pre>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IVector {
    
    /**
     * 向量工厂方法（float数组） / Vector factory method (float array)
     * <p>
     * 使用给定的float数组创建向量实例
     * Creates a vector instance with the given float array
     * </p>
     * 
     * @param data float数组，表示向量数据 / float array representing vector data
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector of(float[] data) {
        return new RereVector(data);
    }

    /**
     * 向量工厂方法（Float包装类数组） / Vector factory method (Float wrapper array)
     * <p>
     * 使用给定的Float包装类数组创建向量实例
     * Creates a vector instance with the given Float wrapper array
     * </p>
     * 
     * @param data Float包装类数组 / Float wrapper array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector of(Float[] data) {
        return of(RereMathUtil.toPrimitive(data));
    }

    /**
     * 向量工厂方法（double数组） / Vector factory method (double array)
     * <p>
     * 使用给定的double数组创建向量实例，自动转换为float
     * Creates a vector instance with the given double array, automatically converted to float
     * </p>
     * 
     * @param data double数组 / double array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector of(double[] data) {
        return of(RereMathUtil.doubleToFloat(data));
    }

    /**
     * 向量工厂方法（Double包装类数组） / Vector factory method (Double wrapper array)
     * <p>
     * 使用给定的Double包装类数组创建向量实例
     * Creates a vector instance with the given Double wrapper array
     * </p>
     * 
     * @param data Double包装类数组 / Double wrapper array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector of(Double[] data) {
        return of(RereMathUtil.toPrimitive(data));
    }
    
    /**
     * 向量工厂方法（int数组） / Vector factory method (int array)
     * <p>
     * 使用给定的int数组创建向量实例，自动转换为float
     * Creates a vector instance with the given int array, automatically converted to float
     * </p>
     * 
     * @param data int数组 / int array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector of(int[] data) {
        return of(RereMathUtil.intToFloat(data));
    }
    
    /**
     * 向量工厂方法（Integer包装类数组） / Vector factory method (Integer wrapper array)
     * <p>
     * 使用给定的Integer包装类数组创建向量实例
     * Creates a vector instance with the given Integer wrapper array
     * </p>
     * 
     * @param data Integer包装类数组 / Integer wrapper array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IVector of(Integer[] data) {
        return of(RereMathUtil.toPrimitive(data));
    }

    /**
     * 创建范围向量（带步长） / Create range vector (with step)
     * <p>
     * 创建一个从start到end（不包含end）、步长为step的向量
     * Creates a vector from start to end (exclusive) with specified step
     * </p>
     * 
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @param step 步长 / Step size
     * @return 范围向量 / Range vector
     * @throws IllegalArgumentException 如果step为0或负数 / if step is 0 or negative
     */
    public static IVector range(int start, int end, int step) {
        List<Float> ls = new ArrayList();
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
     * 创建一个从start到end（不包含end）、步长为1的向量
     * Creates a vector from start to end (exclusive) with step size 1
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
     * 创建一个从0到end（不包含end）、步长为1的向量
     * Creates a vector from 0 to end (exclusive) with step size 1
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
     * 创建一个指定长度的向量，所有元素都为1
     * Creates a vector of specified length with all elements set to 1
     * </p>
     * 
     * @param len 向量长度 / IVector length
     * @return 全1向量 / Ones vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or equal to 0
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
     * 创建一个指定长度的向量，所有元素都为0
     * Creates a vector of specified length with all elements set to 0
     * </p>
     * 
     * @param len 向量长度 / IVector length
     * @return 零向量 / Zeros vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or equal to 0
     */
    public static IVector zeros(int len) {
        float[] v = new float[len];
        // Java数组默认初始化为0，所以不需要显式设置 / Java arrays are initialized to 0 by default
        return IVector.of(v);
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
    public IVector sub(IVector vec);

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
    public IVector add(IVector vec);

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
    public IVector multiply(IVector vec);
    
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
    public float innerProduct(IVector vec);

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
    public IMatrix dot(IMatrix m);

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
    public boolean[] equals(IVector other);

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
    public boolean[] lessThan(IVector other);

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
    public boolean[] greaterThan(IVector other);

    /**
     * 向量减标量 / Vector sub scalar
     * <p>
     * 向量中每个元素减去标量值
     * Subtracts a scalar value from each element in the vector
     * </p>
     * 
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含减法结果 / New vector object containing subtraction result
     */
    public IVector subScalar(float p);

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
    public IVector addScalar(float p);

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
    public IVector multiplyScalar(float p);

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
    public IVector divideByScalar(float p);
    
    /**
     * 向量元素求和 / Vector sum
     * <p>
     * 计算向量中所有元素的和
     * Calculates the sum of all elements in the vector
     * </p>
     * 
     * @return 元素和 / Sum of elements
     */
    public float sum();

    /**
     * 向量最小值 / Vector minimum
     * <p>
     * 找到向量中的最小元素值
     * Finds the minimum element value in the vector
     * </p>
     * 
     * @return 最小值 / Minimum value
     */
    public float min();

    /**
     * 向量最大值 / Vector maximum
     * <p>
     * 找到向量中的最大元素值
     * Finds the maximum element value in the vector
     * </p>
     * 
     * @return 最大值 / Maximum value
     */
    public float max();

    /**
     * 最小值索引 / Index of minimum value
     * <p>
     * 返回向量中最小元素的索引位置
     * Returns the index position of the minimum element in the vector
     * </p>
     * 
     * @return 最小值的索引 / Index of minimum value
     */
    public int argMin();

    /**
     * 最大值索引 / Index of maximum value
     * <p>
     * 返回向量中最大元素的索引位置
     * Returns the index position of the maximum element in the vector
     * </p>
     * 
     * @return 最大值的索引 / Index of maximum value
     */
    public int argMax();

    /**
     * 向量均值 / Vector mean
     * <p>
     * 计算向量中所有元素的平均值
     * Calculates the mean of all elements in the vector
     * </p>
     * 
     * @return 平均值 / Mean value
     */
    public float mean();

    /**
     * 向量标准差 / Vector standard deviation
     * <p>
     * 计算向量的标准差（除以N）
     * Calculates the standard deviation of the vector (divided by N)
     * </p>
     * 
     * @return 标准差 / Standard deviation
     */
    public float std();

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
    public float std(int ddof);

    /**
     * 向量方差 / Vector variance
     * <p>
     * 计算向量的方差（除以N）
     * Calculates the variance of the vector (divided by N)
     * </p>
     * 
     * @return 方差 / Variance
     */
    public float var();

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
    public float var(int ddof);

    /**
     * 向量元素乘积 / Vector product
     * <p>
     * 计算向量中所有元素的乘积
     * Calculates the product of all elements in the vector
     * </p>
     * 
     * @return 元素乘积 / Product of elements
     */
    public float prod();

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
    public IVector clip(float lower, float upper);

    /**
     * 向量峰峰值 / Vector peak-to-peak value
     * <p>
     * 计算向量的峰峰值（最大值减最小值）
     * Calculates the peak-to-peak value of the vector (max - min)
     * </p>
     * 
     * @return 峰峰值 / Peak-to-peak value
     */
    public float ptp();

    /**
     * 向量绝对值 / Vector absolute value
     * <p>
     * 对向量中每个元素取绝对值
     * Takes the absolute value of each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含绝对值结果 / New vector object containing absolute value results
     */
    public IVector abs();

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
    public IVector sqrt();

    /**
     * 向量平方 / Vector square
     * <p>
     * 对向量中每个元素进行平方运算
     * Performs square operation on each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含平方结果 / New vector object containing square results
     */
    public IVector squre();

    /**
     * 向量指数运算 / Vector exponential
     * <p>
     * 对向量中每个元素进行指数运算（e^x）
     * Performs exponential operation (e^x) on each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含指数运算结果 / New vector object containing exponential results
     */
    public IVector exp();

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
    public IVector log();

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
    public IVector log10();

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
    public IVector pow(float m);

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
    public IVector remainder(float value);

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
    public float get(int position);
    
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
    public IVector slice(int end);

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
    public IVector slice(int start, int end);

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
    public IVector slice(int start, int end, int step);

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
    public IVector fancyGet(int[] positions);

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
    public IVector booleanGet(boolean[] booleanIndex);

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
    public IVector set(int position, float value);

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
    public IVector setFromTo(int start, int end, int step, float[] values);
    
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
    public IVector setFromTo(int start, int end, float[] values);

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
    public IVector fill(float value);
    
    /**
     * 向量2范数 / Vector L2 norm
     * <p>
     * 计算向量的2范数（欧几里得范数）
     * Calculates the L2 norm (Euclidean norm) of the vector
     * </p>
     * 
     * @return 2范数 / L2 norm
     */
    public float norm2();

    /**
     * 向量1范数 / Vector L1 norm
     * <p>
     * 计算向量的1范数（曼哈顿范数）
     * Calculates the L1 norm (Manhattan norm) of the vector
     * </p>
     * 
     * @return 1范数 / L1 norm
     */
    public float norm1();

    /**
     * 向量排序 / Vector sorting
     * <p>
     * 对向量中的元素进行升序排序
     * Sorts the elements in the vector in ascending order
     * </p>
     * 
     * @return 排序后的向量（就地操作） / Sorted vector (in-place operation)
     */
    public IVector sort();
    
    /**
     * 向量反转 / Vector reverse
     * <p>
     * 反转向量中元素的顺序
     * Reverses the order of elements in the vector
     * </p>
     * 
     * @return 反转后的向量（就地操作） / Reversed vector (in-place operation)
     */
    public IVector reverse();
    
    /**
     * 获取向量数据数组 / Get vector data array
     * <p>
     * 返回向量的内部数据数组引用
     * Returns a reference to the internal data array of the vector
     * </p>
     * 
     * @return 向量的数据数组 / Data array of the vector
     */
    public float[] getData();

    /**
     * 获取向量长度 / Get vector length
     * <p>
     * 返回向量的长度（元素个数）
     * Returns the length (number of elements) of the vector
     * </p>
     * 
     * @return 向量长度 / IVector length
     */
    public int length();

    /**
     * 向量复制 / Vector copy
     * <p>
     * 创建向量的深拷贝
     * Creates a deep copy of the vector
     * </p>
     * 
     * @return 新的向量对象，与原向量内容相同 / New vector object with the same content as the original
     */
    public IVector copy();
    
    /**
     * 转换为整数数组 / Convert to integer array
     * <p>
     * 将向量转换为整数数组
     * Converts the vector to an integer array
     * </p>
     * 
     * @return 整数数组 / Integer array
     */
    public int[] asIntArray();
    
    /**
     * 转换为双精度数组 / Convert to double array
     * <p>
     * 将向量转换为双精度数组
     * Converts the vector to a double array
     * </p>
     * 
     * @return 双精度数组 / Double array
     */
    public double[] asDoubleArray();

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
    public float euclideanDistance(IVector other);

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
    public float manhattanDistance(IVector other);

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
    public float cosineSimilarity(IVector other);

    // ========== 三角函数操作 / Trigonometric Functions ==========
    
    /**
     * 向量正弦函数 / Vector sine function
     * <p>
     * 对向量中每个元素计算正弦值
     * Computes sine value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含正弦值结果 / New vector object containing sine results
     */
    public IVector sin();
    
    /**
     * 向量余弦函数 / Vector cosine function
     * <p>
     * 对向量中每个元素计算余弦值
     * Computes cosine value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含余弦值结果 / New vector object containing cosine results
     */
    public IVector cos();
    
    /**
     * 向量正切函数 / Vector tangent function
     * <p>
     * 对向量中每个元素计算正切值
     * Computes tangent value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含正切值结果 / New vector object containing tangent results
     */
    public IVector tan();
    
    /**
     * 向量反正弦函数 / Vector arcsine function
     * <p>
     * 对向量中每个元素计算反正弦值
     * Computes arcsine value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含反正弦值结果 / New vector object containing arcsine results
     * @throws ArithmeticException 如果元素值超出[-1,1]范围 / if any element value is outside [-1,1] range
     */
    public IVector arcsin();
    
    /**
     * 向量反余弦函数 / Vector arccosine function
     * <p>
     * 对向量中每个元素计算反余弦值
     * Computes arccosine value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含反余弦值结果 / New vector object containing arccosine results
     * @throws ArithmeticException 如果元素值超出[-1,1]范围 / if any element value is outside [-1,1] range
     */
    public IVector arccos();
    
    /**
     * 向量反正切函数 / Vector arctangent function
     * <p>
     * 对向量中每个元素计算反正切值
     * Computes arctangent value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含反正切值结果 / New vector object containing arctangent results
     */
    public IVector arctan();

    // ========== 双曲函数 / Hyperbolic Functions ==========
    
    /**
     * 向量双曲正弦函数 / Vector hyperbolic sine function
     * <p>
     * 对向量中每个元素计算双曲正弦值
     * Computes hyperbolic sine value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含双曲正弦值结果 / New vector object containing hyperbolic sine results
     */
    public IVector sinh();
    
    /**
     * 向量双曲余弦函数 / Vector hyperbolic cosine function
     * <p>
     * 对向量中每个元素计算双曲余弦值
     * Computes hyperbolic cosine value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含双曲余弦值结果 / New vector object containing hyperbolic cosine results
     */
    public IVector cosh();
    
    /**
     * 向量双曲正切函数 / Vector hyperbolic tangent function
     * <p>
     * 对向量中每个元素计算双曲正切值
     * Computes hyperbolic tangent value for each element in the vector
     * </p>
     * 
     * @return 新的向量对象，包含双曲正切值结果 / New vector object containing hyperbolic tangent results
     */
    public IVector tanh();

    // ========== 舍入函数 / Rounding Functions ==========
    
    /**
     * 向量四舍五入 / Vector round function
     * <p>
     * 对向量中每个元素进行四舍五入
     * Rounds each element in the vector to the nearest integer
     * </p>
     * 
     * @return 新的向量对象，包含四舍五入结果 / New vector object containing rounded results
     */
    public IVector round();
    
    /**
     * 向量向下取整 / Vector floor function
     * <p>
     * 对向量中每个元素进行向下取整
     * Rounds each element in the vector down to the nearest integer
     * </p>
     * 
     * @return 新的向量对象，包含向下取整结果 / New vector object containing floored results
     */
    public IVector floor();
    
    /**
     * 向量向上取整 / Vector ceiling function
     * <p>
     * 对向量中每个元素进行向上取整
     * Rounds each element in the vector up to the nearest integer
     * </p>
     * 
     * @return 新的向量对象，包含向上取整结果 / New vector object containing ceiling results
     */
    public IVector ceil();
    
    /**
     * 向量截断取整 / Vector truncate function
     * <p>
     * 对向量中每个元素进行截断取整（向零方向取整）
     * Truncates each element in the vector towards zero
     * </p>
     * 
     * @return 新的向量对象，包含截断取整结果 / New vector object containing truncated results
     */
    public IVector trunc();

    // ========== 逻辑运算 / Logical Operations ==========
    
    /**
     * 向量逻辑与运算 / Vector logical AND operation
     * <p>
     * 对向量中每个元素进行逻辑与运算（非零为true，零为false）
     * Performs logical AND operation on each element (non-zero as true, zero as false)
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 新的向量对象，包含逻辑与运算结果 / New vector object containing logical AND results
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    public IVector logicalAnd(IVector other);
    
    /**
     * 向量逻辑或运算 / Vector logical OR operation
     * <p>
     * 对向量中每个元素进行逻辑或运算（非零为true，零为false）
     * Performs logical OR operation on each element (non-zero as true, zero as false)
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 新的向量对象，包含逻辑或运算结果 / New vector object containing logical OR results
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    public IVector logicalOr(IVector other);
    
    /**
     * 向量逻辑非运算 / Vector logical NOT operation
     * <p>
     * 对向量中每个元素进行逻辑非运算（非零为false，零为true）
     * Performs logical NOT operation on each element (non-zero as false, zero as true)
     * </p>
     * 
     * @return 新的向量对象，包含逻辑非运算结果 / New vector object containing logical NOT results
     */
    public IVector logicalNot();
    
    /**
     * 向量逻辑异或运算 / Vector logical XOR operation
     * <p>
     * 对向量中每个元素进行逻辑异或运算（非零为true，零为false）
     * Performs logical XOR operation on each element (non-zero as true, zero as false)
     * </p>
     * 
     * @param other 另一个向量 / The other vector
     * @return 新的向量对象，包含逻辑异或运算结果 / New vector object containing logical XOR results
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    public IVector logicalXor(IVector other);

    // ========== 累积操作 / Cumulative Operations ==========
    
    /**
     * 向量累积求和 / Vector cumulative sum
     * <p>
     * 计算向量中元素的累积和
     * Computes the cumulative sum of elements in the vector
     * </p>
     * 
     * @return 新的向量对象，包含累积和结果 / New vector object containing cumulative sum results
     */
    public IVector cumsum();
    
    /**
     * 向量累积乘积 / Vector cumulative product
     * <p>
     * 计算向量中元素的累积乘积
     * Computes the cumulative product of elements in the vector
     * </p>
     * 
     * @return 新的向量对象，包含累积乘积结果 / New vector object containing cumulative product results
     */
    public IVector cumprod();

    // ========== 差分操作 / Difference Operations ==========
    
    /**
     * 向量差分 / Vector difference
     * <p>
     * 计算向量中相邻元素的差分
     * Computes the difference between adjacent elements in the vector
     * </p>
     * 
     * @return 新的向量对象，包含差分结果 / New vector object containing difference results
     */
    public IVector diff();
    
    /**
     * 向量差分（指定阶数） / Vector difference (specified order)
     * <p>
     * 计算向量中元素的n阶差分
     * Computes the n-th order difference of elements in the vector
     * </p>
     * 
     * @param n 差分阶数 / Difference order
     * @return 新的向量对象，包含差分结果 / New vector object containing difference results
     * @throws IllegalArgumentException 如果阶数小于1或大于等于向量长度 / if order is less than 1 or greater than or equal to vector length
     */
    public IVector diff(int n);

    // ========== 条件操作 / Conditional Operations ==========
    
    /**
     * 向量条件选择 / Vector where operation
     * <p>
     * 根据条件选择元素值
     * Selects element values based on conditions
     * </p>
     * 
     * @param condition 条件向量（布尔数组） / Condition vector (boolean array)
     * @param x 条件为true时的值 / Value when condition is true
     * @param y 条件为false时的值 / Value when condition is false
     * @return 新的向量对象，包含条件选择结果 / New vector object containing conditional selection results
     * @throws IllegalArgumentException 如果条件向量长度不匹配 / if condition vector length doesn't match
     */
    public IVector where(boolean[] condition, float x, float y);
    
    /**
     * 向量条件选择（向量值） / Vector where operation (vector values)
     * <p>
     * 根据条件从两个向量中选择元素值
     * Selects element values from two vectors based on conditions
     * </p>
     * 
     * @param condition 条件向量（布尔数组） / Condition vector (boolean array)
     * @param x 条件为true时的值向量 / Value vector when condition is true
     * @param y 条件为false时的值向量 / Value vector when condition is false
     * @return 新的向量对象，包含条件选择结果 / New vector object containing conditional selection results
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't match
     */
    public IVector where(boolean[] condition, IVector x, IVector y);

    // ========== 重复和连接操作 / Repeat and Concatenation Operations ==========
    
    /**
     * 向量重复 / Vector repeat
     * <p>
     * 重复向量中的每个元素指定次数
     * Repeats each element in the vector specified number of times
     * </p>
     * 
     * @param repeats 每个元素重复的次数 / Number of times to repeat each element
     * @return 新的向量对象，包含重复结果 / New vector object containing repeated results
     * @throws IllegalArgumentException 如果重复次数小于1 / if repeat count is less than 1
     */
    public IVector repeat(int repeats);
    
    /**
     * 向量平铺 / Vector tile
     * <p>
     * 将整个向量重复指定次数
     * Repeats the entire vector specified number of times
     * </p>
     * 
     * @param reps 重复次数 / Number of repetitions
     * @return 新的向量对象，包含平铺结果 / New vector object containing tiled results
     * @throws IllegalArgumentException 如果重复次数小于1 / if repeat count is less than 1
     */
    public IVector tile(int reps);

    // ========== 统计扩展操作 / Extended Statistical Operations ==========
    
    /**
     * 向量中位数 / Vector median
     * <p>
     * 计算向量中元素的中位数
     * Computes the median of elements in the vector
     * </p>
     * 
     * @return 中位数 / Median value
     */
    public float median();
    
    /**
     * 向量百分位数 / Vector percentile
     * <p>
     * 计算向量中元素的指定百分位数
     * Computes the specified percentile of elements in the vector
     * </p>
     * 
     * @param q 百分位数（0-100） / Percentile (0-100)
     * @return 百分位数 / Percentile value
     * @throws IllegalArgumentException 如果百分位数超出[0,100]范围 / if percentile is outside [0,100] range
     */
    public float percentile(float q);
    
    /**
     * 向量众数 / Vector mode
     * <p>
     * 计算向量中出现频率最高的元素值
     * Computes the most frequently occurring element value in the vector
     * </p>
     * 
     * @return 众数 / Mode value
     */
    public float mode();

    // ========== 线性代数扩展操作 / Extended Linear Algebra Operations ==========
    
    /**
     * 向量p范数 / Vector Lp norm
     * <p>
     * 计算向量的p范数
     * Calculates the Lp norm of the vector
     * </p>
     * 
     * @param p 范数阶数 / Norm order
     * @return p范数 / Lp norm
     * @throws IllegalArgumentException 如果p小于1 / if p is less than 1
     */
    public float norm(float p);
    
    /**
     * 向量无穷范数 / Vector L-infinity norm
     * <p>
     * 计算向量的无穷范数（最大绝对值）
     * Calculates the L-infinity norm (maximum absolute value) of the vector
     * </p>
     * 
     * @return 无穷范数 / L-infinity norm
     */
    public float normInf();
    
    /**
     * 向量归一化 / Vector normalization
     * <p>
     * 将向量归一化为单位向量（L2范数为1）
     * Normalizes the vector to unit vector (L2 norm equals 1)
     * </p>
     * 
     * @return 新的向量对象，包含归一化结果 / New vector object containing normalized results
     * @throws ArithmeticException 如果向量L2范数为零 / if vector L2 norm is zero
     */
    public IVector normalize();

    // ========== 随机数生成 / Random Number Generation ==========
    
    /**
     * 创建随机向量 / Create random vector
     * <p>
     * 创建指定长度的随机向量，元素值在[0,1)范围内
     * Creates a random vector of specified length with elements in [0,1) range
     * </p>
     * 
     * @param length 向量长度 / IVector length
     * @return 随机向量 / Random vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or equal to 0
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
     * 创建指定长度的正态分布随机向量
     * Creates a normal distribution random vector of specified length
     * </p>
     * 
     * @param length 向量长度 / IVector length
     * @param mean 均值 / Mean
     * @param std 标准差 / Standard deviation
     * @return 正态分布随机向量 / Normal distribution random vector
     * @throws IllegalArgumentException 如果长度小于等于0或标准差小于等于0 / if length is less than or equal to 0 or std is less than or equal to 0
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

    // ========== 线性空间生成 / Linear Space Generation ==========
    
    /**
     * 创建线性空间向量 / Create linear space vector
     * <p>
     * 创建指定数量的等间距数值向量
     * Creates a vector with specified number of equally spaced values
     * </p>
     * 
     * @param start 起始值 / Start value
     * @param stop 结束值 / Stop value
     * @param num 元素数量 / Number of elements
     * @return 线性空间向量 / Linear space vector
     * @throws IllegalArgumentException 如果元素数量小于2 / if number of elements is less than 2
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
     * 创建指定数量的对数等间距数值向量
     * Creates a vector with specified number of logarithmically equally spaced values
     * </p>
     * 
     * @param start 起始值（10^start） / Start value (10^start)
     * @param stop 结束值（10^stop） / Stop value (10^stop)
     * @param num 元素数量 / Number of elements
     * @return 对数空间向量 / Logarithmic space vector
     * @throws IllegalArgumentException 如果元素数量小于2 / if number of elements is less than 2
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
