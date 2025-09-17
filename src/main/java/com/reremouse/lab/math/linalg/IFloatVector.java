package com.reremouse.lab.math.linalg;

import com.reremouse.lab.math.RereMathUtil;
import java.util.ArrayList;
import java.util.List;

/**
 * Float类型向量操作接口 / Float Vector Operations Interface
 * <p>
 * 本接口定义了Float类型向量的常用操作，包括基本的数学运算、统计运算、切片索引、通用函数等功能。
 * 这是IVectorGeneric<Double>的别名，保持向后兼容性。
 * </p>
 * <p>
 This interface defines common vector operations for Float type vectors, including basic mathematical
 operations, statistical operations, slicing and indexing, universal functions
 and other functionalities. This is an alias for IVector<Double> to maintain backward compatibility.
 * </p>
 *
 * <h3>主要功能 / Main Features:</h3>
 * <ul>
 * <li>基本数学运算：加法、减法、乘法、内积 / Basic math operations: add, subtract, multiply,
 * inner product</li>
 * <li>标量运算：与标量的四则运算 / Scalar operations: arithmetic operations with
 * scalars</li>
 * <li>统计运算：求和、均值、方差、标准差、最值 / Statistical operations: sum, mean, variance,
 * standard deviation, min/max</li>
 * <li>通用函数：开方、平方、指数、对数、幂运算 / Universal functions: sqrt, square, exp, log,
 * power</li>
 * <li>切片索引：范围切片、花式索引、布尔索引 / Slicing and indexing: range slicing, fancy
 * indexing, boolean indexing</li>
 * <li>比较运算：元素级比较操作 / Comparison operations: element-wise comparison</li>
 * <li>数据转换：类型转换、数据获取 / Data conversion: type conversion, data access</li>
 * <li>工厂方法：创建范围向量、全1向量、零向量 / Factory methods: create range, ones, zeros
 * vectors</li>
 * </ul>
 *
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 // 创建向量 / Create vectors
 IFloatVector v1 = IFloatVector.of(new float[]{1, 2, 3, 4});
 IFloatVector v2 = IFloatVector.range(10);
 IFloatVector v3 = IFloatVector.ones(5);

 // 基本运算 / Basic operations
 IFloatVector sum = v1.add(v2.slice(4));
 Float dotProduct = v1.innerProduct(v2.slice(4));

 // 统计运算 / Statistical operations
 Float mean = v1.mean();
 Float std = v1.std();

 // 通用函数 / Universal functions
 IFloatVector squared = v1.squre();
 IFloatVector normalized = v1.divideByScala(v1.norm2());

 // 切片和索引 / Slicing and indexing
 IFloatVector slice = v2.slice(2, 8, 2);
 IFloatVector fancy = v1.fancyGet(new int[]{0, 2, 3});
 }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IFloatVector extends IVector<Float> {

    /**
     * 向量工厂方法（Float数组） / Vector factory method (Float array)
     * <p>
     * 使用给定的Float数组创建向量实例 Creates a vector instance with the given Float array
     * </p>
     *
     * @param data Float数组，表示向量数据 / Float array representing vector data
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IFloatVector of(float[] data) {
        return new RereFloatVector(data);
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
    public static IFloatVector of(Float[] data) {
        return of(RereMathUtil.toPrimitive(data));
    }

    /**
     * 向量工厂方法（double数组） / Vector factory method (double array)
     * <p>
     * 使用给定的double数组创建向量实例，自动转换为Float Creates a vector instance with the given
     * double array, automatically converted to Float
     * </p>
     *
     * @param data double数组 / double array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IFloatVector of(double[] data) {
        return of(RereMathUtil.doubleToFloat(data));
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
    public static IFloatVector of(Double[] data) {
        return of(RereMathUtil.toPrimitive(data));
    }

    /**
     * 向量工厂方法（int数组） / Vector factory method (int array)
     * <p>
     * 使用给定的int数组创建向量实例，自动转换为Float Creates a vector instance with the given int
     * array, automatically converted to Float
     * </p>
     *
     * @param data int数组 / int array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IFloatVector of(int[] data) {
        return of(RereMathUtil.intToFloat(data));
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
    public static IFloatVector of(Integer[] data) {
        return of(RereMathUtil.toPrimitive(data));
    }
    
    
        /**
     * 向量工厂方法（单个值） / Vector factory method (single value)
     * <p>
     * 创建包含单个值的向量 Creates a vector containing a single value
     * </p>
     *
     * @param value 单个值 / Single value
     * @return 包含单个值的向量 / Vector containing single value
     */
    public static IFloatVector of(Float value) {
        return of(new float[]{value});
    }

    /**
     * 向量工厂方法（两个值） / Vector factory method (two values)
     * <p>
     * 创建包含两个值的向量 Creates a vector containing two values
     * </p>
     *
     * @param value1 第一个值 / First value
     * @param value2 第二个值 / Second value
     * @return 包含两个值的向量 / Vector containing two values
     */
    public static IFloatVector of(Float value1, Float value2) {
        return of(new float[]{value1, value2});
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
    public static IFloatVector range(int start, int end, int step) {
        List<Float> ls = new ArrayList<>();
        int p = start;
        while (p < end) {
            ls.add((float) p);
            p += step;
        }
        var as = ls.toArray(Float[]::new);
        return IFloatVector.of(as);
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
    public static IFloatVector range(int start, int end) {
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
    public static IFloatVector range(int end) {
        return range(0, end, 1);
    }

    /**
     * 创建全1向量 / Create ones vector
     * <p>
     * 创建一个指定长度的向量，所有元素都为1 Creates a vector of specified length with all
     * elements set to 1
     * </p>
     *
     * @param len 向量长度 / IFloatVector length
     * @return 全1向量 / Ones vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IFloatVector ones(int len) {
        float[] v = new float[len];
        for (int i = 0; i < len; i++) {
            v[i] = 1f;
        }
        return IFloatVector.of(v);
    }

    /**
     * 创建零向量 / Create zeros vector
     * <p>
     * 创建一个指定长度的向量，所有元素都为0 Creates a vector of specified length with all
     * elements set to 0
     * </p>
     *
     * @param len 向量长度 / IFloatVector length
     * @return 零向量 / Zeros vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IFloatVector zeros(int len) {
        float[] v = new float[len];
        // Java数组默认初始化为0，所以不需要显式设置 / Java arrays are initialized to 0 by default
        return IFloatVector.of(v);
    }

    // ========== 随机数生成 / Random Number Generation ==========
    /**
     * 创建随机向量 / Create random vector
     * <p>
     * 创建指定长度的随机向量，元素值在[0,1)范围内 Creates a random vector of specified length with
     * elements in [0,1) range
     * </p>
     *
     * @param length 向量长度 / IFloatVector length
     * @return 随机向量 / Random vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IFloatVector rand(int length) {
        if (length <= 0) {
            throw new IllegalArgumentException("向量长度必须大于0 / IVector length must be greater than 0");
        }
        float[] v = new float[length];
        for (int i = 0; i < length; i++) {
            v[i] = (float) Math.random();
        }
        return IFloatVector.of(v);
    }

    /**
     * 创建正态分布随机向量 / Create normal distribution random vector
     * <p>
     * 创建指定长度的正态分布随机向量 Creates a normal distribution random vector of specified
     * length
     * </p>
     *
     * @param length 向量长度 / IFloatVector length
     * @param mean 均值 / Mean
     * @param std 标准差 / Standard deviation
     * @return 正态分布随机向量 / Normal distribution random vector
     * @throws IllegalArgumentException 如果长度小于等于0或标准差小于等于0 / if length is less
     * than or equal to 0 or std is less than or equal to 0
     */
    public static IFloatVector randn(int length, Float mean, Float std) {
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
        return IFloatVector.of(v);
    }
    
        public static IFloatVector randn(int length) {
        if (length <= 0) {
            throw new IllegalArgumentException("向量长度必须大于0 / IVector length must be greater than 0");
        }

        float[] v = new float[length];
        for (int i = 0; i < length; i++) {
            // Box-Muller变换生成正态分布随机数 / Box-Muller transform for normal distribution
            double u1 = Math.random();
            double u2 = Math.random();
            double z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            v[i] = (float) z0;
        }
        return IFloatVector.of(v);
    }

    // ========== 线性空间生成 / Linear Space Generation ==========
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
    public static IFloatVector linspace(Float start, Float stop, int num) {
        if (num < 2) {
            throw new IllegalArgumentException("元素数量必须大于等于2 / Number of elements must be greater than or equal to 2");
        }
        float[] v = new float[num];
        Float step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            v[i] = start + i * step;
        }
        return IFloatVector.of(v);
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
    public static IFloatVector logspace(Float start, Float stop, int num) {
        if (num < 2) {
            throw new IllegalArgumentException("元素数量必须大于等于2 / Number of elements must be greater than or equal to 2");
        }
        float[] v = new float[num];
        Float step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            v[i] = (float) Math.pow(10, start + i * step);
        }
        return IFloatVector.of(v);
    }

    /**
     * 获取向量数据数组 / Get vector data array
     * <p>
     * 返回向量的内部数据数组引用 Returns a reference to the internal data array of the
     * vector
     * </p>
     *
     * @return 向量的数据数组 / Data array of the vector
     */
    public float[] getData();


}
