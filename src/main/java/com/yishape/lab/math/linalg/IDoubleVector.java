package com.yishape.lab.math.linalg;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.random.RngFactory;
import com.yishape.lab.math.random.RngProvider;

/**
 * Double类型向量操作接口 / Double Vector Operations Interface
 * <p>
 * 本接口定义了Double类型向量的常用操作，包括基本的数学运算、统计运算、切片索引、通用函数等功能。
 * 这是IVectorGeneric<Double>的别名，保持向后兼容性。
 * </p>
 * <p>
 * This interface defines common vector operations for Double type vectors,
 * including basic mathematical operations, statistical operations, slicing and
 * indexing, universal functions and other functionalities. This is an alias for
 * IVector<Double> to maintain backward compatibility.
 * </p>
 * <p>
 * <strong>数组导出：</strong>{@link IVector#toDoubleArray()} 在
 * {@link RereDoubleVector} 等实现中返回<strong>防御性拷贝</strong>；需要与内部共享且可写的后备数组请使用
 * {@link #getData()}。
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
 * // 创建向量 / Create vectors
 * IDoubleVector v1 = IDoubleVector.of(new double[]{1, 2, 3, 4});
 * IDoubleVector v2 = IDoubleVector.range(10);
 * IDoubleVector v3 = IDoubleVector.ones(5);
 *
 * // 基本运算 / Basic operations
 * IDoubleVector sum = v1.add(v2.slice(4));
 * Double dotProduct = v1.innerProduct(v2.slice(4));
 *
 * // 统计运算 / Statistical operations
 * Double mean = v1.mean();
 * Double std = v1.std();
 *
 * // 通用函数 / Universal functions
 * IDoubleVector squared = v1.squre();
 * IDoubleVector normalized = v1.divideByScala(v1.norm2());
 *
 * // 切片和索引 / Slicing and indexing
 * IDoubleVector slice = v2.slice(2, 8, 2);
 * IDoubleVector fancy = v1.fancyGet(new int[]{0, 2, 3});
 * }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IDoubleVector extends IVector<Double> {

    /**
     * 向量工厂方法（Double数组） / Vector factory method (Double array)
     * <p>
     * 使用给定的Double数组创建向量实例 Creates a vector instance with the given Double array
     * </p>
     *
     * @param data Double数组，表示向量数据 / Double array representing vector data
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IDoubleVector of(double[] data) {
        return new RereDoubleVector(data);
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
    public static IDoubleVector of(Double[] data) {
        return of(RereMathUtil.toPrimitive(data));
    }

    /**
     * 向量工厂方法（double数组） / Vector factory method (double array)
     * <p>
     * 使用给定的double数组创建向量实例，自动转换为Double Creates a vector instance with the given
     * double array, automatically converted to Double
     * </p>
     *
     * @param data double数组 / double array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IDoubleVector of(float[] data) {
        return of(RereMathUtil.floatToDouble(data));
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
    public static IDoubleVector of(Float[] data) {
        return of(RereMathUtil.toPrimitive(data));
    }

    /**
     * 向量工厂方法（int数组） / Vector factory method (int array)
     * <p>
     * 使用给定的int数组创建向量实例，自动转换为Double Creates a vector instance with the given int
     * array, automatically converted to Double
     * </p>
     *
     * @param data int数组 / int array
     * @return 新的向量实例 / New vector instance
     * @throws IllegalArgumentException 如果数据为null / if data is null
     */
    public static IDoubleVector of(int[] data) {
        return of(RereMathUtil.intToDouble(data));
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
    public static IDoubleVector of(Integer[] data) {
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
    public static IDoubleVector of(double value) {
        return of(new double[]{value});
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
    public static IDoubleVector of(double value1, double value2) {
        return of(new double[]{value1, value2});
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
    public static IDoubleVector range(int start, int end, int step) {
        int count = 0;
        for (int p = start; p < end; p += step) {
            count++;
        }
        double[] result = new double[count];
        int idx = 0;
        for (int p = start; p < end; p += step) {
            result[idx++] = p;
        }
        return IDoubleVector.of(result);
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
    public static IDoubleVector range(int start, int end) {
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
    public static IDoubleVector range(int end) {
        return range(0, end, 1);
    }

    /**
     * 创建全1向量 / Create ones vector
     * <p>
     * 创建一个指定长度的向量，所有元素都为1 Creates a vector of specified length with all
     * elements set to 1
     * </p>
     *
     * @param len 向量长度 / IDoubleVector length
     * @return 全1向量 / Ones vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IDoubleVector ones(int len) {
        double[] v = new double[len];
        for (int i = 0; i < len; i++) {
            v[i] = 1f;
        }
        return IDoubleVector.of(v);
    }

    /**
     * 创建零向量 / Create zeros vector
     * <p>
     * 创建一个指定长度的向量，所有元素都为0 Creates a vector of specified length with all
     * elements set to 0
     * </p>
     *
     * @param len 向量长度 / IDoubleVector length
     * @return 零向量 / Zeros vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IDoubleVector zeros(int len) {
        double[] v = new double[len];
        // Java数组默认初始化为0，所以不需要显式设置 / Java arrays are initialized to 0 by default
        return IDoubleVector.of(v);
    }

    // ========== 随机数生成 / Random Number Generation ==========
    /**
     * 创建随机向量 / Create random vector
     * <p>
     * 创建指定长度的随机向量，元素值在[0,1)范围内 Creates a random vector of specified length with
     * elements in [0,1) range
     * </p>
     *
     * @param length 向量长度 / IDoubleVector length
     * @return 随机向量 / Random vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    public static IDoubleVector rand(int length) {
        return rand(length, RngFactory.createDefault());
    }

    public static IDoubleVector rand(int length, RngProvider rng) {
        if (length <= 0) {
            throw new IllegalArgumentException("向量长度必须大于0 / IVector length must be greater than 0");
        }
        double[] v = new double[length];
        for (int i = 0; i < length; i++) {
            v[i] = rng.nextDouble();
        }
        return IDoubleVector.of(v);
    }

    public static IDoubleVector randn(int length) {
        return randn(length, RngFactory.createDefault());
    }

    public static IDoubleVector randn(int length, RngProvider rng) {
        if (length <= 0) {
            throw new IllegalArgumentException("向量长度必须大于0 / IVector length must be greater than 0");
        }
        double[] v = new double[length];
        for (int i = 0; i < length; i++) {
            v[i] = rng.nextGaussian();
        }
        return IDoubleVector.of(v);
    }

    public static IDoubleVector randn(int length, Double mean, Double std) {
        if (length <= 0) {
            throw new IllegalArgumentException("向量长度必须大于0 / IVector length must be greater than 0");
        }
        if (std <= 0) {
            throw new IllegalArgumentException("标准差必须大于0 / Standard deviation must be greater than 0");
        }
        RngProvider rng = RngFactory.createDefault();
        double[] v = new double[length];
        for (int i = 0; i < length; i++) {
            v[i] = mean + std * rng.nextGaussian();
        }
        return IDoubleVector.of(v);
    }

    public static IDoubleVector randn(int length, Double mean, Double std, RngProvider rng) {
        if (length <= 0) {
            throw new IllegalArgumentException("向量长度必须大于0 / IVector length must be greater than 0");
        }
        if (std <= 0) {
            throw new IllegalArgumentException("标准差必须大于0 / Standard deviation must be greater than 0");
        }
        double[] v = new double[length];
        for (int i = 0; i < length; i++) {
            v[i] = mean + std * rng.nextGaussian();
        }
        return IDoubleVector.of(v);
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
    public static IDoubleVector linspace(Double start, Double stop, int num) {
        if (num < 2) {
            throw new IllegalArgumentException("元素数量必须大于等于2 / Number of elements must be greater than or equal to 2");
        }
        double[] v = new double[num];
        Double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            v[i] = start + i * step;
        }
        return IDoubleVector.of(v);
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
    public static IDoubleVector logspace(Double start, Double stop, int num) {
        if (num < 2) {
            throw new IllegalArgumentException("元素数量必须大于等于2 / Number of elements must be greater than or equal to 2");
        }
        double[] v = new double[num];
        Double step = (stop - start) / (num - 1);
        for (int i = 0; i < num; i++) {
            v[i] = (float) Math.pow(10, start + i * step);
        }
        return IDoubleVector.of(v);
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
    public double[] getData();

    // ========== 一维双精度工具（见 {@link IVector} 同名静态方法）==========
    static IVector.HistogramResult histogram(double[] x, int bins) {
        return IVector.histogram(x, bins);
    }

    static IVector.HistogramResult histogram(IVector<Double> x, int bins) {
        return IVector.histogram(x, bins);
    }

    static int[] digitize(double[] x, double[] bins) {
        return IVector.digitize(x, bins);
    }

    static int[] digitize(IVector<Double> x, double[] bins) {
        return IVector.digitize(x, bins);
    }

    static double[] polyfit(double[] x, double[] y, int deg) {
        return IVector.polyfit(x, y, deg);
    }

    @SuppressWarnings("unchecked")
    static IVector<Double> polyfit(IVector<Double> x, IVector<Double> y, int deg) {
        return (IVector<Double>) IVector.polyfit(x, y, deg);
    }

    static double[] where(boolean[] cond, double x, double y) {
        return IVector.where(cond, x, y);
    }

    static double[] where(boolean[] cond, double[] x, double[] y) {
        return IVector.where(cond, x, y);
    }

    static IVector<Double> where(IVector<Double> v, boolean[] cond, double x, double y) {
        return IVector.where(v, cond, x, y);
    }

    // ========== 协变返回类型覆盖 / Covariant Return Type Overrides ==========
    @Override
    public IDoubleVector divideByScalar(double p);

    @Override
    public IDoubleVector subScalar(double p);

    @Override
    public IDoubleVector multiplyByScalar(double p);

    @Override
    public IDoubleVector addScalar(double p);

    @Override
    public IDoubleVector divideInPlace(double alpha);

    @Override
    public IDoubleVector copy();

    @Override
    public IDoubleVector slice(int start, int end);

    @Override
    public IDoubleVector slice(int start);

    @Override
    public IDoubleVector abs();

    @Override
    public IDoubleVector sqrt();

    @Override
    public IDoubleVector square();

    @Override
    public IDoubleVector exp();

    @Override
    public IDoubleVector log();

    @Override
    public IDoubleVector log10();

    @Override
    public IDoubleVector sigmoid();

    @Override
    public IDoubleVector relu();

    @Override
    public IDoubleVector pow(double m);

    @Override
    public IDoubleVector sin();

    @Override
    public IDoubleVector cos();

    @Override
    public IDoubleVector tan();

    @Override
    public IDoubleVector arcsin();

    @Override
    public IDoubleVector arccos();

    @Override
    public IDoubleVector arctan();

    @Override
    public IDoubleVector sinh();

    @Override
    public IDoubleVector cosh();

    @Override
    public IDoubleVector tanh();

    @Override
    public IDoubleVector round();

    @Override
    public IDoubleVector floor();

    @Override
    public IDoubleVector ceil();

    @Override
    public IDoubleVector trunc();

    @Override
    public IDoubleVector slice(int start, int end, int step);

    @Override
    public IDoubleVector slice(String sliceExpression);

    @Override
    public IDoubleVector fancyGet(int[] positions);

    @Override
    public IDoubleVector booleanGet(boolean[] booleanIndex);

    @Override
    public IDoubleVector add(IVector<Double> vec);

    @Override
    public IDoubleVector sub(IVector<Double> vec);

    @Override
    public IDoubleVector multiply(IVector<Double> vec);

    @Override
    public IDoubleVector mmul(IMatrix<Double> other);

    @Override
    public IDoubleVector dot(IMatrix<Double> m);

    @Override
    public IDoubleVector clip(double lower, double upper);

    @Override
    public IDoubleVector sort();

    @Override
    public IDoubleVector reverse();

    @Override
    public IDoubleVector cumsum();

    @Override
    public IDoubleVector cumprod();

    @Override
    public IDoubleVector diff();

    @Override
    public IDoubleVector diff(int n);

    @Override
    public IDoubleVector where(boolean[] condition, Double x, Double y);

    @Override
    public IDoubleVector where(boolean[] condition, IVector<Double> x, IVector<Double> y);

    @Override
    public IDoubleVector repeat(int repeats);

    @Override
    public IDoubleVector tile(int reps);

    @Override
    public IDoubleVector normalize();

    @Override
    public IDoubleVector reciprocal();

    @Override
    public IDoubleVector cross(IVector<Double> other);

    @Override
    public IDoubleVector map(java.util.function.Function<Double, Double> fun);

    @Override
    public IDoubleVector concat(IVector<Double> other);

    @Override
    public IDoubleVector sign();

    @Override
    public IDoubleMatrix asColumnVector();

    @Override
    public default IDoubleVector divide(IVector<Double> other) {
        var oo = other.reciprocal();
        return this.multiply(oo);
    }

    // ========== 协变归约返回类型 / Covariant Reduction Return Types ==========
    @Override public IDoubleVector sum();
    @Override public IDoubleVector mean();
    @Override public IDoubleVector min();
    @Override public IDoubleVector max();
    @Override public IDoubleVector std();
    @Override public IDoubleVector std(int ddof);
    @Override public IDoubleVector var();
    @Override public IDoubleVector var(int ddof);
    @Override public IDoubleVector prod();
    @Override public IDoubleVector norm2();
    @Override public IDoubleVector norm1();
    @Override public IDoubleVector ptp();
    @Override public IDoubleVector innerProduct(IVector<Double> vec);
    @Override public IDoubleVector dot(IVector<Double> vec);

    // ========== 激活函数协变覆盖 / Activation Function Covariant Overrides ==========

    @Override default IDoubleVector softmax() { return (IDoubleVector) IVector.super.softmax(); }
    @Override default IDoubleVector logSoftmax() { return (IDoubleVector) IVector.super.logSoftmax(); }
    @Override default IDoubleVector gelu() { return (IDoubleVector) IVector.super.gelu(); }
    @Override default IDoubleVector silu() { return (IDoubleVector) IVector.super.silu(); }
    @Override default IDoubleVector leakyRelu(double alpha) { return (IDoubleVector) IVector.super.leakyRelu(alpha); }
    @Override default IDoubleVector elu(double alpha) { return (IDoubleVector) IVector.super.elu(alpha); }
    @Override default IDoubleVector selu() { return (IDoubleVector) IVector.super.selu(); }
    @Override default IDoubleVector mish() { return (IDoubleVector) IVector.super.mish(); }
    @Override default IDoubleVector softplus(double beta) { return (IDoubleVector) IVector.super.softplus(beta); }
    @Override default IDoubleVector hardtanh(double minVal, double maxVal) { return (IDoubleVector) IVector.super.hardtanh(minVal, maxVal); }
    @Override default IDoubleVector clamp(double min, double max) { return (IDoubleVector) IVector.super.clamp(min, max); }
    @Override default IDoubleVector neg() { return (IDoubleVector) IVector.super.neg(); }
    @Override default IDoubleVector rsub(double scalar) { return (IDoubleVector) IVector.super.rsub(scalar); }
    @Override default IDoubleVector rdiv(double scalar) { return (IDoubleVector) IVector.super.rdiv(scalar); }
    @Override default IDoubleVector layerNorm(IVector<Double> gamma, IVector<Double> beta, double eps) { return (IDoubleVector) IVector.super.layerNorm(gamma, beta, eps); }
    @Override default IDoubleVector dropout(double p) { return (IDoubleVector) IVector.super.dropout(p); }

}
