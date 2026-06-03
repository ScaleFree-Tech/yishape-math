package com.yishape.lab.math.linalg;

import com.yishape.lab.math.RereMathUtil;
import java.io.Serializable;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * 泛型向量操作接口 / Generic Vector Operations Interface
 * <p>
 * 本接口定义了支持泛型数值类型的向量操作，包括基本的数学运算、统计运算、切片索引、通用函数等功能。
 * 提供了创建向量的静态工厂方法和各种向量运算的抽象方法定义。
 * </p>
 * <p>
 * This interface defines vector operations supporting generic numeric types,
 * including basic mathematical operations, statistical operations, slicing and
 * indexing, universal functions and other functionalities. Provides static
 * factory methods for creating vectors and abstract method definitions for
 * various vector operations.
 * </p>
 *
 * <h3>静态工厂方法架构 / Static Factory Method Architecture:</h3>
 * <p>
 * 本接口采用委托模式，静态工厂方法委托给具体实现类： This interface uses delegation pattern, static
 * factory methods delegate to concrete implementations:
 * </p>
 * <ul>
 * <li>**用户调用** / **User calls**: {@code IVector.of(data)} 或
 * {@code Linalg.vector(data)}</li>
 * <li>**委托链** / **Delegation chain**:
 * {@code Linalg → IVector → IFloatVector/IDoubleVector}</li>
 * <li>**类型推断** / **Type inference**: 根据输入数据类型自动选择合适的实现 / Automatically selects
 * appropriate implementation based on input data type</li>
 * <li>**支持类型** / **Supported types**: Float, Double（默认）/ Float, Double
 * (default)</li>
 * </ul>
 *
 * <h3>使用示例 / Usage Examples:</h3>
 * <pre>
 * {@code
 * // 通过IVector创建 / Create via IVector
 * IVector<Double> v1 = IVector.of(new double[]{1, 2, 3});
 * IVector<Float> v2 = IVector.of(new float[]{1f, 2f, 3f});
 *
 * // 通过Linalg创建（推荐）/ Create via Linalg (recommended)
 * IVector<Double> v3 = Linalg.vector(new double[]{1, 2, 3});
 * IVector<Float> v4 = Linalg.vector(new float[]{1f, 2f, 3f});
 *
 * // 范围向量 / Range vectors
 * IVector<Double> range = IVector.range(0, 10, 2);  // [0, 2, 4, 6, 8]
 * IVector<Float> rangeF = IVector.range(5, Float.class);  // [0, 1, 2, 3, 4]
 *
 * // 特殊向量 / Special vectors
 * IVector<Double> ones = IVector.ones(5);     // [1, 1, 1, 1, 1]
 * IVector<Double> zeros = IVector.zeros(3);   // [0, 0, 0]
 * IVector<Double> random = IVector.rand(4);   // 随机向量 / Random vector
 * }
 * </pre>
 *
 * @param <T> 数值类型，必须继承自Number / Numeric type, must extend Number
 * @author lteb2
 * @version 1.0
 * @since 1.0
 * @see Linalg 推荐使用Linalg类的工厂方法 / Recommended to use Linalg class factory
 * methods
 * @see IFloatVector Float类型向量的具体实现 / Concrete implementation for Float type
 * vectors
 * @see IDoubleVector Double类型向量的具体实现 / Concrete implementation for Double type
 * vectors
 */
public interface IVector<T extends Number> extends Serializable{

    public static final long serialVersionUID = 3L;
    
    // ========== 泛型工厂方法支持 / Generic Factory Method Support ==========
    // 注意：这些方法委托给具体实现类，不包含具体逻辑。推荐使用 Linalg 类的工厂方法。
    // Note: These methods delegate to concrete implementations and contain no specific logic. 
    // It's recommended to use factory methods from the Linalg class.
    /**
     * 向量工厂方法（double数组） / Vector factory method (double array)
     * <p>
     * 使用给定的double数组创建向量实例。此方法委托给 {@link IDoubleVector#of(double[])} 实现。 Creates
     * a vector instance with the given double array. This method delegates to
     * {@link IDoubleVector#of(double[])}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(data)} 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(data)} instead of
     * calling this method directly.
     * </p>
     *
     * @param data double数组，表示向量数据 / double array representing vector data
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see Linalg#vector(double[]) 推荐的工厂方法 / Recommended factory method
     * @see IDoubleVector#of(double[]) 实际实现方法 / Actual implementation method
     */
    public static IDoubleVector of(double...data) {
        return IDoubleVector.of(data);
    }

    /**
     * 向量工厂方法（Double包装类数组） / Vector factory method (Double wrapper array)
     * <p>
     * 使用给定的Double包装类数组创建向量实例。此方法委托给 {@link IDoubleVector#of(Double[])} 实现。
     * Creates a vector instance with the given Double wrapper array. This
     * method delegates to {@link IDoubleVector#of(Double[])}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(data)} 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(data)} instead of
     * calling this method directly.
     * </p>
     *
     * @param data Double包装类数组，表示向量数据 / Double wrapper array representing vector
     * data
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see Linalg#vector(Double[]) 推荐的工厂方法 / Recommended factory method
     * @see IDoubleVector#of(Double[]) 实际实现方法 / Actual implementation method
     */
    public static IDoubleVector of(Double[] data) {
        return IDoubleVector.of(data);
    }

    /**
     * 从Double List创建向量
     *
     * @param data
     * @return
     */
    public static IDoubleVector of(List<Double> data) {
        return IVector.of(data.toArray(Double[]::new));
    }

    /**
     * 向量工厂方法（float数组） / Vector factory method (float array)
     * <p>
     * 使用给定的float数组创建向量实例。此方法委托给 {@link IFloatVector#of(float[])} 实现。 Creates a
     * vector instance with the given float array. This method delegates to
     * {@link IFloatVector#of(float[])}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(data)} 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(data)} instead of
     * calling this method directly.
     * </p>
     *
     * @param data float数组，表示向量数据 / float array representing vector data
     * @return 新的向量实例（IFloatVector）/ New vector instance (IFloatVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see Linalg#vector(float[]) 推荐的工厂方法 / Recommended factory method
     * @see IFloatVector#of(float[]) 实际实现方法 / Actual implementation method
     */
    public static IFloatVector of(float[] data) {
        return IFloatVector.of(data);
    }

    /**
     * 向量工厂方法（Float包装类数组） / Vector factory method (Float wrapper array)
     * <p>
     * 使用给定的Float包装类数组创建向量实例。此方法委托给 {@link IFloatVector#of(Float[])} 实现。 Creates
     * a vector instance with the given Float wrapper array. This method
     * delegates to {@link IFloatVector#of(Float[])}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(data)} 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(data)} instead of
     * calling this method directly.
     * </p>
     *
     * @param data Float包装类数组，表示向量数据 / Float wrapper array representing vector
     * data
     * @return 新的向量实例（IFloatVector）/ New vector instance (IFloatVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see Linalg#vector(Float[]) 推荐的工厂方法 / Recommended factory method
     * @see IFloatVector#of(Float[]) 实际实现方法 / Actual implementation method
     */
    public static IFloatVector of(Float[] data) {
        return IFloatVector.of(data);
    }

    /**
     *
     * @param data
     * @return
     */
    public static IFloatVector ofFloatList(List<Float> data) {
        return IVector.of(data.toArray(Float[]::new));
    }

    /**
     * 向量工厂方法（int数组） / Vector factory method (int array)
     * <p>
     * 使用给定的int数组创建向量实例，自动转换为Double类型。此方法委托给 {@link IDoubleVector#of(int[])} 实现。
     * Creates a vector instance with the given int array, automatically
     * converted to Double type. This method delegates to
     * {@link IDoubleVector#of(int[])}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(data)} 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(data)} instead of
     * calling this method directly.
     * </p>
     *
     * @param data int数组，表示向量数据 / int array representing vector data
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see Linalg#vector(int[]) 推荐的工厂方法 / Recommended factory method
     * @see IDoubleVector#of(int[]) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IVector<T> of(int[] data) {
        return (IVector<T>) IDoubleVector.of(data);
    }

    /**
     * 向量工厂方法（Integer包装类数组） / Vector factory method (Integer wrapper array)
     * <p>
     * 使用给定的Integer包装类数组创建向量实例，自动转换为Double类型。此方法委托给
     * {@link IDoubleVector#of(Integer[])} 实现。 Creates a vector instance with
     * the given Integer wrapper array, automatically converted to Double type.
     * This method delegates to {@link IDoubleVector#of(Integer[])}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(data)} 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(data)} instead of
     * calling this method directly.
     * </p>
     *
     * @param data Integer包装类数组，表示向量数据 / Integer wrapper array representing
     * vector data
     * @return 新的向量实例（IDoubleVector）/ New vector instance (IDoubleVector)
     * @throws IllegalArgumentException 如果数据为null / if data is null
     * @see Linalg#vector(Integer[]) 推荐的工厂方法 / Recommended factory method
     * @see IDoubleVector#of(Integer[]) 实际实现方法 / Actual implementation method
     */
    public static <T extends Number> IVector<T> of(Integer[] data) {
        return (IVector<T>) IDoubleVector.of(data);
    }

    /**
     * 向量工厂方法（单个Double值） / Vector factory method (single Double value)
     * <p>
     * 创建包含单个 double 值的向量。此方法委托给 {@link IDoubleVector#of(double)} 实现。 Creates a
     * vector containing a single double value. This method delegates to
     * {@link IDoubleVector#of(double)}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(value)} 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(value)} instead of
     * calling this method directly.
     * </p>
     *
     * @param value 单个Double值，表示向量数据 / Single Double value representing vector
     * data
     * @return 包含单个值的向量（IDoubleVector）/ Vector containing single value
     * (IDoubleVector)
     * @throws IllegalArgumentException 如果值为null / if value is null
     * @see Linalg#vector(double) 推荐的工厂方法 / Recommended factory method
     * @see IDoubleVector#of(double) 实际实现方法 / Actual implementation method
     */
    public static IVector<Double> of(double value) {
        return (IVector<Double>) IDoubleVector.of(value);
    }

    /**
     * 向量工厂方法（两个Double值） / Vector factory method (two Double values)
     * <p>
     * 创建包含两个 double 值的向量。此方法委托给 {@link IDoubleVector#of(double, double)} 实现。
     * Creates a vector containing two double values. This method delegates to
     * {@link IDoubleVector#of(double, double)}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(value1, value2)}
     * 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(value1, value2)}
     * instead of calling this method directly.
     * </p>
     *
     * @param value1 第一个Double值 / First Double value
     * @param value2 第二个Double值 / Second Double value
     * @return 包含两个值的向量（IDoubleVector）/ Vector containing two values
     * (IDoubleVector)
     * @throws IllegalArgumentException 如果任一值为null / if any value is null
     * @see Linalg#vector(double, double) 推荐的工厂方法 / Recommended factory method
     * @see IDoubleVector#of(double, double) 实际实现方法 / Actual implementation
     * method
     */
    public static IVector<Double> of(double value1, double value2) {
        return (IVector<Double>) IDoubleVector.of(value1, value2);
    }

    /**
     * 向量工厂方法（单个Float值） / Vector factory method (single Float value)
     * <p>
     * 创建包含单个 float 值的向量。此方法委托给 {@link IFloatVector#of(float)} 实现。 Creates a
     * vector containing a single float value. This method delegates to
     * {@link IFloatVector#of(float)}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(value)} 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(value)} instead of
     * calling this method directly.
     * </p>
     *
     * @param value 单个Float值，表示向量数据 / Single Float value representing vector
     * data
     * @return 包含单个值的向量（IFloatVector）/ Vector containing single value
     * (IFloatVector)
     * @throws IllegalArgumentException 如果值为null / if value is null
     * @see Linalg#vector(float) 推荐的工厂方法 / Recommended factory method
     * @see IFloatVector#of(float) 实际实现方法 / Actual implementation method
     */
    public static IVector<Float> of(float value) {
        return (IVector<Float>) IFloatVector.of(value);
    }

    /**
     * 向量工厂方法（两个Float值） / Vector factory method (two Float values)
     * <p>
     * 创建包含两个 float 值的向量。此方法委托给 {@link IFloatVector#of(float, float)} 实现。 Creates
     * a vector containing two float values. This method delegates to
     * {@link IFloatVector#of(float, float)}.
     * </p>
     * <p>
     * <strong>推荐使用：</strong>{@code Linalg.vector(value1, value2)}
     * 替代直接调用此方法。<br>
     * <strong>Recommended:</strong> Use {@code Linalg.vector(value1, value2)}
     * instead of calling this method directly.
     * </p>
     *
     * @param value1 第一个Float值 / First Float value
     * @param value2 第二个Float值 / Second Float value
     * @return 包含两个值的向量（IFloatVector）/ Vector containing two values
     * (IFloatVector)
     * @throws IllegalArgumentException 如果任一值为null / if any value is null
     * @see Linalg#vector(float, float) 推荐的工厂方法 / Recommended factory method
     * @see IFloatVector#of(float, float) 实际实现方法 / Actual implementation method
     */
    public static IVector<Float> of(float value1, float value2) {
        return (IVector<Float>) IFloatVector.of(value1, value2);
    }

    // ========== 范围向量创建 / Range Vector Creation ==========
    /**
     * 创建指定类型的范围向量（带步长） / Create range vector of specified type (with step)
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @param step 步长 / Step size
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 范围向量 / Range vector
     * @throws IllegalArgumentException 如果step为0或负数 / if step is 0 or negative
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> range(int start, int end, int step, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.range(start, end, step);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.range(start, end, step);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建范围向量（带步长，默认Double类型） / Create range vector (with step, default Double
     * type)
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
    public static IVector<Double> range(int start, int end, int step) {
        return (IVector<Double>) IDoubleVector.range(start, end, step);
    }

    /**
     * 创建指定类型的范围向量（步长为1） / Create range vector of specified type (step size 1)
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 范围向量 / Range vector
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> range(int start, int end, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.range(start, end);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.range(start, end);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建范围向量（步长为1，默认Double类型） / Create range vector (step size 1, default
     * Double type)
     * <p>
     * 创建一个从start到end（不包含end）、步长为1的向量 Creates a vector from start to end
     * (exclusive) with step size 1
     * </p>
     *
     * @param start 起始值 / Start value
     * @param end 结束值（不包含） / End value (exclusive)
     * @return 范围向量 / Range vector
     */
    public static IVector<Double> range(int start, int end) {
        return (IVector<Double>) IDoubleVector.range(start, end);
    }

    /**
     * 创建指定类型的范围向量（从0开始） / Create range vector of specified type (starting from
     * 0)
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param end 结束值（不包含） / End value (exclusive)
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 范围向量 / Range vector
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> range(int end, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.range(end);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.range(end);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建范围向量（从0开始，默认Double类型） / Create range vector (starting from 0, default
     * Double type)
     * <p>
     * 创建一个从0到end（不包含end）、步长为1的向量 Creates a vector from 0 to end (exclusive)
     * with step size 1
     * </p>
     *
     * @param end 结束值（不包含） / End value (exclusive)
     * @return 范围向量 / Range vector
     */
    public static IVector<Double> range(int end) {
        return (IVector<Double>) IDoubleVector.range(end);
    }

    // ========== 特殊向量创建 / Special Vector Creation ==========
    /**
     * 创建指定类型的全1向量 / Create ones vector of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param len 向量长度 / Vector length
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 全1向量 / Ones vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> ones(int len, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.ones(len);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.ones(len);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建全1向量（默认Double类型） / Create ones vector (default Double type)
     * <p>
     * 创建一个指定长度的向量，所有元素都为1 Creates a vector of specified length with all
     * elements set to 1
     * </p>
     *
     * @param len 向量长度 / Vector length
     * @return 全1向量 / Ones vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    static IVector<Double> ones(int len) {
        return (IVector<Double>) IDoubleVector.ones(len);
    }

    /**
     * 创建指定类型的零向量 / Create zeros vector of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param len 向量长度 / Vector length
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 零向量 / Zeros vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> zeros(int len, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.zeros(len);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.zeros(len);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建零向量（默认Double类型） / Create zeros vector (default Double type)
     * <p>
     * 创建一个指定长度的向量，所有元素都为0 Creates a vector of specified length with all
     * elements set to 0
     * </p>
     *
     * @param len 向量长度 / Vector length
     * @return 零向量 / Zeros vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    static IVector<Double> zeros(int len) {
        return (IVector<Double>) IDoubleVector.zeros(len);
    }

    // ========== 随机数生成 / Random Number Generation ==========
    /**
     * 创建指定类型的随机向量 / Create random vector of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param length 向量长度 / Vector length
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 随机向量 / Random vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> rand(int length, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.rand(length);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.rand(length);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建随机向量（默认Double类型） / Create random vector (default Double type)
     * <p>
     * 创建指定长度的随机向量，元素值在[0,1)范围内 Creates a random vector of specified length with
     * elements in [0,1) range
     * </p>
     *
     * @param length 向量长度 / Vector length
     * @return 随机向量 / Random vector
     * @throws IllegalArgumentException 如果长度小于等于0 / if length is less than or
     * equal to 0
     */
    static IVector<Double> rand(int length) {
        return (IVector<Double>) IDoubleVector.rand(length);
    }

    /**
     * 创建指定类型的正态分布随机向量 / Create normal distribution random vector of specified
     * type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param length 向量长度 / Vector length
     * @param mean 均值 / Mean
     * @param std 标准差 / Standard deviation
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 正态分布随机向量 / Normal distribution random vector
     * @throws IllegalArgumentException 如果长度小于等于0或标准差小于等于0 / if length is less
     * than or equal to 0 or std is less than or equal to 0
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> randn(int length, T mean, T std, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.randn(length, (Float) mean, (Float) std);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.randn(length, (Double) mean, (Double) std);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建正态分布随机向量（默认Double类型） / Create normal distribution random vector
     * (default Double type)
     * <p>
     * 创建指定长度的正态分布随机向量 Creates a normal distribution random vector of specified
     * length
     * </p>
     *
     * @param length 向量长度 / Vector length
     * @param mean 均值 / Mean
     * @param std 标准差 / Standard deviation
     * @return 正态分布随机向量 / Normal distribution random vector
     * @throws IllegalArgumentException 如果长度小于等于0或标准差小于等于0 / if length is less
     * than or equal to 0 or std is less than or equal to 0
     */
    static IVector<Double> randn(int length, double mean, double std) {
        return (IVector<Double>) IDoubleVector.randn(length, mean, std);
    }
    
    static <T extends Number> IVector<T> randn(int length, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.randn(length);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.randn(length);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }
    
    static IVector<Double> randn(int length) {
        return (IVector<Double>) IDoubleVector.randn(length);
    }

    // ========== 线性空间生成 / Linear Space Generation ==========
    /**
     * 创建指定类型的线性空间向量 / Create linear space vector of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param start 起始值 / Start value
     * @param stop 结束值 / Stop value
     * @param num 元素数量 / Number of elements
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 线性空间向量 / Linear space vector
     * @throws IllegalArgumentException 如果元素数量小于2 / if number of elements is
     * less than 2
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> linspace(T start, T stop, int num, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.linspace((Float) start, (Float) stop, num);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.linspace((Double) start, (Double) stop, num);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建线性空间向量（默认Double类型） / Create linear space vector (default Double type)
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
    static IVector<Double> linspace(double start, double stop, int num) {
        return (IVector<Double>) IDoubleVector.linspace(start, stop, num);
    }

    /**
     * 创建指定类型的对数空间向量 / Create logarithmic space vector of specified type
     * <p>
     * 此方法提供泛型支持，具体实现由子接口提供 This method provides generic support, concrete
     * implementation provided by sub-interfaces
     * </p>
     *
     * @param start 起始值（type^start） / Start value (type^start)
     * @param stop 结束值（type^stop） / Stop value (type^stop)
     * @param num 元素数量 / Number of elements
     * @param type 数值类型的类对象 / Class object of the numeric type
     * @param <T> 数值类型 / Numeric type
     * @return 对数空间向量 / Logarithmic space vector
     * @throws IllegalArgumentException 如果元素数量小于2 / if number of elements is
     * less than 2
     */
    @SuppressWarnings("unchecked")
    static <T extends Number> IVector<T> logspace(T start, T stop, int num, Class<T> type) {
        if (type == Float.class) {
            return (IVector<T>) IFloatVector.logspace((Float) start, (Float) stop, num);
        } else if (type == Double.class) {
            return (IVector<T>) IDoubleVector.logspace((Double) start, (Double) stop, num);
        } else {
            throw new UnsupportedOperationException("不支持的数值类型: " + type.getSimpleName() + " / Unsupported numeric type: " + type.getSimpleName());
        }
    }

    /**
     * 创建对数空间向量（默认Double类型） / Create logarithmic space vector (default Double
     * type)
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
    static IVector<Double> logspace(double start, double stop, int num) {
        return (IVector<Double>) IDoubleVector.logspace(start, stop, num);
    }

    // ========== 一维样本工具（实现见 {@link RereDoubleVector}；float 经 double 再转回）==========
    /**
     * 一维等宽直方图结果（计数与分箱边界；边界为 double[]）。
     */
    final class HistogramResult {
        
        public final long[] counts;
        public final double[] binEdges;
        
        public HistogramResult(long[] counts, double[] binEdges) {
            this.counts = counts;
            this.binEdges = binEdges;
        }
    }
    
    static HistogramResult histogram(double[] x, int bins) {
        return RereDoubleVector.histogram(x, bins);
    }
    
    static HistogramResult histogram(float[] x, int bins) {
        return RereDoubleVector.histogram(RereMathUtil.floatToDouble(x), bins);
    }

    /**
     * 样本来自向量（与 {@link IDoubleVector}/{@link IFloatVector} 兼容；避免与 float/double
     * 重载产生相同擦除疑符而合并为一）。
     */
    static HistogramResult histogram(IVector<? extends Number> x, int bins) {
        Objects.requireNonNull(x, "x");
        if (x instanceof IDoubleVector) {
            return RereDoubleVector.histogram((IVector<Double>) x, bins);
        }
        if (x instanceof IFloatVector) {
            return RereDoubleVector.histogram(RereMathUtil.floatToDouble(((IFloatVector) x).getData()), bins);
        }
        throw new IllegalArgumentException("仅支持 Double 或 Float 元素向量 / Only Double or Float element vectors");
    }
    
    static int[] digitize(double[] x, double[] bins) {
        return RereDoubleVector.digitize(x, bins);
    }
    
    static int[] digitize(float[] x, float[] bins) {
        return RereDoubleVector.digitize(RereMathUtil.floatToDouble(x), RereMathUtil.floatToDouble(bins));
    }
    
    static int[] digitize(IVector<? extends Number> x, double[] bins) {
        Objects.requireNonNull(x, "x");
        if (x instanceof IDoubleVector) {
            return RereDoubleVector.digitize((IVector<Double>) x, bins);
        }
        if (x instanceof IFloatVector) {
            return RereDoubleVector.digitize(RereMathUtil.floatToDouble(((IFloatVector) x).getData()), bins);
        }
        throw new IllegalArgumentException("仅支持 Double 或 Float 元素向量 / Only Double or Float element vectors");
    }
    
    static double[] polyfit(double[] x, double[] y, int deg) {
        return RereDoubleVector.polyfit(x, y, deg);
    }
    
    static float[] polyfit(float[] x, float[] y, int deg) {
        return RereMathUtil.doubleToFloat(RereDoubleVector.polyfit(
                RereMathUtil.floatToDouble(x), RereMathUtil.floatToDouble(y), deg));
    }
    
    @SuppressWarnings("unchecked")
    static IVector<? extends Number> polyfit(IVector<? extends Number> x, IVector<? extends Number> y, int deg) {
        Objects.requireNonNull(x, "x");
        Objects.requireNonNull(y, "y");
        if (x instanceof IDoubleVector && y instanceof IDoubleVector) {
            return RereDoubleVector.polyfit((IVector<Double>) x, (IVector<Double>) y, deg);
        }
        if (x instanceof IFloatVector && y instanceof IFloatVector) {
            return IFloatVector.of(IVector.polyfit(
                    ((IFloatVector) x).getData(), ((IFloatVector) y).getData(), deg));
        }
        throw new IllegalArgumentException("x 与 y 须同为 Double 或同为 Float 向量 / x and y must both be Double or both be Float");
    }
    
    static double[] where(boolean[] cond, double x, double y) {
        return RereDoubleVector.where(cond, x, y);
    }
    
    static float[] where(boolean[] cond, float x, float y) {
        return RereMathUtil.doubleToFloat(RereDoubleVector.where(cond, (double) x, (double) y));
    }
    
    static double[] where(boolean[] cond, double[] x, double[] y) {
        return RereDoubleVector.where(cond, x, y);
    }
    
    static float[] where(boolean[] cond, float[] x, float[] y) {
        return RereMathUtil.doubleToFloat(RereDoubleVector.where(
                cond, RereMathUtil.floatToDouble(x), RereMathUtil.floatToDouble(y)));
    }
    
    static <T extends Number> IVector<T> where(IVector<T> v, boolean[] cond, T x, T y) {
        Objects.requireNonNull(v, "v");
        return v.where(cond, x, y);
    }

    // ========== 向量实例方法 / Vector Instance Methods ==========
    // 工厂方法将在具体实现类中定义
    // ========== 抽象方法定义 / Abstract Method Definitions ==========
    /**
     * 获取向量长度 / Get vector length
     * <p>
     * 返回向量的长度（元素个数） Returns the length (number of elements) of the vector
     * </p>
     *
     * @return 向量长度 / Vector length
     */
    public int length();

    /**
     * 获取指定位置的元素 / Get element at specified position
     * <p>
     * 返回向量中指定位置的元素值 Returns the element value at the specified position in the
     * vector
     * </p>
     *
     * @param position 位置索引（从0开始） / Position index (0-based)
     * @return 指定位置的元素值 / Element value at the specified position
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position index is out
     * of bounds
     */
    public double get(int position);

    /**
     * 设置指定位置的元素 / Set element at specified position
     * <p>
     * 设置向量中指定位置的元素值 Sets the element value at the specified position in the
     * vector
     * </p>
     *
     * @param position 位置索引（从0开始） / Position index (0-based)
     * @param value 要设置的值 / Value to set
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position index is out
     * of bounds
     */
    public void set(int position, double value);

    /**
     * 向量加法运算 / Vector addition
     * <p>
     * 对应元素相加，要求两个向量长度相同 Element-wise addition, requires both vectors to have
     * the same length
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> v1 = Linalg.vector(new double[]{1, 2, 3});
     * IVector<Double> v2 = Linalg.vector(new double[]{4, 5, 6});
     * IVector<Double> result = v1.add(v2);  // 结果: [5, 7, 9]
     * }</pre>
     * </p>
     *
     * @param vec 另一个向量 / The other vector
     * @return 新的向量对象，包含加法结果 / New vector object containing addition result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     * @throws NullPointerException 如果vec为null / if vec is null
     */
    public IVector<T> add(IVector<T> vec);

    /**
     * 向量减法运算 / Vector subtraction
     * <p>
     * 对应元素相减，要求两个向量长度相同 Element-wise subtraction, requires both vectors to have
     * the same length
     * </p>
     *
     * @param vec 另一个向量 / The other vector
     * @return 新的向量对象，包含减法结果 / New vector object containing subtraction result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public IVector<T> sub(IVector<T> vec);

    /**
     * 向量逐元素乘法运算（Hadamard乘积） / Vector element-wise multiplication (Hadamard
     * product)
     * <p>
     * 对应元素相乘，要求两个向量长度相同。这是逐元素乘法，不是内积运算。 Element-wise multiplication, requires
     * both vectors to have the same length. This is element-wise
     * multiplication, not inner product.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> v1 = Linalg.vector(new double[]{1, 2, 3});
     * IVector<Double> v2 = Linalg.vector(new double[]{4, 5, 6});
     * IVector<Double> result = v1.multiply(v2);  // 结果: [4, 10, 18]
     * }</pre>
     * </p>
     * <p>
     * <strong>注意：</strong>此方法执行逐元素乘法，如需计算内积请使用 {@link #dot(IVector)} 或
     * {@link #innerProduct(IVector)}。
     * <br><strong>Note:</strong> This method performs element-wise
     * multiplication. For inner product, use {@link #dot(IVector)} or
     * {@link #innerProduct(IVector)}.
     * </p>
     *
     * @param vec 另一个向量 / The other vector
     * @return 新的向量对象，包含逐元素乘法结果 / New vector object containing element-wise
     * multiplication result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     * @throws NullPointerException 如果vec为null / if vec is null
     * @see #dot(IVector) 内积运算 / Inner product operation
     * @see #innerProduct(IVector) 内积运算（别名）/ Inner product operation (alias)
     */
    public IVector<T> multiply(IVector<T> vec);

    /**
     * 向量与矩阵乘法 / Vector-matrix multiplication
     * <p>
     * 计算行向量与矩阵的乘积，结果仍为行向量。要求向量的长度等于矩阵的行数。 Computes the product of a row vector
     * with a matrix, result is still a row vector. Requires vector length to
     * equal matrix row count.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector = Linalg.vector(new double[]{1, 2});
     * IMatrix<Double> matrix = Linalg.matrix(new double[][]{{3, 4}, {5, 6}});
     * IVector<Double> result = vector.mmul(matrix);  // 结果: [13, 16]
     * }</pre>
     * </p>
     *
     * @param other 要相乘的矩阵 / Matrix to multiply with
     * @return 向量与矩阵的乘积结果 / Vector-matrix multiplication result
     * @throws IllegalArgumentException 如果向量长度与矩阵行数不匹配 / if vector length
     * doesn't match matrix row count
     * @throws NullPointerException 如果other为null / if other is null
     */
    public IVector<T> mmul(IMatrix<T> other);

    /**
     * 向量内积运算 / Vector inner product
     * <p>
     * 计算两个向量的内积（点积），要求两个向量长度相同 Computes the inner product (dot product) of two
     * vectors, requires both vectors to have the same length
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> v1 = Linalg.vector(new double[]{1, 2, 3});
     * IVector<Double> v2 = Linalg.vector(new double[]{4, 5, 6});
     * Double dotProduct = v1.innerProduct(v2);  // 结果: 1*4 + 2*5 + 3*6 = 32
     * }</pre>
     * </p>
     *
     * @param vec 另一个向量 / The other vector
     * @return 内积结果 / Inner product result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     * @throws NullPointerException 如果vec为null / if vec is null
     */
    public IVector<T> innerProduct(IVector<T> vec);

    default double innerProductValue(IVector<T> vec) { return innerProduct(vec).get(0); }

    /**
     * 向量内积运算 / Vector inner product innerProduct的别名（适配NumPy名称）
     *
     * @param vec 另一个向量 / The other vector
     * @return 内积结果 / Inner product result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public IVector<T> dot(IVector<T> vec);

    default double dotValue(IVector<T> vec) { return dot(vec).get(0); }

    /**
     * 逐维缩放的平方欧氏距离：∑_k (perDimScale_k · (this_k − other_k))²。
     * <p>
     * 等价于 ‖diag(perDimScale)(this − other)‖²；适用于对角度量（特征逐维权重）下两点距离。
     * </p>
     *
     * @param other 另一点 / Other point
     * @param perDimScale 与 {@code this} 等长的逐维系数（常为对角度量缩放）/ Per-dimension
     * coefficients
     * @return 标量距离平方 / Squared distance
     * @apiNote
     * <strong>DML：</strong>{@code com.yishape.lab.math.ml.dml.common.MetricTransforms#squaredDiagonal}
     * 委托本方法。
     */
    default IVector<T> diagonalWeightedSquaredDistanceTo(IVector<T> other, IVector<T> perDimScale) {
        Objects.requireNonNull(other, "other");
        Objects.requireNonNull(perDimScale, "perDimScale");
        if (length() != other.length() || length() != perDimScale.length()) {
            throw new IllegalArgumentException("向量长度须一致 / vector lengths must match");
        }
        IVector<T> d = sub(other);
        IVector<T> w = d.multiply(perDimScale);
        return w.innerProduct(w);
    }

    default double diagonalWeightedSquaredDistanceToValue(IVector<T> other, IVector<T> perDimScale) {
        return diagonalWeightedSquaredDistanceTo(other, perDimScale).get(0);
    }

    /**
     * 向量加标量 / Vector plus scalar
     * <p>
     * 向量中每个元素加上标量值 Adds a scalar value to each element in the vector
     * </p>
     *
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含加法结果 / New vector object containing addition result
     */
    public IVector<T> addScalar(double p);

    /**
     * 向量减标量 / Vector sub scalar
     * <p>
     * 向量中每个元素减去标量值 Subtracts a scalar value from each element in the vector
     * </p>
     *
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含减法结果 / New vector object containing subtraction result
     */
    public IVector<T> subScalar(double p);

    /**
     * 向量乘标量 / Vector multiply scalar
     * <p>
     * 向量中每个元素乘以标量值 Multiplies each element in the vector by a scalar value
     * </p>
     *
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含乘法结果 / New vector object containing multiplication
     * result
     */
    public IVector<T> multiplyByScalar(double p);

    /**
     * 向量乘标量 / Vector multiply scalar
     * <p>
     * 向量中每个元素乘以标量值 Multiplies each element in the vector by a scalar value
     * </p>
     *
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含乘法结果 / New vector object containing multiplication
     * result
     */
    public default IVector<T> scale(double p) {
        return this.multiplyByScalar(p);
    }

    /**
     * 向量除标量 / Vector divide by scalar
     * <p>
     * 向量中每个元素除以标量值 Divides each element in the vector by a scalar value
     * </p>
     *
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含除法结果 / New vector object containing division result
     * @throws ArithmeticException 如果标量为零 / if scalar is zero
     */
    public IVector<T> divideByScalar(double p);

    /**
     * 原地 AXPY 运算 / In-place AXPY operation
     * <p>
     * 计算 this = this + alpha * x，修改当前向量并返回自身。
     * Computes this = this + alpha * x, modifies this vector in-place and returns itself.
     * 当 alpha = -1 时等价于 subAxpy。
     * </p>
     *
     * @param alpha 标量乘数 / Scalar multiplier
     * @param x 另一个向量 / Another vector
     * @return 当前向量自身 / This vector (for chaining)
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths mismatch
     */
    default IVector<T> axpy(double alpha, IVector<T> x) {
        if (this.size() != x.size()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.size() + " vs " + x.size());
        }
        for (int i = 0; i < size(); i++) {
            set(i, get(i) + alpha * x.get(i));
        }
        return this;
    }

    /**
     * 原地标量除法 / In-place scalar division
     * <p>
     * 计算 this = this / alpha，修改当前向量并返回自身。
     * Computes this = this / alpha, modifies this vector in-place and returns itself.
     * </p>
     *
     * @param alpha 除数 / Divisor
     * @return 当前向量自身 / This vector (for chaining)
     * @throws ArithmeticException 如果 alpha 为零 / if alpha is zero
     */
    default IVector<T> divideInPlace(double alpha) {
        if (alpha == 0.0) {
            throw new ArithmeticException("除以零 / Division by zero");
        }
        for (int i = 0; i < size(); i++) {
            set(i, get(i) / alpha);
        }
        return this;
    }

    /**
     * 向量加标量（就地） / Vector add scalar (in-place)
     * <p>
     * 计算 this = this + p，修改当前向量并返回自身。
     * Computes this = this + p, modifies this vector in-place and returns itself.
     * </p>
     *
     * @param p 标量值 / Scalar value
     * @return 当前向量自身 / This vector (for chaining)
     */
    default IVector<T> addScalarInPlace(double p) {
        for (int i = 0; i < size(); i++) {
            set(i, get(i) + p);
        }
        return this;
    }

    /**
     * In-place scalar subtraction. Subtracts scalar p from every element.
     * 原地标量减法，每个元素减去标量 p。
     *
     * @param p scalar value to subtract
     * @return this vector (for chaining)
     */
    default IVector<T> subScalarInPlace(double p) {
        for (int i = 0; i < size(); i++) {
            set(i, get(i) - p);
        }
        return this;
    }

    /**
     * In-place scalar multiplication. Multiplies every element by scalar p.
     * 原地标量乘法，每个元素乘以标量 p。
     *
     * @param p scalar multiplier
     * @return this vector (for chaining)
     */
    default IVector<T> multiplyByScalarInPlace(double p) {
        for (int i = 0; i < size(); i++) {
            set(i, get(i) * p);
        }
        return this;
    }

    /**
     * In-place element-wise addition. Adds vec to this vector element-wise.
     * 原地逐元素加法，要求两个向量长度相同。
     *
     * @param vec vector to add (must have same length)
     * @return this vector (for chaining)
     */
    default IVector<T> addInPlace(IVector<T> vec) {
        for (int i = 0; i < size(); i++) {
            set(i, get(i) + vec.get(i));
        }
        return this;
    }

    /**
     * In-place element-wise subtraction. Subtracts vec from this vector element-wise.
     * 原地逐元素减法，要求两个向量长度相同。
     *
     * @param vec vector to subtract (must have same length)
     * @return this vector (for chaining)
     */
    default IVector<T> subInPlace(IVector<T> vec) {
        for (int i = 0; i < size(); i++) {
            set(i, get(i) - vec.get(i));
        }
        return this;
    }

    /**
     * In-place element-wise multiplication (Hadamard product).
     * 原地逐元素乘法（Hadamard 积），要求两个向量长度相同。
     *
     * @param vec vector to multiply element-wise (must have same length)
     * @return this vector (for chaining)
     */
    default IVector<T> multiplyInPlace(IVector<T> vec) {
        for (int i = 0; i < size(); i++) {
            set(i, get(i) * vec.get(i));
        }
        return this;
    }

    /**
     * In-place negation. Negates every element.
     * 原地取反，每个元素变为其相反数。
     *
     * @return this vector (for chaining)
     */
    default IVector<T> negInPlace() {
        for (int i = 0; i < size(); i++) {
            set(i, -get(i));
        }
        return this;
    }

    /**
     * 向量逐元素除法运算 / Vector element-wise division
     * <p>
     * 对应元素相除，要求两个向量长度相同。此方法通过计算除数向量的倒数然后进行逐元素乘法来实现。 Element-wise division,
     * requires both vectors to have the same length. This method is implemented
     * by computing the reciprocal of the divisor vector and then performing
     * element-wise multiplication.
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> v1 = Linalg.vector(new double[]{8, 12, 15});
     * IVector<Double> v2 = Linalg.vector(new double[]{2, 3, 5});
     * IVector<Double> result = v1.divide(v2);  // 结果: [4, 4, 3]
     * }</pre>
     * </p>
     * <p>
     * <strong>注意：</strong>此方法执行逐元素除法，如果除数向量中包含零值，将产生无穷大或NaN结果。
     * <br><strong>Note:</strong> This method performs element-wise division. If
     * the divisor vector contains zero values, it will produce infinity or NaN
     * results.
     * </p>
     *
     * @param other 除数向量 / Divisor vector
     * @return 新的向量对象，包含逐元素除法结果 / New vector object containing element-wise
     * division result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     * @throws NullPointerException 如果other为null / if other is null
     * @see #multiply(IVector) 逐元素乘法 / Element-wise multiplication
     * @see #reciprocal() 倒数运算 / Reciprocal operation
     */
    public default IVector<T> divide(IVector<T> other) {
        var oo = other.reciprocal();
        return this.multiply(oo);
    }

    /**
     * 向量元素求和 / Vector sum
     * <p>
     * 计算向量中所有元素的和 Calculates the sum of all elements in the vector
     * </p>
     *
     * @return 元素和 / Sum of elements
     */
    public IVector<T> sum();

    default double sumValue() { return sum().get(0); }

    /**
     * 向量最小值 / Vector minimum
     * <p>
     * 找到向量中的最小元素值 Finds the minimum element value in the vector
     * </p>
     *
     * @return 最小值 / Minimum value
     */
    public IVector<T> min();

    default double minValue() { return min().get(0); }

    /**
     * 向量最大值 / Vector maximum
     * <p>
     * 找到向量中的最大元素值 Finds the maximum element value in the vector
     * </p>
     *
     * @return 最大值 / Maximum value
     */
    public IVector<T> max();

    default double maxValue() { return max().get(0); }

    /**
     * 向量均值 / Vector mean
     * <p>
     * 计算向量中所有元素的平均值 Calculates the mean of all elements in the vector
     * </p>
     *
     * @return 平均值 / Mean value
     */
    public IVector<T> mean();

    default double meanValue() { return mean().get(0); }

    /**
     * 向量复制 / Vector copy
     * <p>
     * 创建向量的深拷贝 Creates a deep copy of the vector
     * </p>
     *
     * @return 新的向量对象，与原向量内容相同 / New vector object with the same content as the
     * original
     */
    public IVector<T> copy();

    /**
     * 向量切片（指定开始和结束位置） / Vector slice (specified start and end positions)
     * <p>
     * 返回从指定开始位置到结束位置的向量切片 Returns a vector slice from specified start position
     * to end position
     * </p>
     *
     * @param start 开始位置 / Start position
     * @param end 结束位置（不包含） / End position (exclusive)
     * @return 切片向量 / Sliced vector
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are
     * out of bounds
     */
    public IVector<T> slice(int start, int end);

    /**
     * 向量切片（从指定位置到末尾） / Vector slice (from specified position to end)
     * <p>
     * 返回从指定位置到向量末尾的切片 Returns a vector slice from the specified position to the
     * end of the vector
     * </p>
     *
     * @param start 起始位置 / Start position
     * @return 切片向量 / Sliced vector
     * @throws IndexOutOfBoundsException 如果起始位置超出范围 / if start position is out of
     * bounds
     */
    public IVector<T> slice(int start);

    /**
     * 向量2范数 / Vector L2 norm
     * <p>
     * 计算向量的2范数（欧几里得范数） Calculates the L2 norm (Euclidean norm) of the vector
     * </p>
     *
     * @return 2范数 / L2 norm
     */
    public IVector<T> norm2();

    default double norm2Value() { return norm2().get(0); }

    /**
     * 向量1范数 / Vector L1 norm
     * <p>
     * 计算向量的1范数（曼哈顿范数） Calculates the L1 norm (Manhattan norm) of the vector
     * </p>
     *
     * @return 1范数 / L1 norm
     */
    public IVector<T> norm1();

    default double norm1Value() { return norm1().get(0); }

    // ========== 扩展的统计操作 / Extended Statistical Operations ==========
    /**
     * 向量标准差 / Vector standard deviation
     * <p>
     * 计算向量中所有元素的标准差，使用样本标准差公式（分母为n-1） Calculates the standard deviation of all
     * elements in the vector using sample standard deviation formula
     * (denominator is n-1)
     * </p>
     * <p>
     * 公式：std = √(Σ(xi - μ)² / (n-1)) Formula: std = √(Σ(xi - μ)² / (n-1))
     * </p>
     *
     * @return 标准差 / Standard deviation
     * @throws ArithmeticException 如果向量长度小于2 / if vector length is less than 2
     */
    public IVector<T> std();

    default double stdValue() { return std().get(0); }

    /**
     * 向量标准差（自由度修正） / Vector standard deviation (degrees of freedom correction)
     * <p>
     * 计算向量中所有元素的标准差，使用指定的自由度修正 Calculates the standard deviation of all
     * elements in the vector using specified degrees of freedom correction
     * </p>
     * <p>
     * 公式：std = √(Σ(xi - μ)² / (n-ddof)) Formula: std = √(Σ(xi - μ)² / (n-ddof))
     * </p>
     *
     * @param ddof 自由度修正值，通常为0（总体标准差）或1（样本标准差）/ Degrees of freedom correction,
     * usually 0 (population std) or 1 (sample std)
     * @return 标准差 / Standard deviation
     * @throws ArithmeticException 如果向量长度小于等于ddof / if vector length is less
     * than or equal to ddof
     * @throws IllegalArgumentException 如果ddof为负数 / if ddof is negative
     */
    public IVector<T> std(int ddof);

    default double stdValue(int ddof) { return std(ddof).get(0); }

    /**
     * 向量方差 / Vector variance
     * <p>
     * 计算向量中所有元素的方差，使用样本方差公式（分母为n-1） Calculates the variance of all elements in
     * the vector using sample variance formula (denominator is n-1)
     * </p>
     * <p>
     * 公式：var = Σ(xi - μ)² / (n-1) Formula: var = Σ(xi - μ)² / (n-1)
     * </p>
     *
     * @return 方差 / Variance
     * @throws ArithmeticException 如果向量长度小于2 / if vector length is less than 2
     */
    public IVector<T> var();

    default double varValue() { return var().get(0); }

    /**
     * 向量方差（自由度修正） / Vector variance (degrees of freedom correction)
     * <p>
     * 计算向量中所有元素的方差，使用指定的自由度修正 Calculates the variance of all elements in the
     * vector using specified degrees of freedom correction
     * </p>
     * <p>
     * 公式：var = Σ(xi - μ)² / (n-ddof) Formula: var = Σ(xi - μ)² / (n-ddof)
     * </p>
     *
     * @param ddof 自由度修正值，通常为0（总体方差）或1（样本方差）/ Degrees of freedom correction,
     * usually 0 (population var) or 1 (sample var)
     * @return 方差 / Variance
     * @throws ArithmeticException 如果向量长度小于等于ddof / if vector length is less
     * than or equal to ddof
     * @throws IllegalArgumentException 如果ddof为负数 / if ddof is negative
     */
    public IVector<T> var(int ddof);

    default double varValue(int ddof) { return var(ddof).get(0); }

    /**
     * 最小值索引 / Index of minimum value
     * <p>
     * 返回向量中最小元素的索引位置 Returns the index position of the minimum element in the
     * vector
     * </p>
     *
     * @return 最小元素的索引（从0开始）/ Index of minimum element (0-based)
     * @throws IllegalStateException 如果向量为空 / if vector is empty
     */
    public int argMin();

    /**
     * 最大值索引 / Index of maximum value
     * <p>
     * 返回向量中最大元素的索引位置 Returns the index position of the maximum element in the
     * vector
     * </p>
     *
     * @return 最大元素的索引（从0开始）/ Index of maximum element (0-based)
     * @throws IllegalStateException 如果向量为空 / if vector is empty
     */
    public int argMax();

    /**
     * 向量元素乘积 / Vector product
     * <p>
     * 计算向量中所有元素的乘积 Calculates the product of all elements in the vector
     * </p>
     * <p>
     * 公式：prod = x₁ × x₂ × ... × xₙ Formula: prod = x₁ × x₂ × ... × xₙ
     * </p>
     *
     * @return 所有元素的乘积 / Product of all elements
     * @throws IllegalStateException 如果向量为空 / if vector is empty
     */
    public IVector<T> prod();

    default double prodValue() { return prod().get(0); }

    // ========== 数学函数 / Mathematical Functions ==========
    /**
     * 向量绝对值 / Vector absolute value
     * <p>
     * 对向量中每个元素进行绝对值运算（|x|） Performs absolute value operation (|x|) on each
     * element in the vector
     * </p>
     *
     * @return 新的向量对象，包含绝对值运算结果 / New vector object containing absolute value
     * operation results
     */
    public IVector<T> abs();

    /**
     * 向量开方 / Vector square root
     * <p>
     * 对向量中每个元素进行平方根运算（√x） Performs square root operation (√x) on each element
     * in the vector
     * </p>
     *
     * @return 新的向量对象，包含平方根运算结果 / New vector object containing square root
     * operation results
     * @throws ArithmeticException 如果任何元素为负数 / if any element is negative
     */
    public IVector<T> sqrt();

    /**
     * 向量平方 / Vector square
     * <p>
     * 对向量中每个元素进行平方运算（x²） Performs square operation (x²) on each element in the
     * vector
     * </p>
     *
     * @return 新的向量对象，包含平方运算结果 / New vector object containing square operation
     * results
     */
    public IVector<T> square();

    /**
     * 向量指数运算 / Vector exponential
     * <p>
     * 对向量中每个元素进行指数运算（e^x） Performs exponential operation (e^x) on each element
     * in the vector
     * </p>
     *
     * @return 新的向量对象，包含指数运算结果 / New vector object containing exponential
     * operation results
     */
    public IVector<T> exp();

    /**
     * 向量自然对数 / Vector natural logarithm
     * <p>
     * 对向量中每个元素进行自然对数运算（ln(x)） Performs natural logarithm operation (ln(x)) on
     * each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含自然对数运算结果 / New vector object containing natural
     * logarithm operation results
     * @throws ArithmeticException 如果任何元素小于等于0 / if any element is less than or
     * equal to 0
     */
    public IVector<T> log();

    /**
     * 向量以10为底的对数 / Vector base-10 logarithm
     * <p>
     * 对向量中每个元素进行以10为底的对数运算（log₁₀(x)） Performs base-10 logarithm operation
     * (log₁₀(x)) on each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含以10为底的对数运算结果 / New vector object containing base-10
     * logarithm operation results
     * @throws ArithmeticException 如果任何元素小于等于0 / if any element is less than or
     * equal to 0
     */
    public IVector<T> log10();

    /**
     * 向量sigmoid激活函数 / Vector sigmoid activation function
     * <p>
     * 对向量中每个元素进行sigmoid函数运算（1/(1+e^(-x))）
     * Performs sigmoid function operation (1/(1+e^(-x))) on each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含sigmoid运算结果（double精度）/ New vector object containing
     * sigmoid operation results (double precision)
     */
    public IVector<T> sigmoid();

    /**
     * 向量ReLU激活函数 / Vector ReLU activation function
     * <p>
     * 对向量中每个元素进行ReLU函数运算（max(0, x)）
     * Performs ReLU function operation (max(0, x)) on each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含ReLU运算结果（double精度）/ New vector object containing ReLU
     * operation results (double precision)
     */
    public IVector<T> relu();

    /**
     * 向量幂运算 / Vector power operation
     * <p>
     * 对向量中每个元素进行幂运算（x^m） Performs power operation (x^m) on each element in the
     * vector
     * </p>
     *
     * @param m 幂指数 / Power exponent
     * @return 新的向量对象，包含幂运算结果 / New vector object containing power operation
     * results
     * @throws IllegalArgumentException 如果幂指数为null / if power exponent is null
     * @throws ArithmeticException 如果底数为负数且幂指数不是整数 / if base is negative and
     * exponent is not an integer
     */
    public IVector<T> pow(double m);

    // ========== 三角函数 / Trigonometric Functions ==========
    /**
     * 向量正弦函数 / Vector sine function
     * <p>
     * 对向量中每个元素进行正弦函数运算（sin(x)） Performs sine function operation (sin(x)) on
     * each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含正弦函数运算结果 / New vector object containing sine function
     * operation results
     */
    public IVector<T> sin();

    /**
     * 向量余弦函数 / Vector cosine function
     * <p>
     * 对向量中每个元素进行余弦函数运算（cos(x)） Performs cosine function operation (cos(x)) on
     * each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含余弦函数运算结果 / New vector object containing cosine function
     * operation results
     */
    public IVector<T> cos();

    /**
     * 向量正切函数 / Vector tangent function
     * <p>
     * 对向量中每个元素进行正切函数运算（tan(x)） Performs tangent function operation (tan(x)) on
     * each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含正切函数运算结果 / New vector object containing tangent function
     * operation results
     */
    public IVector<T> tan();

    /**
     * 向量反正弦函数 / Vector arcsine function
     * <p>
     * 对向量中每个元素进行反正弦函数运算（arcsin(x)） Performs arcsine function operation
     * (arcsin(x)) on each element in the vector
     * </p>
     * <p>
     * 输入值必须在[-1, 1]范围内，输出值在[-π/2, π/2]范围内 Input values must be in [-1, 1]
     * range, output values are in [-π/2, π/2] range
     * </p>
     *
     * @return 新的向量对象，包含反正弦函数运算结果 / New vector object containing arcsine
     * function operation results
     * @throws ArithmeticException 如果任何元素超出[-1, 1]范围 / if any element is outside
     * [-1, 1] range
     */
    public IVector<T> arcsin();

    /**
     * 向量反余弦函数 / Vector arccosine function
     * <p>
     * 对向量中每个元素进行反余弦函数运算（arccos(x)） Performs arccosine function operation
     * (arccos(x)) on each element in the vector
     * </p>
     * <p>
     * 输入值必须在[-1, 1]范围内，输出值在[0, π]范围内 Input values must be in [-1, 1] range,
     * output values are in [0, π] range
     * </p>
     *
     * @return 新的向量对象，包含反余弦函数运算结果 / New vector object containing arccosine
     * function operation results
     * @throws ArithmeticException 如果任何元素超出[-1, 1]范围 / if any element is outside
     * [-1, 1] range
     */
    public IVector<T> arccos();

    /**
     * 向量反正切函数 / Vector arctangent function
     * <p>
     * 对向量中每个元素进行反正切函数运算（arctan(x)） Performs arctangent function operation
     * (arctan(x)) on each element in the vector
     * </p>
     * <p>
     * 输出值在[-π/2, π/2]范围内 Output values are in [-π/2, π/2] range
     * </p>
     *
     * @return 新的向量对象，包含反正切函数运算结果 / New vector object containing arctangent
     * function operation results
     */
    public IVector<T> arctan();

    // ========== 双曲函数 / Hyperbolic Functions ==========
    /**
     * 向量双曲正弦函数 / Vector hyperbolic sine function
     * <p>
     * 对向量中每个元素进行双曲正弦函数运算（sinh(x)） Performs hyperbolic sine function operation
     * (sinh(x)) on each element in the vector
     * </p>
     * <p>
     * 公式：sinh(x) = (e^x - e^(-x)) / 2 Formula: sinh(x) = (e^x - e^(-x)) / 2
     * </p>
     *
     * @return 新的向量对象，包含双曲正弦函数运算结果 / New vector object containing hyperbolic
     * sine function operation results
     */
    public IVector<T> sinh();

    /**
     * 向量双曲余弦函数 / Vector hyperbolic cosine function
     * <p>
     * 对向量中每个元素进行双曲余弦函数运算（cosh(x)） Performs hyperbolic cosine function operation
     * (cosh(x)) on each element in the vector
     * </p>
     * <p>
     * 公式：cosh(x) = (e^x + e^(-x)) / 2 Formula: cosh(x) = (e^x + e^(-x)) / 2
     * </p>
     *
     * @return 新的向量对象，包含双曲余弦函数运算结果 / New vector object containing hyperbolic
     * cosine function operation results
     */
    public IVector<T> cosh();

    /**
     * 向量双曲正切函数 / Vector hyperbolic tangent function
     * <p>
     * 对向量中每个元素进行双曲正切函数运算（tanh(x)） Performs hyperbolic tangent function
     * operation (tanh(x)) on each element in the vector
     * </p>
     * <p>
     * 公式：tanh(x) = sinh(x) / cosh(x) Formula: tanh(x) = sinh(x) / cosh(x)
     * </p>
     * <p>
     * 输出值在(-1, 1)范围内 Output values are in (-1, 1) range
     * </p>
     *
     * @return 新的向量对象，包含双曲正切函数运算结果 / New vector object containing hyperbolic
     * tangent function operation results
     */
    public IVector<T> tanh();

    // ========== 舍入函数 / Rounding Functions ==========
    /**
     * 向量四舍五入 / Vector round function
     * <p>
     * 对向量中每个元素进行四舍五入运算 Performs rounding operation on each element in the
     * vector
     * </p>
     * <p>
     * 使用标准的四舍五入规则：0.5向上舍入 Uses standard rounding rules: 0.5 rounds up
     * </p>
     *
     * @return 新的向量对象，包含四舍五入运算结果 / New vector object containing rounded results
     */
    public IVector<T> round();

    /**
     * 向量向下取整 / Vector floor function
     * <p>
     * 对向量中每个元素进行向下取整运算（向负无穷方向取整） Performs floor operation on each element in
     * the vector (rounds towards negative infinity)
     * </p>
     * <p>
     * 例如：floor(3.7) = 3, floor(-2.3) = -3 Example: floor(3.7) = 3, floor(-2.3)
     * = -3
     * </p>
     *
     * @return 新的向量对象，包含向下取整运算结果 / New vector object containing floor operation
     * results
     */
    public IVector<T> floor();

    /**
     * 向量向上取整 / Vector ceiling function
     * <p>
     * 对向量中每个元素进行向上取整运算（向正无穷方向取整） Performs ceiling operation on each element in
     * the vector (rounds towards positive infinity)
     * </p>
     * <p>
     * 例如：ceil(3.2) = 4, ceil(-2.7) = -2 Example: ceil(3.2) = 4, ceil(-2.7) = -2
     * </p>
     *
     * @return 新的向量对象，包含向上取整运算结果 / New vector object containing ceiling
     * operation results
     */
    public IVector<T> ceil();

    /**
     * 向量截断取整 / Vector truncate function
     * <p>
     * 对向量中每个元素进行截断取整运算（向零方向取整） Performs truncate operation on each element in
     * the vector (rounds towards zero)
     * </p>
     * <p>
     * 例如：trunc(3.7) = 3, trunc(-2.7) = -2 Example: trunc(3.7) = 3, trunc(-2.7)
     * = -2
     * </p>
     *
     * @return 新的向量对象，包含截断取整运算结果 / New vector object containing truncate
     * operation results
     */
    public IVector<T> trunc();

    // ========== 高级索引和切片操作 / Advanced Indexing and Slicing ==========
    /**
     * 向量切片（指定开始、结束位置和步长） / Vector slice with step
     * <p>
     * 返回从指定开始位置到结束位置、按指定步长取样的向量切片，支持负步长（反向取样）
     * Returns a vector slice from specified start position to end position with
     * specified step size, supports negative step (reverse)
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector = Linalg.vector(new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
     * IVector<Double> slice = vector.slice(0, 10, 2);   // 结果: [0, 2, 4, 6, 8]
     * IVector<Double> reverse = vector.slice(9, -1, -1); // 结果: [9, 8, 7, 6, 5, 4, 3, 2, 1]
     * }</pre>
     * </p>
     *
     * @param start 开始位置 / Start position
     * @param end 结束位置（不包含） / End position (exclusive)
     * @param step 步长（正数正向，负数反向） / Step size (positive forward, negative reverse)
     * @return 切片向量 / Sliced vector
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are
     * out of bounds
     * @throws IllegalArgumentException 如果step为0 / if step is 0
     */
    public IVector<T> slice(int start, int end, int step);

    /**
     * 向量切片（字符串表达式） / Vector slice with string expression
     * <p>
     * 根据字符串表达式对向量进行切片操作，支持Python风格的切片语法 Performs vector slicing based on string
     * expression, supports Python-style slice syntax
     * </p>
     * <p>
     * <strong>支持的表达式 / Supported expressions:</strong>
     * <ul>
     * <li>"1:5" - 从索引1到4（不包含5）/ From index 1 to 4 (exclusive of 5)</li>
     * <li>":5" - 从开始到索引4 / From start to index 4</li>
     * <li>"1:" - 从索引1到结束 / From index 1 to end</li>
     * <li>"::2" - 每隔2个元素取一个 / Every 2nd element</li>
     * <li>"1:5:2" - 从索引1到4，每隔2个元素 / From index 1 to 4, every 2nd element</li>
     * </ul>
     * </p>
     *
     * @param sliceExpression 切片表达式，如 "1:5", "::2" / Slice expression, e.g.
     * "1:5", "::2"
     * @return 切片向量 / Sliced vector
     * @throws IllegalArgumentException 如果切片表达式无效 / if slice expression is
     * invalid
     */
    public IVector<T> slice(String sliceExpression);

    /**
     * 花式索引 / Fancy indexing
     * <p>
     * 根据位置索引数组获取对应位置的元素组成新向量 Gets elements at specified positions to form a new
     * vector
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector = Linalg.vector(new double[]{10, 20, 30, 40, 50});
     * int[] positions = {0, 2, 4};
     * IVector<Double> result = vector.fancyGet(positions);  // 结果: [10, 30, 50]
     * }</pre>
     * </p>
     *
     * @param positions 位置索引数组 / Array of position indices
     * @return 新的向量对象，包含指定位置的元素 / New vector object containing elements at
     * specified positions
     * @throws IndexOutOfBoundsException 如果任何索引超出范围 / if any index is out of
     * bounds
     * @throws IllegalArgumentException 如果positions为null或空 / if positions is
     * null or empty
     */
    public IVector<T> fancyGet(int[] positions);

    /**
     * 花式索引赋值 / Fancy indexing set
     * <p>
     * 根据位置索引数组对向量对应位置赋值。Sets values at specified positions using fancy indexing.
     * </p>
     *
     * @param positions 位置索引数组 / Array of position indices
     * @param values 要设置的值数组 / Array of values to set
     * @throws IndexOutOfBoundsException 如果任何索引超出范围 / if any index is out of bounds
     * @throws IllegalArgumentException 如果数组长度不匹配 / if array length mismatch
     */
    public void fancySet(int[] positions, T[] values);

    /**
     * 花式索引赋值（标量）/ Fancy indexing set (scalar)
     * <p>
     * 根据位置索引数组对向量对应位置赋值同一个标量值。Sets a scalar value at
     * specified positions.
     * </p>
     *
     * @param positions 位置索引数组 / Array of position indices
     * @param value 要设置的标量值 / Scalar value to set
     * @throws IndexOutOfBoundsException 如果任何索引超出范围 / if any index is out of bounds
     */
    public void fancySetScalar(int[] positions, T value);

    /**
     * 布尔索引赋值 / Boolean indexing set
     * <p>
     * 根据布尔数组对向量中满足条件的元素赋值。Sets values where the corresponding
     * boolean index is true.
     * </p>
     *
     * @param booleanIndex 布尔索引数组 / Boolean index array
     * @param values 要设置的值数组（长度需与true位置的数量匹配）/ Array of values to set
     * @throws IllegalArgumentException 如果布尔数组长度与向量长度不匹配 / if boolean array length doesn't match vector length
     */
    public void booleanSet(boolean[] booleanIndex, T[] values);

    /**
     * 布尔索引赋值（标量）/ Boolean indexing set (scalar)
     * <p>
     * 根据布尔数组对向量中满足条件的元素赋值同一个标量值。Sets a scalar value where
     * the corresponding boolean index is true.
     * </p>
     *
     * @param booleanIndex 布尔索引数组 / Boolean index array
     * @param value 要设置的标量值 / Scalar value to set
     * @throws IllegalArgumentException 如果布尔数组长度与向量长度不匹配 / if boolean array length doesn't match vector length
     */
    public void booleanSetScalar(boolean[] booleanIndex, T value);

    /**
     * 向量元素倒数 / Vector reciprocal
     * <p>
     * 对向量中每个元素计算倒数（1/x） Computes the reciprocal (1/x) of each element in the
     * vector
     * </p>
     *
     * @return 新的向量对象，包含倒数结果 / New vector object containing reciprocal results
     * @throws ArithmeticException 如果任何元素值为零 / if any element value is zero
     */
    public IVector<T> reciprocal();

    /**
     * 向量外积 / Vector outer product
     * <p>
     * 计算当前向量与另一个向量的外积（张量积） Computes the outer product (tensor product) of
     * current vector with another vector
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 外积矩阵 / Outer product matrix
     * @throws IllegalArgumentException 如果向量为null / if vector is null
     */
    public IMatrix<T> outer(IVector<T> other);

    /**
     * 三维欧氏向量叉积 / Cross product (aligned with {@code numpy.cross} for 3-vectors)
     *
     * @throws IllegalArgumentException 若任一向量长度不为 3 / unless both lengths are 3
     */
    IVector<T> cross(IVector<T> other);

    /**
     * 升序向量中查找插入位置（左边界，等价 {@code numpy.searchsorted(..., side='left')}）/ Sorted
     * search index
     *
     * @throws IllegalArgumentException 若向量未按非降序排列 / if not sorted
     * non-decreasing
     */
    int searchSorted(double value);

    // ========== IVector<T> 特有方法 / IVector<T>-specific Methods ==========
    // Note: 大部分方法现在在 IVector 中定义，这里保留类型特化的方法
    // Note: Most methods are now defined in IVector, keeping type-specific methods here
    /**
     * <p>
     * 行向量与矩阵相乘（与 NumPy {@code np.dot(v, M)}、{@link #mmul(IMatrix)} 一致） / Row
     * vector times matrix, aligned with NumPy {@code np.dot(v, M)} and
     * {@link #mmul(IMatrix)}
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector = Linalg.vector(new double[]{1, 2, 3});
     * IMatrix<Double> matrix = Linalg.matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
     * IVector<Double> result = vector.dot(matrix);  // [22, 28]，等价 vector.mmul(matrix)
     * }</pre>
     * </p>
     *
     * @param m 矩阵 / Matrix
     * @return 长度为矩阵列数的行向量结果 / Row-shaped result vector (length = matrix
     * columns)
     * @throws IllegalArgumentException 如果矩阵为null或维度不匹配 / if matrix is null or
     * dimensions don't match
     */
    public IVector<T> dot(IMatrix<T> m);

    /**
     * 向量相等比较 / Vector equality comparison
     * <p>
     * 逐元素比较两个向量是否相等 Compares two vectors element-wise for equality
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector1 = Linalg.vector(new double[]{1, 2, 3});
     * IVector<Double> vector2 = Linalg.vector(new double[]{1, 3, 3});
     * boolean[] result = vector1.equals(vector2);  // 结果: [true, false, true]
     * }</pre>
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示每个元素的比较结果 / Boolean array representing comparison result
     * for each element
     * @throws IllegalArgumentException 如果向量为null或长度不匹配 / if vector is null or
     * lengths don't match
     */
    public boolean[] eq(IVector<T> other);

    /**
     * 向量小于比较 / Vector less-than comparison
     * <p>
     * 逐元素比较当前向量是否小于另一个向量 Compares current vector element-wise to check if less
     * than another vector
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector1 = Linalg.vector(new double[]{1, 3, 2});
     * IVector<Double> vector2 = Linalg.vector(new double[]{2, 2, 3});
     * boolean[] result = vector1.lessThan(vector2);  // 结果: [true, false, true]
     * }</pre>
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示每个元素的比较结果 / Boolean array representing comparison result
     * for each element
     * @throws IllegalArgumentException 如果向量为null或长度不匹配 / if vector is null or
     * lengths don't match
     */
    public boolean[] lt(IVector<T> other);

    /**
     * 向量大于比较 / Vector greater-than comparison
     * <p>
     * 逐元素比较当前向量是否大于另一个向量 Compares current vector element-wise to check if
     * greater than another vector
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector1 = Linalg.vector(new double[]{3, 1, 4});
     * IVector<Double> vector2 = Linalg.vector(new double[]{2, 2, 3});
     * boolean[] result = vector1.greaterThan(vector2);  // 结果: [true, false, true]
     * }</pre>
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示每个元素的比较结果 / Boolean array representing comparison result
     * for each element
     * @throws IllegalArgumentException 如果向量为null或长度不匹配 / if vector is null or
     * lengths don't match
     */
    public boolean[] gt(IVector<T> other);
    
    public boolean[] ge(IVector<T> other);
    
    public boolean[] le(IVector<T> other);

    // Note: 基本数学函数、三角函数、双曲函数、舝入函数、切片操作等
    // 现在都在 IVector 中定义，从 IVector<T> 继承
    // Note: Basic math functions, trigonometric functions, hyperbolic functions, rounding functions, slicing operations
    // are now defined in IVector and inherited by IVector<T>
    // ========== Float特有方法 / Float-specific Methods ==========
    /**
     * 向量裁剪 / Vector clipping
     * <p>
     * 将向量中的元素值限制在指定范围内 Clips vector elements to the specified range
     * </p>
     *
     * @param lower 下界 / Lower bound
     * @param upper 上界 / Upper bound
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     * @throws IllegalArgumentException 如果lower > upper / if lower > upper
     */
    public IVector<T> clip(double lower, double upper);

    /**
     * 向量峰峰值 / Vector peak-to-peak value
     * <p>
     * 计算向量的峰峰值（最大值减最小值） Calculates the peak-to-peak value of the vector (max -
     * min)
     * </p>
     *
     * @return 峰峰值 / Peak-to-peak value
     */
    public IVector<T> ptp();

    default double ptpValue() { return ptp().get(0); }

    /**
     * 向量取余运算 / Vector remainder operation
     * <p>
     * 对向量中每个元素进行取余运算 Performs remainder operation on each element in the vector
     * </p>
     *
     * @param value 除数 / Divisor
     * @return 新的向量对象，包含取余运算结果 / New vector object containing remainder results
     * @throws ArithmeticException 如果除数为零 / if divisor is zero
     */
    public IVector<T> remainder(T value);

    /**
     * 布尔索引 / Boolean indexing
     * <p>
     * 根据布尔数组获取对应位置为true的元素组成新向量 Gets elements where the corresponding boolean
     * index is true to form a new vector
     * </p>
     *
     * @param booleanIndex 布尔索引数组 / Boolean index array
     * @return 新的向量对象，包含布尔索引为true位置的元素 / New vector object containing elements
     * where boolean index is true
     * @throws IllegalArgumentException 如果布尔数组长度与向量长度不匹配 / if boolean array
     * length doesn't match vector length
     */
    public IVector<T> booleanGet(boolean[] booleanIndex);

    /**
     * 范围设置值（带步长） / Range set values (with step)
     * <p>
     * 设置指定范围内、指定步长位置的元素值 Sets element values at positions within specified
     * range with specified step
     * </p>
     *
     * @param start 开始位置 / Start position
     * @param end 结束位置（不包含） / End position (exclusive)
     * @param step 步长 / Step size
     * @param values 要设置的值数组 / Array of values to set
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are
     * out of bounds
     * @throws IllegalArgumentException 如果值数组长度不匹配 / if values array length
     * doesn't match
     */
    public void setFromTo(int start, int end, int step, T[] values);

    /**
     * 范围设置值 / Range set values
     * <p>
     * 设置指定范围内的元素值（步长为1） Sets element values within specified range (step size
     * 1)
     * </p>
     *
     * @param start 开始位置 / Start position
     * @param end 结束位置（不包含） / End position (exclusive)
     * @param values 要设置的值数组 / Array of values to set
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are
     * out of bounds
     * @throws IllegalArgumentException 如果值数组长度不匹配 / if values array length
     * doesn't match
     */
    public void setFromTo(int start, int end, T[] values);

    /**
     * 向量填充 / Vector fill
     * <p>
     * 将向量中所有元素设置为指定值 Sets all elements in the vector to the specified value
     * </p>
     *
     * @param value 填充值 / Fill value
     */
    public void fill(double value);

    /**
     * 向量排序 / Vector sorting
     * <p>
     * 对向量中的元素进行升序排序 Sorts the elements in the vector in ascending order
     * </p>
     *
     * @return 排序后的向量（就地操作） / Sorted vector (in-place operation)
     */
    public IVector<T> sort();

    /**
     * 向量反转 / Vector reverse
     * <p>
     * 反转向量中元素的顺序 Reverses the order of elements in the vector
     * </p>
     *
     * @return 反转后的向量（就地操作） / Reversed vector (in-place operation)
     */
    public IVector<T> reverse();

    /**
     * 获取向量长度 / Get vector size (The same as length)
     * <p>
     * 返回向量的长度（元素个数） Returns the size (number of elements) of the vector
     * </p>
     *
     * @return 向量长度 / IVector<T> size
     */
    public int size();

    /**
     * 转换为整数数组 / Convert to integer array
     * <p>
     * 将向量转换为整数数组 Converts the vector to an integer array
     * </p>
     *
     * @return 整数数组 / Integer array
     */
    public int[] toIntArray();

    /**
     * 转换为双精度数组 / Convert to double array
     * <p>
     * 将向量转换为双精度数组 Converts the vector to a double array
     * </p>
     *
     * @return 双精度数组 / Double array
     * @apiNote 对 {@link com.yishape.lab.math.linalg.RereDoubleVector}
     * 等存储型实现，返回<strong>防御性拷贝</strong>； 需共享可写后备数组请使用
     * {@link IDoubleVector#getData()} 等。
     */
    public double[] toDoubleArray();

    /**
     * 转换为单精度浮点数组 / Convert to float array
     * <p>
     * 将向量转换为单精度浮点数组 Converts the vector to a float array
     * </p>
     *
     * @return 单精度浮点数组 / Float array
     * @apiNote 对 {@link com.yishape.lab.math.linalg.RereFloatVector}
     * 等存储型实现，返回<strong>防御性拷贝</strong>； 需共享可写后备数组请使用
     * {@link IFloatVector#getData()} 等。
     */
    public float[] toFloatArray();

    /**
     * 计算与另一个向量的欧几里得距离 / Compute Euclidean distance to another vector
     * <p>
     * 计算当前向量与另一个向量之间的欧几里得距离 Computes the Euclidean distance between current
     * vector and another vector
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 欧几里得距离 / Euclidean distance
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public double euclideanDistance(IVector<T> other);

    /**
     * 计算与另一个向量的曼哈顿距离 / Compute Manhattan distance to another vector
     * <p>
     * 计算当前向量与另一个向量之间的曼哈顿距离（L1距离） Computes the Manhattan distance (L1 distance)
     * between current vector and another vector
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 曼哈顿距离 / Manhattan distance
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public double manhattanDistance(IVector<T> other);

    /**
     * 计算与另一个向量的余弦相似度 / Compute cosine similarity to another vector
     * <p>
     * 计算当前向量与另一个向量之间的余弦相似度 Computes the cosine similarity between current
     * vector and another vector
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 余弦相似度 / Cosine similarity
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     * @throws ArithmeticException 如果向量长度为零 / if vector norm is zero
     */
    public double cosineSimilarity(IVector<T> other);

    // ========== 三角函数操作 / Trigonometric Functions ==========
    // ========== 逻辑运算 / Logical Operations ==========
    /**
     * 向量逻辑与运算 / Vector logical AND operation
     * <p>
     * 对向量中每个元素进行逻辑与运算（非零为true，零为false） Performs logical AND operation on each
     * element (non-zero as true, zero as false)
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 新的向量对象，包含逻辑与运算结果 / New vector object containing logical AND
     * results
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public boolean[] logicalAnd(IVector<T> other);

    /**
     * 向量逻辑或运算 / Vector logical OR operation
     * <p>
     * 对向量中每个元素进行逻辑或运算（非零为true，零为false） Performs logical OR operation on each
     * element (non-zero as true, zero as false)
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 新的向量对象，包含逻辑或运算结果 / New vector object containing logical OR
     * results
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public boolean[] logicalOr(IVector<T> other);

    /**
     * 向量逻辑非运算 / Vector logical NOT operation
     * <p>
     * 对向量中每个元素进行逻辑非运算（非零为false，零为true） Performs logical NOT operation on each
     * element (non-zero as false, zero as true)
     * </p>
     *
     * @return 新的向量对象，包含逻辑非运算结果 / New vector object containing logical NOT
     * results
     */
    public boolean[] logicalNot();

    /**
     * 向量逻辑异或运算 / Vector logical XOR operation
     * <p>
     * 对向量中每个元素进行逻辑异或运算（非零为true，零为false） Performs logical XOR operation on each
     * element (non-zero as true, zero as false)
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 新的向量对象，包含逻辑异或运算结果 / New vector object containing logical XOR
     * results
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public boolean[] logicalXor(IVector<T> other);

    // ========== 累积操作 / Cumulative Operations ==========
    /**
     * 向量累积求和 / Vector cumulative sum
     * <p>
     * 计算向量中元素的累积和 Computes the cumulative sum of elements in the vector
     * </p>
     *
     * @return 新的向量对象，包含累积和结果 / New vector object containing cumulative sum
     * results
     */
    public IVector<T> cumsum();

    /**
     * 向量累积乘积 / Vector cumulative product
     * <p>
     * 计算向量中元素的累积乘积 Computes the cumulative product of elements in the vector
     * </p>
     *
     * @return 新的向量对象，包含累积乘积结果 / New vector object containing cumulative product
     * results
     */
    public IVector<T> cumprod();

    // ========== 差分操作 / Difference Operations ==========
    /**
     * 向量差分 / Vector difference
     * <p>
     * 计算向量中相邻元素的差分 Computes the difference between adjacent elements in the
     * vector
     * </p>
     *
     * @return 新的向量对象，包含差分结果 / New vector object containing difference results
     */
    public IVector<T> diff();

    /**
     * 向量差分（指定阶数） / Vector difference (specified order)
     * <p>
     * 计算向量中元素的n阶差分 Computes the n-th order difference of elements in the vector
     * </p>
     *
     * @param n 差分阶数 / Difference order
     * @return 新的向量对象，包含差分结果 / New vector object containing difference results
     * @throws IllegalArgumentException 如果阶数小于1或大于等于向量长度 / if order is less than
     * 1 or greater than or equal to vector length
     */
    public IVector<T> diff(int n);

    // ========== 条件操作 / Conditional Operations ==========
    /**
     * 向量条件选择 / Vector where operation
     * <p>
     * 根据条件选择元素值 Selects element values based on conditions
     * </p>
     *
     * @param condition 条件向量（布尔数组） / Condition vector (boolean array)
     * @param x 条件为true时的值 / Value when condition is true
     * @param y 条件为false时的值 / Value when condition is false
     * @return 新的向量对象，包含条件选择结果 / New vector object containing conditional
     * selection results
     * @throws IllegalArgumentException 如果条件向量长度不匹配 / if condition vector length
     * doesn't match
     */
    public IVector<T> where(boolean[] condition, T x, T y);

    /**
     * 向量条件选择（向量值） / Vector where operation (vector values)
     * <p>
     * 根据条件从两个向量中选择元素值 Selects element values from two vectors based on
     * conditions
     * </p>
     *
     * @param condition 条件向量（布尔数组） / Condition vector (boolean array)
     * @param x 条件为true时的值向量 / Value vector when condition is true
     * @param y 条件为false时的值向量 / Value vector when condition is false
     * @return 新的向量对象，包含条件选择结果 / New vector object containing conditional
     * selection results
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public IVector<T> where(boolean[] condition, IVector<T> x, IVector<T> y);

    // ========== 重复和连接操作 / Repeat and Concatenation Operations ==========
    /**
     * 向量重复 / Vector repeat
     * <p>
     * 重复向量中的每个元素指定次数 Repeats each element in the vector specified number of
     * times
     * </p>
     *
     * @param repeats 每个元素重复的次数 / Number of times to repeat each element
     * @return 新的向量对象，包含重复结果 / New vector object containing repeated results
     * @throws IllegalArgumentException 如果重复次数小于1 / if repeat count is less than
     * 1
     */
    public IVector<T> repeat(int repeats);

    /**
     * 向量平铺 / Vector tile
     * <p>
     * 将整个向量重复指定次数 Repeats the entire vector specified number of times
     * </p>
     *
     * @param reps 重复次数 / Number of repetitions
     * @return 新的向量对象，包含平铺结果 / New vector object containing tiled results
     * @throws IllegalArgumentException 如果重复次数小于1 / if repeat count is less than
     * 1
     */
    public IVector<T> tile(int reps);

    // ========== 统计扩展操作 / Extended Statistical Operations ==========
    /**
     * 向量中位数 / Vector median
     * <p>
     * 计算向量中元素的中位数 Computes the median of elements in the vector
     * </p>
     *
     * @return 中位数 / Median value
     */
    public double median();

    /**
     * 向量百分位数 / Vector percentile
     * <p>
     * 计算向量中元素的指定百分位数 Computes the specified percentile of elements in the
     * vector
     * </p>
     *
     * @param q 百分位数（0-100） / Percentile (0-100)
     * @return 百分位数 / Percentile value
     * @throws IllegalArgumentException 如果百分位数超出[0,100]范围 / if percentile is
     * outside [0,100] range
     */
    public double percentile(double q);

    /**
     * 向量众数 / Vector mode
     * <p>
     * 计算向量中出现频率最高的元素值 Computes the most frequently occurring element value in
     * the vector
     * </p>
     *
     * @return 众数 / Mode value
     */
    public double mode();

    // ========== 线性代数扩展操作 / Extended Linear Algebra Operations ==========
    /**
     * 向量p范数 / Vector Lp norm
     * <p>
     * 计算向量的p范数 Calculates the Lp norm of the vector
     * </p>
     *
     * @param p 范数阶数 / Norm order
     * @return p范数 / Lp norm
     * @throws IllegalArgumentException 如果p小于1 / if p is less than 1
     */
    public double norm(double p);

    /**
     * 向量无穷范数 / Vector L-infinity norm
     * <p>
     * 计算向量的无穷范数（最大绝对值） Calculates the L-infinity norm (maximum absolute value)
     * of the vector
     * </p>
     *
     * @return 无穷范数 / L-infinity norm
     */
    public double normInf();

    /**
     * 向量归一化 / Vector normalization
     * <p>
     * 将向量归一化为单位向量（L2范数为1） Normalizes the vector to unit vector (L2 norm equals
     * 1)
     * </p>
     *
     * @return 新的向量对象，包含归一化结果 / New vector object containing normalized results
     * @throws ArithmeticException 如果向量L2范数为零 / if vector L2 norm is zero
     */
    public IVector<T> normalize();

    /**
     * 计算向量偏度 / Calculate vector skewness
     * <p>
     * 计算向量中元素的偏度（三阶中心矩），用于衡量数据分布的对称性 Calculates the skewness (third central
     * moment) of elements in the vector, used to measure the symmetry of data
     * distribution
     * </p>
     * <p>
     * 偏度公式：skewness = E[(X - μ)³] / σ³ Skewness formula: skewness = E[(X - μ)³]
     * / σ³
     * </p>
     * <ul>
     * <li>skewness = 0: 对称分布 / Symmetric distribution</li>
     * <li>skewness > 0: 右偏分布（正偏） / Right-skewed distribution (positive
     * skew)</li>
     * <li>skewness &lt; 0: 左偏分布（负偏） / Left-skewed distribution (negative
     * skew)</li> </ul>
     *
     * @return 偏度值 / Skewness value
     * @throws ArithmeticException 如果向量长度小于3或标准差为0 / if vector length is less
     * than 3 or standard deviation is 0
     */
    public double skewness();

    /**
     * 计算向量峰度 / Calculate vector kurtosis
     * <p>
     * 计算向量中元素的峰度（四阶中心矩），用于衡量数据分布的尖锐程度 Calculates the kurtosis (fourth central
     * moment) of elements in the vector, used to measure the sharpness of data
     * distribution
     * </p>
     * <p>
     * 峰度公式：kurtosis = E[(X - μ)⁴] / σ⁴ - 3 Kurtosis formula: kurtosis = E[(X -
     * μ)⁴] / σ⁴ - 3
     * </p>
     * <ul>
     * <li>kurtosis = 0: 正态分布峰度 / Normal distribution kurtosis</li>
     * <li>kurtosis > 0: 尖峰分布（重尾） / Leptokurtic distribution (heavy tails)</li>
     * <li>kurtosis &lt; 0: 平峰分布（轻尾） / Platykurtic distribution (light
     * tails)</li>
     * </ul>
     *
     * @return 峰度值 / Kurtosis value
     * @throws ArithmeticException 如果向量长度小于4或标准差为0 / if vector length is less
     * than 4 or standard deviation is 0
     */
    public double kurtosis();

    /**
     * 作为列向量，实质是一个m*1的矩阵
     * <p>
     * 将向量转换为列向量矩阵，即m×1的矩阵，其中m是向量的长度。 向量的每个元素成为矩阵对应行的第一列元素。
     * </p>
     * <p>
     * Converts the vector to a column vector matrix, i.e., an m×1 matrix where
     * m is the vector length. Each element of the vector becomes the first
     * column element of the corresponding row in the matrix.
     * </p>
     *
     * @return 列向量矩阵（m×1）/ Column vector matrix (m×1)
     */
    public IMatrix<T> asColumnVector();

    /**
     * 动态时间规整距离 / Dynamic Time Warping Distance
     * <p>
     * 计算两个向量之间的动态时间规整距离，用于时间序列数据的相似性度量 Computes the Dynamic Time Warping (DTW)
     * distance between two vectors, used for similarity measurement of time
     * series data
     * </p>
     * <p>
     * DTW是一种用于比较两个时间序列相似性的算法，允许时间轴的非线性对齐 DTW is an algorithm for comparing the
     * similarity of two time series, allowing for non-linear alignment of the
     * time axis
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> series1 = Linalg.vector(new double[]{1, 2, 3, 4, 5});
     * IVector<Double> series2 = Linalg.vector(new double[]{1, 1, 3, 5, 5});
     * Double dtwDistance = series1.dtw(series2);
     * }</pre>
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return DTW距离 / DTW distance
     * @throws IllegalArgumentException 如果向量为null / if vector is null
     */
    public double dtw(IVector<T> other);

    /**
     * 皮尔逊相关系数 / Pearson Correlation Coefficient
     * <p>
     * 计算两个向量之间的皮尔逊相关系数，衡量线性相关性 Computes the Pearson correlation coefficient
     * between two vectors, measuring linear correlation
     * </p>
     * <p>
     * 相关系数范围在[-1, 1]之间： Correlation coefficient ranges from [-1, 1]:
     * <ul>
     * <li>1: 完全正相关 / Perfect positive correlation</li>
     * <li>0: 无线性相关 / No linear correlation</li>
     * <li>-1: 完全负相关 / Perfect negative correlation</li>
     * </ul>
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
     * IVector<Double> y = Linalg.vector(new double[]{2, 4, 6, 8, 10});
     * Double correlation = x.corr(y);  // 结果接近1.0
     * }</pre>
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 皮尔逊相关系数 / Pearson correlation coefficient
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     * @throws ArithmeticException 如果任一向量的标准差为0 / if standard deviation of any
     * vector is 0
     */
    public double corr(IVector<T> other);

    /**
     * 协方差 / Covariance
     * <p>
     * 计算两个向量之间的协方差，衡量联合变异性 Computes the covariance between two vectors,
     * measuring joint variability
     * </p>
     * <p>
     * 协方差公式：cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] Covariance formula: cov(X,Y) =
     * E[(X-μₓ)(Y-μᵧ)]
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> x = Linalg.vector(new double[]{1, 2, 3, 4, 5});
     * IVector<Double> y = Linalg.vector(new double[]{2, 4, 6, 8, 10});
     * Double covariance = x.cov(y);
     * }</pre>
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 协方差 / Covariance
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    public double cov(IVector<T> other);

    /**
     * 向量元素映射操作 / Vector element mapping operation
     * <p>
     * 对向量中的每个元素应用指定的函数，返回同形状的新向量 Applies the specified function to each element
     * in the vector, returns a new vector of the same shape
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector = Linalg.vector(new double[]{1, 2, 3, 4, 5});
     * IVector<Double> squared = vector.apply(x -> x * x);  // 结果: [1, 4, 9, 16, 25]
     * IVector<Double> doubled = vector.apply(x -> x * 2);  // 结果: [2, 4, 6, 8, 10]
     * }</pre>
     * </p>
     *
     * @param fun 要应用的函数，接受一个元素并返回转换后的元素 / Function to apply, takes an element
     * and returns transformed element
     * @return 新的向量对象，包含应用函数后的结果 / New vector object containing results after
     * applying the function
     * @throws IllegalArgumentException 如果fun为null / if fun is null
     */
    public IVector<T> map(Function<T, T> fun);
    
    /**
     * map相同类型方法
     * @param fun
     * @return 
     */
    default IVector<T> apply(Function<T, T> fun) {
        return this.map(fun);
    }

    /**
     * 第一四分位数（25%分位数）/ First quartile (25th percentile)
     * <p>
     * 计算向量的第一四分位数，即25%分位数。第一四分位数是将数据按升序排列后， 位于25%位置的值，表示有25%的数据小于等于该值。
     * </p>
     * <p>
     * 计算方法：
     * <ul>
     * <li>将向量数据按升序排序</li>
     * <li>计算位置索引：index = 0.25 * (n - 1)，其中n是数据长度</li>
     * <li>如果索引是整数，直接取该位置的值</li>
     * <li>如果索引不是整数，使用线性插值计算</li>
     * </ul>
     * </p>
     *
     * @return 第一四分位数 / First quartile
     * @throws ArithmeticException 如果向量为空 / if vector is empty
     */
    public default double q1() {
        if (this.length() == 0) {
            throw new ArithmeticException("空向量无法计算第三四分位数 / Cannot compute third quartile for empty vector");
        }

        // 使用已有的percentile方法计算25%分位数
        return this.percentile(25.0);
    }

    /**
     * 第三四分位数（75%分位数）/ Third quartile (75th percentile)
     * <p>
     * 计算向量的第三四分位数，即75%分位数。第三四分位数是将数据按升序排列后， 位于75%位置的值，表示有75%的数据小于等于该值。
     * </p>
     * <p>
     * 计算方法：
     * <ul>
     * <li>将向量数据按升序排序</li>
     * <li>计算位置索引：index = 0.75 * (n - 1)，其中n是数据长度</li>
     * <li>如果索引是整数，直接取该位置的值</li>
     * <li>如果索引不是整数，使用线性插值计算</li>
     * </ul>
     * </p>
     *
     * @return 第三四分位数 / Third quartile
     * @throws ArithmeticException 如果向量为空 / if vector is empty
     */
    public default double q3() {
        if (this.length() == 0) {
            throw new ArithmeticException("空向量无法计算第三四分位数 / Cannot compute third quartile for empty vector");
        }

        // 使用已有的percentile方法计算75%分位数
        return this.percentile(75.0);
    }

    /**
     * 向量连接 / Vector concatenation
     * <p>
     * 将当前向量与另一个向量连接，形成一个新的向量 Concatenates current vector with another vector to
     * form a new vector
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector1 = Linalg.vector(new double[]{1, 2, 3});
     * IVector<Double> vector2 = Linalg.vector(new double[]{4, 5, 6});
     * IVector<Double> result = vector1.concat(vector2);  // 结果: [1, 2, 3, 4, 5, 6]
     * }</pre>
     * </p>
     *
     * @param other 要连接的向量 / Vector to concatenate
     * @return 连接后的新向量 / New concatenated vector
     * @throws IllegalArgumentException 如果向量为null / if vector is null
     */
    public IVector<T> concat(IVector<T> other);

    /**
     * 向量符号函数 / Vector sign function
     * <p>
     * 对向量中每个元素计算符号函数：如果元素为零则返回0，如果元素大于零则返回1，如果元素小于零则返回-1 Computes the sign
     * function for each element in the vector: returns 0 if element is zero, 1
     * if element is greater than zero, -1 if element is less than zero
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector = Linalg.vector(new double[]{-3, 0, 2.5});
     * IVector<Double> result = vector.sign();  // 结果: [-1, 0, 1]
     * }</pre>
     * </p>
     *
     * @return 符号函数结果向量 / Sign function result vector
     */
    public IVector<T> sign();

    /**
     * 向量重塑为矩阵 / Reshape vector to matrix
     * <p>
     * 将向量重新构型为指定行数和列数的矩阵 Reshapes the vector into a matrix with specified rows
     * and columns
     * </p>
     * <p>
     * <strong>使用示例 / Usage Example:</strong>
     * <pre>{@code
     * IVector<Double> vector = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6});
     * IMatrix<Double> matrix = vector.reshape(2, 3);  // 结果: [[1, 2, 3], [4, 5, 6]]
     * }</pre>
     * </p>
     * <p>
     * <strong>注意 / Note:</strong> 向量长度必须等于 rows × cols Vector length must equal
     * rows × cols
     * </p>
     *
     * @param rows 矩阵行数 / Number of rows in the matrix
     * @param cols 矩阵列数 / Number of columns in the matrix
     * @return 重塑后的矩阵 / Reshaped matrix
     * @throws IllegalArgumentException 如果rows或cols小于等于0，或者rows×cols不等于向量长度 / if
     * rows or cols is less than or equal to 0, or rows×cols doesn't equal
     * vector length
     */
    public IMatrix<T> reshape(int rows, int cols);

    /**
     * 计算向量的Hessian矩阵 / Compute the Hessian matrix of the vector
     * <p>
     * 该方法是针对逻辑回归等机器学习场景的特殊实现，其中向量元素被视为逻辑函数的输入（logits）。
     * 计算得到的Hessian矩阵是一个对角矩阵，其对角线元素为：H[i,i] = sigmoid(f[i]) * (1 -
     * sigmoid(f[i]))
     * </p>
     *
     * <h3>数学原理 / Mathematical Principle:</h3>
     * <p>
     * 在逻辑回归中，对数似然函数的二阶导数（Hessian）具有特殊形式： H = X^T * W * X，其中W是对角权重矩阵，W[i,i] =
     * p[i] * (1 - p[i]) 这里p[i] = sigmoid(f[i])是预测概率。
     * </p>
     *
     * <h3>实现特点 / Implementation Features:</h3>
     * <ul>
     * <li><strong>输入验证</strong>：检查向量中是否包含NaN或无穷大值</li>
     * <li><strong>数值稳定性</strong>：处理sigmoid计算中的数值边缘情况</li>
     * <li><strong>性能优化</strong>：根据向量大小采用不同的计算策略</li>
     * <li><strong>内存效率</strong>：只计算对角线元素，其余默认为0</li>
     * </ul>
     *
     * <h3>使用场景 / Usage Scenarios:</h3>
     * <ul>
     * <li>逻辑回归的牛顿-拉夫逊优化 / Newton-Raphson optimization for logistic
     * regression</li>
     * <li>神经网络反向传播中的曲率信息 / Curvature information in neural network
     * backpropagation</li>
     * <li>统计学习中的Fisher信息矩阵 / Fisher information matrix in statistical
     * learning</li>
     * </ul>
     *
     *
     * @return 对角Hessian矩阵，类型为IMatrix<Float>
     * @throws IllegalArgumentException 如果向量包含NaN或无穷大值
     * @see com.yishape.lab.math.RereMathUtil#sigmoid(double)
     * @see IMatrix
     */
    public IMatrix<T> hessianMatrix();

    /**
     * 返回由向量元素构成的单列矩阵（列向量）
     *
     * @return
     */
    public default IMatrix<T> asColumnMatrix() {
        return this.asColumnVector();
    }

    /**
     * 返回由向量元素构成的单行矩阵
     *
     * @return
     */
    public default IMatrix<T> asRowMatrix() {
        return this.asColumnVector().transpose();
    }

    /**
     * 转换为List
     *
     * @return
     */
    @SuppressWarnings("unchecked")
    public default List<T> toList() {
        List ls = new ArrayList();
        int size = this.size();
        for (int i = 0; i < size; i++) {
            ls.add(get(i));
        }
        return ls;
    }

    /**
     * 将向量转换为Double类型
     * <p>
     * 如果向量已经是Double类型，直接返回当前向量；否则创建新的Double类型向量
     * </p>
     *
     * @return Double类型的向量
     */
    public default IVector<Double> toDoubleVector() {
        // 如果原vector本来就是double，则不需要创建，直接返回
        if (this instanceof IDoubleVector) {
            @SuppressWarnings("unchecked")
            IVector<Double> doubleVector = (IVector<Double>) this;
            return doubleVector;
        }
        
        int length = this.length();
        double[] doubleData = new double[length];

        // 复制数据并转换为double类型
        for (int i = 0; i < length; i++) {
            doubleData[i] = this.get(i);
        }

        // 创建并返回新的Double向量
        return IVector.of(doubleData);
    }

    /**
     * 将向量转换为Float类型
     * <p>
     * 如果向量已经是Float类型，直接返回当前向量；否则创建新的Float类型向量
     * </p>
     *
     * @return Float类型的向量
     */
    public default IVector<Float> toFloatVector() {
        // 如果原vector本来就是float，则不需要创建，直接返回
        if (this instanceof IFloatVector) {
            @SuppressWarnings("unchecked")
            IVector<Float> floatVector = (IVector<Float>) this;
            return floatVector;
        }
        
        int length = this.length();
        float[] floatData = new float[length];

        // 复制数据并转换为float类型
        for (int i = 0; i < length; i++) {
            floatData[i] = (float) this.get(i);
        }

        // 创建并返回新的Float向量
        return IVector.of(floatData);
    }

    // ========== 激活函数 / Activation Functions ==========

    /** Numerically stable softmax: exp(x-max) / sum(exp(x-max)). */
    public default IVector<T> softmax() {
        double max = this.maxValue();
        IVector<T> shifted = this.subScalar(max);
        IVector<T> exps = shifted.exp();
        return exps.divideByScalar(exps.sumValue());
    }

    /** Numerically stable log-softmax: log(softmax(x)). */
    public default IVector<T> logSoftmax() {
        double max = this.maxValue();
        IVector<T> shifted = this.subScalar(max);
        double logSum = Math.log(shifted.exp().sumValue());
        return shifted.subScalar(logSum);
    }

    /**
     * Gaussian Error Linear Unit.
     * 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     */
    public default IVector<T> gelu() {
        IVector<T> x = this;
        IVector<T> x3 = x.pow(3.0);
        double coeff = Math.sqrt(2.0 / Math.PI);
        IVector<T> inner = x.add(x3.multiplyByScalar(0.044715)).multiplyByScalar(coeff);
        return x.multiply(inner.tanh().addScalar(1.0)).multiplyByScalar(0.5);
    }

    /** SiLU / Swish: x * sigmoid(x). */
    public default IVector<T> silu() {
        return this.multiply(this.sigmoid());
    }

    /** Leaky ReLU: x if x > 0, alpha * x otherwise. */
    public default IVector<T> leakyRelu(double alpha) {
        IVector<T> r = this.relu();
        return r.add(this.sub(r).multiplyByScalar(alpha));
    }

    /**
     * Exponential Linear Unit.
     * x if x > 0, alpha * (exp(x) - 1) if x <= 0.
     * Implemented as: relu(x) + alpha * (exp(x - relu(x)) - 1)
     */
    public default IVector<T> elu(double alpha) {
        IVector<T> r = this.relu();
        return r.add(this.sub(r).exp().subScalar(1.0).multiplyByScalar(alpha));
    }

    /**
     * Scaled Exponential Linear Unit (self-normalizing).
     * scale * (x if x > 0, alpha * (exp(x) - 1) if x <= 0)
     * with scale≈1.0507, alpha≈1.6733.
     */
    public default IVector<T> selu() {
        final double SCALE = 1.0507009873554804934193349852946;
        final double ALPHA = 1.6732632423543772848170429916717;
        IVector<T> r = this.relu();
        return r.add(this.sub(r).exp().subScalar(1.0).multiplyByScalar(ALPHA)).multiplyByScalar(SCALE);
    }

    /** Mish: x * tanh(softplus(x)). */
    public default IVector<T> mish() {
        return this.multiply(this.softplus(1.0).tanh());
    }

    /** Softplus: log(1 + exp(beta * x)) / beta. */
    public default IVector<T> softplus(double beta) {
        return this.multiplyByScalar(beta).exp().addScalar(1.0).log().divideByScalar(beta);
    }

    /** Hard tanh: clip to [minVal, maxVal]. */
    public default IVector<T> hardtanh(double minVal, double maxVal) {
        return this.clip(minVal, maxVal);
    }

    /** Alias for {@link #clip(double, double)}. */
    public default IVector<T> clamp(double min, double max) {
        return this.clip(min, max);
    }

    /** Negation: -this. */
    public default IVector<T> neg() {
        return this.multiplyByScalar(-1.0);
    }

    /** Reverse subtraction: scalar - this. */
    public default IVector<T> rsub(double scalar) {
        return this.neg().addScalar(scalar);
    }

    /** Reverse division: scalar / this. */
    public default IVector<T> rdiv(double scalar) {
        return this.reciprocal().multiplyByScalar(scalar);
    }

    /**
     * Fused Layer Normalization: y = gamma * (x - mean) / (std + eps) + beta.
     * @param gamma scale parameter
     * @param beta  shift parameter
     * @param eps   small constant for numerical stability
     */
    public default IVector<T> layerNorm(IVector<T> gamma, IVector<T> beta, double eps) {
        double mean = this.meanValue();
        double std = this.stdValue();
        return this.subScalar(mean).divideByScalar(std + eps).multiply(gamma).add(beta);
    }

    /** Dropout at inference: identity (returns this). */
    public default IVector<T> dropout(double p) {
        return this;
    }

}
