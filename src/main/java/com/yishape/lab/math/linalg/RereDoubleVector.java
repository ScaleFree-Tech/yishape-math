package com.yishape.lab.math.linalg;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.util.StringUtils;
import com.yishape.lab.math.compute.IDoubleVectorComputer;
import java.io.Serializable;

import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.concurrent.*;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.List;
import java.util.ArrayList;
import java.util.function.Function;

/**
 * 向量操作实现类 / Vector Operations Implementation Class
 * <p>
 * 本类实现了向量的常用操作，包括基本的数学运算、统计运算、切片索引、通用函数等功能。 基于一维Double数组实现，提供高效的向量计算能力。
 * </p>
 * <p>
 * This class implements common vector operations including basic mathematical
 * operations, statistical operations, slicing and indexing, universal functions
 * and other functionalities. Based on 1D Double array implementation, providing
 * efficient vector computation capabilities.
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
 * <li>线性代数：范数计算、排序、反转 / Linear algebra: norm calculation, sorting,
 * reversing</li>
 * </ul>
 *
 * <h3>使用示例 / Usage Example:</h3>
 * <pre>
 * {@code
 * // 创建向量 / Create vector
 * double[] data = {1.0f, 2.0f, 3.0f, 4.0f};
 * IVector<Double> vector = new RereDoubleVector(data);
 *
 * // 基本运算 / Basic operations
 * IVector<Double> doubled = vector.multiplyScala(2.0f);
 * Double norm = vector.norm2();
 *
 * // 统计运算 / Statistical operations
 * Double mean = vector.mean();
 * Double std = vector.std();
 *
 * // 切片操作 / Slicing operations
 * IVector<Double> slice = vector.slice(1, 3);
 * IVector<Double> squared = vector.squre();
 * }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereDoubleVector implements IDoubleVector,Serializable {

    private final IDoubleVectorComputer computer = new DoubleVectorComputer();

    /**
     * 向量数据存储数组 / Vector data storage array
     * <p>
     * 使用一维Double数组存储向量数据，data[i]表示第i个元素 Uses 1D Double array to store vector
     * data, data[i] represents the i-th element
     * </p>
     */
    private final double data[];

    // ========== 性能优化相关字段 / Performance Optimization Fields ==========
    /**
     * 向量对象池 / Vector object pool
     */
    private static final Queue<RereDoubleVector> VECTOR_POOL = new ConcurrentLinkedQueue<>();

    /**
     * 预分配的内存缓冲区 / Pre-allocated memory buffers
     */
    private static final ThreadLocal<double[]> BUFFER
            = ThreadLocal.withInitial(() -> new double[1024 * 1024]);

    /**
     * 是否启用并行计算 / Whether parallel computation is enabled
     */
    private static volatile boolean PARALLEL_ENABLED = true;

    /**
     * 线程池用于并行计算 / Thread pool for parallel computation
     */
    private static final ExecutorService THREAD_POOL
            = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

    // ========== 对象池化和内存管理方法 / Object Pooling and Memory Management Methods ==========
    /**
     * 从对象池获取向量实例 / Get vector instance from object pool
     */
    public static RereDoubleVector borrowVector(int length) {
        RereDoubleVector vector = VECTOR_POOL.poll();
        if (vector == null) {
            return new RereDoubleVector(length);
        }
        return vector.reset(length);
    }

    /**
     * 将向量返回到对象池 / Return vector to object pool
     */
    public void returnToPool() {
        if (this.data != null && this.data.length > 0) {
            VECTOR_POOL.offer(this);
        }
    }

    /**
     * 重置向量大小 / Reset vector size
     */
    private RereDoubleVector reset(int length) {
        // 由于data是final的，无法重置，直接返回新实例
        return new RereDoubleVector(length);
    }

    /**
     * 预分配内存的构造函数 / Constructor with pre-allocated memory
     */
    public RereDoubleVector(int length) {
        this.data = new double[length];
    }

    /**
     * 获取预分配缓冲区 / Get pre-allocated buffer
     */
    private static double[] getBuffer(int size) {
        double[] buffer = BUFFER.get();
        if (buffer.length < size) {
            return new double[size];
        }
        return buffer;
    }

    /**
     * 设置并行计算开关 / Set parallel computation switch
     */
    public static void setParallelEnabled(boolean enabled) {
        PARALLEL_ENABLED = enabled;
    }

    /**
     * 关闭线程池 / Shutdown thread pool
     */
    public static void shutdown() {
        THREAD_POOL.shutdown();
    }

    /**
     * 向量构造函数（Vector Constructor）
     *
     * <p>
     * 使用给定的Double数组创建向量实例。这是RereVector的主要构造函数，
     * 用于将一维Double数组包装为向量对象，提供丰富的向量运算功能。</p>
     *
     * <p>
     * 设计特点：</p>
     * <ul>
     * <li><strong>数据封装</strong>：将原始数组封装为向量对象，提供面向对象的接口</li>
     * <li><strong>不可变性</strong>：内部数据数组为final，确保向量创建后数据不变</li>
     * <li><strong>空值检查</strong>：严格检查输入参数，防止空指针异常</li>
     * <li><strong>性能优化</strong>：直接引用输入数组，避免不必要的内存拷贝</li>
     * </ul>
     *
     * <p>
     * 使用场景：</p>
     * <ul>
     * <li>从外部数据源创建向量（文件读取、网络传输等）</li>
     * <li>将计算结果转换为向量对象</li>
     * <li>与其他向量运算方法配合使用</li>
     * </ul>
     *
     * <p>
     * 注意事项：</p>
     * <ul>
     * <li>输入数组不应为null，否则抛出IllegalArgumentException</li>
     * <li>向量创建后，内部数据与输入数组共享引用</li>
     * <li>修改输入数组会影响向量数据，建议传入数组的副本</li>
     * </ul>
     *
     * @param data 一维Double数组，表示向量数据。数组长度决定向量维度
     * @throws IllegalArgumentException 当输入数据为null时抛出异常
     */
    public RereDoubleVector(double[] data) {
        // 参数验证：确保输入数据不为null
        if (data == null) {
            throw new IllegalArgumentException("向量数据不能为null / Vector data cannot be null");
        }
        // 直接引用输入数组，避免内存拷贝，提高性能
        this.data = data;
    }

    public RereDoubleVector(Double[] data) {
        // 参数验证：确保输入数据不为null
        if (data == null) {
            throw new IllegalArgumentException("向量数据不能为null / Vector data cannot be null");
        }
        // 直接引用输入数组，避免内存拷贝，提高性能
        this.data = RereMathUtil.toPrimitive(data);
    }

    /**
     * 向量减法运算（Vector Subtraction）
     *
     * <p>
     * 计算两个向量的元素级减法，即result[i] = this[i] - vec[i]。
     * 这是线性代数中的基本运算，广泛应用于向量计算、数值分析和机器学习等领域。</p>
     *
     * <p>
     * 数学定义：</p>
     * <ul>
     * <li>对于向量a = [a₁, a₂, ..., aₙ]和b = [b₁, b₂, ..., bₙ]</li>
     * <li>减法结果c = a - b = [a₁-b₁, a₂-b₂, ..., aₙ-bₙ]</li>
     * <li>要求两个向量具有相同的维度</li>
     * </ul>
     *
     * <p>
     * 性能优化策略：</p>
     * <ul>
     * <li><strong>GPU加速</strong>：对于大向量（超过GPU阈值），优先使用GPU并行计算</li>
     * <li><strong>CPU回退</strong>：GPU计算失败时自动回退到CPU计算</li>
     * <li><strong>内存优化</strong>：使用预分配的结果数组，避免频繁内存分配</li>
     * </ul>
     *
     * <p>
     * 算法复杂度：</p>
     * <ul>
     * <li>时间复杂度：O(n)，其中n是向量长度</li>
     * <li>空间复杂度：O(n)，需要存储结果向量</li>
     * <li>并行度：GPU版本可达到O(n/p)，其中p是并行处理器数量</li>
     * </ul>
     *
     * <p>
     * 应用场景：</p>
     * <ul>
     * <li>梯度下降算法中的参数更新</li>
     * <li>向量场计算中的差分运算</li>
     * <li>数值分析中的误差计算</li>
     * <li>机器学习中的损失函数计算</li>
     * </ul>
     *
     * @param vec 被减向量，必须与当前向量具有相同的长度
     * @return 新的向量对象，包含减法运算结果
     * @throws IllegalArgumentException 当输入向量为null或长度不匹配时抛出异常
     */
    @Override
    public IVector<Double> sub(IVector<Double> vec) {
        var res = this.computer.binaryOperate(data, vec.toDoubleArray(), IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        var vv = IDoubleVector.of(res);  // 创建结果向量对象
        return vv;
    }

    /**
     * 向量加法运算（Vector Addition）
     *
     * <p>
     * 计算两个向量的元素级加法，即result[i] = this[i] + vec[i]。
     * 这是线性代数中最基本的运算之一，是构建复杂数学运算的基础。</p>
     *
     * <p>
     * 数学定义：</p>
     * <ul>
     * <li>对于向量a = [a₁, a₂, ..., aₙ]和b = [b₁, b₂, ..., bₙ]</li>
     * <li>加法结果c = a + b = [a₁+b₁, a₂+b₂, ..., aₙ+bₙ]</li>
     * <li>满足交换律：a + b = b + a</li>
     * <li>满足结合律：(a + b) + c = a + (b + c)</li>
     * </ul>
     *
     * <p>
     * 性能优化策略：</p>
     * <ul>
     * <li><strong>GPU加速</strong>：大向量使用GPU并行计算，显著提升性能</li>
     * <li><strong>CPU优化</strong>：小向量使用优化的CPU算法，包括SIMD指令和缓存优化</li>
     * <li><strong>自适应选择</strong>：根据向量大小自动选择最优计算策略</li>
     * <li><strong>容错机制</strong>：GPU失败时自动回退到CPU计算</li>
     * </ul>
     *
     * <p>
     * 算法复杂度：</p>
     * <ul>
     * <li>时间复杂度：O(n)，其中n是向量长度</li>
     * <li>空间复杂度：O(n)，需要存储结果向量</li>
     * <li>并行度：GPU版本可达到O(n/p)，其中p是并行处理器数量</li>
     * </ul>
     *
     * <p>
     * 应用场景：</p>
     * <ul>
     * <li>线性组合计算：c₁v₁ + c₂v₂ + ... + cₙvₙ</li>
     * <li>神经网络中的前向传播</li>
     * <li>数值积分和微分方程求解</li>
     * <li>信号处理中的信号叠加</li>
     * </ul>
     *
     * @param vec 加数向量，必须与当前向量具有相同的长度
     * @return 新的向量对象，包含加法运算结果
     * @throws IllegalArgumentException 当输入向量为null或长度不匹配时抛出异常
     */
    @Override
    public IVector<Double> add(IVector<Double> vec) {
        var res = this.computer.binaryOperate(data, vec.toDoubleArray(), IDoubleVectorComputer.BinaryOperation.ADD);
        var vv = IDoubleVector.of(res);  // 创建结果向量对象
        return vv;
    }

    /**
     * 向量内积运算 / Vector inner product
     * <p>
     * 计算两个向量的内积（点积），要求两个向量长度相同 Computes the inner product (dot product) of two
     * vectors, requires both vectors to have the same length
     * </p>
     *
     * @param vec 另一个向量 / The other vector
     * @return 内积结果 / Inner product result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    @Override
    public Double innerProduct(IVector<Double> vec) {
        double res = this.computer.binaryReduceOperate(data, vec.toDoubleArray(), IDoubleVectorComputer.BinaryReduceOperation.DOT);
        return res;
    }

    @Override
    public Double dot(IVector<Double> vec) {
        return this.innerProduct(vec);
    }

    /**
     * 向量2范数 / Vector L2 norm
     * <p>
     * 计算向量的2范数（欧几里得范数），即所有元素平方和的开方 Calculates the L2 norm (Euclidean norm) of
     * the vector, which is the square root of the sum of squares
     * </p>
     *
     * @return 2范数 / L2 norm
     */
    @Override
    public Double norm2() {
        double[] squared = computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.POW, 2.0);
        return Math.sqrt(computer.reduceOperate(squared, IDoubleVectorComputer.ReduceOperation.SUM));
    }

    /**
     * 向量1范数 / Vector L1 norm
     * <p>
     * 计算向量的1范数（曼哈顿范数），即所有元素绝对值的和 Calculates the L1 norm (Manhattan norm) of the
     * vector, which is the sum of absolute values
     * </p>
     *
     * @return 1范数 / L1 norm
     */
    @Override
    public Double norm1() {
        double[] squared = computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.ABS, 0.0);
        return computer.reduceOperate(squared, IDoubleVectorComputer.ReduceOperation.SUM);
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
    @Override
    public IVector<Double> divideByScalar(Double p) {
        var res = this.computer.binaryOperate(data, p, IDoubleVectorComputer.BinaryOperation.DIVIDE);
        return IDoubleVector.of(res);
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
    @Override
    public double[] getData() {
        return this.data;
    }

    /**
     * 向量乘法运算（元素级） / Vector multiplication (element-wise)
     * <p>
     * 对应元素相乘，要求两个向量长度相同 Element-wise multiplication, requires both vectors to
     * have the same length
     * </p>
     *
     * @param vec1 另一个向量 / The other vector
     * @return 新的向量对象，包含乘法结果 / New vector object containing multiplication
     * result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    @Override
    public IVector<Double> multiply(IVector<Double> vec1) {
        var res = this.computer.binaryOperate(data, vec1.toDoubleArray(), IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        return IDoubleVector.of(res);
    }

    /**
     * 向量与矩阵的点积 / Vector-matrix dot product
     * <p>
     * 计算向量与矩阵的点积，向量长度必须与矩阵行数相等 Computes the dot product of vector and matrix,
     * vector length must equal matrix row count
     * </p>
     *
     * @param m 矩阵 / IDoubleMatrix
     * @return 结果矩阵 / Result matrix
     * @throws IllegalArgumentException 如果向量长度与矩阵行数不匹配 / if vector length
     * doesn't match matrix row count
     */
    @Override
    public IMatrix<Double> dot(IMatrix<Double> m) {
        IDoubleMatrix m0 = (IDoubleMatrix) m;
        double[][] mm = new double[m.getRowNum()][m.getColNum()];
        for (int i = 0; i < data.length; i++) {
            double w = data[i];
            double[] v = m0.getData()[i];
            for (int j = 0; j < v.length; j++) {
                v[j] = w * v[j];
            }
            mm[i] = v;
        }
        return IDoubleMatrix.of(mm);
    }

    /**
     * 行向量与矩阵相乘 / Row vector matrix multiplication
     * <p>
     * 计算行向量与矩阵的乘积，结果仍是行向量 Computes the product of row vector and matrix, result
     * is still a row vector
     * </p>
     * <p>
     * 数学公式：v * M = [v1, v2, ..., vn] * [[m11, m12, ...], [m21, m22, ...], ...]
     * 结果：[v1*m11 + v2*m21 + ..., v1*m12 + v2*m22 + ..., ...]
     * </p>
     *
     * @param matrix 矩阵 / Matrix
     * @return 结果行向量 / Result row vector
     * @throws IllegalArgumentException 如果向量长度与矩阵行数不匹配 / if vector length
     * doesn't match matrix row count
     * @throws NullPointerException 如果matrix为null / if matrix is null
     */
    @Override
    public IVector<Double> mmul(IMatrix<Double> matrix) {
        if (matrix == null) {
            throw new NullPointerException("矩阵不能为null / Matrix cannot be null");
        }

        IDoubleMatrix m = (IDoubleMatrix) matrix;
        int vectorLen = this.data.length;
        int matrixRows = m.getRowNum();

        if (vectorLen != matrixRows) {
            throw new IllegalArgumentException(
                    String.format("向量长度与矩阵行数不匹配: %d != %d / Vector length doesn't match matrix row count: %d != %d",
                            vectorLen, matrixRows, vectorLen, matrixRows));
        }

        var x1 = this.asRowMatrix().toDoubleArray();
        var x2 = matrix.toDoubleArray();
        var result = this.computer.mmul(x1, x2);
        return IDoubleVector.of(result[0]);
    }

    /**
     * 向量减标量 / Vector sub scalar
     * <p>
     * 向量中每个元素减去标量值 Subtracts a scalar value from each element in the vector
     * </p>
     *
     * @param p 标量值 / Scalar value
     * @return 新的向量对象，包含减法结果 / New vector object containing subtraction result
     */
    @Override
    public IVector<Double> subScalar(Double p) {
        var res = this.computer.binaryOperate(data, p, IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        return IDoubleVector.of(res);
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
    @Override
    public IVector<Double> addScalar(Double p) {
        var res = this.computer.binaryOperate(data, p, IDoubleVectorComputer.BinaryOperation.ADD);
        return IDoubleVector.of(res);
    }

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
    @Override
    public IVector<Double> multiplyScalar(Double p) {
        var res = this.computer.binaryOperate(data, p, IDoubleVectorComputer.BinaryOperation.MULTIPLY);
        return IDoubleVector.of(res);
    }

    /**
     * 向量元素求和 / Vector sum
     * <p>
     * 计算向量中所有元素的和 Calculates the sum of all elements in the vector
     * </p>
     *
     * @return 元素和 / Sum of elements
     */
    @Override
    public Double sum() {
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.SUM);
        return res;
    }

    /**
     * 向量最小值 / Vector minimum
     * <p>
     * 找到向量中的最小元素值 Finds the minimum element value in the vector
     * </p>
     *
     * @return 最小值 / Minimum value
     */
    @Override
    public Double min() {
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.MIN);
        return res;
    }

    /**
     * 向量最大值 / Vector maximum
     * <p>
     * 找到向量中的最大元素值 Finds the maximum element value in the vector
     * </p>
     *
     * @return 最大值 / Maximum value
     */
    @Override
    public Double max() {
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.MAX);
        return res;
    }

    /**
     * 最小值索引 / Index of minimum value
     * <p>
     * 返回向量中最小元素的索引位置 Returns the index position of the minimum element in the
     * vector
     * </p>
     *
     * @return 最小值的索引 / Index of minimum value
     */
    @Override
    public int argMin() {
        int len = this.data.length;
        double min = Double.MAX_VALUE;
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
     * 返回向量中最大元素的索引位置 Returns the index position of the maximum element in the
     * vector
     * </p>
     *
     * @return 最大值的索引 / Index of maximum value
     */
    @Override
    public int argMax() {
        int len = this.data.length;
        double max = Double.MIN_VALUE;
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
     * 计算向量中所有元素的平均值 Calculates the mean of all elements in the vector
     * </p>
     *
     * @return 平均值 / Mean value
     */
    @Override
    public Double mean() {
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.MEAN);
        return res;
    }

    /**
     * 向量标准差 / Vector standard deviation
     * <p>
     * 计算向量的标准差（除以N） Calculates the standard deviation of the vector (divided by
     * N)
     * </p>
     *
     * @return 标准差 / Standard deviation
     */
    @Override
    public Double std() {
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.STANDARD_DEVIATION);
        return res;
    }

    /**
     * 向量标准差（自由度修正） / Vector standard deviation (degrees of freedom correction)
     * <p>
     * 计算向量的标准差，使用自由度修正（除以N-ddof） Calculates the standard deviation of the
     * vector with degrees of freedom correction (divided by N-ddof)
     * </p>
     *
     * @param ddof 自由度修正值 / Degrees of freedom correction
     * @return 标准差 / Standard deviation
     */
    @Override
    public Double std(int ddof) {
        return Math.sqrt(this.var(ddof));
    }

    /**
     * 向量方差 / Vector variance
     * <p>
     * 计算向量的方差（除以N） Calculates the variance of the vector (divided by N)
     * </p>
     *
     * @return 方差 / Variance
     */
    @Override
    public Double var() {
        var res = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.VARIANCE);
        return res;
    }

    /**
     * 向量方差（自由度修正） / Vector variance (degrees of freedom correction)
     * <p>
     * 计算向量的方差，使用自由度修正（除以N-ddof） Calculates the variance of the vector with
     * degrees of freedom correction (divided by N-ddof)
     * </p>
     *
     * @param ddof 自由度修正值 / Degrees of freedom correction
     * @return 方差 / Variance
     */
    @Override
    public Double var(int ddof) {
        return this.subScalar(this.mean()).pow(2d).sum() / (double) (this.length() - ddof);
    }

    /**
     * 向量偏度 / Vector skewness
     * <p>
     * 计算向量的偏度，衡量数据分布的不对称性 偏度 = E[(X-μ)³] / σ³，其中μ是均值，σ是标准差 Calculates the
     * skewness of the vector, measuring the asymmetry of data distribution
     * Skewness = E[(X-μ)³] / σ³, where μ is mean and σ is standard deviation
     * </p>
     *
     * @return 偏度值 / Skewness value
     * @throws ArithmeticException 如果向量长度小于3或标准差为0 / if vector length is less
     * than 3 or standard deviation is 0
     */
    @Override
    public Double skewness() {
        if (this.length() < 3) {
            throw new ArithmeticException("向量长度必须大于等于3才能计算偏度 / Vector length must be at least 3 to calculate skewness");
        }

        double mean = this.mean();
        double std = this.std();

        if (std == 0) {
            throw new ArithmeticException("标准差为0，无法计算偏度 / Standard deviation is 0, cannot calculate skewness");
        }

        var diff = this.computer.binaryOperate(data, mean, IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        var d3 = this.computer.universalOperate(diff, IDoubleVectorComputer.UniversalOperation.POW, 3);
        double sum = this.computer.reduceOperate(d3, IDoubleVectorComputer.ReduceOperation.SUM);

        return sum / (this.length() * std * std * std);
    }

    /**
     * 向量峰度 / Vector kurtosis
     * <p>
     * 计算向量的峰度，衡量数据分布的尖锐程度 峰度 = E[(X-μ)⁴] / σ⁴ - 3，其中μ是均值，σ是标准差 减去3是为了使正态分布的峰度为0
     * Calculates the kurtosis of the vector, measuring the peakedness of data
     * distribution Kurtosis = E[(X-μ)⁴] / σ⁴ - 3, where μ is mean and σ is
     * standard deviation Subtracting 3 makes normal distribution kurtosis equal
     * to 0
     * </p>
     *
     * @return 峰度值 / Kurtosis value
     * @throws ArithmeticException 如果向量长度小于4或标准差为0 / if vector length is less
     * than 4 or standard deviation is 0
     */
    @Override
    public Double kurtosis() {
        if (this.length() < 4) {
            throw new ArithmeticException("向量长度必须大于等于4才能计算峰度 / Vector length must be at least 4 to calculate kurtosis");
        }

        double mean = this.mean();
        double std = this.std();

        if (std == 0) {
            throw new ArithmeticException("标准差为0，无法计算峰度 / Standard deviation is 0, cannot calculate kurtosis");
        }

        var diff = this.computer.binaryOperate(data, mean, IDoubleVectorComputer.BinaryOperation.SUBTRACT);
        var d3 = this.computer.universalOperate(diff, IDoubleVectorComputer.UniversalOperation.POW, 4);
        double sum = this.computer.reduceOperate(d3, IDoubleVectorComputer.ReduceOperation.SUM);

        return (sum / (this.length() * std * std * std * std)) - 3.0f;
    }

    /**
     * 获取向量长度 / Get vector length
     * <p>
     * 返回向量的长度（元素个数） Returns the length (number of elements) of the vector
     * </p>
     *
     * @return 向量长度 / IVector<Double> length
     */
    @Override
    public int length() {
        return data.length;
    }

    @Override
    public int size() {
        return this.length();
    }

    /**
     * 获取指定位置的元素 / Get element at specified position
     * <p>
     * 返回向量中指定位置的元素值，支持负数索引 Returns the element value at the specified position
     * in the vector, supports negative indexing
     * </p>
     *
     * @param position 位置索引（从0开始，支持负数索引） / Position index (0-based, supports
     * negative indexing)
     * @return 指定位置的元素值 / Element value at the specified position
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position index is out
     * of bounds
     */
    @Override
    public Double get(int position) {
        int actualPosition = position;
        if (position < 0) {
            actualPosition = data.length + position;
        }
        if (actualPosition < 0 || actualPosition >= data.length) {
            throw new IndexOutOfBoundsException("位置索引超出范围: " + position + " / Position index out of bounds: " + position);
        }
        return data[actualPosition];
    }

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
    @Override
    public void set(int position, Double value) {
        if (position < 0 || position >= data.length) {
            throw new IndexOutOfBoundsException("位置索引超出范围: " + position + " / Position index out of bounds: " + position);
        }
        data[position] = value;
    }

    /**
     * 向量切片（到指定结束位置） / Vector slice (to specified end position)
     * <p>
     * 返回从开始到指定结束位置的向量切片 Returns a vector slice from start to the specified end
     * position
     * </p>
     *
     * @param start 开始位置（包含） / Start position (inclusive)
     * @return 切片向量 / Sliced vector
     * @throws IndexOutOfBoundsException 如果结束位置超出范围 / if end position is out of
     * bounds
     */
    @Override
    public IVector<Double> slice(int start) {
        return this.slice(start, this.length(), 1);
    }

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
    @Override
    public IVector<Double> slice(int start, int end) {
        return this.slice(start, end, 1);
    }

    /**
     * 向量切片（指定开始、结束位置和步长） / Vector slice (specified start, end positions and
     * step)
     * <p>
     * 返回从指定开始位置到结束位置、指定步长的向量切片，支持负数索引 Returns a vector slice from specified
     * start position to end position with specified step, supports negative
     * indexing
     * </p>
     *
     * @param start 开始位置（支持负数索引） / Start position (supports negative indexing)
     * @param end 结束位置（不包含，支持负数索引） / End position (exclusive, supports negative
     * indexing)
     * @param step 步长 / Step size
     * @return 切片向量 / Sliced vector
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are
     * out of bounds
     * @throws IllegalArgumentException 如果步长小于等于0 / if step is less than or
     * equal to 0
     */
    @Override
    public IVector<Double> slice(int start, int end, int step) {
        // 处理负数索引
        if (start < 0) {
            start = data.length + start;
        }
        if (end < 0) {
            end = data.length + end;
        }

        // 验证参数
        if (step <= 0) {
            throw new IllegalArgumentException("步长必须大于0: " + step + " / Step must be greater than 0: " + step);
        }
        if (start < 0 || start >= data.length) {
            throw new IndexOutOfBoundsException("起始位置超出范围: " + start + " / Start position out of bounds: " + start);
        }
        if (end < 0 || end > data.length) {
            throw new IndexOutOfBoundsException("结束位置超出范围: " + end + " / End position out of bounds: " + end);
        }

        if (start >= end) {
            return IDoubleVector.of(new double[0]); // 返回空向量
        }

        int[] inds = IDoubleVector.range(start, end, step).toIntArray();
        double[] v = new double[inds.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[inds[i]];
        }
        return IDoubleVector.of(v);
    }

    /**
     * 向量切片（字符串表达式） / Vector slice (string expression)
     * <p>
     * 根据切片表达式对向量进行切片操作，支持负数索引 Performs vector slicing based on slice
     * expression, supports negative indexing
     * </p>
     *
     * @param sliceExpression 切片表达式，如 "1:3", ":-1", "::2" / Slice expression,
     * e.g. "1:3", ":-1", "::2"
     * @return 切片向量 / Sliced vector
     * @throws IllegalArgumentException 如果切片表达式无效 / if slice expression is
     * invalid
     */
    @Override
    public IVector<Double> slice(String sliceExpression) {
        if (sliceExpression == null || sliceExpression.trim().isEmpty()) {
            throw new IllegalArgumentException("切片表达式不能为空 / Slice expression cannot be empty");
        }

        // 使用统一的切片表达式解析器
        SliceExpressionParser.SliceResult result = SliceExpressionParser.parse(sliceExpression, data.length);

        if (result.actualStart >= result.actualEnd) {
            return IDoubleVector.of(new double[0]);
        }

        return slice(result.actualStart, result.actualEnd, result.step);
    }

    /**
     * 花式索引 / Fancy indexing
     * <p>
     * 根据给定的位置数组获取对应位置的元素组成新向量，支持负数索引 Gets elements at specified positions to
     * form a new vector, supports negative indexing
     * </p>
     *
     * @param positions 位置索引数组（支持负数索引） / Array of position indices (supports
     * negative indexing)
     * @return 新的向量对象，包含指定位置的元素 / New vector object containing elements at
     * specified positions
     * @throws IndexOutOfBoundsException 如果任何位置索引超出范围 / if any position index is
     * out of bounds
     */
    @Override
    public IVector<Double> fancyGet(int[] positions) {
        double[] v = new double[positions.length];
        for (int i = 0; i < v.length; i++) {
            int actualPosition = positions[i];
            if (actualPosition < 0) {
                actualPosition = data.length + actualPosition;
            }
            if (actualPosition < 0 || actualPosition >= data.length) {
                throw new IndexOutOfBoundsException("位置索引超出范围: " + positions[i] + " / Position index out of bounds: " + positions[i]);
            }
            v[i] = data[actualPosition];
        }
        return IDoubleVector.of(v);
    }

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
    @Override
    public IVector<Double> booleanGet(boolean[] booleanIndex) {
        if (booleanIndex == null) {
            throw new IllegalArgumentException("布尔索引数组不能为null / Boolean index array cannot be null");
        }
        if (booleanIndex.length != this.data.length) {
            throw new IllegalArgumentException("布尔索引数组长度与向量长度不匹配: " + booleanIndex.length + " != " + this.data.length
                    + " / Boolean index array length doesn't match vector length: " + booleanIndex.length + " != " + this.data.length);
        }

        List<Double> ls = new ArrayList<>();
        for (int i = 0; i < booleanIndex.length; i++) {
            if (booleanIndex[i]) {
                ls.add(data[i]);
            }
        }
        Double[] v = ls.toArray(Double[]::new);
        return IDoubleVector.of(v);
    }



    /**
     * 范围设置值（带步长） / Range set values (with step)
     * <p>
     * 设置指定范围内、指定步长位置的元素值，支持负数索引 Sets element values at positions within
     * specified range with specified step, supports negative indexing
     * </p>
     *
     * @param start 开始位置（支持负数索引） / Start position (supports negative indexing)
     * @param end 结束位置（不包含，支持负数索引） / End position (exclusive, supports negative
     * indexing)
     * @param step 步长 / Step size
     * @param values 要设置的值数组 / Array of values to set
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are
     * out of bounds
     * @throws IllegalArgumentException 如果值数组长度不匹配 / if values array length
     * doesn't match
     */
    @Override
    public void setFromTo(int start, int end, int step, Double[] values) {
        // 处理负数索引
        int actualStart = start;
        int actualEnd = end;
        if (start < 0) {
            actualStart = data.length + start;
        }
        if (end < 0) {
            actualEnd = data.length + end;
        }

        int[] inds = IDoubleVector.range(actualStart, actualEnd, step).toIntArray();
        for (int i = 0; i < inds.length; i++) {
            if (inds[i] < 0 || inds[i] >= data.length) {
                throw new IndexOutOfBoundsException("位置索引超出范围: " + inds[i] + " / Position index out of bounds: " + inds[i]);
            }
            data[inds[i]] = values[i];
        }
    }

    /**
     * 范围设置值 / Range set values
     * <p>
     * 设置指定范围内的元素值（步长为1），支持负数索引 Sets element values within specified range (step
     * size 1), supports negative indexing
     * </p>
     *
     * @param start 开始位置（支持负数索引） / Start position (supports negative indexing)
     * @param end 结束位置（不包含，支持负数索引） / End position (exclusive, supports negative
     * indexing)
     * @param values 要设置的值数组 / Array of values to set
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position indices are
     * out of bounds
     * @throws IllegalArgumentException 如果值数组长度不匹配 / if values array length
     * doesn't match
     */
    @Override
    public void setFromTo(int start, int end, Double[] values) {
        this.setFromTo(start, end, 1, values);
    }

    /**
     * 向量相等比较 / Vector equality comparison
     * <p>
     * 对应元素进行相等比较，返回布尔数组 Element-wise equality comparison, returns boolean array
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示对应位置元素是否相等 / Boolean array indicating whether
     * corresponding elements are equal
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    @Override
    public boolean[] eq(IVector<Double> other) {
        var v = this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.EQUALS);
        return v;
    }

    /**
     * 向量小于比较 / Vector less-than comparison
     * <p>
     * 对应元素进行小于比较，返回布尔数组 Element-wise less-than comparison, returns boolean
     * array
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示对应位置元素是否小于 / Boolean array indicating whether
     * corresponding elements are less than
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    @Override
    public boolean[] lt(IVector<Double> other) {

        var v = this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.LESS_THAN);

        return v;
    }

    /**
     * 向量大于比较 / Vector greater-than comparison
     * <p>
     * 对应元素进行大于比较，返回布尔数组 Element-wise greater-than comparison, returns boolean
     * array
     * </p>
     *
     * @param other 另一个向量 / The other vector
     * @return 布尔数组，表示对应位置元素是否大于 / Boolean array indicating whether
     * corresponding elements are greater than
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    @Override
    public boolean[] gt(IVector<Double> other) {
        var v = this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.GREATER_THAN);
        return v;
    }
    
    @Override
    public boolean[] ge(IVector<Double> other) {
        var v = this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.GREATER_THAN_OR_EQUALS);
        return v;
    }
    
    @Override
    public boolean[] le(IVector<Double> other) {
        var v = this.computer.logicalCompare(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalCompare.LESS_THAN_OR_EQUALS);
        return v;
    }

    /**
     * 向量元素乘积 / Vector product
     * <p>
     * 计算向量中所有元素的乘积 Calculates the product of all elements in the vector
     * </p>
     *
     * @return 元素乘积 / Product of elements
     */
    @Override
    public Double prod() {
        var v = this.computer.reduceOperate(data, IDoubleVectorComputer.ReduceOperation.PROD);
        return v;
    }

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
    @Override
    public IVector<Double> clip(Double lower, Double upper) {
        if (lower > upper) {
            throw new IllegalArgumentException("下界不能大于上界: " + lower + " > " + upper
                    + " / Lower bound cannot be greater than upper bound: " + lower + " > " + upper);
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
     * 计算向量的峰峰值（最大值减最小值） Calculates the peak-to-peak value of the vector (max -
     * min)
     * </p>
     *
     * @return 峰峰值 / Peak-to-peak value
     */
    @Override
    public Double ptp() {
        return this.max() - this.min();
    }

    /**
     * 向量绝对值 / Vector absolute value
     * <p>
     * 对向量中每个元素取绝对值 Takes the absolute value of each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含绝对值结果 / New vector object containing absolute value
     * results
     */
    @Override
    public IVector<Double> abs() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.ABS, 0);
        return IDoubleVector.of(v1);
    }

    /**
     * 向量填充 / Vector fill
     * <p>
     * 将向量中所有元素设置为指定值 Sets all elements in the vector to the specified value
     * </p>
     *
     * @param value 填充值 / Fill value
     */
    @Override
    public void fill(Double value) {
        for (int i = 0; i < this.length(); i++) {
            data[i] = value;
        }
    }

    /**
     * 向量排序 / Vector sorting
     * <p>
     * 对向量中的元素进行升序排序 Sorts the elements in the vector in ascending order
     * </p>
     *
     * @return 排序后的向量（就地操作） / Sorted vector (in-place operation)
     */
    @Override
    public IVector<Double> sort() {
        Arrays.sort(data);
        return this;
    }

    /**
     * 向量反转 / Vector reverse
     * <p>
     * 反转向量中元素的顺序 Reverses the order of elements in the vector
     * </p>
     *
     * @return 反转后的向量（就地操作） / Reversed vector (in-place operation)
     */
    @Override
    public IVector<Double> reverse() {
        for (int i = 0; i < data.length / 2; i++) {
            double temp = data[i];
            data[i] = data[data.length - i - 1];
            data[data.length - i - 1] = temp;
        }
        return this;
    }

    /**
     * 向量复制 / Vector copy
     * <p>
     * 创建向量的深拷贝 Creates a deep copy of the vector
     * </p>
     *
     * @return 新的向量对象，与原向量内容相同 / New vector object with the same content as the
     * original
     */
    @Override
    public IVector<Double> copy() {
        double[] v = new double[this.length()];
        System.arraycopy(data, 0, v, 0, this.length());
        return IDoubleVector.of(v);
    }

    /**
     * 向量开方 / Vector square root
     * <p>
     * 对向量中每个元素进行开方运算 Performs square root operation on each element in the
     * vector
     * </p>
     *
     * @return 新的向量对象，包含开方结果 / New vector object containing square root results
     * @throws ArithmeticException 如果元素值为负数 / if any element value is negative
     */
    @Override
    public IVector<Double> sqrt() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SQRT, 0);
        return IDoubleVector.of(v1);
    }

    /**
     * 向量平方 / Vector square
     * <p>
     * 对向量中每个元素进行平方运算 Performs square operation on each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含平方结果 / New vector object containing square results
     */
    @Override
    public IVector<Double> square() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.POW, 2);
        return IDoubleVector.of(v1);
    }

    /**
     * 向量指数运算 / Vector exponential
     * <p>
     * 对向量中每个元素进行指数运算（e^x） Performs exponential operation (e^x) on each element
     * in the vector
     * </p>
     *
     * @return 新的向量对象，包含指数运算结果 / New vector object containing exponential
     * results
     */
    @Override
    public IVector<Double> exp() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.EXP, 0);
        return IDoubleVector.of(v1);
    }

    /**
     * 向量自然对数 / Vector natural logarithm
     * <p>
     * 对向量中每个元素进行自然对数运算（ln(x)） Performs natural logarithm operation (ln(x)) on
     * each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含对数运算结果 / New vector object containing logarithm results
     * @throws ArithmeticException 如果元素值小于等于0 / if any element value is less
     * than or equal to 0
     */
    @Override
    public IVector<Double> log() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.LOG, 0);
        return IDoubleVector.of(v1);
    }

    /**
     * 向量以10为底的对数 / Vector base-10 logarithm
     * <p>
     * 对向量中每个元素进行以10为底的对数运算（log10(x)） Performs base-10 logarithm operation
     * (log10(x)) on each element in the vector
     * </p>
     *
     * @return 新的向量对象，包含对数运算结果 / New vector object containing logarithm results
     * @throws ArithmeticException 如果元素值小于等于0 / if any element value is less
     * than or equal to 0
     */
    @Override
    public IVector<Double> log10() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.LOG10, 0);
        return IDoubleVector.of(v1);
    }

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
     */
    @Override
    public IVector<Double> pow(Double m) {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.POW, m);
        return IDoubleVector.of(v1);
    }

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
    @Override
    public IVector<Double> remainder(Double value) {
        var data2 = this.computer.binaryOperate(data, value, IDoubleVectorComputer.BinaryOperation.REMAINDER);
        return IDoubleVector.of(data2);
    }

    /**
     * 转换为整数数组 / Convert to integer array
     * <p>
     * 将向量转换为整数数组 Converts the vector to an integer array
     * </p>
     *
     * @return 整数数组 / Integer array
     */
    @Override
    public int[] toIntArray() {
        return RereMathUtil.doubleToInt(data);
    }

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
    @Override
    public Double euclideanDistance(IVector<Double> other) {
        var v1 = this.computer.binaryReduceOperate(data, other.toDoubleArray(), IDoubleVectorComputer.BinaryReduceOperation.L2_NORM);
        return v1;
    }

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
    @Override
    public Double manhattanDistance(IVector<Double> other) {
        var v1 = this.computer.binaryReduceOperate(data, other.toDoubleArray(), IDoubleVectorComputer.BinaryReduceOperation.L1_NORM);
        return v1;
    }

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
    @Override
    public Double cosineSimilarity(IVector<Double> other) {
        var dotProduct = this.computer.binaryReduceOperate(data, other.toDoubleArray(), IDoubleVectorComputer.BinaryReduceOperation.DOT);
        var norm1 = this.norm2();
        var norm2 = other.norm2();
        return dotProduct / (norm1 * norm2);
    }

    @Override
    public String toString() {
        var ls = IntStream.range(0, this.data.length)
                .mapToObj(i -> String.format("Value: %.6f", this.data[i]))
                .toArray(String[]::new);
        return StringUtils.join(ls, ", ");
    }

    // ========== 三角函数操作实现 / Trigonometric Functions Implementation ==========
    @Override
    public IVector<Double> sin() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SIN, 0);
        return IDoubleVector.of(v1);
    }

    @Override
    public IVector<Double> cos() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.COS, 0);
        return IDoubleVector.of(v1);
    }

    @Override
    public IVector<Double> tan() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.TAN, 0);
        return IDoubleVector.of(v1);
    }

    @Override
    public IVector<Double> arcsin() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.ASIN, 0);
        return IDoubleVector.of(v1);
    }

    @Override
    public IVector<Double> arccos() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.ACOS, 0);
        return IDoubleVector.of(v1);
    }

    @Override
    public IVector<Double> arctan() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.ATAN, 0);
        return IDoubleVector.of(v1);
    }

    // ========== 双曲函数实现 / Hyperbolic Functions Implementation ==========
    @Override
    public IVector<Double> sinh() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.SINH, 0);
        return IDoubleVector.of(v1);
    }

    @Override
    public IVector<Double> cosh() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.COSH, 0);
        return IDoubleVector.of(v1);
    }

    @Override
    public IVector<Double> tanh() {
        var v1 = this.computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.COSH, 0);
        return IDoubleVector.of(v1);
    }

    // ========== 舍入函数实现 / Rounding Functions Implementation ==========
    @Override
    public IVector<Double> round() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = Math.round(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> floor() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = Math.floor(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> ceil() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = Math.ceil(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> trunc() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = Math.rint(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    // ========== 逻辑运算实现 / Logical Operations Implementation ==========
    @Override
    public boolean[] logicalAnd(IVector<Double> other) {
        var result = this.computer.logicalOperate(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalOperation.AND);
        return result;
    }

    @Override
    public boolean[] logicalOr(IVector<Double> other) {
        var result = this.computer.logicalOperate(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalOperation.OR);
        return result;
    }

    @Override
    public boolean[] logicalNot() {
        var result = this.computer.logicalOperate(data, IDoubleVectorComputer.LogicalOperation.NOT);
        return result;
    }

    @Override
    public boolean[] logicalXor(IVector<Double> other) {
        var result = this.computer.logicalOperate(data, other.toDoubleArray(), IDoubleVectorComputer.LogicalOperation.XOR);
        return result;
    }

    // ========== 累积操作实现 / Cumulative Operations Implementation ==========
    @Override
    public IVector<Double> cumsum() {
        double[] result = new double[this.data.length];
        double sum = 0.0;
        for (int i = 0; i < this.data.length; i++) {
            sum += this.data[i];
            result[i] = sum;
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> cumprod() {
        double[] result = new double[this.data.length];
        double product = 1.0;
        for (int i = 0; i < this.data.length; i++) {
            product *= this.data[i];
            result[i] = product;
        }
        return IDoubleVector.of(result);
    }

    // ========== 差分操作实现 / Difference Operations Implementation ==========
    @Override
    public IVector<Double> diff() {
        return diff(1);
    }

    @Override
    public IVector<Double> diff(int n) {
        var res = this.computer.diff(data, n);
        return IDoubleVector.of(res);
    }

    // ========== 条件操作实现 / Conditional Operations Implementation ==========
    @Override
    public IVector<Double> where(boolean[] condition, Double x, Double y) {
        if (condition == null) {
            throw new IllegalArgumentException("条件数组不能为null / Condition array cannot be null");
        }
        if (condition.length != this.data.length) {
            throw new IllegalArgumentException("条件数组长度与向量长度不匹配: " + condition.length + " != " + this.data.length
                    + " / Condition array length doesn't match vector length: " + condition.length + " != " + this.data.length);
        }

        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = condition[i] ? x : y;
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> where(boolean[] condition, IVector<Double> x, IVector<Double> y) {
        if (condition == null) {
            throw new IllegalArgumentException("条件数组不能为null / Condition array cannot be null");
        }
        if (x == null || y == null) {
            throw new IllegalArgumentException("值向量不能为null / Value vectors cannot be null");
        }
        if (condition.length != this.data.length || x.length() != this.data.length || y.length() != this.data.length) {
            throw new IllegalArgumentException("向量长度不匹配 / Vector lengths don't match");
        }

        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = condition[i] ? x.get(i) : y.get(i);
        }
        return IDoubleVector.of(result);
    }

    // ========== 重复和连接操作实现 / Repeat and Concatenation Operations Implementation ==========
    @Override
    public IVector<Double> repeat(int repeats) {
        if (repeats < 1) {
            throw new IllegalArgumentException("重复次数必须大于等于1: " + repeats + " / Repeat count must be greater than or equal to 1: " + repeats);
        }

        double[] result = new double[this.data.length * repeats];
        for (int i = 0; i < this.data.length; i++) {
            for (int j = 0; j < repeats; j++) {
                result[i * repeats + j] = this.data[i];
            }
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> tile(int reps) {
        if (reps < 1) {
            throw new IllegalArgumentException("重复次数必须大于等于1: " + reps + " / Repeat count must be greater than or equal to 1: " + reps);
        }

        double[] result = new double[this.data.length * reps];
        for (int i = 0; i < reps; i++) {
            System.arraycopy(this.data, 0, result, i * this.data.length, this.data.length);
        }
        return IDoubleVector.of(result);
    }

    // ========== 统计扩展操作实现 / Extended Statistical Operations Implementation ==========
    @Override
    public Double median() {
        if (this.data.length == 0) {
            throw new ArithmeticException("空向量无法计算中位数 / Cannot compute median for empty vector");
        }

        double[] sorted = this.data.clone();
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
    public Double percentile(double q) {
        if (q < 0.0f || q > 100.0f) {
            throw new IllegalArgumentException("百分位数必须在[0,100]范围内: " + q + " / Percentile must be in range [0,100]: " + q);
        }
        if (this.data.length == 0) {
            throw new ArithmeticException("空向量无法计算百分位数 / Cannot compute percentile for empty vector");
        }

        double[] sorted = this.data.clone();
        Arrays.sort(sorted);

        if (q == 0.0f) {
            return sorted[0];
        }
        if (q == 100.0f) {
            return sorted[sorted.length - 1];
        }

        double index = (q / 100.0f) * (sorted.length - 1);
        int lowerIndex = (int) Math.floor(index);
        int upperIndex = (int) Math.ceil(index);

        if (lowerIndex == upperIndex) {
            return sorted[lowerIndex];
        }

        double weight = index - lowerIndex;
        return sorted[lowerIndex] * (1.0f - weight) + sorted[upperIndex] * weight;
    }

    @Override
    public Double mode() {
        if (this.data.length == 0) {
            throw new ArithmeticException("空向量无法计算众数 / Cannot compute mode for empty vector");
        }

        // 使用HashMap统计频率 / Use HashMap to count frequency
        java.util.Map<Double, Integer> frequency = new java.util.HashMap<>();
        for (double value : this.data) {
            frequency.put(value, frequency.getOrDefault(value, 0) + 1);
        }

        double mode = this.data[0];
        int maxFreq = 1;

        for (java.util.Map.Entry<Double, Integer> entry : frequency.entrySet()) {
            if (entry.getValue() > maxFreq) {
                maxFreq = entry.getValue();
                mode = entry.getKey();
            }
        }

        return mode;
    }

    // ========== 线性代数扩展操作实现 / Extended Linear Algebra Operations Implementation ==========
    @Override
    public Double norm(Double p) {
        if (p < 1.0f) {
            throw new IllegalArgumentException("范数阶数必须大于等于1: " + p + " / Norm order must be greater than or equal to 1: " + p);
        }

        if (p == 1.0) {
            return this.norm1();
        }
        if (p == 2.0) {
            return this.norm2();
        }
        if (Double.isInfinite(p)) {
            return this.normInf();
        }

        double[] pd = computer.universalOperate(data, IDoubleVectorComputer.UniversalOperation.POW, p);
        double sum = computer.reduceOperate(pd, IDoubleVectorComputer.ReduceOperation.SUM);

        return Math.pow(sum, 1.0 / p);
    }

    @Override
    public Double normInf() {
        double maxAbs = 0.0;
        for (int i = 0; i < this.data.length; i++) {
            double abs = Math.abs(this.data[i]);
            if (abs > maxAbs) {
                maxAbs = abs;
            }
        }
        return maxAbs;
    }

    @Override
    public IVector<Double> normalize() {
        double norm = this.norm2();
        if (norm == 0.0f) {
            throw new ArithmeticException("向量L2范数为零，无法归一化 / Vector L2 norm is zero, cannot normalize");
        }
        return this.divideByScalar(norm);
    }

    @Override
    public IVector<Double> reciprocal() {
        return this.pow(-1.0);
    }

    @Override
    public IDoubleMatrix outer(IVector<Double> other) {
        var dd = this.computer.outer(data, other.toDoubleArray());
        return IDoubleMatrix.of(dd);
    }

    @Override
    public double[] toDoubleArray() {
        return this.data;
    }

    /**
     * 转换为双精度数组 / Convert to double array
     * <p>
     * 将向量转换为双精度数组 Converts the vector to a double array
     * </p>
     *
     * @return 双精度数组 / Double array
     */
    @Override
    public float[] toFloatArray() {
        return RereMathUtil.doubleToFloat(data);
    }

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
    @Override
    public IMatrix<Double> asColumnVector() {
        var res = this.computer.transpose(data);
        return new RereDoubleMatrix(res);
    }

    /**
     * 动态时间规整（Dynamic Time Warping）算法
     * <p>
     * 计算两个时间序列之间的DTW距离，用于衡量时间序列的相似性。 DTW算法能够处理不同长度的时间序列，并找到最优的对齐路径。
     * </p>
     * <p>
     * 算法原理：
     * <ul>
     * <li>构建距离矩阵：计算两个序列中每对元素之间的距离</li>
     * <li>动态规划：使用DP算法找到从起点到终点的最小累积距离路径</li>
     * <li>路径约束：通常使用Sakoe-Chiba band或Itakura parallelogram约束搜索空间</li>
     * </ul>
     * </p>
     * <p>
     * 时间复杂度：O(m×n)，其中m和n是两个序列的长度 空间复杂度：O(m×n)，用于存储距离矩阵
     * </p>
     *
     * @param other 另一个时间序列向量
     * @return DTW距离值，值越小表示序列越相似
     * @throws IllegalArgumentException 如果输入向量为null
     */
    @Override
    public Double dtw(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }

        int m = this.data.length;
        int n = other.length();

        // 如果任一序列为空，返回无穷大距离
        if (m == 0 || n == 0) {
            return Double.POSITIVE_INFINITY;
        }

        // 如果两个序列完全相同，直接返回0
        if (m == n) {
            boolean identical = true;
            for (int i = 0; i < m; i++) {
                if (Math.abs(this.data[i] - other.get(i)) > 1e-10) {
                    identical = false;
                    break;
                }
            }
            if (identical) {
                return 0.0;
            }
        }

        // 创建距离矩阵，使用动态规划
        double[][] dtwMatrix = new double[m + 1][n + 1];

        // 初始化边界条件
        for (int i = 0; i <= m; i++) {
            dtwMatrix[i][0] = Double.POSITIVE_INFINITY;
        }
        for (int j = 0; j <= n; j++) {
            dtwMatrix[0][j] = Double.POSITIVE_INFINITY;
        }
        dtwMatrix[0][0] = 0.0;

        // 填充DTW矩阵
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // 计算当前元素对的距离
                double distance = Math.abs(this.data[i - 1] - other.get(j - 1));

                // 取三个方向的最小值：左、上、左上
                double minPrev = Math.min(
                        Math.min(dtwMatrix[i - 1][j], dtwMatrix[i][j - 1]),
                        dtwMatrix[i - 1][j - 1]
                );

                dtwMatrix[i][j] = distance + minPrev;
            }
        }

        return dtwMatrix[m][n];
    }

    /**
     * 计算与另一个向量的皮尔逊相关系数 / Compute Pearson correlation coefficient with another
     * vector
     * <p>
     * 计算当前向量与另一个向量之间的皮尔逊相关系数，衡量两个向量的线性相关性。 相关系数的取值范围为[-1,
     * 1]，其中1表示完全正相关，-1表示完全负相关，0表示无线性相关。
     * </p>
     * <p>
     * 数学公式：r = cov(X,Y) / (σX × σY)
     * <ul>
     * <li>cov(X,Y) 是两个向量的协方差</li>
     * <li>σX 和 σY 分别是两个向量的标准差</li>
     * </ul>
     * </p>
     *
     * @param other 另一个向量，必须与当前向量长度相同
     * @return 皮尔逊相关系数，取值范围[-1, 1]
     * @throws IllegalArgumentException 如果输入向量为null或长度不匹配
     * @throws ArithmeticException 如果任一向量的标准差为0（无法计算相关系数）
     */
    @Override
    public Double corr(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + other.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + other.length());
        }

        int len = this.data.length;

        // 如果向量长度为0或1，无法计算相关系数
        if (len < 2) {
            throw new ArithmeticException("向量长度必须大于等于2才能计算相关系数 / Vector length must be at least 2 to calculate correlation");
        }

        // 计算协方差：调用cov方法
        double covariance = this.cov(other);

        // 计算标准差
        double stdX = this.std();
        double stdY = other.std();

        // 检查分母是否为零
        if (stdX == 0.0 || stdY == 0.0) {
            throw new ArithmeticException("向量标准差为0，无法计算相关系数 / Vector standard deviation is 0, cannot calculate correlation");
        }

        // 计算相关系数
        return covariance / (stdX * stdY);
    }

    /**
     * 计算与另一个向量的协方差 / Compute covariance with another vector
     * <p>
     * 计算当前向量与另一个向量之间的协方差，衡量两个向量的线性相关性。 协方差的计算公式为：cov(X,Y) = E[(X-μX)(Y-μY)] =
     * E[XY] - μXμY
     * </p>
     * <p>
     * 协方差的性质：
     * <ul>
     * <li>cov(X,Y) > 0: 正相关，X增大时Y倾向于增大</li>
     * <li>cov(X,Y) < 0: 负相关，X增大时Y倾向于减小</li> <li>cov(X,Y ) = 0: 无线性相关</li>
     * <li>cov(X,X) = var(X): 自协方差等于方差</li>
     * </ul>
     * </p>
     *
     * @param other 另一个向量，必须与当前向量长度相同
     * @return 协方差值
     * @throws IllegalArgumentException 如果输入向量为null或长度不匹配
     * @throws ArithmeticException 如果向量长度为0（无法计算协方差）
     */
    @Override
    public Double cov(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + other.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + other.length());
        }

        int len = this.data.length;

        // 如果向量长度为0，无法计算协方差
        if (len == 0) {
            throw new ArithmeticException("向量长度为0，无法计算协方差 / Vector length is 0, cannot calculate covariance");
        }

        // 如果向量长度为1，协方差为0（只有一个点无法计算协方差）
        if (len == 1) {
            return 0.0;
        }

        // 计算协方差：使用中心化向量的内积
        IVector<Double> centeredX = this.subScalar(this.mean());
        IVector<Double> centeredY = other.subScalar(other.mean());
        double covariance = centeredX.innerProduct(centeredY) / len;

        return covariance;
    }

    @Override
    public IVector<Double> apply(Function<Double, Double> fun) {
        if (fun == null) {
            throw new IllegalArgumentException("函数不能为null / Function cannot be null");
        }

        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = fun.apply(data[i]);
        }

        return new RereDoubleVector(result);
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
    @Override
    public Double q1() {
        if (this.data.length == 0) {
            throw new ArithmeticException("空向量无法计算第一四分位数 / Cannot compute first quartile for empty vector");
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
    @Override
    public Double q3() {
        if (this.data.length == 0) {
            throw new ArithmeticException("空向量无法计算第三四分位数 / Cannot compute third quartile for empty vector");
        }

        // 使用已有的percentile方法计算75%分位数
        return this.percentile(75.0);
    }

    @Override
    public IVector<Double> concat(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }

        int thisLen = this.data.length;
        int otherLen = other.length();
        double[] result = new double[thisLen + otherLen];

        // 复制当前向量的数据
        System.arraycopy(this.data, 0, result, 0, thisLen);

        // 复制另一个向量的数据
        for (int i = 0; i < otherLen; i++) {
            result[thisLen + i] = other.get(i);
        }

        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> sign() {
        var res = this.computer.sign(data);
        return IDoubleVector.of(res);

    }

    @Override
    public IMatrix<Double> reshape(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("行数和列数必须大于0: rows=" + rows + ", cols=" + cols + " / Rows and columns must be greater than 0: rows=" + rows + ", cols=" + cols);
        }

        int totalElements = rows * cols;
        if (totalElements != this.data.length) {
            throw new IllegalArgumentException("重塑后的元素总数必须等于原向量长度: " + totalElements + " != " + this.data.length + " / Reshaped total elements must equal original vector length: " + totalElements + " != " + this.data.length);
        }

        double[][] result = new double[rows][cols];

        // 简单循环通常足够高效
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data, i * cols, result[i], 0, cols);
        }

        return IDoubleMatrix.of(result);
    }

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
     * <h3>性能说明 / Performance Notes:</h3>
     * <ul>
     * <li>小向量（≤100元素）：直接顺序计算</li>
     * <li>大向量（>100元素）：分块处理以提高缓存命中率</li>
     * <li>时间复杂度：O(n)，空间复杂度：O(n²)</li>
     * </ul>
     *
     * @return 对角Hessian矩阵，类型为IMatrix<Double>
     * @throws IllegalArgumentException 如果向量包含NaN或无穷大值
     * @see RereMathUtil#sigmoid(double)
     * @see IMatrix
     */
    @Override
    public IMatrix<Double> hessianMatrix() {
        int n = size();
        if (n == 0) {
            return IDoubleMatrix.of(new double[0][0]);
        }

        // Input validation
        for (int i = 0; i < n; i++) {
            double val = get(i);
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException("向量包含无效值: " + val + " 在位置 " + i + " / Vector contains invalid value: " + val + " at position " + i);
            }
        }

        IMatrix<Double> result = IDoubleMatrix.zeros(n, n);

        // For efficiency with small vectors, use direct computation
        if (n <= 100) {
            for (int i = 0; i < n; i++) {
                double fi = get(i);
                double pi = RereMathUtil.sigmoid(fi);
                double wi = pi * (1 - pi);

                // Handle numerical edge cases
                if (Double.isNaN(wi) || Double.isInfinite(wi) || wi < 0) {
                    wi = 0.0; // Set to zero for numerical stability
                }

                result.set(i, i, wi);
            }
        } else {
            // For larger vectors, process in chunks for better cache performance
            int chunkSize = 64; // Common cache line size
            for (int chunkStart = 0; chunkStart < n; chunkStart += chunkSize) {
                int chunkEnd = Math.min(chunkStart + chunkSize, n);
                for (int i = chunkStart; i < chunkEnd; i++) {
                    double fi = get(i);
                    double pi = RereMathUtil.sigmoid(fi);
                    double wi = pi * (1 - pi);

                    // Handle numerical edge cases
                    if (Double.isNaN(wi) || Double.isInfinite(wi) || wi < 0) {
                        wi = 0.0;
                    }

                    result.set(i, i, wi);
                }
            }
        }

        return result;
    }

}
