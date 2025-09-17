package com.reremouse.lab.math.linalg;

import com.reremouse.lab.math.RereMathUtil;
import com.reremouse.lab.util.StringUtils;
import com.reremouse.lab.math.compute.GPUComputeDoubleUtils;
import com.reremouse.lab.math.compute.GPUConfig;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
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
 // 创建向量 / Create vector
 double[] data = {1.0f, 2.0f, 3.0f, 4.0f};
 IVector<Double> vector = new RereDoubleVector(data);

 // 基本运算 / Basic operations
 IVector<Double> doubled = vector.multiplyScala(2.0f);
 Double norm = vector.norm2();

 // 统计运算 / Statistical operations
 Double mean = vector.mean();
 Double std = vector.std();

 // 切片操作 / Slicing operations
 IVector<Double> slice = vector.slice(1, 3);
 IVector<Double> squared = vector.squre();
 }
 * </pre>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereDoubleVector implements IDoubleVector {

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

    // ========== GPU相关字段 / GPU Related Fields ==========
    /**
     * GPU是否可用 / Whether GPU is available
     */
    private static final boolean GPU_ENABLED = GPUComputeDoubleUtils.isGPUAvailable();

    /**
     * GPU计算阈值（元素数） / GPU computation threshold (elements)
     */
    private static final int GPU_THRESHOLD = GPUConfig.GPU_THRESHOLD;

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
        GPUComputeDoubleUtils.cleanup(); // 清理GPU资源
    }


    /**
     * 检查GPU是否启用 / Check if GPU is enabled
     */
    public static boolean isGPUEnabled() {
        return GPU_ENABLED;
    }

    /**
     * 获取GPU信息 / Get GPU information
     */
    public static String getGPUInfo() {
        return GPUComputeDoubleUtils.getGPUInfo();
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
    public IVector<Double> sub(IVector<Double> vec1) {
        // 参数验证：确保输入向量不为null
        if (vec1 == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        // 维度检查：确保两个向量长度相同
        if (this.data.length != vec1.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + vec1.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + vec1.length());
        }
        IDoubleVector vec0 = (IDoubleVector)vec1;

        int len = this.data.length;  // 向量长度

        // GPU加速策略：对于大向量优先使用GPU计算
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                // 尝试使用GPU进行向量减法计算
                return GPUComputeDoubleUtils.gpuVectorSub(this, vec0);
            } catch (Exception e) {
                // GPU计算失败时的容错处理：自动回退到CPU计算
                System.out.println("GPU向量减法失败，回退到CPU: " + e.getMessage());
            }
        }

        // CPU计算：使用简单的循环实现元素级减法
        double[] v = new double[len];  // 预分配结果数组
        for (int i = 0; i < len; i++) {
            v[i] = this.data[i] - vec0.getData()[i];  // 对应元素相减
        }
        var vv = IDoubleVector.of(v);  // 创建结果向量对象
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
        // 参数验证：确保输入向量不为null
        if (vec == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        // 维度检查：确保两个向量长度相同
        if (this.data.length != vec.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + vec.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + vec.length());
        }

        int len = this.data.length;  // 向量长度

        // GPU加速策略：对于大向量优先使用GPU计算
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                // 尝试使用GPU进行向量加法计算
                return GPUComputeDoubleUtils.gpuVectorAdd(this, (IDoubleVector)vec);
            } catch (Exception e) {
                // GPU计算失败时的容错处理：自动回退到CPU计算
                System.out.println("GPU向量加法失败，回退到CPU: " + e.getMessage());
            }
        }

        // CPU计算：使用优化的CPU算法
        return cpuVectorAdd(vec);
    }

    /**
     * CPU向量加法优化实现（CPU Vector Addition Optimization）
     *
     * <p>
     * 这是CPU版本的向量加法实现，采用多层次的性能优化策略。 根据向量大小自动选择最优的计算方法，在保证数值精度的同时最大化计算性能。</p>
     *
     * <p>
     * 优化策略分层：</p>
     * <ul>
     * <li><strong>小向量（≤512）</strong>：使用缓存友好的基础算法，避免并行开销</li>
     * <li><strong>中等向量（513-50000）</strong>：使用SIMD向量化指令，提高单线程性能</li>
     * <li><strong>大向量（>50000）</strong>：使用多线程并行计算，充分利用多核CPU</li>
     * </ul>
     *
     * <p>
     * 性能优化技术：</p>
     * <ul>
     * <li><strong>循环展开</strong>：减少循环控制开销，提高指令级并行度</li>
     * <li><strong>SIMD指令</strong>：利用CPU的向量处理单元，一次处理多个数据</li>
     * <li><strong>缓存优化</strong>：优化内存访问模式，提高缓存命中率</li>
     * <li><strong>并行计算</strong>：利用多核CPU的并行处理能力</li>
     * </ul>
     *
     * <p>
     * 算法选择逻辑：</p>
     * <ul>
     * <li>首先检查向量大小和并行计算是否启用</li>
     * <li>根据大小阈值选择相应的优化算法</li>
     * <li>确保所有情况下都能获得最佳性能</li>
     * </ul>
     *
     * @param vec 加数向量，必须与当前向量具有相同的长度
     * @return 新的向量对象，包含CPU优化的加法运算结果
     */
    private IVector<Double> cpuVectorAdd(IVector<Double> vec1) {
        int len = this.data.length;        // 向量长度
        IDoubleVector vec0 = (IDoubleVector)vec1;
        double[] otherData = vec0.getData(); // 获取加数向量的数据
        double[] result = new double[len];   // 预分配结果数组

        // 根据向量大小选择最优的优化策略
        if (len > 50000 && PARALLEL_ENABLED) {
            // 大向量策略：使用多线程并行计算
            // 阈值50000确保并行计算的开销小于收益
            parallelAdd(otherData, result, len);
        } else if (len > 512) {
            // 中等向量策略：使用SIMD优化的向量化操作
            // 阈值512确保SIMD指令的收益大于循环展开的开销
            simdVectorizedAdd(otherData, result, len);
        } else {
            // 小向量策略：使用缓存友好的基础算法
            // 对于小向量，简单循环比复杂优化更高效
            cacheOptimizedAdd(otherData, result, len);
        }

        return IDoubleVector.of(result);  // 创建并返回结果向量
    }

    /**
     * 缓存优化的基础向量加法 / Cache-optimized basic vector addition
     */
    private void cacheOptimizedAdd(double[] otherData, double[] result, int len) {
        // 对于小向量，简单的顺序访问最优
        for (int i = 0; i < len; i++) {
            result[i] = this.data[i] + otherData[i];
        }
    }

    /**
     * SIMD优化的向量化加法 / SIMD-optimized vectorized addition
     */
    private void simdVectorizedAdd(double[] otherData, double[] result, int len) {
        int i = 0;

        // 8路循环展开，更好地利用CPU流水线和SIMD指令
        final int unrollFactor = 8;
        final int alignedLen = len - (len % unrollFactor);

        for (; i < alignedLen; i += unrollFactor) {
            // 手动展开循环，编译器更容易优化为SIMD指令
            result[i] = this.data[i] + otherData[i];
            result[i + 1] = this.data[i + 1] + otherData[i + 1];
            result[i + 2] = this.data[i + 2] + otherData[i + 2];
            result[i + 3] = this.data[i + 3] + otherData[i + 3];
            result[i + 4] = this.data[i + 4] + otherData[i + 4];
            result[i + 5] = this.data[i + 5] + otherData[i + 5];
            result[i + 6] = this.data[i + 6] + otherData[i + 6];
            result[i + 7] = this.data[i + 7] + otherData[i + 7];
        }

        // 处理剩余元素
        for (; i < len; i++) {
            result[i] = this.data[i] + otherData[i];
        }
    }

    /**
     * 优化的并行向量加法 / Optimized parallel vector addition
     */
    private void parallelAdd(double[] otherData, double[] result, int len) {
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                Math.max(1, len / 10000)); // 动态调整线程数

        // 确保每个线程至少处理足够的元素以摊销线程开销
        final int minElementsPerThread = 10000;
        if (len / numThreads < minElementsPerThread) {
            numThreads = Math.max(1, len / minElementsPerThread);
        }

        final int elementsPerThread = len / numThreads;
        final int remainder = len % numThreads;

        List<Future<Void>> futures = new ArrayList<>(numThreads);

        for (int t = 0; t < numThreads; t++) {
            final int startIdx = t * elementsPerThread;
            final int endIdx = (t == numThreads - 1)
                    ? startIdx + elementsPerThread + remainder
                    : startIdx + elementsPerThread;

            futures.add(THREAD_POOL.submit(() -> {
                // 在每个线程内部也使用SIMD优化
                simdVectorizedAddRange(otherData, result, startIdx, endIdx);
                return null;
            }));
        }

        // 等待所有线程完成
        for (Future<Void> future : futures) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("并行向量加法失败", e);
            }
        }
    }

    /**
     * 范围内的SIMD向量化加法 / SIMD vectorized addition within range
     */
    private void simdVectorizedAddRange(double[] otherData, double[] result, int start, int end) {
        int i = start;
        final int unrollFactor = 8;
        final int alignedEnd = end - ((end - start) % unrollFactor);

        for (; i < alignedEnd; i += unrollFactor) {
            result[i] = this.data[i] + otherData[i];
            result[i + 1] = this.data[i + 1] + otherData[i + 1];
            result[i + 2] = this.data[i + 2] + otherData[i + 2];
            result[i + 3] = this.data[i + 3] + otherData[i + 3];
            result[i + 4] = this.data[i + 4] + otherData[i + 4];
            result[i + 5] = this.data[i + 5] + otherData[i + 5];
            result[i + 6] = this.data[i + 6] + otherData[i + 6];
            result[i + 7] = this.data[i + 7] + otherData[i + 7];
        }

        for (; i < end; i++) {
            result[i] = this.data[i] + otherData[i];
        }
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
        if (vec == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != vec.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + vec.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + vec.length());
        }

        int len = this.data.length;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuVectorDot(this, (IDoubleVector)vec);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU向量内积失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU优化算法
        return cpuInnerProduct(vec);
    }


    /**
     * CPU向量内积（原有算法） / CPU vector inner product (original algorithm)
     */
    private Double cpuInnerProduct(IVector<Double> vec1) {
        int len = this.data.length;
        IDoubleVector vec0 = (IDoubleVector)vec1;
        double[] otherData = vec0.getData();

        // 优化的算法选择策略
        if (len < 64) {
            // 极小向量：直接计算，避免函数调用开销
            return directInnerProduct(otherData, len);
        } else if (len < 1024) {
            // 小向量：循环展开优化
            return unrolledInnerProduct(otherData, len);
        } else if (len < 50000) {
            // 中等向量：SIMD风格优化
            return simdStyleInnerProduct(otherData, len);
        } else {
            // 大向量：并行计算（提高阈值减少线程开销）
            return parallelInnerProduct(otherData, len);
        }
    }

    /**
     * 直接内积计算（极小向量） / Direct inner product calculation for tiny vectors
     */
    private Double directInnerProduct(double[] otherData, int len) {
        double sum = 0.0;
        for (int i = 0; i < len; i++) {
            sum += this.data[i] * otherData[i];
        }
        return sum;
    }

    /**
     * 循环展开的内积计算 / Unrolled inner product calculation
     */
    private Double unrolledInnerProduct(double[] otherData, int len) {
        double sum = 0.0;
        int i = 0;

        // 4路循环展开
        for (; i < len - 3; i += 4) {
            sum += this.data[i] * otherData[i]
                    + this.data[i + 1] * otherData[i + 1]
                    + this.data[i + 2] * otherData[i + 2]
                    + this.data[i + 3] * otherData[i + 3];
        }

        // 处理剩余元素
        for (; i < len; i++) {
            sum += this.data[i] * otherData[i];
        }

        return sum;
    }

    /**
     * SIMD风格的内积计算 / SIMD-style inner product calculation
     */
    private Double simdStyleInnerProduct(double[] otherData, int len) {
        // 使用多个累加器减少数据依赖
        double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
        int i = 0;

        // 8路展开，使用4个累加器
        for (; i < len - 7; i += 8) {
            sum1 += this.data[i] * otherData[i] + this.data[i + 4] * otherData[i + 4];
            sum2 += this.data[i + 1] * otherData[i + 1] + this.data[i + 5] * otherData[i + 5];
            sum3 += this.data[i + 2] * otherData[i + 2] + this.data[i + 6] * otherData[i + 6];
            sum4 += this.data[i + 3] * otherData[i + 3] + this.data[i + 7] * otherData[i + 7];
        }

        // 处理剩余元素
        double remainderSum = 0.0;
        for (; i < len; i++) {
            remainderSum += this.data[i] * otherData[i];
        }

        return sum1 + sum2 + sum3 + sum4 + remainderSum;
    }

    /**
     * 基础内积计算 / Basic inner product calculation
     */
    private Double basicInnerProduct(double[] otherData, int len) {
        double sum = 0;
        for (int i = 0; i < len; i++) {
            sum += this.data[i] * otherData[i];
        }
        return sum;
    }

    /**
     * 使用Kahan求和算法的向量化内积计算 / Kahan summation vectorized inner product
     * calculation
     */
    private Double kahanVectorizedInnerProduct(double[] otherData, int len) {
        // 使用多个累加器减少数据依赖，提高并行度
        double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
        double c1 = 0, c2 = 0, c3 = 0, c4 = 0; // Kahan求和的补偿项

        int i = 0;
        final int unrollFactor = 4;
        final int alignedLen = len - (len % unrollFactor);

        // 4路展开，每路使用Kahan求和
        for (; i < alignedLen; i += unrollFactor) {
            // 第一路
            double prod1 = this.data[i] * otherData[i];
            double y1 = prod1 - c1;
            double temp1 = sum1 + y1;
            c1 = (temp1 - sum1) - y1;
            sum1 = temp1;

            // 第二路
            double prod2 = this.data[i + 1] * otherData[i + 1];
            double y2 = prod2 - c2;
            double temp2 = sum2 + y2;
            c2 = (temp2 - sum2) - y2;
            sum2 = temp2;

            // 第三路
            double prod3 = this.data[i + 2] * otherData[i + 2];
            double y3 = prod3 - c3;
            double temp3 = sum3 + y3;
            c3 = (temp3 - sum3) - y3;
            sum3 = temp3;

            // 第四路
            double prod4 = this.data[i + 3] * otherData[i + 3];
            double y4 = prod4 - c4;
            double temp4 = sum4 + y4;
            c4 = (temp4 - sum4) - y4;
            sum4 = temp4;
        }

        // 处理剩余元素
        double remainderSum = 0;
        for (; i < len; i++) {
            remainderSum += this.data[i] * otherData[i];
        }

        return sum1 + sum2 + sum3 + sum4 + remainderSum;
    }

    /**
     * 优化的并行内积计算 / Optimized parallel inner product calculation
     */
    private Double parallelInnerProduct(double[] otherData, int len) {
        // 更保守的线程数计算，避免过度并行化
        int numThreads = Math.min(4, Math.max(1, len / 100000));

        if (numThreads == 1) {
            // 单线程情况下使用SIMD风格计算
            return simdStyleInnerProduct(otherData, len);
        }

        final int elementsPerThread = len / numThreads;
        final int remainder = len % numThreads;

        List<Future<Double>> futures = new ArrayList<>(numThreads);

        for (int t = 0; t < numThreads; t++) {
            final int startIdx = t * elementsPerThread;
            final int endIdx = (t == numThreads - 1)
                    ? startIdx + elementsPerThread + remainder
                    : startIdx + elementsPerThread;

            futures.add(THREAD_POOL.submit(() -> {
                // 每个线程内部使用SIMD风格计算
                return simdInnerProductRange(otherData, startIdx, endIdx);
            }));
        }

        // 简单求和合并结果
        double totalSum = 0.0;
        for (Future<Double> future : futures) {
            try {
                totalSum += future.get();
            } catch (InterruptedException | ExecutionException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("并行内积计算失败", e);
            }
        }

        return totalSum;
    }

    /**
     * 范围内的SIMD风格内积计算 / SIMD-style inner product for range
     */
    private Double simdInnerProductRange(double[] otherData, int start, int end) {
        double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
        int i = start;

        // 4路展开
        for (; i < end - 3; i += 4) {
            sum1 += this.data[i] * otherData[i];
            sum2 += this.data[i + 1] * otherData[i + 1];
            sum3 += this.data[i + 2] * otherData[i + 2];
            sum4 += this.data[i + 3] * otherData[i + 3];
        }

        // 处理剩余元素
        double remainderSum = 0.0;
        for (; i < end; i++) {
            remainderSum += this.data[i] * otherData[i];
        }

        return sum1 + sum2 + sum3 + sum4 + remainderSum;
    }

    /**
     * 范围内的Kahan求和内积计算 / Kahan summation inner product within range
     */
    private double kahanInnerProductRange(double[] otherData, int start, int end) {
        double sum = 0;
        double c = 0; // Kahan求和补偿项

        for (int i = start; i < end; i++) {
            double prod = (double) this.data[i] * otherData[i];
            double y = prod - c;
            double temp = sum + y;
            c = (temp - sum) - y;
            sum = temp;
        }

        return sum;
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
        int len = this.data.length;

        if (len == 0) {
            return 0.0;
        }
        if (len == 1) {
            return Math.abs(this.data[0]);
        }

        // 使用优化的平方和计算，避免Math.pow的开销
        if (len > 100000 && PARALLEL_ENABLED) {
            return Math.sqrt(parallelSumOfSquares());
        } else if (len > 256) {
            return Math.sqrt(kahanSumOfSquares());
        } else {
            return Math.sqrt(basicSumOfSquares());
        }
    }

    /**
     * 基础平方和计算 / Basic sum of squares calculation
     */
    private double basicSumOfSquares() {
        double sum = 0;
        for (int i = 0; i < this.data.length; i++) {
            double val = this.data[i];
            sum += val * val; // 避免Math.pow的函数调用开销
        }
        return sum;
    }

    /**
     * Kahan求和的平方和计算 / Kahan summation sum of squares calculation
     */
    private double kahanSumOfSquares() {
        double sum = 0;
        double c = 0; // 补偿项

        for (int i = 0; i < this.data.length; i++) {
            double val = this.data[i];
            double square = val * val;
            double y = square - c;
            double temp = sum + y;
            c = (temp - sum) - y;
            sum = temp;
        }

        return sum;
    }

    /**
     * 并行平方和计算 / Parallel sum of squares calculation
     */
    private double parallelSumOfSquares() {
        int len = this.data.length;
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                Math.max(1, len / 25000));

        final int elementsPerThread = len / numThreads;
        final int remainder = len % numThreads;

        List<Future<Double>> futures = new ArrayList<>(numThreads);

        for (int t = 0; t < numThreads; t++) {
            final int startIdx = t * elementsPerThread;
            final int endIdx = (t == numThreads - 1)
                    ? startIdx + elementsPerThread + remainder
                    : startIdx + elementsPerThread;

            futures.add(THREAD_POOL.submit(() -> {
                double sum = 0;
                double c = 0;

                for (int i = startIdx; i < endIdx; i++) {
                    double val = this.data[i];
                    double square = val * val;
                    double y = square - c;
                    double temp = sum + y;
                    c = (temp - sum) - y;
                    sum = temp;
                }

                return sum;
            }));
        }

        // 合并结果
        double totalSum = 0;
        double c = 0;
        for (Future<Double> future : futures) {
            try {
                double partialSum = future.get();
                double y = partialSum - c;
                double t = totalSum + y;
                c = (t - totalSum) - y;
                totalSum = t;
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("并行平方和计算失败", e);
            }
        }

        return totalSum;
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
        int len = this.data.length;
        double sum = 0;
        for (int i = 0; i < len; i++) {
            sum += Math.abs(this.data[i]);
        }
        return sum;
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
        if (p == 0.0f) {
            throw new ArithmeticException("除数不能为零 / Divisor cannot be zero");
        }

        double[] v = new double[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] / p;
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
     * @param vec 另一个向量 / The other vector
     * @return 新的向量对象，包含乘法结果 / New vector object containing multiplication
     * result
     * @throws IllegalArgumentException 如果向量长度不匹配 / if vector lengths don't
     * match
     */
    @Override
    public IVector<Double> multiply(IVector<Double> vec1) {
        IDoubleVector vec0 = (IDoubleVector)vec1;
        if (vec1 == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != vec1.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + vec1.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + vec1.length());
        }

        int len = this.data.length;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuVectorMultiply(this, vec0);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU向量乘法失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU算法
        double[] v = new double[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] * vec0.getData()[i];
        }
        return IDoubleVector.of(v);
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
        IDoubleMatrix m0 = (IDoubleMatrix)m;
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
     * 计算行向量与矩阵的乘积，结果仍是行向量 Computes the product of row vector and matrix,
     * result is still a row vector
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
        int matrixCols = m.getColNum();
        
        if (vectorLen != matrixRows) {
            throw new IllegalArgumentException(
                String.format("向量长度与矩阵行数不匹配: %d != %d / Vector length doesn't match matrix row count: %d != %d", 
                    vectorLen, matrixRows, vectorLen, matrixRows));
        }
        
        // 计算行向量与矩阵的乘积
        double[] result = new double[matrixCols];
        for (int j = 0; j < matrixCols; j++) {
            double sum = 0.0;
            for (int i = 0; i < vectorLen; i++) {
                sum += this.data[i] * m.get(i, j);
            }
            result[j] = sum;
        }
        
        return IDoubleVector.of(result);
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
        int len = this.data.length;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuVectorScalarSub(this, p);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU向量标量减法失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU算法
        double[] v = new double[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] - p;
        }
        return IDoubleVector.of(v);
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
        int len = this.data.length;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuVectorScalarAdd(this, p);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU向量标量加法失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU算法
        double[] v = new double[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] + p;
        }
        return IDoubleVector.of(v);
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
        int len = this.data.length;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuVectorScalarMultiply(this, p);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU向量标量乘法失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU算法
        double[] v = new double[this.data.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = data[i] * p;
        }
        return IDoubleVector.of(v);
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
        int len = this.data.length;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuVectorSum(this);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU向量求和失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU算法
        double sum = 0;
        for (int i = 0; i < len; i++) {
            sum += this.data[i];
        }
        return sum;
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
        int len = this.data.length;
        double min = Double.MAX_VALUE;
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
     * 找到向量中的最大元素值 Finds the maximum element value in the vector
     * </p>
     *
     * @return 最大值 / Maximum value
     */
    @Override
    public Double max() {
        int len = this.data.length;
        double max = Double.MIN_VALUE;
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
        return this.sum() / (double) this.length();
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
        return Math.sqrt(this.var());
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
        return this.subScalar(this.mean()).pow(2d).sum() / (double) this.length();
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

        double sum = 0.0;
        for (int i = 0; i < this.length(); i++) {
            double diff = data[i] - mean;
            sum += diff * diff * diff; // (x - μ)³
        }

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

        double sum = 0.0;
        for (int i = 0; i < this.length(); i++) {
            double diff = data[i] - mean;
            sum += diff * diff * diff * diff; // (x - μ)⁴
        }

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
     * 设置指定位置的元素 / Set element at specified position
     * <p>
     * 设置向量中指定位置的元素值，支持负数索引 Sets the element value at the specified position in
     * the vector, supports negative indexing
     * </p>
     *
     * @param position 位置索引（从0开始，支持负数索引） / Position index (0-based, supports
     * negative indexing)
     * @param value 要设置的值 / Value to set
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     * @throws IndexOutOfBoundsException 如果位置索引超出范围 / if position index is out
     * of bounds
     */
    @Override
    public IVector<Double> set(int position, Double value) {
        int actualPosition = position;
        if (position < 0) {
            actualPosition = data.length + position;
        }
        if (actualPosition < 0 || actualPosition >= data.length) {
            throw new IndexOutOfBoundsException("位置索引超出范围: " + position + " / Position index out of bounds: " + position);
        }
        data[actualPosition] = value;
        return this;
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
    public IVector<Double> setFromTo(int start, int end, int step, Double[] values) {
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
        return this;
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
    public IVector<Double> setFromTo(int start, int end, Double[] values) {
        return this.setFromTo(start, end, 1, values);
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
    public boolean[] equals(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.length() != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.length() + " != " + other.length()
                    + " / Vector lengths don't match: " + this.length() + " != " + other.length());
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
    public boolean[] lessThan(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.length() != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.length() + " != " + other.length()
                    + " / Vector lengths don't match: " + this.length() + " != " + other.length());
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
    public boolean[] greaterThan(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.length() != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.length() + " != " + other.length()
                    + " / Vector lengths don't match: " + this.length() + " != " + other.length());
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
     * 计算向量中所有元素的乘积 Calculates the product of all elements in the vector
     * </p>
     *
     * @return 元素乘积 / Product of elements
     */
    @Override
    public Double prod() {
        double p = 1;
        for (var e : data) {
            p *= e;
        }
        return p;
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
        var data2 = new double[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = Math.abs(data[i]);
        }
        return IDoubleVector.of(data2);
    }

    /**
     * 向量填充 / Vector fill
     * <p>
     * 将向量中所有元素设置为指定值 Sets all elements in the vector to the specified value
     * </p>
     *
     * @param value 填充值 / Fill value
     * @return 修改后的向量（就地操作） / Modified vector (in-place operation)
     */
    @Override
    public IVector<Double> fill(Double value) {
        for (int i = 0; i < this.length(); i++) {
            data[i] = value;
        }
        return this;

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
        int len = this.data.length;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuVectorSqrt(this);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU向量开方失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU算法
        var data2 = new double[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = Math.sqrt(data[i]);
        }
        return IDoubleVector.of(data2);
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
        int len = this.data.length;

        // 尝试GPU计算（如果启用且满足条件）
        if (GPU_ENABLED && len > GPU_THRESHOLD) {
            try {
                return GPUComputeDoubleUtils.gpuVectorSquare(this);
            } catch (Exception e) {
                // GPU失败时回退到CPU
                System.out.println("GPU向量平方失败，回退到CPU: " + e.getMessage());
            }
        }

        // 使用原有的CPU算法
        var data2 = new double[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] =  Math.pow(data[i], 2);
        }
        return IDoubleVector.of(data2);
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
        var data2 = new double[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = Math.exp(data[i]);
        }
        return IDoubleVector.of(data2);
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
        var data2 = new double[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = Math.log(data[i]);
        }
        return IDoubleVector.of(data2);
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
        var data2 = new double[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = Math.log10(data[i]);
        }
        return IDoubleVector.of(data2);
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
        var data2 = new double[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = Math.pow(data[i], m);
        }
        return IDoubleVector.of(data2);
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
        if (value == 0.0f) {
            throw new ArithmeticException("除数不能为零 / Divisor cannot be zero");
        }

        var data2 = new double[this.data.length];
        for (int i = 0; i < this.length(); i++) {
            data2[i] = data[i] % value;
        }
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
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配 / Vector lengths don't match");
        }

        IVector<Double> diff = this.sub(other);
        return Math.sqrt(diff.innerProduct(diff));
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
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配 / Vector lengths don't match");
        }

        IVector<Double> diff = this.sub(other);
        return diff.abs().sum();
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
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配 / Vector lengths don't match");
        }

        double dotProduct = this.innerProduct(other);
        double norm1 = this.norm2();
        double norm2 = other.norm2();

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
        return StringUtils.join(ls, ", ");
    }

    // ========== 三角函数操作实现 / Trigonometric Functions Implementation ==========
    @Override
    public IVector<Double> sin() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.sin(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> cos() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.cos(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> tan() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.tan(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> arcsin() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            if (this.data[i] < -1.0f || this.data[i] > 1.0f) {
                throw new ArithmeticException("反正弦函数输入值超出范围[-1,1]: " + this.data[i]
                        + " / Arcsine input value outside range [-1,1]: " + this.data[i]);
            }
            result[i] =  Math.asin(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> arccos() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            if (this.data[i] < -1.0f || this.data[i] > 1.0f) {
                throw new ArithmeticException("反余弦函数输入值超出范围[-1,1]: " + this.data[i]
                        + " / Arccosine input value outside range [-1,1]: " + this.data[i]);
            }
            result[i] =  Math.acos(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> arctan() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.atan(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    // ========== 双曲函数实现 / Hyperbolic Functions Implementation ==========
    @Override
    public IVector<Double> sinh() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.sinh(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> cosh() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.cosh(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> tanh() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.tanh(this.data[i]);
        }
        return IDoubleVector.of(result);
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
            result[i] =  Math.floor(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> ceil() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.ceil(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> trunc() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] =  Math.rint(this.data[i]);
        }
        return IDoubleVector.of(result);
    }

    // ========== 逻辑运算实现 / Logical Operations Implementation ==========
    @Override
    public IVector<Double> logicalAnd(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + other.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + other.length());
        }

        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (this.data[i] != 0.0f && other.get(i) != 0.0f) ? 1.0f : 0.0f;
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> logicalOr(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + other.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + other.length());
        }

        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (this.data[i] != 0.0f || other.get(i) != 0.0f) ? 1.0f : 0.0f;
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> logicalNot() {
        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            result[i] = (this.data[i] == 0.0f) ? 1.0f : 0.0f;
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> logicalXor(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("输入向量不能为null / Input vector cannot be null");
        }
        if (this.data.length != other.length()) {
            throw new IllegalArgumentException("向量长度不匹配: " + this.data.length + " != " + other.length()
                    + " / Vector lengths don't match: " + this.data.length + " != " + other.length());
        }

        double[] result = new double[this.data.length];
        for (int i = 0; i < this.data.length; i++) {
            boolean a = this.data[i] != 0.0f;
            boolean b = other.get(i) != 0.0f;
            result[i] = (a != b) ? 1.0f : 0.0f;
        }
        return IDoubleVector.of(result);
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
        if (this.data.length < 2) {
            return IDoubleVector.of(new double[0]);
        }

        double[] result = new double[this.data.length - 1];
        for (int i = 0; i < result.length; i++) {
            result[i] = this.data[i + 1] - this.data[i];
        }
        return IDoubleVector.of(result);
    }

    @Override
    public IVector<Double> diff(int n) {
        if (n < 1) {
            throw new IllegalArgumentException("差分阶数必须大于等于1: " + n + " / Difference order must be greater than or equal to 1: " + n);
        }
        if (n >= this.data.length) {
            throw new IllegalArgumentException("差分阶数必须小于向量长度: " + n + " >= " + this.data.length
                    + " / Difference order must be less than vector length: " + n + " >= " + this.data.length);
        }

        IVector<Double> current = this.copy();
        for (int i = 0; i < n; i++) {
            current = current.diff();
        }
        return current;
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

        if (p == 1.0f) {
            return this.norm1();
        }
        if (p == 2.0f) {
            return this.norm2();
        }
        if (Double.isInfinite(p)) {
            return this.normInf();
        }

        double sum = 0.0f;
        for (int i = 0; i < this.data.length; i++) {
            sum += Math.pow(Math.abs(this.data[i]), p);
        }
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
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            if (data[i] == 0.0f) {
                throw new ArithmeticException("向量包含零元素，无法计算倒数 / Vector contains zero element, cannot compute reciprocal");
            }
            result[i] = 1.0f / data[i];
        }
        return new RereDoubleVector(result);
    }

    @Override
    public IDoubleMatrix outer(IVector<Double> other) {
        if (other == null) {
            throw new IllegalArgumentException("向量不能为null / Vector cannot be null");
        }

        int rows = this.data.length;
        int cols = other.length();
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = this.data[i] * other.get(j);
            }
        }

        return new RereDoubleMatrix(result);
    }


    @Override
    public double[] toDoubleArray() {
        return this.data;}
    
    
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
     * 将向量转换为列向量矩阵，即m×1的矩阵，其中m是向量的长度。
     * 向量的每个元素成为矩阵对应行的第一列元素。
     * </p>
     * <p>
     * Converts the vector to a column vector matrix, i.e., an m×1 matrix where m is the vector length.
     * Each element of the vector becomes the first column element of the corresponding row in the matrix.
     * </p>
     *
     * @return 列向量矩阵（m×1）/ Column vector matrix (m×1)
     */
    @Override
    public IMatrix<Double> asColumnVector() {
        int len = this.data.length;
        double[][] columnMatrix = new double[len][1];
        
        // 将向量的每个元素放入矩阵的第一列
        for (int i = 0; i < len; i++) {
            columnMatrix[i][0] = this.data[i];
        }
        
        return new RereDoubleMatrix(columnMatrix);
    }

    /**
     * 动态时间规整（Dynamic Time Warping）算法
     * <p>
     * 计算两个时间序列之间的DTW距离，用于衡量时间序列的相似性。
     * DTW算法能够处理不同长度的时间序列，并找到最优的对齐路径。
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
     * 时间复杂度：O(m×n)，其中m和n是两个序列的长度
     * 空间复杂度：O(m×n)，用于存储距离矩阵
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
     * 计算与另一个向量的皮尔逊相关系数 / Compute Pearson correlation coefficient with another vector
     * <p>
     * 计算当前向量与另一个向量之间的皮尔逊相关系数，衡量两个向量的线性相关性。
     * 相关系数的取值范围为[-1, 1]，其中1表示完全正相关，-1表示完全负相关，0表示无线性相关。
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
     * 计算当前向量与另一个向量之间的协方差，衡量两个向量的线性相关性。
     * 协方差的计算公式为：cov(X,Y) = E[(X-μX)(Y-μY)] = E[XY] - μXμY
     * </p>
     * <p>
     * 协方差的性质：
     * <ul>
     * <li>cov(X,Y) > 0: 正相关，X增大时Y倾向于增大</li>
     * <li>cov(X,Y) < 0: 负相关，X增大时Y倾向于减小</li>
     * <li>cov(X,Y) = 0: 无线性相关</li>
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
     * 计算向量的第一四分位数，即25%分位数。第一四分位数是将数据按升序排列后，
     * 位于25%位置的值，表示有25%的数据小于等于该值。
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
     * 计算向量的第三四分位数，即75%分位数。第三四分位数是将数据按升序排列后，
     * 位于75%位置的值，表示有75%的数据小于等于该值。
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

    

    

}
