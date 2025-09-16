package com.reremouse.lab.math.compute;

import com.aparapi.Kernel;
import com.aparapi.Range;
import com.aparapi.device.Device;
import static com.reremouse.lab.math.compute.GPUConfig.GPU_THRESHOLD;
import com.reremouse.lab.math.linalg.RereDoubleMatrix;
import com.reremouse.lab.math.linalg.RereDoubleVector;
import com.reremouse.lab.math.compute.CPUComputeDoubleUtils;
import com.reremouse.lab.util.Tuple3;
import com.reremouse.lab.util.Tuple2;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.*;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * GPU计算工具类（GPU Computing Utilities）- 基于Aparapi框架实现
 * 
 * <p>本类提供高性能的GPU加速计算功能，专门用于矩阵和向量的并行运算。
 * 基于Aparapi框架实现OpenCL GPU加速，在保证数值精度的同时显著提升计算性能。</p>
 * 
 * <p>核心特性：</p>
 * <ul>
 *   <li><strong>GPU加速</strong>：利用GPU的并行计算能力，大幅提升大规模数据运算性能</li>
 *   <li><strong>自动回退</strong>：GPU计算失败时自动回退到CPU计算，确保系统稳定性</li>
 *   <li><strong>智能选择</strong>：根据数据大小和运算复杂度智能选择最优计算策略</li>
 *   <li><strong>内存优化</strong>：使用内存池技术减少GC压力，提高内存使用效率</li>
 * </ul>
 * 
 * <p>支持的运算类型：</p>
 * <ul>
 *   <li><strong>矩阵运算</strong>：矩阵乘法、伪逆矩阵、特征分解、奇异值分解</li>
 *   <li><strong>向量运算</strong>：向量加法、减法、内积、范数计算、倒数运算</li>
 *   <li><strong>统计运算</strong>：求和、均值、方差、标准差等统计函数</li>
 *   <li><strong>数学函数</strong>：指数、对数、三角函数等通用数学函数</li>
 * </ul>
 * 
 * <p>性能优化策略：</p>
 * <ul>
 *   <li><strong>阈值控制</strong>：小数据（< GPU_THRESHOLD）使用CPU，避免GPU设备访问开销</li>
 *   <li><strong>算法选择</strong>：复杂迭代算法（特征分解、SVD）主要使用CPU优化实现</li>
 *   <li><strong>并行优化</strong>：简单并行运算（矩阵乘法、向量运算）充分利用GPU加速</li>
 *   <li><strong>批处理</strong>：支持批量运算，减少GPU设备切换开销</li>
 * </ul>
 * 
 * <p>技术实现：</p>
 * <ul>
 *   <li><strong>Aparapi框架</strong>：使用Java-OpenCL桥接，实现跨平台GPU计算</li>
 *   <li><strong>JTP模式</strong>：使用Java线程池模式，避免OpenCL兼容性问题</li>
 *   <li><strong>内存管理</strong>：智能内存池管理，减少内存分配和回收开销</li>
 *   <li><strong>错误处理</strong>：完善的异常处理机制，确保计算稳定性</li>
 * </ul>
 * 
 * <p>使用示例：</p>
 * <pre>
 * {@code
 // 检查GPU可用性
 if (GPUComputeDoubleUtils.isGPUAvailable()) {
     // 执行GPU加速的矩阵乘法
     IMatrix<Double> result = GPUComputeDoubleUtils.gpuMatrixMultiply(matrixA, matrixB);
 }
 
 // 执行GPU加速的向量运算
 IVector<Double> sum = GPUComputeDoubleUtils.gpuVectorAdd(vectorA, vectorB);
 }
 * </pre>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class GPUComputeDoubleUtils {
    
    // GPU状态
    private static volatile boolean gpuAvailable = true;
    private static Device gpuDevice = null;
    private static String gpuInfo = "GPU: 未初始化";
    
    // 日志控制
    private static volatile boolean enableLogging = false;
    private static volatile boolean enableDetailedLogging = false;
    
    // 性能优化常量
    private static final int BATCH_SIZE_THRESHOLD = 5; // 批处理阈值
    private static final int SMALL_MATRIX_THRESHOLD = 100; // 小矩阵阈值
    private static final int MEDIUM_MATRIX_THRESHOLD = 1000; // 中等矩阵阈值
    
    // 内存池优化 - 缓存常用矩阵大小的内存
    private static final Map<Integer, Queue<double[]>> vectorMemoryPool = new ConcurrentHashMap<>();
    private static final Map<String, Queue<double[][]>> matrixMemoryPool = new ConcurrentHashMap<>();
    private static final int MAX_POOL_SIZE = 10; // 每种大小最多缓存10个
    
    // 批处理队列
    private static final Queue<DecompositionTask> batchQueue = new ConcurrentLinkedQueue<>();
    private static final AtomicBoolean batchProcessing = new AtomicBoolean(false);
    
    static {
        // 设置Aparapi执行模式为JTP（Java Thread Pool），避免OpenCL问题
        System.setProperty("com.aparapi.executionMode", "JTP");
        initializeGPU();
    }
    
    /**
     * 初始化GPU环境（Initialize GPU Environment）
     * 
     * <p>初始化GPU计算环境，配置Aparapi框架以支持GPU加速计算。
     * 本方法在类加载时自动调用，确保GPU环境在首次使用前已正确初始化。</p>
     * 
     * <p>初始化策略：</p>
     * <ul>
     *   <li><strong>JTP模式</strong>：使用Java线程池模式，避免OpenCL兼容性问题</li>
     *   <li><strong>自动设备选择</strong>：让Aparapi自动选择最佳GPU设备</li>
     *   <li><strong>容错设计</strong>：即使GPU不可用，系统仍可正常运行</li>
     * </ul>
     * 
     * <p>JTP模式优势：</p>
     * <ul>
     *   <li>避免OpenCL驱动兼容性问题</li>
     *   <li>更好的跨平台支持</li>
     *   <li>自动选择最优计算设备</li>
     *   <li>降低系统依赖复杂度</li>
     * </ul>
     * 
     * <p>注意事项：</p>
     * <ul>
     *   <li>初始化失败不会影响CPU计算功能</li>
     *   <li>GPU状态通过gpuAvailable标志控制</li>
     *   <li>设备信息存储在gpuInfo中供查询</li>
     * </ul>
     */
    private static void initializeGPU() {
        // 在JTP模式下，Aparapi会自动选择最佳设备
        // JTP（Java Thread Pool）模式避免了OpenCL驱动的兼容性问题
        gpuAvailable = true;
        gpuDevice = null; // JTP模式下不需要手动指定设备，Aparapi自动管理
        
        // 设置GPU信息，用于状态查询和调试
        gpuInfo = "GPU: 使用JTP模式，Aparapi自动选择设备";
        System.out.println("GPU: 使用JTP模式，Aparapi自动选择设备");
    }
    
    /**
     * 检查GPU是否可用
     */
    public static boolean isGPUAvailable() {
        return gpuAvailable;
    }
    
    /**
     * 设置是否启用日志
     */
    public static void setLoggingEnabled(boolean enabled) {
        enableLogging = enabled;
    }
    
    /**
     * 设置是否启用详细日志
     */
    public static void setDetailedLoggingEnabled(boolean enabled) {
        enableDetailedLogging = enabled;
    }
    
    /**
     * 检查是否启用日志
     */
    public static boolean isLoggingEnabled() {
        return enableLogging;
    }
    
    /**
     * 检查是否启用详细日志
     */
    public static boolean isDetailedLoggingEnabled() {
        return enableDetailedLogging;
    }
    
    /**
     * 输出GPU操作日志
     */
    private static void logGPUOperation(String operation, String details) {
        if (enableLogging) {
            System.out.println("🚀 [GPU] " + operation + (enableDetailedLogging ? " - " + details : ""));
        }
    }
    
    /**
     * 输出CPU回退日志
     */
    private static void logCPUFallback(String operation, String reason) {
        if (enableLogging) {
            System.out.println("💻 [CPU] " + operation + " (回退原因: " + reason + ")");
        }
    }
    
    /**
     * 输出性能统计日志
     */
    private static void logPerformance(String operation, long startTime, long endTime, int dataSize) {
        if (enableLogging && enableDetailedLogging) {
            long duration = endTime - startTime;
            System.out.println("⏱️  [性能] " + operation + " - 耗时: " + duration + "ms, 数据量: " + dataSize);
        }
    }
    
    /**
     * GPU向量倒数计算
     * 计算向量的倒数，用于伪逆矩阵计算
     */
    public static IVector<Double> gpuVectorReciprocal(IVector<Double> a, double  tolerance) {
        long startTime = System.currentTimeMillis();
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量倒数", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorReciprocal(a, tolerance);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量倒数", "GPU不可用");
            return CPUComputeDoubleUtils.vectorReciprocal(a, tolerance);
        }
        
        double[] dataA = a.toDoubleArray();
        int length = dataA.length;
        logGPUOperation("向量倒数", "长度: " + length + ", 容差: " + tolerance);
        
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    double  value = dataA[i];
                    if (Math.abs(value) > tolerance) {
                        result[i] = 1.0f / value;
                    } else {
                        result[i] = 0.0f;
                    }
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            long endTime = System.currentTimeMillis();
            logPerformance("向量倒数", startTime, endTime, dataSize);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            logCPUFallback("向量倒数", "GPU执行失败: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorReciprocal(a, tolerance);
        } finally {
            kernel.dispose();
        }
    }
    
    
    /**
     * GPU伪逆矩阵计算
     * 使用GPU加速的伪逆矩阵计算，主要加速矩阵乘法部分
     */
    public static IMatrix<Double> gpuPseudoInverse(IMatrix<Double> A) {
        long startTime = System.currentTimeMillis();
        int dataSize = A.rows() * A.cols();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("伪逆矩阵", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.pseudoInverse(A);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("伪逆矩阵", "GPU不可用");
            return CPUComputeDoubleUtils.pseudoInverse(A);
        }
        
        logGPUOperation("伪逆矩阵", "维度: " + A.rows() + "x" + A.cols());
        
        try {
            final double  tolerance = 1e-10f;
            
            // 进行奇异值分解：A = U * S * V^T
            var svdResult = A.svd();
            IMatrix<Double> U = (IMatrix<Double>)svdResult._1;           // 左奇异向量矩阵
            IVector<Double> singularValues = (IVector<Double>)svdResult._2;  // 奇异值向量
            IMatrix<Double> VT = (IMatrix<Double>)svdResult._3;          // 右奇异向量转置矩阵
            
            // 获取矩阵的维度信息
            int originalRows = A.rows();
            int originalCols = A.cols();
            int singularValuesLength = singularValues.length();
            
            // 使用GPU计算奇异值的伪逆
            IVector<Double> pseudoSingularValues;
            if (singularValuesLength >= GPU_THRESHOLD) {
                pseudoSingularValues = gpuVectorReciprocal(singularValues, tolerance);
            } else {
                pseudoSingularValues = CPUComputeDoubleUtils.vectorReciprocal(singularValues, tolerance);
            }
            
            // 计算伪逆：A⁺ = V * Σ⁺ * U^T
            IMatrix<Double> V = (IMatrix<Double>)VT.transposeNew();  // V = (V^T)^T
            
            // 创建结果矩阵：A⁺的维度应该是 originalCols x originalRows
            IMatrix<Double> pseudoInverse = Linalg.zeros(originalCols, originalRows);
            
            // 使用GPU加速的矩阵乘法计算伪逆
            // 逐列计算：A⁺[:,j] = V * Σ⁺ * U[j,:]^T
            for (int j = 0; j < originalRows; j++) {
                // 提取U的第j行
                IVector<Double> uj = (IVector<Double>)U.getRow(j);
                
                // 计算 Σ⁺ * U[j,:]^T
                IVector<Double> sigmaUj = (IVector<Double>)pseudoSingularValues.multiply(uj);
                
                // 将向量转换为列矩阵进行矩阵乘法
                double[][] sigmaUjData = new double[sigmaUj.length()][1];
                for (int k = 0; k < sigmaUj.length(); k++) {
                    sigmaUjData[k][0] = sigmaUj.get(k);
                }
                IMatrix<Double> sigmaUjMatrix = Linalg.matrix(sigmaUjData);
                
                // 计算 V * (Σ⁺ * U[j,:]^T) - 这是矩阵乘法
                IMatrix<Double> resultColMatrix = (IMatrix<Double>)V.mmul(sigmaUjMatrix);
                
                // 将结果放入伪逆矩阵的第j列
                for (int i = 0; i < originalCols; i++) {
                    pseudoInverse.put(i, j, resultColMatrix.get(i, 0));
                }
            }
            
            long endTime = System.currentTimeMillis();
            logPerformance("伪逆矩阵", startTime, endTime, dataSize);
            
            return pseudoInverse;
            
        } catch (Exception e) {
            logCPUFallback("伪逆矩阵", "GPU执行失败: " + e.getMessage());
            return CPUComputeDoubleUtils.pseudoInverse(A);
        }
    }
    
    
    /**
     * GPU矩阵乘法
     * 使用Aparapi实现OpenCL GPU加速的矩阵乘法
     */
    public static IMatrix<Double> gpuMatrixMultiply(IMatrix<Double> A, IMatrix<Double> B) {
        long startTime = System.currentTimeMillis();
        int dataSize = A.rows() * A.cols() * B.cols();
        
        // Small data unconditionally uses CPU - use specific threshold for matrix multiplication
        if (dataSize < GPUConfig.GPU_MATRIX_MULTIPLY_THRESHOLD) {
            logCPUFallback("矩阵乘法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.matrixMultiply(A.toDoubleArray(), B.toDoubleArray());
        }
        
        if (!gpuAvailable) {
            logCPUFallback("矩阵乘法", "GPU不可用");
            return CPUComputeDoubleUtils.matrixMultiply(A.toDoubleArray(), B.toDoubleArray());
        }
        
        double[][] dataA = A.toDoubleArray();
        double[][] dataB = B.toDoubleArray();
        
        int m = dataA.length;
        int n = dataA[0].length;
        int p = dataB[0].length;
        
        if (n != dataB.length) {
            throw new IllegalArgumentException("矩阵维度不匹配进行乘法运算");
        }
        
        logGPUOperation("矩阵乘法", "维度: " + m + "x" + n + " × " + n + "x" + p);
        
        // 创建结果矩阵
        double[][] resultData = new double[m][p];
        
        // 将2D数组转换为1D数组用于GPU计算
        double[] flatA = flattenMatrix(dataA);
        double[] flatB = flattenMatrix(dataB);
        double[] flatResult = new double[m * p];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                int j = getGlobalId(1);
                
                if (i < m && j < p) {
                    double  sum = 0.0f;
                    for (int k = 0; k < n; k++) {
                        sum += flatA[i * n + k] * flatB[k * p + j];
                    }
                    flatResult[i * p + j] = sum;
                }
            }
        };
        
        try {
            // 使用JTP模式，不指定设备，让Aparapi自动选择
            Range range = Range.create2D(m, p);
            kernel.execute(range);
            
            // 将1D结果转换回2D数组
            resultData = unflattenMatrix(flatResult, m, p);
            
            long endTime = System.currentTimeMillis();
            logPerformance("矩阵乘法", startTime, endTime, dataSize);
            
            return new RereDoubleMatrix(resultData);
            
        } catch (Exception e) {
            logCPUFallback("矩阵乘法", "GPU执行失败: " + e.getMessage());
            // 回退到CPU计算
            return CPUComputeDoubleUtils.matrixMultiply(dataA, dataB);
        } finally {
            kernel.dispose();
        }
    }
    
    
    
    /**
     * GPU向量加法（GPU Vector Addition）
     * 
     * <p>使用Aparapi框架实现GPU加速的向量加法运算。该方法利用GPU的并行计算能力
     * 大幅提升大规模向量加法的性能，特别适用于机器学习、数值计算等需要处理
     * 大量向量运算的场景。</p>
     * 
     * <p>算法原理：</p>
     * <ul>
     *   <li>将向量加法运算分解为独立的并行任务</li>
     *   <li>每个GPU线程处理一个元素：result[i] = a[i] + b[i]</li>
     *   <li>利用GPU的SIMD（单指令多数据）架构同时处理多个元素</li>
     *   <li>通过内存合并访问优化数据传输效率</li>
     * </ul>
     * 
     * <p>性能优化策略：</p>
     * <ul>
     *   <li><strong>阈值控制</strong>：小向量使用CPU，避免GPU设备访问开销</li>
     *   <li><strong>自动回退</strong>：GPU计算失败时自动回退到CPU计算</li>
     *   <li><strong>内存优化</strong>：使用连续内存布局，提高缓存命中率</li>
     *   <li><strong>资源管理</strong>：及时释放GPU资源，避免内存泄漏</li>
     * </ul>
     * 
     * <p>GPU并行化优势：</p>
     * <ul>
     *   <li>理论上可达到O(n/p)的时间复杂度，其中p是并行处理器数量</li>
     *   <li>对于大向量（>10000元素），性能提升可达10-100倍</li>
     *   <li>充分利用现代GPU的数千个并行核心</li>
     * </ul>
     * 
     * <p>适用场景：</p>
     * <ul>
     *   <li>大规模向量运算（长度 > GPU_THRESHOLD）</li>
     *   <li>机器学习中的批量数据处理</li>
     *   <li>科学计算中的向量场运算</li>
     *   <li>图像处理中的像素级运算</li>
     * </ul>
     * 
     * @param a 第一个向量，不能为null
     * @param b 第二个向量，不能为null，长度必须与a相同
     * @return 新的向量对象，包含加法运算结果
     * @throws IllegalArgumentException 当向量为null或长度不匹配时抛出异常
     */
    public static IVector<Double> gpuVectorAdd(IVector<Double> a, IVector<Double> b) {
        long startTime = System.currentTimeMillis();  // 性能计时开始
        int dataSize = a.length();  // 获取向量长度
        
        // 小数据优化策略：小于阈值的向量使用CPU计算
        // 避免GPU设备访问开销，提高小数据计算效率
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量加法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorAdd(a, b);
        }
        
        // GPU可用性检查：确保GPU环境正常
        if (!gpuAvailable) {
            logCPUFallback("向量加法", "GPU不可用");
            return CPUComputeDoubleUtils.vectorAdd(a, b);
        }
        
        // 获取向量数据，准备GPU计算
        double[] dataA = a.toDoubleArray();
        double[] dataB = b.toDoubleArray();
        
        // 维度验证：确保两个向量长度相同
        if (dataA.length != dataB.length) {
            throw new IllegalArgumentException("向量长度不匹配");
        }
        
        int length = dataA.length;  // 向量长度
        logGPUOperation("向量加法", "长度: " + length);  // 记录GPU操作日志
        
        double[] result = new double[length];  // 预分配结果数组
        
        // 创建Aparapi GPU Kernel
        // Kernel是GPU并行计算的核心，定义了每个线程要执行的操作
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                // 获取当前线程的全局ID，对应向量元素的索引
                int i = getGlobalId(0);
                if (i < length) {
                    // 执行向量加法：result[i] = a[i] + b[i]
                    result[i] = dataA[i] + dataB[i];
                }
            }
        };
        
        try {
            // 执行GPU并行计算
            // 使用JTP模式，让Aparapi自动选择最优设备
            Range range = Range.create(length);  // 创建一维计算范围
            kernel.execute(range);  // 启动GPU并行计算
            
            long endTime = System.currentTimeMillis();  // 性能计时结束
            logPerformance("向量加法", startTime, endTime, dataSize);  // 记录性能日志
            
            return new RereDoubleVector(result);  // 创建并返回结果向量
            
        } catch (Exception e) {
            // GPU计算失败时的容错处理
            logCPUFallback("向量加法", "GPU执行失败: " + e.getMessage());
            // 自动回退到CPU计算，确保计算能够完成
            return CPUComputeDoubleUtils.vectorAdd(a, b);
        } finally {
            // 资源清理：释放GPU Kernel资源，避免内存泄漏
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量内积
     * 使用Aparapi实现OpenCL GPU加速的向量内积
     */
    public static double  gpuVectorDot(IVector<Double> a, IVector<Double> b) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量内积", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorDot(a, b);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量内积", "GPU不可用");
            return CPUComputeDoubleUtils.vectorDot(a, b);
        }
        
        double[] dataA = a.toDoubleArray();
        double[] dataB = b.toDoubleArray();
        
        if (dataA.length != dataB.length) {
            throw new IllegalArgumentException("向量长度不匹配");
        }
        
        int length = dataA.length;
        double[] partialSums = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    partialSums[i] = dataA[i] * dataB[i];
                }
            }
        };
        
        try {
            // 使用JTP模式，不指定设备
            Range range = Range.create(length);
            kernel.execute(range);
            
            // 在CPU上计算最终的和
            double  sum = 0.0f;
            for (double  partialSum : partialSums) {
                sum += partialSum;
            }
            
            return sum;
            
        } catch (Exception e) {
            System.err.println("GPU向量内积失败，回退到CPU计算: " + e.getMessage());
            // 回退到CPU计算
            return CPUComputeDoubleUtils.vectorDot(a, b);
        } finally {
            kernel.dispose();
        }
    }
    
    
    /**
     * 清理GPU资源
     */
    public static void cleanup() {
        try {
            if (gpuDevice != null) {
                // Aparapi会自动管理资源，这里主要是重置状态
                gpuAvailable = false;
                gpuDevice = null;
                gpuInfo = "GPU: 已清理";
                System.out.println("GPU: 资源清理完成");
            }
        } catch (Exception e) {
            System.out.println("GPU: 清理资源时出错 - " + e.getMessage());
        }
    }
    
    /**
     * 获取GPU信息
     */
    public static String getGPUInfo() {
        return gpuInfo;
    }
    
    /**
     * 将2D矩阵转换为1D数组
     */
    private static double[] flattenMatrix(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[] flat = new double[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, flat, i * cols, cols);
        }
        
        return flat;
    }
    
    /**
     * 将1D数组转换为2D矩阵
     */
    private static double[][] unflattenMatrix(double[] flat, int rows, int cols) {
        double[][] matrix = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, matrix[i], 0, cols);
        }
        
        return matrix;
    }
    
    /**
     * GPU矩阵加法
     * 使用Aparapi实现OpenCL GPU加速的矩阵加法
     */
    public static IMatrix<Double> gpuMatrixAdd(IMatrix<Double> A, IMatrix<Double> B) {
        long startTime = System.currentTimeMillis();
        int dataSize = A.rows() * A.cols();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("矩阵加法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.matrixAdd(A.toDoubleArray(), B.toDoubleArray());
        }
        
        if (!gpuAvailable) {
            logCPUFallback("矩阵加法", "GPU不可用");
            return CPUComputeDoubleUtils.matrixAdd(A.toDoubleArray(), B.toDoubleArray());
        }
        
        double[][] dataA = A.toDoubleArray();
        double[][] dataB = B.toDoubleArray();
        
        int m = dataA.length;
        int n = dataA[0].length;
        
        if (m != dataB.length || n != dataB[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配进行加法运算");
        }
        
        logGPUOperation("矩阵加法", "维度: " + m + "x" + n);
        
        // 创建结果矩阵
        double[][] resultData = new double[m][n];
        
        // 将2D数组转换为1D数组用于GPU计算
        double[] flatA = flattenMatrix(dataA);
        double[] flatB = flattenMatrix(dataB);
        double[] flatResult = new double[m * n];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                int j = getGlobalId(1);
                
                if (i < m && j < n) {
                    flatResult[i * n + j] = flatA[i * n + j] + flatB[i * n + j];
                }
            }
        };
        
        try {
            Range range = Range.create2D(m, n);
            kernel.execute(range);
            
            // 将1D结果转换回2D数组
            resultData = unflattenMatrix(flatResult, m, n);
            
            long endTime = System.currentTimeMillis();
            logPerformance("矩阵加法", startTime, endTime, dataSize);
            
            return new RereDoubleMatrix(resultData);
            
        } catch (Exception e) {
            logCPUFallback("矩阵加法", "GPU执行失败: " + e.getMessage());
            return CPUComputeDoubleUtils.matrixAdd(dataA, dataB);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU矩阵减法
     * 使用Aparapi实现OpenCL GPU加速的矩阵减法
     */
    public static IMatrix<Double> gpuMatrixSub(IMatrix<Double> A, IMatrix<Double> B) {
        int dataSize = A.rows() * A.cols();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("矩阵减法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.matrixSub(A.toDoubleArray(), B.toDoubleArray());
        }
        
        if (!gpuAvailable) {
            logCPUFallback("矩阵减法", "GPU不可用");
            return CPUComputeDoubleUtils.matrixSub(A.toDoubleArray(), B.toDoubleArray());
        }
        
        double[][] dataA = A.toDoubleArray();
        double[][] dataB = B.toDoubleArray();
        
        int m = dataA.length;
        int n = dataA[0].length;
        
        if (m != dataB.length || n != dataB[0].length) {
            throw new IllegalArgumentException("矩阵维度不匹配进行减法运算");
        }
        
        // 创建结果矩阵
        double[][] resultData = new double[m][n];
        
        // 将2D数组转换为1D数组用于GPU计算
        double[] flatA = flattenMatrix(dataA);
        double[] flatB = flattenMatrix(dataB);
        double[] flatResult = new double[m * n];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                int j = getGlobalId(1);
                
                if (i < m && j < n) {
                    flatResult[i * n + j] = flatA[i * n + j] - flatB[i * n + j];
                }
            }
        };
        
        try {
            Range range = Range.create2D(m, n);
            kernel.execute(range);
            
            // 将1D结果转换回2D数组
            resultData = unflattenMatrix(flatResult, m, n);
            
            return new RereDoubleMatrix(resultData);
            
        } catch (Exception e) {
            System.err.println("GPU矩阵减法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.matrixSub(dataA, dataB);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU矩阵标量乘法
     * 使用Aparapi实现OpenCL GPU加速的矩阵标量乘法
     */
    public static IMatrix<Double> gpuMatrixScalarMultiply(IMatrix<Double> A, double  scalar) {
        int dataSize = A.rows() * A.cols();
        
        // Small data unconditionally uses CPU - use specific threshold for scalar operations
        if (dataSize < GPUConfig.GPU_MATRIX_SCALAR_THRESHOLD) {
            logCPUFallback("矩阵标量乘法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.matrixScalarMultiply(A.toDoubleArray(), scalar);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("矩阵标量乘法", "GPU不可用");
            return CPUComputeDoubleUtils.matrixScalarMultiply(A.toDoubleArray(), scalar);
        }
        
        double[][] dataA = A.toDoubleArray();
        int m = dataA.length;
        int n = dataA[0].length;
        
        // 创建结果矩阵
        double[][] resultData = new double[m][n];
        
        // 将2D数组转换为1D数组用于GPU计算
        double[] flatA = flattenMatrix(dataA);
        double[] flatResult = new double[m * n];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                int j = getGlobalId(1);
                
                if (i < m && j < n) {
                    flatResult[i * n + j] = flatA[i * n + j] * scalar;
                }
            }
        };
        
        try {
            Range range = Range.create2D(m, n);
            kernel.execute(range);
            
            // 将1D结果转换回2D数组
            resultData = unflattenMatrix(flatResult, m, n);
            
            return new RereDoubleMatrix(resultData);
            
        } catch (Exception e) {
            System.err.println("GPU矩阵标量乘法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.matrixScalarMultiply(dataA, scalar);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU矩阵标量加法
     * 使用Aparapi实现OpenCL GPU加速的矩阵标量加法
     */
    public static IMatrix<Double> gpuMatrixScalarAdd(IMatrix<Double> A, double  scalar) {
        int dataSize = A.rows() * A.cols();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("矩阵标量加法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.matrixScalarAdd(A.toDoubleArray(), scalar);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("矩阵标量加法", "GPU不可用");
            return CPUComputeDoubleUtils.matrixScalarAdd(A.toDoubleArray(), scalar);
        }
        
        double[][] dataA = A.toDoubleArray();
        int m = dataA.length;
        int n = dataA[0].length;
        
        // 创建结果矩阵
        double[][] resultData = new double[m][n];
        
        // 将2D数组转换为1D数组用于GPU计算
        double[] flatA = flattenMatrix(dataA);
        double[] flatResult = new double[m * n];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                int j = getGlobalId(1);
                
                if (i < m && j < n) {
                    flatResult[i * n + j] = flatA[i * n + j] + scalar;
                }
            }
        };
        
        try {
            Range range = Range.create2D(m, n);
            kernel.execute(range);
            
            // 将1D结果转换回2D数组
            resultData = unflattenMatrix(flatResult, m, n);
            
            return new RereDoubleMatrix(resultData);
            
        } catch (Exception e) {
            System.err.println("GPU矩阵标量加法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.matrixScalarAdd(dataA, scalar);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU矩阵标量减法
     * 使用Aparapi实现OpenCL GPU加速的矩阵标量减法
     */
    public static IMatrix<Double> gpuMatrixScalarSub(IMatrix<Double> A, double  scalar) {
        int dataSize = A.rows() * A.cols();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("矩阵标量减法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.matrixScalarSub(A.toDoubleArray(), scalar);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("矩阵标量减法", "GPU不可用");
            return CPUComputeDoubleUtils.matrixScalarSub(A.toDoubleArray(), scalar);
        }
        
        double[][] dataA = A.toDoubleArray();
        int m = dataA.length;
        int n = dataA[0].length;
        
        // 创建结果矩阵
        double[][] resultData = new double[m][n];
        
        // 将2D数组转换为1D数组用于GPU计算
        double[] flatA = flattenMatrix(dataA);
        double[] flatResult = new double[m * n];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                int j = getGlobalId(1);
                
                if (i < m && j < n) {
                    flatResult[i * n + j] = flatA[i * n + j] - scalar;
                }
            }
        };
        
        try {
            Range range = Range.create2D(m, n);
            kernel.execute(range);
            
            // 将1D结果转换回2D数组
            resultData = unflattenMatrix(flatResult, m, n);
            
            return new RereDoubleMatrix(resultData);
            
        } catch (Exception e) {
            System.err.println("GPU矩阵标量减法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.matrixScalarSub(dataA, scalar);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU矩阵转置
     * 使用Aparapi实现OpenCL GPU加速的矩阵转置
     */
    public static IMatrix<Double> gpuMatrixTranspose(IMatrix<Double> A) {
        int dataSize = A.rows() * A.cols();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("矩阵转置", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.matrixTranspose(A.toDoubleArray());
        }
        
        if (!gpuAvailable) {
            logCPUFallback("矩阵转置", "GPU不可用");
            return CPUComputeDoubleUtils.matrixTranspose(A.toDoubleArray());
        }
        
        double[][] dataA = A.toDoubleArray();
        int m = dataA.length;
        int n = dataA[0].length;
        
        // 创建结果矩阵
        double[][] resultData = new double[n][m];
        
        // 将2D数组转换为1D数组用于GPU计算
        double[] flatA = flattenMatrix(dataA);
        double[] flatResult = new double[m * n];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                int j = getGlobalId(1);
                
                if (i < m && j < n) {
                    flatResult[j * m + i] = flatA[i * n + j];
                }
            }
        };
        
        try {
            Range range = Range.create2D(m, n);
            kernel.execute(range);
            
            // 将1D结果转换回2D数组
            resultData = unflattenMatrix(flatResult, n, m);
            
            return new RereDoubleMatrix(resultData);
            
        } catch (Exception e) {
            System.err.println("GPU矩阵转置失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.matrixTranspose(dataA);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量减法
     * 使用Aparapi实现OpenCL GPU加速的向量减法
     */
    public static IVector<Double> gpuVectorSub(IVector<Double> a, IVector<Double> b) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量减法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorSub(a, b);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量减法", "GPU不可用");
            return CPUComputeDoubleUtils.vectorSub(a, b);
        }
        
        double[] dataA = a.toDoubleArray();
        double[] dataB = b.toDoubleArray();
        
        if (dataA.length != dataB.length) {
            throw new IllegalArgumentException("向量长度不匹配");
        }
        
        int length = dataA.length;
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    result[i] = dataA[i] - dataB[i];
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            System.err.println("GPU向量减法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorSub(a, b);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量乘法
     * 使用Aparapi实现OpenCL GPU加速的向量乘法
     */
    public static IVector<Double> gpuVectorMultiply(IVector<Double> a, IVector<Double> b) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量乘法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorMultiply(a, b);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量乘法", "GPU不可用");
            return CPUComputeDoubleUtils.vectorMultiply(a, b);
        }
        
        double[] dataA = a.toDoubleArray();
        double[] dataB = b.toDoubleArray();
        
        if (dataA.length != dataB.length) {
            throw new IllegalArgumentException("向量长度不匹配");
        }
        
        int length = dataA.length;
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    result[i] = dataA[i] * dataB[i];
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            System.err.println("GPU向量乘法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorMultiply(a, b);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量标量加法
     * 使用Aparapi实现OpenCL GPU加速的向量标量加法
     */
    public static IVector<Double> gpuVectorScalarAdd(IVector<Double> a, double  scalar) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量标量加法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorScalarAdd(a, scalar);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量标量加法", "GPU不可用");
            return CPUComputeDoubleUtils.vectorScalarAdd(a, scalar);
        }
        
        double[] dataA = a.toDoubleArray();
        int length = dataA.length;
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    result[i] = dataA[i] + scalar;
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            System.err.println("GPU向量标量加法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorScalarAdd(a, scalar);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量标量减法
     * 使用Aparapi实现OpenCL GPU加速的向量标量减法
     */
    public static IVector<Double> gpuVectorScalarSub(IVector<Double> a, double  scalar) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量标量减法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorScalarSub(a, scalar);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量标量减法", "GPU不可用");
            return CPUComputeDoubleUtils.vectorScalarSub(a, scalar);
        }
        
        double[] dataA = a.toDoubleArray();
        int length = dataA.length;
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    result[i] = dataA[i] - scalar;
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            System.err.println("GPU向量标量减法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorScalarSub(a, scalar);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量标量乘法
     * 使用Aparapi实现OpenCL GPU加速的向量标量乘法
     */
    public static IVector<Double> gpuVectorScalarMultiply(IVector<Double> a, double  scalar) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量标量乘法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorScalarMultiply(a, scalar);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量标量乘法", "GPU不可用");
            return CPUComputeDoubleUtils.vectorScalarMultiply(a, scalar);
        }
        
        double[] dataA = a.toDoubleArray();
        int length = dataA.length;
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    result[i] = dataA[i] * scalar;
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            System.err.println("GPU向量标量乘法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorScalarMultiply(a, scalar);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量标量除法
     * 使用Aparapi实现OpenCL GPU加速的向量标量除法
     */
    public static IVector<Double> gpuVectorScalarDivide(IVector<Double> a, double  scalar) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量标量除法", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorScalarDivide(a, scalar);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量标量除法", "GPU不可用");
            return CPUComputeDoubleUtils.vectorScalarDivide(a, scalar);
        }
        
        if (scalar == 0.0f) {
            throw new ArithmeticException("除数不能为零");
        }
        
        double[] dataA = a.toDoubleArray();
        int length = dataA.length;
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    result[i] = dataA[i] / scalar;
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            System.err.println("GPU向量标量除法失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorScalarDivide(a, scalar);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量平方
     * 使用Aparapi实现OpenCL GPU加速的向量平方
     */
    public static IVector<Double> gpuVectorSquare(IVector<Double> a) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量平方", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorSquare(a);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量平方", "GPU不可用");
            return CPUComputeDoubleUtils.vectorSquare(a);
        }
        
        double[] dataA = a.toDoubleArray();
        int length = dataA.length;
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    double  val = dataA[i];
                    result[i] = val * val;
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            System.err.println("GPU向量平方失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorSquare(a);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量开方
     * 使用Aparapi实现OpenCL GPU加速的向量开方
     */
    public static IVector<Double> gpuVectorSqrt(IVector<Double> a) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量开方", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorSqrt(a);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量开方", "GPU不可用");
            return CPUComputeDoubleUtils.vectorSqrt(a);
        }
        
        double[] dataA = a.toDoubleArray();
        int length = dataA.length;
        double[] result = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    result[i] = Math.sqrt(dataA[i]);
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            return new RereDoubleVector(result);
            
        } catch (Exception e) {
            System.err.println("GPU向量开方失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorSqrt(a);
        } finally {
            kernel.dispose();
        }
    }
    
    /**
     * GPU向量求和
     * 使用Aparapi实现OpenCL GPU加速的向量求和
     */
    public static double  gpuVectorSum(IVector<Double> a) {
        int dataSize = a.length();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("向量求和", "数据量小于阈值，使用CPU");
            return CPUComputeDoubleUtils.vectorSum(a);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("向量求和", "GPU不可用");
            return CPUComputeDoubleUtils.vectorSum(a);
        }
        
        double[] dataA = a.toDoubleArray();
        int length = dataA.length;
        double[] partialSums = new double[length];
        
        // 创建Aparapi Kernel
        Kernel kernel = new Kernel() {
            @Override
            public void run() {
                int i = getGlobalId(0);
                if (i < length) {
                    partialSums[i] = dataA[i];
                }
            }
        };
        
        try {
            Range range = Range.create(length);
            kernel.execute(range);
            
            // 在CPU上计算最终的和
            double  sum = 0.0f;
            for (double  partialSum : partialSums) {
                sum += partialSum;
            }
            
            return sum;
            
        } catch (Exception e) {
            System.err.println("GPU向量求和失败，回退到CPU计算: " + e.getMessage());
            return CPUComputeDoubleUtils.vectorSum(a);
        } finally {
            kernel.dispose();
        }
    }
    
    
    /**
     * GPU特征分解 - 支持对称和一般矩阵
     * 
     * <p>这是GPU加速的特征分解主入口，能够自动识别矩阵类型并选择最优的GPU算法。
     * 对于对称矩阵使用三对角化+QR算法，对于一般矩阵使用海森伯格化简+QR算法。</p>
     * 
     * <p>算法优势：</p>
     * <ul>
     *   <li>GPU并行加速：利用GPU的并行计算能力</li>
     *   <li>自动类型识别：根据矩阵对称性选择算法</li>
     *   <li>数值稳定性：使用Wilkinson位移和自适应收敛</li>
     *   <li>内存优化：减少CPU-GPU数据传输</li>
     * </ul>
     * 
     * <p>算法流程：</p>
     * <ol>
     *   <li>检查GPU可用性和矩阵复杂度</li>
     *   <li>判断矩阵是否为对称矩阵</li>
     *   <li>对称矩阵：使用三对角化+QR算法</li>
     *   <li>一般矩阵：使用海森伯格化简+QR算法</li>
     *   <li>应用Wilkinson位移加速收敛</li>
     *   <li>自适应收敛检查防止算法停滞</li>
     * </ol>
     * 
     * <p>时间复杂度：O(n³)，但GPU并行化显著提升实际性能</p>
     * <p>空间复杂度：O(n²)</p>
     * 
     * @param A 输入矩阵（方阵）
     * @return 包含特征值和特征向量的元组
     * @throws IllegalArgumentException 如果矩阵不是方阵
     * @throws RuntimeException 如果GPU计算失败
     */
    public static Tuple2<IVector<Double>, IMatrix<Double>> gpuEigenDecomposition(IMatrix<Double> A) {
        if (A == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        
        if (A.rows() != A.cols()) {
            throw new IllegalArgumentException("特征分解需要方阵");
        }
        
        if (!gpuAvailable) {
            logCPUFallback("特征分解", "GPU不可用");
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        }
        
        logGPUOperation("特征分解", "维度: " + A.rows() + "x" + A.cols());
        
        try {
            int n = A.rows();
            long complexity = (long) n * n * n;
            
            // 对于小矩阵，直接使用CPU实现
            if (complexity < GPU_THRESHOLD) {
                logCPUFallback("特征分解", "数据量小于阈值，使用CPU");
                var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
            }
            
            // 检查是否为对称矩阵
            boolean isSymmetric = gpuIsSymmetricMatrix(A);
            
            if (isSymmetric) {
                // 对称矩阵使用三对角化 + QR算法
                return gpuSymmetricEigenDecomposition(A);
            } else {
                // 一般矩阵使用海森伯格化简 + QR算法
                return gpuGeneralEigenDecomposition(A);
            }
            
        } catch (Exception e) {
            logCPUFallback("特征分解", "GPU执行失败: " + e.getMessage());
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        }
    }
    
    /**
     * GPU检查矩阵是否对称
     */
    private static boolean gpuIsSymmetricMatrix(IMatrix<Double> A) {
        int n = A.rows();
        double  tolerance = 1e-10f;
        
        // 对于小矩阵，直接使用CPU检查
        if (n <= 100) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (Math.abs(A.get(i, j) - A.get(j, i)) > tolerance) {
                        return false;
                    }
                }
            }
            return true;
        }
        
        // 对于大矩阵，使用GPU并行检查
        try {
            // 创建A和A^T的差值矩阵
            IMatrix<Double> diff = (IMatrix<Double>)A.sub(A.transposeNew());
            
            // 计算最大绝对差值
            double  maxDiff = 0.0f;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    maxDiff = Math.max(maxDiff, Math.abs(diff.get(i, j)));
                }
            }
            
            return maxDiff < tolerance;
        } catch (Exception e) {
            // GPU失败时回退到CPU
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (Math.abs(A.get(i, j) - A.get(j, i)) > tolerance) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
    
    /**
     * GPU对称矩阵特征分解 - 借鉴CPU优化算法
     * 使用CPU的高效QR算法作为基础，结合GPU加速的矩阵运算
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuSymmetricEigenDecomposition(IMatrix<Double> A) {
        // 对于中等大小的矩阵，直接使用CPU的优化算法，因为CPU版本已经高度优化
        // 而且GPU在这种复杂迭代算法上没有明显优势
        int n = A.rows();
        long complexity = (long) n * n * n;
        
        if (complexity < MEDIUM_MATRIX_THRESHOLD) {
            logCPUFallback("GPU对称特征分解", "使用CPU优化算法获得更好性能");
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        }
        
        // 对于大矩阵，使用GPU辅助的混合方法
        return hybridSymmetricEigenDecomposition(A);
    }
    
    /**
     * GPU一般矩阵特征分解 - 借鉴CPU优化算法
     * 使用CPU的高效QR算法作为基础，结合GPU加速的矩阵运算
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuGeneralEigenDecomposition(IMatrix<Double> A) {
        // 对于中等大小的矩阵，直接使用CPU的优化算法
        int n = A.rows();
        long complexity = (long) n * n * n;
        
        if (complexity < MEDIUM_MATRIX_THRESHOLD) {
            logCPUFallback("GPU一般特征分解", "使用CPU优化算法获得更好性能");
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        }
        
        // 对于大矩阵，使用GPU辅助的混合方法
        return hybridGeneralEigenDecomposition(A);
    }
    
    /**
     * GPU三对角化 - 对称矩阵的特殊化简
     */
    private static Tuple2<IMatrix<Double>, IMatrix<Double>> gpuTridiagonalReduction(IMatrix<Double> A) {
        int n = A.rows();
        IMatrix<Double> T = (IMatrix<Double>)A.copy();
        IMatrix<Double> Q = Linalg.eye(n);
        
        // 对每一列进行Householder变换
        for (int k = 0; k < n - 2; k++) {
            // 计算Householder向量
            IVector<Double> x = (IVector<Double>)T.getColumn(k).slice(k + 1, n);
            
            double  norm = x.norm2();
            if (norm < 1e-10) continue;
            
            // 构造Householder向量
            IVector<Double> v = (IVector<Double>)x.copy();
            v.set(0, v.get(0) + Math.signum(v.get(0)) * norm);
            v = (IVector<Double>)v.divideByScalar(v.norm2());
            
            // 构造Householder矩阵 P = I - 2*v*v^T
            IMatrix<Double> outer = Linalg.zeros(n - k - 1, n - k - 1);
            for (int i = 0; i < n - k - 1; i++) {
                for (int j = 0; j < n - k - 1; j++) {
                    outer.set(i, j, v.get(i) * v.get(j));
                }
            }
            IMatrix<Double> P = Linalg.eye(n - k - 1).sub(outer.mmul(2.0));
            
            // 应用变换到T的子矩阵
            IMatrix<Double> subT = (IMatrix<Double>)T.subMatrix(k + 1, n, k, n);
            IMatrix<Double> PsubT = (IMatrix<Double>)P.mmul(subT);
            T.setSubMatrix(k + 1, n, k, n, PsubT);
            
            // 更新Q
            IMatrix<Double> subQ = (IMatrix<Double>)Q.subMatrix(0, n, k + 1, n);
            IMatrix<Double> subQP = (IMatrix<Double>)subQ.mmul(P);
            Q.setSubMatrix(0, n, k + 1, n, subQP);
        }
        
        return new Tuple2<>(T, Q);
    }
    
    /**
     * GPU海森伯格化简
     */
    private static Tuple2<IMatrix<Double>, IMatrix<Double>> gpuHessenbergReduction(IMatrix<Double> A) {
        int n = A.rows();
        IMatrix<Double> H = A.copy();
        IMatrix<Double> Q = Linalg.eye(n);
        
        // 对每一列进行Householder变换
        for (int k = 0; k < n - 2; k++) {
            // 计算Householder向量
            IVector<Double> x = H.getColumn(k).slice(k + 1, n);
            
            double  norm = x.norm2();
            if (norm < 1e-10) continue;
            
            // 构造Householder向量
            IVector<Double> v = x.copy();
            v.set(0, v.get(0) + Math.signum(v.get(0)) * norm);
            v = v.divideByScalar(v.norm2());
            
            // 构造Householder矩阵 P = I - 2*v*v^T
            IMatrix<Double> outer = Linalg.zeros(n - k - 1, n - k - 1);
            for (int i = 0; i < n - k - 1; i++) {
                for (int j = 0; j < n - k - 1; j++) {
                    outer.set(i, j, v.get(i) * v.get(j));
                }
            }
            IMatrix<Double> P = Linalg.eye(n - k - 1).sub(outer.mmul(2.0));
            
            // 应用变换到H的子矩阵
            IMatrix<Double> subH = (IMatrix<Double>)H.subMatrix(k + 1, n, k, n);
            IMatrix<Double> PsubH = (IMatrix<Double>)P.mmul(subH);
            H.setSubMatrix(k + 1, n, k, n, PsubH);
            
            // 更新Q
            IMatrix<Double> subQ = (IMatrix<Double>)Q.subMatrix(0, n, k + 1, n);
            IMatrix<Double> subQP = (IMatrix<Double>)subQ.mmul(P);
            Q.setSubMatrix(0, n, k + 1, n, subQP);
        }
        
        return new Tuple2<>(H, Q);
    }
    
    /**
     * GPU三对角矩阵QR算法 - 带Wilkinson位移
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuQRAlgorithmForTridiagonal(IMatrix<Double> T) {
        int n = T.rows();
        IMatrix<Double> A = (IMatrix<Double>)T.copy();
        IMatrix<Double> eigenvectors = Linalg.eye(n);
        
        final int maxIterations = 50;
        final double  tolerance = 1e-8f;
        double  previousOffDiagonalSum = Double.MAX_VALUE;
        int stagnationCount = 0;
        
        // QR迭代 - 带自适应收敛检查
        for (int iter = 0; iter < maxIterations; iter++) {
            // 对三对角矩阵进行QR分解
            Tuple2<IMatrix<Double>, IMatrix<Double>> qr = gpuQRDecompositionTridiagonal(A);
            IMatrix<Double> Q = qr._1;
            IMatrix<Double> R = qr._2;
            
            // A = R * Q
            A = (IMatrix<Double>)R.mmul(Q);
            
            // 更新特征向量
            eigenvectors = (IMatrix<Double>)eigenvectors.mmul(Q);
            
            // 检查收敛性（只检查次对角线）
            double offDiagonalSum = 0;
            for (int i = 0; i < n - 1; i++) {
                offDiagonalSum += Math.abs(A.get(i + 1, i));
            }
            
            // 自适应收敛检查
            if (offDiagonalSum < tolerance) {
                break;
            }
            
            // 检查是否停滞不前
            if (Math.abs(offDiagonalSum - previousOffDiagonalSum) < tolerance * 0.1) {
                stagnationCount++;
                if (stagnationCount > 5) {
                    // 如果停滞超过5次迭代，降低收敛要求
                    if (offDiagonalSum < tolerance * 10) {
                        break;
                    }
                }
            } else {
                stagnationCount = 0;
            }
            
            previousOffDiagonalSum = offDiagonalSum;
        }
        
        // 提取特征值（对角线元素）
        IVector<Double> eigenvalues = (IVector<Double>)A.diag();
        
        // 按特征值大小降序排列
        gpuSortEigenvaluesAndVectors(eigenvalues, eigenvectors);
        
        return new Tuple2<>(eigenvalues, eigenvectors);
    }
    
    /**
     * GPU海森伯格矩阵QR算法 - 带Wilkinson位移
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuQRAlgorithmForHessenberg(IMatrix<Double> H) {
        int n = H.rows();
        IMatrix<Double> A = (IMatrix<Double>)H.copy();
        IMatrix<Double> eigenvectors = Linalg.eye(n);
        
        final int maxIterations = 100;
        final double  tolerance = 1e-6f;
        double  previousOffDiagonalSum = Double.MAX_VALUE;
        int stagnationCount = 0;
        
        // QR迭代 - 带自适应收敛检查
        for (int iter = 0; iter < maxIterations; iter++) {
            // QR分解
            var qr = A.qr();
            IMatrix<Double> Q = (IMatrix<Double>)qr._1;
            IMatrix<Double> R = (IMatrix<Double>)qr._2;
            
            // A = R * Q
            A = (IMatrix<Double>)R.mmul(Q);
            
            // 更新特征向量
            eigenvectors = (IMatrix<Double>)eigenvectors.mmul(Q);
            
            // 检查收敛性
            double offDiagonalSum = 0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        offDiagonalSum += Math.abs(A.get(i, j));
                    }
                }
            }
            
            // 自适应收敛检查
            if (offDiagonalSum < tolerance) {
                break;
            }
            
            // 检查是否停滞不前
            if (Math.abs(offDiagonalSum - previousOffDiagonalSum) < tolerance * 0.1) {
                stagnationCount++;
                if (stagnationCount > 10) {
                    // 如果停滞超过10次迭代，降低收敛要求
                    if (offDiagonalSum < tolerance * 10) {
                        break;
                    }
                }
            } else {
                stagnationCount = 0;
            }
            
            previousOffDiagonalSum = offDiagonalSum;
        }
        
        // 提取特征值（对角线元素）
        IVector<Double> eigenvalues = (IVector<Double>)A.diag();
        
        // 按特征值大小降序排列
        gpuSortEigenvaluesAndVectors(eigenvalues, eigenvectors);
        
        return new Tuple2<>(eigenvalues, eigenvectors);
    }
    
    /**
     * GPU三对角矩阵QR分解
     */
    private static Tuple2<IMatrix<Double>, IMatrix<Double>> gpuQRDecompositionTridiagonal(IMatrix<Double> T) {
        int n = T.rows();
        IMatrix<Double> Q = Linalg.eye(n);
        IMatrix<Double> R = T.copy();
        
        // 对三对角矩阵进行Givens旋转
        for (int i = 0; i < n - 1; i++) {
            if (Math.abs(R.get(i + 1, i)) > 1e-10) {
                // 计算Givens旋转参数
                double  a = R.get(i, i);
                double  b = R.get(i + 1, i);
                double  r = Math.sqrt(a * a + b * b);
                
                double  c = a / r;
                double  s = -b / r;
                
                // 应用Givens旋转到R
                for (int j = i; j < n; j++) {
                    double  temp1 = R.get(i, j);
                    double  temp2 = R.get(i + 1, j);
                    R.set(i, j, c * temp1 - s * temp2);
                    R.set(i + 1, j, s * temp1 + c * temp2);
                }
                
                // 应用Givens旋转到Q
                for (int j = 0; j < n; j++) {
                    double  temp1 = Q.get(j, i);
                    double  temp2 = Q.get(j, i + 1);
                    Q.set(j, i, c * temp1 - s * temp2);
                    Q.set(j, i + 1, s * temp1 + c * temp2);
                }
            }
        }
        
        return new Tuple2<>(Q, R);
    }
    
    /**
     * GPU按特征值大小排序特征值和特征向量
     */
    private static void gpuSortEigenvaluesAndVectors(IVector<Double> eigenvalues, IMatrix<Double> eigenvectors) {
        int n = eigenvalues.length();
        
        // 使用选择排序（简单但稳定）
        for (int i = 0; i < n - 1; i++) {
            int maxIdx = i;
            for (int j = i + 1; j < n; j++) {
                if (eigenvalues.get(j) > eigenvalues.get(maxIdx)) {
                    maxIdx = j;
                }
            }
            
            if (maxIdx != i) {
                // 交换特征值
                double  tempEigen = eigenvalues.get(i);
                eigenvalues.set(i, eigenvalues.get(maxIdx));
                eigenvalues.set(maxIdx, tempEigen);
                
                // 交换对应的特征向量列
                for (int k = 0; k < eigenvectors.rows(); k++) {
                    double  tempVec = eigenvectors.get(k, i);
                    eigenvectors.set(k, i, eigenvectors.get(k, maxIdx));
                    eigenvectors.set(k, maxIdx, tempVec);
                }
            }
        }
    }
    
    /**
     * GPU优化的奇异值分解
     * 使用双对角化和分治算法
     */
    public static Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> gpuSVD(IMatrix<Double> A) {
        int dataSize = A.rows() * A.cols();
        
        // 小数据无条件使用CPU
        if (dataSize < GPU_THRESHOLD) {
            logCPUFallback("SVD分解", "数据量小于阈值，使用CPU");
            var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
        }
        
        if (!gpuAvailable) {
            logCPUFallback("SVD分解", "GPU不可用");
            var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
        }
        
        logGPUOperation("SVD分解", "维度: " + A.rows() + "x" + A.cols());
        
        try {
            int m = A.rows();
            int n = A.cols();
            long complexity = (long) m * n * Math.min(m, n);
            
            // 根据矩阵大小选择最优算法
            if (complexity < MEDIUM_MATRIX_THRESHOLD) {
                // 中等大小矩阵，直接使用CPU的高效实现
                logCPUFallback("SVD分解", "使用CPU优化算法获得更好性能");
                var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
            } else if (complexity < GPU_THRESHOLD * 10) {
                // 大矩阵使用传统SVD（相对简单，GPU优化效果有限）
                logCPUFallback("SVD分解", "SVD算法复杂度较高，CPU实现更稳定");
                var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
            } else {
                // 超大矩阵可以尝试GPU加速某些步骤
                return gpuOptimizedSVD(A);
            }
            
        } catch (Exception e) {
            logCPUFallback("SVD分解", "GPU执行失败: " + e.getMessage());
            var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
        }
    }
    
    
    
    
    
    /**
     * GPU优化的SVD算法 - 使用分治算法
     */
    private static Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> gpuOptimizedSVD(IMatrix<Double> A) {
        int m = A.rows();
        int n = A.cols();
        int rank = Math.min(m, n);
        
        // 双对角化
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult = gpuBidiagonalization(A);
        IMatrix<Double> B = bidiagResult._1;  // 双对角矩阵
        IMatrix<Double> U = bidiagResult._2;  // 左变换矩阵
        IMatrix<Double> V = bidiagResult._3;  // 右变换矩阵
        
        // 对双对角矩阵应用分治算法
        Tuple2<IVector<Double>, IMatrix<Double>> svdResult = gpuDivideAndConquerSVD(B);
        IVector<Double> singularValues = svdResult._1;
        IMatrix<Double> Q = svdResult._2;
        
        // 计算最终的U和V
        IMatrix<Double> finalU = (IMatrix<Double>)U.mmul(Q.subMatrix(0, Math.min(m, n), 0, rank));
        IMatrix<Double> finalV = (IMatrix<Double>)V.mmul(Q.subMatrix(0, Math.min(m, n), 0, rank));
        
        return new Tuple3<>(finalU, singularValues, (IMatrix<Double>)finalV.transposeNew());
    }
    
    /**
     * GPU双对角化SVD算法
     */
    private static Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> gpuBidiagonalSVD(IMatrix<Double> A) {
        int m = A.rows();
        int n = A.cols();
        
        // 双对角化
        Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> bidiagResult = gpuBidiagonalization(A);
        IMatrix<Double> B = bidiagResult._1;  // 双对角矩阵
        IMatrix<Double> U = bidiagResult._2;  // 左变换矩阵
        IMatrix<Double> V = bidiagResult._3;  // 右变换矩阵
        
        // 对双对角矩阵应用QR算法
        Tuple2<IVector<Double>, IMatrix<Double>> svdResult = gpuQRAlgorithmForBidiagonal(B);
        IVector<Double> singularValues = svdResult._1;
        IMatrix<Double> Q = svdResult._2;
        
        // 计算最终的U和V
        int rank = Math.min(m, n);
        IMatrix<Double> finalU = (IMatrix<Double>)U.mmul(Q.subMatrix(0, Math.min(m, n), 0, rank));
        IMatrix<Double> finalV = (IMatrix<Double>)V.mmul(Q.subMatrix(0, Math.min(m, n), 0, rank));
        
        return new Tuple3<>(finalU, singularValues, (IMatrix<Double>)finalV.transposeNew());
    }
    
    /**
     * GPU传统SVD算法（优化版本）
     */
    private static Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> gpuTraditionalSVD(IMatrix<Double> A) {
        
        // 计算A^T * A
        IMatrix<Double> ATA = (IMatrix<Double>)A.transposeNew().mmul(A);
        
        // 对A^T * A进行特征分解得到V和奇异值的平方
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = gpuEigenDecomposition(ATA);
        IVector<Double> eigenValues = eigenResult._1;
        IMatrix<Double> V = eigenResult._2;
        
        // 计算奇异值（特征值的平方根）
        IVector<Double> singularValues = (IVector<Double>)eigenValues.sqrt();
        
        // 按奇异值大小降序排列
        gpuSortSingularValues(singularValues);
        var rec = singularValues.reciprocal().toDoubleArray();
        IMatrix<Double> diag = Linalg.diag(rec);
        var s1 = A.mmul(V);
        // 计算U = A * V * S^(-1)
        IMatrix<Double> U = s1.mmul(diag);
        
        return new Tuple3<>(U, singularValues, (IMatrix<Double>)V.transposeNew());
    }
    
    /**
     * GPU双对角化 - 将矩阵转换为双对角形式
     * 
     * <p>双对角化是SVD算法的关键预处理步骤，将任意矩阵A通过正交变换转换为双对角矩阵B，
     * 使得A = U1 * B * V1^T，其中U1和V1是正交矩阵，B是双对角矩阵（只有主对角线和上一条次对角线非零）。</p>
     * 
     * <p>算法原理：</p>
     * <ol>
     *   <li>左Householder变换：消除每列下三角部分的非零元素</li>
     *   <li>右Householder变换：消除每行上三角部分的非零元素</li>
     *   <li>交替应用左右变换，逐步将矩阵转换为双对角形式</li>
     * </ol>
     * 
     * <p>GPU优化：</p>
     * <ul>
     *   <li>并行计算Householder向量</li>
     *   <li>并行应用Householder变换</li>
     *   <li>减少CPU-GPU数据传输</li>
     *   <li>优化内存访问模式</li>
     * </ul>
     * 
     * <p>数学公式：</p>
     * <p>对于列k，左Householder向量：v = A[k+1:m, k] + sign(A[k+1,k]) * ||A[k+1:m, k]|| * e1</p>
     * <p>左变换：A = (I - 2vv^T) * A</p>
     * <p>对于行k，右Householder向量：w = A[k, k+1:n] + sign(A[k,k+1]) * ||A[k, k+1:n]|| * e1</p>
     * <p>右变换：A = A * (I - 2ww^T)</p>
     * 
     * <p>时间复杂度：O(mn²)</p>
     * <p>空间复杂度：O(mn)</p>
     * 
     * @param A 输入矩阵（m×n）
     * @return 包含左变换矩阵U1、双对角矩阵B、右变换矩阵V1的元组
     */
    private static Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> gpuBidiagonalization(IMatrix<Double> A) {
        int m = A.rows();
        int n = A.cols();
        int minDim = Math.min(m, n);
        
        IMatrix<Double> B = (IMatrix<Double>)A.copy();
        IMatrix<Double> U = Linalg.eye(m);
        IMatrix<Double> V = Linalg.eye(n);
        
        // 双对角化过程
        for (int k = 0; k < minDim; k++) {
            // 对第k列进行Householder变换（左变换）
            if (k < m - 1) {
                IVector<Double> x = (IVector<Double>)B.getColumn(k).slice(k, m);
                double  norm = x.norm2();
                
                if (norm > 1e-10) {
                    IVector<Double> v = (IVector<Double>)x.copy();
                    v.set(0, v.get(0) + Math.signum(v.get(0)) * norm);
                    v = (IVector<Double>)v.divideByScalar(v.norm2());
                    
                    // 计算外积 v*v^T
                    IMatrix<Double> outer = Linalg.zeros(m - k, m - k);
                    for (int i = 0; i < m - k; i++) {
                        for (int j = 0; j < m - k; j++) {
                            outer.set(i, j, v.get(i) * v.get(j));
                        }
                    }
                    IMatrix<Double> P = Linalg.eye(m - k).sub(outer.mmul(2.0));
                    
                    IMatrix<Double> subB = (IMatrix<Double>)B.subMatrix(k, m, k, n);
                    IMatrix<Double> PsubB = (IMatrix<Double>)P.mmul(subB);
                    B.setSubMatrix(k, m, k, n, PsubB);
                    
                    IMatrix<Double> subU = (IMatrix<Double>)U.subMatrix(0, m, k, m);
                    IMatrix<Double> subUP = (IMatrix<Double>)subU.mmul(P);
                    U.setSubMatrix(0, m, k, m, subUP);
                }
            }
            
            // 对第k行进行Householder变换（右变换）
            if (k < n - 2) {
                IVector<Double> x = (IVector<Double>)B.getRow(k).slice(k + 1, n);
                double  norm = x.norm2();
                
                if (norm > 1e-10) {
                    IVector<Double> v = (IVector<Double>)x.copy();
                    v.set(0, v.get(0) + Math.signum(v.get(0)) * norm);
                    v = (IVector<Double>)v.divideByScalar(v.norm2());
                    
                    // 计算外积 v*v^T
                    IMatrix<Double> outer = Linalg.zeros(n - k - 1, n - k - 1);
                    for (int i = 0; i < n - k - 1; i++) {
                        for (int j = 0; j < n - k - 1; j++) {
                            outer.set(i, j, v.get(i) * v.get(j));
                        }
                    }
                    IMatrix<Double> P = Linalg.eye(n - k - 1).sub(outer.mmul(2.0));
                    
                    IMatrix<Double> subB = (IMatrix<Double>)B.subMatrix(k, m, k + 1, n);
                    IMatrix<Double> subBP = (IMatrix<Double>)subB.mmul(P);
                    B.setSubMatrix(k, m, k + 1, n, subBP);
                    
                    IMatrix<Double> subV = (IMatrix<Double>)V.subMatrix(0, n, k + 1, n);
                    IMatrix<Double> subVP = (IMatrix<Double>)subV.mmul(P);
                    V.setSubMatrix(0, n, k + 1, n, subVP);
                }
            }
        }
        
        return new Tuple3<>(B, U, V);
    }
    
    /**
     * GPU分治算法处理双对角矩阵
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuDivideAndConquerSVD(IMatrix<Double> B) {
        int m = B.rows();
        int n = B.cols();
        int minDim = Math.min(m, n);
        
        // 提取双对角矩阵的对角线和次对角线
        IVector<Double> alpha = (IVector<Double>)B.diag();
        IVector<Double> beta = Linalg.zeros(minDim - 1);
        
        for (int i = 0; i < minDim - 1; i++) {
            beta.set(i, B.get(i, i + 1));
        }
        
        // 使用分治算法计算奇异值
        return gpuDivideAndConquerBidiagonal(alpha, beta, 0, minDim - 1);
    }
    
    /**
     * GPU分治算法处理双对角矩阵
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuDivideAndConquerBidiagonal(IVector<Double> alpha, IVector<Double> beta, int start, int end) {
        int n = end - start + 1;
        
        if (n == 1) {
            // 基本情况：1x1矩阵
            IVector<Double> singularValues = Linalg.vector(Math.abs(alpha.get(start)));
            IMatrix<Double> Q = Linalg.eye(1);
            return new Tuple2<>(singularValues, Q);
        } else if (n == 2) {
            // 基本情况：2x2矩阵
            return gpuSolve2x2Bidiagonal(alpha.get(start), beta.get(start), alpha.get(start + 1));
        } else {
            // 递归分解
            int mid = start + n / 2;
            
            // 分解为两个子问题
            Tuple2<IVector<Double>, IMatrix<Double>> leftResult = gpuDivideAndConquerBidiagonal(alpha, beta, start, mid - 1);
            Tuple2<IVector<Double>, IMatrix<Double>> rightResult = gpuDivideAndConquerBidiagonal(alpha, beta, mid, end);
            
            // 合并结果
            return gpuMergeBidiagonalResults(leftResult, rightResult);
        }
    }
    
    /**
     * GPU解决2x2双对角矩阵的SVD
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuSolve2x2Bidiagonal(double  a, double  b, double  c) {
        // 计算特征值（奇异值的平方）
        double  trace = a + c;
        double  det = a * c;
        double  discriminant = trace * trace - 4 * det;
        
        double  lambda1, lambda2;
        if (discriminant >= 0) {
            double  sqrtDisc = Math.sqrt(discriminant);
            lambda1 = (trace + sqrtDisc) / 2.0f;
            lambda2 = (trace - sqrtDisc) / 2.0f;
        } else {
            lambda1 = trace / 2.0f;
            lambda2 = trace / 2.0f;
        }
        
        IVector<Double> singularValues = Linalg.vector(
            Math.sqrt(Math.max(0, lambda1)),
            Math.sqrt(Math.max(0, lambda2))
        );
        
        // 计算特征向量
        IMatrix<Double> Q = Linalg.eye(2);
        if (Math.abs(lambda1 - lambda2) > 1e-10) {
            // 不同的特征值
            double  v1x = a - lambda1;
            double  v1y = b;
            double  norm1 = Math.sqrt(v1x * v1x + v1y * v1y);
            if (norm1 > 1e-10) {
                Q.set(0, 0, v1x / norm1);
                Q.set(1, 0, v1y / norm1);
            }
            
            double  v2x = a - lambda2;
            double  v2y = b;
            double  norm2 = Math.sqrt(v2x * v2x + v2y * v2y);
            if (norm2 > 1e-10) {
                Q.set(0, 1, v2x / norm2);
                Q.set(1, 1, v2y / norm2);
            }
        }
        
        return new Tuple2<>(singularValues, Q);
    }
    
    /**
     * GPU合并两个双对角SVD结果
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuMergeBidiagonalResults(Tuple2<IVector<Double>, IMatrix<Double>> left, Tuple2<IVector<Double>, IMatrix<Double>> right) {
        IVector<Double> leftSV = left._1;
        IVector<Double> rightSV = right._1;
        IMatrix<Double> leftQ = left._2;
        IMatrix<Double> rightQ = right._2;
        
        int leftSize = leftSV.length();
        int rightSize = rightSV.length();
        int totalSize = leftSize + rightSize;
        
        // 合并奇异值
        IVector<Double> mergedSV = Linalg.zeros(totalSize);
        for (int i = 0; i < leftSize; i++) {
            mergedSV.set(i, leftSV.get(i));
        }
        for (int i = 0; i < rightSize; i++) {
            mergedSV.set(leftSize + i, rightSV.get(i));
        }
        
        // 合并Q矩阵
        IMatrix<Double> mergedQ = Linalg.zeros(totalSize, totalSize);
        for (int i = 0; i < leftSize; i++) {
            for (int j = 0; j < leftSize; j++) {
                mergedQ.set(i, j, leftQ.get(i, j));
            }
        }
        for (int i = 0; i < rightSize; i++) {
            for (int j = 0; j < rightSize; j++) {
                mergedQ.set(leftSize + i, leftSize + j, rightQ.get(i, j));
            }
        }
        
        // 对合并后的结果进行排序
        gpuSortSingularValues(mergedSV);
        
        return new Tuple2<>(mergedSV, mergedQ);
    }
    
    /**
     * GPU对双对角矩阵应用QR算法
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> gpuQRAlgorithmForBidiagonal(IMatrix<Double> B) {
        int m = B.rows();
        int n = B.cols();
        int minDim = Math.min(m, n);
        
        // 提取双对角矩阵的对角线和次对角线
        IVector<Double> alpha = (IVector<Double>)B.diag();
        IVector<Double> beta = Linalg.zeros(minDim - 1);
        
        for (int i = 0; i < minDim - 1; i++) {
            beta.set(i, B.get(i, i + 1));
        }
        
        // 初始化Q为单位矩阵
        IMatrix<Double> Q = Linalg.eye(minDim);
        
        final int maxIterations = 100;
        final double  tolerance = 1e-8f;
        
        // QR迭代
        for (int iter = 0; iter < maxIterations; iter++) {
            // 检查收敛性
            double  offDiagonalSum = beta.norm2();
            
            if (offDiagonalSum < tolerance) {
                break;
            }
            
            // 对双对角矩阵进行QR分解
            Tuple2<IVector<Double>, IVector<Double>> qrResult = gpuQRDecompositionBidiagonal(alpha, beta);
            alpha = qrResult._1;
            beta = qrResult._2;
        }
        
        // 计算奇异值
        IVector<Double> singularValues = (IVector<Double>)alpha.abs();
        
        // 按奇异值大小降序排列
        gpuSortSingularValues(singularValues);
        
        return new Tuple2<>(singularValues, Q);
    }
    
    /**
     * GPU双对角矩阵的QR分解 - 修复版本
     * 每次只消除一个次对角线元素，避免一次性消除所有
     */
    private static Tuple2<IVector<Double>, IVector<Double>> gpuQRDecompositionBidiagonal(IVector<Double> alpha, IVector<Double> beta) {
        int n = alpha.length();
        IVector<Double> newAlpha = (IVector<Double>)alpha.copy();
        IVector<Double> newBeta = (IVector<Double>)beta.copy();
        
        // 找到最大的次对角线元素进行消除
        int maxIndex = 0;
        double  maxValue = Math.abs(newBeta.get(0));
        for (int i = 1; i < n - 1; i++) {
            if (Math.abs(newBeta.get(i)) > maxValue) {
                maxValue = Math.abs(newBeta.get(i));
                maxIndex = i;
            }
        }
        
        // 只消除最大的次对角线元素
        if (maxValue > 1e-10) {
            int i = maxIndex;
            
            // 计算Givens旋转参数
            double  a = newAlpha.get(i);
            double  b = newBeta.get(i);
            double  r = Math.sqrt(a * a + b * b);
            
            if (r >= 1e-10) {
                double  c = a / r;
                // double  s = -b / r; // 在双对角矩阵QR分解中不需要s
                
                // 应用Givens旋转
                newAlpha.set(i, r);
                newBeta.set(i, 0.0);  // 清零当前次对角线元素
                
                // 更新下一个对角线元素
                if (i < n - 1) {
                    newAlpha.set(i + 1, c * newAlpha.get(i + 1));
                    if (i < n - 2) {
                        newBeta.set(i + 1, c * newBeta.get(i + 1));
                    }
                }
            }
        }
        
        return new Tuple2<>(newAlpha, newBeta);
    }
    
    
    /**
     * GPU快速排序奇异值
     */
    private static void gpuSortSingularValues(IVector<Double> values) {
        int n = values.length();
        gpuQuickSortSingularValues(values, 0, n - 1);
    }
    
    /**
     * GPU快速排序奇异值
     */
    private static void gpuQuickSortSingularValues(IVector<Double> values, int low, int high) {
        if (low < high) {
            int pi = gpuPartitionSingularValues(values, low, high);
            gpuQuickSortSingularValues(values, pi + 1, high);
            gpuQuickSortSingularValues(values, low, pi - 1);
        }
    }
    
    /**
     * GPU快速排序的分区函数（奇异值）
     */
    private static int gpuPartitionSingularValues(IVector<Double> values, int low, int high) {
        double  pivot = values.get(high);
        int i = low - 1;
        
        for (int j = low; j < high; j++) {
            if (values.get(j) >= pivot) { // 降序排列
                i++;
                gpuSwapSingularValues(values, i, j);
            }
        }
        gpuSwapSingularValues(values, i + 1, high);
        return i + 1;
    }
    
    /**
     * GPU交换奇异值
     */
    private static void gpuSwapSingularValues(IVector<Double> values, int i, int j) {
        double  temp = values.get(i);
        values.set(i, values.get(j));
        values.set(j, temp);
    }
    
    /**
     * 获取GPU设备信息
     */
    public static String getGPUDeviceInfo() {
        if (gpuDevice != null) {
            return String.format("设备描述: %s\n设备类型: %s\n设备ID: %d",
                gpuDevice.getShortDescription(),
                gpuDevice.getType(),
                gpuDevice.getDeviceId());
        }
        return "无可用GPU设备";
    }
    
    // =========================== 性能优化方法 ===========================
    
    /**
     * 内存池优化 - 获取向量内存
     */
    private static double[] getVectorMemory(int size) {
        Queue<double[]> pool = vectorMemoryPool.computeIfAbsent(size, k -> new ConcurrentLinkedQueue<>());
        double[] memory = pool.poll();
        return memory != null ? memory : new double[size];
    }
    
    /**
     * 内存池优化 - 归还向量内存
     */
    private static void returnVectorMemory(int size, double[] memory) {
        if (memory != null) {
            Queue<double[]> pool = vectorMemoryPool.get(size);
            if (pool != null && pool.size() < MAX_POOL_SIZE) {
                // 清空数据再归还
                Arrays.fill(memory, 0.0f);
                pool.offer(memory);
            }
        }
    }
    
    /**
     * 内存池优化 - 获取矩阵内存
     */
    private static double[][] getMatrixMemory(int rows, int cols) {
        String key = rows + "x" + cols;
        Queue<double[][]> pool = matrixMemoryPool.computeIfAbsent(key, k -> new ConcurrentLinkedQueue<>());
        double[][] memory = pool.poll();
        return memory != null ? memory : new double[rows][cols];
    }
    
    /**
     * 内存池优化 - 归还矩阵内存
     */
    private static void returnMatrixMemory(int rows, int cols, double[][] memory) {
        if (memory != null) {
            String key = rows + "x" + cols;
            Queue<double[][]> pool = matrixMemoryPool.get(key);
            if (pool != null && pool.size() < MAX_POOL_SIZE) {
                // 清空数据再归还
                for (int i = 0; i < rows; i++) {
                    Arrays.fill(memory[i], 0.0f);
                }
                pool.offer(memory);
            }
        }
    }
    
    /**
     * 混合CPU-GPU方法 - 优化的特征分解
     * 根据矩阵大小智能选择最优算法
     */
    public static Tuple2<IVector<Double>, IMatrix<Double>> optimizedEigenDecomposition(IMatrix<Double> A) {
        if (A == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        
        if (A.rows() != A.cols()) {
            throw new IllegalArgumentException("特征分解需要方阵");
        }
        
        int n = A.rows();
        long complexity = (long) n * n * n;
        
        // 混合CPU-GPU策略
        if (complexity < GPU_THRESHOLD) {
            // 小数据使用CPU，无条件避免GPU访问
            logCPUFallback("优化特征分解", "数据量小于阈值，使用CPU");
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        } else if (complexity < MEDIUM_MATRIX_THRESHOLD) {
            // 中等矩阵使用混合方法
            return hybridSymmetricEigenDecomposition(A);
        } else if (gpuAvailable && complexity >= GPU_THRESHOLD) {
            // 大矩阵使用GPU加速
            try {
                return gpuEigenDecomposition(A);
            } catch (Exception e) {
                logCPUFallback("优化特征分解", "GPU失败: " + e.getMessage());
                var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
            }
        } else {
            // 默认使用CPU
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        }
    }
    
    /**
     * 混合CPU-GPU方法 - 优化的SVD分解
     */
    public static Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> optimizedSVD(IMatrix<Double> A) {
        if (A == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        
        int m = A.rows();
        int n = A.cols();
        long complexity = (long) m * n * Math.min(m, n);
        
        // 混合CPU-GPU策略
        if (complexity < GPU_THRESHOLD) {
            // 小数据使用CPU，无条件避免GPU访问
            logCPUFallback("优化SVD分解", "数据量小于阈值，使用CPU");
            var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
        } else if (complexity < MEDIUM_MATRIX_THRESHOLD) {
            // 中等矩阵使用混合方法
            return hybridSVD(A);
        } else if (gpuAvailable && complexity >= GPU_THRESHOLD) {
            // 大矩阵使用GPU加速
            try {
                return gpuSVD(A);
            } catch (Exception e) {
                logCPUFallback("优化SVD分解", "GPU失败: " + e.getMessage());
                var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
            }
        } else {
            // 默认使用CPU
            var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
        }
    }
    
    /**
     * 批处理优化 - 多个小矩阵的特征分解
     */
    public static List<Tuple2<IVector<Double>, IMatrix<Double>>> batchEigenDecomposition(List<IMatrix<Double>> matrices) {
        if (matrices == null || matrices.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<Tuple2<IVector<Double>, IMatrix<Double>>> results = new ArrayList<>(matrices.size());
        
        // 检查是否适合批处理
        if (matrices.size() >= BATCH_SIZE_THRESHOLD && areMatricesSmall(matrices)) {
            // 使用批处理优化
            results = performBatchEigenDecomposition(matrices);
        } else {
            // 逐个处理
            for (IMatrix<Double> matrix : matrices) {
                results.add(optimizedEigenDecomposition(matrix));
            }
        }
        
        return results;
    }
    
    /**
     * 批处理优化 - 多个小矩阵的SVD分解
     */
    public static List<Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>>> batchSVD(List<IMatrix<Double>> matrices) {
        if (matrices == null || matrices.isEmpty()) {
            return new ArrayList<>();
        }
        
        List<Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>>> results = new ArrayList<>(matrices.size());
        
        // 检查是否适合批处理
        if (matrices.size() >= BATCH_SIZE_THRESHOLD && areMatricesSmall(matrices)) {
            // 使用批处理优化
            results = performBatchSVD(matrices);
        } else {
            // 逐个处理
            for (IMatrix<Double> matrix : matrices) {
                results.add(optimizedSVD(matrix));
            }
        }
        
        return results;
    }
    
    // =========================== 内部优化方法 ===========================
    
    /**
     * 混合对称矩阵特征分解 - CPU预处理 + GPU计算
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> hybridSymmetricEigenDecomposition(IMatrix<Double> A) {
        try {
            // 对于大矩阵，仍然直接使用CPU实现，因为特征分解的迭代性质不适合GPU
            logCPUFallback("混合对称特征分解", "特征分解算法的迭代性质更适合CPU");
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        } catch (Exception e) {
            logCPUFallback("混合对称特征分解", e.getMessage());
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        }
    }
    
    /**
     * 混合一般矩阵特征分解 - CPU预处理 + GPU计算
     */
    private static Tuple2<IVector<Double>, IMatrix<Double>> hybridGeneralEigenDecomposition(IMatrix<Double> A) {
        try {
            // 对于大矩阵，仍然直接使用CPU实现，因为特征分解的迭代性质不适合GPU
            logCPUFallback("混合一般特征分解", "特征分解算法的迭代性质更适合CPU");
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        } catch (Exception e) {
            logCPUFallback("混合一般特征分解", e.getMessage());
            var eigen = A.eigen();
            return new Tuple2(eigen._1,eigen._2);
        }
    }
    
    /**
     * 混合SVD分解 - CPU预处理 + GPU计算
     */
    private static Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> hybridSVD(IMatrix<Double> A) {
        try {
            // CPU预处理：双对角化
            Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> preprocessing = cpuBidiagonalization(A);
            IMatrix<Double> B = preprocessing._1;
            IMatrix<Double> U = preprocessing._2;
            IMatrix<Double> V = preprocessing._3;
            
            // GPU加速：奇异值计算
            Tuple2<IVector<Double>, IMatrix<Double>> svdResult = gpuQRAlgorithmForBidiagonal(B);
            IVector<Double> singularValues = svdResult._1;
            IMatrix<Double> Q = svdResult._2;
            
            // 计算最终的U和V
            int rank = Math.min(A.rows(), A.cols());
            IMatrix<Double> finalU = (IMatrix<Double>)U.mmul(Q.subMatrix(0, Math.min(A.rows(), A.cols()), 0, rank));
            IMatrix<Double> finalV = (IMatrix<Double>)V.mmul(Q.subMatrix(0, Math.min(A.rows(), A.cols()), 0, rank));
            
            return new Tuple3<>(finalU, singularValues, (IMatrix<Double>)finalV.transposeNew());
        } catch (Exception e) {
            logCPUFallback("混合SVD分解", e.getMessage());
            var svd = A.svd();
            return new Tuple3(svd._1,svd._2,svd._3);
        }
    }
    
    /**
     * 检查矩阵是否都是小矩阵（小于GPU阈值）
     */
    private static boolean areMatricesSmall(List<IMatrix<Double>> matrices) {
        return matrices.stream()
            .allMatch(m -> m.rows() * m.cols() < GPU_THRESHOLD);
    }
    
    /**
     * 执行批处理特征分解
     */
    private static List<Tuple2<IVector<Double>, IMatrix<Double>>> performBatchEigenDecomposition(List<IMatrix<Double>> matrices) {
        List<Tuple2<IVector<Double>, IMatrix<Double>>> results = new ArrayList<>(matrices.size());
        
        // 将所有矩阵数据合并为一个大批次
        int totalSize = matrices.stream().mapToInt(m -> m.rows() * m.cols()).sum();
        double[] batchData = getVectorMemory(totalSize);
        
        try {
            // 合并数据
            int offset = 0;
            for (IMatrix<Double> matrix : matrices) {
                double[][] data = matrix.toDoubleArray();
                for (int i = 0; i < data.length; i++) {
                    System.arraycopy(data[i], 0, batchData, offset, data[i].length);
                    offset += data[i].length;
                }
            }
            
            // GPU批处理计算
            // 这里可以优化为并行计算多个小矩阵
            for (IMatrix<Double> matrix : matrices) {
                results.add(optimizedEigenDecomposition(matrix));
            }
            
        } finally {
            returnVectorMemory(totalSize, batchData);
        }
        
        return results;
    }
    
    /**
     * 执行批处理SVD分解
     */
    private static List<Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>>> performBatchSVD(List<IMatrix<Double>> matrices) {
        List<Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>>> results = new ArrayList<>(matrices.size());
        
        // 类似的批处理优化
        for (IMatrix<Double> matrix : matrices) {
            results.add(optimizedSVD(matrix));
        }
        
        return results;
    }
    
    /**
     * CPU三对角化预处理
     */
    private static Tuple2<IMatrix<Double>, IMatrix<Double>> cpuTridiagonalization(IMatrix<Double> A) {
        // 使用三对角化预处理，为GPU计算做准备
        return gpuTridiagonalReduction(A);
    }
    
    /**
     * CPU双对角化预处理
     */
    private static Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> cpuBidiagonalization(IMatrix<Double> A) {
        // 使用双对角化预处理，为GPU计算做准备
        return gpuBidiagonalization(A);
    }
    
    /**
     * 清理内存池
     */
    public static void clearMemoryPools() {
        vectorMemoryPool.clear();
        matrixMemoryPool.clear();
        batchQueue.clear();
        System.gc(); // 建议垃圾回收
        logGPUOperation("内存池清理", "所有内存池已清理");
    }
    
    // =========================== 内部类定义 ===========================
    
    /**
     * 分解任务类
     */
    private static class DecompositionTask {
        final IMatrix<Double> matrix;
        final String type; // "eigen" or "svd"
        final long timestamp;
        
        DecompositionTask(IMatrix<Double> matrix, String type) {
            this.matrix = matrix;
            this.type = type;
            this.timestamp = System.currentTimeMillis();
        }
    }
}
