package com.yishape.lab.math.compute;

import static com.yishape.lab.math.compute.gpu.GpuReduce.PROD;
import com.yishape.lab.math.compute.hpc.HpcGemm;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import static com.yishape.lab.math.compute.ops.BinaryOperation.ADD;
import static com.yishape.lab.math.compute.ops.BinaryOperation.DIVIDE;
import static com.yishape.lab.math.compute.ops.BinaryOperation.MULTIPLY;
import static com.yishape.lab.math.compute.ops.BinaryOperation.REMAINDER;
import static com.yishape.lab.math.compute.ops.BinaryOperation.SUBTRACT;
import com.yishape.lab.math.compute.ops.BinaryReduceOperation;
import static com.yishape.lab.math.compute.ops.BinaryReduceOperation.DOT;
import static com.yishape.lab.math.compute.ops.BinaryReduceOperation.L1_NORM;
import static com.yishape.lab.math.compute.ops.BinaryReduceOperation.L2_NORM;
import com.yishape.lab.math.compute.ops.LogicalCompare;
import static com.yishape.lab.math.compute.ops.LogicalCompare.EQUALS;
import static com.yishape.lab.math.compute.ops.LogicalCompare.GREATER_THAN;
import static com.yishape.lab.math.compute.ops.LogicalCompare.GREATER_THAN_OR_EQUALS;
import static com.yishape.lab.math.compute.ops.LogicalCompare.LESS_THAN;
import static com.yishape.lab.math.compute.ops.LogicalCompare.LESS_THAN_OR_EQUALS;
import static com.yishape.lab.math.compute.ops.LogicalCompare.NOT_EQUALS;
import com.yishape.lab.math.compute.ops.LogicalOperation;
import static com.yishape.lab.math.compute.ops.LogicalOperation.AND;
import static com.yishape.lab.math.compute.ops.LogicalOperation.NOT;
import static com.yishape.lab.math.compute.ops.LogicalOperation.OR;
import static com.yishape.lab.math.compute.ops.LogicalOperation.XOR;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MAX;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MEAN;
import static com.yishape.lab.math.compute.ops.ReduceOperation.MIN;
import static com.yishape.lab.math.compute.ops.ReduceOperation.STANDARD_DEVIATION;
import static com.yishape.lab.math.compute.ops.ReduceOperation.SUM;
import static com.yishape.lab.math.compute.ops.ReduceOperation.VARIANCE;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import static com.yishape.lab.math.compute.ops.UniversalOperation.ABS;
import static com.yishape.lab.math.compute.ops.UniversalOperation.ACOS;
import static com.yishape.lab.math.compute.ops.UniversalOperation.ASIN;
import static com.yishape.lab.math.compute.ops.UniversalOperation.ATAN;
import static com.yishape.lab.math.compute.ops.UniversalOperation.CBRT;
import static com.yishape.lab.math.compute.ops.UniversalOperation.COS;
import static com.yishape.lab.math.compute.ops.UniversalOperation.COSH;
import static com.yishape.lab.math.compute.ops.UniversalOperation.EXP;
import static com.yishape.lab.math.compute.ops.UniversalOperation.EXPM1;
import static com.yishape.lab.math.compute.ops.UniversalOperation.GELU;
import static com.yishape.lab.math.compute.ops.UniversalOperation.LOG;
import static com.yishape.lab.math.compute.ops.UniversalOperation.LOG10;
import static com.yishape.lab.math.compute.ops.UniversalOperation.LOG1P;
import static com.yishape.lab.math.compute.ops.UniversalOperation.POW;
import static com.yishape.lab.math.compute.ops.UniversalOperation.RELU;
import static com.yishape.lab.math.compute.ops.UniversalOperation.SIGMOID;
import static com.yishape.lab.math.compute.ops.UniversalOperation.SIN;
import static com.yishape.lab.math.compute.ops.UniversalOperation.SINH;
import static com.yishape.lab.math.compute.ops.UniversalOperation.SQRT;
import static com.yishape.lab.math.compute.ops.UniversalOperation.TAN;
import static com.yishape.lab.math.compute.ops.UniversalOperation.TANH;
import com.yishape.lab.math.linalg.RereDoubleMatrix;

import com.yishape.lab.util.YishapeConfig;

import com.yishape.lab.util.YishapeLogger;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD运算 使用Java Vector API实现
 * 针对长度1000+的向量进行深度优化
 *
 * @author lteb2
 */
public class SIMDDoubleComputer implements IDoubleVectorComputer,Serializable {

    private static final YishapeLogger log = YishapeLogger.getLogger(SIMDDoubleComputer.class);

    /** ThreadLocal pool of scratch double[half][half] buffers for Strassen recursion. */
    private static final ThreadLocal<Deque<double[][]>> STRASSEN_POOL =
        ThreadLocal.withInitial(ArrayDeque::new);

    private static double[][] acquireScratch(int d) {
        Deque<double[][]> pool = STRASSEN_POOL.get();
        double[][] buf = pool.poll();
        if (buf != null && buf.length == d && buf[0].length == d) {
            for (int i = 0; i < d; i++) Arrays.fill(buf[i], 0.0);
            return buf;
        }
        return new double[d][d];
    }

    private static void releaseScratch(double[][] buf) {
        if (buf == null) return;
        Deque<double[][]> pool = STRASSEN_POOL.get();
        if (pool.size() < 16) pool.push(buf);
    }

    // 性能优化常量
    private static final VectorSpecies<Double> PREFERRED_SPECIES;
    private static final int VECTOR_LENGTH;
    private static final int OPTIMAL_BLOCK_SIZE;
    private static final int CACHE_LINE_SIZE = 64; // 典型缓存行大小
    private static final int L1_CACHE_SIZE = 32 * 1024; // 典型L1缓存大小
    private static final int L2_CACHE_SIZE = 256 * 1024; // 典型L2缓存大小

    // 性能监控 — 系统属性 + YishapeConfig profile 桥接
    private static boolean isPerformanceMonitoring() {
        return YishapeConfig.isSimdPerformanceMonitoring();
    }
    private static boolean isDetailedLogging() {
        return YishapeConfig.isSimdDetailedLogging();
    }

    // Strassen矩阵乘法常量
    /** 低于此尺寸使用标准分块乘法 */
    private static final int STRASSEN_THRESHOLD = 2048;
    /** 复杂度阈值 (m*n*p)，超过时才启用Strassen */
    private static final long STRASSEN_COMPLEXITY_THRESHOLD = 8_589_934_592L; // 2048³
    /** 系统属性开关 */
    private static final boolean USE_STRASSEN = !"false".equals(
        System.getProperty("yishape.strassen.enabled", "true"));

    // 性能统计 (AtomicLong for thread-safe increment)
    private static final AtomicLong totalOperations = new AtomicLong(0);
    private static final AtomicLong totalVectorizedOperations = new AtomicLong(0);
    private static final AtomicLong totalScalarFallbacks = new AtomicLong(0);

    static {
        // 智能选择最优向量规格
        PREFERRED_SPECIES = selectOptimalSpecies();
        VECTOR_LENGTH = PREFERRED_SPECIES.length();
        OPTIMAL_BLOCK_SIZE = calculateOptimalBlockSize();
    }
    
    /**
     * 智能选择最优向量规格。
     * 直接使用 SPECIES_PREFERRED — JDK 根据实际 CPU 特性选择最佳宽度，
     * 避免在无 AVX-512 硬件上误用软件模拟的 512-bit 向量导致性能急剧退化。
     */
    private static VectorSpecies<Double> selectOptimalSpecies() {
        return DoubleVector.SPECIES_PREFERRED;
    }
    
    /**
     * 计算最优块大小
     */
    private static int calculateOptimalBlockSize() {
        // 基于L1缓存大小和向量长度计算最优块大小
        int elementsInL1 = L1_CACHE_SIZE / (Double.BYTES * 2); // 考虑读写两个数组
        int blockSize = Math.max(64, elementsInL1 / 4); // 保留1/4的L1缓存空间
        
        // 确保块大小是向量长度的倍数
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;
        int vectorLen = species.length();
        return ((blockSize + vectorLen - 1) / vectorLen) * vectorLen;
    }
    
    /**
     * 根据数据大小动态计算展开因子
     * 优化了展开因子以平衡性能和复杂度
     */
    private static int calculateUnrollFactor(int dataSize) {
        if (dataSize < 512) return 2;      // 小数据集使用最小展开
        if (dataSize < 2048) return 3;     // 中等数据集适度展开
        if (dataSize < 8192) return 4;     // 大数据集使用中等展开
        return 6; // 非常大的数据才使用较大展开因子
    }
    
    /**
     * 性能监控：记录操作统计
     */
    private static void recordOperation(boolean vectorized, int dataSize) {
        if (isPerformanceMonitoring()) {
            totalOperations.incrementAndGet();
            if (vectorized) {
                totalVectorizedOperations.incrementAndGet();
            } else {
                totalScalarFallbacks.incrementAndGet();
            }

            if (isDetailedLogging()) {
                long ops = totalOperations.get();
                if (ops % 10000 == 0) {
                    double vectorizationRate = (double) totalVectorizedOperations.get() / ops * 100;
                    log.debug(String.format("SIMD Stats: Operations=%d, Vectorization Rate=%.2f%%, Data Size=%d%n",
                        ops, vectorizationRate, dataSize));
                }
            }
        }
    }
    
    /**
     * 性能监控：获取性能统计
     */
    public static String getPerformanceStats() {
        if (!isPerformanceMonitoring()) {
            return "Performance monitoring is disabled. Enable with -Dsimd.performance.monitoring=true";
        }

        long ops = totalOperations.get();
        long vecOps = totalVectorizedOperations.get();
        long scalarOps = totalScalarFallbacks.get();
        double vectorizationRate = ops > 0 ? (double) vecOps / ops * 100 : 0;

        return String.format(
            "SIMD Performance Statistics:%n" +
            "Total Operations: %d%n" +
            "Vectorized Operations: %d%n" +
            "Scalar Fallbacks: %d%n" +
            "Vectorization Rate: %.2f%%%n" +
            "Optimal Vector Length: %d%n" +
            "Optimal Block Size: %d",
            ops, vecOps, scalarOps,
            vectorizationRate, VECTOR_LENGTH, OPTIMAL_BLOCK_SIZE
        );
    }
    
    /**
     * 改进的参数验证
     */
    private static void validateVectorInputs(double[] x1, double[] x2, String operationName) {
        if (x1 == null) {
            throw new IllegalArgumentException(String.format(
                "[%s] 第一个输入向量不能为null", operationName));
        }
        if (x2 == null) {
            throw new IllegalArgumentException(String.format(
                "[%s] 第二个输入向量不能为null", operationName));
        }
        if (x1.length != x2.length) {
            throw new IllegalArgumentException(String.format(
                "[%s] 向量长度不匹配: x1.length=%d, x2.length=%d",
                operationName, x1.length, x2.length));
        }

        // 检查数据合法性
        if (x1.length > Integer.MAX_VALUE / 8) {
            throw new IllegalArgumentException(String.format(
                "[%s] 向量太大，可能导致内存问题: length=%d",
                operationName, x1.length));
        }
    }

    /**
     * 为矩阵运算选择最优的向量species。
     * 至少需要能容纳4个元素才使用preferred，否则降级。
     */
    private static VectorSpecies<Double> selectMatrixSpecies() {
        return DoubleVector.SPECIES_PREFERRED;
    }
    
    /**
     * 单个向量的参数验证
     */
    private static void validateVectorInput(double[] x, String operationName) {
        if (x == null) {
            throw new IllegalArgumentException(String.format(
                "[%s] 输入向量不能为null", operationName));
        }
        
        if (x.length > Integer.MAX_VALUE / 8) {
            throw new IllegalArgumentException(String.format(
                "[%s] 向量太大，可能导致内存问题: length=%d", 
                operationName, x.length));
        }
    }

    /**
     * 检查是否支持Java Vector API计算（加载、运算、归约烟测；失败则回退 SISD）
     *
     * @return true if Vector API is supported, false otherwise
     */
    public static boolean checkIfSupport() {
        try {
            VectorSpecies<Double> species = PREFERRED_SPECIES;
            int vl = species.length();
            if (vl <= 0) {
                return false;
            }
            double[] a = new double[vl];
            double[] b = new double[vl];
            java.util.Arrays.fill(a, 1.0);
            java.util.Arrays.fill(b, 1.0);
            DoubleVector va = DoubleVector.fromArray(species, a, 0);
            DoubleVector vb = DoubleVector.fromArray(species, b, 0);
            DoubleVector vs = va.add(vb);
            double sum = vs.reduceLanes(VectorOperators.ADD);
            if (Double.isNaN(sum) || Double.isInfinite(sum)) {
                return false;
            }
            double expected = 2.0 * vl;
            return Math.abs(sum - expected) <= Math.ulp(expected) * 4;
        } catch (LinkageError | RuntimeException ex) {
            return false;
        }
    }

    /**
     * 基于Java Vector API计算向量二元运算
     * 针对长度1000+的向量进行深度优化
     *
     * @param x1 第一个向量
     * @param x2 第二个向量
     * @param operation 二元运算操作
     * @return 二元运算结果
     * @throws IllegalArgumentException 如果输入为null或长度不匹配
     */
    @Override
    public double[] binaryOperate(double[] x1, double[] x2, BinaryOperation operation) {
        // 改进的参数验证和性能监控
        validateVectorInputs(x1, x2, "binaryOperate(vector, vector)");
        recordOperation(true, x1.length);

        final int length = x1.length;
        final double[] result = new double[length];
        
        // 对于小数据集，使用简单的向量化
        if (length < VECTOR_LENGTH * 2) {
            return binaryOperateSmall(x1, x2, operation, result, length);
        }
        
        // 对于大数据集，使用优化的块处理
        return binaryOperateOptimized(x1, x2, operation, result, length);
    }
    
    /**
     * 小数据集的二元运算处理
     */
    private double[] binaryOperateSmall(double[] x1, double[] x2, BinaryOperation operation, 
                                       double[] result, int length) {
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x1, i);
            DoubleVector b = DoubleVector.fromArray(PREFERRED_SPECIES, x2, i);
            DoubleVector c = performBinaryOperation(a, b, operation);
            c.intoArray(result, i);
        }

        // 处理剩余元素
        performScalarRemainder(x1, x2, result, operation, i, length);
        return result;
    }
    
    /**
     * 大数据集的优化二元运算处理
     */
    private double[] binaryOperateOptimized(double[] x1, double[] x2, BinaryOperation operation, 
                                           double[] result, int length) {
        final int unrollFactor = calculateUnrollFactor(length);
        final int blockSize = unrollFactor * VECTOR_LENGTH;
        final int upperBound = (length / blockSize) * blockSize;
        
        int i = 0;
        
        // 循环展开处理主要数据
        for (; i < upperBound; i += blockSize) {
            performUnrolledBinaryOperation(x1, x2, result, operation, i, unrollFactor);
        }
        
        // 处理剩余的向量化部分
        int vectorBound = PREFERRED_SPECIES.loopBound(length);
        for (; i < vectorBound; i += VECTOR_LENGTH) {
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x1, i);
            DoubleVector b = DoubleVector.fromArray(PREFERRED_SPECIES, x2, i);
            DoubleVector c = performBinaryOperation(a, b, operation);
            c.intoArray(result, i);
        }
        
        // 处理最后的标量元素
        performScalarRemainder(x1, x2, result, operation, i, length);
        return result;
    }
    
    /**
     * 执行展开的二元运算
     */
    private void performUnrolledBinaryOperation(double[] x1, double[] x2, double[] result, 
                                              BinaryOperation operation, int startIndex, int unrollFactor) {
        // 根据展开因子动态处理
        for (int u = 0; u < unrollFactor; u++) {
            int idx = startIndex + u * VECTOR_LENGTH;
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x1, idx);
            DoubleVector b = DoubleVector.fromArray(PREFERRED_SPECIES, x2, idx);
            DoubleVector c = performBinaryOperation(a, b, operation);
            c.intoArray(result, idx);
        }
    }
    
    /**
     * 执行具体的二元运算
     */
    private DoubleVector performBinaryOperation(DoubleVector a, DoubleVector b, BinaryOperation operation) {
        return switch (operation) {
            case ADD -> a.add(b);
            case SUBTRACT -> a.sub(b);
            case MULTIPLY -> a.mul(b);
            case DIVIDE -> a.div(b);
            case REMAINDER -> {
                // Optimized remainder via D2L/L2D truncation (same approach as float SIMD)
                DoubleVector quotient = a.div(b);
                var longQuotient = quotient.convert(VectorOperators.D2L, 0);
                var truncated = longQuotient.convert(VectorOperators.L2D, 0);
                yield a.sub(truncated.mul(b));
            }
        };
    }

    /**
     * 处理标量剩余元素
     */
    private void performScalarRemainder(double[] x1, double[] x2, double[] result,
                                       BinaryOperation operation, int startIndex, int length) {
        for (int i = startIndex; i < length; i++) {
            result[i] = switch (operation) {
                case ADD -> x1[i] + x2[i];
                case SUBTRACT -> x1[i] - x2[i];
                case MULTIPLY -> x1[i] * x2[i];
                case DIVIDE -> x1[i] / x2[i];
                case REMAINDER -> x1[i] % x2[i];
            };
        }
    }

    /**
     * 基于Java Vector API计算向量与标量二元运算
     * 针对长度1000+的向量进行深度优化
     * 
     * @param x1 向量
     * @param x2 标量值
     * @param operation 二元运算操作
     * @return 二元运算结果
     * @throws IllegalArgumentException 如果输入向量为null
     */
    @Override
    public double[] binaryOperate(double[] x1, double x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        final int length = x1.length;
        final double[] result = new double[length];
        
        // 对于小数据集，使用简单的向量化
        if (length < VECTOR_LENGTH * 4) {
            return binaryOperateWithScalarSmall(x1, x2, operation, result, length);
        }
        
        // 对于大数据集，使用优化的块处理
        return binaryOperateWithScalarOptimized(x1, x2, operation, result, length);
    }
    
    /**
     * 小数据集的向量与标量二元运算处理
     */
    private double[] binaryOperateWithScalarSmall(double[] x1, double x2, BinaryOperation operation, 
                                                  double[] result, int length) {
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(length);
        DoubleVector scalarVector = DoubleVector.broadcast(PREFERRED_SPECIES, x2);

        // 简化的向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x1, i);
            DoubleVector c = performBinaryOperationWithScalar(a, scalarVector, x2, operation);
            c.intoArray(result, i);
        }

        // 处理剩余元素
        performScalarRemainderWithScalar(x1, x2, result, operation, i, length);
        return result;
    }
    
    /**
     * 大数据集的优化向量与标量二元运算处理
     * 简化了乘法操作的特殊处理，减少分支开销
     */
    private double[] binaryOperateWithScalarOptimized(double[] x1, double x2, BinaryOperation operation, 
                                                      double[] result, int length) {
        final int unrollFactor = Math.min(4, calculateUnrollFactor(length)); // 限制展开因子
        final int blockSize = unrollFactor * VECTOR_LENGTH;
        final int upperBound = (length / blockSize) * blockSize;
        final DoubleVector scalarVector = DoubleVector.broadcast(PREFERRED_SPECIES, x2);
        
        int i = 0;
        
        // 简化的循环展开处理，去除乘法特殊处理
        for (; i < upperBound; i += blockSize) {
            performUnrolledBinaryOperationWithScalar(x1, scalarVector, x2, result, operation, i, unrollFactor);
        }
        
        // 处理剩余的向量化部分
        int vectorBound = PREFERRED_SPECIES.loopBound(length);
        for (; i < vectorBound; i += VECTOR_LENGTH) {
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x1, i);
            DoubleVector c = performBinaryOperationWithScalar(a, scalarVector, x2, operation);
            c.intoArray(result, i);
        }
        
        // 处理最后的标量元素
        performScalarRemainderWithScalar(x1, x2, result, operation, i, length);
        return result;
    }
    
    /**
     * 执行展开的向量与标量二元运算
     */
    private void performUnrolledBinaryOperationWithScalar(double[] x1, DoubleVector scalarVector, double x2, 
                                                          double[] result, BinaryOperation operation, 
                                                          int startIndex, int unrollFactor) {
        for (int u = 0; u < unrollFactor; u++) {
            int idx = startIndex + u * VECTOR_LENGTH;
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x1, idx);
            DoubleVector c = performBinaryOperationWithScalar(a, scalarVector, x2, operation);
            c.intoArray(result, idx);
        }
    }
    
    /**
     * 执行具体的向量与标量二元运算
     */
    private DoubleVector performBinaryOperationWithScalar(DoubleVector a, DoubleVector scalarVector, 
                                                          double scalarValue, BinaryOperation operation) {
        return switch (operation) {
            case ADD -> a.add(scalarVector);
            case SUBTRACT -> a.sub(scalarVector);
            case MULTIPLY -> a.mul(scalarVector);
            case DIVIDE -> a.div(scalarVector);
            case REMAINDER -> {
                // Optimized remainder via D2L/L2D truncation (same approach as float SIMD)
                DoubleVector quotient = a.div(scalarVector);
                var longQuotient = quotient.convert(VectorOperators.D2L, 0);
                var truncated = longQuotient.convert(VectorOperators.L2D, 0);
                yield a.sub(truncated.mul(scalarVector));
            }
        };
    }
    
    /**
     * 处理向量与标量的标量剩余元素
     */
    private void performScalarRemainderWithScalar(double[] x1, double x2, double[] result, 
                                                  BinaryOperation operation, int startIndex, int length) {
        for (int i = startIndex; i < length; i++) {
            result[i] = switch (operation) {
                case ADD -> x1[i] + x2;
                case SUBTRACT -> x1[i] - x2;
                case MULTIPLY -> x1[i] * x2;
                case DIVIDE -> x1[i] / x2;
                case REMAINDER -> x1[i] % x2;
            };
        }
    }

    /**
     * 基于Java Vector API计算矩阵二元运算
     *
     * @param x1 第一个矩阵
     * @param x2 第二个矩阵
     * @param operation 二元运算操作
     * @return 二元运算结果
     * @throws IllegalArgumentException 如果输入为null或维度不匹配
     */
    @Override
    public double[][] binaryOperate(double[][] x1, double[][] x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("矩阵行数必须相同");
        }

        if (x1.length > 0) {
            int cols = x1[0].length;
            if (cols != x2[0].length) {
                throw new IllegalArgumentException("矩阵列数必须相同");
            }
            // 检查 ragged array：验证每行长度一致性
            for (int row = 0; row < x1.length; row++) {
                if (x1[row] == null || x1[row].length != cols) {
                    throw new IllegalArgumentException("矩阵列数必须相同 (第 " + row + " 行的 x1 异常)");
                }
                if (x2[row] == null || x2[row].length != cols) {
                    throw new IllegalArgumentException("矩阵列数必须相同 (第 " + row + " 行的 x2 异常)");
                }
            }
        }

        if (x1.length == 0) return new double[0][0];

        // 创建结果矩阵
        double[][] result = new double[x1.length][x1[0].length];

        // 对每一行进行向量化二元运算
        for (int row = 0; row < x1.length; row++) {
            result[row] = binaryOperate(x1[row], x2[row], operation);
        }

        return result;
    }

    /**
     * 基于Java Vector API计算矩阵与标量二元运算
     *
     * @param x1 矩阵
     * @param x2 标量值
     * @param operation 二元运算操作
     * @return 二元运算结果
     * @throws IllegalArgumentException 如果输入矩阵为null
     */
    @Override
    public double[][] binaryOperate(double[][] x1, double x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length == 0) return new double[0][0];

        // 创建结果矩阵
        double[][] result = new double[x1.length][x1[0].length];

        // 对每一行进行向量化二元运算
        for (int row = 0; row < x1.length; row++) {
            result[row] = binaryOperate(x1[row], x2, operation);
        }

        return result;
    }

    /**
     * 基于Java Vector API计算向量通用运算
     *
     * @param x 向量
     * @param operation 通用运算操作
     * @param additionalParam
     * @return 运算结果
     * @throws IllegalArgumentException 如果输入为null
     */
    @Override
    public double[] universalOperate(double[] x, UniversalOperation operation, double additionalParam) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;

        // 创建结果数组
        double[] result = new double[x.length];

        // 使用向量API进行通用运算
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(x.length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x, i);

            // 执行相应的通用运算
            DoubleVector c;
            switch (operation) {
                case EXP ->
                    c = a.lanewise(VectorOperators.EXP);
                case LOG ->
                    c = a.lanewise(VectorOperators.LOG);
                case LOG10 ->
                    c = a.lanewise(VectorOperators.LOG10);
                case SIN ->
                    c = a.lanewise(VectorOperators.SIN);
                case COS ->
                    c = a.lanewise(VectorOperators.COS);
                case TAN ->
                    c = a.lanewise(VectorOperators.TAN);
                case ASIN ->
                    c = a.lanewise(VectorOperators.ASIN);
                case ACOS ->
                    c = a.lanewise(VectorOperators.ACOS);
                case ATAN ->
                    c = a.lanewise(VectorOperators.ATAN);
                case SQRT ->
                    c = a.lanewise(VectorOperators.SQRT);
                case ABS ->
                    c = a.abs(); // Use the abs() method directly
                case POW ->
                    c = a.lanewise(VectorOperators.POW, additionalParam);
                case CBRT ->
                    c = a.lanewise(VectorOperators.CBRT);
                case COSH ->
                    c = a.lanewise(VectorOperators.COSH);
                case SINH ->
                    c = a.lanewise(VectorOperators.SINH);
                case TANH ->
                    c = a.lanewise(VectorOperators.TANH);
                case EXPM1 ->
                    c = a.lanewise(VectorOperators.EXPM1);
                case LOG1P ->
                    c = a.lanewise(VectorOperators.LOG1P);
                case RELU ->
                    c = a.max(0.0);
                case SIGMOID ->
                    c = a.mul(0.5).lanewise(VectorOperators.TANH).mul(0.5).add(0.5);
                case GELU -> {
                    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    DoubleVector xCubed = a.mul(a).mul(a);
                    DoubleVector inner = a.add(xCubed.mul(0.044715)).mul(0.7978845608028654);
                    DoubleVector tanhV = inner.lanewise(VectorOperators.TANH);
                    c = a.mul(0.5).mul(tanhV.add(1.0));
                }
                default ->
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }

            // 将结果存储回数组
            c.intoArray(result, i);
        }

        // 处理剩余元素
        for (; i < x.length; i++) {
            switch (operation) {
                case EXP:
                    result[i] = Math.exp(x[i]);
                    break;
                case LOG:
                    result[i] = Math.log(x[i]);
                    break;
                case LOG10:
                    result[i] = Math.log10(x[i]);
                    break;
                case SIN:
                    result[i] = Math.sin(x[i]);
                    break;
                case COS:
                    result[i] = Math.cos(x[i]);
                    break;
                case TAN:
                    result[i] = Math.tan(x[i]);
                    break;
                case ASIN:
                    result[i] = Math.asin(x[i]);
                    break;
                case ACOS:
                    result[i] = Math.acos(x[i]);
                    break;
                case ATAN:
                    result[i] = Math.atan(x[i]);
                    break;
                case SQRT:
                    result[i] = Math.sqrt(x[i]);
                    break;
                case ABS:
                    result[i] = Math.abs(x[i]);
                    break;
                case POW:
                    result[i] = Math.pow(x[i], additionalParam);
                    break;
                case CBRT:
                    result[i] = Math.cbrt(x[i]);
                    break;
                case COSH:
                    result[i] = Math.cosh(x[i]);
                    break;
                case SINH:
                    result[i] = Math.sinh(x[i]);
                    break;
                case TANH:
                    result[i] = Math.tanh(x[i]);
                    break;
                case EXPM1:
                    result[i] = Math.expm1(x[i]);
                    break;
                case LOG1P:
                    result[i] = Math.log1p(x[i]);
                    break;
                case RELU:
                    result[i] = Math.max(0.0, x[i]);
                    break;
                case SIGMOID:
                    result[i] = 0.5 * Math.tanh(x[i] * 0.5) + 0.5;
                    break;
                case GELU:
                    double xi = x[i];
                    result[i] = 0.5 * xi * (1.0 + Math.tanh(0.7978845608028654 * (xi + 0.044715 * xi * xi * xi)));
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }

        return result;
    }

    /**
     * 基于Java Vector API计算矩阵通用运算
     *
     * @param x 矩阵
     * @param operation 通用运算操作
     * @param additionalParam 额外参数（用于需要第二个参数的操作，如POW）
     * @return 运算结果
     * @throws IllegalArgumentException 如果输入为null
     */
    @Override
    public double[][] universalOperate(double[][] x, UniversalOperation operation, double additionalParam) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x.length == 0) return new double[0][0];

        // 创建结果矩阵
        double[][] result = new double[x.length][x[0].length];

        // 对每一行进行向量化通用运算
        for (int row = 0; row < x.length; row++) {
            result[row] = universalOperate(x[row], operation, additionalParam);
        }

        return result;
    }

    /**
     * 基于Java Vector API计算向量最小值操作
     *
     * @param x1 第一个向量
     * @param x2 第二个向量
     * @return 两个向量对应元素的最小值
     * @throws IllegalArgumentException 如果输入为null或长度不匹配
     */
    @Override
    public double[] elementWiseMin(double[] x1, double[] x2) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;

        // 创建结果数组
        double[] result = new double[x1.length];

        // 使用向量API进行最小值运算
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x1, i);
            DoubleVector b = DoubleVector.fromArray(PREFERRED_SPECIES, x2, i);

            // 执行向量最小值操作
            DoubleVector c = a.lanewise(VectorOperators.MIN, b);

            // 将结果存储回数组
            c.intoArray(result, i);
        }

        // 处理剩余元素
        for (; i < x1.length; i++) {
            result[i] = Math.min(x1[i], x2[i]);
        }

        return result;
    }

    /**
     * 基于Java Vector API计算向量最大值操作
     *
     * @param x1 第一个向量
     * @param x2 第二个向量
     * @return 两个向量对应元素的最大值
     * @throws IllegalArgumentException 如果输入为null或长度不匹配
     */
    @Override
    public double[] elementWiseMax(double[] x1, double[] x2) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;

        // 创建结果数组
        double[] result = new double[x1.length];

        // 使用向量API进行最大值运算
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x1, i);
            DoubleVector b = DoubleVector.fromArray(PREFERRED_SPECIES, x2, i);

            // 执行向量最大值操作
            DoubleVector c = a.lanewise(VectorOperators.MAX, b);

            // 将结果存储回数组
            c.intoArray(result, i);
        }

        // 处理剩余元素
        for (; i < x1.length; i++) {
            result[i] = Math.max(x1[i], x2[i]);
        }

        return result;
    }

    /**
     * 基于Java Vector API计算向量取反操作
     *
     * @param x 向量
     * @return 向量取反结果
     * @throws IllegalArgumentException 如果输入为null
     */
    @Override
    public double[] negate(double[] x) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;

        // 创建结果数组
        double[] result = new double[x.length];

        // 使用向量API进行取反运算
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(x.length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(PREFERRED_SPECIES, x, i);

            // 执行向量取反操作
            DoubleVector c = a.lanewise(VectorOperators.NEG);

            // 将结果存储回数组
            c.intoArray(result, i);
        }

        // 处理剩余元素
        for (; i < x.length; i++) {
            result[i] = -x[i];
        }

        return result;
    }

    /**
     * 基于Java Vector API计算矩阵最小值操作
     *
     * @param x1 第一个矩阵
     * @param x2 第二个矩阵
     * @return 两个矩阵对应元素的最小值
     * @throws IllegalArgumentException 如果输入为null或维度不匹配
     */
    @Override
    public double[][] elementWiseMin(double[][] x1, double[][] x2) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("矩阵行数必须相同");
        }

        if (x1.length > 0 && x1[0].length != x2[0].length) {
            throw new IllegalArgumentException("矩阵列数必须相同");
        }

        if (x1.length == 0) return new double[0][0];

        // 创建结果矩阵
        double[][] result = new double[x1.length][x1[0].length];

        // 对每一行进行向量化最小值操作
        for (int row = 0; row < x1.length; row++) {
            result[row] = elementWiseMin(x1[row], x2[row]);
        }

        return result;
    }

    /**
     * 基于Java Vector API计算矩阵最大值操作
     *
     * @param x1 第一个矩阵
     * @param x2 第二个矩阵
     * @return 两个矩阵对应元素的最大值
     * @throws IllegalArgumentException 如果输入为null或维度不匹配
     */
    @Override
    public double[][] elementWiseMax(double[][] x1, double[][] x2) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("矩阵行数必须相同");
        }

        if (x1.length > 0 && x1[0].length != x2[0].length) {
            throw new IllegalArgumentException("矩阵列数必须相同");
        }

        if (x1.length == 0) return new double[0][0];

        // 创建结果矩阵
        double[][] result = new double[x1.length][x1[0].length];

        // 对每一行进行向量化最大值操作
        for (int row = 0; row < x1.length; row++) {
            result[row] = elementWiseMax(x1[row], x2[row]);
        }

        return result;
    }

    /**
     * 基于Java Vector API计算矩阵取反操作
     *
     * @param x 矩阵
     * @return 矩阵取反结果
     * @throws IllegalArgumentException 如果输入为null
     */
    @Override
    public double[][] negate(double[][] x) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x.length == 0) return new double[0][0];

        // 创建结果矩阵
        double[][] result = new double[x.length][x[0].length];

        // 对每一行进行向量化取反操作
        for (int row = 0; row < x.length; row++) {
            result[row] = negate(x[row]);
        }

        return result;
    }

    /**
     * 基于Java Vector API计算向量归约操作
     *
     * @param x 向量
     * @param operation 归约操作类型
     * @return 归约操作结果
     * @throws IllegalArgumentException 如果输入为null或不支持的操作
     */
    @Override
    public double reduceOperate(double[] x, ReduceOperation operation) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // Empty array contract: return identity/default for each operation
        if (x.length == 0) {
            return switch (operation) {
                case SUM, MEAN, VARIANCE, STANDARD_DEVIATION -> 0.0;
                case PROD -> 1.0;
                case MAX -> Double.NEGATIVE_INFINITY;
                case MIN -> Double.POSITIVE_INFINITY;
            };
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;

        switch (operation) {
            case SUM -> {
                return reduceOperateSum(x, species);
            }

            case MEAN -> {
                // 计算总和并除以长度
                return reduceOperateSum(x, species) / x.length;
            }

            case MIN -> {
                return reduceOperateMinMax(x, species, true);
            }

            case MAX -> {
                return reduceOperateMinMax(x, species, false);
            }

            case VARIANCE -> {
                // 计算均值
                double mean = reduceOperate(x, ReduceOperation.MEAN);
                return reduceOperateVariance(x, species, mean);
            }

            case STANDARD_DEVIATION -> {
                return Math.sqrt(reduceOperate(x, ReduceOperation.VARIANCE));
            }

            case PROD -> {
                return reduceOperateProduct(x, species);
            }

            default ->
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    /**
     * 基于Java Vector API计算向量元素之和
     * 进一步简化策略，专注于最优性能
     *
     * @param x 向量
     * @param species 向量规格
     * @return 向量元素之和
     */
    private static double reduceOperateSum(double[] x, VectorSpecies<Double> species) {
        final int length = x.length;
        
        // 进一步简化：只对非常大的向量使用循环展开
        if (length < VECTOR_LENGTH * 16) {
            return reduceOperateSumSimple(x, species, length);
        }
        
        return reduceOperateSumOptimizedLight(x, species, length);
    }
    
    /**
     * 简单高效的向量求和实现
     */
    private static double reduceOperateSumSimple(double[] x, VectorSpecies<Double> species, int length) {
        int i = 0;
        int upperBound = species.loopBound(length);
        DoubleVector acc = DoubleVector.zero(species);
        DoubleVector compensation = DoubleVector.zero(species);

        for (; i < upperBound; i += species.length()) {
            DoubleVector a = DoubleVector.fromArray(species, x, i);
            DoubleVector y = a.sub(compensation);
            DoubleVector t = acc.add(y);
            compensation = t.sub(acc).sub(y);
            acc = t;
        }

        double sumResult = acc.reduceLanes(VectorOperators.ADD);

        for (; i < length; i++) {
            sumResult += x[i];
        }

        return sumResult;
    }
    
    /**
     * 轻量级优化的向量求和实现
     * 仅使用2个累加器以平衡性能和复杂度
     */
    private static double reduceOperateSumOptimizedLight(double[] x, VectorSpecies<Double> species, int length) {
        final int vectorLength = species.length();
        final int blockSize = vectorLength * 4; // 降低展开因子
        final int upperBound = (length / blockSize) * blockSize;
        
        // 使用2个累加器，避免过度复杂化
        DoubleVector acc1 = DoubleVector.zero(species);
        DoubleVector acc2 = DoubleVector.zero(species);
        
        int i = 0;
        
        // 简化的循环展开
        for (; i < upperBound; i += blockSize) {
            DoubleVector a1 = DoubleVector.fromArray(species, x, i);
            DoubleVector a2 = DoubleVector.fromArray(species, x, i + vectorLength);
            DoubleVector a3 = DoubleVector.fromArray(species, x, i + 2 * vectorLength);
            DoubleVector a4 = DoubleVector.fromArray(species, x, i + 3 * vectorLength);
            
            acc1 = acc1.add(a1).add(a3);
            acc2 = acc2.add(a2).add(a4);
        }
        
        // 合并累加器
        double sumResult = acc1.add(acc2).reduceLanes(VectorOperators.ADD);
        
        // 处理剩余的向量化部分
        int vectorBound = species.loopBound(length);
        for (; i < vectorBound; i += vectorLength) {
            DoubleVector a = DoubleVector.fromArray(species, x, i);
            sumResult += a.reduceLanes(VectorOperators.ADD);
        }
        
        // 处理最后的标量元素
        for (; i < length; i++) {
            sumResult += x[i];
        }

        return sumResult;
    }

    /**
     * 基于Java Vector API计算向量元素的最小值或最大值
     *
     * @param x 向量
     * @param species 向量规格
     * @param isMin 是否计算最小值（true为最小值，false为最大值）
     * @return 向量元素的最小值或最大值
     */
    private static double reduceOperateMinMax(double[] x, VectorSpecies<Double> species, boolean isMin) {
        // 使用向量API进行最小值或最大值运算
        int i = 0;
        int upperBound = species.loopBound(x.length);

        // 特殊处理：如果向量长度为0，直接返回
        if (x.length == 0) {
            return isMin ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }

        // 特殊处理：如果向量长度小于species.length，则直接使用标量计算
        if (x.length < species.length()) {
            double result = x[0];
            for (int j = 1; j < x.length; j++) {
                if (isMin) {
                    result = Math.min(result, x[j]);
                } else {
                    result = Math.max(result, x[j]);
                }
            }
            return result;
        }

        // 初始化结果向量为第一个元素
        DoubleVector resultVector = DoubleVector.fromArray(species, x, i);
        i += species.length();

        // 向量化处理 — 仍有余量时使用分治法减少比较次数
        if (i < upperBound) {
            DoubleVector acc = resultVector;

            while (i < upperBound) {
                int groupEnd = Math.min(i + species.length() * 4, upperBound);
                DoubleVector groupResult = DoubleVector.fromArray(species, x, i);
                i += species.length();

                while (i < groupEnd) {
                    DoubleVector a = DoubleVector.fromArray(species, x, i);
                    groupResult = groupResult.lanewise(isMin ? VectorOperators.MIN : VectorOperators.MAX, a);
                    i += species.length();
                }

                acc = acc.lanewise(isMin ? VectorOperators.MIN : VectorOperators.MAX, groupResult);
            }

            resultVector = acc;
        }

        // 直接使用向量归约方法，避免临时数组分配
        double result = resultVector.reduceLanes(isMin ? VectorOperators.MIN : VectorOperators.MAX);

        // 处理剩余元素
        for (; i < x.length; i++) {
            if (isMin) {
                result = Math.min(result, x[i]);
            } else {
                result = Math.max(result, x[i]);
            }
        }

        return result;
    }

    /**
     * 基于Java Vector API计算向量元素的方差
     * 使用Kahan求和提升数值稳定性
     *
     * @param x 向量
     * @param species 向量规格
     * @param mean 均值
     * @return 向量元素的方差
     */
    private static double reduceOperateVariance(double[] x, VectorSpecies<Double> species, double mean) {
        int i = 0;
        int upperBound = species.loopBound(x.length);

        // Kahan求和累加器
        DoubleVector acc = DoubleVector.zero(species);
        DoubleVector compensation = DoubleVector.zero(species);
        DoubleVector meanVector = DoubleVector.broadcast(species, mean);

        for (; i < upperBound; i += species.length()) {
            DoubleVector a = DoubleVector.fromArray(species, x, i);
            DoubleVector diff = a.sub(meanVector);
            DoubleVector squared = diff.mul(diff);

            // Kahan求和：y = squared - compensation, t = acc + y, compensation = (t - acc) - y
            DoubleVector y = squared.sub(compensation);
            DoubleVector t = acc.add(y);
            compensation = t.sub(acc).sub(y);
            acc = t;
        }

        double varResult = acc.reduceLanes(VectorOperators.ADD);

        for (; i < x.length; i++) {
            double diff = x[i] - mean;
            varResult += diff * diff;
        }

        return varResult / x.length;
    }

    /**
     * 基于Java Vector API计算向量元素的乘积
     *
     * @param x 向量
     * @param species 向量规格
     * @return 向量元素的乘积
     */
    private static double reduceOperateProduct(double[] x, VectorSpecies<Double> species) {
        // 使用向量API进行乘积运算
        int i = 0;
        int upperBound = species.loopBound(x.length);

        // 初始化累乘器向量为1
        DoubleVector acc = DoubleVector.broadcast(species, 1.0);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(species, x, i);

            // 累乘到累乘器
            acc = acc.mul(a);
        }

        // 归约累乘器中的值
        double productResult = acc.reduceLanes(VectorOperators.MUL);

        // 处理剩余元素
        for (; i < x.length; i++) {
            productResult *= x[i];
        }

        return productResult;
    }

    /**
     * 基于Java Vector API计算矩阵归约操作
     *
     * @param x 矩阵
     * @param operation 归约操作类型
     * @return 归约操作结果
     * @throws IllegalArgumentException 如果输入为null或不支持的操作
     */
    @Override
    public double reduceOperate(double[][] x, ReduceOperation operation) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // Empty matrix contract: return identity/default for each operation
        if (x.length == 0 || x[0].length == 0) {
            return switch (operation) {
                case SUM, MEAN, VARIANCE, STANDARD_DEVIATION -> 0.0;
                case PROD -> 1.0;
                case MAX -> Double.NEGATIVE_INFINITY;
                case MIN -> Double.POSITIVE_INFINITY;
            };
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = selectMatrixSpecies();

        switch (operation) {
            case SUM -> {
                double sumResult = 0.0;
                // 优化：使用向量化操作直接计算
                for (double[] row : x) {
                    sumResult += reduceOperateSum(row, species);
                }
                return sumResult;
            }

            case MEAN -> {
                double totalSum = 0.0;
                int totalElements = 0;
                // 优化：使用向量化操作直接计算
                for (double[] row : x) {
                    totalSum += reduceOperateSum(row, species);
                    totalElements += row.length;
                }
                return totalSum / totalElements;
            }

            case MIN -> {
                double minResult = Double.POSITIVE_INFINITY;
                // 优化：使用向量化操作直接计算
                for (double[] row : x) {
                    if (row.length > 0) {
                        // 只处理非空行
                        minResult = Math.min(minResult, reduceOperateMinMax(row, species, true));
                    }
                }
                return minResult;
            }

            case MAX -> {
                double maxResult = Double.NEGATIVE_INFINITY;
                // 优化：使用向量化操作直接计算
                for (double[] row : x) {
                    if (row.length > 0) {
                        // 只处理非空行
                        maxResult = Math.max(maxResult, reduceOperateMinMax(row, species, false));
                    }
                }
                return maxResult;
            }

            case VARIANCE -> {
                // 计算均值
                double mean = reduceOperate(x, ReduceOperation.MEAN);
                // 计算方差：reduceOperateVariance divides by row length internally,
                // so multiply back by row length to accumulate raw SSD, then divide by totalCount
                double varianceSum = 0.0;
                int totalCount = 0;
                for (double[] rowArray : x) {
                    varianceSum += reduceOperateVariance(rowArray, species, mean) * rowArray.length;
                    totalCount += rowArray.length;
                }
                return varianceSum / totalCount;
            }

            case STANDARD_DEVIATION -> {
                return Math.sqrt(reduceOperate(x, ReduceOperation.VARIANCE));
            }

            case PROD -> {
                double productResult = 1.0;
                // 优化：使用向量化操作直接计算
                for (double[] row : x) {
                    productResult *= reduceOperateProduct(row, species);
                }
                return productResult;
            }

            default ->
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    /**
     * 基于Java Vector API计算向量二元归约运算
     *
     * @param x1 第一个向量
     * @param x2 第二个向量
     * @param operation 二元归约运算操作
     * @return 二元归约运算结果
     * @throws IllegalArgumentException 如果输入为null或长度不匹配
     */
    @Override
    public double binaryReduceOperate(double[] x1, double[] x2, BinaryReduceOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = selectMatrixSpecies();

        switch (operation) {
            case DOT -> {
                // 直接计算点积，避免中间数组分配
                return directDotProduct(x1, x2, species);
            }

            case L2_NORM -> {
                // 直接计算L2范数，避免中间数组分配
                return Math.sqrt(directSumOfSquaredDifferences(x1, x2, species));
            }

            case L1_NORM -> {
                // 直接计算L1范数，避免中间数组分配
                return directSumOfAbsoluteDifferences(x1, x2, species);
            }

            default ->
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    /**
     * 直接计算两个向量的点积，避免中间数组分配
     */
    private static double directDotProduct(double[] x1, double[] x2, VectorSpecies<Double> species) {
        int i = 0;
        int upperBound = species.loopBound(x1.length);
        
        // 初始化累加器向量
        DoubleVector acc = DoubleVector.zero(species);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(species, x1, i);
            DoubleVector b = DoubleVector.fromArray(species, x2, i);

            // 累加到累加器
            acc = acc.add(a.mul(b));
        }

        // 归约累加器中的值
        double sumResult = acc.reduceLanes(VectorOperators.ADD);

        // 处理剩余元素
        for (; i < x1.length; i++) {
            sumResult += x1[i] * x2[i];
        }

        return sumResult;
    }

    /**
     * 直接计算两个向量差的平方和，避免中间数组分配
     */
    private static double directSumOfSquaredDifferences(double[] x1, double[] x2, VectorSpecies<Double> species) {
        int i = 0;
        int upperBound = species.loopBound(x1.length);
        
        // 初始化累加器向量
        DoubleVector acc = DoubleVector.zero(species);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(species, x1, i);
            DoubleVector b = DoubleVector.fromArray(species, x2, i);

            // 计算差值
            DoubleVector diff = a.sub(b);
            
            // 计算平方并累加到累加器
            acc = acc.add(diff.mul(diff));
        }

        // 归约累加器中的值
        double sumResult = acc.reduceLanes(VectorOperators.ADD);

        // 处理剩余元素
        for (; i < x1.length; i++) {
            double diff = x1[i] - x2[i];
            sumResult += diff * diff;
        }

        return sumResult;
    }

    /**
     * 直接计算两个向量差的绝对值和，避免中间数组分配
     */
    private static double directSumOfAbsoluteDifferences(double[] x1, double[] x2, VectorSpecies<Double> species) {
        int i = 0;
        int upperBound = species.loopBound(x1.length);
        
        // 初始化累加器向量
        DoubleVector acc = DoubleVector.zero(species);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(species, x1, i);
            DoubleVector b = DoubleVector.fromArray(species, x2, i);

            // 计算差值的绝对值并累加到累加器
            acc = acc.add(a.sub(b).abs());
        }

        // 归约累加器中的值
        double sumResult = acc.reduceLanes(VectorOperators.ADD);

        // 处理剩余元素
        for (; i < x1.length; i++) {
            sumResult += Math.abs(x1[i] - x2[i]);
        }

        return sumResult;
    }

    @Override
    public double binaryReduceOperate(double[][] x1, double[][] x2, BinaryReduceOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length != x2.length || (x1.length > 0 && x1[0].length != x2[0].length)) {
            throw new IllegalArgumentException("矩阵维度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = selectMatrixSpecies();

        switch (operation) {
            case DOT -> {
                // 直接计算矩阵点积，避免中间数组分配
                return directMatrixDotProduct(x1, x2, species);
            }

            case L2_NORM -> {
                // 直接计算矩阵L2范数，避免中间数组分配
                return Math.sqrt(directMatrixSumOfSquaredDifferences(x1, x2, species));
            }

            case L1_NORM -> {
                // 直接计算矩阵L1范数，避免中间数组分配
                return directMatrixSumOfAbsoluteDifferences(x1, x2, species);
            }

            default ->
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    /**
     * 直接计算两个矩阵的点积，避免中间数组分配
     */
    private static double directMatrixDotProduct(double[][] x1, double[][] x2, VectorSpecies<Double> species) {
        double sumResult = 0.0;
        
        // 对每一行进行向量化点积计算
        for (int row = 0; row < x1.length; row++) {
            sumResult += directDotProduct(x1[row], x2[row], species);
        }
        
        return sumResult;
    }

    /**
     * 直接计算两个矩阵差的平方和，避免中间数组分配
     */
    private static double directMatrixSumOfSquaredDifferences(double[][] x1, double[][] x2, VectorSpecies<Double> species) {
        double sumResult = 0.0;
        
        // 对每一行进行向量化计算
        for (int row = 0; row < x1.length; row++) {
            sumResult += directSumOfSquaredDifferences(x1[row], x2[row], species);
        }
        
        return sumResult;
    }

    /**
     * 直接计算两个矩阵差的绝对值和，避免中间数组分配
     */
    private static double directMatrixSumOfAbsoluteDifferences(double[][] x1, double[][] x2, VectorSpecies<Double> species) {
        double sumResult = 0.0;
        
        // 对每一行进行向量化计算
        for (int row = 0; row < x1.length; row++) {
            sumResult += directSumOfAbsoluteDifferences(x1[row], x2[row], species);
        }
        
        return sumResult;
    }

    /**
     * 基于Java Vector API计算向量逻辑运算
     *
     * @param x1 第一个向量
     * @param x2 第二个向量
     * @param operation 逻辑运算操作
     * @return 布尔数组，表示x1中对应元素是否满足指定逻辑运算条件于x2中对应元素
     * @throws IllegalArgumentException 如果输入为null或长度不匹配
     */
    @Override
    public boolean[] logicalCompare(double[] x1, double[] x2, LogicalCompare operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用向量API进行逻辑运算
        int i = 0;
        int upperBound = species.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(species, x1, i);
            DoubleVector b = DoubleVector.fromArray(species, x2, i);

            // 执行向量逻辑运算操作
            VectorMask<Double> mask;
            switch (operation) {
                case EQUALS ->
                    mask = a.compare(VectorOperators.EQ, b);
                case NOT_EQUALS ->
                    mask = a.compare(VectorOperators.NE, b);
                case LESS_THAN ->
                    mask = a.compare(VectorOperators.LT, b);
                case LESS_THAN_OR_EQUALS ->
                    mask = a.compare(VectorOperators.LE, b);
                case GREATER_THAN ->
                    mask = a.compare(VectorOperators.GT, b);
                case GREATER_THAN_OR_EQUALS ->
                    mask = a.compare(VectorOperators.GE, b);
                default ->
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }

            // 将结果存储回布尔数组
            mask.intoArray(result, i);
        }

        // 处理剩余元素
        for (; i < x1.length; i++) {
            switch (operation) {
                case EQUALS ->
                    result[i] = x1[i] == x2[i];
                case NOT_EQUALS ->
                    result[i] = x1[i] != x2[i];
                case LESS_THAN ->
                    result[i] = x1[i] < x2[i];
                case LESS_THAN_OR_EQUALS ->
                    result[i] = x1[i] <= x2[i];
                case GREATER_THAN ->
                    result[i] = x1[i] > x2[i];
                case GREATER_THAN_OR_EQUALS ->
                    result[i] = x1[i] >= x2[i];
                default ->
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }

        return result;
    }

    @Override
    public boolean[] logicalOperate(double[] x1, double[] x2, LogicalOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用向量API进行逻辑运算
        int i = 0;
        int upperBound = species.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(species, x1, i);
            DoubleVector b = DoubleVector.fromArray(species, x2, i);

            // 创建掩码：非零值为true，零值为false
            VectorMask<Double> maskA = a.compare(VectorOperators.NE, 0.0);
            VectorMask<Double> maskB = b.compare(VectorOperators.NE, 0.0);

            // 执行向量逻辑运算操作
            VectorMask<Double> mask;
            switch (operation) {
                case AND ->
                    mask = maskA.and(maskB);
                case OR ->
                    mask = maskA.or(maskB);
                case XOR ->
                    mask = maskA.xor(maskB);
                default ->
                    throw new IllegalArgumentException("不支持的操作: " + operation + "。logicalOperate方法仅支持AND、OR、XOR操作。对于NOT操作，请使用单参数版本。");
            }

            // 将结果存储回布尔数组
            mask.intoArray(result, i);
        }

        // 处理剩余元素
        for (; i < x1.length; i++) {
            // 将double值转换为布尔值：非零为true，零为false
            boolean boolA = (x1[i] != 0.0);
            boolean boolB = (x2[i] != 0.0);

            switch (operation) {
                case AND ->
                    result[i] = boolA && boolB;
                case OR ->
                    result[i] = boolA || boolB;
                case XOR ->
                    result[i] = boolA ^ boolB;
                default ->
                    throw new IllegalArgumentException("不支持的操作: " + operation + "。logicalOperate方法仅支持AND、OR、XOR操作。对于NOT操作，请使用单参数版本。");
            }
        }

        return result;
    }

    @Override
    public boolean[] logicalOperate(double[] x1, LogicalOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 获取最优向量规格
        VectorSpecies<Double> species = DoubleVector.SPECIES_PREFERRED;

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用向量API进行逻辑运算
        int i = 0;
        int upperBound = species.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            DoubleVector a = DoubleVector.fromArray(species, x1, i);

            // 执行向量逻辑运算操作
            VectorMask<Double> mask;
            switch (operation) {
                case NOT ->
                    // 对于NOT操作，我们将非零值视为true，零值视为false，然后取反
                    // 所以0变成true，非零变成false
                    mask = a.compare(VectorOperators.EQ, 0.0);
                default ->
                    throw new IllegalArgumentException("不支持的操作: " + operation + "。logicalOperate方法仅支持NOT操作。对于AND、OR、XOR操作，请使用biLogicalOperate方法。");
            }

            // 将结果存储回布尔数组
            mask.intoArray(result, i);
        }

        // 处理剩余元素
        for (; i < x1.length; i++) {
            switch (operation) {
                case NOT ->
                    // 对于NOT操作，0变成true，非零变成false
                    result[i] = (x1[i] == 0.0);
                default ->
                    throw new IllegalArgumentException("不支持的操作: " + operation + "。logicalOperate方法仅支持NOT操作。对于AND、OR、XOR操作，请使用biLogicalOperate方法。");
            }
        }

        return result;
    }

    @Override
    public boolean[][] logicalCompare(double[][] x1, double[][] x2, LogicalCompare operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length != x2.length || (x1.length > 0 && x1[0].length != x2[0].length)) {
            throw new IllegalArgumentException("矩阵维度必须相同");
        }

        // 获取矩阵维度
        int rows = x1.length;
        int cols = x1.length > 0 ? x1[0].length : 0;
        
        // 创建结果数组
        boolean[][] result = new boolean[rows][cols];
        
        // 对每一行进行向量化逻辑比较运算
        for (int row = 0; row < rows; row++) {
            result[row] = logicalCompare(x1[row], x2[row], operation);
        }

        return result;
    }

    @Override
    public boolean[][] logicalOperate(double[][] x1, LogicalOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // 获取矩阵维度
        int rows = x1.length;
        int cols = x1.length > 0 ? x1[0].length : 0;
        
        // 创建结果数组
        boolean[][] result = new boolean[rows][cols];
        
        // 对每一行进行向量化逻辑运算
        for (int row = 0; row < rows; row++) {
            result[row] = logicalOperate(x1[row], operation);
        }

        return result;
    }

    @Override
    public boolean[][] logicalOperate(double[][] x1, double[][] x2, LogicalOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length != x2.length || (x1.length > 0 && x1[0].length != x2[0].length)) {
            throw new IllegalArgumentException("矩阵维度必须相同");
        }

        // 获取矩阵维度
        int rows = x1.length;
        int cols = x1.length > 0 ? x1[0].length : 0;
        
        // 创建结果数组
        boolean[][] result = new boolean[rows][cols];
        
        // 对每一行进行向量化逻辑运算
        for (int row = 0; row < rows; row++) {
            result[row] = logicalOperate(x1[row], x2[row], operation);
        }

        return result;
    }
    
    
    

    @Override
    public double[][] transpose(double[][] matrix) {
        return transposeOptimized(matrix);
    }

    @Override
    public double[][] transpose(double[] rowVector) {
        int length = rowVector.length;
        double[][] columnMatrix = new double[length][1];
        for (int i = 0; i < length; i++) {
            columnMatrix[i][0] = rowVector[i];
        }
        return columnMatrix;
    }

    @Override
    public double[][] mmul(double[][] a, double[][] b) {
        int m = a.length;          // 当前矩阵行数
        if (m == 0) return new double[0][];
        int otherRows = b.length;
        if (otherRows == 0) return new double[0][];
        int n = a[0].length;       // 当前矩阵列数
        int p = b[0].length;       // 另一个矩阵列数
        if (n != otherRows) {
            throw new IllegalArgumentException(
                "Matrix dimension mismatch for mmul: A is " + m + "x" + n + ", B is " + otherRows + "x" + p);
        }

        double[][] ob = HpcGemm.tryMatMul(a, b);
        if (ob != null) {
            return ob;
        }

        // 计算复杂度
        long complexity = (long) m * n * p;

        // Strassen路径：大矩阵且启用时使用O(n^2.807)算法
        if (USE_STRASSEN && complexity >= STRASSEN_COMPLEXITY_THRESHOLD
                && Math.min(m, Math.min(n, p)) >= STRASSEN_THRESHOLD) {
            return mmulStrassen(a, b, m, n, p);
        }

        if (complexity < 4000) {
            return mmulDirectOptimized(a, b, m, n, p);
        } else if (complexity < 8000000) {
            return mmulBlockOptimized(a, b, m, n, p);
        } else {
            return mmulBlockParallel(a, b, m, n, p);
        }
    }
    
    /**
     * 优化的直接矩阵乘法，针对长度1000+对应优化
     */
    private double[][] mmulDirectOptimized(double[][] a, double[][] b, int m, int n, int p) {
        // 转置b矩阵以优化内存访问模式
        double[][] bT = transposeOptimized(b);
        double[][] c = new double[m][p];
        
        final int unrollFactor = calculateUnrollFactor(n);
        final int vectorLength = VECTOR_LENGTH;
        final int blockSize = unrollFactor * vectorLength;
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                c[i][j] = computeDotProductOptimized(a[i], bT[j], n, blockSize, unrollFactor);
            }
        }
        
        return c;
    }
    
    /**
     * 优化的点积计算，针对长向量优化。
     * 使用安全边界检查确保不越界读取。
     */
    private double computeDotProductOptimized(double[] x, double[] y, int length, int blockSize, int unrollFactor) {
        if (length < blockSize) {
            return computeDotProductSimple(x, y, length);
        }

        DoubleVector acc1 = DoubleVector.zero(PREFERRED_SPECIES);
        DoubleVector acc2 = DoubleVector.zero(PREFERRED_SPECIES);
        DoubleVector acc3 = DoubleVector.zero(PREFERRED_SPECIES);
        DoubleVector acc4 = DoubleVector.zero(PREFERRED_SPECIES);

        int i = 0;
        final int upperBound = (length / blockSize) * blockSize;

        for (; i < upperBound; i += blockSize) {
            final int blockEnd = Math.min(i + blockSize, length);
            for (int u = 0; u < unrollFactor; ) {
                int idx = i + u * VECTOR_LENGTH;
                // 计算当前索引开始还能安全读取多少个向量
                int availableVectors = (blockEnd - idx) / VECTOR_LENGTH;

                if (availableVectors >= 4) {
                    DoubleVector ax1 = DoubleVector.fromArray(PREFERRED_SPECIES, x, idx);
                    DoubleVector ay1 = DoubleVector.fromArray(PREFERRED_SPECIES, y, idx);
                    DoubleVector ax2 = DoubleVector.fromArray(PREFERRED_SPECIES, x, idx + VECTOR_LENGTH);
                    DoubleVector ay2 = DoubleVector.fromArray(PREFERRED_SPECIES, y, idx + VECTOR_LENGTH);
                    DoubleVector ax3 = DoubleVector.fromArray(PREFERRED_SPECIES, x, idx + 2 * VECTOR_LENGTH);
                    DoubleVector ay3 = DoubleVector.fromArray(PREFERRED_SPECIES, y, idx + 2 * VECTOR_LENGTH);
                    DoubleVector ax4 = DoubleVector.fromArray(PREFERRED_SPECIES, x, idx + 3 * VECTOR_LENGTH);
                    DoubleVector ay4 = DoubleVector.fromArray(PREFERRED_SPECIES, y, idx + 3 * VECTOR_LENGTH);
                    acc1 = acc1.add(ax1.mul(ay1));
                    acc2 = acc2.add(ax2.mul(ay2));
                    acc3 = acc3.add(ax3.mul(ay3));
                    acc4 = acc4.add(ax4.mul(ay4));
                    u += 4;
                } else if (availableVectors >= 2) {
                    DoubleVector ax1 = DoubleVector.fromArray(PREFERRED_SPECIES, x, idx);
                    DoubleVector ay1 = DoubleVector.fromArray(PREFERRED_SPECIES, y, idx);
                    DoubleVector ax2 = DoubleVector.fromArray(PREFERRED_SPECIES, x, idx + VECTOR_LENGTH);
                    DoubleVector ay2 = DoubleVector.fromArray(PREFERRED_SPECIES, y, idx + VECTOR_LENGTH);
                    acc1 = acc1.add(ax1.mul(ay1));
                    acc2 = acc2.add(ax2.mul(ay2));
                    u += 2;
                } else if (availableVectors >= 1) {
                    DoubleVector ax = DoubleVector.fromArray(PREFERRED_SPECIES, x, idx);
                    DoubleVector ay = DoubleVector.fromArray(PREFERRED_SPECIES, y, idx);
                    acc1 = acc1.add(ax.mul(ay));
                    u += 1;
                } else {
                    break;
                }
            }
        }

        DoubleVector totalAcc = acc1.add(acc2).add(acc3).add(acc4);
        double result = totalAcc.reduceLanes(VectorOperators.ADD);

        int vectorBound = PREFERRED_SPECIES.loopBound(length);
        for (; i < vectorBound; i += VECTOR_LENGTH) {
            DoubleVector ax = DoubleVector.fromArray(PREFERRED_SPECIES, x, i);
            DoubleVector ay = DoubleVector.fromArray(PREFERRED_SPECIES, y, i);
            result += ax.mul(ay).reduceLanes(VectorOperators.ADD);
        }

        for (; i < length; i++) {
            result += x[i] * y[i];
        }

        return result;
    }
    
    /**
     * 简单的点积计算
     */
    private double computeDotProductSimple(double[] x, double[] y, int length) {
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(length);
        DoubleVector acc = DoubleVector.zero(PREFERRED_SPECIES);

        for (; i < upperBound; i += VECTOR_LENGTH) {
            DoubleVector ax = DoubleVector.fromArray(PREFERRED_SPECIES, x, i);
            DoubleVector ay = DoubleVector.fromArray(PREFERRED_SPECIES, y, i);
            acc = acc.add(ax.mul(ay));
        }

        double result = acc.reduceLanes(VectorOperators.ADD);
        
        for (; i < length; i++) {
            result += x[i] * y[i];
        }

        return result;
    }
    
    /**
     * 优化的矩阵转置操作
     */
    private double[][] transposeOptimized(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] result = new double[cols][rows];
        
        // 对于小矩阵，使用简单的转置
        if (rows * cols < OPTIMAL_BLOCK_SIZE * OPTIMAL_BLOCK_SIZE) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result[j][i] = matrix[i][j];
                }
            }
            return result;
        }
        
        // 对于大矩阵，使用分块转置以优化缓存局部性
        final int blockSize = Math.min(OPTIMAL_BLOCK_SIZE, Math.min(rows, cols));
        
        for (int i = 0; i < rows; i += blockSize) {
            for (int j = 0; j < cols; j += blockSize) {
                int iEnd = Math.min(i + blockSize, rows);
                int jEnd = Math.min(j + blockSize, cols);
                
                // 块内转置
                for (int ii = i; ii < iEnd; ii++) {
                    for (int jj = j; jj < jEnd; jj++) {
                        result[jj][ii] = matrix[ii][jj];
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * 优化的分块矩阵乘法
     */
    /**
     * Blocked GEMM with Vector API FMA, using i-k-j loop order.
     * Reuses b[k] row in cache and broadcasts aik across j-dimension vectors.
     */
    private double[][] mmulBlockOptimized(double[][] a, double[][] b, int m, int n, int p) {
        double[][] c = new double[m][p];
        int blockSize = calculateMatrixBlockSize(m, n, p);
        final int vl = VECTOR_LENGTH;

        // ii-kk-jj outer blocking for cache, i-k-j inner for b[k] reuse
        for (int ii = 0; ii < m; ii += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                for (int jj = 0; jj < p; jj += blockSize) {
                    int iEnd = Math.min(ii + blockSize, m);
                    int jEnd = Math.min(jj + blockSize, p);
                    int kEnd = Math.min(kk + blockSize, n);

                    for (int i = ii; i < iEnd; i++) {
                        double[] aRow = a[i];
                        double[] cRow = c[i];
                        for (int k = kk; k < kEnd; k++) {
                            double aik = aRow[k];
                            if (aik == 0.0) continue;
                            double[] bRow = b[k];
                            DoubleVector aikVec = DoubleVector.broadcast(PREFERRED_SPECIES, aik);

                            int j = jj;
                            // 4x unrolled FMA: cRow[j] += aik * bRow[j]
                            for (; j + vl * 4 <= jEnd; j += vl * 4) {
                                DoubleVector c0 = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j);
                                c0 = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j), c0);
                                c0.intoArray(cRow, j);

                                DoubleVector c1 = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j + vl);
                                c1 = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j + vl), c1);
                                c1.intoArray(cRow, j + vl);

                                DoubleVector c2 = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j + vl * 2);
                                c2 = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j + vl * 2), c2);
                                c2.intoArray(cRow, j + vl * 2);

                                DoubleVector c3 = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j + vl * 3);
                                c3 = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j + vl * 3), c3);
                                c3.intoArray(cRow, j + vl * 3);
                            }
                            // Single vector cleanup
                            for (; j + vl <= jEnd; j += vl) {
                                DoubleVector cVec = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j);
                                cVec = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j), cVec);
                                cVec.intoArray(cRow, j);
                            }
                            // Scalar cleanup
                            for (; j < jEnd; j++) {
                                cRow[j] += aik * bRow[j];
                            }
                        }
                    }
                }
            }
        }
        return c;
    }

    /**
     * 并行分块 GEMM：按行粗粒度分区，每分区内执行 i-k-j FMA 微核。
     * 仅用于大矩阵（complexity >= 8M），避免小矩阵的并行调度开销。
     */
    private double[][] mmulBlockParallel(double[][] a, double[][] b, int m, int n, int p) {
        double[][] c = new double[m][p];
        int blockSize = calculateMatrixBlockSize(m, n, p);
        java.util.concurrent.ForkJoinPool pool = RereDoubleMatrix.getThreadPool();

        if (pool == null || pool.isShutdown()) {
            return mmulBlockOptimized(a, b, m, n, p);
        }

        int numThreads = pool.getParallelism();
        if (numThreads <= 1 || m < blockSize * 2) {
            return mmulBlockOptimized(a, b, m, n, p);
        }

        int rowsPerTask = (m + numThreads - 1) / numThreads;
        List<Future<Void>> futures = new ArrayList<>();
        final int vl = VECTOR_LENGTH;

        for (int t = 0; t < numThreads; t++) {
            final int startRow = t * rowsPerTask;
            final int endRow = Math.min(startRow + rowsPerTask, m);
            if (startRow >= endRow) break;
            futures.add(pool.submit(() -> {
                for (int kk = 0; kk < n; kk += blockSize) {
                    for (int jj = 0; jj < p; jj += blockSize) {
                        int kEnd = Math.min(kk + blockSize, n);
                        int jEnd = Math.min(jj + blockSize, p);

                        for (int i = startRow; i < endRow; i++) {
                            double[] aRow = a[i];
                            double[] cRow = c[i];
                            for (int k = kk; k < kEnd; k++) {
                                double aik = aRow[k];
                                if (aik == 0.0) continue;
                                double[] bRow = b[k];
                                DoubleVector aikVec = DoubleVector.broadcast(PREFERRED_SPECIES, aik);

                                int j = jj;
                                for (; j + vl * 4 <= jEnd; j += vl * 4) {
                                    DoubleVector c0 = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j);
                                    c0 = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j), c0);
                                    c0.intoArray(cRow, j);

                                    DoubleVector c1 = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j + vl);
                                    c1 = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j + vl), c1);
                                    c1.intoArray(cRow, j + vl);

                                    DoubleVector c2 = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j + vl * 2);
                                    c2 = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j + vl * 2), c2);
                                    c2.intoArray(cRow, j + vl * 2);

                                    DoubleVector c3 = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j + vl * 3);
                                    c3 = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j + vl * 3), c3);
                                    c3.intoArray(cRow, j + vl * 3);
                                }
                                for (; j + vl <= jEnd; j += vl) {
                                    DoubleVector cv = DoubleVector.fromArray(PREFERRED_SPECIES, cRow, j);
                                    cv = aikVec.fma(DoubleVector.fromArray(PREFERRED_SPECIES, bRow, j), cv);
                                    cv.intoArray(cRow, j);
                                }
                                for (; j < jEnd; j++) {
                                    cRow[j] += aik * bRow[j];
                                }
                            }
                        }
                    }
                }
                return null;
            }));
        }

        try {
            for (Future<Void> f : futures) f.get();
        } catch (Exception e) {
            log.warn("Parallel GEMM failed, falling back to sequential: {}", e.getMessage());
            return mmulBlockOptimized(a, b, m, n, p);
        }
        return c;
    }

    /**
     * 计算矩阵乘法的最优块大小
     */
    private static int calculateMatrixBlockSize(int m, int n, int p) {
        // 基于L2缓存大小和矩阵维度计算最优块大小
        long elementsInL2 = L2_CACHE_SIZE / (Double.BYTES * 3); // 3个矩阵的数据
        int blockSize = (int) Math.cbrt(elementsInL2 / 3); // 立方根估算

        // 限制在合理范围内
        blockSize = Math.max(32, Math.min(blockSize, 256));

        // 确保是向量长度的倍数
        return ((blockSize + VECTOR_LENGTH - 1) / VECTOR_LENGTH) * VECTOR_LENGTH;
    }

    // 针对特定形状优化的外积
    @Override
    public double[][] outer(double[] a, double[] b) {
        int m = a.length;
        int n = b.length;
        double[][] result = new double[m][n];
        return outerProductByRow(a, b, result);
    }

    private double[][] outerProductByRow(double[] a, double[] b, double[][] result) {
        int m = a.length;
        int n = b.length;
        final VectorSpecies<Double> species = selectMatrixSpecies();
        int vectorLength = species.length();

        // Loop unrolling factor for better performance
        final int UNROLL_FACTOR = 4;

        for (int i = 0; i < m; i++) {
            DoubleVector aBroadcast = DoubleVector.broadcast(species, a[i]);
            int j = 0;

            // Process multiple elements at once with loop unrolling
            int unrolledEnd = (n / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR);
            for (; j < unrolledEnd; j += vectorLength * UNROLL_FACTOR) {
                // Unroll 4 iterations to improve instruction-level parallelism
                DoubleVector bVec0 = DoubleVector.fromArray(species, b, j);
                DoubleVector bVec1 = DoubleVector.fromArray(species, b, j + vectorLength);
                DoubleVector bVec2 = DoubleVector.fromArray(species, b, j + 2 * vectorLength);
                DoubleVector bVec3 = DoubleVector.fromArray(species, b, j + 3 * vectorLength);
                
                DoubleVector result0 = aBroadcast.mul(bVec0);
                DoubleVector result1 = aBroadcast.mul(bVec1);
                DoubleVector result2 = aBroadcast.mul(bVec2);
                DoubleVector result3 = aBroadcast.mul(bVec3);
                
                result0.intoArray(result[i], j);
                result1.intoArray(result[i], j + vectorLength);
                result2.intoArray(result[i], j + 2 * vectorLength);
                result3.intoArray(result[i], j + 3 * vectorLength);
            }

            // Process vectorized elements
            int vectorizedEnd = (n / vectorLength) * vectorLength;
            for (; j < vectorizedEnd; j += vectorLength) {
                DoubleVector bVec = DoubleVector.fromArray(species, b, j);
                aBroadcast.mul(bVec).intoArray(result[i], j);
            }

            for (; j < n; j++) {
                result[i][j] = a[i] * b[j];
            }
        }
        return result;
    }

    @Override
    public double[] sign(double[] array) {
        double[] result = new double[array.length];
        final VectorSpecies<Double> species = selectMatrixSpecies();
        int vectorLength = species.length();
        int i = 0;

        DoubleVector zero = DoubleVector.zero(species);
        DoubleVector one = DoubleVector.broadcast(species, 1.0);
        DoubleVector minusOne = DoubleVector.broadcast(species, -1.0);

        // Loop unrolling factor for better performance
        final int UNROLL_FACTOR = 4;
        
        // Process multiple elements at once with loop unrolling
        int unrolledEnd = (array.length / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR);
        for (; i < unrolledEnd; i += vectorLength * UNROLL_FACTOR) {
            // Unroll 4 iterations to improve instruction-level parallelism
            DoubleVector vec0 = DoubleVector.fromArray(species, array, i);
            DoubleVector vec1 = DoubleVector.fromArray(species, array, i + vectorLength);
            DoubleVector vec2 = DoubleVector.fromArray(species, array, i + 2 * vectorLength);
            DoubleVector vec3 = DoubleVector.fromArray(species, array, i + 3 * vectorLength);

            // 使用三元逻辑实现
            VectorMask<Double> gtZero0 = vec0.compare(VectorOperators.GT, 0.0);
            VectorMask<Double> ltZero0 = vec0.lt(0.0);
            VectorMask<Double> isNaN0 = vec0.eq(vec0).not();
            VectorMask<Double> gtZero1 = vec1.compare(VectorOperators.GT, 0.0);
            VectorMask<Double> ltZero1 = vec1.lt(0.0);
            VectorMask<Double> isNaN1 = vec1.eq(vec1).not();
            VectorMask<Double> gtZero2 = vec2.compare(VectorOperators.GT, 0.0);
            VectorMask<Double> ltZero2 = vec2.lt(0.0);
            VectorMask<Double> isNaN2 = vec2.eq(vec2).not();
            VectorMask<Double> gtZero3 = vec3.compare(VectorOperators.GT, 0.0);
            VectorMask<Double> ltZero3 = vec3.lt(0.0);
            VectorMask<Double> isNaN3 = vec3.eq(vec3).not();

            DoubleVector signVec0 = zero.blend(one, gtZero0).blend(minusOne, ltZero0).blend(vec0, isNaN0);
            DoubleVector signVec1 = zero.blend(one, gtZero1).blend(minusOne, ltZero1).blend(vec1, isNaN1);
            DoubleVector signVec2 = zero.blend(one, gtZero2).blend(minusOne, ltZero2).blend(vec2, isNaN2);
            DoubleVector signVec3 = zero.blend(one, gtZero3).blend(minusOne, ltZero3).blend(vec3, isNaN3);

            signVec0.intoArray(result, i);
            signVec1.intoArray(result, i + vectorLength);
            signVec2.intoArray(result, i + 2 * vectorLength);
            signVec3.intoArray(result, i + 3 * vectorLength);
        }

        // Process vectorized elements
        int vectorizedEnd = (array.length / vectorLength) * vectorLength;
        for (; i < vectorizedEnd; i += vectorLength) {
            DoubleVector vec = DoubleVector.fromArray(species, array, i);

            // 使用三元逻辑实现
            VectorMask<Double> gtZero = vec.compare(VectorOperators.GT, 0.0);
            VectorMask<Double> ltZero = vec.lt(0.0);
            VectorMask<Double> isNaN = vec.eq(vec).not();

            DoubleVector signVec = zero.blend(one, gtZero).blend(minusOne, ltZero).blend(vec, isNaN);
            signVec.intoArray(result, i);
        }

        // 处理尾部
        for (; i < array.length; i++) {
            result[i] = signScalar(array[i]);
        }

        return result;
    }

    @Override
    public double[][] sign(double[][] array) {
        // 参数验证
        if (array == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // 获取矩阵维度
        int rows = array.length;
        int cols = array.length > 0 ? array[0].length : 0;
        
        // 创建结果矩阵
        double[][] result = new double[rows][cols];
        
        // 对每一行进行向量化符号运算
        for (int row = 0; row < rows; row++) {
            result[row] = sign(array[row]);
        }

        return result;
    }
    
    private double signScalar(double value) {
        if (Double.isNaN(value)) {
            return Double.NaN;
        }
        if (value > 0) {
            return 1.0;
        }
        if (value < 0) {
            return -1.0;
        }
        return 0.0;
    }

    @Override
    public double[] fill(int size, double value) {
        double[] result = new double[size];
        java.util.Arrays.fill(result, value);
        return result;
    }

    @Override
    public double[] diff(double[] array, int stride) {
        if (stride <= 0) {
            throw new IllegalArgumentException("步长必须为正整数");
        }
        if (array.length <= stride) {
            return new double[0];
        }
        final VectorSpecies<Double> species = selectMatrixSpecies();
        int vectorLength = species.length();
        int resultLength = array.length - stride;
        double[] result = new double[resultLength];
        int i = 0;

        final int UNROLL_FACTOR = 4;

        // 安全边界：确保 i + stride + vectorLength * UNROLL_FACTOR 不越界
        int safeLimit = Math.max(0, array.length - vectorLength * UNROLL_FACTOR);
        int unrolledEnd = Math.min(
            (resultLength / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR),
            Math.min(resultLength, Math.max(0, safeLimit - stride))
        );

        for (; i < unrolledEnd; i += vectorLength * UNROLL_FACTOR) {
            DoubleVector current0 = DoubleVector.fromArray(species, array, i);
            DoubleVector future0 = DoubleVector.fromArray(species, array, i + stride);
            DoubleVector current1 = DoubleVector.fromArray(species, array, i + vectorLength);
            DoubleVector future1 = DoubleVector.fromArray(species, array, i + stride + vectorLength);
            DoubleVector current2 = DoubleVector.fromArray(species, array, i + 2 * vectorLength);
            DoubleVector future2 = DoubleVector.fromArray(species, array, i + stride + 2 * vectorLength);
            DoubleVector current3 = DoubleVector.fromArray(species, array, i + 3 * vectorLength);
            DoubleVector future3 = DoubleVector.fromArray(species, array, i + stride + 3 * vectorLength);

            DoubleVector diff0 = future0.sub(current0);
            DoubleVector diff1 = future1.sub(current1);
            DoubleVector diff2 = future2.sub(current2);
            DoubleVector diff3 = future3.sub(current3);

            diff0.intoArray(result, i);
            diff1.intoArray(result, i + vectorLength);
            diff2.intoArray(result, i + 2 * vectorLength);
            diff3.intoArray(result, i + 3 * vectorLength);
        }

        // 安全边界：确保 i + stride + vectorLength 不越界
        int safeVectorLimit = Math.max(0, array.length - vectorLength - stride);
        int vectorizedEnd = Math.min(
            (resultLength / vectorLength) * vectorLength,
            safeVectorLimit
        );
        for (; i < vectorizedEnd; i += vectorLength) {
            DoubleVector current = DoubleVector.fromArray(species, array, i);
            DoubleVector future = DoubleVector.fromArray(species, array, i + stride);

            DoubleVector diff = future.sub(current);
            diff.intoArray(result, i);
        }

        for (; i < resultLength; i++) {
            result[i] = array[i + stride] - array[i];
        }

        return result;
    }

    // ======================== Strassen矩阵乘法 ========================

    /**
     * Strassen O(n^2.807) matrix multiplication for large matrices.
     * Recursion thresholds to mmulBlockOptimized when submatrix dimension &lt; STRASSEN_THRESHOLD.
     * Odd dimensions are padded and the result stripped.
     */
    private double[][] mmulStrassen(double[][] a, double[][] b, int m, int n, int p) {
        // Pad to even dimensions
        int mPad = (m + 1) & ~1;
        int nPad = (n + 1) & ~1;
        int pPad = (p + 1) & ~1;
        boolean padded = (mPad != m || nPad != n || pPad != p);

        double[][] aPad, bPad;
        if (padded) {
            aPad = new double[mPad][nPad];
            bPad = new double[nPad][pPad];
            for (int i = 0; i < m; i++) System.arraycopy(a[i], 0, aPad[i], 0, n);
            for (int i = 0; i < n; i++) System.arraycopy(b[i], 0, bPad[i], 0, p);
        } else {
            aPad = a;
            bPad = b;
        }

        // Use block-optimized for near-threshold
        if (mPad <= STRASSEN_THRESHOLD || nPad <= STRASSEN_THRESHOLD || pPad <= STRASSEN_THRESHOLD) {
            return mmulBlockOptimized(aPad, bPad, mPad, nPad, pPad);
        }

        int dim = Math.max(mPad, Math.max(nPad, pPad));
        int effective = Math.min(Math.min(mPad, nPad), pPad);
        // Use the smallest common even dimension for recursion
        effective = (effective + 1) & ~1;

        double[][] resultPad = strassenSquareRecursive(aPad, bPad, effective,
                0, 0, 0, 0, 0, 0, true);

        // Strip padding
        if (padded) {
            double[][] result = new double[m][p];
            for (int i = 0; i < m; i++) System.arraycopy(resultPad[i], 0, result[i], 0, p);
            return result;
        }
        return resultPad;
    }

    /**
     * Core Strassen recursion for common even dimension d.
     */
    private double[][] strassenSquareRecursive(double[][] a, double[][] b, int d,
            int aRowOff, int aColOff, int bRowOff, int bColOff,
            int cRowOff, int cColOff, boolean topLevel) {
        if (d <= STRASSEN_THRESHOLD) {
            double[][] aSub = acquireScratch(d);
            double[][] bSub = acquireScratch(d);
            for (int i = 0; i < d; i++) {
                System.arraycopy(a[aRowOff + i], aColOff, aSub[i], 0, d);
                System.arraycopy(b[bRowOff + i], bColOff, bSub[i], 0, d);
            }
            double[][] leafResult = mmulBlockOptimized(aSub, bSub, d, d, d);
            releaseScratch(aSub);
            releaseScratch(bSub);
            return leafResult;
        }

        int half = d >>> 1;
        double[][] M1, M2, M3, M4, M5, M6, M7;

        ExecutorService pool = RereDoubleMatrix.getThreadPool();
        boolean parallel = topLevel && pool != null && !pool.isShutdown();

        if (parallel) {
            List<Future<double[][]>> futures = new ArrayList<>(7);

            double[][] A11pA22 = addSubmatrices(a, aRowOff, aColOff, aRowOff + half, aColOff + half, half, true);
            double[][] B11pB22 = addSubmatrices(b, bRowOff, bColOff, bRowOff + half, bColOff + half, half, true);
            double[][] A21pA22 = addSubmatrices(a, aRowOff + half, aColOff, aRowOff + half, aColOff + half, half, true);
            double[][] B12mB22 = addSubmatrices(b, bRowOff, bColOff + half, bRowOff + half, bColOff + half, half, false);
            double[][] B21mB11 = addSubmatrices(b, bRowOff + half, bColOff, bRowOff, bColOff, half, false);
            double[][] A11pA12 = addSubmatrices(a, aRowOff, aColOff, aRowOff, aColOff + half, half, true);
            double[][] A21mA11 = addSubmatrices(a, aRowOff + half, aColOff, aRowOff, aColOff, half, false);
            double[][] B11pB12 = addSubmatrices(b, bRowOff, bColOff, bRowOff, bColOff + half, half, true);
            double[][] A12mA22 = addSubmatrices(a, aRowOff, aColOff + half, aRowOff + half, aColOff + half, half, false);
            double[][] B21pB22 = addSubmatrices(b, bRowOff + half, bColOff, bRowOff + half, bColOff + half, half, true);

            futures.add(pool.submit(() -> mmulBlockOptimized(A11pA22, B11pB22, half, half, half)));
            futures.add(pool.submit(() -> mmulBlockOptimized(A21pA22, getSub(b, bRowOff, bColOff, half), half, half, half)));
            futures.add(pool.submit(() -> mmulBlockOptimized(getSub(a, aRowOff, aColOff, half), B12mB22, half, half, half)));
            futures.add(pool.submit(() -> mmulBlockOptimized(getSub(a, aRowOff + half, aColOff + half, half), B21mB11, half, half, half)));
            futures.add(pool.submit(() -> mmulBlockOptimized(A11pA12, getSub(b, bRowOff + half, bColOff + half, half), half, half, half)));
            futures.add(pool.submit(() -> mmulBlockOptimized(A21mA11, B11pB12, half, half, half)));
            futures.add(pool.submit(() -> mmulBlockOptimized(A12mA22, B21pB22, half, half, half)));

            try {
                M1 = futures.get(0).get();
                M2 = futures.get(1).get();
                M3 = futures.get(2).get();
                M4 = futures.get(3).get();
                M5 = futures.get(4).get();
                M6 = futures.get(5).get();
                M7 = futures.get(6).get();
            } catch (Exception e) {
                log.warn("Parallel Strassen failed, falling back to sequential: {}", e.getMessage());
                return mmulBlockOptimized(a, b, d, d, d);
            }
        } else {
            // Sequential Strassen with scratch buffer reuse
            double[][] a11 = acquireScratch(half);
            double[][] a22 = acquireScratch(half);
            double[][] b11 = acquireScratch(half);
            double[][] b22 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff, a11);
            getSubInto(a, aRowOff + half, aColOff + half, a22);
            getSubInto(b, bRowOff, bColOff, b11);
            getSubInto(b, bRowOff + half, bColOff + half, b22);

            double[][] scratch = acquireScratch(half);

            // M1 = (a11+a22) @ (b11+b22)
            addInto(a11, a22, scratch);
            double[][] m1a = acquireScratch(half);
            addInto(b11, b22, m1a);
            M1 = mmulBlockOptimized(scratch, m1a, half, half, half);
            releaseScratch(m1a);

            // M2 = (a21+a22) @ b11
            double[][] a21 = acquireScratch(half);
            getSubInto(a, aRowOff + half, aColOff, a21);
            addInto(a21, a22, scratch);
            M2 = mmulBlockOptimized(scratch, b11, half, half, half);

            // M3 = a11 @ (b12-b22)
            double[][] b12 = acquireScratch(half);
            getSubInto(b, bRowOff, bColOff + half, b12);
            subInto(b12, b22, scratch);
            M3 = mmulBlockOptimized(a11, scratch, half, half, half);
            releaseScratch(a11);

            // M4 = a22 @ (b21-b11)
            double[][] b21 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff, b21);
            subInto(b21, b11, scratch);
            releaseScratch(b11);
            M4 = mmulBlockOptimized(a22, scratch, half, half, half);
            releaseScratch(a22);

            // M5 = (a11+a12) @ b22
            double[][] a12 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff + half, a12);
            a11 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff, a11);
            addInto(a11, a12, scratch);
            releaseScratch(a11);
            releaseScratch(a12);
            M5 = mmulBlockOptimized(scratch, b22, half, half, half);
            releaseScratch(b22);

            // M6 = (a21-a11) @ (b11+b12)
            a11 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff, a11);
            subInto(a21, a11, scratch);
            releaseScratch(a21);
            double[][] b11b = acquireScratch(half);
            b11 = acquireScratch(half);
            getSubInto(b, bRowOff, bColOff, b11);
            addInto(b11, b12, b11b);
            releaseScratch(b11);
            releaseScratch(b12);
            M6 = mmulBlockOptimized(scratch, b11b, half, half, half);
            releaseScratch(b11b);

            // M7 = (a12-a22) @ (b21+b22)
            a12 = acquireScratch(half);
            a22 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff + half, a12);
            getSubInto(a, aRowOff + half, aColOff + half, a22);
            subInto(a12, a22, scratch);
            releaseScratch(a12);
            releaseScratch(a22);
            double[][] b22b = acquireScratch(half);
            b22 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff + half, b22);
            b21 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff, b21);
            addInto(b21, b22, b22b);
            releaseScratch(b21);
            releaseScratch(b22);
            M7 = mmulBlockOptimized(scratch, b22b, half, half, half);
            releaseScratch(b22b);
            releaseScratch(a11);
            releaseScratch(scratch);
        }

        // Combine: C11, C12, C21, C22
        double[][] result = new double[d][d];
        for (int i = 0; i < half; i++) {
            for (int j = 0; j < half; j++) {
                result[i][j]               = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
                result[i][j + half]        = M3[i][j] + M5[i][j];
                result[i + half][j]        = M2[i][j] + M4[i][j];
                result[i + half][j + half] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
            }
        }
        return result;
    }

    /** Element-wise matrix addition: a + b */
    private static double[][] add(double[][] a, double[][] b) {
        int m = a.length, n = a[0].length;
        double[][] r = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                r[i][j] = a[i][j] + b[i][j];
        return r;
    }

    /** Element-wise matrix subtraction: a - b */
    private static double[][] sub(double[][] a, double[][] b) {
        int m = a.length, n = a[0].length;
        double[][] r = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                r[i][j] = a[i][j] - b[i][j];
        return r;
    }

    /** Extract a submatrix of size d×d */
    private static double[][] getSub(double[][] a, int rowOff, int colOff, int d) {
        double[][] r = new double[d][d];
        for (int i = 0; i < d; i++)
            System.arraycopy(a[rowOff + i], colOff, r[i], 0, d);
        return r;
    }

    /** Add or subtract two submatrices of a common source array */
    private static double[][] addSubmatrices(double[][] a, int r1, int c1, int r2, int c2, int d, boolean add) {
        double[][] r = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                r[i][j] = add ? a[r1 + i][c1 + j] + a[r2 + i][c2 + j]
                              : a[r1 + i][c1 + j] - a[r2 + i][c2 + j];
            }
        }
        return r;
    }

    /** Element-wise matrix addition into pre-allocated output: out = a + b */
    private static void addInto(double[][] a, double[][] b, double[][] out) {
        int m = a.length;
        for (int i = 0; i < m; i++) {
            double[] ai = a[i], bi = b[i], oi = out[i];
            for (int j = 0; j < ai.length; j++) oi[j] = ai[j] + bi[j];
        }
    }

    /** Element-wise matrix subtraction into pre-allocated output: out = a - b */
    private static void subInto(double[][] a, double[][] b, double[][] out) {
        int m = a.length;
        for (int i = 0; i < m; i++) {
            double[] ai = a[i], bi = b[i], oi = out[i];
            for (int j = 0; j < ai.length; j++) oi[j] = ai[j] - bi[j];
        }
    }

    /** Extract a submatrix into pre-allocated output */
    private static void getSubInto(double[][] a, int rowOff, int colOff, double[][] out) {
        int d = out.length;
        for (int i = 0; i < d; i++)
            System.arraycopy(a[rowOff + i], colOff, out[i], 0, d);
    }

    /** Overload: mmulBlockOptimized with offset parameters for submatrix multiplication */
    private double[][] mmulBlockOptimized(double[][] a, double[][] b,
            int aRowOff, int aColOff, int aRows, int aCols, int bCols) {
        double[][] aSub = new double[aRows][aCols];
        for (int i = 0; i < aRows; i++)
            System.arraycopy(a[aRowOff + i], aColOff, aSub[i], 0, aCols);
        return mmulBlockOptimized(aSub, b, aRows, aCols, bCols);
    }

    // ===================== Flat GEMM (row-major double[]) =====================

    /**
     * Row-major flat matrix multiply: C[m*n] = A[m*k] @ B[k*n].
     * Dispatch: HPC faer → Java SIMD blocked GEMM.
     *
     * @param a row-major flat A, length m*k
     * @param m rows of A
     * @param k cols of A (= rows of B)
     * @param b row-major flat B, length k*n
     * @param n cols of B
     * @return row-major flat C, length m*n
     */
    public static double[] flatMmul(double[] a, int m, int k, double[] b, int n) {
        double[] c = new double[m * n];
        if (HpcIm2col.tryFlatDgemm(m, n, k, a, b, c)) {
            return c;
        }
        flatMmulBlocked(a, 0, m, k, b, 0, n, c, 0);
        return c;
    }

    /**
     * Offset-aware flat GEMM for batched use.
     * C = A[aOff .. aOff+m*k] @ B[bOff .. bOff+k*n].
     */
    public static double[] flatMmul(double[] a, int aOff, int m, int k,
                                    double[] b, int bOff, int n) {
        double[] aSlice = java.util.Arrays.copyOfRange(a, aOff, aOff + m * k);
        double[] bSlice = java.util.Arrays.copyOfRange(b, bOff, bOff + k * n);
        return flatMmul(aSlice, m, k, bSlice, n);
    }

    /**
     * SIMD blocked GEMM on flat row-major arrays (i-k-j FMA micro-kernel).
     * Adapted from {@link #mmulBlockOptimized(double[][], double[][], int, int, int)}.
     */
    private static void flatMmulBlocked(double[] a, int aOff, int m, int k,
                                         double[] b, int bOff, int n,
                                         double[] c, int cOff) {
        int blockSize = calculateMatrixBlockSize(m, k, n);
        final int vl = VECTOR_LENGTH;
        final VectorSpecies<Double> species = PREFERRED_SPECIES;

        for (int ii = 0; ii < m; ii += blockSize) {
            for (int kk = 0; kk < k; kk += blockSize) {
                for (int jj = 0; jj < n; jj += blockSize) {
                    int iEnd = Math.min(ii + blockSize, m);
                    int kEnd = Math.min(kk + blockSize, k);
                    int jEnd = Math.min(jj + blockSize, n);

                    for (int i = ii; i < iEnd; i++) {
                        int aRow = aOff + i * k;
                        int cRow = cOff + i * n;
                        for (int kk2 = kk; kk2 < kEnd; kk2++) {
                            double aik = a[aRow + kk2];
                            if (aik == 0.0) continue;
                            int bRow = bOff + kk2 * n;
                            DoubleVector aikVec = DoubleVector.broadcast(species, aik);

                            int j = jj;
                            for (; j + vl * 4 <= jEnd; j += vl * 4) {
                                DoubleVector c0 = DoubleVector.fromArray(species, c, cRow + j);
                                c0 = aikVec.fma(DoubleVector.fromArray(species, b, bRow + j), c0);
                                c0.intoArray(c, cRow + j);

                                DoubleVector c1 = DoubleVector.fromArray(species, c, cRow + j + vl);
                                c1 = aikVec.fma(DoubleVector.fromArray(species, b, bRow + j + vl), c1);
                                c1.intoArray(c, cRow + j + vl);

                                DoubleVector c2 = DoubleVector.fromArray(species, c, cRow + j + vl * 2);
                                c2 = aikVec.fma(DoubleVector.fromArray(species, b, bRow + j + vl * 2), c2);
                                c2.intoArray(c, cRow + j + vl * 2);

                                DoubleVector c3 = DoubleVector.fromArray(species, c, cRow + j + vl * 3);
                                c3 = aikVec.fma(DoubleVector.fromArray(species, b, bRow + j + vl * 3), c3);
                                c3.intoArray(c, cRow + j + vl * 3);
                            }
                            for (; j + vl <= jEnd; j += vl) {
                                DoubleVector cv = DoubleVector.fromArray(species, c, cRow + j);
                                cv = aikVec.fma(DoubleVector.fromArray(species, b, bRow + j), cv);
                                cv.intoArray(c, cRow + j);
                            }
                            for (; j < jEnd; j++) {
                                c[cRow + j] += aik * b[bRow + j];
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Cache-blocked transpose of a row-major flat matrix.
     * dst[j*m+i] = src[i*n+j].
     *
     * @param src source flat array (m rows, n cols)
     * @param m   rows
     * @param n   cols
     * @return transposed flat array (n rows, m cols)
     */
    public static double[] flatTranspose(double[] src, int m, int n) {
        double[] dst = new double[n * m];
        int blockSize = Math.min(OPTIMAL_BLOCK_SIZE, Math.min(m, n));
        for (int i = 0; i < m; i += blockSize) {
            int iEnd = Math.min(i + blockSize, m);
            for (int j = 0; j < n; j += blockSize) {
                int jEnd = Math.min(j + blockSize, n);
                for (int ii = i; ii < iEnd; ii++) {
                    for (int jj = j; jj < jEnd; jj++) {
                        dst[jj * m + ii] = src[ii * n + jj];
                    }
                }
            }
        }
        return dst;
    }

    // ===================== Batch Flat GEMM =====================

    /**
     * Batch flat row-major GEMM: for each b in 0..batch-1,
     * C_b[m×n] = A_b[m×k] @ B_b[k×n].
     * a, b are contiguous arrays of batch*m*k and batch*k*n elements.
     * Returns a new double[batch*m*n] with all results.
     */
    public static double[] flatMmulBatched(double[] a, double[] b,
                                            int batch, int m, int k, int n) {
        double[] c = new double[batch * m * n];
        if (HpcIm2col.tryFlatDgemmBatch(batch, m, n, k, a, b, c)) {
            return c;
        }
        // Fallback: per-batch flat GEMM with SIMD blocked kernel
        int mk = m * k;
        int kn = k * n;
        int mn = m * n;
        for (int bi = 0; bi < batch; bi++) {
            flatMmulBlocked(a, bi * mk, m, k, b, bi * kn, n, c, bi * mn);
        }
        return c;
    }
}
