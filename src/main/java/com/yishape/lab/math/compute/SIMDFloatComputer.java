package com.yishape.lab.math.compute;

import com.yishape.lab.util.YishapeConfig;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.linalg.RereDoubleMatrix;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * SIMD运算 使用Java Vector API实现
 * 针对长度1000+的向量进行深度优化
 *
 * @author lteb2
 */
public class SIMDFloatComputer implements IFloatVectorComputer,Serializable {

    private static final YishapeLogger log = YishapeLogger.getLogger(SIMDFloatComputer.class);

    
    // 性能优化常量
    private static final VectorSpecies<Float> PREFERRED_SPECIES;
    private static final int VECTOR_LENGTH;
    private static final int OPTIMAL_BLOCK_SIZE;
    private static final int CACHE_LINE_SIZE = 64; // 典型缓存行大小
    private static final int L1_CACHE_SIZE = 32 * 1024; // 典型L1缓存大小
    private static final int L2_CACHE_SIZE = 256 * 1024; // 典型L2缓存大小

    // Strassen矩阵乘法常量
    /** 低于此尺寸使用标准分块乘法 */
    private static final int STRASSEN_THRESHOLD = 2048;
    /** 复杂度阈值 (m*n*p)，超过时才启用Strassen */
    private static final long STRASSEN_COMPLEXITY_THRESHOLD = 8_589_934_592L; // 2048³
    /** 系统属性开关 */
    private static final boolean USE_STRASSEN = !"false".equals(
        System.getProperty("yishape.strassen.enabled", "true"));

    /** ThreadLocal pool of scratch float[half][half] buffers for Strassen recursion. */
    private static final ThreadLocal<Deque<float[][]>> STRASSEN_POOL =
        ThreadLocal.withInitial(ArrayDeque::new);

    private static float[][] acquireScratch(int d) {
        Deque<float[][]> pool = STRASSEN_POOL.get();
        float[][] buf = pool.poll();
        if (buf != null && buf.length == d && buf[0].length == d) {
            for (int i = 0; i < d; i++) Arrays.fill(buf[i], 0.0f);
            return buf;
        }
        return new float[d][d];
    }

    private static void releaseScratch(float[][] buf) {
        if (buf == null) return;
        Deque<float[][]> pool = STRASSEN_POOL.get();
        if (pool.size() < 16) pool.push(buf);
    }

    // 性能监控 — 系统属性 + YishapeConfig profile 桥接
    private static boolean isPerformanceMonitoring() {
        return YishapeConfig.isSimdPerformanceMonitoring();
    }
    private static boolean isDetailedLogging() {
        return YishapeConfig.isSimdDetailedLogging();
    }
    
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
     * 智能选择最优向量规格
     */
    private static VectorSpecies<Float> selectOptimalSpecies() {
        return FloatVector.SPECIES_PREFERRED;
    }
    
    /**
     * 计算最优块大小
     */
    private static int calculateOptimalBlockSize() {
        // 基于L1缓存大小和向量长度计算最优块大小
        int elementsInL1 = L1_CACHE_SIZE / (Float.BYTES * 2); // 考虑读写两个数组
        int blockSize = Math.max(64, elementsInL1 / 4); // 保留1/4的L1缓存空间
        
        // 确保块大小是向量长度的倍数
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;
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
                    float vectorizationRate = (float) totalVectorizedOperations.get() / ops * 100;
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
        float vectorizationRate = ops > 0 ? (float) vecOps / ops * 100 : 0;

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
    private static void validateVectorInputs(float[] x1, float[] x2, String operationName) {
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
     * 单个向量的参数验证
     */
    private static void validateVectorInput(float[] x, String operationName) {
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
     * 为矩阵运算选择最优的向量species。
     * 至少需要能容纳4个元素才使用preferred，否则降级。
     */
    private static VectorSpecies<Float> selectMatrixSpecies() {
        return FloatVector.SPECIES_PREFERRED;
    }

    /**
     * 检查是否支持Java Vector API计算（加载、运算、归约烟测；失败则回退 SISD）
     *
     * @return true if Vector API is supported, false otherwise
     */
    public static boolean checkIfSupport() {
        try {
            VectorSpecies<Float> species = PREFERRED_SPECIES;
            int vl = species.length();
            if (vl <= 0) {
                return false;
            }
            float[] a = new float[vl];
            float[] b = new float[vl];
            java.util.Arrays.fill(a, 1.0f);
            java.util.Arrays.fill(b, 1.0f);
            FloatVector va = FloatVector.fromArray(species, a, 0);
            FloatVector vb = FloatVector.fromArray(species, b, 0);
            FloatVector vs = va.add(vb);
            float sum = vs.reduceLanes(VectorOperators.ADD);
            if (Float.isNaN(sum) || Float.isInfinite(sum)) {
                return false;
            }
            float expected = 2.0f * vl;
            return Math.abs(sum - expected) <= Math.ulp(expected) * 8f;
        } catch (Throwable ex) {
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
    public float[] binaryOperate(float[] x1, float[] x2, IFloatVectorComputer.BinaryOperation operation) {
        // 改进的参数验证和性能监控
        validateVectorInputs(x1, x2, "binaryOperate(vector, vector)");
        recordOperation(true, x1.length);

        final int length = x1.length;
        final float[] result = new float[length];
        
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
    private float[] binaryOperateSmall(float[] x1, float[] x2, IFloatVectorComputer.BinaryOperation operation, 
                                       float[] result, int length) {
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x1, i);
            FloatVector b = FloatVector.fromArray(PREFERRED_SPECIES, x2, i);
            FloatVector c = performBinaryOperation(a, b, operation);
            c.intoArray(result, i);
        }

        // 处理剩余元素
        performScalarRemainder(x1, x2, result, operation, i, length);
        return result;
    }
    
    /**
     * 大数据集的优化二元运算处理
     */
    private float[] binaryOperateOptimized(float[] x1, float[] x2, IFloatVectorComputer.BinaryOperation operation, 
                                           float[] result, int length) {
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
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x1, i);
            FloatVector b = FloatVector.fromArray(PREFERRED_SPECIES, x2, i);
            FloatVector c = performBinaryOperation(a, b, operation);
            c.intoArray(result, i);
        }
        
        // 处理最后的标量元素
        performScalarRemainder(x1, x2, result, operation, i, length);
        return result;
    }
    
    /**
     * 执行展开的二元运算
     */
    private void performUnrolledBinaryOperation(float[] x1, float[] x2, float[] result, 
                                              IFloatVectorComputer.BinaryOperation operation, int startIndex, int unrollFactor) {
        // 根据展开因子动态处理
        for (int u = 0; u < unrollFactor; u++) {
            int idx = startIndex + u * VECTOR_LENGTH;
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x1, idx);
            FloatVector b = FloatVector.fromArray(PREFERRED_SPECIES, x2, idx);
            FloatVector c = performBinaryOperation(a, b, operation);
            c.intoArray(result, idx);
        }
    }
    
    /**
     * 执行具体的二元运算
     */
    private FloatVector performBinaryOperation(FloatVector a, FloatVector b, IFloatVectorComputer.BinaryOperation operation) {
        return switch (operation) {
            case ADD -> a.add(b);
            case SUBTRACT -> a.sub(b);
            case MULTIPLY -> a.mul(b);
            case DIVIDE -> a.div(b);
            case REMAINDER -> {
                // 优化的取余计算
                FloatVector quotient = a.div(b);
                var intQuotient = quotient.convert(VectorOperators.F2I, 0);
                var truncated = intQuotient.convert(VectorOperators.I2F, 0);
                yield a.sub(truncated.mul(b));
            }
        };
    }
    
    /**
     * 处理标量剩余元素
     */
    private void performScalarRemainder(float[] x1, float[] x2, float[] result, 
                                       IFloatVectorComputer.BinaryOperation operation, int startIndex, int length) {
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
    public float[] binaryOperate(float[] x1, float x2, IFloatVectorComputer.BinaryOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        final int length = x1.length;
        final float[] result = new float[length];
        
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
    private float[] binaryOperateWithScalarSmall(float[] x1, float x2, IFloatVectorComputer.BinaryOperation operation, 
                                                  float[] result, int length) {
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(length);
        FloatVector scalarVector = FloatVector.broadcast(PREFERRED_SPECIES, x2);

        // 简化的向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x1, i);
            FloatVector c = performBinaryOperationWithScalar(a, scalarVector, x2, operation);
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
    private float[] binaryOperateWithScalarOptimized(float[] x1, float x2, IFloatVectorComputer.BinaryOperation operation, 
                                                      float[] result, int length) {
        final int unrollFactor = Math.min(4, calculateUnrollFactor(length)); // 限制展开因子
        final int blockSize = unrollFactor * VECTOR_LENGTH;
        final int upperBound = (length / blockSize) * blockSize;
        final FloatVector scalarVector = FloatVector.broadcast(PREFERRED_SPECIES, x2);
        
        int i = 0;
        
        // 简化的循环展开处理，去除乘法特殊处理
        for (; i < upperBound; i += blockSize) {
            performUnrolledBinaryOperationWithScalar(x1, scalarVector, x2, result, operation, i, unrollFactor);
        }
        
        // 处理剩余的向量化部分
        int vectorBound = PREFERRED_SPECIES.loopBound(length);
        for (; i < vectorBound; i += VECTOR_LENGTH) {
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x1, i);
            FloatVector c = performBinaryOperationWithScalar(a, scalarVector, x2, operation);
            c.intoArray(result, i);
        }
        
        // 处理最后的标量元素
        performScalarRemainderWithScalar(x1, x2, result, operation, i, length);
        return result;
    }
    
    /**
     * 执行展开的向量与标量二元运算
     */
    private void performUnrolledBinaryOperationWithScalar(float[] x1, FloatVector scalarVector, float x2, 
                                                          float[] result, IFloatVectorComputer.BinaryOperation operation, 
                                                          int startIndex, int unrollFactor) {
        for (int u = 0; u < unrollFactor; u++) {
            int idx = startIndex + u * VECTOR_LENGTH;
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x1, idx);
            FloatVector c = performBinaryOperationWithScalar(a, scalarVector, x2, operation);
            c.intoArray(result, idx);
        }
    }
    
    /**
     * 执行具体的向量与标量二元运算
     */
    private FloatVector performBinaryOperationWithScalar(FloatVector a, FloatVector scalarVector, 
                                                          float scalarValue, IFloatVectorComputer.BinaryOperation operation) {
        return switch (operation) {
            case ADD -> a.add(scalarVector);
            case SUBTRACT -> a.sub(scalarVector);
            case MULTIPLY -> a.mul(scalarVector);
            case DIVIDE -> a.div(scalarVector);
            case REMAINDER -> {
                // 优化的取余计算
                FloatVector quotient = a.div(scalarValue);
                var intQuotient = quotient.convert(VectorOperators.F2I, 0);
                FloatVector truncated = (FloatVector) intQuotient.convert(VectorOperators.I2F, 0);
                yield a.sub(truncated.mul(scalarValue));
            }
        };
    }
    
    /**
     * 处理向量与标量的标量剩余元素
     */
    private void performScalarRemainderWithScalar(float[] x1, float x2, float[] result, 
                                                  IFloatVectorComputer.BinaryOperation operation, int startIndex, int length) {
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
    public float[][] binaryOperate(float[][] x1, float[][] x2, IFloatVectorComputer.BinaryOperation operation) {
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

        if (x1.length == 0) return new float[0][0];

        // 创建结果矩阵
        float[][] result = new float[x1.length][x1[0].length];

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
    public float[][] binaryOperate(float[][] x1, float x2, IFloatVectorComputer.BinaryOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length == 0) return new float[0][0];

        // 创建结果矩阵
        float[][] result = new float[x1.length][x1[0].length];

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
    public float[] universalOperate(float[] x, IFloatVectorComputer.UniversalOperation operation, float additionalParam) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

        // 创建结果数组
        float[] result = new float[x.length];

        // 使用向量API进行通用运算
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(x.length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x, i);

            // 执行相应的通用运算
            FloatVector c;
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
                    c = a.max(0.0f);
                case SIGMOID ->
                    c = a.mul(0.5f).lanewise(VectorOperators.TANH).mul(0.5f).add(0.5f);
                case GELU -> {
                    // inner = 0.7978845608 * (x + 0.044715 * x^3)
                    FloatVector xCubed = a.mul(a).mul(a);
                    FloatVector inner = a.add(xCubed.mul(0.044715f)).mul(0.7978845608028654f);
                    FloatVector tanhV = inner.lanewise(VectorOperators.TANH);
                    // gelu = 0.5 * x * (1 + tanh)
                    c = a.mul(0.5f).mul(tanhV.add(1.0f));
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
                    result[i] = (float)Math.exp(x[i]);
                    break;
                case LOG:
                    result[i] = (float)Math.log(x[i]);
                    break;
                case LOG10:
                    result[i] = (float)Math.log10(x[i]);
                    break;
                case SIN:
                    result[i] = (float)Math.sin(x[i]);
                    break;
                case COS:
                    result[i] = (float)Math.cos(x[i]);
                    break;
                case TAN:
                    result[i] = (float)Math.tan(x[i]);
                    break;
                case ASIN:
                    result[i] = (float)Math.asin(x[i]);
                    break;
                case ACOS:
                    result[i] = (float)Math.acos(x[i]);
                    break;
                case ATAN:
                    result[i] = (float)Math.atan(x[i]);
                    break;
                case SQRT:
                    result[i] = (float)Math.sqrt(x[i]);
                    break;
                case ABS:
                    result[i] = Math.abs(x[i]);
                    break;
                case POW:
                    result[i] = (float)Math.pow(x[i], additionalParam);
                    break;
                case CBRT:
                    result[i] = (float)Math.cbrt(x[i]);
                    break;
                case COSH:
                    result[i] = (float)Math.cosh(x[i]);
                    break;
                case SINH:
                    result[i] = (float)Math.sinh(x[i]);
                    break;
                case TANH:
                    result[i] = (float)Math.tanh(x[i]);
                    break;
                case EXPM1:
                    result[i] = (float)Math.expm1(x[i]);
                    break;
                case LOG1P:
                    result[i] = (float)Math.log1p(x[i]);
                    break;
                case RELU:
                    result[i] = Math.max(0.0f, x[i]);
                    break;
                case SIGMOID:
                    result[i] = 0.5f * (float) Math.tanh(x[i] * 0.5f) + 0.5f;
                    break;
                case GELU: {
                    float inner = 0.7978845608028654f * (x[i] + 0.044715f * x[i] * x[i] * x[i]);
                    result[i] = 0.5f * x[i] * (1f + (float) Math.tanh(inner));
                    break;
                }
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
    public float[][] universalOperate(float[][] x, IFloatVectorComputer.UniversalOperation operation, float additionalParam) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x.length == 0) return new float[0][0];

        // 创建结果矩阵
        float[][] result = new float[x.length][x[0].length];

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
    public float[] elementWiseMin(float[] x1, float[] x2) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

        // 创建结果数组
        float[] result = new float[x1.length];

        // 使用向量API进行最小值运算
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x1, i);
            FloatVector b = FloatVector.fromArray(PREFERRED_SPECIES, x2, i);

            // 执行向量最小值操作
            FloatVector c = a.lanewise(VectorOperators.MIN, b);

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
    public float[] elementWiseMax(float[] x1, float[] x2) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

        // 创建结果数组
        float[] result = new float[x1.length];

        // 使用向量API进行最大值运算
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x1, i);
            FloatVector b = FloatVector.fromArray(PREFERRED_SPECIES, x2, i);

            // 执行向量最大值操作
            FloatVector c = a.lanewise(VectorOperators.MAX, b);

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
    public float[] negate(float[] x) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

        // 创建结果数组
        float[] result = new float[x.length];

        // 使用向量API进行取反运算
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(x.length);

        // 向量化处理
        for (; i < upperBound; i += VECTOR_LENGTH) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(PREFERRED_SPECIES, x, i);

            // 执行向量取反操作
            FloatVector c = a.lanewise(VectorOperators.NEG);

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
    public float[][] elementWiseMin(float[][] x1, float[][] x2) {
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

        if (x1.length == 0) return new float[0][0];

        // 创建结果矩阵
        float[][] result = new float[x1.length][x1[0].length];

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
    public float[][] elementWiseMax(float[][] x1, float[][] x2) {
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

        if (x1.length == 0) return new float[0][0];

        // 创建结果矩阵
        float[][] result = new float[x1.length][x1[0].length];

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
    public float[][] negate(float[][] x) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x.length == 0) return new float[0][0];

        // 创建结果矩阵
        float[][] result = new float[x.length][x[0].length];

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
    public float reduceOperate(float[] x, IFloatVectorComputer.ReduceOperation operation) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // Empty array contract: return identity/default for each operation
        if (x.length == 0) {
            return switch (operation) {
                case SUM, MEAN, VARIANCE, STANDARD_DEVIATION -> 0.0f;
                case PROD -> 1.0f;
                case MAX -> Float.NEGATIVE_INFINITY;
                case MIN -> Float.POSITIVE_INFINITY;
            };
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

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
                float mean = reduceOperate(x, IFloatVectorComputer.ReduceOperation.MEAN);
                return reduceOperateVariance(x, species, mean);
            }

            case STANDARD_DEVIATION -> {
                return (float)Math.sqrt(reduceOperate(x, IFloatVectorComputer.ReduceOperation.VARIANCE));
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
    private static float reduceOperateSum(float[] x, VectorSpecies<Float> species) {
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
    private static float reduceOperateSumSimple(float[] x, VectorSpecies<Float> species, int length) {
        int i = 0;
        int upperBound = species.loopBound(length);
        FloatVector acc = FloatVector.zero(species);
        FloatVector compensation = FloatVector.zero(species);

        for (; i < upperBound; i += species.length()) {
            FloatVector a = FloatVector.fromArray(species, x, i);
            FloatVector y = a.sub(compensation);
            FloatVector t = acc.add(y);
            compensation = t.sub(acc).sub(y);
            acc = t;
        }

        float sumResult = acc.reduceLanes(VectorOperators.ADD);

        for (; i < length; i++) {
            sumResult += x[i];
        }

        return sumResult;
    }
    
    /**
     * 轻量级优化的向量求和实现
     * 仅使用2个累加器以平衡性能和复杂度
     */
    private static float reduceOperateSumOptimizedLight(float[] x, VectorSpecies<Float> species, int length) {
        final int vectorLength = species.length();
        final int blockSize = vectorLength * 4; // 降低展开因子
        final int upperBound = (length / blockSize) * blockSize;
        
        // 使用2个累加器，避免过度复杂化
        FloatVector acc1 = FloatVector.zero(species);
        FloatVector acc2 = FloatVector.zero(species);
        
        int i = 0;
        
        // 简化的循环展开
        for (; i < upperBound; i += blockSize) {
            FloatVector a1 = FloatVector.fromArray(species, x, i);
            FloatVector a2 = FloatVector.fromArray(species, x, i + vectorLength);
            FloatVector a3 = FloatVector.fromArray(species, x, i + 2 * vectorLength);
            FloatVector a4 = FloatVector.fromArray(species, x, i + 3 * vectorLength);
            
            acc1 = acc1.add(a1).add(a3);
            acc2 = acc2.add(a2).add(a4);
        }
        
        // 合并累加器
        float sumResult = acc1.add(acc2).reduceLanes(VectorOperators.ADD);
        
        // 处理剩余的向量化部分
        int vectorBound = species.loopBound(length);
        for (; i < vectorBound; i += vectorLength) {
            FloatVector a = FloatVector.fromArray(species, x, i);
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
    private static float reduceOperateMinMax(float[] x, VectorSpecies<Float> species, boolean isMin) {
        // 使用向量API进行最小值或最大值运算
        int i = 0;
        int upperBound = species.loopBound(x.length);

        // 特殊处理：如果向量长度为0，直接返回
        if (x.length == 0) {
            return isMin ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
        }

        // 特殊处理：如果向量长度小于species.length，则直接使用标量计算
        if (x.length < species.length()) {
            float result = x[0];
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
        FloatVector resultVector = FloatVector.fromArray(species, x, i);
        i += species.length();

        // 向量化处理 — 仍有余量时使用分治法减少比较次数
        if (i < upperBound) {
            FloatVector acc = resultVector;

            while (i < upperBound) {
                int groupEnd = Math.min(i + species.length() * 4, upperBound);
                FloatVector groupResult = FloatVector.fromArray(species, x, i);
                i += species.length();

                while (i < groupEnd) {
                    FloatVector a = FloatVector.fromArray(species, x, i);
                    groupResult = groupResult.lanewise(isMin ? VectorOperators.MIN : VectorOperators.MAX, a);
                    i += species.length();
                }

                acc = acc.lanewise(isMin ? VectorOperators.MIN : VectorOperators.MAX, groupResult);
            }

            resultVector = acc;
        }

        // 直接使用向量归约方法，避免临时数组分配
        float result = resultVector.reduceLanes(isMin ? VectorOperators.MIN : VectorOperators.MAX);

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
    private static float reduceOperateVariance(float[] x, VectorSpecies<Float> species, float mean) {
        int i = 0;
        int upperBound = species.loopBound(x.length);

        // Kahan求和累加器
        FloatVector acc = FloatVector.zero(species);
        FloatVector compensation = FloatVector.zero(species);
        FloatVector meanVector = FloatVector.broadcast(species, mean);

        for (; i < upperBound; i += species.length()) {
            FloatVector a = FloatVector.fromArray(species, x, i);
            FloatVector diff = a.sub(meanVector);
            FloatVector squared = diff.mul(diff);

            // Kahan求和：y = squared - compensation, t = acc + y, compensation = (t - acc) - y
            FloatVector y = squared.sub(compensation);
            FloatVector t = acc.add(y);
            compensation = t.sub(acc).sub(y);
            acc = t;
        }

        float varResult = acc.reduceLanes(VectorOperators.ADD);

        for (; i < x.length; i++) {
            float diff = x[i] - mean;
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
    private static float reduceOperateProduct(float[] x, VectorSpecies<Float> species) {
        // 使用向量API进行乘积运算
        int i = 0;
        int upperBound = species.loopBound(x.length);

        // 初始化累乘器向量为1
        FloatVector acc = FloatVector.broadcast(species, 1.0f);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(species, x, i);

            // 累乘到累乘器
            acc = acc.mul(a);
        }

        // 归约累乘器中的值
        float productResult = acc.reduceLanes(VectorOperators.MUL);

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
    public float reduceOperate(float[][] x, IFloatVectorComputer.ReduceOperation operation) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // Empty matrix contract: return identity/default for each operation
        if (x.length == 0 || x[0].length == 0) {
            return switch (operation) {
                case SUM, MEAN, VARIANCE, STANDARD_DEVIATION -> 0.0f;
                case PROD -> 1.0f;
                case MAX -> Float.NEGATIVE_INFINITY;
                case MIN -> Float.POSITIVE_INFINITY;
            };
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = selectMatrixSpecies();

        switch (operation) {
            case SUM -> {
                float sumResult = 0.0f;
                // 优化：使用向量化操作直接计算
                for (float[] row : x) {
                    sumResult += reduceOperateSum(row, species);
                }
                return sumResult;
            }

            case MEAN -> {
                float totalSum = 0.0f;
                int totalElements = 0;
                // 优化：使用向量化操作直接计算
                for (float[] row : x) {
                    totalSum += reduceOperateSum(row, species);
                    totalElements += row.length;
                }
                return totalSum / totalElements;
            }

            case MIN -> {
                float minResult = Float.POSITIVE_INFINITY;
                // 优化：使用向量化操作直接计算
                for (float[] row : x) {
                    if (row.length > 0) {
                        // 只处理非空行
                        minResult = Math.min(minResult, reduceOperateMinMax(row, species, true));
                    }
                }
                return minResult;
            }

            case MAX -> {
                float maxResult = Float.NEGATIVE_INFINITY;
                // 优化：使用向量化操作直接计算
                for (float[] row : x) {
                    if (row.length > 0) {
                        // 只处理非空行
                        maxResult = Math.max(maxResult, reduceOperateMinMax(row, species, false));
                    }
                }
                return maxResult;
            }

            case VARIANCE -> {
                // 计算均值
                float mean = reduceOperate(x, IFloatVectorComputer.ReduceOperation.MEAN);
                // 计算方差：reduceOperateVariance divides by row length internally,
                // so multiply back by row length to accumulate raw SSD, then divide by totalCount
                float varianceSum = 0.0f;
                int totalCount = 0;
                for (float[] rowArray : x) {
                    varianceSum += reduceOperateVariance(rowArray, species, mean) * rowArray.length;
                    totalCount += rowArray.length;
                }
                return varianceSum / totalCount;
            }

            case STANDARD_DEVIATION -> {
                return (float)Math.sqrt(reduceOperate(x, IFloatVectorComputer.ReduceOperation.VARIANCE));
            }

            case PROD -> {
                float productResult = 1.0f;
                // 优化：使用向量化操作直接计算
                for (float[] row : x) {
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
    public float binaryReduceOperate(float[] x1, float[] x2, IFloatVectorComputer.BinaryReduceOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = selectMatrixSpecies();

        switch (operation) {
            case DOT -> {
                // 直接计算点积，避免中间数组分配
                return directDotProduct(x1, x2, species);
            }

            case L2_NORM -> {
                // 直接计算L2范数，避免中间数组分配
                return (float)Math.sqrt(directSumOfSquaredDifferences(x1, x2, species));
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
    private static float directDotProduct(float[] x1, float[] x2, VectorSpecies<Float> species) {
        int i = 0;
        int upperBound = species.loopBound(x1.length);
        
        // 初始化累加器向量
        FloatVector acc = FloatVector.zero(species);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(species, x1, i);
            FloatVector b = FloatVector.fromArray(species, x2, i);

            // 累加到累加器
            acc = acc.add(a.mul(b));
        }

        // 归约累加器中的值
        float sumResult = acc.reduceLanes(VectorOperators.ADD);

        // 处理剩余元素
        for (; i < x1.length; i++) {
            sumResult += x1[i] * x2[i];
        }

        return sumResult;
    }

    /**
     * 直接计算两个向量差的平方和，避免中间数组分配
     */
    private static float directSumOfSquaredDifferences(float[] x1, float[] x2, VectorSpecies<Float> species) {
        int i = 0;
        int upperBound = species.loopBound(x1.length);
        
        // 初始化累加器向量
        FloatVector acc = FloatVector.zero(species);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(species, x1, i);
            FloatVector b = FloatVector.fromArray(species, x2, i);

            // 计算差值
            FloatVector diff = a.sub(b);
            
            // 计算平方并累加到累加器
            acc = acc.add(diff.mul(diff));
        }

        // 归约累加器中的值
        float sumResult = acc.reduceLanes(VectorOperators.ADD);

        // 处理剩余元素
        for (; i < x1.length; i++) {
            float diff = x1[i] - x2[i];
            sumResult += diff * diff;
        }

        return sumResult;
    }

    /**
     * 直接计算两个向量差的绝对值和，避免中间数组分配
     */
    private static float directSumOfAbsoluteDifferences(float[] x1, float[] x2, VectorSpecies<Float> species) {
        int i = 0;
        int upperBound = species.loopBound(x1.length);
        
        // 初始化累加器向量
        FloatVector acc = FloatVector.zero(species);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(species, x1, i);
            FloatVector b = FloatVector.fromArray(species, x2, i);

            // 计算差值的绝对值并累加到累加器
            acc = acc.add(a.sub(b).abs());
        }

        // 归约累加器中的值
        float sumResult = acc.reduceLanes(VectorOperators.ADD);

        // 处理剩余元素
        for (; i < x1.length; i++) {
            sumResult += Math.abs(x1[i] - x2[i]);
        }

        return sumResult;
    }

    @Override
    public float binaryReduceOperate(float[][] x1, float[][] x2, IFloatVectorComputer.BinaryReduceOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length != x2.length || (x1.length > 0 && x1[0].length != x2[0].length)) {
            throw new IllegalArgumentException("矩阵维度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = selectMatrixSpecies();

        switch (operation) {
            case DOT -> {
                // 直接计算矩阵点积，避免中间数组分配
                return directMatrixDotProduct(x1, x2, species);
            }

            case L2_NORM -> {
                // 直接计算矩阵L2范数，避免中间数组分配
                return (float)Math.sqrt(directMatrixSumOfSquaredDifferences(x1, x2, species));
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
    private static float directMatrixDotProduct(float[][] x1, float[][] x2, VectorSpecies<Float> species) {
        float sumResult = 0.0f;
        
        // 对每一行进行向量化点积计算
        for (int row = 0; row < x1.length; row++) {
            sumResult += directDotProduct(x1[row], x2[row], species);
        }
        
        return sumResult;
    }

    /**
     * 直接计算两个矩阵差的平方和，避免中间数组分配
     */
    private static float directMatrixSumOfSquaredDifferences(float[][] x1, float[][] x2, VectorSpecies<Float> species) {
        float sumResult = 0.0f;
        
        // 对每一行进行向量化计算
        for (int row = 0; row < x1.length; row++) {
            sumResult += directSumOfSquaredDifferences(x1[row], x2[row], species);
        }
        
        return sumResult;
    }

    /**
     * 直接计算两个矩阵差的绝对值和，避免中间数组分配
     */
    private static float directMatrixSumOfAbsoluteDifferences(float[][] x1, float[][] x2, VectorSpecies<Float> species) {
        float sumResult = 0.0f;
        
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
    public boolean[] logicalCompare(float[] x1, float[] x2, IFloatVectorComputer.LogicalCompare operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用向量API进行逻辑运算
        int i = 0;
        int upperBound = species.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(species, x1, i);
            FloatVector b = FloatVector.fromArray(species, x2, i);

            // 执行向量逻辑运算操作
            VectorMask<Float> mask;
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
    public boolean[] logicalOperate(float[] x1, float[] x2, IFloatVectorComputer.LogicalOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用向量API进行逻辑运算
        int i = 0;
        int upperBound = species.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(species, x1, i);
            FloatVector b = FloatVector.fromArray(species, x2, i);

            // 创建掩码：非零值为true，零值为false
            VectorMask<Float> maskA = a.compare(VectorOperators.NE, 0.0f);
            VectorMask<Float> maskB = b.compare(VectorOperators.NE, 0.0f);

            // 执行向量逻辑运算操作
            VectorMask<Float> mask;
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
            // 将float值转换为布尔值：非零为true，零为false
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
    public boolean[] logicalOperate(float[] x1, IFloatVectorComputer.LogicalOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED;

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用向量API进行逻辑运算
        int i = 0;
        int upperBound = species.loopBound(x1.length);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(species, x1, i);

            // 执行向量逻辑运算操作
            VectorMask<Float> mask;
            switch (operation) {
                case NOT ->
                    // 对于NOT操作，我们将非零值视为true，零值视为false，然后取反
                    // 所以0变成true，非零变成false
                    mask = a.compare(VectorOperators.EQ, 0.0f);
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
    public boolean[][] logicalCompare(float[][] x1, float[][] x2, IFloatVectorComputer.LogicalCompare operation) {
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
    public boolean[][] logicalOperate(float[][] x1, IFloatVectorComputer.LogicalOperation operation) {
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
    public boolean[][] logicalOperate(float[][] x1, float[][] x2, IFloatVectorComputer.LogicalOperation operation) {
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
    public float[][] transpose(float[][] matrix) {
        return transposeOptimized(matrix);
    }

    @Override
    public float[][] transpose(float[] rowVector) {
        int length = rowVector.length;
        float[][] columnMatrix = new float[length][1];
        for (int i = 0; i < length; i++) {
            columnMatrix[i][0] = rowVector[i];
        }
        return columnMatrix;
    }

    @Override
    public float[][] mmul(float[][] a, float[][] b) {
        int m = a.length;          // 当前矩阵行数
        if (m == 0) return new float[0][];
        int otherRows = b.length;
        if (otherRows == 0) return new float[0][];
        int n = a[0].length;       // 当前矩阵列数
        int p = b[0].length;       // 另一个矩阵列数
        if (n != otherRows) {
            throw new IllegalArgumentException(
                "Matrix dimension mismatch for mmul: A is " + m + "x" + n + ", B is " + otherRows + "x" + p);
        }

        // 计算复杂度
        long complexity = (long) m * n * p;

        // Strassen路径：大矩阵且启用时使用O(n^2.807)算法
        if (USE_STRASSEN && complexity >= STRASSEN_COMPLEXITY_THRESHOLD
                && Math.min(m, Math.min(n, p)) >= STRASSEN_THRESHOLD) {
            return mmulStrassen(a, b, m, n, p);
        }

        // 阈值与 SISDDoubleComputer.mmul 对齐
        if (complexity < 4000) {
            // 极小矩阵：直接计算
            return naiveMultiply(a, b, m, n, p);
        } else if (complexity < 32000) {
            // 小矩阵：循环展开优化
            return unrolledMultiply(a, b, m, n, p);
        } else if (complexity < 8000000) {
            // 中等矩阵：分块算法 (~n<200)
            return blockMultiply(a, b, m, n, p);
        } else {
            // 大矩阵：并行分块算法
            return parallelBlockMultiply(a, b, m, n, p);
        }
    }
    
    // ===================== Matrix multiplication implementations =====================

    /**
     * 朴素矩阵乘法 / Naive matrix multiplication
     * i-k-j 顺序：缓存友好，b[k] 整行连续访问
     */
    private float[][] naiveMultiply(float[][] data, float[][] otherData, int m, int n, int p) {
        float[][] result = new float[m][p];

        for (int i = 0; i < m; i++) {
            float[] thisRow = data[i];
            float[] resultRow = result[i];

            for (int k = 0; k < n; k++) {
                float aik = thisRow[k];
                float[] otherRow = otherData[k];
                for (int j = 0; j < p; j++) {
                    resultRow[j] += aik * otherRow[j];
                }
            }
        }

        return result;
    }

    /**
     * 循环展开优化的矩阵乘法 / Loop unrolled matrix multiplication
     * i-k-j 顺序 + 4路 j 展开：缓存友好
     */
    private float[][] unrolledMultiply(float[][] data, float[][] otherData, int m, int n, int p) {
        float[][] result = new float[m][p];

        for (int i = 0; i < m; i++) {
            float[] thisRow = data[i];
            float[] resultRow = result[i];

            for (int k = 0; k < n; k++) {
                float aik = thisRow[k];
                float[] otherRow = otherData[k];

                int j = 0;
                for (; j < p - 3; j += 4) {
                    resultRow[j]     += aik * otherRow[j];
                    resultRow[j + 1] += aik * otherRow[j + 1];
                    resultRow[j + 2] += aik * otherRow[j + 2];
                    resultRow[j + 3] += aik * otherRow[j + 3];
                }
                for (; j < p; j++) {
                    resultRow[j] += aik * otherRow[j];
                }
            }
        }

        return result;
    }

    /**
     * 分块矩阵乘法 with SIMD FMA micro-kernel, using i-k-j loop order.
     * Reuses b[k] row in cache and broadcasts aik across j-dimension vectors.
     */
    private float[][] blockMultiply(float[][] data, float[][] otherData, int m, int n, int p) {
        float[][] result = new float[m][p];
        int blockSize = calculateMatrixBlockSize(m, n, p);
        final int vl = VECTOR_LENGTH;

        for (int ii = 0; ii < m; ii += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                for (int jj = 0; jj < p; jj += blockSize) {
                    int iEnd = Math.min(ii + blockSize, m);
                    int jEnd = Math.min(jj + blockSize, p);
                    int kEnd = Math.min(kk + blockSize, n);

                    for (int i = ii; i < iEnd; i++) {
                        float[] aRow = data[i];
                        float[] cRow = result[i];
                        for (int k = kk; k < kEnd; k++) {
                            float aik = aRow[k];
                            if (aik == 0.0f) continue;
                            float[] bRow = otherData[k];
                            FloatVector aikVec = FloatVector.broadcast(PREFERRED_SPECIES, aik);

                            int j = jj;
                            // 4x unrolled FMA: cRow[j] += aik * bRow[j]
                            for (; j + vl * 4 <= jEnd; j += vl * 4) {
                                FloatVector c0 = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j);
                                c0 = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j), c0);
                                c0.intoArray(cRow, j);

                                FloatVector c1 = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j + vl);
                                c1 = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j + vl), c1);
                                c1.intoArray(cRow, j + vl);

                                FloatVector c2 = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j + vl * 2);
                                c2 = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j + vl * 2), c2);
                                c2.intoArray(cRow, j + vl * 2);

                                FloatVector c3 = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j + vl * 3);
                                c3 = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j + vl * 3), c3);
                                c3.intoArray(cRow, j + vl * 3);
                            }
                            // Single vector cleanup
                            for (; j + vl <= jEnd; j += vl) {
                                FloatVector cVec = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j);
                                cVec = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j), cVec);
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
        return result;
    }

    /**
     * 并行分块 GEMM：按行粗粒度分区，每分区内执行 i-k-j FMA 微核。
     * 仅用于大矩阵（complexity >= 8M），避免小矩阵的并行调度开销。
     */
    private float[][] parallelBlockMultiply(float[][] a, float[][] b, int m, int n, int p) {
        float[][] c = new float[m][p];
        int blockSize = calculateMatrixBlockSize(m, n, p);
        java.util.concurrent.ForkJoinPool pool = RereDoubleMatrix.getThreadPool();

        if (pool == null || pool.isShutdown()) {
            return blockMultiply(a, b, m, n, p);
        }

        int numThreads = pool.getParallelism();
        if (numThreads <= 1 || m < blockSize * 2) {
            return blockMultiply(a, b, m, n, p);
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
                            float[] aRow = a[i];
                            float[] cRow = c[i];
                            for (int k = kk; k < kEnd; k++) {
                                float aik = aRow[k];
                                if (aik == 0.0f) continue;
                                float[] bRow = b[k];
                                FloatVector aikVec = FloatVector.broadcast(PREFERRED_SPECIES, aik);

                                int j = jj;
                                for (; j + vl * 4 <= jEnd; j += vl * 4) {
                                    FloatVector c0 = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j);
                                    c0 = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j), c0);
                                    c0.intoArray(cRow, j);

                                    FloatVector c1 = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j + vl);
                                    c1 = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j + vl), c1);
                                    c1.intoArray(cRow, j + vl);

                                    FloatVector c2 = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j + vl * 2);
                                    c2 = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j + vl * 2), c2);
                                    c2.intoArray(cRow, j + vl * 2);

                                    FloatVector c3 = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j + vl * 3);
                                    c3 = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j + vl * 3), c3);
                                    c3.intoArray(cRow, j + vl * 3);
                                }
                                for (; j + vl <= jEnd; j += vl) {
                                    FloatVector cv = FloatVector.fromArray(PREFERRED_SPECIES, cRow, j);
                                    cv = aikVec.fma(FloatVector.fromArray(PREFERRED_SPECIES, bRow, j), cv);
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
            return blockMultiply(a, b, m, n, p);
        }
        return c;
    }

    /**
     * 计算矩阵乘法的最优块大小
     */
    private static int calculateMatrixBlockSize(int m, int n, int p) {
        // 基于L2缓存大小和矩阵维度计算最优块大小
        long elementsInL2 = L2_CACHE_SIZE / (Float.BYTES * 3); // 3个矩阵的数据
        int blockSize = (int) Math.cbrt(elementsInL2 / 3); // 立方根估算

        // 限制在合理范围内
        blockSize = Math.max(32, Math.min(blockSize, 256));

        // 确保是向量长度的倍数
        return ((blockSize + VECTOR_LENGTH - 1) / VECTOR_LENGTH) * VECTOR_LENGTH;
    }

    // ===================== Optimized transpose =====================

    /**
     * 优化的矩阵转置操作 (kept for non-mmul callers)
     */
    private float[][] transposeOptimized(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] result = new float[cols][rows];

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

    // ======================== Strassen matrix multiplication ========================

    /**
     * Strassen O(n^2.807) matrix multiplication for large matrices.
     * Odd dimensions are padded and the result stripped.
     */
    private float[][] mmulStrassen(float[][] a, float[][] b, int m, int n, int p) {
        int mPad = (m + 1) & ~1;
        int nPad = (n + 1) & ~1;
        int pPad = (p + 1) & ~1;
        boolean padded = (mPad != m || nPad != n || pPad != p);

        float[][] aPad, bPad;
        if (padded) {
            aPad = new float[mPad][nPad];
            bPad = new float[nPad][pPad];
            for (int i = 0; i < m; i++) System.arraycopy(a[i], 0, aPad[i], 0, n);
            for (int i = 0; i < n; i++) System.arraycopy(b[i], 0, bPad[i], 0, p);
        } else {
            aPad = a;
            bPad = b;
        }

        if (mPad <= STRASSEN_THRESHOLD || nPad <= STRASSEN_THRESHOLD || pPad <= STRASSEN_THRESHOLD) {
            return blockMultiply(aPad, bPad, mPad, nPad, pPad);
        }

        int effective = Math.min(Math.min(mPad, nPad), pPad);
        effective = (effective + 1) & ~1;

        float[][] resultPad = strassenSquareRecursive(aPad, bPad, effective,
                0, 0, 0, 0, 0, 0, true);

        if (padded) {
            float[][] result = new float[m][p];
            for (int i = 0; i < m; i++) System.arraycopy(resultPad[i], 0, result[i], 0, p);
            return result;
        }
        return resultPad;
    }

    /**
     * Core Strassen recursion for common even dimension d.
     */
    private float[][] strassenSquareRecursive(float[][] a, float[][] b, int d,
            int aRowOff, int aColOff, int bRowOff, int bColOff,
            int cRowOff, int cColOff, boolean topLevel) {
        if (d <= STRASSEN_THRESHOLD) {
            float[][] aSub = acquireScratch(d);
            float[][] bSub = acquireScratch(d);
            for (int i = 0; i < d; i++) {
                System.arraycopy(a[aRowOff + i], aColOff, aSub[i], 0, d);
                System.arraycopy(b[bRowOff + i], bColOff, bSub[i], 0, d);
            }
            float[][] leafResult = blockMultiply(aSub, bSub, d, d, d);
            releaseScratch(aSub);
            releaseScratch(bSub);
            return leafResult;
        }

        int half = d >>> 1;
        float[][] M1, M2, M3, M4, M5, M6, M7;

        ExecutorService pool = RereDoubleMatrix.getThreadPool();
        boolean parallel = topLevel && pool != null && !pool.isShutdown();

        if (parallel) {
            List<Future<float[][]>> futures = new ArrayList<>(7);

            float[][] A11pA22 = addSubmatrices(a, aRowOff, aColOff, aRowOff + half, aColOff + half, half, true);
            float[][] B11pB22 = addSubmatrices(b, bRowOff, bColOff, bRowOff + half, bColOff + half, half, true);
            float[][] A21pA22 = addSubmatrices(a, aRowOff + half, aColOff, aRowOff + half, aColOff + half, half, true);
            float[][] B12mB22 = addSubmatrices(b, bRowOff, bColOff + half, bRowOff + half, bColOff + half, half, false);
            float[][] B21mB11 = addSubmatrices(b, bRowOff + half, bColOff, bRowOff, bColOff, half, false);
            float[][] A11pA12 = addSubmatrices(a, aRowOff, aColOff, aRowOff, aColOff + half, half, true);
            float[][] A21mA11 = addSubmatrices(a, aRowOff + half, aColOff, aRowOff, aColOff, half, false);
            float[][] B11pB12 = addSubmatrices(b, bRowOff, bColOff, bRowOff, bColOff + half, half, true);
            float[][] A12mA22 = addSubmatrices(a, aRowOff, aColOff + half, aRowOff + half, aColOff + half, half, false);
            float[][] B21pB22 = addSubmatrices(b, bRowOff + half, bColOff, bRowOff + half, bColOff + half, half, true);

            futures.add(pool.submit(() -> blockMultiply(A11pA22, B11pB22, half, half, half)));
            futures.add(pool.submit(() -> blockMultiply(A21pA22, getSub(b, bRowOff, bColOff, half), half, half, half)));
            futures.add(pool.submit(() -> blockMultiply(getSub(a, aRowOff, aColOff, half), B12mB22, half, half, half)));
            futures.add(pool.submit(() -> blockMultiply(getSub(a, aRowOff + half, aColOff + half, half), B21mB11, half, half, half)));
            futures.add(pool.submit(() -> blockMultiply(A11pA12, getSub(b, bRowOff + half, bColOff + half, half), half, half, half)));
            futures.add(pool.submit(() -> blockMultiply(A21mA11, B11pB12, half, half, half)));
            futures.add(pool.submit(() -> blockMultiply(A12mA22, B21pB22, half, half, half)));

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
                return blockMultiply(a, b, d, d, d);
            }
        } else {
            // Sequential Strassen with scratch buffer reuse
            float[][] a11 = acquireScratch(half);
            float[][] a22 = acquireScratch(half);
            float[][] b11 = acquireScratch(half);
            float[][] b22 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff, a11);
            getSubInto(a, aRowOff + half, aColOff + half, a22);
            getSubInto(b, bRowOff, bColOff, b11);
            getSubInto(b, bRowOff + half, bColOff + half, b22);

            float[][] scratch = acquireScratch(half);

            // M1 = (a11+a22) @ (b11+b22)
            addInto(a11, a22, scratch);
            float[][] m1a = acquireScratch(half);
            addInto(b11, b22, m1a);
            M1 = blockMultiply(scratch, m1a, half, half, half);
            releaseScratch(m1a);

            // M2 = (a21+a22) @ b11
            float[][] a21 = acquireScratch(half);
            getSubInto(a, aRowOff + half, aColOff, a21);
            addInto(a21, a22, scratch);
            M2 = blockMultiply(scratch, b11, half, half, half);

            // M3 = a11 @ (b12-b22)
            float[][] b12 = acquireScratch(half);
            getSubInto(b, bRowOff, bColOff + half, b12);
            subInto(b12, b22, scratch);
            M3 = blockMultiply(a11, scratch, half, half, half);
            releaseScratch(a11);

            // M4 = a22 @ (b21-b11)
            float[][] b21 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff, b21);
            subInto(b21, b11, scratch);
            releaseScratch(b11);
            M4 = blockMultiply(a22, scratch, half, half, half);
            releaseScratch(a22);

            // M5 = (a11+a12) @ b22
            float[][] a12 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff + half, a12);
            a11 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff, a11);
            addInto(a11, a12, scratch);
            releaseScratch(a11);
            releaseScratch(a12);
            M5 = blockMultiply(scratch, b22, half, half, half);
            releaseScratch(b22);

            // M6 = (a21-a11) @ (b11+b12)
            a11 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff, a11);
            subInto(a21, a11, scratch);
            releaseScratch(a21);
            float[][] b11b = acquireScratch(half);
            b11 = acquireScratch(half);
            getSubInto(b, bRowOff, bColOff, b11);
            addInto(b11, b12, b11b);
            releaseScratch(b11);
            releaseScratch(b12);
            M6 = blockMultiply(scratch, b11b, half, half, half);
            releaseScratch(b11b);

            // M7 = (a12-a22) @ (b21+b22)
            a12 = acquireScratch(half);
            a22 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff + half, a12);
            getSubInto(a, aRowOff + half, aColOff + half, a22);
            subInto(a12, a22, scratch);
            releaseScratch(a12);
            releaseScratch(a22);
            float[][] b22b = acquireScratch(half);
            b22 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff + half, b22);
            b21 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff, b21);
            addInto(b21, b22, b22b);
            releaseScratch(b21);
            releaseScratch(b22);
            M7 = blockMultiply(scratch, b22b, half, half, half);
            releaseScratch(b22b);
            releaseScratch(a11);
            releaseScratch(scratch);
        }

        // Combine: C11, C12, C21, C22
        float[][] result = new float[d][d];
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

    /** Element-wise matrix addition into pre-allocated output: out = a + b */
    private static void addInto(float[][] a, float[][] b, float[][] out) {
        int m = a.length;
        for (int i = 0; i < m; i++) {
            float[] ai = a[i], bi = b[i], oi = out[i];
            for (int j = 0; j < ai.length; j++) oi[j] = ai[j] + bi[j];
        }
    }

    /** Element-wise matrix subtraction into pre-allocated output: out = a - b */
    private static void subInto(float[][] a, float[][] b, float[][] out) {
        int m = a.length;
        for (int i = 0; i < m; i++) {
            float[] ai = a[i], bi = b[i], oi = out[i];
            for (int j = 0; j < ai.length; j++) oi[j] = ai[j] - bi[j];
        }
    }

    /** Extract a submatrix of size d x d (allocates result) */
    private static float[][] getSub(float[][] a, int rowOff, int colOff, int d) {
        float[][] r = new float[d][d];
        for (int i = 0; i < d; i++)
            System.arraycopy(a[rowOff + i], colOff, r[i], 0, d);
        return r;
    }

    /** Extract a submatrix into pre-allocated output */
    private static void getSubInto(float[][] a, int rowOff, int colOff, float[][] out) {
        int d = out.length;
        for (int i = 0; i < d; i++)
            System.arraycopy(a[rowOff + i], colOff, out[i], 0, d);
    }

    /** Add or subtract two submatrices of a common source array */
    private static float[][] addSubmatrices(float[][] a, int r1, int c1, int r2, int c2, int d, boolean add) {
        float[][] r = new float[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                r[i][j] = add ? a[r1 + i][c1 + j] + a[r2 + i][c2 + j]
                              : a[r1 + i][c1 + j] - a[r2 + i][c2 + j];
            }
        }
        return r;
    }

    // 针对特定形状优化的外积
    @Override
    public float[][] outer(float[] a, float[] b) {
        int m = a.length;
        int n = b.length;
        float[][] result = new float[m][n];
        return outerProductByRow(a, b, result);
    }

    private float[][] outerProductByRow(float[] a, float[] b, float[][] result) {
        int m = a.length;
        int n = b.length;
        final VectorSpecies<Float> species = selectMatrixSpecies();
        int vectorLength = species.length();

        // Loop unrolling factor for better performance
        final int UNROLL_FACTOR = 4;

        for (int i = 0; i < m; i++) {
            FloatVector aBroadcast = FloatVector.broadcast(species, a[i]);
            int j = 0;

            // Process multiple elements at once with loop unrolling
            int unrolledEnd = (n / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR);
            for (; j < unrolledEnd; j += vectorLength * UNROLL_FACTOR) {
                // Unroll 4 iterations to improve instruction-level parallelism
                FloatVector bVec0 = FloatVector.fromArray(species, b, j);
                FloatVector bVec1 = FloatVector.fromArray(species, b, j + vectorLength);
                FloatVector bVec2 = FloatVector.fromArray(species, b, j + 2 * vectorLength);
                FloatVector bVec3 = FloatVector.fromArray(species, b, j + 3 * vectorLength);
                
                FloatVector result0 = aBroadcast.mul(bVec0);
                FloatVector result1 = aBroadcast.mul(bVec1);
                FloatVector result2 = aBroadcast.mul(bVec2);
                FloatVector result3 = aBroadcast.mul(bVec3);
                
                result0.intoArray(result[i], j);
                result1.intoArray(result[i], j + vectorLength);
                result2.intoArray(result[i], j + 2 * vectorLength);
                result3.intoArray(result[i], j + 3 * vectorLength);
            }

            // Process vectorized elements
            int vectorizedEnd = (n / vectorLength) * vectorLength;
            for (; j < vectorizedEnd; j += vectorLength) {
                FloatVector bVec = FloatVector.fromArray(species, b, j);
                aBroadcast.mul(bVec).intoArray(result[i], j);
            }

            for (; j < n; j++) {
                result[i][j] = a[i] * b[j];
            }
        }
        return result;
    }

    @Override
    public float[] sign(float[] array) {
        float[] result = new float[array.length];
        final VectorSpecies<Float> species = selectMatrixSpecies();
        int vectorLength = species.length();
        int i = 0;

        FloatVector zero = FloatVector.zero(species);
        FloatVector one = FloatVector.broadcast(species, 1.0f);
        FloatVector minusOne = FloatVector.broadcast(species, -1.0f);

        // Loop unrolling factor for better performance
        final int UNROLL_FACTOR = 4;
        
        // Process multiple elements at once with loop unrolling
        int unrolledEnd = (array.length / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR);
        for (; i < unrolledEnd; i += vectorLength * UNROLL_FACTOR) {
            // Unroll 4 iterations to improve instruction-level parallelism
            FloatVector vec0 = FloatVector.fromArray(species, array, i);
            FloatVector vec1 = FloatVector.fromArray(species, array, i + vectorLength);
            FloatVector vec2 = FloatVector.fromArray(species, array, i + 2 * vectorLength);
            FloatVector vec3 = FloatVector.fromArray(species, array, i + 3 * vectorLength);

            // 使用三元逻辑实现
            VectorMask<Float> gtZero0 = vec0.compare(VectorOperators.GT, 0.0f);
            VectorMask<Float> ltZero0 = vec0.lt(0.0f);
            VectorMask<Float> isNaN0 = vec0.eq(vec0).not();
            VectorMask<Float> gtZero1 = vec1.compare(VectorOperators.GT, 0.0f);
            VectorMask<Float> ltZero1 = vec1.lt(0.0f);
            VectorMask<Float> isNaN1 = vec1.eq(vec1).not();
            VectorMask<Float> gtZero2 = vec2.compare(VectorOperators.GT, 0.0f);
            VectorMask<Float> ltZero2 = vec2.lt(0.0f);
            VectorMask<Float> isNaN2 = vec2.eq(vec2).not();
            VectorMask<Float> gtZero3 = vec3.compare(VectorOperators.GT, 0.0f);
            VectorMask<Float> ltZero3 = vec3.lt(0.0f);
            VectorMask<Float> isNaN3 = vec3.eq(vec3).not();

            FloatVector signVec0 = zero.blend(one, gtZero0).blend(minusOne, ltZero0).blend(vec0, isNaN0);
            FloatVector signVec1 = zero.blend(one, gtZero1).blend(minusOne, ltZero1).blend(vec1, isNaN1);
            FloatVector signVec2 = zero.blend(one, gtZero2).blend(minusOne, ltZero2).blend(vec2, isNaN2);
            FloatVector signVec3 = zero.blend(one, gtZero3).blend(minusOne, ltZero3).blend(vec3, isNaN3);

            signVec0.intoArray(result, i);
            signVec1.intoArray(result, i + vectorLength);
            signVec2.intoArray(result, i + 2 * vectorLength);
            signVec3.intoArray(result, i + 3 * vectorLength);
        }

        // Process vectorized elements
        int vectorizedEnd = (array.length / vectorLength) * vectorLength;
        for (; i < vectorizedEnd; i += vectorLength) {
            FloatVector vec = FloatVector.fromArray(species, array, i);

            // 使用三元逻辑实现
            VectorMask<Float> gtZero = vec.compare(VectorOperators.GT, 0.0f);
            VectorMask<Float> ltZero = vec.lt(0.0f);
            VectorMask<Float> isNaN = vec.eq(vec).not();

            FloatVector signVec = zero.blend(one, gtZero).blend(minusOne, ltZero).blend(vec, isNaN);
            signVec.intoArray(result, i);
        }

        // 处理尾部
        for (; i < array.length; i++) {
            result[i] = signScalar(array[i]);
        }

        return result;
    }

    @Override
    public float[][] sign(float[][] array) {
        // 参数验证
        if (array == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // 获取矩阵维度
        int rows = array.length;
        int cols = array.length > 0 ? array[0].length : 0;
        
        // 创建结果矩阵
        float[][] result = new float[rows][cols];
        
        // 对每一行进行向量化符号运算
        for (int row = 0; row < rows; row++) {
            result[row] = sign(array[row]);
        }

        return result;
    }
    
    private float signScalar(float value) {
        if (Float.isNaN(value)) {
            return Float.NaN;
        }
        if (value > 0) {
            return 1.0f;
        }
        if (value < 0) {
            return -1.0f;
        }
        return 0.0f;
    }

    @Override
    public float[] diff(float[] array, int stride) {
        if (stride <= 0) {
            throw new IllegalArgumentException("步长必须为正整数");
        }
        if (array.length <= stride) {
            return new float[0];
        }
        final VectorSpecies<Float> species = PREFERRED_SPECIES;
        int vectorLength = species.length();
        int resultLength = array.length - stride;
        float[] result = new float[resultLength];
        int i = 0;

        final int UNROLL_FACTOR = 4;

        // 安全边界：确保 i + stride + vectorLength * UNROLL_FACTOR 不越界
        int safeLimit = Math.max(0, array.length - vectorLength * UNROLL_FACTOR);
        int unrolledEnd = Math.min(
            (resultLength / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR),
            Math.min(resultLength, Math.max(0, safeLimit - stride))
        );

        for (; i < unrolledEnd; i += vectorLength * UNROLL_FACTOR) {
            FloatVector current0 = FloatVector.fromArray(species, array, i);
            FloatVector future0 = FloatVector.fromArray(species, array, i + stride);
            FloatVector current1 = FloatVector.fromArray(species, array, i + vectorLength);
            FloatVector future1 = FloatVector.fromArray(species, array, i + stride + vectorLength);
            FloatVector current2 = FloatVector.fromArray(species, array, i + 2 * vectorLength);
            FloatVector future2 = FloatVector.fromArray(species, array, i + stride + 2 * vectorLength);
            FloatVector current3 = FloatVector.fromArray(species, array, i + 3 * vectorLength);
            FloatVector future3 = FloatVector.fromArray(species, array, i + stride + 3 * vectorLength);

            FloatVector diff0 = future0.sub(current0);
            FloatVector diff1 = future1.sub(current1);
            FloatVector diff2 = future2.sub(current2);
            FloatVector diff3 = future3.sub(current3);

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
            FloatVector current = FloatVector.fromArray(species, array, i);
            FloatVector future = FloatVector.fromArray(species, array, i + stride);

            FloatVector diff = future.sub(current);
            diff.intoArray(result, i);
        }

        for (; i < resultLength; i++) {
            result[i] = array[i + stride] - array[i];
        }

        return result;
    }
}
