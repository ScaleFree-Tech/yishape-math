package com.yishape.lab.math.compute;

import java.io.Serializable;
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
    
    // 性能优化常量
    private static final VectorSpecies<Float> PREFERRED_SPECIES;
    private static final int VECTOR_LENGTH;
    private static final int OPTIMAL_BLOCK_SIZE;
    private static final int CACHE_LINE_SIZE = 64; // 典型缓存行大小
    private static final int L1_CACHE_SIZE = 32 * 1024; // 典型L1缓存大小
    private static final int L2_CACHE_SIZE = 256 * 1024; // 典型L2缓存大小
    
    // 性能监控常量
    private static final boolean ENABLE_PERFORMANCE_MONITORING = Boolean.parseBoolean(
        System.getProperty("simd.performance.monitoring", "false"));
    private static final boolean ENABLE_DETAILED_LOGGING = Boolean.parseBoolean(
        System.getProperty("simd.detailed.logging", "false"));
    
    // 性能统计
    private static volatile long totalOperations = 0;
    private static volatile long totalVectorizedOperations = 0;
    private static volatile long totalScalarFallbacks = 0;
    
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
        VectorSpecies<Float> preferred = FloatVector.SPECIES_PREFERRED;
        
        // 对于长度1000+的向量，优先使用较大的向量长度
        if (preferred.length() >= 8) {
            return preferred;
        } else if (FloatVector.SPECIES_512.length() > 0) {
            return FloatVector.SPECIES_512;
        } else if (FloatVector.SPECIES_256.length() > 0) {
            return FloatVector.SPECIES_256;
        } else {
            return FloatVector.SPECIES_128;
        }
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
        if (ENABLE_PERFORMANCE_MONITORING) {
            totalOperations++;
            if (vectorized) {
                totalVectorizedOperations++;
            } else {
                totalScalarFallbacks++;
            }
            
            if (ENABLE_DETAILED_LOGGING && totalOperations % 10000 == 0) {
                float vectorizationRate = (float) totalVectorizedOperations / totalOperations * 100;
                System.out.printf("SIMD Stats: Operations=%d, Vectorization Rate=%.2f%%, Data Size=%d%n", 
                    totalOperations, vectorizationRate, dataSize);
            }
        }
    }
    
    /**
     * 性能监控：获取性能统计
     */
    public static String getPerformanceStats() {
        if (!ENABLE_PERFORMANCE_MONITORING) {
            return "Performance monitoring is disabled. Enable with -Dsimd.performance.monitoring=true";
        }
        
        float vectorizationRate = totalOperations > 0 ? 
            (float) totalVectorizedOperations / totalOperations * 100 : 0;
        
        return String.format(
            "SIMD Performance Statistics:%n" +
            "Total Operations: %d%n" +
            "Vectorized Operations: %d%n" +
            "Scalar Fallbacks: %d%n" +
            "Vectorization Rate: %.2f%%%n" +
            "Optimal Vector Length: %d%n" +
            "Optimal Block Size: %d",
            totalOperations, totalVectorizedOperations, totalScalarFallbacks,
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
     * 检查是否支持Java Vector API计算
     *
     * @return true if Vector API is supported, false otherwise
     */
    public static boolean checkIfSupport() {
        try {
            // 测试基本的Vector API操作
            VectorSpecies<Float> species = PREFERRED_SPECIES;
            System.out.println("Optimal SPECIES Length: " + species.length());
            System.out.println("Optimal Block Size: " + OPTIMAL_BLOCK_SIZE);
            return true;
        } catch (UnsupportedOperationException | NoClassDefFoundError | ExceptionInInitializerError e) {
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

        if (x1.length > 0 && x1[0].length != x2[0].length) {
            throw new IllegalArgumentException("矩阵列数必须相同");
        }

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

        // 对于需要非空检查的操作
        if ((operation == IFloatVectorComputer.ReduceOperation.MEAN
                || operation == IFloatVectorComputer.ReduceOperation.MIN
                || operation == IFloatVectorComputer.ReduceOperation.MAX
                || operation == IFloatVectorComputer.ReduceOperation.VARIANCE
                || operation == IFloatVectorComputer.ReduceOperation.STANDARD_DEVIATION
                || operation == IFloatVectorComputer.ReduceOperation.PROD)
                && x.length == 0) {
            if (operation == IFloatVectorComputer.ReduceOperation.PROD) {
                // 空向量的乘积定义为1
                return 1.0f;
            }
            throw new IllegalArgumentException("向量不能为空");
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

        // 简单的向量化循环，避免过度优化
        for (; i < upperBound; i += species.length()) {
            FloatVector a = FloatVector.fromArray(species, x, i);
            acc = acc.add(a);
        }

        float sumResult = acc.reduceLanes(VectorOperators.ADD);
        
        // 处理剩余元素
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

        // 向量化处理 - 使用分治法减少比较次数
        int remainingElements = upperBound - i;
        if (remainingElements >= species.length()) {
            // 如果剩余元素足够多，使用分治法
            FloatVector acc = resultVector;
            
            // 分组处理以减少比较次数
            while (i < upperBound) {
                int groupEnd = Math.min(i + species.length() * 4, upperBound); // 处理4个向量组
                FloatVector groupResult = FloatVector.fromArray(species, x, i);
                i += species.length();
                
                // 在组内进行比较
                while (i < groupEnd) {
                    FloatVector a = FloatVector.fromArray(species, x, i);
                    if (isMin) {
                        groupResult = groupResult.lanewise(VectorOperators.MIN, a);
                    } else {
                        groupResult = groupResult.lanewise(VectorOperators.MAX, a);
                    }
                    i += species.length();
                }
                
                // 将组结果与累积结果比较
                if (isMin) {
                    acc = acc.lanewise(VectorOperators.MIN, groupResult);
                } else {
                    acc = acc.lanewise(VectorOperators.MAX, groupResult);
                }
            }
            
            resultVector = acc;
        } else {
            // 常规处理
            for (; i < upperBound; i += species.length()) {
                // 从数组加载向量
                FloatVector a = FloatVector.fromArray(species, x, i);

                // 更新最小值或最大值
                if (isMin) {
                    resultVector = resultVector.lanewise(VectorOperators.MIN, a);
                } else {
                    resultVector = resultVector.lanewise(VectorOperators.MAX, a);
                }
            }
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
     *
     * @param x 向量
     * @param species 向量规格
     * @param mean 均值
     * @return 向量元素的方差
     */
    private static float reduceOperateVariance(float[] x, VectorSpecies<Float> species, float mean) {
        // 使用向量API进行方差运算
        int i = 0;
        int upperBound = species.loopBound(x.length);

        // 初始化累加器向量
        FloatVector acc = FloatVector.zero(species);

        // 创建均值广播向量
        FloatVector meanVector = FloatVector.broadcast(species, mean);

        // 向量化处理
        for (; i < upperBound; i += species.length()) {
            // 从数组加载向量
            FloatVector a = FloatVector.fromArray(species, x, i);

            // 计算 (x - mean)
            FloatVector diff = a.sub(meanVector);

            // 计算 (x - mean)^2
            FloatVector squared = diff.mul(diff);

            // 累加到累加器
            acc = acc.add(squared);
        }

        // 归约累加器中的值
        float varResult = acc.reduceLanes(VectorOperators.ADD);

        // 处理剩余元素
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

        // 对于需要非空检查的操作
        if ((operation == IFloatVectorComputer.ReduceOperation.MEAN
                || operation == IFloatVectorComputer.ReduceOperation.MIN
                || operation == IFloatVectorComputer.ReduceOperation.MAX
                || operation == IFloatVectorComputer.ReduceOperation.VARIANCE
                || operation == IFloatVectorComputer.ReduceOperation.STANDARD_DEVIATION
                || operation == IFloatVectorComputer.ReduceOperation.PROD)
                && (x.length == 0 || x[0].length == 0)) {
            throw new IllegalArgumentException("矩阵不能为空");
        }

        // 获取最优向量规格
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;

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
                // 计算方差
                float varianceSum = 0.0f;
                int totalCount = 0;
                // 优化：使用向量化操作直接计算
                for (float[] rowArray : x) {
                    varianceSum += reduceOperateVariance(rowArray, species, mean);
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
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;

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
        VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;

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
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] result = new float[cols][rows];

        final VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;
        int vectorLength = species.length();

        // Loop unrolling factor for better performance
        final int UNROLL_FACTOR = 4;

        for (int i = 0; i < rows; i++) {
            int j = 0;
            // Process multiple elements at once with loop unrolling
            int unrolledEnd = (cols / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR);
            for (; j < unrolledEnd; j += vectorLength * UNROLL_FACTOR) {
                // Unroll 4 iterations to improve instruction-level parallelism
                FloatVector vec0 = FloatVector.fromArray(species, matrix[i], j);
                FloatVector vec1 = FloatVector.fromArray(species, matrix[i], j + vectorLength);
                FloatVector vec2 = FloatVector.fromArray(species, matrix[i], j + 2 * vectorLength);
                FloatVector vec3 = FloatVector.fromArray(species, matrix[i], j + 3 * vectorLength);

                // 将向量元素存储到转置矩阵的对应位置
                for (int k = 0; k < vectorLength; k++) {
                    result[j + k][i] = vec0.lane(k);
                    result[j + k + vectorLength][i] = vec1.lane(k);
                    result[j + k + 2 * vectorLength][i] = vec2.lane(k);
                    result[j + k + 3 * vectorLength][i] = vec3.lane(k);
                }
            }
            
            // 向量化处理
            int vectorizedEnd = (cols / vectorLength) * vectorLength;
            for (; j < vectorizedEnd; j += vectorLength) {
                FloatVector vec = FloatVector.fromArray(species, matrix[i], j);
                // 将向量元素存储到转置矩阵的对应位置
                for (int k = 0; k < vectorLength; k++) {
                    result[j + k][i] = vec.lane(k);
                }
            }
            
            // 处理剩余元素
            for (; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    @Override
    public float[][] transpose(float[] rowVector) {
        int length = rowVector.length;
        float[][] columnMatrix = new float[length][1];
        final VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;
        int vectorLength = species.length();

        int i = 0;
        
        // Loop unrolling factor for better performance
        final int UNROLL_FACTOR = 4;
        
        // Process multiple elements at once with loop unrolling
        int unrolledEnd = (length / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR);
        for (; i < unrolledEnd; i += vectorLength * UNROLL_FACTOR) {
            // Unroll 4 iterations to improve instruction-level parallelism
            FloatVector vec0 = FloatVector.fromArray(species, rowVector, i);
            FloatVector vec1 = FloatVector.fromArray(species, rowVector, i + vectorLength);
            FloatVector vec2 = FloatVector.fromArray(species, rowVector, i + 2 * vectorLength);
            FloatVector vec3 = FloatVector.fromArray(species, rowVector, i + 3 * vectorLength);

            // 将向量中的每个元素存储到列矩阵中
            for (int j = 0; j < vectorLength; j++) {
                columnMatrix[i + j][0] = vec0.lane(j);
                columnMatrix[i + j + vectorLength][0] = vec1.lane(j);
                columnMatrix[i + j + 2 * vectorLength][0] = vec2.lane(j);
                columnMatrix[i + j + 3 * vectorLength][0] = vec3.lane(j);
            }
        }

        // 向量化处理
        int vectorizedEnd = (length / vectorLength) * vectorLength;
        for (; i < vectorizedEnd; i += vectorLength) {
            FloatVector vec = FloatVector.fromArray(species, rowVector, i);

            // 将向量中的每个元素存储到列矩阵中
            for (int j = 0; j < vectorLength; j++) {
                columnMatrix[i + j][0] = vec.lane(j);
            }
        }

        // 处理剩余元素
        for (; i < length; i++) {
            columnMatrix[i][0] = rowVector[i];
        }

        return columnMatrix;
    }

    @Override
    public float[][] mmul(float[][] a, float[][] b) {
        int m = a.length;          // 当前矩阵行数
        int n = a[0].length;       // 当前矩阵列数
        int p = b[0].length;       // 另一个矩阵列数

        // 计算复杂度
        long complexity = (long) m * n * p;

        if (complexity < 8000) {
            // 小矩阵：直接计算
            return mmulDirectOptimized(a, b, m, n, p);
        } else {
            // 大矩阵：分块算法优化
            return mmulBlockOptimized(a, b, m, n, p);
        }
    }
    
    /**
     * 优化的直接矩阵乘法，针对长度1000+对应优化
     */
    private float[][] mmulDirectOptimized(float[][] a, float[][] b, int m, int n, int p) {
        // 转置b矩阵以优化内存访问模式
        float[][] bT = transposeOptimized(b);
        float[][] c = new float[m][p];
        
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
     * 优化的点积计算，针对长向量优化
     */
    private float computeDotProductOptimized(float[] x, float[] y, int length, int blockSize, int unrollFactor) {
        if (length < blockSize) {
            return computeDotProductSimple(x, y, length);
        }
        
        // 使用多个累加器提高指令级并行
        FloatVector acc1 = FloatVector.zero(PREFERRED_SPECIES);
        FloatVector acc2 = FloatVector.zero(PREFERRED_SPECIES);
        FloatVector acc3 = FloatVector.zero(PREFERRED_SPECIES);
        FloatVector acc4 = FloatVector.zero(PREFERRED_SPECIES);
        
        int i = 0;
        final int upperBound = (length / blockSize) * blockSize;
        
        // 循环展开处理主要数据
        for (; i < upperBound; i += blockSize) {
            for (int u = 0; u < unrollFactor; u += 4) {
                int idx = i + u * VECTOR_LENGTH;
                
                if (idx + 3 * VECTOR_LENGTH < upperBound) {
                    FloatVector ax1 = FloatVector.fromArray(PREFERRED_SPECIES, x, idx);
                    FloatVector ay1 = FloatVector.fromArray(PREFERRED_SPECIES, y, idx);
                    FloatVector ax2 = FloatVector.fromArray(PREFERRED_SPECIES, x, idx + VECTOR_LENGTH);
                    FloatVector ay2 = FloatVector.fromArray(PREFERRED_SPECIES, y, idx + VECTOR_LENGTH);
                    FloatVector ax3 = FloatVector.fromArray(PREFERRED_SPECIES, x, idx + 2 * VECTOR_LENGTH);
                    FloatVector ay3 = FloatVector.fromArray(PREFERRED_SPECIES, y, idx + 2 * VECTOR_LENGTH);
                    FloatVector ax4 = FloatVector.fromArray(PREFERRED_SPECIES, x, idx + 3 * VECTOR_LENGTH);
                    FloatVector ay4 = FloatVector.fromArray(PREFERRED_SPECIES, y, idx + 3 * VECTOR_LENGTH);
                    
                    acc1 = acc1.add(ax1.mul(ay1));
                    acc2 = acc2.add(ax2.mul(ay2));
                    acc3 = acc3.add(ax3.mul(ay3));
                    acc4 = acc4.add(ax4.mul(ay4));
                } else {
                    // 处理剩余的向量
                    for (int v = u; v < unrollFactor && idx < upperBound; v++, idx += VECTOR_LENGTH) {
                        FloatVector ax = FloatVector.fromArray(PREFERRED_SPECIES, x, idx);
                        FloatVector ay = FloatVector.fromArray(PREFERRED_SPECIES, y, idx);
                        acc1 = acc1.add(ax.mul(ay));
                    }
                    break;
                }
            }
        }
        
        // 合并累加器
        FloatVector totalAcc = acc1.add(acc2).add(acc3).add(acc4);
        float result = totalAcc.reduceLanes(VectorOperators.ADD);
        
        // 处理剩余的向量化部分
        int vectorBound = PREFERRED_SPECIES.loopBound(length);
        for (; i < vectorBound; i += VECTOR_LENGTH) {
            FloatVector ax = FloatVector.fromArray(PREFERRED_SPECIES, x, i);
            FloatVector ay = FloatVector.fromArray(PREFERRED_SPECIES, y, i);
            result += ax.mul(ay).reduceLanes(VectorOperators.ADD);
        }
        
        // 处理最后的标量元素
        for (; i < length; i++) {
            result += x[i] * y[i];
        }
        
        return result;
    }
    
    /**
     * 简单的点积计算
     */
    private float computeDotProductSimple(float[] x, float[] y, int length) {
        int i = 0;
        int upperBound = PREFERRED_SPECIES.loopBound(length);
        FloatVector acc = FloatVector.zero(PREFERRED_SPECIES);

        for (; i < upperBound; i += VECTOR_LENGTH) {
            FloatVector ax = FloatVector.fromArray(PREFERRED_SPECIES, x, i);
            FloatVector ay = FloatVector.fromArray(PREFERRED_SPECIES, y, i);
            acc = acc.add(ax.mul(ay));
        }

        float result = acc.reduceLanes(VectorOperators.ADD);
        
        for (; i < length; i++) {
            result += x[i] * y[i];
        }

        return result;
    }
    
    /**
     * 优化的矩阵转置操作
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
    
    /**
     * 优化的分块矩阵乘法
     */
    private float[][] mmulBlockOptimized(float[][] a, float[][] b, int m, int n, int p) {
        float[][] c = new float[m][p];
        float[][] bT = transposeOptimized(b);
        
        // 动态计算最优块大小
        int optimalBlockSize = calculateMatrixBlockSize(m, n, p);
        
        for (int i0 = 0; i0 < m; i0 += optimalBlockSize) {
            for (int j0 = 0; j0 < p; j0 += optimalBlockSize) {
                for (int k0 = 0; k0 < n; k0 += optimalBlockSize) {
                    int iEnd = Math.min(i0 + optimalBlockSize, m);
                    int jEnd = Math.min(j0 + optimalBlockSize, p);
                    int kEnd = Math.min(k0 + optimalBlockSize, n);
                    
                    processMatrixBlockOptimized(a, bT, c, i0, iEnd, j0, jEnd, k0, kEnd);
                }
            }
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
    
    /**
     * 优化的矩阵块处理
     */
    private void processMatrixBlockOptimized(float[][] a, float[][] bT, float[][] c,
            int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd) {
        
        for (int i = iStart; i < iEnd; i++) {
            for (int j = jStart; j < jEnd; j++) {
                // 使用优化的点积计算部分和
                float partialSum = computePartialDotProduct(a[i], bT[j], kStart, kEnd);
                c[i][j] += partialSum;
            }
        }
    }
    
    /**
     * 计算部分点积的优化实现
     */
    private float computePartialDotProduct(float[] x, float[] y, int start, int end) {
        final int length = end - start;
        final int unrollFactor = calculateUnrollFactor(length);
        final int blockSize = unrollFactor * VECTOR_LENGTH;
        
        if (length < blockSize) {
            return computePartialDotProductSimple(x, y, start, end);
        }
        
        // 使用多个累加器
        FloatVector acc1 = FloatVector.zero(PREFERRED_SPECIES);
        FloatVector acc2 = FloatVector.zero(PREFERRED_SPECIES);
        
        int i = start;
        final int upperBound = start + (length / blockSize) * blockSize;
        
        for (; i < upperBound; i += blockSize) {
            for (int u = 0; u < unrollFactor; u += 2) {
                int idx = i + u * VECTOR_LENGTH;
                
                if (idx + VECTOR_LENGTH < upperBound) {
                    FloatVector ax1 = FloatVector.fromArray(PREFERRED_SPECIES, x, idx);
                    FloatVector ay1 = FloatVector.fromArray(PREFERRED_SPECIES, y, idx);
                    FloatVector ax2 = FloatVector.fromArray(PREFERRED_SPECIES, x, idx + VECTOR_LENGTH);
                    FloatVector ay2 = FloatVector.fromArray(PREFERRED_SPECIES, y, idx + VECTOR_LENGTH);
                    
                    acc1 = acc1.add(ax1.mul(ay1));
                    acc2 = acc2.add(ax2.mul(ay2));
                } else {
                    for (int v = u; v < unrollFactor && idx < upperBound; v++, idx += VECTOR_LENGTH) {
                        FloatVector ax = FloatVector.fromArray(PREFERRED_SPECIES, x, idx);
                        FloatVector ay = FloatVector.fromArray(PREFERRED_SPECIES, y, idx);
                        acc1 = acc1.add(ax.mul(ay));
                    }
                    break;
                }
            }
        }
        
        float result = acc1.add(acc2).reduceLanes(VectorOperators.ADD);
        
        // 处理剩余部分
        for (; i < end; i++) {
            result += x[i] * y[i];
        }
        
        return result;
    }
    
    /**
     * 简单的部分点积计算
     */
    private float computePartialDotProductSimple(float[] x, float[] y, int start, int end) {
        float result = 0.0f;
        for (int i = start; i < end; i++) {
            result += x[i] * y[i];
        }
        return result;
    }

    // 针对特定形状优化的外积
    @Override
    public float[][] outer(float[] a, float[] b) {
        int m = a.length;
        int n = b.length;
        float[][] result = new float[m][n];

        // 根据向量长度选择最优策略
        if (m >= n) {
            // a 较长，按行处理
            return outerProductByRow(a, b, result);
        } else {
            // b 较长，按列处理
            return outerProductByColumn(a, b, result);
        }
    }

    private float[][] outerProductByRow(float[] a, float[] b, float[][] result) {
        int m = a.length;
        int n = b.length;
        final VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;
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

    private float[][] outerProductByColumn(float[] a, float[] b, float[][] result) {
        int m = a.length;
        int n = b.length;
        final VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;
        int vectorLength = species.length();

        // Loop unrolling factor for better performance
        final int UNROLL_FACTOR = 4;

        for (int j = 0; j < n; j++) {
            FloatVector bBroadcast = FloatVector.broadcast(species, b[j]);
            int i = 0;

            // Process multiple elements at once with loop unrolling
            int unrolledEnd = (m / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR);
            for (; i < unrolledEnd; i += vectorLength * UNROLL_FACTOR) {
                // Unroll 4 iterations to improve instruction-level parallelism
                FloatVector aVec0 = FloatVector.fromArray(species, a, i);
                FloatVector aVec1 = FloatVector.fromArray(species, a, i + vectorLength);
                FloatVector aVec2 = FloatVector.fromArray(species, a, i + 2 * vectorLength);
                FloatVector aVec3 = FloatVector.fromArray(species, a, i + 3 * vectorLength);
                
                FloatVector product0 = aVec0.mul(bBroadcast);
                FloatVector product1 = aVec1.mul(bBroadcast);
                FloatVector product2 = aVec2.mul(bBroadcast);
                FloatVector product3 = aVec3.mul(bBroadcast);

                // 存储到结果矩阵的列中
                for (int k = 0; k < vectorLength; k++) {
                    result[i + k][j] = product0.lane(k);
                    result[i + k + vectorLength][j] = product1.lane(k);
                    result[i + k + 2 * vectorLength][j] = product2.lane(k);
                    result[i + k + 3 * vectorLength][j] = product3.lane(k);
                }
            }

            // Process vectorized elements
            int vectorizedEnd = (m / vectorLength) * vectorLength;
            for (; i < vectorizedEnd; i += vectorLength) {
                FloatVector aVec = FloatVector.fromArray(species, a, i);
                FloatVector product = aVec.mul(bBroadcast);

                // 存储到结果矩阵的列中
                for (int k = 0; k < vectorLength; k++) {
                    result[i + k][j] = product.lane(k);
                }
            }

            for (; i < m; i++) {
                result[i][j] = a[i] * b[j];
            }
        }
        return result;
    }

    @Override
    public float[] sign(float[] array) {
        float[] result = new float[array.length];
        final VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;
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
            VectorMask<Float> gtZero1 = vec1.compare(VectorOperators.GT, 0.0f);
            VectorMask<Float> ltZero1 = vec1.lt(0.0f);
            VectorMask<Float> gtZero2 = vec2.compare(VectorOperators.GT, 0.0f);
            VectorMask<Float> ltZero2 = vec2.lt(0.0f);
            VectorMask<Float> gtZero3 = vec3.compare(VectorOperators.GT, 0.0f);
            VectorMask<Float> ltZero3 = vec3.lt(0.0f);

            FloatVector signVec0 = zero.blend(one, gtZero0).blend(minusOne, ltZero0);
            FloatVector signVec1 = zero.blend(one, gtZero1).blend(minusOne, ltZero1);
            FloatVector signVec2 = zero.blend(one, gtZero2).blend(minusOne, ltZero2);
            FloatVector signVec3 = zero.blend(one, gtZero3).blend(minusOne, ltZero3);

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

            FloatVector signVec = zero.blend(one, gtZero).blend(minusOne, ltZero);
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
        if (array.length <= stride) {
            return new float[0];
        }
        final VectorSpecies<Float> species = FloatVector.SPECIES_PREFERRED.length() >= 4 ? 
            FloatVector.SPECIES_PREFERRED.loopBound(4) >= 4 ? FloatVector.SPECIES_PREFERRED : 
            FloatVector.SPECIES_256 : FloatVector.SPECIES_128;
        int vectorLength = species.length();
        int resultLength = array.length - stride;
        float[] result = new float[resultLength];
        int i = 0;

        // Loop unrolling factor for better performance
        final int UNROLL_FACTOR = 4;
        
        // Process multiple elements at once with loop unrolling
        int unrolledEnd = (resultLength / (vectorLength * UNROLL_FACTOR)) * (vectorLength * UNROLL_FACTOR);
        for (; i < unrolledEnd; i += vectorLength * UNROLL_FACTOR) {
            // Unroll 4 iterations to improve instruction-level parallelism
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

        // Process vectorized elements
        int vectorizedEnd = (resultLength / vectorLength) * vectorLength;
        for (; i < vectorizedEnd; i += vectorLength) {
            FloatVector current = FloatVector.fromArray(species, array, i);
            FloatVector future = FloatVector.fromArray(species, array, i + stride);

            FloatVector diff = future.sub(current);
            diff.intoArray(result, i);
        }

        // Handle remaining elements
        for (; i < resultLength; i++) {
            result[i] = array[i + stride] - array[i];
        }

        return result;
    }
}
