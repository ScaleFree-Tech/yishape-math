package com.yishape.lab.math.compute;

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
import static com.yishape.lab.math.compute.ops.ReduceOperation.PROD;
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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

/**
 * SISD运算 Java Vector API计算的回退类，使用普通的JAVA循环实现运算
 *
 * @author lteb2
 */
public class SISDFloatComputer implements IFloatVectorComputer,Serializable {

    /** Minimum array length to use parallel execution for element-wise ops. */
    private static final int PARALLEL_THRESHOLD = 50_000;

    /** Task that processes a range [start, end). */
    @FunctionalInterface
    private interface RangeTask { void accept(int start, int end); }

    /** Execute a task in parallel over array chunks if length exceeds threshold. */
    private static void parallelForEach(int length, RangeTask task) {
        ForkJoinPool pool = RereDoubleMatrix.getThreadPool();
        if (length < PARALLEL_THRESHOLD || pool == null || pool.isShutdown()) {
            task.accept(0, length);
            return;
        }
        int nThreads = pool.getParallelism();
        if (nThreads <= 1) { task.accept(0, length); return; }
        int chunk = (length + nThreads - 1) / nThreads;
        List<Future<?>> futures = new ArrayList<>(nThreads);
        for (int t = 0; t < nThreads; t++) {
            int start = t * chunk;
            int end = Math.min(start + chunk, length);
            if (start >= end) break;
            final int s = start, e = end;
            futures.add(pool.submit(() -> task.accept(s, e)));
        }
        for (Future<?> f : futures) {
            try { f.get(); } catch (Exception e) { throw new RuntimeException(e); }
        }
    }

    @Override
    public float[] binaryOperate(float[] x1, float[] x2, BinaryOperation operation) {
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("The length of the vector must be the same. First length: "+x1.length+", second length: "+x2.length);
        }

        int n = x1.length;
        float[] result = new float[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                switch (operation) {
                    case ADD -> result[i] = x1[i] + x2[i];
                    case SUBTRACT -> result[i] = x1[i] - x2[i];
                    case MULTIPLY -> result[i] = x1[i] * x2[i];
                    case DIVIDE -> result[i] = x1[i] / x2[i];
                    case REMAINDER -> result[i] = x1[i] % x2[i];
                    default -> throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        });
        return result;
    }

    @Override
    public float[] binaryOperate(float[] x1, float x2, BinaryOperation operation) {
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        int n = x1.length;
        float[] result = new float[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                switch (operation) {
                    case ADD -> result[i] = x1[i] + x2;
                    case SUBTRACT -> result[i] = x1[i] - x2;
                    case MULTIPLY -> result[i] = x1[i] * x2;
                    case DIVIDE -> result[i] = x1[i] / x2;
                    case REMAINDER -> result[i] = x1[i] % x2;
                    default -> throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        });
        return result;
    }

    @Override
    public float[][] binaryOperate(float[][] x1, float[][] x2, BinaryOperation operation) {
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
        int rows = x1.length;
        int cols = x1[0].length;
        float[][] result = new float[rows][cols];

        // Parallelize over rows (each row processed sequentially to avoid nested parallelism)
        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                float[] r1 = x1[row], r2 = x2[row], out = result[row];
                for (int j = 0; j < cols; j++) {
                    switch (operation) {
                        case ADD -> out[j] = r1[j] + r2[j];
                        case SUBTRACT -> out[j] = r1[j] - r2[j];
                        case MULTIPLY -> out[j] = r1[j] * r2[j];
                        case DIVIDE -> out[j] = r1[j] / r2[j];
                        case REMAINDER -> out[j] = r1[j] % r2[j];
                        default -> throw new IllegalArgumentException("不支持的操作: " + operation);
                    }
                }
            }
        });
        return result;
    }

    @Override
    public float[][] binaryOperate(float[][] x1, float x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length == 0) return new float[0][0];

        int rows = x1.length;
        int cols = x1[0].length;
        float[][] result = new float[rows][cols];

        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                float[] r1 = x1[row], out = result[row];
                for (int j = 0; j < cols; j++) {
                    switch (operation) {
                        case ADD -> out[j] = r1[j] + x2;
                        case SUBTRACT -> out[j] = r1[j] - x2;
                        case MULTIPLY -> out[j] = r1[j] * x2;
                        case DIVIDE -> out[j] = r1[j] / x2;
                        case REMAINDER -> out[j] = r1[j] % x2;
                        default -> throw new IllegalArgumentException("不支持的操作: " + operation);
                    }
                }
            }
        });

        return result;
    }

    @Override
    public float[] universalOperate(float[] x, UniversalOperation operation, float additionalParam) {
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        int n = x.length;
        float[] result = new float[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                switch (operation) {
                    case EXP -> result[i] = (float) Math.exp(x[i]);
                    case LOG -> result[i] = (float) Math.log(x[i]);
                    case LOG10 -> result[i] = (float) Math.log10(x[i]);
                    case SIN -> result[i] = (float) Math.sin(x[i]);
                    case COS -> result[i] = (float) Math.cos(x[i]);
                    case TAN -> result[i] = (float) Math.tan(x[i]);
                    case ASIN -> result[i] = (float) Math.asin(x[i]);
                    case ACOS -> result[i] = (float) Math.acos(x[i]);
                    case ATAN -> result[i] = (float) Math.atan(x[i]);
                    case SQRT -> result[i] = (float) Math.sqrt(x[i]);
                    case ABS -> result[i] = Math.abs(x[i]);
                    case POW -> result[i] = (float) Math.pow(x[i], additionalParam);
                    case CBRT -> result[i] = (float) Math.cbrt(x[i]);
                    case COSH -> result[i] = (float) Math.cosh(x[i]);
                    case SINH -> result[i] = (float) Math.sinh(x[i]);
                    case TANH -> result[i] = (float) Math.tanh(x[i]);
                    case EXPM1 -> result[i] = (float) Math.expm1(x[i]);
                    case LOG1P -> result[i] = (float) Math.log1p(x[i]);
                    case RELU -> result[i] = Math.max(0.0f, x[i]);
                    case SIGMOID -> result[i] = (float) (1.0 / (1.0 + Math.exp(-x[i])));
                    case GELU -> {
                        float inner = 0.7978845608028654f * (x[i] + 0.044715f * x[i] * x[i] * x[i]);
                        result[i] = 0.5f * x[i] * (1f + (float) Math.tanh(inner));
                    }
                    default -> throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        });
        return result;
    }

    @Override
    public float[][] universalOperate(float[][] x, UniversalOperation operation, float additionalParam) {
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x.length == 0) return new float[0][0];

        int rows = x.length;
        int cols = x[0].length;
        float[][] result = new float[rows][cols];

        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                result[row] = universalOperate(x[row], operation, additionalParam);
            }
        });
        return result;
    }

    @Override
    public float reduceOperate(float[] x, ReduceOperation operation) {
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

        switch (operation) {
            case SUM: {
                float sum = 0.0f;
                for (int i = 0; i < x.length; i++) {
                    sum += x[i];
                }
                return sum;
            }

            case MEAN: {
                // 计算总和并除以长度
                float sum = 0.0f;
                for (int i = 0; i < x.length; i++) {
                    sum += x[i];
                }
                return sum / x.length;
            }

            case MIN: {
                float min = Float.POSITIVE_INFINITY;
                for (int i = 0; i < x.length; i++) {
                    if (x[i] < min) {
                        min = x[i];
                    }
                }
                return min;
            }

            case MAX: {
                float max = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < x.length; i++) {
                    if (x[i] > max) {
                        max = x[i];
                    }
                }
                return max;
            }

            case VARIANCE: {
                // Kahan compensated summation for mean
                float sum = 0.0f, c = 0.0f;
                for (int i = 0; i < x.length; i++) {
                    float y = x[i] - c;
                    float t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                float mean = sum / x.length;

                // Kahan compensated summation for variance
                float varianceSum = 0.0f;
                float cv = 0.0f;
                for (int i = 0; i < x.length; i++) {
                    float diff = x[i] - mean;
                    float y = diff * diff - cv;
                    float t = varianceSum + y;
                    cv = (t - varianceSum) - y;
                    varianceSum = t;
                }
                return varianceSum / x.length;
            }

            case STANDARD_DEVIATION: {
                return (float) Math.sqrt(reduceOperate(x, ReduceOperation.VARIANCE));
            }

            case PROD: {
                float p = 1.0f;
                for (int i = 0; i < x.length; i++) {
                    p *= x[i];
                }
                return p;
            }

            default:
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    @Override
    public float reduceOperate(float[][] x, ReduceOperation operation) {
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

        switch (operation) {
            case SUM: {
                float sumResult = 0.0f;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        sumResult += x[i][j];
                    }
                }
                return sumResult;
            }

            case MEAN: {
                float totalSum = 0.0f;
                int totalElements = 0;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        totalSum += x[i][j];
                        totalElements++;
                    }
                }
                return totalSum / totalElements;
            }

            case MIN: {
                float minResult = Float.POSITIVE_INFINITY;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        if (x[i][j] < minResult) {
                            minResult = x[i][j];
                        }
                    }
                }
                return minResult;
            }

            case MAX: {
                float maxResult = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        if (x[i][j] > maxResult) {
                            maxResult = x[i][j];
                        }
                    }
                }
                return maxResult;
            }

            case VARIANCE: {
                // Kahan compensated summation for mean
                float totalSum = 0.0f, c = 0.0f;
                int totalElements = 0;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        float y = x[i][j] - c;
                        float t = totalSum + y;
                        c = (t - totalSum) - y;
                        totalSum = t;
                        totalElements++;
                    }
                }
                float mean = totalSum / totalElements;

                // Kahan compensated summation for variance
                float varianceSum = 0.0f;
                float cv = 0.0f;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        float diff = x[i][j] - mean;
                        float y = diff * diff - cv;
                        float t = varianceSum + y;
                        cv = (t - varianceSum) - y;
                        varianceSum = t;
                    }
                }
                return varianceSum / totalElements;
            }

            case STANDARD_DEVIATION: {
                return (float) Math.sqrt(reduceOperate(x, ReduceOperation.VARIANCE));
            }

            case PROD: {
                float productResult = 1.0f;
                for (float[] row : x) {
                    for (int j = 0; j < row.length; j++) {
                        productResult *= row[j];
                    }
                }
                return productResult;
            }

            default:
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    @Override
    public float binaryReduceOperate(float[] x1, float[] x2, BinaryReduceOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        switch (operation) {
            case DOT: {
                // 计算点积：两个向量对应元素相乘后求和
                float sum = 0.0f;
                for (int i = 0; i < x1.length; i++) {
                    sum += x1[i] * x2[i];
                }
                return sum;
            }

            case L2_NORM: {
                // 计算两个向量差的L2范数
                float sum = 0.0f;
                for (int i = 0; i < x1.length; i++) {
                    float diff = x1[i] - x2[i];
                    sum += diff * diff;
                }
                return (float)Math.sqrt(sum);
            }

            case L1_NORM: {
                // 计算两个向量差的L1范数
                float sum = 0.0f;
                for (int i = 0; i < x1.length; i++) {
                    sum += Math.abs(x1[i] - x2[i]);
                }
                return sum;
            }

            default:
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    @Override
    public float binaryReduceOperate(float[][] x1, float[][] x2, BinaryReduceOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length != x2.length || (x1.length > 0 && x1[0].length != x2[0].length)) {
            throw new IllegalArgumentException("矩阵维度必须相同");
        }

        switch (operation) {
            case DOT: {
                // 计算矩阵点积：两个矩阵对应元素相乘后求和
                float sum = 0.0f;
                for (int i = 0; i < x1.length; i++) {
                    for (int j = 0; j < x1[i].length; j++) {
                        sum += x1[i][j] * x2[i][j];
                    }
                }
                return sum;
            }

            case L2_NORM: {
                // 计算两个矩阵差的L2范数（弗罗贝尼乌斯范数）
                float sum = 0.0f;
                for (int i = 0; i < x1.length; i++) {
                    for (int j = 0; j < x1[i].length; j++) {
                        float diff = x1[i][j] - x2[i][j];
                        sum += diff * diff;
                    }
                }
                return (float)Math.sqrt(sum);
            }

            case L1_NORM: {
                // 计算两个矩阵差的L1范数
                float sum = 0.0f;
                for (int i = 0; i < x1.length; i++) {
                    for (int j = 0; j < x1[i].length; j++) {
                        sum += Math.abs(x1[i][j] - x2[i][j]);
                    }
                }
                return sum;
            }

            default:
                throw new IllegalArgumentException("不支持的操作: " + operation);
        }
    }

    @Override
    public float[] elementWiseMin(float[] x1, float[] x2) {
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        int n = x1.length;
        float[] result = new float[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                result[i] = Math.min(x1[i], x2[i]);
            }
        });
        return result;
    }

    @Override
    public float[][] elementWiseMin(float[][] x1, float[][] x2) {
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

        int rows = x1.length;
        float[][] result = new float[rows][x1[0].length];
        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                float[] r1 = x1[row], r2 = x2[row], out = result[row];
                for (int j = 0; j < r1.length; j++) {
                    out[j] = Math.min(r1[j], r2[j]);
                }
            }
        });
        return result;
    }

    @Override
    public float[] elementWiseMax(float[] x1, float[] x2) {
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        int n = x1.length;
        float[] result = new float[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                result[i] = Math.max(x1[i], x2[i]);
            }
        });
        return result;
    }

    @Override
    public float[][] elementWiseMax(float[][] x1, float[][] x2) {
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

        int rows = x1.length;
        float[][] result = new float[rows][x1[0].length];
        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                float[] r1 = x1[row], r2 = x2[row], out = result[row];
                for (int j = 0; j < r1.length; j++) {
                    out[j] = Math.max(r1[j], r2[j]);
                }
            }
        });
        return result;
    }

    @Override
    public float[] negate(float[] x) {
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        int n = x.length;
        float[] result = new float[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                result[i] = -x[i];
            }
        });
        return result;
    }

    @Override
    public float[][] negate(float[][] x) {
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x.length == 0) return new float[0][0];

        int rows = x.length;
        float[][] result = new float[rows][x[0].length];
        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                float[] xr = x[row], out = result[row];
                for (int j = 0; j < xr.length; j++) {
                    out[j] = -xr[j];
                }
            }
        });
        return result;
    }

    @Override
    public boolean[] logicalCompare(float[] x1, float[] x2, LogicalCompare operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用普通Java循环进行逻辑运算
        for (int i = 0; i < x1.length; i++) {
            switch (operation) {
                case EQUALS:
                    result[i] = x1[i] == x2[i];
                    break;
                case NOT_EQUALS:
                    result[i] = x1[i] != x2[i];
                    break;
                case LESS_THAN:
                    result[i] = x1[i] < x2[i];
                    break;
                case LESS_THAN_OR_EQUALS:
                    result[i] = x1[i] <= x2[i];
                    break;
                case GREATER_THAN:
                    result[i] = x1[i] > x2[i];
                    break;
                case GREATER_THAN_OR_EQUALS:
                    result[i] = x1[i] >= x2[i];
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }

        return result;
    }

    @Override
    public boolean[] logicalOperate(float[] x1, float[] x2, LogicalOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用普通Java循环进行逻辑运算
        for (int i = 0; i < x1.length; i++) {
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
    public boolean[] logicalOperate(float[] x1, LogicalOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用普通Java循环进行逻辑运算
        for (int i = 0; i < x1.length; i++) {
            // 将float值转换为布尔值：非零为true，零为false
            boolean boolA = (x1[i] != 0.0);

            switch (operation) {
                case NOT:
                    result[i] = !boolA;
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation + "。logicalOperate(float[], LogicalOperation)方法仅支持NOT操作。对于AND、OR、XOR操作，请使用双参数版本。");
            }
        }

        return result;
    }

    @Override
    public float[][] transpose(float[][] matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        if (matrix.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }

        int m = matrix.length;
        int n = matrix[0].length;
        float[][] result = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    @Override
    public float[][] transpose(float[] rowVector) {
        // 参数验证
        if (rowVector == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        int length = rowVector.length;
        float[][] result = new float[length][1];
        
        // 将行向量转置为列向量
        for (int i = 0; i < length; i++) {
            result[i][0] = rowVector[i];
        }
        
        return result;
    }

    @Override
    public float[][] mmul(float[][] data, float[][] otherData) {
        int m = data.length;          // 当前矩阵行数
        if (m == 0) return new float[0][];
        int otherRows = otherData.length;
        if (otherRows == 0) return new float[0][];
        int n = data[0].length;       // 当前矩阵列数
        int p = otherData[0].length;  // 另一个矩阵列数
        if (n != otherRows) {
            throw new IllegalArgumentException(
                "Matrix dimension mismatch for mmul: A is " + m + "x" + n + ", B is " + otherRows + "x" + p);
        }

        // 计算复杂度
        long complexity = (long) m * n * p;

        if (complexity < 1000) {
            // 极小矩阵：直接计算
            return naiveMultiply(data, otherData, m, n, p);
        } else if (complexity < 64000) {
            // 小矩阵：循环展开优化
            return unrolledMultiply(data, otherData, m, n, p);
        } else if (complexity < 8000000) {
            // 中等矩阵：分块算法
            return blockMultiply(data, otherData, m, n, p);
        } else {
            // 大矩阵：并行分块算法
            return parallelBlockMultiply(data, otherData, m, n, p);
        }

    }

    /**
     * 朴素矩阵乘法 / Naive matrix multiplication
     * i-k-j 顺序：缓存友好，otherData[k] 整行连续访问
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
     * i-k-j 顺序 + 4路 j 展开：缓存友好，otherData[k] 整行连续访问
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
     * 分块矩阵乘法 / Block matrix multiplication
     */
    private float[][] blockMultiply(float[][] data, float[][] otherData, int m, int n, int p) {
        float[][] result = new float[m][p];

        // 动态计算最优块大小
        int blockSize = calculateOptimalBlockSize(m, n, p);

        for (int ii = 0; ii < m; ii += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                for (int jj = 0; jj < p; jj += blockSize) {

                    int iEnd = Math.min(ii + blockSize, m);
                    int kEnd = Math.min(kk + blockSize, n);
                    int jEnd = Math.min(jj + blockSize, p);

                    // 块内 i-k-j 顺序
                    for (int i = ii; i < iEnd; i++) {
                        float[] thisRow = data[i];
                        float[] resultRow = result[i];

                        for (int k = kk; k < kEnd; k++) {
                            float aik = thisRow[k];
                            float[] otherRow = otherData[k];

                            int j = jj;
                            for (; j < jEnd - 3; j += 4) {
                                resultRow[j]     += aik * otherRow[j];
                                resultRow[j + 1] += aik * otherRow[j + 1];
                                resultRow[j + 2] += aik * otherRow[j + 2];
                                resultRow[j + 3] += aik * otherRow[j + 3];
                            }
                            for (; j < jEnd; j++) {
                                resultRow[j] += aik * otherRow[j];
                            }
                        }
                    }
                }
            }
        }

        return result;
    }

    /**
     * 并行分块矩阵乘法 / Parallel block matrix multiplication
     */
    private float[][] parallelBlockMultiply(float[][] data, float[][] otherData, int m, int n, int p) {
        float[][] result = new float[m][p];

        // 计算最优块大小和线程数
        int blockSize = calculateOptimalBlockSize(m, n, p);
        ForkJoinPool pool = RereDoubleMatrix.getThreadPool();

        if (pool == null || pool.isShutdown()) {
            return blockMultiply(data, otherData, m, n, p);
        }

        int numThreads = pool.getParallelism();
        if (numThreads <= 1 || m < blockSize * 2) {
            return blockMultiply(data, otherData, m, n, p);
        }

        List<Future<Void>> futures = new ArrayList<>();

        // 按行分块并行计算
        int rowsPerThread = (m + numThreads - 1) / numThreads;

        for (int t = 0; t < numThreads; t++) {
            final int startRow = t * rowsPerThread;
            final int endRow = Math.min(startRow + rowsPerThread, m);

            if (startRow >= endRow) {
                break;
            }

            Future<Void> future = pool.submit(() -> {
                // 每个线程处理指定行范围的计算
                for (int ii = startRow; ii < endRow; ii += blockSize) {
                    for (int jj = 0; jj < p; jj += blockSize) {
                        for (int kk = 0; kk < n; kk += blockSize) {

                            int iEnd = Math.min(ii + blockSize, endRow);
                            int jEnd = Math.min(jj + blockSize, p);
                            int kEnd = Math.min(kk + blockSize, n);

                            // 块内计算
                            for (int i = ii; i < iEnd; i++) {
                                float[] thisRow = data[i];
                                float[] resultRow = result[i];

                                for (int j = jj; j < jEnd; j++) {
                                    float sum = resultRow[j];

                                    // 内层循环展开
                                    int k = kk;
                                    for (; k < kEnd - 3; k += 4) {
                                        sum += thisRow[k] * otherData[k][j]
                                                + thisRow[k + 1] * otherData[k + 1][j]
                                                + thisRow[k + 2] * otherData[k + 2][j]
                                                + thisRow[k + 3] * otherData[k + 3][j];
                                    }

                                    for (; k < kEnd; k++) {
                                        sum += thisRow[k] * otherData[k][j];
                                    }

                                    resultRow[j] = sum;
                                }
                            }
                        }
                    }
                }
                return null;
            });

            futures.add(future);
        }

        // 等待所有线程完成
        try {
            for (Future<Void> future : futures) {
                future.get();
            }
        } catch (Exception e) {
            return blockMultiply(data, otherData, m, n, p);
        }

        return result;
    }

    /**
     * 计算最优块大小 / Calculate optimal block size
     *
     * @param m
     * @param n
     * @param p
     * @return
     */
    public static int calculateOptimalBlockSize(int m, int n, int p) {
        // 基于L1缓存大小估算最优块大小
        // 假设L1缓存32KB，Float占4字节
        int l1CacheSize = 32 * 1024 / 4; // 8192个Float

        // 考虑三个矩阵块：A_block, B_block, C_block
        // 每个块大约占用 blockSize^2 个元素
        int maxBlockSize = (int) Math.sqrt(l1CacheSize / 3);

        // 限制在合理范围内
        maxBlockSize = Math.max(16, Math.min(maxBlockSize, 128));

        // 选择能整除矩阵维度的块大小
        for (int blockSize = maxBlockSize; blockSize >= 16; blockSize--) {
            if (m % blockSize == 0 || n % blockSize == 0 || p % blockSize == 0) {
                return blockSize;
            }
        }

        return 32; // 默认块大小
    }

    @Override
    public float[][] outer(float[] a, float[] b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("向量不能为null / Vector cannot be null");
        }

        int rows = a.length;
        int cols = b.length;
        float[][] result = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i] * b[j];  // Fixed: was using a[j] instead of b[j]
            }
        }
        return result;
    }

    @Override
    public float[] sign(float[] data) {
        float[] result = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            if (data[i] > 0) {
                result[i] = 1.0f;
            } else if (data[i] < 0) {
                result[i] = -1.0f;
            } else {
                result[i] = 0.0f;
            }
        }
        return result;
    }

    @Override
    public float[][] sign(float[][] array) {
        // 参数验证
        if (array == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // 创建结果矩阵
        float[][] result = new float[array.length][];
        
        // 对每一行计算符号函数
        for (int i = 0; i < array.length; i++) {
            result[i] = sign(array[i]);
        }

        return result;
    }

    @Override
    public float[] diff(float[] data, int n) {
        // 参数验证
        if (data == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        if (n <= 0) {
            throw new IllegalArgumentException("步长必须为正整数");
        }

        // 如果步长大于等于数组长度，返回空数组
        if (n >= data.length) {
            return new float[0];
        }

        // 创建结果数组，长度为 data.length - n
        float[] result = new float[data.length - n];

        // 计算差分：result[i] = data[i + n] - data[i]
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i + n] - data[i];
        }

        return result;
    }

    @Override
    public boolean[][] logicalCompare(float[][] x1, float[][] x2, LogicalCompare operation) {
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

        if (x1.length == 0) return new boolean[0][0];

        // 创建结果矩阵
        boolean[][] result = new boolean[x1.length][x1[0].length];

        // 使用普通Java循环进行逻辑比较运算
        for (int i = 0; i < x1.length; i++) {
            for (int j = 0; j < x1[i].length; j++) {
                switch (operation) {
                    case EQUALS:
                        result[i][j] = x1[i][j] == x2[i][j];
                        break;
                    case NOT_EQUALS:
                        result[i][j] = x1[i][j] != x2[i][j];
                        break;
                    case LESS_THAN:
                        result[i][j] = x1[i][j] < x2[i][j];
                        break;
                    case LESS_THAN_OR_EQUALS:
                        result[i][j] = x1[i][j] <= x2[i][j];
                        break;
                    case GREATER_THAN:
                        result[i][j] = x1[i][j] > x2[i][j];
                        break;
                    case GREATER_THAN_OR_EQUALS:
                        result[i][j] = x1[i][j] >= x2[i][j];
                        break;
                    default:
                        throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        }

        return result;
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, LogicalOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length == 0) return new boolean[0][0];

        // 创建结果矩阵
        boolean[][] result = new boolean[x1.length][x1[0].length];

        // 使用普通Java循环进行逻辑运算
        for (int i = 0; i < x1.length; i++) {
            for (int j = 0; j < x1[i].length; j++) {
                // 将float值转换为布尔值：非零为true，零为false
                boolean boolA = (x1[i][j] != 0.0);

                switch (operation) {
                    case NOT:
                        result[i][j] = !boolA;
                        break;
                    default:
                        throw new IllegalArgumentException("不支持的操作: " + operation + "。logicalOperate方法仅支持NOT操作。对于AND、OR、XOR操作，请使用双参数版本。");
                }
            }
        }

        return result;
    }

    @Override
    public boolean[][] logicalOperate(float[][] x1, float[][] x2, LogicalOperation operation) {
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

        if (x1.length == 0) return new boolean[0][0];

        // 创建结果矩阵
        boolean[][] result = new boolean[x1.length][x1[0].length];

        // 使用普通Java循环进行逻辑运算
        for (int i = 0; i < x1.length; i++) {
            for (int j = 0; j < x1[i].length; j++) {
                // 将float值转换为布尔值：非零为true，零为false
                boolean boolA = (x1[i][j] != 0.0);
                boolean boolB = (x2[i][j] != 0.0);

                switch (operation) {
                    case AND:
                        result[i][j] = boolA && boolB;
                        break;
                    case OR:
                        result[i][j] = boolA || boolB;
                        break;
                    case XOR:
                        result[i][j] = boolA ^ boolB;
                        break;
                    default:
                        throw new IllegalArgumentException("不支持的操作: " + operation + "。logicalOperate方法仅支持AND、OR、XOR操作。对于NOT操作，请使用单参数版本。");
                }
            }
        }

        return result;
    }
    
    

}
