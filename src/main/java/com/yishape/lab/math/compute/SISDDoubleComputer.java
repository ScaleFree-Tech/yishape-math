package com.yishape.lab.math.compute;

import static com.yishape.lab.math.compute.gpu.GpuReduce.MIN;
import com.yishape.lab.math.compute.hpc.HpcGemm;
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

import com.yishape.lab.util.YishapeLogger;

import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.Future;

/**
 * SISD运算 Java Vector API计算的回退类，使用普通的JAVA循环实现运算
 *
 * @author lteb2
 */
public class SISDDoubleComputer implements IDoubleVectorComputer,Serializable {

    private static final YishapeLogger log = YishapeLogger.getLogger(SISDDoubleComputer.class);

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

// Strassen矩阵乘法常量
    private static final int STRASSEN_THRESHOLD = 2048;
    private static final long STRASSEN_COMPLEXITY_THRESHOLD = 8_589_934_592L; // 2048³
    private static final boolean USE_STRASSEN = !"false".equals(
        System.getProperty("yishape.strassen.enabled", "true"));

    /** Minimum array length to use parallel execution for element-wise ops. */
    private static final int PARALLEL_THRESHOLD = 50_000;

    @Override
    public double[] binaryOperate(double[] x1, double[] x2, BinaryOperation operation) {
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("The length of the vector must be the same. First length: "+x1.length+", second length: "+x2.length);
        }

        int n = x1.length;
        double[] result = new double[n];
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
    public double[] binaryOperate(double[] x1, double x2, BinaryOperation operation) {
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        int n = x1.length;
        double[] result = new double[n];
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
        int rows = x1.length;
        int cols = x1[0].length;
        double[][] result = new double[rows][cols];

        // Parallelize over rows (each row processed sequentially to avoid nested parallelism)
        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                double[] r1 = x1[row], r2 = x2[row], out = result[row];
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
    public double[][] binaryOperate(double[][] x1, double x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x1.length == 0) return new double[0][0];

        int rows = x1.length;
        int cols = x1[0].length;
        double[][] result = new double[rows][cols];

        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                double[] r1 = x1[row], out = result[row];
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
    public double[] universalOperate(double[] x, UniversalOperation operation, double additionalParam) {
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        int n = x.length;
        double[] result = new double[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                switch (operation) {
                    case EXP -> result[i] = Math.exp(x[i]);
                    case LOG -> result[i] = Math.log(x[i]);
                    case LOG10 -> result[i] = Math.log10(x[i]);
                    case SIN -> result[i] = Math.sin(x[i]);
                    case COS -> result[i] = Math.cos(x[i]);
                    case TAN -> result[i] = Math.tan(x[i]);
                    case ASIN -> result[i] = Math.asin(x[i]);
                    case ACOS -> result[i] = Math.acos(x[i]);
                    case ATAN -> result[i] = Math.atan(x[i]);
                    case SQRT -> result[i] = Math.sqrt(x[i]);
                    case ABS -> result[i] = Math.abs(x[i]);
                    case POW -> result[i] = Math.pow(x[i], additionalParam);
                    case CBRT -> result[i] = Math.cbrt(x[i]);
                    case COSH -> result[i] = Math.cosh(x[i]);
                    case SINH -> result[i] = Math.sinh(x[i]);
                    case TANH -> result[i] = Math.tanh(x[i]);
                    case EXPM1 -> result[i] = Math.expm1(x[i]);
                    case LOG1P -> result[i] = Math.log1p(x[i]);
                    case RELU -> result[i] = Math.max(0.0, x[i]);
                    case SIGMOID -> result[i] = 1.0 / (1.0 + Math.exp(-x[i]));
                    case GELU -> {
                        double xi = x[i];
                        result[i] = 0.5 * xi * (1.0 + Math.tanh(0.7978845608028654 * (xi + 0.044715 * xi * xi * xi)));
                    }
                    default -> throw new IllegalArgumentException("不支持的操作: " + operation);
                }
            }
        });
        return result;
    }

    @Override
    public double[][] universalOperate(double[][] x, UniversalOperation operation, double additionalParam) {
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x.length == 0) return new double[0][0];

        int rows = x.length;
        int cols = x[0].length;
        double[][] result = new double[rows][cols];

        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                double[] xr = x[row], out = result[row];
                for (int j = 0; j < cols; j++) {
                    switch (operation) {
                        case EXP -> out[j] = Math.exp(xr[j]);
                        case LOG -> out[j] = Math.log(xr[j]);
                        case LOG10 -> out[j] = Math.log10(xr[j]);
                        case SIN -> out[j] = Math.sin(xr[j]);
                        case COS -> out[j] = Math.cos(xr[j]);
                        case TAN -> out[j] = Math.tan(xr[j]);
                        case ASIN -> out[j] = Math.asin(xr[j]);
                        case ACOS -> out[j] = Math.acos(xr[j]);
                        case ATAN -> out[j] = Math.atan(xr[j]);
                        case SQRT -> out[j] = Math.sqrt(xr[j]);
                        case ABS -> out[j] = Math.abs(xr[j]);
                        case POW -> out[j] = Math.pow(xr[j], additionalParam);
                        case CBRT -> out[j] = Math.cbrt(xr[j]);
                        case COSH -> out[j] = Math.cosh(xr[j]);
                        case SINH -> out[j] = Math.sinh(xr[j]);
                        case TANH -> out[j] = Math.tanh(xr[j]);
                        case EXPM1 -> out[j] = Math.expm1(xr[j]);
                        case LOG1P -> out[j] = Math.log1p(xr[j]);
                        case RELU -> out[j] = Math.max(0.0, xr[j]);
                        case SIGMOID -> out[j] = 1.0 / (1.0 + Math.exp(-xr[j]));
                        case GELU -> {
                            double xi = xr[j];
                            out[j] = 0.5 * xi * (1.0 + Math.tanh(0.7978845608028654 * (xi + 0.044715 * xi * xi * xi)));
                        }
                        default -> throw new IllegalArgumentException("不支持的操作: " + operation);
                    }
                }
            }
        });
        return result;
    }

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

        switch (operation) {
            case SUM: {
                double sum = 0.0;
                for (int i = 0; i < x.length; i++) {
                    sum += x[i];
                }
                return sum;
            }

            case MEAN: {
                // 计算总和并除以长度
                double sum = 0.0;
                for (int i = 0; i < x.length; i++) {
                    sum += x[i];
                }
                return sum / x.length;
            }

            case MIN: {
                double min = Double.POSITIVE_INFINITY;
                for (int i = 0; i < x.length; i++) {
                    if (x[i] < min) {
                        min = x[i];
                    }
                }
                return min;
            }

            case MAX: {
                double max = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < x.length; i++) {
                    if (x[i] > max) {
                        max = x[i];
                    }
                }
                return max;
            }

            case VARIANCE: {
                // Kahan compensated summation for mean (consistent with SIMD path)
                double sum = 0.0;
                double c = 0.0;
                for (int i = 0; i < x.length; i++) {
                    double y = x[i] - c;
                    double t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                double mean = sum / x.length;

                // Kahan compensated summation for variance
                double varianceSum = 0.0;
                double cv = 0.0;
                for (int i = 0; i < x.length; i++) {
                    double diff = x[i] - mean;
                    double y = diff * diff - cv;
                    double t = varianceSum + y;
                    cv = (t - varianceSum) - y;
                    varianceSum = t;
                }
                return varianceSum / x.length;
            }

            case STANDARD_DEVIATION: {
                return Math.sqrt(reduceOperate(x, ReduceOperation.VARIANCE));
            }

            case PROD: {
                double p = 1.0;
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

        switch (operation) {
            case SUM: {
                double sumResult = 0.0;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        sumResult += x[i][j];
                    }
                }
                return sumResult;
            }

            case MEAN: {
                double totalSum = 0.0;
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
                double minResult = Double.POSITIVE_INFINITY;
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
                double maxResult = Double.NEGATIVE_INFINITY;
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
                // 计算均值
                double totalSum = 0.0;
                int totalElements = 0;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        totalSum += x[i][j];
                        totalElements++;
                    }
                }
                double mean = totalSum / totalElements;

                // 计算方差
                double varianceSum = 0.0;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        double diff = x[i][j] - mean;
                        varianceSum += diff * diff;
                    }
                }
                return varianceSum / totalElements;
            }

            case STANDARD_DEVIATION: {
                // 计算均值
                double totalSum = 0.0;
                int totalElements = 0;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        totalSum += x[i][j];
                        totalElements++;
                    }
                }
                double mean = totalSum / totalElements;

                // 计算方差
                double varianceSum = 0.0;
                for (int i = 0; i < x.length; i++) {
                    for (int j = 0; j < x[i].length; j++) {
                        double diff = x[i][j] - mean;
                        varianceSum += diff * diff;
                    }
                }
                double variance = varianceSum / totalElements;

                return Math.sqrt(variance);
            }

            case PROD: {
                double productResult = 1.0;
                for (double[] row : x) {
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
    public double binaryReduceOperate(double[] x1, double[] x2, BinaryReduceOperation operation) {
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
                double sum = 0.0;
                for (int i = 0; i < x1.length; i++) {
                    sum += x1[i] * x2[i];
                }
                return sum;
            }

            case L2_NORM: {
                // 计算两个向量差的L2范数
                double sum = 0.0;
                for (int i = 0; i < x1.length; i++) {
                    double diff = x1[i] - x2[i];
                    sum += diff * diff;
                }
                return Math.sqrt(sum);
            }

            case L1_NORM: {
                // 计算两个向量差的L1范数
                double sum = 0.0;
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
    public double binaryReduceOperate(double[][] x1, double[][] x2, BinaryReduceOperation operation) {
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
                double sum = 0.0;
                for (int i = 0; i < x1.length; i++) {
                    for (int j = 0; j < x1[i].length; j++) {
                        sum += x1[i][j] * x2[i][j];
                    }
                }
                return sum;
            }

            case L2_NORM: {
                // 计算两个矩阵差的L2范数（弗罗贝尼乌斯范数）
                double sum = 0.0;
                for (int i = 0; i < x1.length; i++) {
                    for (int j = 0; j < x1[i].length; j++) {
                        double diff = x1[i][j] - x2[i][j];
                        sum += diff * diff;
                    }
                }
                return Math.sqrt(sum);
            }

            case L1_NORM: {
                // 计算两个矩阵差的L1范数
                double sum = 0.0;
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
    public double[] elementWiseMin(double[] x1, double[] x2) {
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        int n = x1.length;
        double[] result = new double[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                result[i] = Math.min(x1[i], x2[i]);
            }
        });
        return result;
    }

    @Override
    public double[][] elementWiseMin(double[][] x1, double[][] x2) {
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

        int rows = x1.length;
        double[][] result = new double[rows][x1[0].length];
        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                double[] r1 = x1[row], r2 = x2[row], out = result[row];
                for (int j = 0; j < r1.length; j++) {
                    out[j] = Math.min(r1[j], r2[j]);
                }
            }
        });
        return result;
    }

    @Override
    public double[] elementWiseMax(double[] x1, double[] x2) {
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        int n = x1.length;
        double[] result = new double[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                result[i] = Math.max(x1[i], x2[i]);
            }
        });
        return result;
    }

    @Override
    public double[][] elementWiseMax(double[][] x1, double[][] x2) {
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

        int rows = x1.length;
        double[][] result = new double[rows][x1[0].length];
        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                double[] r1 = x1[row], r2 = x2[row], out = result[row];
                for (int j = 0; j < r1.length; j++) {
                    out[j] = Math.max(r1[j], r2[j]);
                }
            }
        });
        return result;
    }

    @Override
    public double[] negate(double[] x) {
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        int n = x.length;
        double[] result = new double[n];
        parallelForEach(n, (start, end) -> {
            for (int i = start; i < end; i++) {
                result[i] = -x[i];
            }
        });
        return result;
    }

    @Override
    public double[][] negate(double[][] x) {
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        if (x.length == 0) return new double[0][0];

        int rows = x.length;
        double[][] result = new double[rows][x[0].length];
        parallelForEach(rows, (start, end) -> {
            for (int row = start; row < end; row++) {
                double[] xr = x[row], out = result[row];
                for (int j = 0; j < xr.length; j++) {
                    out[j] = -xr[j];
                }
            }
        });
        return result;
    }

    @Override
    public boolean[] logicalCompare(double[] x1, double[] x2, LogicalCompare operation) {
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
    public boolean[] logicalOperate(double[] x1, double[] x2, LogicalOperation operation) {
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

        // 创建结果数组
        boolean[] result = new boolean[x1.length];

        // 使用普通Java循环进行逻辑运算
        for (int i = 0; i < x1.length; i++) {
            // 将double值转换为布尔值：非零为true，零为false
            boolean boolA = (x1[i] != 0.0);

            switch (operation) {
                case NOT:
                    result[i] = !boolA;
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation + "。logicalOperate(double[], LogicalOperation)方法仅支持NOT操作。对于AND、OR、XOR操作，请使用双参数版本。");
            }
        }

        return result;
    }

    @Override
    public double[][] transpose(double[][] matrix) {
        if (matrix == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }
        if (matrix.length == 0) {
            throw new IllegalArgumentException("矩阵不能为空");
        }

        int m = matrix.length;
        int n = matrix[0].length;
        double[][] result = new double[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }

    @Override
    public double[][] transpose(double[] rowVector) {
        // 参数验证
        if (rowVector == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        int length = rowVector.length;
        double[][] result = new double[length][1];
        
        // 将行向量转置为列向量
        for (int i = 0; i < length; i++) {
            result[i][0] = rowVector[i];
        }
        
        return result;
    }

    @Override
    public double[][] mmul(double[][] data, double[][] otherData) {
        int m = data.length;          // 当前矩阵行数
        if (m == 0) return new double[0][];
        int otherRows = otherData.length;
        if (otherRows == 0) return new double[0][];
        int n = data[0].length;       // 当前矩阵列数
        int p = otherData[0].length;  // 另一个矩阵列数
        if (n != otherRows) {
            throw new IllegalArgumentException(
                "Matrix dimension mismatch for mmul: A is " + m + "x" + n + ", B is " + otherRows + "x" + p);
        }

        double[][] ob = HpcGemm.tryMatMul(data, otherData);
        if (ob != null) {
            return ob;
        }

        // 计算复杂度
        long complexity = (long) m * n * p;

        // Strassen路径：大矩阵且启用时使用O(n^2.807)算法
        if (USE_STRASSEN && complexity >= STRASSEN_COMPLEXITY_THRESHOLD
                && Math.min(m, Math.min(n, p)) >= STRASSEN_THRESHOLD) {
            return mmulStrassen(data, otherData, m, n, p);
        }

        // 阈值与 SIMDDoubleComputer.mmul 对齐
        if (complexity < 4000) {
            // 极小矩阵：直接计算
            return naiveMultiply(data, otherData, m, n, p);
        } else if (complexity < 32000) {
            // 小矩阵：循环展开优化
            return unrolledMultiply(data, otherData, m, n, p);
        } else if (complexity < 8000000) {
            // 中等矩阵：分块算法 (~n<200)
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
    private double[][] naiveMultiply(double[][] data, double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        for (int i = 0; i < m; i++) {
            double[] thisRow = data[i];
            double[] resultRow = result[i];

            for (int k = 0; k < n; k++) {
                double aik = thisRow[k];
                double[] otherRow = otherData[k];
                for (int j = 0; j < p; j++) {
                    resultRow[j] += aik * otherRow[j];
                }
            }
        }

        return result;
    }

    /**
     * 循环展开优化的矩阵乘法 / Loop unrolled matrix multiplication
     * i-k-j 顺序：缓存友好，otherData[k] 整行连续访问
     */
    private double[][] unrolledMultiply(double[][] data, double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        for (int i = 0; i < m; i++) {
            double[] thisRow = data[i];
            double[] resultRow = result[i];

            for (int k = 0; k < n; k++) {
                double aik = thisRow[k];
                double[] otherRow = otherData[k];

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
     * ii-kk-jj 外层分块 + i-k-j 内层 + 8路展开：缓存友好，otherData[k] 整行连续访问
     */
    private double[][] blockMultiply(double[][] data, double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        int blockSize = calculateOptimalBlockSize(m, n, p);

        for (int ii = 0; ii < m; ii += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                for (int jj = 0; jj < p; jj += blockSize) {

                    int iEnd = Math.min(ii + blockSize, m);
                    int jEnd = Math.min(jj + blockSize, p);
                    int kEnd = Math.min(kk + blockSize, n);

                    for (int i = ii; i < iEnd; i++) {
                        double[] thisRow = data[i];
                        double[] resultRow = result[i];

                        for (int k = kk; k < kEnd; k++) {
                            double aik = thisRow[k];
                            double[] otherRow = otherData[k];
                            double r0, r1, r2, r3, r4, r5, r6, r7;

                            int j = jj;
                            for (; j < jEnd - 7; j += 8) {
                                r0 = resultRow[j]     + aik * otherRow[j];
                                r1 = resultRow[j + 1] + aik * otherRow[j + 1];
                                r2 = resultRow[j + 2] + aik * otherRow[j + 2];
                                r3 = resultRow[j + 3] + aik * otherRow[j + 3];
                                r4 = resultRow[j + 4] + aik * otherRow[j + 4];
                                r5 = resultRow[j + 5] + aik * otherRow[j + 5];
                                r6 = resultRow[j + 6] + aik * otherRow[j + 6];
                                r7 = resultRow[j + 7] + aik * otherRow[j + 7];
                                resultRow[j]     = r0;
                                resultRow[j + 1] = r1;
                                resultRow[j + 2] = r2;
                                resultRow[j + 3] = r3;
                                resultRow[j + 4] = r4;
                                resultRow[j + 5] = r5;
                                resultRow[j + 6] = r6;
                                resultRow[j + 7] = r7;
                            }
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
     * 并行分块矩阵乘法，使用 ForkJoinPool 的 work-stealing 平衡负载。
     * 按行粗粒度分区（每分区 = m/numThreads 行），减少任务数避免 B 矩阵缓存争用。
     */
    private double[][] parallelBlockMultiply(double[][] data, double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];
        int blockSize = calculateOptimalBlockSize(m, n, p);
        java.util.concurrent.ForkJoinPool pool = RereDoubleMatrix.getThreadPool();

        if (pool == null || pool.isShutdown()) {
            return blockMultiply(data, otherData, m, n, p);
        }

        int numThreads = pool.getParallelism();
        if (numThreads <= 1 || m < blockSize * 2) {
            return blockMultiply(data, otherData, m, n, p);
        }

        // 粗粒度行分区：将 m 行分给 numThreads 个任务，减少 B 矩阵跨核缓存争用
        int rowsPerTask = (m + numThreads - 1) / numThreads;
        List<Future<Void>> futures = new ArrayList<>();
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
                            double[] thisRow = data[i];
                            double[] resultRow = result[i];

                            for (int k = kk; k < kEnd; k++) {
                                double aik = thisRow[k];
                                double[] otherRow = otherData[k];
                                double r0, r1, r2, r3, r4, r5, r6, r7;

                                int j = jj;
                                for (; j < jEnd - 7; j += 8) {
                                    r0 = resultRow[j]     + aik * otherRow[j];
                                    r1 = resultRow[j + 1] + aik * otherRow[j + 1];
                                    r2 = resultRow[j + 2] + aik * otherRow[j + 2];
                                    r3 = resultRow[j + 3] + aik * otherRow[j + 3];
                                    r4 = resultRow[j + 4] + aik * otherRow[j + 4];
                                    r5 = resultRow[j + 5] + aik * otherRow[j + 5];
                                    r6 = resultRow[j + 6] + aik * otherRow[j + 6];
                                    r7 = resultRow[j + 7] + aik * otherRow[j + 7];
                                    resultRow[j]     = r0;
                                    resultRow[j + 1] = r1;
                                    resultRow[j + 2] = r2;
                                    resultRow[j + 3] = r3;
                                    resultRow[j + 4] = r4;
                                    resultRow[j + 5] = r5;
                                    resultRow[j + 6] = r6;
                                    resultRow[j + 7] = r7;
                                }
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
                return null;
            }));
        }

        try {
            for (Future<Void> f : futures) f.get();
        } catch (Exception e) {
            log.warn("Parallel block-multiply failed, falling back to sequential: {}", e.getMessage());
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
        // 假设L1缓存32KB，Double占8字节
        int l1CacheSize = 32 * 1024 / 8; // 4096个Double

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
    public double[][] outer(double[] a, double[] b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("向量不能为null / Vector cannot be null");
        }

        int rows = a.length;
        int cols = b.length;
        double[][] result = new double[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i][j] = a[i] * b[j];  // Fixed: was using a[j] instead of b[j]
            }
        }
        return result;
    }

    @Override
    public double[] sign(double[] data) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            double v = data[i];
            if (Double.isNaN(v)) {
                result[i] = Double.NaN; // preserve NaN (consistent with SIMD path)
            } else if (v > 0) {
                result[i] = 1.0;
            } else if (v < 0) {
                result[i] = -1.0;
            } else {
                result[i] = 0.0;
            }
        }
        return result;
    }

    @Override
    public double[][] sign(double[][] array) {
        // 参数验证
        if (array == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // 创建结果矩阵
        double[][] result = new double[array.length][];
        
        // 对每一行计算符号函数
        for (int i = 0; i < array.length; i++) {
            result[i] = sign(array[i]);
        }

        return result;
    }

    @Override
    public double[] diff(double[] data, int n) {
        // 参数验证
        if (data == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }
        
        if (n <= 0) {
            throw new IllegalArgumentException("步长必须为正整数");
        }

        // 如果步长大于等于数组长度，返回空数组
        if (n >= data.length) {
            return new double[0];
        }

        // 创建结果数组，长度为 data.length - n
        double[] result = new double[data.length - n];

        // 计算差分：result[i] = data[i + n] - data[i]
        for (int i = 0; i < result.length; i++) {
            result[i] = data[i + n] - data[i];
        }

        return result;
    }

    @Override
    public boolean[][] logicalCompare(double[][] x1, double[][] x2, LogicalCompare operation) {
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
    public boolean[][] logicalOperate(double[][] x1, LogicalOperation operation) {
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
                // 将double值转换为布尔值：非零为true，零为false
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
    public boolean[][] logicalOperate(double[][] x1, double[][] x2, LogicalOperation operation) {
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
                // 将double值转换为布尔值：非零为true，零为false
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

    // ========== Strassen矩阵乘法 ==========

    /**
     * Strassen O(n^2.807) matrix multiplication for large matrices.
     * Odd dimensions are padded and the result stripped.
     */
    private double[][] mmulStrassen(double[][] a, double[][] b, int m, int n, int p) {
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

        if (mPad <= STRASSEN_THRESHOLD || nPad <= STRASSEN_THRESHOLD || pPad <= STRASSEN_THRESHOLD) {
            return blockMultiply(aPad, bPad, mPad, nPad, pPad);
        }

        int effective = Math.min(Math.min(mPad, nPad), pPad);
        effective = (effective + 1) & ~1;

        double[][] resultPad = strassenSquareRecursive(aPad, bPad, effective,
                0, 0, 0, 0, 0, 0, true);

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
            double[][] leafResult = blockMultiply(aSub, bSub, d, d, d);
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

            futures.add(pool.submit(() -> blockMultiply(A11pA22, B11pB22, half, half, half)));   // M1
            futures.add(pool.submit(() -> blockMultiply(A21pA22, getSub(b, bRowOff, bColOff, half), half, half, half))); // M2
            futures.add(pool.submit(() -> blockMultiply(getSub(a, aRowOff, aColOff, half), B12mB22, half, half, half))); // M3
            futures.add(pool.submit(() -> blockMultiply(getSub(a, aRowOff + half, aColOff + half, half), B21mB11, half, half, half))); // M4
            futures.add(pool.submit(() -> blockMultiply(A11pA12, getSub(b, bRowOff + half, bColOff + half, half), half, half, half))); // M5
            futures.add(pool.submit(() -> blockMultiply(A21mA11, B11pB12, half, half, half))); // M6
            futures.add(pool.submit(() -> blockMultiply(A12mA22, B21pB22, half, half, half))); // M7

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
            addInto(a11, a22, scratch);               // scratch = a11+a22
            double[][] m1a = acquireScratch(half);
            addInto(b11, b22, m1a);                   // m1a = b11+b22
            M1 = blockMultiply(scratch, m1a, half, half, half);
            releaseScratch(m1a);

            // M2 = (a21+a22) @ b11
            double[][] a21 = acquireScratch(half);
            getSubInto(a, aRowOff + half, aColOff, a21);
            addInto(a21, a22, scratch);               // scratch = a21+a22
            M2 = blockMultiply(scratch, b11, half, half, half);

            // M3 = a11 @ (b12-b22)
            double[][] b12 = acquireScratch(half);
            getSubInto(b, bRowOff, bColOff + half, b12);
            subInto(b12, b22, scratch);               // scratch = b12-b22
            M3 = blockMultiply(a11, scratch, half, half, half);
            releaseScratch(a11);

            // M4 = a22 @ (b21-b11)
            double[][] b21 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff, b21);
            subInto(b21, b11, scratch);               // scratch = b21-b11
            releaseScratch(b11);
            M4 = blockMultiply(a22, scratch, half, half, half);
            releaseScratch(a22);

            // M5 = (a11+a12) @ b22 — a11 already released, reuse it
            double[][] a12 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff + half, a12);
            a11 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff, a11);
            addInto(a11, a12, scratch);               // scratch = a11+a12
            releaseScratch(a11);
            releaseScratch(a12);
            M5 = blockMultiply(scratch, b22, half, half, half);
            releaseScratch(b22);

            // M6 = (a21-a11) @ (b11+b12)
            a11 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff, a11);
            subInto(a21, a11, scratch);               // scratch = a21-a11
            releaseScratch(a21);
            double[][] b11b = acquireScratch(half);
            b11 = acquireScratch(half);
            getSubInto(b, bRowOff, bColOff, b11);
            addInto(b11, b12, b11b);                  // b11b = b11+b12
            releaseScratch(b11);
            releaseScratch(b12);
            M6 = blockMultiply(scratch, b11b, half, half, half);
            releaseScratch(b11b);

            // M7 = (a12-a22) @ (b21+b22)
            a12 = acquireScratch(half);
            a22 = acquireScratch(half);
            getSubInto(a, aRowOff, aColOff + half, a12);
            getSubInto(a, aRowOff + half, aColOff + half, a22);
            subInto(a12, a22, scratch);               // scratch = a12-a22
            releaseScratch(a12);
            releaseScratch(a22);
            double[][] b22b = acquireScratch(half);
            b22 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff + half, b22);
            b21 = acquireScratch(half);
            getSubInto(b, bRowOff + half, bColOff, b21);
            addInto(b21, b22, b22b);                  // b22b = b21+b22
            releaseScratch(b21);
            releaseScratch(b22);
            M7 = blockMultiply(scratch, b22b, half, half, half);
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

    /** Element-wise matrix addition: a + b (allocates result) */
    private static double[][] add(double[][] a, double[][] b) {
        int m = a.length, n = a[0].length;
        double[][] r = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                r[i][j] = a[i][j] + b[i][j];
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

    /** Element-wise matrix subtraction: a - b (allocates result) */
    private static double[][] sub(double[][] a, double[][] b) {
        int m = a.length, n = a[0].length;
        double[][] r = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                r[i][j] = a[i][j] - b[i][j];
        return r;
    }

    /** Element-wise matrix subtraction into pre-allocated output: out = a - b */
    private static void subInto(double[][] a, double[][] b, double[][] out) {
        int m = a.length;
        for (int i = 0; i < m; i++) {
            double[] ai = a[i], bi = b[i], oi = out[i];
            for (int j = 0; j < ai.length; j++) oi[j] = ai[j] - bi[j];
        }
    }

    /** Extract a submatrix of size d×d (allocates result) */
    private static double[][] getSub(double[][] a, int rowOff, int colOff, int d) {
        double[][] r = new double[d][d];
        for (int i = 0; i < d; i++)
            System.arraycopy(a[rowOff + i], colOff, r[i], 0, d);
        return r;
    }

    /** Extract a submatrix into pre-allocated output */
    private static void getSubInto(double[][] a, int rowOff, int colOff, double[][] out) {
        int d = out.length;
        for (int i = 0; i < d; i++)
            System.arraycopy(a[rowOff + i], colOff, out[i], 0, d);
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

    // ===================== Parallel dispatch helpers =====================

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

    /**
     * Parallel reduce over a 1D array. Splits into chunks, reduces each in parallel,
     * then combines partial results.
     */
    private static double parallelReduce(double[] x, double identity,
            java.util.function.DoubleBinaryOperator combiner,
            java.util.function.Function<double[], Double> reducer) {
        int n = x.length;
        ForkJoinPool pool = RereDoubleMatrix.getThreadPool();
        if (n < PARALLEL_THRESHOLD || pool == null || pool.isShutdown()) {
            return reducer.apply(x);
        }
        int nThreads = pool.getParallelism();
        if (nThreads <= 1) return reducer.apply(x);
        int chunk = (n + nThreads - 1) / nThreads;
        double[] partials = new double[nThreads];
        List<Future<?>> futures = new ArrayList<>(nThreads);
        for (int t = 0; t < nThreads; t++) {
            final int idx = t;
            int start = t * chunk;
            int end = Math.min(start + chunk, n);
            if (start >= end) { partials[idx] = identity; continue; }
            final int s = start, e = end;
            futures.add(pool.submit(() -> {
                double[] slice = new double[e - s];
                System.arraycopy(x, s, slice, 0, e - s);
                partials[idx] = reducer.apply(slice);
            }));
        }
        for (Future<?> f : futures) {
            try { f.get(); } catch (Exception e) { throw new RuntimeException(e); }
        }
        double result = identity;
        for (double p : partials) result = combiner.applyAsDouble(result, p);
        return result;
    }

}
