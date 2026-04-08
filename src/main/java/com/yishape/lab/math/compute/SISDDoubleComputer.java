package com.yishape.lab.math.compute;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * SISD运算 Java Vector API计算的回退类，使用普通的JAVA循环实现运算
 *
 * @author lteb2
 */
public class SISDDoubleComputer implements IDoubleVectorComputer,Serializable {

    @Override
    public double[] binaryOperate(double[] x1, double[] x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("The length of the vector must be the same. First length: "+x1.length+", second length: "+x2.length);
        }

        // 创建结果数组
        double[] result = new double[x1.length];

        // 使用普通Java循环进行二元运算
        for (int i = 0; i < x1.length; i++) {
            switch (operation) {
                case ADD:
                    result[i] = x1[i] + x2[i];
                    break;
                case SUBTRACT:
                    result[i] = x1[i] - x2[i];
                    break;
                case MULTIPLY:
                    result[i] = x1[i] * x2[i];
                    break;
                case DIVIDE:
                    result[i] = x1[i] / x2[i];
                    break;
                case REMAINDER:
                    result[i] = x1[i] % x2[i];
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }

        return result;
    }

    @Override
    public double[] binaryOperate(double[] x1, double x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 创建结果数组
        double[] result = new double[x1.length];

        // 使用普通Java循环进行二元运算
        for (int i = 0; i < x1.length; i++) {
            switch (operation) {
                case ADD:
                    result[i] = x1[i] + x2;
                    break;
                case SUBTRACT:
                    result[i] = x1[i] - x2;
                    break;
                case MULTIPLY:
                    result[i] = x1[i] * x2;
                    break;
                case DIVIDE:
                    result[i] = x1[i] / x2;
                    break;
                case REMAINDER:
                    result[i] = x1[i] % x2;
                    break;
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }

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

        if (x1.length > 0 && x1[0].length != x2[0].length) {
            throw new IllegalArgumentException("矩阵列数必须相同");
        }

        // 创建结果矩阵
        double[][] result = new double[x1.length][x1[0].length];

        // 对每一行进行二元运算
        for (int row = 0; row < x1.length; row++) {
            result[row] = binaryOperate(x1[row], x2[row], operation);
        }

        return result;
    }

    @Override
    public double[][] binaryOperate(double[][] x1, double x2, BinaryOperation operation) {
        // 参数验证
        if (x1 == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // 创建结果矩阵
        double[][] result = new double[x1.length][x1[0].length];

        // 对每一行进行二元运算
        for (int row = 0; row < x1.length; row++) {
            result[row] = binaryOperate(x1[row], x2, operation);
        }

        return result;
    }

    @Override
    public double[] universalOperate(double[] x, UniversalOperation operation, double additionalParam) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 创建结果数组
        double[] result = new double[x.length];

        // 使用普通Java循环进行通用运算
        for (int i = 0; i < x.length; i++) {
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
                default:
                    throw new IllegalArgumentException("不支持的操作: " + operation);
            }
        }

        return result;
    }

    @Override
    public double[][] universalOperate(double[][] x, UniversalOperation operation, double additionalParam) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // 创建结果矩阵
        double[][] result = new double[x.length][x[0].length];

        // 对每一行进行通用运算
        for (int row = 0; row < x.length; row++) {
            result[row] = universalOperate(x[row], operation, additionalParam);
        }

        return result;
    }

    @Override
    public double reduceOperate(double[] x, ReduceOperation operation) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 对于需要非空检查的操作
        if ((operation == ReduceOperation.MEAN
                || operation == ReduceOperation.MIN
                || operation == ReduceOperation.MAX
                || operation == ReduceOperation.VARIANCE
                || operation == ReduceOperation.STANDARD_DEVIATION
                || operation == ReduceOperation.PROD)
                && x.length == 0) {
            if (operation == ReduceOperation.PROD) {
                return 1.0;
            }
            throw new IllegalArgumentException("向量不能为空");
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
                // 计算均值
                double sum = 0.0;
                for (int i = 0; i < x.length; i++) {
                    sum += x[i];
                }
                double mean = sum / x.length;

                // 计算方差
                double varianceSum = 0.0;
                for (int i = 0; i < x.length; i++) {
                    double diff = x[i] - mean;
                    varianceSum += diff * diff;
                }
                return varianceSum / x.length;
            }

            case STANDARD_DEVIATION: {
                // 计算均值
                double sum = 0.0;
                for (int i = 0; i < x.length; i++) {
                    sum += x[i];
                }
                double mean = sum / x.length;

                // 计算方差
                double varianceSum = 0.0;
                for (int i = 0; i < x.length; i++) {
                    double diff = x[i] - mean;
                    varianceSum += diff * diff;
                }
                double variance = varianceSum / x.length;

                return Math.sqrt(variance);
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

        // 对于需要非空检查的操作
        if ((operation == ReduceOperation.MEAN
                || operation == ReduceOperation.MIN
                || operation == ReduceOperation.MAX
                || operation == ReduceOperation.VARIANCE
                || operation == ReduceOperation.STANDARD_DEVIATION
                || operation == ReduceOperation.PROD)
                && (x.length == 0 || x[0].length == 0)) {
            throw new IllegalArgumentException("矩阵不能为空");
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
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 创建结果数组
        double[] result = new double[x1.length];

        // 使用普通Java循环计算最小值
        for (int i = 0; i < x1.length; i++) {
            result[i] = Math.min(x1[i], x2[i]);
        }

        return result;
    }

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

        // 创建结果矩阵
        double[][] result = new double[x1.length][x1[0].length];

        // 对每一行计算最小值
        for (int row = 0; row < x1.length; row++) {
            result[row] = elementWiseMin(x1[row], x2[row]);
        }

        return result;
    }

    @Override
    public double[] elementWiseMax(double[] x1, double[] x2) {
        // 参数验证
        if (x1 == null || x2 == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        if (x1.length != x2.length) {
            throw new IllegalArgumentException("向量长度必须相同");
        }

        // 创建结果数组
        double[] result = new double[x1.length];

        // 使用普通Java循环计算最大值
        for (int i = 0; i < x1.length; i++) {
            result[i] = Math.max(x1[i], x2[i]);
        }

        return result;
    }

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

        // 创建结果矩阵
        double[][] result = new double[x1.length][x1[0].length];

        // 对每一行计算最大值
        for (int row = 0; row < x1.length; row++) {
            result[row] = elementWiseMax(x1[row], x2[row]);
        }

        return result;
    }

    @Override
    public double[] negate(double[] x) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入向量不能为null");
        }

        // 创建结果数组
        double[] result = new double[x.length];

        // 使用普通Java循环进行取反运算
        for (int i = 0; i < x.length; i++) {
            result[i] = -x[i];
        }

        return result;
    }

    @Override
    public double[][] negate(double[][] x) {
        // 参数验证
        if (x == null) {
            throw new IllegalArgumentException("输入矩阵不能为null");
        }

        // 创建结果矩阵
        double[][] result = new double[x.length][x[0].length];

        // 对每一行进行取反运算
        for (int row = 0; row < x.length; row++) {
            result[row] = negate(x[row]);
        }

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
        int n = data[0].length;       // 当前矩阵列数
        int p = otherData[0].length;  // 另一个矩阵列数

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
     */
    private double[][] naiveMultiply(double[][] data, double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        for (int i = 0; i < m; i++) {
            double[] thisRow = data[i];
            double[] resultRow = result[i];

            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += thisRow[k] * otherData[k][j];
                }
                resultRow[j] = sum;
            }
        }

        return result;
    }

    /**
     * 循环展开优化的矩阵乘法 / Loop unrolled matrix multiplication
     */
    private double[][] unrolledMultiply(double[][] data, double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        for (int i = 0; i < m; i++) {
            double[] thisRow = data[i];
            double[] resultRow = result[i];

            for (int j = 0; j < p; j++) {
                double sum = 0.0;

                // 循环展开，每次处理4个元素
                int k = 0;
                for (; k < n - 3; k += 4) {
                    sum += thisRow[k] * otherData[k][j]
                            + thisRow[k + 1] * otherData[k + 1][j]
                            + thisRow[k + 2] * otherData[k + 2][j]
                            + thisRow[k + 3] * otherData[k + 3][j];
                }

                // 处理剩余元素
                for (; k < n; k++) {
                    sum += thisRow[k] * otherData[k][j];
                }

                resultRow[j] = sum;
            }
        }

        return result;
    }

    /**
     * 分块矩阵乘法 / Block matrix multiplication
     */
    private double[][] blockMultiply(double[][] data, double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        // 动态计算最优块大小
        int blockSize = calculateOptimalBlockSize(m, n, p);

        for (int ii = 0; ii < m; ii += blockSize) {
            for (int jj = 0; jj < p; jj += blockSize) {
                for (int kk = 0; kk < n; kk += blockSize) {

                    int iEnd = Math.min(ii + blockSize, m);
                    int jEnd = Math.min(jj + blockSize, p);
                    int kEnd = Math.min(kk + blockSize, n);

                    // 块内计算
                    for (int i = ii; i < iEnd; i++) {
                        double[] thisRow = data[i];
                        double[] resultRow = result[i];

                        for (int j = jj; j < jEnd; j++) {
                            double sum = resultRow[j];

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

        return result;
    }

    /**
     * 并行分块矩阵乘法 / Parallel block matrix multiplication
     */
    private double[][] parallelBlockMultiply(double[][] data, double[][] otherData, int m, int n, int p) {
        double[][] result = new double[m][p];

        // 计算最优块大小和线程数
        int blockSize = calculateOptimalBlockSize(m, n, p);
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(),
                Math.max(1, (m + blockSize - 1) / blockSize));

        if (numThreads == 1) {
            // 单线程情况下使用普通分块算法
            return blockMultiply(data, otherData, m, n, p);
        }

        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<Void>> futures = new ArrayList<>();

        // 按行分块并行计算
        int rowsPerThread = (m + numThreads - 1) / numThreads;

        for (int t = 0; t < numThreads; t++) {
            final int startRow = t * rowsPerThread;
            final int endRow = Math.min(startRow + rowsPerThread, m);

            if (startRow >= endRow) {
                break;
            }

            Future<Void> future = executor.submit(() -> {
                // 每个线程处理指定行范围的计算
                for (int ii = startRow; ii < endRow; ii += blockSize) {
                    for (int jj = 0; jj < p; jj += blockSize) {
                        for (int kk = 0; kk < n; kk += blockSize) {

                            int iEnd = Math.min(ii + blockSize, endRow);
                            int jEnd = Math.min(jj + blockSize, p);
                            int kEnd = Math.min(kk + blockSize, n);

                            // 块内计算
                            for (int i = ii; i < iEnd; i++) {
                                double[] thisRow = data[i];
                                double[] resultRow = result[i];

                                for (int j = jj; j < jEnd; j++) {
                                    double sum = resultRow[j];

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
        } catch (InterruptedException | ExecutionException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("并行计算失败 / Parallel computation failed", e);
        } finally {
            executor.shutdown();
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
        // 假设L1缓存32KB，Double占4字节
        int l1CacheSize = 32 * 1024 / 4; // 8192个Double

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
            if (data[i] > 0) {
                result[i] = 1.0;
            } else if (data[i] < 0) {
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
    
    

}
