package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.BinaryReduceOperation;
import com.yishape.lab.math.compute.ops.LogicalCompare;
import com.yishape.lab.math.compute.ops.LogicalOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;

/**
 * Double Vector Computer Interface
 * 双精度向量计算器接口
 *
 * This interface defines comprehensive mathematical operations for double-precision vectors and matrices.
 * It supports element-wise operations, reduction operations, logical operations, matrix operations,
 * and various mathematical transformations commonly used in scientific computing and machine learning.
 *
 * 该接口定义了双精度向量和矩阵的综合数学运算。
 * 支持元素级运算、归约运算、逻辑运算、矩阵运算以及科学计算和机器学习中常用的各种数学变换。
 *
 * @author lteb2
 * @version 1.0
 * @since 2024
 */
public interface IDoubleVectorComputer {





    /**
     * Perform binary operation between two vectors.
     * 对两个向量执行二元运算。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @param operation the binary operation to perform / 要执行的二元运算
     * @return result vector after applying the operation / 运算后的结果向量
     */
    public double[] binaryOperate(double[] x1, double[] x2, BinaryOperation operation);

    /**
     * Perform binary operation between a vector and a scalar.
     * 对向量和标量执行二元运算。
     *
     * @param x1 input vector / 输入向量
     * @param x2 scalar value / 标量值
     * @param operation the binary operation to perform / 要执行的二元运算
     * @return result vector after applying the operation / 运算后的结果向量
     */
    public double[] binaryOperate(double[] x1, double x2, BinaryOperation operation);

    /**
     * Perform binary operation between two matrices.
     * 对两个矩阵执行二元运算。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @param operation the binary operation to perform / 要执行的二元运算
     * @return result matrix after applying the operation / 运算后的结果矩阵
     */
    public double[][] binaryOperate(double[][] x1, double[][] x2, BinaryOperation operation);

    /**
     * Perform binary operation between a matrix and a scalar.
     * 对矩阵和标量执行二元运算。
     *
     * @param x1 input matrix / 输入矩阵
     * @param x2 scalar value / 标量值
     * @param operation the binary operation to perform / 要执行的二元运算
     * @return result matrix after applying the operation / 运算后的结果矩阵
     */
    public double[][] binaryOperate(double[][] x1, double x2, BinaryOperation operation);

    /**
     * Apply universal operation to a vector with an additional parameter.
     * 对向量应用通用运算（带附加参数）。
     *
     * @param x input vector / 输入向量
     * @param operation the universal operation to apply / 要应用的通用运算
     * @param additionalParam additional parameter for the operation / 运算的附加参数
     * @return result vector after applying the operation / 运算后的结果向量
     */
    public double[] universalOperate(double[] x, UniversalOperation operation, double additionalParam);

    /**
     * Apply universal operation to a matrix with an additional parameter.
     * 对矩阵应用通用运算（带附加参数）。
     *
     * @param x input matrix / 输入矩阵
     * @param operation the universal operation to apply / 要应用的通用运算
     * @param additionalParam additional parameter for the operation / 运算的附加参数
     * @return result matrix after applying the operation / 运算后的结果矩阵
     */
    public double[][] universalOperate(double[][] x, UniversalOperation operation, double additionalParam);

    /**
     * Reduce a vector to a scalar using the specified reduction operation.
     * 使用指定的归约运算将向量归约为标量。
     *
     * @param x input vector / 输入向量
     * @param operation the reduction operation to perform / 要执行的归约运算
     * @return scalar result of the reduction / 归约的标量结果
     */
    public double reduceOperate(double[] x, ReduceOperation operation);

    /**
     * Reduce a matrix to a scalar using the specified reduction operation.
     * 使用指定的归约运算将矩阵归约为标量。
     *
     * @param x input matrix / 输入矩阵
     * @param operation the reduction operation to perform / 要执行的归约运算
     * @return scalar result of the reduction / 归约的标量结果
     */
    public double reduceOperate(double[][] x, ReduceOperation operation);

    /**
     * Perform binary reduction operation between two vectors.
     * 对两个向量执行二元归约运算。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @param operation the binary reduction operation to perform / 要执行的二元归约运算
     * @return scalar result of the binary reduction / 二元归约的标量结果
     */
    public double binaryReduceOperate(double[] x1, double[] x2, BinaryReduceOperation operation);

    /**
     * Perform binary reduction operation between two matrices.
     * 对两个矩阵执行二元归约运算。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @param operation the binary reduction operation to perform / 要执行的二元归约运算
     * @return scalar result of the binary reduction / 二元归约的标量结果
     */
    public double binaryReduceOperate(double[][] x1, double[][] x2, BinaryReduceOperation operation);

    /**
     * Compute element-wise minimum between two vectors.
     * 计算两个向量的逐元素最小值。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @return vector containing element-wise minimum values / 包含逐元素最小值的向量
     */
    public double[] elementWiseMin(double[] x1, double[] x2);

    /**
     * Compute element-wise minimum between two matrices.
     * 计算两个矩阵的逐元素最小值。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @return matrix containing element-wise minimum values / 包含逐元素最小值的矩阵
     */
    public double[][] elementWiseMin(double[][] x1, double[][] x2);

    /**
     * Compute element-wise maximum between two vectors.
     * 计算两个向量的逐元素最大值。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @return vector containing element-wise maximum values / 包含逐元素最大值的向量
     */
    public double[] elementWiseMax(double[] x1, double[] x2);

    /**
     * Compute element-wise maximum between two matrices.
     * 计算两个矩阵的逐元素最大值。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @return matrix containing element-wise maximum values / 包含逐元素最大值的矩阵
     */
    public double[][] elementWiseMax(double[][] x1, double[][] x2);

    /**
     * Compute element-wise negation of a vector.
     * 计算向量的逐元素取反。
     *
     * @param x input vector / 输入向量
     * @return vector with negated values / 取反后的向量
     */
    public double[] negate(double[] x);

    /**
     * Create a new vector filled with a scalar value.
     * 创建填充指定标量值的新向量。
     *
     * @param size vector size / 向量大小
     * @param value fill value / 填充值
     * @return new vector with all elements set to value / 所有元素为value的新向量
     */
    public double[] fill(int size, double value);

    /**
     * Compute element-wise negation of a matrix.
     * 计算矩阵的逐元素取反。
     *
     * @param x input matrix / 输入矩阵
     * @return matrix with negated values / 取反后的矩阵
     */
    public double[][] negate(double[][] x);

    /**
     * Perform element-wise logical comparison between two vectors.
     * 对两个向量执行逐元素逻辑比较。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @param operation the logical comparison operation / 逻辑比较运算
     * @return boolean array with comparison results / 比较结果的布尔数组
     */
    public boolean[] logicalCompare(double[] x1, double[] x2, LogicalCompare operation);

    /**
     * Perform element-wise logical comparison between a vector and a scalar.
     * 对向量和标量执行逐元素逻辑比较。
     *
     * @param x input vector / 输入向量
     * @param scalar scalar value to compare against / 比较标量
     * @param operation the logical comparison operation / 逻辑比较运算
     * @return boolean array with comparison results / 比较结果的布尔数组
     */
    public boolean[] logicalCompare(double[] x, double scalar, LogicalCompare operation);

    /**
     * Perform logical operation on a vector with a custom predicate.
     * 使用自定义谓词对向量执行逻辑运算。
     *
     * @param x input vector / 输入向量
     * @param predicate element-wise predicate / 逐元素谓词
     * @return boolean array with predicate results / 谓词结果的布尔数组
     */
    public boolean[] logicalOperate(double[] x, java.util.function.DoublePredicate predicate);

    /**
     * Perform logical operation on a 2D vector array with a custom predicate.
     * 使用自定义谓词对二维向量数组执行逻辑运算。
     *
     * @param x input 2D vector array / 输入二维向量数组
     * @param predicate element-wise predicate / 逐元素谓词
     * @return boolean 2D array with predicate results / 谓词结果的布尔二维数组
     */
    public boolean[][] logicalOperate(double[][] x, java.util.function.DoublePredicate predicate);

    /**
     * Perform logical operation on a vector.
     * 对向量执行逻辑运算。
     *
     * @param x1 input vector / 输入向量
     * @param operation the logical operation to perform / 要执行的逻辑运算
     * @return boolean array with operation results / 运算结果的布尔数组
     */
    public boolean[] logicalOperate(double[] x1, LogicalOperation operation);

    /**
     * Perform logical operation between two vectors.
     * 对两个向量执行逻辑运算。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @param operation the logical operation to perform / 要执行的逻辑运算
     * @return boolean array with operation results / 运算结果的布尔数组
     */
    public boolean[] logicalOperate(double[] x1, double[] x2, LogicalOperation operation);

    /**
     * Perform element-wise logical comparison between two matrices.
     * 对两个矩阵执行逐元素逻辑比较。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @param operation the logical comparison operation / 逻辑比较运算
     * @return boolean matrix with comparison results / 比较结果的布尔矩阵
     */
    public boolean[][] logicalCompare(double[][] x1, double[][] x2, LogicalCompare operation);

    /**
     * Perform logical operation on a matrix.
     * 对矩阵执行逻辑运算。
     *
     * @param x1 input matrix / 输入矩阵
     * @param operation the logical operation to perform / 要执行的逻辑运算
     * @return boolean matrix with operation results / 运算结果的布尔矩阵
     */
    public boolean[][] logicalOperate(double[][] x1, LogicalOperation operation);

    /**
     * Perform logical operation between two matrices.
     * 对两个矩阵执行逻辑运算。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @param operation the logical operation to perform / 要执行的逻辑运算
     * @return boolean matrix with operation results / 运算结果的布尔矩阵
     */
    public boolean[][] logicalOperate(double[][] x1, double[][] x2, LogicalOperation operation);

    /**
     * Transpose a matrix.
     * 转置矩阵。
     *
     * @param matrix input matrix to transpose / 要转置的输入矩阵
     * @return transposed matrix / 转置后的矩阵
     */
    public double[][] transpose(double[][] matrix);

    /**
     * Convert a row vector to a column vector (single row to single column).
     * 将行向量转换为列向量（单行转单列）。
     *
     * @param rowVector input row vector / 输入行向量
     * @return column vector representation / 列向量表示
     */
    public double[][] transpose(double[] rowVector);

    /**
     * Perform matrix multiplication between two matrices.
     * 执行两个矩阵的矩阵乘法。
     *
     * @param a first matrix / 第一个矩阵
     * @param b second matrix / 第二个矩阵
     * @return result of matrix multiplication / 矩阵乘法的结果
     */
    public double[][] mmul(double[][] a, double[][] b);

    /**
     * Compute outer product between two vectors.
     * 计算两个向量的外积。
     *
     * @param a first vector / 第一个向量
     * @param b second vector / 第二个向量
     * @return outer product matrix / 外积矩阵
     */
    public double[][] outer(double[] a, double[] b);

    /**
     * Compute element-wise sign of vector elements.
     * 计算向量元素的逐元素符号。
     *
     * @param array input vector / 输入向量
     * @return vector with sign values (-1, 0, 1) / 符号值向量（-1, 0, 1）
     */
    public double[] sign(double[] array);

    /**
     * Compute element-wise sign of matrix elements.
     * 计算矩阵元素的逐元素符号。
     *
     * @param array input matrix / 输入矩阵
     * @return matrix with sign values (-1, 0, 1) / 符号值矩阵（-1, 0, 1）
     */
    public double[][] sign(double[][] array);

    /**
     * Compute differences between consecutive elements with specified stride.
     * 以指定步长计算连续元素间的差分。
     *
     * @param array input array / 输入数组
     * @param stride step size for computing differences / 计算差分的步长
     * @return array of differences / 差分数组
     */
    public double[] diff(double[] array, int stride);

    /**
     * Element-wise conditional selection: result[i] = mask[i] ? a[i] : b[i].
     * Equivalent to numpy.where(mask, a, b).
     * 逐元素条件选择：mask 为 true 取 a，false 取 b。
     *
     * @param mask boolean mask array / 布尔掩码数组
     * @param a    values where mask is true / mask 为 true 时的取值
     * @param b    values where mask is false / mask 为 false 时的取值
     * @return result array / 结果数组
     */
    public double[] where(boolean[] mask, double[] a, double[] b);

    /**
     * Element-wise conditional selection for matrices.
     * 矩阵逐元素条件选择。
     */
    public double[][] where(boolean[][] mask, double[][] a, double[][] b);

    /**
     * In-place binary operation: {@code target[i] = target[i] OP source[i]}.
     * Modifies {@code target} directly without allocating a new array.
     * Supports ADD, SUBTRACT, MULTIPLY and DIVIDE.
     *
     * @param target    first operand AND output array (modified in-place)
     * @param source    second operand array (read-only)
     * @param operation the binary operation (ADD, SUBTRACT, MULTIPLY, DIVIDE)
     */
    void binaryOperateInPlace(double[] target, double[] source, BinaryOperation operation);

    /**
     * In-place binary operation over a sub-range:
     * {@code target[targetOffset + i] = target[targetOffset + i] OP source[sourceOffset + i]}
     * for {@code 0 <= i < length}. Enables strided/row-wise accumulation (e.g. bias broadcast)
     * without allocating tiled buffers.
     *
     * @param target        first operand AND output array (modified in-place)
     * @param targetOffset  start index in {@code target}
     * @param source        second operand array (read-only)
     * @param sourceOffset  start index in {@code source}
     * @param length        number of elements to process
     * @param operation     the binary operation (ADD, SUBTRACT, MULTIPLY, DIVIDE)
     */
    void binaryOperateInPlace(double[] target, int targetOffset,
                              double[] source, int sourceOffset, int length,
                              BinaryOperation operation);

    /**
     * In-place clamp: {@code data[i] = min(max(data[i], min), max)}.
     * Modifies {@code data} directly without allocating a new array.
     *
     * @param data operand AND output array (modified in-place)
     * @param min  lower bound
     * @param max  upper bound
     */
    void clampInPlace(double[] data, double min, double max);

    /**
     * In-place binary operation with scalar: {@code target[i] = target[i] OP scalar}.
     * Modifies {@code target} directly without allocating a new array.
     *
     * @param target    operand AND output array (modified in-place)
     * @param scalar    scalar operand (read-only)
     * @param operation the binary operation (ADD, SUBTRACT, MULTIPLY, DIVIDE)
     */
    void binaryOperateInPlace(double[] target, double scalar, BinaryOperation operation);

}
