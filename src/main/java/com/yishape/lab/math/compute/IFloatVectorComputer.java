package com.yishape.lab.math.compute;

import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.BinaryReduceOperation;
import com.yishape.lab.math.compute.ops.LogicalCompare;
import com.yishape.lab.math.compute.ops.LogicalOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;

/**
 * Float Vector Computer Interface
 * 单精度向量计算器接口
 *
 * This interface defines comprehensive mathematical operations for single-precision float vectors and matrices.
 * It provides the same functionality as IDoubleVectorComputer but optimized for single-precision floating-point arithmetic,
 * offering better memory efficiency and performance for applications where double precision is not required.
 *
 * 该接口定义了单精度浮点向量和矩阵的综合数学运算。
 * 提供与IDoubleVectorComputer相同的功能，但针对单精度浮点运算进行优化，
 * 在不需要双精度的情况下提供更好的内存效率和性能。
 *
 * @author lteb2
 * @version 1.0
 * @since 2024
 */
public interface IFloatVectorComputer {
    
    
    /**
     * Perform binary operation between two float vectors.
     * 对两个单精度向量执行二元运算。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @param operation the binary operation to perform / 要执行的二元运算
     * @return result vector after applying the operation / 运算后的结果向量
     */
    public float[] binaryOperate(float[] x1, float[] x2, BinaryOperation operation);
    
    /**
     * Perform binary operation between a float vector and a scalar.
     * 对单精度向量和标量执行二元运算。
     *
     * @param x1 input vector / 输入向量
     * @param x2 scalar value / 标量值
     * @param operation the binary operation to perform / 要执行的二元运算
     * @return result vector after applying the operation / 运算后的结果向量
     */
    public float[] binaryOperate(float[] x1, float x2, BinaryOperation operation);
    
    /**
     * Perform binary operation between two float matrices.
     * 对两个单精度矩阵执行二元运算。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @param operation the binary operation to perform / 要执行的二元运算
     * @return result matrix after applying the operation / 运算后的结果矩阵
     */
    public float[][] binaryOperate(float[][] x1, float[][] x2, BinaryOperation operation);
    
    /**
     * Perform binary operation between a float matrix and a scalar.
     * 对单精度矩阵和标量执行二元运算。
     *
     * @param x1 input matrix / 输入矩阵
     * @param x2 scalar value / 标量值
     * @param operation the binary operation to perform / 要执行的二元运算
     * @return result matrix after applying the operation / 运算后的结果矩阵
     */
    public float[][] binaryOperate(float[][] x1, float x2, BinaryOperation operation);
    
    /**
     * Apply universal operation to a float vector with an additional parameter.
     * 对单精度向量应用通用运算（带附加参数）。
     *
     * @param x input vector / 输入向量
     * @param operation the universal operation to apply / 要应用的通用运算
     * @param additionalParam additional parameter for the operation / 运算的附加参数
     * @return result vector after applying the operation / 运算后的结果向量
     */
    public float[] universalOperate(float[] x, UniversalOperation operation, float additionalParam);
    
    /**
     * Apply universal operation to a float matrix with an additional parameter.
     * 对单精度矩阵应用通用运算（带附加参数）。
     *
     * @param x input matrix / 输入矩阵
     * @param operation the universal operation to apply / 要应用的通用运算
     * @param additionalParam additional parameter for the operation / 运算的附加参数
     * @return result matrix after applying the operation / 运算后的结果矩阵
     */
    public float[][] universalOperate(float[][] x, UniversalOperation operation, float additionalParam);

    /**
     * Reduce a float vector to a scalar using the specified reduction operation.
     * 使用指定的归约运算将单精度向量归约为标量。
     *
     * @param x input vector / 输入向量
     * @param operation the reduction operation to perform / 要执行的归约运算
     * @return scalar result of the reduction / 归约的标量结果
     */
    public float reduceOperate(float[] x, ReduceOperation operation);

    /**
     * Reduce a float matrix to a scalar using the specified reduction operation.
     * 使用指定的归约运算将单精度矩阵归约为标量。
     *
     * @param x input matrix / 输入矩阵
     * @param operation the reduction operation to perform / 要执行的归约运算
     * @return scalar result of the reduction / 归约的标量结果
     */
    public float reduceOperate(float[][] x, ReduceOperation operation);

    /**
     * Perform binary reduction operation between two float vectors.
     * 对两个单精度向量执行二元归约运算。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @param operation the binary reduction operation to perform / 要执行的二元归约运算
     * @return scalar result of the binary reduction / 二元归约的标量结果
     */
    public float binaryReduceOperate(float[] x1, float[] x2, BinaryReduceOperation operation);

    /**
     * Perform binary reduction operation between two float matrices.
     * 对两个单精度矩阵执行二元归约运算。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @param operation the binary reduction operation to perform / 要执行的二元归约运算
     * @return scalar result of the binary reduction / 二元归约的标量结果
     */
    public float binaryReduceOperate(float[][] x1, float[][] x2, BinaryReduceOperation operation);

    /**
     * Compute element-wise minimum between two float vectors.
     * 计算两个单精度向量的逐元素最小值。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @return vector containing element-wise minimum values / 包含逐元素最小值的向量
     */
    public float[] elementWiseMin(float[] x1, float[] x2);

    /**
     * Compute element-wise minimum between two float matrices.
     * 计算两个单精度矩阵的逐元素最小值。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @return matrix containing element-wise minimum values / 包含逐元素最小值的矩阵
     */
    public float[][] elementWiseMin(float[][] x1, float[][] x2);

    /**
     * Compute element-wise maximum between two float vectors.
     * 计算两个单精度向量的逐元素最大值。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @return vector containing element-wise maximum values / 包含逐元素最大值的向量
     */
    public float[] elementWiseMax(float[] x1, float[] x2);

    /**
     * Compute element-wise maximum between two float matrices.
     * 计算两个单精度矩阵的逐元素最大值。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @return matrix containing element-wise maximum values / 包含逐元素最大值的矩阵
     */
    public float[][] elementWiseMax(float[][] x1, float[][] x2);

    /**
     * Compute element-wise negation of a float vector.
     * 计算单精度向量的逐元素取反。
     *
     * @param x input vector / 输入向量
     * @return vector with negated values / 取反后的向量
     */
    public float[] negate(float[] x);

    /**
     * Compute element-wise negation of a float matrix.
     * 计算单精度矩阵的逐元素取反。
     *
     * @param x input matrix / 输入矩阵
     * @return matrix with negated values / 取反后的矩阵
     */
    public float[][] negate(float[][] x);

    /**
     * Perform element-wise logical comparison between two float vectors.
     * 对两个单精度向量执行逐元素逻辑比较。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @param operation the logical comparison operation / 逻辑比较运算
     * @return boolean array with comparison results / 比较结果的布尔数组
     */
    public boolean[] logicalCompare(float[] x1, float[] x2,LogicalCompare operation);

    /**
     * Perform logical operation on a float vector.
     * 对单精度向量执行逻辑运算。
     *
     * @param x1 input vector / 输入向量
     * @param operation the logical operation to perform / 要执行的逻辑运算
     * @return boolean array with operation results / 运算结果的布尔数组
     */
    public boolean[] logicalOperate(float[] x1, LogicalOperation operation);

    /**
     * Perform logical operation between two float vectors.
     * 对两个单精度向量执行逻辑运算。
     *
     * @param x1 first vector / 第一个向量
     * @param x2 second vector / 第二个向量
     * @param operation the logical operation to perform / 要执行的逻辑运算
     * @return boolean array with operation results / 运算结果的布尔数组
     */
    public boolean[] logicalOperate(float[] x1,float[] x2, LogicalOperation operation);

    /**
     * Perform element-wise logical comparison between two float matrices.
     * 对两个单精度矩阵执行逐元素逻辑比较。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @param operation the logical comparison operation / 逻辑比较运算
     * @return boolean matrix with comparison results / 比较结果的布尔矩阵
     */
    public boolean[][] logicalCompare(float[][] x1, float[][] x2,LogicalCompare operation);

    /**
     * Perform logical operation on a float matrix.
     * 对单精度矩阵执行逻辑运算。
     *
     * @param x1 input matrix / 输入矩阵
     * @param operation the logical operation to perform / 要执行的逻辑运算
     * @return boolean matrix with operation results / 运算结果的布尔矩阵
     */
    public boolean[][] logicalOperate(float[][] x1, LogicalOperation operation);

    /**
     * Perform logical operation between two float matrices.
     * 对两个单精度矩阵执行逻辑运算。
     *
     * @param x1 first matrix / 第一个矩阵
     * @param x2 second matrix / 第二个矩阵
     * @param operation the logical operation to perform / 要执行的逻辑运算
     * @return boolean matrix with operation results / 运算结果的布尔矩阵
     */
    public boolean[][] logicalOperate(float[][] x1,float[][] x2, LogicalOperation operation);

    /**
     * Transpose a float matrix.
     * 转置单精度矩阵。
     *
     * @param matrix input matrix to transpose / 要转置的输入矩阵
     * @return transposed matrix / 转置后的矩阵
     */
    public float[][] transpose(float[][] matrix);

    /**
     * Convert a float row vector to a column vector (single row to single column).
     * 将单精度行向量转换为列向量（单行转单列）。
     *
     * @param rowVector input row vector / 输入行向量
     * @return column vector representation / 列向量表示
     */
    public float[][] transpose(float[] rowVector);

    /**
     * Perform matrix multiplication between two float matrices.
     * 执行两个单精度矩阵的矩阵乘法。
     *
     * @param a first matrix / 第一个矩阵
     * @param b second matrix / 第二个矩阵
     * @return result of matrix multiplication / 矩阵乘法的结果
     */
    public float[][] mmul(float[][] a, float[][] b);

    /**
     * Compute outer product between two float vectors.
     * 计算两个单精度向量的外积。
     *
     * @param a first vector / 第一个向量
     * @param b second vector / 第二个向量
     * @return outer product matrix / 外积矩阵
     */
    public float[][] outer(float[] a, float[] b);

    /**
     * Compute element-wise sign of float vector elements.
     * 计算单精度向量元素的逐元素符号。
     *
     * @param array input vector / 输入向量
     * @return vector with sign values (-1, 0, 1) / 符号值向量（-1, 0, 1）
     */
    public float[] sign(float[] array);

    /**
     * Compute element-wise sign of float matrix elements.
     * 计算单精度矩阵元素的逐元素符号。
     *
     * @param array input matrix / 输入矩阵
     * @return matrix with sign values (-1, 0, 1) / 符号值矩阵（-1, 0, 1）
     */
    public float[][] sign(float[][] array);

    /**
     * Compute differences between consecutive float elements with specified stride.
     * 以指定步长计算单精度连续元素间的差分。
     *
     * @param array input array / 输入数组
     * @param stride step size for computing differences / 计算差分的步长
     * @return array of differences / 差分数组
     */
    public float[] diff(float[] array, int stride);
 
}
