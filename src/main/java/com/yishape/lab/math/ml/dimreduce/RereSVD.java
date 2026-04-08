package com.yishape.lab.math.ml.dimreduce;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.io.*;

/**
 * SVD降维算法实现类 / SVD Dimensionality Reduction Algorithm Implementation
 * <p>
 * 实现基于奇异值分解（SVD）的降维算法。SVD是一种线性降维技术，
 * 能够在保持数据主要特征的同时减少数据的维度。
 * </p>
 * <p>
 * Implements dimensionality reduction algorithm based on Singular Value Decomposition (SVD).
 * SVD is a linear dimensionality reduction technique that can reduce data dimensions
 * while preserving the main characteristics of the data.
 * </p>
 *
 * @author lteb2
 */
public class RereSVD implements IDimReduce, ISerializableModel {

    private static final Logger log = LoggerFactory.getLogger(RereSVD.class);

    
    private static final long serialVersionUID = 1L;
    
    /**
     * 用SVD方法降维 / Dimensionality reduction using SVD method
     * <p>
     * 通过奇异值分解（SVD）对输入数据进行降维处理。
     * 算法步骤：
     * 1. 对原始数据矩阵进行SVD分解：A = U * S * V^T
     * 2. 选择前dim个最大的奇异值及对应的奇异向量
     * 3. 将原始数据投影到低维空间：A_reduced = A * V[:, :dim]
     * </p>
     * <p>
     * Performs dimensionality reduction on input data through Singular Value Decomposition (SVD).
     * Algorithm steps:
     * 1. Perform SVD decomposition on original data matrix: A = U * S * V^T
     * 2. Select the first dim largest singular values and corresponding singular vectors
     * 3. Project original data to low-dimensional space: A_reduced = A * V[:, :dim]
     * </p>
     *
     * @param originalData 原数据矩阵，每行表示一个样本，每列表示一个特征
     *                     Original data matrix, each row represents a sample, each column represents a feature
     * @param dim 目标维度，即降维后的列数，必须小于等于原始数据的列数
     *            Target dimension, i.e., number of columns after dimensionality reduction, 
     *            must be less than or equal to the number of columns in original data
     * @return 降维后的矩阵，行数保持不变，列数为dim
     *         Dimensionally reduced matrix, row count remains the same, column count is dim
     * @throws IllegalArgumentException 如果dim大于原始数据的列数或小于等于0
     *                                  if dim is greater than the number of columns in original data or less than or equal to 0
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim){
        
        // 参数验证 / Parameter validation
        if (originalData == null) {
            throw new IllegalArgumentException("原始数据不能为空 / Original data cannot be null");
        }
        
        int originalRows = originalData.getRowNum();
        int originalCols = originalData.getColNum();
        
        if (dim <= 0) {
            throw new IllegalArgumentException("目标维度必须大于0 / Target dimension must be greater than 0");
        }
        
        if (dim > originalCols) {
            throw new IllegalArgumentException("目标维度不能大于原始数据的列数 / Target dimension cannot be greater than original column count");
        }
        
        // 如果目标维度等于原始维度，直接返回原数据的副本
        // If target dimension equals original dimension, return a copy of original data
        if (dim == originalCols) {
            return IMatrix.of(originalData.toDoubleArray());
        }
        
        // 执行SVD分解 / Perform SVD decomposition
        // A = U * S * V^T
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> svdResult = originalData.svd();
        IMatrix U = (IMatrix)svdResult._1;           // 左奇异向量矩阵 / Left singular vectors matrix
        IVector singularValues = (IVector)svdResult._2; // 奇异值向量 / Singular values vector
        IMatrix VT = (IMatrix)svdResult._3;          // 右奇异向量转置矩阵 / Right singular vectors transpose matrix
        
        // 从VT中提取前dim个主成分（V的前dim列，即VT的前dim行）
        // Extract first dim principal components from VT (first dim columns of V, i.e., first dim rows of VT)
        // 使用slice方法直接提取前dim行，替代手动循环
        IMatrix projectionMatrix = (IMatrix)VT.slice(0, dim, 0, originalCols).transposeNew();
        
        // 将原始数据投影到低维空间：A_reduced = A * V[:, :dim]
        // Project original data to low-dimensional space: A_reduced = A * V[:, :dim]
        IMatrix result = (IMatrix)originalData.mmul(projectionMatrix);
        
        return result;
    }
    
    /**
     * 将模型保存在本地
     * @param path 保存路径
     */
    @Override
    public void save(String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        } catch (IOException e) {
            log.error("exception", e);
        }
    }
}