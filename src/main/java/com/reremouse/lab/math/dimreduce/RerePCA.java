package com.reremouse.lab.math.dimreduce;

import com.reremouse.lab.util.Tuple2;
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;

/**
 * PCA (主成分分析) 降维算法实现类 / PCA (Principal Component Analysis) Dimensionality Reduction Algorithm Implementation
 * <p>
 * 实现基于主成分分析（PCA）的降维算法。PCA是一种线性降维技术，
 * 能够找到数据的主要变化方向，在保持数据主要特征的同时减少数据的维度。
 * </p>
 * <p>
 * Implements dimensionality reduction algorithm based on Principal Component Analysis (PCA).
 * PCA is a linear dimensionality reduction technique that can find the main directions
 * of data variation and reduce data dimensions while preserving main characteristics.
 * </p>
 * 
 * <h3>算法步骤 / Algorithm Steps:</h3>
 * <ol>
 *   <li>数据中心化：减去每列的均值 / Data centering: subtract mean of each column</li>
 *   <li>计算协方差矩阵 / Compute covariance matrix</li>
 *   <li>对协方差矩阵进行特征分解 / Eigendecomposition of covariance matrix</li>
 *   <li>选择前k个最大特征值对应的特征向量 / Select k eigenvectors with largest eigenvalues</li>
 *   <li>将原数据投影到主成分空间 / Project original data to principal component space</li>
 * </ol>
 * 
 * @author lteb2
 */
public class RerePCA {
    
    
    /**
     * 用PCA方法降维 / Dimensionality reduction using PCA method
     * <p>
     * 通过主成分分析（PCA）对输入数据进行降维处理。
     * 算法步骤：
     * 1. 对原始数据进行中心化处理：X_centered = X - mean(X)
     * 2. 计算协方差矩阵：C = (X_centered^T * X_centered) / (n-1)
     * 3. 对协方差矩阵进行特征分解：C = V * Λ * V^T
     * 4. 选择前dim个最大特征值对应的特征向量作为主成分
     * 5. 将原始数据投影到主成分空间：X_reduced = X_centered * V[:, :dim]
     * </p>
     * <p>
     * Performs dimensionality reduction on input data through Principal Component Analysis (PCA).
     * Algorithm steps:
     * 1. Center the original data: X_centered = X - mean(X)
     * 2. Compute covariance matrix: C = (X_centered^T * X_centered) / (n-1)
     * 3. Eigendecomposition of covariance matrix: C = V * Λ * V^T
     * 4. Select the first dim eigenvectors with largest eigenvalues as principal components
     * 5. Project original data to principal component space: X_reduced = X_centered * V[:, :dim]
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
            return IMatrix.of(originalData.getData());
        }
        
        // 步骤1：数据中心化 / Step 1: Data centering
        IMatrix centeredData = originalData.center();
        
        // 步骤2：计算协方差矩阵 / Step 2: Compute covariance matrix
        IMatrix covarianceMatrix = centeredData.covarianceFromCentered();
        
        // 步骤3：对协方差矩阵进行特征分解 / Step 3: Eigendecomposition of covariance matrix
        Tuple2<IVector, IMatrix> eigenResult = covarianceMatrix.eigen();
        IVector eigenValues = eigenResult._1;    // 特征值（已按大小降序排列）/ Eigenvalues (sorted in descending order)
        IMatrix eigenVectors = eigenResult._2;   // 特征向量（列为特征向量）/ Eigenvectors (columns are eigenvectors)
        
        // 步骤4：选择前dim个主成分 / Step 4: Select first dim principal components
        float[][] principalComponentsData = new float[originalCols][dim];
        for (int i = 0; i < dim; i++) {
            IVector eigenVec = eigenVectors.getColunm(i);
            // 将特征向量作为主成分矩阵的列
            for (int j = 0; j < originalCols; j++) {
                principalComponentsData[j][i] = eigenVec.get(j);
            }
        }
        IMatrix principalComponents = IMatrix.of(principalComponentsData);
        
        // 步骤5：将中心化数据投影到主成分空间 / Step 5: Project centered data to principal component space
        IMatrix reducedData = centeredData.mmul(principalComponents);
        
        return reducedData;
    }
    
}
