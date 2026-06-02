package com.yishape.lab.math.ml.dr;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.preprocessing.ITransform;
import com.yishape.lab.util.Tuple3;
import com.yishape.lab.math.linalg.IMatrix;

import java.util.LinkedHashMap;
import java.util.Map;

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
public class RereSVD implements ITransform<Double>, ISerializableModel {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereSVD.class);

        private IMatrix<Double> feature;
    private int nComponents = -1;
    
    @Override
    public boolean ifTrained() {
        return nComponents > 0 && feature != null;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null) {
            throw new IllegalArgumentException("特征数据不能为空 / Feature data cannot be null");
        }
        if (nComponents <= 0) {
            throw new IllegalStateException("必须先设置目标维度 / Target dimension must be set first (use setNComponents)");
        }
        this.feature = feature;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!ifTrained()) {
            throw new IllegalStateException("模型尚未训练，请先调用fit / Model not trained, call fit first");
        }
        
        int originalCols = feature.getColNum();
        
        if (originalCols != this.feature.getColNum()) {
            throw new IllegalArgumentException("特征维度不匹配 / Feature dimension mismatch");
        }
        
        // 如果目标维度等于原始维度，直接返回原数据的副本
        if (nComponents == originalCols) {
            return IMatrix.of(feature.toDoubleArray());
        }
        
        // 执行SVD分解 / Perform SVD decomposition
        // A = U * S * V^T
        Tuple3<?, ?, ?> svdResult = feature.svd();
        IMatrix U = (IMatrix)svdResult._1;
        IMatrix VT = (IMatrix)svdResult._3;
        
        // 从VT中提取前nComponents个主成分
        IMatrix projectionMatrix = (IMatrix)VT.slice(0, nComponents, 0, originalCols).transposeNew();
        
        // 将原始数据投影到低维空间
        IMatrix result = (IMatrix)feature.mmul(projectionMatrix);
        
        return result;
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    /**
     * 设置目标维度
     * @param nComponents 目标维度
     * @return 当前实例
     */
    public RereSVD setNComponents(int nComponents) {
        this.nComponents = nComponents;
        return this;
    }

    /**
     * 获取目标维度
     * @return 目标维度
     */
    public int getNComponents() {
        return nComponents;
    }

    /**
     * 用SVD方法降维（便捷方法）
     * @param originalData 原数据矩阵
     * @param dim 目标维度
     * @return 降维后的矩阵
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim){
        return setNComponents(dim).fit(originalData).transform(originalData);
    }

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("nComponents", nComponents);
        return p;
    }

    @Override
    public void fromParams(Map<String, Object> p) {
        this.nComponents = ((Number) p.get("nComponents")).intValue();
    }

}