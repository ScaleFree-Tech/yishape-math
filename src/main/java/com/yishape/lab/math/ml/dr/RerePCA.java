package com.yishape.lab.math.ml.dr;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.preprocessing.ITransform;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.LinkedHashMap;
import java.util.Map;

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
 *   <li>数据中心化：减去训练集每列的均值 / Data centering: subtract training set column means</li>
 *   <li>计算协方差矩阵 / Compute covariance matrix</li>
 *   <li>对协方差矩阵进行特征分解 / Eigendecomposition of covariance matrix</li>
 *   <li>选择前k个最大特征值对应的特征向量 / Select k eigenvectors with largest eigenvalues</li>
 *   <li>将数据投影到主成分空间 / Project data to principal component space</li>
 * </ol>
 *
 * @author lteb2
 */
public class RerePCA implements ITransform<Double>, ISerializableModel{

    private static final YishapeLogger log = YishapeLogger.getLogger(RerePCA.class);

    private IMatrix<Double> feature;
    private int nComponents = -1;
    private IVector<Double> mean;
    private IMatrix<Double> principalComponents;

    /** 是否已完成拟合（已设置目标维度且已调用 {@link #fit}）。 */
    @Override
    public boolean ifTrained() {
        return nComponents > 0 && feature != null && mean != null && principalComponents != null;
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
        int nCols = feature.getColNum();

        this.mean = feature.colMeans();

        IMatrix centeredData = (IMatrix) feature.center();
        IMatrix covarianceMatrix = (IMatrix) centeredData.covarianceFromCentered();
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = covarianceMatrix.eigen();
        IMatrix eigenVectors = (IMatrix) eigenResult._2;
        this.principalComponents = (IMatrix) eigenVectors.slice(0, nCols, 0, Math.min(nComponents, nCols));

        return this;
    }

    /**
     * 用 PCA 将数据投影到主成分空间（使用训练集均值和主成分）。
     *
     * @param feature 待转换数据矩阵，每行一个样本、每列一个特征；列数须与拟合时一致
     * @return 降维后的矩阵，行数不变，列数为 {@link #getNComponents()}
     * @throws IllegalStateException 尚未调用 {@link #fit}
     * @throws IllegalArgumentException 特征列数与拟合时不一致
     */
    @Override
    public IMatrix transform(IMatrix feature) {
        if (!ifTrained()) {
            throw new IllegalStateException("模型尚未训练，请先调用fit / Model not trained, call fit first");
        }

        int originalCols = feature.getColNum();

        if (originalCols != this.feature.getColNum()) {
            throw new IllegalArgumentException("特征维度不匹配 / Feature dimension mismatch");
        }

        if (nComponents == originalCols) {
            return IMatrix.of(feature.toDoubleArray());
        }

        IMatrix centeredData = (IMatrix) feature.broadcastSubRow(this.mean);
        return (IMatrix) centeredData.mmul(this.principalComponents);
    }

    @Override
    public IMatrix getFeature() {
        return feature;
    }

    /**
     * 设置目标维度（主成分数量）
     * @param nComponents 目标维度
     * @return 当前实例
     */
    public RerePCA setNComponents(int nComponents) {
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
     * 用PCA方法降维（便捷方法，等同于先设置nComponents再fit+transform）
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
        if (mean != null) {
            double[] meanArr = new double[mean.size()];
            for (int i = 0; i < meanArr.length; i++) meanArr[i] = mean.get(i);
            p.put("mean", meanArr);
        }
        if (principalComponents != null) {
            p.put("components", principalComponents.toDoubleArray());
        }
        return p;
    }

    @Override
    public void fromParams(Map<String, Object> p) {
        this.nComponents = ((Number) p.get("nComponents")).intValue();
        double[] meanArr = (double[]) p.get("mean");
        if (meanArr != null) {
            this.mean = IVector.of(meanArr);
        }
        double[][] compArr = (double[][]) p.get("components");
        if (compArr != null) {
            this.principalComponents = IMatrix.of(compArr);
        }
        if (this.mean != null && this.principalComponents != null) {
            this.feature = this.principalComponents;
        }
    }

}
