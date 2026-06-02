package com.yishape.lab.math.ml.dr;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.preprocessing.ITransform;
import com.yishape.lab.util.Tuple2;

import com.yishape.lab.util.YishapeLogger;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Arrays;

/**
 * Kernel PCA (核主成分分析) 降维算法实现类
 * <p>
 * 实现基于核主成分分析（Kernel PCA）的降维算法。Kernel PCA是PCA的非线性扩展，
 * 通过核函数将数据映射到高维特征空间，然后在该空间中进行PCA。
 * 支持多种核函数：linear, rbf, poly, sigmoid, cosine。
 * </p>
 *
 * @author lteb2
 */
public class RereKernelPCA implements ITransform<Double>, ISerializableModel {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereKernelPCA.class);

    public enum KernelType {
        LINEAR,
        RBF,
        POLY,
        SIGMOID,
        COSINE
    }

    private KernelType kernel = KernelType.RBF;
    private int nComponents = -1;
    private double gamma = -1.0;
    private int degree = 3;
    private double coef0 = 1.0;
    private boolean removeZeroEig = false;

    private IMatrix<Double> XFit;
    private double[] eigenvalues;
    private double[][] eigenvectors;
    private double[][] scaledEigenvectors;
    private boolean fitted = false;
    private int nFeaturesIn;

    /** ITransform接口需要的字段 */
    private IMatrix<Double> feature;

    public RereKernelPCA() {
    }

    public RereKernelPCA(KernelType kernel) {
        this.kernel = kernel;
    }

    public RereKernelPCA(KernelType kernel, int nComponents) {
        this.kernel = kernel;
        this.nComponents = nComponents;
    }

    public RereKernelPCA(KernelType kernel, int nComponents, double gamma) {
        this.kernel = kernel;
        this.nComponents = nComponents;
        this.gamma = gamma;
    }

    @Override
    public boolean ifTrained() {
        return fitted && feature != null;
    }

    @Override
    @SuppressWarnings("unchecked")
    public ITransform<Double> fit(IMatrix feature) {
        if (feature == null) {
            throw new IllegalArgumentException("特征数据不能为空");
        }

        int nSamples = feature.getRowNum();
        int nFeatures = feature.getColNum();

        if (nSamples < 2) {
            throw new IllegalArgumentException("至少需要2个样本 / At least 2 samples required");
        }

        if (gamma <= 0) {
            gamma = 1.0 / nFeatures;
        }

        this.feature = feature;
        this.nFeaturesIn = nFeatures;
        this.XFit = (IMatrix<Double>) feature;

        int effectiveDim = (nComponents > 0) ? nComponents : nSamples;
        if (effectiveDim > nSamples) {
            effectiveDim = nSamples;
        }

        double[][] K = computeKernelMatrix((IMatrix)feature, (IMatrix)feature);
        double[][] Kc = centerKernelMatrix(K);

        IMatrix<Double> KcIMatrix = IMatrix.of(Kc);
        Tuple2<IVector<Double>, IMatrix<Double>> eigenResult = KcIMatrix.eigen();

        this.eigenvalues = new double[eigenResult._1.size()];
        for (int i = 0; i < eigenvalues.length; i++) {
            eigenvalues[i] = (Double) eigenResult._1.get(i);
        }

        this.eigenvectors = new double[eigenvalues.length][eigenvalues.length];
        for (int i = 0; i < eigenvalues.length; i++) {
            for (int j = 0; j < eigenvalues.length; j++) {
                eigenvectors[i][j] = (Double) eigenResult._2.get(i, j);
            }
        }

        Integer[] indices = new Integer[eigenvalues.length];
        for (int i = 0; i < eigenvalues.length; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(eigenvalues[b], eigenvalues[a]));

        double[] sortedEigenvalues = new double[eigenvalues.length];
        double[][] sortedEigenvectors = new double[eigenvectors.length][eigenvectors.length];
        for (int i = 0; i < eigenvalues.length; i++) {
            sortedEigenvalues[i] = eigenvalues[indices[i]];
            for (int j = 0; j < eigenvalues.length; j++) {
                sortedEigenvectors[j][i] = eigenvectors[j][indices[i]];
            }
        }
        this.eigenvalues = sortedEigenvalues;
        this.eigenvectors = sortedEigenvectors;

        int nonZeroCount = 0;
        for (double ev : eigenvalues) {
            if (ev > 0) nonZeroCount++;
        }
        int finalDim = Math.min(effectiveDim, nonZeroCount);
        if (finalDim < effectiveDim) {
            log.warn("有效特征值数量 ({}) 少于请求的维度 ({})", finalDim, effectiveDim);
        }

        this.scaledEigenvectors = new double[nSamples][finalDim];
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < finalDim; j++) {
                if (eigenvalues[j] > 0) {
                    scaledEigenvectors[i][j] = eigenvectors[i][j] / Math.sqrt(eigenvalues[j]);
                }
            }
        }

        this.fitted = true;
        return this;
    }

    @Override
    public IMatrix transform(IMatrix feature) {
        if (!ifTrained()) {
            throw new IllegalStateException("模型尚未训练，请先调用fit / Model not trained, call fit first");
        }

        int nSamples = feature.getRowNum();
        if (feature.getColNum() != nFeaturesIn) {
            throw new IllegalArgumentException("特征维度不匹配 / Feature dimension mismatch");
        }

        double[][] K = computeKernelMatrix((IMatrix)feature, XFit);
        double[][] Kc = centerKernelMatrix(K);

        int nComponents = scaledEigenvectors[0].length;
        int nFitSamples = XFit.getRowNum();
        double[][] result = new double[nSamples][nComponents];

        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nComponents; j++) {
                double sum = 0;
                for (int k = 0; k < nFitSamples; k++) {
                    sum += Kc[i][k] * scaledEigenvectors[k][j];
                }
                result[i][j] = sum;
            }
        }

        return IMatrix.of(result);
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
    public RereKernelPCA setNComponents(int nComponents) {
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
     * 用KernelPCA方法降维（便捷方法）
     * @param originalData 原数据矩阵
     * @param dim 目标维度
     * @return 降维后的矩阵
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim) {
        return setNComponents(dim).fit(originalData).transform(originalData);
    }

    private double[][] computeKernelMatrix(IMatrix X, IMatrix Y) {
        int nX = X.getRowNum();
        int nY = Y.getRowNum();
        double[][] K = new double[nX][nY];

        if (kernel == KernelType.LINEAR) {
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double sum = 0;
                    for (int k = 0; k < nFeaturesIn; k++) {
                        sum += (Double) X.get(i, k) * (Double) Y.get(j, k);
                    }
                    K[i][j] = sum;
                }
            }
        } else if (kernel == KernelType.RBF) {
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double sum = 0;
                    for (int k = 0; k < nFeaturesIn; k++) {
                        double diff = (Double) X.get(i, k) - (Double) Y.get(j, k);
                        sum += diff * diff;
                    }
                    K[i][j] = Math.exp(-gamma * sum);
                }
            }
        } else if (kernel == KernelType.POLY) {
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double sum = 0;
                    for (int k = 0; k < nFeaturesIn; k++) {
                        sum += (Double) X.get(i, k) * (Double) Y.get(j, k);
                    }
                    K[i][j] = Math.pow(gamma * sum + coef0, degree);
                }
            }
        } else if (kernel == KernelType.SIGMOID) {
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double sum = 0;
                    for (int k = 0; k < nFeaturesIn; k++) {
                        sum += (Double) X.get(i, k) * (Double) Y.get(j, k);
                    }
                    K[i][j] = Math.tanh(gamma * sum + coef0);
                }
            }
        } else if (kernel == KernelType.COSINE) {
            for (int i = 0; i < nX; i++) {
                for (int j = 0; j < nY; j++) {
                    double dot = 0, normX = 0, normY = 0;
                    for (int k = 0; k < nFeaturesIn; k++) {
                        dot += (Double) X.get(i, k) * (Double) Y.get(j, k);
                        normX += (Double) X.get(i, k) * (Double) X.get(i, k);
                        normY += (Double) Y.get(j, k) * (Double) Y.get(j, k);
                    }
                    double normProduct = Math.sqrt(normX) * Math.sqrt(normY);
                    K[i][j] = (normProduct > 0) ? dot / normProduct : 0;
                }
            }
        }

        return K;
    }

    private double[][] centerKernelMatrix(double[][] K) {
        int n = K.length;
        double[][] Kc = new double[n][n];

        double[] rowMeans = new double[n];
        double totalMean = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                rowMeans[i] += K[i][j];
            }
            rowMeans[i] /= n;
            totalMean += rowMeans[i];
        }
        totalMean /= n;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Kc[i][j] = K[i][j] - rowMeans[i] - rowMeans[j] + totalMean;
            }
        }

        return Kc;
    }

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("kernel", kernel.name());
        p.put("nComponents", nComponents);
        p.put("gamma", gamma);
        p.put("degree", degree);
        p.put("coef0", coef0);
        p.put("fitted", fitted);
        p.put("nFeaturesIn", nFeaturesIn);
        if (eigenvalues != null) p.put("eigenvalues", eigenvalues.clone());
        if (eigenvectors != null) p.put("eigenvectors", eigenvectors.clone());
        if (scaledEigenvectors != null) p.put("scaledEigenvectors", scaledEigenvectors.clone());
        return p;
    }

    @Override
    public void fromParams(Map<String, Object> p) {
        this.kernel = KernelType.valueOf((String) p.get("kernel"));
        this.nComponents = ((Number) p.get("nComponents")).intValue();
        this.gamma = ((Number) p.get("gamma")).doubleValue();
        this.degree = ((Number) p.get("degree")).intValue();
        this.coef0 = ((Number) p.get("coef0")).doubleValue();
        this.fitted = (Boolean) p.get("fitted");
        this.nFeaturesIn = ((Number) p.get("nFeaturesIn")).intValue();
        this.eigenvalues = (double[]) p.get("eigenvalues");
        this.eigenvectors = (double[][]) p.get("eigenvectors");
        this.scaledEigenvectors = (double[][]) p.get("scaledEigenvectors");
    }

    public KernelType getKernel() { return kernel; }
    public void setKernel(KernelType kernel) { this.kernel = kernel; }
    public double getGamma() { return gamma; }
    public void setGamma(double gamma) { this.gamma = gamma; }
    public int getDegree() { return degree; }
    public void setDegree(int degree) { this.degree = degree; }
    public double getCoef0() { return coef0; }
    public void setCoef0(double coef0) { this.coef0 = coef0; }
    public double[] getEigenvalues() { return eigenvalues; }
    public double[][] getEigenvectors() { return eigenvectors; }
    public boolean isFitted() { return fitted; }
}
