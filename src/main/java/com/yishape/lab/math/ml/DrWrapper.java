package com.yishape.lab.math.ml;

import com.yishape.lab.math.ml.dr.RereFA;
import com.yishape.lab.math.ml.dr.RereKernelPCA;
import com.yishape.lab.math.ml.dr.RereNMF;
import com.yishape.lab.math.ml.dr.RerePCA;
import com.yishape.lab.math.ml.dr.RereSVD;
import com.yishape.lab.math.ml.dr.RereTSNE;
import com.yishape.lab.math.ml.dr.RereUMAP;
import com.yishape.lab.math.ml.preprocessing.ITransform;

/**
 * 降维算法工厂类 / Dimensionality Reduction Factory
 *
 * <h2>使用方式</h2>
 * <pre>{@code
 * // 方式1: 使用工厂方法（推荐）
 * IMatrix<Double> reduced = ML.dr.pca(2).fitTransform(data);
 *
 * // 方式2: 使用便捷方法
 * IMatrix<Double> reduced = ML.dr.pca(2).fitTransform(data);
 *
 * // 方式3: 链式调用
 * IMatrix<Double> reduced = ML.dr.pca().setNComponents(2).fit(data).transform(data);
 * }</pre>
 *
 * @author lteb2
 * @see ITransform
 */
public class DrWrapper {

    // ========== 降维 / Dimensionality reduction ==========

    /**
     * PCA 降维器。
     *
     * @param nComponents 目标维度（主成分数量）
     * @return 配置好的 PCA 实例
     */
    public ITransform<Double> pca(int nComponents) {
        return new RerePCA().setNComponents(nComponents);
    }

    /**
     * 基于 SVD 的线性降维器。
     *
     * @param nComponents 目标维度
     * @return 配置好的 SVD 实例
     */
    public ITransform<Double> svd(int nComponents) {
        return new RereSVD().setNComponents(nComponents);
    }

    /**
     * t-SNE 非线性降维器。
     *
     * @param nComponents 目标维度（通常为2-3用于可视化）
     * @return 配置好的 t-SNE 实例
     */
    public ITransform<Double> tsne(int nComponents) {
        return new RereTSNE().setNComponents(nComponents);
    }

    /**
     * UMAP 非线性降维器。
     *
     * @param nComponents 目标维度（通常为2-3用于可视化）
     * @return 配置好的 UMAP 实例
     */
    public ITransform<Double> umap(int nComponents) {
        return new RereUMAP().setNComponents(nComponents);
    }

    /**
     * 因子分析降维器。
     *
     * @param nFactors 因子数量
     * @return 配置好的 FA 实例
     */
    public ITransform<Double> fa(int nFactors) {
        return new RereFA().setNComponents(nFactors);
    }

    /**
     * 非负矩阵分解降维器。
     *
     * @param nComponents 成分数量
     * @return 配置好的 NMF 实例
     */
    public ITransform<Double> nmf(int nComponents) {
        return new RereNMF().setNComponents(nComponents);
    }

    /**
     * 核PCA降维器（默认RBF核）。
     *
     * @param nComponents 目标维度
     * @return 配置好的 KernelPCA 实例
     */
    public ITransform<Double> kernelPca(int nComponents) {
        return new RereKernelPCA().setNComponents(nComponents);
    }

    /**
     * 核PCA降维器（指定核类型）。
     *
     * @param kernel 核类型
     * @param nComponents 目标维度
     * @return 配置好的 KernelPCA 实例
     */
    public ITransform<Double> kernelPca(RereKernelPCA.KernelType kernel, int nComponents) {
        return new RereKernelPCA(kernel).setNComponents(nComponents);
    }

    /**
     * 核PCA降维器（指定核类型和gamma参数）。
     *
     * @param kernel 核类型
     * @param nComponents 目标维度
     * @param gamma gamma参数（RBF核使用 gamma = 1/(2*sigma^2)）
     * @return 配置好的 KernelPCA 实例
     */
    public ITransform<Double> kernelPca(RereKernelPCA.KernelType kernel, int nComponents, double gamma) {
        return new RereKernelPCA(kernel, nComponents, gamma);
    }
}
