package com.yishape.lab.math.ml;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.clu.ClusteringMetrics;
import com.yishape.lab.math.ml.clu.GMMClustering;
import com.yishape.lab.math.ml.clu.IClustering;
import com.yishape.lab.math.ml.clu.KMeansPlusPlus;

import com.yishape.lab.util.YishapeLogger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * 聚类算法工厂与评估入口，API 设计与 {@link ClfWrapper} 一致。
 *
 * @author lteb2
 */
public class CluWrapper {

    private static final YishapeLogger log = YishapeLogger.getLogger(CluWrapper.class);

    /**
     * 创建默认 K-means++（等价于 {@code new KMeansPlusPlus()}）。
     *
     * @return 默认聚类数等资源由实现内部设定
     * @see KMeansPlusPlus#KMeansPlusPlus()
     */
    public IClustering kMeans() {
        return new KMeansPlusPlus();
    }

    /**
     * K-means++，指定随机种子。
     *
     * @param randomSeed 随机种子
     * @return K-means++ 实例
     * @see KMeansPlusPlus#KMeansPlusPlus(long)
     */
    public IClustering kMeans(long randomSeed) {
        return new KMeansPlusPlus(randomSeed);
    }

    /**
     * K-means++，指定簇数 {@code numClusters}。
     *
     * @param numClusters 簇数（须为正）
     * @return K-means++ 实例
     * @see KMeansPlusPlus#KMeansPlusPlus(int)
     */
    public IClustering kMeans(int numClusters) {
        return new KMeansPlusPlus(numClusters);
    }

    /**
     * K-means++，指定簇数与随机种子。
     *
     * @param numClusters 簇数
     * @param randomSeed  随机种子
     * @return K-means++ 实例
     * @see KMeansPlusPlus#KMeansPlusPlus(int, long)
     */
    public IClustering kMeans(int numClusters, long randomSeed) {
        return new KMeansPlusPlus(numClusters, randomSeed);
    }

    /**
     * K-means++，完全指定随机源、最大迭代与收敛阈值。
     *
     * @param random                 随机数生成器
     * @param maxIterations          最大迭代次数
     * @param convergenceThreshold   收敛阈值
     * @return K-means++ 实例
     * @see KMeansPlusPlus#KMeansPlusPlus(int, Random, int, double)
     */
    public IClustering kMeans(Random random, int maxIterations, double convergenceThreshold) {
        return new KMeansPlusPlus(KMeansPlusPlus.DEFAULT_K, random, maxIterations, convergenceThreshold);
    }

    /**
     * 高斯混合模型聚类（EM），默认构造。
     *
     * @return GMM 实例
     * @see GMMClustering#GMMClustering()
     */
    public IClustering gmm() {
        return new GMMClustering();
    }

    /**
     * GMM，指定簇数。
     *
     * @param numClusters 组分（簇）数
     * @return GMM 实例
     */
    public IClustering gmm(int numClusters) {
        return new GMMClustering(numClusters);
    }

    /**
     * GMM，指定 EM 迭代、容忍度、重启次数与初始化选项。
     *
     * @param maxIterations    最大迭代
     * @param tolerance        收敛容忍度
     * @param numRestarts      随机重启次数
     * @param useKMeansInit    是否用 K-means++ 初始化
     * @param randomSeed       随机种子
     * @param verbose          是否详细日志
     * @return GMM 实例
     * @see GMMClustering#GMMClustering(int, long, int, double, int, boolean, boolean)
     */
    public IClustering gmm(int maxIterations, double tolerance, int numRestarts,
            boolean useKMeansInit, long randomSeed, boolean verbose) {
        return new GMMClustering(GMMClustering.DEFAULT_K, randomSeed,
                maxIterations, tolerance, numRestarts, useKMeansInit, verbose);
    }

    /**
     * 在已拟合的聚类模型上计算质量指标（惯性、轮廓系数、CH、DB 等）。
     * 数据须与训练时顺序、规模一致，以便与内部标签对齐。
     *
     * @param model 已 {@link IClustering#fit} 的模型
     * @param data  样本列表
     * @return 聚类指标
     * @see ClusteringMetrics
     */
    public static ClusteringMetrics clusteringMetrics(IClustering model, List<IVector<Double>> data) {
        return model.evaluateQuality(data);
    }

    /**
     * 矩阵形式 {@link #clusteringMetrics(IClustering, List)}：每行一个样本。
     *
     * @param model 已拟合模型
     * @param data  n×d 特征矩阵
     * @return 聚类指标
     */
    public static ClusteringMetrics clusteringMetrics(IClustering model, IMatrix<Double> data) {
        if (data == null || data.getRowNum() == 0) {
            throw new IllegalArgumentException("数据矩阵不能为空");
        }
        List<IVector<Double>> rows = new ArrayList<>(data.getRowNum());
        for (int i = 0; i < data.getRowNum(); i++) {
            rows.add(data.getRow(i));
        }
        return model.evaluateQuality(rows);
    }

    /**
     * 从本地序列化文件加载聚类模型。
     *
     * @param modelPath 模型路径
     * @return 聚类模型
     * @throws IllegalStateException 加载失败
     */
    public static IClustering loadClustering(String modelPath) {
        try {
            ISerializableModel model = ISerializableModel.load(modelPath);
            return (IClustering) model;
        } catch (Exception e) {
            log.error("Failed to load clustering model from: {}", modelPath, e);
            throw new IllegalStateException("Failed to load clustering model from: " + modelPath, e);
        }
    }

    /**
     * 将聚类模型保存到本地路径。
     *
     * @param clustering 聚类模型
     * @param modelPath  保存路径
     */
    public static void saveClustering(IClustering clustering, String modelPath) {
        clustering.save(modelPath);
    }
}
