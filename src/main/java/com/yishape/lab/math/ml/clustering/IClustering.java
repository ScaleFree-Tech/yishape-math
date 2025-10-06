package com.yishape.lab.math.ml.clustering;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;

import java.util.List;

/**
 * 聚类算法通用接口
 * Generic interface for clustering algorithms
 * 
 * 定义了所有聚类算法应该实现的标准方法，采用类似Scikit-learn的API设计
 * Defines standard methods that all clustering algorithms should implement, using Scikit-learn-like API design
 * 
 * @author reremouse
 */
public interface IClustering extends ISerializableModel{
    
    /**
     * 训练聚类模型
     * Fit the clustering model
     * 
     * @param data 输入数据向量列表 / Input data vector list
     * @return 当前实例，支持链式调用 / Current instance for method chaining
     */
    IClustering fit(List<IVector<Double>> data);
    
    /**
     * 训练聚类模型
     * Fit the clustering model
     * 
     * @param data 输入数据矩阵 (n x d) / Input data matrix (n x d)
     * @return 当前实例，支持链式调用 / Current instance for method chaining
     */
    IClustering fit(IMatrix<Double> data);
    
    /**
     * 训练聚类模型并返回聚类标签
     * Fit the clustering model and return cluster labels
     * 
     * @param data 输入数据向量列表 / Input data vector list
     * @return 聚类标签数组 / Array of cluster labels
     */
    int[] fitPredict(List<IVector<Double>> data);
    
    /**
     * 训练聚类模型并返回聚类标签
     * Fit the clustering model and return cluster labels
     * 
     * @param data 输入数据矩阵 (n x d) / Input data matrix (n x d)
     * @return 聚类标签数组 / Array of cluster labels
     */
    int[] fitPredict(IMatrix<Double> data);
    
    /**
     * 预测新数据点的聚类标签
     * Predict cluster labels for new data points
     * 
     * @param data 新数据点 / New data points
     * @return 聚类标签数组 / Array of cluster labels
     */
    int[] predict(List<IVector<Double>> data);
    
    /**
     * 预测单个数据点的聚类标签
     * Predict cluster label for a single data point
     * 
     * @param point 数据点 / Data point
     * @return 聚类标签 / Cluster label
     */
    int predict(IVector<Double> point);
    
    /**
     * 获取聚类中心
     * Get cluster centers
     * 
     * @return 聚类中心列表 / List of cluster centers
     */
    List<IVector<Double>> getClusterCenters();
    
    /**
     * 获取聚类标签
     * Get cluster labels for training data
     * 
     * @return 聚类标签数组 / Array of cluster labels
     */
    int[] getLabels();
    
    /**
     * 获取聚类数量
     * Get number of clusters
     * 
     * @return 聚类数量 / Number of clusters
     */
    int getNumClusters();
    
    /**
     * 获取数据维度
     * Get data dimensionality
     * 
     * @return 数据维度 / Data dimensionality
     */
    int getDimension();
    
    /**
     * 获取聚类惯性（质量指标）
     * Get clustering inertia (quality metric)
     * 
     * @return 聚类惯性 / Clustering inertia
     */
    double getInertia();
    
    /**
     * 是否收敛
     * Whether converged
     * 
     * @return 是否收敛 / Whether converged
     */
    boolean isConverged();
    
    /**
     * 获取迭代次数
     * Get number of iterations
     * 
     * @return 迭代次数 / Number of iterations
     */
    int getIterations();
    
    /**
     * 计算聚类质量评估指标
     * Compute clustering quality metrics
     * 
     * @param data 原始数据 / Original data
     * @return 评估指标 / Evaluation metrics
     */
    ClusteringMetrics evaluateQuality(List<IVector<Double>> data);
    
    /**
     * 获取算法名称
     * Get algorithm name
     * 
     * @return 算法名称 / Algorithm name
     */
    String getAlgorithmName();
    
    /**
     * 设置算法参数
     * Set algorithm parameters
     * 
     * @param parameters 参数映射 / Parameter map
     */
    void setParameters(java.util.Map<String, Object> parameters);
    
    /**
     * 获取算法参数
     * Get algorithm parameters
     * 
     * @return 参数映射 / Parameter map
     */
    java.util.Map<String, Object> getParameters();
    
}