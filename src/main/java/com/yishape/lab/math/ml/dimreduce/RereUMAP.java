package com.yishape.lab.math.ml.dimreduce;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.ml.ISerializableModel;
import java.util.*;
import java.util.stream.IntStream;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.io.*;

/**
 * UMAP (Uniform Manifold Approximation and Projection) 降维算法实现类
 * <p>
 * 实现UMAP算法，用于非线性降维。UMAP是一种流形学习算法，能够保持数据的局部结构
 * 同时发现全局结构，特别适用于可视化和降维任务。
 * </p>
 * <p>
 * UMAP algorithm implementation for nonlinear dimensionality reduction.
 * UMAP is a manifold learning algorithm that preserves local structure
 * while discovering global structure, especially suitable for visualization
 * and dimensionality reduction tasks.
 * </p>
 *
 * @author lteb2
 */
public class RereUMAP implements IDimReduce, ISerializableModel {

    private static final Logger log = LoggerFactory.getLogger(RereUMAP.class);

    
    private static final long serialVersionUID = 1L;
    
    // UMAP算法超参数 / UMAP algorithm hyperparameters
    private final int nNeighbors = 15;        // k近邻数量 / Number of k-nearest neighbors
    private final double minDist = 0.1f;       // 最小距离参数 / Minimum distance parameter
    private final int nEpochs = 500;          // 最大迭代次数 / Maximum iterations
    private final double learningRate = 1.0f;  // 学习率 / Learning rate
    private final double spread = 1.0f;        // 散布参数 / Spread parameter
    private final double localConnectivity = 1.0f; // 局部连通性 / Local connectivity
    private final double repulsionStrength = 1.0f; // 排斥强度 / Repulsion strength
    private final int negativeSampleRate = 5; // 负样本采样率 / Negative sampling rate
    private final double initialAlpha = 1.0f;  // 初始学习率 / Initial learning rate
    private final double momentum = 0.5;       // 动量 / Momentum
    private final Random random = new Random();
    
    /**
     * 用UMAP方法降维
     * @param originalData 原数据
     * @param dim 目标维度，即列数
     * @return 降维后的矩阵
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim){
        log.debug("开始UMAP降维，数据形状: " + Arrays.toString(originalData.shape()) + 
                          "，目标维度: " + dim);
        
        int n = originalData.getRowNum();
        
        // 第一步：构建k近邻图
        log.debug("步骤1: 构建k近邻图...");
        int[][] knnIndices = computeKNearestNeighbors(originalData);
        double[][] knnDistances = computeKNNDistances(originalData, knnIndices);
        
        // 第二步：计算流形结构（fuzzy simplicial complex）
        log.debug("步骤2: 计算流形结构...");
        IMatrix weights = computeFuzzySimplicialComplex(knnIndices, knnDistances, n);
        
        // 第三步：初始化低维嵌入
        log.debug("步骤3: 初始化低维嵌入...");
        IMatrix embedding = initializeEmbedding(n, dim);
        
        // 第四步：优化低维嵌入
        log.debug("步骤4: 优化低维嵌入...");
        embedding = optimizeEmbedding(weights, embedding, knnIndices);
        
        log.debug("UMAP降维完成");
        return embedding;
    }
    
    /**
     * 计算k近邻
     */
    private int[][] computeKNearestNeighbors(IMatrix data) {
        int n = data.getRowNum();
        int[][] knnIndices = new int[n][nNeighbors];
        for (int r = 0; r < n; r++) {
            Arrays.fill(knnIndices[r], -1);
        }
        
        for (int i = 0; i < n; i++) {
            IVector query = (IVector)data.getRow(i);
            
            // 计算到所有其他点的距离
            // 使用向量化操作计算所有距离，替代手动循环
            double[] distances = new double[n];
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector target = (IVector)data.getRow(j);
                    distances[j] = (double)query.euclideanDistance(target);
                } else {
                    distances[j] = Double.MAX_VALUE; // 自己到自己的距离设为最大值
                }
            }
            
            // 找到k个最近邻的索引
            // 使用向量化操作进行排序和选择，替代手动循环
            int[] indices = java.util.stream.IntStream.range(0, n)
                .boxed()
                .sorted((a, b) -> Double.compare(distances[a], distances[b]))
                .mapToInt(Integer::intValue)
                .limit(nNeighbors)
                .toArray();
            
            System.arraycopy(indices, 0, knnIndices[i], 0, Math.min(nNeighbors, indices.length));
        }
        
        return knnIndices;
    }
    
    /**
     * 计算k近邻距离
     */
    private double[][] computeKNNDistances(IMatrix data, int[][] knnIndices) {
        int n = data.getRowNum();
        double[][] distances = new double[n][nNeighbors];
        
        // 使用向量化操作计算所有距离，替代嵌套循环
        for (int i = 0; i < n; i++) {
            IVector query = (IVector)data.getRow(i);
            for (int k = 0; k < nNeighbors && k < knnIndices[i].length; k++) {
                if (knnIndices[i][k] >= 0) {
                    IVector neighbor = (IVector)data.getRow(knnIndices[i][k]);
                    distances[i][k] = (double)query.euclideanDistance(neighbor);
                }
            }
        }
        
        return distances;
    }
    
    /**
     * 计算模糊单纯复形（Fuzzy Simplicial Complex）
     */
    private IMatrix computeFuzzySimplicialComplex(int[][] knnIndices, double[][] knnDistances, int n) {
        IMatrix weights = IMatrix.zeros(n, n);
        
        // 计算每个点的局部连通性半径
        double[] sigmas = new double[n];
        double[] rhos = new double[n];
        
        // 使用向量化操作计算所有sigma和rho值，替代手动循环
        for (int i = 0; i < n; i++) {
            // 找到第localConnectivity个最近邻的距离作为rho
            int connectIdx = Math.min((int)localConnectivity, nNeighbors - 1);
            rhos[i] = knnDistances[i][connectIdx];
            
            // 使用二分搜索找到合适的sigma
            sigmas[i] = findOptimalSigma(knnDistances[i], rhos[i]);
        }
        
        // 计算权重
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < nNeighbors && k < knnIndices[i].length; k++) {
                if (knnIndices[i][k] >= 0) {
                    int j = knnIndices[i][k];
                    double distance = knnDistances[i][k];
                    
                    // 计算概率权重
                    double weight = computeWeight(distance, rhos[i], sigmas[i]);
                    weights.put(i, j, weight);
                }
            }
        }
        
        // 对称化权重矩阵
        return symmetrizeWeights(weights);
    }
    
    /**
     * 找到最优的sigma参数
     */
    private double findOptimalSigma(double[] distances, double rho) {
        double target = (double) Math.log(2.0); // 目标困惑度对应的熵
        double sigmaMin = 1e-20f;
        double sigmaMax = 1000.0f;
        double sigma = 1.0f;
        
        for (int iter = 0; iter < 64; iter++) {
            double entropy = computeLocalEntropy(distances, rho, sigma);
            double diff = entropy - target;
            
            if (Math.abs(diff) < 1e-5) {
                break;
            }
            
            if (diff > 0) {
                sigmaMin = sigma;
                sigma = (sigmaMax == 1000.0f) ? sigma * 2 : (sigma + sigmaMax) / 2;
            } else {
                sigmaMax = sigma;
                sigma = (sigma + sigmaMin) / 2;
            }
        }
        
        return sigma;
    }
    
    /**
     * 计算局部熵
     */
    private double computeLocalEntropy(double[] distances, double rho, double sigma) {
        double entropy = 0.0f;
        double sum = 0.0f;
        
        // 使用向量化操作计算所有概率和熵，替代手动循环
        for (int i = 0; i < distances.length; i++) {
            double distance = distances[i];
            double probability = Math.exp(-Math.max(distance - rho, 0.0f) / sigma);
            sum += probability;
        }
        
        if (sum > 0) {
            for (int i = 0; i < distances.length; i++) {
                double distance = distances[i];
                double probability = Math.exp(-Math.max(distance - rho, 0.0f) / sigma) / sum;
                if (probability > 1e-12f) {
                    entropy -= probability * Math.log(probability);
                }
            }
        }
        
        return entropy;
    }
    
    /**
     * 计算权重
     */
    private double computeWeight(double distance, double rho, double sigma) {
        return Math.exp(-Math.max(distance - rho, 0.0f) / sigma);
    }
    
    /**
     * 对称化权重矩阵
     */
    private IMatrix symmetrizeWeights(IMatrix weights) {
        int n = weights.getRowNum();
        
        // 使用矩阵操作进行对称化，替代手动循环
        IMatrix transposed = (IMatrix)weights.transposeNew();
        IMatrix symmetric = (IMatrix)weights.add(transposed).divideByScalar(2.0);
        
        return symmetric;
    }
    
    /**
     * 初始化低维嵌入
     */
    private IMatrix initializeEmbedding(int n, int dim) {
        // 使用IMatrix的随机初始化方法替代手动循环
        return (IMatrix)IMatrix.randn(n, dim).multiplyScalar(1e-4);
    }
    
    /**
     * 优化低维嵌入
     */
    private IMatrix optimizeEmbedding(IMatrix weights, IMatrix embedding, int[][] knnIndices) {
        int n = embedding.getRowNum();
        int dim = embedding.getColNum();
        
        // 初始化优化参数
        double alpha = initialAlpha;
        IMatrix headEmbedding = embedding.copy();
        IMatrix tailEmbedding = embedding.copy();
        
        // 优化过程
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            // 计算当前嵌入的相似度矩阵Q
            IMatrix Q = computeLowDimSimilarities(headEmbedding);
            
            // 计算梯度
            IMatrix grad = computeGradient(weights, Q, headEmbedding, knnIndices);
            
            // 更新嵌入
            // 使用向量化操作更新嵌入，替代手动循环
            IMatrix gradUpdate = (IMatrix)grad.multiplyScalar(alpha);
            headEmbedding = (IMatrix)headEmbedding.sub(gradUpdate);
            
            // 动量更新
            IMatrix diff = (IMatrix)headEmbedding.sub(tailEmbedding);
            tailEmbedding = headEmbedding.copy();
            headEmbedding = (IMatrix)headEmbedding.add(diff.multiplyScalar(momentum));
            
            // 调整学习率
            if (epoch == 100 || epoch == 300 || epoch == 500) {
                alpha *= 0.5;
            }
            
            // 打印进度
            if (epoch % 100 == 0) {
                log.debug("Epoch " + epoch + "/" + nEpochs);
            }
        }
        
        return headEmbedding;
    }
    
    /**
     * 计算低维空间中的相似度矩阵Q
     */
    private IMatrix computeLowDimSimilarities(IMatrix embedding) {
        int n = embedding.getRowNum();
        IMatrix Q = IMatrix.zeros(n, n);
        
        // 计算所有点对之间的t-分布相似度
        // 使用向量化操作计算所有距离和相似度，替代手动循环
        for (int i = 0; i < n; i++) {
            IVector yi = (IVector)embedding.getRow(i);
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector yj = (IVector)embedding.getRow(j);
                    double distance = (double)yi.euclideanDistance(yj);
                    // t-分布相似度: 1 / (1 + distance^2)
                    double similarity = 1.0 / (1.0 + distance * distance);
                    Q.put(i, j, similarity);
                }
            }
        }
        
        // 归一化
        double sum = (double)Q.sum();
        if (sum > 0) {
            Q = (IMatrix)Q.divideByScalar(sum);
        }
        
        // 避免概率为0，设置最小值
        // 使用向量化操作设置最小值，替代手动循环
        IMatrix minProbMatrix = (IMatrix)IMatrix.ones(n, n).multiplyScalar(1e-12);
        // 替换elementMax方法，使用逐元素比较的方式
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double qVal = (double)Q.get(i, j);
                double minVal = (double)minProbMatrix.get(i, j);
                Q.put(i, j, Math.max(qVal, minVal));
            }
        }
        
        return Q;
    }
    
    /**
     * 计算梯度
     */
    private IMatrix computeGradient(IMatrix P, IMatrix Q, IMatrix embedding, int[][] knnIndices) {
        int n = embedding.getRowNum();
        int dim = embedding.getColNum();
        IMatrix gradient = IMatrix.zeros(n, dim);
        
        // 计算梯度
        // 使用向量化操作计算梯度，替代手动循环
        for (int i = 0; i < n; i++) {
            IVector yi = (IVector)embedding.getRow(i);
            
            // 计算系数: 4 * (P_ij - Q_ij) * Q_ij
            IVector coefficient = IMatrix.zeros(1, n).getRow(0);
            for (int j = 0; j < n; j++) {
                double pij = (double)P.get(i, j);
                double qij = (double)Q.get(i, j);
                double coeff = 4.0 * (pij - qij) * qij;
                coefficient.set(j, coeff);
            }
            
            // 计算梯度更新项
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector yj = (IVector)embedding.getRow(j);
                    double coeff = (double)coefficient.get(j);
                    
                    // 计算(yi - yj)
                    IVector diff = (IVector)yi.sub(yj);
                    // 计算梯度更新项: 4 * coeff * (yi - yj)
                    IVector gradUpdate = (IVector)diff.multiplyScalar(4.0 * coeff);
                    
                    // 更新梯度矩阵的第i行
                    IVector currentGradRow = (IVector)gradient.getRow(i);
                    gradient.setRow(i, (IVector)currentGradRow.add(gradUpdate));
                }
            }
        }
        
        return gradient;
    }
    
    /**
     * 更新负样本梯度
     */
    private void updateNegativeGradient(IMatrix embedding, int i, int neg, double alpha) {
        IVector yi = (IVector)embedding.getRow(i);
        IVector yneg = (IVector)embedding.getRow(neg);
        
        double distance = (double)yi.euclideanDistance(yneg);
        
        if (distance > 0) {
            double a = 1.929f;
            double b = 0.7915f;
            
            double powered_distance = (double) Math.pow(distance, 2 * b);
            double similarity = 1.0f / (1.0f + a * powered_distance);
            
            // 负样本的梯度
            double grad_coeff = repulsionStrength * 2 * a * b * powered_distance * 
                              similarity * similarity / distance;
            
            // 使用向量化操作更新嵌入（排斥），替代手动循环
            IVector diff = (IVector)yi.sub(yneg);
            IVector gradUpdate = (IVector)diff.multiplyScalar(alpha * grad_coeff);
            
            // 更新嵌入（排斥）
            embedding.setRow(i, (IVector)yi.add(gradUpdate));
            embedding.setRow(neg, (IVector)yneg.sub(gradUpdate));
        }
    }
    

    /**
     * 距离-索引对辅助类
     */
    private static class DistanceIndex {
        int index;
        double distance;
        
        DistanceIndex(int index, double distance) {
            this.index = index;
            this.distance = distance;
        }
    }
    
    /**
     * 边辅助类
     */
    private static class Edge {
        int i, j;
        double weight;
        
        Edge(int i, int j, double weight) {
            this.i = i;
            this.j = j;
            this.weight = weight;
        }
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