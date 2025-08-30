package com.reremouse.lab.math.dimreduce;

import java.util.*;
import java.util.stream.IntStream;
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;

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
public class RereUMAP {
    
    // UMAP算法超参数 / UMAP algorithm hyperparameters
    private final int nNeighbors = 15;        // k近邻数量 / Number of k-nearest neighbors
    private final float minDist = 0.1f;       // 最小距离参数 / Minimum distance parameter
    private final int nEpochs = 500;          // 最大迭代次数 / Maximum iterations
    private final float learningRate = 1.0f;  // 学习率 / Learning rate
    private final float spread = 1.0f;        // 散布参数 / Spread parameter
    private final float localConnectivity = 1.0f; // 局部连通性 / Local connectivity
    private final float repulsionStrength = 1.0f; // 排斥强度 / Repulsion strength
    private final int negativeSampleRate = 5; // 负样本采样率 / Negative sampling rate
    private final float initialAlpha = 1.0f;  // 初始学习率 / Initial learning rate
    private final Random random = new Random();
    
    /**
     * 用UMAP方法降维
     * @param originalData 原数据
     * @param dim 目标维度，即列数
     * @return 降维后的矩阵
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim){
        System.out.println("开始UMAP降维，数据形状: " + Arrays.toString(originalData.shape()) + 
                          "，目标维度: " + dim);
        
        int n = originalData.getRowNum();
        
        // 第一步：构建k近邻图
        System.out.println("步骤1: 构建k近邻图...");
        int[][] knnIndices = computeKNearestNeighbors(originalData);
        float[][] knnDistances = computeKNNDistances(originalData, knnIndices);
        
        // 第二步：计算流形结构（fuzzy simplicial complex）
        System.out.println("步骤2: 计算流形结构...");
        IMatrix weights = computeFuzzySimplicialComplex(knnIndices, knnDistances, n);
        
        // 第三步：初始化低维嵌入
        System.out.println("步骤3: 初始化低维嵌入...");
        IMatrix embedding = initializeEmbedding(n, dim);
        
        // 第四步：优化低维嵌入
        System.out.println("步骤4: 优化低维嵌入...");
        embedding = optimizeEmbedding(weights, embedding, knnIndices);
        
        System.out.println("UMAP降维完成");
        return embedding;
    }
    
    /**
     * 计算k近邻
     */
    private int[][] computeKNearestNeighbors(IMatrix data) {
        int n = data.getRowNum();
        int[][] knnIndices = new int[n][nNeighbors];
        
        for (int i = 0; i < n; i++) {
            IVector query = data.getRow(i);
            
            // 计算到所有其他点的距离
            List<DistanceIndex> distances = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector target = data.getRow(j);
                    float distance = computeEuclideanDistance(query, target);
                    distances.add(new DistanceIndex(j, distance));
                }
            }
            
            // 排序并选择k个最近邻
            distances.sort(Comparator.comparingDouble(d -> d.distance));
            for (int k = 0; k < Math.min(nNeighbors, distances.size()); k++) {
                knnIndices[i][k] = distances.get(k).index;
            }
        }
        
        return knnIndices;
    }
    
    /**
     * 计算k近邻距离
     */
    private float[][] computeKNNDistances(IMatrix data, int[][] knnIndices) {
        int n = data.getRowNum();
        float[][] distances = new float[n][nNeighbors];
        
        for (int i = 0; i < n; i++) {
            IVector query = data.getRow(i);
            for (int k = 0; k < nNeighbors && k < knnIndices[i].length; k++) {
                if (knnIndices[i][k] >= 0) {
                    IVector neighbor = data.getRow(knnIndices[i][k]);
                    distances[i][k] = computeEuclideanDistance(query, neighbor);
                }
            }
        }
        
        return distances;
    }
    
    /**
     * 计算模糊单纯复形（Fuzzy Simplicial Complex）
     */
    private IMatrix computeFuzzySimplicialComplex(int[][] knnIndices, float[][] knnDistances, int n) {
        IMatrix weights = IMatrix.zeros(n, n);
        
        // 计算每个点的局部连通性半径
        float[] sigmas = new float[n];
        float[] rhos = new float[n];
        
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
                    float distance = knnDistances[i][k];
                    
                    // 计算概率权重
                    float weight = computeWeight(distance, rhos[i], sigmas[i]);
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
    private float findOptimalSigma(float[] distances, float rho) {
        float target = (float) Math.log(2.0); // 目标困惑度对应的熵
        float sigmaMin = 1e-20f;
        float sigmaMax = 1000.0f;
        float sigma = 1.0f;
        
        for (int iter = 0; iter < 64; iter++) {
            float entropy = computeLocalEntropy(distances, rho, sigma);
            float diff = entropy - target;
            
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
    private float computeLocalEntropy(float[] distances, float rho, float sigma) {
        float entropy = 0.0f;
        float sum = 0.0f;
        
        for (float distance : distances) {
            if (distance > 0) {
                float adjustedDist = Math.max(0, distance - rho);
                float prob = (float) Math.exp(-adjustedDist / sigma);
                sum += prob;
            }
        }
        
        if (sum > 0) {
            for (float distance : distances) {
                if (distance > 0) {
                    float adjustedDist = Math.max(0, distance - rho);
                    float prob = (float) Math.exp(-adjustedDist / sigma) / sum;
                    if (prob > 1e-12) {
                        entropy -= prob * Math.log(prob);
                    }
                }
            }
        }
        
        return entropy;
    }
    
    /**
     * 计算权重
     */
    private float computeWeight(float distance, float rho, float sigma) {
        float adjustedDist = Math.max(0, distance - rho);
        return (float) Math.exp(-adjustedDist / sigma);
    }
    
    /**
     * 对称化权重矩阵
     */
    private IMatrix symmetrizeWeights(IMatrix weights) {
        int n = weights.getRowNum();
        IMatrix symmetric = IMatrix.zeros(n, n);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float wij = weights.get(i, j);
                float wji = weights.get(j, i);
                // 使用模糊并集操作: a + b - a*b
                float combined = wij + wji - wij * wji;
                symmetric.put(i, j, combined);
            }
        }
        
        return symmetric;
    }
    
    /**
     * 初始化低维嵌入
     */
    private IMatrix initializeEmbedding(int n, int dim) {
        float[][] data = new float[n][dim];
        
        // 使用随机初始化
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                data[i][j] = (float) (random.nextGaussian() * 10.0);
            }
        }
        
        return IMatrix.of(data);
    }
    
    /**
     * 优化低维嵌入
     */
    private IMatrix optimizeEmbedding(IMatrix weights, IMatrix embedding, int[][] knnIndices) {
        int n = embedding.getRowNum();
        int dim = embedding.getColNum();
        
        // 准备边列表
        List<Edge> edges = prepareEdges(weights, knnIndices);
        
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            float alpha = initialAlpha * (1.0f - (float)epoch / nEpochs);
            
            // 处理所有边
            for (Edge edge : edges) {
                // 正样本梯度更新
                updatePositiveGradient(embedding, edge, alpha);
                
                // 负样本梯度更新
                for (int neg = 0; neg < negativeSampleRate; neg++) {
                    int negativeIndex = random.nextInt(n);
                    if (negativeIndex != edge.i && negativeIndex != edge.j) {
                        updateNegativeGradient(embedding, edge.i, negativeIndex, alpha);
                    }
                }
            }
            
            // 每50个epoch输出进度
            if (epoch % 50 == 0) {
                System.out.println("UMAP优化进度: " + epoch + "/" + nEpochs);
            }
        }
        
        return embedding;
    }
    
    /**
     * 准备边列表
     */
    private List<Edge> prepareEdges(IMatrix weights, int[][] knnIndices) {
        List<Edge> edges = new ArrayList<>();
        int n = weights.getRowNum();
        
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < nNeighbors && k < knnIndices[i].length; k++) {
                if (knnIndices[i][k] >= 0) {
                    int j = knnIndices[i][k];
                    float weight = weights.get(i, j);
                    if (weight > 0) {
                        edges.add(new Edge(i, j, weight));
                    }
                }
            }
        }
        
        return edges;
    }
    
    /**
     * 更新正样本梯度
     */
    private void updatePositiveGradient(IMatrix embedding, Edge edge, float alpha) {
        IVector yi = embedding.getRow(edge.i);
        IVector yj = embedding.getRow(edge.j);
        
        float distance = computeEuclideanDistance(yi, yj);
        
        if (distance > 0) {
            // UMAP的低维相似性函数: 1 / (1 + a * d^(2b))
            float a = 1.929f; // 这些参数可以从minDist和spread计算得出
            float b = 0.7915f;
            
            float powered_distance = (float) Math.pow(distance, 2 * b);
            float similarity = 1.0f / (1.0f + a * powered_distance);
            
            // 计算梯度
            float grad_coeff = edge.weight * 2 * a * b * powered_distance * similarity / 
                              (distance * (1.0f + a * powered_distance));
            
            // 更新嵌入
            for (int d = 0; d < embedding.getColNum(); d++) {
                float grad = grad_coeff * (yi.get(d) - yj.get(d));
                float newYi = yi.get(d) + alpha * grad;
                float newYj = yj.get(d) - alpha * grad;
                
                embedding.put(edge.i, d, newYi);
                embedding.put(edge.j, d, newYj);
            }
        }
    }
    
    /**
     * 更新负样本梯度
     */
    private void updateNegativeGradient(IMatrix embedding, int i, int neg, float alpha) {
        IVector yi = embedding.getRow(i);
        IVector yneg = embedding.getRow(neg);
        
        float distance = computeEuclideanDistance(yi, yneg);
        
        if (distance > 0) {
            float a = 1.929f;
            float b = 0.7915f;
            
            float powered_distance = (float) Math.pow(distance, 2 * b);
            float similarity = 1.0f / (1.0f + a * powered_distance);
            
            // 负样本的梯度
            float grad_coeff = repulsionStrength * 2 * a * b * powered_distance * 
                              similarity * similarity / distance;
            
            // 更新嵌入（排斥）
            for (int d = 0; d < embedding.getColNum(); d++) {
                float grad = grad_coeff * (yi.get(d) - yneg.get(d));
                float newYi = yi.get(d) + alpha * grad;
                float newYneg = yneg.get(d) - alpha * grad;
                
                embedding.put(i, d, newYi);
                embedding.put(neg, d, newYneg);
            }
        }
    }
    
    /**
     * 计算欧几里得距离
     */
    private float computeEuclideanDistance(IVector v1, IVector v2) {
        return v1.euclideanDistance(v2);
        
    }
    
    /**
     * 距离-索引对辅助类
     */
    private static class DistanceIndex {
        int index;
        float distance;
        
        DistanceIndex(int index, float distance) {
            this.index = index;
            this.distance = distance;
        }
    }
    
    /**
     * 边辅助类
     */
    private static class Edge {
        int i, j;
        float weight;
        
        Edge(int i, int j, float weight) {
            this.i = i;
            this.j = j;
            this.weight = weight;
        }
    }
}
