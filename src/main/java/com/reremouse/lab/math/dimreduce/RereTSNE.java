package com.reremouse.lab.math.dimreduce;

import java.util.Random;
import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;

/**
 * t-SNE降维算法实现类 / t-SNE Dimensionality Reduction Algorithm Implementation
 * <p>
 * 实现t-distributed Stochastic Neighbor Embedding (t-SNE)算法，用于非线性降维
 * Implements t-distributed Stochastic Neighbor Embedding (t-SNE) for nonlinear dimensionality reduction
 * </p>
 *
 * @author lteb2
 */
public class RereTSNE {
    
    // 算法超参数 / Algorithm hyperparameters
    private final double perplexity = 30.0;    // 困惑度 / Perplexity
    private final int maxIter = 1000;          // 最大迭代次数 / Maximum iterations
    private final double learningRate = 200.0; // 学习率 / Learning rate
    private final double momentum = 0.8;       // 动量 / Momentum
    private final double tolerance = 1e-4;     // 收敛阈值 / Convergence tolerance
    
    /**
     * 用t-SNE方法降维
     * @param originalData 原数据，每行为一个样本
     * @param dim 目标维度，即列数
     * @return 降维后的矩阵
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim) {
        if (originalData == null || dim <= 0) {
            throw new IllegalArgumentException("输入数据不能为空，目标维度必须大于0");
        }
        
        int n = originalData.getRowNum(); // 样本数量
        int originalDim = originalData.getColNum(); // 原始维度
        
        if (n < 2) {
            throw new IllegalArgumentException("样本数量必须至少为2");
        }
        
        System.out.println("开始t-SNE降维：样本数=" + n + "，原始维度=" + originalDim + "，目标维度=" + dim);
        
        // 步骤1：计算高维空间中的相似度矩阵P
        IMatrix P = computeHighDimSimilarities(originalData);
        
        // 步骤2：初始化低维嵌入Y
        IMatrix Y = initializeLowDimEmbedding(n, dim);
        
        // 步骤3：使用梯度下降优化Y
        Y = optimizeEmbedding(P, Y);
        
        System.out.println("t-SNE降维完成");
        return Y;
    }
    
    /**
     * 计算高维空间中的相似度矩阵P
     */
    private IMatrix computeHighDimSimilarities(IMatrix X) {
        int n = X.getRowNum();
        IMatrix P = IMatrix.zeros(n, n);
        
        // 为每个点寻找合适的sigma（方差）
        for (int i = 0; i < n; i++) {
            IVector xi = X.getRow(i);
            double sigma = findOptimalSigma(xi, X, i);
            
            // 计算第i行的概率分布
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector xj = X.getRow(j);
                    double distance = computeEuclideanDistance(xi, xj);
                    double prob = Math.exp(-distance * distance / (2 * sigma * sigma));
                    P.put(i, j, (float) prob);
                    sum += prob;
                }
            }
            
            // 归一化
            if (sum > 0) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        P.put(i, j, (float) (P.get(i, j) / sum));
                    }
                }
            }
        }
        
        // 对称化：P_ij = (P_ij + P_ji) / (2*n)
        IMatrix P_symmetric = IMatrix.zeros(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double symmetric_prob = (P.get(i, j) + P.get(j, i)) / (2.0 * n);
                    P_symmetric.put(i, j, (float) Math.max(symmetric_prob, 1e-12)); // 避免概率为0
                }
            }
        }
        
        return P_symmetric;
    }
    
    /**
     * 为给定点寻找最优的sigma（方差）参数
     */
    private double findOptimalSigma(IVector xi, IMatrix X, int i) {
        double sigmaMin = 1e-20;
        double sigmaMax = 1e20;
        double sigma = 1.0;
        double tolerance = 1e-5;
        int maxIterations = 50;
        
        for (int iter = 0; iter < maxIterations; iter++) {
            double entropy = computeEntropy(xi, X, i, sigma);
            double perplexityDiff = entropy - Math.log(perplexity);
            
            if (Math.abs(perplexityDiff) < tolerance) {
                break;
            }
            
            if (perplexityDiff > 0) {
                sigmaMin = sigma;
                sigma = (sigmaMax == 1e20) ? sigma * 2 : (sigma + sigmaMax) / 2;
            } else {
                sigmaMax = sigma;
                sigma = (sigma + sigmaMin) / 2;
            }
        }
        
        return sigma;
    }
    
    /**
     * 计算给定sigma下的熵（用于二分搜索最优sigma）
     */
    private double computeEntropy(IVector xi, IMatrix X, int i, double sigma) {
        double sum = 0.0;
        double entropy = 0.0;
        int n = X.getRowNum();
        
        for (int j = 0; j < n; j++) {
            if (i != j) {
                IVector xj = X.getRow(j);
                double distance = computeEuclideanDistance(xi, xj);
                double prob = Math.exp(-distance * distance / (2 * sigma * sigma));
                sum += prob;
            }
        }
        
        if (sum > 0) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector xj = X.getRow(j);
                    double distance = computeEuclideanDistance(xi, xj);
                    double prob = Math.exp(-distance * distance / (2 * sigma * sigma)) / sum;
                    if (prob > 1e-12) {
                        entropy -= prob * Math.log(prob);
                    }
                }
            }
        }
        
        return entropy;
    }
    
    /**
     * 计算两个向量之间的欧几里得距离
     */
    private double computeEuclideanDistance(IVector v1, IVector v2) {
        IVector diff = v1.sub(v2);
        return Math.sqrt(diff.innerProduct(diff));
    }
    
    /**
     * 初始化低维嵌入Y
     */
    private IMatrix initializeLowDimEmbedding(int n, int dim) {
        Random random = new Random();
        float[][] data = new float[n][dim];
        
        // 使用小的随机值初始化
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                data[i][j] = (float) (random.nextGaussian() * 1e-4);
            }
        }
        
        return IMatrix.of(data);
    }
    
    /**
     * 使用梯度下降优化低维嵌入Y
     */
    private IMatrix optimizeEmbedding(IMatrix P, IMatrix Y) {
        int n = Y.getRowNum();
        int dim = Y.getColNum();
        IMatrix velocity = IMatrix.zeros(n, dim); // 动量项
        
        for (int iter = 0; iter < maxIter; iter++) {
            // 计算低维相似度矩阵Q
            IMatrix Q = computeLowDimSimilarities(Y);
            
            // 计算梯度
            IMatrix gradient = computeGradient(P, Q, Y);
            
            // 更新速度（应用动量）
            velocity = velocity.mmul(momentum).sub(gradient.mmul(learningRate));
            
            // 更新Y
            Y = Y.add(velocity);
            
            // 每50次迭代输出一次进度
            if (iter % 50 == 0) {
                double cost = computeKLDivergence(P, Q);
                System.out.println("迭代 " + iter + "，KL散度: " + cost);
            }
            
            // 检查收敛性
            if (iter > 100) {
                IMatrix Q_prev = computeLowDimSimilarities(Y.sub(velocity));
                double cost_current = computeKLDivergence(P, Q);
                double cost_prev = computeKLDivergence(P, Q_prev);
                
                if (Math.abs(cost_current - cost_prev) < tolerance) {
                    System.out.println("收敛于迭代 " + iter);
                    break;
                }
            }
        }
        
        return Y;
    }
    
    /**
     * 计算低维空间中的相似度矩阵Q（使用t分布）
     */
    private IMatrix computeLowDimSimilarities(IMatrix Y) {
        int n = Y.getRowNum();
        IMatrix Q = IMatrix.zeros(n, n);
        double sum = 0.0;
        
        // 计算分子
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector yi = Y.getRow(i);
                    IVector yj = Y.getRow(j);
                    double distance = computeEuclideanDistance(yi, yj);
                    double similarity = 1.0 / (1.0 + distance * distance); // t分布
                    Q.put(i, j, (float) similarity);
                    sum += similarity;
                }
            }
        }
        
        // 归一化
        if (sum > 0) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        Q.put(i, j, (float) Math.max(Q.get(i, j) / sum, 1e-12));
                    }
                }
            }
        }
        
        return Q;
    }
    
    /**
     * 计算t-SNE的梯度
     */
    private IMatrix computeGradient(IMatrix P, IMatrix Q, IMatrix Y) {
        int n = Y.getRowNum();
        int dim = Y.getColNum();
        IMatrix gradient = IMatrix.zeros(n, dim);
        
        for (int i = 0; i < n; i++) {
            IVector yi = Y.getRow(i);
            
            for (int d = 0; d < dim; d++) {
                double gradSum = 0.0;
                
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        IVector yj = Y.getRow(j);
                        double distance_sq = computeEuclideanDistance(yi, yj);
                        distance_sq = distance_sq * distance_sq;
                        
                        double pij = P.get(i, j);
                        double qij = Q.get(i, j);
                        
                        double coefficient = (pij - qij) / (1.0 + distance_sq);
                        gradSum += coefficient * (yi.get(d) - yj.get(d));
                    }
                }
                
                gradient.put(i, d, (float) (4.0 * gradSum));
            }
        }
        
        return gradient;
    }
    
    /**
     * 计算KL散度作为损失函数
     */
    private double computeKLDivergence(IMatrix P, IMatrix Q) {
        int n = P.getRowNum();
        double kl = 0.0;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double pij = P.get(i, j);
                    double qij = Q.get(i, j);
                    
                    if (pij > 1e-12 && qij > 1e-12) {
                        kl += pij * Math.log(pij / qij);
                    }
                }
            }
        }
        
        return kl;
    }
}
