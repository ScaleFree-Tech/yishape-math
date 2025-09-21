package com.reremouse.lab.math.ml.dimreduce;

import java.util.Random;
import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;

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
            IVector xi = (IVector)X.getRow(i);
            double sigma = findOptimalSigma(xi, X, i);
            
            // 计算第i行的概率分布
            // 使用向量化操作计算所有距离和概率，替代手动循环
            IMatrix distances = computePairwiseDistances(X, xi); // 计算xi到所有点的距离
            // 使用pow方法来计算平方
            IMatrix squaredDistances = (IMatrix)distances.pow(2.0);
            IMatrix scaledDistances = (IMatrix)squaredDistances.multiplyScalar(-1.0 / (2 * sigma * sigma));
            IMatrix probs = (IMatrix)scaledDistances.exp();
            
            // 将对角线元素设为0（自己到自己的概率为0）
            probs.put(i, 0, 0.0);
            
            // 归一化
            double sum = (double)probs.sum();
            if (sum > 0) {
                // 使用正确的API方法
                IVector row = (IVector)probs.getRow(0);
                IVector normalizedRow = (IVector)row.divideByScalar(sum);
                P.setRow(i, normalizedRow);
            }
        }
        
        // 对称化：P_ij = (P_ij + P_ji) / (2*n)
        // 使用矩阵操作进行对称化，替代手动循环
        IMatrix P_transpose = (IMatrix)P.transposeNew();
        IMatrix P_symmetric = (IMatrix)P.add(P_transpose).divideByScalar(2.0 * n);
        
        // 避免概率为0，设置最小值
        // 使用向量化操作设置最小值，替代手动循环
        IMatrix minProbMatrix = (IMatrix)IMatrix.ones(n, n).multiplyScalar(1e-12);
        // 使用新的multiply方法进行元素级乘法
        IMatrix maxMatrix = (IMatrix)P_symmetric.multiply(minProbMatrix);
        P_symmetric = maxMatrix;
        
        // 确保对角线为0
        for (int i = 0; i < n; i++) {
            P_symmetric.put(i, i, 0.0);
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
        
        // 使用向量化操作进行二分搜索，替代手动循环
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
        int n = X.getRowNum();
        
        // 使用向量化操作计算所有概率，替代手动循环
        // 计算所有点到xi的距离
        IMatrix distances = computePairwiseDistances(X, xi);
        // 计算概率: exp(-distance^2 / (2 * sigma^2))
        IMatrix probs = (IMatrix)distances.pow(2.0).multiplyScalar(-1.0 / (2 * sigma * sigma)).exp();
        
        // 将对角线元素设为0（自己到自己的概率为0）
        probs.put(i, 0, 0.0);
        
        double sum = (double)probs.sum();
        double entropy = 0.0;
        
        if (sum > 0) {
            // 归一化概率
            IMatrix normalizedProbs = (IMatrix)probs.divideByScalar(sum);
            
            // 使用向量化操作计算熵，替代手动循环
            // 熵 = -sum(p * log(p))，其中p > 1e-12
            IMatrix p = normalizedProbs;
            IMatrix logP = (IMatrix)p.log();
            
            // 由于没有直接的矩阵比较方法，我们逐行处理
            double[] pArray = p.flatten().toDoubleArray();
            double[] logPArray = logP.flatten().toDoubleArray();
            double[] pLogPArray = new double[pArray.length];
            
            // 计算p * log(p)，但只对满足条件的元素(p > 1e-12)
            for (int k = 0; k < pArray.length; k++) {
                if (pArray[k] > 1e-12) {
                    pLogPArray[k] = pArray[k] * logPArray[k];
                } else {
                    pLogPArray[k] = 0.0;
                }
            }
            
            // 计算熵
            double pLogPSum = 0.0;
            for (int k = 0; k < pLogPArray.length; k++) {
                pLogPSum += pLogPArray[k];
            }
            entropy = -pLogPSum;
        }
        
        return entropy;
    }

    /**
     * 初始化低维嵌入Y
     */
    private IMatrix initializeLowDimEmbedding(int n, int dim) {
        // 使用IMatrix的随机初始化方法替代手动循环
        return (IMatrix)IMatrix.randn(n, dim).multiplyScalar(1e-4);
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
            velocity = (IMatrix)velocity.multiplyScalar(momentum).sub(gradient.multiplyScalar(learningRate));
            
            // 更新Y
            Y = (IMatrix)Y.add(velocity);
            
            // 每50次迭代输出一次进度
            if (iter % 50 == 0) {
                double cost = computeKLDivergence(P, Q);
                System.out.println("迭代 " + iter + "，KL散度: " + cost);
            }
            
            // 检查收敛性
            if (iter > 100) {
                IMatrix Q_prev = computeLowDimSimilarities((IMatrix)Y.sub(velocity));
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
        
        // 计算分子: 1 / (1 + ||yi - yj||^2)
        // 使用向量化操作替代嵌套循环
        for (int i = 0; i < n; i++) {
            IVector yi = (IVector)Y.getRow(i);
            // 计算yi到所有点的距离
            IMatrix distances = computePairwiseDistances(Y, yi);
            // 计算相似度: 1 / (1 + distance^2)
            IMatrix squaredDistances = (IMatrix)distances.pow(2.0);
            IMatrix onesMatrix = (IMatrix)IMatrix.ones(1, n);
            IMatrix denominator = (IMatrix)squaredDistances.add(onesMatrix);
            // 计算倒数: 1 / denominator
            IMatrix onesNumerator = (IMatrix)IMatrix.ones(1, n);
            IMatrix similarities = (IMatrix)onesNumerator.divide(denominator);
            // 将对角线元素设为0（自己到自己的相似度为0）
            similarities.put(i, 0, 0.0);
            // 设置第i行
            Q.setRow(i, (IVector)similarities.getRow(0));
            // 累加总和
            sum += (double)similarities.sum();
        }
        
        // 归一化
        if (sum > 0) {
            // 使用向量化操作进行归一化，替代手动循环
            Q = (IMatrix)Q.divideByScalar(sum);
            
            // 确保最小值，替代手动循环
            IMatrix minMatrix = (IMatrix)IMatrix.ones(n, n).multiplyScalar(1e-12);
            // 使用新的multiply方法进行元素级乘法
            IMatrix maxMatrix = (IMatrix)Q.multiply(minMatrix);
            Q = maxMatrix;
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
        
        // 使用向量化操作计算梯度，减少嵌套循环
        for (int i = 0; i < n; i++) {
            IVector yi = (IVector)Y.getRow(i);
            
            // 计算yi到所有点的距离
            IMatrix distances = computePairwiseDistances(Y, yi);
            // 计算距离的平方
            IMatrix squaredDistances = (IMatrix)distances.pow(2.0);
            // 计算分母: 1 + distance^2
            IMatrix onesMatrix = (IMatrix)IMatrix.ones(1, n);
            IMatrix denominator = (IMatrix)squaredDistances.add(onesMatrix);
            // 计算系数: (pij - qij) / (1 + distance^2)
            IVector piRow = (IVector)P.getRow(i);
            IVector qiRow = (IVector)Q.getRow(i);
            IVector diffPQ = (IVector)piRow.sub(qiRow);
            // 将分母矩阵转换为向量
            IVector denominatorVector = (IVector)denominator.getRow(0);
            // 计算系数
            IVector coefficient = (IVector)diffPQ.divide(denominatorVector);
            
            // 计算梯度更新项
            // 对于每个j，梯度更新为: 4 * (pij - qij) / (1 + ||yi - yj||^2) * (yi - yj)
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector yj = (IVector)Y.getRow(j);
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
     * 计算KL散度作为损失函数
     */
    private double computeKLDivergence(IMatrix P, IMatrix Q) {
        int n = P.getRowNum();
        double kl = 0.0;
        
        // 使用向量化操作计算KL散度，替代手动循环
        // KL散度 = sum(pij * log(pij / qij))，其中pij > 1e-12且qij > 1e-12
        for (int i = 0; i < n; i++) {
            IVector piRow = (IVector)P.getRow(i);
            IVector qiRow = (IVector)Q.getRow(i);
            
            // 计算log(pij / qij)，但只对满足条件的元素
            double[] piArray = piRow.toDoubleArray();
            double[] qiArray = qiRow.toDoubleArray();
            double[] logRatioArray = new double[piArray.length];
            
            double rowSum = 0.0;
            for (int j = 0; j < piArray.length; j++) {
                if (i != j && piArray[j] > 1e-12 && qiArray[j] > 1e-12) {
                    logRatioArray[j] = piArray[j] * Math.log(piArray[j] / qiArray[j]);
                } else {
                    logRatioArray[j] = 0.0;
                }
                rowSum += logRatioArray[j];
            }
            kl += rowSum;
        }
        
        return kl;
    }
    
    /**
     * 计算向量与矩阵中所有行的欧几里得距离
     */
    private IMatrix computePairwiseDistances(IMatrix X, IVector xi) {
        int n = X.getRowNum();
        IMatrix distances = IMatrix.zeros(1, n);
        
        // 使用向量化操作计算所有距离，替代手动循环
        for (int j = 0; j < n; j++) {
            IVector xj = (IVector)X.getRow(j);
            double distance = (double)xi.euclideanDistance(xj);
            distances.put(0, j, distance);
        }
        
        return distances;
    }
}