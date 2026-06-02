package com.yishape.lab.math.ml.dr;

import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.preprocessing.ITransform;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * t-SNE降维算法实现类 / t-SNE Dimensionality Reduction Algorithm Implementation
 * <p>
 * 实现t-distributed Stochastic Neighbor Embedding (t-SNE)算法，用于非线性降维
 * Implements t-distributed Stochastic Neighbor Embedding (t-SNE) for nonlinear dimensionality reduction
 * </p>
 *
 * @author lteb2
 */
public class RereTSNE implements ITransform<Double>, ISerializableModel {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereTSNE.class);

        private IMatrix<Double> feature;
    private int nComponents = -1;

    // 算法超参数 / Algorithm hyperparameters
    private double perplexity = 30.0;    // 困惑度 / Perplexity
    private int maxIter = 1000;          // 最大迭代次数 / Maximum iterations
    private double learningRate = 200.0; // 学习率 / Learning rate
    private double momentum = 0.8;       // 动量 / Momentum
    private double tolerance = 1e-4;     // 收敛阈值 / Convergence threshold

    /**
     * 与 {@link com.yishape.lab.math.linalg.RereDoubleMatrix#divideByScalar(Double)} 一致：
     * 除数绝对值 ≤1e-12 会抛异常，故归一化前概率质量须显著大于该门限（极小 sigma 等会导致 underflow）。
     */
    private static final double MIN_PROB_MASS_FOR_NORMALIZE = 1e-10;

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
        
        int n = feature.getRowNum();
        int originalDim = feature.getColNum();
        
        if (n < 2) {
            throw new IllegalArgumentException("样本数量必须至少为2 / At least 2 samples required");
        }
        
        log.debug("开始t-SNE降维：样本数=" + n + "，原始维度=" + originalDim + "，目标维度=" + nComponents);
        
        // 步骤1：计算高维空间中的相似度矩阵P
        IMatrix P = computeHighDimSimilarities((IMatrix)feature);
        
        // 步骤2：初始化低维嵌入Y
        IMatrix Y = initializeLowDimEmbedding(n, nComponents);
        
        // 步骤3：使用梯度下降优化Y
        Y = optimizeEmbedding(P, Y);
        
        log.debug("t-SNE降维完成");
        return Y;
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
    public RereTSNE setNComponents(int nComponents) {
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
     * 用t-SNE方法降维（便捷方法）
     * @param originalData 原数据，每行为一个样本
     * @param dim 目标维度，即列数
     * @return 降维后的矩阵
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim) {
        return setNComponents(dim).fit(originalData).transform(originalData);
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
            IMatrix scaledDistances = (IMatrix)squaredDistances.multiplyByScalar(-1.0 / (2 * sigma * sigma));
            IMatrix probs = (IMatrix)scaledDistances.exp();
            
            // 将对角线元素设为0（自己到自己的概率为0）；distances 为 1×n，坐标为 (0, j)
            probs.put(0, i, 0.0);
            
            // 归一化
            double sum = probs.sumValue();
            if (sum > MIN_PROB_MASS_FOR_NORMALIZE) {
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
        IMatrix minProbMatrix = (IMatrix)IMatrix.ones(n, n).multiplyByScalar(1e-12);
        // 替换multiply方法，使用逐元素比较的方式
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double pVal = (double)P_symmetric.get(i, j);
                double minVal = (double)minProbMatrix.get(i, j);
                P_symmetric.put(i, j, Math.max(pVal, minVal));
            }
        }
        
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
        IMatrix probs = (IMatrix)distances.pow(2.0).multiplyByScalar(-1.0 / (2 * sigma * sigma)).exp();
        
        // 将对角线元素设为0；distances 为 1×n
        probs.put(0, i, 0.0);
        
        double sum = probs.sumValue();
        double entropy = 0.0;
        
        if (sum > MIN_PROB_MASS_FOR_NORMALIZE) {
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
        return (IMatrix)IMatrix.randn(n, dim).multiplyByScalar(1e-4);
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
            
            // 更新动量项
            velocity = (IMatrix)velocity.multiplyByScalar(momentum).sub(gradient.multiplyByScalar(learningRate));
            
            // 更新嵌入
            Y = (IMatrix)Y.add(velocity);
            
            // 检查收敛性
            if (iter % 100 == 0) {
                double klDivergence = computeKLDivergence(P, Q);
                log.debug("Iteration " + iter + ", KL divergence: " + klDivergence);
            }
        }
        
        return Y;
    }
    
    /**
     * 计算低维空间中的相似度矩阵Q
     */
    private IMatrix computeLowDimSimilarities(IMatrix Y) {
        int n = Y.getRowNum();
        IMatrix Q = IMatrix.zeros(n, n);
        
        // 计算所有点对之间的t-分布相似度
        for (int i = 0; i < n; i++) {
            IVector yi = (IVector)Y.getRow(i);
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector yj = (IVector)Y.getRow(j);
                    double distance = (double)yi.euclideanDistance(yj);
                    // t-分布相似度: 1 / (1 + distance^2)
                    double similarity = 1.0 / (1.0 + distance * distance);
                    Q.put(i, j, similarity);
                }
            }
        }
        
        // 归一化
        double sum = Q.sumValue();
        if (sum > MIN_PROB_MASS_FOR_NORMALIZE) {
            Q = (IMatrix)Q.divideByScalar(sum);
        }
        
        // 避免概率为0，设置最小值
        int nRows = Q.getRowNum();
        int nCols = Q.getColNum();
        for (int i = 0; i < nRows; i++) {
            for (int j = 0; j < nCols; j++) {
                double val = (double)Q.get(i, j);
                Q.put(i, j, Math.max(val, 1e-12));
            }
        }
        
        // 确保对角线为0
        for (int i = 0; i < n; i++) {
            Q.put(i, i, 0.0);
        }
        
        return Q;
    }
    
    /**
     * 计算梯度
     */
    private IMatrix computeGradient(IMatrix P, IMatrix Q, IMatrix Y) {
        int n = Y.getRowNum();
        int dim = Y.getColNum();
        IMatrix gradient = IMatrix.zeros(n, dim);
        
        // 计算梯度
        for (int i = 0; i < n; i++) {
            IVector yi = (IVector)Y.getRow(i);
            IVector gradRow = (IVector)gradient.getRow(i);
            
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    IVector yj = (IVector)Y.getRow(j);
                    double pij = (double)P.get(i, j);
                    double qij = (double)Q.get(i, j);
                    
                    // 计算系数: 4 * (P_ij - Q_ij) * Q_ij
                    double coeff = 4.0 * (pij - qij) * qij;
                    
                    // 计算(yi - yj)
                    IVector diff = (IVector)yi.sub(yj);
                    
                    // 计算梯度更新项: coeff * (yi - yj)
                    IVector gradUpdate = (IVector)diff.multiplyByScalar(coeff);
                    
                    // 更新梯度矩阵的第i行
                    gradRow = (IVector)gradRow.add(gradUpdate);
                }
            }
            
            gradient.setRow(i, gradRow);
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

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("nComponents", nComponents);
        p.put("perplexity", perplexity);
        p.put("maxIter", maxIter);
        p.put("learningRate", learningRate);
        p.put("momentum", momentum);
        p.put("tolerance", tolerance);
        return p;
    }

    @Override
    public void fromParams(Map<String, Object> p) {
        this.nComponents = ((Number) p.get("nComponents")).intValue();
        this.perplexity = ((Number) p.get("perplexity")).doubleValue();
        this.maxIter = ((Number) p.get("maxIter")).intValue();
        this.learningRate = ((Number) p.get("learningRate")).doubleValue();
        this.momentum = ((Number) p.get("momentum")).doubleValue();
        this.tolerance = ((Number) p.get("tolerance")).doubleValue();
    }

}