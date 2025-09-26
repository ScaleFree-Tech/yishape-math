package com.reremouse.lab.math.ml.cls.tree;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import java.util.ArrayList;
import java.util.List;

/**
 * 决策树类（用于XGBoost）
 * <p>
 * 实现XGBoost中使用的回归树，支持基于梯度和海塞矩阵的分裂准则。
 * 每个叶子节点输出一个数值，用于梯度提升算法。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class XGTree {
    
    /** 根节点 */
    private XGTreeNode root;
    
    /** 最大深度 */
    private int maxDepth;
    
    /** 最小分裂样本数 */
    private int minSamplesSplit;
    
    /** 最小叶子样本数 */
    private int minSamplesLeaf;
    
    /** L1正则化参数 */
    private double alpha;
    
    /** L2正则化参数 */
    private double lambda;
    
    /** 学习率 */
    private double learningRate;
    
    /**
     * 构造函数
     * @param maxDepth 最大深度
     * @param minSamplesSplit 最小分裂样本数
     * @param minSamplesLeaf 最小叶子样本数
     * @param alpha L1正则化参数
     * @param lambda L2正则化参数
     * @param learningRate 学习率
     */
    public XGTree(int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                       double alpha, double lambda, double learningRate) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.alpha = alpha;
        this.lambda = lambda;
        this.learningRate = learningRate;
    }
    
    /**
     * 训练决策树
     * @param features 特征矩阵
     * @param gradients 梯度向量
     * @param hessians 海塞向量
     */
    public void fit(IMatrix features, IVector gradients, IVector hessians) {
        int numSamples = features.rows();
        int[] sampleIndices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            sampleIndices[i] = i;
        }
        
        this.root = buildTree(features, gradients, hessians, sampleIndices, 0);
    }
    
    /**
     * 递归构建决策树
     * @param features 特征矩阵
     * @param gradients 梯度向量
     * @param hessians 海塞向量
     * @param sampleIndices 样本索引
     * @param depth 当前深度
     * @return 树节点
     */
    private XGTreeNode buildTree(IMatrix features, IVector gradients, IVector hessians,
                              int[] sampleIndices, int depth) {
        
        int numSamples = sampleIndices.length;
        
        // 计算当前节点的梯度和海塞和
        double gradSum = 0.0;
        double hessSum = 0.0;
        for (int idx : sampleIndices) {
            gradSum += gradients.get(idx).doubleValue();
            hessSum += hessians.get(idx).doubleValue();
        }
        
        // 计算叶子节点的最优权重
        double leafValue = computeLeafWeight(gradSum, hessSum);
        
        // 停止条件检查
        if (shouldStop(depth, numSamples, hessSum)) {
            return new XGTreeNode(leafValue, depth, numSamples);
        }
        
        // 寻找最佳分裂
        SplitResult bestSplit = findBestSplit(features, gradients, hessians, sampleIndices);
        
        if (bestSplit == null || bestSplit.gain <= 0) {
            return new XGTreeNode(leafValue, depth, numSamples);
        }
        
        // 分裂样本
        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();
        
        for (int idx : sampleIndices) {
            if (features.get(idx, bestSplit.featureIndex).doubleValue() <= bestSplit.threshold) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }
        
        // 检查分裂后的样本数量
        if (leftIndices.size() < minSamplesLeaf || rightIndices.size() < minSamplesLeaf) {
            return new XGTreeNode(leafValue, depth, numSamples);
        }
        
        // 递归构建左右子树
        XGTreeNode leftChild = buildTree(features, gradients, hessians, 
                                     leftIndices.stream().mapToInt(i -> i).toArray(), depth + 1);
        XGTreeNode rightChild = buildTree(features, gradients, hessians, 
                                      rightIndices.stream().mapToInt(i -> i).toArray(), depth + 1);
        
        return new XGTreeNode(bestSplit.featureIndex, bestSplit.threshold, 
                           leftChild, rightChild, depth, numSamples, bestSplit.gain);
    }
    
    /**
     * 计算叶子节点的最优权重
     * @param gradSum 梯度和
     * @param hessSum 海塞和
     * @return 最优权重
     */
    private double computeLeafWeight(double gradSum, double hessSum) {
        if (hessSum + lambda == 0) {
            return 0.0;
        }
        
        // XGBoost叶子权重公式：-G/(H+λ)，其中G是梯度和，H是海塞和
        double weight = -gradSum / (hessSum + lambda);
        
        // 应用L1正则化（软阈值）
        if (alpha > 0) {
            if (weight > alpha) {
                weight -= alpha;
            } else if (weight < -alpha) {
                weight += alpha;
            } else {
                weight = 0.0;
            }
        }
        
        return weight * learningRate;
    }
    
    /**
     * 检查是否应该停止分裂
     * @param depth 当前深度
     * @param numSamples 样本数量
     * @param hessSum 海塞和
     * @return 是否停止
     */
    private boolean shouldStop(int depth, int numSamples, double hessSum) {
        return depth >= maxDepth || 
               numSamples < minSamplesSplit || 
               hessSum < 1e-8;
    }
    
    /**
     * 寻找最佳分裂
     * @param features 特征矩阵
     * @param gradients 梯度向量
     * @param hessians 海塞向量
     * @param sampleIndices 样本索引
     * @return 最佳分裂结果
     */
    private SplitResult findBestSplit(IMatrix features, IVector gradients, IVector hessians,
                                     int[] sampleIndices) {
        
        int numFeatures = features.cols();
        SplitResult bestSplit = null;
        double bestGain = 0.0;
        
        // 计算当前节点的梯度和海塞和
        double totalGradSum = 0.0;
        double totalHessSum = 0.0;
        for (int idx : sampleIndices) {
            totalGradSum += gradients.get(idx).doubleValue();
            totalHessSum += hessians.get(idx).doubleValue();
        }
        
        // 遍历所有特征
        for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
            
            // 获取当前特征的所有唯一值作为候选分裂点
            List<Double> candidateThresholds = getCandidateThresholds(features, sampleIndices, featureIdx);
            
            // 遍历所有候选分裂点
            for (double threshold : candidateThresholds) {
                
                // 计算分裂后的增益
                double gain = computeSplitGain(features, gradients, hessians, sampleIndices,
                                             featureIdx, threshold, totalGradSum, totalHessSum);
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = new SplitResult(featureIdx, threshold, gain);
                }
            }
        }
        
        return bestSplit;
    }
    
    /**
     * 获取候选分裂阈值
     * @param features 特征矩阵
     * @param sampleIndices 样本索引
     * @param featureIdx 特征索引
     * @return 候选阈值列表
     */
    private List<Double> getCandidateThresholds(IMatrix features, int[] sampleIndices, int featureIdx) {
        List<Double> values = new ArrayList<>();
        for (int idx : sampleIndices) {
            values.add(features.get(idx, featureIdx).doubleValue());
        }
        
        values.sort(Double::compareTo);
        
        List<Double> thresholds = new ArrayList<>();
        for (int i = 0; i < values.size() - 1; i++) {
            if (!values.get(i).equals(values.get(i + 1))) {
                thresholds.add((values.get(i) + values.get(i + 1)) / 2.0);
            }
        }
        
        return thresholds;
    }
    
    /**
     * 计算分裂增益
     * @param features 特征矩阵
     * @param gradients 梯度向量
     * @param hessians 海塞向量
     * @param sampleIndices 样本索引
     * @param featureIdx 特征索引
     * @param threshold 分裂阈值
     * @param totalGradSum 总梯度和
     * @param totalHessSum 总海塞和
     * @return 分裂增益
     */
    private double computeSplitGain(IMatrix features, IVector gradients, IVector hessians,
                                   int[] sampleIndices, int featureIdx, double threshold,
                                   double totalGradSum, double totalHessSum) {
        
        double leftGradSum = 0.0;
        double leftHessSum = 0.0;
        int leftCount = 0;
        
        for (int idx : sampleIndices) {
            if (features.get(idx, featureIdx).doubleValue() <= threshold) {
                leftGradSum += gradients.get(idx).doubleValue();
                leftHessSum += hessians.get(idx).doubleValue();
                leftCount++;
            }
        }
        
        double rightGradSum = totalGradSum - leftGradSum;
        double rightHessSum = totalHessSum - leftHessSum;
        int rightCount = sampleIndices.length - leftCount;
        
        // 检查分裂后的样本数量
        if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) {
            return 0.0;
        }
        
        // 计算增益：XGBoost分裂增益公式
        double leftScore = (leftGradSum * leftGradSum) / (leftHessSum + lambda);
        double rightScore = (rightGradSum * rightGradSum) / (rightHessSum + lambda);
        double parentScore = (totalGradSum * totalGradSum) / (totalHessSum + lambda);
        
        double gain = 0.5 * (leftScore + rightScore - parentScore);
        
        // 减去复杂度惩罚（树的复杂度）
        gain -= alpha;
        
        return gain;
    }
    
    /**
     * 预测单个样本
     * @param features 特征向量
     * @return 预测值
     */
    public double predict(IVector features) {
        if (root == null) {
            return 0.0;
        }
        
        double[] featureArray = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureArray[i] = features.get(i).doubleValue();
        }
        
        return root.predict(featureArray);
    }
    
    /**
     * 批量预测
     * @param features 特征矩阵
     * @return 预测值数组
     */
    public double[] predict(IMatrix features) {
        int numSamples = features.rows();
        double[] predictions = new double[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            predictions[i] = predict(features.getRow(i));
        }
        
        return predictions;
    }
    
    /**
     * 计算特征重要性
     * @param numFeatures 特征数量
     * @return 特征重要性数组
     */
    public double[] computeFeatureImportance(int numFeatures) {
        double[] importance = new double[numFeatures];
        if (root != null) {
            root.computeFeatureImportance(importance);
        }
        return importance;
    }
    
    /**
     * 获取树的深度
     * @return 树的深度
     */
    public int getDepth() {
        return root != null ? root.getTreeDepth() : 0;
    }
    
    /**
     * 获取叶子节点数量
     * @return 叶子节点数量
     */
    public int getLeafCount() {
        return root != null ? root.getLeafCount() : 0;
    }
    
    // ==================== Getters and Setters ====================
    
    public XGTreeNode getRoot() {
        return root;
    }
    
    public int getMaxDepth() {
        return maxDepth;
    }
    
    public int getMinSamplesSplit() {
        return minSamplesSplit;
    }
    
    public int getMinSamplesLeaf() {
        return minSamplesLeaf;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public double getLambda() {
        return lambda;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * 分裂结果内部类
     */
    private static class SplitResult {
        int featureIndex;
        double threshold;
        double gain;
        
        SplitResult(int featureIndex, double threshold, double gain) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.gain = gain;
        }
    }
}