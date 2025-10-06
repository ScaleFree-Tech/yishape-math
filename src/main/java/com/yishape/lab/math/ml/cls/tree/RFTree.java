package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.*;
import java.util.stream.IntStream;

/**
 * 分类决策树类（用于随机森林）
 * <p>
 * 实现传统的分类决策树，使用信息增益或基尼不纯度作为分裂准则。
 * 支持特征随机选择和Bootstrap采样。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RFTree {
    
    /** 根节点 */
    private RFTreeNode root;
    
    /** 最大深度 */
    private int maxDepth;
    
    /** 最小分裂样本数 */
    private int minSamplesSplit;
    
    /** 最小叶子样本数 */
    private int minSamplesLeaf;
    
    /** 每次分裂时考虑的最大特征数 */
    private int maxFeatures;
    
    /** 分裂准则 */
    private SplitCriterion criterion;
    
    /** 随机数生成器 */
    private Random random;
    
    /** 特征重要性 */
    private double[] featureImportance;
    
    /** 类别数量 */
    private int numClasses;
    
    /** 标签映射 */
    private Map<String, Integer> labelMapping;
    
    /** 反向标签映射 */
    private Map<Integer, String> reverseLabelMapping;
    
    /**
     * 分裂准则枚举
     */
    public enum SplitCriterion {
        GINI,           // 基尼不纯度
        ENTROPY         // 信息熵
    }
    
    /**
     * 构造函数
     * @param maxDepth 最大深度
     * @param minSamplesSplit 最小分裂样本数
     * @param minSamplesLeaf 最小叶子样本数
     * @param maxFeatures 最大特征数
     * @param criterion 分裂准则
     * @param randomSeed 随机种子
     */
    public RFTree(int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                                    int maxFeatures, SplitCriterion criterion, long randomSeed) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.criterion = criterion;
        this.random = new Random(randomSeed);
    }
    
    /**
     * 训练决策树
     * @param features 特征矩阵
     * @param labels 标签数组
     * @param sampleIndices 样本索引（用于Bootstrap采样）
     */
    public void fit(IMatrix features, String[] labels, int[] sampleIndices) {
        // 预处理标签
        preprocessLabels(labels);
        
        // 初始化特征重要性
        this.featureImportance = new double[features.cols()];
        
        // 转换标签为数值
        int[] numericLabels = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            numericLabels[i] = labelMapping.get(labels[i]);
        }
        
        // 构建决策树
        this.root = buildTree(features, numericLabels, sampleIndices, 0, null);
    }
    
    /**
     * 递归构建决策树
     * @param features 特征矩阵
     * @param labels 数值标签数组
     * @param sampleIndices 样本索引
     * @param depth 当前深度
     * @param availableFeatures 可用特征索引（用于特征随机选择）
     * @return 树节点
     */
    private RFTreeNode buildTree(IMatrix features, int[] labels, int[] sampleIndices, 
                                           int depth, Set<Integer> availableFeatures) {
        
        int numSamples = sampleIndices.length;
        
        // 计算当前节点的类别分布
        Map<Integer, Integer> classCounts = new HashMap<>();
        for (int idx : sampleIndices) {
            int label = labels[idx];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        
        // 找到多数类
        int majorityClass = classCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get().getKey();
        
        String majorityLabel = reverseLabelMapping.get(majorityClass);
        
        // 停止条件检查
        if (shouldStop(depth, numSamples, classCounts)) {
            return new RFTreeNode(majorityLabel, depth, numSamples, classCounts);
        }
        
        // 随机选择特征子集
        Set<Integer> candidateFeatures = selectRandomFeatures(features.cols(), availableFeatures);
        
        // 寻找最佳分裂
        SplitResult bestSplit = findBestSplit(features, labels, sampleIndices, candidateFeatures);
        
        if (bestSplit == null || bestSplit.gain <= 0) {
            return new RFTreeNode(majorityLabel, depth, numSamples, classCounts);
        }
        
        // 更新特征重要性
        featureImportance[bestSplit.featureIndex] += bestSplit.gain * numSamples;
        
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
            return new RFTreeNode(majorityLabel, depth, numSamples, classCounts);
        }
        
        // 递归构建左右子树
        RFTreeNode leftChild = buildTree(features, labels, 
                leftIndices.stream().mapToInt(i -> i).toArray(), depth + 1, candidateFeatures);
        RFTreeNode rightChild = buildTree(features, labels, 
                rightIndices.stream().mapToInt(i -> i).toArray(), depth + 1, candidateFeatures);
        
        return new RFTreeNode(bestSplit.featureIndex, bestSplit.threshold, 
                leftChild, rightChild, depth, numSamples, bestSplit.gain, classCounts);
    }
    
    /**
     * 随机选择特征子集
     * @param totalFeatures 总特征数
     * @param availableFeatures 可用特征集合
     * @return 选中的特征索引集合
     */
    private Set<Integer> selectRandomFeatures(int totalFeatures, Set<Integer> availableFeatures) {
        Set<Integer> allFeatures;
        if (availableFeatures != null) {
            allFeatures = availableFeatures;
        } else {
            allFeatures = new HashSet<>();
            for (int i = 0; i < totalFeatures; i++) {
                allFeatures.add(i);
            }
        }
        
        int numFeaturesToSelect = Math.min(maxFeatures, allFeatures.size());
        
        List<Integer> featureList = new ArrayList<>(allFeatures);
        Collections.shuffle(featureList, random);
        
        return new HashSet<>(featureList.subList(0, numFeaturesToSelect));
    }
    
    /**
     * 检查是否应该停止分裂
     * @param depth 当前深度
     * @param numSamples 样本数量
     * @param classCounts 类别计数
     * @return 是否停止
     */
    private boolean shouldStop(int depth, int numSamples, Map<Integer, Integer> classCounts) {
        return depth >= maxDepth || 
               numSamples < minSamplesSplit || 
               classCounts.size() <= 1; // 纯节点
    }
    
    /**
     * 寻找最佳分裂
     * @param features 特征矩阵
     * @param labels 标签数组
     * @param sampleIndices 样本索引
     * @param candidateFeatures 候选特征
     * @return 最佳分裂结果
     */
    private SplitResult findBestSplit(IMatrix features, int[] labels, int[] sampleIndices,
                                     Set<Integer> candidateFeatures) {
        
        SplitResult bestSplit = null;
        double bestGain = 0.0;
        
        // 计算当前节点的不纯度
        double currentImpurity = computeImpurity(labels, sampleIndices);
        
        // 遍历候选特征
        for (int featureIdx : candidateFeatures) {
            
            // 获取候选分裂点
            List<Double> candidateThresholds = getCandidateThresholds(features, sampleIndices, featureIdx);
            
            // 遍历候选分裂点
            for (double threshold : candidateThresholds) {
                
                // 计算分裂后的增益
                double gain = computeSplitGain(features, labels, sampleIndices,
                        featureIdx, threshold, currentImpurity);
                
                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = new SplitResult(featureIdx, threshold, gain);
                }
            }
        }
        
        return bestSplit;
    }
    
    /**
     * 计算不纯度
     * @param labels 标签数组
     * @param sampleIndices 样本索引
     * @return 不纯度值
     */
    private double computeImpurity(int[] labels, int[] sampleIndices) {
        Map<Integer, Integer> classCounts = new HashMap<>();
        for (int idx : sampleIndices) {
            int label = labels[idx];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        
        int totalSamples = sampleIndices.length;
        
        if (criterion == SplitCriterion.GINI) {
            return computeGiniImpurity(classCounts, totalSamples);
        } else {
            return computeEntropyImpurity(classCounts, totalSamples);
        }
    }
    
    /**
     * 计算基尼不纯度
     * @param classCounts 类别计数
     * @param totalSamples 总样本数
     * @return 基尼不纯度
     */
    private double computeGiniImpurity(Map<Integer, Integer> classCounts, int totalSamples) {
        // 使用向量运算计算基尼不纯度
        double[] probabilities = new double[classCounts.size()];
        int index = 0;
        for (int count : classCounts.values()) {
            probabilities[index++] = (double) count / totalSamples;
        }
        
        IVector probVector = Linalg.vector(probabilities);
        // 计算概率平方和：sum(p_i^2)
        double sumSquaredProbs = (double)probVector.square().sum();
//        for (int i = 0; i < probVector.size(); i++) {
//            double prob = probVector.get(i).doubleValue();
//            sumSquaredProbs += prob * prob;
//        }
        
        return 1.0 - sumSquaredProbs;
    }
    
    /**
     * 计算信息熵
     * @param classCounts 类别计数
     * @param totalSamples 总样本数
     * @return 信息熵
     */
    private double computeEntropyImpurity(Map<Integer, Integer> classCounts, int totalSamples) {
        // 使用向量运算计算信息熵
        double entropy = 0.0;
        for (int count : classCounts.values()) {
            if (count > 0) {
                double probability = (double) count / totalSamples;
                entropy -= probability * (Math.log(probability) / Math.log(2));
            }
        }
        return entropy;
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
     * @param labels 标签数组
     * @param sampleIndices 样本索引
     * @param featureIdx 特征索引
     * @param threshold 分裂阈值
     * @param currentImpurity 当前不纯度
     * @return 分裂增益
     */
    private double computeSplitGain(IMatrix features, int[] labels, int[] sampleIndices,
                                   int featureIdx, double threshold, double currentImpurity) {
        
        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();
        
        for (int idx : sampleIndices) {
            if (features.get(idx, featureIdx).doubleValue() <= threshold) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }
        
        // 检查分裂后的样本数量
        if (leftIndices.size() < minSamplesLeaf || rightIndices.size() < minSamplesLeaf) {
            return 0.0;
        }
        
        // 计算分裂后的加权不纯度
        int totalSamples = sampleIndices.length;
        double leftWeight = (double) leftIndices.size() / totalSamples;
        double rightWeight = (double) rightIndices.size() / totalSamples;
        
        double leftImpurity = computeImpurity(labels, leftIndices.stream().mapToInt(i -> i).toArray());
        double rightImpurity = computeImpurity(labels, rightIndices.stream().mapToInt(i -> i).toArray());
        
        double weightedImpurity = leftWeight * leftImpurity + rightWeight * rightImpurity;
        
        return currentImpurity - weightedImpurity;
    }
    
    /**
     * 预测单个样本
     * @param features 特征向量
     * @return 预测标签
     */
    public String predict(IVector features) {
        if (root == null) {
            throw new IllegalStateException("模型尚未训练");
        }
        return root.predict(features);
    }
    
    /**
     * 批量预测
     * @param features 特征矩阵
     * @return 预测标签数组
     */
    public String[] predict(IMatrix features) {
        int numSamples = features.rows();
        String[] predictions = new String[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            predictions[i] = predict(features.getRow(i));
        }
        
        return predictions;
    }
    
    /**
     * 标签预处理
     */
    private void preprocessLabels(String[] labels) {
        labelMapping = new HashMap<>();
        reverseLabelMapping = new HashMap<>();
        
        int nextLabel = 0;
        for (String label : labels) {
            if (!labelMapping.containsKey(label)) {
                labelMapping.put(label, nextLabel);
                reverseLabelMapping.put(nextLabel, label);
                nextLabel++;
            }
        }
        
        this.numClasses = labelMapping.size();
    }
    
    // ==================== Getters ====================
    
    public double[] getFeatureImportance() {
        return featureImportance.clone();
    }
    
    public int getDepth() {
        return root != null ? root.getTreeDepth() : 0;
    }
    
    public int getLeafCount() {
        return root != null ? root.getLeafCount() : 0;
    }
    
    public RFTreeNode getRoot() {
        return root;
    }
    
    public Map<String, Integer> getLabelMapping() {
        return new HashMap<>(labelMapping);
    }
    
    public Map<Integer, String> getReverseLabelMapping() {
        return new HashMap<>(reverseLabelMapping);
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