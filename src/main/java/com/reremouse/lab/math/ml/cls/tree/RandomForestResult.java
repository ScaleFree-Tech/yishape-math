package com.reremouse.lab.math.ml.cls.tree;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.ml.cls.ClassificationResult;
import java.util.List;
import java.util.Map;

/**
 * 随机森林分类结果类
 * <p>
 * 包含随机森林训练后的模型信息，包括决策树集合、特征重要性、训练参数等。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RandomForestResult extends ClassificationResult {
    
    /** 决策树集合 */
    private List<RFTree> trees;
    
    /** 树的数量 */
    private int nEstimators;
    
    /** 最大深度 */
    private int maxDepth;
    
    /** 最小分裂样本数 */
    private int minSamplesSplit;
    
    /** 最小叶子样本数 */
    private int minSamplesLeaf;
    
    /** 每次分裂时考虑的最大特征数 */
    private int maxFeatures;
    
    /** 是否使用Bootstrap采样 */
    private boolean bootstrap;
    
    /** 随机种子 */
    private long randomSeed;
    
    /** 袋外误差率 */
    private double oobScore;
    
    /**
     * 构造函数
     */
    public RandomForestResult() {
        super();
        this.oobScore = -1.0; // 初始化为-1表示未计算
    }
    
    /**
     * 构造函数
     * @param trees 决策树集合
     * @param nEstimators 树的数量
     * @param maxDepth 最大深度
     * @param minSamplesSplit 最小分裂样本数
     * @param minSamplesLeaf 最小叶子样本数
     * @param maxFeatures 最大特征数
     * @param bootstrap 是否使用Bootstrap采样
     * @param randomSeed 随机种子
     * @param labelMapping 标签映射
     * @param reverseLabelMapping 反向标签映射
     */
    public RandomForestResult(List<RFTree> trees, int nEstimators, int maxDepth,
                             int minSamplesSplit, int minSamplesLeaf, int maxFeatures,
                             boolean bootstrap, long randomSeed,
                             Map<String, Integer> labelMapping,
                             Map<Integer, String> reverseLabelMapping) {
        super();
        this.trees = trees;
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.randomSeed = randomSeed;
        this.setLabelMapping(labelMapping);
        this.setReverseLabelMapping(reverseLabelMapping);
        this.oobScore = -1.0;
    }
    
    /**
     * 完整构造函数（包含特征重要性和评估指标）
     * @param trees 决策树列表
     * @param nEstimators 树的数量
     * @param maxDepth 最大深度
     * @param maxFeatures 最大特征数
     * @param bootstrap 是否使用Bootstrap
     * @param randomSeed 随机种子
     * @param featureImportance 特征重要性数组
     * @param numClasses 类别数量
     * @param labelMapping 标签映射
     * @param reverseLabelMapping 反向标签映射
     * @param oobScore OOB分数
     * @param trainAccuracy 训练准确率
     */
    public RandomForestResult(List<RFTree> trees, int nEstimators, int maxDepth,
                             int maxFeatures, boolean bootstrap, long randomSeed,
                             double[] featureImportance, int numClasses,
                             Map<String, Integer> labelMapping,
                             Map<Integer, String> reverseLabelMapping,
                             double oobScore, double trainAccuracy) {
        super();
        this.trees = trees;
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.randomSeed = randomSeed;
        this.setNumClasses(numClasses);
        this.setLabelMapping(labelMapping);
        this.setReverseLabelMapping(reverseLabelMapping);
        this.oobScore = oobScore;
        this.setTrainAccuracy(trainAccuracy);
        
        // 转换特征重要性数组为IVector
        if (featureImportance != null) {
            this.setFeatureImportance(com.reremouse.lab.math.linalg.Linalg.vector(featureImportance));
        }
    }
    
    // ==================== Getters and Setters ====================
    
    public List<RFTree> getTrees() {
        return trees;
    }
    
    public void setTrees(List<RFTree> trees) {
        this.trees = trees;
    }
    
    public int getNEstimators() {
        return nEstimators;
    }
    
    public void setNEstimators(int nEstimators) {
        this.nEstimators = nEstimators;
    }
    
    public int getMaxDepth() {
        return maxDepth;
    }
    
    public void setMaxDepth(int maxDepth) {
        this.maxDepth = maxDepth;
    }
    
    public int getMinSamplesSplit() {
        return minSamplesSplit;
    }
    
    public void setMinSamplesSplit(int minSamplesSplit) {
        this.minSamplesSplit = minSamplesSplit;
    }
    
    public int getMinSamplesLeaf() {
        return minSamplesLeaf;
    }
    
    public void setMinSamplesLeaf(int minSamplesLeaf) {
        this.minSamplesLeaf = minSamplesLeaf;
    }
    
    public int getMaxFeatures() {
        return maxFeatures;
    }
    
    public void setMaxFeatures(int maxFeatures) {
        this.maxFeatures = maxFeatures;
    }
    
    public boolean isBootstrap() {
        return bootstrap;
    }
    
    public void setBootstrap(boolean bootstrap) {
        this.bootstrap = bootstrap;
    }
    
    public long getRandomSeed() {
        return randomSeed;
    }
    
    public void setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
    }
    

    
    public double getOobScore() {
        return oobScore;
    }
    
    public void setOobScore(double oobScore) {
        this.oobScore = oobScore;
    }
    

    
    // ==================== 辅助方法 ====================
    
    /**
     * 获取模型类型描述
     */
    @Override
    public String getModelTypeDescription() {
        if (isBinaryClassification()) {
            return String.format("二分类随机森林 (%d棵树)", nEstimators);
        } else {
            return String.format("多分类随机森林 (%d类, %d棵树)", getNumClasses(), nEstimators);
        }
    }
    
    /**
     * 获取模型摘要信息
     */
    @Override
    public String getModelSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 随机森林模型摘要 ===\n");
        sb.append(getBasicStats()).append("\n");
        sb.append(String.format("树的数量: %d\n", nEstimators));
        sb.append(String.format("最大深度: %d\n", maxDepth));
        sb.append(String.format("最小分裂样本数: %d\n", minSamplesSplit));
        sb.append(String.format("最小叶子样本数: %d\n", minSamplesLeaf));
        sb.append(String.format("最大特征数: %d\n", maxFeatures));
        sb.append(String.format("Bootstrap采样: %s\n", bootstrap ? "是" : "否"));
        
        if (oobScore >= 0) {
            sb.append(String.format("袋外得分: %.4f\n", oobScore));
        }
        
        sb.append("模型状态: ").append(isTrained() ? "已训练" : "未训练");
        return sb.toString();
    }
    
    /**
     * 检查模型是否已训练
     */
    @Override
    public boolean isTrained() {
        return trees != null && !trees.isEmpty();
    }
}