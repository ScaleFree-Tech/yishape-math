package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.cls.ClassificationResult;

import java.util.List;
import java.util.Map;

/**
 * XGBoost分类结果类
 * <p>
 * 包含XGBoost训练后的模型信息，包括决策树集合、特征重要性、训练损失等。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class XGBoostResult extends ClassificationResult {
    
    /** 决策树集合 */
    private List<XGTree> trees;
    
    /** 学习率 */
    private double learningRate;
    
    /** 训练损失历史 */
    private double[] lossHistory;
    

    
    /** 正则化参数 */
    private double lambda;
    private double alpha;
    
    /** 最大树深度 */
    private int maxDepth;
    
    /** 最小分裂样本数 */
    private int minSamplesSplit;
    
    /** 最小叶子样本数 */
    private int minSamplesLeaf;
    
    /**
     * 构造函数
     */
    public XGBoostResult() {
        super();
    }
    
    /**
     * 构造函数
     * @param trees 决策树集合
     * @param learningRate 学习率
     * @param lossHistory 损失历史
     * @param featureImportance 特征重要性
     * @param numClasses 类别数量
     * @param labelMapping 标签映射
     * @param reverseLabelMapping 反向标签映射
     */
    public XGBoostResult(List<XGTree> trees, double learningRate, 
                        double[] lossHistory, IVector featureImportance,
                        int numClasses, Map<String, Integer> labelMapping,
                        Map<Integer, String> reverseLabelMapping) {
        super();
        this.trees = trees;
        this.learningRate = learningRate;
        this.lossHistory = lossHistory;
        this.setFeatureImportance(featureImportance);
        this.setNumClasses(numClasses);
        this.setLabelMapping(labelMapping);
        this.setReverseLabelMapping(reverseLabelMapping);
    }
    
    // ==================== Getters and Setters ====================
    
    public List<XGTree> getTrees() {
        return trees;
    }
    
    public void setTrees(List<XGTree> trees) {
        this.trees = trees;
    }
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public double[] getLossHistory() {
        return lossHistory;
    }
    
    public void setLossHistory(double[] lossHistory) {
        this.lossHistory = lossHistory;
    }
    
    public void setLossHistory(List<Double> lossHistory) {
        if (lossHistory != null) {
            this.lossHistory = lossHistory.stream().mapToDouble(Double::doubleValue).toArray();
        }
    }
    
    public void setBinary(boolean isBinary) {
        this.setBinaryClassification(isBinary);
    }
    
    public void setLabelToIndex(Map<String, Integer> labelToIndex) {
        this.setLabelMapping(labelToIndex);
    }
    
    public void setIndexToLabel(Map<Integer, String> indexToLabel) {
        this.setReverseLabelMapping(indexToLabel);
    }
    
    public double getLambda() {
        return lambda;
    }
    
    public void setLambda(double lambda) {
        this.lambda = lambda;
    }
    
    public double getAlpha() {
        return alpha;
    }
    
    public void setAlpha(double alpha) {
        this.alpha = alpha;
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
    
    /**
     * 获取树的数量
     * @return 树的数量
     */
    public int getNumTrees() {
        return trees != null ? trees.size() : 0;
    }
    
    /**
     * 获取最终训练损失
     * @return 最终训练损失
     */
    public double getFinalLoss() {
        return lossHistory != null && lossHistory.length > 0 ? 
               lossHistory[lossHistory.length - 1] : Double.NaN;
    }
    
    // ==================== 实现抽象方法 ====================
    
    @Override
    public String getModelTypeDescription() {
        return String.format("XGBoost (%s, %d类)", 
                           isBinaryClassification() ? "二分类" : "多分类", 
                           getNumClasses());
    }
    
    @Override
    public String getModelSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== XGBoost模型摘要 ===\n");
        sb.append(getBasicStats()).append("\n");
        sb.append(String.format("树数量: %d\n", getNumTrees()));
        sb.append(String.format("学习率: %.4f\n", learningRate));
        sb.append(String.format("正则化参数 - Lambda: %.4f, Alpha: %.4f\n", lambda, alpha));
        sb.append(String.format("最大深度: %d\n", maxDepth));
        sb.append(String.format("最小分裂样本数: %d\n", minSamplesSplit));
        sb.append(String.format("最小叶子样本数: %d\n", minSamplesLeaf));
        
        if (!Double.isNaN(getFinalLoss())) {
            sb.append(String.format("最终损失: %.6f\n", getFinalLoss()));
        }
        
        sb.append("模型状态: ").append(isTrained() ? "已训练" : "未训练");
        return sb.toString();
    }
    
    @Override
    public boolean isTrained() {
        return trees != null && !trees.isEmpty();
    }
    
    // ==================== 其他方法 ====================
    
    /**
     * 获取模型描述信息（保持向后兼容性）
     * @return 模型描述
     */
    public String getModelDescription() {
        return getModelTypeDescription() + " - 树数量: " + getNumTrees() + 
               ", 学习率: " + learningRate + 
               ", 最终损失: " + String.format("%.6f", getFinalLoss());
    }
}