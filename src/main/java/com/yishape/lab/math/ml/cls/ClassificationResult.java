package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.linalg.IVector;
import java.io.Serializable;

import java.util.Map;

/**
 * 分类结果抽象基类
 * <p>
 * 包含所有分类算法结果的共同属性和方法，为不同分类算法提供统一的接口。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public abstract class ClassificationResult implements Serializable{
    
    // ==================== 共同基础属性 ====================
    
    /** 类别数量 */
    protected int numClasses;
    
    /** 是否为二分类 */
    protected boolean isBinaryClassification;
    
    /** 标签映射：标签名称 -> 索引 */
    protected Map<String, Integer> labelMapping;
    
    /** 反向标签映射：索引 -> 标签名称 */
    protected Map<Integer, String> reverseLabelMapping;
    
    /** 训练准确率 */
    protected double trainAccuracy;
    
    /** 训练损失 */
    protected double loss;
    
    /** 特征重要性分数 */
    protected IVector featureImportance;
    
    /** 特征数量 */
    protected int numFeatures;
    
    // ==================== 构造函数 ====================
    
    /**
     * 默认构造函数
     */
    public ClassificationResult() {
        this.trainAccuracy = -1.0; // 初始化为-1表示未计算
        this.loss = -1.0;
        this.numClasses = 0;
        this.numFeatures = 0;
        this.isBinaryClassification = false;
    }
    
    // ==================== 共同的Getter和Setter方法 ====================
    
    /**
     * 获取类别数量
     * @return 类别数量
     */
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * 设置类别数量
     * @param numClasses 类别数量
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
        this.isBinaryClassification = (numClasses == 2);
    }
    
    /**
     * 是否为二分类
     * @return true表示二分类，false表示多分类
     */
    public boolean isBinaryClassification() {
        return isBinaryClassification;
    }
    
    /**
     * 设置是否为二分类
     * @param isBinaryClassification 是否为二分类
     */
    public void setBinaryClassification(boolean isBinaryClassification) {
        this.isBinaryClassification = isBinaryClassification;
    }
    
    /**
     * 获取标签映射
     * @return 标签映射
     */
    public Map<String, Integer> getLabelMapping() {
        return labelMapping;
    }
    
    /**
     * 设置标签映射
     * @param labelMapping 标签映射
     */
    public void setLabelMapping(Map<String, Integer> labelMapping) {
        this.labelMapping = labelMapping;
        if (labelMapping != null) {
            this.numClasses = labelMapping.size();
            this.isBinaryClassification = (numClasses == 2);
        }
    }
    
    /**
     * 获取反向标签映射
     * @return 反向标签映射
     */
    public Map<Integer, String> getReverseLabelMapping() {
        return reverseLabelMapping;
    }
    
    /**
     * 设置反向标签映射
     * @param reverseLabelMapping 反向标签映射
     */
    public void setReverseLabelMapping(Map<Integer, String> reverseLabelMapping) {
        this.reverseLabelMapping = reverseLabelMapping;
    }
    
    /**
     * 获取训练准确率
     * @return 训练准确率
     */
    public double getTrainAccuracy() {
        return trainAccuracy;
    }
    
    /**
     * 设置训练准确率
     * @param trainAccuracy 训练准确率
     */
    public void setTrainAccuracy(double trainAccuracy) {
        this.trainAccuracy = trainAccuracy;
    }
    
    /**
     * 获取训练损失
     * @return 训练损失
     */
    public double getLoss() {
        return loss;
    }
    
    /**
     * 设置训练损失
     * @param loss 训练损失
     */
    public void setLoss(double loss) {
        this.loss = loss;
    }
    
    /**
     * 获取特征重要性
     * @return 特征重要性向量
     */
    public IVector getFeatureImportance() {
        return featureImportance;
    }
    
    /**
     * 设置特征重要性
     * @param featureImportance 特征重要性向量
     */
    public void setFeatureImportance(IVector featureImportance) {
        this.featureImportance = featureImportance;
        if (featureImportance != null) {
            this.numFeatures = featureImportance.size();
        }
    }
    
    /**
     * 获取特征数量
     * @return 特征数量
     */
    public int getNumFeatures() {
        return numFeatures;
    }
    
    /**
     * 设置特征数量
     * @param numFeatures 特征数量
     */
    public void setNumFeatures(int numFeatures) {
        this.numFeatures = numFeatures;
    }
    
    // ==================== 抽象方法 ====================
    
    /**
     * 获取模型类型描述
     * @return 模型类型描述字符串
     */
    public abstract String getModelTypeDescription();
    
    /**
     * 获取模型摘要信息
     * @return 模型摘要字符串
     */
    public abstract String getModelSummary();
    
    /**
     * 检查模型是否已训练
     * @return true表示已训练，false表示未训练
     */
    public abstract boolean isTrained();
    
    // ==================== 通用工具方法 ====================
    
    /**
     * 获取类别标签数组
     * @return 类别标签数组，如果标签映射为空则返回null
     */
    public String[] getClassLabels() {
        if (reverseLabelMapping == null || reverseLabelMapping.isEmpty()) {
            return null;
        }
        
        String[] labels = new String[numClasses];
        for (int i = 0; i < numClasses; i++) {
            labels[i] = reverseLabelMapping.get(i);
        }
        return labels;
    }
    
    /**
     * 根据标签名称获取索引
     * @param label 标签名称
     * @return 标签索引，如果不存在则返回-1
     */
    public int getLabelIndex(String label) {
        if (labelMapping == null) {
            return -1;
        }
        return labelMapping.getOrDefault(label, -1);
    }
    
    /**
     * 根据索引获取标签名称
     * @param index 标签索引
     * @return 标签名称，如果不存在则返回null
     */
    public String getLabelName(int index) {
        if (reverseLabelMapping == null) {
            return null;
        }
        return reverseLabelMapping.get(index);
    }
    
    /**
     * 检查是否有特征重要性信息
     * @return true表示有特征重要性信息，false表示没有
     */
    public boolean hasFeatureImportance() {
        return featureImportance != null && featureImportance.size() > 0;
    }
    
    /**
     * 获取基础统计信息
     * @return 包含基础统计信息的字符串
     */
    public String getBasicStats() {
        StringBuilder sb = new StringBuilder();
        sb.append("分类类型: ").append(isBinaryClassification ? "二分类" : "多分类").append("\n");
        sb.append("类别数量: ").append(numClasses).append("\n");
        sb.append("特征数量: ").append(numFeatures).append("\n");
        if (trainAccuracy >= 0) {
            sb.append("训练准确率: ").append(String.format("%.4f", trainAccuracy)).append("\n");
        }
        if (loss >= 0) {
            sb.append("训练损失: ").append(String.format("%.6f", loss)).append("\n");
        }
        sb.append("特征重要性: ").append(hasFeatureImportance() ? "可用" : "不可用");
        return sb.toString();
    }
}
