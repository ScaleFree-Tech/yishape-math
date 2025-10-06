package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.linalg.IVector;

/**
 * 逻辑回归分类结果类
 * <p>
 * 包含逻辑回归模型的训练结果，包括权重、偏置等参数。
 * 继承自ClassificationResult，使用父类的共同属性。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class LogisticRegressionResult extends ClassificationResult {

    /** 模型权重向量 */
    private IVector weights;

    /** 模型偏置向量 */
    private IVector bias;

    // ==================== 构造函数 ====================
    
    /**
     * 默认构造函数
     */
    public LogisticRegressionResult() {
        super();
    }
    
    /**
     * 带参数的构造函数
     * @param weights 权重向量
     * @param bias 偏置向量
     */
    public LogisticRegressionResult(IVector weights, IVector bias) {
        super();
        this.weights = weights;
        this.bias = bias;
        if (weights != null) {
            this.setNumFeatures(weights.size());
        }
    }

    // ==================== Getter和Setter方法 ====================
    
    /**
     * 获取权重向量
     * @return 权重向量
     */
    public IVector getWeights() {
        return weights;
    }

    /**
     * 设置权重向量
     * @param weights 权重向量
     */
    public void setWeights(IVector weights) {
        this.weights = weights;
        if (weights != null) {
            this.setNumFeatures(weights.size());
        }
    }

    /**
     * 获取偏置向量
     * @return 偏置向量
     */
    public IVector getBias() {
        return bias;
    }

    /**
     * 设置偏置向量
     * @param bias 偏置向量
     */
    public void setBias(IVector bias) {
        this.bias = bias;
    }

    // ==================== 实现抽象方法 ====================
    
    @Override
    public String getModelTypeDescription() {
        return "逻辑回归 (Logistic Regression)";
    }

    @Override
    public String getModelSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 逻辑回归模型摘要 ===\n");
        sb.append(getBasicStats()).append("\n");
        
        if (weights != null) {
            sb.append("权重向量维度: ").append(weights.size()).append("\n");
        }
        if (bias != null) {
            sb.append("偏置向量维度: ").append(bias.size()).append("\n");
        }
        
        sb.append("模型状态: ").append(isTrained() ? "已训练" : "未训练");
        return sb.toString();
    }

    @Override
    public boolean isTrained() {
        return weights != null && bias != null;
    }
    
    // ==================== 其他方法 ====================
    
    /**
     * 获取权重向量的L2范数
     * @return 权重向量的L2范数，如果权重为null则返回0
     */
    public double getWeightsL2Norm() {
        if (weights == null) {
            return 0.0;
        }
        double sum = 0.0;
        for (int i = 0; i < weights.size(); i++) {
            double w = weights.get(i).doubleValue();
            sum += w * w;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * 检查模型参数是否有效
     * @return true表示参数有效，false表示参数无效
     */
    public boolean hasValidParameters() {
        return weights != null && bias != null && 
               weights.size() > 0 && bias.size() > 0;
    }
}
