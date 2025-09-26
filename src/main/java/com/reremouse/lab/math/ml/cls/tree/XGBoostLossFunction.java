package com.reremouse.lab.math.ml.cls.tree;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;

/**
 * XGBoost损失函数类
 * <p>
 * 实现XGBoost中使用的各种损失函数，包括二分类的logistic loss和多分类的softmax loss。
 * 提供梯度和海塞矩阵的计算功能。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class XGBoostLossFunction {
    
    /** 损失函数类型枚举 */
    public enum LossType {
        /** 二分类logistic损失 */
        BINARY_LOGISTIC,
        /** 多分类softmax损失 */
        MULTICLASS_SOFTMAX
    }
    
    /** 损失函数类型 */
    private LossType lossType;
    
    /** 类别数量（多分类时使用） */
    private int numClasses;
    
    /** 数值稳定性的小常数 */
    private static final double EPS = 1e-15;
    
    /**
     * 构造函数
     * @param lossType 损失函数类型
     * @param numClasses 类别数量
     */
    public XGBoostLossFunction(LossType lossType, int numClasses) {
        this.lossType = lossType;
        this.numClasses = numClasses;
    }
    
    /**
     * 计算损失值
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 损失值
     */
    public double computeLoss(IMatrix predictions, IMatrix labels) {
        switch (lossType) {
            case BINARY_LOGISTIC:
                return computeBinaryLogisticLoss(predictions, labels);
            case MULTICLASS_SOFTMAX:
                return computeMulticlassSoftmaxLoss(predictions, labels);
            default:
                throw new IllegalArgumentException("Unsupported loss type: " + lossType);
        }
    }
    
    /**
     * 计算梯度
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 梯度矩阵
     */
    public IMatrix computeGradients(IMatrix predictions, IMatrix labels) {
        switch (lossType) {
            case BINARY_LOGISTIC:
                return computeBinaryLogisticGradients(predictions, labels);
            case MULTICLASS_SOFTMAX:
                return computeMulticlassSoftmaxGradients(predictions, labels);
            default:
                throw new IllegalArgumentException("Unsupported loss type: " + lossType);
        }
    }
    
    /**
     * 计算海塞矩阵（对角线元素）
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 海塞矩阵对角线元素
     */
    public IMatrix computeHessians(IMatrix predictions, IMatrix labels) {
        switch (lossType) {
            case BINARY_LOGISTIC:
                return computeBinaryLogisticHessians(predictions, labels);
            case MULTICLASS_SOFTMAX:
                return computeMulticlassSoftmaxHessians(predictions, labels);
            default:
                throw new IllegalArgumentException("Unsupported loss type: " + lossType);
        }
    }
    
    /**
     * 计算二分类logistic损失
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 损失值
     */
    private double computeBinaryLogisticLoss(IMatrix predictions, IMatrix labels) {
        int numSamples = predictions.rows();
        double totalLoss = 0.0;
        
        for (int i = 0; i < numSamples; i++) {
            double pred = predictions.get(i, 0).doubleValue();
            double label = labels.get(i, 0).doubleValue();
            
            // 应用sigmoid函数
            double prob = sigmoid(pred);
            prob = Math.max(EPS, Math.min(1.0 - EPS, prob)); // 数值稳定性
            
            // 计算交叉熵损失
            totalLoss += -(label * Math.log(prob) + (1 - label) * Math.log(1 - prob));
        }
        
        return totalLoss / numSamples;
    }
    
    /**
     * 计算二分类logistic梯度
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 梯度矩阵
     */
    private IMatrix computeBinaryLogisticGradients(IMatrix predictions, IMatrix labels) {
        int numSamples = predictions.rows();
        IMatrix gradients = Linalg.zeros(numSamples, 1);
        
        for (int i = 0; i < numSamples; i++) {
            double pred = predictions.get(i, 0).doubleValue();
            double label = labels.get(i, 0).doubleValue();
            
            // 梯度：sigmoid(pred) - label
            double prob = sigmoid(pred);
            double gradient = prob - label;
            
            gradients.set(i, 0, gradient);
        }
        
        return gradients;
    }
    
    /**
     * 计算二分类logistic海塞矩阵
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 海塞矩阵对角线元素
     */
    private IMatrix computeBinaryLogisticHessians(IMatrix predictions, IMatrix labels) {
        int numSamples = predictions.rows();
        IMatrix hessians = Linalg.zeros(numSamples, 1);
        
        for (int i = 0; i < numSamples; i++) {
            double pred = predictions.get(i, 0).doubleValue();
            
            // 海塞矩阵：sigmoid(pred) * (1 - sigmoid(pred))
            double prob = sigmoid(pred);
            double hessian = prob * (1 - prob);
            hessian = Math.max(EPS, hessian); // 数值稳定性
            
            hessians.set(i, 0, hessian);
        }
        
        return hessians;
    }
    
    /**
     * 计算多分类softmax损失
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 损失值
     */
    private double computeMulticlassSoftmaxLoss(IMatrix predictions, IMatrix labels) {
        int numSamples = predictions.rows();
        double totalLoss = 0.0;
        
        for (int i = 0; i < numSamples; i++) {
            // 获取当前样本的预测值和标签
            IVector predRow = predictions.getRow(i);
            IVector labelRow = labels.getRow(i);
            
            // 计算softmax概率
            IVector<Double> probs = softmax(predRow);
            
            // 使用向量运算计算交叉熵损失
            // 先对概率应用log函数，确保数值稳定性
            IVector logProbs = probs.apply(p -> Math.log(Math.max(EPS, p.doubleValue())));
            
            // 使用向量内积计算损失：-sum(label * log(prob))
            double sampleLoss = -labelRow.dot(logProbs).doubleValue();
            totalLoss += sampleLoss;
        }
        
        return totalLoss / numSamples;
    }
    
    /**
     * 计算多分类softmax梯度
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 梯度矩阵
     */
    private IMatrix computeMulticlassSoftmaxGradients(IMatrix predictions, IMatrix labels) {
        int numSamples = predictions.rows();
        IMatrix gradients = Linalg.zeros(numSamples, numClasses);
        
        for (int i = 0; i < numSamples; i++) {
            // 获取当前样本的预测值和标签
            IVector predRow = predictions.getRow(i);
            IVector labelRow = labels.getRow(i);
            
            // 计算softmax概率
            IVector probs = softmax(predRow);
            
            // 使用向量运算计算梯度：softmax(pred) - label
            IVector gradient = probs.sub(labelRow);
            
            // 将梯度向量设置到结果矩阵中
            for (int j = 0; j < numClasses; j++) {
                gradients.set(i, j, gradient.get(j).doubleValue());
            }
        }
        
        return gradients;
    }
    
    /**
     * 计算多分类softmax海塞矩阵
     * @param predictions 预测值
     * @param labels 真实标签
     * @return 海塞矩阵对角线元素
     */
    private IMatrix computeMulticlassSoftmaxHessians(IMatrix predictions, IMatrix labels) {
        int numSamples = predictions.rows();
        IMatrix hessians = Linalg.zeros(numSamples, numClasses);
        
        for (int i = 0; i < numSamples; i++) {
            // 获取当前样本的预测值
            IVector predRow = predictions.getRow(i);
            
            // 计算softmax概率
            IVector probs = softmax(predRow);
            
            // 使用向量运算计算海塞矩阵对角线元素：softmax(pred) * (1 - softmax(pred))
            IVector ones = Linalg.ones(numClasses);
            IVector oneMinusProbs = ones.sub(probs);
            IVector<Double> hessianRow = probs.multiply(oneMinusProbs);
            
            // 应用数值稳定性约束
            hessianRow = hessianRow.apply(h -> Math.max(EPS, h.doubleValue()));
            
            // 将海塞矩阵向量设置到结果矩阵中
            for (int j = 0; j < numClasses; j++) {
                hessians.set(i, j, hessianRow.get(j).doubleValue());
            }
        }
        
        return hessians;
    }
    
    /**
     * Sigmoid激活函数
     * @param x 输入值
     * @return sigmoid(x)
     */
    private double sigmoid(double x) {
        // 数值稳定的sigmoid实现
        if (x > 0) {
            double exp_neg_x = Math.exp(-x);
            return 1.0 / (1.0 + exp_neg_x);
        } else {
            double exp_x = Math.exp(x);
            return exp_x / (1.0 + exp_x);
        }
    }
    
    /**
     * Softmax激活函数
     * @param x 输入向量
     * @return softmax(x)
     */
    private IVector softmax(IVector x) {
        // 使用向量运算优化softmax计算
        
        // 找到最大值以提高数值稳定性
        double maxVal = x.max().doubleValue();
        
        // 计算exp(x - max) - 使用向量运算
        IVector shifted = x.subScalar(maxVal);
        IVector expValues = shifted.exp();
        
        // 归一化 - 使用向量运算
        double sum = expValues.sum().doubleValue();
        return expValues.divideByScalar(sum);
    }
    
    /**
     * 将预测值转换为概率
     * @param predictions 预测值
     * @return 概率矩阵
     */
    public IMatrix predictProba(IMatrix predictions) {
        switch (lossType) {
            case BINARY_LOGISTIC:
                return predictBinaryProba(predictions);
            case MULTICLASS_SOFTMAX:
                return predictMulticlassProba(predictions);
            default:
                throw new IllegalArgumentException("Unsupported loss type: " + lossType);
        }
    }
    
    /**
     * 二分类概率预测
     * @param predictions 预测值
     * @return 概率矩阵
     */
    private IMatrix predictBinaryProba(IMatrix predictions) {
        int numSamples = predictions.rows();
        IMatrix probabilities = Linalg.zeros(numSamples, 2);
        
        for (int i = 0; i < numSamples; i++) {
            double pred = predictions.get(i, 0).doubleValue();
            double prob1 = sigmoid(pred);
            double prob0 = 1.0 - prob1;
            
            probabilities.set(i, 0, prob0);
            probabilities.set(i, 1, prob1);
        }
        
        return probabilities;
    }
    
    /**
     * 多分类概率预测
     * @param predictions 预测值
     * @return 概率矩阵
     */
    private IMatrix predictMulticlassProba(IMatrix predictions) {
        int numSamples = predictions.rows();
        IMatrix probabilities = Linalg.zeros(numSamples, numClasses);
        
        for (int i = 0; i < numSamples; i++) {
            IVector predRow = predictions.getRow(i);
            IVector probs = softmax(predRow);
            
            for (int j = 0; j < numClasses; j++) {
                probabilities.set(i, j, probs.get(j).doubleValue());
            }
        }
        
        return probabilities;
    }
    
    // ==================== Getters and Setters ====================
    
    public LossType getLossType() {
        return lossType;
    }
    
    public int getNumClasses() {
        return numClasses;
    }
    
    public void setLossType(LossType lossType) {
        this.lossType = lossType;
    }
    
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }
}