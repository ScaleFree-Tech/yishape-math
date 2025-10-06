package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.newton.RereLBFGS;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * 统一逻辑回归实现类
 * <p>
 * 本类实现了逻辑回归算法，自动检测并支持二分类和多分类问题：
 * - 二分类：使用sigmoid函数，输出单个概率值
 * - 多分类：使用softmax函数，输出多个类别的概率分布
 * </p>
 * 
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class RereLogisticRegression implements IClassification, IGradientFunction, IObjectiveFunction, ISerializableModel {
    
    private static final long serialVersionUID = 1L;
    
    // ==================== 模型参数 ====================
    
    /** 权重矩阵：多分类时每行对应一个类别，二分类时只有一行 */
    private IMatrix weights;
    
    /** 偏置向量：多分类时每个类别一个偏置，二分类时只有一个偏置 */
    private IVector bias;
    
    /** 学习率 */
    private double learningRate = 0.01;
    
    /** 最大迭代次数 */
    private int maxIterations = 1000;
    
    /** 收敛阈值 */
    private double tolerance = 1e-6;
    
    /** L1正则化系数（λ₁） */
    private double lambda1 = 0.0;
    
    /** L2正则化系数（λ₂） */
    private double lambda2 = 0.0;
    
    /**
     * 正则化类型枚举
     */
    public enum RegularizationType {
        /** 无正则化 */
        NONE,
        /** L1正则化（Lasso） */
        L1,
        /** L2正则化（Ridge） */
        L2,
        /** ElasticNet正则化（L1 + L2的组合） */
        ELASTIC_NET
    }
    
    /** 正则化类型 */
    private RegularizationType regularizationType = RegularizationType.NONE;
    
    /** 标签映射：将字符串标签映射为数值 */
    private Map<String, Integer> labelMapping;
    
    /** 反向标签映射：将数值映射回字符串标签 */
    private Map<Integer, String> reverseLabelMapping;
    
    /** 训练特征矩阵 */
    private IMatrix trainingFeatures;
    
    /** 训练标签数组（数值化后） */
    private int[] trainingLabels;
    
    /** 是否已训练 */
    private boolean isTrained = false;
    
    /** 分类类型：true为二分类，false为多分类 */
    private boolean isBinaryClassification = true;
    
    /** 类别数量 */
    private int numClasses = 2;
    
    /** 特征维度 */
    private int featureDimension = 0;
    
    // ==================== 构造函数 ====================
    
    /**
     * 默认构造函数
     */
    public RereLogisticRegression() {
        this.labelMapping = new HashMap<>();
        this.reverseLabelMapping = new HashMap<>();
    }
    
    /**
     * 参数化构造函数
     */
    public RereLogisticRegression(double learningRate, int maxIterations, double tolerance, 
                                 double lambda1, double lambda2) {
        this();
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.tolerance = tolerance;
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.regularizationType = inferRegularizationType(lambda1, lambda2);
    }
    
    // ==================== 核心训练方法 ====================
    
    @Override
    public LogisticRegressionResult fit(IMatrix feature, String[] labels) {
        if (feature == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签数组不能为null");
        }
        
        if (feature.getRowNum() != labels.length) {
            throw new IllegalArgumentException("特征矩阵行数与标签数组长度不匹配");
        }
        
        if (feature.getRowNum() == 0) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        
        // 保存训练数据
        this.trainingFeatures = feature;
        this.featureDimension = feature.getColNum();
        
        // 标签预处理：将字符串标签转换为数值，并检测分类类型
        preprocessLabels(labels);
        
        // 初始化模型参数
        initializeParameters();
        
        // 使用LBFGS优化器训练模型
        trainWithOptimizer();
        
        // 标记模型已训练
        this.isTrained = true;
        
        // 计算最终损失
        double finalLoss = computeObjective(createParameterVector());
        
        // 创建并返回训练结果
        LogisticRegressionResult result = new LogisticRegressionResult();
        
        // 正确设置权重和偏置
        if (isBinaryClassification) {
            result.setWeights(weights.getRow(0));
            result.setBias(Linalg.vector(new double[]{(double)bias.get(0)}));
        } else {
            // 对于多分类，返回第一个类别的权重和偏置作为默认值
            // 注意：完整权重信息存储在模型内部，可通过getWeights()获取
            result.setWeights(weights.getRow(0));
            result.setBias(Linalg.vector(new double[]{(double)bias.get(0)}));
        }
        
        result.setLoss(finalLoss);
        
        return result;
    }
    
    /**
     * 使用优化器训练模型
     */
    private void trainWithOptimizer() {
        // 创建初始参数向量
        IVector initParams = createParameterVector();
        
        // 创建LBFGS优化器
        IOptimizer optimizer = new RereLBFGS();
        
        // 执行优化
        var optimizationResult = optimizer.optimize(initParams, this, this);
        
        // 从优化结果中提取参数
        IVector optimalParams = optimizationResult.getOptimalPoint();
        extractParametersFromVector(optimalParams);
    }
    
    // ==================== 预测方法 ====================
    
    @Override
    public String predict(IVector x) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        if (x == null) {
            throw new IllegalArgumentException("输入特征向量不能为null");
        }
        
        if (x.length() != featureDimension) {
            throw new IllegalArgumentException("输入特征维度与训练特征维度不匹配");
        }
        
        if (isBinaryClassification) {
            // 二分类：使用sigmoid函数
            double probability = predictProbability(x);
            return probability >= 0.5 ? reverseLabelMapping.get(1) : reverseLabelMapping.get(0);
        } else {
            // 多分类：使用softmax函数
            double[] probabilities = predictProbabilities(x);
            // 使用向量的argMax方法找到最大概率的类别
            IVector<Double> probVector = Linalg.vector(probabilities);
            int predictedClass = probVector.argMax();
            return reverseLabelMapping.get(predictedClass);
        }
    }
    
    /**
     * 预测样本属于正类的概率（仅适用于二分类）
     */
    public double predictProbability(IVector x) {
        if (!isBinaryClassification) {
            throw new IllegalStateException("predictProbability方法仅适用于二分类模型");
        }
        
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        // 使用向量内积计算线性组合
        double z = (double)weights.getRow(0).innerProduct(x) + (double)bias.get(0);
        
        // 应用sigmoid函数：P(y=1|x) = 1 / (1 + e^(-z))
        return sigmoid(z);
    }
    
    /**
     * 预测样本属于每个类别的概率（适用于多分类）
     * @param x
     * @return 
     */
    public double[] predictProbabilities(IVector x) {
        if (isBinaryClassification) {
            throw new IllegalStateException("predictProbabilities方法仅适用于多分类模型");
        }
        
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        // 使用矩阵运算计算所有类别的线性组合，替代手动循环
        // 计算 W * x + b
        IVector<Double> logits = weights.mmul(x).add(bias);
        
        // 应用softmax函数
        return softmax(logits.toDoubleArray());
    }
    
    /**
     * 内部预测概率方法（不检查训练状态）
     */
    private Object predictProbabilityInternal(IVector x) {
        if (isBinaryClassification) {
            // 二分类：返回单个概率值
            // 使用向量内积计算线性组合
            double z = (double)weights.getRow(0).innerProduct(x) + (double)bias.get(0);
            return sigmoid(z);
        } else {
            // 多分类：返回概率数组
            // 使用矩阵运算计算所有类别的线性组合，替代手动循环
            IVector<Double> logits = weights.mmul(x).add(bias);
            return softmax(logits.toDoubleArray());
        }
    }
    
    /**
     * 批量预测
     */
    public String[] predictBatch(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        if (features == null) {
            throw new IllegalArgumentException("特征矩阵不能为null");
        }
        
        if (features.getColNum() != featureDimension) {
            throw new IllegalArgumentException("特征维度与训练特征维度不匹配");
        }
        
        if (isBinaryClassification) {
            // 二分类：使用矩阵乘法进行批量预测
            // 计算线性组合：Z = X * W^T + b
            IMatrix weightMatrix = weights.transpose(); // 转置权重矩阵以匹配维度
            IMatrix linearOutput = features.mmul(weightMatrix);
            
            // 添加偏置值到每一行
            final double biasValue = (double)bias.get(0);
            linearOutput = linearOutput.apply(x -> (double)x + biasValue);
            
            // 应用sigmoid函数
            IMatrix probabilities = linearOutput.apply(x -> sigmoid((double)x));
            
            // 转换为预测标签
            // 使用向量化操作替代手动循环
            String[] predictions = new String[features.getRowNum()];
            IVector<Double> probVector = probabilities.getColumn(0);
            for (int i = 0; i < features.getRowNum(); i++) {
                double prob = (double)probVector.get(i);
                predictions[i] = prob >= 0.5 ? reverseLabelMapping.get(1) : reverseLabelMapping.get(0);
            }
            return predictions;
        } else {
            // 多分类：使用矩阵运算进行批量预测
            // 计算线性组合：Z = X * W^T + B (广播加法)
            IMatrix weightMatrix = weights.transpose(); // 转置权重矩阵以匹配维度
            IMatrix linearOutput = features.mmul(weightMatrix);
            
            // 使用广播加法添加偏置，替代手动循环
            linearOutput = linearOutput.broadcastAddRow(bias);
            
            // 应用softmax函数（按行）
            // 先计算指数值
            IMatrix expOutput = linearOutput.apply(x -> Math.exp((double)x));
            
            // 对每行进行归一化处理
            for (int i = 0; i < expOutput.getRowNum(); i++) {
                IVector<Double> row = expOutput.getRow(i);
                double rowSum = (double)row.sum();
                if (rowSum != 0) {
                    IVector<Double> normalizedRow = row.divideByScalar(rowSum);
                    expOutput.setRow(i, normalizedRow);
                }
            }
            
            // 转换为预测标签
            String[] result = new String[features.getRowNum()];
            for (int i = 0; i < features.getRowNum(); i++) {
                IVector<Double> row = expOutput.getRow(i);
                int predictedClass = row.argMax(); // 使用argMax方法找到最大概率的类别
                result[i] = reverseLabelMapping.get(predictedClass);
            }
            return result;
        }
    }
    
    // ==================== 损失函数和梯度计算 ====================
    
    @Override
    public double computeObjective(IVector x) {
        if (trainingFeatures == null || trainingLabels == null) {
            throw new IllegalStateException("训练数据未设置");
        }
        
        // 从参数向量中提取权重和偏置
        extractParametersFromVector(x);
        
        double totalLoss = 0.0;
        int m = trainingFeatures.getRowNum();
        
        // 计算每个样本的损失
        for (int i = 0; i < m; i++) {
            IVector sample = (IVector)trainingFeatures.getRow(i);
            int label = trainingLabels[i];
            
            if (isBinaryClassification) {
                // 二分类：使用sigmoid和交叉熵损失
                double probability = (Double) predictProbabilityInternal(sample);
                double sampleLoss = -label * Math.log(probability + 1e-15) - 
                                  (1 - label) * Math.log(1 - probability + 1e-15);
                totalLoss += sampleLoss;
            } else {
                // 多分类：使用softmax和交叉熵损失
                double[] probabilities = (double[]) predictProbabilityInternal(sample);
                double sampleLoss = -Math.log(probabilities[label] + 1e-15);
                totalLoss += sampleLoss;
            }
        }
        
        // 平均损失
        totalLoss /= m;
        
        // 添加正则化项
        totalLoss += computeRegularizationTerm();
        
        return totalLoss;
    }
    
    @Override
    public IVector computeGradient(IVector x) {
        if (trainingFeatures == null || trainingLabels == null) {
            throw new IllegalStateException("训练数据未设置");
        }
        
        // 从参数向量中提取权重和偏置
        extractParametersFromVector(x);
        
        int m = trainingFeatures.getRowNum();
        
        if (isBinaryClassification) {
            // 二分类梯度计算
            return computeBinaryClassificationGradient(m);
        } else {
            // 多分类梯度计算
            return computeMulticlassClassificationGradient(m);
        }
    }
    
    /**
     * 计算二分类梯度
     */
    private IVector computeBinaryClassificationGradient(int m) {
        // 使用矩阵运算计算梯度，而不是手动循环
        
        // 计算所有样本的预测概率
        IMatrix predictions = trainingFeatures.mmul(weights.transpose());
        final double biasValue = (double)bias.get(0);
        predictions = predictions.apply(x -> sigmoid((double)x + biasValue));
        
        // 计算误差矩阵 (predictions - labels)
        // 创建标签矩阵
        double[][] labelArray = new double[m][1];
        for (int i = 0; i < m; i++) {
            labelArray[i][0] = trainingLabels[i];
        }
        IMatrix labelsMatrix = Linalg.matrix(labelArray);
        
        // 使用矩阵运算计算误差
        IMatrix errors = predictions.sub(labelsMatrix);
        
        // 计算权重梯度：features^T * errors / m
        IMatrix weightGradientsMatrix = trainingFeatures.transpose().mmul(errors).divideByScalar((double)m);
        IVector weightGradients = weightGradientsMatrix.getColumn(0);
        
        // 计算偏置梯度：sum(errors) / m
        double biasGradient = (double)errors.sum() / m;
        
        // 添加正则化梯度
        // 直接使用向量操作，避免手动循环
        double[] weightGradArray = weightGradients.toDoubleArray();
        addRegularizationGradients(weightGradArray);
        
        // 重新创建权重梯度向量
        weightGradients = Linalg.vector(weightGradArray);
        
        // 创建梯度向量
        return createGradientVector(weightGradArray, biasGradient);
    }
    
    /**
     * 计算多分类梯度
     */
    private IVector computeMulticlassClassificationGradient(int m) {
        // 使用矩阵运算计算梯度，而不是手动循环
        
        // 计算所有样本的线性输出
        IMatrix linearOutput = trainingFeatures.mmul(weights.transpose());
        
        // 添加偏置到每一行
        // 使用广播加法替代手动循环
        linearOutput = linearOutput.broadcastAddRow(bias);
        
        // 应用softmax函数到每一行
        IMatrix probabilities = linearOutput.copy();
        // 先计算指数值
        probabilities = probabilities.apply(x -> Math.exp((double)x));
        
        // 对每行进行归一化处理
        for (int i = 0; i < probabilities.getRowNum(); i++) {
            IVector<Double> row = probabilities.getRow(i);
            double rowSum = (double)row.sum();
            if (rowSum != 0) {
                IVector<Double> normalizedRow = row.divideByScalar(rowSum);
                probabilities.setRow(i, normalizedRow);
            }
        }
        
        // 计算误差矩阵 (probabilities - one_hot_labels)
        // 创建one-hot编码的标签矩阵
        double[][] labelArray = new double[m][numClasses];
        for (int i = 0; i < m; i++) {
            int trueLabel = trainingLabels[i];
            labelArray[i][trueLabel] = 1.0;
        }
        IMatrix oneHotLabels = Linalg.matrix(labelArray);
        
        // 使用矩阵运算计算误差
        IMatrix errors = probabilities.sub(oneHotLabels);
        
        // 计算权重梯度：features^T * errors / m
        IMatrix weightGradientsMatrix = trainingFeatures.transpose().mmul(errors).divideByScalar((double)m);
        
        // 计算偏置梯度：colSums(errors) / m
        // 使用向量的toDoubleArray方法替代手动循环
        IVector biasGradientsVector = errors.colSums().divideByScalar((double)m);
        double[] biasGradients = biasGradientsVector.toDoubleArray();
        
        // 转换权重梯度矩阵为数组格式
        // 使用矩阵的转置和flatten方法替代手动循环
        IMatrix weightGradientsMatrixT = weightGradientsMatrix.transpose();
        double[] weightGradientsFlat = weightGradientsMatrixT.flatten().toDoubleArray();
        
        // 重塑为二维数组格式
        double[][] weightGradients = new double[numClasses][featureDimension];
        for (int k = 0; k < numClasses; k++) {
            System.arraycopy(weightGradientsFlat, k * featureDimension, weightGradients[k], 0, featureDimension);
        }
        
        // 添加正则化梯度
        addRegularizationGradients(weightGradients);
        
        // 创建梯度向量
        return createGradientVector(weightGradients, biasGradients);
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 标签预处理
     */
    private void preprocessLabels(String[] labels) {
        labelMapping.clear();
        reverseLabelMapping.clear();
        
        int nextLabel = 0;
        for (String label : labels) {
            if (!labelMapping.containsKey(label)) {
                labelMapping.put(label, nextLabel);
                reverseLabelMapping.put(nextLabel, label);
                nextLabel++;
            }
        }
        
        this.numClasses = labelMapping.size();
        
        if (numClasses < 2) {
            throw new IllegalArgumentException("至少需要2个类别，当前标签数量：" + numClasses);
        }
        
        // 判断分类类型
        this.isBinaryClassification = (numClasses == 2);
        
        // 转换为数值标签数组
        this.trainingLabels = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            this.trainingLabels[i] = labelMapping.get(labels[i]);
        }
    }
    
    /**
     * 初始化模型参数
     */
    private void initializeParameters() {
        Random random = new Random();
        
        if (isBinaryClassification) {
            // 二分类：权重向量 + 单个偏置
            double scale = Math.sqrt(2.0 / featureDimension);
            
            // 使用Linalg工厂方法创建随机矩阵
            this.weights = Linalg.randn(1, featureDimension, 0.0, scale);
            this.bias = Linalg.zeros(1);
        } else {
            // 多分类：权重矩阵 + 偏置向量
            double scale = Math.sqrt(2.0 / featureDimension);
            
            // 使用Linalg工厂方法创建随机矩阵
            this.weights = Linalg.randn(numClasses, featureDimension, 0.0, scale);
            this.bias = Linalg.zeros(numClasses);
        }
    }
    
    /**
     * 创建参数向量
     */
    private IVector createParameterVector() {
        if (isBinaryClassification) {
            // 二分类：[w1, w2, ..., wn, b]
            // 使用矩阵和向量的连接操作来创建参数向量
            IVector weightVector = weights.getRow(0);
            IVector biasScalar = Linalg.vector(new double[]{(double)bias.get(0)});
            return weightVector.concat(biasScalar);
        } else {
            // 多分类：[w11, w12, ..., w1n, w21, w22, ..., w2n, ..., wk1, wk2, ..., wkn, b1, b2, ..., bk]
            // 将权重矩阵展平并连接偏置向量
            IVector weightVector = weights.flatten();
            return weightVector.concat(bias);
        }
    }
    
    /**
     * 从参数向量中提取权重和偏置
     */
    private void extractParametersFromVector(IVector paramVector) {
        if (isBinaryClassification) {
            // 二分类：提取权重向量和单个偏置
            int n = paramVector.length() - 1;
            
            // 使用切片操作提取权重和偏置
            IVector weightVector = paramVector.slice(0, n);
            this.weights = weightVector.asColumnVector().transpose();
            
            // 提取偏置
            double biasValue = (double)paramVector.get(n);
            this.bias = Linalg.vector(new double[]{biasValue});
        } else {
            // 多分类：提取权重矩阵和偏置向量
            int weightElements = numClasses * featureDimension;
            
            // 使用切片操作提取权重和偏置
            IVector weightVector = paramVector.slice(0, weightElements);
            IVector biasVector = paramVector.slice(weightElements, paramVector.length());
            
            // 使用reshape方法重塑权重向量为矩阵
            this.weights = weightVector.reshape(numClasses, featureDimension);
            this.bias = biasVector;
        }
    }
    
    /**
     * 创建梯度向量（二分类）
     */
    private IVector createGradientVector(double[] weightGradients, double biasGradient) {
        // 使用Linalg工厂方法创建向量并连接
        IVector weightGradVector = Linalg.vector(weightGradients);
        IVector biasGradVector = Linalg.vector(new double[]{biasGradient});
        return weightGradVector.concat(biasGradVector);
    }
    
    /**
     * 创建梯度向量（多分类）
     */
    private IVector createGradientVector(double[][] weightGradients, double[] biasGradients) {
        // 使用Linalg工厂方法创建矩阵和向量并连接
        IMatrix weightGradMatrix = Linalg.matrix(weightGradients);
        IVector weightGradVector = weightGradMatrix.flatten();
        IVector biasGradVector = Linalg.vector(biasGradients);
        return weightGradVector.concat(biasGradVector);
    }
    
    /**
     * 根据lambda1和lambda2的值自动判断正则化类型
     */
    private RegularizationType inferRegularizationType(double lambda1, double lambda2) {
        if (lambda1 > 0 && lambda2 > 0) {
            return RegularizationType.ELASTIC_NET;
        } else if (lambda1 > 0 && lambda2 <= 0) {
            return RegularizationType.L1;
        } else if (lambda1 <= 0 && lambda2 > 0) {
            return RegularizationType.L2;
        } else {
            return RegularizationType.NONE;
        }
    }
    
    /**
     * 计算正则化项
     */
    private double computeRegularizationTerm() {
        double regularizationTerm = 0.0;
        
        switch (regularizationType) {
            case L1:
                // 使用矩阵的L1范数计算正则化项
                regularizationTerm = lambda1 * (double)weights.abs().sum();
                break;
            case L2:
                // 使用矩阵的Frobenius范数计算L2正则化项
                double frobenius = (double)weights.frobeniusNorm();
                regularizationTerm = lambda2 * frobenius * frobenius / 2.0;
                break;
            case ELASTIC_NET:
                // L1部分
                double l1Term = (double)weights.abs().sum();
                // L2部分
                double frobeniusNorm = (double)weights.frobeniusNorm();
                double l2Term = frobeniusNorm * frobeniusNorm;
                regularizationTerm = lambda1 * l1Term + lambda2 * l2Term / 2.0;
                break;
            case NONE:
            default:
                regularizationTerm = 0.0;
                break;
        }
        
        return regularizationTerm;
    }
    
    /**
     * 添加正则化梯度（二分类）
     */
    private void addRegularizationGradients(double[] weightGradients) {
        switch (regularizationType) {
            case L1:
                // 使用向量操作计算L1正则化梯度
                IVector weightsRow = weights.getRow(0);
                IVector signVector = weightsRow.sign(); // 使用新添加的sign方法
                IVector l1Grad = signVector.multiplyScalar(lambda1);
                IVector weightGradVector = Linalg.vector(weightGradients);
                IVector result = weightGradVector.add(l1Grad);
                // 使用toDoubleArray方法替代手动循环
                double[] resultArray = result.toDoubleArray();
                System.arraycopy(resultArray, 0, weightGradients, 0, weightGradients.length);
                break;
            case L2:
                // 使用向量操作计算L2正则化梯度
                IVector l2Grad = weights.getRow(0).multiplyScalar(lambda2);
                IVector weightGradVector2 = Linalg.vector(weightGradients);
                IVector result2 = weightGradVector2.add(l2Grad);
                // 使用toDoubleArray方法替代手动循环
                double[] resultArray2 = result2.toDoubleArray();
                System.arraycopy(resultArray2, 0, weightGradients, 0, weightGradients.length);
                break;
            case ELASTIC_NET:
                // 使用向量操作计算ElasticNet正则化梯度
                IVector weightsRow2 = weights.getRow(0);
                IVector signVector2 = weightsRow2.sign(); // 使用新添加的sign方法
                IVector l1Component = signVector2.multiplyScalar(lambda1);
                IVector l2Component = weights.getRow(0).multiplyScalar(lambda2);
                IVector elasticGrad = l1Component.add(l2Component);
                IVector weightGradVector3 = Linalg.vector(weightGradients);
                IVector result3 = weightGradVector3.add(elasticGrad);
                // 使用toDoubleArray方法替代手动循环
                double[] resultArray3 = result3.toDoubleArray();
                System.arraycopy(resultArray3, 0, weightGradients, 0, weightGradients.length);
                break;
            case NONE:
            default:
                // 无正则化，不添加梯度
                break;
        }
    }
    
    /**
     * 添加正则化梯度（多分类）
     */
    private void addRegularizationGradients(double[][] weightGradients) {
        switch (regularizationType) {
            case L1:
                // 使用矩阵操作计算L1正则化梯度
                IMatrix signMatrix = weights.sign(); // 使用sign方法
                IMatrix l1Grad = signMatrix.multiplyScalar(lambda1);
                IMatrix weightGradMatrix = Linalg.matrix(weightGradients);
                IMatrix result = weightGradMatrix.add(l1Grad);
                // 使用toDoubleArray方法替代手动循环
                double[][] resultArray = new double[numClasses][featureDimension];
                for (int k = 0; k < numClasses; k++) {
                    IVector<Double> row = result.getRow(k);
                    resultArray[k] = row.toDoubleArray();
                }
                // 复制结果回原数组
                for (int k = 0; k < numClasses; k++) {
                    System.arraycopy(resultArray[k], 0, weightGradients[k], 0, featureDimension);
                }
                break;
            case L2:
                // 使用矩阵操作计算L2正则化梯度
                IMatrix l2Grad = weights.multiplyScalar(lambda2);
                IMatrix weightGradMatrix2 = Linalg.matrix(weightGradients);
                IMatrix result2 = weightGradMatrix2.add(l2Grad);
                // 使用toDoubleArray方法替代手动循环
                double[][] resultArray2 = new double[numClasses][featureDimension];
                for (int k = 0; k < numClasses; k++) {
                    IVector<Double> row = result2.getRow(k);
                    resultArray2[k] = row.toDoubleArray();
                }
                // 复制结果回原数组
                for (int k = 0; k < numClasses; k++) {
                    System.arraycopy(resultArray2[k], 0, weightGradients[k], 0, featureDimension);
                }
                break;
            case ELASTIC_NET:
                // 使用矩阵操作计算ElasticNet正则化梯度
                IMatrix signMatrix2 = weights.sign(); // 使用sign方法
                IMatrix l1Component = signMatrix2.multiplyScalar(lambda1);
                IMatrix l2Component = weights.multiplyScalar(lambda2);
                IMatrix elasticGrad = l1Component.add(l2Component);
                IMatrix weightGradMatrix3 = Linalg.matrix(weightGradients);
                IMatrix result3 = weightGradMatrix3.add(elasticGrad);
                // 使用toDoubleArray方法替代手动循环
                double[][] resultArray3 = new double[numClasses][featureDimension];
                for (int k = 0; k < numClasses; k++) {
                    IVector<Double> row = result3.getRow(k);
                    resultArray3[k] = row.toDoubleArray();
                }
                // 复制结果回原数组
                for (int k = 0; k < numClasses; k++) {
                    System.arraycopy(resultArray3[k], 0, weightGradients[k], 0, featureDimension);
                }
                break;
            case NONE:
            default:
                // 无正则化，不添加梯度
                break;
        }
    }
    
    /**
     * Sigmoid激活函数
     */
    private double sigmoid(double z) {
        // 处理数值稳定性
        if (z >= 0) {
            double expNegZ = Math.exp(-z);
            return 1.0 / (1.0 + expNegZ);
        } else {
            double expZ = Math.exp(z);
            return expZ / (1.0 + expZ);
        }
    }
    
    /**
     * Softmax激活函数
     */
    private double[] softmax(double[] logits) {
        // 使用向量操作来实现softmax函数，提高数值稳定性和效率
        IVector<Double> logitVector = Linalg.vector(logits);
        
        // 找到最大值以提高数值稳定性
        double maxLogit = (double)logitVector.max();
        
        // 计算指数并求和: exp(logits - maxLogit)
        IVector<Double> shifted = logitVector.subScalar(maxLogit);
        IVector<Double> expVector = shifted.exp();
        double sum = (double)expVector.sum();
        
        // 避免除零错误并归一化
        if (sum == 0) {
            // 如果所有概率都接近零，返回均匀分布
            IVector<Double> uniform = Linalg.ones(numClasses).divideByScalar((double)numClasses);
            return uniform.toDoubleArray();
        } else {
            // 归一化
            IVector<Double> probabilities = expVector.divideByScalar(sum);
            return probabilities.toDoubleArray();
        }
    }
    
    // ==================== Getter和Setter方法 ====================
    
    /**
     * 获取权重矩阵
     */
    public IMatrix getWeights() {
        return weights;
    }
    
    /**
     * 获取偏置向量
     */
    public IVector getBias() {
        return bias;
    }
    
    /**
     * 检查是否为二分类
     */
    public boolean isBinaryClassification() {
        return isBinaryClassification;
    }
    
    /**
     * 获取类别数量
     */
    public int getNumClasses() {
        return numClasses;
    }
    
    /**
     * 获取特征维度
     */
    public int getFeatureDimension() {
        return featureDimension;
    }
    
    /**
     * 获取学习率
     */
    public double getLearningRate() {
        return learningRate;
    }
    
    /**
     * 设置学习率
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * 获取最大迭代次数
     */
    public int getMaxIterations() {
        return maxIterations;
    }
    
    /**
     * 设置最大迭代次数
     */
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }
    
    /**
     * 获取收敛阈值
     */
    public double getTolerance() {
        return tolerance;
    }
    
    /**
     * 设置收敛阈值
     */
    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }
    
    /**
     * 获取L1正则化系数
     */
    public double getLambda1() {
        return lambda1;
    }
    
    /**
     * 设置L1正则化系数
     */
    public void setLambda1(double lambda1) {
        this.lambda1 = lambda1;
        this.regularizationType = inferRegularizationType(this.lambda1, this.lambda2);
    }
    
    /**
     * 获取L2正则化系数
     */
    public double getLambda2() {
        return lambda2;
    }
    
    /**
     * 设置L2正则化系数
     */
    public void setLambda2(double lambda2) {
        this.lambda2 = lambda2;
        this.regularizationType = inferRegularizationType(this.lambda1, this.lambda2);
    }
    
    /**
     * 设置正则化参数
     */
    public void setRegularization(double lambda1, double lambda2) {
        this.lambda1 = lambda1;
        this.lambda2 = lambda2;
        this.regularizationType = inferRegularizationType(lambda1, lambda2);
    }
    
    /**
     * 获取正则化类型
     */
    public RegularizationType getRegularizationType() {
        return regularizationType;
    }
    
    /**
     * 获取正则化描述信息
     */
    public String getRegularizationDescription() {
        switch (regularizationType) {
            case L1:
                return String.format("L1正则化 (λ₁ = %.4f)", lambda1);
            case L2:
                return String.format("L2正则化 (λ₂ = %.4f)", lambda2);
            case ELASTIC_NET:
                return String.format("ElasticNet正则化 (λ₁ = %.4f, λ₂ = %.4f)", lambda1, lambda2);
            case NONE:
            default:
                return "无正则化";
        }
    }
    
    /**
     * 检查模型是否已训练
     */
    public boolean isTrained() {
        return isTrained;
    }
    
    /**
     * 获取标签映射
     */
    public Map<String, Integer> getLabelMapping() {
        return new HashMap<>(labelMapping);
    }
    
    /**
     * 获取模型类型描述
     */
    public String getModelTypeDescription() {
        if (isBinaryClassification) {
            return "二分类逻辑回归";
        } else {
            return String.format("多分类逻辑回归 (%d类)", numClasses);
        }
    }
    
    /**
     * 将模型保存在本地
     * @param path 保存路径
     */
    @Override
    public void save(String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
