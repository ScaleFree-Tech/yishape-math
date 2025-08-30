package com.reremouse.lab.math.ml.cls;

import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.RereMatrix;
import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.optimize.IGradientFunction;
import com.reremouse.lab.math.optimize.IObjectiveFunction;
import com.reremouse.lab.math.optimize.IOptimizer;
import com.reremouse.lab.math.optimize.RereLBFGS;
import com.reremouse.lab.util.Tuple2;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

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
public class RereLogisticRegression implements IClassification, IGradientFunction, IObjectiveFunction {
    
    // ==================== 模型参数 ====================
    
    /** 权重矩阵：多分类时每行对应一个类别，二分类时只有一行 */
    private IMatrix weights;
    
    /** 偏置向量：多分类时每个类别一个偏置，二分类时只有一个偏置 */
    private IVector bias;
    
    /** 学习率 */
    private float learningRate = 0.01f;
    
    /** 最大迭代次数 */
    private int maxIterations = 1000;
    
    /** 收敛阈值 */
    private float tolerance = 1e-6f;
    
    /** L1正则化系数（λ₁） */
    private float lambda1 = 0.0f;
    
    /** L2正则化系数（λ₂） */
    private float lambda2 = 0.0f;
    
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
    public RereLogisticRegression(float learningRate, int maxIterations, float tolerance, 
                                 float lambda1, float lambda2) {
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
        float finalLoss = computeObjective(createParameterVector());
        
        // 创建并返回训练结果
        LogisticRegressionResult result = new LogisticRegressionResult();
        
        // 对于多分类，使用第一个类别的权重作为兼容
        if (isBinaryClassification) {
            result.setWeights(weights.getRow(0));
        } else {
            result.setWeights(weights.getRow(0));
        }
        
        // 设置偏置（多分类时使用第一个偏置作为兼容）
        if (isBinaryClassification) {
            result.setBias(new RereVector(new float[]{bias.get(0)}));
        } else {
            result.setBias(new RereVector(new float[]{bias.get(0)}));
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
        Tuple2<Float, IVector> optimizationResult = optimizer.optimize(initParams, this, this);
        
        // 从优化结果中提取参数
        IVector optimalParams = optimizationResult._2;
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
            float probability = predictProbability(x);
            return probability >= 0.5 ? reverseLabelMapping.get(1) : reverseLabelMapping.get(0);
        } else {
            // 多分类：使用softmax函数
            float[] probabilities = predictProbabilities(x);
            int predictedClass = 0;
            float maxProb = probabilities[0];
            for (int i = 1; i < numClasses; i++) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i];
                    predictedClass = i;
                }
            }
            return reverseLabelMapping.get(predictedClass);
        }
    }
    
    /**
     * 预测样本属于正类的概率（仅适用于二分类）
     */
    public float predictProbability(IVector x) {
        if (!isBinaryClassification) {
            throw new IllegalStateException("predictProbability方法仅适用于二分类模型");
        }
        
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        // 计算线性组合：z = w^T * x + b
        float z = weights.getRow(0).innerProduct(x) + bias.get(0);
        
        // 应用sigmoid函数：P(y=1|x) = 1 / (1 + e^(-z))
        return sigmoid(z);
    }
    
    /**
     * 预测样本属于每个类别的概率（适用于多分类）
     */
    public float[] predictProbabilities(IVector x) {
        if (isBinaryClassification) {
            throw new IllegalStateException("predictProbabilities方法仅适用于多分类模型");
        }
        
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }
        
        // 计算每个类别的线性组合
        float[] logits = new float[numClasses];
        for (int k = 0; k < numClasses; k++) {
            IVector classWeights = weights.getRow(k);
            logits[k] = classWeights.innerProduct(x) + bias.get(k);
        }
        
        // 应用softmax函数
        return softmax(logits);
    }
    
    /**
     * 内部预测概率方法（不检查训练状态）
     */
    private Object predictProbabilityInternal(IVector x) {
        if (isBinaryClassification) {
            // 二分类：返回单个概率值
            float z = weights.getRow(0).innerProduct(x) + bias.get(0);
            return sigmoid(z);
        } else {
            // 多分类：返回概率数组
            float[] logits = new float[numClasses];
            for (int k = 0; k < numClasses; k++) {
                IVector classWeights = weights.getRow(k);
                logits[k] = classWeights.innerProduct(x) + bias.get(k);
            }
            return softmax(logits);
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
        
        String[] predictions = new String[features.getRowNum()];
        
        for (int i = 0; i < features.getRowNum(); i++) {
            IVector sample = features.getRow(i);
            predictions[i] = predict(sample);
        }
        
        return predictions;
    }
    
    // ==================== 损失函数和梯度计算 ====================
    
    @Override
    public float computeObjective(IVector x) {
        if (trainingFeatures == null || trainingLabels == null) {
            throw new IllegalStateException("训练数据未设置");
        }
        
        // 从参数向量中提取权重和偏置
        extractParametersFromVector(x);
        
        float totalLoss = 0.0f;
        int m = trainingFeatures.getRowNum();
        
        // 计算每个样本的损失
        for (int i = 0; i < m; i++) {
            IVector sample = trainingFeatures.getRow(i);
            int label = trainingLabels[i];
            
            if (isBinaryClassification) {
                // 二分类：使用sigmoid和交叉熵损失
                float probability = (Float) predictProbabilityInternal(sample);
                float sampleLoss = -label * (float) Math.log(probability + 1e-15f) - 
                                  (1 - label) * (float) Math.log(1 - probability + 1e-15f);
                totalLoss += sampleLoss;
            } else {
                // 多分类：使用softmax和交叉熵损失
                float[] probabilities = (float[]) predictProbabilityInternal(sample);
                float sampleLoss = -(float) Math.log(probabilities[label] + 1e-15f);
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
        int n = featureDimension;
        
        // 初始化梯度向量
        float[] weightGradients = new float[n];
        float biasGradient = 0.0f;
        
        // 计算每个样本的梯度
        for (int i = 0; i < m; i++) {
            IVector sample = trainingFeatures.getRow(i);
            int label = trainingLabels[i];
            
            // 计算预测概率
            float probability = (Float) predictProbabilityInternal(sample);
            
            // 计算误差
            float error = probability - label;
            
            // 累积权重梯度
            for (int j = 0; j < n; j++) {
                weightGradients[j] += error * sample.get(j);
            }
            
            // 累积偏置梯度
            biasGradient += error;
        }
        
        // 平均梯度
        for (int j = 0; j < n; j++) {
            weightGradients[j] /= m;
        }
        biasGradient /= m;
        
        // 添加正则化梯度
        addRegularizationGradients(weightGradients);
        
        // 创建梯度向量
        return createGradientVector(weightGradients, biasGradient);
    }
    
    /**
     * 计算多分类梯度
     */
    private IVector computeMulticlassClassificationGradient(int m) {
        // 初始化梯度矩阵和向量
        float[][] weightGradients = new float[numClasses][featureDimension];
        float[] biasGradients = new float[numClasses];
        
        // 计算每个样本的梯度
        for (int i = 0; i < m; i++) {
            IVector sample = trainingFeatures.getRow(i);
            int label = trainingLabels[i];
            
            // 计算预测概率
            float[] probabilities = (float[]) predictProbabilityInternal(sample);
            
            // 计算每个类别的梯度
            for (int k = 0; k < numClasses; k++) {
                float error = probabilities[k] - (k == label ? 1.0f : 0.0f);
                
                // 权重梯度
                for (int j = 0; j < featureDimension; j++) {
                    weightGradients[k][j] += error * sample.get(j);
                }
                
                // 偏置梯度
                biasGradients[k] += error;
            }
        }
        
        // 平均梯度
        for (int k = 0; k < numClasses; k++) {
            for (int j = 0; j < featureDimension; j++) {
                weightGradients[k][j] /= m;
            }
            biasGradients[k] /= m;
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
            float scale = (float) Math.sqrt(2.0 / featureDimension);
            float[] weightData = new float[featureDimension];
            for (int i = 0; i < featureDimension; i++) {
                weightData[i] = (random.nextFloat() - 0.5f) * 2.0f * scale;
            }
            
            this.weights = new RereMatrix(new float[][]{weightData});
            this.bias = new RereVector(new float[]{0.0f});
        } else {
            // 多分类：权重矩阵 + 偏置向量
            float scale = (float) Math.sqrt(2.0 / featureDimension);
            float[][] weightData = new float[numClasses][featureDimension];
            for (int k = 0; k < numClasses; k++) {
                for (int j = 0; j < featureDimension; j++) {
                    weightData[k][j] = (random.nextFloat() - 0.5f) * 2.0f * scale;
                }
            }
            
            this.weights = new RereMatrix(weightData);
            float[] biasData = new float[numClasses];
            for (int k = 0; k < numClasses; k++) {
                biasData[k] = 0.0f;
            }
            this.bias = new RereVector(biasData);
        }
    }
    
    /**
     * 创建参数向量
     */
    private IVector createParameterVector() {
        if (isBinaryClassification) {
            // 二分类：[w1, w2, ..., wn, b]
            int n = featureDimension;
            float[] paramData = new float[n + 1];
            
            // 复制权重
            for (int i = 0; i < n; i++) {
                paramData[i] = weights.get(0, i);
            }
            
            // 添加偏置
            paramData[n] = bias.get(0);
            
            return new RereVector(paramData);
        } else {
            // 多分类：[w11, w12, ..., w1n, w21, w22, ..., w2n, ..., wk1, wk2, ..., wkn, b1, b2, ..., bk]
            int totalParams = numClasses * featureDimension + numClasses;
            float[] paramData = new float[totalParams];
            
            int index = 0;
            
            // 复制权重
            for (int k = 0; k < numClasses; k++) {
                for (int j = 0; j < featureDimension; j++) {
                    paramData[index++] = weights.get(k, j);
                }
            }
            
            // 复制偏置
            for (int k = 0; k < numClasses; k++) {
                paramData[index++] = bias.get(k);
            }
            
            return new RereVector(paramData);
        }
    }
    
    /**
     * 从参数向量中提取权重和偏置
     */
    private void extractParametersFromVector(IVector paramVector) {
        if (isBinaryClassification) {
            // 二分类：提取权重向量和单个偏置
            int n = paramVector.length() - 1;
            
            // 提取权重
            float[] weightData = new float[n];
            for (int i = 0; i < n; i++) {
                weightData[i] = paramVector.get(i);
            }
            this.weights = new RereMatrix(new float[][]{weightData});
            
            // 提取偏置
            float biasValue = paramVector.get(n);
            this.bias = new RereVector(new float[]{biasValue});
        } else {
            // 多分类：提取权重矩阵和偏置向量
            int index = 0;
            
            // 提取权重
            float[][] weightData = new float[numClasses][featureDimension];
            for (int k = 0; k < numClasses; k++) {
                for (int j = 0; j < featureDimension; j++) {
                    weightData[k][j] = paramVector.get(index++);
                }
            }
            this.weights = new RereMatrix(weightData);
            
            // 提取偏置
            float[] biasData = new float[numClasses];
            for (int k = 0; k < numClasses; k++) {
                biasData[k] = paramVector.get(index++);
            }
            this.bias = new RereVector(biasData);
        }
    }
    
    /**
     * 创建梯度向量（二分类）
     */
    private IVector createGradientVector(float[] weightGradients, float biasGradient) {
        int n = weightGradients.length;
        float[] gradientData = new float[n + 1];
        
        // 复制权重梯度
        System.arraycopy(weightGradients, 0, gradientData, 0, n);
        
        // 添加偏置梯度
        gradientData[n] = biasGradient;
        
        return new RereVector(gradientData);
    }
    
    /**
     * 创建梯度向量（多分类）
     */
    private IVector createGradientVector(float[][] weightGradients, float[] biasGradients) {
        int totalParams = numClasses * featureDimension + numClasses;
        float[] gradientData = new float[totalParams];
        
        int index = 0;
        
        // 复制权重梯度
        for (int k = 0; k < numClasses; k++) {
            for (int j = 0; j < featureDimension; j++) {
                gradientData[index++] = weightGradients[k][j];
            }
        }
        
        // 复制偏置梯度
        for (int k = 0; k < numClasses; k++) {
            gradientData[index++] = biasGradients[k];
        }
        
        return new RereVector(gradientData);
    }
    
    /**
     * 根据lambda1和lambda2的值自动判断正则化类型
     */
    private RegularizationType inferRegularizationType(float lambda1, float lambda2) {
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
    private float computeRegularizationTerm() {
        float regularizationTerm = 0.0f;
        
        switch (regularizationType) {
            case L1:
                for (int i = 0; i < weights.getRowNum(); i++) {
                    for (int j = 0; j < weights.getColNum(); j++) {
                        regularizationTerm += Math.abs(weights.get(i, j));
                    }
                }
                regularizationTerm *= lambda1;
                break;
            case L2:
                for (int i = 0; i < weights.getRowNum(); i++) {
                    for (int j = 0; j < weights.getColNum(); j++) {
                        regularizationTerm += weights.get(i, j) * weights.get(i, j);
                    }
                }
                regularizationTerm *= lambda2 / 2.0f;
                break;
            case ELASTIC_NET:
                // L1部分
                float l1Term = 0.0f;
                for (int i = 0; i < weights.getRowNum(); i++) {
                    for (int j = 0; j < weights.getColNum(); j++) {
                        l1Term += Math.abs(weights.get(i, j));
                    }
                }
                // L2部分
                float l2Term = 0.0f;
                for (int i = 0; i < weights.getRowNum(); i++) {
                    for (int j = 0; j < weights.getColNum(); j++) {
                        l2Term += weights.get(i, j) * weights.get(i, j);
                    }
                }
                regularizationTerm = lambda1 * l1Term + lambda2 * l2Term / 2.0f;
                break;
            case NONE:
            default:
                regularizationTerm = 0.0f;
                break;
        }
        
        return regularizationTerm;
    }
    
    /**
     * 添加正则化梯度（二分类）
     */
    private void addRegularizationGradients(float[] weightGradients) {
        switch (regularizationType) {
            case L1:
                for (int i = 0; i < weightGradients.length; i++) {
                    weightGradients[i] += lambda1 * Math.signum(weights.get(0, i));
                }
                break;
            case L2:
                for (int i = 0; i < weightGradients.length; i++) {
                    weightGradients[i] += lambda2 * weights.get(0, i);
                }
                break;
            case ELASTIC_NET:
                for (int i = 0; i < weightGradients.length; i++) {
                    weightGradients[i] += lambda1 * Math.signum(weights.get(0, i)) + lambda2 * weights.get(0, i);
                }
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
    private void addRegularizationGradients(float[][] weightGradients) {
        switch (regularizationType) {
            case L1:
                for (int k = 0; k < numClasses; k++) {
                    for (int j = 0; j < featureDimension; j++) {
                        weightGradients[k][j] += lambda1 * Math.signum(weights.get(k, j));
                    }
                }
                break;
            case L2:
                for (int k = 0; k < numClasses; k++) {
                    for (int j = 0; j < featureDimension; j++) {
                        weightGradients[k][j] += lambda2 * weights.get(k, j);
                    }
                }
                break;
            case ELASTIC_NET:
                for (int k = 0; k < numClasses; k++) {
                    for (int j = 0; j < featureDimension; j++) {
                        weightGradients[k][j] += lambda1 * Math.signum(weights.get(k, j)) + lambda2 * weights.get(k, j);
                    }
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
    private float sigmoid(float z) {
        // 处理数值稳定性
        if (z > 0) {
            return 1.0f / (1.0f + (float) Math.exp(-z));
        } else {
            float expZ = (float) Math.exp(z);
            return expZ / (1.0f + expZ);
        }
    }
    
    /**
     * Softmax激活函数
     */
    private float[] softmax(float[] logits) {
        float[] probabilities = new float[numClasses];
        
        // 找到最大值以提高数值稳定性
        float maxLogit = logits[0];
        for (int i = 1; i < numClasses; i++) {
            if (logits[i] > maxLogit) {
                maxLogit = logits[i];
            }
        }
        
        // 计算指数并求和
        float sum = 0.0f;
        for (int i = 0; i < numClasses; i++) {
            probabilities[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += probabilities[i];
        }
        
        // 归一化
        for (int i = 0; i < numClasses; i++) {
            probabilities[i] /= sum;
        }
        
        return probabilities;
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
    public float getLearningRate() {
        return learningRate;
    }
    
    /**
     * 设置学习率
     */
    public void setLearningRate(float learningRate) {
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
    public float getTolerance() {
        return tolerance;
    }
    
    /**
     * 设置收敛阈值
     */
    public void setTolerance(float tolerance) {
        this.tolerance = tolerance;
    }
    
    /**
     * 获取L1正则化系数
     */
    public float getLambda1() {
        return lambda1;
    }
    
    /**
     * 设置L1正则化系数
     */
    public void setLambda1(float lambda1) {
        this.lambda1 = lambda1;
        this.regularizationType = inferRegularizationType(this.lambda1, this.lambda2);
    }
    
    /**
     * 获取L2正则化系数
     */
    public float getLambda2() {
        return lambda2;
    }
    
    /**
     * 设置L2正则化系数
     */
    public void setLambda2(float lambda2) {
        this.lambda2 = lambda2;
        this.regularizationType = inferRegularizationType(this.lambda1, this.lambda2);
    }
    
    /**
     * 设置正则化参数
     */
    public void setRegularization(float lambda1, float lambda2) {
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
}
