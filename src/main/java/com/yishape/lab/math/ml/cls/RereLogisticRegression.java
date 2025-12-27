package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOptimizer;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.newton.RereLBFGS;

/**
 * 统一逻辑回归实现类
 * <p>
 * 本类实现了逻辑回归算法，自动检测并支持二分类和多分类问题： - 二分类：使用sigmoid函数，输出单个概率值 -
 * 多分类：使用softmax函数，输出多个类别的概率分布
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class RereLogisticRegression implements IClassification, IGradientFunction, IObjectiveFunction, ISerializableModel {

    private static final long serialVersionUID = 1L;

    // 创建LBFGS优化器
    IOptimizer optimizer = new RereLBFGS();
//    IOptimizer optimizer = new RereConjugateGradient();
//    IOptimizer optimizer = new RereSteepestDescent();

    // ==================== 模型参数 ====================
    /**
     * 权重矩阵：多分类时为(K-1) x n矩阵，每行对应一个类别（最后一个类作为参考），二分类时为1 x n矩阵
     */
    private IMatrix weights;

    /**
     * 偏置向量：多分类时为(K-1)维向量，二分类时为1维向量
     */
    private IVector bias;

    /**
     * 学习率
     */
    private double learningRate = 0.001;

    /**
     * 最大迭代次数
     */
    private int maxIterations = 1000;

    /**
     * 收敛阈值
     */
    private double tolerance = 1e-6;

    /**
     * L1正则化系数（λ₁）
     */
    private double lambda1 = 0.0;

    /**
     * L2正则化系数（λ₂）
     */
    private double lambda2 = 0.0;

    /**
     * 正则化类型枚举
     */
    public enum RegularizationType {
        /**
         * 无正则化
         */
        NONE,
        /**
         * L1正则化（Lasso）
         */
        L1,
        /**
         * L2正则化（Ridge）
         */
        L2,
        /**
         * ElasticNet正则化（L1 + L2的组合）
         */
        ELASTIC_NET
    }

    /**
     * 正则化类型
     */
    private RegularizationType regularizationType = RegularizationType.NONE;

    /**
     * 标签映射：将字符串标签映射为数值
     */
    private Map<String, Integer> labelMapping;

    /**
     * 反向标签映射：将数值映射回字符串标签
     */
    private Map<Integer, String> reverseLabelMapping;

    /**
     * 训练特征矩阵
     */
    private IMatrix trainingFeatures;

    /**
     * 训练标签数组（数值化后）
     */
    private int[] trainingLabels;

    /**
     * 缓存的训练标签矩阵（用于加速梯度计算）
     */
    private IMatrix cachedLabelsMatrix;

    /**
     * 缓存的one-hot标签矩阵（用于多分类梯度计算）
     */
    private IMatrix cachedOneHotLabels;

    /**
     * 缓存的线性组合输出（X * W^T + b）
     */
    private IMatrix cachedLinearOutput;

    /**
     * 缓存的激活函数输出（sigmoid/softmax结果）
     */
    private IMatrix cachedProbabilities;

    /**
     * 缓存的训练特征矩阵转置（用于加速梯度计算）
     */
    private IMatrix cachedTrainingFeaturesTranspose;

    /**
     * 缓存的权重矩阵转置（用于加速线性组合计算）
     */
    private IMatrix cachedWeightsTranspose;

    /**
     * 当前缓存对应的权重矩阵（用于检测权重是否改变）
     */
    private IMatrix cachedWeightsForTranspose;

    /**
     * 线性组合缓存是否有效
     */
    private boolean linearOutputValid = false;

    /**
     * 激活函数输出缓存是否有效
     */
    private boolean probabilitiesValid = false;

    /**
     * 当前缓存对应的参数向量（用于检查参数是否改变）
     */
    private IVector currentParamVector;

    /**
     * 是否已训练
     */
    private boolean isTrained = false;

    /**
     * 分类类型：true为二分类，false为多分类
     */
    private boolean isBinaryClassification = true;

    /**
     * 类别数量
     */
    private int numClasses = 2;

    /**
     * 特征维度
     */
    private int featureDimension = 0;

    /**
     * 随机种子，用于确保参数初始化的一致性
     */
    private long randomSeed = System.currentTimeMillis();

    /**
     * 优化器收敛重试次数
     */
    private int maxRetries = 1;

    /**
     * 是否启用特征归一化
     */
    private boolean standardizeFeatures = true;

    /**
     * 特征第5百分位数向量（用于鲁棒归一化）
     */
    private IVector featureP5;

    /**
     * 特征第95百分位数向量（用于鲁棒归一化）
     */
    private IVector featureP95;

    /**
     * 类别权重向量（用于处理类别不平衡）
     */
    private IVector classWeights;

    /**
     * 是否启用类别权重（用于处理类别不平衡）
     */
    private boolean useClassWeights = false;

    /**
     * 训练集中每个类别的样本数量（用于参数初始化）
     */
    private int[] classCounts;

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
     *
     * @param learningRate
     * @param maxIterations
     * @param tolerance
     * @param lambda1
     * @param lambda2
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

    /**
     * 带随机种子的构造函数
     *
     * @param learningRate
     * @param maxIterations
     * @param tolerance
     * @param lambda1
     * @param lambda2
     * @param randomSeed
     */
    public RereLogisticRegression(double learningRate, int maxIterations, double tolerance,
            double lambda1, double lambda2, long randomSeed) {
        this(learningRate, maxIterations, tolerance, lambda1, lambda2);
        this.randomSeed = randomSeed;
    }

    public RereLogisticRegression(double lambda1, double lambda2) {
        this();
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

        // 检查特征矩阵是否包含无效值
        for (int i = 0; i < feature.getRowNum(); i++) {
            for (int j = 0; j < feature.getColNum(); j++) {
                double val = (double) feature.get(i, j);
                if (Double.isNaN(val) || Double.isInfinite(val)) {
                    throw new IllegalArgumentException(
                            String.format("训练特征矩阵包含无效值：行%d，列%d，值%s", i, j, val));
                }
            }
        }

        // 特征预处理：归一化（如果启用）
        IMatrix processedFeatures = feature;
        if (standardizeFeatures) {
            processedFeatures = normalizeTrainingFeatures(feature);
        }

        // 保存训练数据
        this.trainingFeatures = processedFeatures;
        this.featureDimension = processedFeatures.getColNum();

        // 清空训练特征缓存，确保使用新的训练数据
        this.cachedTrainingFeaturesTranspose = null;

        // 标签预处理：将字符串标签转换为数值，并检测分类类型
        preprocessLabels(labels);

        // 缓存训练标签矩阵用于加速梯度计算
        int m = trainingLabels.length;
        double[][] labelArray = new double[m][1];
        for (int i = 0; i < m; i++) {
            labelArray[i][0] = trainingLabels[i];
        }
        this.cachedLabelsMatrix = Linalg.matrix(labelArray);

        // 缓存one-hot标签矩阵用于多分类梯度计算
        if (!isBinaryClassification) {
            double[][] oneHotArray = new double[m][numClasses];
            for (int i = 0; i < m; i++) {
                int trueLabel = trainingLabels[i];
                oneHotArray[i][trueLabel] = 1.0;
            }
            this.cachedOneHotLabels = Linalg.matrix(oneHotArray);
        }

        // 初始化模型参数
        initializeParameters();

        // 使用LBFGS优化器训练模型
        trainWithOptimizer();

        // 标记模型已训练
        this.isTrained = true;

        // 计算最终损失和性能指标
        double finalLoss = computeObjective(createParameterVector());

        // 输出训练摘要
        System.out.println("=== 逻辑回归训练完成 ===");
        System.out.println("模型类型: " + getModelTypeDescription());
        System.out.println("特征维度: " + featureDimension);
        System.out.println("训练样本数: " + trainingFeatures.getRowNum());
        System.out.println("最终损失: " + String.format("%.6f", finalLoss));
        System.out.println("正则化: " + getRegularizationDescription());
        System.out.println("特征归一化: " + (standardizeFeatures ? "启用" : "禁用"));
        System.out.println("类别权重: " + (classWeights != null ? "启用" : "禁用"));
        System.out.println("随机种子: " + randomSeed);

        // 创建并返回训练结果
        LogisticRegressionResult result = new LogisticRegressionResult();

        // 设置权重和偏置
        if (isBinaryClassification) {
            result.setWeights(weights.getRow(0));
            result.setBias(Linalg.vector(new double[]{(double) bias.get(0)}));
        } else {
            // 对于多分类，返回完整权重矩阵和平移后的偏置向量
            // 这样可以保持与二分类接口的一致性，同时提供完整信息
            result.setWeights(weights.flatten()); // 展平权重矩阵
            result.setBias(bias); // 返回完整偏置向量
        }

        result.setLoss(finalLoss);

        return result;
    }

    /**
     * 使用优化器训练模型
     */
    private void trainWithOptimizer() {
        IVector optimalParams = null;
        double bestLoss = Double.MAX_VALUE;
        boolean converged = false;

        // 尝试多次训练以提高收敛稳定性
        for (int retry = 0; retry <= maxRetries; retry++) {
            try {
                // 如果不是第一次尝试，使用不同的随机种子
                if (retry > 0) {
                    this.randomSeed += retry; // 改变种子以获得不同初始化
                    initializeParameters();
                }

                // 创建初始参数向量
                IVector initParams = createParameterVector();
                long start = System.currentTimeMillis();
                System.out.println("Start to optimize...");
                // 执行优化
                var optimizationResult = optimizer.optimize(initParams, this, this);
                System.out.println("Optimization finished.");
                long end = System.currentTimeMillis();
                System.out.println("Optimization time cost (s):" + (end - start) / 1000.0);
                // 计算最终损失
                double finalLoss = computeObjective(optimizationResult.getOptimalPoint());

                // 如果收敛了或者损失更小，更新最佳结果
                if (optimizationResult.isConverged() || finalLoss < bestLoss) {
                    optimalParams = optimizationResult.getOptimalPoint();
                    bestLoss = finalLoss;
                    converged = optimizationResult.isConverged();

                    // 如果已经收敛，提前退出
                    if (converged) {
                        break;
                    }
                }

                if (retry < maxRetries) {
                    System.out.println(String.format("训练尝试 %d 未收敛或损失较高 (损失: %.6f)，尝试重新初始化...",
                            retry + 1, finalLoss));
                }

            } catch (Exception e) {
                System.out.println("训练尝试 " + (retry + 1) + " 失败: " + e.getMessage());
                if (retry == maxRetries) {
                    throw new RuntimeException("所有训练尝试都失败", e);
                }
            }
        }

        // 检查最终结果
        if (!converged) {
            System.out.println(String.format("警告：经过 %d 次尝试，优化器仍未完全收敛，最终损失值：%.6f",
                    maxRetries + 1, bestLoss));
        }

        // 从最佳结果中提取参数
        if (optimalParams != null) {
            extractParametersFromVector(optimalParams);
        } else {
            throw new RuntimeException("训练失败，无法获得有效的参数");
        }
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
            throw new IllegalArgumentException(
                    String.format("输入特征维度不匹配：期望%d维，实际%d维",
                            featureDimension, x.length()));
        }

        // 归一化输入特征（如果启用）
        IVector standardizedX = normalizePredictionFeatures(x);

        // 检查输入向量是否包含无效值
        for (int i = 0; i < standardizedX.length(); i++) {
            double val = (double) standardizedX.get(i);
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                throw new IllegalArgumentException(
                        String.format("输入特征向量包含无效值：位置%d，值%s", i, val));
            }
        }

        if (isBinaryClassification) {
            // 二分类：使用sigmoid函数
            double probability = predictProbability(standardizedX);
            return probability >= 0.5 ? reverseLabelMapping.get(1) : reverseLabelMapping.get(0);
        } else {
            // 多分类：使用softmax函数
            double[] probabilities = predictProbabilities(standardizedX);
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
        double z = (double) weights.getRow(0).innerProduct(x) + (double) bias.get(0);

        // 应用sigmoid函数：P(y=1|x) = 1 / (1 + e^(-z))
        return sigmoid(z);
    }

    /**
     * 预测样本属于每个类别的概率（适用于多分类）
     *
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

        // 使用K参数向量计算所有类的logits
        double[] logits = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            logits[i] = (double) weights.getRow(i).innerProduct(x) + (double) bias.get(i);
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
            // 使用向量内积计算线性组合
            double z = (double) weights.getRow(0).innerProduct(x) + (double) bias.get(0);
            return sigmoid(z);
        } else {
            // 多分类：返回概率数组，使用K参数向量
            double[] logits = new double[numClasses];
            for (int i = 0; i < numClasses; i++) {
                logits[i] = (double) weights.getRow(i).innerProduct(x) + (double) bias.get(i);
            }
            return softmax(logits);
        }
    }

    /**
     * 批量预测
     *
     * @param features
     * @return
     */
    @Override
    public String[] predictBatch(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }

        if (features == null) {
            throw new IllegalArgumentException("特征矩阵不能为null");
        }

        if (features.getColNum() != (standardizeFeatures ? featureDimension : featureDimension)) {
            throw new IllegalArgumentException("特征维度与训练特征维度不匹配");
        }

        // 归一化批量特征（如果启用）
        IMatrix processedFeatures = features;
        if (standardizeFeatures) {
            processedFeatures = normalizeBatchPredictionFeatures(features);
        }

        if (isBinaryClassification) {
            // 二分类：使用矩阵乘法进行批量预测
            // 计算线性组合：Z = X * W^T + b
            // 使用缓存的转置权重矩阵
            IMatrix weightMatrix = getWeightsTranspose();
            IMatrix linearOutput = processedFeatures.mmul(weightMatrix);

            // 添加偏置值到每一行
            final double biasValue = (double) bias.get(0);
            linearOutput = linearOutput.apply(x -> (double) x + biasValue);

            // 应用sigmoid函数
            IMatrix probabilities = linearOutput.apply(x -> sigmoid((double) x));

            // 转换为预测标签
            // 使用向量化操作替代手动循环
            String[] predictions = new String[features.getRowNum()];
            IVector<Double> probVector = probabilities.getColumn(0);
            for (int i = 0; i < features.getRowNum(); i++) {
                double prob = (double) probVector.get(i);
                predictions[i] = prob >= 0.5 ? reverseLabelMapping.get(1) : reverseLabelMapping.get(0);
            }
            return predictions;
        } else {
            // 多分类：使用K参数向量进行批量预测
            // 计算线性组合：Z = X * W^T + B (广播加法)
            // 使用缓存的转置权重矩阵
            IMatrix weightMatrix = getWeightsTranspose();
            IMatrix linearOutput = processedFeatures.mmul(weightMatrix);

            // 使用广播加法添加偏置
            linearOutput = linearOutput.broadcastAddRow(bias);

            // 构建完整的logits并应用softmax
            String[] result = new String[features.getRowNum()];
            for (int i = 0; i < features.getRowNum(); i++) {
                double[] logits = new double[numClasses];
                for (int j = 0; j < numClasses; j++) {
                    logits[j] = (double) linearOutput.get(i, j);
                }

                double[] probabilities = softmax(logits);
                IVector<Double> probVector = Linalg.vector(probabilities);
                int predictedClass = probVector.argMax();
                result[i] = reverseLabelMapping.get(predictedClass);
            }
            return result;
        }
    }

    /**
     * 批量预测（同时返回标签和概率）
     *
     * @param features 特征矩阵，每行代表一个样本
     * @return 包含预测标签和概率的BatchPredictionResult对象
     */
    @Override
    public BatchPredictionResult predictBatchWithProbabilities(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练，请先调用fit方法");
        }

        if (features == null) {
            throw new IllegalArgumentException("特征矩阵不能为null");
        }

        if (features.getColNum() != (standardizeFeatures ? featureDimension : featureDimension)) {
            throw new IllegalArgumentException("特征维度与训练特征维度不匹配");
        }

        // 归一化批量特征（如果启用）
        IMatrix processedFeatures = features;
        if (standardizeFeatures) {
            processedFeatures = normalizeBatchPredictionFeatures(features);
        }

        if (isBinaryClassification) {
            // 二分类：使用矩阵乘法进行批量预测
            // 计算线性组合：Z = X * W^T + b
            // 使用缓存的转置权重矩阵
            IMatrix weightMatrix = getWeightsTranspose();
            IMatrix linearOutput = processedFeatures.mmul(weightMatrix);

            // 添加偏置值到每一行
            final double biasValue = (double) bias.get(0);
            linearOutput = linearOutput.apply(x -> (double) x + biasValue);

            // 应用sigmoid函数
            IMatrix probabilities = linearOutput.apply(x -> sigmoid((double) x));

            // 转换为预测标签
            String[] predictions = new String[features.getRowNum()];
            IVector<Double> probVector = probabilities.getColumn(0);
            double[] probArray = probVector.toDoubleArray();

            for (int i = 0; i < features.getRowNum(); i++) {
                double prob = probArray[i];
                predictions[i] = prob >= 0.5 ? reverseLabelMapping.get(1) : reverseLabelMapping.get(0);
            }

            return new BatchPredictionResult(predictions, probArray);
        } else {
            // 多分类：使用K参数向量进行批量预测
            // 计算线性组合：Z = X * W^T + B (广播加法)
            // 使用缓存的转置权重矩阵
            IMatrix weightMatrix = getWeightsTranspose();
            IMatrix linearOutput = processedFeatures.mmul(weightMatrix);

            // 使用广播加法添加偏置
            linearOutput = linearOutput.broadcastAddRow(bias);

            // 构建完整的logits并应用softmax
            String[] predictions = new String[features.getRowNum()];
            double[][] classProbabilities = new double[features.getRowNum()][numClasses];

            for (int i = 0; i < features.getRowNum(); i++) {
                double[] logits = new double[numClasses];
                for (int j = 0; j < numClasses; j++) {
                    logits[j] = (double) linearOutput.get(i, j);
                }

                double[] probabilities = softmax(logits);
                classProbabilities[i] = probabilities;

                IVector<Double> probVector = Linalg.vector(probabilities);
                int predictedClass = probVector.argMax();
                predictions[i] = reverseLabelMapping.get(predictedClass);
            }

            return new BatchPredictionResult(predictions, classProbabilities);
        }
    }

    // ==================== 损失函数和梯度计算 ====================
    @Override
    public double computeObjective(IVector x) {
        if (trainingFeatures == null || trainingLabels == null) {
            throw new IllegalStateException("训练数据未设置");
        }

        // 从参数向量中提取权重和偏置（会自动清空缓存）
        extractParametersFromVector(x);

        // 使用缓存计算概率
        IMatrix probabilities = computeProbabilities();
        int m = trainingFeatures.getRowNum();

        double totalLoss = 0.0;

        if (isBinaryClassification) {
            // 二分类：使用缓存的概率计算交叉熵损失
            for (int i = 0; i < m; i++) {
                int label = trainingLabels[i];
                double weight = classWeights != null ? (double) classWeights.get(label) : 1.0;
                double probability = (double) probabilities.get(i, 0);

                // 确保概率在有效范围内，避免log(0)
                probability = Math.max(1e-15, Math.min(1.0 - 1e-15, probability));
                double sampleLoss = -label * Math.log(probability)
                        - (1.0 - label) * Math.log(1.0 - probability);
                totalLoss += weight * sampleLoss;
            }
        } else {
            // 多分类：使用缓存的概率计算交叉熵损失
            for (int i = 0; i < m; i++) {
                int label = trainingLabels[i];
                double weight = classWeights != null ? (double) classWeights.get(label) : 1.0;
                double probability = (double) probabilities.get(i, label);

                double sampleLoss = -Math.log(probability + 1e-15);
                totalLoss += weight * sampleLoss;
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
        // 使用缓存的概率计算梯度
        IMatrix probabilities = computeProbabilities();

        // 验证维度
        if (probabilities.getColNum() != 1) {
            throw new IllegalStateException(
                String.format("二分类概率矩阵列数应为1，实际为%d", probabilities.getColNum()));
        }

        // 计算误差矩阵 (probabilities - labels)
        IMatrix errors = probabilities.sub(cachedLabelsMatrix);

        // 应用类别权重：为每个样本根据其真实标签应用对应的类别权重
        if (classWeights != null) {
            double[][] weightedErrors = new double[m][1];
            for (int i = 0; i < m; i++) {
                int trueLabel = trainingLabels[i];
                double weight = (double) classWeights.get(trueLabel);
                weightedErrors[i][0] = (double) errors.get(i, 0) * weight;
            }
            errors = Linalg.matrix(weightedErrors);
        }

        // 使用缓存的训练特征转置计算权重梯度：features^T * errors / m
        IMatrix featuresTranspose = getTrainingFeaturesTranspose();
        
        // 验证矩阵乘法维度兼容性
        if (featuresTranspose.getColNum() != errors.getRowNum()) {
            throw new IllegalStateException(String.format(
                "矩阵乘法维度不兼容: features^T (%dx%d) * errors (%dx%d)",
                featuresTranspose.getRowNum(), featuresTranspose.getColNum(),
                errors.getRowNum(), errors.getColNum()));
        }
        
        IMatrix weightGradientsMatrix = featuresTranspose.mmul(errors).divideByScalar((double) m);
        IVector weightGradients = weightGradientsMatrix.getColumn(0);

        // 计算偏置梯度：sum(errors) / m
        double biasGradient = (double) errors.sum() / m;

        // 添加正则化梯度
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
        // 使用缓存的概率计算梯度
        IMatrix probabilities = computeProbabilities();

        // 验证维度
        if (probabilities.getColNum() != numClasses) {
            throw new IllegalStateException(String.format(
                "概率矩阵列数(%d)与类别数(%d)不匹配", 
                probabilities.getColNum(), numClasses));
        }

        // 计算误差矩阵 (probabilities - one_hot_labels)
        // 使用缓存的one-hot标签矩阵
        IMatrix oneHotLabels;
        if (cachedOneHotLabels != null && cachedOneHotLabels.getRowNum() == m 
            && cachedOneHotLabels.getColNum() == numClasses) {
            oneHotLabels = cachedOneHotLabels;
        } else {
            // 如果缓存为空或维度不匹配，临时创建one-hot标签矩阵
            double[][] labelArray = new double[m][numClasses];
            for (int i = 0; i < m; i++) {
                int trueLabel = trainingLabels[i];
                labelArray[i][trueLabel] = 1.0;
            }
            oneHotLabels = Linalg.matrix(labelArray);
        }

        IMatrix errors = probabilities.sub(oneHotLabels);

        // 应用类别权重
        if (classWeights != null) {
            // 为每一行应用对应的类别权重
            double[][] weightedErrors = new double[errors.getRowNum()][errors.getColNum()];
            for (int i = 0; i < errors.getRowNum(); i++) {
                int trueLabel = trainingLabels[i];
                double weight = (double) classWeights.get(trueLabel);
                for (int j = 0; j < errors.getColNum(); j++) {
                    weightedErrors[i][j] = (double) errors.get(i, j) * weight;
                }
            }
            errors = Linalg.matrix(weightedErrors);
        }

        // 使用缓存的训练特征转置计算权重梯度：features^T * errors / m
        IMatrix featuresTranspose = getTrainingFeaturesTranspose();
        
        // 验证矩阵乘法维度兼容性
        if (featuresTranspose.getColNum() != errors.getRowNum()) {
            throw new IllegalStateException(String.format(
                "矩阵乘法维度不兼容: features^T (%dx%d) * errors (%dx%d)",
                featuresTranspose.getRowNum(), featuresTranspose.getColNum(),
                errors.getRowNum(), errors.getColNum()));
        }
        
        // errors 维度: m x numClasses
        // weightGradientsMatrix 维度: featureDimension x numClasses
        IMatrix weightGradientsMatrix = featuresTranspose.mmul(errors).divideByScalar((double) m);

        // 计算偏置梯度：colSums(errors) / m
        IVector biasGradientsVector = errors.colSums().divideByScalar((double) m);
        double[] biasGradients = biasGradientsVector.toDoubleArray();

        // 转换权重梯度矩阵为数组格式
        // weightGradientsMatrix 维度: featureDimension x numClasses
        // 需要转置为 numClasses x featureDimension
        IMatrix weightGradientsMatrixT = weightGradientsMatrix.transpose();

        // 重塑为二维数组格式 (K个权重向量)
        double[][] weightGradients = new double[numClasses][featureDimension];
        for (int k = 0; k < numClasses; k++) {
            for (int j = 0; j < featureDimension; j++) {
                weightGradients[k][j] = (double) weightGradientsMatrixT.get(k, j);
            }
        }

        // 添加正则化梯度
        addRegularizationGradients(weightGradients);

        // 创建梯度向量
        return createGradientVector(weightGradients, biasGradients);
    }

    // ==================== 辅助方法 ====================
    /**
     * 特征归一化预处理（使用鲁棒的百分位数方法）
     */
    private IMatrix normalizeTrainingFeatures(IMatrix features) {
        int numSamples = features.getRowNum();
        int numFeatures = features.getColNum();

        // 计算每个特征的第5和第95百分位数
        double[] p5s = new double[numFeatures];
        double[] p95s = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double[] values = new double[numSamples];
            for (int i = 0; i < numSamples; i++) {
                values[i] = (double) features.get(i, j);
            }
            java.util.Arrays.sort(values);

            int p5Index = (int) Math.floor(0.05 * numSamples);
            int p95Index = (int) Math.ceil(0.95 * numSamples) - 1;
            if (p95Index >= numSamples) {
                p95Index = numSamples - 1;
            }
            if (p5Index < 0) {
                p5Index = 0;
            }

            double p5 = values[p5Index];
            double p95 = values[p95Index];

            // 如果百分位数相等，使用最小最大值
            if (p5 == p95) {
                p5 = values[0];
                p95 = values[numSamples - 1];
            }

            p5s[j] = p5;
            p95s[j] = p95;
        }

        // 保存归一化参数
        this.featureP5 = Linalg.vector(p5s);
        this.featureP95 = Linalg.vector(p95s);

        // 创建归一化后的特征矩阵
        double[][] normalizedData = new double[numSamples][numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                double val = (double) features.get(i, j);
                double p5 = p5s[j];
                double p95 = p95s[j];
                if (p95 == p5) {
                    normalizedData[i][j] = 0.0;
                } else {
                    normalizedData[i][j] = (val - p5) / (p95 - p5);
                }
            }
        }

        return Linalg.matrix(normalizedData);
    }

    /**
     * 归一化预测特征
     */
    private IVector normalizePredictionFeatures(IVector features) {
        if (!standardizeFeatures || featureP5 == null || featureP95 == null) {
            return features;
        }

        double[] normalized = new double[features.length()];
        for (int j = 0; j < features.length(); j++) {
            double val = (double) features.get(j);
            double p5 = (double) featureP5.get(j);
            double p95 = (double) featureP95.get(j);
            if (p95 == p5) {
                normalized[j] = 0.0;
            } else {
                normalized[j] = (val - p5) / (p95 - p5);
            }
        }

        return Linalg.vector(normalized);
    }

    /**
     * 归一化批量预测特征
     */
    private IMatrix normalizeBatchPredictionFeatures(IMatrix features) {
        if (!standardizeFeatures || featureP5 == null || featureP95 == null) {
            return features;
        }

        int numSamples = features.getRowNum();
        int numFeatures = features.getColNum();
        double[][] normalizedData = new double[numSamples][numFeatures];

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                double val = (double) features.get(i, j);
                double p5 = (double) featureP5.get(j);
                double p95 = (double) featureP95.get(j);
                if (p95 == p5) {
                    normalizedData[i][j] = 0.0;
                } else {
                    normalizedData[i][j] = (val - p5) / (p95 - p5);
                }
            }
        }

        return Linalg.matrix(normalizedData);
    }

    /**
     * 计算类别权重（用于处理类别不平衡）
     */
    private void computeClassWeights() {
        // 计算每个类别的样本数量
        int[] classCounts = new int[numClasses];
        for (int label : trainingLabels) {
            classCounts[label]++;
        }

        // 计算类别权重：总样本数 / (类别数 * 类别样本数)
        // 这样稀有类别的权重更高
        double[] weights = new double[numClasses];
        int totalSamples = trainingLabels.length;

        for (int i = 0; i < numClasses; i++) {
            if (classCounts[i] > 0) {
                weights[i] = (double) totalSamples / (numClasses * classCounts[i]);
            } else {
                weights[i] = 1.0; // 避免除零
            }
        }

        this.classWeights = Linalg.vector(weights);
    }

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

        // 计算类别计数（用于参数初始化）
        this.classCounts = new int[numClasses];
        for (int label : trainingLabels) {
            classCounts[label]++;
        }

        // 计算类别权重（仅在启用时）
        if (useClassWeights) {
            computeClassWeights();
        } else {
            this.classWeights = null;
        }

        // 清空标签矩阵缓存，因为标签映射已改变
        this.cachedLabelsMatrix = null;
        this.cachedOneHotLabels = null;
    }

    /**
     * 初始化模型参数
     */
    private void initializeParameters() {
        Random random = new Random(randomSeed);

        if (isBinaryClassification) {
            // 二分类：权重向量 + 单个偏置
            double scale = Math.sqrt(2.0 / featureDimension);

            // 手动创建随机权重矩阵以确保可重现性
            double[][] weightArray = new double[1][featureDimension];
            for (int j = 0; j < featureDimension; j++) {
                weightArray[0][j] = random.nextGaussian() * scale;
            }
            this.weights = Linalg.matrix(weightArray);
            this.bias = Linalg.zeros(1);
        } else {
            // 多分类：使用K参数向量（完整权重矩阵）
            double[][] weightArray = new double[numClasses][featureDimension];
            double[] biasArray = new double[numClasses];

            // 权重使用小随机值初始化（He初始化更合适）
            double scale = Math.sqrt(2.0 / featureDimension);
            for (int i = 0; i < numClasses; i++) {
                for (int j = 0; j < featureDimension; j++) {
                    weightArray[i][j] = random.nextGaussian() * scale;
                }
                // 偏置初始化为0（对称初始化）
                biasArray[i] = 0.0;
            }

            this.weights = Linalg.matrix(weightArray);
            this.bias = Linalg.vector(biasArray);
        }
    }

    /**
     * 创建参数向量（用于测试）
     */
    public IVector createParameterVector() {
        if (isBinaryClassification) {
            // 二分类：[w1, w2, ..., wn, b]
            // 使用矩阵和向量的连接操作来创建参数向量
            IVector weightVector = weights.getRow(0);
            IVector biasScalar = Linalg.vector(new double[]{(double) bias.get(0)});
            return weightVector.concat(biasScalar);
        } else {
            // 多分类：[w11, w12, ..., w1n, w21, w22, ..., w2n, ..., w(k-1)1, w(k-1)2, ..., w(k-1)n, b1, b2, ..., b(k-1)]
            // 将权重矩阵展平并连接偏置向量
            IVector weightVector = weights.flatten();
            return weightVector.concat(bias);
        }
    }

    /**
     * 从参数向量中提取权重和偏置
     */
    private void extractParametersFromVector(IVector paramVector) {
        // 检查参数是否真正改变，只有改变时才清空缓存
        if (currentParamVector == null || !paramVector.equals(currentParamVector)) {
            if (isBinaryClassification) {
                // 二分类：提取权重向量和单个偏置
                int n = paramVector.length() - 1;

                // 使用切片操作提取权重和偏置
                IVector weightVector = paramVector.slice(0, n);
                this.weights = weightVector.asColumnVector().transpose();

                // 提取偏置
                double biasValue = (double) paramVector.get(n);
                this.bias = Linalg.vector(new double[]{biasValue});
            } else {
                // 多分类：提取权重矩阵 (numClasses x featureDimension) 和偏置向量 (numClasses)
                int weightElements = numClasses * featureDimension;

                // 使用切片操作提取权重和偏置
                IVector weightVector = paramVector.slice(0, weightElements);
                IVector biasVector = paramVector.slice(weightElements, paramVector.length());

                // 使用reshape方法重塑权重向量为矩阵
                this.weights = weightVector.reshape(numClasses, featureDimension);
                this.bias = biasVector;
            }

            // 参数更新后，清空相关缓存
            invalidateCache();

            // 更新当前参数向量
            this.currentParamVector = paramVector.copy();
        }
    }

    /**
     * 清空缓存（当参数更新时调用）
     * 注意：cachedLabelsMatrix和cachedOneHotLabels不在此处清除，
     * 因为它们基于训练标签，在优化过程中不会改变
     */
    private void invalidateCache() {
        linearOutputValid = false;
        probabilitiesValid = false;
        cachedLinearOutput = null;
        cachedProbabilities = null;
        // 权重相关缓存需要更新
        cachedWeightsTranspose = null;
        cachedWeightsForTranspose = null;
        // cachedOneHotLabels 和 cachedLabelsMatrix 基于训练标签，不在此处清除
    }

    /**
     * 获取缓存的权重矩阵转置
     */
    private IMatrix getWeightsTranspose() {
        if (cachedWeightsTranspose == null || cachedWeightsForTranspose != weights) {
            cachedWeightsTranspose = weights.transpose();
            cachedWeightsForTranspose = weights;
        }
        return cachedWeightsTranspose;
    }

    /**
     * 获取缓存的训练特征矩阵转置
     */
    private IMatrix getTrainingFeaturesTranspose() {
        if (cachedTrainingFeaturesTranspose == null) {
            cachedTrainingFeaturesTranspose = trainingFeatures.transpose();
        } else {
            // 验证缓存是否仍然有效
            int expectedRows = trainingFeatures.getColNum();
            int expectedCols = trainingFeatures.getRowNum();
            if (cachedTrainingFeaturesTranspose.getRowNum() != expectedRows ||
                cachedTrainingFeaturesTranspose.getColNum() != expectedCols) {
                cachedTrainingFeaturesTranspose = trainingFeatures.transpose();
            }
        }
        return cachedTrainingFeaturesTranspose;
    }

    /**
     * 计算线性组合输出（X * W^T + b）
     */
    private IMatrix computeLinearOutput() {
        if (!linearOutputValid || cachedLinearOutput == null) {
            // 使用缓存的转置权重矩阵
            IMatrix weightMatrix = getWeightsTranspose();
            IMatrix linearOutput = trainingFeatures.mmul(weightMatrix);

            if (isBinaryClassification) {
                // 二分类：添加单个偏置值到每一行
                final double biasValue = (double) bias.get(0);
                cachedLinearOutput = linearOutput.apply(x -> (double) x + biasValue);
            } else {
                // 多分类：使用广播加法添加偏置向量
                cachedLinearOutput = linearOutput.broadcastAddRow(bias);
            }
            linearOutputValid = true;
        }
        return cachedLinearOutput;
    }

    /**
     * 计算激活函数输出
     */
    private IMatrix computeProbabilities() {
        if (!probabilitiesValid || cachedProbabilities == null) {
            IMatrix linearOutput = computeLinearOutput();
            int m = linearOutput.getRowNum();

            if (isBinaryClassification) {
                // 二分类：应用sigmoid函数
                cachedProbabilities = linearOutput.apply(x -> sigmoid((double) x));
            } else {
                // 多分类：应用softmax函数到每一行
                double[][] probArray = new double[m][numClasses];
                for (int i = 0; i < m; i++) {
                    double[] logits = new double[numClasses];
                    for (int k = 0; k < numClasses; k++) {
                        logits[k] = (double) linearOutput.get(i, k);
                    }
                    double[] probs = softmax(logits);
                    IVector<Double> probVector = Linalg.vector(probs);
                    probArray[i] = probVector.toDoubleArray();
                }
                cachedProbabilities = Linalg.matrix(probArray);
            }
            probabilitiesValid = true;
        }
        return cachedProbabilities;
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
        // 如果两个参数都很小，认为是无正则化
        if (lambda1 < 1e-6 && lambda2 < 1e-6) {
            return RegularizationType.NONE;
        } // 如果L1明显大于L2，使用L1正则化
        else if (lambda1 > lambda2 * 10) {
            return RegularizationType.L1;
        } // 如果L2明显大于L1，使用L2正则化
        else if (lambda2 > lambda1 * 10) {
            return RegularizationType.L2;
        } // 否则使用ElasticNet
        else {
            return RegularizationType.ELASTIC_NET;
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
                regularizationTerm = lambda1 * (double) weights.abs().sum();
                break;
            case L2:
                // 使用矩阵的Frobenius范数计算L2正则化项
                double frobenius = (double) weights.frobeniusNorm();
                regularizationTerm = lambda2 * frobenius * frobenius / 2.0;
                break;
            case ELASTIC_NET:
                // L1部分
                double l1Term = (double) weights.abs().sum();
                // L2部分
                double frobeniusNorm = (double) weights.frobeniusNorm();
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
        int numWeightClasses = numClasses; // K个权重向量
        switch (regularizationType) {
            case L1:
                // 使用矩阵操作计算L1正则化梯度
                IMatrix signMatrix = weights.sign(); // weights是(numClasses) x n
                IMatrix l1Grad = signMatrix.multiplyScalar(lambda1);
                IMatrix weightGradMatrix = Linalg.matrix(weightGradients);
                IMatrix result = weightGradMatrix.add(l1Grad);
                // 使用toDoubleArray方法替代手动循环
                double[][] resultArray = new double[numWeightClasses][featureDimension];
                for (int k = 0; k < numWeightClasses; k++) {
                    IVector<Double> row = result.getRow(k);
                    resultArray[k] = row.toDoubleArray();
                }
                // 复制结果回原数组
                for (int k = 0; k < numWeightClasses; k++) {
                    System.arraycopy(resultArray[k], 0, weightGradients[k], 0, featureDimension);
                }
                break;
            case L2:
                // 使用矩阵操作计算L2正则化梯度
                IMatrix l2Grad = weights.multiplyScalar(lambda2);
                IMatrix weightGradMatrix2 = Linalg.matrix(weightGradients);
                IMatrix result2 = weightGradMatrix2.add(l2Grad);
                // 使用toDoubleArray方法替代手动循环
                double[][] resultArray2 = new double[numWeightClasses][featureDimension];
                for (int k = 0; k < numWeightClasses; k++) {
                    IVector<Double> row = result2.getRow(k);
                    resultArray2[k] = row.toDoubleArray();
                }
                // 复制结果回原数组
                for (int k = 0; k < numWeightClasses; k++) {
                    System.arraycopy(resultArray2[k], 0, weightGradients[k], 0, featureDimension);
                }
                break;
            case ELASTIC_NET:
                // 使用矩阵操作计算ElasticNet正则化梯度
                IMatrix signMatrix2 = weights.sign(); // weights是(numClasses) x n
                IMatrix l1Component = signMatrix2.multiplyScalar(lambda1);
                IMatrix l2Component = weights.multiplyScalar(lambda2);
                IMatrix elasticGrad = l1Component.add(l2Component);
                IMatrix weightGradMatrix3 = Linalg.matrix(weightGradients);
                IMatrix result3 = weightGradMatrix3.add(elasticGrad);
                // 使用toDoubleArray方法替代手动循环
                double[][] resultArray3 = new double[numWeightClasses][featureDimension];
                for (int k = 0; k < numWeightClasses; k++) {
                    IVector<Double> row = result3.getRow(k);
                    resultArray3[k] = row.toDoubleArray();
                }
                // 复制结果回原数组
                for (int k = 0; k < numWeightClasses; k++) {
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
     * 计算log(exp(logOfX) + exp(logOfY))，避免数值溢出
     */
    private double logOfSum(double logOfX, double logOfY) {
        // 检查NaN值
        if (Double.isNaN(logOfX)) {
            return logOfY;
        }
        if (Double.isNaN(logOfY)) {
            return logOfX;
        }

        // 使用数值稳定的计算
        if (logOfX > logOfY) {
            return logOfX + Math.log(1 + Math.exp(logOfY - logOfX));
        } else {
            return logOfY + Math.log(1 + Math.exp(logOfX - logOfY));
        }
    }

    /**
     * Softmax激活函数（数值稳定版本，优化版）
     */
    private double[] softmax(double[] logits) {
        if (logits == null || logits.length == 0) {
            throw new IllegalArgumentException("logits数组不能为空");
        }

        // 使用数值稳定的softmax计算
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        double[] expValues = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            expValues[i] = Math.exp(logits[i] - maxLogit);
            sum += expValues[i];
        }

        // 归一化（检查sum是否为0或无效）
        if (sum == 0 || Double.isNaN(sum) || Double.isInfinite(sum)) {
            // 如果计算出错，返回均匀分布
            double uniformProb = 1.0 / logits.length;
            for (int i = 0; i < logits.length; i++) {
                expValues[i] = uniformProb;
            }
        } else {
            double invSum = 1.0 / sum;
            for (int i = 0; i < logits.length; i++) {
                expValues[i] *= invSum;
            }
        }
        return expValues;
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
     * 获取随机种子
     */
    public long getRandomSeed() {
        return randomSeed;
    }

    /**
     * 设置随机种子
     */
    public void setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
    }

    /**
     * 获取是否启用特征归一化
     */
    public boolean isStandardizeFeatures() {
        return standardizeFeatures;
    }

    /**
     * 设置是否启用特征归一化
     */
    public void setStandardizeFeatures(boolean standardizeFeatures) {
        this.standardizeFeatures = standardizeFeatures;
    }

    /**
     * 获取是否启用类别权重
     */
    public boolean isUseClassWeights() {
        return useClassWeights;
    }

    /**
     * 设置是否启用类别权重（用于处理类别不平衡）
     */
    public void setUseClassWeights(boolean useClassWeights) {
        this.useClassWeights = useClassWeights;
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
     * @return 
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
     *
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
