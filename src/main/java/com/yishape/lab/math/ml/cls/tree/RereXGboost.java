package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.cls.IClassification;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.math.optimize.newton.RereOnlineSGD;
import com.yishape.lab.math.optimize.newton.RereOnlineAdam;

import java.util.*;
import java.io.*;

/**
 * RereXGboost分类器
 * <p>
 * 实现XGBoost（eXtreme Gradient Boosting）算法，支持二分类和多分类。
 * 使用梯度提升决策树（GBDT）作为基学习器，通过迭代训练多个决策树来提升模型性能。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereXGboost implements IClassification, IGradientFunction, IObjectiveFunction, ISerializableModel {
    
    private static final long serialVersionUID = 1L;
    
    // ==================== 模型参数 ====================
    
    /** 决策树列表 */
    private List<XGTree> trees;
    
    /** 损失函数 */
    private XGBoostLossFunction lossFunction;
    
    /** 学习率 */
    private double learningRate = 0.1;
    
    /** 最大迭代次数（树的数量） */
    private int nEstimators = 100;
    
    /** 树的最大深度 */
    private int maxDepth = 6;
    
    /** 最小分裂样本数 */
    private int minSamplesSplit = 2;
    
    /** 最小叶子样本数 */
    private int minSamplesLeaf = 1;
    
    /** L1正则化参数 */
    private double alpha = 0.0;
    
    /** L2正则化参数 */
    private double lambda = 1.0;
    
    /** 早停轮数 */
    private int earlyStoppingRounds = 10;
    
    /** 验证集比例 */
    private double validationFraction = 0.1;
    
    /** 收敛容忍度 */
    private double tolerance = 1e-6;
    
    /** 随机种子 */
    private long randomSeed = 42;
    
    // ==================== 训练状态 ====================
    
    /** 类别标签映射 */
    private Map<String, Integer> labelToIndex;
    
    /** 索引到类别标签映射 */
    private Map<Integer, String> indexToLabel;
    
    /** 类别数量 */
    private int numClasses;
    
    /** 是否为二分类 */
    private boolean isBinary;
    
    /** 特征数量 */
    private int numFeatures;
    
    /** 训练损失历史 */
    private List<Double> trainLossHistory;
    
    /** 验证损失历史 */
    private List<Double> validationLossHistory;
    
    /** 特征重要性 */
    private IVector featureImportance;
    
    /** 初始预测值 */
    private IMatrix initialPredictions;
    
    /** 当前预测矩阵，用于优化计算 */
    private IMatrix predictions;
    
    // 优化器相关字段
    private IOnlineOptimizer optimizer;
    private String optimizerType = "adam"; // 默认使用SGD
    private double optimizerLearningRate = 0.01;
    
    /**
     * 默认构造函数
     */
    public RereXGboost() {
        this.trees = new ArrayList<>();
        this.trainLossHistory = new ArrayList<>();
        this.validationLossHistory = new ArrayList<>();
    }
    
    /**
     * 带参数的构造函数
     * @param learningRate 学习率
     * @param nEstimators 树的数量
     * @param maxDepth 最大深度
     * @param alpha L1正则化参数
     * @param lambda L2正则化参数
     */
    public RereXGboost(double learningRate, int nEstimators, int maxDepth, 
                      double alpha, double lambda) {
        this();
        this.learningRate = learningRate;
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth;
        this.alpha = alpha;
        this.lambda = lambda;
    }
    
    @Override
    public XGBoostResult fit(IMatrix features, String[] labels) {
        // 初始化
        initializeModel(features, labels);
        
        // 准备训练数据
        IMatrix labelMatrix = prepareLabels(labels);
        
        // 分割训练集和验证集
        DataSplit dataSplit = splitData(features, labelMatrix);
        
        // 初始化预测值
        IMatrix trainPredictions = initializePredictions(dataSplit.trainFeatures);
        IMatrix validPredictions = null;
        if (dataSplit.validFeatures != null) {
            validPredictions = initializePredictions(dataSplit.validFeatures);
        }
        
        // 梯度提升训练
        int bestIteration = 0;
        double bestValidLoss = Double.MAX_VALUE;
        int noImprovementCount = 0;
        
        // 初始化优化器参数向量（学习率）
        IVector optimizerParams = Linalg.vector(new double[]{learningRate});
        if (optimizer != null) {
            optimizer.initialize(optimizerParams);
        }
        
        for (int iteration = 0; iteration < nEstimators; iteration++) {
            
            // 计算梯度和海塞矩阵
            IMatrix gradients = lossFunction.computeGradients(trainPredictions, dataSplit.trainLabels);
            IMatrix hessians = lossFunction.computeHessians(trainPredictions, dataSplit.trainLabels);
            
            // 训练决策树
            List<XGTree> iterationTrees = trainTrees(dataSplit.trainFeatures, gradients, hessians);
            trees.addAll(iterationTrees);
            
            // 更新预测值 - 设置当前预测矩阵并调用优化后的方法
            this.predictions = trainPredictions;
            updatePredictions(dataSplit.trainFeatures, iterationTrees);
            trainPredictions = this.predictions;
            
            if (validPredictions != null) {
                this.predictions = validPredictions;
                updatePredictions(dataSplit.validFeatures, iterationTrees);
                validPredictions = this.predictions;
            }
            
            // 计算损失
            double trainLoss = lossFunction.computeLoss(trainPredictions, dataSplit.trainLabels);
            trainLossHistory.add(trainLoss);
            
            // 使用优化器动态调整学习率
            if (optimizer != null && iteration > 0) {
                // 计算学习率的梯度（基于损失变化）
                double lossGradient = trainLossHistory.get(iteration) - trainLossHistory.get(iteration - 1);
                IVector lrGradient = Linalg.vector(new double[]{lossGradient});
                
                // 使用优化器更新学习率
                IVector newParams = optimizer.step(lrGradient, trainLoss);
                double newLearningRate = Math.max(0.001, Math.min(1.0, newParams.get(0).doubleValue())); // 限制学习率范围
                this.learningRate = newLearningRate;
            }
            
            if (validPredictions != null) {
                double validLoss = lossFunction.computeLoss(validPredictions, dataSplit.validLabels);
                validationLossHistory.add(validLoss);
                
                // 早停检查
                if (validLoss < bestValidLoss - tolerance) {
                    bestValidLoss = validLoss;
                    bestIteration = iteration;
                    noImprovementCount = 0;
                } else {
                    noImprovementCount++;
                    if (noImprovementCount >= earlyStoppingRounds) {
                        System.out.println("Early stopping at iteration " + iteration);
                        break;
                    }
                }
            }
            
            // 打印进度
            if ((iteration + 1) % 10 == 0) {
                System.out.printf("Iteration %d: Train Loss = %.6f", iteration + 1, trainLoss);
                if (validPredictions != null) {
                    System.out.printf(", Valid Loss = %.6f", validationLossHistory.get(iteration));
                }
                System.out.println();
            }
        }
        
        // 计算特征重要性
        computeFeatureImportance();
        
        // 创建结果对象
        return createResult();
    }
    
    @Override
    public String predict(IVector x) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Model has not been trained yet.");
        }
        
        // 获取预测概率
        IVector probabilities = predictProba(x);
        
        // 使用向量API找到最大概率对应的类别索引
        int maxIndex = probabilities.argMax();
        
        return indexToLabel.get(maxIndex);
    }
    
    /**
     * 预测概率
     * @param x 特征向量
     * @return 各类别的概率
     */
    public IVector predictProba(IVector x) {
        if (trees.isEmpty()) {
            throw new IllegalStateException("Model has not been trained yet.");
        }
        
        // 初始化预测值
        IMatrix predictions = Linalg.zeros(1, isBinary ? 1 : numClasses);
        
        // 设置初始预测值
        for (int j = 0; j < predictions.cols(); j++) {
            predictions.set(0, j, initialPredictions.get(0, j).doubleValue());
        }
        
        // 累加所有树的预测
        int treeIndex = 0;
        for (int i = 0; i < nEstimators && treeIndex < trees.size(); i++) {
            for (int j = 0; j < (isBinary ? 1 : numClasses); j++) {
                if (treeIndex < trees.size()) {
                    double treePred = trees.get(treeIndex).predict(x);
                    double currentPred = predictions.get(0, j).doubleValue();
                    predictions.set(0, j, currentPred + treePred);
                    treeIndex++;
                }
            }
        }
        
        // 转换为概率
        IMatrix probMatrix = lossFunction.predictProba(predictions);
        return probMatrix.getRow(0);
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
     * 批量预测概率
     * @param features 特征矩阵
     * @return 概率矩阵
     */
    public IMatrix predictProba(IMatrix features) {
        int numSamples = features.rows();
        int outputDim = isBinary ? 2 : numClasses;
        IMatrix probabilities = Linalg.zeros(numSamples, outputDim);
        
        // 批量初始化预测值
        IMatrix predictions = Linalg.zeros(numSamples, isBinary ? 1 : numClasses);
        
        // 设置初始预测值 - 使用矩阵运算
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < predictions.cols(); j++) {
                predictions.set(i, j, initialPredictions.get(0, j).doubleValue());
            }
        }
        
        // 累加所有树的预测
        int treeIndex = 0;
        for (int iter = 0; iter < nEstimators && treeIndex < trees.size(); iter++) {
            for (int classIdx = 0; classIdx < (isBinary ? 1 : numClasses); classIdx++) {
                if (treeIndex < trees.size()) {
                    XGTree tree = trees.get(treeIndex);
                    // 对每个样本进行预测
                    for (int i = 0; i < numSamples; i++) {
                        double treePred = tree.predict(features.getRow(i));
                        double currentPred = predictions.get(i, classIdx).doubleValue();
                        predictions.set(i, classIdx, currentPred + treePred);
                    }
                    treeIndex++;
                }
            }
        }
        
        // 批量转换为概率
        return lossFunction.predictProba(predictions);
    }
    
    @Override
    public IVector computeGradient(IVector x) {
        // 计算当前模型参数的梯度
        // 这里x代表模型的参数向量，我们计算相对于这些参数的梯度
        if (predictions == null || lossFunction == null) {
            return Linalg.vector(new double[x.size()]);
        }
        
        // 计算梯度：对于XGBoost，梯度是损失函数相对于预测值的导数
        IMatrix gradients = lossFunction.computeGradients(predictions, predictions); // 使用当前预测作为标签的近似
        
        // 将梯度矩阵展平为向量
        double[] gradArray = new double[gradients.rows() * gradients.cols()];
        int idx = 0;
        for (int i = 0; i < gradients.rows(); i++) {
            for (int j = 0; j < gradients.cols(); j++) {
                gradArray[idx++] = gradients.get(i, j).doubleValue();
            }
        }
        
        return Linalg.vector(gradArray);
    }
    
    @Override
    public double computeObjective(IVector x) {
        // 计算当前模型的目标函数值（损失值）
        if (predictions == null || lossFunction == null) {
            return 0.0;
        }
        
        // 计算当前预测的损失
        return lossFunction.computeLoss(predictions, predictions); // 使用当前预测作为标签的近似
    }
    
    // ==================== 私有辅助方法 ====================
    
    /**
     * 初始化模型
     * @param features 特征矩阵
     * @param labels 标签数组
     */
    private void initializeModel(IMatrix features, String[] labels) {
        this.numFeatures = features.cols();
        
        // 构建标签映射
        buildLabelMapping(labels);
        
        // 初始化损失函数
        XGBoostLossFunction.LossType lossType = isBinary ? 
            XGBoostLossFunction.LossType.BINARY_LOGISTIC : 
            XGBoostLossFunction.LossType.MULTICLASS_SOFTMAX;
        this.lossFunction = new XGBoostLossFunction(lossType, numClasses);
        
        // 清空之前的训练状态
        this.trees.clear();
        this.trainLossHistory.clear();
        this.validationLossHistory.clear();
        
        // 初始化优化器
        initializeOptimizer();
    }
    
    /**
     * 构建标签映射
     * @param labels 标签数组
     */
    private void buildLabelMapping(String[] labels) {
        Set<String> uniqueLabels = new HashSet<>(Arrays.asList(labels));
        this.numClasses = uniqueLabels.size();
        this.isBinary = numClasses == 2;
        
        this.labelToIndex = new HashMap<>();
        this.indexToLabel = new HashMap<>();
        
        int index = 0;
        for (String label : uniqueLabels) {
            labelToIndex.put(label, index);
            indexToLabel.put(index, label);
            index++;
        }
    }
    
    /**
     * 准备标签矩阵
     * @param labels 标签数组
     * @return 标签矩阵
     */
    private IMatrix prepareLabels(String[] labels) {
        int numSamples = labels.length;
        
        if (isBinary) {
            // 二分类：使用0/1编码
            IMatrix labelMatrix = Linalg.zeros(numSamples, 1);
            for (int i = 0; i < numSamples; i++) {
                int labelIndex = labelToIndex.get(labels[i]);
                labelMatrix.set(i, 0, (double) labelIndex);
            }
            return labelMatrix;
        } else {
            // 多分类：使用one-hot编码
            IMatrix labelMatrix = Linalg.zeros(numSamples, numClasses);
            for (int i = 0; i < numSamples; i++) {
                int labelIndex = labelToIndex.get(labels[i]);
                for (int j = 0; j < numClasses; j++) {
                    labelMatrix.set(i, j, j == labelIndex ? 1.0 : 0.0);
                }
            }
            return labelMatrix;
        }
    }
    
    /**
     * 分割训练集和验证集
     * @param features 特征矩阵
     * @param labels 标签矩阵
     * @return 数据分割结果
     */
    private DataSplit splitData(IMatrix features, IMatrix labels) {
        int numSamples = features.rows();
        
        if (validationFraction <= 0 || validationFraction >= 1) {
            // 不使用验证集
            return new DataSplit(features, labels, null, null);
        }
        
        int validSize = (int) (numSamples * validationFraction);
        int trainSize = numSamples - validSize;
        
        // 如果验证集大小为0，直接返回全部数据作为训练集
        if (validSize == 0) {
            return new DataSplit(features, labels, null, null);
        }
        
        // 随机打乱索引
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random(randomSeed));
        
        // 分割数据
        IMatrix trainFeatures = Linalg.zeros(trainSize, features.cols());
        IMatrix trainLabels = Linalg.zeros(trainSize, labels.cols());
        IMatrix validFeatures = Linalg.zeros(validSize, features.cols());
        IMatrix validLabels = Linalg.zeros(validSize, labels.cols());
        
        for (int i = 0; i < trainSize; i++) {
            int idx = indices.get(i);
            for (int j = 0; j < features.cols(); j++) {
                trainFeatures.set(i, j, features.get(idx, j).doubleValue());
            }
            for (int j = 0; j < labels.cols(); j++) {
                trainLabels.set(i, j, labels.get(idx, j).doubleValue());
            }
        }
        
        for (int i = 0; i < validSize; i++) {
            int idx = indices.get(trainSize + i);
            for (int j = 0; j < features.cols(); j++) {
                validFeatures.set(i, j, features.get(idx, j).doubleValue());
            }
            for (int j = 0; j < labels.cols(); j++) {
                validLabels.set(i, j, labels.get(idx, j).doubleValue());
            }
        }
        
        return new DataSplit(trainFeatures, trainLabels, validFeatures, validLabels);
    }
    
    /**
     * 初始化预测值
     * @param features 特征矩阵
     * @return 初始预测矩阵
     */
    private IMatrix initializePredictions(IMatrix features) {
        int numSamples = features.rows();
        int predCols = isBinary ? 1 : numClasses;
        
        IMatrix predictions = Linalg.zeros(numSamples, predCols);
        
        // 初始化为0（对数几率为0，对应概率为0.5）
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < predCols; j++) {
                predictions.set(i, j, 0.0);
            }
        }
        
        // 保存初始预测值
        if (initialPredictions == null) {
            initialPredictions = Linalg.zeros(1, predCols);
            for (int j = 0; j < predCols; j++) {
                initialPredictions.set(0, j, 0.0);
            }
        }
        
        return predictions;
    }
    
    /**
     * 训练决策树
     * @param features 特征矩阵
     * @param gradients 梯度矩阵
     * @param hessians 海塞矩阵
     * @return 训练好的决策树列表
     */
    private List<XGTree> trainTrees(IMatrix features, IMatrix gradients, IMatrix hessians) {
        List<XGTree> iterationTrees = new ArrayList<>();
        
        int numTrees = isBinary ? 1 : numClasses;
        
        for (int treeIdx = 0; treeIdx < numTrees; treeIdx++) {
            // 提取当前树对应的梯度和海塞
            IVector treeGradients = gradients.getColumn(treeIdx);
            IVector treeHessians = hessians.getColumn(treeIdx);
            
            // 创建并训练决策树
            XGTree tree = new XGTree(maxDepth, minSamplesSplit, minSamplesLeaf,
                                               alpha, lambda, learningRate);
            tree.fit(features, treeGradients, treeHessians);
            
            iterationTrees.add(tree);
        }
        
        return iterationTrees;
    }
    
    /**
     * 更新预测值
     */
    private void updatePredictions(IMatrix features, List<XGTree> newTrees) {
        int numSamples = features.rows();
        
        // 为每个新树更新预测
        int treeIndex = 0;
        for (int classIdx = 0; classIdx < (isBinary ? 1 : numClasses); classIdx++) {
            if (treeIndex < newTrees.size()) {
                XGTree tree = newTrees.get(treeIndex);
                
                // 批量预测所有样本
                for (int i = 0; i < numSamples; i++) {
                    double treePred = tree.predict(features.getRow(i));
                    double currentPred = predictions.get(i, classIdx).doubleValue();
                    predictions.set(i, classIdx, currentPred + treePred);
                }
                treeIndex++;
            }
        }
    }
    
    /**
     * 计算特征重要性
     */
    private void computeFeatureImportance() {
        // 使用向量运算优化特征重要性计算
        IVector importance = Linalg.zeros(numFeatures);
        
        for (XGTree tree : trees) {
            double[] treeImportance = tree.computeFeatureImportance(numFeatures);
            IVector treeImportanceVector = Linalg.vector(treeImportance);
            // 使用向量加法替代手动循环
            importance = importance.add(treeImportanceVector);
        }
        
        // 归一化 - 使用向量运算
        double totalImportance = importance.sum().doubleValue();
        if (totalImportance > 0) {
            // 使用向量除法进行归一化
            importance = importance.divideByScalar(totalImportance);
        }
        
        this.featureImportance = importance;
    }
    
    /**
     * 创建结果对象
     * @return XGBoost训练结果
     */
    private XGBoostResult createResult() {
        XGBoostResult result = new XGBoostResult();
        
        result.setTrees(new ArrayList<>(trees));
        result.setLearningRate(learningRate);
        result.setLossHistory(new ArrayList<>(trainLossHistory));
        result.setFeatureImportance(featureImportance);
        result.setNumClasses(numClasses);
        result.setBinary(isBinary);
        result.setLabelToIndex(new HashMap<>(labelToIndex));
        result.setIndexToLabel(new HashMap<>(indexToLabel));
        result.setLambda(lambda);
        result.setAlpha(alpha);
        result.setMaxDepth(maxDepth);
        result.setMinSamplesSplit(minSamplesSplit);
        result.setMinSamplesLeaf(minSamplesLeaf);
        
        return result;
    }
    
    // ==================== Getters and Setters ====================
    
    public double getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
    
    public int getNEstimators() {
        return nEstimators;
    }
    
    public int getNumEstimators() {
        return nEstimators;
    }

    public void setNEstimators(int nEstimators) {
        this.nEstimators = nEstimators;
    }
    
    public void setNumEstimators(int numEstimators) {
        this.nEstimators = numEstimators;
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
    
    public double getAlpha() {
        return alpha;
    }
    
    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }
    
    public double getLambda() {
        return lambda;
    }
    
    public void setLambda(double lambda) {
        this.lambda = lambda;
    }
    
    public int getEarlyStoppingRounds() {
        return earlyStoppingRounds;
    }
    
    public void setEarlyStoppingRounds(int earlyStoppingRounds) {
        this.earlyStoppingRounds = earlyStoppingRounds;
    }
    
    public double getValidationFraction() {
        return validationFraction;
    }
    
    public void setValidationFraction(double validationFraction) {
        this.validationFraction = validationFraction;
    }
    
    public double getTolerance() {
        return tolerance;
    }
    
    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }
    
    public long getRandomSeed() {
        return randomSeed;
    }
    
    public void setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
    }
    
    public List<Double> getTrainLossHistory() {
        return new ArrayList<>(trainLossHistory);
    }
    
    public List<Double> getValidationLossHistory() {
        return new ArrayList<>(validationLossHistory);
    }
    
    public IVector getFeatureImportance() {
        return featureImportance;
    }
    
    public int getNumClasses() {
        return numClasses;
    }
    
    public boolean isBinary() {
        return isBinary;
    }
    
    public boolean isEarlyStopping() {
        return earlyStoppingRounds > 0;
    }
    
    public void setEarlyStopping(boolean earlyStopping) {
        this.earlyStoppingRounds = earlyStopping ? 10 : 0;
    }
    
    // ==================== 优化器相关方法 ====================
    
    /**
     * 设置优化器类型
     * @param optimizerType 优化器类型 ("sgd", "adam")
     */
    public void setOptimizerType(String optimizerType) {
        this.optimizerType = optimizerType.toLowerCase();
    }
    
    /**
     * 获取优化器类型
     * @return 优化器类型
     */
    public String getOptimizerType() {
        return optimizerType;
    }
    
    /**
     * 设置优化器学习率
     * @param optimizerLearningRate 优化器学习率
     */
    public void setOptimizerLearningRate(double optimizerLearningRate) {
        this.optimizerLearningRate = optimizerLearningRate;
    }
    
    /**
     * 获取优化器学习率
     * @return 优化器学习率
     */
    public double getOptimizerLearningRate() {
        return optimizerLearningRate;
    }

    /**
     * 获取优化器
     * @return 当前使用的优化器
     */
    public IOnlineOptimizer getOptimizer() {
        return optimizer;
    }

    /**
     * 初始化优化器
     */
    private void initializeOptimizer() {
        switch (optimizerType) {
            case "sgd":
                optimizer = new RereOnlineSGD(optimizerLearningRate, 0.9); // 学习率和动量
                break;
            case "adam":
                optimizer = new RereOnlineAdam(optimizerLearningRate);
                break;
            default:
                // 对于无效的优化器类型，默认使用SGD
                System.out.println("警告: 不支持的优化器类型 '" + optimizerType + "'，使用默认的SGD优化器");
                optimizer = new RereOnlineSGD(optimizerLearningRate, 0.9);
                optimizerType = "sgd"; // 更新为实际使用的类型
                break;
        }
    }
    
    /**
     * 数据分割结果内部类
     */
    private static class DataSplit {
        IMatrix trainFeatures;
        IMatrix trainLabels;
        IMatrix validFeatures;
        IMatrix validLabels;
        
        DataSplit(IMatrix trainFeatures, IMatrix trainLabels, 
                 IMatrix validFeatures, IMatrix validLabels) {
            this.trainFeatures = trainFeatures;
            this.trainLabels = trainLabels;
            this.validFeatures = validFeatures;
            this.validLabels = validLabels;
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
