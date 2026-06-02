package com.yishape.lab.math.ml.clf.ensemble;

import com.yishape.lab.math.ml.clf.lr.RereLogisticRegression;
import com.yishape.lab.util.YishapeLogger;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.clf.BatchPredResult;
import com.yishape.lab.math.ml.clf.ClfResult;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.ml.clf.tree.RereRandomForest;
import com.yishape.lab.math.ml.clf.tree.RereXGboost;
import com.yishape.lab.math.ml.clf.tree.RFTree;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;

import java.util.*;

/**
 * 集成分类器
 * <p>
 * 结合多种分类算法进行集成学习，包括：
 * - 随机森林 (Random Forest)
 * - 逻辑回归 (Logistic Regression)
 * - XGBoost
 * 
 * 支持多种集成策略：
 * - 投票法 (Voting)
 * - 加权投票法 (Weighted Voting)
 * - 堆叠法 (Stacking)
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class EnsembleClassifier implements IClassifier {

    private static final YishapeLogger log = YishapeLogger.getLogger(EnsembleClassifier.class);

    /** 随机森林分类器 */
    private RereRandomForest randomForest;
    
    /** 逻辑回归分类器 */
    private RereLogisticRegression logisticRegression;
    
    /** XGBoost分类器 */
    private RereXGboost xgboost;
    
    /** 分类器权重 */
    private IVector classifierWeights;
    
    /** 集成策略 */
    private EnsembleStrategy strategy;
    
    /** 是否已训练 */
    private boolean isTrained;
    
    /** 类别标签 */
    private String[] classLabels;
    
    /** 随机种子 */
    private long randomSeed;
    
    private ClassificationMetrics metrics;
    private EnsembleResult result;
    
    /**
     * 集成策略枚举
     */
    public enum EnsembleStrategy {
        /** 简单投票 */
        VOTING,
        /** 加权投票 */
        WEIGHTED_VOTING,
        /** 堆叠 */
        STACKING
    }
    
    /**
     * 集成结果类
     */
    public static class EnsembleResult extends ClfResult {
        public String[] predictions;
        public IMatrix probabilities;
        public double accuracy;
        public Map<String, Double> classifierAccuracies;
        public int numEstimators;
        public String ensembleStrategy;
        
        public EnsembleResult() {
            super();
        }
        
        public EnsembleResult(String[] predictions, IMatrix probabilities, 
                            double accuracy, Map<String, Double> classifierAccuracies) {
            super();
            this.predictions = predictions;
            this.probabilities = probabilities;
            this.accuracy = accuracy;
            this.classifierAccuracies = classifierAccuracies;
        }
        
        @Override
        public String getModelTypeDescription() {
            return "Ensemble Classifier (" + ensembleStrategy + ")";
        }
        
        @Override
        public String getModelSummary() {
            return String.format("Ensemble: %s, %d estimators, %d classes", 
                    ensembleStrategy, numEstimators, numClasses);
        }
        
        @Override
        public boolean isTrained() {
            return numEstimators > 0;
        }
    }
    

    
    /**
     * 默认构造函数
     * @param strategy 集成策略
     * @param randomSeed 随机种子
     */
    public EnsembleClassifier(EnsembleStrategy strategy, long randomSeed) {
        this.strategy = strategy;
        this.randomSeed = randomSeed;
        this.isTrained = false;
        
        // 初始化分类器
        initializeClassifiers();
        
        // 初始化权重（均等权重）
        this.classifierWeights = Linalg.vector(new double[]{1.0/3, 1.0/3, 1.0/3});
    }
    
    /**
     * 带权重的构造函数
     * @param strategy 集成策略
     * @param weights 分类器权重 [RF, LR, XGB]
     * @param randomSeed 随机种子
     */
    public EnsembleClassifier(EnsembleStrategy strategy, double[] weights, long randomSeed) {
        this.strategy = strategy;
        this.randomSeed = randomSeed;
        this.isTrained = false;
        
        // 初始化分类器
        initializeClassifiers();
        
        // 设置权重并归一化
        IVector weightVector = Linalg.vector(weights);
        double weightSum = weightVector.sumValue();
        this.classifierWeights = weightVector.multiplyByScalar(1.0 / weightSum);
    }
    
    /**
     * 初始化分类器
     */
    private void initializeClassifiers() {
        // 初始化随机森林
        this.randomForest = new RereRandomForest(
            100, 10, 2, 1, -1, true, 
            RFTree.SplitCriterion.GINI, randomSeed
        );
        
        // 初始化逻辑回归
        this.logisticRegression = new RereLogisticRegression();
        
        // 初始化XGBoost
        this.xgboost = new RereXGboost(0.1, 100, 6, 0.0, 1.0);
        this.xgboost.setRandomSeed(randomSeed);
    }
    
    /**
     * 训练集成分类器
     * @param features 特征矩阵
     * @param labels 标签数组
     * @return 当前实例，支持链式调用
     */
    @Override
    public IClassifier fit(IMatrix features, String[] labels) {
        if (features == null || labels == null) {
            throw new IllegalArgumentException("Features and labels cannot be null");
        }

        if (features.rows() != labels.length) {
            throw new IllegalArgumentException("Number of samples in features and labels must match");
        }

        // 获取唯一类别标签
        Set<String> uniqueLabels = new HashSet<>(Arrays.asList(labels));
        this.classLabels = uniqueLabels.toArray(new String[0]);
        Arrays.sort(this.classLabels);

        // 训练各个分类器
        log.debug("Training Random Forest...");
        randomForest.fit(features, labels);

        log.debug("Training Logistic Regression...");
        logisticRegression.fit(features, labels);

        log.debug("Training XGBoost...");
        xgboost.fit(features, labels);

        // 如果使用加权投票，根据验证性能调整权重
        if (strategy == EnsembleStrategy.WEIGHTED_VOTING) {
            adjustWeightsByValidation(features, labels);
        }

        this.isTrained = true;

        // 创建结果对象
        EnsembleResult er = new EnsembleResult();
        er.predictions = predictBatch(features);
        er.setNumClasses(classLabels.length);
        er.numEstimators = 3;
        er.ensembleStrategy = strategy.name();
        this.result = er;

        log.debug("Ensemble training completed!");

        return this;
    }

    @Override
    public String[] fitPredict(IMatrix features, String[] labels) {
        fit(features, labels);
        return predict(features).predictions;
    }

    @Override
    public EnsembleResult getResult() {
        return result;
    }
    
    /**
     * 预测
     * @param features 特征矩阵
     * @return 预测结果
     */
    public EnsembleResult predict(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        // 获取各分类器的预测
        String[] rfPredictions = randomForest.predictBatch(features);
        String[] lrPredictions = logisticRegression.predictBatch(features);
        String[] xgbPredictions = xgboost.predictBatch(features);
        
        // 获取各分类器的概率预测
        IMatrix rfProbs = convertPredictionsToProbs(rfPredictions);
        IMatrix lrProbs = convertPredictionsToProbs(lrPredictions);
        IMatrix xgbProbs = convertPredictionsToProbs(xgbPredictions);
        
        // 根据策略进行集成
        String[] ensemblePredictions;
        IMatrix ensembleProbs;
        
        switch (strategy) {
            case VOTING:
                ensemblePredictions = votingPredict(rfPredictions, lrPredictions, xgbPredictions);
                ensembleProbs = votingProbabilities(rfProbs, lrProbs, xgbProbs);
                break;
            case WEIGHTED_VOTING:
                ensemblePredictions = weightedVotingPredict(rfPredictions, lrPredictions, xgbPredictions);
                ensembleProbs = weightedVotingProbabilities(rfProbs, lrProbs, xgbProbs);
                break;
            case STACKING:
                ensemblePredictions = stackingPredict(rfProbs, lrProbs, xgbProbs);
                ensembleProbs = stackingProbabilities(rfProbs, lrProbs, xgbProbs);
                break;
            default:
                throw new IllegalStateException("Unknown ensemble strategy: " + strategy);
        }
        
        // 计算各分类器准确率（如果有真实标签的话，这里简化处理）
        Map<String, Double> classifierAccuracies = new HashMap<>();
        classifierAccuracies.put("RandomForest", 0.0);
        classifierAccuracies.put("LogisticRegression", 0.0);
        classifierAccuracies.put("XGBoost", 0.0);
        
        return new EnsembleResult(ensemblePredictions, ensembleProbs, 0.0, classifierAccuracies);
    }
    
    /**
     * 简单投票预测
     */
    private String[] votingPredict(String[] rfPreds, String[] lrPreds, String[] xgbPreds) {
        String[] result = new String[rfPreds.length];
        
        for (int i = 0; i < rfPreds.length; i++) {
            Map<String, Integer> votes = new HashMap<>();
            votes.put(rfPreds[i], votes.getOrDefault(rfPreds[i], 0) + 1);
            votes.put(lrPreds[i], votes.getOrDefault(lrPreds[i], 0) + 1);
            votes.put(xgbPreds[i], votes.getOrDefault(xgbPreds[i], 0) + 1);
            
            // 找到得票最多的类别
            result[i] = votes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get().getKey();
        }
        
        return result;
    }
    
    /**
     * 加权投票预测
     */
    private String[] weightedVotingPredict(String[] rfPreds, String[] lrPreds, String[] xgbPreds) {
        String[] result = new String[rfPreds.length];
        
        for (int i = 0; i < rfPreds.length; i++) {
            Map<String, Double> weightedVotes = new HashMap<>();
            
            // 加权投票
            weightedVotes.put(rfPreds[i], 
                weightedVotes.getOrDefault(rfPreds[i], 0.0) + classifierWeights.get(0));
            weightedVotes.put(lrPreds[i], 
                weightedVotes.getOrDefault(lrPreds[i], 0.0) + classifierWeights.get(1));
            weightedVotes.put(xgbPreds[i], 
                weightedVotes.getOrDefault(xgbPreds[i], 0.0) + classifierWeights.get(2));
            
            // 找到加权得票最多的类别
            result[i] = weightedVotes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get().getKey();
        }
        
        return result;
    }
    
    /**
     * 堆叠预测（简化版本）
     */
    private String[] stackingPredict(IMatrix rfProbs, IMatrix lrProbs, IMatrix xgbProbs) {
        // 简化的堆叠：使用加权平均
        IMatrix avgProbs = weightedVotingProbabilities(rfProbs, lrProbs, xgbProbs);
        
        String[] result = new String[avgProbs.rows()];
        for (int i = 0; i < avgProbs.rows(); i++) {
            int maxIndex = 0;
            double maxProb = avgProbs.get(i, 0);
            
            for (int j = 1; j < avgProbs.cols(); j++) {
                double prob = avgProbs.get(i, j);
                if (prob > maxProb) {
                    maxProb = prob;
                    maxIndex = j;
                }
            }
            
            result[i] = classLabels[maxIndex];
        }
        
        return result;
    }
    
    /**
     * 简单投票概率
     */
    private IMatrix votingProbabilities(IMatrix rfProbs, IMatrix lrProbs, IMatrix xgbProbs) {
        int rows = rfProbs.rows();
        int cols = rfProbs.cols();
        double[][] avgProbs = new double[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                avgProbs[i][j] = (rfProbs.get(i, j) + 
                                 lrProbs.get(i, j) + 
                                 xgbProbs.get(i, j)) / 3.0;
            }
        }
        
        return Linalg.matrix(avgProbs);
    }
    
    /**
     * 加权投票概率
     */
    private IMatrix weightedVotingProbabilities(IMatrix rfProbs, IMatrix lrProbs, IMatrix xgbProbs) {
        int rows = rfProbs.rows();
        int cols = rfProbs.cols();
        double[][] weightedProbs = new double[rows][cols];
        
        double rfWeight = classifierWeights.get(0);
        double lrWeight = classifierWeights.get(1);
        double xgbWeight = classifierWeights.get(2);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weightedProbs[i][j] = rfWeight * rfProbs.get(i, j) + 
                                     lrWeight * lrProbs.get(i, j) + 
                                     xgbWeight * xgbProbs.get(i, j);
            }
        }
        
        return Linalg.matrix(weightedProbs);
    }
    
    /**
     * 堆叠概率（简化版本）
     */
    private IMatrix stackingProbabilities(IMatrix rfProbs, IMatrix lrProbs, IMatrix xgbProbs) {
        // 简化的堆叠：使用加权平均
        return weightedVotingProbabilities(rfProbs, lrProbs, xgbProbs);
    }
    
    /**
     * 将预测转换为概率矩阵
     */
    private IMatrix convertPredictionsToProbs(String[] predictions) {
        int numSamples = predictions.length;
        int numClasses = classLabels.length;
        double[][] probs = new double[numSamples][numClasses];
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (classLabels[j].equals(predictions[i])) {
                    probs[i][j] = 1.0;
                } else {
                    probs[i][j] = 0.0;
                }
            }
        }
        
        return Linalg.matrix(probs);
    }
    
    /**
     * 根据验证性能调整权重
     */
    private void adjustWeightsByValidation(IMatrix features, String[] labels) {
        // 简单的交叉验证来评估各分类器性能
        int folds = 5;
        int foldSize = features.rows() / folds;
        
        double[] accuracies = new double[3]; // RF, LR, XGB
        
        for (int fold = 0; fold < folds; fold++) {
            int startIdx = fold * foldSize;
            int endIdx = (fold == folds - 1) ? features.rows() : (fold + 1) * foldSize;
            
            // 分割训练和验证集
            List<Integer> trainIndices = new ArrayList<>();
            List<Integer> validIndices = new ArrayList<>();
            
            for (int i = 0; i < features.rows(); i++) {
                if (i >= startIdx && i < endIdx) {
                    validIndices.add(i);
                } else {
                    trainIndices.add(i);
                }
            }
            
            // 创建训练和验证数据
            IMatrix trainFeatures = extractRows(features, trainIndices);
            String[] trainLabels = extractLabels(labels, trainIndices);
            IMatrix validFeatures = extractRows(features, validIndices);
            String[] validLabels = extractLabels(labels, validIndices);
            
            // 训练和评估各分类器
            RereRandomForest tempRF = new RereRandomForest(
                50, 10, 2, 1, -1, true, 
                RFTree.SplitCriterion.GINI, randomSeed + fold
            );
            tempRF.fit(trainFeatures, trainLabels);
            accuracies[0] += computeAccuracy(tempRF.predictBatch(validFeatures), validLabels);
            
            RereLogisticRegression tempLR = new RereLogisticRegression();
            tempLR.fit(trainFeatures, trainLabels);
            accuracies[1] += computeAccuracy(tempLR.predictBatch(validFeatures), validLabels);
            
            RereXGboost tempXGB = new RereXGboost(0.1, 50, 6, 0.0, 1.0);
            tempXGB.setRandomSeed(randomSeed + fold);
            tempXGB.fit(trainFeatures, trainLabels);
            accuracies[2] += computeAccuracy(tempXGB.predictBatch(validFeatures), validLabels);
        }
        
        // 平均准确率
        for (int i = 0; i < 3; i++) {
            accuracies[i] /= folds;
        }
        
        // 根据准确率调整权重
        double totalAccuracy = accuracies[0] + accuracies[1] + accuracies[2];
        if (totalAccuracy > 0) {
            this.classifierWeights = Linalg.vector(new double[]{
                accuracies[0] / totalAccuracy,
                accuracies[1] / totalAccuracy,
                accuracies[2] / totalAccuracy
            });
        }
        
        log.debug("Adjusted weights based on validation: " + 
                          "RF=" + String.format("%.3f", classifierWeights.get(0)) +
                          ", LR=" + String.format("%.3f", classifierWeights.get(1)) +
                          ", XGB=" + String.format("%.3f", classifierWeights.get(2)));
    }
    
    /**
     * 提取指定行的特征矩阵
     */
    private IMatrix extractRows(IMatrix matrix, List<Integer> indices) {
        double[][] data = new double[indices.size()][matrix.cols()];
        for (int i = 0; i < indices.size(); i++) {
            for (int j = 0; j < matrix.cols(); j++) {
                data[i][j] = matrix.get(indices.get(i), j);
            }
        }
        return Linalg.matrix(data);
    }
    
    /**
     * 提取指定索引的标签
     */
    private String[] extractLabels(String[] labels, List<Integer> indices) {
        String[] result = new String[indices.size()];
        for (int i = 0; i < indices.size(); i++) {
            result[i] = labels[indices.get(i)];
        }
        return result;
    }
    
    /**
     * 计算准确率
     */
    private double computeAccuracy(String[] predictions, String[] trueLabels) {
        int correct = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i].equals(trueLabels[i])) {
                correct++;
            }
        }
        return (double) correct / predictions.length;
    }
    
    /**
     * 单样本预测 - 实现IClassification接口要求
     * @param features 单个样本的特征向量
     * @return 预测的类别标签
     */
    @Override
    public String predict(IVector features) {
        if (!isTrained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        
        // 将向量转换为1x特征数的矩阵
        double[][] featureMatrix = new double[1][features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureMatrix[0][i] = features.get(i);
        }
        IMatrix singleSampleMatrix = Linalg.matrix(featureMatrix);
        
        // 使用现有的批量预测方法
        EnsembleResult result = predict(singleSampleMatrix);
        
        // 返回第一个（也是唯一一个）预测结果
        return result.predictions[0];
    }
    
    /**
     * 批量预测 - 返回String数组（为了与IClassification接口兼容）
     * @param features 特征矩阵
     * @return 预测的类别标签数组
     */
    @Override
    public String[] predictBatch(IMatrix features) {
        EnsembleResult result = predict(features);
        return result.predictions;
    }

    @Override
    public Map<String, Double> predictProb(IVector x) {
        if (!isTrained) {
            throw new IllegalStateException("模型必须先训练才能进行预测");
        }

        if (x == null) {
            throw new IllegalArgumentException("输入特征向量不能为null");
        }

        // 将向量转换为1x特征数的矩阵
        double[][] featureMatrix = new double[1][x.size()];
        for (int i = 0; i < x.size(); i++) {
            featureMatrix[0][i] = x.get(i);
        }
        IMatrix singleSampleMatrix = Linalg.matrix(featureMatrix);

        // 使用现有的predictBatchWithProbs方法
        BatchPredResult batchResult = predictBatchWithProbs(singleSampleMatrix);

        // 转换为Map<String, Double>格式
        Map<String, Double> result = new HashMap<>();
        
        if (batchResult.isBinaryClassification()) {
            // 二分类：probabilities是double[]，存储正类概率
            double positiveProb = batchResult.getProbabilities()[0];
            double negativeProb = 1.0 - positiveProb;
            // 根据classLabels确定哪个是positive/negative
            if (classLabels.length >= 2) {
                result.put(classLabels[0], negativeProb);
                result.put(classLabels[1], positiveProb);
            }
        } else {
            // 多分类：classProbabilities是double[][]
            double[][] probs = batchResult.getClassProbabilities();
            for (int j = 0; j < classLabels.length; j++) {
                result.put(classLabels[j], probs[0][j]);
            }
        }

        return result;
    }
    
    

    @Override
    public BatchPredResult predictBatchWithProbs(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("模型必须先训练才能进行预测");
        }

        if (features == null) {
            throw new IllegalArgumentException("特征矩阵不能为null");
        }

        // 使用现有的predict方法获取集成结果
        EnsembleResult result = predict(features);

        // 将IMatrix概率转换为double[][]
        int numSamples = result.probabilities.rows();
        int numClasses = result.probabilities.cols();
        double[][] classProbabilities = new double[numSamples][numClasses];

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numClasses; j++) {
                classProbabilities[i][j] = result.probabilities.get(i, j);
            }
        }

        return new BatchPredResult(result.predictions, classProbabilities);
    }
    
    
    
    // Getters
    @Override
    public boolean isTrained() { return isTrained; }
    public String[] getClassLabels() { return classLabels; }
    public IVector getClassifierWeights() { return classifierWeights; }
    public EnsembleStrategy getStrategy() { return strategy; }

    @Override
    public ClassificationMetrics getMetrics() {
        return metrics;
    }

    @Override
    public void setMetrics(ClassificationMetrics metrics) {
        this.metrics = metrics;
    }
    
    
    
    
    /**
     * 将模型保存在本地
     * @param path 保存路径
     */
    // save() inherited from ISerializableModel default

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("strategy", strategy.name());
        p.put("isTrained", isTrained);
        p.put("randomSeed", randomSeed);
        if (classLabels != null) p.put("classLabels", new ArrayList<>(Arrays.asList(classLabels)));
        if (classifierWeights != null) p.put("classifierWeights", classifierWeights.toDoubleArray());
        if (randomForest != null) p.put("randomForest", randomForest.toParams());
        if (logisticRegression != null) p.put("logisticRegression", logisticRegression.toParams());
        if (xgboost != null) p.put("xgboost", xgboost.toParams());
        return p;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void fromParams(Map<String, Object> p) {
        this.strategy = EnsembleStrategy.valueOf((String) p.get("strategy"));
        this.isTrained = (Boolean) p.get("isTrained");
        this.randomSeed = ((Number) p.get("randomSeed")).longValue();
        List<String> labelsList = (List<String>) p.get("classLabels");
        this.classLabels = labelsList != null ? labelsList.toArray(new String[0]) : null;
        double[] w = (double[]) p.get("classifierWeights");
        if (w != null) this.classifierWeights = Linalg.vector(w);
        Map<String, Object> rfParams = (Map<String, Object>) p.get("randomForest");
        if (rfParams != null) {
            this.randomForest = new RereRandomForest();
            this.randomForest.fromParams(rfParams);
        }
        Map<String, Object> lrParams = (Map<String, Object>) p.get("logisticRegression");
        if (lrParams != null) {
            this.logisticRegression = new RereLogisticRegression();
            this.logisticRegression.fromParams(lrParams);
        }
        Map<String, Object> xgbParams = (Map<String, Object>) p.get("xgboost");
        if (xgbParams != null) {
            this.xgboost = new RereXGboost();
            this.xgboost.fromParams(xgbParams);
        }
    }
}