package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.cls.BatchPredictionResult;
import com.yishape.lab.math.ml.cls.IClassification;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.io.*;

/**
 * 随机森林分类器
 * <p>
 * 实现基于Bootstrap聚合和特征随机选择的随机森林算法。
 * 支持多线程训练、特征重要性计算和袋外评估。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereRandomForest implements IClassification, IGradientFunction, IObjectiveFunction, ISerializableModel {
    
    private static final long serialVersionUID = 1L;
    
    /** 决策树数量 */
    private int nEstimators;
    
    /** 最大深度 */
    private int maxDepth;
    
    /** 最小分裂样本数 */
    private int minSamplesSplit;
    
    /** 最小叶子样本数 */
    private int minSamplesLeaf;
    
    /** 每棵树考虑的最大特征数 */
    private int maxFeatures;
    
    /** 是否使用Bootstrap采样 */
    private boolean bootstrap;
    
    /** 分裂准则 */
    private RFTree.SplitCriterion criterion;
    
    /** 随机种子 */
    private long randomSeed;
    
    /** 随机数生成器 */
    private Random random;
    
    /** 决策树列表 */
    private List<RFTree> trees;
    
    /** 训练结果 */
    private RandomForestResult result;
    
    /** 是否已训练 */
    private boolean isTrained;
    
    /** 特征数量 */
    private int numFeatures;
    
    /** 样本数量 */
    private int numSamples;
    
    /** 类别标签映射 */
    private Map<String, Integer> labelMapping;
    
    /** 反向标签映射 */
    private Map<Integer, String> reverseLabelMapping;
    
    /** 类别数量 */
    private int numClasses;
    
    /**
     * 默认构造函数
     */
    public RereRandomForest() {
        this(100, -1, 2, 1, -1, true, 
             RFTree.SplitCriterion.GINI, 42L);
    }
    
    /**
     * 完整构造函数
     * @param nEstimators 决策树数量
     * @param maxDepth 最大深度（-1表示无限制）
     * @param minSamplesSplit 最小分裂样本数
     * @param minSamplesLeaf 最小叶子样本数
     * @param maxFeatures 最大特征数（-1表示sqrt(n_features)）
     * @param bootstrap 是否使用Bootstrap采样
     * @param criterion 分裂准则
     * @param randomSeed 随机种子
     */
    public RereRandomForest(int nEstimators, int maxDepth, int minSamplesSplit, 
                           int minSamplesLeaf, int maxFeatures, boolean bootstrap,
                           RFTree.SplitCriterion criterion, long randomSeed) {
        this.nEstimators = nEstimators;
        this.maxDepth = maxDepth > 0 ? maxDepth : Integer.MAX_VALUE;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.bootstrap = bootstrap;
        this.criterion = criterion;
        this.randomSeed = randomSeed;
        this.random = new Random(randomSeed);
        this.trees = new ArrayList<>();
        this.isTrained = false;
    }
    
    @Override
    public RandomForestResult fit(IMatrix features, String[] labels) {
        if (features == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签不能为空");
        }
        
        if (features.rows() != labels.length) {
            throw new IllegalArgumentException("特征矩阵行数与标签数量不匹配");
        }
        
        this.numSamples = features.rows();
        this.numFeatures = features.cols();
        
        // 预处理标签
        preprocessLabels(labels);
        
        // 设置默认的maxFeatures
        if (maxFeatures <= 0) {
            this.maxFeatures = (int) Math.sqrt(numFeatures);
        }
        
        // 初始化决策树列表
        trees.clear();
        
        // 训练多个决策树
        List<int[]> bootstrapSamples = new ArrayList<>();
        List<int[]> oobSamples = new ArrayList<>();
        
        for (int i = 0; i < nEstimators; i++) {
            // 生成Bootstrap样本
            int[] bootstrapIndices;
            int[] oobIndices;
            
            if (bootstrap) {
                BootstrapResult bootstrapResult = generateBootstrapSample(numSamples, random);
                bootstrapIndices = bootstrapResult.inBagIndices;
                oobIndices = bootstrapResult.outOfBagIndices;
            } else {
                bootstrapIndices = new int[numSamples];
                for (int j = 0; j < numSamples; j++) {
                    bootstrapIndices[j] = j;
                }
                oobIndices = new int[0];
            }
            
            bootstrapSamples.add(bootstrapIndices);
            oobSamples.add(oobIndices);
            
            // 创建并训练决策树
            RFTree tree = new RFTree(
                    maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, 
                    criterion, random.nextLong());
            
            tree.fit(features, labels, bootstrapIndices);
            trees.add(tree);
        }
        
        // 计算特征重要性
        double[] featureImportance = computeFeatureImportance();
        
        // 计算袋外评估
        double oobScore = computeOOBScore(features, labels, oobSamples);
        
        // 设置训练状态为true，以便可以调用predict方法
        this.isTrained = true;
        
        // 计算训练准确率
        String[] trainPredictions = predictBatch(features);
        double trainAccuracy = computeAccuracy(labels, trainPredictions);
        
        // 创建结果对象
        this.result = new RandomForestResult(
                trees, nEstimators, maxDepth, maxFeatures, bootstrap, randomSeed,
                featureImportance, numClasses, labelMapping, reverseLabelMapping,
                oobScore, trainAccuracy);
        
        return result;
    }
    
    @Override
    public String predict(IVector x) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练");
        }
        
        // 收集所有树的预测结果
        Map<String, Integer> votes = new HashMap<>();
        
        for (RFTree tree : trees) {
            String prediction = tree.predict(x);
            votes.put(prediction, votes.getOrDefault(prediction, 0) + 1);
        }
        
        // 返回得票最多的类别
        return votes.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get().getKey();
    }
    
    /**
     * 批量预测
     * @param features 特征矩阵
     * @return 预测标签数组
     */
    @Override
    public String[] predictBatch(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练");
        }
        
        int numSamples = features.rows();
        String[] predictions = new String[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            predictions[i] = predict(features.getRow(i));
        }
        
        return predictions;
    }

    @Override
    public BatchPredictionResult predictBatchWithProbabilities(IMatrix features) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练");
        }

        int numSamples = features.rows();
        String[] predictions = new String[numSamples];
        double[][] classProbabilities = new double[numSamples][numClasses];

        // 获取反向标签映射的列表（按索引排序）
        List<String> classLabelsList = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            classLabelsList.add(reverseLabelMapping.get(i));
        }

        for (int i = 0; i < numSamples; i++) {
            IVector instance = features.getRow(i);
            Map<String, Double> probMap = predictProba(instance);

            // 找到概率最大的类别作为预测
            String predictedClass = null;
            double maxProb = -1.0;

            // 获取预测标签和概率
            for (Map.Entry<String, Double> entry : probMap.entrySet()) {
                String className = entry.getKey();
                double prob = entry.getValue();

                // 填充概率矩阵
                int classIdx = labelMapping.get(className);
                classProbabilities[i][classIdx] = prob;

                // 记录最大概率的类别
                if (prob > maxProb) {
                    maxProb = prob;
                    predictedClass = className;
                }
            }

            predictions[i] = predictedClass;
        }

        return new BatchPredictionResult(predictions, classProbabilities);
    }

    
    
    
    /**
     * 预测概率分布
     * @param x 特征向量
     * @return 各类别的概率分布
     */
    public Map<String, Double> predictProba(IVector x) {
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练");
        }
        
        // 使用向量来统计各类别的投票
        IVector classCounts = Linalg.zeros(numClasses);
        
        for (RFTree tree : trees) {
            String prediction = tree.predict(x);
            int classIndex = labelMapping.get(prediction);
            // 增加对应类别的计数
            double currentCount = classCounts.get(classIndex).doubleValue();
            classCounts.set(classIndex, currentCount + 1.0);
        }
        
        // 使用向量归一化计算概率
        IVector probabilities = classCounts.normalize();
        
        // 转换为Map格式
        Map<String, Double> result = new HashMap<>();
        for (Map.Entry<Integer, String> entry : reverseLabelMapping.entrySet()) {
            int classIndex = entry.getKey();
            String className = entry.getValue();
            result.put(className, probabilities.get(classIndex).doubleValue());
        }
        
        return result;
    }
    
    @Override
    public IVector computeGradient(IVector x) {
        // 对于随机森林，梯度计算通常用于特征重要性分析
        // 这里实现一个简化版本，计算预测概率对输入特征的敏感性
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练");
        }
        
        double[] gradient = new double[x.size()];
        double epsilon = 1e-6;
        
        // 获取原始预测概率
        Map<String, Double> originalProba = predictProba(x);
        String predictedClass = predict(x);
        double originalProb = originalProba.get(predictedClass);
        
        // 计算数值梯度
        for (int i = 0; i < x.size(); i++) {
            // 创建扰动后的特征向量
            IVector perturbedX = x.copy();
            perturbedX.set(i, x.get(i).doubleValue() + epsilon);
            
            // 计算扰动后的预测概率
            Map<String, Double> perturbedProba = predictProba(perturbedX);
            double perturbedProb = perturbedProba.get(predictedClass);
            
            // 计算梯度
            gradient[i] = (perturbedProb - originalProb) / epsilon;
        }
        
        return Linalg.vector(gradient);
    }
    
    @Override
    public double computeObjective(IVector x) {
        // 对于随机森林，目标函数可以定义为预测的不确定性（熵）
        if (!isTrained) {
            throw new IllegalStateException("模型尚未训练");
        }
        
        Map<String, Double> probabilities = predictProba(x);
        double entropy = 0.0;
        
        for (double prob : probabilities.values()) {
            if (prob > 0) {
                entropy -= prob * (Math.log(prob) / Math.log(2));
            }
        }
        
        return entropy;
    }
    
    /**
     * 生成Bootstrap样本
     * @param numSamples 样本数量
     * @param random 随机数生成器
     * @return Bootstrap结果
     */
    private BootstrapResult generateBootstrapSample(int numSamples, Random random) {
        Set<Integer> inBagSet = new HashSet<>();
        List<Integer> inBagList = new ArrayList<>();
        
        // 有放回抽样
        for (int i = 0; i < numSamples; i++) {
            int index = random.nextInt(numSamples);
            inBagList.add(index);
            inBagSet.add(index);
        }
        
        // 计算袋外样本
        List<Integer> oobList = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            if (!inBagSet.contains(i)) {
                oobList.add(i);
            }
        }
        
        int[] inBagIndices = inBagList.stream().mapToInt(i -> i).toArray();
        int[] oobIndices = oobList.stream().mapToInt(i -> i).toArray();
        
        return new BootstrapResult(inBagIndices, oobIndices);
    }
    
    /**
     * 计算特征重要性
     * @return 特征重要性数组
     */
    private double[] computeFeatureImportance() {
        // 使用IVector进行向量运算
        IVector totalImportance = Linalg.zeros(numFeatures);
        
        for (RFTree tree : trees) {
            double[] treeImportance = tree.getFeatureImportance();
            IVector treeImportanceVector = Linalg.vector(treeImportance);
            totalImportance = totalImportance.add(treeImportanceVector);
        }
        
        // 使用向量的归一化方法
        IVector normalizedImportance = totalImportance.normalize();
        
        // 转换回数组格式以保持接口兼容性
        double[] result = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            result[i] = normalizedImportance.get(i).doubleValue();
        }
        
        return result;
    }
    
    /**
     * 计算袋外评估分数
     * @param features 特征矩阵
     * @param labels 真实标签
     * @param oobSamples 袋外样本索引列表
     * @return OOB分数
     */
    private double computeOOBScore(IMatrix features, String[] labels, List<int[]> oobSamples) {
        if (!bootstrap) {
            return Double.NaN; // 没有Bootstrap采样时无法计算OOB分数
        }
        
        Map<Integer, Map<String, Integer>> oobVotes = new HashMap<>();
        
        // 收集每个样本的袋外预测
        for (int treeIdx = 0; treeIdx < trees.size(); treeIdx++) {
            RFTree tree = trees.get(treeIdx);
            int[] oobIndices = oobSamples.get(treeIdx);
            
            for (int sampleIdx : oobIndices) {
                String prediction = tree.predict(features.getRow(sampleIdx));
                
                oobVotes.computeIfAbsent(sampleIdx, k -> new HashMap<>())
                        .merge(prediction, 1, Integer::sum);
            }
        }
        
        // 计算OOB准确率
        int correctPredictions = 0;
        int totalPredictions = 0;
        
        for (Map.Entry<Integer, Map<String, Integer>> entry : oobVotes.entrySet()) {
            int sampleIdx = entry.getKey();
            Map<String, Integer> votes = entry.getValue();
            
            String predictedLabel = votes.entrySet().stream()
                    .max(Map.Entry.comparingByValue())
                    .get().getKey();
            
            if (predictedLabel.equals(labels[sampleIdx])) {
                correctPredictions++;
            }
            totalPredictions++;
        }
        
        return totalPredictions > 0 ? (double) correctPredictions / totalPredictions : 0.0;
    }
    
    /**
     * 计算准确率
     * @param trueLabels 真实标签
     * @param predictedLabels 预测标签
     * @return 准确率
     */
    private double computeAccuracy(String[] trueLabels, String[] predictedLabels) {
        if (trueLabels.length != predictedLabels.length) {
            throw new IllegalArgumentException("标签数组长度不匹配");
        }
        
        // 使用向量化操作计算准确率
        int correct = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i].equals(predictedLabels[i])) {
                correct++;
            }
        }
        
        return (double) correct / trueLabels.length;
    }
    
    /**
     * 预处理标签
     * @param labels 原始标签数组
     */
    private void preprocessLabels(String[] labels) {
        labelMapping = new HashMap<>();
        reverseLabelMapping = new HashMap<>();
        
        int nextLabel = 0;
        for (String label : labels) {
            if (!labelMapping.containsKey(label)) {
                labelMapping.put(label, nextLabel);
                reverseLabelMapping.put(nextLabel, label);
                nextLabel++;
            }
        }
        
        this.numClasses = labelMapping.size();
    }
    
    // ==================== Getters and Setters ====================
    
    public int getnEstimators() {
        return nEstimators;
    }
    
    public void setnEstimators(int nEstimators) {
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
    
    public RFTree.SplitCriterion getCriterion() {
        return criterion;
    }
    
    public void setCriterion(RFTree.SplitCriterion criterion) {
        this.criterion = criterion;
    }
    
    public long getRandomSeed() {
        return randomSeed;
    }
    
    public void setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
        this.random = new Random(randomSeed);
    }
    
    public List<RFTree> getTrees() {
        return new ArrayList<>(trees);
    }
    
    public RandomForestResult getResult() {
        return result;
    }
    
    public boolean isTrained() {
        return isTrained;
    }
    
    public double[] getFeatureImportance() {
        if (result != null && result.getFeatureImportance() != null) {
            IVector featureImportanceVector = result.getFeatureImportance();
            double[] featureImportanceArray = new double[featureImportanceVector.size()];
            for (int i = 0; i < featureImportanceVector.size(); i++) {
                featureImportanceArray[i] = featureImportanceVector.get(i).doubleValue();
            }
            return featureImportanceArray;
        }
        return null;
    }
    
    public Map<String, Integer> getLabelMapping() {
        return labelMapping != null ? new HashMap<>(labelMapping) : null;
    }
    
    public Map<Integer, String> getReverseLabelMapping() {
        return reverseLabelMapping != null ? new HashMap<>(reverseLabelMapping) : null;
    }
    
    /**
     * Bootstrap结果内部类
     */
    private static class BootstrapResult {
        final int[] inBagIndices;
        final int[] outOfBagIndices;
        
        BootstrapResult(int[] inBagIndices, int[] outOfBagIndices) {
            this.inBagIndices = inBagIndices;
            this.outOfBagIndices = outOfBagIndices;
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
