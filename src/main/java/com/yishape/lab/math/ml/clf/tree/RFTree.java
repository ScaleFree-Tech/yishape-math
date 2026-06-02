package com.yishape.lab.math.ml.clf.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.*;

/**
 * 分类决策树类（用于随机森林）
 * <p>
 * 实现传统的分类决策树，使用信息增益或基尼不纯度作为分裂准则。
 * 支持特征随机选择和Bootstrap采样。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RFTree {
    
    /** 根节点 */
    private RFTreeNode root;
    
    /** 最大深度 */
    private int maxDepth;
    
    /** 最小分裂样本数 */
    private int minSamplesSplit;
    
    /** 最小叶子样本数 */
    private int minSamplesLeaf;
    
    /** 每次分裂时考虑的最大特征数 */
    private int maxFeatures;
    
    /** 分裂准则 */
    private SplitCriterion criterion;
    
    /** 随机数生成器 */
    private Random random;
    
    /** 特征重要性 */
    private double[] featureImportance;
    
    /** 类别数量 */
    private int numClasses;
    
    /** 标签映射 */
    private Map<String, Integer> labelMapping;
    
    /** 反向标签映射 */
    private Map<Integer, String> reverseLabelMapping;

    /** 训练集特征缓存（行主序），加速分裂搜索 */
    private double[][] Xdata;
    
    /**
     * 分裂准则枚举
     */
    public enum SplitCriterion {
        GINI,           // 基尼不纯度
        ENTROPY         // 信息熵
    }
    
    /**
     * 构造函数
     * @param maxDepth 最大深度
     * @param minSamplesSplit 最小分裂样本数
     * @param minSamplesLeaf 最小叶子样本数
     * @param maxFeatures 最大特征数
     * @param criterion 分裂准则
     * @param randomSeed 随机种子
     */
    public RFTree(int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                                    int maxFeatures, SplitCriterion criterion, long randomSeed) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.maxFeatures = maxFeatures;
        this.criterion = criterion;
        this.random = new Random(randomSeed);
    }
    
    /**
     * 训练决策树
     * @param features 特征矩阵
     * @param labels 标签数组
     * @param sampleIndices 样本索引（用于Bootstrap采样）
     */
    public void fit(IMatrix features, String[] labels, int[] sampleIndices) {
        preprocessLabels(labels);

        this.featureImportance = new double[features.cols()];
        this.Xdata = imatrixToDouble2d(features);

        int[] numericLabels = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            numericLabels[i] = labelMapping.get(labels[i]);
        }

        this.root = buildTree(numericLabels, sampleIndices, 0, null);
    }
    
    private static double[][] imatrixToDouble2d(IMatrix features) {
        int r = features.rows();
        int c = features.cols();
        double[][] x = new double[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                x[i][j] = features.get(i, j);
            }
        }
        return x;
    }

    /**
     * 递归构建决策树
     * @param labels 数值标签数组（全局下标 → 类号）
     * @param sampleIndices 本结点样本行号
     * @param depth 当前深度
     * @param availableFeatures 可用特征索引（用于特征随机选择）
     * @return 树节点
     */
    private RFTreeNode buildTree(int[] labels, int[] sampleIndices,
                                           int depth, Set<Integer> availableFeatures) {

        int numSamples = sampleIndices.length;

        Map<Integer, Integer> classCounts = new HashMap<>(numClasses * 2);
        for (int idx : sampleIndices) {
            int label = labels[idx];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }

        int majorityClass = classCounts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .get().getKey();

        String majorityLabel = reverseLabelMapping.get(majorityClass);

        if (shouldStop(depth, numSamples, classCounts)) {
            return new RFTreeNode(majorityLabel, depth, numSamples, classCounts);
        }

        Set<Integer> candidateFeatures = selectRandomFeatures(Xdata[0].length, availableFeatures);

        SplitResult bestSplit = findBestSplit(labels, sampleIndices, candidateFeatures);

        if (bestSplit == null || bestSplit.gain <= 0) {
            return new RFTreeNode(majorityLabel, depth, numSamples, classCounts);
        }

        featureImportance[bestSplit.featureIndex] += bestSplit.gain * numSamples;

        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();

        for (int idx : sampleIndices) {
            if (Xdata[idx][bestSplit.featureIndex] <= bestSplit.threshold) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }

        if (leftIndices.size() < minSamplesLeaf || rightIndices.size() < minSamplesLeaf) {
            return new RFTreeNode(majorityLabel, depth, numSamples, classCounts);
        }

        RFTreeNode leftChild = buildTree(labels,
                leftIndices.stream().mapToInt(i -> i).toArray(), depth + 1, candidateFeatures);
        RFTreeNode rightChild = buildTree(labels,
                rightIndices.stream().mapToInt(i -> i).toArray(), depth + 1, candidateFeatures);

        return new RFTreeNode(bestSplit.featureIndex, bestSplit.threshold,
                leftChild, rightChild, depth, numSamples, bestSplit.gain, classCounts);
    }
    
    /**
     * 随机选择特征子集
     * @param totalFeatures 总特征数
     * @param availableFeatures 可用特征集合
     * @return 选中的特征索引集合
     */
    private Set<Integer> selectRandomFeatures(int totalFeatures, Set<Integer> availableFeatures) {
        Set<Integer> allFeatures;
        if (availableFeatures != null) {
            allFeatures = availableFeatures;
        } else {
            allFeatures = new HashSet<>();
            for (int i = 0; i < totalFeatures; i++) {
                allFeatures.add(i);
            }
        }
        
        int numFeaturesToSelect = Math.min(maxFeatures, allFeatures.size());
        
        List<Integer> featureList = new ArrayList<>(allFeatures);
        Collections.shuffle(featureList, random);
        
        return new HashSet<>(featureList.subList(0, numFeaturesToSelect));
    }
    
    /**
     * 检查是否应该停止分裂
     * @param depth 当前深度
     * @param numSamples 样本数量
     * @param classCounts 类别计数
     * @return 是否停止
     */
    private boolean shouldStop(int depth, int numSamples, Map<Integer, Integer> classCounts) {
        return depth >= maxDepth || 
               numSamples < minSamplesSplit || 
               classCounts.size() <= 1; // 纯节点
    }
    
    /**
     * 在候选特征上寻找最佳分裂（按特征值排序后 O(n) 扫描，避免对每个阈值重复划分样本）。
     */
    private SplitResult findBestSplit(int[] labels, int[] sampleIndices,
                                     Set<Integer> candidateFeatures) {

        SplitResult bestSplit = null;
        double bestGain = 0.0;
        double currentImpurity = computeImpurity(labels, sampleIndices);
        int n = sampleIndices.length;

        int[] totalByClass = new int[numClasses];
        for (int idxRow : sampleIndices) {
            totalByClass[labels[idxRow]]++;
        }

        int[] leftCount = new int[numClasses];
        int[] rightCount = new int[numClasses];

        for (int featureIdx : candidateFeatures) {
            Integer[] ord = new Integer[n];
            for (int t = 0; t < n; t++) {
                ord[t] = sampleIndices[t];
            }
            final int fj = featureIdx;
            Arrays.sort(ord, (a, b) -> Double.compare(Xdata[a][fj], Xdata[b][fj]));

            Arrays.fill(leftCount, 0);
            int leftSize = 0;

            for (int i = 0; i < n - 1; i++) {
                int row = ord[i];
                leftCount[labels[row]]++;
                leftSize++;
                int rightSize = n - leftSize;

                if (Double.compare(Xdata[ord[i]][fj], Xdata[ord[i + 1]][fj]) == 0) {
                    continue;
                }
                if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) {
                    continue;
                }

                double threshold = (Xdata[ord[i]][fj] + Xdata[ord[i + 1]][fj]) / 2.0;

                for (int c = 0; c < numClasses; c++) {
                    rightCount[c] = totalByClass[c] - leftCount[c];
                }

                double leftImpurity;
                double rightImpurity;
                if (criterion == SplitCriterion.GINI) {
                    leftImpurity = giniFromIntCounts(leftCount, leftSize);
                    rightImpurity = giniFromIntCounts(rightCount, rightSize);
                } else {
                    leftImpurity = entropyFromIntCounts(leftCount, leftSize);
                    rightImpurity = entropyFromIntCounts(rightCount, rightSize);
                }

                double weightedImpurity =
                        (leftSize / (double) n) * leftImpurity + (rightSize / (double) n) * rightImpurity;
                double gain = currentImpurity - weightedImpurity;

                if (gain > bestGain) {
                    bestGain = gain;
                    bestSplit = new SplitResult(featureIdx, threshold, gain);
                }
            }
        }

        return bestSplit;
    }

    private static double giniFromIntCounts(int[] counts, int total) {
        double sumSq = 0.0;
        for (int c : counts) {
            if (c > 0) {
                double p = (double) c / total;
                sumSq += p * p;
            }
        }
        return 1.0 - sumSq;
    }

    private static double entropyFromIntCounts(int[] counts, int total) {
        double e = 0.0;
        for (int c : counts) {
            if (c > 0) {
                double p = (double) c / total;
                e -= p * (Math.log(p) / Math.log(2.0));
            }
        }
        return e;
    }
    
    /**
     * 计算不纯度
     * @param labels 标签数组
     * @param sampleIndices 样本索引
     * @return 不纯度值
     */
    private double computeImpurity(int[] labels, int[] sampleIndices) {
        Map<Integer, Integer> classCounts = new HashMap<>();
        for (int idx : sampleIndices) {
            int label = labels[idx];
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        
        int totalSamples = sampleIndices.length;
        
        if (criterion == SplitCriterion.GINI) {
            return computeGiniImpurity(classCounts, totalSamples);
        } else {
            return computeEntropyImpurity(classCounts, totalSamples);
        }
    }
    
    /**
     * 计算基尼不纯度
     * @param classCounts 类别计数
     * @param totalSamples 总样本数
     * @return 基尼不纯度
     */
    private double computeGiniImpurity(Map<Integer, Integer> classCounts, int totalSamples) {
        double sumSquaredProbs = 0.0;
        for (int count : classCounts.values()) {
            double p = (double) count / totalSamples;
            sumSquaredProbs += p * p;
        }
        return 1.0 - sumSquaredProbs;
    }
    
    /**
     * 计算信息熵
     * @param classCounts 类别计数
     * @param totalSamples 总样本数
     * @return 信息熵
     */
    private double computeEntropyImpurity(Map<Integer, Integer> classCounts, int totalSamples) {
        double entropy = 0.0;
        for (int count : classCounts.values()) {
            if (count > 0) {
                double probability = (double) count / totalSamples;
                entropy -= probability * (Math.log(probability) / Math.log(2));
            }
        }
        return entropy;
    }

    /**
     * 预测单个样本
     * @param features 特征向量
     * @return 预测标签
     */
    public String predict(IVector features) {
        if (root == null) {
            throw new IllegalStateException("模型尚未训练");
        }
        return root.predict(features);
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
     * 标签预处理
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
    
    // ==================== Getters ====================
    
    public double[] getFeatureImportance() {
        return featureImportance.clone();
    }
    
    public int getDepth() {
        return root != null ? root.getTreeDepth() : 0;
    }
    
    public int getLeafCount() {
        return root != null ? root.getLeafCount() : 0;
    }
    
    public RFTreeNode getRoot() {
        return root;
    }
    
    public Map<String, Integer> getLabelMapping() {
        return new HashMap<>(labelMapping);
    }
    
    public Map<Integer, String> getReverseLabelMapping() {
        return new HashMap<>(reverseLabelMapping);
    }
    
    // ==================== JSON persistence ====================

    Map<String, Object> toParams() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("maxDepth", maxDepth);
        m.put("minSamplesSplit", minSamplesSplit);
        m.put("minSamplesLeaf", minSamplesLeaf);
        m.put("maxFeatures", maxFeatures);
        m.put("criterion", criterion.name());
        m.put("numClasses", numClasses);
        m.put("labelMapping", new HashMap<>(labelMapping));
        m.put("reverseLabelMapping", reverseLabelMappingToString());
        m.put("featureImportance", featureImportance != null ? featureImportance.clone() : null);
        if (root != null) m.put("root", root.toParams());
        return m;
    }

    private Map<String, String> reverseLabelMappingToString() {
        Map<String, String> m = new LinkedHashMap<>();
        for (Map.Entry<Integer, String> e : reverseLabelMapping.entrySet()) {
            m.put(String.valueOf(e.getKey()), e.getValue());
        }
        return m;
    }

    @SuppressWarnings("unchecked")
    static RFTree fromParams(Map<String, Object> m) {
        int maxDepth = ((Number) m.get("maxDepth")).intValue();
        int minSamplesSplit = ((Number) m.get("minSamplesSplit")).intValue();
        int minSamplesLeaf = ((Number) m.get("minSamplesLeaf")).intValue();
        int maxFeatures = ((Number) m.get("maxFeatures")).intValue();
        SplitCriterion criterion = SplitCriterion.valueOf((String) m.get("criterion"));
        int numClasses = ((Number) m.get("numClasses")).intValue();
        Map<String, Integer> labelMapping = (Map<String, Integer>) m.get("labelMapping");
        Map<String, String> revStr = (Map<String, String>) m.get("reverseLabelMapping");
        Map<Integer, String> reverseLabelMapping = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : revStr.entrySet()) {
            reverseLabelMapping.put(Integer.parseInt(e.getKey()), e.getValue());
        }
        double[] featureImportance = (double[]) m.get("featureImportance");
        Map<String, Object> rootMap = (Map<String, Object>) m.get("root");

        RFTree tree = new RFTree(maxDepth, minSamplesSplit, minSamplesLeaf, maxFeatures, criterion, 0);
        tree.numClasses = numClasses;
        tree.labelMapping = labelMapping;
        tree.reverseLabelMapping = reverseLabelMapping;
        tree.featureImportance = featureImportance;
        if (rootMap != null) {
            tree.root = RFTreeNode.fromParams(rootMap);
        }
        return tree;
    }

    /**
     * 分裂结果内部类
     */
    private static class SplitResult {
        int featureIndex;
        double threshold;
        double gain;
        
        SplitResult(int featureIndex, double threshold, double gain) {
            this.featureIndex = featureIndex;
            this.threshold = threshold;
            this.gain = gain;
        }
    }
}