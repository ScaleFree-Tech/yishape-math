package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.cls.BatchPredictionResult;
import com.yishape.lab.math.ml.cls.ClassificationResult;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.ml.cls.IClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.*;

/**
 * 单棵决策树分类器：连续特征上的二元分裂；支持 CART（基尼/熵）与 C4.5 <em>风格</em>增益率。
 * 若需 Weka {@code J48} 级别的完整 C4.5（名目属性、缺失值、剪枝策略等），请在外部扩展或使用专用库。
 * <p>
 * 标签索引按字典序稳定分配（与 {@link com.yishape.lab.math.ml.cls.tree.RereXGboost} 一致）。
 * </p>
 */
public class RereDecisionTree implements IClassifier, ISerializableModel {

    private static final Logger log = LoggerFactory.getLogger(RereDecisionTree.class);
    private static final long serialVersionUID = 1L;

    private DecisionTreeCriterion criterion = DecisionTreeCriterion.CART_GINI;
    private int maxDepth = Integer.MAX_VALUE;
    private int minSamplesSplit = 2;
    private int minSamplesLeaf = 1;

    private Map<String, Integer> labelToIndex = new HashMap<>();
    private Map<Integer, String> indexToLabel = new HashMap<>();
    private int numClasses;
    private DecisionTreeNode root;
    private double[] featureImportance;
    private int numFeatures;
    private boolean trained;
    private ClassificationMetrics metrics;

    public RereDecisionTree() {
    }

    public RereDecisionTree(DecisionTreeCriterion criterion, int maxDepth,
                            int minSamplesSplit, int minSamplesLeaf) {
        this.criterion = criterion != null ? criterion : DecisionTreeCriterion.CART_GINI;
        this.maxDepth = maxDepth > 0 ? maxDepth : Integer.MAX_VALUE;
        this.minSamplesSplit = Math.max(2, minSamplesSplit);
        this.minSamplesLeaf = Math.max(1, minSamplesLeaf);
    }

    public DecisionTreeCriterion getCriterion() {
        return criterion;
    }

    public void setCriterion(DecisionTreeCriterion criterion) {
        this.criterion = criterion != null ? criterion : DecisionTreeCriterion.CART_GINI;
    }

    public int getMaxDepth() {
        return maxDepth;
    }

    public void setMaxDepth(int maxDepth) {
        this.maxDepth = maxDepth > 0 ? maxDepth : Integer.MAX_VALUE;
    }

    public int getMinSamplesSplit() {
        return minSamplesSplit;
    }

    public void setMinSamplesSplit(int minSamplesSplit) {
        this.minSamplesSplit = Math.max(2, minSamplesSplit);
    }

    public int getMinSamplesLeaf() {
        return minSamplesLeaf;
    }

    public void setMinSamplesLeaf(int minSamplesLeaf) {
        this.minSamplesLeaf = Math.max(1, minSamplesLeaf);
    }

    @Override
    public ClassificationResult fit(IMatrix feature, String[] labels) {
        validateTrain(feature, labels);
        buildLabelMapping(labels);
        int n = feature.getRowNum();
        int p = feature.getColNum();
        this.numFeatures = p;
        featureImportance = new double[p];

        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = labelToIndex.get(labels[i]);
        }
        int[] allIdx = new int[n];
        for (int i = 0; i < n; i++) {
            allIdx[i] = i;
        }

        root = buildTree(feature, y, allIdx, 0);

        trained = true;

        DecisionTreeResult result = new DecisionTreeResult();
        result.setCriterion(criterion);
        result.setMaxDepthParam(maxDepth == Integer.MAX_VALUE ? -1 : maxDepth);
        result.setTreeDepth(root != null ? root.depthBelow() : 0);
        result.setLeafCount(root != null ? root.leafCount() : 0);
        result.setTrained(true);
        result.setNumClasses(numClasses);
        result.setLabelMapping(new HashMap<>(labelToIndex));
        result.setReverseLabelMapping(new HashMap<>(indexToLabel));
        result.setNumFeatures(p);
        result.setFeatureImportanceFromArray(featureImportance);

        ClassificationMetrics m = ClassificationMetrics.compute(this, feature, labels);
        result.setTrainAccuracy(m.getAccuracy());
        this.metrics = m;
        log.debug("DecisionTree fit: acc={}, depth={}, leaves={}",
                m.getAccuracy(), result.getTreeDepth(), result.getLeafCount());
        return result;
    }

    private void validateTrain(IMatrix feature, String[] labels) {
        if (feature == null || labels == null) {
            throw new IllegalArgumentException("特征矩阵和标签不能为 null");
        }
        if (feature.getRowNum() != labels.length) {
            throw new IllegalArgumentException("行数与标签长度不一致");
        }
        if (feature.getRowNum() == 0) {
            throw new IllegalArgumentException("训练数据不能为空");
        }
        if (feature.getColNum() == 0) {
            throw new IllegalArgumentException("特征维度不能为 0");
        }
        for (int i = 0; i < feature.getRowNum(); i++) {
            for (int j = 0; j < feature.getColNum(); j++) {
                double v = feature.get(i, j).doubleValue();
                if (Double.isNaN(v) || Double.isInfinite(v)) {
                    throw new IllegalArgumentException("无效特征值: (" + i + "," + j + ")");
                }
            }
        }
    }

    private void buildLabelMapping(String[] labels) {
        TreeSet<String> sorted = new TreeSet<>(Arrays.asList(labels));
        labelToIndex.clear();
        indexToLabel.clear();
        int id = 0;
        for (String s : sorted) {
            labelToIndex.put(s, id);
            indexToLabel.put(id, s);
            id++;
        }
        numClasses = sorted.size();
        if (numClasses < 2) {
            throw new IllegalArgumentException("至少需要 2 个类别");
        }
    }

    private DecisionTreeNode buildTree(IMatrix X, int[] y, int[] idx, int depth) {
        int numSamples = idx.length;
        Map<Integer, Integer> counts = classCounts(y, idx);
        int majority = majorityClass(counts);

        if (shouldStop(depth, numSamples, counts)) {
            return DecisionTreeNode.leaf(indexToLabel.get(majority), depth, numSamples, counts);
        }

        SplitPick pick = findBestSplit(X, y, idx);
        if (pick == null || pick.score <= 0) {
            return DecisionTreeNode.leaf(indexToLabel.get(majority), depth, numSamples, counts);
        }

        featureImportance[pick.feature] += pick.rawGain * numSamples;

        List<Integer> left = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        for (int i : idx) {
            if (X.get(i, pick.feature).doubleValue() <= pick.threshold) {
                left.add(i);
            } else {
                right.add(i);
            }
        }
        if (left.size() < minSamplesLeaf || right.size() < minSamplesLeaf) {
            return DecisionTreeNode.leaf(indexToLabel.get(majority), depth, numSamples, counts);
        }

        int[] li = left.stream().mapToInt(Integer::intValue).toArray();
        int[] ri = right.stream().mapToInt(Integer::intValue).toArray();
        DecisionTreeNode L = buildTree(X, y, li, depth + 1);
        DecisionTreeNode R = buildTree(X, y, ri, depth + 1);
        return DecisionTreeNode.internal(pick.feature, pick.threshold, L, R, depth, numSamples, pick.rawGain, counts);
    }

    private boolean shouldStop(int depth, int numSamples, Map<Integer, Integer> counts) {
        return depth >= maxDepth
                || numSamples < minSamplesSplit
                || counts.size() <= 1;
    }

    private Map<Integer, Integer> classCounts(int[] y, int[] idx) {
        Map<Integer, Integer> m = new HashMap<>();
        for (int i : idx) {
            m.merge(y[i], 1, Integer::sum);
        }
        return m;
    }

    private int majorityClass(Map<Integer, Integer> counts) {
        return counts.entrySet().stream().max(Map.Entry.<Integer, Integer>comparingByValue()
                .thenComparing(Map.Entry.comparingByKey())).get().getKey();
    }

    private SplitPick findBestSplit(IMatrix X, int[] y, int[] idx) {
        int n = idx.length;
        int p = X.getColNum();

        double parentGini = giniImpurity(y, idx);
        double parentEntropy = entropyImpurity(y, idx);

        SplitPick best = null;
        for (int j = 0; j < p; j++) {
            List<Double> thr = candidateThresholds(X, idx, j);
            for (double t : thr) {
                List<Integer> left = new ArrayList<>();
                List<Integer> right = new ArrayList<>();
                for (int row : idx) {
                    if (X.get(row, j).doubleValue() <= t) {
                        left.add(row);
                    } else {
                        right.add(row);
                    }
                }
                if (left.size() < minSamplesLeaf || right.size() < minSamplesLeaf) {
                    continue;
                }

                double score;
                double rawGain;
                double nL = left.size();
                double nR = right.size();

                if (criterion == DecisionTreeCriterion.CART_GINI) {
                    double gL = giniImpurity(y, listToArr(left));
                    double gR = giniImpurity(y, listToArr(right));
                    rawGain = parentGini - (nL / n) * gL - (nR / n) * gR;
                    score = rawGain;
                } else if (criterion == DecisionTreeCriterion.CART_ENTROPY) {
                    double eL = entropyImpurity(y, listToArr(left));
                    double eR = entropyImpurity(y, listToArr(right));
                    rawGain = parentEntropy - (nL / n) * eL - (nR / n) * eR;
                    score = rawGain;
                } else {
                    // C4.5 gain ratio on entropy-based info gain
                    double eL = entropyImpurity(y, listToArr(left));
                    double eR = entropyImpurity(y, listToArr(right));
                    double infoGain = parentEntropy - (nL / n) * eL - (nR / n) * eR;
                    rawGain = infoGain;
                    double pl = nL / n;
                    double pr = nR / n;
                    double splitInfo = -safeLog2(pl) * pl - safeLog2(pr) * pr;
                    if (splitInfo > 1e-12) {
                        score = infoGain / splitInfo;
                    } else {
                        score = infoGain;
                    }
                }

                if (best == null || better(score, j, t, best.score, best.feature, best.threshold)) {
                    best = new SplitPick(j, t, score, rawGain);
                }
            }
        }
        return best;
    }

    private static boolean better(double sNew, int fNew, double tNew,
                                  double sBest, int fBest, double tBest) {
        if (sNew > sBest + 1e-15) {
            return true;
        }
        if (Math.abs(sNew - sBest) <= 1e-15) {
            if (fNew != fBest) {
                return fNew < fBest;
            }
            return Double.compare(tNew, tBest) < 0;
        }
        return false;
    }

    private static int[] listToArr(List<Integer> list) {
        return list.stream().mapToInt(Integer::intValue).toArray();
    }

    private static List<Double> candidateThresholds(IMatrix X, int[] idx, int j) {
        double[] vals = new double[idx.length];
        for (int i = 0; i < idx.length; i++) {
            vals[i] = X.get(idx[i], j).doubleValue();
        }
        Arrays.sort(vals);
        List<Double> out = new ArrayList<>();
        for (int i = 0; i < vals.length - 1; i++) {
            if (Double.compare(vals[i], vals[i + 1]) != 0) {
                out.add((vals[i] + vals[i + 1]) / 2.0);
            }
        }
        return out;
    }

    private static double giniImpurity(int[] y, int[] idx) {
        Map<Integer, Integer> c = new HashMap<>();
        for (int i : idx) {
            c.merge(y[i], 1, Integer::sum);
        }
        int total = idx.length;
        double sumSq = 0;
        for (int v : c.values()) {
            double p = (double) v / total;
            sumSq += p * p;
        }
        return 1.0 - sumSq;
    }

    private static double entropyImpurity(int[] y, int[] idx) {
        Map<Integer, Integer> c = new HashMap<>();
        for (int i : idx) {
            c.merge(y[i], 1, Integer::sum);
        }
        int total = idx.length;
        double e = 0;
        for (int v : c.values()) {
            if (v > 0) {
                double p = (double) v / total;
                e -= p * (Math.log(p) / LN2);
            }
        }
        return e;
    }

    private static double safeLog2(double p) {
        if (p <= 0) {
            return 0;
        }
        return Math.log(p) / LN2;
    }

    private static final double LN2 = Math.log(2.0);

    private static final class SplitPick {
        final int feature;
        final double threshold;
        final double score;
        final double rawGain;

        SplitPick(int feature, double threshold, double score, double rawGain) {
            this.feature = feature;
            this.threshold = threshold;
            this.score = score;
            this.rawGain = rawGain;
        }
    }

    @Override
    public String predict(IVector x) {
        if (!trained || root == null) {
            throw new IllegalStateException("模型尚未训练");
        }
        if (x.length() != numFeatures) {
            throw new IllegalArgumentException("特征维度不匹配");
        }
        return root.predict(x);
    }

    @Override
    public Map<String, Double> predictProb(IVector x) {
        if (!trained || root == null) {
            throw new IllegalStateException("模型尚未训练");
        }
        return root.predictProb(x, indexToLabel, numClasses);
    }

    @Override
    public String[] predictBatch(IMatrix features) {
        int m = features.getRowNum();
        String[] out = new String[m];
        for (int i = 0; i < m; i++) {
            out[i] = predict(features.getRow(i));
        }
        return out;
    }

    @Override
    public BatchPredictionResult predictBatchWithProbs(IMatrix features) {
        int m = features.getRowNum();
        String[] labels = new String[m];
        int kOut = numClasses;
        double[][] probs = new double[m][kOut];
        for (int i = 0; i < m; i++) {
            IVector row = features.getRow(i);
            Map<String, Double> pm = predictProb(row);
            labels[i] = predict(row);
            for (int c = 0; c < numClasses; c++) {
                probs[i][c] = pm.getOrDefault(indexToLabel.get(c), 0.0);
            }
        }
        return new BatchPredictionResult(labels, probs);
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public ClassificationMetrics getMetrics() {
        return metrics;
    }

    @Override
    public void setMetrics(ClassificationMetrics metrics) {
        this.metrics = metrics;
    }

    @Override
    public void save(String path) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        } catch (IOException e) {
            log.error("save decision tree failed", e);
            throw new IllegalStateException(e);
        }
    }

    /** 节点（单文件内聚）。 */
    static final class DecisionTreeNode implements Serializable {
        private static final long serialVersionUID = 1L;

        boolean leaf;
        String leafLabel;
        Map<Integer, Integer> classCounts;
        int feature;
        double threshold;
        DecisionTreeNode left;
        DecisionTreeNode right;
        int depth;
        int sampleCount;

        static DecisionTreeNode leaf(String label, int depth, int n, Map<Integer, Integer> counts) {
            DecisionTreeNode nd = new DecisionTreeNode();
            nd.leaf = true;
            nd.leafLabel = label;
            nd.classCounts = new HashMap<>(counts);
            nd.depth = depth;
            nd.sampleCount = n;
            return nd;
        }

        static DecisionTreeNode internal(int f, double t, DecisionTreeNode L, DecisionTreeNode R,
                                         int depth, int n, double gain, Map<Integer, Integer> counts) {
            DecisionTreeNode nd = new DecisionTreeNode();
            nd.leaf = false;
            nd.feature = f;
            nd.threshold = t;
            nd.left = L;
            nd.right = R;
            nd.depth = depth;
            nd.sampleCount = n;
            nd.classCounts = new HashMap<>(counts);
            return nd;
        }

        String predict(IVector x) {
            if (leaf) {
                return leafLabel;
            }
            if (x.get(feature).doubleValue() <= threshold) {
                return left.predict(x);
            }
            return right.predict(x);
        }

        Map<String, Double> predictProb(IVector x, Map<Integer, String> idxToLab, int numClasses) {
            DecisionTreeNode cur = this;
            while (!cur.leaf) {
                if (x.get(cur.feature).doubleValue() <= cur.threshold) {
                    cur = cur.left;
                } else {
                    cur = cur.right;
                }
            }
            Map<String, Double> out = new LinkedHashMap<>();
            int sum = cur.classCounts.values().stream().mapToInt(Integer::intValue).sum();
            if (sum <= 0) {
                for (int c = 0; c < numClasses; c++) {
                    out.put(idxToLab.get(c), 1.0 / numClasses);
                }
                return out;
            }
            for (int c = 0; c < numClasses; c++) {
                int cnt = cur.classCounts.getOrDefault(c, 0);
                out.put(idxToLab.get(c), (double) cnt / sum);
            }
            return out;
        }

        int depthBelow() {
            if (leaf) {
                return depth;
            }
            return Math.max(left.depthBelow(), right.depthBelow());
        }

        int leafCount() {
            if (leaf) {
                return 1;
            }
            return left.leafCount() + right.leafCount();
        }
    }
}
