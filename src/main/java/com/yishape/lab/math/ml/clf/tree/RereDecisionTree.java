package com.yishape.lab.math.ml.clf.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.clf.BatchPredResult;
import com.yishape.lab.math.ml.clf.ClfResult;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.util.YishapeLogger;

import java.util.*;

/**
 * 单棵决策树分类器：连续特征上的二元分裂；支持 CART（基尼/熵）与 C4.5 <em>风格</em>增益率。
 * 若需 Weka {@code J48} 级别的完整 C4.5（名目属性、缺失值、剪枝策略等），请在外部扩展或使用专用库。
 * <p>
 * 标签索引按字典序稳定分配（与 {@link com.yishape.lab.math.ml.clf.tree.RereXGboost} 一致）。
 * </p>
 */
public class RereDecisionTree implements IClassifier {

    private static final YishapeLogger log = YishapeLogger.getLogger(RereDecisionTree.class);

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
    private DecisionTreeResult result;

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
    public IClassifier fit(IMatrix feature, String[] labels) {
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

        double[][] X = extractTrainingMatrix(feature);
        root = buildTree(X, y, allIdx, 0);

        trained = true;

        ClassificationMetrics m = ClassificationMetrics.compute(this, feature, labels);
        this.metrics = m;

        this.result = new DecisionTreeResult();
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
        result.setTrainAccuracy(m.getAccuracy());

        log.debug("DecisionTree fit: acc={}, depth={}, leaves={}",
                m.getAccuracy(), root != null ? root.depthBelow() : 0, root != null ? root.leafCount() : 0);
        return this;
    }

    @Override
    public String[] fitPredict(IMatrix feature, String[] labels) {
        fit(feature, labels);
        return predictBatch(feature);
    }

    @Override
    public DecisionTreeResult getResult() {
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
                double v = feature.get(i, j);
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

    private static double[][] extractTrainingMatrix(IMatrix feature) {
        int r = feature.getRowNum();
        int c = feature.getColNum();
        double[][] x = new double[r][c];
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                x[i][j] = feature.get(i, j);
            }
        }
        return x;
    }

    private DecisionTreeNode buildTree(double[][] X, int[] y, int[] idx, int depth) {
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
            if (X[i][pick.feature] <= pick.threshold) {
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

    private SplitPick findBestSplit(double[][] X, int[] y, int[] idx) {
        int n = idx.length;
        int p = X[0].length;

        double parentGini = giniImpurity(y, idx);
        double parentEntropy = entropyImpurity(y, idx);

        int[] totalByClass = new int[numClasses];
        for (int row : idx) {
            totalByClass[y[row]]++;
        }

        int[] leftCount = new int[numClasses];
        int[] rightCount = new int[numClasses];

        SplitPick best = null;
        for (int j = 0; j < p; j++) {
            Integer[] ord = new Integer[n];
            for (int t = 0; t < n; t++) {
                ord[t] = idx[t];
            }
            final int fj = j;
            Arrays.sort(ord, (a, b) -> Double.compare(X[a][fj], X[b][fj]));

            Arrays.fill(leftCount, 0);
            int leftSize = 0;

            for (int i = 0; i < n - 1; i++) {
                int row = ord[i];
                leftCount[y[row]]++;
                leftSize++;
                int rightSize = n - leftSize;

                if (Double.compare(X[ord[i]][fj], X[ord[i + 1]][fj]) == 0) {
                    continue;
                }
                if (leftSize < minSamplesLeaf || rightSize < minSamplesLeaf) {
                    continue;
                }

                double threshold = (X[ord[i]][fj] + X[ord[i + 1]][fj]) / 2.0;
                double score;
                double rawGain;

                if (criterion == DecisionTreeCriterion.CART_GINI) {
                    double gL = giniFromClassCounts(leftCount, leftSize);
                    for (int c = 0; c < numClasses; c++) {
                        rightCount[c] = totalByClass[c] - leftCount[c];
                    }
                    double gR = giniFromClassCounts(rightCount, rightSize);
                    rawGain = parentGini - (leftSize / (double) n) * gL - (rightSize / (double) n) * gR;
                    score = rawGain;
                } else if (criterion == DecisionTreeCriterion.CART_ENTROPY) {
                    double eL = entropyFromClassCounts(leftCount, leftSize);
                    for (int c = 0; c < numClasses; c++) {
                        rightCount[c] = totalByClass[c] - leftCount[c];
                    }
                    double eR = entropyFromClassCounts(rightCount, rightSize);
                    rawGain = parentEntropy - (leftSize / (double) n) * eL - (rightSize / (double) n) * eR;
                    score = rawGain;
                } else {
                    double eL = entropyFromClassCounts(leftCount, leftSize);
                    for (int c = 0; c < numClasses; c++) {
                        rightCount[c] = totalByClass[c] - leftCount[c];
                    }
                    double eR = entropyFromClassCounts(rightCount, rightSize);
                    double infoGain = parentEntropy - (leftSize / (double) n) * eL - (rightSize / (double) n) * eR;
                    rawGain = infoGain;
                    double pl = leftSize / (double) n;
                    double pr = rightSize / (double) n;
                    double splitInfo = -safeLog2(pl) * pl - safeLog2(pr) * pr;
                    if (splitInfo > 1e-12) {
                        score = infoGain / splitInfo;
                    } else {
                        score = infoGain;
                    }
                }

                if (best == null || better(score, j, threshold, best.score, best.feature, best.threshold)) {
                    best = new SplitPick(j, threshold, score, rawGain);
                }
            }
        }
        return best;
    }

    private static double giniFromClassCounts(int[] counts, int total) {
        double sumSq = 0.0;
        for (int c : counts) {
            if (c > 0) {
                double p = (double) c / total;
                sumSq += p * p;
            }
        }
        return 1.0 - sumSq;
    }

    private static double entropyFromClassCounts(int[] counts, int total) {
        double e = 0.0;
        for (int c : counts) {
            if (c > 0) {
                double p = (double) c / total;
                e -= p * (Math.log(p) / LN2);
            }
        }
        return e;
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
    public BatchPredResult predictBatchWithProbs(IMatrix features) {
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
        return new BatchPredResult(labels, probs);
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

    // ==================== JSON persistence ====================

    @Override
    public Map<String, Object> toParams() {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("criterion", criterion.name());
        p.put("maxDepth", maxDepth);
        p.put("minSamplesSplit", minSamplesSplit);
        p.put("minSamplesLeaf", minSamplesLeaf);
        p.put("numClasses", numClasses);
        p.put("numFeatures", numFeatures);
        p.put("trained", trained);
        p.put("labelToIndex", new HashMap<>(labelToIndex));
        p.put("indexToLabel", indexToLabelToString());
        if (featureImportance != null) p.put("featureImportance", featureImportance.clone());
        if (root != null) p.put("root", nodeToParams(root));
        return p;
    }

    private Map<String, String> indexToLabelToString() {
        Map<String, String> m = new LinkedHashMap<>();
        for (Map.Entry<Integer, String> e : indexToLabel.entrySet()) {
            m.put(String.valueOf(e.getKey()), e.getValue());
        }
        return m;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void fromParams(Map<String, Object> p) {
        this.criterion = DecisionTreeCriterion.valueOf((String) p.get("criterion"));
        this.maxDepth = ((Number) p.get("maxDepth")).intValue();
        this.minSamplesSplit = ((Number) p.get("minSamplesSplit")).intValue();
        this.minSamplesLeaf = ((Number) p.get("minSamplesLeaf")).intValue();
        this.numClasses = ((Number) p.get("numClasses")).intValue();
        this.numFeatures = ((Number) p.get("numFeatures")).intValue();
        this.trained = (Boolean) p.get("trained");
        this.labelToIndex = new HashMap<>((Map<String, Integer>) p.get("labelToIndex"));
        this.indexToLabel = indexToLabelFromString((Map<String, String>) p.get("indexToLabel"));
        this.featureImportance = (double[]) p.get("featureImportance");
        Map<String, Object> rootMap = (Map<String, Object>) p.get("root");
        if (rootMap != null) this.root = nodeFromParams(rootMap);
        if (trained) {
            this.result = new DecisionTreeResult();
            this.result.setCriterion(criterion);
            this.result.setMaxDepthParam(maxDepth == Integer.MAX_VALUE ? -1 : maxDepth);
            this.result.setTreeDepth(root != null ? root.depthBelow() : 0);
            this.result.setLeafCount(root != null ? root.leafCount() : 0);
            this.result.setTrained(true);
            this.result.setNumClasses(numClasses);
            this.result.setLabelMapping(new HashMap<>(labelToIndex));
            this.result.setReverseLabelMapping(new HashMap<>(indexToLabel));
            this.result.setNumFeatures(numFeatures);
            if (featureImportance != null) {
                this.result.setFeatureImportanceFromArray(featureImportance);
            }
        }
    }

    private Map<Integer, String> indexToLabelFromString(Map<String, String> m) {
        Map<Integer, String> result = new LinkedHashMap<>();
        for (Map.Entry<String, String> e : m.entrySet()) {
            result.put(Integer.parseInt(e.getKey()), e.getValue());
        }
        return result;
    }

    private static Map<String, Object> nodeToParams(DecisionTreeNode node) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("leaf", node.leaf);
        m.put("depth", node.depth);
        m.put("sampleCount", node.sampleCount);
        Map<String, Integer> countsStr = new LinkedHashMap<>();
        if (node.classCounts != null) {
            for (Map.Entry<Integer, Integer> e : node.classCounts.entrySet()) {
                countsStr.put(String.valueOf(e.getKey()), e.getValue());
            }
        }
        m.put("classCounts", countsStr);
        if (node.leaf) {
            m.put("leafLabel", node.leafLabel);
        } else {
            m.put("feature", node.feature);
            m.put("threshold", node.threshold);
            m.put("left", nodeToParams(node.left));
            m.put("right", nodeToParams(node.right));
        }
        return m;
    }

    @SuppressWarnings("unchecked")
    private static DecisionTreeNode nodeFromParams(Map<String, Object> m) {
        DecisionTreeNode nd = new DecisionTreeNode();
        nd.leaf = (Boolean) m.get("leaf");
        nd.depth = ((Number) m.get("depth")).intValue();
        nd.sampleCount = ((Number) m.get("sampleCount")).intValue();
        Map<String, Integer> countsStr = (Map<String, Integer>) m.get("classCounts");
        nd.classCounts = new HashMap<>();
        if (countsStr != null) {
            for (Map.Entry<String, Integer> e : countsStr.entrySet()) {
                nd.classCounts.put(Integer.parseInt(e.getKey()), e.getValue());
            }
        }
        if (nd.leaf) {
            nd.leafLabel = (String) m.get("leafLabel");
        } else {
            nd.feature = ((Number) m.get("feature")).intValue();
            nd.threshold = ((Number) m.get("threshold")).doubleValue();
            nd.left = nodeFromParams((Map<String, Object>) m.get("left"));
            nd.right = nodeFromParams((Map<String, Object>) m.get("right"));
        }
        return nd;
    }

    /** 节点（单文件内聚）。 */
    static final class DecisionTreeNode {

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
            if (x.get(feature) <= threshold) {
                return left.predict(x);
            }
            return right.predict(x);
        }

        Map<String, Double> predictProb(IVector x, Map<Integer, String> idxToLab, int numClasses) {
            DecisionTreeNode cur = this;
            while (!cur.leaf) {
                if (x.get(cur.feature) <= cur.threshold) {
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
