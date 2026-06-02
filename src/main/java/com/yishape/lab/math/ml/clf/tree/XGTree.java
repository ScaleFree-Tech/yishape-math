package com.yishape.lab.math.ml.clf.tree;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 决策树类（用于 XGBoost 风格梯度提升）。
 * <p>
 * 支持精确贪心分裂（{@link XGBoostTreeMethod#EXACT}）与直方图近似分裂（{@link XGBoostTreeMethod#HIST}）。
 * 直方图分箱可为全局 min/max {@linkplain XGBoostHistogramBinning#UNIFORM 均匀}，
 * 或 {@linkplain XGBoostHistogramBinning#QUANTILE_WEIGHTED_SKETCH Hessian 加权分位数}（对齐常用 weighted quantile 候选）。
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class XGTree {

    /** 根节点 */
    private XGTreeNode root;

    /** 最大深度 */
    private int maxDepth;

    /** 最小分裂样本数 */
    private int minSamplesSplit;

    /** 最小叶子样本数（样本个数） */
    private int minSamplesLeaf;

    /** L1 正则化参数（作用于叶子权重推导中的近似项） */
    private double alpha;

    /** L2 正则化参数 λ */
    private double lambda;

    /** 学习率（收缩系数） */
    private double learningRate;

    /** 建树策略 */
    private XGBoostTreeMethod treeMethod = XGBoostTreeMethod.EXACT;

    /** 直方图最大箱数（仅 HIST） */
    private int maxBin = 256;

    /**
     * 最小分裂增益 γ：best_gain &gt; γ 才分裂（对齐 XGBoost min_split_loss 语义；0 表示仅禁止非正增益）。
     */
    private double gamma = 0.0;

    /**
     * 子结点最小 Hessian 和（对齐 XGBoost min_child_weight；0 表示不启用）。
     */
    private double minChildWeight = 0.0;

    /** 训练矩阵列数（全局分箱） */
    private int globalNumCols;

    /** 各列特征全局 [min,max]，长度 = globalNumCols */
    private double[] featMin;

    private double[] featMax;

    /** HIST + {@link XGBoostHistogramBinning#QUANTILE_WEIGHTED_SKETCH} 时非 null，形状 {@code [col][maxBin+1]} */
    private double[][] histQuantileEdges;

    /** 直方图分箱策略（仅 HIST） */
    private XGBoostHistogramBinning histogramBinning = XGBoostHistogramBinning.UNIFORM;

    /** 直方图临时缓冲 */
    private double[] histGradBuf;

    private double[] histHessBuf;

    private int[] histCountBuf;

    /**
     * 经典构造函数：精确贪心，γ=0，min_child_weight=0。
     */
    public XGTree(int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                  double alpha, double lambda, double learningRate) {
        this(maxDepth, minSamplesSplit, minSamplesLeaf, alpha, lambda, learningRate,
                XGBoostTreeMethod.EXACT, 256, 0.0, 0.0, XGBoostHistogramBinning.UNIFORM);
    }

    /**
     * 完整构造函数（工业参数）。
     *
     * @param maxBin       直方图箱数，至少 2
     * @param gamma        最小分裂增益
     * @param minChildWeight 子结点最小 Σh（hessian）
     */
    public XGTree(int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                  double alpha, double lambda, double learningRate,
                  XGBoostTreeMethod treeMethod, int maxBin, double gamma, double minChildWeight) {
        this(maxDepth, minSamplesSplit, minSamplesLeaf, alpha, lambda, learningRate,
                treeMethod, maxBin, gamma, minChildWeight, XGBoostHistogramBinning.UNIFORM);
    }

    /**
     * 完整构造函数（含直方图分箱策略）。
     */
    public XGTree(int maxDepth, int minSamplesSplit, int minSamplesLeaf,
                  double alpha, double lambda, double learningRate,
                  XGBoostTreeMethod treeMethod, int maxBin, double gamma, double minChildWeight,
                  XGBoostHistogramBinning histogramBinning) {
        this.maxDepth = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
        this.minSamplesLeaf = minSamplesLeaf;
        this.alpha = alpha;
        this.lambda = lambda;
        this.learningRate = learningRate;
        this.treeMethod = treeMethod != null ? treeMethod : XGBoostTreeMethod.EXACT;
        this.maxBin = Math.max(2, maxBin);
        this.gamma = gamma;
        this.minChildWeight = Math.max(0.0, minChildWeight);
        this.histogramBinning = histogramBinning != null ? histogramBinning : XGBoostHistogramBinning.UNIFORM;
    }

    public XGBoostTreeMethod getTreeMethod() {
        return treeMethod;
    }

    public int getMaxBin() {
        return maxBin;
    }

    public double getGamma() {
        return gamma;
    }

    public double getMinChildWeight() {
        return minChildWeight;
    }

    public XGBoostHistogramBinning getHistogramBinning() {
        return histogramBinning;
    }

    /**
     * 在全训练行上建树（子采样请在调用方传入 {@code sampleIndices}）。
     */
    public void fit(IMatrix features, IVector gradients, IVector hessians) {
        int numSamples = features.rows();
        int[] sampleIndices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            sampleIndices[i] = i;
        }
        fit(features, gradients, hessians, sampleIndices, null);
    }

    /**
     * @param sampleIndices 参与本棵树训练的样本行索引（全局行号）
     * @param featureSubset 参与分裂的特征列索引；{@code null} 表示全部特征
     */
    public void fit(IMatrix features, IVector gradients, IVector hessians,
                    int[] sampleIndices, int[] featureSubset) {
        globalNumCols = features.cols();
        computeGlobalBinning(features);
        if (treeMethod == XGBoostTreeMethod.HIST
                && histogramBinning == XGBoostHistogramBinning.QUANTILE_WEIGHTED_SKETCH) {
            histQuantileEdges = XgbWeightedQuantileSketch.buildEdges(features, hessians, sampleIndices, maxBin);
        } else {
            histQuantileEdges = null;
        }
        ensureHistBuffers();
        this.root = buildTree(features, gradients, hessians, sampleIndices, featureSubset, 0);
    }

    private void computeGlobalBinning(IMatrix features) {
        featMin = new double[globalNumCols];
        featMax = new double[globalNumCols];
        int n = features.rows();
        Arrays.fill(featMin, Double.POSITIVE_INFINITY);
        Arrays.fill(featMax, Double.NEGATIVE_INFINITY);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < globalNumCols; j++) {
                double v = features.get(i, j);
                if (v < featMin[j]) {
                    featMin[j] = v;
                }
                if (v > featMax[j]) {
                    featMax[j] = v;
                }
            }
        }
    }

    private void ensureHistBuffers() {
        if (treeMethod != XGBoostTreeMethod.HIST) {
            return;
        }
        if (histGradBuf == null || histGradBuf.length < maxBin) {
            histGradBuf = new double[maxBin];
            histHessBuf = new double[maxBin];
            histCountBuf = new int[maxBin];
        }
    }

    private XGTreeNode buildTree(IMatrix features, IVector gradients, IVector hessians,
                                 int[] sampleIndices, int[] featureSubset, int depth) {

        int numSamples = sampleIndices.length;

        double gradSum = 0.0;
        double hessSum = 0.0;
        for (int idx : sampleIndices) {
            gradSum += gradients.get(idx);
            hessSum += hessians.get(idx);
        }

        double leafValue = computeLeafWeight(gradSum, hessSum);

        if (shouldStop(depth, numSamples, hessSum)) {
            return new XGTreeNode(leafValue, depth, numSamples);
        }

        SplitResult bestSplit = switch (treeMethod) {
            case EXACT -> findBestSplitExact(features, gradients, hessians, sampleIndices, featureSubset);
            case HIST -> findBestSplitHist(features, gradients, hessians, sampleIndices, featureSubset);
        };

        if (bestSplit == null || bestSplit.gain <= gamma) {
            return new XGTreeNode(leafValue, depth, numSamples);
        }

        List<Integer> leftIndices = new ArrayList<>();
        List<Integer> rightIndices = new ArrayList<>();

        for (int idx : sampleIndices) {
            if (features.get(idx, bestSplit.featureIndex) <= bestSplit.threshold) {
                leftIndices.add(idx);
            } else {
                rightIndices.add(idx);
            }
        }

        if (leftIndices.size() < minSamplesLeaf || rightIndices.size() < minSamplesLeaf) {
            return new XGTreeNode(leafValue, depth, numSamples);
        }

        XGTreeNode leftChild = buildTree(features, gradients, hessians,
                leftIndices.stream().mapToInt(Integer::intValue).toArray(), featureSubset, depth + 1);
        XGTreeNode rightChild = buildTree(features, gradients, hessians,
                rightIndices.stream().mapToInt(Integer::intValue).toArray(), featureSubset, depth + 1);

        return new XGTreeNode(bestSplit.featureIndex, bestSplit.threshold,
                leftChild, rightChild, depth, numSamples, bestSplit.gain);
    }

    private double computeLeafWeight(double gradSum, double hessSum) {
        if (hessSum + lambda == 0) {
            return 0.0;
        }

        double weight = -gradSum / (hessSum + lambda);

        if (alpha > 0) {
            if (weight > alpha) {
                weight -= alpha;
            } else if (weight < -alpha) {
                weight += alpha;
            } else {
                weight = 0.0;
            }
        }

        return weight * learningRate;
    }

    private boolean shouldStop(int depth, int numSamples, double hessSum) {
        return depth >= maxDepth
                || numSamples < minSamplesSplit
                || hessSum < 1e-8;
    }

    private int[] effectiveFeatureList(int[] featureSubset, int numFeatures) {
        if (featureSubset != null) {
            return featureSubset;
        }
        int[] all = new int[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            all[i] = i;
        }
        return all;
    }

    private SplitResult findBestSplitExact(IMatrix features, IVector gradients, IVector hessians,
                                           int[] sampleIndices, int[] featureSubset) {
        int numFeatures = features.cols();
        int[] feats = effectiveFeatureList(featureSubset, numFeatures);

        SplitResult bestSplit = null;
        double bestGain = Double.NEGATIVE_INFINITY;

        double totalGradSum = 0.0;
        double totalHessSum = 0.0;
        for (int idx : sampleIndices) {
            totalGradSum += gradients.get(idx);
            totalHessSum += hessians.get(idx);
        }

        int n = sampleIndices.length;

        for (int fi : feats) {
            final int fcol = fi;
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            Arrays.sort(order, Comparator.comparingDouble(
                    i -> features.get(sampleIndices[i], fcol)));

            double leftGradSum = 0.0;
            double leftHessSum = 0.0;

            for (int i = 0; i < n - 1; i++) {
                int row = sampleIndices[order[i]];
                leftGradSum += gradients.get(row);
                leftHessSum += hessians.get(row);

                double v0 = features.get(row, fi);
                double v1 = features.get(sampleIndices[order[i + 1]], fi);
                if (Double.compare(v0, v1) == 0) {
                    continue;
                }

                int leftCount = i + 1;
                int rightCount = n - leftCount;
                if (leftCount < minSamplesLeaf || rightCount < minSamplesLeaf) {
                    continue;
                }

                double rightHessSum = totalHessSum - leftHessSum;
                if (leftHessSum < minChildWeight || rightHessSum < minChildWeight) {
                    continue;
                }

                double gain = gainAtSplit(totalGradSum, totalHessSum, leftGradSum, leftHessSum);
                if (gain > bestGain) {
                    bestGain = gain;
                    double threshold = (v0 + v1) / 2.0;
                    bestSplit = new SplitResult(fi, threshold, gain);
                }
            }
        }

        return bestSplit;
    }

    private SplitResult findBestSplitHist(IMatrix features, IVector gradients, IVector hessians,
                                          int[] sampleIndices, int[] featureSubset) {
        int numFeatures = features.cols();
        int[] feats = effectiveFeatureList(featureSubset, numFeatures);

        SplitResult bestSplit = null;
        double bestGain = Double.NEGATIVE_INFINITY;

        double totalGradSum = 0.0;
        double totalHessSum = 0.0;
        for (int idx : sampleIndices) {
            totalGradSum += gradients.get(idx);
            totalHessSum += hessians.get(idx);
        }

        int n = sampleIndices.length;

        boolean useQuantile = histQuantileEdges != null;

        for (int fi : feats) {
            double mn = useQuantile ? histQuantileEdges[fi][0] : featMin[fi];
            double mx = useQuantile ? histQuantileEdges[fi][maxBin] : featMax[fi];
            if (!(mx > mn + 1e-12)) {
                continue;
            }

            Arrays.fill(histGradBuf, 0, maxBin, 0.0);
            Arrays.fill(histHessBuf, 0, maxBin, 0.0);
            Arrays.fill(histCountBuf, 0, maxBin, 0);

            double width = useQuantile ? Double.NaN : (mx - mn) / maxBin;

            for (int idx : sampleIndices) {
                double v = features.get(idx, fi);
                int b = useQuantile ? binForQuantile(v, histQuantileEdges[fi]) : binUniform(v, mn, width);
                histGradBuf[b] += gradients.get(idx);
                histHessBuf[b] += hessians.get(idx);
                histCountBuf[b]++;
            }

            double lg = 0.0;
            double lh = 0.0;
            int lc = 0;

            for (int b = 0; b < maxBin - 1; b++) {
                lg += histGradBuf[b];
                lh += histHessBuf[b];
                lc += histCountBuf[b];

                int rc = n - lc;
                if (lc < minSamplesLeaf || rc < minSamplesLeaf) {
                    continue;
                }

                double rh = totalHessSum - lh;
                if (lh < minChildWeight || rh < minChildWeight) {
                    continue;
                }

                double gain = gainAtSplit(totalGradSum, totalHessSum, lg, lh);
                if (gain > bestGain) {
                    bestGain = gain;
                    double boundary = useQuantile ? histQuantileEdges[fi][b + 1] : mn + (b + 1) * width;
                    bestSplit = new SplitResult(fi, boundary, gain);
                }
            }
        }

        return bestSplit;
    }

    private int binUniform(double v, double mn, double width) {
        int b = (int) Math.floor((v - mn) / width);
        if (b < 0) {
            return 0;
        }
        if (b >= maxBin) {
            return maxBin - 1;
        }
        return b;
    }

    private static int binForQuantile(double v, double[] edges) {
        int bmax = edges.length - 1;
        int b = 0;
        while (b < bmax - 1 && v > edges[b + 1]) {
            b++;
        }
        return b;
    }

    private double gainAtSplit(double totalGradSum, double totalHessSum,
                               double leftGradSum, double leftHessSum) {
        double rightGradSum = totalGradSum - leftGradSum;
        double rightHessSum = totalHessSum - leftHessSum;

        double denomL = leftHessSum + lambda;
        double denomR = rightHessSum + lambda;
        double denomP = totalHessSum + lambda;
        if (denomL <= 0 || denomR <= 0 || denomP <= 0) {
            return 0.0;
        }

        double leftScore = (leftGradSum * leftGradSum) / denomL;
        double rightScore = (rightGradSum * rightGradSum) / denomR;
        double parentScore = (totalGradSum * totalGradSum) / denomP;

        // 与工业 XGBoost 常用近似一致：分裂打分不含 leaf-L1(alpha)；alpha 仅作用于叶子权重闭合式
        return 0.5 * (leftScore + rightScore - parentScore);
    }

    public double predict(IVector features) {
        if (root == null) {
            return 0.0;
        }

        double[] featureArray = new double[features.size()];
        for (int i = 0; i < features.size(); i++) {
            featureArray[i] = features.get(i);
        }

        return root.predict(featureArray);
    }

    public double[] predict(IMatrix features) {
        int numSamples = features.rows();
        double[] predictions = new double[numSamples];

        for (int i = 0; i < numSamples; i++) {
            predictions[i] = predict(features.getRow(i));
        }

        return predictions;
    }

    public double[] computeFeatureImportance(int numFeatures) {
        double[] importance = new double[numFeatures];
        if (root != null) {
            root.computeFeatureImportance(importance);
        }
        return importance;
    }

    public int getDepth() {
        return root != null ? root.getTreeDepth() : 0;
    }

    public int getLeafCount() {
        return root != null ? root.getLeafCount() : 0;
    }

    public XGTreeNode getRoot() {
        return root;
    }

    public int getMaxDepth() {
        return maxDepth;
    }

    public int getMinSamplesSplit() {
        return minSamplesSplit;
    }

    public int getMinSamplesLeaf() {
        return minSamplesLeaf;
    }

    public double getAlpha() {
        return alpha;
    }

    public double getLambda() {
        return lambda;
    }

    public double getLearningRate() {
        return learningRate;
    }

    // ==================== JSON persistence ====================

    Map<String, Object> toParams() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("maxDepth", maxDepth);
        m.put("minSamplesSplit", minSamplesSplit);
        m.put("minSamplesLeaf", minSamplesLeaf);
        m.put("alpha", alpha);
        m.put("lambda", lambda);
        m.put("learningRate", learningRate);
        m.put("treeMethod", treeMethod.name());
        m.put("maxBin", maxBin);
        m.put("gamma", gamma);
        m.put("minChildWeight", minChildWeight);
        m.put("histogramBinning", histogramBinning.name());
        if (root != null) m.put("root", root.toParams());
        return m;
    }

    @SuppressWarnings("unchecked")
    static XGTree fromParams(Map<String, Object> m) {
        int maxDepth = ((Number) m.get("maxDepth")).intValue();
        int minSamplesSplit = ((Number) m.get("minSamplesSplit")).intValue();
        int minSamplesLeaf = ((Number) m.get("minSamplesLeaf")).intValue();
        double alpha = ((Number) m.get("alpha")).doubleValue();
        double lambda = ((Number) m.get("lambda")).doubleValue();
        double learningRate = ((Number) m.get("learningRate")).doubleValue();
        XGBoostTreeMethod treeMethod = XGBoostTreeMethod.valueOf((String) m.get("treeMethod"));
        int maxBin = ((Number) m.get("maxBin")).intValue();
        double gamma = ((Number) m.get("gamma")).doubleValue();
        double minChildWeight = ((Number) m.get("minChildWeight")).doubleValue();
        XGBoostHistogramBinning histogramBinning = XGBoostHistogramBinning.valueOf((String) m.get("histogramBinning"));
        XGTree tree = new XGTree(maxDepth, minSamplesSplit, minSamplesLeaf, alpha, lambda, learningRate,
                treeMethod, maxBin, gamma, minChildWeight, histogramBinning);
        Map<String, Object> rootMap = (Map<String, Object>) m.get("root");
        if (rootMap != null) tree.root = XGTreeNode.fromParams(rootMap);
        return tree;
    }

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
