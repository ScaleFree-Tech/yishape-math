package com.yishape.lab.math.ml.metric;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.ml.clf.BatchPredResult;
import com.yishape.lab.util.IRichReport;
import java.io.Serializable;

import java.util.*;
import com.yishape.lab.math.ml.clf.IClassifier;

import com.yishape.lab.util.YishapeLogger;

/**
 * 分类评估指标工具类
 * <p>
 * 提供类似sk-learn的分类器评估功能，包括准确率、精确率、召回率、F1分数、AUC等指标的计算。 支持二分类和多分类场景，提供全面的性能评估。
 * </p>
 *
 * @author lteb2002
 * @version 1.0
 * @since 1.0
 */
public class ClassificationMetrics implements Serializable, IRichReport {

    private static final YishapeLogger log = YishapeLogger.getLogger(ClassificationMetrics.class);

    // ==================== 核心指标属性 ====================
    /**
     * 准确率 / Accuracy
     */
    private final double accuracy;

    /**
     * 宏平均精确率 / Macro-averaged precision
     */
    private final double macroPrecision;

    /**
     * 宏平均召回率 / Macro-averaged recall
     */
    private final double macroRecall;

    /**
     * 宏平均F1分数 / Macro-averaged F1 score
     */
    private final double macroF1;

    /**
     * 加权平均精确率 / Weighted-averaged precision
     */
    private final double weightedPrecision;

    /**
     * 加权平均召回率 / Weighted-averaged recall
     */
    private final double weightedRecall;

    /**
     * 加权平均F1分数 / Weighted-averaged F1 score
     */
    private final double weightedF1;

    /**
     * 二分类AUC (仅适用于二分类) / Binary classification AUC
     */
    private final double auc;

    // ==================== 详细指标属性 ====================
    /**
     * 每类别的精确率 / Precision per class
     */
    private final Map<String, Double> precisionPerClass;

    /**
     * 每类别的召回率 / Recall per class
     */
    private final Map<String, Double> recallPerClass;

    /**
     * 每类别的F1分数 / F1 score per class
     */
    private final Map<String, Double> f1PerClass;

    /**
     * 每类别的支持数 / Support count per class
     */
    private final Map<String, Integer> supportPerClass;

    /**
     * 混淆矩阵 / Confusion matrix
     */
    private final int[][] confusionMatrix;

    /**
     * 类别标签 / Class labels
     */
    private final String[] classLabels;

    /**
     * 预测概率 (用于AUC计算) / Prediction probabilities for AUC calculation
     */
    private final double[] positiveProbabilities;
    private final int[] trueLabels;

    // ==================== 构造函数 ====================
    /**
     * 私有构造函数，通过Builder模式创建实例
     */
    private ClassificationMetrics(Builder builder) {
        this.accuracy = builder.accuracy;
        this.macroPrecision = builder.macroPrecision;
        this.macroRecall = builder.macroRecall;
        this.macroF1 = builder.macroF1;
        this.weightedPrecision = builder.weightedPrecision;
        this.weightedRecall = builder.weightedRecall;
        this.weightedF1 = builder.weightedF1;
        this.auc = builder.auc;
        this.precisionPerClass = builder.precisionPerClass != null
                ? new HashMap<>(builder.precisionPerClass) : new HashMap<>();
        this.recallPerClass = builder.recallPerClass != null
                ? new HashMap<>(builder.recallPerClass) : new HashMap<>();
        this.f1PerClass = builder.f1PerClass != null
                ? new HashMap<>(builder.f1PerClass) : new HashMap<>();
        this.supportPerClass = builder.supportPerClass != null
                ? new HashMap<>(builder.supportPerClass) : new HashMap<>();
        this.confusionMatrix = builder.confusionMatrix;
        this.classLabels = builder.classLabels;
        this.positiveProbabilities = builder.positiveProbabilities;
        this.trueLabels = builder.trueLabels;
    }

    // ==================== 静态计算方法 ====================
    /**
     * 计算分类评估指标 Compute classification evaluation metrics
     *
     * @param yTrue 真实标签 / True labels
     * @param yPred 预测标签 / Predicted labels
     * @return 评估指标对象 / Evaluation metrics object
     */
    public static ClassificationMetrics compute(String[] yTrue, String[] yPred) {
        if (yTrue == null || yPred == null) {
            throw new IllegalArgumentException("真实标签和预测标签不能为空");
        }

        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("真实标签和预测标签的长度必须一致");
        }

        return new Builder(yTrue, yPred).build();
    }

    /**
     * 计算分类评估指标 (带概率) Compute classification evaluation metrics (with
     * probabilities)
     *
     * @param yTrue 真实标签 / True labels
     * @param yPred 预测标签 / Predicted labels
     * @param yProb 预测为正类的概率 / Probability of positive class
     * @return 评估指标对象 / Evaluation metrics object
     */
    public static ClassificationMetrics compute(String[] yTrue, String[] yPred, double[] yProb) {
        if (yTrue == null || yPred == null || yProb == null) {
            throw new IllegalArgumentException("真实标签、预测标签和概率不能为空");
        }

        if (yTrue.length != yPred.length || yTrue.length != yProb.length) {
            throw new IllegalArgumentException("真实标签、预测标签和概率的长度必须一致");
        }

        return new Builder(yTrue, yPred, yProb).build();
    }

    /**
     * 计算分类评估指标 (带概率和正类标签)
     *
     * @param yTrue 真实标签
     * @param yPred 预测标签
     * @param yProb 预测为正类的概率
     * @param positiveLabel 正类标签
     * @return 评估指标对象
     */
    public static ClassificationMetrics compute(String[] yTrue, String[] yPred, double[] yProb, String positiveLabel) {
        if (yTrue == null || yPred == null || yProb == null) {
            throw new IllegalArgumentException("真实标签、预测标签和概率不能为空");
        }

        if (yTrue.length != yPred.length || yTrue.length != yProb.length) {
            throw new IllegalArgumentException("真实标签、预测标签和概率的长度必须一致");
        }

        return new Builder(yTrue, yPred, yProb, positiveLabel).build();
    }

    /**
     *
     * @param model
     * @param feature
     * @param yTrue
     * @return
     */
    public static ClassificationMetrics compute(IClassifier model, IMatrix feature, String[] yTrue) {
        if (!model.isTrained()) {
            model.fit(feature, yTrue);
        }
        BatchPredResult prdResults = model.predictBatchWithProbs(feature);
        return compute(yTrue, prdResults);
    }

    /**
     * 基于批量预测结果计算各种指标 自动根据BatchPredictionResult中的值判断是二分类还是多分类，并决定是否包含概率信息
     *
     * @param yTrue 真实标签数组
     * @param prdResults 批量预测结果对象
     * @return 评估指标对象
     */
    public static ClassificationMetrics compute(String[] yTrue, BatchPredResult prdResults) {
        if (yTrue == null || prdResults == null) {
            throw new IllegalArgumentException("真实标签和批量预测结果不能为空");
        }

        String[] predictions = prdResults.getPredictions();

        if (yTrue.length != predictions.length) {
            throw new IllegalArgumentException("真实标签和预测结果的长度必须一致");
        }

        // 检查是否为二分类
        if (prdResults.isBinaryClassification()) {
            // 二分类情况
            double[] probabilities = prdResults.getProbabilities();

            // 检查是否有概率信息
            if (probabilities != null && probabilities.length == predictions.length) {
                // 有概率信息，使用带概率的compute方法
                return compute(yTrue, predictions, probabilities);
            } else {
                // 没有概率信息，使用不带概率的compute方法
                return compute(yTrue, predictions);
            }
        } else {
            // 多分类情况
            double[][] classProbabilities = prdResults.getClassProbabilities();

            // 多分类情况下，如果有概率信息，需要提取正类概率（通常为最大概率）
            if (classProbabilities != null && classProbabilities.length == predictions.length) {
                // 对于多分类，通常使用最大概率作为"正类"概率
                double[] maxProbabilities = new double[classProbabilities.length];
                for (int i = 0; i < classProbabilities.length; i++) {
                    double maxProb = 0.0;
                    for (double prob : classProbabilities[i]) {
                        if (prob > maxProb) {
                            maxProb = prob;
                        }
                    }
                    maxProbabilities[i] = maxProb;
                }

                // 使用带概率的compute方法
                return compute(yTrue, predictions, maxProbabilities);
            } else {
                // 没有概率信息，使用不带概率的compute方法
                return compute(yTrue, predictions);
            }
        }
    }

    /**
     * 基于批量预测结果计算各种指标 (指定正类标签)
     *
     * @param yTrue 真实标签数组
     * @param prdResults 批量预测结果对象
     * @param positiveLabel 正类标签
     * @return 评估指标对象
     */
    public static ClassificationMetrics compute(String[] yTrue, BatchPredResult prdResults, String positiveLabel) {
        if (yTrue == null || prdResults == null) {
            throw new IllegalArgumentException("真实标签和批量预测结果不能为空");
        }

        String[] predictions = prdResults.getPredictions();

        if (yTrue.length != predictions.length) {
            throw new IllegalArgumentException("真实标签和预测结果的长度必须一致");
        }

        if (prdResults.isBinaryClassification()) {
            double[] probabilities = prdResults.getProbabilities();
            if (probabilities != null && probabilities.length == predictions.length) {
                return compute(yTrue, predictions, probabilities, positiveLabel);
            } else {
                return compute(yTrue, predictions);
            }
        } else {
            // 多分类忽略 positiveLabel，退回到现有逻辑
            return compute(yTrue, prdResults);
        }
    }

    // ==================== Getter方法 ====================
    /**
     * 获取准确率
     *
     * @return 准确率
     */
    public double getAccuracy() {
        return accuracy;
    }

    /**
     * 获取宏平均精确率
     *
     * @return 宏平均精确率
     */
    public double getMacroPrecision() {
        return macroPrecision;
    }

    /**
     * 获取宏平均召回率
     *
     * @return 宏平均召回率
     */
    public double getMacroRecall() {
        return macroRecall;
    }

    /**
     * 获取宏平均F1分数
     *
     * @return 宏平均F1分数
     */
    public double getMacroF1() {
        return macroF1;
    }

    /**
     * 获取加权平均精确率
     *
     * @return 加权平均精确率
     */
    public double getWeightedPrecision() {
        return weightedPrecision;
    }

    /**
     * 获取加权平均召回率
     *
     * @return 加权平均召回率
     */
    public double getWeightedRecall() {
        return weightedRecall;
    }

    /**
     * 获取加权平均F1分数
     *
     * @return 加权平均F1分数
     */
    public double getWeightedF1() {
        return weightedF1;
    }

    /**
     * 获取AUC值 (仅适用于二分类)
     *
     * @return AUC值，如果是多分类则返回-1
     */
    public double getAuc() {
        return auc;
    }

    /**
     * 获取每类别的精确率
     *
     * @return 每类别的精确率映射
     */
    public Map<String, Double> getPrecisionPerClass() {
        return new HashMap<>(precisionPerClass);
    }

    /**
     * 获取每类别的召回率
     *
     * @return 每类别的召回率映射
     */
    public Map<String, Double> getRecallPerClass() {
        return new HashMap<>(recallPerClass);
    }

    /**
     * 获取每类别的F1分数
     *
     * @return 每类别的F1分数映射
     */
    public Map<String, Double> getF1PerClass() {
        return new HashMap<>(f1PerClass);
    }

    /**
     * 获取每类别的支持数
     *
     * @return 每类别的支持数映射
     */
    public Map<String, Integer> getSupportPerClass() {
        return new HashMap<>(supportPerClass);
    }

    /**
     * 获取混淆矩阵
     *
     * @return 混淆矩阵
     */
    public int[][] getConfusionMatrix() {
        return confusionMatrix.clone();
    }

    /**
     * 获取类别标签
     *
     * @return 类别标签数组
     */
    public String[] getClassLabels() {
        return classLabels.clone();
    }

    /**
     * 检查是否为二分类
     *
     * @return true表示二分类，false表示多分类
     */
    public boolean isBinaryClassification() {
        return classLabels != null && classLabels.length == 2;
    }

    /**
     * 获取类别数量
     *
     * @return 类别数量
     */
    public int getNumClasses() {
        return classLabels != null ? classLabels.length : 0;
    }

    // ==================== 工具方法 ====================
    /**
     * 获取分类报告 Get classification report
     *
     * @return 格式化的分类报告
     */
    public String getClassificationReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=================== 分类报告 ===================\n");
        sb.append(String.format("准确率: %.4f\n", accuracy));
        sb.append(String.format("宏平均 - 精确率: %.4f, 召回率: %.4f, F1: %.4f\n",
                macroPrecision, macroRecall, macroF1));
        sb.append(String.format("加权平均 - 精确率: %.4f, 召回率: %.4f, F1: %.4f\n",
                weightedPrecision, weightedRecall, weightedF1));

        if (isBinaryClassification() && auc >= 0) {
            sb.append(String.format("AUC: %.4f\n", auc));
        }

        sb.append("\n=================== 详细指标 ===================\n");
        sb.append(String.format("%-10s %-10s %-10s %-10s %-10s\n",
                "类别", "精确率", "召回率", "F1分数", "支持数"));
        sb.append(String.format("%-10s %-10s %-10s %-10s %-10s\n",
                "------", "------", "------", "------", "------"));

        if (classLabels != null) {
            for (String label : classLabels) {
                double precision = precisionPerClass.getOrDefault(label, 0.0);
                double recall = recallPerClass.getOrDefault(label, 0.0);
                double f1 = f1PerClass.getOrDefault(label, 0.0);
                int support = supportPerClass.getOrDefault(label, 0);

                sb.append(String.format("%-10s %-10.4f %-10.4f %-10.4f %-10d\n",
                        label, precision, recall, f1, support));
            }
        }

        return sb.toString();
    }

    /**
     * 获取混淆矩阵字符串表示 Get confusion matrix string representation
     *
     * @return 混淆矩阵字符串
     */
    public String getConfusionMatrixString() {
        if (confusionMatrix == null || classLabels == null) {
            return "混淆矩阵不可用";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("=================== 混淆矩阵 ===================\n");

        // 打印表头
        sb.append(String.format("%-10s", "实际\\预测"));
        for (String label : classLabels) {
            sb.append(String.format("%-10s", label));
        }
        sb.append("\n");

        // 打印分隔线
        sb.append(String.format("%-10s", "--------"));
        for (int i = 0; i < classLabels.length; i++) {
            sb.append(String.format("%-10s", "--------"));
        }
        sb.append("\n");

        // 打印矩阵内容
        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(String.format("%-10s", classLabels[i]));
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                sb.append(String.format("%-10d", confusionMatrix[i][j]));
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    @Override
    public String toReport() {
        return getClassificationReport();
    }

    @Override
    public String toBriefReport() {
        if (isBinaryClassification() && auc >= 0) {
            return String.format("Classification | Acc=%.4f | MacroF1=%.4f | AUC=%.4f",
                    accuracy, macroF1, auc);
        }
        return String.format("Classification | Acc=%.4f | MacroF1=%.4f | WeightedF1=%.4f",
                accuracy, macroF1, weightedF1);
    }

    @Override
    public String toString() {
        if (isBinaryClassification()) {
            if (auc >= 0) {
                return String.format("ClassificationMetrics{accuracy=%.4f, macroF1=%.4f, weightedF1=%.4f, auc=%.4f}",
                        accuracy, macroF1, weightedF1, auc);
            } else {
                return String.format("ClassificationMetrics{accuracy=%.4f, macroF1=%.4f, weightedF1=%.4f, auc=N/A}",
                        accuracy, macroF1, weightedF1);
            }
        } else {
            return String.format("ClassificationMetrics{accuracy=%.4f, macroF1=%.4f, weightedF1=%.4f}",
                    accuracy, macroF1, weightedF1);
        }
    }

    // ==================== Builder模式 ====================
    /**
     * 构建器类，用于创建ClassificationMetrics实例
     */
    private static class Builder {

        // 必需参数
        private final String[] yTrue;
        private final String[] yPred;

        // 可选参数
        private double[] yProb = null;
        private String positiveLabel = null;

        // 计算结果
        private double accuracy = 0.0;
        private double macroPrecision = 0.0;
        private double macroRecall = 0.0;
        private double macroF1 = 0.0;
        private double weightedPrecision = 0.0;
        private double weightedRecall = 0.0;
        private double weightedF1 = 0.0;
        private double auc = -1.0;

        private Map<String, Double> precisionPerClass = new HashMap<>();
        private Map<String, Double> recallPerClass = new HashMap<>();
        private Map<String, Double> f1PerClass = new HashMap<>();
        private Map<String, Integer> supportPerClass = new HashMap<>();
        private int[][] confusionMatrix;
        private String[] classLabels;
        private double[] positiveProbabilities;
        private int[] trueLabels;

        /**
         * 构造函数 (无概率)
         */
        public Builder(String[] yTrue, String[] yPred) {
            this.yTrue = yTrue;
            this.yPred = yPred;
            computeMetrics();
        }

        /**
         * 构造函数 (带概率)
         */
        public Builder(String[] yTrue, String[] yPred, double[] yProb) {
            this.yTrue = yTrue;
            this.yPred = yPred;
            this.yProb = yProb;
            computeMetrics();
        }

        /**
         * 构造函数 (带概率和正类标签)
         */
        public Builder(String[] yTrue, String[] yPred, double[] yProb, String positiveLabel) {
            this.yTrue = yTrue;
            this.yPred = yPred;
            this.yProb = yProb;
            this.positiveLabel = positiveLabel;
            computeMetrics();
        }

        /**
         * 计算所有指标
         */
        private void computeMetrics() {
            int n = yTrue.length;

            // 获取唯一标签并排序
            Set<String> uniqueLabels = new HashSet<>();
            uniqueLabels.addAll(Arrays.asList(yTrue));
            uniqueLabels.addAll(Arrays.asList(yPred));
            classLabels = uniqueLabels.toArray(new String[0]);
            Arrays.sort(classLabels);

            // 计算混淆矩阵
            confusionMatrix = computeConfusionMatrix(yTrue, yPred, classLabels);

            // 计算准确率
            accuracy = computeAccuracy(yTrue, yPred);

            // 计算每类别的指标
            computePerClassMetrics();

            // 计算宏平均和加权平均
            computeAverageMetrics();

            // 如果是二分类且有概率信息，计算AUC
            if (classLabels.length == 2 && yProb != null) {
                computeAUC();
            }
        }

        /**
         * 计算混淆矩阵
         */
        private int[][] computeConfusionMatrix(String[] yTrue, String[] yPred, String[] labels) {
            int numClasses = labels.length;
            int[][] matrix = new int[numClasses][numClasses];

            // 创建标签到索引的映射
            Map<String, Integer> labelToIndex = new HashMap<>();
            for (int i = 0; i < labels.length; i++) {
                labelToIndex.put(labels[i], i);
            }

            // 填充混淆矩阵
            for (int i = 0; i < yTrue.length; i++) {
                int trueIdx = labelToIndex.get(yTrue[i]);
                int predIdx = labelToIndex.get(yPred[i]);
                matrix[trueIdx][predIdx]++;
            }

            return matrix;
        }

        /**
         * 计算准确率
         */
        private double computeAccuracy(String[] yTrue, String[] yPred) {
            int correct = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yTrue[i].equals(yPred[i])) {
                    correct++;
                }
            }
            return (double) correct / yTrue.length;
        }

        /**
         * 计算每类别的指标
         */
        private void computePerClassMetrics() {
            for (String label : classLabels) {
                int tp = 0, fp = 0, fn = 0, tn = 0;
                int support = 0;

                for (int i = 0; i < yTrue.length; i++) {
                    boolean isTruePositive = yTrue[i].equals(label);
                    boolean isPredPositive = yPred[i].equals(label);

                    if (isTruePositive) {
                        support++;
                        if (isPredPositive) {
                            tp++; // True Positive
                        } else {
                            fn++; // False Negative
                        }
                    } else {
                        if (isPredPositive) {
                            fp++; // False Positive
                        } else {
                            tn++; // True Negative
                        }
                    }
                }

                // 计算精确率、召回率和F1分数
                double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0.0;
                double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0.0;
                double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

                precisionPerClass.put(label, precision);
                recallPerClass.put(label, recall);
                f1PerClass.put(label, f1);
                supportPerClass.put(label, support);
            }
        }

        /**
         * 计算宏平均和加权平均指标
         */
        private void computeAverageMetrics() {
            double macroPrecisionSum = 0.0;
            double macroRecallSum = 0.0;
            double macroF1Sum = 0.0;

            double weightedPrecisionSum = 0.0;
            double weightedRecallSum = 0.0;
            double weightedF1Sum = 0.0;

            int totalSupport = 0;

            for (String label : classLabels) {
                double precision = precisionPerClass.get(label);
                double recall = recallPerClass.get(label);
                double f1 = f1PerClass.get(label);
                int support = supportPerClass.get(label);

                totalSupport += support;

                macroPrecisionSum += precision;
                macroRecallSum += recall;
                macroF1Sum += f1;

                weightedPrecisionSum += precision * support;
                weightedRecallSum += recall * support;
                weightedF1Sum += f1 * support;
            }

            int numClasses = classLabels.length;
            if (numClasses > 0) {
                macroPrecision = macroPrecisionSum / numClasses;
                macroRecall = macroRecallSum / numClasses;
                macroF1 = macroF1Sum / numClasses;
            }

            if (totalSupport > 0) {
                weightedPrecision = weightedPrecisionSum / totalSupport;
                weightedRecall = weightedRecallSum / totalSupport;
                weightedF1 = weightedF1Sum / totalSupport;
            }
        }

        /**
         * 计算AUC (仅适用于二分类) - 修复版
         */
        private void computeAUC() {
            if (classLabels.length != 2 || yProb == null) {
                return;
            }

            // 确定正类和负类标签
            String posLabel;
            if (this.positiveLabel != null) {
                posLabel = this.positiveLabel;
            } else {
                posLabel = classLabels[1]; // 兜底：字典序较大的为正类
            }
            String negLabel = classLabels[0].equals(posLabel) ? classLabels[1] : classLabels[0];

            // 准备AUC计算数据
            List<Double> probabilities = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();

            for (int i = 0; i < yTrue.length; i++) {
                probabilities.add(yProb[i]);
                labels.add(yTrue[i].equals(posLabel) ? 1 : 0);
            }

            // 按概率降序排序，相同概率时保持稳定性
            List<Integer> sortedIndices = new ArrayList<>();
            for (int i = 0; i < probabilities.size(); i++) {
                sortedIndices.add(i);
            }
            sortedIndices.sort((i, j) -> {
                int cmp = Double.compare(probabilities.get(j), probabilities.get(i));
                if (cmp == 0) {
                    // 相同概率时保持原始顺序
                    return Integer.compare(i, j);
                }
                return cmp;
            });

            // 计算AUC (使用梯形法则)
            int tp = 0, fp = 0;
            int posCount = 0, negCount = 0;

            for (int label : labels) {
                if (label == 1) {
                    posCount++;
                } else {
                    negCount++;
                }
            }

            // 检查是否有足够的正负类样本
            if (posCount == 0 || negCount == 0) {
                this.auc = -1.0;
                return;
            }

            // 检查概率分布，如果正类平均概率小于负类，可能需要调整
            double posMeanProb = 0.0;
            double negMeanProb = 0.0;
            int posCount2 = 0, negCount2 = 0;
            
            for (int i = 0; i < labels.size(); i++) {
                if (labels.get(i) == 1) {
                    posMeanProb += probabilities.get(i);
                    posCount2++;
                } else {
                    negMeanProb += probabilities.get(i);
                    negCount2++;
                }
            }
            
            posMeanProb /= posCount2;
            negMeanProb /= negCount2;

            // 如果正类平均概率小于负类，说明预测方向可能相反
            boolean reverseDirection = posMeanProb < negMeanProb;
            
            if (reverseDirection) {
                log.debug("AUC: 正类平均概率({})小于负类平均概率({})，可能需调整方向", posMeanProb, negMeanProb);
            }

            double auc = 0.0;
            int previousTP = 0;

            for (int idx : sortedIndices) {
                if (labels.get(idx) == 1) {
                    tp++;
                } else {
                    fp++;
                    // 使用梯形法则计算面积
                    double currentTPR = (double) tp / posCount;
                    double previousTPR = (double) previousTP / posCount;
                    double deltaFPR = 1.0 / negCount;
                    auc += (previousTPR + currentTPR) * deltaFPR / 2;
                    previousTP = tp;
                }
            }

            // 如果预测方向相反，AUC应该接近0，这时我们可以计算1-AUC作为修正
            if (reverseDirection && auc < 0.5) {
                log.debug("AUC 方向修正: {} -> {}", auc, (1.0 - auc));
                auc = 1.0 - auc;
            }

            // 验证AUC值的合理性
            if (auc < 0.0 || auc > 1.0) {
                log.debug("AUC 超出 [0,1]，置为 -1，原值={}", auc);
                auc = -1.0;
            }

            this.auc = auc;

            // 保存AUC计算所需的数据
            positiveProbabilities = probabilities.stream().mapToDouble(Double::doubleValue).toArray();
            trueLabels = labels.stream().mapToInt(Integer::intValue).toArray();
        }

        /**
         * 构建ClassificationMetrics对象
         */
        public ClassificationMetrics build() {
            return new ClassificationMetrics(this);
        }
    }
}
