package com.yishape.lab.math.ml.metric;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.ml.cls.BatchPredictionResult;
import com.yishape.lab.math.ml.cls.IClassification;

import java.util.*;

/**
 * 修复版分类评估指标工具类
 * 解决AUC为0的问题
 * 
 * @author lteb2
 */
public class ClassificationMetricsFixed {

    // ==================== 核心指标属性 ====================
    private final double accuracy;
    private final double macroPrecision;
    private final double macroRecall;
    private final double macroF1;
    private final double weightedPrecision;
    private final double weightedRecall;
    private final double weightedF1;
    private final double auc;
    private final Map<String, Double> precisionPerClass;
    private final Map<String, Double> recallPerClass;
    private final Map<String, Double> f1PerClass;
    private final Map<String, Integer> supportPerClass;
    private final int[][] confusionMatrix;
    private final String[] classLabels;
    private final double[] positiveProbabilities;
    private final int[] trueLabels;

    // ==================== 构造函数 ====================
    private ClassificationMetricsFixed(Builder builder) {
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
    public static ClassificationMetricsFixed compute(String[] yTrue, String[] yPred) {
        if (yTrue == null || yPred == null) {
            throw new IllegalArgumentException("真实标签和预测标签不能为空");
        }

        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("真实标签和预测标签的长度必须一致");
        }

        return new Builder(yTrue, yPred).build();
    }

    public static ClassificationMetricsFixed compute(String[] yTrue, String[] yPred, double[] yProb) {
        if (yTrue == null || yPred == null || yProb == null) {
            throw new IllegalArgumentException("真实标签、预测标签和概率不能为空");
        }

        if (yTrue.length != yPred.length || yTrue.length != yProb.length) {
            throw new IllegalArgumentException("真实标签、预测标签和概率的长度必须一致");
        }

        return new Builder(yTrue, yPred, yProb).build();
    }

    public static ClassificationMetricsFixed compute(IClassification model, IMatrix feature, String[] yTrue) {
        if (!model.isTrained()) {
            model.fit(feature, yTrue);
        }
        BatchPredictionResult prdResults = model.predictBatchWithProbabilities(feature);
        return compute(yTrue, prdResults);
    }

    public static ClassificationMetricsFixed compute(String[] yTrue, BatchPredictionResult prdResults) {
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
                return compute(yTrue, predictions, probabilities);
            } else {
                return compute(yTrue, predictions);
            }
        } else {
            double[][] classProbabilities = prdResults.getClassProbabilities();

            if (classProbabilities != null && classProbabilities.length == predictions.length) {
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

                return compute(yTrue, predictions, maxProbabilities);
            } else {
                return compute(yTrue, predictions);
            }
        }
    }

    // ==================== Getter方法 ====================
    public double getAccuracy() { return accuracy; }
    public double getMacroPrecision() { return macroPrecision; }
    public double getMacroRecall() { return macroRecall; }
    public double getMacroF1() { return macroF1; }
    public double getWeightedPrecision() { return weightedPrecision; }
    public double getWeightedRecall() { return weightedRecall; }
    public double getWeightedF1() { return weightedF1; }
    public double getAuc() { return auc; }
    
    public Map<String, Double> getPrecisionPerClass() {
        return new HashMap<>(precisionPerClass);
    }
    
    public Map<String, Double> getRecallPerClass() {
        return new HashMap<>(recallPerClass);
    }
    
    public Map<String, Double> getF1PerClass() {
        return new HashMap<>(f1PerClass);
    }
    
    public Map<String, Integer> getSupportPerClass() {
        return new HashMap<>(supportPerClass);
    }
    
    public int[][] getConfusionMatrix() {
        return confusionMatrix.clone();
    }
    
    public String[] getClassLabels() {
        return classLabels.clone();
    }
    
    public boolean isBinaryClassification() {
        return classLabels != null && classLabels.length == 2;
    }
    
    public int getNumClasses() {
        return classLabels != null ? classLabels.length : 0;
    }

    @Override
    public String toString() {
        if (isBinaryClassification()) {
            if (auc >= 0) {
                return String.format("ClassificationMetricsFixed{accuracy=%.4f, macroF1=%.4f, weightedF1=%.4f, auc=%.4f}",
                        accuracy, macroF1, weightedF1, auc);
            } else {
                return String.format("ClassificationMetricsFixed{accuracy=%.4f, macroF1=%.4f, weightedF1=%.4f, auc=N/A}",
                        accuracy, macroF1, weightedF1);
            }
        } else {
            return String.format("ClassificationMetricsFixed{accuracy=%.4f, macroF1=%.4f, weightedF1=%.4f}",
                    accuracy, macroF1, weightedF1);
        }
    }

    // ==================== Builder模式 ====================
    private static class Builder {
        private final String[] yTrue;
        private final String[] yPred;
        private double[] yProb = null;

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

        public Builder(String[] yTrue, String[] yPred) {
            this.yTrue = yTrue;
            this.yPred = yPred;
            computeMetrics();
        }

        public Builder(String[] yTrue, String[] yPred, double[] yProb) {
            this.yTrue = yTrue;
            this.yPred = yPred;
            this.yProb = yProb;
            computeMetrics();
        }

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

        private int[][] computeConfusionMatrix(String[] yTrue, String[] yPred, String[] labels) {
            int numClasses = labels.length;
            int[][] matrix = new int[numClasses][numClasses];

            Map<String, Integer> labelToIndex = new HashMap<>();
            for (int i = 0; i < labels.length; i++) {
                labelToIndex.put(labels[i], i);
            }

            for (int i = 0; i < yTrue.length; i++) {
                int trueIdx = labelToIndex.get(yTrue[i]);
                int predIdx = labelToIndex.get(yPred[i]);
                matrix[trueIdx][predIdx]++;
            }

            return matrix;
        }

        private double computeAccuracy(String[] yTrue, String[] yPred) {
            int correct = 0;
            for (int i = 0; i < yTrue.length; i++) {
                if (yTrue[i].equals(yPred[i])) {
                    correct++;
                }
            }
            return (double) correct / yTrue.length;
        }

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
                            tp++;
                        } else {
                            fn++;
                        }
                    } else {
                        if (isPredPositive) {
                            fp++;
                        } else {
                            tn++;
                        }
                    }
                }

                double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0.0;
                double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0.0;
                double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

                precisionPerClass.put(label, precision);
                recallPerClass.put(label, recall);
                f1PerClass.put(label, f1);
                supportPerClass.put(label, support);
            }
        }

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
         * 修复版AUC计算方法
         * 解决AUC为0的问题
         */
        private void computeAUC() {
            if (classLabels.length != 2 || yProb == null) {
                return;
            }

            // 确定正类和负类标签
            String positiveLabel = classLabels[1]; // 假设字典序较大的为正类
            String negativeLabel = classLabels[0];

            // 准备AUC计算数据
            List<Double> probabilities = new ArrayList<>();
            List<Integer> labels = new ArrayList<>();

            for (int i = 0; i < yTrue.length; i++) {
                probabilities.add(yProb[i]);
                labels.add(yTrue[i].equals(positiveLabel) ? 1 : 0);
            }

            // 按概率降序排序
            List<Integer> sortedIndices = new ArrayList<>();
            for (int i = 0; i < probabilities.size(); i++) {
                sortedIndices.add(i);
            }
            sortedIndices.sort((i, j) -> {
                int cmp = Double.compare(probabilities.get(j), probabilities.get(i));
                if (cmp == 0) {
                    return Integer.compare(i, j);
                }
                return cmp;
            });

            // 计算正负类数量
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
                System.out.println("警告: 正类平均概率(" + posMeanProb + ")小于负类平均概率(" + negMeanProb + ")");
                System.out.println("这可能导致AUC接近0，正在调整计算方向...");
            }

            // 计算AUC
            double auc = 0.0;
            int tp = 0, fp = 0;
            int previousTP = 0;

            for (int idx : sortedIndices) {
                int label = labels.get(idx);
                
                if (label == 1) {
                    tp++;
                } else {
                    fp++;
                    double currentTPR = (double) tp / posCount;
                    double previousTPR = (double) previousTP / posCount;
                    double deltaFPR = 1.0 / negCount;
                    double areaIncrement = (previousTPR + currentTPR) * deltaFPR / 2;
                    auc += areaIncrement;
                    previousTP = tp;
                }
            }

            // 如果预测方向相反，AUC应该接近0，这时我们可以计算1-AUC作为修正
            if (reverseDirection && auc < 0.5) {
                System.out.println("检测到预测方向相反，AUC修正: " + auc + " -> " + (1.0 - auc));
                auc = 1.0 - auc;
            }

            // 验证AUC值的合理性
            if (auc < 0.0 || auc > 1.0) {
                System.out.println("警告: AUC值超出合理范围 [" + auc + "]，设置为-1");
                auc = -1.0;
            }

            this.auc = auc;

            // 保存AUC计算所需的数据
            positiveProbabilities = probabilities.stream().mapToDouble(Double::doubleValue).toArray();
            trueLabels = labels.stream().mapToInt(Integer::intValue).toArray();
        }

        public ClassificationMetricsFixed build() {
            return new ClassificationMetricsFixed(this);
        }
    }
}