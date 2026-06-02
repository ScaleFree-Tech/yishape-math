package com.yishape.lab.math.ml.metric;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 交叉验证结果类
 *
 * @author lteb2
 */
public class CrossValidationResult {

    /**
     * 每次折叠的准确率
     */
    private final List<Double> accuracyScores;

    /**
     * 每次折叠的F1分数
     */
    private final List<Double> f1Scores;

    /**
     * 每次折叠的精确率
     */
    private final List<Double> precisionScores;

    /**
     * 每次折叠的召回率
     */
    private final List<Double> recallScores;

    /**
     * 每次折叠的AUC (如果是二分类)
     */
    private final List<Double> aucScores;

    /**
     * 平均准确率
     */
    private final double meanAccuracy;

    /**
     * 准确率标准差
     */
    private final double stdAccuracy;

    /**
     * 平均F1分数
     */
    private final double meanF1;

    /**
     * F1分数标准差
     */
    private final double stdF1;

    /**
     * 平均精确率
     */
    private final double meanPrecision;

    /**
     * 平均召回率
     */
    private final double meanRecall;

    /**
     * 平均AUC
     */
    private final double meanAuc;

    /**
     * 总验证次数
     */
    private final int totalFolds;

    /**
     * 使用的验证类型
     */
    private final CrossValidation.CrossValidationType validationType;

    /**
     * 每折的训练时间 (毫秒)
     */
    private final List<Long> trainingTimes;

    /**
     * 每折的预测时间 (毫秒)
     */
    private final List<Long> predictionTimes;

    /**
     * 构造函数
     */
    public CrossValidationResult(List<Double> accuracyScores, List<Double> f1Scores,
            List<Double> precisionScores, List<Double> recallScores,
            List<Double> aucScores, List<Long> trainingTimes,
            List<Long> predictionTimes, CrossValidation.CrossValidationType validationType) {
        this.accuracyScores = new ArrayList<>(accuracyScores);
        this.f1Scores = new ArrayList<>(f1Scores);
        this.precisionScores = new ArrayList<>(precisionScores);
        this.recallScores = new ArrayList<>(recallScores);
        this.aucScores = new ArrayList<>(aucScores);
        this.trainingTimes = new ArrayList<>(trainingTimes);
        this.predictionTimes = new ArrayList<>(predictionTimes);
        this.validationType = validationType;
        this.totalFolds = accuracyScores.size();

        // 计算统计指标
        this.meanAccuracy = calculateMean(accuracyScores);
        this.stdAccuracy = calculateStd(accuracyScores, meanAccuracy);
        this.meanF1 = calculateMean(f1Scores);
        this.stdF1 = calculateStd(f1Scores, meanF1);
        this.meanPrecision = calculateMean(precisionScores);
        this.meanRecall = calculateMean(recallScores);
        this.meanAuc = aucScores.isEmpty() ? -1.0 : calculateMean(aucScores);
    }

    /**
     * 计算平均值
     */
    private double calculateMean(List<Double> values) {
        if (values.isEmpty()) {
            return 0.0;
        }
        return values.stream().mapToDouble(Double::doubleValue).sum() / values.size();
    }

    /**
     * 计算标准差
     */
    private double calculateStd(List<Double> values, double mean) {
        if (values.size() <= 1) {
            return 0.0;
        }
        double variance = values.stream()
                .mapToDouble(v -> (v - mean) * (v - mean))
                .sum() / (values.size() - 1);
        return Math.sqrt(variance);
    }

    // ==================== Getter方法 ====================
    public List<Double> getAccuracyScores() {
        return new ArrayList<>(accuracyScores);
    }

    public List<Double> getF1Scores() {
        return new ArrayList<>(f1Scores);
    }

    public List<Double> getPrecisionScores() {
        return new ArrayList<>(precisionScores);
    }

    public List<Double> getRecallScores() {
        return new ArrayList<>(recallScores);
    }

    public List<Double> getAucScores() {
        return new ArrayList<>(aucScores);
    }

    public List<Long> getTrainingTimes() {
        return new ArrayList<>(trainingTimes);
    }

    public List<Long> getPredictionTimes() {
        return new ArrayList<>(predictionTimes);
    }

    public double getMeanAccuracy() {
        return meanAccuracy;
    }

    public double getStdAccuracy() {
        return stdAccuracy;
    }

    public double getMeanF1() {
        return meanF1;
    }

    public double getStdF1() {
        return stdF1;
    }

    public double getMeanPrecision() {
        return meanPrecision;
    }

    public double getMeanRecall() {
        return meanRecall;
    }

    public double getMeanAuc() {
        return meanAuc;
    }

    public int getTotalFolds() {
        return totalFolds;
    }

    public CrossValidation.CrossValidationType getValidationType() {
        return validationType;
    }

    /**
     * 获取95%置信区间
     */
    public double[] getAccuracy95Percentile() {
        return getPercentile(accuracyScores, 2.5, 97.5);
    }

    /**
     * 获取百分位数
     */
    private double[] getPercentile(List<Double> values, double... percentiles) {
        if (values.isEmpty()) {
            return new double[percentiles.length];
        }

        List<Double> sortedValues = new ArrayList<>(values);
        Collections.sort(sortedValues);

        double[] result = new double[percentiles.length];
        for (int i = 0; i < percentiles.length; i++) {
            double p = percentiles[i];
            double index = p / 100.0 * (sortedValues.size() - 1);
            int lower = (int) Math.floor(index);
            int upper = (int) Math.ceil(index);

            if (lower == upper) {
                result[i] = sortedValues.get(lower);
            } else {
                double weight = index - lower;
                result[i] = sortedValues.get(lower) * (1 - weight) + sortedValues.get(upper) * weight;
            }
        }
        return result;
    }

    /**
     * 获取详细报告
     */
    public String getDetailedReport() {
        StringBuilder sb = new StringBuilder();
        sb.append("=================== 交叉验证结果报告 ===================\n");
        sb.append(String.format("验证类型: %s\n", validationType));
        sb.append(String.format("验证次数: %d\n", totalFolds));

        sb.append("\n=== 准确率统计 ===\n");
        sb.append(String.format("平均值: %.4f ± %.4f\n", meanAccuracy, stdAccuracy));
        sb.append(String.format("最小值: %.4f\n", Collections.min(accuracyScores)));
        sb.append(String.format("最大值: %.4f\n", Collections.max(accuracyScores)));
        sb.append(String.format("95%%置信区间: [%.4f, %.4f]\n",
                getAccuracy95Percentile()[0], getAccuracy95Percentile()[1]));

        sb.append("\n=== F1分数统计 ===\n");
        sb.append(String.format("平均值: %.4f ± %.4f\n", meanF1, stdF1));
        sb.append(String.format("最小值: %.4f\n", Collections.min(f1Scores)));
        sb.append(String.format("最大值: %.4f\n", Collections.max(f1Scores)));

        sb.append("\n=== 精确率和召回率 ===\n");
        sb.append(String.format("平均精确率: %.4f\n", meanPrecision));
        sb.append(String.format("平均召回率: %.4f\n", meanRecall));

        if (meanAuc >= 0) {
            sb.append("\n=== AUC统计 ===\n");
            sb.append(String.format("平均AUC: %.4f\n", meanAuc));
            sb.append(String.format("最小AUC: %.4f\n", Collections.min(aucScores)));
            sb.append(String.format("最大AUC: %.4f\n", Collections.max(aucScores)));
        }

        sb.append("\n=== 时间性能 ===\n");
        long totalTrainingTime = trainingTimes.stream().mapToLong(Long::longValue).sum();
        long totalPredictionTime = predictionTimes.stream().mapToLong(Long::longValue).sum();
        sb.append(String.format("总训练时间: %d ms\n", totalTrainingTime));
        sb.append(String.format("总预测时间: %d ms\n", totalPredictionTime));
        sb.append(String.format("平均每折训练时间: %.2f ms\n",
                (double) totalTrainingTime / totalFolds));
        sb.append(String.format("平均每折预测时间: %.2f ms\n",
                (double) totalPredictionTime / totalFolds));

        sb.append("\n=== 每折详细结果 ===\n");
        sb.append(String.format("%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n",
                "折数", "准确率", "F1分数", "精确率", "召回率", "AUC", "训练时间(ms)"));
        sb.append(String.format("%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n",
                "----", "------", "-------", "-------", "-------", "---", "----------"));

        for (int i = 0; i < totalFolds; i++) {
            sb.append(String.format("%-6d %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10d\n",
                    i + 1, accuracyScores.get(i), f1Scores.get(i),
                    precisionScores.get(i), recallScores.get(i),
                    aucScores.isEmpty() ? -1.0 : aucScores.get(i),
                    trainingTimes.get(i)));
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("CrossValidationResult{meanAccuracy=%.4f±%.4f, meanF1=%.4f±%.4f, folds=%d}",
                meanAccuracy, stdAccuracy, meanF1, stdF1, totalFolds);
    }
}
