package com.yishape.lab.math.ml.clf;

import com.yishape.lab.util.IRichReport;
import com.yishape.lab.util.ReportBuilder;

/**
 * 批量预测结果类，包含预测标签和概率
 *
 * @author lteb2
 */
public class BatchPredResult implements IRichReport {

    private final String[] predictions;
    private final double[] probabilities; // 二分类时使用
    private final double[][] classProbabilities; // 多分类时使用
    private final boolean isBinaryClassification;

    /**
     * 构造函数（二分类）
     * @param predictions
     * @param probabilities
     */
    public BatchPredResult(String[] predictions, double[] probabilities) {
        this.predictions = predictions;
        this.probabilities = probabilities;
        this.classProbabilities = null;
        this.isBinaryClassification = true;
    }

    /**
     * 构造函数（多分类）
     * @param predictions
     * @param classProbabilities
     */
    public BatchPredResult(String[] predictions, double[][] classProbabilities) {
        this.predictions = predictions;
        this.probabilities = null;
        this.classProbabilities = classProbabilities;
        this.isBinaryClassification = false;
    }

    /**
     * 获取预测标签数组
     * @return 
     */
    public String[] getPredictions() {
        return predictions;
    }

    /**
     * 获取二分类概率数组（仅适用于二分类）
     * @return 
     */
    public double[] getProbabilities() {
        if (!isBinaryClassification) {
            throw new IllegalStateException("getProbabilities()方法仅适用于二分类模型");
        }
        return probabilities;
    }

    /**
     * 获取多分类概率矩阵（仅适用于多分类）
     * @return 
     */
    public double[][] getClassProbabilities() {
        if (isBinaryClassification) {
            throw new IllegalStateException("getClassProbabilities()方法仅适用于多分类模型");
        }
        return classProbabilities;
    }

    /**
     * 检查是否为二分类结果
     * @return 
     */
    public boolean isBinaryClassification() {
        return isBinaryClassification;
    }

    /**
     * 获取样本数量
     * @return 
     */
    public int getSampleCount() {
        return predictions.length;
    }

    @Override
    public String toReport() {
        ReportBuilder rb = new ReportBuilder("Batch Prediction Result");
        rb.kv("Sample count", getSampleCount());
        rb.kv("Type", isBinaryClassification ? "Binary" : "Multiclass");
        if (isBinaryClassification && probabilities != null && probabilities.length > 0) {
            double min = Double.MAX_VALUE, max = Double.MIN_VALUE, sum = 0;
            for (double p : probabilities) {
                if (p < min) min = p;
                if (p > max) max = p;
                sum += p;
            }
            rb.kv("Prob range", String.format("[%.4f, %.4f]", min, max));
            rb.kv("Prob mean", sum / probabilities.length);
        } else if (!isBinaryClassification && classProbabilities != null && classProbabilities.length > 0) {
            rb.kv("Class count", classProbabilities[0].length);
        }
        return rb.build();
    }

    @Override
    public String toBriefReport() {
        return String.format("BatchPred | samples=%d | %s",
                getSampleCount(), isBinaryClassification ? "binary" : "multiclass");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("BatchPredictionResult{");
        sb.append("样本数=").append(getSampleCount());
        sb.append(", 类型=").append(isBinaryClassification ? "二分类" : "多分类");
        if (isBinaryClassification) {
            sb.append(", 概率范围=[");
            if (probabilities.length > 0) {
                double min = Double.MAX_VALUE;
                double max = Double.MIN_VALUE;
                for (double p : probabilities) {
                    if (p < min) {
                        min = p;
                    }
                    if (p > max) {
                        max = p;
                    }
                }
                sb.append(String.format("%.4f, %.4f", min, max));
            }
            sb.append("]");
        } else {
            sb.append(", 类别数=").append(classProbabilities[0].length);
        }
        sb.append("}");
        return sb.toString();
    }
}
