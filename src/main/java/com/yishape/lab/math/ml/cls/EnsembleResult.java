package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.linalg.IVector;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author lteb2
 */
/**
 * 集成分类结果类 - 实现IClassification接口所需的返回类型
 */
public class EnsembleResult extends ClassificationResult {

    private final EnsembleClassifier.EnsembleStrategy strategy;
    private final IVector classifierWeights;
    private final Map<String, Double> classifierAccuracies;
    private final boolean trained;

    public EnsembleResult(EnsembleClassifier.EnsembleStrategy strategy,
            IVector classifierWeights,
            String[] classLabels,
            double accuracy,
            Map<String, Double> classifierAccuracies) {
        super();
        this.strategy = strategy;
        this.classifierWeights = classifierWeights;
        this.classifierAccuracies = classifierAccuracies;
        this.trained = true;

        // 使用父类的属性
        this.setTrainAccuracy(accuracy);
        this.setNumClasses(classLabels.length);

        // 构建标签映射
        Map<String, Integer> labelMapping = new HashMap<>();
        Map<Integer, String> reverseLabelMapping = new HashMap<>();
        for (int i = 0; i < classLabels.length; i++) {
            labelMapping.put(classLabels[i], i);
            reverseLabelMapping.put(i, classLabels[i]);
        }
        this.setLabelMapping(labelMapping);
        this.setReverseLabelMapping(reverseLabelMapping);
    }

    public EnsembleClassifier.EnsembleStrategy getStrategy() {
        return strategy;
    }

    public IVector getClassifierWeights() {
        return classifierWeights;
    }

    public Map<String, Double> getClassifierAccuracies() {
        return classifierAccuracies;
    }

    // ==================== 实现抽象方法 ====================
    @Override
    public String getModelTypeDescription() {
        return String.format("集成分类器 (%s, %d类)",
                strategy.toString(), getNumClasses());
    }

    @Override
    public String getModelSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 集成分类器模型摘要 ===\n");
        sb.append(getBasicStats()).append("\n");
        sb.append(String.format("集成策略: %s\n", strategy));
        sb.append("分类器权重: [");
        for (int i = 0; i < classifierWeights.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(String.format("%.3f", classifierWeights.get(i).doubleValue()));
        }
        sb.append("]\n");

        if (classifierAccuracies != null && !classifierAccuracies.isEmpty()) {
            sb.append("各分类器准确率:\n");
            for (Map.Entry<String, Double> entry : classifierAccuracies.entrySet()) {
                sb.append(String.format("  %s: %.4f\n", entry.getKey(), entry.getValue()));
            }
        }

        sb.append("模型状态: ").append(isTrained() ? "已训练" : "未训练");
        return sb.toString();
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    // ==================== 向后兼容性方法 ====================
    public String[] getClassLabels() {
        return super.getClassLabels();
    }

    public double getAccuracy() {
        return getTrainAccuracy();
    }
}
