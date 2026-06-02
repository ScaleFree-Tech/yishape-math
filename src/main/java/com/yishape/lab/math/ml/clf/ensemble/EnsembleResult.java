package com.yishape.lab.math.ml.clf.ensemble;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.clf.ClfResult;

import java.util.HashMap;
import java.util.Map;

/**
 * 集成分类结果类 / Ensemble Classification Result
 * <p>
 * 实现IClassification接口所需的返回类型，封装集成分类器的结果。
 * Implements IClassification interface return type, encapsulating ensemble classifier results.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class EnsembleResult extends ClfResult {

    private final EnsembleClassifier.EnsembleStrategy strategy;
    private final IVector classifierWeights;
    private final Map<String, Double> classifierAccuracies;
    private final boolean trained;

    /**
     * 构造函数 / Constructor
     * <p>
     * 创建集成分类结果。
     * Create ensemble classification result.
     * </p>
     *
     * @param strategy 集成策略 / Ensemble strategy
     * @param classifierWeights 分类器权重 / Classifier weights
     * @param classLabels 类别标签 / Class labels
     * @param accuracy 准确率 / Accuracy
     * @param classifierAccuracies 各分类器准确率 / Individual classifier accuracies
     */
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

    /**
     * 获取集成策略 / Get ensemble strategy
     *
     * @return 集成策略 / Ensemble strategy
     */
    public EnsembleClassifier.EnsembleStrategy getStrategy() {
        return strategy;
    }

    /**
     * 获取分类器权重 / Get classifier weights
     *
     * @return 分类器权重向量 / Classifier weights vector
     */
    public IVector getClassifierWeights() {
        return classifierWeights;
    }

    /**
     * 获取各分类器准确率 / Get classifier accuracies
     *
     * @return 分类器准确率映射 / Classifier accuracies map
     */
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
            sb.append(String.format("%.3f", classifierWeights.get(i)));
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
