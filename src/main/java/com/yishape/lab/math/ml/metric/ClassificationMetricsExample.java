package com.yishape.lab.math.ml.metric;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * ClassificationMetrics工具类使用示例
 * 演示如何使用ClassificationMetrics对分类器进行评估
 * 
 * @author yishape
 * @version 1.0
 */
public class ClassificationMetricsExample {

    public static void main(String[] args) {
        System.out.println("=== ClassificationMetrics 使用示例 ===\n");
        
        // 示例1: 二分类评估 (无概率)
        exampleBinaryClassificationWithoutProb();
        
        // 示例2: 二分类评估 (带概率，计算AUC)
        exampleBinaryClassificationWithProb();
        
        // 示例3: 多分类评估
        exampleMulticlassClassification();
        
        // 示例4: 完美分类和最差分类
        examplePerfectAndWorstClassification();
    }
    
    /**
     * 示例1: 二分类评估 (无概率)
     */
    private static void exampleBinaryClassificationWithoutProb() {
        System.out.println("=== 示例1: 二分类评估 (无概率) ===");
        
        // 模拟真实标签和预测标签
        String[] yTrue = {"spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham", "ham", "spam"};
        String[] yPred = {"spam", "ham", "ham", "ham", "spam", "ham", "spam", "ham", "spam", "spam"};
        
        // 计算评估指标
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        
        // 输出评估结果
        System.out.println("分类报告:");
        System.out.println(metrics.getClassificationReport());
        
        System.out.println("\n混淆矩阵:");
        System.out.println(metrics.getConfusionMatrixString());
        
        // 获取具体指标值
        System.out.printf("准确率: %.4f\n", metrics.getAccuracy());
        System.out.printf("宏平均F1分数: %.4f\n", metrics.getMacroF1());
        System.out.printf("加权平均F1分数: %.4f\n", metrics.getWeightedF1());
        System.out.println();
    }
    
    /**
     * 示例2: 二分类评估 (带概率，计算AUC)
     */
    private static void exampleBinaryClassificationWithProb() {
        System.out.println("=== 示例2: 二分类评估 (带概率，计算AUC) ===");
        
        // 模拟真实标签和预测标签 (疾病诊断)
        String[] yTrue = {"healthy", "sick", "sick", "healthy", "sick", "healthy", "sick", "healthy"};
        String[] yPred = {"healthy", "sick", "healthy", "healthy", "sick", "sick", "sick", "healthy"};
        
        // 模拟预测概率 (模型预测为"sick"的概率)
        double[] yProb = {0.1, 0.9, 0.6, 0.2, 0.8, 0.7, 0.95, 0.15};
        
        // 计算评估指标
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred, yProb);
        
        // 输出评估结果
        System.out.println("分类报告 (包含AUC):");
        System.out.println(metrics.getClassificationReport());
        
        System.out.println("\n混淆矩阵:");
        System.out.println(metrics.getConfusionMatrixString());
        
        // 获取AUC值
        if (metrics.isBinaryClassification() && metrics.getAuc() >= 0) {
            System.out.printf("AUC (ROC曲线下面积): %.4f\n", metrics.getAuc());
            
            if (metrics.getAuc() > 0.9) {
                System.out.println("AUC解释: 优秀的分类性能 (>0.9)");
            } else if (metrics.getAuc() > 0.8) {
                System.out.println("AUC解释: 良好的分类性能 (>0.8)");
            } else if (metrics.getAuc() > 0.7) {
                System.out.println("AUC解释: 一般的分类性能 (>0.7)");
            } else {
                System.out.println("AUC解释: 较差的分类性能 (≤0.7)");
            }
        }
        System.out.println();
    }
    
    /**
     * 示例3: 多分类评估
     */
    private static void exampleMulticlassClassification() {
        System.out.println("=== 示例3: 多分类评估 ===");
        
        // 模拟图像分类结果 (猫、狗、鸟)
        String[] yTrue = {"cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog", "bird"};
        String[] yPred = {"cat", "dog", "bird", "dog", "dog", "bird", "cat", "bird", "bird", "cat", "dog", "cat"};
        
        // 计算评估指标
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        
        // 输出评估结果
        System.out.println("分类报告:");
        System.out.println(metrics.getClassificationReport());
        
        System.out.println("\n混淆矩阵:");
        System.out.println(metrics.getConfusionMatrixString());
        
        // 显示每类别的详细指标
        System.out.println("每类别的详细指标:");
        metrics.getPrecisionPerClass().forEach((label, precision) -> {
            System.out.printf("  %s: 精确率=%.4f, 召回率=%.4f, F1=%.4f, 支持数=%d\n",
                    label,
                    metrics.getPrecisionPerClass().get(label),
                    metrics.getRecallPerClass().get(label),
                    metrics.getF1PerClass().get(label),
                    metrics.getSupportPerClass().get(label));
        });
        
        // 多分类不计算AUC
        System.out.printf("AUC: %.4f (多分类问题不计算AUC)\n", metrics.getAuc());
        System.out.println();
    }
    
    /**
     * 示例4: 完美分类和最差分类
     */
    private static void examplePerfectAndWorstClassification() {
        System.out.println("=== 示例4: 完美分类和最差分类 ===");
        
        // 完美分类
        System.out.println("--- 完美分类示例 ---");
        String[] yTrue = {"A", "B", "C", "A", "B", "C"};
        String[] yPred = {"A", "B", "C", "A", "B", "C"};
        
        ClassificationMetrics perfectMetrics = ClassificationMetrics.compute(yTrue, yPred);
        System.out.println(perfectMetrics.getClassificationReport());
        
        // 最差分类
        System.out.println("\n--- 最差分类示例 ---");
        String[] worstYPred = {"B", "C", "A", "C", "A", "B"};
        
        ClassificationMetrics worstMetrics = ClassificationMetrics.compute(yTrue, worstYPred);
        System.out.println(worstMetrics.getClassificationReport());
        
        // 性能对比
        System.out.println("\n--- 性能对比 ---");
        System.out.printf("完美分类准确率: %.4f\n", perfectMetrics.getAccuracy());
        System.out.printf("最差分类准确率: %.4f\n", worstMetrics.getAccuracy());
        System.out.printf("性能差距: %.4f\n", perfectMetrics.getAccuracy() - worstMetrics.getAccuracy());
        
        System.out.println();
    }
    
    /**
     * 附加功能示例: 自定义评估指标
     */
    public static void demonstrateAdvancedFeatures() {
        System.out.println("=== 高级功能演示 ===");
        
        // 创建测试数据
        String[] yTrue = {"cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog", "bird"};
        String[] yPred = {"cat", "dog", "bird", "dog", "dog", "bird", "cat", "bird", "bird"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        
        // 1. 获取原始数据
        System.out.println("1. 获取原始数据:");
        System.out.println("真实标签: " + Arrays.toString(yTrue));
        System.out.println("预测标签: " + Arrays.toString(yPred));
        System.out.println("类别标签: " + Arrays.toString(metrics.getClassLabels()));
        
        // 2. 获取混淆矩阵进行自定义分析
        System.out.println("\n2. 混淆矩阵分析:");
        int[][] confusionMatrix = metrics.getConfusionMatrix();
        for (int i = 0; i < confusionMatrix.length; i++) {
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                System.out.printf("%4d ", confusionMatrix[i][j]);
            }
            System.out.println();
        }
        
        // 3. 自定义指标计算
        System.out.println("\n3. 自定义指标:");
        double accuracy = metrics.getAccuracy();
        double macroF1 = metrics.getMacroF1();
        double weightedF1 = metrics.getWeightedF1();
        
        // 计算平衡准确率 (对类别不平衡不敏感)
        double balancedAccuracy = calculateBalancedAccuracy(yTrue, yPred);
        System.out.printf("准确率 (Accuracy): %.4f\n", accuracy);
        System.out.printf("平衡准确率: %.4f\n", balancedAccuracy);
        System.out.printf("宏平均F1: %.4f\n", macroF1);
        System.out.printf("加权平均F1: %.4f\n", weightedF1);
        
        // 4. 性能诊断
        System.out.println("\n4. 性能诊断:");
        diagnoseClassificationPerformance(metrics);
    }
    
    /**
     * 计算平衡准确率
     */
    private static double calculateBalancedAccuracy(String[] yTrue, String[] yPred) {
        Map<String, int[]> classStats = new HashMap<>();
        
        // 统计每类别的TP, TN, FP, FN
        for (int i = 0; i < yTrue.length; i++) {
            String trueLabel = yTrue[i];
            String predLabel = yPred[i];
            
            classStats.putIfAbsent(trueLabel, new int[4]); // TP, TN, FP, FN
            
            int[] stats = classStats.get(trueLabel);
            if (trueLabel.equals(predLabel)) {
                stats[0]++; // TP
                // 减少其他类别的TN
                for (Map.Entry<String, int[]> entry : classStats.entrySet()) {
                    if (!entry.getKey().equals(trueLabel)) {
                        entry.getValue()[1]++; // TN
                    }
                }
            } else {
                stats[3]++; // FN
                // 增加对应预测类别的FP
                classStats.putIfAbsent(predLabel, new int[4]);
                classStats.get(predLabel)[2]++; // FP
            }
        }
        
        // 计算每类别的召回率并平均
        double sumRecall = 0.0;
        int numClasses = classStats.size();
        
        for (int[] stats : classStats.values()) {
            int tp = stats[0];
            int fn = stats[3];
            double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0.0;
            sumRecall += recall;
        }
        
        return numClasses > 0 ? sumRecall / numClasses : 0.0;
    }
    
    /**
     * 性能诊断
     */
    private static void diagnoseClassificationPerformance(ClassificationMetrics metrics) {
        double accuracy = metrics.getAccuracy();
        double macroF1 = metrics.getMacroF1();
        double weightedF1 = metrics.getWeightedF1();
        
        // 准确率分析
        if (accuracy > 0.9) {
            System.out.println("✓ 准确率很高，分类器表现优秀");
        } else if (accuracy > 0.8) {
            System.out.println("△ 准确率良好，但仍有改进空间");
        } else {
            System.out.println("✗ 准确率较低，需要优化模型");
        }
        
        // F1分数分析
        if (Math.abs(macroF1 - weightedF1) > 0.1) {
            System.out.println("⚠ 宏平均和加权平均F1差异较大，可能存在类别不平衡问题");
        } else {
            System.out.println("✓ 宏平均和加权平均F1差异较小，类别分布相对平衡");
        }
        
        // 类别不平衡分析
        Map<String, Integer> support = metrics.getSupportPerClass();
        long maxSupport = support.values().stream().mapToLong(Integer::longValue).max().orElse(0);
        long minSupport = support.values().stream().mapToLong(Integer::longValue).min().orElse(0);
        
        if ((double) maxSupport / minSupport > 3) {
            System.out.println("⚠ 存在明显的类别不平衡，建议使用加权指标或采样技术");
        }
    }
}