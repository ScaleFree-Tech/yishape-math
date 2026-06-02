package com.yishape.lab.math.ml.metric;

public class ClassificationMetricsDemo {

    public static void main(String[] args) {
        System.out.println("=== ClassificationMetrics toString() 改进测试 ===\n");

        // 测试1: 二分类（带概率）
        System.out.println("测试1: 二分类（带概率）");
        String[] yTrue1 = {"0", "0", "1", "1", "1", "0", "1", "0", "1", "1"};
        String[] yPred1 = {"0", "0", "1", "1", "0", "0", "1", "1", "1", "1"};
        double[] yProb1 = {0.2, 0.3, 0.8, 0.9, 0.4, 0.1, 0.95, 0.6, 0.85, 0.9};

        ClassificationMetrics metrics1 = ClassificationMetrics.compute(yTrue1, yPred1, yProb1);
        System.out.println("toString(): " + metrics1.toString());
        System.out.println("AUC值: " + metrics1.getAuc());
        System.out.println("是否二分类: " + metrics1.isBinaryClassification());
        System.out.println();

        // 测试2: 二分类（不带概率）
        System.out.println("测试2: 二分类（不带概率）");
        String[] yTrue2 = {"0", "0", "1", "1", "1", "0", "1", "0", "1", "1"};
        String[] yPred2 = {"0", "0", "1", "1", "0", "0", "1", "1", "1", "1"};

        ClassificationMetrics metrics2 = ClassificationMetrics.compute(yTrue2, yPred2);
        System.out.println("toString(): " + metrics2.toString());
        System.out.println("AUC值: " + metrics2.getAuc());
        System.out.println("是否二分类: " + metrics2.isBinaryClassification());
        System.out.println();

        // 测试3: 多分类
        System.out.println("测试3: 多分类");
        String[] yTrue3 = {"A", "B", "C", "A", "B", "C", "A", "B", "C", "A"};
        String[] yPred3 = {"A", "B", "A", "A", "C", "C", "A", "B", "C", "B"};

        ClassificationMetrics metrics3 = ClassificationMetrics.compute(yTrue3, yPred3);
        System.out.println("toString(): " + metrics3.toString());
        System.out.println("AUC值: " + metrics3.getAuc());
        System.out.println("是否二分类: " + metrics3.isBinaryClassification());
        System.out.println();

        // 测试4: 展示详细报告
        System.out.println("=== 详细报告示例（二分类带概率）===");
        System.out.println(metrics1.getClassificationReport());
        System.out.println(metrics1.getConfusionMatrixString());
    }
}