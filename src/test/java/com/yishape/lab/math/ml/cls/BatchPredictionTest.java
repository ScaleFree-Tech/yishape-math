package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;

public class BatchPredictionTest {

    public static void main(String[] args) {
        System.out.println("=== 批量预测概率功能测试 ===\n");

        // 测试1: 二分类批量预测
        System.out.println("测试1: 二分类批量预测");
        testBinaryClassification();

        // 测试2: 多分类批量预测
        System.out.println("\n测试2: 多分类批量预测");
        testMulticlassClassification();

        // 测试3: 使用ClassificationMetrics计算指标
        System.out.println("\n测试3: 使用ClassificationMetrics计算指标");
        testMetricsWithBatchPrediction();
    }

    private static void testBinaryClassification() {
        // 创建二分类训练数据
        double[][] trainData = {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
            {5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0}
        };
        String[] trainLabels = {"0", "0", "0", "0", "1", "1", "1", "1"};

        IMatrix trainFeatures = Linalg.matrix(trainData);

        // 训练模型
        RereLogisticRegression model = new RereLogisticRegression(0.01, 1000, 1e-6, 0.0, 0.0);
        model.fit(trainFeatures, trainLabels);

        // 创建测试数据
        double[][] testData = {
            {1.5, 2.5}, {3.5, 4.5}, {5.5, 6.5}, {7.5, 8.5}
        };
        IMatrix testFeatures = Linalg.matrix(testData);

        // 使用新的批量预测方法
        RereLogisticRegression.BatchPredictionResult result = model.predictBatchWithProbabilities(testFeatures);

        System.out.println("预测结果: " + result);
        System.out.println("预测标签: " + java.util.Arrays.toString(result.getPredictions()));
        System.out.println("预测概率: " + java.util.Arrays.toString(result.getProbabilities()));

        // 验证概率范围
        double[] probabilities = result.getProbabilities();
        for (int i = 0; i < probabilities.length; i++) {
            System.out.printf("样本%d: 概率=%.4f, 预测=%s\n", i, probabilities[i], result.getPredictions()[i]);
        }
    }

    private static void testMulticlassClassification() {
        // 创建多分类训练数据
        double[][] trainData = {
            {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0},
            {5.0, 1.0}, {6.0, 2.0}, {7.0, 3.0}, {8.0, 4.0},
            {1.0, 5.0}, {2.0, 6.0}, {3.0, 7.0}, {4.0, 8.0}
        };
        String[] trainLabels = {"A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"};

        IMatrix trainFeatures = Linalg.matrix(trainData);

        // 训练模型
        RereLogisticRegression model = new RereLogisticRegression(0.01, 1000, 1e-6, 0.0, 0.0);
        model.fit(trainFeatures, trainLabels);

        // 创建测试数据
        double[][] testData = {
            {1.5, 1.5}, {5.5, 1.5}, {1.5, 5.5}
        };
        IMatrix testFeatures = Linalg.matrix(testData);

        // 使用新的批量预测方法
        RereLogisticRegression.BatchPredictionResult result = model.predictBatchWithProbabilities(testFeatures);

        System.out.println("预测结果: " + result);
        System.out.println("预测标签: " + java.util.Arrays.toString(result.getPredictions()));

        // 显示每个样本的详细概率
        double[][] classProbabilities = result.getClassProbabilities();
        String[] labels = model.getLabelMapping().keySet().toArray(new String[0]);
        java.util.Arrays.sort(labels);

        for (int i = 0; i < classProbabilities.length; i++) {
            System.out.printf("样本%d: 预测=%s\n", i, result.getPredictions()[i]);
            for (int j = 0; j < classProbabilities[i].length; j++) {
                System.out.printf("  %s: %.4f\n", labels[j], classProbabilities[i][j]);
            }
        }
    }

    private static void testMetricsWithBatchPrediction() {
        // 创建二分类训练数据
        double[][] trainData = {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
            {5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0}
        };
        String[] trainLabels = {"0", "0", "0", "0", "1", "1", "1", "1"};

        IMatrix trainFeatures = Linalg.matrix(trainData);

        // 训练模型
        RereLogisticRegression model = new RereLogisticRegression(0.01, 1000, 1e-6, 0.0, 0.0);
        model.fit(trainFeatures, trainLabels);

        // 创建测试数据和真实标签
        double[][] testData = {
            {1.5, 2.5}, {3.5, 4.5}, {5.5, 6.5}, {7.5, 8.5}
        };
        String[] trueLabels = {"0", "0", "1", "1"}; // 真实标签
        IMatrix testFeatures = Linalg.matrix(testData);

        // 使用新的批量预测方法
        RereLogisticRegression.BatchPredictionResult result = model.predictBatchWithProbabilities(testFeatures);

        System.out.println("批量预测结果: " + result);
        System.out.println("预测标签: " + java.util.Arrays.toString(result.getPredictions()));
        System.out.println("真实标签: " + java.util.Arrays.toString(trueLabels));

        // 使用ClassificationMetrics计算指标
        ClassificationMetrics metrics = ClassificationMetrics.compute(trueLabels, result);

        System.out.println("\n分类指标:");
        System.out.println(metrics.toString());
        System.out.println("\n详细报告:");
        System.out.println(metrics.getClassificationReport());
        System.out.println(metrics.getConfusionMatrixString());
    }
}