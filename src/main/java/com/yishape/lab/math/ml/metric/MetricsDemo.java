package com.yishape.lab.math.ml.metric;

import com.yishape.lab.math.linalg.IMatrix;

/**
 * 机器学习评估指标演示
 * 展示如何使用ClassificationMetrics和CrossValidation工具类
 * 
 * @author yishape
 * @version 1.0
 */
public class MetricsDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 机器学习评估指标工具演示 ===\n");
        
        // 演示ClassificationMetrics的使用
        demonstrateClassificationMetrics();
        
        // 演示CrossValidation的使用
        demonstrateCrossValidation();
    }
    
    /**
     * 演示ClassificationMetrics的使用
     */
    private static void demonstrateClassificationMetrics() {
        System.out.println("=== ClassificationMetrics 演示 ===");
        
        // 示例1: 二分类评估
        System.out.println("\n--- 示例1: 二分类评估 ---");
        String[] yTrue = {"cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"};
        String[] yPred = {"cat", "dog", "dog", "dog", "cat", "cat", "cat", "dog"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        System.out.println("分类报告:");
        System.out.println(metrics.getClassificationReport());
        System.out.println("\n混淆矩阵:");
        System.out.println(metrics.getConfusionMatrixString());
        
        // 示例2: 带概率的二分类评估（包含AUC）
        System.out.println("\n--- 示例2: 带AUC的二分类评估 ---");
        String[] yTrue2 = {"healthy", "sick", "sick", "healthy", "sick", "healthy", "sick", "healthy"};
        String[] yPred2 = {"healthy", "sick", "healthy", "healthy", "sick", "sick", "sick", "healthy"};
        double[] yProb = {0.1, 0.9, 0.6, 0.2, 0.8, 0.7, 0.95, 0.15};
        
        ClassificationMetrics metrics2 = ClassificationMetrics.compute(yTrue2, yPred2, yProb);
        System.out.println("分类报告 (包含AUC):");
        System.out.println(metrics2.getClassificationReport());
        
        // 示例3: 多分类评估
        System.out.println("\n--- 示例3: 多分类评估 ---");
        String[] yTrue3 = {"cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog", "bird"};
        String[] yPred3 = {"cat", "dog", "bird", "dog", "dog", "bird", "cat", "bird", "bird"};
        
        ClassificationMetrics metrics3 = ClassificationMetrics.compute(yTrue3, yPred3);
        System.out.println("分类报告:");
        System.out.println(metrics3.getClassificationReport());
        System.out.println("\n混淆矩阵:");
        System.out.println(metrics3.getConfusionMatrixString());
        
        // 显示每类别的详细指标
        System.out.println("每类别的详细指标:");
        metrics3.getPrecisionPerClass().forEach((label, precision) -> {
            System.out.printf("  %s: 精确率=%.4f, 召回率=%.4f, F1=%.4f, 支持数=%d\n",
                    label,
                    metrics3.getPrecisionPerClass().get(label),
                    metrics3.getRecallPerClass().get(label),
                    metrics3.getF1PerClass().get(label),
                    metrics3.getSupportPerClass().get(label));
        });
    }
    
    /**
     * 演示CrossValidation的使用
     */
    private static void demonstrateCrossValidation() {
        System.out.println("\n\n=== CrossValidation 演示 ===");
        
        // 创建模拟数据
        System.out.println("\n--- 创建模拟数据集 ---");
        int nSamples = 100;
        int nFeatures = 5;
        
        // 创建特征矩阵 (简化示例，实际应用中需要真实数据)
        double[][] data = new double[nSamples][nFeatures];
        String[] labels = new String[nSamples];
        
        // 生成随机数据和标签
        for (int i = 0; i < nSamples; i++) {
            for (int j = 0; j < nFeatures; j++) {
                data[i][j] = Math.random();
            }
            // 简单的标签生成逻辑
            labels[i] = (data[i][0] + data[i][1] > 1.0) ? "class_A" : "class_B";
        }
        
        IMatrix<Double> X = IMatrix.of(data);
        System.out.println("数据集创建完成:");
        System.out.println("  样本数: " + nSamples);
        System.out.println("  特征数: " + nFeatures);
        System.out.println("  类别分布: class_A=" + 
                java.util.Arrays.stream(labels).filter(l -> l.equals("class_A")).count() + 
                ", class_B=" + 
                java.util.Arrays.stream(labels).filter(l -> l.equals("class_B")).count());
        
        // 注意：这里只是演示API使用，实际使用需要真实的分类器
        System.out.println("\n--- 交叉验证方法演示 ---");
        
        System.out.println("1. K折交叉验证 (k=5):");
        System.out.println("   CrossValidation.kFoldCrossValidation(classifier, X, labels, 5)");
        
        System.out.println("\n2. 分层K折交叉验证 (k=5):");
        System.out.println("   CrossValidation.stratifiedKFoldCrossValidation(classifier, X, labels, 5)");
        
        System.out.println("\n3. 随机分割交叉验证 (测试集20%, 分割5次):");
        System.out.println("   CrossValidation.randomSplitCrossValidation(classifier, X, labels, 0.2, 5, 42)");
        
        System.out.println("\n4. 留一法交叉验证:");
        System.out.println("   CrossValidation.leaveOneOutCrossValidation(classifier, X, labels)");
        
        System.out.println("\n5. 多次K折交叉验证 (k=5, 重复3次):");
        System.out.println("   CrossValidation.repeatedKFoldCrossValidation(classifier, X, labels, 5, 3, 42)");
        
        // 显示预期的输出格式
        System.out.println("\n--- 交叉验证结果示例 ---");
        System.out.println("交叉验证结果将包含以下信息:");
        System.out.println("  • 每次折叠的准确率、F1分数、精确率、召回率");
        System.out.println("  • 平均值和标准差");
        System.out.println("  • 95%置信区间");
        System.out.println("  • 训练和预测时间统计");
        System.out.println("  • 详细的分类报告");
        System.out.println("  • 混淆矩阵");
        
        System.out.println("\n使用示例:");
        System.out.println("```java");
        System.out.println("// 创建分类器实例");
        System.out.println("IClassification classifier = new YourClassifier();");
        System.out.println("");
        System.out.println("// 执行5折交叉验证");
        System.out.println("CrossValidation.CrossValidationResult result = ");
        System.out.println("    CrossValidation.kFoldCrossValidation(classifier, X, labels, 5);");
        System.out.println("");
        System.out.println("// 获取详细报告");
        System.out.println("System.out.println(result.getDetailedReport());");
        System.out.println("```");
    }
}