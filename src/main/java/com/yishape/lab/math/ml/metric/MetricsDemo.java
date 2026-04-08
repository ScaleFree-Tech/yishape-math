package com.yishape.lab.math.ml.metric;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.yishape.lab.math.linalg.IMatrix;

/**
 * 机器学习评估指标演示
 * 展示如何使用ClassificationMetrics和CrossValidation工具类
 * 
 * @author yishape
 * @version 1.0
 */
public class MetricsDemo {

    private static final Logger log = LoggerFactory.getLogger(MetricsDemo.class);

    
    public static void main(String[] args) {
        log.debug("=== 机器学习评估指标工具演示 ===\n");
        
        // 演示ClassificationMetrics的使用
        demonstrateClassificationMetrics();
        
        // 演示CrossValidation的使用
        demonstrateCrossValidation();
    }
    
    /**
     * 演示ClassificationMetrics的使用
     */
    private static void demonstrateClassificationMetrics() {
        log.debug("=== ClassificationMetrics 演示 ===");
        
        // 示例1: 二分类评估
        log.debug("\n--- 示例1: 二分类评估 ---");
        String[] yTrue = {"cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"};
        String[] yPred = {"cat", "dog", "dog", "dog", "cat", "cat", "cat", "dog"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        log.debug("分类报告:");
        log.debug(metrics.getClassificationReport());
        log.debug("\n混淆矩阵:");
        log.debug(metrics.getConfusionMatrixString());
        
        // 示例2: 带概率的二分类评估（包含AUC）
        log.debug("\n--- 示例2: 带AUC的二分类评估 ---");
        String[] yTrue2 = {"healthy", "sick", "sick", "healthy", "sick", "healthy", "sick", "healthy"};
        String[] yPred2 = {"healthy", "sick", "healthy", "healthy", "sick", "sick", "sick", "healthy"};
        double[] yProb = {0.1, 0.9, 0.6, 0.2, 0.8, 0.7, 0.95, 0.15};
        
        ClassificationMetrics metrics2 = ClassificationMetrics.compute(yTrue2, yPred2, yProb);
        log.debug("分类报告 (包含AUC):");
        log.debug(metrics2.getClassificationReport());
        
        // 示例3: 多分类评估
        log.debug("\n--- 示例3: 多分类评估 ---");
        String[] yTrue3 = {"cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog", "bird"};
        String[] yPred3 = {"cat", "dog", "bird", "dog", "dog", "bird", "cat", "bird", "bird"};
        
        ClassificationMetrics metrics3 = ClassificationMetrics.compute(yTrue3, yPred3);
        log.debug("分类报告:");
        log.debug(metrics3.getClassificationReport());
        log.debug("\n混淆矩阵:");
        log.debug(metrics3.getConfusionMatrixString());
        
        // 显示每类别的详细指标
        log.debug("每类别的详细指标:");
        metrics3.getPrecisionPerClass().forEach((label, precision) -> {
            log.debug(String.format("  %s: 精确率=%.4f, 召回率=%.4f, F1=%.4f, 支持数=%d\n",
                    label,
                    metrics3.getPrecisionPerClass().get(label),
                    metrics3.getRecallPerClass().get(label),
                    metrics3.getF1PerClass().get(label),
                    metrics3.getSupportPerClass().get(label)));
        });
    }
    
    /**
     * 演示CrossValidation的使用
     */
    private static void demonstrateCrossValidation() {
        log.debug("\n\n=== CrossValidation 演示 ===");
        
        // 创建模拟数据
        log.debug("\n--- 创建模拟数据集 ---");
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
        log.debug("数据集创建完成:");
        log.debug("  样本数: " + nSamples);
        log.debug("  特征数: " + nFeatures);
        log.debug("  类别分布: class_A=" + 
                java.util.Arrays.stream(labels).filter(l -> l.equals("class_A")).count() + 
                ", class_B=" + 
                java.util.Arrays.stream(labels).filter(l -> l.equals("class_B")).count());
        
        // 注意：这里只是演示API使用，实际使用需要真实的分类器
        log.debug("\n--- 交叉验证方法演示 ---");
        
        log.debug("1. K折交叉验证 (k=5):");
        log.debug("   CrossValidation.kFoldCrossValidation(classifier, X, labels, 5)");
        
        log.debug("\n2. 分层K折交叉验证 (k=5):");
        log.debug("   CrossValidation.stratifiedKFoldCrossValidation(classifier, X, labels, 5)");
        
        log.debug("\n3. 随机分割交叉验证 (测试集20%, 分割5次):");
        log.debug("   CrossValidation.randomSplitCrossValidation(classifier, X, labels, 0.2, 5, 42)");
        
        log.debug("\n4. 留一法交叉验证:");
        log.debug("   CrossValidation.leaveOneOutCrossValidation(classifier, X, labels)");
        
        log.debug("\n5. 多次K折交叉验证 (k=5, 重复3次):");
        log.debug("   CrossValidation.repeatedKFoldCrossValidation(classifier, X, labels, 5, 3, 42)");
        
        // 显示预期的输出格式
        log.debug("\n--- 交叉验证结果示例 ---");
        log.debug("交叉验证结果将包含以下信息:");
        log.debug("  • 每次折叠的准确率、F1分数、精确率、召回率");
        log.debug("  • 平均值和标准差");
        log.debug("  • 95%置信区间");
        log.debug("  • 训练和预测时间统计");
        log.debug("  • 详细的分类报告");
        log.debug("  • 混淆矩阵");
        
        log.debug("\n使用示例:");
        log.debug("```java");
        log.debug("// 创建分类器实例");
        log.debug("IClassification classifier = new YourClassifier();");
        log.debug("");
        log.debug("// 执行5折交叉验证");
        log.debug("CrossValidation.CrossValidationResult result = ");
        log.debug("    CrossValidation.kFoldCrossValidation(classifier, X, labels, 5);");
        log.debug("");
        log.debug("// 获取详细报告");
        log.debug("log.debug(result.getDetailedReport());");
        log.debug("```");
    }
}