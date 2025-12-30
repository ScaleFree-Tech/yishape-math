package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.ml.metric.ClassificationMetricsFixed;

/**
 * AUC修复效果测试
 * 
 * @author lteb2
 */
public class AUCFixTest {

    public static void main(String args[]) {
        // 测试文件路径
        String path = "C:\\Users\\lteb2\\Downloads\\d9c2cb80-3944-4f82-b884-93cad3e586fc.csv";
        
        try {
            System.out.println("=== AUC修复效果测试 ===");
            
            // 读取数据
            var df = DataFrame.readCsv(path);
            var feature = df.sliceColumn(1, -1).toMatrix();
            var labels = df.get(df.getColumnCount()-1).toStringArray();
            
            System.out.println("数据集信息:");
            System.out.println("特征维度: " + feature.getColNum());
            System.out.println("样本数量: " + feature.getRowNum());
            
            // 训练模型
            var lr = new RereLogisticRegression(0.0, 0.1);
            var res = lr.fit(feature, labels);
            System.out.println("模型训练完成");
            
            // 获取预测结果
            var predicted = lr.predictBatchWithProbabilities(feature);
            
            // 使用原始ClassificationMetrics计算
            System.out.println("\n=== 原始ClassificationMetrics结果 ===");
            ClassificationMetrics originalMetrics = ClassificationMetrics.compute(labels, predicted);
            System.out.println("原始AUC: " + originalMetrics.getAuc());
            System.out.println("原始准确率: " + originalMetrics.getAccuracy());
            
            // 使用修复版ClassificationMetricsFixed计算
            System.out.println("\n=== 修复版ClassificationMetricsFixed结果 ===");
            ClassificationMetricsFixed fixedMetrics = ClassificationMetricsFixed.compute(labels, predicted);
            System.out.println("修复后AUC: " + fixedMetrics.getAuc());
            System.out.println("修复后准确率: " + fixedMetrics.getAccuracy());
            
            // 对比分析
            System.out.println("\n=== 对比分析 ===");
            double originalAUC = originalMetrics.getAuc();
            double fixedAUC = fixedMetrics.getAuc();
            
            if (originalAUC == 0.0 && fixedAUC > 0.0) {
                System.out.println("✅ 修复成功！AUC从0.0提升到" + fixedAUC);
            } else if (originalAUC == fixedAUC) {
                System.out.println("ℹ️  AUC值相同，可能原本就没有问题或问题类型不同");
            } else {
                System.out.println("⚠️  AUC值发生变化，需要进一步分析");
            }
            
            // 详细分析
            analyzeAUCDifference(labels, predicted, originalAUC, fixedAUC);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /**
     * 分析AUC差异
     */
    private static void analyzeAUCDifference(String[] labels, BatchPredictionResult predicted, 
                                           double originalAUC, double fixedAUC) {
        System.out.println("\n=== AUC差异详细分析 ===");
        
        double[] probabilities = predicted.getProbabilities();
        
        // 获取唯一标签
        java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
        for (String label : labels) {
            uniqueLabels.add(label);
        }
        
        String[] labelsArray = uniqueLabels.toArray(new String[0]);
        String negativeLabel = labelsArray[0];
        String positiveLabel = labelsArray[1];
        
        // 计算正负类概率统计
        java.util.List<Double> negativeProbs = new java.util.ArrayList<>();
        java.util.List<Double> positiveProbs = new java.util.ArrayList<>();
        
        for (int i = 0; i < labels.length; i++) {
            if (labels[i].equals(negativeLabel)) {
                negativeProbs.add(probabilities[i]);
            } else {
                positiveProbs.add(probabilities[i]);
            }
        }
        
        double posMean = positiveProbs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double negMean = negativeProbs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        System.out.println("正类平均概率: " + posMean);
        System.out.println("负类平均概率: " + negMean);
        System.out.println("正类平均概率 < 负类平均概率: " + (posMean < negMean));
        
        if (posMean < negMean) {
            System.out.println("发现: 正类平均概率小于负类平均概率");
            System.out.println("这解释了为什么原始AUC为0");
            System.out.println("修复版应该通过1-AUC来修正这个问题");
            
            double expectedFixedAUC = 1.0 - originalAUC;
            System.out.println("期望的修复后AUC: " + expectedFixedAUC);
            System.out.println("实际的修复后AUC: " + fixedAUC);
            System.out.println("是否匹配: " + (Math.abs(expectedFixedAUC - fixedAUC) < 1e-6));
        }
        
        // 检查概率分布
        double minProb = java.util.Arrays.stream(probabilities).min().orElse(0.0);
        double maxProb = java.util.Arrays.stream(probabilities).max().orElse(1.0);
        System.out.println("概率范围: [" + minProb + ", " + maxProb + "]");
        
        boolean allSame = java.util.Arrays.stream(probabilities)
            .allMatch(p -> Math.abs(p - probabilities[0]) < 1e-10);
        System.out.println("所有概率是否相同: " + allSame);
        
        if (allSame) {
            System.out.println("所有概率相同是AUC为0的另一个可能原因");
        }
    }
}