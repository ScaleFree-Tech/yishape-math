package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;

/**
 * 最终AUC修复验证测试
 * 
 * @author lteb2
 */
public class FinalAUCTest {

    public static void main(String args[]) {
        // 测试文件路径
        String path = "C:\\Users\\lteb2\\Downloads\\d9c2cb80-3944-4f82-b884-93cad3e586fc.csv";
        
        try {
            System.out.println("=== 最终AUC修复验证测试 ===");
            System.out.println("测试文件: " + path);
            
            // 读取数据
            var df = DataFrame.readCsv(path);
            var feature = df.sliceColumn(1, -1).toMatrix();
            var labels = df.get(df.getColumnCount()-1).toStringArray();
            
            System.out.println("\n数据集信息:");
            System.out.println("特征维度: " + feature.getColNum());
            System.out.println("样本数量: " + feature.getRowNum());
            
            // 检查标签
            java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
            for (String label : labels) {
                uniqueLabels.add(label);
            }
            System.out.println("唯一标签: " + uniqueLabels);
            
            if (uniqueLabels.size() != 2) {
                System.out.println("警告: 这不是二分类问题，AUC计算可能不适用");
                return;
            }
            
            // 训练模型
            System.out.println("\n训练模型...");
            var lr = new RereLogisticRegression(0.0, 0.1);
            var res = lr.fit(feature, labels);
            System.out.println("模型训练完成");
            
            // 获取预测结果
            System.out.println("\n获取预测结果...");
            var predicted = lr.predictBatchWithProbabilities(feature);
            
            // 检查预测结果
            String[] predLabels = predicted.getPredictions();
            double[] probabilities = predicted.getProbabilities();
            
            System.out.println("预测标签数量: " + predLabels.length);
            System.out.println("概率数组数量: " + probabilities.length);
            
            // 检查概率分布
            double minProb = Double.MAX_VALUE;
            double maxProb = Double.MIN_VALUE;
            for (double prob : probabilities) {
                if (prob < minProb) minProb = prob;
                if (prob > maxProb) maxProb = prob;
            }
            System.out.println("概率范围: [" + String.format("%.6f", minProb) + ", " + String.format("%.6f", maxProb) + "]");
            
            // 检查标签分布
            java.util.Map<String, Integer> labelCounts = new java.util.HashMap<>();
            for (String label : labels) {
                labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
            }
            System.out.println("标签分布: " + labelCounts);
            
            // 计算分类指标（使用修复后的ClassificationMetrics）
            System.out.println("\n计算分类指标...");
            ClassificationMetrics metrics = ClassificationMetrics.compute(labels, predicted);
            
            // 输出结果
            System.out.println("\n=== 最终结果 ===");
            System.out.println("准确率: " + String.format("%.4f", metrics.getAccuracy()));
            System.out.println("宏平均F1: " + String.format("%.4f", metrics.getMacroF1()));
            System.out.println("加权平均F1: " + String.format("%.4f", metrics.getWeightedF1()));
            System.out.println("AUC: " + String.format("%.4f", metrics.getAuc()));
            
            // 分析结果
            double auc = metrics.getAuc();
            if (auc == -1.0) {
                System.out.println("\n❌ AUC计算失败（返回-1）");
                System.out.println("可能原因: 缺少正类或负类样本");
            } else if (auc == 0.0) {
                System.out.println("\n❌ AUC仍为0，修复失败");
                System.out.println("需要进一步调查其他原因");
            } else if (auc < 0.5) {
                System.out.println("\n⚠️  AUC较低 (" + auc + ")，模型性能不佳");
                System.out.println("但至少不再是0，修复部分成功");
            } else if (auc >= 0.5 && auc < 0.7) {
                System.out.println("\n✅ AUC中等 (" + auc + ")，修复成功");
                System.out.println("模型有一定区分能力");
            } else if (auc >= 0.7 && auc < 0.9) {
                System.out.println("\n✅ AUC良好 (" + auc + ")，修复成功");
                System.out.println("模型有较好的区分能力");
            } else {
                System.out.println("\n🎉 AUC优秀 (" + auc + ")，修复成功");
                System.out.println("模型有很好的区分能力");
            }
            
            // 详细分析
            analyzeFinalResults(labels, probabilities, auc);
            
        } catch (Exception e) {
            System.out.println("\n❌ 测试过程中发生错误:");
            e.printStackTrace();
        }
    }
    
    /**
     * 详细分析最终结果
     */
    private static void analyzeFinalResults(String[] labels, double[] probabilities, double auc) {
        System.out.println("\n=== 详细分析 ===");
        
        // 获取唯一标签
        java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
        for (String label : labels) {
            uniqueLabels.add(label);
        }
        
        String[] labelsArray = uniqueLabels.toArray(new String[0]);
        String negativeLabel = labelsArray[0];
        String positiveLabel = labelsArray[1];
        
        System.out.println("负类标签: " + negativeLabel);
        System.out.println("正类标签: " + positiveLabel);
        
        // 分离正负类概率
        java.util.List<Double> negativeProbs = new java.util.ArrayList<>();
        java.util.List<Double> positiveProbs = new java.util.ArrayList<>();
        
        for (int i = 0; i < labels.length; i++) {
            if (labels[i].equals(negativeLabel)) {
                negativeProbs.add(probabilities[i]);
            } else {
                positiveProbs.add(probabilities[i]);
            }
        }
        
        // 计算统计信息
        double negMean = negativeProbs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double posMean = positiveProbs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double negMin = negativeProbs.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double negMax = negativeProbs.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        double posMin = positiveProbs.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double posMax = positiveProbs.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        
        System.out.println("负类统计 - 平均值: " + String.format("%.4f", negMean) + 
                          ", 范围: [" + String.format("%.4f", negMin) + ", " + String.format("%.4f", negMax) + "]");
        System.out.println("正类统计 - 平均值: " + String.format("%.4f", posMean) + 
                          ", 范围: [" + String.format("%.4f", posMin) + ", " + String.format("%.4f", posMax) + "]");
        
        System.out.println("正类平均概率 > 负类平均概率: " + (posMean > negMean));
        
        if (posMean < negMean) {
            System.out.println("⚠️  正类平均概率仍小于负类，这可能影响AUC值");
        }
        
        // 检查是否所有概率都相同
        boolean allSame = true;
        if (probabilities.length > 1) {
            double first = probabilities[0];
            for (int i = 1; i < probabilities.length; i++) {
                if (Math.abs(probabilities[i] - first) > 1e-10) {
                    allSame = false;
                    break;
                }
            }
        }
        System.out.println("所有概率是否相同: " + allSame);
        
        if (allSame) {
            System.out.println("⚠️  所有概率相同是AUC为0的另一个可能原因");
        }
        
        // 检查极端概率值
        int zeroCount = 0, oneCount = 0;
        for (double prob : probabilities) {
            if (prob == 0.0) zeroCount++;
            if (prob == 1.0) oneCount++;
        }
        System.out.println("概率为0的数量: " + zeroCount);
        System.out.println("概率为1的数量: " + oneCount);
        
        if (zeroCount + oneCount == probabilities.length) {
            System.out.println("⚠️  所有概率都是极端值，可能是数值稳定性问题");
        }
        
        System.out.println("\n=== 修复总结 ===");
        if (auc > 0.0) {
            System.out.println("✅ 修复成功！AUC从0提升到" + auc);
            System.out.println("修复机制: 检测到预测方向相反时，使用1-AUC进行修正");
        } else {
            System.out.println("❌ 修复失败，AUC仍为0");
            System.out.println("可能需要检查:");
            System.out.println("1. 数据质量问题");
            System.out.println("2. 模型训练问题");
            System.out.println("3. 其他数值稳定性问题");
        }
    }
}