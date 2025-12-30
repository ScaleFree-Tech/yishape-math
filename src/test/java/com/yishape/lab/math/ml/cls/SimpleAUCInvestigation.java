package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;

/**
 * 简化的AUC问题调查
 * 
 * @author lteb2
 */
public class SimpleAUCInvestigation {

    public static void main(String args[]) {
        // 测试文件路径
        String path = "C:\\Users\\lteb2\\Downloads\\d9c2cb80-3944-4f82-b884-93cad3e586fc.csv";
        
        try {
            System.out.println("=== 简化AUC问题调查 ===");
            
            // 读取数据
            var df = DataFrame.readCsv(path);
            var feature = df.sliceColumn(1, -1).toMatrix();
            var labels = df.get(df.getColumnCount()-1).toStringArray();
            
            System.out.println("数据集信息:");
            System.out.println("特征维度: " + feature.getColNum());
            System.out.println("样本数量: " + feature.getRowNum());
            
            // 检查标签
            java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
            for (String label : labels) {
                uniqueLabels.add(label);
            }
            System.out.println("唯一标签: " + uniqueLabels);
            
            if (uniqueLabels.size() != 2) {
                System.out.println("警告: 这不是二分类问题");
                return;
            }
            
            // 训练模型
            var lr = new RereLogisticRegression(0.0, 0.1);
            var res = lr.fit(feature, labels);
            System.out.println("模型训练完成");
            
            // 获取预测结果
            var predicted = lr.predictBatchWithProbabilities(feature);
            
            // 检查预测结果
            String[] predLabels = predicted.getPredictions();
            double[] probabilities = predicted.getProbabilities();
            
            System.out.println("预测结果检查:");
            System.out.println("预测标签数量: " + predLabels.length);
            System.out.println("概率数组数量: " + probabilities.length);
            
            // 检查概率范围
            double minProb = Double.MAX_VALUE;
            double maxProb = Double.MIN_VALUE;
            for (double prob : probabilities) {
                if (prob < minProb) minProb = prob;
                if (prob > maxProb) maxProb = prob;
            }
            System.out.println("概率范围: [" + minProb + ", " + maxProb + "]");
            
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
            
            // 检查标签分布
            java.util.Map<String, Integer> labelCounts = new java.util.HashMap<>();
            for (String label : labels) {
                labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
            }
            System.out.println("标签分布: " + labelCounts);
            
            // 计算分类指标
            ClassificationMetrics metrics = ClassificationMetrics.compute(labels, predicted);
            System.out.println("AUC值: " + metrics.getAuc());
            
            // 如果AUC为0，进行详细分析
            if (metrics.getAuc() == 0.0) {
                System.out.println("\n=== AUC为0的详细分析 ===");
                analyzeAUCZeroCase(labels, probabilities);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /**
     * 分析AUC为0的情况
     */
    private static void analyzeAUCZeroCase(String[] trueLabels, double[] probabilities) {
        // 获取唯一标签
        java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
        for (String label : trueLabels) {
            uniqueLabels.add(label);
        }
        
        String[] labelsArray = uniqueLabels.toArray(new String[0]);
        String negativeLabel = labelsArray[0];
        String positiveLabel = labelsArray[1];
        
        // 分离正负类概率
        java.util.List<Double> negativeProbs = new java.util.ArrayList<>();
        java.util.List<Double> positiveProbs = new java.util.ArrayList<>();
        
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i].equals(negativeLabel)) {
                negativeProbs.add(probabilities[i]);
            } else {
                positiveProbs.add(probabilities[i]);
            }
        }
        
        // 计算统计信息
        double negMin = negativeProbs.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double negMax = negativeProbs.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        double posMin = positiveProbs.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double posMax = positiveProbs.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
        
        System.out.println("负类概率范围: [" + negMin + ", " + negMax + "]");
        System.out.println("正类概率范围: [" + posMin + ", " + posMax + "]");
        
        // 检查是否正类概率都小于负类概率
        boolean allPositiveLessThanNegative = true;
        for (double posProb : positiveProbs) {
            for (double negProb : negativeProbs) {
                if (posProb >= negProb) {
                    allPositiveLessThanNegative = false;
                    break;
                }
            }
            if (!allPositiveLessThanNegative) break;
        }
        
        if (allPositiveLessThanNegative) {
            System.out.println("发现: 所有正类样本的概率都小于所有负类样本的概率");
            System.out.println("这表明模型的预测方向完全相反");
        }
        
        // 检查是否所有概率都为0或1
        boolean allExtreme = true;
        for (double prob : probabilities) {
            if (prob != 0.0 && prob != 1.0) {
                allExtreme = false;
                break;
            }
        }
        
        if (allExtreme) {
            System.out.println("发现: 所有概率都是极端值(0或1)");
            System.out.println("可能是数值稳定性或模型训练问题");
        }
    }
}