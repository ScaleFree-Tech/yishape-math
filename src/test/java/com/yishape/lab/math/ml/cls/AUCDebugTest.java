package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import com.yishape.lab.math.ml.metric.CrossValidation;

/**
 * AUC调试测试类
 * 用于调查二分类AUC为0的问题
 * 
 * @author lteb2
 */
public class AUCDebugTest {

    public static void main(String args[]) {
        // 测试文件路径
        String path = "C:\\Users\\lteb2\\Downloads\\d9c2cb80-3944-4f82-b884-93cad3e586fc.csv";
        
        try {
            System.out.println("=== AUC调试测试开始 ===");
            
            // 读取数据
            var df = DataFrame.readCsv(path);
            var feature = df.sliceColumn(1, -1).toMatrix();
            var labels = df.get(df.getColumnCount()-1).toStringArray();
            
            System.out.println("数据集信息:");
            System.out.println("特征维度: " + feature.getColNum());
            System.out.println("样本数量: " + feature.getRowNum());
            System.out.println("标签类型: " + getLabelInfo(labels));
            
            // 检查标签分布
            checkLabelDistribution(labels);
            
            // 训练模型
            var lr = new RereLogisticRegression(0.0, 0.1);
            var res = lr.fit(feature, labels);
            System.out.println("模型训练完成");
            
            // 获取预测结果
            var predicted = lr.predictBatchWithProbabilities(feature);
            
            // 检查预测结果
            checkPredictionResults(labels, predicted);
            
            // 计算分类指标
            ClassificationMetrics metrics = ClassificationMetrics.compute(labels, predicted);
            System.out.println("分类指标计算完成");
            System.out.println("AUC值: " + metrics.getAuc());
            
            // 详细分析AUC计算过程
            analyzeAUCCalculation(labels, predicted);
            
            // 交叉验证
            var cvResult = CrossValidation.kFoldCrossValidation(lr, feature, labels, 3);
            System.out.println("交叉验证结果: " + cvResult);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /**
     * 获取标签信息
     */
    private static String getLabelInfo(String[] labels) {
        java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
        for (String label : labels) {
            uniqueLabels.add(label);
        }
        return String.join(", ", uniqueLabels);
    }
    
    /**
     * 检查标签分布
     */
    private static void checkLabelDistribution(String[] labels) {
        System.out.println("\n=== 标签分布分析 ===");
        
        java.util.Map<String, Integer> labelCounts = new java.util.HashMap<>();
        for (String label : labels) {
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }
        
        int total = labels.length;
        for (String label : labelCounts.keySet()) {
            int count = labelCounts.get(label);
            double percentage = (double) count / total * 100;
            System.out.println("标签 '" + label + "': " + count + " (" + String.format("%.2f%%", percentage) + ")");
        }
        
        // 检查是否为二分类
        if (labelCounts.size() != 2) {
            System.out.println("警告: 这不是二分类问题，AUC计算可能不适用");
        }
    }
    
    /**
     * 检查预测结果
     */
    private static void checkPredictionResults(String[] trueLabels, BatchPredictionResult predicted) {
        System.out.println("\n=== 预测结果分析 ===");
        
        String[] predLabels = predicted.getPredictions();
        double[] probabilities = predicted.getProbabilities();
        
        System.out.println("预测标签数量: " + predLabels.length);
        System.out.println("概率数组数量: " + probabilities.length);
        
        // 检查概率分布
        double minProb = Double.MAX_VALUE;
        double maxProb = Double.MIN_VALUE;
        int zeroCount = 0;
        int oneCount = 0;
        
        for (double prob : probabilities) {
            if (prob < minProb) minProb = prob;
            if (prob > maxProb) maxProb = prob;
            if (prob == 0.0) zeroCount++;
            if (prob == 1.0) oneCount++;
        }
        
        System.out.println("概率范围: [" + String.format("%.6f", minProb) + ", " + String.format("%.6f", maxProb) + "]");
        System.out.println("概率为0的数量: " + zeroCount);
        System.out.println("概率为1的数量: " + oneCount);
        
        // 检查预测准确性
        int correct = 0;
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i].equals(predLabels[i])) {
                correct++;
            }
        }
        double accuracy = (double) correct / trueLabels.length;
        System.out.println("预测准确率: " + String.format("%.4f", accuracy));
        
        // 检查概率与真实标签的关系
        checkProbabilityLabelRelationship(trueLabels, probabilities);
    }
    
    /**
     * 检查概率与真实标签的关系
     */
    private static void checkProbabilityLabelRelationship(String[] trueLabels, double[] probabilities) {
        System.out.println("\n=== 概率与标签关系分析 ===");
        
        // 获取唯一标签
        java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
        for (String label : trueLabels) {
            uniqueLabels.add(label);
        }
        
        String[] labelsArray = uniqueLabels.toArray(new String[0]);
        if (labelsArray.length != 2) {
            System.out.println("不是二分类问题");
            return;
        }
        
        // 假设第一个标签为负类，第二个为正类
        String negativeLabel = labelsArray[0];
        String positiveLabel = labelsArray[1];
        
        java.util.List<Double> negativeProbs = new java.util.ArrayList<>();
        java.util.List<Double> positiveProbs = new java.util.ArrayList<>();
        
        for (int i = 0; i < trueLabels.length; i++) {
            if (trueLabels[i].equals(negativeLabel)) {
                negativeProbs.add(probabilities[i]);
            } else if (trueLabels[i].equals(positiveLabel)) {
                positiveProbs.add(probabilities[i]);
            }
        }
        
        // 计算负类和正类的概率统计
        double negMean = negativeProbs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double posMean = positiveProbs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        
        System.out.println("负类样本数: " + negativeProbs.size() + ", 平均概率: " + String.format("%.4f", negMean));
        System.out.println("正类样本数: " + positiveProbs.size() + ", 平均概率: " + String.format("%.4f", posMean));
        
        // 检查是否出现概率倒置（负类概率 > 正类概率）
        int invertedCount = 0;
        for (double negProb : negativeProbs) {
            for (double posProb : positiveProbs) {
                if (negProb > posProb) {
                    invertedCount++;
                }
            }
        }
        
        int totalPairs = negativeProbs.size() * positiveProbs.size();
        System.out.println("概率倒置对数: " + invertedCount + " / " + totalPairs);
        System.out.println("倒置比例: " + String.format("%.4f", (double) invertedCount / totalPairs));
    }
    
    /**
     * 详细分析AUC计算过程
     */
    private static void analyzeAUCCalculation(String[] trueLabels, BatchPredictionResult predicted) {
        System.out.println("\n=== AUC计算过程分析 ===");
        
        double[] probabilities = predicted.getProbabilities();
        
        // 获取唯一标签并排序
        java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
        uniqueLabels.addAll(java.util.Arrays.asList(trueLabels));
        String[] classLabels = uniqueLabels.toArray(new String[0]);
        java.util.Arrays.sort(classLabels);
        
        if (classLabels.length != 2) {
            System.out.println("不是二分类问题，跳过AUC分析");
            return;
        }
        
        // 确定正类和负类
        String positiveLabel = classLabels[1]; // 假设字典序较大的为正类
        String negativeLabel = classLabels[0];
        
        System.out.println("负类标签: " + negativeLabel);
        System.out.println("正类标签: " + positiveLabel);
        
        // 准备AUC计算数据
        java.util.List<Double> probList = new java.util.ArrayList<>();
        java.util.List<Integer> labelList = new java.util.ArrayList<>();
        
        for (int i = 0; i < trueLabels.length; i++) {
            probList.add(probabilities[i]);
            labelList.add(trueLabels[i].equals(positiveLabel) ? 1 : 0);
        }
        
        // 按概率降序排序
        java.util.List<Integer> sortedIndices = new java.util.ArrayList<>();
        for (int i = 0; i < probList.size(); i++) {
            sortedIndices.add(i);
        }
        sortedIndices.sort((i, j) -> {
            int cmp = Double.compare(probList.get(j), probList.get(i));
            if (cmp == 0) {
                return Integer.compare(i, j);
            }
            return cmp;
        });
        
        // 计算AUC
        int tp = 0, fp = 0;
        int posCount = 0, negCount = 0;
        
        for (int label : labelList) {
            if (label == 1) {
                posCount++;
            } else {
                negCount++;
            }
        }
        
        System.out.println("正类样本数: " + posCount);
        System.out.println("负类样本数: " + negCount);
        
        if (posCount == 0 || negCount == 0) {
            System.out.println("错误: 缺少正类或负类样本，无法计算AUC");
            return;
        }
        
        double auc = 0.0;
        int previousTP = 0;
        java.util.List<Double> tprList = new java.util.ArrayList<>();
        java.util.List<Double> fprList = new java.util.ArrayList<>();
        
        for (int idx : sortedIndices) {
            if (labelList.get(idx) == 1) {
                tp++;
            } else {
                fp++;
                double currentTPR = (double) tp / posCount;
                double previousTPR = (double) previousTP / posCount;
                double deltaFPR = 1.0 / negCount;
                auc += (previousTPR + currentTPR) * deltaFPR / 2;
                previousTP = tp;
                
                // 记录ROC点
                tprList.add(currentTPR);
                fprList.add((double) fp / negCount);
            }
        }
        
        System.out.println("计算得到的AUC: " + String.format("%.6f", auc));
        
        // 检查AUC为0的可能原因
        if (auc == 0.0) {
            System.out.println("\n=== AUC为0的可能原因分析 ===");
            
            // 检查是否所有正类都在负类之前（完美分类的反面）
            int lastPositiveIndex = -1;
            int lastNegativeIndex = -1;
            
            for (int i = 0; i < sortedIndices.size(); i++) {
                int idx = sortedIndices.get(i);
                if (labelList.get(idx) == 1) {
                    lastPositiveIndex = i;
                } else {
                    lastNegativeIndex = i;
                }
            }
            
            if (lastPositiveIndex < lastNegativeIndex && lastPositiveIndex != -1) {
                System.out.println("原因1: 所有正类样本的概率都小于所有负类样本的概率");
                System.out.println("这表明模型的预测方向完全相反");
            }
            
            // 检查是否所有概率都相同
            boolean allSame = true;
            double firstProb = probList.get(0);
            for (double prob : probList) {
                if (Math.abs(prob - firstProb) > 1e-10) {
                    allSame = false;
                    break;
                }
            }
            
            if (allSame) {
                System.out.println("原因2: 所有样本的概率都相同，无法区分正负类");
            }
            
            // 检查是否只有极端概率值
            int extremeCount = 0;
            for (double prob : probList) {
                if (prob == 0.0 || prob == 1.0) {
                    extremeCount++;
                }
            }
            
            if (extremeCount == probList.size()) {
                System.out.println("原因3: 所有概率都是极端值(0或1)，可能是数值稳定性问题");
            }
        }
    }
}