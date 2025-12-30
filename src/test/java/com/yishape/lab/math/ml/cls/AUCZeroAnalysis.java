package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.data.DataFrame;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;

/**
 * AUC为0问题的深入分析
 * 
 * @author lteb2
 */
public class AUCZeroAnalysis {

    public static void main(String args[]) {
        // 测试文件路径
        String path = "C:\\Users\\lteb2\\Downloads\\d9c2cb80-3944-4f82-b884-93cad3e586fc.csv";
        
        try {
            System.out.println("=== AUC为0问题深入分析 ===");
            
            // 读取数据
            var df = DataFrame.readCsv(path);
            var feature = df.sliceColumn(1, -1).toMatrix();
            var labels = df.get(df.getColumnCount()-1).toStringArray();
            
            // 检查数据
            System.out.println("数据集信息:");
            System.out.println("特征维度: " + feature.getColNum());
            System.out.println("样本数量: " + feature.getRowNum());
            
            // 检查标签分布
            java.util.Map<String, Integer> labelCounts = new java.util.HashMap<>();
            for (String label : labels) {
                labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
            }
            System.out.println("标签分布: " + labelCounts);
            
            // 训练模型
            var lr = new RereLogisticRegression(0.0, 0.1);
            var res = lr.fit(feature, labels);
            System.out.println("模型训练完成");
            
            // 获取预测结果
            var predicted = lr.predictBatchWithProbabilities(feature);
            String[] predLabels = predicted.getPredictions();
            double[] probabilities = predicted.getProbabilities();
            
            // 分析AUC计算过程
            analyzeAUCComputation(labels, probabilities);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /**
     * 分析AUC计算过程
     */
    private static void analyzeAUCComputation(String[] yTrue, double[] yProb) {
        System.out.println("\n=== AUC计算过程分析 ===");
        
        // 1. 获取唯一标签并排序
        java.util.Set<String> uniqueLabels = new java.util.HashSet<>();
        uniqueLabels.addAll(java.util.Arrays.asList(yTrue));
        String[] classLabels = uniqueLabels.toArray(new String[0]);
        java.util.Arrays.sort(classLabels);
        
        System.out.println("类别标签: " + java.util.Arrays.toString(classLabels));
        
        if (classLabels.length != 2) {
            System.out.println("不是二分类问题，跳过AUC分析");
            return;
        }
        
        // 2. 确定正类和负类标签
        String positiveLabel = classLabels[1]; // 字典序较大的为正类
        String negativeLabel = classLabels[0];
        
        System.out.println("负类标签: " + negativeLabel);
        System.out.println("正类标签: " + positiveLabel);
        
        // 3. 准备AUC计算数据
        java.util.List<Double> probabilities = new java.util.ArrayList<>();
        java.util.List<Integer> labels = new java.util.ArrayList<>();
        
        for (int i = 0; i < yTrue.length; i++) {
            probabilities.add(yProb[i]);
            labels.add(yTrue[i].equals(positiveLabel) ? 1 : 0);
        }
        
        System.out.println("正类样本数: " + java.util.Collections.frequency(labels, 1));
        System.out.println("负类样本数: " + java.util.Collections.frequency(labels, 0));
        
        // 4. 按概率降序排序
        java.util.List<Integer> sortedIndices = new java.util.ArrayList<>();
        for (int i = 0; i < probabilities.size(); i++) {
            sortedIndices.add(i);
        }
        sortedIndices.sort((i, j) -> {
            int cmp = Double.compare(probabilities.get(j), probabilities.get(i));
            if (cmp == 0) {
                return Integer.compare(i, j);
            }
            return cmp;
        });
        
        // 5. 计算AUC (使用梯形法则)
        int tp = 0, fp = 0;
        int posCount = 0, negCount = 0;
        
        for (int label : labels) {
            if (label == 1) {
                posCount++;
            } else {
                negCount++;
            }
        }
        
        System.out.println("正类总数: " + posCount);
        System.out.println("负类总数: " + negCount);
        
        if (posCount == 0 || negCount == 0) {
            System.out.println("错误: 缺少正类或负类样本");
            return;
        }
        
        double auc = 0.0;
        int previousTP = 0;
        
        // 详细记录计算过程
        java.util.List<Double> tprList = new java.util.ArrayList<>();
        java.util.List<Double> fprList = new java.util.ArrayList<>();
        
        System.out.println("\nAUC计算详细过程:");
        System.out.println("排序索引\t概率值\t\t真实标签\tTP\tFP\tTPR\t\tFPR\t\t增量面积");
        
        for (int idx : sortedIndices) {
            double prob = probabilities.get(idx);
            int label = labels.get(idx);
            
            if (label == 1) {
                tp++;
            } else {
                fp++;
                // 使用梯形法则计算面积
                double currentTPR = (double) tp / posCount;
                double previousTPR = (double) previousTP / posCount;
                double deltaFPR = 1.0 / negCount;
                double areaIncrement = (previousTPR + currentTPR) * deltaFPR / 2;
                auc += areaIncrement;
                previousTP = tp;
                
                // 记录ROC点
                tprList.add(currentTPR);
                fprList.add((double) fp / negCount);
                
                System.out.printf("%d\t\t%.6f\t%d\t\t%d\t%d\t%.4f\t\t%.4f\t\t%.6f\n",
                    idx, prob, label, tp, fp, currentTPR, (double) fp / negCount, areaIncrement);
            }
        }
        
        System.out.println("\n最终AUC值: " + auc);
        
        // 6. 分析AUC为0的可能原因
        if (auc == 0.0) {
            System.out.println("\n=== AUC为0的原因分析 ===");
            
            // 检查是否所有正类都在负类之前（完美分类的反面）
            int lastPositiveIndex = -1;
            int lastNegativeIndex = -1;
            
            for (int i = 0; i < sortedIndices.size(); i++) {
                int idx = sortedIndices.get(i);
                if (labels.get(idx) == 1) {
                    lastPositiveIndex = i;
                } else {
                    lastNegativeIndex = i;
                }
            }
            
            System.out.println("最后正类排序位置: " + lastPositiveIndex);
            System.out.println("最后负类排序位置: " + lastNegativeIndex);
            
            if (lastPositiveIndex < lastNegativeIndex && lastPositiveIndex != -1) {
                System.out.println("原因1: 所有正类样本的概率都小于所有负类样本的概率");
                System.out.println("这表明模型的预测方向完全相反");
                
                // 检查概率分布
                double[] posProbs = new double[posCount];
                double[] negProbs = new double[negCount];
                int posIdx = 0, negIdx = 0;
                
                for (int i = 0; i < labels.size(); i++) {
                    if (labels.get(i) == 1) {
                        posProbs[posIdx++] = probabilities.get(i);
                    } else {
                        negProbs[negIdx++] = probabilities.get(i);
                    }
                }
                
                double maxPosProb = java.util.Arrays.stream(posProbs).max().orElse(0.0);
                double minNegProb = java.util.Arrays.stream(negProbs).min().orElse(1.0);
                
                System.out.println("正类最大概率: " + maxPosProb);
                System.out.println("负类最小概率: " + minNegProb);
                System.out.println("是否所有正类概率 < 所有负类概率: " + (maxPosProb < minNegProb));
            }
            
            // 检查是否所有概率都相同
            boolean allSame = true;
            double firstProb = probabilities.get(0);
            for (double prob : probabilities) {
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
            for (double prob : probabilities) {
                if (prob == 0.0 || prob == 1.0) {
                    extremeCount++;
                }
            }
            
            if (extremeCount == probabilities.size()) {
                System.out.println("原因3: 所有概率都是极端值(0或1)，可能是数值稳定性问题");
            }
        }
        
        // 7. 检查标签映射是否正确
        System.out.println("\n=== 标签映射检查 ===");
        System.out.println("在ClassificationMetrics中，正类被定义为: " + positiveLabel);
        System.out.println("负类被定义为: " + negativeLabel);
        
        // 检查这是否合理
        double posMeanProb = 0.0;
        double negMeanProb = 0.0;
        int posCount2 = 0, negCount2 = 0;
        
        for (int i = 0; i < labels.size(); i++) {
            if (labels.get(i) == 1) {
                posMeanProb += probabilities.get(i);
                posCount2++;
            } else {
                negMeanProb += probabilities.get(i);
                negCount2++;
            }
        }
        
        posMeanProb /= posCount2;
        negMeanProb /= negCount2;
        
        System.out.println("正类平均概率: " + posMeanProb);
        System.out.println("负类平均概率: " + negMeanProb);
        
        if (posMeanProb < negMeanProb) {
            System.out.println("警告: 正类平均概率小于负类平均概率，这可能导致AUC接近0");
            System.out.println("可能的原因:");
            System.out.println("1. 标签映射不正确（正负类定义颠倒）");
            System.out.println("2. 模型训练有问题，预测方向相反");
            System.out.println("3. 数据预处理有问题");
        }
    }
}