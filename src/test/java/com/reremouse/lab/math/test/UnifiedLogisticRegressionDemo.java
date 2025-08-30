package com.reremouse.lab.math.test;

import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.RereMatrix;
import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.ml.cls.LogisticRegressionResult;
import com.reremouse.lab.math.ml.cls.RereLogisticRegression;
import java.util.Map;

/**
 * 统一逻辑回归演示类
 * <p>
 * 本类演示了改进后的RereLogisticRegression类的功能，
 * 展示其既适用于二分类问题又适用于多分类问题的能力。
 * </p>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class UnifiedLogisticRegressionDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 统一逻辑回归演示 ===\n");
        
        // 测试二分类功能
        testBinaryClassification();
        System.out.println();
        
        // 测试多分类功能
        testMulticlassClassification();
        System.out.println();
        
        // 测试不同正则化方法
        testRegularizationMethods();
    }
    
    /**
     * 测试二分类功能
     */
    private static void testBinaryClassification() {
        System.out.println("=== 二分类测试 ===\n");
        
        // 创建二分类训练数据
        float[][] featureData = {
            // 正类样本 (标签: "正")
            {1.0f, 2.0f, 3.0f},  // 样本1
            {2.0f, 3.0f, 4.0f},  // 样本2
            {3.0f, 4.0f, 5.0f},  // 样本3
            {4.0f, 5.0f, 6.0f},  // 样本4
            {5.0f, 6.0f, 7.0f},  // 样本5
            
            // 负类样本 (标签: "负")
            {-1.0f, -2.0f, -3.0f},  // 样本6
            {-2.0f, -3.0f, -4.0f},  // 样本7
            {-3.0f, -4.0f, -5.0f},  // 样本8
            {-4.0f, -5.0f, -6.0f},  // 样本9
            {-5.0f, -6.0f, -7.0f}   // 样本10
        };
        
        String[] labels = {
            "正", "正", "正", "正", "正",
            "负", "负", "负", "负", "负"
        };
        
        // 创建特征矩阵
        IMatrix featureMatrix = new RereMatrix(featureData);
        
        System.out.println("二分类训练数据：");
        System.out.println("特征矩阵形状: " + featureMatrix.getRowNum() + " x " + featureMatrix.getColNum());
        System.out.println("标签数量: " + labels.length);
        System.out.println("类别: 正、负");
        System.out.println();
        
        // 创建并训练二分类模型
        RereLogisticRegression lr = new RereLogisticRegression();
        lr.setLearningRate(0.01f);
        lr.setMaxIterations(1000);
        lr.setTolerance(1e-6f);
        lr.setRegularization(0.0f, 0.01f);  // 使用L2正则化
        
        try {
            long startTime = System.currentTimeMillis();
            LogisticRegressionResult result = lr.fit(featureMatrix, labels);
            long endTime = System.currentTimeMillis();
            
            System.out.println("二分类模型训练完成！");
            System.out.println("模型类型: " + lr.getModelTypeDescription());
            System.out.println("训练耗时: " + (endTime - startTime) + "ms");
            System.out.println("最终损失: " + result.getLoss());
            System.out.println("正则化: " + lr.getRegularizationDescription());
            System.out.println();
            
            // 评估二分类模型
            evaluateBinaryClassificationModel(lr, featureMatrix, labels);
            
        } catch (Exception e) {
            System.err.println("二分类训练失败：" + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试多分类功能
     */
    private static void testMulticlassClassification() {
        System.out.println("=== 多分类测试 ===\n");
        
        // 创建三分类训练数据
        float[][] featureData = {
            // A类样本
            {95, 92, 5.0f, 9.5f},  // 样本1
            {98, 95, 6.0f, 9.8f},  // 样本2
            {92, 88, 4.5f, 9.0f},  // 样本3
            {96, 90, 5.5f, 9.2f},  // 样本4
            {94, 89, 4.8f, 8.8f},  // 样本5
            
            // B类样本
            {85, 82, 3.5f, 7.5f},  // 样本6
            {88, 85, 4.0f, 8.0f},  // 样本7
            {82, 80, 3.2f, 7.0f},  // 样本8
            {87, 83, 3.8f, 7.8f},  // 样本9
            {86, 81, 3.6f, 7.2f},  // 样本10
            
            // C类样本
            {70, 68, 2.0f, 5.5f},  // 样本11
            {72, 70, 2.2f, 6.0f},  // 样本12
            {68, 65, 1.8f, 5.0f},  // 样本13
            {75, 72, 2.5f, 6.5f},  // 样本14
            {71, 69, 2.1f, 5.8f}   // 样本15
        };
        
        String[] labels = {
            "A", "A", "A", "A", "A",
            "B", "B", "B", "B", "B",
            "C", "C", "C", "C", "C"
        };
        
        // 创建特征矩阵
        IMatrix featureMatrix = new RereMatrix(featureData);
        
        System.out.println("多分类训练数据：");
        System.out.println("特征矩阵形状: " + featureMatrix.getRowNum() + " x " + featureMatrix.getColNum());
        System.out.println("标签数量: " + labels.length);
        System.out.println("类别: A、B、C");
        System.out.println();
        
        // 创建并训练多分类模型
        RereLogisticRegression lr = new RereLogisticRegression();
        lr.setLearningRate(0.01f);
        lr.setMaxIterations(1000);
        lr.setTolerance(1e-6f);
        lr.setRegularization(0.0f, 0.01f);  // 使用L2正则化
        
        try {
            long startTime = System.currentTimeMillis();
            LogisticRegressionResult result = lr.fit(featureMatrix, labels);
            long endTime = System.currentTimeMillis();
            
            System.out.println("多分类模型训练完成！");
            System.out.println("模型类型: " + lr.getModelTypeDescription());
            System.out.println("训练耗时: " + (endTime - startTime) + "ms");
            System.out.println("最终损失: " + result.getLoss());
            System.out.println("正则化: " + lr.getRegularizationDescription());
            System.out.println();
            
            // 评估多分类模型
            evaluateMulticlassModel(lr, featureMatrix, labels);
            
        } catch (Exception e) {
            System.err.println("多分类训练失败：" + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试不同正则化方法
     */
    private static void testRegularizationMethods() {
        System.out.println("=== 正则化方法测试 ===\n");
        
        // 使用多分类数据进行正则化测试
        float[][] featureData = {
            {95, 92, 5.0f, 9.5f}, {98, 95, 6.0f, 9.8f}, {92, 88, 4.5f, 9.0f},
            {85, 82, 3.5f, 7.5f}, {88, 85, 4.0f, 8.0f}, {82, 80, 3.2f, 7.0f},
            {70, 68, 2.0f, 5.5f}, {72, 70, 2.2f, 6.0f}, {68, 65, 1.8f, 5.0f}
        };
        
        String[] labels = {"A", "A", "A", "B", "B", "B", "C", "C", "C"};
        
        IMatrix featureMatrix = new RereMatrix(featureData);
        
        // 测试L1正则化
        System.out.println("--- L1正则化测试 ---");
        testRegularization(featureMatrix, labels, 0.01f, 0.0f, "L1正则化");
        
        // 测试L2正则化
        System.out.println("--- L2正则化测试 ---");
        testRegularization(featureMatrix, labels, 0.0f, 0.01f, "L2正则化");
        
        // 测试ElasticNet正则化
        System.out.println("--- ElasticNet正则化测试 ---");
        testRegularization(featureMatrix, labels, 0.005f, 0.005f, "ElasticNet正则化");
        
        // 测试无正则化
        System.out.println("--- 无正则化测试 ---");
        testRegularization(featureMatrix, labels, 0.0f, 0.0f, "无正则化");
    }
    
    /**
     * 测试特定正则化方法
     */
    private static void testRegularization(IMatrix featureMatrix, String[] labels, 
                                         float lambda1, float lambda2, String methodName) {
        RereLogisticRegression lr = new RereLogisticRegression();
        lr.setLearningRate(0.01f);
        lr.setMaxIterations(1000);
        lr.setTolerance(1e-6f);
        lr.setRegularization(lambda1, lambda2);
        
        try {
            long startTime = System.currentTimeMillis();
            LogisticRegressionResult result = lr.fit(featureMatrix, labels);
            long endTime = System.currentTimeMillis();
            
            System.out.println("模型类型: " + lr.getModelTypeDescription());
            System.out.println("训练耗时: " + (endTime - startTime) + "ms");
            System.out.println("最终损失: " + result.getLoss());
            System.out.println("正则化: " + lr.getRegularizationDescription());
            
            // 计算准确率
            String[] predictions = lr.predictBatch(featureMatrix);
            int correct = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (predictions[i].equals(labels[i])) {
                    correct++;
                }
            }
            float accuracy = (float) correct / predictions.length * 100;
            System.out.printf("准确率: %.1f%% (%d/%d)%n", accuracy, correct, predictions.length);
            System.out.println();
            
        } catch (Exception e) {
            System.err.println(methodName + "训练失败：" + e.getMessage());
        }
    }
    
    /**
     * 评估二分类模型
     */
    private static void evaluateBinaryClassificationModel(RereLogisticRegression lr, 
                                                        IMatrix featureMatrix, String[] labels) {
        System.out.println("二分类模型评估：");
        
        // 显示标签映射
        System.out.println("标签映射: " + lr.getLabelMapping());
        
        // 在训练集上进行预测
        String[] predictions = lr.predictBatch(featureMatrix);
        
        // 计算准确率
        int correctPredictions = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i].equals(labels[i])) {
                correctPredictions++;
            }
        }
        float accuracy = (float) correctPredictions / predictions.length * 100;
        
        System.out.printf("准确率: %.1f%% (%d/%d)%n", accuracy, correctPredictions, predictions.length);
        
        // 显示详细预测结果
        System.out.println("详细预测结果：");
        for (int i = 0; i < featureMatrix.getRowNum(); i++) {
            IVector sample = featureMatrix.getRow(i);
            float probability = lr.predictProbability(sample);
            
            System.out.printf("  样本%d: 预测=%s, 实际=%s, 正类概率=%.3f%n", 
                i + 1, predictions[i], labels[i], probability);
        }
        
        // 预测新样本
        System.out.println("\n新样本预测：");
        float[][] newSamples = {
            {2.5f, 3.5f, 4.5f},   // 新样本1：介于正负类之间
            {-2.5f, -3.5f, -4.5f}, // 新样本2：负类
            {6.0f, 7.0f, 8.0f}     // 新样本3：正类
        };
        
        for (int i = 0; i < newSamples.length; i++) {
            IVector newSample = new RereVector(newSamples[i]);
            String prediction = lr.predict(newSample);
            float probability = lr.predictProbability(newSample);
            
            System.out.printf("  新样本%d [%.1f, %.1f, %.1f]: 预测=%s, 正类概率=%.3f%n", 
                i + 1, newSamples[i][0], newSamples[i][1], newSamples[i][2], 
                prediction, probability);
        }
        
        // 显示权重信息
        System.out.println("\n权重信息：");
        IMatrix weightsMatrix = lr.getWeights();
        IVector biasVector = lr.getBias();
        
        System.out.println("权重向量: [");
        for (int j = 0; j < weightsMatrix.getColNum(); j++) {
            if (j > 0) System.out.print(", ");
            System.out.printf("%.4f", weightsMatrix.get(0, j));
        }
        System.out.printf("], 偏置: %.4f%n", biasVector.get(0));
    }
    
    /**
     * 评估多分类模型
     */
    private static void evaluateMulticlassModel(RereLogisticRegression lr, 
                                              IMatrix featureMatrix, String[] labels) {
        System.out.println("多分类模型评估：");
        
        // 显示标签映射
        System.out.println("标签映射: " + lr.getLabelMapping());
        
        // 在训练集上进行预测
        String[] predictions = lr.predictBatch(featureMatrix);
        
        // 计算准确率
        int correctPredictions = 0;
        for (int i = 0; i < predictions.length; i++) {
            if (predictions[i].equals(labels[i])) {
                correctPredictions++;
            }
        }
        float accuracy = (float) correctPredictions / predictions.length * 100;
        
        System.out.printf("准确率: %.1f%% (%d/%d)%n", accuracy, correctPredictions, predictions.length);
        
        // 显示详细预测结果（只显示前5个样本）
        System.out.println("详细预测结果（前5个样本）：");
        int showCount = Math.min(5, featureMatrix.getRowNum());
        for (int i = 0; i < showCount; i++) {
            IVector sample = featureMatrix.getRow(i);
            float[] probabilities = lr.predictProbabilities(sample);
            
            System.out.printf("  样本%d: 预测=%s, 实际=%s, 概率分布: A=%.3f, B=%.3f, C=%.3f%n", 
                i + 1, predictions[i], labels[i], 
                probabilities[0], probabilities[1], probabilities[2]);
        }
        
        // 预测新样本
        System.out.println("\n新样本预测：");
        float[][] newSamples = {
            {90, 88, 4.2f, 8.5f},  // 新样本1：介于A和B之间
            {78, 75, 2.8f, 6.5f},  // 新样本2：介于B和C之间
            {99, 96, 6.5f, 9.9f}   // 新样本3：A类
        };
        
        for (int i = 0; i < newSamples.length; i++) {
            IVector newSample = new RereVector(newSamples[i]);
            String prediction = lr.predict(newSample);
            float[] probabilities = lr.predictProbabilities(newSample);
            
            System.out.printf("  新样本%d [数学:%.0f, 英语:%.0f, 学习时间:%.1f小时, 参与度:%.1f]: ", 
                i + 1, newSamples[i][0], newSamples[i][1], newSamples[i][2], newSamples[i][3]);
            System.out.printf("预测=%s, 概率分布: A=%.3f, B=%.3f, C=%.3f%n", 
                prediction, probabilities[0], probabilities[1], probabilities[2]);
        }
        
        // 显示权重信息
        System.out.println("\n权重信息：");
        IMatrix weightsMatrix = lr.getWeights();
        IVector biasVector = lr.getBias();
        
        System.out.println("权重矩阵 (每行对应一个类别):");
        for (int k = 0; k < lr.getNumClasses(); k++) {
            final int classIndex = k;
            IVector classWeights = weightsMatrix.getRow(k);
            System.out.printf("  类别%d (%s): [", k, lr.getLabelMapping().entrySet().stream()
                .filter(entry -> entry.getValue() == classIndex)
                .findFirst()
                .map(Map.Entry::getKey)
                .orElse("未知"));
            
            for (int j = 0; j < classWeights.length(); j++) {
                if (j > 0) System.out.print(", ");
                System.out.printf("%.4f", classWeights.get(j));
            }
            System.out.printf("], 偏置: %.4f%n", biasVector.get(k));
        }
    }
}
