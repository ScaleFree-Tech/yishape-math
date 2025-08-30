package com.reremouse.lab.math.test;

import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.RereMatrix;
import com.reremouse.lab.math.RereVector;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.ml.cls.LogisticRegressionResult;
import com.reremouse.lab.math.ml.cls.RereLogisticRegression;

/**
 * 逻辑斯蒂回归演示类
 * <p>
 * 本类演示了如何使用RereLogisticRegression进行二分类任务。
 * 包含了一个简单的示例：根据学生的考试成绩预测是否通过考试。
 * </p>
 * 
 * <h3>示例说明：</h3>
 * <ul>
 *   <li>特征：数学成绩、英语成绩、学习时间</li>
 *   <li>标签：通过(1)、不通过(0)</li>
 *   <li>模型：逻辑斯蒂回归</li>
 * </ul>
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class LogisticRegressionDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 逻辑斯蒂回归演示 ===\n");
        
        // 创建训练数据
        // 特征矩阵：每行是一个学生，每列是一个特征
        // 特征：[数学成绩, 英语成绩, 学习时间(小时)]
        float[][] featureData = {
            {85, 78, 3.5f},  // 学生1：数学85，英语78，学习3.5小时
            {92, 85, 4.0f},  // 学生2：数学92，英语85，学习4.0小时
            {78, 82, 2.5f},  // 学生3：数学78，英语82，学习2.5小时
            {95, 90, 5.0f},  // 学生4：数学95，英语90，学习5.0小时
            {70, 75, 2.0f},  // 学生5：数学70，英语75，学习2.0小时
            {88, 88, 3.8f},  // 学生6：数学88，英语88，学习3.8小时
            {82, 79, 3.0f},  // 学生7：数学82，英语79，学习3.0小时
            {90, 92, 4.5f},  // 学生8：数学90，英语92，学习4.5小时
            {75, 70, 2.2f},  // 学生9：数学75，英语70，学习2.2小时
            {87, 85, 3.2f}   // 学生10：数学87，英语85，学习3.2小时
        };
        
        // 标签：1表示通过，0表示不通过
        String[] labels = {"1", "1", "0", "1", "0", "1", "1", "1", "0", "1"};
        
        // 创建特征矩阵
        IMatrix featureMatrix = new RereMatrix(featureData);
        
        System.out.println("训练数据：");
        System.out.println("特征矩阵形状: " + featureMatrix.getRowNum() + " x " + featureMatrix.getColNum());
        System.out.println("标签数量: " + labels.length);
        System.out.println();
        
        // 创建逻辑斯蒂回归模型
        System.out.println("创建逻辑斯蒂回归模型...");
        RereLogisticRegression lr = new RereLogisticRegression();
        
        // 设置模型参数
        lr.setLearningRate(0.01f);
        lr.setMaxIterations(1000);
        lr.setTolerance(1e-6f);
        lr.setRegularization(0.05f, 0.1f);  // ElasticNet正则化：L1=0.05, L2=0.1
        
        System.out.println("模型参数：");
        System.out.println("  学习率: " + lr.getLearningRate());
        System.out.println("  最大迭代次数: " + lr.getMaxIterations());
        System.out.println("  收敛阈值: " + lr.getTolerance());
        System.out.println("  正则化: " + lr.getRegularizationDescription());
        System.out.println();
        
        try {
            // 训练模型
            System.out.println("开始训练模型...");
            long startTime = System.currentTimeMillis();
            
            LogisticRegressionResult result = lr.fit(featureMatrix, labels);
            
            long endTime = System.currentTimeMillis();
            System.out.println("训练完成！耗时: " + (endTime - startTime) + "ms");
            System.out.println();
            
            // 显示训练结果
            System.out.println("训练结果：");
            System.out.println("  最终损失: " + result.getLoss());
            
            // 手动格式化权重向量输出，避免依赖问题
            IVector weights = result.getWeights();
            System.out.print("  权重向量: [");
            for (int i = 0; i < weights.length(); i++) {
                if (i > 0) System.out.print(", ");
                System.out.printf("%.4f", weights.get(i));
            }
            System.out.println("]");
            
            // 手动格式化偏置项输出
            IVector bias = result.getBias();
            System.out.printf("  偏置项: %.4f%n", bias.get(0));
            System.out.println();
            
            // 显示标签映射
            System.out.println("标签映射：");
            System.out.println("  " + lr.getLabelMapping());
            System.out.println();
            
            // 在训练集上进行预测
            System.out.println("训练集预测结果：");
            String[] predictions = lr.predictBatch(featureMatrix);
            for (int i = 0; i < featureMatrix.getRowNum(); i++) {
                IVector sample = featureMatrix.getRow(i);
                float probability = lr.predictProbability(sample);
                System.out.printf("  学生%d: 预测=%s, 概率=%.3f, 实际=%s%n", 
                    i + 1, predictions[i], probability, labels[i]);
            }
            System.out.println();
            
            // 预测新样本
            System.out.println("新样本预测：");
            float[][] newSamples = {
                {80, 80, 3.0f},  // 新学生1
                {95, 95, 5.0f},  // 新学生2
                {65, 65, 1.5f}   // 新学生3
            };
            
            for (int i = 0; i < newSamples.length; i++) {
                IVector newSample = new RereVector(newSamples[i]);
                String prediction = lr.predict(newSample);
                float probability = lr.predictProbability(newSample);
                
                System.out.printf("  新学生%d [数学:%.0f, 英语:%.0f, 学习时间:%.1f小时]: ", 
                    i + 1, newSamples[i][0], newSamples[i][1], newSamples[i][2]);
                System.out.printf("预测=%s, 通过概率=%.3f%n", prediction, probability);
            }
            System.out.println();
            
            // 模型评估
            System.out.println("模型评估：");
            int correctPredictions = 0;
            for (int i = 0; i < predictions.length; i++) {
                if (predictions[i].equals(labels[i])) {
                    correctPredictions++;
                }
            }
            float accuracy = (float) correctPredictions / predictions.length * 100;
            System.out.printf("  准确率: %.1f%% (%d/%d)%n", accuracy, correctPredictions, predictions.length);
            
        } catch (Exception e) {
            System.err.println("训练过程中发生错误：");
            e.printStackTrace();
        }
    }
}
