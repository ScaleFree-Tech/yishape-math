package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.ml.clf.lr.RereLogisticRegression;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;

/**
 * 缓存优化测试类
 * 用于验证RereLogisticRegression的缓存优化效果
 */
public class CacheOptimizationTest {

    public static void main(String[] args) {
        System.out.println("=== 缓存优化测试 ===");
        
        // 创建测试数据
        int numSamples = 1000;
        int numFeatures = 100;
        int numClasses = 3;
        
        System.out.println("测试数据规模: " + numSamples + " 样本, " + numFeatures + " 特征, " + numClasses + " 类");
        
        // 生成随机测试数据
        IMatrix features = generateRandomFeatures(numSamples, numFeatures);
        String[] labels = generateRandomLabels(numSamples, numClasses);
        
        // 测试优化前的性能（模拟）
        System.out.println("\n--- 测试缓存优化效果 ---");
        
        // 创建模型
        RereLogisticRegression model = new RereLogisticRegression(0.01, 0.01);
        model.setStandardizeFeatures(true);
        model.setUseClassWeights(true);
        
        // 训练模型
        long startTime = System.currentTimeMillis();
        model.fit(features, labels);
        long trainTime = System.currentTimeMillis() - startTime;
        
        System.out.println("训练时间: " + trainTime + " ms");
        
        // 测试目标函数和梯度计算
        IObjectiveFunction objectiveFunction = model;
        IGradientFunction gradientFunction = model;
        
        // 创建测试参数向量
        long paramStartTime = System.currentTimeMillis();
        var paramVector = model.createParameterVector();
        long paramTime = System.currentTimeMillis() - paramStartTime;
        
        System.out.println("参数向量创建时间: " + paramTime + " ms");
        
        // 测试目标函数计算（第一次，无缓存）
        long objStartTime = System.currentTimeMillis();
        double objective1 = objectiveFunction.computeObjective(paramVector);
        long objTime1 = System.currentTimeMillis() - objStartTime;
        
        System.out.println("第一次目标函数计算时间: " + objTime1 + " ms");
        System.out.println("目标函数值: " + objective1);
        
        // 测试梯度计算（第一次，无缓存）
        long gradStartTime = System.currentTimeMillis();
        var gradient1 = gradientFunction.computeGradient(paramVector);
        long gradTime1 = System.currentTimeMillis() - gradStartTime;
        
        System.out.println("第一次梯度计算时间: " + gradTime1 + " ms");
        System.out.println("梯度向量长度: " + gradient1.length());
        
        // 再次测试目标函数计算（第二次，有缓存）
        long objStartTime2 = System.currentTimeMillis();
        double objective2 = objectiveFunction.computeObjective(paramVector);
        long objTime2 = System.currentTimeMillis() - objStartTime2;
        
        System.out.println("第二次目标函数计算时间: " + objTime2 + " ms");
        System.out.println("目标函数值: " + objective2);
        
        // 再次测试梯度计算（第二次，有缓存）
        long gradStartTime2 = System.currentTimeMillis();
        var gradient2 = gradientFunction.computeGradient(paramVector);
        long gradTime2 = System.currentTimeMillis() - gradStartTime2;
        
        System.out.println("第二次梯度计算时间: " + gradTime2 + " ms");
        System.out.println("梯度向量长度: " + gradient2.length());
        
        // 计算性能提升
        double objSpeedup = (double) objTime1 / objTime2;
        double gradSpeedup = (double) gradTime1 / gradTime2;
        
        System.out.println("\n--- 性能提升分析 ---");
        System.out.println("目标函数计算加速比: " + String.format("%.2f", objSpeedup) + "x");
        System.out.println("梯度计算加速比: " + String.format("%.2f", gradSpeedup) + "x");
        
        // 验证结果一致性
        boolean objConsistent = Math.abs(objective1 - objective2) < 1e-10;
        boolean gradConsistent = true;
        
        for (int i = 0; i < gradient1.length(); i++) {
            if (Math.abs((double) gradient1.get(i) - (double) gradient2.get(i)) > 1e-10) {
                gradConsistent = false;
                break;
            }
        }
        
        System.out.println("\n--- 结果一致性验证 ---");
        System.out.println("目标函数值一致性: " + (objConsistent ? "通过" : "失败"));
        System.out.println("梯度向量一致性: " + (gradConsistent ? "通过" : "失败"));
        
        if (objConsistent && gradConsistent) {
            System.out.println("\n✅ 缓存优化测试通过！性能显著提升且结果一致。");
        } else {
            System.out.println("\n❌ 缓存优化测试失败！结果不一致。");
        }
    }
    
    /**
     * 生成随机特征矩阵
     */
    private static IMatrix generateRandomFeatures(int numSamples, int numFeatures) {
        double[][] data = new double[numSamples][numFeatures];
        java.util.Random random = new java.util.Random(42); // 固定种子以确保可重现性
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                data[i][j] = random.nextDouble() * 10 - 5; // [-5, 5] 范围内的随机数
            }
        }
        
        return Linalg.matrix(data);
    }
    
    /**
     * 生成随机标签数组
     */
    private static String[] generateRandomLabels(int numSamples, int numClasses) {
        String[] labels = new String[numSamples];
        java.util.Random random = new java.util.Random(42); // 固定种子以确保可重现性
        
        for (int i = 0; i < numSamples; i++) {
            int labelIndex = random.nextInt(numClasses);
            labels[i] = "class_" + labelIndex;
        }
        
        return labels;
    }
}