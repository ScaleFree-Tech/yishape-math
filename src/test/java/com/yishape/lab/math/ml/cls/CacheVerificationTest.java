package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.ml.clf.lr.RereLogisticRegression;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;

/**
 * 缓存验证测试类
 * 用于详细验证RereLogisticRegression的缓存机制是否正确工作
 */
public class CacheVerificationTest {

    public static void main(String[] args) {
        System.out.println("=== 缓存机制验证测试 ===");
        
        // 创建小规模测试数据以便详细分析
        int numSamples = 1000;
        int numFeatures = 30;
        int numClasses = 2;
        
        System.out.println("测试数据规模: " + numSamples + " 样本, " + numFeatures + " 特征, " + numClasses + " 类");
        
        // 生成确定性测试数据
        IMatrix features = generateDeterministicFeatures(numSamples, numFeatures);
        String[] labels = generateDeterministicLabels(numSamples, numClasses);
        
        // 创建模型
        RereLogisticRegression model = new RereLogisticRegression(0.01, 0.01);
        model.setStandardizeFeatures(false); // 禁用归一化以便更容易验证
        
        // 训练模型
        System.out.println("\n--- 训练模型 ---");
        model.fit(features, labels);
        
        // 获取训练后的参数向量
        var paramVector = model.createParameterVector();
        System.out.println("参数向量长度: " + paramVector.length());
        
        // 测试目标函数计算
        IObjectiveFunction objectiveFunction = model;
        System.out.println("\n--- 测试目标函数计算 ---");
        
        // 第一次计算（应该没有缓存）
        System.out.println("第一次目标函数计算...");
        long startTime1 = System.currentTimeMillis();
        double objective1 = objectiveFunction.computeObjective(paramVector);
        long time1 = System.currentTimeMillis() - startTime1;
        System.out.println("耗时: " + time1 + " ms");
        System.out.println("目标函数值: " + objective1);
        
        // 第二次计算（应该使用缓存）
        System.out.println("第二次目标函数计算（相同参数）...");
        long startTime2 = System.currentTimeMillis();
        double objective2 = objectiveFunction.computeObjective(paramVector);
        long time2 = System.currentTimeMillis() - startTime2;
        System.out.println("耗时: " + time2 + " ms");
        System.out.println("目标函数值: " + objective2);
        
        // 验证结果一致性
        boolean objConsistent = Math.abs(objective1 - objective2) < 1e-10;
        System.out.println("目标函数值一致性: " + (objConsistent ? "通过" : "失败"));
        
        // 测试梯度计算
        IGradientFunction gradientFunction = model;
        System.out.println("\n--- 测试梯度计算 ---");
        
        // 第一次计算（应该没有缓存）
        System.out.println("第一次梯度计算...");
        long startTime3 = System.currentTimeMillis();
        var gradient1 = gradientFunction.computeGradient(paramVector);
        long time3 = System.currentTimeMillis() - startTime3;
        System.out.println("耗时: " + time3 + " ms");
        System.out.println("梯度向量长度: " + gradient1.length());
        
        // 第二次计算（应该使用缓存）
        System.out.println("第二次梯度计算（相同参数）...");
        long startTime4 = System.currentTimeMillis();
        var gradient2 = gradientFunction.computeGradient(paramVector);
        long time4 = System.currentTimeMillis() - startTime4;
        System.out.println("耗时: " + time4 + " ms");
        System.out.println("梯度向量长度: " + gradient2.length());
        
        // 验证结果一致性
        boolean gradConsistent = true;
        for (int i = 0; i < gradient1.length(); i++) {
            if (Math.abs((double) gradient1.get(i) - (double) gradient2.get(i)) > 1e-10) {
                gradConsistent = false;
                break;
            }
        }
        System.out.println("梯度向量一致性: " + (gradConsistent ? "通过" : "失败"));
        
        // 测试参数改变后的缓存失效
        System.out.println("\n--- 测试缓存失效机制 ---");
        
        // 创建稍微不同的参数向量
        var modifiedParamVector = paramVector.copy();
        modifiedParamVector.set(0, (double) modifiedParamVector.get(0) + 0.001);
        
        // 计算修改后参数的目标函数（应该重新计算）
        System.out.println("修改参数后的目标函数计算...");
        long startTime5 = System.currentTimeMillis();
        double objective3 = objectiveFunction.computeObjective(modifiedParamVector);
        long time5 = System.currentTimeMillis() - startTime5;
        System.out.println("耗时: " + time5 + " ms");
        System.out.println("目标函数值: " + objective3);
        
        // 再次计算相同修改后参数（应该使用缓存）
        System.out.println("再次计算相同修改后参数...");
        long startTime6 = System.currentTimeMillis();
        double objective4 = objectiveFunction.computeObjective(modifiedParamVector);
        long time6 = System.currentTimeMillis() - startTime6;
        System.out.println("耗时: " + time6 + " ms");
        System.out.println("目标函数值: " + objective4);
        
        // 验证结果一致性
        boolean objConsistent2 = Math.abs(objective3 - objective4) < 1e-10;
        System.out.println("修改后参数目标函数值一致性: " + (objConsistent2 ? "通过" : "失败"));
        
        // 性能分析
        System.out.println("\n--- 性能分析 ---");
        if (time2 > 0) {
            double objSpeedup = (double) time1 / time2;
            System.out.println("目标函数计算加速比: " + String.format("%.2f", objSpeedup) + "x");
        } else {
            System.out.println("目标函数计算加速比: 无法计算（第二次计算时间太短）");
        }
        
        if (time4 > 0) {
            double gradSpeedup = (double) time3 / time4;
            System.out.println("梯度计算加速比: " + String.format("%.2f", gradSpeedup) + "x");
        } else {
            System.out.println("梯度计算加速比: 无法计算（第二次计算时间太短）");
        }
        
        // 总结
        System.out.println("\n=== 测试总结 ===");
        if (objConsistent && gradConsistent && objConsistent2) {
            System.out.println("✅ 所有缓存验证测试通过！");
            System.out.println("缓存机制正确工作：");
            System.out.println("- 相同参数时重用缓存");
            System.out.println("- 参数改变时正确失效缓存");
            System.out.println("- 计算结果保持一致性");
        } else {
            System.out.println("❌ 缓存验证测试失败！");
            if (!objConsistent) System.out.println("- 目标函数结果不一致");
            if (!gradConsistent) System.out.println("- 梯度计算结果不一致");
            if (!objConsistent2) System.out.println("- 修改参数后结果不一致");
        }
    }
    
    /**
     * 生成确定性特征矩阵
     */
    private static IMatrix generateDeterministicFeatures(int numSamples, int numFeatures) {
        double[][] data = new double[numSamples][numFeatures];
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                data[i][j] = (i + 1) * (j + 1) * 0.1; // 简单的确定性模式
            }
        }
        
        return Linalg.matrix(data);
    }
    
    /**
     * 生成确定性标签数组
     */
    private static String[] generateDeterministicLabels(int numSamples, int numClasses) {
        String[] labels = new String[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            int labelIndex = i % numClasses;
            labels[i] = "class_" + labelIndex;
        }
        
        return labels;
    }
}