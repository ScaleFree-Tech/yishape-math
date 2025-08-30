package com.reremouse.lab.math.test;

import com.reremouse.lab.math.IMatrix;
import com.reremouse.lab.math.IVector;
import com.reremouse.lab.math.ml.lr.RegressionResult;
import com.reremouse.lab.math.ml.lr.RereLinearRegression;

/**
 * 线性回归演示类
 * <p>
 * 本类展示了如何使用RereLinearRegression类进行线性回归模型的训练和预测，
 * 包括各种正则化选项（L1、L2、ElasticNet）
 * </p>
 * 
 * <h3>演示内容 / Demo Content:</h3>
 * <ul>
 *   <li>创建线性回归模型</li>
 *   <li>准备训练数据</li>
 *   <li>训练模型（无正则化）</li>
 *   <li>训练模型（L1正则化）</li>
 *   <li>训练模型（L2正则化）</li>
 *   <li>训练模型（ElasticNet正则化）</li>
 *   <li>进行预测和结果分析</li>
 * </ul>
 * 
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public class LinearRegressionDemo {
    
    public static void main(String[] args) {
        System.out.println("=== 线性回归演示（支持多种正则化） ===\n");
        
        // 1. 创建线性回归模型
        System.out.println("1. 创建线性回归模型");
        RereLinearRegression linearRegression = new RereLinearRegression();
        System.out.println("   - 包含偏置项: " + linearRegression.isIncludeBias());
        System.out.println("   - 正则化类型: " + linearRegression.getRegularizationDescription());
        System.out.println();
        
        // 2. 准备训练数据
        System.out.println("2. 准备训练数据");
        System.out.println("   假设我们要拟合的线性关系: y = 2*x1 + 3*x2 + 1");
        
        // 创建特征矩阵：5个样本，2个特征
        float[][] features = {
            {1.0f, 2.0f},  // 样本1：x1=1, x2=2
            {2.0f, 3.0f},  // 样本2：x1=2, x2=3
            {3.0f, 4.0f},  // 样本3：x1=3, x2=4
            {4.0f, 5.0f},  // 样本4：x1=4, x2=5
            {5.0f, 6.0f}   // 样本5：x1=5, x2=6
        };
        
        // 创建对应的标签向量
        float[] labels = new float[features.length];
        for (int i = 0; i < features.length; i++) {
            // 计算真实值：y = 2*x1 + 3*x2 + 1
            labels[i] = 2.0f * features[i][0] + 3.0f * features[i][1] + 1.0f;
        }
        
        IMatrix featureMatrix = IMatrix.of(features);
        IVector labelVector = IVector.of(labels);
        
        System.out.println("   特征矩阵形状: " + featureMatrix.getRowNum() + " x " + featureMatrix.getColNum());
        System.out.println("   标签向量长度: " + labelVector.length());
        System.out.println("   训练样本:");
        for (int i = 0; i < features.length; i++) {
            System.out.printf("     样本%d: x1=%.1f, x2=%.1f, y=%.1f\n", 
                i+1, features[i][0], features[i][1], labels[i]);
        }
        System.out.println();
        
        // 3. 训练无正则化模型
        System.out.println("3. 训练无正则化模型");
        try {
            RegressionResult result = linearRegression.fit(featureMatrix, labelVector);
            System.out.println("   训练完成！");
            System.out.println("   - 最终损失值: " + result.getLoss());
            
            // 显示分离后的权重和偏置项
            System.out.println("   - 特征权重向量长度: " + result.getWeights().length());
            System.out.println("   - 偏置项向量长度: " + result.getBias().length());
            System.out.println("   - 特征权重向量:");
            for (int i = 0; i < result.getWeights().length(); i++) {
                System.out.printf("     w%d = %.6f (特征%d的系数)\n", i, result.getWeights().getData()[i], i);
            }
            System.out.printf("   - 偏置项: b = %.6f\n", result.getBias().getData()[0]);
            
            // 使用新的getter方法获取权重信息
            System.out.println("   - 使用getter方法获取:");
            System.out.printf("     - 特征权重数量: %d\n", linearRegression.getFeatureWeights() != null ? linearRegression.getFeatureWeights().length() : 0);
            System.out.printf("     - 偏置项值: %.6f\n", linearRegression.getBias());
            System.out.printf("     - 完整权重向量长度: %d\n", linearRegression.getFullWeights() != null ? linearRegression.getFullWeights().length() : 0);
            
        } catch (Exception e) {
            System.err.println("   训练失败: " + e.getMessage());
            e.printStackTrace();
        }
        
        // 4. 进行预测
        System.out.println("4. 进行预测");
        
        // 测试新样本
        float[][] testFeatures = {
            {1.5f, 2.5f},  // 测试样本1
            {2.5f, 3.5f},  // 测试样本2
            {3.5f, 4.5f}   // 测试样本3
        };
        
        System.out.println("   测试样本预测结果:");
        for (int i = 0; i < testFeatures.length; i++) {
            IVector testFeature = IVector.of(testFeatures[i]);
            float prediction = linearRegression.predict(testFeature);
            
            // 计算真实值
            float actual = 2.0f * testFeatures[i][0] + 3.0f * testFeatures[i][1] + 1.0f;
            float error = Math.abs(prediction - actual);
            
            System.out.printf("     测试样本%d: x1=%.1f, x2=%.1f\n", i+1, testFeatures[i][0], testFeatures[i][1]);
            System.out.printf("       预测值: %.6f\n", prediction);
            System.out.printf("       真实值: %.6f\n", actual);
            System.out.printf("       误差: %.6f\n", error);
            System.out.println();
        }
        
        // 5. 模型分析
        System.out.println("5. 模型分析");
        System.out.println("   - 样本数量: " + linearRegression.getSampleCount());
        System.out.println("   - 特征数量: " + linearRegression.getFeatureCount());
        System.out.println("   - 是否包含偏置项: " + linearRegression.isIncludeBias());
        System.out.println("   - 正则化类型: " + linearRegression.getRegularizationDescription());
        
        // 6. 演示不同正则化类型
        System.out.println("\n6. 演示不同正则化类型");
        
        // L1正则化模型（使用自动推断）
        System.out.println("   L1正则化模型 (Lasso) - 自动推断:");
        try {
            RereLinearRegression lrL1 = new RereLinearRegression(true, 0.1f, 0.0f);
            System.out.println("   - 自动推断的正则化类型: " + lrL1.getRegularizationDescription());
            
            RegressionResult resultL1 = lrL1.fit(featureMatrix, labelVector);
            System.out.println("   - 损失值: " + resultL1.getLoss());
            System.out.println("   - 权重向量长度: " + resultL1.getWeights().length());
            
            // 显示L1正则化后的权重和偏置项
            System.out.println("   - L1正则化后的权重和偏置项:");
            System.out.println("     - 特征权重:");
            for (int i = 0; i < resultL1.getWeights().length(); i++) {
                System.out.printf("       w%d = %.6f\n", i, resultL1.getWeights().getData()[i]);
            }
            System.out.printf("     - 偏置项: b = %.6f\n", resultL1.getBias().getData()[0]);
        } catch (Exception e) {
            System.err.println("   L1正则化训练失败: " + e.getMessage());
        }
        System.out.println();
        
        // L2正则化模型（使用自动推断）
        System.out.println("   L2正则化模型 (Ridge) - 自动推断:");
        try {
            RereLinearRegression lrL2 = new RereLinearRegression(true, 0.0f, 0.1f);
            System.out.println("   - 自动推断的正则化类型: " + lrL2.getRegularizationDescription());
            
            RegressionResult resultL2 = lrL2.fit(featureMatrix, labelVector);
            System.out.println("   - 损失值: " + resultL2.getLoss());
            System.out.println("   - 权重向量长度: " + resultL2.getWeights().length());
            
            // 显示L2正则化后的权重和偏置项
            System.out.println("   - L2正则化后的权重和偏置项:");
            System.out.println("     - 特征权重:");
            for (int i = 0; i < resultL2.getWeights().length(); i++) {
                System.out.printf("       w%d = %.6f\n", i, resultL2.getWeights().getData()[i]);
            }
            System.out.printf("     - 偏置项: b = %.6f\n", resultL2.getBias().getData()[0]);
        } catch (Exception e) {
            System.err.println("   L2正则化训练失败: " + e.getMessage());
        }
        System.out.println();
        
        // ElasticNet正则化模型（使用自动推断）
        System.out.println("   ElasticNet正则化模型 - 自动推断:");
        try {
            RereLinearRegression lrElasticNet = new RereLinearRegression(true, 0.05f, 0.05f);
            System.out.println("   - 自动推断的正则化类型: " + lrElasticNet.getRegularizationDescription());
            
            RegressionResult resultElasticNet = lrElasticNet.fit(featureMatrix, labelVector);
            System.out.println("   - 损失值: " + resultElasticNet.getLoss());
            System.out.println("   - 权重向量长度: " + resultElasticNet.getWeights().length());
            
            // 显示ElasticNet正则化后的权重和偏置项
            System.out.println("   - ElasticNet正则化后的权重和偏置项:");
            System.out.println("     - 特征权重:");
            for (int i = 0; i < resultElasticNet.getWeights().length(); i++) {
                System.out.printf("       w%d = %.6f\n", i, resultElasticNet.getWeights().getData()[i]);
            }
            System.out.printf("     - 偏置项: b = %.6f\n", resultElasticNet.getBias().getData()[0]);
        } catch (Exception e) {
            System.err.println("   ElasticNet正则化训练失败: " + e.getMessage());
        }
        System.out.println();
        
        // 7. 演示参数动态调整和自动推断
        System.out.println("7. 演示参数动态调整和自动推断");
        try {
            RereLinearRegression lrDynamic = new RereLinearRegression();
            System.out.println("   - 初始状态: " + lrDynamic.getRegularizationDescription());
            
            // 设置L1正则化（自动推断）
            lrDynamic.setRegularization(0.1f, 0.0f);
            System.out.println("   - 设置(0.1, 0.0)后: " + lrDynamic.getRegularizationDescription());
            
            // 训练模型
            RegressionResult resultDynamic = lrDynamic.fit(featureMatrix, labelVector);
            System.out.println("   - 训练完成，损失值: " + resultDynamic.getLoss());
            
            // 动态调整L1系数（自动推断）
            lrDynamic.setLambda1(0.2f);
            System.out.println("   - 调整L1系数为0.2后: " + lrDynamic.getRegularizationDescription());
            
            // 切换到ElasticNet（自动推断）
            lrDynamic.setRegularization(0.1f, 0.1f);
            System.out.println("   - 设置(0.1, 0.1)后: " + lrDynamic.getRegularizationDescription());
            
            // 切换到L2（自动推断）
            lrDynamic.setRegularization(0.0f, 0.2f);
            System.out.println("   - 设置(0.0, 0.2)后: " + lrDynamic.getRegularizationDescription());
            
            // 切换到无正则化（自动推断）
            lrDynamic.setRegularization(0.0f, 0.0f);
            System.out.println("   - 设置(0.0, 0.0)后: " + lrDynamic.getRegularizationDescription());
            
        } catch (Exception e) {
            System.err.println("   参数调整演示失败: " + e.getMessage());
        }
        
        // 8. 演示显式设置正则化类型
        System.out.println("\n8. 演示显式设置正则化类型");
        try {
            // 显式设置L1正则化
            RereLinearRegression lrExplicit = new RereLinearRegression(true, 
                RereLinearRegression.RegularizationType.L1, 0.1f, 0.0f);
            System.out.println("   - 显式设置L1正则化: " + lrExplicit.getRegularizationDescription());
            
            // 显式设置ElasticNet
            lrExplicit.setRegularization(RereLinearRegression.RegularizationType.ELASTIC_NET, 0.05f, 0.05f);
            System.out.println("   - 显式设置ElasticNet: " + lrExplicit.getRegularizationDescription());
            
        } catch (Exception e) {
            System.err.println("   显式设置演示失败: " + e.getMessage());
        }
        
        System.out.println("\n=== 演示完成 ===");
    }
}
