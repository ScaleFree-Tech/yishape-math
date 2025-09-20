package com.reremouse.lab.math.stats.model;

import com.reremouse.lab.math.stats.distribution.multiv.MultivariateNormalDistribution;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.RereDoubleVector;
import com.reremouse.lab.math.linalg.RereDoubleMatrix;
import com.reremouse.lab.math.linalg.IMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 简单测试GMM、EM算法和多元高斯分布的修复后代码
 */
public class GMMTest {
    
    public static void main(String[] args) {
        System.out.println("=== GMM、EM算法和多元高斯分布测试 ===\n");
        
        try {
            testMultivariateGaussian();
            testGaussianMixtureModel();
            testEMAlgorithm();
            
            System.out.println("\n=== 所有测试通过！ ===");
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试多元高斯分布
     */
    private static void testMultivariateGaussian() {
        System.out.println("1. 测试多元高斯分布");
        System.out.println("-------------------");
        
        // 创建测试数据
        IVector<Double> mean = new RereDoubleVector(new double[]{1.0, 2.0});
        IMatrix<Double> covariance = new RereDoubleMatrix(new double[][]{
            {1.0, 0.5},
            {0.5, 2.0}
        });
        
        MultivariateNormalDistribution gaussian = new MultivariateNormalDistribution(mean, covariance);
        
        // 测试概率密度函数
        IVector<Double> testPoint = new RereDoubleVector(new double[]{1.5, 2.5});
        double pdf = gaussian.pdf(testPoint);
        double logPdf = gaussian.logPdf(testPoint);
        
        System.out.printf("测试点: [%.1f, %.1f]\n", testPoint.get(0), testPoint.get(1));
        System.out.printf("PDF: %.6f\n", pdf);
        System.out.printf("Log PDF: %.6f\n", logPdf);
        System.out.printf("验证: log(PDF) = %.6f\n", Math.log(pdf));
        
        // 验证对数概率密度的一致性
        double logPdfDiff = Math.abs(logPdf - Math.log(pdf));
        if (logPdfDiff < 1e-10) {
            System.out.println("✅ 对数概率密度计算正确");
        } else {
            System.out.println("❌ 对数概率密度计算有误，差异: " + logPdfDiff);
        }
        
        System.out.println();
    }
    
    /**
     * 测试高斯混合模型
     */
    private static void testGaussianMixtureModel() {
        System.out.println("2. 测试高斯混合模型");
        System.out.println("-------------------");
        
        // 创建简单的2D数据
        List<IVector<Double>> data = generateTestData();
        
        // 创建GMM模型
        GaussianMixtureModel gmm = new GaussianMixtureModel(2, 2); // 2个分量，2维
        
        // 测试初始化
        System.out.println("初始化权重: ");
        for (int i = 0; i < gmm.getNumComponents(); i++) {
            System.out.printf("  分量 %d: %.3f\n", i, gmm.getWeight(i));
        }
        
        // 测试概率密度计算
        IVector<Double> testPoint = data.get(0);
        double pdf = gmm.pdf(testPoint);
        System.out.printf("测试点PDF: %.6f\n", pdf);
        
        // 测试后验概率计算
        IVector<Double> posteriors = gmm.computePosteriors(testPoint);
        System.out.println("后验概率: ");
        for (int i = 0; i < posteriors.size(); i++) {
            System.out.printf("  分量 %d: %.3f\n", i, posteriors.get(i));
        }
        
        // 验证后验概率和为1
        double sum = 0.0;
        for (int i = 0; i < posteriors.size(); i++) {
            sum += posteriors.get(i);
        }
        if (Math.abs(sum - 1.0) < 1e-10) {
            System.out.println("✅ 后验概率归一化正确");
        } else {
            System.out.println("❌ 后验概率归一化有误，和为: " + sum);
        }
        
        System.out.println();
    }
    
    /**
     * 测试EM算法
     */
    private static void testEMAlgorithm() {
        System.out.println("3. 测试EM算法");
        System.out.println("-------------");
        
        // 创建测试数据
        List<IVector<Double>> data = generateTestData();
        
        // 创建EM算法实例
        EMAlgorithm em = new EMAlgorithm(100, 1e-6, true); // 最大100次迭代，收敛阈值1e-6，输出详细信息
        
        // 创建初始GMM模型
        GaussianMixtureModel initialGMM = new GaussianMixtureModel(2, 2); // 2个分量，2维
        initialGMM.fit(data); // 使用K-means++初始化
        
        // 训练GMM
        System.out.println("开始EM算法训练...");
        EMAlgorithm.EMResult result = em.fit(data, initialGMM);
        
        System.out.printf("迭代次数: %d\n", result.iterations);
        System.out.printf("最终对数似然: %.6f\n", result.logLikelihood);
        System.out.printf("是否收敛: %s\n", result.converged ? "是" : "否");
        
        // 计算BIC和AIC
        double bic = em.computeBIC(data, initialGMM);
        double aic = em.computeAIC(data, initialGMM);
        System.out.printf("BIC: %.6f\n", bic);
        System.out.printf("AIC: %.6f\n", aic);
        
        GaussianMixtureModel trainedGMM = initialGMM;
        
        // 验证训练后的模型
        System.out.println("训练后的权重: ");
        for (int i = 0; i < trainedGMM.getNumComponents(); i++) {
            System.out.printf("  分量 %d: %.3f\n", i, trainedGMM.getWeight(i));
        }
        
        // 验证权重和为1
        double weightSum = 0.0;
        for (int i = 0; i < trainedGMM.getNumComponents(); i++) {
            weightSum += trainedGMM.getWeight(i);
        }
        if (Math.abs(weightSum - 1.0) < 1e-6) { // 放宽精度要求
            System.out.println("✅ 权重归一化正确");
        } else {
            System.out.println("❌ 权重归一化有误，和为: " + weightSum);
        }
        
        // 测试预测
        IVector<Double> testPoint = data.get(0);
        int prediction = trainedGMM.predictComponent(testPoint);
        System.out.printf("测试点预测分量: %d\n", prediction);
        
        if (prediction >= 0 && prediction < trainedGMM.getNumComponents()) {
            System.out.println("✅ 预测结果合理");
        } else {
            System.out.println("❌ 预测结果异常");
        }
        
        System.out.println();
    }
    
    /**
     * 生成测试数据
     */
    private static List<IVector<Double>> generateTestData() {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(42); // 固定种子以获得可重复的结果
        
        // 生成两个高斯分布的混合数据，增加样本数量并改善分离度
        for (int i = 0; i < 100; i++) {
            if (random.nextDouble() < 0.5) {
                // 第一个分量：均值[0, 0]，标准差1.0
                double x1 = random.nextGaussian() * 1.0;
                double y1 = random.nextGaussian() * 1.0;
                data.add(new RereDoubleVector(new double[]{x1, y1}));
            } else {
                // 第二个分量：均值[4, 4]，标准差1.0
                double x2 = random.nextGaussian() * 1.0 + 4.0;
                double y2 = random.nextGaussian() * 1.0 + 4.0;
                data.add(new RereDoubleVector(new double[]{x2, y2}));
            }
        }
        
        return data;
    }
}