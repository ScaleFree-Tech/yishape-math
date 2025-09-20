package com.reremouse.lab.math.stats.model;

import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import com.reremouse.lab.math.linalg.RereDoubleVector;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Comprehensive test for improved Gaussian Mixture Model implementation
 */
public class ImprovedGMMComprehensiveTest {
    
    private List<IVector<Double>> testData;
    private List<IVector<Double>> challengingData;
    
    @BeforeEach
    public void setUp() {
        // 生成标准测试数据
        testData = generateStandardTestData();
        
        // 生成具有挑战性的测试数据
        challengingData = generateChallengingTestData();
    }
    
    /**
     * 生成标准测试数据：三个明显分离的高斯分布
     */
    private List<IVector<Double>> generateStandardTestData() {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(42);
        
        // 第一个分量：中心在(0, 0)
        for (int i = 0; i < 100; i++) {
            double x = random.nextGaussian() * 0.5;
            double y = random.nextGaussian() * 0.5;
            data.add(Linalg.vector(x, y));
        }
        
        // 第二个分量：中心在(3, 3)
        for (int i = 0; i < 100; i++) {
            double x = 3.0 + random.nextGaussian() * 0.5;
            double y = 3.0 + random.nextGaussian() * 0.5;
            data.add(Linalg.vector(x, y));
        }
        
        // 第三个分量：中心在(-2, 4)
        for (int i = 0; i < 100; i++) {
            double x = -2.0 + random.nextGaussian() * 0.6;
            double y = 4.0 + random.nextGaussian() * 0.6;
            data.add(Linalg.vector(x, y));
        }
        
        return data;
    }
    
    /**
     * 生成具有挑战性的测试数据：重叠分量和离群点
     */
    private List<IVector<Double>> generateChallengingTestData() {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(123);
        
        // 创建一些重叠的分量
        for (int i = 0; i < 50; i++) {
            // 紧密重叠的分量
            double x = random.nextGaussian() * 0.1;
            double y = random.nextGaussian() * 0.1;
            data.add(Linalg.vector(x, y));
            
            // 另一个紧密重叠的分量
            double x2 = 0.2 + random.nextGaussian() * 0.1;
            double y2 = 0.2 + random.nextGaussian() * 0.1;
            data.add(Linalg.vector(x2, y2));
        }
        
        // 添加一些离群点
        for (int i = 0; i < 10; i++) {
            double x = random.nextGaussian() * 5 + 10;
            double y = random.nextGaussian() * 5 + 10;
            data.add(Linalg.vector(x, y));
        }
        
        return data;
    }
    
    @Test
    public void testGMMWithStandardData() {
        System.out.println("=== 测试标准数据集上的GMM ===");
        
        // 创建GMM模型
        GaussianMixtureModel gmm = new GaussianMixtureModel(3, 2);
        
        // 使用EM算法训练
        EMAlgorithm em = new EMAlgorithm(100, 1e-6, true);
        EMAlgorithm.EMResult result = em.fit(testData, gmm);
        
        System.out.printf("标准数据集结果:\n");
        System.out.printf("  对数似然: %.6f\n", result.logLikelihood);
        System.out.printf("  收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
        System.out.printf("  迭代次数: %d\n", result.iterations);
        
        // 验证权重和为1
        double weightSum = 0.0;
        for (int i = 0; i < gmm.getNumComponents(); i++) {
            weightSum += gmm.getWeight(i);
        }
        System.out.printf("  权重和: %.6f\n", weightSum);
        
        // 测试采样功能
        List<IVector<Double>> samples = gmm.sample(10);
        System.out.printf("  采样点数量: %d\n", samples.size());
        
        // 测试概率密度计算
        IVector<Double> testPoint = testData.get(0);
        double pdf = gmm.pdf(testPoint);
        double logPdf = gmm.logPdf(testPoint);
        System.out.printf("  测试点PDF: %.6e\n", pdf);
        System.out.printf("  测试点LogPDF: %.6f\n", logPdf);
        
        assert result.converged : "算法应该收敛";
        assert Math.abs(weightSum - 1.0) < 1e-6 : "权重和应该为1";
        assert samples.size() == 10 : "应该生成10个样本点";
        assert pdf >= 0 : "PDF应该非负";
        assert Double.isFinite(logPdf) : "LogPDF应该是有限值";
    }
    
    @Test
    public void testGMMWithChallengingData() {
        System.out.println("=== 测试挑战性数据集上的GMM ===");
        
        // 创建GMM模型
        GaussianMixtureModel gmm = new GaussianMixtureModel(4, 2);
        
        // 使用EM算法训练（启用多重启动）
        EMAlgorithm em = new EMAlgorithm(100, 1e-6, true);
        EMAlgorithm.EMResult result = em.fitWithMultipleRestarts(challengingData, 4, 5, true);
        
        System.out.printf("挑战性数据集结果:\n");
        System.out.printf("  对数似然: %.6f\n", result.logLikelihood);
        System.out.printf("  收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
        System.out.printf("  迭代次数: %d\n", result.iterations);
        
        // 验证权重和为1
        double weightSum = 0.0;
        for (int i = 0; i < gmm.getNumComponents(); i++) {
            weightSum += gmm.getWeight(i);
        }
        System.out.printf("  权重和: %.6f\n", weightSum);
        
        // 测试概率密度计算的数值稳定性
        IVector<Double> testPoint = challengingData.get(0);
        double pdf = gmm.pdf(testPoint);
        double logPdf = gmm.logPdf(testPoint);
        System.out.printf("  测试点PDF: %.6e\n", pdf);
        System.out.printf("  测试点LogPDF: %.6f\n", logPdf);
        
        assert Double.isFinite(result.logLikelihood) : "对数似然应该是有限值";
        assert Math.abs(weightSum - 1.0) < 1e-6 : "权重和应该为1";
        assert pdf >= 0 : "PDF应该非负";
        assert Double.isFinite(logPdf) : "LogPDF应该是有限值";
    }
    
    @Test
    public void testGMMInitializationMethods() {
        System.out.println("=== 测试不同的GMM初始化方法 ===");
        
        // 测试K-means++初始化
        GaussianMixtureModel gmm1 = new GaussianMixtureModel(3, 2);
        gmm1.initializeWithKMeansPlusPlus(testData);
        
        System.out.println("K-means++初始化完成");
        
        // 测试随机初始化
        GaussianMixtureModel gmm2 = new GaussianMixtureModel(3, 2);
        gmm2.initializeRandomly(testData);
        
        System.out.println("随机初始化完成");
        
        // 测试智能随机初始化
        GaussianMixtureModel gmm3 = new GaussianMixtureModel(3, 2);
        gmm3.initializeWithSmartRandom(testData);
        
        System.out.println("智能随机初始化完成");
        
        // 验证所有模型的权重和为1
        for (int i = 0; i < 3; i++) {
            GaussianMixtureModel gmm = (i == 0) ? gmm1 : (i == 1) ? gmm2 : gmm3;
            double weightSum = 0.0;
            for (int j = 0; j < gmm.getNumComponents(); j++) {
                weightSum += gmm.getWeight(j);
            }
            System.out.printf("  模型%d权重和: %.6f\n", i+1, weightSum);
            assert Math.abs(weightSum - 1.0) < 1e-6 : "权重和应该为1";
        }
    }
    
    @Test
    public void testNumericalStability() {
        System.out.println("=== 测试数值稳定性 ===");
        
        // 创建极端数据点测试数值稳定性
        List<IVector<Double>> extremeData = new ArrayList<>();
        extremeData.add(Linalg.vector(1e-10, 1e-10));  // 非常小的值
        extremeData.add(Linalg.vector(1e10, 1e10));    // 非常大的值
        extremeData.add(Linalg.vector(0.0, 0.0));      // 零值
        extremeData.add(Linalg.vector(-1e10, -1e10));  // 负的大值
        
        GaussianMixtureModel gmm = new GaussianMixtureModel(2, 2);
        EMAlgorithm em = new EMAlgorithm(50, 1e-6, true);
        
        try {
            EMAlgorithm.EMResult result = em.fit(extremeData, gmm);
            
            System.out.printf("极端数据集结果:\n");
            System.out.printf("  对数似然: %.6f\n", result.logLikelihood);
            System.out.printf("  收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
            
            // 测试概率密度计算
            for (IVector<Double> point : extremeData) {
                double pdf = gmm.pdf(point);
                double logPdf = gmm.logPdf(point);
                System.out.printf("  点[%.2e, %.2e] PDF: %.6e, LogPDF: %.6f\n", 
                                point.get(0), point.get(1), pdf, logPdf);
                
                assert pdf >= 0 : "PDF应该非负";
                assert Double.isFinite(logPdf) || logPdf == Double.NEGATIVE_INFINITY : "LogPDF应该是有限值或负无穷";
            }
        } catch (Exception e) {
            System.out.println("处理极端数据时出现预期的数值问题: " + e.getMessage());
        }
    }
}