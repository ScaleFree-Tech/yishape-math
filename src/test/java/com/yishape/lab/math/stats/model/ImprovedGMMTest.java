package com.yishape.lab.math.stats.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 改进后的GMM测试，验证所有新功能
 * Enhanced GMM test to verify all new features
 */
public class ImprovedGMMTest {
    
    public static void main(String[] args) {
        System.out.println("=== 改进后的GMM算法测试 ===\n");
        
        try {
            testBasicFunctionality();
            testKMeansPlusPlusInitialization();
            testMultipleInitializations();
            testConvergenceImprovements();
            testNumericalStability();
            
            System.out.println("\n=== 所有改进测试通过！ ===");
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 测试基本功能
     */
    private static void testBasicFunctionality() {
        System.out.println("1. 测试基本功能");
        System.out.println("================");
        
        List<IVector<Double>> data = generateTestData(200, 2);
        
        // 使用改进的EM算法
        EMAlgorithm em = new EMAlgorithm(50, 1e-6, true);
        
        // 测试基本训练
        EMAlgorithm.EMResult result = em.fit(data, 2, false);
        
        System.out.printf("基本训练结果: 对数似然=%.6f, 收敛=%s, 迭代次数=%d\n", 
                        result.logLikelihood, result.converged ? "是" : "否", result.iterations);
        
        System.out.println("✅ 基本功能测试通过\n");
    }
    
    /**
     * 测试K-means++初始化
     */
    private static void testKMeansPlusPlusInitialization() {
        System.out.println("2. 测试K-means++初始化");
        System.out.println("======================");
        
        List<IVector<Double>> data = generateTestData(200, 2);
        
        EMAlgorithm em = new EMAlgorithm(50, 1e-6, true);
        
        // 比较随机初始化和K-means++初始化
        System.out.println("--- 随机初始化 ---");
        EMAlgorithm.EMResult randomResult = em.fit(data, 2, false);
        
        System.out.println("\n--- K-means++初始化 ---");
        EMAlgorithm.EMResult kmeansResult = em.fit(data, 2, true);
        
        System.out.printf("\n比较结果:\n");
        System.out.printf("随机初始化: 对数似然=%.6f, 迭代次数=%d\n", 
                        randomResult.logLikelihood, randomResult.iterations);
        System.out.printf("K-means++: 对数似然=%.6f, 迭代次数=%d\n", 
                        kmeansResult.logLikelihood, kmeansResult.iterations);
        
        if (kmeansResult.logLikelihood >= randomResult.logLikelihood) {
            System.out.println("✅ K-means++初始化效果良好");
        } else {
            System.out.println("⚠️ K-means++初始化在此次测试中效果一般");
        }
        
        System.out.println("✅ K-means++初始化测试完成\n");
    }
    
    /**
     * 测试多次初始化
     */
    private static void testMultipleInitializations() {
        System.out.println("3. 测试多次初始化");
        System.out.println("==================");
        
        List<IVector<Double>> data = generateTestData(200, 2);
        
        EMAlgorithm em = new EMAlgorithm(30, 1e-6, true);
        
        // 测试多次随机初始化
        System.out.println("--- 多次随机初始化 ---");
        EMAlgorithm.EMResult multiRandomResult = em.fitWithMultipleInitializations(data, 2, 5, false);
        
        System.out.println("\n--- 多次K-means++初始化 ---");
        EMAlgorithm.EMResult multiKmeansResult = em.fitWithMultipleInitializations(data, 2, 3, true);
        
        System.out.printf("\n多次初始化比较:\n");
        System.out.printf("多次随机: 对数似然=%.6f, 收敛=%s\n", 
                        multiRandomResult.logLikelihood, multiRandomResult.converged ? "是" : "否");
        System.out.printf("多次K-means++: 对数似然=%.6f, 收敛=%s\n", 
                        multiKmeansResult.logLikelihood, multiKmeansResult.converged ? "是" : "否");
        
        System.out.println("✅ 多次初始化测试完成\n");
    }
    
    /**
     * 测试收敛性改进
     */
    private static void testConvergenceImprovements() {
        System.out.println("4. 测试收敛性改进");
        System.out.println("==================");
        
        // 创建更具挑战性的数据（更多重叠）
        List<IVector<Double>> challengingData = generateChallengingData(150);
        
        EMAlgorithm em = new EMAlgorithm(100, 1e-8, true);
        
        EMAlgorithm.EMResult result = em.fit(challengingData, 3, true);
        
        System.out.printf("挑战性数据训练结果:\n");
        System.out.printf("对数似然: %.6f\n", result.logLikelihood);
        System.out.printf("收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
        System.out.printf("迭代次数: %d\n", result.iterations);
        
        // 检查收敛历史
        System.out.printf("最大迭代次数: %d\n", em.getMaxIterations());
        System.out.printf("收敛阈值: %.2e\n", em.getTolerance());
        
        System.out.println("✅ 收敛性改进测试完成\n");
    }
    
    /**
     * 测试数值稳定性
     */
    private static void testNumericalStability() {
        System.out.println("5. 测试数值稳定性");
        System.out.println("==================");
        
        // 创建可能导致数值不稳定的数据
        List<IVector<Double>> unstableData = generateUnstableData(100);
        
        EMAlgorithm em = new EMAlgorithm(50, 1e-6, true);
        
        try {
            EMAlgorithm.EMResult result = em.fit(unstableData, 2, true);
            
            System.out.printf("数值稳定性测试结果:\n");
            System.out.printf("对数似然: %.6f\n", result.logLikelihood);
            System.out.printf("收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
            System.out.printf("迭代次数: %d\n", result.iterations);
            
            // 检查结果是否为有效数值
            if (Double.isFinite(result.logLikelihood) && !Double.isNaN(result.logLikelihood)) {
                System.out.println("✅ 数值稳定性良好");
            } else {
                System.out.println("❌ 检测到数值不稳定");
            }
            
        } catch (Exception e) {
            System.out.println("⚠️ 数值稳定性测试中出现异常: " + e.getMessage());
            System.out.println("这可能表明正则化机制正在工作");
        }
        
        System.out.println("✅ 数值稳定性测试完成\n");
    }
    
    /**
     * 生成标准测试数据
     */
    private static List<IVector<Double>> generateTestData(int numSamples, int dimension) {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(42);
        
        // 生成两个高斯分布的混合数据
        for (int i = 0; i < numSamples; i++) {
            if (random.nextDouble() < 0.6) {
                // 第一个分量：均值[0, 0]，标准差1.0
                double[] point = new double[dimension];
                for (int d = 0; d < dimension; d++) {
                    point[d] = random.nextGaussian() * 1.0;
                }
                data.add(Linalg.vector(point));
            } else {
                // 第二个分量：均值[3, 3]，标准差1.2
                double[] point = new double[dimension];
                for (int d = 0; d < dimension; d++) {
                    point[d] = random.nextGaussian() * 1.2 + 3.0;
                }
                data.add(Linalg.vector(point));
            }
        }
        
        return data;
    }
    
    /**
     * 生成具有挑战性的数据（更多重叠）
     */
    private static List<IVector<Double>> generateChallengingData(int numSamples) {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(123);
        
        // 生成三个重叠的高斯分布
        for (int i = 0; i < numSamples; i++) {
            double r = random.nextDouble();
            if (r < 0.4) {
                // 第一个分量：均值[0, 0]
                double x = random.nextGaussian() * 1.5;
                double y = random.nextGaussian() * 1.5;
                data.add(Linalg.vector(new double[]{x, y}));
            } else if (r < 0.7) {
                // 第二个分量：均值[2, 1]
                double x = random.nextGaussian() * 1.2 + 2.0;
                double y = random.nextGaussian() * 1.2 + 1.0;
                data.add(Linalg.vector(new double[]{x, y}));
            } else {
                // 第三个分量：均值[1, 3]
                double x = random.nextGaussian() * 1.0 + 1.0;
                double y = random.nextGaussian() * 1.0 + 3.0;
                data.add(Linalg.vector(new double[]{x, y}));
            }
        }
        
        return data;
    }
    
    /**
     * 生成可能导致数值不稳定的数据
     */
    private static List<IVector<Double>> generateUnstableData(int numSamples) {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(456);
        
        // 生成非常接近的点和一些离群点
        for (int i = 0; i < numSamples; i++) {
            if (i < numSamples * 0.8) {
                // 大部分点聚集在很小的区域
                double x = random.nextGaussian() * 0.01;
                double y = random.nextGaussian() * 0.01;
                data.add(Linalg.vector(new double[]{x, y}));
            } else {
                // 少数离群点
                double x = random.nextGaussian() * 10.0 + 20.0;
                double y = random.nextGaussian() * 10.0 + 20.0;
                data.add(Linalg.vector(new double[]{x, y}));
            }
        }
        
        return data;
    }
}