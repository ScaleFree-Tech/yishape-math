package com.yishape.lab.math.stats.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 改进后的EM算法测试类
 */
public class ImprovedEMAlgorithmTest {
    
    private EMAlgorithm emAlgorithm;
    private List<IVector<Double>> testData;
    
    @BeforeEach
    public void setUp() {
        emAlgorithm = new EMAlgorithm(100, 1e-6, true);
        
        // 生成测试数据：三个高斯分布的混合
        testData = generateMixtureData();
    }
    
    /**
     * 生成混合高斯分布的测试数据
     */
    private List<IVector<Double>> generateMixtureData() {
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
            double x = 2.0 + random.nextGaussian() * 0.5;
            double y = 2.0 + random.nextGaussian() * 0.5;
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
    
    @Test
    public void testOriginalKMeansInitialization() {
        System.out.println("=== 测试原始K-means++初始化 ===");
        
        EMAlgorithm.EMResult result = emAlgorithm.fit(testData, 3, true);
        
        System.out.printf("原始方法结果:\n");
        System.out.printf("  对数似然: %.6f\n", result.logLikelihood);
        System.out.printf("  收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
        System.out.printf("  迭代次数: %d\n", result.iterations);
        System.out.println();
    }
    
    @Test
    public void testImprovedKMeansInitialization() {
        System.out.println("=== 测试改进的K-means++初始化 ===");
        
        EMAlgorithm.EMResult result = emAlgorithm.fit(testData, 3, true);
        
        System.out.printf("改进方法结果:\n");
        System.out.printf("  对数似然: %.6f\n", result.logLikelihood);
        System.out.printf("  收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
        System.out.printf("  迭代次数: %d\n", result.iterations);
        System.out.println();
    }
    
    @Test
    public void testMultipleRestarts() {
        System.out.println("=== 测试多重启动策略 ===");
        
        EMAlgorithm.EMResult result = emAlgorithm.fitWithMultipleRestarts(testData, 3, 5, true);
        
        System.out.printf("多重启动结果:\n");
        System.out.printf("  对数似然: %.6f\n", result.logLikelihood);
        System.out.printf("  收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
        System.out.printf("  迭代次数: %d\n", result.iterations);
        System.out.println();
    }
    
    @Test
    public void testComparisonBetweenMethods() {
        System.out.println("=== 比较不同方法的性能 ===");
        
        // 测试原始多次初始化方法
        System.out.println("1. 原始多次初始化方法:");
        EMAlgorithm.EMResult originalResult = emAlgorithm.fitWithMultipleInitializations(testData, 3, 5, true);
        System.out.printf("   对数似然: %.6f, 收敛: %s, 迭代: %d\n", 
                        originalResult.logLikelihood, 
                        originalResult.converged ? "是" : "否", 
                        originalResult.iterations);
        
        // 测试新的多重启动方法（使用改进的随机初始化）
        System.out.println("2. 新的多重启动方法（智能随机初始化）:");
        EMAlgorithm.EMResult newResult = emAlgorithm.fitWithMultipleRestarts(testData, 3, 5, false);
        System.out.printf("   对数似然: %.6f, 收敛: %s, 迭代: %d\n", 
                        newResult.logLikelihood, 
                        newResult.converged ? "是" : "否", 
                        newResult.iterations);
        
        // 比较结果
        System.out.println("3. 性能比较:");
        double improvement = newResult.logLikelihood - originalResult.logLikelihood;
        System.out.printf("   对数似然改进: %.6f\n", improvement);
        System.out.printf("   改进百分比: %.2f%%\n", 
                        Math.abs(originalResult.logLikelihood) > 1e-10 ? 
                        (improvement / Math.abs(originalResult.logLikelihood)) * 100 : 0);
        
        if (improvement > 0) {
            System.out.println("   ✅ 新方法表现更好");
        } else if (improvement < -1e-6) {
            System.out.println("   ❌ 新方法表现较差");
        } else {
            System.out.println("   ➖ 两种方法表现相当");
        }
    }
    
    @Test
    public void testNumericalStability() {
        System.out.println("=== 测试数值稳定性改进 ===");
        
        // 创建一个更具挑战性的数据集
        List<IVector<Double>> challengingData = generateChallengingData();
        
        System.out.println("使用具有挑战性的数据集测试数值稳定性...");
        
        EMAlgorithm.EMResult result = emAlgorithm.fitWithMultipleRestarts(challengingData, 4, 3, true);
        
        System.out.printf("挑战性数据结果:\n");
        System.out.printf("  对数似然: %.6f\n", result.logLikelihood);
        System.out.printf("  收敛状态: %s\n", result.converged ? "收敛" : "未收敛");
        System.out.printf("  迭代次数: %d\n", result.iterations);
        
        if (Double.isFinite(result.logLikelihood) && result.logLikelihood > Double.NEGATIVE_INFINITY) {
            System.out.println("  ✅ 数值稳定性良好");
        } else {
            System.out.println("  ❌ 存在数值稳定性问题");
        }
    }
    
    /**
     * 生成具有挑战性的数据集（用于测试数值稳定性）
     */
    private List<IVector<Double>> generateChallengingData() {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(123);
        
        // 创建一些重叠的分量和离群点
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
}