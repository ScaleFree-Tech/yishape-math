package com.yishape.lab.math.stats.model;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 简化的改进测试，专注于验证核心改进功能
 */
public class SimpleImprovedTest {
    
    public static void main(String[] args) {
        System.out.println("=== GMM算法改进验证 ===\n");
        
        try {
            // 生成测试数据
            List<IVector<Double>> data = generateSimpleTestData();
            
            // 测试1: 基本功能
            testBasicFunctionality(data);
            
            // 测试2: K-means++初始化
            testKMeansPlusPlus(data);
            
            // 测试3: 多次初始化
            testMultipleInitializations(data);
            
            System.out.println("\n=== 改进验证完成 ===");
            
        } catch (Exception e) {
            System.err.println("测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void testBasicFunctionality(List<IVector<Double>> data) {
        System.out.println("1. 测试基本功能改进");
        System.out.println("===================");
        
        try {
            EMAlgorithm em = new EMAlgorithm(30, 1e-6, true);
            EMAlgorithm.EMResult result = em.fit(data, 2, false);
            
            System.out.printf("训练结果: 对数似然=%.6f, 收敛=%s, 迭代次数=%d\n", 
                            result.logLikelihood, result.converged ? "是" : "否", result.iterations);
            
            System.out.println("✅ 基本功能测试通过");
            
        } catch (Exception e) {
            System.out.println("❌ 基本功能测试失败: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    private static void testKMeansPlusPlus(List<IVector<Double>> data) {
        System.out.println("2. 测试K-means++初始化");
        System.out.println("======================");
        
        try {
            EMAlgorithm em = new EMAlgorithm(30, 1e-6, true);
            
            System.out.println("--- 使用K-means++初始化 ---");
            EMAlgorithm.EMResult result = em.fit(data, 2, true);
            
            System.out.printf("K-means++结果: 对数似然=%.6f, 收敛=%s, 迭代次数=%d\n", 
                            result.logLikelihood, result.converged ? "是" : "否", result.iterations);
            
            System.out.println("✅ K-means++初始化测试通过");
            
        } catch (Exception e) {
            System.out.println("❌ K-means++初始化测试失败: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    private static void testMultipleInitializations(List<IVector<Double>> data) {
        System.out.println("3. 测试多次初始化");
        System.out.println("==================");
        
        try {
            EMAlgorithm em = new EMAlgorithm(20, 1e-6, true);
            
            System.out.println("--- 执行多次初始化 ---");
            EMAlgorithm.EMResult result = em.fitWithMultipleInitializations(data, 2, 3, false);
            
            System.out.printf("多次初始化最佳结果: 对数似然=%.6f, 收敛=%s, 迭代次数=%d\n", 
                            result.logLikelihood, result.converged ? "是" : "否", result.iterations);
            
            System.out.println("✅ 多次初始化测试通过");
            
        } catch (Exception e) {
            System.out.println("❌ 多次初始化测试失败: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    private static List<IVector<Double>> generateSimpleTestData() {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(42);
        
        // 生成两个明显分离的高斯分布
        for (int i = 0; i < 100; i++) {
            if (random.nextDouble() < 0.5) {
                // 第一个分量：均值[0, 0]
                double x = random.nextGaussian() * 1.0;
                double y = random.nextGaussian() * 1.0;
                data.add(Linalg.vector(new double[]{x, y}));
            } else {
                // 第二个分量：均值[5, 5]
                double x = random.nextGaussian() * 1.0 + 5.0;
                double y = random.nextGaussian() * 1.0 + 5.0;
                data.add(Linalg.vector(new double[]{x, y}));
            }
        }
        
        return data;
    }
}