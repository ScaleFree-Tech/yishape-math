package com.yishape.lab.math.ml.clustering;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.stats.model.GaussianMixtureModel;
import com.yishape.lab.math.stats.model.EMAlgorithm;
import com.yishape.lab.math.stats.distribution.multiv.MultivariateNormalDistribution;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class DebugClusteringTest {
    
    @Test
    public void testClusteringBehavior() {
        // 生成简单的测试数据
        List<IVector<Double>> data = generateSimpleTestData();
        
        // 打印前5个和后5个数据点
        System.out.println("前5个数据点:");
        for (int i = 0; i < 5; i++) {
            IVector<Double> point = data.get(i);
            System.out.printf("  点 %d: (%.2f, %.2f)\n", i, point.get(0), point.get(1));
        }
        
        System.out.println("后5个数据点:");
        for (int i = data.size() - 5; i < data.size(); i++) {
            IVector<Double> point = data.get(i);
            System.out.printf("  点 %d: (%.2f, %.2f)\n", i, point.get(0), point.get(1));
        }
        
        // 测试K-Means算法
        System.out.println("\n=== K-Means算法测试 ===");
        KMeansPlusPlus kmeans = new KMeansPlusPlus();
        // 设置聚类数量为2
        kmeans.setParameters(Map.of("n_clusters", 2));
        kmeans.fit(data);
        
        System.out.println("K-Means聚类结果:");
        int[] kmeansLabels = kmeans.getLabels();
        int numClusters = kmeans.getNumClusters();
        int[] kmeansCounts = new int[numClusters];
        for (int label : kmeansLabels) {
            kmeansCounts[label]++;
        }
        for (int i = 0; i < numClusters; i++) {
            System.out.printf("  聚类%d: %d个数据点\n", i, kmeansCounts[i]);
        }
        
        // 测试GMM算法 - 使用正确的训练方式
        System.out.println("\n=== GMM算法详细测试 ===");
        
        // 创建GMM模型并使用正确的fit方法
        GaussianMixtureModel gmmModel = new GaussianMixtureModel(2, 2);
        
        // 使用详细的EM算法
        System.out.println("1. 使用GMM.fit()方法训练:");
        EMAlgorithm emAlgorithm = new EMAlgorithm(50, 1e-6, true); // 启用详细输出
        EMAlgorithm.EMResult emResult = gmmModel.fit(data, emAlgorithm);
        
        System.out.printf("EM结果: 对数似然=%.6f, 收敛=%s, 迭代次数=%d\n", 
                        emResult.logLikelihood, emResult.converged ? "是" : "否", emResult.iterations);
        
        // 2. 检查训练后的状态
        System.out.println("\n2. 训练后的GMM状态:");
        for (int i = 0; i < gmmModel.getNumComponents(); i++) {
            System.out.printf("分量 %d:\n", i);
            System.out.printf("  权重: %.4f\n", gmmModel.getWeight(i));
            
            MultivariateNormalDistribution component = gmmModel.getComponent(i);
            IVector<Double> mean = component.getMean();
            System.out.printf("  均值: (%.2f, %.2f)\n", mean.get(0), mean.get(1));
        }
        
        // 3. 检查聚类结果
        System.out.println("\n3. 最终聚类结果:");
        int[] gmmLabels = new int[data.size()];
        int[] gmmCounts = new int[2];
        for (int i = 0; i < data.size(); i++) {
            gmmLabels[i] = gmmModel.predictComponent(data.get(i));
            gmmCounts[gmmLabels[i]]++;
        }
        System.out.printf("  聚类0: %d个数据点\n", gmmCounts[0]);
        System.out.printf("  聚类1: %d个数据点\n", gmmCounts[1]);
        
        // 4. 检查前5个数据点的后验概率
        System.out.println("\n4. 前5个数据点的后验概率:");
        for (int i = 0; i < 5; i++) {
            IVector<Double> point = data.get(i);
            IVector<Double> posteriors = gmmModel.predict(point);
            int predictedComponent = gmmModel.predictComponent(point);
            System.out.printf("  点 %d: 后验概率 [%.4f, %.4f], 预测分量: %d\n", 
                            i, posteriors.get(0), posteriors.get(1), predictedComponent);
        }
        
        // 6. 使用GMMClustering类进行对比
        System.out.println("\n6. GMMClustering类结果对比:");
        GMMClustering gmmClustering = new GMMClustering(50, 1e-6, 10, true, 42L, true); // 启用verbose
        // 设置聚类数量为2
        gmmClustering.setParameters(Map.of("n_components", 2));
        gmmClustering.fit(data);
        
        int[] gmmClusteringLabels = gmmClustering.getLabels();
        int gmmNumClusters = gmmClustering.getNumClusters();
        int[] gmmClusteringCounts = new int[gmmNumClusters];
        for (int label : gmmClusteringLabels) {
            gmmClusteringCounts[label]++;
        }
        for (int i = 0; i < gmmNumClusters; i++) {
            System.out.printf("  聚类%d: %d个数据点\n", i, gmmClusteringCounts[i]);
        }
    }
    
    private List<IVector<Double>> generateSimpleTestData() {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(12345);
        
        // 聚类1：中心在(0, 0)
        for (int i = 0; i < 30; i++) {
            IVector<Double> point = Linalg.zeros(2);
            point.set(0, random.nextGaussian() * 0.3);
            point.set(1, random.nextGaussian() * 0.3);
            data.add(point);
        }
        
        // 聚类2：中心在(3, 3)
        for (int i = 0; i < 30; i++) {
            IVector<Double> point = Linalg.zeros(2);
            point.set(0, 3.0 + random.nextGaussian() * 0.3);
            point.set(1, 3.0 + random.nextGaussian() * 0.3);
            data.add(point);
        }
        
        // 打印前几个和后几个数据点来验证
        System.out.println("前5个数据点（应该在(0,0)附近）:");
        for (int i = 0; i < 5; i++) {
            System.out.println("  " + data.get(i));
        }
        System.out.println("后5个数据点（应该在(3,3)附近）:");
        for (int i = data.size() - 5; i < data.size(); i++) {
            System.out.println("  " + data.get(i));
        }
        
        return data;
    }
    
}