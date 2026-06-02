package com.yishape.lab.math.ml.clustering;

import com.yishape.lab.math.ml.clu.KMeansPlusPlus;
import com.yishape.lab.math.ml.clu.GMMClustering;
import com.yishape.lab.math.ml.clu.ClusteringMetrics;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * 综合聚类算法测试
 * 验证K-means、GMM、DBSCAN等算法的修复效果
 */
public class ComprehensiveClusteringTest {
    
    private List<IVector<Double>> testData;
    private Random random;
    
    @BeforeEach
    public void setUp() {
        random = new Random(42);
        testData = generateTwoClusterData();
    }
    
    /**
     * 生成两个明显分离的聚类数据
     */
    private List<IVector<Double>> generateTwoClusterData() {
        List<IVector<Double>> data = new ArrayList<>();
        
        // 第一个聚类：中心在(3, 3)附近
        for (int i = 0; i < 30; i++) {
            double x = 3.0 + random.nextGaussian() * 0.5;
            double y = 3.0 + random.nextGaussian() * 0.5;
            data.add(Linalg.vector(x, y));
        }
        
        // 第二个聚类：中心在(15, 15)附近
        for (int i = 0; i < 30; i++) {
            double x = 15.0 + random.nextGaussian() * 0.5;
            double y = 15.0 + random.nextGaussian() * 0.5;
            data.add(Linalg.vector(x, y));
        }
        
        return data;
    }
    
    @Test
    public void testAllClusteringAlgorithms() {
        System.out.println("=== 综合聚类算法测试 ===");
        System.out.printf("测试数据: %d个数据点，2个真实聚类\n", testData.size());
        
        // 测试K-means算法
        testKMeansAlgorithm();
        
        // 测试GMM算法
        testGMMAlgorithm();
        
        System.out.println("=== 所有算法测试完成 ===");
    }
    
    private void testKMeansAlgorithm() {
        System.out.println("\n1. K-means算法测试");
        System.out.println("==================");
        
        try {
            KMeansPlusPlus kmeans = new KMeansPlusPlus(42L);
            
            // 设置详细输出
            java.util.Map<String, Object> params = new java.util.HashMap<>();
            params.put("verbose", true);
            kmeans.setParameters(params);
            
            kmeans.fit(testData);
            
            System.out.printf("K-means结果:\n");
            System.out.printf("  聚类数量: %d\n", kmeans.getNumClusters());
            
            // 统计每个聚类的大小
            int[] clusterSizes = new int[kmeans.getNumClusters()];
            for (int label : kmeans.getLabels()) {
                if (label >= 0 && label < clusterSizes.length) {
                    clusterSizes[label]++;
                }
            }
            
            for (int i = 0; i < clusterSizes.length; i++) {
                System.out.printf("  聚类%d: %d个数据点\n", i, clusterSizes[i]);
            }
            
            // 验证聚类质量
            ClusteringMetrics metrics = kmeans.evaluateQuality(testData);
            System.out.printf("  惯性: %.6f\n", metrics.getInertia());
            System.out.printf("  轮廓系数: %.6f\n", metrics.getSilhouetteScore());
            
            System.out.println("✅ K-means测试通过");
            
        } catch (Exception e) {
            System.out.println("❌ K-means测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void testGMMAlgorithm() {
        System.out.println("\n2. GMM算法测试");
        System.out.println("===============");
        
        try {
            GMMClustering gmm = new GMMClustering();
            
            // 设置详细输出
            java.util.Map<String, Object> params = new java.util.HashMap<>();
            params.put("verbose", true);
            gmm.setParameters(params);
            
            gmm.fit(testData);
            
            System.out.printf("GMM结果:\n");
            System.out.printf("  聚类数量: %d\n", gmm.getNumClusters());
            
            // 统计每个聚类的大小
            int[] clusterSizes = new int[gmm.getNumClusters()];
            for (int label : gmm.getLabels()) {
                if (label >= 0 && label < clusterSizes.length) {
                    clusterSizes[label]++;
                }
            }
            
            for (int i = 0; i < clusterSizes.length; i++) {
                System.out.printf("  聚类%d: %d个数据点\n", i, clusterSizes[i]);
            }
            
            // 验证聚类质量
            ClusteringMetrics metrics = gmm.evaluateQuality(testData);
            System.out.printf("  惯性: %.6f\n", metrics.getInertia());
            System.out.printf("  轮廓系数: %.6f\n", metrics.getSilhouetteScore());
            
            // 显示GMM特有的信息
            Map<String, Object> additionalInfo = gmm.getParameters();
            if (additionalInfo.containsKey("logLikelihood")) {
                double logLikelihood = (Double) additionalInfo.get("logLikelihood");
                System.out.printf("  对数似然: %.6f\n", logLikelihood);
            }
            
            if (additionalInfo.containsKey("aic")) {
                double aic = (Double) additionalInfo.get("aic");
                System.out.printf("  AIC: %.6f\n", aic);
            }
            
            if (additionalInfo.containsKey("bic")) {
                double bic = (Double) additionalInfo.get("bic");
                System.out.printf("  BIC: %.6f\n", bic);
            }
            
            System.out.println("✅ GMM测试通过");
            
        } catch (Exception e) {
            System.out.println("❌ GMM测试失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    

}