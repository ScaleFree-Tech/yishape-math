package com.yishape.lab.math.ml.clustering;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 聚类接口兼容性测试
 * 验证不同聚类算法实现的接口一致性
 * 
 * @author reremouse
 */
public class ClusteringInterfaceTest {
    
    private List<IVector<Double>> testData;
    private List<IClustering> algorithms;
    
    @BeforeEach
    void setUp() {
        // 创建测试数据
        testData = generateSimpleTestData();
        
        // 初始化所有聚类算法
        algorithms = new ArrayList<>();
        
        KMeansPlusPlus kMeans = new KMeansPlusPlus(12345L);
        GMMClustering gmm = new GMMClustering();
        
        algorithms.add(kMeans);
        algorithms.add(gmm);
        
        // 为所有算法设置相同的基本参数
        Map<String, Object> params = new HashMap<>();
        params.put("numClusters", 2);
        
        for (IClustering algorithm : algorithms) {
            algorithm.setParameters(params);
        }
    }
    
    /**
     * 生成简单的测试数据：两个聚类
     */
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
        
        return data;
    }
    
    @Test
    void testAllAlgorithmsImplementInterface() {
        for (IClustering algorithm : algorithms) {
            assertNotNull(algorithm);
            assertTrue(algorithm instanceof IClustering);
            
            // 验证基本方法存在
            assertNotNull(algorithm.getAlgorithmName());
            assertNotNull(algorithm.getParameters());
            
            // 验证算法名称不为空
            assertFalse(algorithm.getAlgorithmName().trim().isEmpty());
        }
    }
    
    @Test
    void testClusteringResultConsistency() {
        for (IClustering algorithm : algorithms) {
            algorithm.fit(testData);
            
            assertNotNull(algorithm.getClusterCenters(), "聚类中心不应为空");
            assertEquals(2, algorithm.getNumClusters(), "聚类数量应为2");
            assertEquals(testData.size(), algorithm.getLabels().length, "分配数组长度应等于数据点数量");
            assertEquals(2, algorithm.getDimension(), "数据维度应为2");
            
            // 验证聚类中心
            List<IVector<Double>> centers = algorithm.getClusterCenters();
            assertNotNull(centers, "聚类中心不应为空");
            assertEquals(2, centers.size(), "应有2个聚类中心");
            
            // 验证每个聚类都有数据点
            int[] labels = algorithm.getLabels();
            for (int cluster = 0; cluster < 2; cluster++) {
                List<Integer> pointsInCluster = new ArrayList<>();
                for (int i = 0; i < labels.length; i++) {
                    if (labels[i] == cluster) {
                        pointsInCluster.add(i);
                    }
                }
                assertNotNull(pointsInCluster, "聚类" + cluster + "的数据点列表不应为空");
                assertTrue(pointsInCluster.size() > 0, "聚类" + cluster + "应包含至少一个数据点");
            }
            
            // 验证标签范围
            for (int i = 0; i < labels.length; i++) {
                int label = labels[i];
                assertTrue(label >= 0 && label < 2, 
                    "算法 " + algorithm.getAlgorithmName() + " 在索引 " + i + " 处的标签 " + label + " 超出范围");
            }
            
            // 验证惯性值
            assertTrue(algorithm.getInertia() >= 0, 
                "算法 " + algorithm.getAlgorithmName() + " 的惯性值为负数");
        }
    }
    
    @Test
    void testParameterManagement() {
        for (IClustering algorithm : algorithms) {
            // 获取初始参数
            Map<String, Object> initialParams = algorithm.getParameters();
            assertNotNull(initialParams, "算法 " + algorithm.getAlgorithmName() + " 返回null参数");
            
            // 设置新参数
            Map<String, Object> newParams = new HashMap<>();
            newParams.put("numClusters", 3);
            newParams.put("testParam", "testValue");
            
            algorithm.setParameters(newParams);
            
            // 验证参数已更新
            Map<String, Object> updatedParams = algorithm.getParameters();
            assertTrue(updatedParams.containsKey("numClusters"), 
                "算法 " + algorithm.getAlgorithmName() + " 未保存numClusters参数");
            assertEquals(3, updatedParams.get("numClusters"), 
                "算法 " + algorithm.getAlgorithmName() + " numClusters参数值不正确");
        }
    }
    
    @Test
    void testQualityEvaluation() {
        for (IClustering algorithm : algorithms) {
            algorithm.fit(testData);
            ClusteringMetrics metrics = algorithm.evaluateQuality(testData);
            
            assertNotNull(metrics, "算法 " + algorithm.getAlgorithmName() + " 返回null评估指标");
            assertTrue(metrics.getInertia() >= 0, 
                "算法 " + algorithm.getAlgorithmName() + " 惯性值为负数");
            assertTrue(Double.isFinite(metrics.getSilhouetteScore()), 
                "算法 " + algorithm.getAlgorithmName() + " 轮廓系数不是有限数");
            assertTrue(metrics.getCalinskiHarabaszIndex() >= 0, 
                "算法 " + algorithm.getAlgorithmName() + " CH指数为负数");
            assertTrue(metrics.getDaviesBouldinIndex() >= 0, 
                "算法 " + algorithm.getAlgorithmName() + " DB指数为负数");
        }
    }
    
    @ParameterizedTest
    @ValueSource(ints = {1, 2, 3, 5})
    void testDifferentClusterNumbers(int numClusters) {
        // 跳过聚类数大于数据点数的情况
        if (numClusters > testData.size()) {
            return;
        }
        
        for (IClustering algorithm : algorithms) {
            Map<String, Object> params = new HashMap<>();
            params.put("numClusters", numClusters);
            algorithm.setParameters(params);
            
            try {
                algorithm.fit(testData);
                assertEquals(numClusters, algorithm.getNumClusters(), 
                    "算法 " + algorithm.getAlgorithmName() + " 聚类数量设置失败");
                
                // 验证所有聚类标签都在有效范围内
                for (int label : algorithm.getLabels()) {
                    assertTrue(label >= 0 && label < numClusters, 
                        "算法 " + algorithm.getAlgorithmName() + " 标签超出范围");
                }
                
            } catch (Exception e) {
                fail("算法 " + algorithm.getAlgorithmName() + " 在聚类数为 " + numClusters + " 时失败: " + e.getMessage());
            }
        }
    }
    
    @Test
    void testClusteringResultUtilityMethods() {
        for (IClustering algorithm : algorithms) {
            algorithm.fit(testData);
            
            // 测试聚类大小
            int[] labels = algorithm.getLabels();
            int[] clusterSizes = new int[algorithm.getNumClusters()];
            for (int label : labels) {
                clusterSizes[label]++;
            }
            assertEquals(algorithm.getNumClusters(), clusterSizes.length, 
                "算法 " + algorithm.getAlgorithmName() + " 聚类大小数组长度不正确");
            
            int totalSize = 0;
            for (int size : clusterSizes) {
                assertTrue(size >= 0, "算法 " + algorithm.getAlgorithmName() + " 聚类大小为负数");
                totalSize += size;
            }
            assertEquals(testData.size(), totalSize, 
                "算法 " + algorithm.getAlgorithmName() + " 聚类大小总和不等于数据点数");
            
            // 测试获取聚类中的数据点
            for (int cluster = 0; cluster < algorithm.getNumClusters(); cluster++) {
                List<Integer> indices = new ArrayList<>();
                for (int i = 0; i < labels.length; i++) {
                    if (labels[i] == cluster) {
                        indices.add(i);
                    }
                }
                assertNotNull(indices, "算法 " + algorithm.getAlgorithmName() + " 返回null索引列表");
                assertEquals(clusterSizes[cluster], indices.size(), 
                    "算法 " + algorithm.getAlgorithmName() + " 聚类 " + cluster + " 的索引数量不匹配");
                
                // 验证索引的有效性
                for (int index : indices) {
                    assertTrue(index >= 0 && index < testData.size(), 
                        "算法 " + algorithm.getAlgorithmName() + " 索引超出范围");
                    assertEquals(cluster, labels[index], 
                        "算法 " + algorithm.getAlgorithmName() + " 索引与标签不匹配");
                }
            }
        }
    }
    
    @Test
    void testErrorHandling() {
        for (IClustering algorithm : algorithms) {
            // 测试空数据处理
            assertThrows(IllegalArgumentException.class, () -> {
                algorithm.fit(new ArrayList<>());
            }, "算法 " + algorithm.getAlgorithmName() + " 未正确处理空数据");
            
            // 测试null数据处理
            assertThrows(IllegalArgumentException.class, () -> {
                algorithm.fit((List<IVector<Double>>) null);
            }, "算法 " + algorithm.getAlgorithmName() + " 未正确处理null数据");
        }
    }
    
    @Test
    void testAlgorithmSpecificBehavior() {
        // 测试K-Means特定行为
        KMeansPlusPlus kMeans = new KMeansPlusPlus(12345L);
        Map<String, Object> params = new HashMap<>();
        params.put("numClusters", 2);
        kMeans.setParameters(params);
        
        kMeans.fit(testData);
        assertTrue(kMeans.isConverged(), "K-Means应该总是收敛");
        
        // 测试K-Means预测功能
        assertDoesNotThrow(() -> {
            kMeans.predict(testData);
        });
        
        // 测试GMM特定行为
        GMMClustering gmm = new GMMClustering();
        gmm.setParameters(params);
        
        gmm.fit(testData);
        
        // GMM应该支持预测
        assertDoesNotThrow(() -> {
            gmm.predict(testData);
        });
        
        // 测试单点预测
        IVector<Double> testPoint = Linalg.zeros(2);
        testPoint.set(0, 1.5);
        testPoint.set(1, 1.5);
        
        assertDoesNotThrow(() -> {
            int prediction = gmm.predict(testPoint);
            assertTrue(prediction >= 0 && prediction < 2);
        });
    }
}