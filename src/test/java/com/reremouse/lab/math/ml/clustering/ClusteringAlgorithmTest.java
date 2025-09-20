package com.reremouse.lab.math.ml.clustering;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.linalg.IVector;
import com.reremouse.lab.math.linalg.Linalg;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 聚类算法测试类
 * 测试通用聚类接口和具体实现
 * 
 * @author reremouse
 */
public class ClusteringAlgorithmTest {
    
    private List<IVector<Double>> testData;
    private IMatrix<Double> testMatrix;
    private KMeansPlusPlus kMeans;
    private GMMClustering gmm;
    
    @BeforeEach
    void setUp() {
        // 创建测试数据：三个明显分离的聚类
        testData = generateTestData();
        testMatrix = convertToMatrix(testData);
        
        // 初始化算法
        kMeans = new KMeansPlusPlus(12345L); // 固定种子确保可重复性
        gmm = new GMMClustering();
        
        // 设置参数
        Map<String, Object> params = new HashMap<>();
        params.put("numClusters", 3);
        kMeans.setParameters(params);
        gmm.setParameters(params);
    }
    
    /**
     * 生成测试数据：三个高斯分布的聚类
     */
    private List<IVector<Double>> generateTestData() {
        List<IVector<Double>> data = new ArrayList<>();
        Random random = new Random(12345);
        
        // 聚类1：中心在(0, 0)
        for (int i = 0; i < 50; i++) {
            IVector<Double> point = Linalg.zeros(2);
            point.set(0, random.nextGaussian() * 0.5);
            point.set(1, random.nextGaussian() * 0.5);
            data.add(point);
        }
        
        // 聚类2：中心在(5, 0)
        for (int i = 0; i < 50; i++) {
            IVector<Double> point = Linalg.zeros(2);
            point.set(0, 5.0 + random.nextGaussian() * 0.5);
            point.set(1, random.nextGaussian() * 0.5);
            data.add(point);
        }
        
        // 聚类3：中心在(2.5, 4)
        for (int i = 0; i < 50; i++) {
            IVector<Double> point = Linalg.zeros(2);
            point.set(0, 2.5 + random.nextGaussian() * 0.5);
            point.set(1, 4.0 + random.nextGaussian() * 0.5);
            data.add(point);
        }
        
        return data;
    }
    
    /**
     * 将向量列表转换为矩阵
     */
    private IMatrix<Double> convertToMatrix(List<IVector<Double>> data) {
        int n = data.size();
        int d = data.get(0).size();
        IMatrix<Double> matrix = Linalg.zeros(n, d);
        
        for (int i = 0; i < n; i++) {
            IVector<Double> row = data.get(i);
            for (int j = 0; j < d; j++) {
                matrix.set(i, j, row.get(j));
            }
        }
        
        return matrix;
    }
    
    @Test
    void testKMeansClusteringWithVectorList() {
        kMeans.fit(testData);
        
        assertNotNull(kMeans.getClusterCenters());
        assertEquals(3, kMeans.getNumClusters());
        assertEquals(2, kMeans.getDimension());
        assertEquals(150, kMeans.getLabels().length);
        assertEquals(3, kMeans.getClusterCenters().size());
        assertTrue(kMeans.isConverged());
        assertTrue(kMeans.getInertia() > 0);
        
        // 验证标签范围
        for (int label : kMeans.getLabels()) {
            assertTrue(label >= 0 && label < 3);
        }
    }
    
    @Test
    void testKMeansClusteringWithMatrix() {
        kMeans.fit(testMatrix);
        
        assertNotNull(kMeans.getClusterCenters());
        assertEquals(3, kMeans.getNumClusters());
        assertEquals(2, kMeans.getDimension());
        assertEquals(150, kMeans.getLabels().length);
        assertEquals(3, kMeans.getClusterCenters().size());
        assertTrue(kMeans.isConverged());
        assertTrue(kMeans.getInertia() > 0);
    }
    
    @Test
    void testGMMClustering() {
        gmm.fit(testData);
        
        assertNotNull(gmm.getClusterCenters());
        assertEquals(3, gmm.getNumClusters());
        assertEquals(2, gmm.getDimension());
        assertEquals(150, gmm.getLabels().length);
        assertEquals(3, gmm.getClusterCenters().size());
        assertTrue(gmm.getInertia() >= 0);
        
        // 验证标签范围
        for (int label : gmm.getLabels()) {
            assertTrue(label >= 0 && label < 3);
        }
    }
    
    @Test
    void testClusteringMetrics() {
        kMeans.fit(testData);
        ClusteringMetrics metrics = kMeans.evaluateQuality(testData);
        
        assertNotNull(metrics);
        assertTrue(metrics.getInertia() > 0);
        assertTrue(metrics.getSilhouetteScore() > 0); // 好的聚类应该有正的轮廓系数
        assertTrue(metrics.getCalinskiHarabaszIndex() > 0);
        assertTrue(metrics.getDaviesBouldinIndex() > 0);
    }
    
    @Test
    void testAlgorithmNames() {
        assertEquals("K-Means++", kMeans.getAlgorithmName());
        assertEquals("Gaussian Mixture Model", gmm.getAlgorithmName());
    }
    
    @Test
    void testParameterManagement() {
        // 测试K-Means参数
        Map<String, Object> kMeansParams = kMeans.getParameters();
        assertTrue(kMeansParams.containsKey("numClusters"));
        assertTrue(kMeansParams.containsKey("maxIterations"));
        assertTrue(kMeansParams.containsKey("convergenceThreshold"));
        assertEquals("K-Means++", kMeansParams.get("algorithmName"));
        
        // 测试GMM参数
        Map<String, Object> gmmParams = gmm.getParameters();
        assertTrue(gmmParams.containsKey("numClusters"));
        assertTrue(gmmParams.containsKey("maxIterations"));
        assertTrue(gmmParams.containsKey("convergenceThreshold"));
        assertEquals("Gaussian Mixture Model", gmmParams.get("algorithmName"));
        
        // 测试参数设置
        Map<String, Object> newParams = new HashMap<>();
        newParams.put("numClusters", 5);
        kMeans.setParameters(newParams);
        assertEquals(5, kMeans.getParameters().get("numClusters"));
    }
    
    @Test
    void testClusteringResultMethods() {
        kMeans.fit(testData);
        
        // 测试聚类大小计算
        int[] clusterSizes = new int[3];
        int[] labels = kMeans.getLabels();
        for (int label : labels) {
            clusterSizes[label]++;
        }
        assertEquals(3, clusterSizes.length);
        
        int totalSize = 0;
        for (int size : clusterSizes) {
            assertTrue(size > 0);
            totalSize += size;
        }
        assertEquals(150, totalSize);
        
        // 测试获取聚类中的数据点索引
        for (int cluster = 0; cluster < 3; cluster++) {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < labels.length; i++) {
                if (labels[i] == cluster) {
                    indices.add(i);
                }
            }
            assertNotNull(indices);
            assertEquals(clusterSizes[cluster], indices.size());
            
            // 验证索引的有效性
            for (int index : indices) {
                assertTrue(index >= 0 && index < 150);
                assertEquals(cluster, labels[index]);
            }
        }
    }
    
    @Test
    void testEmptyDataHandling() {
        List<IVector<Double>> emptyData = new ArrayList<>();
        
        assertThrows(IllegalArgumentException.class, () -> {
            kMeans.fit(emptyData);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            gmm.fit(emptyData);
        });
    }
    
    @Test
    void testInvalidClusterNumber() {
        Map<String, Object> params = new HashMap<>();
        params.put("numClusters", 0);
        
        assertThrows(IllegalArgumentException.class, () -> {
            kMeans.setParameters(params);
            kMeans.fit(testData);
        });
    }
    
    @Test
    void testClusteringConsistency() {
        // 使用相同的种子应该产生一致的结果
        KMeansPlusPlus kMeans1 = new KMeansPlusPlus(12345L);
        KMeansPlusPlus kMeans2 = new KMeansPlusPlus(12345L);
        
        Map<String, Object> params = new HashMap<>();
        params.put("numClusters", 3);
        kMeans1.setParameters(params);
        kMeans2.setParameters(params);
        
        kMeans1.fit(testData);
        kMeans2.fit(testData);
        
        // 结果应该一致（允许标签重新排列）
        assertEquals(kMeans1.getInertia(), kMeans2.getInertia(), 1e-10);
        assertEquals(kMeans1.getNumClusters(), kMeans2.getNumClusters());
    }
    
    @Test
    void testPredictMethod() {
        // 先进行聚类
        kMeans.fit(testData);
        
        // 测试单个数据点预测
        IVector<Double> newPoint = Linalg.zeros(2);
        newPoint.set(0, 5.0);
        newPoint.set(1, 3.0);
        int prediction = kMeans.predict(newPoint);
        assertTrue(prediction >= 0 && prediction < 3);
        
        // 测试批量预测
        IVector<Double> newPoint2 = Linalg.zeros(2);
        newPoint2.set(0, 6.0);
        newPoint2.set(1, 4.0);
        List<IVector<Double>> newData = Arrays.asList(newPoint, newPoint2);
        int[] predictions = kMeans.predict(newData);
        assertEquals(2, predictions.length);
        for (int pred : predictions) {
            assertTrue(pred >= 0 && pred < 3);
        }
    }
    
    @Test
    void testGMMPrediction() {
        // 训练GMM
        gmm.fit(testData);
        
        // 测试预测新数据点
        IVector<Double> newPoint = Linalg.zeros(2);
        newPoint.set(0, 5.0);
        newPoint.set(1, 3.0);
        
        int prediction = gmm.predict(newPoint);
        assertTrue(prediction >= 0 && prediction < 3);
        
        // 测试批量预测
        List<IVector<Double>> newPoints = Arrays.asList(newPoint);
        int[] predictions = gmm.predict(newPoints);
        assertEquals(1, predictions.length);
        assertEquals(prediction, predictions[0]);
    }
}