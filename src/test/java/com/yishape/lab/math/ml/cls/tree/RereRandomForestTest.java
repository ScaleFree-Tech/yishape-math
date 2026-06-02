package com.yishape.lab.math.ml.cls.tree;

import com.yishape.lab.math.ml.clf.tree.RandomForestResult;
import com.yishape.lab.math.ml.clf.tree.RereRandomForest;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;
import java.util.Map;

/**
 * 随机森林分类器测试类
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class RereRandomForestTest {
    
    private RereRandomForest randomForest;
    private IMatrix<Double> features;
    private String[] labels;
    
    @BeforeEach
    public void setUp() {
        // 创建简单的二分类测试数据
        double[][] data = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0},
            {5.0, 6.0},
            {6.0, 7.0},
            {7.0, 8.0},
            {8.0, 9.0}
        };
        features = Linalg.matrix(data);
        
        labels = new String[]{"A", "A", "A", "A", "B", "B", "B", "B"};
        
        // 初始化随机森林分类器
        randomForest = new RereRandomForest();
        randomForest.setnEstimators(5);
        randomForest.setMaxDepth(3);
        randomForest.setMinSamplesSplit(2);
        randomForest.setMinSamplesLeaf(1);
        randomForest.setBootstrap(true);
        randomForest.setRandomSeed(42L);
    }
    
    @Test
    public void testBinaryClassification() {
        // 训练模型
        randomForest.fit(features, labels);
        
        // 验证模型已训练
        assertTrue(randomForest.isTrained());
        assertNotNull(randomForest.getResult());
        
        // 测试预测
        IVector<Double> testSample = Linalg.vector(new double[]{2.5, 3.5});
        String prediction = randomForest.predict(testSample);
        assertNotNull(prediction);
        assertTrue(prediction.equals("A") || prediction.equals("B"));
        
        // 测试批量预测
        double[][] testData = {
            {1.5, 2.5},
            {6.5, 7.5}
        };
        IMatrix<Double> testMatrix = Linalg.matrix(testData);
        String[] predictions = randomForest.predictBatch(testMatrix);
        assertNotNull(predictions);
        assertEquals(2, predictions.length);
        
        // 验证预测结果
        for (String pred : predictions) {
            assertTrue(pred.equals("A") || pred.equals("B"));
        }
    }
    
    @Test
    public void testMultiClassification() {
        // 创建多分类测试数据
        double[][] multiData = {
            {1.0, 1.0}, {1.5, 1.5}, {2.0, 2.0},  // 类别 A
            {4.0, 4.0}, {4.5, 4.5}, {5.0, 5.0},  // 类别 B
            {7.0, 7.0}, {7.5, 7.5}, {8.0, 8.0}   // 类别 C
        };
        IMatrix<Double> multiFeatures = Linalg.matrix(multiData);
        String[] multiLabels = {"A", "A", "A", "B", "B", "B", "C", "C", "C"};
        
        // 训练模型
        randomForest.fit(multiFeatures, multiLabels);
        
        // 验证模型已训练
        assertTrue(randomForest.isTrained());
        
        // 测试预测
        IVector<Double> testSample = Linalg.vector(new double[]{1.2, 1.2});
        String prediction = randomForest.predict(testSample);
        assertNotNull(prediction);
        assertTrue(prediction.equals("A") || prediction.equals("B") || prediction.equals("C"));
    }
    
    @Test
    public void testModelParameters() {
        // 测试参数设置
        randomForest.setnEstimators(10);
        randomForest.setMaxDepth(5);
        randomForest.setMaxFeatures(2); // 使用int而不是String
        randomForest.setBootstrap(false);
        
        // 验证参数设置
        assertEquals(10, randomForest.getnEstimators());
        assertEquals(5, randomForest.getMaxDepth());
        assertEquals(2, randomForest.getMaxFeatures());
        assertFalse(randomForest.isBootstrap());
        
        // 训练模型
        randomForest.fit(features, labels);
        assertTrue(randomForest.isTrained());
    }
    
    @Test
    public void testFeatureImportance() {
        // 训练模型
        randomForest.fit(features, labels);
        
        // 获取特征重要性
        double[] importance = randomForest.getFeatureImportance();
        assertNotNull(importance);
        assertEquals(features.cols(), importance.length);
        
        // 验证特征重要性值
        for (double imp : importance) {
            assertTrue(imp >= 0.0);
        }
    }
    
    @Test
    public void testProbabilityPrediction() {
        // 训练模型
        randomForest.fit(features, labels);
        
        // 测试概率预测
        IVector<Double> testSample = Linalg.vector(new double[]{3.0, 4.0});
        Map<String, Double> probabilities = randomForest.predictProb(testSample);
        assertNotNull(probabilities);
        
        // 验证概率和为1
        double sum = 0.0;
        for (double prob : probabilities.values()) {
            assertTrue(prob >= 0.0 && prob <= 1.0);
            sum += prob;
        }
        assertEquals(1.0, sum, 0.001);
    }
    
    @Test
    public void testOOBScore() {
        // 启用Bootstrap采样
        randomForest.setBootstrap(true);
        randomForest.fit(features, labels);
        
        // 获取OOB分数
        RandomForestResult result = randomForest.getResult();
        assertNotNull(result);
        
        double oobScore = result.getOobScore();
        assertTrue(oobScore >= 0.0 && oobScore <= 1.0);
    }
    
    @Test
    public void testTrainingAccuracy() {
        // 训练模型
        randomForest.fit(features, labels);
        
        // 获取训练准确率
        RandomForestResult result = randomForest.getResult();
        assertNotNull(result);
        
        double trainAccuracy = result.getTrainAccuracy();
        assertTrue(trainAccuracy >= 0.0 && trainAccuracy <= 1.0);
    }
    
    @Test
    public void testModelSummary() {
        // 训练模型
        randomForest.fit(features, labels);
        
        // 获取模型摘要
        RandomForestResult result = randomForest.getResult();
        assertNotNull(result);
        
        String summary = result.getModelSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("随机森林"));
        assertTrue(summary.contains("树的数量"));
        assertTrue(summary.contains("特征数量"));
    }
    
    @Test
    public void testEmptyData() {
        // 测试空数据的处理
        double[][] emptyData = {};
        String[] emptyLabels = {};
        
        assertThrows(IllegalArgumentException.class, () -> {
            randomForest.fit(Linalg.matrix(emptyData), emptyLabels);
        });
    }
    
    @Test
    public void testMismatchedDataLabels() {
        // 测试数据和标签不匹配的情况
        String[] wrongLabels = {"A", "B"}; // 只有2个标签，但有8个样本
        
        assertThrows(IllegalArgumentException.class, () -> {
            randomForest.fit(features, wrongLabels);
        });
    }
}