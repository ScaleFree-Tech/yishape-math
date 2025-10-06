package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.ml.cls.tree.RereXGboost;
import com.yishape.lab.math.ml.cls.tree.XGBoostResult;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * XGBoost分类器测试类
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class XGBoostTest {
    
    private RereXGboost xgboost;
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
            {6.0, 7.0}
        };
        features = Linalg.matrix(data);
        
        labels = new String[]{"A", "A", "A", "B", "B", "B"};
        
        // 初始化XGBoost分类器
        xgboost = new RereXGboost();
        xgboost.setLearningRate(0.1);
        xgboost.setNumEstimators(10);
        xgboost.setMaxDepth(3);
        xgboost.setMinSamplesSplit(2);
        xgboost.setMinSamplesLeaf(1);
    }
    
    @Test
    public void testBinaryClassification() {
        // 训练模型
        XGBoostResult result = (XGBoostResult) xgboost.fit(features, labels);
        
        // 验证结果不为空
        assertNotNull(result);
        assertNotNull(result.getTrees());
        assertTrue(result.getTrees().size() > 0);
        
        // 验证是二分类
        Assertions.assertTrue(result.isBinaryClassification());
        Assertions.assertEquals(2, result.getNumClasses());
        
        // 测试预测
        IVector<Double> testSample = Linalg.vector(new double[]{2.5, 3.5});
        String prediction = xgboost.predict(testSample);
        assertNotNull(prediction);
        assertTrue(prediction.equals("A") || prediction.equals("B"));
        
        // 测试批量预测
        String[] predictions = xgboost.predict(features);
        assertNotNull(predictions);
        assertEquals(features.rows(), predictions.length);
        
        // 测试概率预测
        IMatrix<Double> probabilities = xgboost.predictProba(features);
        assertNotNull(probabilities);
        assertEquals(features.rows(), probabilities.rows());
        assertEquals(2, probabilities.cols()); // 二分类应该有2列概率
        
        // 验证概率和为1
        for (int i = 0; i < probabilities.rows(); i++) {
            double sum = probabilities.get(i, 0).doubleValue() + probabilities.get(i, 1).doubleValue();
            assertEquals(1.0, sum, 1e-6);
        }
    }
    
    @Test
    public void testMultiClassification() {
        // 创建多分类测试数据
        double[][] multiData = {
            {1.0, 1.0}, {1.5, 1.5}, {2.0, 2.0},  // 类别A
            {4.0, 4.0}, {4.5, 4.5}, {5.0, 5.0},  // 类别B
            {7.0, 7.0}, {7.5, 7.5}, {8.0, 8.0}   // 类别C
        };
        IMatrix<Double> multiFeatures = Linalg.matrix(multiData);
        String[] multiLabels = {"A", "A", "A", "B", "B", "B", "C", "C", "C"};
        
        // 训练模型
        XGBoostResult result = (XGBoostResult) xgboost.fit(multiFeatures, multiLabels);
        
        // 验证结果
        assertNotNull(result);
        Assertions.assertFalse(result.isBinaryClassification());
        Assertions.assertEquals(3, result.getNumClasses());
        
        // 测试预测
        String[] predictions = xgboost.predict(multiFeatures);
        assertNotNull(predictions);
        assertEquals(multiFeatures.rows(), predictions.length);
        
        // 测试概率预测
        IMatrix<Double> probabilities = xgboost.predictProba(multiFeatures);
        assertNotNull(probabilities);
        assertEquals(multiFeatures.rows(), probabilities.rows());
        assertEquals(3, probabilities.cols()); // 三分类应该有3列概率
    }
    
    @Test
    public void testModelParameters() {
        // 测试参数设置
        xgboost.setLearningRate(0.05);
        assertEquals(0.05, xgboost.getLearningRate(), 1e-6);
        
        xgboost.setNumEstimators(50);
        assertEquals(50, xgboost.getNumEstimators());
        
        xgboost.setMaxDepth(5);
        assertEquals(5, xgboost.getMaxDepth());
        
        xgboost.setLambda(0.1);
        assertEquals(0.1, xgboost.getLambda(), 1e-6);
        
        xgboost.setAlpha(0.05);
        assertEquals(0.05, xgboost.getAlpha(), 1e-6);
    }
    
    @Test
    public void testFeatureImportance() {
        // 训练模型
        XGBoostResult result = (XGBoostResult) xgboost.fit(features, labels);
        
        // 验证特征重要性
        IVector<Double> importance = result.getFeatureImportance();
        assertNotNull(importance);
        assertEquals(features.cols(), importance.size());
        
        // 特征重要性应该都是非负数
        for (int i = 0; i < importance.size(); i++) {
            assertTrue(importance.get(i).doubleValue() >= 0.0);
        }
    }
    
    @Test
    public void testEarlyStoppingDisabled() {
        // 测试禁用早停
        xgboost.setEarlyStopping(false);
        assertFalse(xgboost.isEarlyStopping());
        
        XGBoostResult result = (XGBoostResult) xgboost.fit(features, labels);
        assertNotNull(result);
        
        // 应该训练完所有的估计器
        assertEquals(xgboost.getNumEstimators(), result.getNumTrees());
    }
}