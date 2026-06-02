package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.ml.clf.tree.RereXGboost;
import com.yishape.lab.math.ml.clf.tree.XGBoostResult;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * XGBoost优化器集成测试类
 * 测试XGBoost与SGD、Adam等优化器的集成功能
 * 
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public class XGBoostOptimizerIntegrationTest {
    
    private RereXGboost xgboost;
    private IMatrix<Double> features;
    private String[] labels;
    
    @BeforeEach
    public void setUp() {
        // 创建测试数据
        double[][] data = {
            {1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0},
            {5.0, 6.0}, {6.0, 7.0}, {7.0, 8.0}, {8.0, 9.0}
        };
        features = Linalg.matrix(data);
        labels = new String[]{"A", "A", "A", "A", "B", "B", "B", "B"};
        
        // 初始化XGBoost分类器（启用实验性「自适应 boosting 学习率」，覆盖默认关闭行为）
        xgboost = new RereXGboost();
        xgboost.setAdaptiveBoostingLearningRate(true);
        xgboost.setLearningRate(0.1);
        xgboost.setNumEstimators(20);
        xgboost.setMaxDepth(3);
    }
    
    @Test
    public void testSGDOptimizerIntegration() {
        // 设置SGD优化器
        xgboost.setOptimizerType("sgd");
        xgboost.setOptimizerLearningRate(0.01);
        
        // 验证参数设置
        assertEquals("sgd", xgboost.getOptimizerType());
        assertEquals(0.01, xgboost.getOptimizerLearningRate(), 1e-6);
        
        // 训练模型
        xgboost.fit(features, labels);
        XGBoostResult result = xgboost.getResult();
        
        // 验证结果
        assertNotNull(result);
        assertNotNull(result.getTrees());
        assertTrue(result.getTrees().size() > 0);
        
        // 测试预测
        String[] predictions = xgboost.predictBatch(features);
        assertNotNull(predictions);
        assertEquals(features.rows(), predictions.length);
        
        // 验证优化器已初始化
        assertNotNull(xgboost.getOptimizer());
        Assertions.assertTrue(xgboost.getOptimizer().isInitialized());
    }
    
    @Test
    public void testAdamOptimizerIntegration() {
        // 设置Adam优化器
        xgboost.setOptimizerType("adam");
        xgboost.setOptimizerLearningRate(0.001);
        
        // 验证参数设置
        assertEquals("adam", xgboost.getOptimizerType());
        assertEquals(0.001, xgboost.getOptimizerLearningRate(), 1e-6);
        
        // 训练模型
        xgboost.fit(features, labels);
        XGBoostResult result = xgboost.getResult();
        
        // 验证结果
        assertNotNull(result);
        assertNotNull(result.getTrees());
        assertTrue(result.getTrees().size() > 0);
        
        // 测试预测
        String[] predictions = xgboost.predictBatch(features);
        assertNotNull(predictions);
        assertEquals(features.rows(), predictions.length);
        
        // 验证优化器已初始化
        assertNotNull(xgboost.getOptimizer());
        Assertions.assertTrue(xgboost.getOptimizer().isInitialized());
    }
    
    @Test
    public void testOptimizerLearningRateAdaptation() {
        // 设置SGD优化器
        xgboost.setOptimizerType("sgd");
        xgboost.setOptimizerLearningRate(0.1);
        
        double initialLearningRate = xgboost.getLearningRate();
        
        // 训练模型
        xgboost.fit(features, labels);
        XGBoostResult result = xgboost.getResult();
        
        // 验证学习率可能已被优化器调整
        assertNotNull(result);
        
        // 学习率应该在合理范围内
        double finalLearningRate = xgboost.getLearningRate();
        assertTrue(finalLearningRate >= 0.001 && finalLearningRate <= 1.0);
    }
    
    @Test
    public void testGradientAndObjectiveComputation() {
        // 初始化模型
        xgboost.setOptimizerType("sgd");
        xgboost.fit(features, labels);
        XGBoostResult result = xgboost.getResult();
        
        // 测试梯度计算 - 在模型训练后进行
        IVector<Double> testParams = Linalg.vector(new double[]{0.1, 0.2});
        IVector<Double> gradient = xgboost.computeGradient(testParams);
        assertNotNull(gradient);
        // 梯度向量应该存在（不检查具体大小，因为可能为0）
        
        // 测试目标函数计算
        double objective = xgboost.computeObjective(testParams);
        // 目标函数应该是有限的数值
        assertTrue(Double.isFinite(objective));
    }
    
    @Test
    public void testOptimizerStateInfo() {
        // 设置Adam优化器
        xgboost.setOptimizerType("adam");
        xgboost.setOptimizerLearningRate(0.01);
        
        // 训练模型
        xgboost.fit(features, labels);
        XGBoostResult result = xgboost.getResult();
        
        // 验证优化器状态
        assertNotNull(xgboost.getOptimizer());
        Assertions.assertTrue(xgboost.getOptimizer().isInitialized());
        assertTrue(xgboost.getOptimizer().getCurrentStep() > 0);
        
        // 获取状态信息
        String stateInfo = xgboost.getOptimizer().getStateInfo();
        assertNotNull(stateInfo);
        assertFalse(stateInfo.isEmpty());
    }
    
    @Test
    public void testInvalidOptimizerType() {
        // 设置无效的优化器类型
        xgboost.setOptimizerType("invalid");
        
        // 应该默认使用SGD
        xgboost.fit(features, labels);
        XGBoostResult result = xgboost.getResult();
        assertNotNull(result);
        
        // 验证优化器已初始化（应该使用默认的SGD）
        assertNotNull(xgboost.getOptimizer());
    }
}