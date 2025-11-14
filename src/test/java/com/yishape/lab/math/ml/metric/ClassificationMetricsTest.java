package com.yishape.lab.math.ml.metric;

import org.junit.jupiter.api.Test;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.*;

/**
 * ClassificationMetrics工具类测试
 * 测试分类评估指标的各种计算功能
 * 
 * @author yishape
 * @version 1.0
 */
public class ClassificationMetricsTest {

    @Test
    public void testBinaryClassificationMetrics() {
        // 二分类测试数据
        String[] yTrue = {"cat", "dog", "cat", "dog", "cat", "dog", "cat", "dog"};
        String[] yPred = {"cat", "dog", "dog", "dog", "cat", "cat", "cat", "dog"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        
        // 验证基本属性
        assertTrue(metrics.isBinaryClassification());
        assertEquals(2, metrics.getNumClasses());
        
        // 验证准确率计算 (6/8 = 0.75)
        assertEquals(0.75, metrics.getAccuracy(), 0.01);
        
        // 验证类别标签
        String[] labels = metrics.getClassLabels();
        assertEquals(2, labels.length);
        
        // 验证分类报告不为空
        String report = metrics.getClassificationReport();
        assertNotNull(report);
        assertTrue(report.contains("准确率"));
        
        System.out.println("=== 二分类测试结果 ===");
        System.out.println(report);
        System.out.println(metrics.getConfusionMatrixString());
    }

    @Test
    public void testBinaryClassificationWithAUC() {
        // 带概率的二分类测试数据
        String[] yTrue = {"negative", "positive", "positive", "negative", "positive", "negative"};
        String[] yPred = {"negative", "positive", "negative", "negative", "positive", "positive"};
        double[] yProb = {0.1, 0.9, 0.4, 0.2, 0.8, 0.7}; // 预测为正类的概率
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred, yProb);
        
        // 验证AUC计算
        assertTrue(metrics.isBinaryClassification());
        assertTrue(metrics.getAuc() >= 0); // AUC应该被计算出来
        
        // 验证分类报告包含AUC
        String report = metrics.getClassificationReport();
        assertTrue(report.contains("AUC"));
        
        System.out.println("=== 带AUC的二分类测试结果 ===");
        System.out.println(report);
    }

    @Test
    public void testMulticlassClassificationMetrics() {
        // 多分类测试数据
        String[] yTrue = {"cat", "dog", "bird", "cat", "dog", "bird", "cat", "dog", "bird"};
        String[] yPred = {"cat", "dog", "bird", "dog", "dog", "bird", "cat", "bird", "bird"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        
        // 验证多分类属性
        assertFalse(metrics.isBinaryClassification());
        assertEquals(3, metrics.getNumClasses());
        
        // 验证准确率计算 (7/9 ≈ 0.78)
        assertEquals(0.7778, metrics.getAccuracy(), 0.01);
        
        // 验证每类别的指标
        Map<String, Double> precisionPerClass = metrics.getPrecisionPerClass();
        assertEquals(3, precisionPerClass.size());
        
        Map<String, Double> recallPerClass = metrics.getRecallPerClass();
        assertEquals(3, recallPerClass.size());
        
        Map<String, Double> f1PerClass = metrics.getF1PerClass();
        assertEquals(3, f1PerClass.size());
        
        System.out.println("=== 多分类测试结果 ===");
        System.out.println(metrics.getClassificationReport());
        System.out.println(metrics.getConfusionMatrixString());
    }

    @Test
    public void testPerfectClassification() {
        // 完美分类测试
        String[] yTrue = {"A", "B", "C", "A", "B", "C"};
        String[] yPred = {"A", "B", "C", "A", "B", "C"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        
        // 验证完美分类结果
        assertEquals(1.0, metrics.getAccuracy(), 0.0001);
        assertEquals(1.0, metrics.getMacroPrecision(), 0.0001);
        assertEquals(1.0, metrics.getMacroRecall(), 0.0001);
        assertEquals(1.0, metrics.getMacroF1(), 0.0001);
        
        System.out.println("=== 完美分类测试结果 ===");
        System.out.println(metrics.getClassificationReport());
    }

    @Test
    public void testWorstClassification() {
        // 最差分类测试 (全部预测错误)
        String[] yTrue = {"A", "B", "C", "A", "B", "C"};
        String[] yPred = {"B", "C", "A", "B", "C", "A"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        
        // 验证最差分类结果
        assertEquals(0.0, metrics.getAccuracy(), 0.0001);
        
        System.out.println("=== 最差分类测试结果 ===");
        System.out.println(metrics.getClassificationReport());
    }

    @Test
    public void testErrorHandling() {
        // 测试异常处理
        assertThrows(IllegalArgumentException.class, () -> {
            ClassificationMetrics.compute(null, new String[]{"A", "B"});
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            ClassificationMetrics.compute(new String[]{"A", "B"}, null);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            ClassificationMetrics.compute(new String[]{"A", "B"}, new String[]{"A"});
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            ClassificationMetrics.compute(new String[]{"A", "B"}, new String[]{"A", "B"}, null);
        });
    }

    @Test
    public void testSingleClass() {
        // 单类别测试
        String[] yTrue = {"A", "A", "A"};
        String[] yPred = {"A", "A", "A"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        
        assertEquals(1, metrics.getNumClasses());
        assertEquals(1.0, metrics.getAccuracy(), 0.0001);
        
        System.out.println("=== 单类别测试结果 ===");
        System.out.println(metrics.getClassificationReport());
    }

    @Test
    public void testToString() {
        String[] yTrue = {"cat", "dog", "cat", "dog"};
        String[] yPred = {"cat", "dog", "dog", "dog"};
        
        ClassificationMetrics metrics = ClassificationMetrics.compute(yTrue, yPred);
        String str = metrics.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("ClassificationMetrics"));
        assertTrue(str.contains("accuracy"));
        
        System.out.println("=== toString测试 ===");
        System.out.println(str);
    }
}