package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.ml.clf.ensemble.EnsembleClassifier;
import com.yishape.lab.math.ml.clf.ensemble.EnsembleResult;
import com.yishape.lab.math.ml.clf.IClassifier;
import com.yishape.lab.math.ml.clf.ClfResult;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试EnsembleClassifier实现IClassification接口的功能
 */
public class EnsembleClassifierInterfaceTest {

    @Test
    public void testIClassificationInterface() {
        // 创建测试数据
        double[][] data = {
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 4.0},
            {4.0, 5.0}
        };
        String[] labels = {"A", "A", "B", "B"};

        IMatrix features = Linalg.matrix(data);

        // 创建EnsembleClassifier实例
        EnsembleClassifier classifier = new EnsembleClassifier(
            EnsembleClassifier.EnsembleStrategy.VOTING, 42L);

        // 验证实现了IClassification接口
        assertTrue(classifier instanceof IClassifier,
                  "EnsembleClassifier应该实现IClassification接口");

        // 测试fit方法返回IClassifier（支持链式调用）
        IClassifier result = classifier.fit(features, labels);
        assertNotNull(result, "fit方法应该返回非空的IClassifier");
        assertSame(classifier, result, "fit方法应该返回this以支持链式调用");

        // 测试单样本预测方法
        IVector singleSample = Linalg.vector(new double[]{1.5, 2.5});
        String prediction = classifier.predict(singleSample);
        assertNotNull(prediction, "单样本预测结果不应为空");
        assertTrue(prediction.equals("A") || prediction.equals("B"),
                  "预测结果应该是A或B");

        // 测试批量预测方法
        String[] batchPredictions = classifier.predictBatch(features);
        assertNotNull(batchPredictions, "批量预测结果不应为空");
        assertEquals(4, batchPredictions.length, "批量预测结果长度应该等于输入样本数");

        // 验证所有预测结果都是有效的类别标签
        for (String pred : batchPredictions) {
            assertTrue(pred.equals("A") || pred.equals("B"),
                      "每个预测结果都应该是A或B");
        }

        System.out.println("EnsembleClassifier成功实现了IClassification接口！");
        System.out.println("单样本预测结果: " + prediction);
        System.out.println("批量预测结果: " + java.util.Arrays.toString(batchPredictions));
    }
    
    @Test
    public void testUntrainedModelException() {
        // 测试未训练模型的异常处理
        EnsembleClassifier classifier = new EnsembleClassifier(
            EnsembleClassifier.EnsembleStrategy.VOTING, 42L);
        
        IVector singleSample = Linalg.vector(new double[]{1.0, 2.0});
        
        // 应该抛出IllegalStateException
        assertThrows(IllegalStateException.class, () -> {
            classifier.predict(singleSample);
        }, "未训练的模型应该抛出IllegalStateException");
    }
}