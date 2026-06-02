package com.yishape.lab.math.ml.clf;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import java.util.Map;

/**
 *
 * @author lteb2
 */
public interface ISeqClassifier extends ISerializableModel {

    /**
     * 适用于序列化训练
     *
     * @param seqFeature
     * @param labels
     * @return
     */
    public ISeqClassifier fit(IMatrix[] seqFeature, String[] labels);

    /**
     * 同时训练与预测
     * @param seqFeatures
     * @param labels
     * @return 
     */
    public default String[] fitPredict(IMatrix[] seqFeatures, String[] labels) {
        var model = this.fit(seqFeatures, labels);
        return model.predictBatch(seqFeatures);
    }

    /**
     * 预测类别 / Predict Class Label
     *
     * @param seqFeature 输入特征向量 / Input feature vector
     * @return 预测的类别标签 / Predicted class label
     */
    public String predict(IMatrix seqFeature);

    /**
     * 
     * @param seqFeature
     * @return 
     */
    public Map<String, Double> predictProb(IMatrix seqFeature);

    /**
     * 
     * @param seqFeatures
     * @return 
     */
    public String[] predictBatch(IMatrix[] seqFeatures);

    /**
     * 
     * @param seqFeatures
     * @return 
     */
    public BatchPredResult predictBatchWithProbs(IMatrix[] seqFeatures);

    /**
     * 
     * @return 
     */
    public boolean isTrained();

    /**
     * 获取分类评估指标 / Get Classification Metrics
     *
     * @return 分类评估指标 / Classification metrics
     */
    public ClassificationMetrics getMetrics();

    /**
     * 设置分类评估指标 / Set Classification Metrics
     *
     * @param metrics 分类评估指标 / Classification metrics
     */
    public void setMetrics(ClassificationMetrics metrics);

    /**
     * 
     * @return 
     */
    default ClfResult getResult() {
        return null;
    }
}
