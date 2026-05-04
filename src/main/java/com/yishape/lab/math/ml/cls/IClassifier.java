package com.yishape.lab.math.ml.cls;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import java.util.Map;

/**
 * 分类器接口 / Classifier Interface
 * <p>
 * 定义分类器的基本接口，所有分类器实现都应遵循此接口。
 * Defines the basic interface for classifiers, all classifier implementations should follow this interface.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IClassifier extends ISerializableModel {


    /**
     * 训练分类器 / Train Classifier
     *
     * @param feature 特征矩阵 / Feature matrix
     * @param labels 标签数组 / Label array
     * @return 训练结果 / Training result
     */
    public ClassificationResult fit(IMatrix feature, String[] labels);


    /**
     * 预测类别 / Predict Class Label
     *
     * @param x 输入特征向量 / Input feature vector
     * @return 预测的类别标签 / Predicted class label
     */
    public String predict(IVector x);

    /**
     * 预测各个类别的概率 / Predict Class Probabilities
     *
     * @param x 输入特征向量 / Input feature vector
     * @return 类别到概率的映射 / Map from class to probability
     */
    public Map<String, Double> predictProb(IVector x);

    /**
     * 批量预测类别 / Batch Predict Class Labels
     *
     * @param features 特征矩阵 / Feature matrix
     * @return 预测的类别标签数组 / Array of predicted class labels
     */
    public String[] predictBatch(IMatrix features);


    /**
     * 批量预测类别及概率 / Batch Predict with Probabilities
     *
     * @param features 特征矩阵 / Feature matrix
     * @return 批量预测结果 / Batch prediction result
     */
    public BatchPredictionResult predictBatchWithProbs(IMatrix features);


    /**
     * 检查分类器是否已训练 / Check if Classifier is Trained
     *
     * @return 是否已训练 / Whether the classifier is trained
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

}
