package com.yishape.lab.math.ml.clf;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;
import com.yishape.lab.math.ml.metric.ClassificationMetrics;
import java.util.Map;

/**
 * 分类器接口 / Classifier Interface
 * <p>
 * 定义分类器的基本接口，所有分类器实现都应遵循此接口。 Defines the basic interface for classifiers, all
 * classifier implementations should follow this interface.
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
     * @return 当前实例，支持链式调用 / Current instance for method chaining
     */
    public IClassifier fit(IMatrix feature, String[] labels);
    
    

    /**
     * 训练分类器并返回预测结果 / Train Classifier and Predict
     *
     * <p>
     * 组合方法，等价于 {@code fit(feature, labels); return predictBatch(feature);}</p>
     *
     * @param feature 特征矩阵 / Feature matrix
     * @param labels 标签数组 / Label array
     * @return 预测的类别标签数组 / Array of predicted class labels
     */
    public default String[] fitPredict(IMatrix feature, String[] labels) {
        var model = this.fit(feature, labels);
        return model.predictBatch(feature);
    }



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
    public BatchPredResult predictBatchWithProbs(IMatrix features);

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

    /**
     * 获取分类器特有信息（可选实现）
     * <p>
     * 各分类器可覆盖此方法返回各自的特有结果对象，如：
     * <ul>
     * <li>{@link com.yishape.lab.math.ml.clf.lr.RereLogisticRegression} 可返回
     * {@link com.yishape.lab.math.ml.clf.lr.LRResult}</li>
     * <li>{@link com.yishape.lab.math.ml.clf.tree.RereRandomForest} 可返回
     * {@link com.yishape.lab.math.ml.clf.tree.RandomForestResult}</li>
     * <li>{@link com.yishape.lab.math.ml.clf.tree.RereXGboost} 可返回
     * {@link com.yishape.lab.math.ml.clf.tree.XGBoostResult}</li>
     * </ul>
     * 如果分类器没有特有信息，可返回 {@code null}。
     *
     * @return 分类器特有结果对象，无则返回 null
     */
    default ClfResult getResult() {
        return null;
    }

}
