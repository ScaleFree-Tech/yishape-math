package com.yishape.lab.math.ml.reg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.ISerializableModel;

/**
 * 回归接口 / Regression Interface
 * <p>
 * 定义回归模型的基本接口，所有回归模型实现都应遵循此接口。
 * Defines the basic interface for regression models, all regression model implementations should follow this interface.
 * </p>
 *
 * @author lteb2
 * @version 2.0
 * @since 1.0
 */
public interface IRegression extends ISerializableModel {

    /**
     * 训练回归模型 / Train Regression Model
     *
     * @param feature 特征矩阵 / Feature matrix
     * @param labels 标签向量 / Label vector
     * @return 当前实例，支持链式调用 / Current instance for method chaining
     */
    public IRegression fit(IMatrix feature, IVector labels);

    /**
     * 训练回归模型并返回预测结果 / Train Regression Model and Predict
     *
     * <p>组合方法，等价于 {@code fit(feature, labels); return predictBatch(feature);}</p>
     *
     * @param feature 特征矩阵 / Feature matrix
     * @param labels 标签向量 / Label vector
     * @return 预测值数组 / Array of predicted values
     */
    public double[] fitPredict(IMatrix feature, IVector labels);

    /**
     * 批量预测 / Batch Predict
     *
     * @param features 特征矩阵 / Feature matrix
     * @return 预测值数组 / Array of predicted values
     */
    public double[] predictBatch(IMatrix features);

    /**
     * 预测数值 / Predict Value
     *
     * @param x 输入特征向量 / Input feature vector
     * @return 预测的数值 / Predicted value
     */
    public double predict(IVector x);

    /**
     * 检查回归模型是否已训练 / Check if Regression Model is Trained
     *
     * @return 是否已训练 / Whether the model is trained
     */
    public boolean isTrained();

    /**
     * 获取回归结果 / Get Regression Result
     *
     * @return 回归结果对象 / Regression result object
     */
    public RegressionResult getResult();

}
