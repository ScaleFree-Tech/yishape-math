package com.yishape.lab.math.ml.lr;

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
 * @version 1.0
 * @since 1.0
 */
public interface IRegression extends ISerializableModel {

    /**
     * 训练回归模型 / Train Regression Model
     *
     * @param feature 特征矩阵 / Feature matrix
     * @param labels 标签向量 / Label vector
     * @return 训练结果 / Training result
     */
    public RegressionResult fit(IMatrix feature, IVector labels);

    /**
     * 预测数值 / Predict Value
     *
     * @param x 输入特征向量 / Input feature vector
     * @return 预测的数值 / Predicted value
     */
    public double predict(IVector x);

}
