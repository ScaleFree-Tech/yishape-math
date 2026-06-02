package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;
import java.io.Serializable;
import java.util.List;

/**
 * 数据预处理接口，实现包括归一化、标准化等等
 *
 * @author lteb2
 * @param <T> 数值类型
 */
public interface ITransform<T extends Number> extends Serializable {

    /**
     * 是否已训练，否则transform不能用
     *
     * @return
     */
    boolean ifTrained();

    /**
     * 训练预处理器或者其它类似使用数据变换的功能
     *
     * @param feature 训练数据
     * @return 当前实例，支持链式调用 / Current instance for method chaining
     */
    ITransform<T> fit(IMatrix feature);

    /**
     * 带标签的拟合与变换，数据预处理一般不需要，设置一个以防有少数算法需要，实现时需覆盖
     * @param feature 训练数据
     * @param labels 标签数组
     * @return 当前实例，支持链式调用 / Current instance for method chaining
     */
    default ITransform<T> fit(IMatrix feature, String[] labels) {
        return fit(feature);
    }

    /**
     * 变换数据，需要指定数据，可是原数据，也可以是新数据
     *
     * @param feature
     * @return
     */
    IMatrix<T> transform(IMatrix feature);

    /**
     * 获得输入模型的特征
     *
     * @return
     */
    IMatrix<T> getFeature();

    /**
     * 默认变换，变换原输入的数据
     *
     * @return
     */
    default IMatrix<T> transform() {
        var m = this.getFeature();
        return this.transform(m);
    }

    /**
     * 同时执行拟合与变换
     *
     * @param feature 训练数据
     * @return 变换后的数据矩阵
     */
    default IMatrix<T> fitTransform(IMatrix feature) {
        this.fit(feature);
        return this.transform(feature);
    }

}
