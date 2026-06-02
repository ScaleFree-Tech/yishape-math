package com.yishape.lab.math.ml.preprocessing;

import com.yishape.lab.math.linalg.IMatrix;

/**
 * 可inverse_transform的scaler接口
 * @author lteb2
 * @param <T> 数值类型
 */
public interface IRereScaler<T extends Number> extends ITransform<T> {

    /**
     * 反变换
     * @param feature 变换后的特征
     * @return 原始尺度的特征
     */
    IMatrix<T> inverseTransform(IMatrix feature);
}
