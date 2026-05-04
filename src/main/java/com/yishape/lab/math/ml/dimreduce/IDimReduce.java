package com.yishape.lab.math.ml.dimreduce;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.ml.ISerializableModel;

/**
 * 降维接口 / Dimensionality Reduction Interface
 * <p>
 * 定义降维算法的标准接口，所有降维算法实现都应遵循此接口。
 * Defines the standard interface for dimensionality reduction algorithms.
 * </p>
 *
 * @author lteb2
 * @version 1.0
 * @since 1.0
 */
public interface IDimReduce extends ISerializableModel {

    /**
     * 执行降维 / Perform Dimensionality Reduction
     *
     * @param originalData 原始数据 / Original data
     * @param dim 目标维度 / Target dimension
     * @return 降维后的数据 / Reduced dimension data
     */
    public IMatrix dimensionReduction(IMatrix originalData, int dim);
}
