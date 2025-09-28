package com.reremouse.lab.math.ml.dimreduce;

import com.reremouse.lab.math.linalg.IMatrix;
import com.reremouse.lab.math.ml.ISerializableModel;

/**
 * 降维类的接口
 * @author lteb2
 * @param <T>
 */
public interface IDimReduce<T extends Number> extends ISerializableModel{
    
    /**
     * 
     * @param originalData 原始数据
     * @param dim 目标维度
     * @return 降维后的数据
     */
    public IMatrix<T> dimensionReduction(IMatrix<T> originalData, int dim);
}
