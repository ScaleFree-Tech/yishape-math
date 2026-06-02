package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * KD-Tree 双精度向量索引，内部委托给 {@link KdTreeFloatVecIdx}。
 */
public final class KdTreeDoubleVecIdx extends AbstractDoubleVecIdx<KdTreeFloatVecIdx> {

    private static final long serialVersionUID = 1L;

    public KdTreeDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        super(validate(data, ids), metric,
                new KdTreeFloatVecIdx(toFloat(data), ids, floatMetric(metric), options));
    }

    public KdTreeDoubleVecIdx(int dimensions, IDisMetric<Double> metric) {
        super(dimensions, metric, new KdTreeFloatVecIdx(dimensions, floatMetric(metric)));
    }
}
