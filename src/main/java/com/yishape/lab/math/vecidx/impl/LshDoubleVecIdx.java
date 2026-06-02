package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * LSH 双精度向量索引，内部委托给 {@link LshFloatVecIdx}。
 */
public final class LshDoubleVecIdx extends AbstractDoubleVecIdx<LshFloatVecIdx> {

    private static final long serialVersionUID = 1L;

    public LshDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        super(validate(data, ids), metric,
                new LshFloatVecIdx(toFloat(data), ids, floatMetric(metric), options));
    }

    public LshDoubleVecIdx(int dimensions, IDisMetric<Double> metric) {
        super(dimensions, metric, new LshFloatVecIdx(dimensions, floatMetric(metric)));
    }
}
