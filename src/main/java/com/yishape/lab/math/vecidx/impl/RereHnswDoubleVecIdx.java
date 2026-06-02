package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * 纯 Java HNSW 双精度向量索引，内部委托给 {@link RereHnswFloatVecIdx}。
 *
 * <p>仅用于 {@link RereHnswDoubleVecIdx} 内部回退，不直接暴露给用户。</p>
 */
public final class RereHnswDoubleVecIdx extends AbstractDoubleVecIdx<RereHnswFloatVecIdx> {

    private static final long serialVersionUID = 1L;

    public RereHnswDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        super(validate(data, ids), metric,
                new RereHnswFloatVecIdx(toFloat(data), ids, floatMetric(metric), options));
    }
}
