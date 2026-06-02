package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * PQ+HNSW 双精度向量索引，内部委托给 {@link PqHnswFloatVecIdx}。
 */
public final class PqHnswDoubleVecIdx extends AbstractDoubleVecIdx<PqHnswFloatVecIdx> {

    private static final long serialVersionUID = 1L;

    public PqHnswDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options, boolean useOpq) {
        super(validate(data, ids), metric,
                new PqHnswFloatVecIdx(toFloat(data), ids, floatMetric(metric), options, useOpq));
    }

    public PqHnswDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        this(data, ids, metric, options, false);
    }

    public PqHnswDoubleVecIdx(int dimensions, IDisMetric<Double> metric,
            VecSearchOption options, boolean useOpq) {
        super(dimensions, metric,
                new PqHnswFloatVecIdx(dimensions, floatMetric(metric), options, useOpq));
    }
}
