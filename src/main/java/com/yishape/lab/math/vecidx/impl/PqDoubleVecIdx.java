package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * PQ 双精度向量索引，内部委托给 {@link PqFloatVecIdx}。
 */
public final class PqDoubleVecIdx extends AbstractDoubleVecIdx<PqFloatVecIdx> {

    private static final long serialVersionUID = 1L;

    public PqDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options, boolean useOpq) {
        super(validate(data, ids), metric,
                new PqFloatVecIdx(toFloat(data), ids, floatMetric(metric), options, useOpq));
    }

    public PqDoubleVecIdx(double[][] data, String[] ids,
            IDisMetric<Double> metric, VecSearchOption options) {
        this(data, ids, metric, options, false);
    }

    public PqDoubleVecIdx(int dimensions, IDisMetric<Double> metric, boolean useOpq) {
        super(dimensions, metric, new PqFloatVecIdx(dimensions, floatMetric(metric), useOpq));
    }
}
