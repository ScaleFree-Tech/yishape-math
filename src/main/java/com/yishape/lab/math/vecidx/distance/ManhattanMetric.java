package com.yishape.lab.math.vecidx.distance;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * 曼哈顿距离（L1 范数）：{@code d(a,b) = Σᵢ |aᵢ − bᵢ|}。
 */
public final class ManhattanMetric<T extends Number> implements IDisMetric<T> {

    public static final ManhattanMetric<Double> DOUBLE = new ManhattanMetric<>();
    public static final ManhattanMetric<Float> FLOAT = new ManhattanMetric<>();

    private ManhattanMetric() {
    }

    @Override
    public MetricType type() {
        return MetricType.MANHATTAN;
    }

    @Override
    public String name() {
        return "manhattan";
    }

    @Override
    public boolean isSimilarity() {
        return false;
    }

    @Override
    public double compute(IVector<T> a, IVector<T> b) {
        return a.manhattanDistance(b);
    }

    private static final long serialVersionUID = 1L;
}
