package com.yishape.lab.math.vecidx.distance;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * 欧几里得距离（L2 范数）：{@code d(a,b) = ‖a−b‖₂}。
 *
 * <p>提供 {@link Double} 与 {@link Float} 两个单例。
 * 数组高性能路径在暴力扫描等实现内部直接调用，不经过本接口。</p>
 */
public final class EuclideanMetric<T extends Number> implements IDisMetric<T> {

    public static final EuclideanMetric<Double> DOUBLE = new EuclideanMetric<>();
    public static final EuclideanMetric<Float> FLOAT = new EuclideanMetric<>();

    private EuclideanMetric() {
    }

    @Override
    public MetricType type() {
        return MetricType.EUCLIDEAN;
    }

    @Override
    public String name() {
        return "euclidean";
    }

    @Override
    public boolean isSimilarity() {
        return false;
    }

    @Override
    public double compute(IVector<T> a, IVector<T> b) {
        return a.euclideanDistance(b);
    }

    private static final long serialVersionUID = 1L;
}
