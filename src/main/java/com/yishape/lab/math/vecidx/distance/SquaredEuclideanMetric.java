package com.yishape.lab.math.vecidx.distance;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * 平方欧几里得距离：{@code d²(a,b) = ‖a−b‖₂²}。
 *
 * <p>排序结果与 {@link EuclideanMetric} 完全一致（平方函数单调），
 * 但省去 {@code sqrt} 开销，适合仅关心相对排序、不对外暴露绝对距离的场景。
 * 若需要精确距离值（如 {@link com.yishape.lab.math.ml.dml.triplet.TripletBuilder}
 * 中三元组距离），应使用 {@link EuclideanMetric}。</p>
 */
public final class SquaredEuclideanMetric<T extends Number> implements IDisMetric<T> {

    public static final SquaredEuclideanMetric<Double> DOUBLE = new SquaredEuclideanMetric<>();
    public static final SquaredEuclideanMetric<Float> FLOAT = new SquaredEuclideanMetric<>();

    private SquaredEuclideanMetric() {
    }

    @Override
    public MetricType type() {
        return MetricType.SQUARED_EUCLIDEAN;
    }

    @Override
    public String name() {
        return "squared_euclidean";
    }

    @Override
    public boolean isSimilarity() {
        return false;
    }

    @Override
    public double compute(IVector<T> a, IVector<T> b) {
        double sum = 0.0;
        int n = a.length();
        for (int i = 0; i < n; i++) {
            double d = a.get(i) - b.get(i);
            sum += d * d;
        }
        return sum;
    }

    private static final long serialVersionUID = 1L;
}
