package com.yishape.lab.math.vecidx.distance;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * 余弦距离：{@code d(a,b) = 1 − cos(θ) = 1 − (a·b)/(‖a‖‖b‖)}。
 *
 * <p>值域 {@code [0, 2]}，0 表示方向完全相同，2 表示方向相反。
 * 若需要原始相似度（越大越近），请使用 {@link InnerProductMetric} 或自行转换。</p>
 */
public final class CosineMetric<T extends Number> implements IDisMetric<T> {

    public static final CosineMetric<Double> DOUBLE = new CosineMetric<>();
    public static final CosineMetric<Float> FLOAT = new CosineMetric<>();

    private CosineMetric() {
    }

    @Override
    public MetricType type() {
        return MetricType.COSINE;
    }

    @Override
    public String name() {
        return "cosine";
    }

    @Override
    public boolean isSimilarity() {
        return false;
    }

    @Override
    public double compute(IVector<T> a, IVector<T> b) {
        double sim = a.cosineSimilarity(b);
        // clamp 防止数值漂移超出 [-1, 1]
        if (sim > 1.0) {
            sim = 1.0;
        } else if (sim < -1.0) {
            sim = -1.0;
        }
        return 1.0 - sim;
    }

    private static final long serialVersionUID = 1L;
}
