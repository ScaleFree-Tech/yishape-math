package com.yishape.lab.math.vecidx.distance;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * 内积（点积）相似度：{@code s(a,b) = a·b}。
 *
 * <p><strong>注意：这是相似度而非距离</strong>，
 * {@link #isSimilarity()} 返回 {@code true}，索引实现据此按降序排列结果。
 * 值越大表示越相近。负内积表示方向相反。</p>
 *
 * <p>在已归一化向量上，内积等价于余弦相似度。
 * 若需要非负距离型度量，请使用 {@link CosineMetric}。</p>
 */
public final class InnerProductMetric<T extends Number> implements IDisMetric<T> {

    public static final InnerProductMetric<Double> DOUBLE = new InnerProductMetric<>();
    public static final InnerProductMetric<Float> FLOAT = new InnerProductMetric<>();

    private InnerProductMetric() {
    }

    @Override
    public MetricType type() {
        return MetricType.INNER_PRODUCT;
    }

    @Override
    public String name() {
        return "inner_product";
    }

    @Override
    public boolean isSimilarity() {
        return true;
    }

    @Override
    public double compute(IVector<T> a, IVector<T> b) {
        return a.innerProductValue(b);
    }

    private static final long serialVersionUID = 1L;
}
