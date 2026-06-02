package com.yishape.lab.math.vecidx.distance;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.dml.DmlMetric;
import com.yishape.lab.math.vecidx.MetricType;

import java.util.Objects;
import com.yishape.lab.math.vecidx.IDisMetric;

/**
 * 将 {@link DmlMetric}（度量学习拟合结果）包装为 {@link IDisMetric}，
 * 使已学习的马氏型度量可直接用于近邻索引。
 *
 * <p>{@link DmlMetric#squaredDistance} 返回平方距离，本适配器做 {@code sqrt}
 * 以与 {@link EuclideanMetric} 保持一致的值域语义。
 * 若仅关心排序一致性，可改用 {@link SquaredEuclideanMetric} 并在外层自行映射，
 * 从而省去 {@code sqrt} 开销。</p>
 *
 * <p>使用示例：
 * <pre>{@code
 * DmlMetric learned = ddml.fit(features, labels);
IDisMetric<Double> metric = DmlMetricAdapter.of(learned);
IDoubleVectorIndex idx = VectorIndexes.buildDouble(data, rowIds, metric, options);
}</pre>
 */
public final class DmlMetricAdapter implements IDisMetric<Double> {

    private final DmlMetric metric;

    private DmlMetricAdapter(DmlMetric metric) {
        this.metric = Objects.requireNonNull(metric, "metric");
    }

    /**
     * 包装已有 {@link DmlMetric}。
     */
    public static DmlMetricAdapter of(DmlMetric metric) {
        return new DmlMetricAdapter(metric);
    }

    @Override
    public MetricType type() {
        return null; // custom learned metric
    }

    @Override
    public String name() {
        return "dml_" + metric.form().name().toLowerCase();
    }

    @Override
    public boolean isSimilarity() {
        return false;
    }

    @Override
    public double compute(IVector<Double> a, IVector<Double> b) {
        return Math.sqrt(metric.squaredDistance(a, b));
    }

    /**
     * 返回被包装的原始 {@link DmlMetric}。
     */
    public DmlMetric dmlMetric() {
        return metric;
    }

    private static final long serialVersionUID = 1L;
}
