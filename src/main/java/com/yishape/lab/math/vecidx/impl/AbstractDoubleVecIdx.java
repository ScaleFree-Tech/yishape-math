package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.SearchHit;

import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IDoubleVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 为 {@code *DoubleVectorIndex} 消除重复的 double→float 转换与委托骨架。
 *
 * <p>子类仅需提供构造器（调用父类构造器传入已构建好的内部 {@link IFloatVecIdx} 实现），
 * 所有 {@link IDoubleVecIdx} / {@link IMutableVecIdx} 方法由本类统一委托。</p>
 *
 * @param <T> 内部单精度索引类型，须同时实现 {@link IFloatVecIdx} 和 {@link IMutableVecIdx}
 */
public abstract class AbstractDoubleVecIdx<T extends IFloatVecIdx & IMutableVecIdx<Float>>
        implements IDoubleVecIdx, IMutableVecIdx<Double> {

    private static final long serialVersionUID = 1L;

    protected final T inner;
    protected final IDisMetric<Double> metric;
    protected final int dimensions;

    protected AbstractDoubleVecIdx(int dimensions, IDisMetric<Double> metric, T inner) {
        this.dimensions = dimensions;
        this.metric = Objects.requireNonNull(metric, "metric");
        this.inner = Objects.requireNonNull(inner, "inner");
    }

    // ==================== 静态工具 ====================

    protected static float[] toFloat(double[] d) {
        float[] f = new float[d.length];
        for (int i = 0; i < d.length; i++) {
            f[i] = (float) d[i];
        }
        return f;
    }

    protected static float[][] toFloat(double[][] data) {
        float[][] f = new float[data.length][];
        for (int i = 0; i < data.length; i++) {
            f[i] = toFloat(data[i]);
        }
        return f;
    }

    protected static double[] toDouble(float[] f) {
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }

    protected static int validate(double[][] data, String[] ids) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        if (ids.length == 0) {
            throw new IllegalArgumentException("ids must not be empty");
        }
        if (data.length != ids.length) {
            throw new IllegalArgumentException("data rows must equal ids length");
        }
        int d = data[0].length;
        for (int i = 1; i < data.length; i++) {
            if (data[i] == null || data[i].length != d) {
                throw new IllegalArgumentException("all vectors must have consistent dimensions");
            }
        }
        return d;
    }

    protected static IDisMetric<Float> floatMetric(IDisMetric<Double> metric) {
        MetricType mt = metric.type();
        if (mt != null) {
            return new IDisMetric<>() {
                @Override
                public MetricType type() {
                    return mt;
                }

                @Override
                public String name() {
                    return metric.name();
                }

                @Override
                public boolean isSimilarity() {
                    return metric.isSimilarity();
                }

                @Override
                public double compute(IVector<Float> a, IVector<Float> b) {
                    float[] fa = a.toFloatArray();
                    float[] fb = b.toFloatArray();
                    return computeFloat(mt, fa, fb);
                }

                private double computeFloat(MetricType t, float[] a, float[] b) {
                    switch (t) {
                        case EUCLIDEAN:
                            return Math.sqrt(Linalg.squaredDistance(a, b));
                        case SQUARED_EUCLIDEAN:
                            return Linalg.squaredDistance(a, b);
                        case MANHATTAN: {
                            double sum = 0.0;
                            for (int i = 0; i < a.length; i++) {
                                sum += Math.abs((double) a[i] - (double) b[i]);
                            }
                            return sum;
                        }
                        case COSINE: {
                            double dot = 0.0, na = 0.0, nb = 0.0;
                            for (int i = 0; i < a.length; i++) {
                                dot += (double) a[i] * (double) b[i];
                                na += (double) a[i] * (double) a[i];
                                nb += (double) b[i] * (double) b[i];
                            }
                            double cos = dot / (Math.sqrt(na) * Math.sqrt(nb));
                            return 1.0 - Math.max(-1.0, Math.min(1.0, cos));
                        }
                        case INNER_PRODUCT: {
                            return Linalg.dot(a, b);
                        }
                        default:
                            return metric.compute(
                                    IVector.of(toDouble(a)),
                                    IVector.of(toDouble(b)));
                    }
                }
            };
        }
        return new IDisMetric<>() {
            @Override
            public MetricType type() {
                return null;
            }

            @Override
            public String name() {
                return metric.name();
            }

            @Override
            public boolean isSimilarity() {
                return metric.isSimilarity();
            }

            @Override
            public double compute(IVector<Float> a, IVector<Float> b) {
                return metric.compute(
                        IVector.of(toDouble(a.toFloatArray())),
                        IVector.of(toDouble(b.toFloatArray())));
            }
        };
    }

    // ==================== 委托方法 ====================

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public int size() {
        return inner.size();
    }

    @Override
    public IDisMetric<Double> metric() {
        return metric;
    }

    @Override
    public boolean isApproximate() {
        return inner.isApproximate();
    }

    @Override
    public boolean isConcurrent() {
        return inner.isConcurrent();
    }

    @Override
    public List<SearchHit> search(double[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        return inner.search(toFloat(query), k, excludeIds, filter);
    }

    @Override
    public List<SearchHit> search(IVector<Double> query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        Objects.requireNonNull(query, "query");
        if (query.length() != dimensions) {
            throw new IllegalArgumentException("query dimension must be " + dimensions);
        }
        double[] q = new double[dimensions];
        for (int i = 0; i < dimensions; i++) {
            q[i] = query.get(i);
        }
        return search(q, k, excludeIds, filter);
    }

    @Override
    public List<SearchHit> rangeSearch(double[] query, double radius, Collection<String> excludeIds, Predicate<String> filter) {
        return inner.rangeSearch(toFloat(query), radius, excludeIds, filter);
    }

    @Override
    public IVector<Double> getVector(String id) {
        IVector<Float> fvec = inner.getVector(id);
        if (fvec == null) {
            return null;
        }
        return IVector.of(toDouble(fvec.toFloatArray()));
    }

    @Override
    public void add(String id, IVector<Double> vector) {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(vector, "vector");
        if (vector.length() != dimensions) {
            throw new IllegalArgumentException("vector dimension must be " + dimensions);
        }
        inner.add(id, IVector.of(toFloat(vector.toDoubleArray())));
    }

    @Override
    public boolean remove(String id) {
        return inner.remove(id);
    }

    @Override
    public boolean contains(String id) {
        return inner.contains(id);
    }

    @Override
    public void clear() {
        inner.clear();
    }

    @Override
    public void close() {
        inner.close();
    }
}
