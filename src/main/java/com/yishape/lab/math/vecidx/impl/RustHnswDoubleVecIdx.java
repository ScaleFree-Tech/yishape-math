package com.yishape.lab.math.vecidx.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.MetricType;
import com.yishape.lab.math.vecidx.SearchHit;
import com.yishape.lab.math.vecidx.VecSearchOption;
import com.yishape.lab.math.compute.hpc.HpcOptionalRuntime;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IDoubleVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 基于 Rust vector-index（FFM 桥接）的 HNSW 双精度向量索引。
 *
 * <p>当 Rust 原生库不可用时，自动回落到纯 Java 实现
 * {@link RereHnswDoubleVecIdx}，调用方无感知。</p>
 *
 * <p>内部以 {@code float[]} 与 Rust 交互（vector-index 0.1 仅支持 f32），
 * 精度损失在典型嵌入/检索场景可忽略。仅支持 {@code euclidean}、
 * {@code squared_euclidean}、{@code cosine} 三种度量。</p>
 */
public class RustHnswDoubleVecIdx implements IDoubleVecIdx, IMutableVecIdx<Double> {

    private final IDoubleVecIdx delegate;
    private final IMutableVecIdx<Double> mutableDelegate;

    public RustHnswDoubleVecIdx(double[][] data, String[] ids, IDisMetric<Double> metric,
            VecSearchOption options) {
        IDoubleVecIdx d = null;
        IMutableVecIdx<Double> m = null;
        // 优先尝试 Rust 原生 HNSW
        if (HpcOptionalRuntime.isHnswNativeAvailable()) {
            try {
                RustBackend backend = new RustBackend(data, ids, metric, options);
                d = backend;
                m = backend;
            } catch (UnsatisfiedLinkError | IllegalStateException e) {
                // Rust 原生库不可用或构建失败，静默回落到 Java HNSW
            }
        }
        // 纯 Java 回落
        if (d == null) {
            RereHnswDoubleVecIdx java = new RereHnswDoubleVecIdx(data, ids, metric, options);
            d = java;
            m = java;
        }
        this.delegate = d;
        this.mutableDelegate = m;
    }

    @Override
    public int dimensions() {
        return delegate.dimensions();
    }

    @Override
    public int size() {
        return delegate.size();
    }

    @Override
    public IDisMetric<Double> metric() {
        return delegate.metric();
    }

    @Override
    public boolean isApproximate() {
        return delegate.isApproximate();
    }

    @Override
    public boolean isConcurrent() {
        return mutableDelegate.isConcurrent();
    }

    @Override
    public List<SearchHit> search(double[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        return delegate.search(query, k, excludeIds, filter);
    }

    @Override
    public List<SearchHit> search(IVector<Double> query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        return delegate.search(query, k, excludeIds, filter);
    }

    @Override
    public List<SearchHit> rangeSearch(double[] query, double radius, Collection<String> excludeIds, Predicate<String> filter) {
        return delegate.rangeSearch(query, radius, excludeIds, filter);
    }

    @Override
    public IVector<Double> getVector(String id) {
        return delegate.getVector(id);
    }

    @Override
    public void add(String id, IVector<Double> vector) {
        mutableDelegate.add(id, vector);
    }

    @Override
    public boolean remove(String id) {
        return mutableDelegate.remove(id);
    }

    @Override
    public boolean contains(String id) {
        return mutableDelegate.contains(id);
    }

    @Override
    public void clear() {
        mutableDelegate.clear();
    }

    @Override
    public void close() {
        mutableDelegate.close();
    }

    // ==================== Rust 后端（原实现）====================

    private static final class RustBackend implements IDoubleVecIdx, IMutableVecIdx<Double> {

        private static final int METRIC_L2 = 0;
        private static final int METRIC_COSINE = 1;

        private final int dimensions;
        private final IDisMetric<Double> metric;
        private final VecSearchOption options;
        private final boolean nativeReturnsSquaredL2; // true: native returns L2-squared, caller's metric may need sqrt

        private long handle;
        private final Object handleLock = new Object();
        private final Map<String, Long> idToRustId = new HashMap<>();
        private final Map<Long, String> rustIdToId = new HashMap<>();
        private final AtomicLong nextRustId = new AtomicLong(0);

        RustBackend(double[][] data, String[] ids, IDisMetric<Double> metric,
                VecSearchOption options) {
            this.dimensions = validateAndGetDims(data, ids);
            this.metric = Objects.requireNonNull(metric, "metric");
            this.options = options != null ? options : VecSearchOption.DEFAULT;
            MetricType mt = metric.type();
            if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
                throw new IllegalArgumentException(
                        "HNSW only supports euclidean, squared_euclidean, cosine metrics; got: " + metric.name());
            }
            this.nativeReturnsSquaredL2 = true;

            int metricType = metric.type() == MetricType.COSINE ? METRIC_COSINE : METRIC_L2;
            long[] rustIds = new long[ids.length];
            for (int i = 0; i < ids.length; i++) {
                long rid = nextRustId.getAndIncrement();
                rustIds[i] = rid;
                idToRustId.put(ids[i], rid);
                rustIdToId.put(rid, ids[i]);
            }

            float[] flat = new float[ids.length * dimensions];
            for (int i = 0; i < ids.length; i++) {
                for (int j = 0; j < dimensions; j++) {
                    flat[i * dimensions + j] = (float) data[i][j];
                }
            }

            Long h = HpcOptionalRuntime.hnswBuildF32(
                    dimensions, flat, rustIds, metricType,
                    this.options.hnswM(), this.options.hnswEfConstruction(), this.options.hnswEfSearch());
            if (h == null || h <= 0) {
                throw new IllegalStateException("HNSW index build failed; ensure HPC extension and native library are loaded.");
            }
            this.handle = h;
        }

        private static int validateAndGetDims(double[][] data, String[] ids) {
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

        @Override
        public int dimensions() {
            return dimensions;
        }

        @Override
        public int size() {
            synchronized (handleLock) {
                long h = handle;
                if (h <= 0) {
                    return 0;
                }
                Integer sz = HpcOptionalRuntime.hnswSize(h);
                return sz != null && sz >= 0 ? sz : 0;
            }
        }

        @Override
        public IDisMetric<Double> metric() {
            return metric;
        }

        @Override
        public boolean isApproximate() {
            return true;
        }

        @Override
        public boolean isConcurrent() {
            return true;
        }

        @Override
        public List<SearchHit> search(double[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
            if (k <= 0) {
                return List.of();
            }
            float[] q = new float[dimensions];
            for (int i = 0; i < dimensions; i++) {
                q[i] = (float) query[i];
            }
            HpcOptionalRuntime.RHnswSearch res;
            synchronized (handleLock) {
                long h = handle;
                if (h <= 0) {
                    return List.of();
                }
                Integer sz = HpcOptionalRuntime.hnswSize(h);
                int total = sz != null && sz >= 0 ? sz : 0;
                int fetch = k + (excludeIds != null ? excludeIds.size() : 0) + 16;
                fetch = Math.min(fetch, total);
                fetch = Math.max(fetch, k);
                res = HpcOptionalRuntime.hnswSearchF32(h, q, fetch);
            }
            if (res == null || !res.ok() || res.ids() == null) {
                return List.of();
            }

            Set<String> ex = excludeIds instanceof Set<String> s ? s
                    : new HashSet<>(excludeIds != null ? excludeIds : List.of());
            List<SearchHit> buf = new ArrayList<>(Math.min(k, res.found()));
            for (int i = 0; i < res.found() && buf.size() < k; i++) {
                String id = rustIdToId.get(res.ids()[i]);
                if (id == null) {
                    continue;
                }
                if (ex.contains(id) || (filter != null && !filter.test(id))) {
                    continue;
                }
                double d = res.distances()[i];
                if (nativeReturnsSquaredL2 && metric.type() != MetricType.SQUARED_EUCLIDEAN) {
                    d = Math.sqrt(d);
                }
                buf.add(new SearchHit(id, d));
            }
            return buf;
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
            throw new UnsupportedOperationException("HNSW does not yet support range search");
        }

        @Override
        public IVector<Double> getVector(String id) {
            Long rustId = idToRustId.get(id);
            if (rustId == null) {
                return null;
            }
            synchronized (handleLock) {
                long h = handle;
                if (h <= 0) {
                    return null;
                }
                float[] buf = new float[dimensions];
                Integer rc = HpcOptionalRuntime.hnswGetF32(h, rustId, buf);
                if (rc == null || rc != 0) {
                    return null;
                }
                double[] d = new double[dimensions];
                for (int i = 0; i < dimensions; i++) {
                    d[i] = buf[i];
                }
                return IVector.of(d);
            }
        }

        @Override
        public void add(String id, IVector<Double> vector) {
            Objects.requireNonNull(id, "id");
            Objects.requireNonNull(vector, "vector");
            if (vector.length() != dimensions) {
                throw new IllegalArgumentException("vector dimension must be " + dimensions);
            }
            synchronized (handleLock) {
                if (idToRustId.containsKey(id)) {
                    throw new IllegalArgumentException("duplicate ID: " + id);
                }
                long h = handle;
                if (h <= 0) {
                    throw new IllegalStateException("index is closed");
                }
                long rustId = nextRustId.getAndIncrement();
                float[] data = new float[dimensions];
                for (int i = 0; i < dimensions; i++) {
                    data[i] = (float)vector.get(i);
                }
                Integer rc = HpcOptionalRuntime.hnswAddF32(h, rustId, data);
                if (rc == null || rc != 0) {
                    throw new IllegalStateException("HNSW add failed, status: " + rc);
                }
                idToRustId.put(id, rustId);
                rustIdToId.put(rustId, id);
            }
        }

        @Override
        public boolean remove(String id) {
            throw new UnsupportedOperationException("vector-index 0.1 does not yet support deletion");
        }

        @Override
        public boolean contains(String id) {
            return idToRustId.containsKey(id);
        }

        @Override
        public void clear() {
            synchronized (handleLock) {
                long h = handle;
                if (h > 0) {
                    HpcOptionalRuntime.hnswFree(h);
                    handle = 0;
                }
                idToRustId.clear();
                rustIdToId.clear();
            }
        }

        @Override
        public void close() {
            clear();
        }
    }
}
