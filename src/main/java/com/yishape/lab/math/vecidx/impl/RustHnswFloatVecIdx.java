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
import java.util.concurrent.locks.StampedLock;
import java.util.function.Predicate;
import com.yishape.lab.math.vecidx.IDisMetric;
import com.yishape.lab.math.vecidx.IFloatVecIdx;
import com.yishape.lab.math.vecidx.IMutableVecIdx;

/**
 * 基于 Rust vector-index（FFM 桥接）的 HNSW 单精度向量索引。
 *
 * <p>当 Rust 原生库不可用时，自动回落到纯 Java 实现
 * {@link RereHnswFloatVecIdx}，调用方无感知。</p>
 *
 * <p>仅支持 {@code euclidean}、{@code squared_euclidean}、{@code cosine} 三种度量；
 * 其余度量会在构造时抛出 {@link IllegalArgumentException}。</p>
 */
public class RustHnswFloatVecIdx implements IFloatVecIdx, IMutableVecIdx<Float> {

    private final IFloatVecIdx delegate;
    private final IMutableVecIdx<Float> mutableDelegate;

    public RustHnswFloatVecIdx(float[][] data, String[] ids, IDisMetric<Float> metric,
            VecSearchOption options) {
        IFloatVecIdx d = null;
        IMutableVecIdx<Float> m = null;
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
            RereHnswFloatVecIdx java = new RereHnswFloatVecIdx(data, ids, metric, options);
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
    public IDisMetric<Float> metric() {
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
    public List<SearchHit> search(float[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        return delegate.search(query, k, excludeIds, filter);
    }

    @Override
    public List<SearchHit> search(IVector<Float> query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        return delegate.search(query, k, excludeIds, filter);
    }

    @Override
    public IVector<Float> getVector(String id) {
        return delegate.getVector(id);
    }

    @Override
    public void add(String id, IVector<Float> vector) {
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

    private static final class RustBackend implements IFloatVecIdx, IMutableVecIdx<Float> {

        private static final int METRIC_L2 = 0;
        private static final int METRIC_COSINE = 1;

        private final int dimensions;
        private final IDisMetric<Float> metric;
        private final VecSearchOption options;
        // 是否需要对原生返回值开方（仅 EUCLIDEAN 度量需要：原生返回 L2-squared）
        private final boolean applySqrt;

        private long handle;
        // StampedLock 替代单一 synchronized：搜索/getVector 可并发，add/clear 互斥
        private final StampedLock rwLock = new StampedLock();
        private final Map<String, Long> idToRustId = new HashMap<>();
        private final Map<Long, String> rustIdToId = new HashMap<>();
        private final AtomicLong nextRustId = new AtomicLong(0);
        // 本地缓存的索引大小，避免每次查询都调 FFI 的 hnswSize
        private int sizeCache;

        RustBackend(float[][] data, String[] ids, IDisMetric<Float> metric,
                VecSearchOption options) {
            this.dimensions = validateAndGetDims(data, ids);
            this.metric = Objects.requireNonNull(metric, "metric");
            this.options = options != null ? options : VecSearchOption.DEFAULT;
            MetricType mt = metric.type();
            if (mt != MetricType.EUCLIDEAN && mt != MetricType.SQUARED_EUCLIDEAN && mt != MetricType.COSINE) {
                throw new IllegalArgumentException(
                        "HNSW 仅支持 euclidean、squared_euclidean、cosine 度量，实为: " + metric.name());
            }
            // hnsw_rs L2 算子返回 squared distance；cosine 返回原始 cosine distance（无需开方）
            this.applySqrt = mt == MetricType.EUCLIDEAN;

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
                System.arraycopy(data[i], 0, flat, i * dimensions, dimensions);
            }

            Long h = HpcOptionalRuntime.hnswBuildF32(
                    dimensions, flat, rustIds, metricType,
                    this.options.hnswM(), this.options.hnswEfConstruction(), this.options.hnswEfSearch());
            if (h == null || h <= 0) {
                throw new IllegalStateException("HNSW 索引构建失败；请确认 HPC 扩展与原生库已正确加载。");
            }
            this.handle = h;
            this.sizeCache = ids.length;
        }

        private static int validateAndGetDims(float[][] data, String[] ids) {
            Objects.requireNonNull(data, "data");
            Objects.requireNonNull(ids, "ids");
            if (ids.length == 0) {
                throw new IllegalArgumentException("ids 不能为空");
            }
            if (data.length != ids.length) {
                throw new IllegalArgumentException("data 行数须等于 ids 长度");
            }
            int d = data[0].length;
            for (int i = 1; i < data.length; i++) {
                if (data[i] == null || data[i].length != d) {
                    throw new IllegalArgumentException("所有向量维度须一致");
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
            long stamp = rwLock.readLock();
            try {
                return handle <= 0 ? 0 : sizeCache;
            } finally {
                rwLock.unlockRead(stamp);
            }
        }

        @Override
        public IDisMetric<Float> metric() {
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
        public List<SearchHit> search(float[] query, int k, Collection<String> excludeIds, Predicate<String> filter) {
            if (k <= 0) {
                return List.of();
            }
            HpcOptionalRuntime.RHnswSearch res;
            long stamp = rwLock.readLock();
            try {
                long h = handle;
                if (h <= 0) {
                    return List.of();
                }
                int total = sizeCache;
                int fetch = k + (excludeIds != null ? excludeIds.size() : 0) + 16;
                fetch = Math.min(fetch, total);
                fetch = Math.max(fetch, k);
                res = HpcOptionalRuntime.hnswSearchF32(h, query, fetch);
            } finally {
                rwLock.unlockRead(stamp);
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
                if (applySqrt) {
                    d = Math.sqrt(d);
                }
                buf.add(new SearchHit(id, d));
            }
            return buf;
        }

        @Override
        public List<SearchHit> search(IVector<Float> query, int k, Collection<String> excludeIds, Predicate<String> filter) {
            Objects.requireNonNull(query, "query");
            if (query.length() != dimensions) {
                throw new IllegalArgumentException("查询向量维度须为 " + dimensions);
            }
            float[] q = new float[dimensions];
            for (int i = 0; i < dimensions; i++) {
                q[i] = (float)query.get(i);
            }
            return search(q, k, excludeIds, filter);
        }

        @Override
        public IVector<Float> getVector(String id) {
            Long rustId = idToRustId.get(id);
            if (rustId == null) {
                return null;
            }
            long stamp = rwLock.readLock();
            try {
                long h = handle;
                if (h <= 0) {
                    return null;
                }
                float[] buf = new float[dimensions];
                Integer rc = HpcOptionalRuntime.hnswGetF32(h, rustId, buf);
                if (rc == null || rc != 0) {
                    return null;
                }
                return IVector.of(buf);
            } finally {
                rwLock.unlockRead(stamp);
            }
        }

        @Override
        public void add(String id, IVector<Float> vector) {
            Objects.requireNonNull(id, "id");
            Objects.requireNonNull(vector, "vector");
            if (vector.length() != dimensions) {
                throw new IllegalArgumentException("向量维度须为 " + dimensions);
            }
            long stamp = rwLock.writeLock();
            try {
                if (idToRustId.containsKey(id)) {
                    throw new IllegalArgumentException("重复 ID: " + id);
                }
                long h = handle;
                if (h <= 0) {
                    throw new IllegalStateException("索引已关闭");
                }
                long rustId = nextRustId.getAndIncrement();
                float[] data = new float[dimensions];
                for (int i = 0; i < dimensions; i++) {
                    data[i] = (float)vector.get(i);
                }
                Integer rc = HpcOptionalRuntime.hnswAddF32(h, rustId, data);
                if (rc == null || rc != 0) {
                    throw new IllegalStateException("HNSW add 失败，状态码: " + rc);
                }
                idToRustId.put(id, rustId);
                rustIdToId.put(rustId, id);
                sizeCache++;
            } finally {
                rwLock.unlockWrite(stamp);
            }
        }

        @Override
        public boolean remove(String id) {
            throw new UnsupportedOperationException("vector-index 0.1 尚未支持删除操作");
        }

        @Override
        public boolean contains(String id) {
            return idToRustId.containsKey(id);
        }

        @Override
        public void clear() {
            long stamp = rwLock.writeLock();
            try {
                long h = handle;
                if (h > 0) {
                    HpcOptionalRuntime.hnswFree(h);
                    handle = 0;
                }
                idToRustId.clear();
                rustIdToId.clear();
                sizeCache = 0;
            } finally {
                rwLock.unlockWrite(stamp);
            }
        }

        @Override
        public void close() {
            clear();
        }
    }
}
