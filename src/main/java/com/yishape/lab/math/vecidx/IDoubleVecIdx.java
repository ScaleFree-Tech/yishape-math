package com.yishape.lab.math.vecidx;

import com.yishape.lab.math.linalg.IVector;

import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * {@link IVecIdx} 的 {@link Double} 类型特化，定义 {@code double[]} 高性能查询路径。
 *
 * <p>连续内存数组接口适合：
 * <ul>
 *   <li>从 {@code double[][]} 物化矩阵直接查询（避免 {@link IVector} 包装开销）</li>
 *   <li>FFM / JNI 桥接至 Rust 等 native 后端（native 侧直接读连续内存）</li>
 * </ul>
 *
 * <p>子类实现者只需覆盖 {@link #search(double[], int, Collection, Predicate)}；
 * {@link IVector} 查询会自动通过 {@code query.toDoubleArray()} 委托至此。</p>
 */
public interface IDoubleVecIdx extends IVecIdx<Double> {

    @Override
    default List<SearchHit> search(IVector<Double> query, int k, Collection<String> excludeIds, Predicate<String> filter) {
        Objects.requireNonNull(query, "query");
        if (query.length() != dimensions()) {
            throw new IllegalArgumentException(
                    "query 维度须为 " + dimensions() + "，实为 " + query.length());
        }
        return search(query.toDoubleArray(), k, excludeIds, filter);
    }

    /**
     * {@code double[]} 高性能查询路径。
     *
     * @param query      查询向量，长度须等于 {@link #dimensions()}
     * @param k          返回近邻数上界；{@code k <= 0} 时返回空列表
     * @param excludeIds 需跳过的条目 id（可为空）
     * @param filter     标量过滤谓词（可为 null）
     * @return 按距离升序排列的列表（若度量为相似度则按相似度降序）
     */
    List<SearchHit> search(double[] query, int k, Collection<String> excludeIds, Predicate<String> filter);

    default List<SearchHit> search(double[] query, int k) {
        return search(query, k, Set.of(), null);
    }

    default List<SearchHit> search(double[] query, int k, Collection<String> excludeIds) {
        return search(query, k, excludeIds, null);
    }

    // ==================== double[] 精确近邻查询 ====================

    /**
     * {@code double[]} 高性能精确近邻查询路径。
     *
     * <p>默认实现：若本索引为精确索引则直接委托 {@link #search(double[], int, Collection, Predicate)}；
     * 否则先用 ANN 粗排取 {@code k * factor} 个候选，再用 {@link #metric()} 精排。</p>
     *
     * @param query      查询向量，长度须等于 {@link #dimensions()}
     * @param k          返回近邻数
     * @param factor     oversampling 系数
     * @param excludeIds 需跳过的条目 id（可为空）
     * @param filter     标量过滤谓词（可为 null）
     * @return 按 {@link #metric()} 精确排序的 Top-k 结果
     */
    default List<SearchHit> searchExact(double[] query, int k, int factor,
            Collection<String> excludeIds, Predicate<String> filter) {
        return IVecIdx.super.searchExact(IVector.of(query), k, factor, excludeIds, filter);
    }

    default List<SearchHit> searchExact(double[] query, int k, int factor) {
        return searchExact(query, k, factor, Set.of(), null);
    }

    default List<SearchHit> searchExact(double[] query, int k, int factor, Collection<String> excludeIds) {
        return searchExact(query, k, factor, excludeIds, null);
    }

    // ==================== double[] 批量查询 ====================

    default List<List<SearchHit>> batchSearch(double[][] queries, int k) {
        return batchSearch(queries, k, Set.of(), null);
    }

    default List<List<SearchHit>> batchSearch(double[][] queries, int k, Collection<String> globalExcludeIds, Predicate<String> filter) {
        Objects.requireNonNull(queries, "queries");
        return java.util.Arrays.stream(queries)
                .map(q -> search(q, k, globalExcludeIds, filter))
                .toList();
    }

    // ==================== double[] 范围查询 ====================

    default List<SearchHit> rangeSearch(double[] query, double radius, Collection<String> excludeIds, Predicate<String> filter) {
        return IVecIdx.super.rangeSearch(com.yishape.lab.math.linalg.IVector.of(query), radius, excludeIds, filter);
    }

    default List<SearchHit> rangeSearch(double[] query, double radius) {
        return rangeSearch(query, radius, Set.of(), null);
    }

    default List<SearchHit> rangeSearch(double[] query, double radius, Collection<String> excludeIds) {
        return rangeSearch(query, radius, excludeIds, null);
    }
}
