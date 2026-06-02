package com.yishape.lab.math.vecidx;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.vecidx.distance.EuclideanMetric;

import java.io.Serializable;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Predicate;

/**
 * 固定向量集合上的 k 近邻查询索引。
 *
 * <p>
 * 设计哲学对标 {@link com.yishape.lab.math.linalg.IVector} 与
 * {@link com.yishape.lab.math.optimize.IOptimizer}：
 * <ul>
 * <li><strong>泛型接口 + 类型特化子接口</strong>：
 * {@link IDoubleVecIdx}、{@link IFloatVecIdx} 在数组路径上做特化， 供连续内存与 FFM /
 * JNI 桥接使用。</li>
 * <li><strong>默认方法提供便捷重载</strong>，抽象方法聚焦核心契约。</li>
 * <li><strong>构建时绑定距离度量</strong>： 索引通过 {@link #metric()} 暴露自身使用的
 * {@link IDisMetric}， 查询方无需每次传递。</li>
 * <li><strong>标量过滤</strong>： 支持 {@code Predicate<String>} 谓词过滤，实现向量相似度 +
 * 元数据条件的混合查询。</li>
 * </ul>
 *
 * <p>
 * 典型使用流程（工厂模式）：
 * <pre>{@code
 * try (IDoubleVecIdx idx = VI.buildDouble(
    data, ids, EuclideanMetric.DOUBLE, VecSearchOption.DEFAULT)) {
    List<SearchHit> hits = idx.search(query, 10, id -> payload.get(id).category().equals("electronics"));
}
}</pre>
 *
 * @param <T> 向量元素类型，{@link Double} 或 {@link Float}
 * @see IDoubleVecIdx
 * @see IFloatVecIdx
 * @see IMutableVecIdx
 * @see VI
 */
public interface IVecIdx<T extends Number> extends Serializable, AutoCloseable {

    /**
     * 向量维度（与存入向量长度一致）。
     */
    int dimensions();

    /**
     * 索引中条目数。
     */
    int size();

    /**
     * 本索引使用的距离度量。
     */
    IDisMetric<T> metric();

    /**
     * 是否为近似索引（如 HNSW），{@code false} 表示结果精确。
     */
    boolean isApproximate();

    // ==================== 基础查询 ====================
    /**
     * 查询 {@code query} 的 {@code k} 个近邻。
     *
     * @param query 查询向量，维度须等于 {@link #dimensions()}
     * @param k 返回近邻数上界；{@code k <= 0} 时返回空列表
     * @return 按“越近越靠前”排序的列表
     */
    default List<SearchHit> search(IVector<T> query, int k) {
        return search(query, k, Set.of(), null);
    }

    /**
     * 查询 {@code query} 的 {@code k} 个近邻，排除指定 ID。
     *
     * @param query 查询向量，维度须等于 {@link #dimensions()}
     * @param k 返回近邻数上界
     * @param excludeIds 需跳过的条目 id（可为空）
     * @return 按“越近越靠前”排序的列表
     */
    default List<SearchHit> search(IVector<T> query, int k, Collection<String> excludeIds) {
        return search(query, k, excludeIds, null);
    }

    /**
     * 查询 {@code query} 的 {@code k} 个近邻，排除指定 ID，并应用标量过滤。
     *
     * <p>
     * <strong>标量过滤语义</strong>：{@code filter} 为 {@code null} 时不过滤； 非 null 时仅返回
     * {@code filter.test(id) == true} 的条目。 后端可选择预过滤（在距离计算前排除）或后过滤（先取超集再过滤），
     * 精确索引通常采用全量扫描预过滤，近似索引通常采用后过滤 + oversampling。</p>
     *
     * @param query 查询向量，维度须等于 {@link #dimensions()}
     * @param k 返回近邻数上界
     * @param excludeIds 需跳过的条目 id（可为空）
     * @param filter 标量过滤谓词（可为 null）
     * @return 按“越近越靠前”排序的列表
     * @throws IllegalArgumentException 维数不匹配
     */
    List<SearchHit> search(IVector<T> query, int k, Collection<String> excludeIds, Predicate<String> filter);

    // ==================== 粗排 + 精排 ====================
    /**
     * 两阶段检索：先用索引自身度量粗排取 {@code ef} 个候选，再用 {@code refineMetric} 精确重算距离。
     *
     * <p>
     * 典型场景：</p>
     * <ul>
     * <li>HNSW 用内积粗排（图遍历快），再用欧氏距离精排（结果更准）；</li>
     * <li>DML 用对角度量粗排，再用满秩度量精排。</li>
     * </ul>
     *
     * <p>
     * 默认实现：委托 {@link #search(IVector, int)} 取 {@code ef} 个候选， 用
     * {@code refineMetric} 重新计算精确距离后取 Top-k。</p>
     *
     * @param query 查询向量
     * @param k 最终返回近邻数
     * @param refineMetric 精排度量（通常比索引度量更精确但更慢）
     * @param ef 粗排候选数（ef >= k）
     * @return 按 {@code refineMetric} 排序的 Top-k 结果
     */
    default List<SearchHit> search(IVector<T> query, int k,
            IDisMetric<T> refineMetric, int ef) {
        return search(query, k, refineMetric, ef, Set.of(), null);
    }

    /**
     * 带过滤的两阶段检索。
     * @param query
     * @param k
     * @param refineMetric
     * @param ef
     * @param excludeIds
     * @param filter
     * @return 
     */
    default List<SearchHit> search(IVector<T> query, int k,
            IDisMetric<T> refineMetric, int ef,
            Collection<String> excludeIds, Predicate<String> filter) {
        Objects.requireNonNull(refineMetric, "refineMetric");
        if (ef < k) {
            throw new IllegalArgumentException("ef 必须 >= k");
        }
        List<SearchHit> coarse = search(query, ef, excludeIds, filter);
        boolean sim = refineMetric.isSimilarity();
        List<SearchHit> refined = coarse.stream()
                .map(h -> {
                    // 通过 getVector 获取原始向量重新计算距离
                    IVector<T> vec = getVector(h.id());
                    if (vec == null) {
                        return h;
                    }
                    double d = refineMetric.compute(query, vec);
                    return new SearchHit(h.id(), d);
                })
                .sorted(sim
                        ? (a, b) -> Double.compare(b.distance(), a.distance())
                        : Comparator.comparingDouble(SearchHit::distance))
                .toList();
        int n = Math.min(k, refined.size());
        return refined.subList(0, n);
    }

    /**
     * 根据 ID 获取索引中的原始向量。
     *
     * <p>
     * 默认实现返回 {@code null}（部分索引不保留原始向量，如量化索引）。 暴力扫描等精确实现应覆盖此方法以支持精排。</p>
     *
     * @param id 条目标识
     * @return 原始向量；若不存在或不支持则返回 {@code null}
     */
    default IVector<T> getVector(String id) {
        return null;
    }

    // ==================== 精确近邻查询（ANN + 精排）====================
    /**
     * 精确近邻查询：对近似索引先用 ANN 粗排取 {@code k * factor} 个候选， 再用本索引度量精排得到精确 Top-k。
     *
     * <p>
     * 此方法解决"上层 API 需要精确结果，但底层是 ANN 索引"的问题： 对精确索引（{@link #isApproximate()} 为
     * {@code false}）直接返回精确结果； 对近似索引则执行 oversampling + 精排两步。</p>
     *
     * <p>
     * 精排计算量与 {@code factor} 成正比，建议对召回率要求高的场景（如 DML 三元组构建） 使用较大的
     * {@code factor}（如 10），对延迟敏感场景使用较小值。</p>
     *
     * @param query 查询向量，维度须等于 {@link #dimensions()}
     * @param k 返回近邻数
     * @param factor oversampling 系数；实际取 {@code k * factor} 个候选后精排
     * @return 按 {@link #metric()} 精确排序的 Top-k 结果
     * @throws IllegalArgumentException factor &lt;= 0
     */
    default List<SearchHit> searchExact(IVector<T> query, int k, int factor) {
        return searchExact(query, k, factor, Set.of(), null);
    }

    /**
     * 带排除和过滤的精确近邻查询。
     * @param query
     * @param k
     * @param factor
     * @param excludeIds
     * @param filter
     * @return 
     */
    default List<SearchHit> searchExact(IVector<T> query, int k, int factor,
            Collection<String> excludeIds, Predicate<String> filter) {
        if (factor <= 0) {
            throw new IllegalArgumentException("factor 必须为正");
        }
        if (k <= 0) {
            return List.of();
        }
        // 精确索引直接走基础查询
        if (!isApproximate()) {
            return search(query, k, excludeIds, filter);
        }
        int ef = Math.multiplyExact(k, factor);
        return search(query, k, metric(), ef, excludeIds, filter);
    }

    // ==================== 批量查询 ====================
    default List<List<SearchHit>> batchSearch(List<? extends IVector<T>> queries, int k) {
        return batchSearch(queries, k, Set.of(), null);
    }

    default List<List<SearchHit>> batchSearch(List<? extends IVector<T>> queries, int k,
            Collection<String> globalExcludeIds, Predicate<String> filter) {
        Objects.requireNonNull(queries, "queries");
        return queries.stream()
                .map(q -> search(q, k, globalExcludeIds, filter))
                .toList();
    }

    // ==================== 范围查询 ====================
    default List<SearchHit> rangeSearch(IVector<T> query, double radius) {
        return rangeSearch(query, radius, Set.of(), null);
    }

    default List<SearchHit> rangeSearch(IVector<T> query, double radius, Collection<String> excludeIds) {
        return rangeSearch(query, radius, excludeIds, null);
    }

    default List<SearchHit> rangeSearch(IVector<T> query, double radius,
            Collection<String> excludeIds, Predicate<String> filter) {
        if (radius < 0) {
            return List.of();
        }
        List<SearchHit> all = search(query, size(), excludeIds, filter);
        boolean sim = metric().isSimilarity();
        return all.stream()
                .filter(h -> sim ? h.distance() >= radius : h.distance() <= radius)
                .toList();
    }

    // ==================== 资源释放 ====================
    @Override
    default void close() {
        // 默认空操作
    }

    /**
     * 根据特征矩阵构建索引，由 {@link IdxType} 指定索引类型。
     *
     * <p>AUTO 路由（基于 2026-05-13 性能基准测试）：</p>
     * <ul>
     *   <li>{@code N < 1,000} → BruteForce</li>
     *   <li>{@code dim ≤ 20} → KD-Tree</li>
     *   <li>{@code N ≥ 50,000} → HNSW（最快查询）</li>
     *   <li>{@code 5,000 ≤ N < 50,000} → LSH（高召回）</li>
     * </ul>
     */
    public static IDoubleVecIdx of(IMatrix<?> feature, IdxType type) {
        Objects.requireNonNull(feature, "feature");
        if (feature.getRowNum() == 0 || feature.getColNum() == 0) {
            throw new IllegalArgumentException("特征矩阵不能为空");
        }
        int n = feature.getRowNum();
        double[][] data = feature.toDoubleArray();
        String[] ids = new String[n];
        for (int i = 0; i < n; i++) ids[i] = String.valueOf(i);
        return of(data, ids, type, EuclideanMetric.DOUBLE);
    }

    public static IDoubleVecIdx of(IMatrix<?> feature, IdxType type, IDisMetric<Double> metric) {
        Objects.requireNonNull(feature, "feature");
        Objects.requireNonNull(metric, "metric");
        int n = feature.getRowNum();
        double[][] data = feature.toDoubleArray();
        String[] ids = new String[n];
        for (int i = 0; i < n; i++) ids[i] = String.valueOf(i);
        return of(data, ids, type, metric);
    }

    public static IDoubleVecIdx of(double[][] data, String[] ids, IdxType type, IDisMetric<Double> metric) {
        Objects.requireNonNull(data, "data");
        Objects.requireNonNull(ids, "ids");
        Objects.requireNonNull(type, "type");
        Objects.requireNonNull(metric, "metric");
        int n = ids.length;
        int dim = data.length > 0 && data[0] != null ? data[0].length : 0;
        if (n == 0 || dim == 0) throw new IllegalArgumentException("数据不能为空");
        VecSearchOption opts = new VecSearchOption(type, 16, 200, 200, 0);
        return VI.buildDouble(data, ids, metric, opts);
    }

    public static IDoubleVecIdx of(IMatrix<?> feature) {
        return of(feature, IdxType.AUTO);
    }

}
