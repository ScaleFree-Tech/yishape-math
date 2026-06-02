package com.yishape.lab.math.vecidx;

import java.io.Serializable;

/**
 * 向量索引的构建选项。
 *
 * <p>只控制 HNSW 的调参；其他索引类型（KDTree、LSH、PQ 等）无需调参，
 * 直接使用默认参数。</p>
 *
 * @param indexType          索引类型；{@code null} 等价于 {@link IdxType#AUTO}
 * @param hnswM               每节点最大双向连接数（默认 16）。
 * @param hnswEfConstruction  建树时贪心列表长度（默认 200）。
 * @param hnswEfSearch       查询时贪心列表长度（默认 200）。
 * @param queryParallelism     批量查询并行度；{@code <= 0} 表示顺序执行。
 */
public record VecSearchOption(
        IdxType indexType,
        int hnswM,
        int hnswEfConstruction,
        int hnswEfSearch,
        int queryParallelism
) implements Serializable {

    private static final long serialVersionUID = 1L;

    /** AUTO：根据数据规模自动选择最优索引类型。 */
    public static final VecSearchOption DEFAULT = new VecSearchOption(IdxType.AUTO, 16, 200, 200, 0);

    /** 强制精确检索（BruteForce）。 */
    public static final VecSearchOption EXACT = new VecSearchOption(IdxType.BRUTE_FORCE, 16, 200, 200, 0);

    /** 合法性校验。 */
    public VecSearchOption {
        if (indexType == null) indexType = IdxType.AUTO;
        if (hnswM <= 0) {
            throw new IllegalArgumentException("hnswM 必须为正");
        }
        if (hnswEfConstruction <= 0 || hnswEfSearch <= 0) {
            throw new IllegalArgumentException("hnswEfConstruction / hnswEfSearch 必须为正");
        }
    }

    public VecSearchOption withQueryParallelism(int queryParallelism) {
        return new VecSearchOption(indexType, hnswM, hnswEfConstruction, hnswEfSearch, queryParallelism);
    }
}
