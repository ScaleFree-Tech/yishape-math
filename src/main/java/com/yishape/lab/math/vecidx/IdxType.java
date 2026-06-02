package com.yishape.lab.math.vecidx;

/**
 * 向量索引的类型
 * @author lteb2
 */
public enum IdxType {
    /** 暴力扫描（精确）。 */
    BRUTE_FORCE,
    HNSW,
    KDTree,
    PQ,
    LSH,
    PQ_HNSW,
    /** 根据数据规模自动选择最优类型。 */
    AUTO
}
