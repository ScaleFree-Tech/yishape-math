package com.yishape.lab.math.vecidx;

import com.yishape.lab.math.linalg.IVector;

import java.util.Collection;
import java.util.Map;

/**
 * 支持动态增删的向量索引，面向内存向量数据库等需要在线更新的场景。
 *
 * <p>并非所有后端都支持在线更新（如静态 HNSW 建完后追加代价高）。
 * 可变语义由具体实现决定：</p>
 * <ul>
 *   <li>简单实现（如暴力扫描）天然支持增删；</li>
 *   <li>图索引（HNSW）可支持增量添加，但删除通常为逻辑删除或重建；</li>
 *   <li>树索引（KDTree）的增量更新可能需要再平衡。</li>
 * </ul>
 *
 * <p><strong>线程安全</strong>：</p>
 * <ul>
 *   <li>{@link #isConcurrent()} 暴露并发能力；返回 {@code true} 时，多读与读写可并发。</li>
 *   <li>返回 {@code false} 时，调用方需自行同步。</li>
 * </ul>
 *
 * @param <T> 向量元素类型
 * @see IVecIdx
 */
public interface IMutableVecIdx<T extends Number> extends IVecIdx<T> {

    /**
     * 是否支持并发查询与更新。
     * 返回 {@code true} 时，实现内部已做线程安全保证。
     */
    boolean isConcurrent();

    /**
     * 向索引中添加一条向量。
     *
     * @param id     全局唯一标识；若已存在则行为由实现决定（覆盖或抛异常）
     * @param vector 向量，维度须等于 {@link #dimensions()}
     * @throws IllegalArgumentException 维数不匹配
     */
    void add(String id, IVector<T> vector);

    /**
     * 原子批量添加。
     *
     * @param vectors id 到向量的映射
     */
    default void addAll(Map<String, ? extends IVector<T>> vectors) {
        for (Map.Entry<String, ? extends IVector<T>> e : vectors.entrySet()) {
            add(e.getKey(), e.getValue());
        }
    }

    /**
     * 从索引中移除指定 ID。
     *
     * @param id 待移除的标识
     * @return 实际存在并移除返回 {@code true}，否则 {@code false}
     */
    boolean remove(String id);

    /**
     * 原子批量删除。
     */
    default void removeAll(Collection<String> ids) {
        for (String id : ids) {
            remove(id);
        }
    }

    /**
     * 是否包含指定 ID。
     */
    boolean contains(String id);

    /**
     * 清空索引中所有向量。
     */
    void clear();
}
