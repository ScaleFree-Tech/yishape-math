package com.yishape.lab.math.vecidx;

import com.yishape.lab.math.linalg.IVector;

import java.io.Serializable;

/**
 * 向量距离度量函数：给定两个同维向量，返回距离值或相似度值。
 *
 * <p>索引实现通过本接口与具体距离解耦，从而支持欧氏、余弦、曼哈顿、内积、
 * 马氏（经 {@link com.yishape.lab.math.ml.dml.DmlMetric} 包装）等多种度量。
 * 所有内置实现均为无状态单例或轻量不可变对象。</p>
 *
 * <p>契约：
 * <ul>
 *   <li>距离型（{@link #isSimilarity()} == false）：{@code compute(a, b) >= 0}，
 *       且 {@code compute(a, a) == 0}；值越小表示越相近。</li>
 *   <li>相似度型（{@link #isSimilarity()} == true）：值越大表示越相近；
 *       索引实现据此自动调整排序方向。</li>
 * </ul>
 *
 * <p>对标 {@link com.yishape.lab.math.linalg.IVector} 的 API 哲学：
 * 泛型接口 + 类型特化实现，静态单例工厂提供常用度量。</p>
 *
 * @param <T> 向量元素类型，{@link Double} 或 {@link Float}
 * @see com.yishape.lab.math.vecidx.distance.EuclideanMetric
 * @see com.yishape.lab.math.vecidx.distance.CosineMetric
 * @see com.yishape.lab.math.vecidx.distance.DmlMetricAdapter
 */
public interface IDisMetric<T extends Number> extends Serializable {

    /**
     * 度量名称，用于日志、调试与序列化识别。
     * @return 
     */
    String name();

    /**
     * 若为内置度量，返回对应的枚举；自定义度量返回 {@code null}。
     *
     * <p>索引实现应优先用 {@link #type()} 做 O(1) 分发，
     * 仅当返回 {@code null} 时才回退到 {@link #compute}。</p>
     * @return 
     */
    MetricType type();

    /**
     * 是否为相似度（越大越近）而非距离（越小越近）。
     * 索引实现据此决定结果排序方向。
     * @return 
     */
    boolean isSimilarity();

    /**
     * 计算两向量的距离或相似度值。
     *
     * @param a 向量，非 null
     * @param b 向量，非 null，与 {@code a} 同维
     * @return 非负标量（距离型）或相似度得分
     * @throws IllegalArgumentException 维数不匹配
     */
    double compute(IVector<T> a, IVector<T> b);
}
