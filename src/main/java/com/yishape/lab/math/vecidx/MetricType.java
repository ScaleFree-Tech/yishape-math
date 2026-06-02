package com.yishape.lab.math.vecidx;

/**
 * 内置距离度量的类型枚举，供索引实现做快速分发。
 *
 * <p>对标 Faiss 的 {@code MetricType}，避免字符串比较带来的性能损耗与笔误风险。
 * 自定义度量（如 DML 学习度量）返回 {@link IDisMetric#type()} = {@code null}，
 * 索引实现应回退到 {@link IDisMetric#compute}。</p>
 */
public enum MetricType {
    EUCLIDEAN,
    SQUARED_EUCLIDEAN,
    MANHATTAN,
    COSINE,
    INNER_PRODUCT
}
