package com.yishape.lab.math.compute.ops;

    /**
     * Reduction Operations Enumeration
     * 归约运算枚举
     *
     * Defines operations that reduce a vector to a single scalar value,
     * such as sum, mean, min, max, and statistical measures.
     *
     * 定义将向量归约为单个标量值的运算，如求和、均值、最小值、最大值和统计量。
     */
public enum ReduceOperation {
    SUM,
        MEAN,
        MIN,
        MAX,
        VARIANCE,
        STANDARD_DEVIATION,
        PROD
}
