package com.yishape.lab.math.compute.ops;

/**
     * Binary Reduction Operations Enumeration
     * 二元归约运算枚举
     *
     * Defines reduction operations that combine two vectors into a scalar result,
     * such as dot product and various distance norms.
     *
     * 定义将两个向量合并为标量结果的归约运算，如点积和各种距离范数。
     */
public enum BinaryReduceOperation {
    DOT,
        L2_NORM,
        L1_NORM
}
