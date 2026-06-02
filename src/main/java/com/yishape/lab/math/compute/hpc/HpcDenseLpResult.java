package com.yishape.lab.math.compute.hpc;

/**
 * 与 yishape-math-hpc 中 {@code LpNonnegativeResult} 同构，供主库在<strong>不依赖</strong>该构件时使用。
 */
public record HpcDenseLpResult(int status, double objective, double[] x) {
}
