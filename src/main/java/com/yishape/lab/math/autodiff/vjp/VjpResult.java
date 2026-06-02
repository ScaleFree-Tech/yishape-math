package com.yishape.lab.math.autodiff.vjp;

import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * Result of a VJP (Vector-Jacobian Product) computation.
 *
 * <p>Holds both the forward output {@code y = fn(x)} and a reusable
 * {@link VjpFunction} that computes J<sup>T</sup> @ g for arbitrary
 * upstream gradients g.
 *
 * <p>VJP 计算的结果。同时持有前向输出 y = fn(x) 和可重用的 VJP 算子。
 */
public record VjpResult(IDiffVector y, VjpFunction vjpFn) {
}
