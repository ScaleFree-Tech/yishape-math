package com.yishape.lab.math.autodiff.vjp;

import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * Reusable Vector-Jacobian Product (VJP) operator.
 *
 * <p>Given an upstream gradient vector g, computes J<sup>T</sup> @ g where
 * J = ∂fn/∂x evaluated at the captured input. Can be called multiple times
 * with different upstream gradients without rebuilding the computation graph.
 *
 * <p>返回可重复使用的向量-雅可比积（VJP）算子。
 * 给定上游梯度向量 g，计算 J<sup>T</sup> @ g，其中 J = ∂fn/∂x 在捕获输入处的值。
 * 可多次调用而无需重建计算图。
 */
@FunctionalInterface
public interface VjpFunction {

    /**
     * Computes J<sup>T</sup> @ g for this operator's captured function at the captured input.
     *
     * @param upstreamGradient the upstream gradient vector g (length = output dimension)
     * @return the VJP result as a differentiable vector (length = input dimension)
     */
    IDiffVector apply(IDiffVector upstreamGradient);
}
