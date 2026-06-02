package com.yishape.lab.math.autodiff.vjp;

import com.yishape.lab.math.autodiff.IDiffVector;
import java.util.List;

/**
 * Result of a batch VJP computation.
 *
 * <p>Holds the forward outputs and reusable VJP functions for each sample
 * in a batch. Each VJP function computes J<sup>T</sup> @ g for the
 * corresponding sample.
 *
 * <p>批量 VJP 计算的结果。持有每样本的前向输出和可重用的 VJP 算子。
 */
public record BatchVjpResult(IDiffVector[] ys, VjpFunction[] vjpFns) {

    /**
     * Returns the batch size.
     */
    public int batchSize() {
        return ys.length;
    }

    /**
     * Convenience: apply a single upstream gradient across all VJP functions.
     *
     * @param upstreamGradient the upstream gradient vector
     * @return per-sample VJP results
     */
    public IDiffVector[] applyAll(IDiffVector upstreamGradient) {
        IDiffVector[] results = new IDiffVector[vjpFns.length];
        for (int i = 0; i < vjpFns.length; i++) {
            results[i] = vjpFns[i].apply(upstreamGradient);
        }
        return results;
    }

    /**
     * Sum all VJP results into a single gradient vector.
     * Equivalent to gradient accumulation across the batch.
     */
    public IDiffVector sumGradients(IDiffVector upstreamGradient) {
        IDiffVector[] grads = applyAll(upstreamGradient);
        IDiffVector sum = grads[0];
        for (int i = 1; i < grads.length; i++) {
            sum = sum.add(grads[i]);
        }
        return sum;
    }

    /**
     * Mean of all VJP results across the batch.
     */
    public IDiffVector meanGradients(IDiffVector upstreamGradient) {
        IDiffVector sum = sumGradients(upstreamGradient);
        return sum.div(batchSize());
    }
}
