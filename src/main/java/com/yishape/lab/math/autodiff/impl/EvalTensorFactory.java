package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;

/**
 * Factory for creating non-differentiable (eval-mode) tensor views
 * from raw double arrays with explicit shape information.
 *
 * Used by CustomOp backward to wrap upstream gradient data as IDiffTensor
 * without requiring gradient tracking.
 */
public final class EvalTensorFactory {
    private EvalTensorFactory() {}

    /**
     * Wrap raw double[] data with a shape as a non-differentiable IDiffTensor.
     * The tensor does not require gradients and has no AD graph overhead.
     *
     * @param data  flat data array (not copied)
     * @param shape desired shape
     * @return IDiffTensor wrapping the data (non-differentiable)
     */
    public static IDiffTensor wrap(double[] data, int[] shape) {
        return new ConstantDiffTensor(new RereDoubleTensor(data, shape));
    }

    /**
     * Wrap raw double[] data as a non-differentiable IDiffVector.
     *
     * @param data flat data array (not copied)
     * @return IDiffVector wrapping the data
     */
    public static IDiffVector wrapVector(double[] data) {
        return RereDiffVector.constant(data);
    }
}
