package com.yishape.lab.math.autodiff.impl;

/**
 * Mixed-precision differentiable tensor: FP32 primal, FP64 gradient accumulation.
 * Extends {@link RereDiffTensor} for seamless DAG integration.
 */
public class FloatDiffTensor extends RereDiffTensor {

    private static final long serialVersionUID = 1L;
    private final float[] floatValue;

    public FloatDiffTensor(float[] data, int... shape) {
        super(floatToDouble(data), shape);
        this.floatValue = data.clone();
    }

    private static double[] floatToDouble(float[] data) {
        double[] d = new double[data.length];
        for (int i = 0; i < data.length; i++) d[i] = data[i];
        return d;
    }

    public float[] getFloatData() {
        return floatValue.clone();
    }
}
