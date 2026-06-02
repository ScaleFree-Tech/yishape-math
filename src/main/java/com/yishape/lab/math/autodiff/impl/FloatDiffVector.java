package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IFloatVector;

/**
 * Mixed-precision differentiable vector: FP32 primal, FP64 gradient accumulation.
 * 混合精度可微向量：FP32 前向值，FP64 梯度累加。
 *
 * <p>Extends {@link RereDiffVector} for seamless DAG integration.
 * 继承 {@link RereDiffVector}，与现有计算图无缝集成。</p>
 */
public class FloatDiffVector extends RereDiffVector {

    private static final long serialVersionUID = 1L;
    private final float[] floatValue;

    public FloatDiffVector(float[] data) {
        super(IDoubleVector.of(data));
        this.floatValue = data.clone();
    }

    public FloatDiffVector(IFloatVector data) {
        this(data.getData());
    }

    public float[] getFloatData() {
        return floatValue.clone();
    }
}
