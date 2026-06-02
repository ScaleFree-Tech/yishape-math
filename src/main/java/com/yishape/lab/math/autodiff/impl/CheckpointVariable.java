package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * Gradient-checkpoint node: discards intermediate activations and recomputes on backward.
 * 梯度检查点节点：丢弃中间激活，反向时重算前向。
 */
public class CheckpointVariable extends RereDiffVector {

    private static final long serialVersionUID = 1L;

    private final Function<IDiffVector, IDiffVector> forwardFn;
    private final RereDiffVector originalInput;

    public CheckpointVariable(IDoubleVector value, Function<IDiffVector, IDiffVector> forwardFn,
            RereDiffVector originalInput) {
        super(value);
        this.forwardFn = forwardFn;
        this.originalInput = originalInput;
        this.isLeaf = false;
        this.inputs = new ArrayList<>();

        CheckpointVariable self = this;
        // Re-run forward from saved input, then backward through recomputed subgraph
        // 从保存的输入重算前向，再对重算子图做反向
        this.backwardFn = (Consumer<IDoubleVector>) (gradOut) -> {
            if (self.gradient == null) {
                self.gradient = gradOut;
            }
            IDoubleVector savedGrad = self.gradient;
            IDiffVector recomputed = forwardFn.apply(originalInput);
            RereDiffVector out = (RereDiffVector) recomputed;
            out.backward(savedGrad);
        };
    }
}
