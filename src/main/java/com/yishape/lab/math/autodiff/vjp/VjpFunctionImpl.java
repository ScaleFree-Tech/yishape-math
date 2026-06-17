package com.yishape.lab.math.autodiff.vjp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.vjp.VjpFunction;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Reusable VJP (Vector-Jacobian Product) operator implementation.
 *
 * <p>Captures a computation graph from a forward pass and replays the backward
 * pass on each {@link #apply(IDiffVector)} call with a fresh upstream gradient.
 * Gradients are reset between calls so the operator is reusable.
 *
 * <p>可重用的 VJP 算子实现。捕获前向计算图，每次 apply 重新执行反向传播。
 * 每次调用之间重置梯度，因此算子可重复使用。
 */
public class VjpFunctionImpl implements VjpFunction {

    private final RereDiffVector root;
    private final RereDiffVector input;
    private final List<RereDiffTensor> order;
    private final int inputSize;

    /**
     * Creates a reusable VJP function from a forward computation graph.
     *
     * @param root  the output node of the computation graph (fn(x))
     * @param input the input leaf node (x)
     */
    public VjpFunctionImpl(RereDiffVector root, RereDiffVector input) {
        this.root = root;
        this.input = input;
        this.inputSize = input.getValue().size();
        this.order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        root.tensor.buildTopo(order, visited);
    }

    @Override
    public IDiffVector apply(IDiffVector upstreamGradient) {
        RereDiffVector ug = (RereDiffVector) upstreamGradient;

        // Reset all gradients in the computation graph to avoid contamination
        // from previous backward passes
        for (RereDiffTensor node : order) {
            node.setGradData(null);
        }

        // Set upstream gradient on the root and run backward pass manually
        // using the pre-built topological order. This bypasses backwardImpl()
        // which would destroy intermediate graph nodes (inputs/backwardFn nullified),
        // making subsequent apply() calls impossible.
        root.tensor.setGradData(ug.tensor.value().toDoubleArray().clone());
        RereDiffTensor.runBackwardOnOrder(order);

        // Read and deep-copy the gradient from the input leaf
        IDoubleVector grad = input.getGradient();
        if (grad == null) {
            return new RereDiffVector(new double[inputSize]);
        }
        return new RereDiffVector(grad.getData().clone());
    }
}
