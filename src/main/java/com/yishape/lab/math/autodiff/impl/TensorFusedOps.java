package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Consumer;

/**
 * Fused element-wise operations on tensors.
 * <p>
 * Chains multiple unary/scalar ops on a single flat buffer to avoid intermediate graph nodes.
 * At {@link #done()}, creates one fused {@link RereDiffTensor} node whose backward applies
 * the adjoint chain in reverse order.
 */
public final class TensorFusedOps {

    private final RereDiffTensor root;
    private final int totalSize;
    private final List<double[]> opInputs = new ArrayList<>();
    private final List<DoubleBinaryOperator> opBackwards = new ArrayList<>();
    private final List<String> opTags = new ArrayList<>();
    private double[] cur;

    public TensorFusedOps(RereDiffTensor input) {
        this.root = input;
        this.totalSize = (int) input.value.totalSize();
        this.cur = input.value.toDoubleArray();
    }

    /** Register a unary op: forward / backward(g, x) → gradient. */
    public TensorFusedOps addOp(DoubleUnaryOperator forward, DoubleBinaryOperator backward, String tag) {
        double[] input = cur;
        double[] output = new double[totalSize];
        for (int i = 0; i < totalSize; i++) output[i] = forward.applyAsDouble(input[i]);
        opInputs.add(input);
        opBackwards.add(backward);
        opTags.add(tag);
        cur = output;
        return this;
    }

    // Pre-defined ops
    public TensorFusedOps exp()  { return addOp(Math::exp,  (g, x) -> g * Math.exp(x), "exp"); }
    public TensorFusedOps log()  { return addOp(Math::log,  (g, x) -> g / x, "log"); }
    public TensorFusedOps sqrt() { return addOp(Math::sqrt, (g, x) -> g / (2.0 * Math.sqrt(x)), "sqrt"); }
    public TensorFusedOps neg()  { return addOp(x -> -x,    (g, x) -> -g, "neg"); }
    public TensorFusedOps relu() { return addOp(x -> x > 0 ? x : 0, (g, x) -> x > 0 ? g : 0, "relu"); }
    public TensorFusedOps square() { return addOp(x -> x*x, (g, x) -> g * 2.0 * x, "square"); }
    public TensorFusedOps sigmoid() { return addOp(x -> 1.0/(1.0+Math.exp(-x)), (g, x) -> { double s = 1.0/(1.0+Math.exp(-x)); return g * s * (1-s); }, "sigmoid"); }
    public TensorFusedOps tanh() { return addOp(Math::tanh, (g, x) -> { double t = Math.tanh(x); return g * (1.0 - t*t); }, "tanh"); }
    public TensorFusedOps abs()  { return addOp(Math::abs,  (g, x) -> x >= 0 ? g : -g, "abs"); }
    public TensorFusedOps add(double s) { return addOp(x -> x + s, (g, x) -> g, "addScalar"); }
    public TensorFusedOps mul(double s) { return addOp(x -> x * s, (g, x) -> g * s, "mulScalar"); }

    /**
     * Finalize and create a fused tensor node.
     * The output shape matches the input shape.
     */
    public IDiffTensor done() {
        if (!root.requiresGrad) {
            return new RereDiffTensor.ConstantDiffTensor(new RereDoubleTensor(cur, root.shape()));
        }
        int n = totalSize;
        double[] out = cur;
        String chainTag = opTags.isEmpty() ? "identity" : String.join("_", opTags);
        List<double[]> savedInputs = new ArrayList<>(opInputs);
        List<DoubleBinaryOperator> savedBackwards = new ArrayList<>(opBackwards);

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] dx = AutodiffBufferPool.acquire(n);
            System.arraycopy(self.grad, 0, dx, 0, n);
            // Apply chain rule backward
            for (int k = savedBackwards.size() - 1; k >= 0; k--) {
                double[] in = savedInputs.get(k);
                DoubleBinaryOperator bwd = savedBackwards.get(k);
                for (int i = 0; i < n; i++) {
                    dx[i] = bwd.applyAsDouble(dx[i], in[i]);
                }
            }
            input.accGradFromPooled(dx, n);
        };
        RereDiffTensor result = new RereDiffTensor(out, root.shape(), List.of(root), bw, chainTag);
        result.exportShape = root.shape();
        return result;
    }
}
