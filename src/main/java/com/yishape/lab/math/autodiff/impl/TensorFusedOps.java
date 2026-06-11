package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.ops.UniversalOperation;
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

    private static final DoubleVectorComputer COMPUTER = new DoubleVectorComputer();

    private final RereDiffTensor root;
    private final int totalSize;
    private final List<double[]> opInputs = new ArrayList<>();
    private final List<DoubleBinaryOperator> opBackwards = new ArrayList<>();
    private final List<String> opTags = new ArrayList<>();
    private double[] cur;

    public TensorFusedOps(RereDiffTensor input) {
        this.root = input;
        this.totalSize = (int) input.value().totalSize();
        this.cur = input.value().toDoubleArray();
    }

    /** Register a unary op with a universal operation (SIMD-accelerated forward). */
    private TensorFusedOps addOp(UniversalOperation uop, DoubleBinaryOperator backward, String tag) {
        double[] input = cur;
        double[] output = COMPUTER.universalOperate(input, uop, 0.0);
        opInputs.add(input);
        opBackwards.add(backward);
        opTags.add(tag);
        cur = output;
        return this;
    }

    /** Register a unary op with a custom lambda forward (no SIMD). */
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

    // Pre-defined ops — SIMD-accelerated where universal op exists
    public TensorFusedOps exp()  { return addOp(UniversalOperation.EXP,  (g, x) -> g * Math.exp(x), "exp"); }
    public TensorFusedOps log()  { return addOp(UniversalOperation.LOG,  (g, x) -> g / x, "log"); }
    public TensorFusedOps sqrt() { return addOp(UniversalOperation.SQRT, (g, x) -> g / (2.0 * Math.sqrt(x)), "sqrt"); }
    public TensorFusedOps sin()  { return addOp(UniversalOperation.SIN,  (g, x) -> g * Math.cos(x), "sin"); }
    public TensorFusedOps cos()  { return addOp(UniversalOperation.COS,  (g, x) -> -g * Math.sin(x), "cos"); }
    public TensorFusedOps sigmoid() { return addOp(UniversalOperation.SIGMOID, (g, x) -> { double s = 1.0/(1.0+Math.exp(-x)); return g * s * (1-s); }, "sigmoid"); }
    public TensorFusedOps tanh() { return addOp(UniversalOperation.TANH, (g, x) -> { double t = Math.tanh(x); return g * (1.0 - t*t); }, "tanh"); }
    public TensorFusedOps abs()  { return addOp(UniversalOperation.ABS,  (g, x) -> x >= 0 ? g : -g, "abs"); }
    public TensorFusedOps relu() { return addOp(UniversalOperation.RELU, (g, x) -> x > 0 ? g : 0, "relu"); }
    public TensorFusedOps gelu() { return addOp(UniversalOperation.GELU, (g, x) -> {
            double c = Math.sqrt(2.0 / Math.PI);
            double t = Math.tanh(c * (x + 0.044715 * x * x * x));
            double dt = 1.0 - t * t;
            double dx = c * (1.0 + 3.0 * 0.044715 * x * x);
            return g * (0.5 * (1.0 + t) + 0.5 * x * dt * dx);
        }, "gelu"); }

    // Ops without universal operation — fall back to scalar forward
    public TensorFusedOps neg()  { return addOp(x -> -x,    (g, x) -> -g, "neg"); }
    public TensorFusedOps square() { return addOp(x -> x*x, (g, x) -> g * 2.0 * x, "square"); }
    public TensorFusedOps silu() { return addOp(x -> x / (1.0 + Math.exp(-x)), (g, x) -> {
            double s = 1.0/(1.0+Math.exp(-x));
            return g * (s + x * s * (1.0 - s));
        }, "silu"); }
    public TensorFusedOps mish() { return addOp(x -> x * Math.tanh(Math.log1p(Math.exp(x))), (g, x) -> {
            double sp = Math.log1p(Math.exp(x));
            double t = Math.tanh(sp);
            double sig = 1.0/(1.0+Math.exp(-x));
            return g * (t + x * (1.0 - t*t) * sig);
        }, "mish"); }
    public TensorFusedOps pow(double n) { return addOp(x -> Math.pow(x, n), (g, x) -> g * n * Math.pow(x, n - 1), "pow"); }
    public TensorFusedOps leakyRelu(double alpha) { return addOp(x -> x > 0 ? x : alpha * x, (g, x) -> x > 0 ? g : g * alpha, "leakyRelu"); }
    public TensorFusedOps elu(double alpha) { return addOp(x -> x > 0 ? x : alpha * (Math.exp(x) - 1), (g, x) -> x > 0 ? g : g * alpha * Math.exp(x), "elu"); }
    public TensorFusedOps selu() {
        final double lambda = 1.0507009873554804934193349852946;
        final double alpha = 1.6732632423543772848170429916717;
        return addOp(x -> x > 0 ? lambda * x : lambda * alpha * (Math.exp(x) - 1),
                     (g, x) -> x > 0 ? g * lambda : g * lambda * alpha * Math.exp(x), "selu");
    }
    public TensorFusedOps softplus() { return addOp(x -> x > 20 ? x : Math.log1p(Math.exp(x)), (g, x) -> x > 20 ? g : g / (1.0 + Math.exp(-x)), "softplus"); }
    public TensorFusedOps hardtanh(double min, double max) { return addOp(x -> x < min ? min : (x > max ? max : x), (g, x) -> (x < min || x > max) ? 0 : g, "hardtanh"); }
    public TensorFusedOps add(double s) { return addOp(x -> x + s, (g, x) -> g, "addScalar"); }
    public TensorFusedOps sub(double s) { return addOp(x -> x - s, (g, x) -> g, "subScalar"); }
    public TensorFusedOps mul(double s) { return addOp(x -> x * s, (g, x) -> g * s, "mulScalar"); }
    public TensorFusedOps div(double s) { return addOp(x -> x / s, (g, x) -> g / s, "divScalar"); }

    /**
     * Finalize and create a fused tensor node.
     * The output shape matches the input shape.
     */
    public IDiffTensor done() {
        if (!root.requiresGrad()) {
            return new ConstantDiffTensor(new RereDoubleTensor(cur, root.shape()));
        }
        int n = totalSize;
        double[] out = cur;
        String chainTag = opTags.isEmpty() ? "identity" : String.join("_", opTags);
        List<double[]> savedInputs = new ArrayList<>(opInputs);
        List<DoubleBinaryOperator> savedBackwards = new ArrayList<>(opBackwards);

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs().get(0);
            double[] dx = AutodiffBufferPool.acquire(n);
            System.arraycopy(self.gradData(), 0, dx, 0, n);
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
        result.setExportShape(root.shape());
        return result;
    }
}
