package com.yishape.lab.math.autodiff;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import com.yishape.lab.math.autodiff.impl.TensorBackedDiffVector;

/**
 * Self-contained custom differentiable operation.
 *
 * <p>Replaces the legacy {@code AD.registerGradient / AD.custom / AD.unregisterGradient}
 * pattern. Each instance manages its own forward-data cache and embeds the backward
 * function directly into graph nodes, eliminating the global
 * {@link CustomGradientRegistry} and preventing memory leaks.</p>
 *
 * <p>Subclass and implement {@link #forward(IDoubleVector[])} and
 * {@link #backward(IDoubleVector, Object)}, then call {@link #apply(IDiffVector...)}
 * to build computation graph nodes.</p>
 *
 * <p>Typical usage:
 * <pre>{@code
 * CustomOp op = new CustomOp() {
 *     protected ForwardResult forward(IDoubleVector[] inputs) {
 *         double[] out = ...;
 *         MyCtx ctx = new MyCtx(...);
 *         return new ForwardResult(IDoubleVector.of(out), ctx);
 *     }
 *     protected IDoubleVector[] backward(IDoubleVector g, Object ctx) {
 *         MyCtx c = (MyCtx) ctx;
 *         return new IDoubleVector[]{ IDoubleVector.of(dX) };
 *     }
 * };
 * IDiffVector result = op.apply(x); // auto nodeId, auto cache, no leaks
 * }</pre>
 */
public abstract class CustomOp {

    static final AtomicLong GLOBAL_COUNTER = new AtomicLong(0);

    final AtomicLong fwdCounter = new AtomicLong(0);

    // Forward-data cache: maps forward pass ID to opaque context for backward.
    // Uses ConcurrentHashMap to avoid the 256-slot overflow bug and synchronized bottleneck.
    private final ConcurrentHashMap<Long, Object> fwdCache = new ConcurrentHashMap<>();

    /**
     * When set, propagated to the graph node's opTag so the GPU/HPC graph executor
     * can dispatch a native kernel for this CustomOp instead of falling back to CPU.
     */
    protected String graphOpTag;

    /** Scalar parameter propagated to the graph node (e.g. eps, outFeatures). */
    protected double graphScalarParam = Double.NaN;

    /** Second scalar parameter propagated to the graph node (e.g. batch size). */
    protected double graphScalarParam2 = Double.NaN;

    /** Set the scalar parameter propagated to the graph node. */
    public void setScalarParam(double v) { graphScalarParam = v; }

    /** Set the second scalar parameter propagated to the graph node. */
    public void setScalarParam2(double v) { graphScalarParam2 = v; }

    /** Set the operation tag for GPU/HPC graph dispatch. */
    public void setGraphOpTag(String tag) { graphOpTag = tag; }

    private void putCache(long id, Object ctx) {
        if (ctx != null) {
            fwdCache.put(id, ctx);
        }
    }

    private Object getCache(long id) {
        return fwdCache.get(id);
    }

    /**
     * Apply this operation to the given inputs.
     * Always creates a computation graph node with embedded backward.
     */
    public IDiffVector apply(IDiffVector... inputs) {
        RereDiffVector[] nodes = new RereDiffVector[inputs.length];
        IDoubleVector[] raw = new IDoubleVector[inputs.length];
        // Track TensorBackedDiffVector inputs for gradient bridging
        RereDiffTensor[] tensorBridges = new RereDiffTensor[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] instanceof TensorBackedDiffVector tbdv) {
                RereDiffTensor t = tbdv.unwrap();
                tensorBridges[i] = t;
                raw[i] = IDoubleVector.of(t.value.toDoubleArray());
                // Resolve tensor chain: walk back through tensor graph to find
                // vectorSource, creating vector nodes for intermediate ops.
                // This connects the model's full computation graph for GPU/HPC execution.
                nodes[i] = resolveTensorChain(t);
                // If resolveTensorChain returns null, no vectorSource found — will create bridge leaf below
            } else {
                nodes[i] = (RereDiffVector) inputs[i];
                raw[i] = nodes[i].getValue();
            }
        }

        ForwardResult fr = forward(raw);

        long id = fwdCounter.incrementAndGet();
        putCache(id, fr.context());

        // Build input list: use chain nodes when available, bridge leaves as fallback
        RereDiffVector[] effectiveNodes = new RereDiffVector[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            if (tensorBridges[i] != null && nodes[i] == null) {
                // No vectorSource found — create disconnected bridge leaf as fallback.
                // This means GPU graph execution will skip (tensorBridge not in SUPPORTED_OPS).
                RereDiffTensor t = tensorBridges[i];
                RereDiffVector bridgeLeaf = new RereDiffVector(IDoubleVector.of(t.value.toDoubleArray()));
                bridgeLeaf.opTag = "tensorBridge";
                effectiveNodes[i] = bridgeLeaf;
            } else {
                // Either non-tensor input, or chain node connected to vectorSource
                effectiveNodes[i] = nodes[i];
            }
        }

        List<RereDiffVector> insList = Arrays.asList(effectiveNodes);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            Object ctx = getCache(id);
            IDoubleVector[] grads = backward(gradOut, ctx);
            for (int j = 0; j < inputs.length && j < grads.length; j++) {
                if (grads[j] != null) {
                    if (tensorBridges[j] != null) {
                        // Bridge gradient back to tensor's grad array, then
                        // propagate through the tensor graph so upstream layers
                        // (Conv2d, Linear, etc.) receive gradients on their parameters.
                        RereDiffTensor t = tensorBridges[j];
                        if (t.grad == null) {
                            t.grad = new double[grads[j].size()];
                        }
                        double[] g = grads[j].getData();
                        for (int k = 0; k < g.length; k++) {
                            t.grad[k] += g[k];
                        }
                        t.propagateGrad();
                    } else if (nodes[j] != null) {
                        nodes[j].accGrad(grads[j]);
                    }
                }
            }
        };

        RereDiffVector result = (RereDiffVector) RereDiffVector.createOpNode(fr.output(), insList, backwardFn);
        if (graphOpTag != null) {
            result.opTag = graphOpTag;
            if (!Double.isNaN(graphScalarParam)) result.scalarParam = graphScalarParam;
            if (!Double.isNaN(graphScalarParam2)) result.scalarParam2 = graphScalarParam2;
        }
        return result;
    }

    /**
     * Apply this operation and wrap the result as an {@link IDiffTensor} with
     * guaranteed gradient flow back through both the vector and tensor graphs.
     *
     * <p>This is the <b>architectural guarantee</b> against gradient flow breakage
     * at the vector↔tensor boundary (see CLAUDE.md §7c). DL layers (Linear, Conv2d,
     * MHA, LSTM, etc.) <b>must</b> use this method instead of calling
     * {@link IDiffTensor#fromDiffVector} directly.</p>
     *
     * <p>Internally:
     * <ol>
     *   <li>Calls {@link #apply(IDiffVector...)} to get a vector-graph output</li>
     *   <li>Wraps via {@code IDiffTensor.fromDiffVector()} to create a tensor</li>
     *   <li>Sets {@code tensor.vectorSource = outputVector} to bridge gradients</li>
     * </ol>
     *
     * <p>During backward, the tensor graph flows into {@code vectorSource},
     * which triggers {@code backwardNested()} on the vector graph, reaching
     * CustomOp parameter leaves (Conv2d weight, Linear weight, etc.).</p>
     *
     * @param shape the desired output tensor shape (must match output vector size)
     * @param inputs differentiable vector inputs to this operation
     * @return differentiable tensor with bridged gradient flow
     */
    public IDiffTensor tensorApply(int[] shape, IDiffVector... inputs) {
        IDiffVector result = apply(inputs);
        IDiffTensor tensor = IDiffTensor.fromDiffVector(result, shape);
        // Architectural guarantee: link tensor leaf back to vector graph node
        // so backward can bridge gradients through the dual AD system.
        if (result instanceof RereDiffVector rv
                && tensor instanceof RereDiffTensor rt) {
            rt.vectorSource = rv;
        }
        return tensor;
    }

    /**
     * Overload accepting a variadic shape for convenience: {@code tensorApply(2, 3, inputs)}.
     */
    public IDiffTensor tensorApply(IDiffVector[] inputs, int... shape) {
        return tensorApply(shape, inputs);
    }

    /**
     * Walk the tensor graph from {@code t} back to the nearest {@code vectorSource},
     * creating vector-graph nodes for intermediate value-changing operations.
     *
     * <p>This connects the model's full computation graph in the vector domain,
     * enabling GPU/HPC graph execution to reach all parameter leaves. Without this,
     * bridge leaves would fragment the vector graph, making GPU backward compute
     * gradients only for the final layer's parameters.</p>
     *
     * <p>Shape-only ops (reshape, flatten, cat, stack, unsqueeze, squeeze, transpose,
     * permute) are treated as pass-through since they don't change values or gradients.
     * Value-changing ops (relu, sigmoid, tanh, gelu, etc.) get explicit vector nodes
     * with correct backward functions for GPU gradient computation.</p>
     *
     * @return a RereDiffVector connected to the tensor's vectorSource chain,
     *         or {@code null} if no vectorSource is reachable
     */
    private static RereDiffVector resolveTensorChain(RereDiffTensor t) {
        // 1. Walk backward through tensor graph, collecting value-changing ops
        List<String> ops = new ArrayList<>();
        RereDiffTensor cur = t;
        RereDiffVector vectorSource = null;
        int maxDepth = 30;

        while (cur != null && maxDepth-- > 0) {
            if (cur.vectorSource != null) {
                vectorSource = cur.vectorSource;
                break;
            }
            if (cur.inputs.isEmpty()) break;

            String op = cur.opTag;
            if (op != null && isValueChangingOp(op)) {
                ops.add(op);
            }
            // For shape-only or multi-input ops, skip to inputs[0]
            cur = cur.inputs.get(0);
        }

        if (vectorSource == null) return null;

        // 2. Check size compatibility: shape-changing ops (cat, stack, reshape)
        //    may have changed the tensor size vs vectorSource.
        //    If sizes differ, we can't use the chain approach — fall back to bridge leaf.
        int tSize = (int) t.value.totalSize();
        int vsSize = vectorSource.value.size();
        if (tSize != vsSize && ops.isEmpty()) {
            // Size changed but no value-changing ops — this is a shape-only transformation
            // (e.g., cat/stack batching). Create a leaf with the stacked data.
            // The gradient for this leaf is computed by the downstream CustomOp but
            // doesn't flow back to any parameter (it's just input data).
            return new RereDiffVector(IDoubleVector.of(t.value.toDoubleArray()));
        }

        // 3. Build vector chain forward (from vectorSource toward t's value)
        //    ops are collected in reverse order (outermost first), so iterate backward
        RereDiffVector chainNode = vectorSource;
        for (int j = ops.size() - 1; j >= 0; j--) {
            chainNode = createChainVectorNode(chainNode, ops.get(j));
        }

        return chainNode;
    }

    /** Check if an op tag represents a value-changing operation (not just shape change). */
    private static boolean isValueChangingOp(String op) {
        return switch (op) {
            case "relu", "sigmoid", "tanh", "gelu", "silu", "mish",
                 "leakyRelu", "elu", "selu", "softplus", "hardtanh",
                 "exp", "log", "sqrt", "square", "abs", "neg",
                 "softmax", "logSoftmax", "dropout" -> true;
            default -> false;
        };
    }

    /**
     * Create a vector-graph node that mirrors a tensor-graph unary op.
     * The node has the correct forward value and backward function for GPU gradient computation.
     */
    private static RereDiffVector createChainVectorNode(RereDiffVector input, String op) {
        double[] inData = input.value.getData();
        int n = inData.length;
        double[] outData = new double[n];

        // Compute forward value
        switch (op) {
            case "relu" -> { for (int i = 0; i < n; i++) outData[i] = Math.max(0, inData[i]); }
            case "sigmoid" -> { for (int i = 0; i < n; i++) outData[i] = 1.0 / (1.0 + Math.exp(-inData[i])); }
            case "tanh" -> { for (int i = 0; i < n; i++) outData[i] = Math.tanh(inData[i]); }
            case "gelu" -> {
                double sqrt2pi = Math.sqrt(2.0 / Math.PI);
                for (int i = 0; i < n; i++) {
                    double x = inData[i];
                    double inner = sqrt2pi * (x + 0.044715 * x * x * x);
                    outData[i] = 0.5 * x * (1.0 + Math.tanh(inner));
                }
            }
            case "silu" -> { for (int i = 0; i < n; i++) { double s = 1.0 / (1.0 + Math.exp(-inData[i])); outData[i] = inData[i] * s; } }
            case "mish" -> { for (int i = 0; i < n; i++) { double sp = Math.log(1 + Math.exp(inData[i])); outData[i] = inData[i] * Math.tanh(sp); } }
            case "exp" -> { for (int i = 0; i < n; i++) outData[i] = Math.exp(inData[i]); }
            case "log" -> { for (int i = 0; i < n; i++) outData[i] = Math.log(inData[i]); }
            case "sqrt" -> { for (int i = 0; i < n; i++) outData[i] = Math.sqrt(inData[i]); }
            case "square" -> { for (int i = 0; i < n; i++) outData[i] = inData[i] * inData[i]; }
            case "abs" -> { for (int i = 0; i < n; i++) outData[i] = Math.abs(inData[i]); }
            case "neg" -> { for (int i = 0; i < n; i++) outData[i] = -inData[i]; }
            default -> { for (int i = 0; i < n; i++) outData[i] = inData[i]; } // fallback: identity
        }

        // Capture forward output for backward derivative computation
        final double[] fwdOut = outData.clone();
        final double[] fwdIn = inData.clone();

        Consumer<IDoubleVector> backwardFn = switch (op) {
            case "relu" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = fwdIn[i] > 0 ? g[i] : 0;
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "sigmoid" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = g[i] * fwdOut[i] * (1 - fwdOut[i]);
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "tanh" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = g[i] * (1 - fwdOut[i] * fwdOut[i]);
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "gelu" -> gradOut -> {
                double sqrt2pi = Math.sqrt(2.0 / Math.PI);
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) {
                    double x = fwdIn[i];
                    double inner = sqrt2pi * (x + 0.044715 * x * x * x);
                    double tanhInner = Math.tanh(inner);
                    double cdf = 0.5 * (1.0 + tanhInner);
                    double pdf = sqrt2pi * (1.0 + 0.134145 * x * x) * (1 - tanhInner * tanhInner);
                    dIn[i] = g[i] * (cdf + 0.5 * x * pdf);
                }
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "silu" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) {
                    double s = 1.0 / (1.0 + Math.exp(-fwdIn[i]));
                    dIn[i] = g[i] * (s + fwdIn[i] * s * (1 - s));
                }
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "mish" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) {
                    double x = fwdIn[i];
                    double sp = Math.log(1 + Math.exp(x));
                    double tsp = Math.tanh(sp);
                    double sig = 1.0 / (1.0 + Math.exp(-x));
                    dIn[i] = g[i] * (tsp + x * sig * (1 - tsp * tsp));
                }
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "exp" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = g[i] * fwdOut[i];
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "log" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = g[i] / fwdIn[i];
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "sqrt" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = g[i] / (2 * fwdOut[i]);
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "square" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = g[i] * 2 * fwdIn[i];
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "abs" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = g[i] * (fwdIn[i] >= 0 ? 1 : -1);
                input.accGrad(IDoubleVector.of(dIn));
            };
            case "neg" -> gradOut -> {
                double[] g = gradOut.getData();
                double[] dIn = new double[n];
                for (int i = 0; i < n; i++) dIn[i] = -g[i];
                input.accGrad(IDoubleVector.of(dIn));
            };
            default -> gradOut -> input.accGrad(gradOut); // identity backward
        };

        RereDiffVector node = (RereDiffVector) RereDiffVector.createOpNode(
                IDoubleVector.of(outData), List.of(input), backwardFn);
        node.opTag = op;
        return node;
    }

    /**
     * Compute the forward pass from raw input values.
     * @return output value plus opaque context accessible in {@link #backward}.
     */
    protected abstract ForwardResult forward(IDoubleVector[] rawInputs);

    /**
     * Compute per-input gradients given the upstream gradient and saved forward context.
     * @param gradOutput upstream gradient from subsequent operations
     * @param forwardContext the context object returned by {@link #forward}
     * @return per-input gradient arrays in the same order as {@link #apply} inputs;
     *         any element may be null if that input doesn't require gradients
     */
    protected abstract IDoubleVector[] backward(IDoubleVector gradOutput, Object forwardContext);

    /** Result of the forward pass: output value plus opaque context for backward. */
    public record ForwardResult(IDoubleVector output, Object context) {}
}
