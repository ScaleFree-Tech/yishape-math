package com.yishape.lab.math.autodiff;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import com.yishape.lab.math.linalg.IDoubleVector;
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
                nodes[i] = null; // will be replaced by bridge leaf
            } else {
                nodes[i] = (RereDiffVector) inputs[i];
                raw[i] = nodes[i].getValue();
            }
        }

        ForwardResult fr = forward(raw);

        long id = fwdCounter.incrementAndGet();
        putCache(id, fr.context());

        // Build input list with bridge leaves for tensor-backed inputs
        RereDiffVector[] effectiveNodes = new RereDiffVector[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            if (tensorBridges[i] != null) {
                // Create a bridge leaf in the vector graph that mirrors the tensor leaf
                RereDiffTensor t = tensorBridges[i];
                effectiveNodes[i] = new RereDiffVector(IDoubleVector.of(t.value.toDoubleArray()));
            } else {
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
                        // Bridge gradient back to tensor's grad array
                        RereDiffTensor t = tensorBridges[j];
                        if (t.grad == null) {
                            t.grad = new double[grads[j].size()];
                        }
                        double[] g = grads[j].getData();
                        for (int k = 0; k < g.length; k++) {
                            t.grad[k] += g[k];
                        }
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
