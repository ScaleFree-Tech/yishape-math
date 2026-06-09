package com.yishape.lab.math.autodiff;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;

/**
 * Self-contained custom differentiable operation.
 *
 * <p>Replaces the legacy {@code AD.registerGradient / AD.custom / AD.unregisterGradient}
 * pattern. Each instance manages its own forward-data cache and embeds the backward
 * function directly into graph nodes, eliminating the global
 * {@link CustomGradientRegistry} and preventing memory leaks.</p>
 *
 * <p>Subclass and implement {@link #forward(IDoubleVector[])} and
 * {@link #backward(IDoubleVector, Object)}.</p>
 *
 * <p><b>Pure tensor graph (all computations in the tensor domain):</b>
 * <pre>{@code
 * CustomOp op = new CustomOp() { ... };
 * IDiffTensor result = op.apply(new int[]{n}, xTensor, wTensor);
 * }</pre>
 *
 * <p><b>Vector-compatible (via AD.op):</b>
 * <pre>{@code
 * CustomOp op = new CustomOp() { ... };
 * IDiffVector result = AD.op(op, xVector, wVector); // unwraps to tensors internally
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

    void putCache(long id, Object ctx) {
        if (ctx != null) {
            fwdCache.put(id, ctx);
        }
    }

    Object getCache(long id) {
        return fwdCache.get(id);
    }

    /** Remove the cached forward context after backward to prevent unbounded growth. */
    void removeCache(long id) {
        fwdCache.remove(id);
    }

    /**
     * Clear all cached forward contexts.
     * Called after GPU/HPC graph execution (which bypasses {@code loss.backward()})
     * to prevent the forward-data cache from growing unboundedly.
     */
    public void clearForwardCache() {
        fwdCache.clear();
    }

    /**
     * Apply this operation to tensor inputs, creating a tensor-native graph node
     * with full gradient flow in the tensor graph (no vector bridge needed).
     *
     * <p>This is the <b>primary API</b> for CustomOp — the replacement for both
     * the deprecated {@link #apply(IDiffVector...)} and the deprecated
     * {@link #tensorApply(int[], IDiffVector...)} bridge.</p>
     *
     * <p>Internally:
     * <ol>
     *   <li>Extracts flat {@code double[]} from each tensor input</li>
     *   <li>Calls {@link #forward(IDoubleVector[])} on the flat data</li>
     *   <li>Creates a {@link RereDiffTensor} graph node wrapping the result</li>
     * </ol>
     *
     * <p>The result is a pure tensor graph node — gradients flow directly through
     * the tensor graph without the old {@code vectorSource} bridge.</p>
     *
     * @param outShape desired output tensor shape (product must equal output size)
     * @param inputs   differentiable tensor inputs to this operation
     * @return differentiable tensor with gradient flow in the tensor graph
     */
    public IDiffTensor apply(int[] outShape, IDiffTensor... inputs) {
        return tensorApply(outShape, inputs);
    }

    /**
     * Convenience overload: {@code op.apply(new IDiffTensor[]{x, w}, outCh, H, W)}.
     */
    public IDiffTensor apply(IDiffTensor[] inputs, int... outShape) {
        return tensorApply(outShape, inputs);
    }

    /**
     * Tensor-native apply: creates tensor graph nodes directly with no vector bridge.
     *
     * <p>New code should use {@link #apply(int[], IDiffTensor...)} instead.
     * This method is retained as the internal implementation delegate.</p>
     *
     * <p>No {@code vectorSource} bridge is needed — gradients flow entirely within
     * the tensor graph.</p>
     *
     * @param outShape output tensor shape (product must equal forward output size)
     * @param inputs differentiable tensor inputs to this operation
     * @return differentiable tensor with embedded CustomOp backward
     */
    public IDiffTensor tensorApply(int[] outShape, IDiffTensor... inputs) {
        RereDiffTensor[] tNodes = new RereDiffTensor[inputs.length];
        IDoubleVector[] rawVectors = new IDoubleVector[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            IDiffTensor in = inputs[i];
            // Unwrap BatchedDiffTensor to get the underlying tensor
            while (in instanceof BatchedDiffTensor bdt) {
                in = bdt.unwrap();
            }
            if (in instanceof RereDiffTensor rt) {
                tNodes[i] = rt;
                rawVectors[i] = IDoubleVector.of(rt.value().toDoubleArray());
            } else {
                // Non-RereDiffTensor impl (e.g., ConstantDiffTensor or user-defined):
                // wrap its data as a leaf or non-leaf node preserving requiresGrad
                double[] data = in.toDoubleArray();
                tNodes[i] = new RereDiffTensor(data, in.shape());
                tNodes[i].setRequiresGrad(in.requiresGrad());
                rawVectors[i] = IDoubleVector.of(data);
            }
        }

        ForwardResult fr = forward(rawVectors);

        long id = fwdCounter.incrementAndGet();
        putCache(id, fr.context());

        double[] outputData = fr.output().getData();
        List<RereDiffTensor> insList = Arrays.asList(tNodes);

        Consumer<RereDiffTensor> backwardFn = (self) -> {
            Object ctx = getCache(id);
            removeCache(id); // release cache entry immediately after retrieval
            IDoubleVector[] grads = backward(IDoubleVector.of(self.gradData()), ctx);
            for (int j = 0; j < tNodes.length && j < grads.length; j++) {
                if (grads[j] != null) {
                    tNodes[j].accGrad(grads[j].getData());
                }
            }
        };

        RereDiffTensor result = new RereDiffTensor(outputData, outShape, insList, backwardFn,
            graphOpTag != null ? graphOpTag : "CustomOp");
        if (graphOpTag != null) {
            if (!Double.isNaN(graphScalarParam)) result.setScalarParam(graphScalarParam);
            if (!Double.isNaN(graphScalarParam2)) result.setScalarParam2(graphScalarParam2);
        }
        return result;
    }

    /**
     * Convenience overload: {@code op.tensorApply(new IDiffTensor[]{x, w}, outCh, H, W)}.
     */
    public IDiffTensor tensorApply(IDiffTensor[] inputs, int... outShape) {
        return tensorApply(outShape, inputs);
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
