package com.yishape.lab.math.autodiff;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import com.yishape.lab.math.linalg.IDoubleVector;
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
        for (int i = 0; i < inputs.length; i++) {
            nodes[i] = (RereDiffVector) inputs[i];
            raw[i] = nodes[i].getValue();
        }

        ForwardResult fr = forward(raw);

        long id = fwdCounter.incrementAndGet();
        putCache(id, fr.context());

        // Arrays.asList wraps the original array (no copy), unlike List.of for N>2
        List<RereDiffVector> insList = Arrays.asList(nodes);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            Object ctx = getCache(id);
            IDoubleVector[] grads = backward(gradOut, ctx);
            for (int j = 0; j < nodes.length && j < grads.length; j++) {
                if (grads[j] != null) {
                    nodes[j].accGrad(grads[j]);
                }
            }
        };

        return RereDiffVector.createOpNode(fr.output(), insList, backwardFn);
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
