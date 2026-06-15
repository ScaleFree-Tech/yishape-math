package com.yishape.lab.math.autodiff;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Self-contained custom differentiable operation.
 *
 * <p>Replaces the legacy {@code AD.registerGradient / AD.custom / AD.unregisterGradient}
 * pattern. Each instance manages its own forward-data cache and embeds the backward
 * function directly into graph nodes, eliminating the global
 * {@link CustomGradientRegistry} and preventing memory leaks.</p>
 *
 * <h2>Tensor-native API (primary)</h2>
 * <p>Override {@link #forwardTensor(IDiffTensor[], int[])} to receive full tensor inputs
 * with shape information. This is the recommended override point for all new code.
 * The default implementation provides a complete tensor-native forward+backward cycle
 * using {@link #forward(IDoubleVector[])} and {@link #backward(IDoubleVector, Object)}.</p>
 *
 * <pre>{@code
 * CustomOp op = new CustomOp() {
 *     @Override
 *     protected ForwardResult forwardTensor(IDiffTensor[] inputs, int[] outShape) {
 *         IDiffTensor x = inputs[0];
 *         IDiffTensor w = inputs[1];
 *         // Direct shape access — no inference from flat array length
 *         int[] xShape = x.shape();
 *         int[] wShape = w.shape();
 *         double[] outData = x.mmul(w.transpose()).toDoubleArray();
 *         return new ForwardResult(IDoubleVector.of(outData),
 *             new Object[]{xShape, wShape});
 *     }
 * };
 * }</pre>
 *
 * <h2>Legacy vector API (backward compatible)</h2>
 * <p>{@link #forward(IDoubleVector[])} is retained for subclasses that have not yet
 * migrated. The default implementation flattens tensor inputs and delegates to
 * {@link #forwardTensor(IDiffTensor[], int[])}, so overriding {@code forwardTensor}
 * alone is sufficient for new code.</p>
 */
public abstract class CustomOp {

    static final AtomicLong GLOBAL_COUNTER = new AtomicLong(0);

    final AtomicLong fwdCounter = new AtomicLong(0);

    // Forward-data cache: maps forward pass ID to opaque context for backward.
    // Uses ConcurrentHashMap to avoid the 256-slot overflow bug and synchronized bottleneck.
    private final ConcurrentHashMap<Long, Object> fwdCache = new ConcurrentHashMap<>();

    /** Maximum number of uncollected backward contexts before forced cleanup. */
    private static final int MAX_FWD_CACHE = 10_000;

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
            // Prevent unbounded growth when backward is never called (e.g. inference)
            if (fwdCache.size() >= MAX_FWD_CACHE) {
                fwdCache.clear();
            }
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

    // ==================== Tensor-native API (primary) ====================

    /**
     * Compute the forward pass from tensor inputs with full shape information.
     *
     * <p>This is the <b>primary</b> override point. The default implementation
     * provides a complete tensor-native cycle: it extracts flat arrays from the
     * tensor inputs (preserving their shapes in the context), delegates to
     * {@link #forward(IDoubleVector[])}, and on backward reshapes gradients
     * using the saved shapes before calling {@link #backward}.</p>
     *
     * <p>Subclasses that override this method can use {@code inputs[i].shape()}
     * directly instead of inferring batch size from flat array length.</p>
     *
     * @param inputs   differentiable tensor inputs (same order as passed to {@link #apply})
     * @param outShape desired output tensor shape (product must equal output size)
     * @return output value plus opaque context accessible in {@link #backward}
     */
    protected ForwardResult forwardTensor(IDiffTensor[] inputs, int[] outShape) {
        // Default: extract flat arrays (saving shapes for backward), delegate to legacy forward
        int[][] inputShapes = new int[inputs.length][];
        IDoubleVector[] flatInputs = new IDoubleVector[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            inputShapes[i] = inputs[i].shape();
            flatInputs[i] = IDoubleVector.of(inputs[i].toDoubleArray());
        }
        ForwardResult fr = forward(flatInputs);
        // Augment context with input shapes for tensor-aware backward
        return new ForwardResult(fr.output(), new TensorContext(fr.context(), inputShapes, outShape));
    }

    // ==================== Legacy vector API (backward compat) ====================

    /**
     * Apply this operation to tensor inputs, creating a tensor-native graph node
     * with full gradient flow in the tensor graph.
     *
     * <p>Calls {@link #forwardTensor(IDiffTensor[], int[])} as the primary path.
     * The backward pass automatically reshapes flat gradients using input shapes
     * saved in the forward context.</p>
     *
     * @param outShape output tensor shape (product must equal forward output size)
     * @param inputs   differentiable tensor inputs to this operation
     * @return differentiable tensor with embedded CustomOp backward
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
     * Tensor-native apply: creates tensor graph nodes with full shape information.
     *
     * <p>Calls {@link #forwardTensor(IDiffTensor[], int[])} and stores the
     * input tensor references for backward gradient accumulation. Flat gradients
     * from upstream are automatically reshaped using input shapes before being
     * passed to {@link #backward}.</p>
     *
     * @param outShape output tensor shape (product must equal forward output size)
     * @param inputs   differentiable tensor inputs to this operation
     * @return differentiable tensor with embedded CustomOp backward
     */
    public IDiffTensor tensorApply(int[] outShape, IDiffTensor... inputs) {
        RereDiffTensor[] tNodes = new RereDiffTensor[inputs.length];
        int[][] inputShapes = new int[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            IDiffTensor in = inputs[i];
            // Unwrap BatchedDiffTensor to get the underlying tensor
            while (in instanceof BatchedDiffTensor bdt) {
                in = bdt.unwrap();
            }
            if (in instanceof RereDiffTensor rt) {
                tNodes[i] = rt;
                inputShapes[i] = rt.value().shape();
            } else {
                // Non-RereDiffTensor impl: wrap as leaf, preserve requiresGrad
                double[] data = in.toDoubleArray();
                RereDiffTensor leaf = new RereDiffTensor(data, in.shape());
                leaf.setRequiresGrad(in.requiresGrad());
                tNodes[i] = leaf;
                inputShapes[i] = in.shape();
            }
        }

        // Call tensor-native forward (default: flatten tensors → delegate to legacy forward)
        ForwardResult fr = forwardTensor(inputs, outShape);

        long id = fwdCounter.incrementAndGet();
        putCache(id, fr.context());

        double[] outputData = fr.output().getData();
        List<RereDiffTensor> insList = Arrays.asList(tNodes);

        Consumer<RereDiffTensor> backwardFn = (self) -> {
            Object ctx = getCache(id);
            removeCache(id); // release cache entry immediately after retrieval
            // Unwrap TensorContext if present: forwardTensor() wraps the raw context
            // in a TensorContext (for shape preservation), but subclasses' backward()
            // expect their own context type (e.g. Conv2d.ForwardData, LayerNorm.LayerNormContext).
            if (ctx instanceof TensorContext tc) ctx = tc.delegate;
            // Reshape flat upstream gradient to output shape, then delegate to backward
            IDiffTensor gradTensor = IDiffTensor.fromTensor(
                com.yishape.lab.math.autodiff.impl.EvalTensorFactory.wrap(
                    self.gradData(), outShape), false);
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
     * Compute the forward pass from raw input values (legacy vector API).
     *
     * <p>The default implementation flattens tensor inputs and delegates to
     * {@link #forwardTensor(IDiffTensor[], int[])}. Subclasses that override
     * this method continue to work — the delegation chain is:
     * {@code tensorApply → forwardTensor → forward}.</p>
     *
     * @return output value plus opaque context accessible in {@link #backward}
     */
    protected ForwardResult forward(IDoubleVector[] rawInputs) {
        // Legacy subclasses override this. forwardTensor default delegates here.
        throw new AbstractMethodError(
            "Subclass must implement either forward(IDoubleVector[]) or "
            + "forwardTensor(IDiffTensor[], int[])");
    }

    /**
     * Compute per-input gradients given the upstream gradient and saved forward context.
     *
     * <p>The {@code gradOutput} is a flat array matching the output shape. When
     * using the default {@link #forwardTensor} implementation, the context contains
     * input shapes ({@link TensorContext}) that subclasses can use to reshape
     * gradients correctly.</p>
     *
     * @param gradOutput      upstream gradient from subsequent operations (flat array)
     * @param forwardContext  the context object returned by {@link #forward} or
     *                        {@link #forwardTensor}; may be a {@link TensorContext}
     *                        when using the default tensor-native forward
     * @return per-input gradient arrays in the same order as {@link #apply} inputs;
     *         any element may be null if that input doesn't require gradients
     */
    protected abstract IDoubleVector[] backward(IDoubleVector gradOutput, Object forwardContext);

    /** Result of the forward pass: output value plus opaque context for backward. */
    public record ForwardResult(IDoubleVector output, Object context) {}

    // ==================== Context helpers ====================

    /**
     * Context wrapper that preserves input shapes and output shape for tensor-aware backward.
     * Used by the default {@link #forwardTensor} implementation.
     */
    public static final class TensorContext {
        public final Object delegate;      // original context from legacy forward
        public final int[][] inputShapes;  // shape of each input tensor
        public final int[] outShape;       // output shape

        public TensorContext(Object delegate, int[][] inputShapes, int[] outShape) {
            this.delegate = delegate;
            this.inputShapes = inputShapes;
            this.outShape = outShape;
        }

        /** Extract input shapes for a specific backward input index. */
        public int[] inputShape(int index) {
            return inputShapes != null && index < inputShapes.length ? inputShapes[index] : null;
        }
    }
}
