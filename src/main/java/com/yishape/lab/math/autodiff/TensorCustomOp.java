package com.yishape.lab.math.autodiff;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;

/**
 * Self-contained custom differentiable operation with tensor shape awareness.
 *
 * <p>Like {@link CustomOp} but forward/backward operate on {@link IDoubleTensor}
 * (with shape metadata) instead of flat {@link IDoubleVector}. This is the long-term
 * replacement for the flatten→CustomOp→wrap bridge pattern.</p>
 *
 * <p>Subclass and implement {@link #forward(IDoubleTensor[])} and
 * {@link #backward(IDoubleTensor, Object)}, then call {@link #apply(IDiffTensor...)}
 * to build computation graph nodes.</p>
 *
 * <p>Typical usage:
 * <pre>{@code
 * TensorCustomOp op = new TensorCustomOp() {
 *     protected ForwardResult forward(IDoubleTensor[] inputs) {
 *         int[] inShape = inputs[0].shape(); // e.g. [C, H, W]
 *         int[] outShape = computeOutputShape(inShape);
 *         double[] out = ...;
 *         MyCtx ctx = new MyCtx(...);
 *         return new ForwardResult(new RereDoubleTensor(out, outShape), ctx);
 *     }
 *     protected IDoubleTensor[] backward(IDoubleTensor g, Object ctx) {
 *         MyCtx c = (MyCtx) ctx;
 *         return new IDoubleTensor[]{ new RereDoubleTensor(dX, c.inputShape()) };
 *     }
 * };
 * IDiffTensor result = op.apply(x); // tensor in, tensor out
 * }</pre>
 */
public abstract class TensorCustomOp {

    final AtomicLong fwdCounter = new AtomicLong(0);

    // Forward-data cache: maps forward pass ID to opaque context for backward.
    private final ConcurrentHashMap<Long, Object> fwdCache = new ConcurrentHashMap<>();

    private void putCache(long id, Object ctx) {
        fwdCache.put(id, ctx);
    }

    private Object getCache(long id) {
        return fwdCache.get(id);
    }

    /**
     * Apply this operation to the given tensors.
     * Always creates a computation graph node with embedded backward.
     */
    public IDiffTensor apply(IDiffTensor... inputs) {
        RereDiffTensor[] tensorNodes = new RereDiffTensor[inputs.length];
        IDoubleTensor[] rawTensors = new IDoubleTensor[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            tensorNodes[i] = (RereDiffTensor) inputs[i];
            rawTensors[i] = inputs[i].detach();
        }

        ForwardResult fr = forward(rawTensors);

        long id = fwdCounter.incrementAndGet();
        putCache(id, fr.context());

        int[] outShape = fr.output().shape();
        double[] outputData = fr.output().toDoubleArray();

        List<RereDiffTensor> insList = Arrays.asList(tensorNodes);
        Consumer<RereDiffTensor> backwardFn = (self) -> {
            Object ctx = getCache(id);
            try {
                IDoubleTensor gradTensor = new RereDoubleTensor(self.gradData(), outShape);
                IDoubleTensor[] grads = backward(gradTensor, ctx);
                for (int j = 0; j < tensorNodes.length && j < grads.length; j++) {
                    if (grads[j] != null) {
                        tensorNodes[j].accGrad(grads[j].toDoubleArray());
                    }
                }
            } finally {
                fwdCache.remove(id);
            }
        };

        return new RereDiffTensor(outputData, outShape, insList, backwardFn, "CustomOp");
    }

    /**
     * Compute the forward pass from raw input tensors.
     * @return output tensor plus opaque context accessible in {@link #backward}.
     */
    protected abstract ForwardResult forward(IDoubleTensor[] rawInputs);

    /**
     * Compute per-input gradients given the upstream gradient and saved forward context.
     * @param gradOutput upstream gradient tensor from subsequent operations
     * @param forwardContext the context object returned by {@link #forward}
     * @return per-input gradient tensors in the same order as {@link #apply} inputs;
     *         any element may be null if that input doesn't require gradients
     */
    protected abstract IDoubleTensor[] backward(IDoubleTensor gradOutput, Object forwardContext);

    /** Result of the forward pass: output tensor plus opaque context for backward. */
    public record ForwardResult(IDoubleTensor output, Object context) {}
}
