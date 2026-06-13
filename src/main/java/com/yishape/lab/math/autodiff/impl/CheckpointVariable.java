package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * Gradient-checkpoint node: discards intermediate activations and recomputes on backward.
 * 梯度检查点节点：丢弃中间激活，反向时重算前向。
 *
 * <h3>IMPORTANT: Weight staleness risk</h3>
 * The {@code forwardFn} lambda captures whatever references were in scope at construction time.
 * If an optimizer updates weight tensors between the original forward and the checkpoint backward,
 * the recomputed subgraph will use the <b>updated</b> weights, producing incorrect gradients.
 *
 * <p>To avoid this, {@code forwardFn} must either:</p>
 * <ol>
 *   <li>Capture <b>immutable value snapshots</b> of all weights (e.g., {@code weight.clone()})
 *       and use those snapshots during recomputation, or</li>
 *   <li>Be used only when optimizer steps occur <b>after</b> all checkpoint backward passes
 *       (e.g., gradient accumulation mode where optimizer.step() is deferred).</li>
 * </ol>
 *
 * <p>The {@code originalInput} is saved by value at construction time and is safe from staleness.</p>
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
        this.tensor.setIsLeaf(false);

        // Re-run forward from saved input, then backward through recomputed subgraph
        // 从保存的输入重算前向，再对重算子图做反向
        this.tensor.setBackwardFn((self) -> {
            double[] g = self.gradData();
            IDiffVector recomputed = forwardFn.apply(originalInput);
            if (!(recomputed instanceof RereDiffVector out)) {
                throw new IllegalStateException(
                    "Checkpoint forwardFn must return RereDiffVector, got " + recomputed.getClass().getName());
            }
            if (!out.tensor.requiresGrad()) return;
            out.tensor.setGradData(g);
            out.tensor.backwardImpl();
        });
    }
}
