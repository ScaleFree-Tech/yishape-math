package com.yishape.lab.math.autodiff.vmap;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.BatchedDiffTensor;
import com.yishape.lab.math.autodiff.BatchedDiffVector;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.util.Messages;

/**
 * JAX-style vmap transform: stacks inputs, wraps in batched tensors, executes
 * {@code fn} once with single-graph batched execution, then unstacks results.
 *
 * <p>This class contains the high-level vmap orchestration logic (stack →
 * wrap → execute → unwrap → unstack). Low-level batched computation primitives
 * are in {@link VMap}.
 *
 * <p>All methods are static. For the public API, use {@link AD#vmap},
 * {@link AD#vmapT}, etc., which delegate here.
 *
 * @author lteb2
 */
public final class VMapTransform {

    private VMapTransform() {
        // utility class
    }

    // ==================== IDiffVector vmap ====================

    /**
     * Vectorized map: stacks inputs into a single flat vector, executes
     * {@code fn} once on a {@link BatchedDiffVector}, then unstacks the result.
     *
     * @param fn  function R^D → R^M operating on single elements
     * @param xs  list of input vectors, all same length D
     * @return array of per-element outputs, each of length M
     */
    public static IDiffVector[] vmap(Function<IDiffVector, IDiffVector> fn, List<? extends IDiffVector> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();
        int d = xs.get(0).size();

        for (int i = 1; i < n; i++) {
            if (xs.get(i).size() != d) {
                throw new IllegalArgumentException(
                    Messages.get("vmap.dimension_mismatch", d, xs.get(i).size()));
            }
        }

        double[][] raw = new double[n][];
        for (int i = 0; i < n; i++) {
            raw[i] = xs.get(i).getValue().getData();
        }
        double[] stacked = VMap.INSTANCE.stack(raw);

        IDiffVector batchedInput = AD.vector(stacked);
        BatchedDiffVector bdv = new BatchedDiffVector(batchedInput, n, d);
        IDiffVector batchedResult = fn.apply(bdv);

        IDiffVector flat = (batchedResult instanceof BatchedDiffVector b)
            ? b.unwrap() : batchedResult;
        int outDim = flat.size() / n;
        IDiffVector[] ys = new IDiffVector[n];
        for (int i = 0; i < n; i++) {
            ys[i] = flat.slice(i * outDim, (i + 1) * outDim);
        }
        return ys;
    }

    /**
     * Convenience: apply fn to each element and sum the results.
     */
    public static IDiffVector vmapSum(Function<IDiffVector, IDiffVector> fn, List<? extends IDiffVector> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();
        int d = xs.get(0).size();

        for (int i = 1; i < n; i++) {
            if (xs.get(i).size() != d) {
                throw new IllegalArgumentException(
                    Messages.get("vmap.dimension_mismatch", d, xs.get(i).size()));
            }
        }

        double[][] raw = new double[n][];
        for (int i = 0; i < n; i++) raw[i] = xs.get(i).getValue().getData();
        double[] stacked = VMap.INSTANCE.stack(raw);

        IDiffVector batchedInput = AD.vector(stacked);
        BatchedDiffVector bdv = new BatchedDiffVector(batchedInput, n, d);
        IDiffVector result = fn.apply(bdv);

        IDiffVector flat = (result instanceof BatchedDiffVector b) ? b.unwrap() : result;
        int outDim = flat.size() / n;
        if (outDim == 1) {
            return flat.sum();
        }
        return flat.reshape(n, outDim).sum(0);
    }

    /**
     * Convenience: apply fn to each element and return the mean.
     */
    public static IDiffVector vmapMean(Function<IDiffVector, IDiffVector> fn, List<? extends IDiffVector> xs) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();
        int d = xs.get(0).size();

        for (int i = 1; i < n; i++) {
            if (xs.get(i).size() != d) {
                throw new IllegalArgumentException(
                    Messages.get("vmap.dimension_mismatch", d, xs.get(i).size()));
            }
        }

        double[][] raw = new double[n][];
        for (int i = 0; i < n; i++) raw[i] = xs.get(i).getValue().getData();
        double[] stacked = VMap.INSTANCE.stack(raw);

        IDiffVector batchedInput = AD.vector(stacked);
        BatchedDiffVector bdv = new BatchedDiffVector(batchedInput, n, d);
        IDiffVector result = fn.apply(bdv);

        IDiffVector flat = (result instanceof BatchedDiffVector b) ? b.unwrap() : result;
        int outDim = flat.size() / n;
        if (outDim == 1) {
            return flat.mean();
        }
        return flat.reshape(n, outDim).sum(0).div(n);
    }

    // ==================== IDiffTensor vmap ====================

    /**
     * JAX-style vmap over tensors. Stacks inputs along {@code inAxes},
     * executes {@code fn} once with single-graph batched execution, then
     * unstacks results along {@code outAxes}.
     *
     * @param fn      function to vectorize (receives BatchedDiffTensor with batch at dim 0)
     * @param xs      input tensors to stack along inAxes
     * @param inAxes  axis to stack inputs along (0 = leading batch dim, -1 = trailing)
     * @param outAxes axis in the output where the batch dim should appear
     * @return unstacked per-sample results
     */
    public static IDiffTensor[] vmapT(Function<IDiffTensor, IDiffTensor> fn,
                                      List<? extends IDiffTensor> xs,
                                      int inAxes, int outAxes) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();

        IDiffTensor[] rest = new IDiffTensor[n - 1];
        for (int i = 1; i < n; i++) {
            rest[i - 1] = xs.get(i);
        }
        IDiffTensor batched = xs.get(0).stack(inAxes, rest);

        if (inAxes != 0) {
            int ndim = batched.rank();
            int[] perm = new int[ndim];
            perm[0] = inAxes < 0 ? ndim - 1 + inAxes : inAxes;
            int idx = 1;
            for (int d = 0; d < ndim; d++) {
                if (d != perm[0]) perm[idx++] = d;
            }
            batched = batched.permute(perm);
        }

        BatchedDiffTensor bdt = new BatchedDiffTensor(batched);
        IDiffTensor result = fn.apply(bdt);

        IDiffTensor flat = (result instanceof BatchedDiffTensor b) ? b.unwrap() : result;

        if (outAxes != 0) {
            int ndim = flat.rank();
            int targetOut = outAxes < 0 ? ndim + outAxes : outAxes;
            int[] perm = new int[ndim];
            int idx = 0;
            for (int d = 1; d <= targetOut; d++) perm[idx++] = d;
            perm[idx++] = 0;
            for (int d = targetOut + 1; d < ndim; d++) perm[idx++] = d;
            flat = flat.permute(perm);
        }

        int unstackDim = outAxes < 0 ? flat.rank() + outAxes : outAxes;
        IDiffTensor[] ys = new IDiffTensor[n];
        for (int i = 0; i < n; i++) {
            ys[i] = flat.select(unstackDim, i);
        }
        return ys;
    }

    /** Shorthand for {@code vmapT(fn, xs, 0, 0)}. */
    public static IDiffTensor[] vmapT(Function<IDiffTensor, IDiffTensor> fn, List<? extends IDiffTensor> xs) {
        return vmapT(fn, xs, 0, 0);
    }

    /**
     * JAX-style vmap that returns a stacked (still-batched) result instead of
     * unstacking. Building block for nested vmap.
     *
     * @param fn     function receiving a BatchedDiffTensor
     * @param xs     input tensors to stack and batch
     * @param inAxes axis along which to stack (0 = leading batch dim)
     * @return BatchedDiffTensor wrapping the stacked result
     */
    public static IDiffTensor vmapStackedT(Function<IDiffTensor, IDiffTensor> fn,
                                            List<? extends IDiffTensor> xs,
                                            int inAxes) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();

        IDiffTensor[] rest = new IDiffTensor[n - 1];
        for (int i = 1; i < n; i++) {
            rest[i - 1] = xs.get(i);
        }
        IDiffTensor batched = xs.get(0).stack(inAxes, rest);

        if (inAxes != 0) {
            int ndim = batched.rank();
            int[] perm = new int[ndim];
            perm[0] = inAxes;
            int idx = 1;
            for (int d = 0; d < ndim; d++) {
                if (d != inAxes) perm[idx++] = d;
            }
            batched = batched.permute(perm);
        }

        BatchedDiffTensor wrapper = new BatchedDiffTensor(batched);
        IDiffTensor result = fn.apply(wrapper);

        if (result instanceof BatchedDiffTensor bdt) {
            return bdt;
        }
        return new BatchedDiffTensor(result);
    }

    /** Shorthand for {@code vmapStackedT(fn, xs, 0)}. */
    public static IDiffTensor vmapStackedT(Function<IDiffTensor, IDiffTensor> fn,
                                            List<? extends IDiffTensor> xs) {
        return vmapStackedT(fn, xs, 0);
    }

    /**
     * JAX-style multi-axis vmap: each input tensor can have its batch dimension
     * at a different axis.
     *
     * @param fn      function receiving list of BatchedDiffTensors
     * @param xs      input tensors, each with its own batch dim
     * @param inAxes  per-input batch axis (must match xs.size()); negative = from end
     * @param outAxes axis in the output where the batch dim should appear
     * @return unstacked per-sample results
     */
    public static IDiffTensor[] vmapMultiT(Function<List<IDiffTensor>, IDiffTensor> fn,
                                            List<? extends IDiffTensor> xs,
                                            int[] inAxes, int outAxes) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        if (inAxes.length != xs.size()) {
            throw new IllegalArgumentException(
                "inAxes.length=" + inAxes.length + " but xs.size()=" + xs.size());
        }
        int n = xs.size();

        List<IDiffTensor> batchedInputs = new ArrayList<>(n);
        int batchSize = -1;
        for (int i = 0; i < n; i++) {
            IDiffTensor x = xs.get(i);
            int ax = inAxes[i];
            int ndim = x.rank();
            int batchAxis = ax < 0 ? ndim + ax : ax;

            int bs = x.dim(batchAxis);
            if (batchSize < 0) batchSize = bs;
            else if (batchSize != bs) {
                throw new IllegalArgumentException(
                    "Inconsistent batch sizes: input[0] batch=" + batchSize
                    + " but input[" + i + "] batch=" + bs + " at axis " + ax);
            }

            if (batchAxis != 0) {
                int[] perm = new int[ndim];
                perm[0] = batchAxis;
                int idx = 1;
                for (int d = 0; d < ndim; d++) {
                    if (d != batchAxis) perm[idx++] = d;
                }
                x = x.permute(perm);
            }
            batchedInputs.add(new BatchedDiffTensor(x));
        }

        IDiffTensor result = fn.apply(batchedInputs);
        IDiffTensor flat = (result instanceof BatchedDiffTensor b) ? b.unwrap() : result;

        if (outAxes != 0) {
            int ndim = flat.rank();
            int targetOut = outAxes < 0 ? ndim + outAxes : outAxes;
            int[] perm = new int[ndim];
            int idx = 0;
            for (int d = 1; d <= targetOut; d++) perm[idx++] = d;
            perm[idx++] = 0;
            for (int d = targetOut + 1; d < ndim; d++) perm[idx++] = d;
            flat = flat.permute(perm);
        }

        int unstackDim = outAxes < 0 ? flat.rank() + outAxes : outAxes;
        IDiffTensor[] ys = new IDiffTensor[batchSize];
        for (int i = 0; i < batchSize; i++) {
            ys[i] = flat.select(unstackDim, i);
        }
        return ys;
    }

    /**
     * JAX-style vmap with multiple input tensors (varargs version).
     *
     * @param fn       function receiving an array of BatchedDiffTensors
     * @param inAxes   per-input stacking axes (length must match inputs.length)
     * @param inputs   list-of-lists: inputs[i] is the list of tensors for the i-th argument
     * @return per-sample results
     */
    @SafeVarargs
    public static IDiffTensor[] vmapMultiT(Function<IDiffTensor[], IDiffTensor> fn,
                                            int[] inAxes,
                                            List<? extends IDiffTensor>... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = inputs[0].size();
        for (int i = 1; i < inputs.length; i++) {
            if (inputs[i].size() != n) {
                throw new IllegalArgumentException(
                    "All input lists must have the same size. Expected " + n
                    + " but input[" + i + "] has " + inputs[i].size());
            }
        }

        IDiffTensor[] batched = new IDiffTensor[inputs.length];
        for (int j = 0; j < inputs.length; j++) {
            List<? extends IDiffTensor> list = inputs[j];
            int ax = inAxes != null && j < inAxes.length ? inAxes[j] : 0;
            IDiffTensor[] rest = new IDiffTensor[n - 1];
            for (int i = 1; i < n; i++) rest[i - 1] = list.get(i);
            batched[j] = list.get(0).stack(ax, rest);
            if (ax != 0) {
                int ndim = batched[j].rank();
                int[] perm = new int[ndim];
                perm[0] = ax < 0 ? ndim - 1 + ax : ax;
                int idx = 1;
                for (int d = 0; d < ndim; d++) {
                    if (d != perm[0]) perm[idx++] = d;
                }
                batched[j] = batched[j].permute(perm);
            }
        }

        BatchedDiffTensor[] bdts = new BatchedDiffTensor[batched.length];
        for (int j = 0; j < batched.length; j++) {
            bdts[j] = new BatchedDiffTensor(batched[j]);
        }

        IDiffTensor result = fn.apply(bdts);
        IDiffTensor flat = (result instanceof BatchedDiffTensor b) ? b.unwrap() : result;

        IDiffTensor[] ys = new IDiffTensor[n];
        for (int i = 0; i < n; i++) {
            ys[i] = flat.select(0, i);
        }
        return ys;
    }

    // ==================== IDiffTensor vmap with reduction ====================

    /**
     * JAX-style vmap with sum reduction.
     */
    public static IDiffTensor vmapSumT(Function<IDiffTensor, IDiffTensor> fn,
                                        List<? extends IDiffTensor> xs,
                                        int inAxes) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();

        IDiffTensor[] rest = new IDiffTensor[n - 1];
        for (int i = 1; i < n; i++) {
            rest[i - 1] = xs.get(i);
        }
        IDiffTensor batched = xs.get(0).stack(inAxes, rest);

        if (inAxes != 0) {
            int ndim = batched.rank();
            int[] perm = new int[ndim];
            perm[0] = inAxes < 0 ? ndim - 1 + inAxes : inAxes;
            int idx = 1;
            for (int d = 0; d < ndim; d++) {
                if (d != perm[0]) perm[idx++] = d;
            }
            batched = batched.permute(perm);
        }

        BatchedDiffTensor bdt = new BatchedDiffTensor(batched);
        IDiffTensor result = fn.apply(bdt);

        IDiffTensor flat = (result instanceof BatchedDiffTensor b) ? b.unwrap() : result;
        return flat.sum(0, false);
    }

    /** Shorthand for {@code vmapSumT(fn, xs, 0)}. */
    public static IDiffTensor vmapSumT(Function<IDiffTensor, IDiffTensor> fn, List<? extends IDiffTensor> xs) {
        return vmapSumT(fn, xs, 0);
    }

    /**
     * JAX-style vmap with mean reduction.
     */
    public static IDiffTensor vmapMeanT(Function<IDiffTensor, IDiffTensor> fn,
                                         List<? extends IDiffTensor> xs,
                                         int inAxes) {
        if (xs.isEmpty()) {
            throw new IllegalArgumentException(Messages.get("vmap.input_empty"));
        }
        int n = xs.size();

        IDiffTensor[] rest = new IDiffTensor[n - 1];
        for (int i = 1; i < n; i++) {
            rest[i - 1] = xs.get(i);
        }
        IDiffTensor batched = xs.get(0).stack(inAxes, rest);

        if (inAxes != 0) {
            int ndim = batched.rank();
            int[] perm = new int[ndim];
            perm[0] = inAxes < 0 ? ndim - 1 + inAxes : inAxes;
            int idx = 1;
            for (int d = 0; d < ndim; d++) {
                if (d != perm[0]) perm[idx++] = d;
            }
            batched = batched.permute(perm);
        }

        BatchedDiffTensor bdt = new BatchedDiffTensor(batched);
        IDiffTensor result = fn.apply(bdt);

        IDiffTensor flat = (result instanceof BatchedDiffTensor b) ? b.unwrap() : result;
        return flat.sum(0, false).div(n);
    }

    /** Shorthand for {@code vmapMeanT(fn, xs, 0)}. */
    public static IDiffTensor vmapMeanT(Function<IDiffTensor, IDiffTensor> fn, List<? extends IDiffTensor> xs) {
        return vmapMeanT(fn, xs, 0);
    }
}
