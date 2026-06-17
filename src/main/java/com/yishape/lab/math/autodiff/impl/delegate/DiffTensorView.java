package com.yishape.lab.math.autodiff.impl.delegate;

import com.yishape.lab.math.autodiff.impl.DiffTensorUtil;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.impl.AutodiffBufferPool;
import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;

/**
 * View operations extracted from {@link RereDiffTensor}.
 * All methods are static, taking the tensor as first parameter.
 */
public final class DiffTensorView {
    private DiffTensorView() { /* utility class */ }

// ==================== View ops ====================

public static IDiffTensor select(RereDiffTensor tensor, int dim, long index) {
    RereDoubleTensor view = (RereDoubleTensor) tensor.value.select(dim, index);
    if (!tensor.requiresGrad) return tensor.toNonDiff(view);
    int[] parentShape = tensor.shape();
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int[] viewShape = view.shape();
    int viewTotal = (int) view.totalSize();
    // Precompute parent flat index for each view position
    int[] parentIdx = new int[viewTotal];
    for (int i = 0; i < viewTotal; i++) {
        int[] vIdx = DiffTensorUtil.unlinearizeInt(i, viewShape);
        int[] pIdx = new int[parentShape.length];
        int vi = 0;
        for (int j = 0; j < parentShape.length; j++) {
            pIdx[j] = (j == d) ? (int) index : vIdx[vi++];
        }
        parentIdx[i] = DiffTensorUtil.flatIndex(pIdx, parentShape);
    }
    // Store parentIdx for GPU/HPC backend graph serialization
    int[] bi = parentIdx;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor parent = self.inputs.get(0);
        int pt = (int) parent.value.totalSize();
        double[] pGrad = AutodiffBufferPool.acquire(pt);
        int[] idx = self.backwardIndices();
        if (idx != null) {
            for (int i = 0; i < viewTotal; i++) pGrad[idx[i]] += self.grad[i];
        } else {
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
        }
        parent.accGradFromPooled(pGrad, pt);
    };
    RereDiffTensor result = new RereDiffTensor(view, List.of(tensor), bw, "select");
    result.backwardIndices = bi;
    return result;
}

public static IDiffTensor slice(RereDiffTensor tensor, int dim, long start, long end) {
    RereDoubleTensor view = (RereDoubleTensor) tensor.value.slice(dim, start, end);
    if (!tensor.requiresGrad) return tensor.toNonDiff(view);
    int[] parentShape = tensor.shape();
    int d = (dim < 0 ? dim + tensor.rank() : dim);
    int[] viewShape = view.shape();
    int viewTotal = (int) view.totalSize();
    int[] parentIdx = new int[viewTotal];
    for (int i = 0; i < viewTotal; i++) {
        int[] vIdx = DiffTensorUtil.unlinearizeInt(i, viewShape);
        int[] pIdx = vIdx.clone();
        pIdx[d] += (int) start;
        parentIdx[i] = DiffTensorUtil.flatIndex(pIdx, parentShape);
    }
    // Store parentIdx for GPU/HPC backend graph serialization (same pattern as select())
    int[] bi = parentIdx;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor parent = self.inputs.get(0);
        int pt = (int) parent.value.totalSize();
        double[] pGrad = AutodiffBufferPool.acquire(pt);
        int[] idx = self.backwardIndices();
        if (idx != null) {
            for (int i = 0; i < viewTotal; i++) pGrad[idx[i]] += self.grad[i];
        } else {
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
        }
        parent.accGradFromPooled(pGrad, pt);
    };
    RereDiffTensor result = new RereDiffTensor(view, List.of(tensor), bw, "slice");
    result.setBackwardIndices(bi);
    return result;
}

public static IDiffTensor narrow(RereDiffTensor tensor, int dim, long start, long length) {
    // narrow is same as slice(dim, start, start+length)
    return slice(tensor, dim, start, start + length);
}

public static IDiffTensor permute(RereDiffTensor tensor, int... dims) {
    RereDoubleTensor view = (RereDoubleTensor) tensor.value.permute(dims);
    if (!tensor.requiresGrad) return tensor.toNonDiff(view);
    int[] parentShape = tensor.shape();
    int[] viewShape = view.shape();
    int viewTotal = (int) view.totalSize();
    // Compute inverse permutation
    int[] invDims = new int[dims.length];
    for (int i = 0; i < dims.length; i++) invDims[dims[i]] = i;
    int[] parentIdx = new int[viewTotal];
    for (int i = 0; i < viewTotal; i++) {
        int[] vIdx = DiffTensorUtil.unlinearizeInt(i, viewShape);
        int[] pIdx = new int[parentShape.length];
        for (int j = 0; j < dims.length; j++) pIdx[dims[j]] = vIdx[j];
        parentIdx[i] = DiffTensorUtil.flatIndex(pIdx, parentShape);
    }
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor parent = self.inputs.get(0);
        int pt = (int) parent.value.totalSize();
        double[] pGrad = AutodiffBufferPool.acquire(pt);
        for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
        parent.accGradFromPooled(pGrad, pt);
    };
    RereDiffTensor result = new RereDiffTensor(view, List.of(tensor), bw, "permute");
    result.backwardIndices = dims;  // axis permutation for HPC/GPU backend
    // Symbolic backward: inverse permutation (tape-of-tape)
    final int[] invDimsSym = invDims.clone();
    result.symbolicBackwardFn = g -> new IDiffTensor[]{ g.permute(invDimsSym) };
    return result;
}

public static IDiffTensor transpose(RereDiffTensor tensor, int dim0, int dim1) {
    int r = tensor.rank();
    int[] perm = new int[r];
    for (int i = 0; i < r; i++) perm[i] = i;
    perm[dim0] = dim1;
    perm[dim1] = dim0;
    return permute(tensor, perm);
}

public static IDiffTensor transpose(RereDiffTensor tensor) {
    if (tensor.rank() != 2) throw new IllegalStateException("transpose() requires rank 2, got " + tensor.rank());
    return transpose(tensor, 0, 1);
}

public static IDiffTensor squeeze(RereDiffTensor tensor, int... dims) {
    RereDoubleTensor view = (RereDoubleTensor) tensor.value.squeeze(dims);
    if (!tensor.requiresGrad) return tensor.toNonDiff(view);
    int[] parentShape = tensor.shape();
    int[] viewShape = view.shape();
    int viewTotal = (int) view.totalSize();
    // Build mapping: which source dim each view dim corresponds to
    boolean[] squeezed = new boolean[parentShape.length];
    if (dims.length == 0) {
        for (int j = 0; j < parentShape.length; j++) squeezed[j] = (parentShape[j] == 1);
    } else {
        for (int d : dims) squeezed[(d < 0 ? d + parentShape.length : d)] = true;
    }
    int[] srcDim = new int[viewShape.length];
    int si = 0;
    for (int j = 0; j < parentShape.length; j++) if (!squeezed[j]) srcDim[si++] = j;
    int[] parentIdx = new int[viewTotal];
    for (int i = 0; i < viewTotal; i++) {
        int[] vIdx = DiffTensorUtil.unlinearizeInt(i, viewShape);
        int[] pIdx = new int[parentShape.length];
        for (int j = 0; j < parentShape.length; j++) pIdx[j] = 0; // squeezed dims stay 0
        for (int j = 0; j < viewShape.length; j++) pIdx[srcDim[j]] = vIdx[j];
        parentIdx[i] = DiffTensorUtil.flatIndex(pIdx, parentShape);
    }
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor parent = self.inputs.get(0);
        int pt = (int) parent.value.totalSize();
        double[] pGrad = AutodiffBufferPool.acquire(pt);
        for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
        parent.accGradFromPooled(pGrad, pt);
    };
    return new RereDiffTensor(view, List.of(tensor), bw, "squeeze");
}

public static IDiffTensor unsqueeze(RereDiffTensor tensor, int dim) {
    RereDoubleTensor view = (RereDoubleTensor) tensor.value.unsqueeze(dim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(view);
    int d = (dim < 0 ? dim + tensor.rank() + 1 : dim);
    int[] parentShape = tensor.shape();
    int[] viewShape = view.shape();
    int viewTotal = (int) view.totalSize();
    int[] parentIdx = new int[viewTotal];
    for (int i = 0; i < viewTotal; i++) {
        int[] vIdx = DiffTensorUtil.unlinearizeInt(i, viewShape);
        int[] pIdx = new int[parentShape.length];
        int pi = 0;
        for (int j = 0; j < viewShape.length; j++) {
            if (j != d) pIdx[pi++] = vIdx[j];
        }
        parentIdx[i] = DiffTensorUtil.flatIndex(pIdx, parentShape);
    }
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor parent = self.inputs.get(0);
        int pt = (int) parent.value.totalSize();
        double[] pGrad = AutodiffBufferPool.acquire(pt);
        for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
        parent.accGradFromPooled(pGrad, pt);
    };
    RereDiffTensor unsqzResult = new RereDiffTensor(view, List.of(tensor), bw, "unsqueeze");
    // Symbolic backward: squeeze the unsqueezed dim (tape-of-tape)
    final int unsqzDim = d;
    unsqzResult.symbolicBackwardFn = g -> new IDiffTensor[]{ g.squeeze(unsqzDim) };
    return unsqzResult;
}

public static IDiffTensor flatten(RereDiffTensor tensor, int startDim, int endDim) {
    RereDoubleTensor view = (RereDoubleTensor) tensor.value.flatten(startDim, endDim);
    if (!tensor.requiresGrad) return tensor.toNonDiff(view);
    int[] parentShape = tensor.shape();
    int[] viewShape = view.shape();
    int viewTotal = (int) view.totalSize();
    // flatten merges dims [start, end] into one. parentShape -> viewShape mapping:
    int pre = startDim, post = parentShape.length - endDim - 1;
    int midParent = 1;
    for (int j = startDim; j <= endDim; j++) midParent *= parentShape[j];
    int[] parentIdx = new int[viewTotal];
    for (int i = 0; i < viewTotal; i++) {
        int[] vIdx = DiffTensorUtil.unlinearizeInt(i, viewShape);
        int[] pIdx = new int[parentShape.length];
        // Copy pre dims
        for (int j = 0; j < startDim; j++) pIdx[j] = vIdx[j];
        // Unflatten mid dim
        int midVal = vIdx[startDim];
        for (int j = endDim; j >= startDim; j--) {
            pIdx[j] = midVal % parentShape[j];
            midVal /= parentShape[j];
        }
        // Copy post dims
        for (int j = 0; j < post; j++) pIdx[endDim + 1 + j] = vIdx[startDim + 1 + j];
        parentIdx[i] = DiffTensorUtil.flatIndex(pIdx, parentShape);
    }
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor parent = self.inputs.get(0);
        int pt = (int) parent.value.totalSize();
        double[] pGrad = AutodiffBufferPool.acquire(pt);
        for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
        parent.accGradFromPooled(pGrad, pt);
    };
    RereDiffTensor flatResult = new RereDiffTensor(view, List.of(tensor), bw, "flatten");
    // Symbolic backward: reshape back to parent shape (tape-of-tape)
    final int[] flatParentShape = tensor.shape().clone();
    flatResult.symbolicBackwardFn = g -> new IDiffTensor[]{ g.reshape(flatParentShape) };
    return flatResult;
}

public static IDiffTensor expand(RereDiffTensor tensor, int... targetShape) {
    // expand repeats data along dimensions where parent shape is 1
    RereDoubleTensor view = (RereDoubleTensor) tensor.value.expand(targetShape);
    if (!tensor.requiresGrad) return tensor.toNonDiff(view);
    int[] parentShape = tensor.shape();
    int[] viewShape = view.shape();
    int viewTotal = (int) view.totalSize();
    int[] parentIdx = new int[viewTotal];
    for (int i = 0; i < viewTotal; i++) {
        int[] vIdx = DiffTensorUtil.unlinearizeInt(i, viewShape);
        int[] pIdx = new int[parentShape.length];
        for (int j = 0; j < parentShape.length; j++) {
            pIdx[j] = (parentShape[j] == 1) ? 0 : vIdx[j];
        }
        parentIdx[i] = DiffTensorUtil.flatIndex(pIdx, parentShape);
    }
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor parent = self.inputs.get(0);
        int pt = (int) parent.value.totalSize();
        double[] pGrad = AutodiffBufferPool.acquire(pt);
        for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
        parent.accGradFromPooled(pGrad, pt);
    };
    RereDiffTensor result = new RereDiffTensor(view, List.of(tensor), bw, "expand");
    result.backwardIndices = parentShape;  // source shape for HPC/GPU backward (reduce-sum over broadcast dims)
    return result;
}

public static IDiffTensor broadcastTo(RereDiffTensor tensor, int... targetShape) {
    return expand(tensor, targetShape);
}

/**
 * Returns a contiguous copy of this tensor, or {@code this} if it is already
 * both logically contiguous AND has zero offset.
 *
 * <p><b>PITFALL — offset ≠ 0 causes infinite recursion:</b>
 * {@link #reshape(int...)} calls {@code tensor.contiguous().reshape(newShape)}.
 * If {@code contiguous()} returns {@code this} (because strides are C-order)
 * but {@code tensor.offset() != 0}, the {@code reshape()} guard {@code tensor.isContiguous() && tensor.offset() == 0}
 * fails, and it calls {@code tensor.contiguous().reshape()} again → StackOverflowError.
 *
 * <p>This happens with tensor views like {@code select(dim, idx)} on a multi-row
 * tensor: row 1 of [2, N] has shape [N], contiguous strides [1], but offset = N.
 * Always check {@code tensor.offset() == 0} alongside {@code tensor.isContiguous()}.</p>
 */
public static IDiffTensor contiguous(RereDiffTensor tensor) {
    if (tensor.value.isContiguous() && tensor.offset() == 0) return tensor;
    if (!tensor.requiresGrad) return tensor.toNonDiff(new RereDoubleTensor(tensor.value.toDoubleArray(), tensor.shape()));
    int[] s = tensor.shape();
    int n = (int) tensor.value.totalSize();
    double[] contigData = new double[n];
    for (int i = 0; i < n; i++) {
        contigData[i] = tensor.value.linearGet(i);
    }
    // For contiguous(), the view IS the same logical elements, just reordered in storage.
    // Since grad is indexed by logical position, the backward is just identity.
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor input = self.inputs.get(0);
        input.accGrad(self.grad.clone());
    };
    return new RereDiffTensor(new RereDoubleTensor(contigData, s), List.of(tensor), bw, "contiguous");
}

public static IDiffTensor reshape(RereDiffTensor tensor, int... newShape) {
    if (tensor.isContiguous() && tensor.offset() == 0) {
        RereDoubleTensor view = (RereDoubleTensor) tensor.value.reshape(newShape);
        if (!tensor.requiresGrad) return tensor.toNonDiff(view);
        // Reshape doesn't change data order for contiguous tensors
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            input.accGrad(self.grad.clone());
        };
        RereDiffTensor reshapeResult = new RereDiffTensor(view, List.of(tensor), bw, "reshape");
        // Symbolic backward: reshape back to parent shape (tape-of-tape)
        final int[] parentShapeSym = tensor.shape().clone();
        reshapeResult.symbolicBackwardFn = g -> new IDiffTensor[]{ g.reshape(parentShapeSym) };
        return reshapeResult;
    }
    return tensor.contiguous().reshape(newShape);
}

public static IDiffTensor tile(RereDiffTensor tensor, int... repeats) {
    int r = tensor.rank();
    int[] rep = new int[r];
    for (int i = 0; i < r; i++) rep[i] = (i < repeats.length) ? repeats[i] : 1;
    int[] inShape = tensor.shape();
    int[] outShape = new int[r];
    for (int i = 0; i < r; i++) outShape[i] = inShape[i] * rep[i];
    int outTotal = 1;
    for (int d : outShape) outTotal *= d;

    if (!tensor.requiresGrad) return tensor.toNonDiff(tensor.value.tile(repeats));

    double[] inData = tensor.value.toDoubleArray();
    double[] outData = new double[outTotal];
    // Precompute parent index for sum-reduce backward
    int[] parentIdx = new int[outTotal];
    for (int flat = 0; flat < outTotal; flat++) {
        int remaining = flat;
        int inFlat = 0;
        int inStride = 1;
        for (int j = r - 1; j >= 0; j--) {
            int coord = remaining % outShape[j];
            remaining /= outShape[j];
            inFlat += (coord % inShape[j]) * inStride;
            inStride *= inShape[j];
        }
        outData[flat] = inData[inFlat];
        parentIdx[flat] = inFlat;
    }
    int fOutTotal = outTotal;
    Consumer<RereDiffTensor> bw = self -> {
        RereDiffTensor parent = self.inputs.get(0);
        int pt = (int) parent.value.totalSize();
        double[] pGrad = AutodiffBufferPool.acquire(pt);
        for (int i = 0; i < fOutTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
        parent.accGradFromPooled(pGrad, pt);
    };
    return new RereDiffTensor(outData, outShape, List.of(tensor), bw, "tile");
}


}
