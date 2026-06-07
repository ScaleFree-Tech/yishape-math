package com.yishape.lab.math.autodiff;

import java.util.List;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;

/**
 * Batch-aware differentiable tensor for single-graph vmap execution over
 * the leading dimension. JAX-style: wraps a tensor of shape {@code [B, D1, D2, ...]}
 * and intercepts dimension-aware and structural operations so that the batch
 * dimension (index 0) is transparent to the vmapped function.
 *
 * <h3>Dimension-shift rule</h3>
 * The batch dimension occupies index 0. All dimension-indexed operations
 * ({@code sum(dim, ...)}, {@code select(dim, ...)}, {@code softmax(dim)},
 * {@code permute(dims)}, {@code transpose(dim0, dim1)}, {@code gather(dim, ...)},
 * etc.) have their {@code dim} parameter shifted by +1 so the user writes
 * them as if the batch dimension doesn't exist.
 *
 * <h3>Structural ops</h3>
 * View operations ({@code reshape}, {@code permute}, {@code transpose},
 * {@code flatten}, {@code contiguous}, {@code tile}, {@code broadcastTo},
 * {@code squeeze}, {@code unsqueeze}, {@code expand}) are intercepted,
 * dim-shifted where needed, and re-wrapped so batch context flows through.
 *
 * <h3>Matrix ops inside vmap</h3>
 * {@code mmul} on {@code [B, M, K] @ [B, K, N]} is executed as bmm.
 * {@code mmul} on {@code [B, M, K] @ [K, N]} broadcasts the weight
 * across the batch dimension. {@code bmm} delegates directly.
 *
 * <h3>Advanced ops</h3>
 * {@code gather}, {@code scatter}, {@code scatterAdd}, {@code indexSelect},
 * {@code where}, {@code pad}, {@code unfold}, {@code cat}, {@code stack},
 * {@code normalize}, {@code argsort}, {@code topk} are all supported with
 * proper dim-shifting.
 *
 * @author lteb2
 */
public final class BatchedDiffTensor implements IDiffTensor {

    private final IDiffTensor data;
    /** Number of leading batch dimensions. Default 1. Nesting increments this. */
    private final int nestingDepth;

    public BatchedDiffTensor(IDiffTensor data) {
        this(data, 1);
    }

    /** Package-private: create with explicit nesting depth (for nested vmap). */
    BatchedDiffTensor(IDiffTensor data, int nestingDepth) {
        // Flat model: unwrap any inner BatchedDiffTensor to avoid double-shifting
        this.data = (data instanceof BatchedDiffTensor bdt) ? bdt.data : data;
        this.nestingDepth = nestingDepth;
    }

    public IDiffTensor unwrap() {
        return data;
    }

    /** @return number of leading batch dimensions (≥1) */
    public int nestingDepth() {
        return nestingDepth;
    }

    private IDiffTensor wrap(IDiffTensor result) {
        if (result == data) return this;
        // Flat model: unwrap any BatchedDiffTensor to preserve single-layer delegation
        IDiffTensor raw = (result instanceof BatchedDiffTensor bdt) ? bdt.data : result;
        return new BatchedDiffTensor(raw, nestingDepth);
    }

    // ---- helpers ----

    private int shift(int dim) {
        return dim + nestingDepth;
    }

    /**
     * Shift all dimension indices in a permute array by nestingDepth and prepend
     * nestingDepth zeros for batch dimensions. E.g., depth=2: user permute(1,0)
     * on [B2,B1,M,N] → we execute permute(0,1,3,2).
     */
    private int[] shiftPermute(int... dims) {
        int[] shifted = new int[dims.length + nestingDepth];
        for (int i = 0; i < nestingDepth; i++) shifted[i] = i; // batch dims keep order
        for (int i = 0; i < dims.length; i++) {
            shifted[i + nestingDepth] = shift(dims[i]);
        }
        return shifted;
    }

    /** Detach if IDiffTensor, return as-is otherwise. */
    private static IDoubleTensor detachOther(IDoubleTensor t) {
        return (t instanceof IDiffTensor dt) ? dt.detach() : t;
    }

    /** Get batch size from leading dimension. */
    private int batchSize() {
        return data.shape()[0];
    }

    /** Get sizes of all nesting batch dimensions (for structural ops). */
    int[] batchSizes() {
        int[] sizes = new int[nestingDepth];
        for (int i = 0; i < nestingDepth; i++) sizes[i] = data.shape()[i];
        return sizes;
    }

    private static UnsupportedOperationException unsupported(String op) {
        return new UnsupportedOperationException(
            op + " is not supported on BatchedDiffTensor. "
            + "Only element-wise ops, reductions, matrix ops, and structural ops "
            + "are allowed inside vmapped functions.");
    }

    // ==================== batch-aware reductions ====================

    @Override
    public IDiffTensor sum() {
        int ndim = data.shape().length;
        IDiffTensor result = data;
        for (int d = ndim - 1; d >= nestingDepth; d--) {
            result = result.sum(d, false);
        }
        return result;
    }

    @Override
    public IDiffTensor sum(int dim, boolean keepdim) {
        return wrap(data.sum(shift(dim), keepdim));
    }

    @Override
    public IDiffTensor mean(int dim, boolean keepdim) {
        return wrap(data.mean(shift(dim), keepdim));
    }

    @Override
    public IDiffTensor max(int dim, boolean keepdim) {
        return wrap(data.max(shift(dim), keepdim));
    }

    @Override
    public IDiffTensor min(int dim, boolean keepdim) {
        return wrap(data.min(shift(dim), keepdim));
    }

    @Override
    public IDiffTensor prod(int dim, boolean keepdim) {
        return wrap(data.prod(shift(dim), keepdim));
    }

    @Override
    public IDiffTensor std(int dim, boolean keepdim) {
        return wrap(data.std(shift(dim), keepdim));
    }

    @Override
    public IDiffTensor var(int dim, boolean keepdim) {
        return wrap(data.var(shift(dim), keepdim));
    }

    // ==================== element-wise arithmetic ====================

    @Override public IDiffTensor add(IDoubleTensor other) { return wrap(data.add(other)); }
    @Override public IDiffTensor sub(IDoubleTensor other) { return wrap(data.sub(other)); }
    @Override public IDiffTensor mul(IDoubleTensor other) { return wrap(data.mul(other)); }
    @Override public IDiffTensor div(IDoubleTensor other) { return wrap(data.div(other)); }
    @Override public IDiffTensor add(double scalar) { return wrap(data.add(scalar)); }
    @Override public IDiffTensor sub(double scalar) { return wrap(data.sub(scalar)); }
    @Override public IDiffTensor mul(double scalar) { return wrap(data.mul(scalar)); }
    @Override public IDiffTensor div(double scalar) { return wrap(data.div(scalar)); }

    // ==================== element-wise unary ====================

    @Override public IDiffTensor neg() { return wrap(data.neg()); }
    @Override public IDiffTensor abs() { return wrap(data.abs()); }
    @Override public IDiffTensor sqrt() { return wrap(data.sqrt()); }
    @Override public IDiffTensor exp() { return wrap(data.exp()); }
    @Override public IDiffTensor log() { return wrap(data.log()); }
    @Override public IDiffTensor sigmoid() { return wrap(data.sigmoid()); }
    @Override public IDiffTensor relu() { return wrap(data.relu()); }
    @Override public IDiffTensor square() { return wrap(data.square()); }
    @Override public IDiffTensor pow(double n) { return wrap(data.pow(n)); }
    @Override public IDiffTensor clamp(double min, double max) { return wrap(data.clamp(min, max)); }
    @Override public IDiffTensor sin() { return wrap(data.sin()); }
    @Override public IDiffTensor cos() { return wrap(data.cos()); }
    @Override public IDiffTensor tan() { return wrap(data.tan()); }
    @Override public IDiffTensor tanh() { return wrap(data.tanh()); }
    @Override public IDiffTensor silu() { return wrap(data.silu()); }
    @Override public IDiffTensor gelu() { return wrap(data.gelu()); }
    @Override public IDiffTensor softplus(double beta) { return wrap(data.softplus(beta)); }
    @Override public IDiffTensor mish() { return wrap(data.mish()); }
    @Override public IDiffTensor elu(double alpha) { return wrap(data.elu(alpha)); }
    @Override public IDiffTensor leakyRelu(double alpha) { return wrap(data.leakyRelu(alpha)); }
    @Override public IDiffTensor selu() { return wrap(data.selu()); }
    @Override public IDiffTensor hardtanh(double minVal, double maxVal) { return wrap(data.hardtanh(minVal, maxVal)); }
    @Override public IDiffTensor dropout(double p) { return wrap(data.dropout(p)); }

    // ==================== reverse ops / norm ====================

    @Override public IDiffTensor rsub(double scalar) { return wrap(data.rsub(scalar)); }
    @Override public IDiffTensor rdiv(double scalar) { return wrap(data.rdiv(scalar)); }
    @Override public IDiffTensor reciprocal() { return wrap(data.reciprocal()); }
    @Override public IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
            int stride, int padding, int dilation) {
        return wrap(data.conv2d(weight, bias, stride, padding, dilation));
    }
    @Override public IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
            IDiffTensor mask, double dropout) {
        return wrap(data.scaledDotProductAttention(key, vTensor, mask, dropout));
    }
    @Override public IDiffTensor layerNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        return wrap(data.layerNorm(gamma, beta, eps));
    }
    @Override public IDiffTensor batchNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        return wrap(data.batchNorm(gamma, beta, eps));
    }

    // ==================== structural with dim (shift) ====================

    @Override
    public IDiffTensor select(int dim, long index) {
        return wrap(data.select(shift(dim), index));
    }

    @Override
    public IDiffTensor slice(int dim, long start, long end) {
        return wrap(data.slice(shift(dim), start, end));
    }

    @Override
    public IDiffTensor narrow(int dim, long start, long length) {
        return wrap(data.narrow(shift(dim), start, length));
    }

    @Override
    public IDiffTensor argmax(int dim) {
        return wrap(data.argmax(shift(dim)));
    }

    @Override
    public IDiffTensor argmin(int dim) {
        return wrap(data.argmin(shift(dim)));
    }

    @Override
    public IDiffTensor softmax(int dim) {
        return wrap(data.softmax(shift(dim)));
    }

    @Override
    public IDiffTensor logSoftmax(int dim) {
        return wrap(data.logSoftmax(shift(dim)));
    }

    @Override
    public IDiffTensor softmaxCrossEntropy(IDoubleTensor labels, int dim) {
        if (labels instanceof BatchedDiffTensor bl) {
            return wrap(data.softmaxCrossEntropy(bl.data, shift(dim)));
        }
        return wrap(data.softmaxCrossEntropy(labels, shift(dim)));
    }

    @Override
    public IDiffTensor cumsum(int dim) {
        return wrap(data.cumsum(shift(dim)));
    }

    @Override
    public IDiffTensor cumprod(int dim) {
        return wrap(data.cumprod(shift(dim)));
    }

    // ==================== structural: delegate WITH wrapping (batch context preserved) ====================

    @Override
    public IDiffTensor reshape(int... newShape) {
        // User provides shape without batch dims; prepend all nesting batch dims
        int[] batchSizes = batchSizes();
        int[] fullShape = new int[newShape.length + nestingDepth];
        System.arraycopy(batchSizes, 0, fullShape, 0, nestingDepth);
        System.arraycopy(newShape, 0, fullShape, nestingDepth, newShape.length);
        return wrap(data.reshape(fullShape));
    }

    @Override
    public IDiffTensor permute(int... dims) {
        // Shift all user dims by +1, keep batch at 0
        return wrap(data.permute(shiftPermute(dims)));
    }

    @Override
    public IDiffTensor transpose(int dim0, int dim1) {
        return wrap(data.transpose(shift(dim0), shift(dim1)));
    }

    @Override
    public IDiffTensor transpose() {
        // Transpose the last two non-batch dims: data has [B, ..., D_{n-1}, D_n]
        int ndim = data.rank();
        return wrap(data.transpose(ndim - 2, ndim - 1));
    }

    @Override
    public IDiffTensor squeeze(int... dims) {
        if (dims.length == 0) return wrap(data.squeeze());
        int[] shifted = new int[dims.length];
        for (int i = 0; i < dims.length; i++) shifted[i] = shift(dims[i]);
        return wrap(data.squeeze(shifted));
    }

    @Override
    public IDiffTensor unsqueeze(int dim) {
        return wrap(data.unsqueeze(shift(dim)));
    }

    @Override
    public IDiffTensor flatten(int startDim, int endDim) {
        return wrap(data.flatten(shift(startDim), shift(endDim)));
    }

    @Override
    public IDiffTensor expand(int... shape) {
        int[] batchSizes = batchSizes();
        int[] fullShape = new int[shape.length + nestingDepth];
        System.arraycopy(batchSizes, 0, fullShape, 0, nestingDepth);
        System.arraycopy(shape, 0, fullShape, nestingDepth, shape.length);
        return wrap(data.expand(fullShape));
    }

    @Override
    public IDiffTensor contiguous() {
        return wrap(data.contiguous());
    }

    @Override
    public IDiffTensor tile(int... repeats) {
        int[] fullRepeats = new int[repeats.length + nestingDepth];
        for (int i = 0; i < nestingDepth; i++) fullRepeats[i] = 1; // don't tile batch dims
        System.arraycopy(repeats, 0, fullRepeats, nestingDepth, repeats.length);
        return wrap(data.tile(fullRepeats));
    }

    @Override
    public IDiffTensor broadcastTo(int... shape) {
        int[] batchSizes = batchSizes();
        int[] fullShape = new int[shape.length + nestingDepth];
        System.arraycopy(batchSizes, 0, fullShape, 0, nestingDepth);
        System.arraycopy(shape, 0, fullShape, nestingDepth, shape.length);
        return wrap(data.broadcastTo(fullShape));
    }

    @Override
    public IDiffTensor clone() {
        return wrap(data.clone());
    }

    // ==================== in-place ops (delegate, maintain wrapper) ====================

    @Override public IDiffTensor add_(IDoubleTensor other) { data.add_(other); return this; }
    @Override public IDiffTensor sub_(IDoubleTensor other) { data.sub_(other); return this; }
    @Override public IDiffTensor mul_(IDoubleTensor other) { data.mul_(other); return this; }
    @Override public IDiffTensor div_(IDoubleTensor other) { data.div_(other); return this; }
    @Override public IDiffTensor fill_(double value) { data.fill_(value); return this; }
    @Override public IDiffTensor copy_(IDoubleTensor src) { data.copy_(src); return this; }

    // ==================== matrix ops ====================

    @Override
    public IDiffTensor mmul(IDoubleTensor other) {
        // Case 1: both are BatchedDiffTensor → bmm
        // Case 2: this BatchedDiffTensor [B, M, K], other plain [K, N] → per-sample mmul via batch-expand
        // Case 3: this plain (shouldn't happen inside vmap) → delegate
        IDiffTensor unwrappedThis = this.data;
        IDoubleTensor otherData;

        if (other instanceof BatchedDiffTensor bo) {
            // Both batched: do bmm on the underlying tensors [B, M, K] @ [B, K, N]
            IDiffTensor result = unwrappedThis.bmm(bo.data);
            return wrap(result);
        }

        // This is batched [B, M, K], other is plain [K, N] (or [N] for vector)
        // Expand other to [B, K, N] and do bmm
        otherData = detachOther(other);
        int B = batchSize();
        int thisRank = unwrappedThis.rank();
        int otherRank = otherData.rank();

        if (thisRank == 3 && otherRank == 2) {
            // [B, M, K] @ [K, N] → expand other to [B, K, N], do bmm → [B, M, N]
            int M = unwrappedThis.dim(1), K = unwrappedThis.dim(2), N = otherData.dim(1);
            IDiffTensor expandedOther;
            if (otherData instanceof IDiffTensor od) {
                // Expand to [B, K, N] by unsqueezing and broadcasting
                expandedOther = od.unsqueeze(0).expand(B, K, N);
            } else {
                expandedOther = IDiffTensor.constantTensor(otherData.toDoubleArray(), otherData.shape())
                    .unsqueeze(0).expand(B, K, N);
            }
            IDiffTensor result = unwrappedThis.bmm(expandedOther);
            return wrap(result);
        }

        if (thisRank == 2 && otherRank == 1) {
            // [B, K] @ [K] = [B] — per-sample dot product
            // Unsqueeze to [B, 1, K] @ [1, K, 1] → [B, 1, 1] → squeeze
            IDiffTensor this3d = unwrappedThis.unsqueeze(1); // [B, 1, K]
            IDiffTensor other3d;
            if (otherData instanceof IDiffTensor od) {
                other3d = od.reshape(1, otherData.dim(0), 1).expand(B, otherData.dim(0), 1); // [B, K, 1]
            } else {
                other3d = IDiffTensor.constantTensor(otherData.toDoubleArray(), otherData.shape())
                    .reshape(1, otherData.dim(0), 1).expand(B, otherData.dim(0), 1);
            }
            IDiffTensor result3d = this3d.bmm(other3d); // [B, 1, 1]
            return wrap(result3d.squeeze(1, 2)); // [B]
        }

        if (thisRank == 2 && otherRank == 2) {
            // [B, K] @ [M, K]^T style — less common, unsqueeze to 3D
            int K1 = unwrappedThis.dim(1), M = otherData.dim(0), K2 = otherData.dim(1);
            if (K1 != K2) throw new IllegalArgumentException(
                "mmul inside vmap: shape mismatch [" + B + "," + K1 + "] @ [" + M + "," + K2 + "]");
            // Treat as [B, 1, K] @ [B, K, M] → [B, 1, M] → squeeze
            IDiffTensor this3d = unwrappedThis.unsqueeze(1); // [B, 1, K]
            IDiffTensor otherT;
            if (otherData instanceof IDiffTensor od) {
                otherT = od.transpose(0, 1).unsqueeze(0).expand(B, K2, M);
            } else {
                otherT = IDiffTensor.constantTensor(otherData.toDoubleArray(), otherData.shape())
                    .transpose(0, 1).unsqueeze(0).expand(B, K2, M);
            }
            IDiffTensor result = this3d.bmm(otherT); // [B, 1, M]
            return wrap(result.squeeze(1)); // [B, M]
        }

        // Fallback: delegate directly (may fail if shapes don't align)
        return wrap(unwrappedThis.mmul(otherData));
    }

    @Override
    public IDiffTensor bmm(IDoubleTensor other) {
        IDiffTensor otherData;
        if (other instanceof BatchedDiffTensor bo) {
            otherData = bo.data;
        } else {
            otherData = (other instanceof IDiffTensor dt) ? dt : null;
            if (otherData == null) {
                otherData = IDiffTensor.constantTensor(other.toDoubleArray(), other.shape());
            }
        }
        return wrap(data.bmm(otherData));
    }

    @Override
    public IDiffTensor einsum(String subscript, IDoubleTensor... others) {
        // For simple cases inside vmap, delegate to data.einsum with proper
        // batch handling. The underlying tensor already has the batch dim.
        // Complex subscripts that need dim shifting are rare in vmap contexts.
        IDiffTensor[] unwrapped = new IDiffTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            if (others[i] instanceof BatchedDiffTensor bo) {
                unwrapped[i] = bo.data;
            } else if (others[i] instanceof IDiffTensor dt) {
                unwrapped[i] = dt;
            } else {
                unwrapped[i] = IDiffTensor.constantTensor(
                    others[i].toDoubleArray(), others[i].shape());
            }
        }
        return wrap(data.einsum(subscript, unwrapped));
    }

    // ==================== advanced ops ====================

    @Override
    public IDiffTensor gather(int dim, IDoubleTensor index) {
        IDoubleTensor idx = (index instanceof BatchedDiffTensor bi) ? bi.data
            : detachOther(index);
        return wrap(data.gather(shift(dim), idx));
    }

    @Override
    public IDiffTensor indexSelect(int dim, IDoubleTensor index) {
        IDoubleTensor idx = (index instanceof BatchedDiffTensor bi) ? bi.data
            : detachOther(index);
        return wrap(data.indexSelect(shift(dim), idx));
    }

    @Override
    public IDiffTensor argsort(int dim, boolean descending) {
        return wrap(data.argsort(shift(dim), descending));
    }

    @Override
    public IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) {
        IDoubleTensor idx = (index instanceof BatchedDiffTensor bi) ? bi.data
            : detachOther(index);
        IDoubleTensor src = (source instanceof BatchedDiffTensor bs) ? bs.data
            : detachOther(source);
        return wrap(data.scatter(shift(dim), idx, src));
    }

    @Override
    public IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) {
        IDoubleTensor idx = (index instanceof BatchedDiffTensor bi) ? bi.data
            : detachOther(index);
        IDoubleTensor src = (source instanceof BatchedDiffTensor bs) ? bs.data
            : detachOther(source);
        return wrap(data.scatterAdd(shift(dim), idx, src));
    }

    @Override
    public IDiffTensor where(IDoubleTensor condition, IDoubleTensor other) {
        IDoubleTensor cond = (condition instanceof BatchedDiffTensor bc) ? bc.data
            : detachOther(condition);
        IDoubleTensor oth = (other instanceof BatchedDiffTensor bo) ? bo.data
            : detachOther(other);
        return wrap(data.where(cond, oth));
    }

    @Override
    public IDiffTensor topk(int k, int dim, boolean largest) {
        return wrap(data.topk(k, shift(dim), largest));
    }

    @Override
    public IDiffTensor pad(int[][] padding, String mode, double value) {
        // Pad spec is [dim0_l, dim0_r], [dim1_l, dim1_r], ...
        // Add [0,0] for each batch dimension
        int[][] fullPad = new int[padding.length + nestingDepth][];
        for (int i = 0; i < nestingDepth; i++) fullPad[i] = new int[]{0, 0};
        System.arraycopy(padding, 0, fullPad, nestingDepth, padding.length);
        return wrap(data.pad(fullPad, mode, value));
    }

    @Override
    public IDiffTensor tril(int diagonal) {
        return wrap(data.tril(diagonal));
    }

    @Override
    public IDiffTensor unfold(int dim, int size, int stride, int dilation) {
        return wrap(data.unfold(shift(dim), size, stride, dilation));
    }

    @Override
    public IDiffTensor nonzero() {
        // nonzero returns variable-length results per sample — delegate but warn
        return wrap(data.nonzero());
    }

    @Override
    public IDiffTensor maskedSelect(IDoubleTensor mask) {
        IDoubleTensor m = (mask instanceof BatchedDiffTensor bm) ? bm.data
            : detachOther(mask);
        return wrap(data.maskedSelect(m));
    }

    @Override
    public IDiffTensor maskedFill(IDoubleTensor mask, double value) {
        IDoubleTensor m = (mask instanceof BatchedDiffTensor bm) ? bm.data
            : detachOther(mask);
        return wrap(data.maskedFill(m, value));
    }

    @Override
    public IDiffTensor cat(int dim, IDoubleTensor... others) {
        IDiffTensor[] unwrapped = new IDiffTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            if (others[i] instanceof BatchedDiffTensor bo) {
                unwrapped[i] = bo.data;
            } else if (others[i] instanceof IDiffTensor dt) {
                unwrapped[i] = dt;
            } else {
                unwrapped[i] = IDiffTensor.constantTensor(
                    others[i].toDoubleArray(), others[i].shape());
            }
        }
        return wrap(data.cat(shift(dim), unwrapped));
    }

    @Override
    public IDiffTensor stack(int dim, IDoubleTensor... others) {
        IDiffTensor[] unwrapped = new IDiffTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            if (others[i] instanceof BatchedDiffTensor bo) {
                unwrapped[i] = bo.data;
            } else if (others[i] instanceof IDiffTensor dt) {
                unwrapped[i] = dt;
            } else {
                unwrapped[i] = IDiffTensor.constantTensor(
                    others[i].toDoubleArray(), others[i].shape());
            }
        }
        return wrap(data.stack(shift(dim), unwrapped));
    }

    @Override
    public IDiffTensor normalize(double p, int dim) {
        return wrap(data.normalize(p, shift(dim)));
    }

    // ==================== accessors ====================

    @Override public int[] shape() { return data.shape(); }
    @Override public int rank() { return data.rank(); }
    @Override public int dim(int axis) { return data.dim(axis); }
    @Override public long totalSize() { return data.totalSize(); }
    @Override public int[] strides() { return data.strides(); }
    @Override public int stride(int axis) { return data.stride(axis); }
    @Override public int offset() { return data.offset(); }
    @Override public boolean isContiguous() { return data.isContiguous(); }
    @Override public double item() { return data.item(); }
    @Override public double get(int... indices) { return data.get(indices); }
    @Override public IDiffTensor set(double value, int... indices) { data.set(value, indices); return this; }
    @Override public IDiffTensor fill(double value) { data.fill(value); return this; }
    @Override public IDiffTensor copy() { return new BatchedDiffTensor(data.clone(), nestingDepth); }
    @Override public double[] toDoubleArray() { return data.toDoubleArray(); }
    @Override public IMatrix toMatrix() { return data.toMatrix(); }
    @Override public IDoubleVector toVector() { return data.toVector(); }
    @Override public IDoubleVector toVectorCopy() { return data.toVectorCopy(); }
    @Override public List<IDoubleTensor> unstack(int dim) { return data.unstack(shift(dim)); }
    @Override public IDiffVector flattenValue() { return data.flattenValue(); }
    @Override public IDiffVector flattenGrad() { return data.flattenGrad(); }
    @Override public IDoubleTensor detach() { return data.detach(); }
    @Override public boolean requiresGrad() { return data.requiresGrad(); }
    @Override public IDiffTensor setRequiresGrad(boolean requiresGrad) { return wrap(data.setRequiresGrad(requiresGrad)); }
    @Override public IDoubleTensor grad() { return data.grad(); }

    // ==================== scalar reductions ====================

    @Override public double sumAll() { return data.sumAll(); }
    @Override public double meanAll() { return data.meanAll(); }
    @Override public double maxAll() { return data.maxAll(); }
    @Override public double minAll() { return data.minAll(); }
    @Override public double prodAll() { return data.prodAll(); }

    @Override public void backward() { data.backward(); }
    @Override public void backward(IDoubleTensor gradient) { data.backward(gradient); }
    @Override public void zeroGradient() { data.zeroGradient(); }
    @Override public void clipGradNorm(double maxNorm) { data.clipGradNorm(maxNorm); }
    @Override public void clipGradValue(double maxValue) { data.clipGradValue(maxValue); }
}
