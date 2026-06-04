package com.yishape.lab.math.autodiff;

import java.util.List;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;

/**
 * Batch-aware differentiable tensor for single-graph vmap execution over
 * the leading dimension.
 *
 * <p>Wraps a tensor of shape {@code [B, D1, D2, ...]} and intercepts
 * {@link #sum()} / {@link #sum(int, boolean)} / {@link #mean(int, boolean)}
 * so that reductions inside a vmapped function operate per-sample.
 *
 * <p>Dimension-shift rule: the batch dimension occupies index 0. All
 * dimension-indexed operations ({@code sum(dim, ...)}, {@code select(dim, ...)})
 * have their {@code dim} parameter shifted by +1.
 *
 * <p>Element-wise operations are delegated and re-wrapped so batch context
 * flows through the computation chain.
 *
 * @author lteb2
 */
final class BatchedDiffTensor implements IDiffTensor {

    private final IDiffTensor data;

    BatchedDiffTensor(IDiffTensor data) {
        this.data = data;
    }

    IDiffTensor unwrap() {
        return data;
    }

    private IDiffTensor wrap(IDiffTensor result) {
        if (result == data) return this;
        if (result instanceof BatchedDiffTensor bdt) return bdt;
        return new BatchedDiffTensor(result);
    }

    // ---- batch-aware reductions ----

    @Override
    public IDiffTensor sum() {
        int ndim = data.shape().length;
        IDiffTensor result = data;
        for (int d = ndim - 1; d >= 1; d--) {
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

    // ---- element-wise arithmetic ----

    @Override public IDiffTensor add(IDoubleTensor other) { return wrap(data.add(other)); }
    @Override public IDiffTensor sub(IDoubleTensor other) { return wrap(data.sub(other)); }
    @Override public IDiffTensor mul(IDoubleTensor other) { return wrap(data.mul(other)); }
    @Override public IDiffTensor div(IDoubleTensor other) { return wrap(data.div(other)); }
    @Override public IDiffTensor add(double scalar) { return wrap(data.add(scalar)); }
    @Override public IDiffTensor sub(double scalar) { return wrap(data.sub(scalar)); }
    @Override public IDiffTensor mul(double scalar) { return wrap(data.mul(scalar)); }
    @Override public IDiffTensor div(double scalar) { return wrap(data.div(scalar)); }

    // ---- element-wise unary ----

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

    // ---- structural with dim (shift) ----

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
        // labels must be a BatchedDiffTensor with same batch handling
        if (labels instanceof BatchedDiffTensor bl) {
            return wrap(data.softmaxCrossEntropy(bl.data, shift(dim)));
        }
        return wrap(data.softmaxCrossEntropy(labels, shift(dim)));
    }

    // ---- accessors ----

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
    @Override public IDiffTensor copy() { return new BatchedDiffTensor((IDiffTensor) data.copy()); }
    @Override public double[] toDoubleArray() { return data.toDoubleArray(); }
    @Override public IMatrix toMatrix() { return data.toMatrix(); }
    @Override public IDoubleVector toVector() { return data.toVector(); }
    @Override public IDoubleVector toVectorCopy() { return data.toVectorCopy(); }
    @Override public List<IDoubleTensor> unstack(int dim) { return data.unstack(dim); }
    @Override public IDiffVector flattenValue() { return data.flattenValue(); }
    @Override public IDiffVector flattenGrad() { return data.flattenGrad(); }
    @Override public IDoubleTensor detach() { return data.detach(); }
    @Override public boolean requiresGrad() { return data.requiresGrad(); }
    @Override public IDiffTensor setRequiresGrad(boolean requiresGrad) { return wrap(data.setRequiresGrad(requiresGrad)); }
    @Override public IDoubleTensor grad() { return data.grad(); }

    // ---- scalar reductions ----

    @Override public double sumAll() { return data.sumAll(); }
    @Override public double meanAll() { return data.meanAll(); }
    @Override public double maxAll() { return data.maxAll(); }
    @Override public double minAll() { return data.minAll(); }
    @Override public double prodAll() { return data.prodAll(); }

    @Override public void backward() { data.backward(); }
    @Override public void backward(IDoubleTensor gradient) { data.backward(gradient); }
    @Override public void zeroGradient() { data.zeroGradient(); }

    // ---- structural: delegate without wrapping (work on [B,...] shape) ----

    @Override public IDiffTensor reshape(int... newShape) { return data.reshape(newShape); }
    @Override public IDiffTensor permute(int... dims) { return data.permute(dims); }
    @Override public IDiffTensor transpose(int dim0, int dim1) { return data.transpose(dim0, dim1); }
    @Override public IDiffTensor transpose() { return data.transpose(); }
    @Override public IDiffTensor squeeze(int... dims) { return data.squeeze(dims); }
    @Override public IDiffTensor unsqueeze(int dim) { return data.unsqueeze(dim); }
    @Override public IDiffTensor flatten(int startDim, int endDim) { return data.flatten(startDim, endDim); }
    @Override public IDiffTensor expand(int... shape) { return data.expand(shape); }
    @Override public IDiffTensor contiguous() { return data.contiguous(); }
    @Override public IDiffTensor tile(int... repeats) { return data.tile(repeats); }
    @Override public IDiffTensor broadcastTo(int... shape) { return data.broadcastTo(shape); }
    @Override public IDiffTensor clone() { return data.clone(); }
    @Override public IDiffTensor add_(IDoubleTensor other) { return data.add_(other); }
    @Override public IDiffTensor sub_(IDoubleTensor other) { return data.sub_(other); }
    @Override public IDiffTensor mul_(IDoubleTensor other) { return data.mul_(other); }
    @Override public IDiffTensor div_(IDoubleTensor other) { return data.div_(other); }
    @Override public IDiffTensor fill_(double value) { return data.fill_(value); }
    @Override public IDiffTensor copy_(IDoubleTensor src) { return data.copy_(src); }

    // ---- structural ops that break batch abstraction ----

    @Override public IDiffTensor cumsum(int dim) { return data.cumsum(shift(dim)); }
    @Override public IDiffTensor cumprod(int dim) { return data.cumprod(shift(dim)); }
    @Override public IDiffTensor mmul(IDoubleTensor other) { throw unsupported("mmul"); }
    @Override public IDiffTensor bmm(IDoubleTensor other) { throw unsupported("bmm"); }
    @Override public IDiffTensor einsum(String subscript, IDoubleTensor... others) { throw unsupported("einsum"); }
    @Override public IDiffTensor gather(int dim, IDoubleTensor index) { throw unsupported("gather"); }
    @Override public IDiffTensor indexSelect(int dim, IDoubleTensor index) { throw unsupported("indexSelect"); }
    @Override public IDiffTensor argsort(int dim, boolean descending) { throw unsupported("argsort"); }
    @Override public IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) { throw unsupported("scatter"); }
    @Override public IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) { throw unsupported("scatterAdd"); }
    @Override public IDiffTensor where(IDoubleTensor condition, IDoubleTensor other) { throw unsupported("where"); }
    @Override public IDiffTensor topk(int k, int dim, boolean largest) { throw unsupported("topk"); }
    @Override public IDiffTensor pad(int[][] padding, String mode, double value) { throw unsupported("pad"); }
    @Override public IDiffTensor unfold(int dim, int size, int stride, int dilation) { throw unsupported("unfold"); }
    @Override public IDiffTensor nonzero() { throw unsupported("nonzero"); }
    @Override public IDiffTensor maskedSelect(IDoubleTensor mask) { throw unsupported("maskedSelect"); }
    @Override public IDiffTensor maskedFill(IDoubleTensor mask, double value) { throw unsupported("maskedFill"); }
    @Override public IDiffTensor cat(int dim, IDoubleTensor... others) { throw unsupported("cat"); }
    @Override public IDiffTensor stack(int dim, IDoubleTensor... others) { throw unsupported("stack"); }
    @Override public IDiffTensor normalize(double p, int dim) { throw unsupported("normalize"); }

    // ---- helpers ----

    private int shift(int dim) {
        return dim + 1;
    }

    private static UnsupportedOperationException unsupported(String op) {
        return new UnsupportedOperationException(
            op + " is not supported on BatchedDiffTensor. "
            + "Only element-wise operations and sum/mean reductions are allowed inside vmapped functions.");
    }
}
