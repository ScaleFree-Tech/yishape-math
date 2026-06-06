package com.yishape.lab.math.autodiff.impl;

import java.util.function.Function;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

/**
 * Lightweight IDiffVector bridge over a RereDiffTensor.
 *
 * <p>Returned by {@link RereDiffTensor#flattenValue()} and
 * {@link RereDiffTensor#flattenGrad()} for optimizer integration.
 * All shape-aware operations delegate to the backing tensor; element-wise
 * operations are unsupported in this flat context.
 *
 * @deprecated Part of the dual-AD bridge being phased out in Tensor-First AD unification.
 *             In the unified architecture, {@code flattenValue()}/{@code flattenGrad()}
 *             will return a rank-1 {@code RereDiffTensor} view instead of this wrapper.
 *             Optimizer integration will use {@code IDiffTensor.flattenGrad()} directly.
 *             Migration tracked in Phase 7 of the AD Unification plan.
 */
@Deprecated
public final class TensorBackedDiffVector implements IDiffVector {

    private final RereDiffTensor tensor;
    private final boolean gradMode;

    /**
     * @param tensor   backing tensor
     * @param gradMode true = flattenGrad (value is grad array), false = flattenValue (value is tensor data)
     */
    TensorBackedDiffVector(RereDiffTensor tensor, boolean gradMode) {
        this.tensor = tensor;
        this.gradMode = gradMode;
    }

    public RereDiffTensor unwrap() { return tensor; }

    // ---- value / gradient ----

    @Override
    public IDoubleVector getValue() {
        if (gradMode) {
            double[] g = tensor.grad;
            return g != null ? IDoubleVector.of(g) : IDoubleVector.of(new double[(int) tensor.value.totalSize()]);
        }
        return IDoubleVector.of(tensor.value.toDoubleArray());
    }

    @Override
    public IDoubleVector getGradient() {
        if (gradMode) return getValue();
        double[] g = tensor.grad;
        return g != null ? IDoubleVector.of(g) : null;
    }

    @Override
    public boolean isLeaf() { return tensor.isLeaf; }

    @Override
    public int size() { return (int) tensor.value.totalSize(); }

    @Override
    public double get(int i) {
        return gradMode ? tensor.grad[i] : tensor.value.linearGet(i);
    }

    // ---- gradient ops (delegate to tensor) ----

    @Override
    public void backward() { tensor.backward(); }

    @Override
    public void backward(IDoubleVector initialGradient) {
        tensor.grad = initialGradient.getData();
        tensor.backward();
    }

    @Override
    public void zeroGradient() { tensor.zeroGradient(); }

    @Override
    public IDiffVector grad() { return tensor.flattenGrad(); }

    @Override
    public IDiffVector copy() {
        RereDiffTensor t = (RereDiffTensor) tensor.clone();
        return new TensorBackedDiffVector(t, gradMode);
    }

    // ---- reductions (delegate to tensor) ----

    @Override
    public IDiffVector sum() {
        IDiffTensor st = tensor.sum();
        return st.flattenValue();
    }

    @Override
    public IDiffVector mean() {
        long n = tensor.value.totalSize();
        IDiffTensor st = tensor.sum();
        return st.div((double) n).flattenValue();
    }

    // ---- in-place operations (leaf-only, delegate to tensor methods) ----

    @Override
    public IDiffVector divideInPlace(double alpha) {
        if (gradMode) throw new UnsupportedOperationException("divideInPlace on grad view");
        for (long i = 0; i < tensor.value.totalSize(); i++) {
            tensor.value.linearSet(i, tensor.value.linearGet(i) / alpha);
        }
        return this;
    }

    @Override
    public IDiffVector addInPlace(IDiffVector other) {
        if (gradMode) throw new UnsupportedOperationException("addInPlace on grad view");
        double[] od = other.getValue().getData();
        long n = tensor.value.totalSize();
        for (long i = 0; i < n; i++) {
            tensor.value.linearSet(i, tensor.value.linearGet(i) + od[(int) i]);
        }
        return this;
    }

    @Override
    public IDiffVector mulInPlace(double scalar) {
        if (gradMode) throw new UnsupportedOperationException("mulInPlace on grad view");
        for (long i = 0; i < tensor.value.totalSize(); i++) {
            tensor.value.linearSet(i, tensor.value.linearGet(i) * scalar);
        }
        return this;
    }

    // ============ Element-wise ops (bridged to RereDiffVector) ============

    /**
     * Creates a RereDiffVector with a backward bridge that routes gradients
     * back to the underlying tensor's grad array. Used to enable element-wise
     * operations on TensorBackedDiffVector.
     */
    private RereDiffVector asRereDiff() {
        RereDiffVector rv = new RereDiffVector(getValue());
        final RereDiffTensor t = this.tensor;
        rv.backwardFn = gradVec -> {
            double[] g = gradVec.getData();
            if (t.grad == null) t.grad = new double[(int) t.value.totalSize()];
            for (int i = 0; i < g.length; i++) t.grad[i] += g[i];
        };
        return rv;
    }

    private IDiffVector unsupported(String op) {
        throw new UnsupportedOperationException(
            op + " not supported on TensorBackedDiffVector — use tensor-level ops instead");
    }

    // Element-wise ops bridged through asRereDiff()
    @Override public IDiffVector add(IDiffVector o) { return asRereDiff().add(o); }
    @Override public IDiffVector sub(IDiffVector o) { return asRereDiff().sub(o); }
    @Override public IDiffVector mul(IDiffVector o) { return asRereDiff().mul(o); }
    @Override public IDiffVector div(IDiffVector o) { return asRereDiff().div(o); }
    @Override public IDiffVector add(double s) { return asRereDiff().add(s); }
    @Override public IDiffVector sub(double s) { return asRereDiff().sub(s); }
    @Override public IDiffVector mul(double s) { return asRereDiff().mul(s); }
    @Override public IDiffVector div(double s) { return asRereDiff().div(s); }
    @Override public IDiffVector rsub(double s) { return asRereDiff().rsub(s); }
    @Override public IDiffVector rdiv(double s) { return asRereDiff().rdiv(s); }
    @Override public IDiffVector neg() { return asRereDiff().neg(); }
    @Override public IDiffVector pow(double n) { return asRereDiff().pow(n); }
    @Override public IDiffVector exp() { return asRereDiff().exp(); }
    @Override public IDiffVector log() { return asRereDiff().log(); }
    @Override public IDiffVector sin() { return asRereDiff().sin(); }
    @Override public IDiffVector cos() { return asRereDiff().cos(); }
    @Override public IDiffVector tan() { return asRereDiff().tan(); }
    @Override public IDiffVector tanh() { return asRereDiff().tanh(); }
    @Override public IDiffVector sigmoid() { return asRereDiff().sigmoid(); }
    @Override public IDiffVector relu() { return asRereDiff().relu(); }
    @Override public IDiffVector gelu() { return asRereDiff().gelu(); }
    @Override public IDiffVector leakyRelu(double a) { return asRereDiff().leakyRelu(a); }
    @Override public IDiffVector elu(double a) { return asRereDiff().elu(a); }
    @Override public IDiffVector selu() { return asRereDiff().selu(); }
    @Override public IDiffVector silu() { return asRereDiff().silu(); }
    @Override public IDiffVector mish() { return asRereDiff().mish(); }
    @Override public IDiffVector softplus(double b) { return asRereDiff().softplus(b); }
    @Override public IDiffVector hardtanh(double a, double b) { return asRereDiff().hardtanh(a, b); }
    @Override public IDiffVector clamp(double a, double b) { return asRereDiff().clamp(a, b); }
    @Override public IDiffVector abs() { return asRereDiff().abs(); }
    @Override public IDiffVector sqrt() { return asRereDiff().sqrt(); }
    @Override public IDiffVector square() { return asRereDiff().square(); }
    @Override public IDiffVector dot(IDiffVector o) { return asRereDiff().dot(o); }
    @Override public IDiffVector broadcast(int n) { return asRereDiff().broadcast(n); }

    // Shape-changing ops kept unsupported (need tensor-level semantics)
    @Override public IDiffVector softmax() { return unsupported("softmax"); }
    @Override public IDiffVector logSoftmax() { return unsupported("logSoftmax"); }
    @Override public IDiffVector layerNorm(IDiffVector g, IDiffVector b, double e) { return unsupported("layerNorm"); }
    @Override public IDiffVector batchNorm(IDiffVector g, IDiffVector b, double e) { return unsupported("batchNorm"); }
    @Override public IDiffVector dropout(double p) { return unsupported("dropout"); }
    @Override
    public IDiffVector slice(int start, int end) {
        int len = end - start;
        double[] data = new double[len];
        double[] full = getValue().getData();
        System.arraycopy(full, start, data, 0, len);
        // Create a RereDiffVector with backward bridge to the tensor's grad
        RereDiffVector vec = new RereDiffVector(IDoubleVector.of(data));
        final RereDiffTensor t = this.tensor;
        final int off = start;
        vec.backwardFn = gradVec -> {
            double[] g = gradVec.getData();
            if (t.grad == null) t.grad = new double[(int) t.value.totalSize()];
            for (int i = 0; i < len; i++) t.grad[off + i] += g[i];
        };
        return vec;
    }

    @Override
    public IDiffVector cat(IDiffVector... others) {
        int total = size();
        for (IDiffVector o : others) total += o.getValue().size();
        double[] data = new double[total];
        System.arraycopy(getValue().getData(), 0, data, 0, size());
        int pos = size();
        for (IDiffVector o : others) {
            double[] od = o.getValue().getData();
            System.arraycopy(od, 0, data, pos, od.length);
            pos += od.length;
        }
        // Build a proper RereDiffVector that bridges gradients back to each input
        RereDiffVector result = new RereDiffVector(IDoubleVector.of(data));
        final IDiffVector[] allParts = new IDiffVector[1 + others.length];
        allParts[0] = this;
        System.arraycopy(others, 0, allParts, 1, others.length);
        final int[] offsets = new int[allParts.length + 1];
        offsets[0] = 0;
        for (int i = 0; i < allParts.length; i++) {
            offsets[i + 1] = offsets[i] + allParts[i].getValue().size();
        }
        result.backwardFn = gradVec -> {
            double[] g = gradVec.getData();
            for (int i = 0; i < allParts.length; i++) {
                int len = offsets[i + 1] - offsets[i];
                double[] partGrad = new double[len];
                System.arraycopy(g, offsets[i], partGrad, 0, len);
                IDiffVector part = allParts[i];
                if (part instanceof TensorBackedDiffVector tb) {
                    double[] tg = tb.tensor.grad;
                    if (tg == null) tb.tensor.grad = new double[(int) tb.tensor.value.totalSize()];
                    for (int j = 0; j < len; j++) tb.tensor.grad[j] += partGrad[j];
                } else if (part instanceof RereDiffVector rv) {
                    rv.accGrad(IDoubleVector.of(partGrad));
                }
            }
        };
        return result;
    }
    @Override public IDiffVector add(IVector<Double> v) { return unsupported("add(IVector)"); }
    @Override public IDiffVector sub(IVector<Double> v) { return unsupported("sub(IVector)"); }
    @Override public IDiffVector multiply(IVector<Double> v) { return unsupported("multiply"); }
    @Override public IDiffVector divide(IVector<Double> o) { return unsupported("divide"); }
    @Override public IDiffVector innerProduct(IVector<Double> v) { return unsupported("innerProduct"); }
    @Override public IDiffVector dot(IVector<Double> v) { return unsupported("dot(IVector)"); }

    // ---- IDoubleVector in-place ops (delegate, no wrapping) ----

    @Override public IDoubleVector negInPlace() { return unsupported("negInPlace"); }
    @Override public IDoubleVector addScalarInPlace(double p) { return unsupported("addScalarInPlace"); }
    @Override public IDoubleVector subScalarInPlace(double p) { return unsupported("subScalarInPlace"); }
    @Override public IDoubleVector multiplyByScalarInPlace(double p) { return unsupported("multiplyByScalarInPlace"); }
    @Override public IDoubleVector addInPlace(IVector<Double> vec) { return unsupported("addInPlace"); }
    @Override public IDoubleVector subInPlace(IVector<Double> vec) { return unsupported("subInPlace"); }
    @Override public IDoubleVector multiplyInPlace(IVector<Double> vec) { return unsupported("multiplyInPlace"); }
}
