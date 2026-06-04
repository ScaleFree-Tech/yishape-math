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
 */
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

    // ============ Unsupported / Delegated element-wise ops ============

    private IDiffVector unsupported(String op) {
        throw new UnsupportedOperationException(
            op + " not supported on TensorBackedDiffVector — use tensor-level ops instead");
    }

    @Override public IDiffVector add(IDiffVector o) { return unsupported("add"); }
    @Override public IDiffVector sub(IDiffVector o) { return unsupported("sub"); }
    @Override public IDiffVector mul(IDiffVector o) { return unsupported("mul"); }
    @Override public IDiffVector div(IDiffVector o) { return unsupported("div"); }
    @Override public IDiffVector add(double s) { return unsupported("add"); }
    @Override public IDiffVector sub(double s) { return unsupported("sub"); }
    @Override public IDiffVector mul(double s) { return unsupported("mul"); }
    @Override public IDiffVector div(double s) { return unsupported("div"); }
    @Override public IDiffVector rsub(double s) { return unsupported("rsub"); }
    @Override public IDiffVector rdiv(double s) { return unsupported("rdiv"); }
    @Override public IDiffVector neg() { return unsupported("neg"); }
    @Override public IDiffVector pow(double n) { return unsupported("pow"); }
    @Override public IDiffVector exp() { return unsupported("exp"); }
    @Override public IDiffVector log() { return unsupported("log"); }
    @Override public IDiffVector sin() { return unsupported("sin"); }
    @Override public IDiffVector cos() { return unsupported("cos"); }
    @Override public IDiffVector tan() { return unsupported("tan"); }
    @Override public IDiffVector tanh() { return unsupported("tanh"); }
    @Override public IDiffVector sigmoid() { return unsupported("sigmoid"); }
    @Override public IDiffVector relu() { return unsupported("relu"); }
    @Override public IDiffVector gelu() { return unsupported("gelu"); }
    @Override public IDiffVector leakyRelu(double a) { return unsupported("leakyRelu"); }
    @Override public IDiffVector elu(double a) { return unsupported("elu"); }
    @Override public IDiffVector selu() { return unsupported("selu"); }
    @Override public IDiffVector silu() { return unsupported("silu"); }
    @Override public IDiffVector mish() { return unsupported("mish"); }
    @Override public IDiffVector softplus(double b) { return unsupported("softplus"); }
    @Override public IDiffVector hardtanh(double a, double b) { return unsupported("hardtanh"); }
    @Override public IDiffVector clamp(double a, double b) { return unsupported("clamp"); }
    @Override public IDiffVector layerNorm(IDiffVector g, IDiffVector b, double e) { return unsupported("layerNorm"); }
    @Override public IDiffVector batchNorm(IDiffVector g, IDiffVector b, double e) { return unsupported("batchNorm"); }
    @Override public IDiffVector dropout(double p) { return unsupported("dropout"); }
    @Override public IDiffVector abs() { return unsupported("abs"); }
    @Override public IDiffVector sqrt() { return unsupported("sqrt"); }
    @Override public IDiffVector square() { return unsupported("square"); }
    @Override public IDiffVector softmax() { return unsupported("softmax"); }
    @Override public IDiffVector logSoftmax() { return unsupported("logSoftmax"); }
    @Override public IDiffVector dot(IDiffVector o) { return unsupported("dot"); }
    @Override public IDiffVector broadcast(int n) { return unsupported("broadcast"); }
    @Override public IDiffVector slice(int s, int e) { return unsupported("slice"); }
    @Override public IDiffVector cat(IDiffVector... o) { return unsupported("cat"); }
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
