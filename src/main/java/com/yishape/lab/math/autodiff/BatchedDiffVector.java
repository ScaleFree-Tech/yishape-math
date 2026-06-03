package com.yishape.lab.math.autodiff;

import java.util.List;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

/**
 * Batch-aware differentiable vector for single-graph vmap execution.
 *
 * <p>Wraps a flat vector of length {@code N * D} (sample-major layout) and
 * intercepts {@link #sum()} / {@link #mean()} to reduce per-sample over the
 * inner dimension {@code D}. Element-wise operations are delegated to the
 * underlying vector and re-wrapped so batch context flows through the chain.
 *
 * <p>Example: {@code v.pow(2).sum()} on a {@code BatchedDiffVector} with N=4, D=3:
 * <ol>
 *   <li>{@code pow(2)} — element-wise, produces BatchedDiffVector[N*D]</li>
 *   <li>{@code sum()} — intercepts, reshapes to [N, D], sums axis=1 → IDiffVector[N]</li>
 * </ol>
 *
 * <p>Structural operations ({@code slice}, {@code dot}, {@code broadcast},
 * {@code cat}, {@code softmax}) are unsupported — they would break the batch
 * abstraction.
 *
 * @author lteb2
 */
final class BatchedDiffVector implements IDiffVector {

    private final IDiffVector data;
    private final int batch;
    private final int dim;

    BatchedDiffVector(IDiffVector data, int batch, int dim) {
        this.data = data;
        this.batch = batch;
        this.dim = dim;
    }

    IDiffVector unwrap() {
        return data;
    }

    int batchSize() {
        return batch;
    }

    int sampleDim() {
        return dim;
    }

    private IDiffVector wrap(IDiffVector result) {
        if (result == data) return this;
        if (result instanceof BatchedDiffVector bdv) return bdv;
        return new BatchedDiffVector(result, batch, dim);
    }

    // ---- batch-aware reductions ----

    @Override
    public IDiffVector sum() {
        IDiffMatrix mat = data.reshape(batch, dim);
        return mat.sum(1);
    }

    @Override
    public IDiffVector mean() {
        IDiffMatrix mat = data.reshape(batch, dim);
        return mat.sum(1).div((double) dim);
    }

    // ---- accessors ----

    @Override
    public IDoubleVector getValue() {
        return data.getValue();
    }

    @Override
    public IDoubleVector getGradient() {
        return data.getGradient();
    }

    @Override
    public boolean isLeaf() {
        return data.isLeaf();
    }

    @Override
    public void backward() {
        data.backward();
    }

    @Override
    public void backward(IDoubleVector initialGradient) {
        data.backward(initialGradient);
    }

    @Override
    public void zeroGradient() {
        data.zeroGradient();
    }

    @Override
    public IDiffVector grad() {
        return data.grad();
    }

    @Override
    public int size() {
        return data.size();
    }

    @Override
    public double get(int i) {
        return data.get(i);
    }

    @Override
    public IDiffVector copy() {
        return new BatchedDiffVector(data.copy(), batch, dim);
    }

    @Override
    public IDiffVector divideInPlace(double alpha) {
        return wrap(data.divideInPlace(alpha));
    }

    @Override
    public IDiffVector addInPlace(IDiffVector other) {
        return wrap(data.addInPlace(unwrapIfNeeded(other)));
    }

    @Override
    public IDiffVector mulInPlace(double scalar) {
        return wrap(data.mulInPlace(scalar));
    }

    // ---- IDoubleVector in-place operations (delegate, no wrapping) ----

    @Override
    public IDoubleVector negInPlace() {
        return data.negInPlace();
    }

    @Override
    public IDoubleVector addScalarInPlace(double p) {
        return data.addScalarInPlace(p);
    }

    @Override
    public IDoubleVector subScalarInPlace(double p) {
        return data.subScalarInPlace(p);
    }

    @Override
    public IDoubleVector multiplyByScalarInPlace(double p) {
        return data.multiplyByScalarInPlace(p);
    }

    @Override
    public IDoubleVector addInPlace(IVector<Double> vec) {
        return data.addInPlace(vec);
    }

    @Override
    public IDoubleVector subInPlace(IVector<Double> vec) {
        return data.subInPlace(vec);
    }

    @Override
    public IDoubleVector multiplyInPlace(IVector<Double> vec) {
        return data.multiplyInPlace(vec);
    }

    // ---- arithmetic with IDiffVector ----

    @Override
    public IDiffVector add(IDiffVector other) {
        return wrap(data.add(unwrapIfNeeded(other)));
    }

    @Override
    public IDiffVector sub(IDiffVector other) {
        return wrap(data.sub(unwrapIfNeeded(other)));
    }

    @Override
    public IDiffVector mul(IDiffVector other) {
        return wrap(data.mul(unwrapIfNeeded(other)));
    }

    @Override
    public IDiffVector div(IDiffVector other) {
        return wrap(data.div(unwrapIfNeeded(other)));
    }

    // ---- arithmetic with scalar ----

    @Override
    public IDiffVector add(double scalar) {
        return wrap(data.add(scalar));
    }

    @Override
    public IDiffVector sub(double scalar) {
        return wrap(data.sub(scalar));
    }

    @Override
    public IDiffVector mul(double scalar) {
        return wrap(data.mul(scalar));
    }

    @Override
    public IDiffVector div(double scalar) {
        return wrap(data.div(scalar));
    }

    @Override
    public IDiffVector rsub(double scalar) {
        return wrap(data.rsub(scalar));
    }

    @Override
    public IDiffVector rdiv(double scalar) {
        return wrap(data.rdiv(scalar));
    }

    // ---- unary ----

    @Override
    public IDiffVector neg() {
        return wrap(data.neg());
    }

    @Override
    public IDiffVector pow(double n) {
        return wrap(data.pow(n));
    }

    // ---- element-wise math ----

    @Override
    public IDiffVector exp() {
        return wrap(data.exp());
    }

    @Override
    public IDiffVector log() {
        return wrap(data.log());
    }

    @Override
    public IDiffVector sin() {
        return wrap(data.sin());
    }

    @Override
    public IDiffVector cos() {
        return wrap(data.cos());
    }

    @Override
    public IDiffVector tan() {
        return wrap(data.tan());
    }

    @Override
    public IDiffVector tanh() {
        return wrap(data.tanh());
    }

    @Override
    public IDiffVector sigmoid() {
        return wrap(data.sigmoid());
    }

    @Override
    public IDiffVector relu() {
        return wrap(data.relu());
    }

    @Override
    public IDiffVector gelu() {
        return wrap(data.gelu());
    }

    @Override
    public IDiffVector leakyRelu(double alpha) {
        return wrap(data.leakyRelu(alpha));
    }

    @Override
    public IDiffVector elu(double alpha) {
        return wrap(data.elu(alpha));
    }

    @Override
    public IDiffVector selu() {
        return wrap(data.selu());
    }

    @Override
    public IDiffVector silu() {
        return wrap(data.silu());
    }

    @Override
    public IDiffVector mish() {
        return wrap(data.mish());
    }

    @Override
    public IDiffVector softplus(double beta) {
        return wrap(data.softplus(beta));
    }

    @Override
    public IDiffVector hardtanh(double minVal, double maxVal) {
        return wrap(data.hardtanh(minVal, maxVal));
    }

    @Override
    public IDiffVector clamp(double min, double max) {
        return wrap(data.clamp(min, max));
    }

    @Override
    public IDiffVector abs() {
        return wrap(data.abs());
    }

    @Override
    public IDiffVector sqrt() {
        return wrap(data.sqrt());
    }

    @Override
    public IDiffVector square() {
        return wrap(data.square());
    }

    // ---- normalization ----

    @Override
    public IDiffVector layerNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        throw unsupported("layerNorm");
    }

    @Override
    public IDiffVector batchNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        throw unsupported("batchNorm");
    }

    @Override
    public IDiffVector dropout(double p) {
        return wrap(data.dropout(p));
    }

    // ---- softmax (structure-dependent on flat representation) ----

    @Override
    public IDiffVector softmax() {
        throw unsupported("softmax");
    }

    @Override
    public IDiffVector logSoftmax() {
        throw unsupported("logSoftmax");
    }

    // ---- structural ops ----

    @Override
    public IDiffVector dot(IDiffVector other) {
        throw unsupported("dot");
    }

    @Override
    public IDiffVector broadcast(int n) {
        throw unsupported("broadcast");
    }

    @Override
    public IDiffVector slice(int start, int end) {
        throw unsupported("slice");
    }

    @Override
    public IDiffVector cat(IDiffVector... others) {
        throw unsupported("cat");
    }

    // ---- IVector<Double> variants ----

    @Override
    public IDiffVector add(IVector<Double> vec) {
        return wrap(data.add(vec));
    }

    @Override
    public IDiffVector sub(IVector<Double> vec) {
        return wrap(data.sub(vec));
    }

    @Override
    public IDiffVector multiply(IVector<Double> vec) {
        return wrap(data.multiply(vec));
    }

    @Override
    public IDiffVector divide(IVector<Double> other) {
        return wrap(data.divide(other));
    }

    @Override
    public IDiffVector innerProduct(IVector<Double> vec) {
        throw unsupported("innerProduct");
    }

    @Override
    public IDiffVector dot(IVector<Double> vec) {
        throw unsupported("dot(IVector)");
    }

    // ---- helpers ----

    private IDiffVector unwrapIfNeeded(IDiffVector other) {
        return (other instanceof BatchedDiffVector bdv) ? bdv.data : other;
    }

    private static UnsupportedOperationException unsupported(String op) {
        return new UnsupportedOperationException(
            op + " is not supported on BatchedDiffVector. "
            + "Only element-wise operations and sum/mean reductions are allowed inside vmapped functions.");
    }
}
