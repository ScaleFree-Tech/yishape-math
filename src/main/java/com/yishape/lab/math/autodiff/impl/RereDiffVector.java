package com.yishape.lab.math.autodiff.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.RereDoubleVector;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.util.YishapeLogger;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;
import com.yishape.lab.math.autodiff.IDiffTensor;

/**
 * Default reverse-mode AD implementation for {@link IDiffVector}.
 * Now a thin proxy over {@link RereDiffTensor} (shape=[n]).
 *
 * <p>All operations delegate to the underlying tensor graph. The tensor graph
 * is the single source of truth for gradient propagation. This class provides
 * {@link IDiffVector} API compatibility while the computation graph lives
 * entirely in the tensor domain.</p>
 */
public class RereDiffVector implements IDiffVector, Serializable {

    private static final long serialVersionUID = 5L;
    private static final YishapeLogger log = YishapeLogger.getLogger(RereDiffVector.class);

    /** Global seed counter for dropout mask generation. Shared across vector, matrix, and tensor dropout. */
    public static final AtomicLong DROPOUT_SEED_COUNTER = new AtomicLong(0);

    // ==================== Core field ====================

    /** Underlying rank-1 tensor (shape=[n]). The single source of truth for the computation graph.
     * Public to allow internal graph-manipulation code (GraphOptimizer, GraphExporter, etc.)
     * to operate directly on the tensor graph. */
    public final RereDiffTensor tensor;

    // ==================== Constructors ====================

    /** Wrap an existing tensor as an IDiffVector. */
    public RereDiffVector(RereDiffTensor tensor) {
        this.tensor = tensor;
    }

    /** Create a leaf vector from an IDoubleVector. The underlying data is NOT defensively copied. */
    public RereDiffVector(IDoubleVector value) {
        this(value.getData());
    }

    /** Create a leaf vector from a raw double array. The array is NOT defensively copied. */
    public RereDiffVector(double[] data) {
        this.tensor = new RereDiffTensor(data, new int[]{data.length});
    }

    // ==================== Package-private accessors ====================

    /** Unwrap to the underlying tensor. Package-private: only for autodiff package use. */
    RereDiffTensor unwrap() { return tensor; }

    /** Wrap a tensor as an IDiffVector. */
    static RereDiffVector wrap(RereDiffTensor t) {
        return new RereDiffVector(t);
    }

    /** Safely wrap any IDiffTensor as an IDiffVector, handling ConstantDiffTensor and other non-RereDiffTensor types. */
    static RereDiffVector wrapSafe(IDiffTensor t) {
        if (t instanceof RereDiffTensor rt) return new RereDiffVector(rt);
        // Constant or non-diff tensor — create a non-diff leaf vector
        double[] data = t.toDoubleArray();
        RereDiffTensor leaf = new RereDiffTensor(data, t.shape());
        leaf.setRequiresGrad(false);
        return new RereDiffVector(leaf);
    }

    // ==================== Value / gradient access ====================

    @Override
    public IDoubleVector getValue() {
        return IDoubleVector.of(tensor.value().toDoubleArray());
    }

    @Override
    public IDoubleVector getGradient() {
        return tensor.gradData() != null ? IDoubleVector.of(tensor.gradData().clone()) : null;
    }

    @Override
    public boolean isLeaf() {
        return tensor.isLeaf();
    }

    // ==================== Gradient operations ====================

    @Override
    public void backward() {
        tensor.backward();
    }

    @Override
    public void backward(boolean retainGraph) {
        tensor.backward(retainGraph);
    }

    @Override
    public void backward(IDoubleVector initialGradient) {
        backward(initialGradient, false);
    }

    @Override
    public void backward(IDoubleVector initialGradient, boolean retainGraph) {
        if (!tensor.requiresGrad()) return;
        tensor.setGradData(initialGradient.getData().clone());
        tensor.backwardImpl(retainGraph);
    }

    @Override
    public void zeroGradient() {
        tensor.zeroGradient();
    }

    @Override
    public IDiffVector grad() {
        if (tensor.gradData() == null) {
            throw new IllegalStateException("Gradient is null — call backward() first");
        }
        RereDiffTensor gradTensor = new RereDiffTensor(tensor.gradData().clone(), tensor.shape());
        gradTensor.setRequiresGrad(true);
        return new RereDiffVector(gradTensor);
    }

    /** Accumulates gradient into the underlying tensor. */
    public void accGrad(IDoubleVector grad) {
        tensor.accGrad(grad.getData());
    }

    /** Takes ownership of a freshly-allocated array, avoiding copy on first gradient accumulation. */
    public void accGradDirect(double[] data) {
        tensor.accGrad(data);
    }

    /** Accumulate from pooled buffer. */
    public void accGradFromPooled(double[] pooledBuf, int n) {
        tensor.accGradFromPooled(pooledBuf, n);
    }

    /** Updates the leaf value in-place. Only valid on leaf nodes. */
    public void updateData(double[] newData) {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("updateData() is only allowed on leaf nodes");
        }
        RereDoubleTensor val = tensor.value();
        double[] dest = val.toDoubleArray();
        System.arraycopy(newData, 0, dest, 0, Math.min(newData.length, dest.length));
        tensor.setValue(new RereDoubleTensor(dest, val.shape()));
        tensor.setGradData(null);
    }

    // ==================== resolveToRere (bridge compatibility) ====================

    /**
     * Resolves any IDiffVector to a RereDiffVector for tensor-graph access.
     */
    public static RereDiffVector resolveToRere(IDiffVector v) {
        if (v instanceof RereDiffVector rv) return rv;
        throw new IllegalArgumentException("Cannot resolve to RereDiffVector: " + v.getClass().getSimpleName());
    }

    // ==================== createNonLeaf (internal AD infrastructure) ====================

    /**
     * Creates a non-leaf computation node with an embedded backward function.
     * Package-private: for use by {@link FusedOps}, {@link FusedReductionOps},
     * and other internal AD infrastructure that build graph nodes directly.
     */
    static RereDiffVector createNonLeaf(double[] data, List<RereDiffVector> inputs,
                                         Consumer<IDoubleVector> backwardFn) {
        List<RereDiffTensor> tensorInputs = new ArrayList<>();
        for (RereDiffVector rv : inputs) {
            tensorInputs.add(rv.tensor);
        }
        Consumer<RereDiffTensor> tensorBackwardFn = self -> backwardFn.accept(IDoubleVector.of(self.gradData()));
        RereDiffTensor t = new RereDiffTensor(data, new int[]{data.length}, tensorInputs, tensorBackwardFn, null);
        return new RereDiffVector(t);
    }

    // ==================== Static helpers ====================

    public static IDiffVector constant(double[] data) {
        RereDiffTensor t = new RereDiffTensor(data.clone(), new int[]{data.length});
        t.setRequiresGrad(false);
        return new RereDiffVector(t);
    }

    // ==================== Higher-order AD ====================

    /**
     * Symbolic reverse over the tensor DAG (tape-of-tape / higher-order AD).
     * Delegates to the tensor graph's symbolic backward mechanism.
     */
    public static IDiffVector[] grad(IDiffVector output, IDiffVector... inputs) {
        RereDiffTensor out = ((RereDiffVector) output).tensor;
        RereDiffTensor[] ins = new RereDiffTensor[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            ins[i] = ((RereDiffVector) inputs[i]).tensor;
        }
        com.yishape.lab.math.autodiff.IDiffTensor[] tensorGrads =
            com.yishape.lab.math.autodiff.AD.grad(out, ins);
        IDiffVector[] result = new IDiffVector[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            result[i] = new RereDiffVector((RereDiffTensor) tensorGrads[i]);
        }
        return result;
    }

    // ==================== Arithmetic with variables ====================

    @Override
    public IDiffVector add(IDiffVector other) {
        RereDiffVector o = resolveToRere(other);
        return wrap((RereDiffTensor) tensor.add(o.tensor));
    }

    @Override
    public IDiffVector sub(IDiffVector other) {
        RereDiffVector o = resolveToRere(other);
        return wrap((RereDiffTensor) tensor.sub(o.tensor));
    }

    @Override
    public IDiffVector mul(IDiffVector other) {
        RereDiffVector o = resolveToRere(other);
        return wrap((RereDiffTensor) tensor.mul(o.tensor));
    }

    @Override
    public IDiffVector div(IDiffVector other) {
        RereDiffVector o = resolveToRere(other);
        return wrap((RereDiffTensor) tensor.div(o.tensor));
    }

    // ==================== Arithmetic with scalars ====================

    @Override
    public IDiffVector add(double scalar) {
        if (scalar == 0.0) return this;  // identity
        return wrap((RereDiffTensor) tensor.add(scalar));
    }

    @Override
    public IDiffVector sub(double scalar) {
        if (scalar == 0.0) return this;  // identity
        return wrap((RereDiffTensor) tensor.sub(scalar));
    }

    @Override
    public IDiffVector mul(double scalar) {
        if (scalar == 1.0) return this;  // identity
        if (scalar == 0.0) {
            // Detached constant — no gradient flow to input
            double[] zeros = new double[(int) tensor.value().totalSize()];
            return constant(zeros);
        }
        return wrap((RereDiffTensor) tensor.mul(scalar));
    }

    @Override
    public IDiffVector div(double scalar) {
        if (scalar == 1.0) return this;  // identity
        return wrap((RereDiffTensor) tensor.div(scalar));
    }

    @Override
    public IDiffVector rsub(double scalar) {
        return wrap((RereDiffTensor) tensor.rsub(scalar));
    }

    @Override
    public IDiffVector rdiv(double scalar) {
        return wrap((RereDiffTensor) tensor.rdiv(scalar));
    }

    // ==================== Unary ====================

    @Override
    public IDiffVector neg() {
        return wrap((RereDiffTensor) tensor.neg());
    }

    @Override
    public IDiffVector pow(double n) {
        if (n == 1.0) return this;  // identity
        if (n == 0.0) {
            // x^0 = 1, detached constant
            double[] ones = new double[(int) tensor.value().totalSize()];
            java.util.Arrays.fill(ones, 1.0);
            return constant(ones);
        }
        return wrap((RereDiffTensor) tensor.pow(n));
    }

    // ==================== Element-wise math ====================

    @Override public IDiffVector exp() { return wrap((RereDiffTensor) tensor.exp()); }
    @Override public IDiffVector log() { return wrap((RereDiffTensor) tensor.log()); }
    @Override public IDiffVector sin() { return wrap((RereDiffTensor) tensor.sin()); }
    @Override public IDiffVector cos() { return wrap((RereDiffTensor) tensor.cos()); }
    @Override public IDiffVector tan() { return wrap((RereDiffTensor) tensor.tan()); }
    @Override public IDiffVector tanh() { return wrap((RereDiffTensor) tensor.tanh()); }
    @Override public IDiffVector sigmoid() { return wrap((RereDiffTensor) tensor.sigmoid()); }
    @Override public IDiffVector relu() { return wrap((RereDiffTensor) tensor.relu()); }
    @Override public IDiffVector gelu() { return wrap((RereDiffTensor) tensor.gelu()); }
    @Override public IDiffVector leakyRelu(double alpha) { return wrap((RereDiffTensor) tensor.leakyRelu(alpha)); }
    @Override public IDiffVector elu(double alpha) { return wrap((RereDiffTensor) tensor.elu(alpha)); }
    @Override public IDiffVector selu() { return wrap((RereDiffTensor) tensor.selu()); }
    @Override public IDiffVector silu() { return wrap((RereDiffTensor) tensor.silu()); }
    @Override public IDiffVector mish() { return wrap((RereDiffTensor) tensor.mish()); }
    @Override public IDiffVector softplus(double beta) { return wrap((RereDiffTensor) tensor.softplus(beta)); }
    @Override public IDiffVector hardtanh(double minVal, double maxVal) { return wrap((RereDiffTensor) tensor.hardtanh(minVal, maxVal)); }
    @Override public IDiffVector clamp(double min, double max) { return wrap((RereDiffTensor) tensor.clamp(min, max)); }
    @Override public IDiffVector abs() { return wrap((RereDiffTensor) tensor.abs()); }
    @Override public IDiffVector sqrt() { return wrap((RereDiffTensor) tensor.sqrt()); }
    @Override public IDiffVector square() { return wrap((RereDiffTensor) tensor.square()); }
    @Override public IDiffVector dropout(double p) { return wrap((RereDiffTensor) tensor.dropout(p)); }

    // ==================== Normalization (fused) ====================

    @Override
    public IDiffVector softmax() {
        return wrap((RereDiffTensor) tensor.softmax(0));
    }

    @Override
    public IDiffVector logSoftmax() {
        return wrap((RereDiffTensor) tensor.logSoftmax(0));
    }

    @Override
    public IDiffVector layerNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector gr = resolveToRere(gamma);
        RereDiffVector br = resolveToRere(beta);
        int features = (int) gr.tensor.value().totalSize();
        int total = (int) tensor.value().totalSize();
        int batch = total / features;
        if (total % features != 0) {
            throw new IllegalArgumentException(
                "Input size (" + total + ") not divisible by features (" + features + ")");
        }
        IDiffTensor reshaped = tensor.reshape(batch, features);
        IDiffTensor normalized = reshaped.layerNorm(gr.tensor, br.tensor, eps);
        return wrap((RereDiffTensor) normalized.reshape(total));
    }

    @Override
    public IDiffVector batchNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector gr = resolveToRere(gamma);
        RereDiffVector br = resolveToRere(beta);
        int features = (int) gr.tensor.value().totalSize();
        int total = (int) tensor.value().totalSize();
        int batch = total / features;
        if (total % features != 0) {
            throw new IllegalArgumentException(
                "Input size (" + total + ") not divisible by features (" + features + ")");
        }
        IDiffTensor reshaped = tensor.reshape(batch, features);
        IDiffTensor normalized = reshaped.batchNorm(gr.tensor, br.tensor, eps);
        return wrap((RereDiffTensor) normalized.reshape(total));
    }

    /**
     * Proxy to {@link IDiffTensor#conv2d(IDiffTensor, IDiffTensor, int, int, int)}.
     * Reshapes this 1-D vector to [N,C,H,W], performs conv2d, and reshapes back.
     */
    public IDiffVector conv2d(IDiffVector weight, IDiffVector bias,
                               int[] inShape, int stride, int padding, int dilation) {
        RereDiffVector wv = resolveToRere(weight);
        RereDiffVector bv = resolveToRere(bias);
        IDiffTensor reshaped = tensor.reshape(inShape);
        IDiffTensor result = reshaped.conv2d(wv.tensor, bv != null ? bv.tensor : null,
            stride, padding, dilation);
        long total = result.totalSize();
        if (total > Integer.MAX_VALUE) throw new IllegalArgumentException("output too large for vector");
        return wrap((RereDiffTensor) result.reshape((int) total));
    }

    /**
     * Proxy to {@link IDiffTensor#scaledDotProductAttention(IDiffTensor, IDiffTensor, IDiffTensor, double)}.
     * Reshapes this 1-D vector to [batch, seqQ, d_k], performs attention, and reshapes back.
     */
    public IDiffVector scaledDotProductAttention(IDiffVector key, IDiffVector vTensor,
                                                   IDiffVector mask, int[] shape,
                                                   double dropout) {
        RereDiffVector kv = resolveToRere(key);
        RereDiffVector vv = resolveToRere(vTensor);
        RereDiffVector mv = resolveToRere(mask);
        IDiffTensor reshaped = tensor.reshape(shape);
        IDiffTensor result = reshaped.scaledDotProductAttention(
            kv.tensor, vv.tensor, mv != null ? mv.tensor : null, dropout);
        long total = result.totalSize();
        if (total > Integer.MAX_VALUE) throw new IllegalArgumentException("output too large for vector");
        return wrap((RereDiffTensor) result.reshape((int) total));
    }

    // ==================== Reductions ====================

    @Override
    public IDiffVector sum() {
        return wrapSafe(tensor.sum());
    }

    /**
     * Mean of all elements. Delegates to {@link RereDiffTensor#mean()} which
     * holds the canonical full-reduce mean definition (fused squareMean/expMean
     * via {@code tryFuseMeanDim}, native "mean" node otherwise). Phase A §7c:
     * no vector-level graph construction.
     */
    @Override
    public IDiffVector mean() {
        return wrapSafe(tensor.mean());
    }

    // ==================== Vector operations ====================

    /**
     * Single-node fused dot product: creates one graph node with 2 inputs (a, b).
     * Forward = sum(a * b), backward: ∂/∂a = grad * b, ∂/∂b = grad * a.
     *
     * <p>Must NOT use the old mul()+sum() chain — that produces a two-node graph
     * where the "dot" tag on the sum node has only 1 input. GPU/HPC "dot" dispatch
     * expects exactly 2 inputs to compute both ∂a and ∂b gradients.</p>
     */
    @Override
    public IDiffVector dot(IDiffVector other) {
        RereDiffVector o = resolveToRere(other);
        // Fused 2-input "dot" node built at the tensor layer (GPU/HPC dispatch +
        // tape-of-tape symbolic backward for MixedMode.hvp). See DiffTensorReduce.dot.
        return wrap((RereDiffTensor) tensor.dot(o.tensor));
    }

    @Override
    public IDiffVector broadcast(int n) {
        if (tensor.value().totalSize() != 1) {
            throw new IllegalArgumentException("broadcast requires scalar (size=1) input, got size=" + tensor.value().totalSize());
        }
        return wrap((RereDiffTensor) tensor.broadcastTo(n));
    }

    @Override
    public boolean isFlat() {
        int[] shape = tensor.shape();
        if (shape.length <= 1) return true;
        return shape[0] == tensor.value().totalSize();
    }

    @Override
    public IDiffVector slice(int start, int end) {
        long totalSize = tensor.value().totalSize();
        if (start < 0 || end > totalSize) {
            throw new IllegalArgumentException(
                "slice(" + start + ", " + end + ") out of bounds for size " + totalSize);
        }
        int[] shape = tensor.shape();
        IDiffTensor sliced;
        if (shape.length > 1 && shape[0] != totalSize) {
            // Multi-D backing (e.g. shape=[B, features]): reshape to flat 1-D first
            // so that slice() interprets start/end as flat element indices, not dim-0 rows.
            sliced = tensor.reshape((int) totalSize).slice(0, start, end);
        } else {
            sliced = tensor.slice(0, start, end);
        }
        return wrap((RereDiffTensor) sliced);
    }

    @Override
    public IDiffVector cat(IDiffVector... others) {
        // Flatten to 1D for vector-level concatenation.
        // Vectors may wrap multi-dim tensors (e.g. from flattenValue()),
        // which would cause DiffTensorAdvanced.cat to compute wrong innerSize.
        RereDiffTensor self1d = (tensor.rank() == 1) ? tensor
            : (RereDiffTensor) tensor.reshape(new int[]{(int) tensor.totalSize()});
        IDiffTensor[] otherTensors = new IDiffTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            RereDiffTensor ot = ((RereDiffVector) others[i]).tensor;
            otherTensors[i] = (ot.rank() == 1) ? ot
                : (RereDiffTensor) ot.reshape(new int[]{(int) ot.totalSize()});
        }
        return wrap((RereDiffTensor) self1d.cat(0, otherTensors));
    }

    // ==================== In-place operations ====================

    @Override
    public IDiffVector addInPlace(IDiffVector other) {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("addInPlace only allowed on leaf variables");
        }
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        double[] otherData = other.toDoubleArray();
        double[] result = new com.yishape.lab.math.compute.DoubleVectorComputer()
            .binaryOperate(data, otherData, com.yishape.lab.math.compute.ops.BinaryOperation.ADD);
        tensor.setValue(new RereDoubleTensor(result, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    @Override
    public IDiffVector mulInPlace(double scalar) {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("mulInPlace only allowed on leaf variables");
        }
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        for (int i = 0; i < data.length; i++) {
            data[i] *= scalar;
        }
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    // ==================== Copy ====================

    @Override
    public IDiffVector copy() {
        return wrap((RereDiffTensor) tensor.clone());
    }

    @Override
    public IDiffVector detach() {
        return wrap((RereDiffTensor) tensor.detach());
    }

    // ==================== IDiffMatrix reshape ====================

    @Override
    public IDiffMatrix reshape(int rows, int cols) {
        int origSize = (int) tensor.value().totalSize();
        if (rows * cols != origSize) {
            throw new IllegalArgumentException(
                "reshape dimensions " + rows + "x" + cols + " must match size " + origSize);
        }
        // Delegate to tensor graph — reshape returns a tensor with new shape.
        // Wrap as RereDiffMatrix for IDiffMatrix API compatibility.
        return new RereDiffMatrix((RereDiffTensor) tensor.reshape(rows, cols));
    }

    // ==================== Covariant overrides from IDoubleVector / IVector ====================

    @Override
    public IDiffVector divideInPlace(double alpha) {
        if (alpha == 1.0) return this;
        return this.mulInPlace(1.0 / alpha);
    }

    @Override
    public IDiffVector addScalarInPlace(double p) {
        if (p == 0.0) return this;
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] += p;
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    @Override
    public IDiffVector subScalarInPlace(double p) {
        if (p == 0.0) return this;
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] -= p;
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    @Override
    public IDiffVector multiplyByScalarInPlace(double p) {
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] *= p;
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    @Override
    public IDiffVector addInPlace(IVector<Double> vec) {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("addInPlace only allowed on leaf variables");
        }
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] += other[i];
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    @Override
    public IDiffVector subInPlace(IVector<Double> vec) {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("subInPlace only allowed on leaf variables");
        }
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] -= other[i];
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    @Override
    public IDiffVector multiplyInPlace(IVector<Double> vec) {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("multiplyInPlace only allowed on leaf variables");
        }
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] *= other[i];
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    @Override
    public IDiffVector negInPlace() {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("negInPlace only allowed on leaf variables");
        }
        RereDoubleTensor val = tensor.value();
        double[] data = val.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] = -data[i];
        tensor.setValue(new RereDoubleTensor(data, val.shape()));
        tensor.setGradData(null);
        return this;
    }

    @Override
    public IDiffVector add(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) return this.add(dv);
        return this.add(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector sub(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) return this.sub(dv);
        return this.sub(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector multiply(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) return this.mul(dv);
        return this.mul(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector divide(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) return this.div(dv);
        return this.div(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector dot(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) return this.dot(dv);
        return this.dot(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector innerProduct(IVector<Double> vec) {
        return this.dot(vec);
    }

    @Override public double dtw(IVector<Double> other) { return getValue().dtw(other); }
    @Override public double normInf() { return getValue().normInf(); }

    // ==================== IDoubleVector bridge methods ====================

    @Override public IDiffVector round() { return wrap((RereDiffTensor) tensor.round()); }
    @Override public IDiffVector floor() { return wrap((RereDiffTensor) tensor.floor()); }
    @Override public IDiffVector ceil() { return wrap((RereDiffTensor) tensor.ceil()); }
    @Override public IDiffVector trunc() { return wrap((RereDiffTensor) tensor.trunc()); }
    @Override public IDiffVector sign() { return wrap((RereDiffTensor) tensor.sign()); }

    @Override
    public IDiffVector log10() {
        // log10(x) = log(x) / log(10)
        return wrap((RereDiffTensor) tensor.log().div(Math.log(10)));
    }

    @Override
    public IDiffVector reciprocal() {
        return wrap((RereDiffTensor) tensor.reciprocal());
    }

    @Override
    public IDiffVector arcsin() {
        return wrap((RereDiffTensor) tensor.arcsin());
    }

    @Override
    public IDiffVector arccos() {
        return wrap((RereDiffTensor) tensor.arccos());
    }

    @Override
    public IDiffVector arctan() {
        return wrap((RereDiffTensor) tensor.arctan());
    }

    @Override
    public IDiffVector sinh() {
        return wrap((RereDiffTensor) tensor.sinh());
    }

    @Override
    public IDiffVector cosh() {
        return wrap((RereDiffTensor) tensor.cosh());
    }

    @Override
    public IDiffVector remainder(Double value) {
        return wrap((RereDiffTensor) tensor.remainder(value));
    }

    // ==================== Reverse ====================

    @Override
    public IDiffVector reverse() {
        return wrap((RereDiffTensor) tensor.flip(0));
    }

    // ==================== Tile / repeat ====================

    @Override
    public IDiffVector tile(int reps) {
        return wrap((RereDiffTensor) tensor.tile(reps));
    }

    @Override
    public IDiffVector repeat(int repeats) {
        // repeatInterleave(repeats, 0) tiles each element `repeats` times: [a,b]→[a,a,b,b],
        // matching y[i] = fwd[i/repeats]; its scatter-add backward sums repeated grads.
        return wrap((RereDiffTensor) tensor.repeatInterleave(repeats, 0));
    }

    // ==================== Slice variants ====================

    @Override
    public IDiffVector slice(int start, int end, int step) {
        // Step-slice has no direct tensor op; gather over a stride-index tensor.
        // Index construction (not data computation) — structural loop, §7a-exempt.
        int len = 0;
        for (int pos = start; pos < end; pos += step) len++;
        double[] idx = new double[len];
        for (int i = 0, pos = start; pos < end; i++, pos += step) idx[i] = pos;
        return wrap((RereDiffTensor) tensor.gather(0, IDiffTensor.constantTensor(idx, len)));
    }

    @Override
    public IDiffVector slice(String sliceExpression) {
        String[] parts = sliceExpression.split(":");
        int start, end, step;
        if (parts.length == 3) {
            start = Integer.parseInt(parts[0]);
            end = Integer.parseInt(parts[1]);
            step = Integer.parseInt(parts[2]);
        } else if (parts.length == 2) {
            start = Integer.parseInt(parts[0]);
            end = Integer.parseInt(parts[1]);
            step = 1;
        } else {
            throw new IllegalArgumentException("Invalid slice expression: " + sliceExpression);
        }
        return slice(start, end, step);
    }

    // ==================== Fancy/boolean indexing ====================

    @Override
    public IDiffVector fancyGet(int[] positions) {
        double[] idx = new double[positions.length];
        for (int i = 0; i < positions.length; i++) idx[i] = positions[i];
        return wrap((RereDiffTensor) tensor.gather(0, IDiffTensor.constantTensor(idx, positions.length)));
    }

    @Override
    public IDiffVector booleanGet(boolean[] booleanIndex) {
        // Gather over the indices where the mask is true.
        int count = 0;
        for (boolean b : booleanIndex) if (b) count++;
        double[] idx = new double[count];
        int k = 0;
        for (int i = 0; i < booleanIndex.length; i++) if (booleanIndex[i]) idx[k++] = i;
        return wrap((RereDiffTensor) tensor.gather(0, IDiffTensor.constantTensor(idx, count)));
    }

    // ==================== Matrix operations ====================

    @Override
    public IDiffVector mmul(IMatrix<Double> other) {
        int rows = other.rows();
        int cols = other.cols();
        double[][] mat = other.toDoubleArray();
        double[] flat = new double[rows * cols];
        for (int r = 0; r < rows; r++) System.arraycopy(mat[r], 0, flat, r * cols, cols);
        // y = mat @ fwd, where mat is a non-diff constant [rows,cols] and fwd the
        // vector reshaped to [cols,1]. mmul routes grad to the reshaped vector
        // (matT.requiresGrad=false but vecCol.requiresGrad=true ⇒ diff path),
        // then reshape-back propagates to this tensor.
        IDiffTensor matT = IDiffTensor.constantTensor(flat, rows, cols);
        IDiffTensor vecCol = tensor.reshape(cols, 1);
        IDiffTensor y = matT.mmul(vecCol);
        return wrap((RereDiffTensor) y.reshape(rows));
    }

    @Override
    public IDiffVector dot(IMatrix<Double> m) {
        return mmul(m);
    }

    @Override
    public IDiffVector cross(IVector<Double> other) {
        double[] oData = other.toDoubleArray();
        IDiffTensor oConst = IDiffTensor.constantTensor(oData, oData.length);
        return wrap((RereDiffTensor) tensor.cross(oConst));
    }

    // ==================== Where ====================

    @Override
    public IDiffVector where(boolean[] condition, Double x, Double y) {
        // result[i] = condition[i] ? self[i] : fill, where fill = x (or y if x null).
        // tensor.where(cond, other) = cond ? self : other; routes grad to self where true.
        int n = (int) tensor.totalSize();
        double[] cond = new double[n];
        for (int i = 0; i < n; i++) cond[i] = condition[i] ? 1.0 : 0.0;
        double fill = (x != null) ? x : y;
        IDiffTensor condT = IDiffTensor.constantTensor(cond, n);
        IDiffTensor fillT = IDiffTensor.constantTensor(new double[]{fill}, 1);
        return wrap((RereDiffTensor) tensor.where(condT, fillT));
    }

    @Override
    public IDiffVector where(boolean[] condition, IVector<Double> x, IVector<Double> y) {
        if (x instanceof IDiffVector dx && y instanceof IDiffVector dy) {
            return where(condition, dx, dy);
        }
        throw new UnsupportedOperationException(
            "where(IVector) with non-differentiable vectors is not supported");
    }

    @Override
    public IDiffVector where(boolean[] condition, IDiffVector x, IDiffVector y) {
        // result = condition ? x : y (uses x/y values, not self). Build as
        // x.where(cond, y): routes grad to x where true, to y where false.
        RereDiffVector xv = resolveToRere(x);
        RereDiffVector yv = resolveToRere(y);
        int n = (int) xv.tensor.totalSize();
        double[] cond = new double[n];
        for (int i = 0; i < n; i++) cond[i] = condition[i] ? 1.0 : 0.0;
        IDiffTensor condT = IDiffTensor.constantTensor(cond, n);
        return wrap((RereDiffTensor) xv.tensor.where(condT, yv.tensor));
    }

    // ==================== Cumulative operations ====================

    @Override
    public IDiffVector cumsum() {
        return wrap((RereDiffTensor) tensor.cumsum(0));
    }

    @Override
    public IDiffVector cumprod() {
        return wrap((RereDiffTensor) tensor.cumprod(0));
    }

    @Override
    public IDiffVector diff() {
        // y[i] = x[i+1] - x[i] = slice(1,n) - slice(0,n-1); chain rule handles the
        // shared-dependence backward (dx[0]=-g0, dx[i]=g[i-1]-g[i], dx[n-1]=g[n-2]).
        int n = (int) tensor.totalSize();
        return wrap((RereDiffTensor) tensor.slice(0, 1, n).sub(tensor.slice(0, 0, n - 1)));
    }

    @Override
    public IDiffVector diff(int order) {
        if (order == 1) return diff();
        IDiffVector result = this;
        for (int i = 0; i < order; i++) result = result.diff();
        return result;
    }

    // ==================== Sort ====================

    @Override
    public IDiffVector sort() {
        // gather along the ascending argsort permutation; scatter-add backward
        // routes each output grad back to its source position.
        return wrap((RereDiffTensor) tensor.gather(0, tensor.argsort(0, false)));
    }

    // ==================== Concat (deprecated) ====================

    @Override
    @Deprecated
    public IDiffVector concat(IVector<Double> other) {
        double[] od = other.toDoubleArray();
        return wrap((RereDiffTensor) tensor.cat(0, IDiffTensor.constantTensor(od, od.length)));
    }

    // ==================== Statistics ====================

    @Override
    public IDiffVector min() {
        return wrap((RereDiffTensor) tensor.min(0, false));
    }

    @Override
    public IDiffVector max() {
        return wrap((RereDiffTensor) tensor.max(0, false));
    }

    @Override
    public IDiffVector prod() {
        return wrap((RereDiffTensor) tensor.prod(0, false));
    }

    @Override
    public IDiffVector norm2() {
        // ||x||_2 = sqrt(sum(x^2)); gradient via chain rule (square→sum→sqrt).
        return wrap((RereDiffTensor) tensor.square().sum().sqrt());
    }

    @Override
    public IDiffVector norm1() {
        // ||x||_1 = sum(|x|); gradient via chain rule (abs→sum).
        return wrap((RereDiffTensor) tensor.abs().sum());
    }

    @Override
    public IDiffVector ptp() {
        // range = max - min; sub routes +g to argmax and -g to argmin.
        return wrap((RereDiffTensor) tensor.max(0, false).sub(tensor.min(0, false)));
    }

    @Override
    public IDiffVector normalize() {
        return wrap((RereDiffTensor) tensor.normalize(2.0, 0));
    }

    @Override
    public IDiffVector std() {
        return wrap((RereDiffTensor) tensor.std(0, false));
    }

    @Override
    public IDiffVector std(int ddof) {
        // sqrt(var(ddof)); composition handles arbitrary ddof (mean-dependence term
        // cancels in the full Jacobian, matching the closed-form 2(x-mean)/(n-ddof)).
        return wrap((RereDiffTensor) varTensor(ddof).sqrt());
    }

    @Override
    public IDiffVector var() {
        return wrap((RereDiffTensor) tensor.var(0, false));
    }

    @Override
    public IDiffVector var(int ddof) {
        return wrap(varTensor(ddof));
    }

    /** var = sum((x - mean)^2) / (n - ddof), built from existing tensor ops. */
    private RereDiffTensor varTensor(int ddof) {
        int n = tensor.shape()[0];
        IDiffTensor mean = tensor.mean(0, true);
        return (RereDiffTensor) tensor.sub(mean).square().sum().div((double) (n - ddof));
    }

    // ==================== Non-differentiable ====================

    @Override
    public IDiffVector map(Function<Double, Double> fun) {
        throw new UnsupportedOperationException(
            "map() with arbitrary function cannot be differentiated. Use specific operations instead.");
    }

    @Override
    public IDoubleMatrix asColumnVector() {
        return getValue().asColumnVector();
    }

    @Override
    public IDoubleMatrix hessianMatrix() {
        return (IDoubleMatrix) getValue().hessianMatrix();
    }

    @Override public double cov(IVector<Double> other) { return getValue().cov(other); }
    @Override public double corr(IVector<Double> other) { return getValue().corr(other); }
}
