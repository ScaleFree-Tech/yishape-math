package com.yishape.lab.math.autodiff.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.RereDoubleVector;
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
        return tensor.gradData() != null ? IDoubleVector.of(tensor.gradData()) : null;
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
    public void backward(IDoubleVector initialGradient) {
        if (!tensor.requiresGrad()) return;
        tensor.setGradData(initialGradient.getData());
        tensor.backwardImpl();
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
        double[] dest = tensor.value().toDoubleArray();
        System.arraycopy(newData, 0, dest, 0, Math.min(newData.length, dest.length));
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

    static IDiffVector constant(double[] data) {
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
        RereDiffVector gr = (RereDiffVector) gamma;
        RereDiffVector br = (RereDiffVector) beta;
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
        RereDiffVector gr = (RereDiffVector) gamma;
        RereDiffVector br = (RereDiffVector) beta;
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
        RereDiffVector wv = (RereDiffVector) weight;
        RereDiffVector bv = (RereDiffVector) bias;
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
    public IDiffVector scaledDotProductAttention(IDiffVector key, IDiffVector value,
                                                   IDiffVector mask, int[] shape,
                                                   double dropout) {
        RereDiffVector kv = (RereDiffVector) key;
        RereDiffVector vv = (RereDiffVector) value;
        RereDiffVector mv = (RereDiffVector) mask;
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
     * Single-node fused mean: creates one graph node that computes mean directly
     * from the input tensor. Unlike the old sum()+div(n) two-node pattern, this
     * produces a single node whose input is the raw data — essential for GPU/HPC
     * backends which expect "mean" to receive the unreduced tensor, not a pre-summed scalar.
     *
     * <p>Fusion detection: if {@code tensor} wraps a pending unary op (relu, square, etc.),
     * we fuse the unary derivative into the backward and produce tags like "reluMean".
     * The fused node takes the ORIGINAL input (before the unary op) as its graph input,
     * bypassing the intermediate unary node — consistent with {@code RereDiffTensor.tryFuseSum()}.</p>
     *
     * <p>Important: both {@code backwardFn} (first-order Consumer) and
     * {@code symbolicBackwardFn} (second-order tape-of-tape) MUST be set.
     * Omitting symbolicBackwardFn causes {@code AD.grad()} to return zero gradients.</p>
     */
    @Override
    public IDiffVector mean() {
        RereDiffTensor t = tensor;
        int n = (int) t.value().totalSize();
        double[] tData = t.value().toDoubleArray();

        // Compute mean value in forward pass
        double sum = 0;
        for (int i = 0; i < n; i++) sum += tData[i];
        double meanVal = sum / n;

        // Detect pending fusable unary op for fused tags like "reluMean"
        String fusedTag = null;
        Consumer<RereDiffTensor> fusedBw = null;
        double[] symFactor = null;
        RereDiffTensor fusedInput = t;
        if (t.inputs() != null && t.inputs().size() == 1 && t.opTag() != null) {
            RereDiffTensor inp = t.inputs().get(0);
            if (inp != null && inp.requiresGrad()) {
                double[] xData = inp.value().toDoubleArray();
                int m = n;
                switch (t.opTag()) {
                    case "square" -> {
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) sf[i] = 2.0 * xData[i] / m;
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "squareMean";
                        symFactor = sf;
                    }
                    case "relu" -> {
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) sf[i] = xData[i] > 0 ? 1.0 / m : 0;
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "reluMean";
                        symFactor = sf;
                    }
                    case "exp" -> {
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) sf[i] = tData[i] / m;
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "expMean";
                        symFactor = sf;
                    }
                    case "abs" -> {
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) sf[i] = (xData[i] >= 0 ? 1.0 : -1.0) / m;
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "absMean";
                        symFactor = sf;
                    }
                    case "log" -> {
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) sf[i] = 1.0 / m / xData[i];
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "logMean";
                        symFactor = sf;
                    }
                    case "sigmoid" -> {
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) { double s = tData[i]; sf[i] = s * (1.0 - s) / m; }
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "sigmoidMean";
                        symFactor = sf;
                    }
                    case "tanh" -> {
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) { double th = tData[i]; sf[i] = (1.0 - th * th) / m; }
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "tanhMean";
                        symFactor = sf;
                    }
                    case "silu" -> {
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) {
                            double xi = xData[i];
                            double sig = 1.0 / (1.0 + Math.exp(-xi));
                            sf[i] = (sig + xi * sig * (1.0 - sig)) / m;
                        }
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "siluMean";
                        symFactor = sf;
                    }
                    case "pow" -> {
                        double scalarP = t.scalarParam();
                        if (Double.isNaN(scalarP)) break;
                        double[] buf = AutodiffBufferPool.acquire(m);
                        double[] sf = new double[m];
                        for (int i = 0; i < m; i++) sf[i] = scalarP * Math.pow(xData[i], scalarP - 1) / m;
                        fusedBw = self -> {
                            double g = self.gradData()[0];
                            for (int i = 0; i < m; i++) buf[i] = g * sf[i];
                            inp.accGradFromPooled(buf, m);
                        };
                        fusedInput = inp;
                        fusedTag = "powMean";
                        symFactor = sf;
                    }
                }
            }
        }

        if (fusedTag != null && fusedBw != null && symFactor != null) {
            RereDiffTensor rt = new RereDiffTensor(new double[]{meanVal}, new int[]{1},
                List.of(fusedInput), fusedBw, fusedTag);
            rt.setExportShape(fusedInput.shape());
            if ("powMean".equals(fusedTag)) {
                rt.setScalarParam(t.scalarParam());
            }
            double[] sfCopy = symFactor;
            int[] shapeCopy = fusedInput.shape().clone();
            // tape-of-tape: g is a scalar tensor, multiply by factor to broadcast to input shape.
            // Without this, AD.grad() returns zero for paths through fused mean nodes.
            rt.setSymbolicBackwardFn(g -> new IDiffTensor[]{
                g.mul(IDiffTensor.constantTensor(sfCopy, shapeCopy))
            });
            return wrap(rt);
        }

        // Simple mean: single fused node avoiding the old sum()+div(n) two-node pattern.
        // GPU "mean" op expects raw input data, not a pre-summed scalar.
        // Uses Arrays.fill() for broadcast (SIMD-friendly via JDK intrinsic, matches sum() pattern).
        double gDivN = 1.0 / n;
        double[] gradBuf = AutodiffBufferPool.acquire(n);
        Consumer<RereDiffTensor> bw = self -> {
            double g = self.gradData()[0];
            double gVal = g * gDivN;
            Arrays.fill(gradBuf, 0, n, gVal);
            t.accGradFromPooled(gradBuf, n);
        };

        RereDiffTensor rt = new RereDiffTensor(new double[]{meanVal}, new int[]{1},
            List.of(t), bw, "mean");
        // exportShape = input shape so GPU/HPC backends know the reduction dimension.
        rt.setExportShape(t.shape());
        // tape-of-tape symbolic backward: g * (ones/n) broadcast → required for AD.grad().
        double[] meanSymFactor = new double[n];
        Arrays.fill(meanSymFactor, gDivN);
        int[] meanShapeCopy = t.shape().clone();
        rt.setSymbolicBackwardFn(g -> new IDiffTensor[]{
            g.mul(IDiffTensor.constantTensor(meanSymFactor, meanShapeCopy))
        });
        return wrap(rt);
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
        // Fused "dot" node: mul + sum in one kernel on GPU, 2 inputs → 2 output grads.
        RereDiffTensor a = tensor;
        RereDiffTensor b = o.tensor;
        int m = (int) a.value().totalSize();
        double[] aData = a.value().toDoubleArray();
        double[] bData = b.value().toDoubleArray();
        double total = 0;
        for (int i = 0; i < m; i++) total += aData[i] * bData[i];

        double[] bCopy = bData.clone();
        double[] aCopy = aData.clone();
        Consumer<RereDiffTensor> bw = self -> {
            double g = self.gradData()[0];
            double[] daBuf = AutodiffBufferPool.acquire(m);
            double[] dbBuf = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < m; i++) {
                daBuf[i] = g * bCopy[i];  // d/da = b * grad
                dbBuf[i] = g * aCopy[i];  // d/db = a * grad
            }
            a.accGradFromPooled(daBuf, m);
            b.accGradFromPooled(dbBuf, m);
        };
        RereDiffTensor rt = new RereDiffTensor(new double[]{total}, new int[]{1},
            List.of(a, b), bw, "dot");
        rt.setExportShape(a.shape());
        // tape-of-tape: d/da = g * b, d/db = g * a.
        // MUST use actual tensor references (a,b) — not constantTensor copies.
        // constantTensor creates dead-end nodes that break the gradient chain
        // for MixedMode.hvp() (Hessian-vector product via tape-of-tape AD).
        RereDiffTensor aRef = a;
        RereDiffTensor bRef = b;
        rt.setSymbolicBackwardFn(g -> new IDiffTensor[]{
            g.mul(bRef),
            g.mul(aRef)
        });
        return wrap(rt);
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
        IDiffTensor[] otherTensors = new IDiffTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            otherTensors[i] = ((RereDiffVector) others[i]).tensor;
        }
        return wrap((RereDiffTensor) tensor.cat(0, otherTensors));
    }

    // ==================== In-place operations ====================

    @Override
    public IDiffVector addInPlace(IDiffVector other) {
        return this.add(other);
    }

    @Override
    public IDiffVector mulInPlace(double scalar) {
        if (!tensor.isLeaf()) {
            throw new IllegalStateException("mulInPlace only allowed on leaf variables");
        }
        double[] data = tensor.value().toDoubleArray();
        for (int i = 0; i < data.length; i++) {
            data[i] *= scalar;
        }
        tensor.setGradData(null);
        return this;
    }

    // ==================== Copy ====================

    @Override
    public IDiffVector copy() {
        return wrap((RereDiffTensor) tensor.clone());
    }

    // ==================== IDiffMatrix reshape ====================

    @Override
    public IDiffMatrix reshape(int rows, int cols) {
        int origSize = (int) tensor.value().totalSize();
        if (rows * cols != origSize) {
            throw new IllegalArgumentException(
                "reshape dimensions " + rows + "x" + cols + " must match size " + origSize);
        }
        IDoubleMatrix resultVal = IDoubleMatrix.fromArray(tensor.value().toDoubleArray(), rows, cols);
        RereDiffTensor self = this.tensor;
        Consumer<IDoubleMatrix> backwardFn = (matrixGrad) -> {
            double[] flatGrad = ((IDoubleVector) matrixGrad.flatten()).getData();
            self.accGrad(flatGrad);
            self.propagateGrad(); // continue gradient propagation through tensor subgraph
        };
        Function<IDiffVector, IDiffVector[]> symbolicBackwardFn = (matrixGrad) ->
            new IDiffVector[] { matrixGrad };
        RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(), backwardFn);
        node.opTag = "reshape";
        node.symbolicBackwardFn = symbolicBackwardFn;
        return node;
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
        double[] data = tensor.value().toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] += p;
        return this;
    }

    @Override
    public IDiffVector subScalarInPlace(double p) {
        if (p == 0.0) return this;
        double[] data = tensor.value().toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] -= p;
        return this;
    }

    @Override
    public IDiffVector multiplyByScalarInPlace(double p) {
        double[] data = tensor.value().toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] *= p;
        return this;
    }

    @Override
    public IDiffVector addInPlace(IVector<Double> vec) {
        double[] data = tensor.value().toDoubleArray();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] += other[i];
        return this;
    }

    @Override
    public IDiffVector subInPlace(IVector<Double> vec) {
        double[] data = tensor.value().toDoubleArray();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] -= other[i];
        return this;
    }

    @Override
    public IDiffVector multiplyInPlace(IVector<Double> vec) {
        double[] data = tensor.value().toDoubleArray();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] *= other[i];
        return this;
    }

    @Override
    public IDiffVector negInPlace() {
        double[] data = tensor.value().toDoubleArray();
        for (int i = 0; i < data.length; i++) data[i] = -data[i];
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

    @Override public IDiffVector round() { return zeroGradUnaryOp(v -> Math.round(v), "round"); }
    @Override public IDiffVector floor() { return zeroGradUnaryOp(Math::floor, "floor"); }
    @Override public IDiffVector ceil() { return zeroGradUnaryOp(Math::ceil, "ceil"); }
    @Override public IDiffVector trunc() { return zeroGradUnaryOp(v -> (double)(long)v, "trunc"); }
    @Override public IDiffVector sign() { return zeroGradUnaryOp(Math::signum, "sign"); }

    /** Build a unary graph node whose gradient is zero everywhere (flat function). */
    private IDiffVector zeroGradUnaryOp(java.util.function.DoubleUnaryOperator forward, String tag) {
        double[] xd = tensor.value().toDoubleArray();
        int n = xd.length;
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = forward.applyAsDouble(xd[i]);
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs().get(0);
            input.accGrad(new double[n]); // gradient is zero everywhere
        };
        RereDiffTensor node = new RereDiffTensor(out, new int[]{n}, List.of(tensor), bw, tag);
        return wrap(node);
    }

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
        // arcsin'(x) = 1/sqrt(1-x^2)
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[fwd.length];
        for (int i = 0; i < fwd.length; i++) {
            double v = fwd[i];
            if (v < -1.0 || v > 1.0) throw new ArithmeticException("arcsin domain: |x| <= 1, got " + v);
            y[i] = Math.asin(v);
        }
        // Create a fused node: forward is arcsin, backward is 1/sqrt(1-x^2) * gradOut
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[fwd.length];
            for (int i = 0; i < fwd.length; i++) {
                dx[i] = gd[i] / Math.sqrt(1.0 - fwd[i] * fwd[i]);
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "arcsin"));
    }

    @Override
    public IDiffVector arccos() {
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[fwd.length];
        for (int i = 0; i < fwd.length; i++) {
            double v = fwd[i];
            if (v < -1.0 || v > 1.0) throw new ArithmeticException("arccos domain: |x| <= 1, got " + v);
            y[i] = Math.acos(v);
        }
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[fwd.length];
            for (int i = 0; i < fwd.length; i++) {
                dx[i] = -gd[i] / Math.sqrt(1.0 - fwd[i] * fwd[i]);
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "arccos"));
    }

    @Override
    public IDiffVector arctan() {
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[fwd.length];
        for (int i = 0; i < fwd.length; i++) y[i] = Math.atan(fwd[i]);
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[fwd.length];
            for (int i = 0; i < fwd.length; i++) {
                dx[i] = gd[i] / (1.0 + fwd[i] * fwd[i]);
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "arctan"));
    }

    @Override
    public IDiffVector sinh() {
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[fwd.length];
        for (int i = 0; i < fwd.length; i++) y[i] = Math.sinh(fwd[i]);
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[fwd.length];
            for (int i = 0; i < fwd.length; i++) dx[i] = gd[i] * Math.cosh(fwd[i]);
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "sinh"));
    }

    @Override
    public IDiffVector cosh() {
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[fwd.length];
        for (int i = 0; i < fwd.length; i++) y[i] = Math.cosh(fwd[i]);
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[fwd.length];
            for (int i = 0; i < fwd.length; i++) dx[i] = gd[i] * Math.sinh(fwd[i]);
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "cosh"));
    }

    @Override
    public IDiffVector remainder(Double value) {
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[fwd.length];
        for (int i = 0; i < fwd.length; i++) y[i] = fwd[i] % value;
        // straight-through estimator for remainder
        RereDiffTensor result = new RereDiffTensor(y, new int[]{y.length});
        return wrap(result);
    }

    // ==================== Reverse ====================

    @Override
    public IDiffVector reverse() {
        int n = (int) tensor.value().totalSize();
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[n];
        for (int i = 0; i < n; i++) y[n - 1 - i] = fwd[i];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) dx[n - 1 - i] = go[i];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{n}, List.of(self), backwardFn, "reverse"));
    }

    // ==================== Tile / repeat ====================

    @Override
    public IDiffVector tile(int reps) {
        int n = (int) tensor.value().totalSize();
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[n * reps];
        for (int i = 0; i < y.length; i++) y[i] = fwd[i % n];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[n];
            for (int i = 0; i < go.length; i++) dx[i % n] += go[i];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "tile"));
    }

    @Override
    public IDiffVector repeat(int repeats) {
        int n = (int) tensor.value().totalSize();
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[n * repeats];
        for (int i = 0; i < y.length; i++) y[i] = fwd[i / repeats];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[n];
            for (int i = 0; i < go.length; i++) dx[i / repeats] += go[i];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "repeat"));
    }

    // ==================== Slice variants ====================

    @Override
    public IDiffVector slice(int start, int end, int step) {
        double[] fwd = tensor.value().toDoubleArray();
        int fullLen = fwd.length;
        int lenTemp = 0;
        for (int pos = start; pos < end; pos += step) lenTemp++;
        final int len = lenTemp;
        double[] y = new double[len];
        for (int i = 0, pos = start; pos < end && i < len; i++, pos += step) y[i] = fwd[pos];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[fullLen];
            for (int i = 0, pos = start; pos < end && i < len; i++, pos += step) dx[pos] = gd[i];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{len}, List.of(self), backwardFn, "slice"));
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
        double[] fwd = tensor.value().toDoubleArray();
        int selfLen = fwd.length;
        double[] y = new double[positions.length];
        for (int i = 0; i < positions.length; i++) y[i] = fwd[positions[i]];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[selfLen];
            for (int i = 0; i < positions.length; i++) dx[positions[i]] += gd[i];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "fancyGet"));
    }

    @Override
    public IDiffVector booleanGet(boolean[] booleanIndex) {
        double[] fwd = tensor.value().toDoubleArray();
        int selfLen = fwd.length;
        int count = 0;
        for (boolean b : booleanIndex) if (b) count++;
        double[] y = new double[count];
        int outIdx = 0;
        for (int i = 0; i < selfLen; i++) if (booleanIndex[i]) y[outIdx++] = fwd[i];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[selfLen];
            int oi = 0;
            for (int i = 0; i < selfLen; i++) if (booleanIndex[i]) dx[i] += gd[oi++];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{count}, List.of(self), backwardFn, "booleanGet"));
    }

    // ==================== Matrix operations ====================

    @Override
    public IDiffVector mmul(IMatrix<Double> other) {
        double[] fwd = tensor.value().toDoubleArray();
        int rows = other.rows();
        int cols = other.cols();
        double[][] mat = other.toDoubleArray();
        double[] y = new double[rows];
        for (int r = 0; r < rows; r++) {
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += fwd[c] * mat[r][c];
            y[r] = sum;
        }
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[cols];
            for (int c = 0; c < cols; c++) {
                double sum = 0;
                for (int r = 0; r < rows; r++) sum += go[r] * mat[r][c];
                dx[c] = sum;
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{rows}, List.of(self), backwardFn, "mmul"));
    }

    @Override
    public IDiffVector dot(IMatrix<Double> m) {
        return mmul(m);
    }

    @Override
    public IDiffVector cross(IVector<Double> other) {
        double[] fwd = tensor.value().toDoubleArray();
        double[] oData = other.toDoubleArray();
        double[] y = new double[]{
            fwd[1] * oData[2] - fwd[2] * oData[1],
            fwd[2] * oData[0] - fwd[0] * oData[2],
            fwd[0] * oData[1] - fwd[1] * oData[0]
        };
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[]{
                go[1] * oData[2] - go[2] * oData[1],
                go[2] * oData[0] - go[0] * oData[2],
                go[0] * oData[1] - go[1] * oData[0]
            };
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{3}, List.of(self), backwardFn, "cross"));
    }

    // ==================== Where ====================

    @Override
    public IDiffVector where(boolean[] condition, Double x, Double y) {
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) result[i] = condition[i] ? fwd[i] : (x != null ? x : y);
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) if (condition[i]) dx[i] += gd[i];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(result, new int[]{n}, List.of(self), backwardFn, "where"));
    }

    @Override
    public IDiffVector where(boolean[] condition, IVector<Double> x, IVector<Double> y) {
        double[] fwd = tensor.value().toDoubleArray();
        double[] xd = x.toDoubleArray();
        double[] yd = y.toDoubleArray();
        int n = fwd.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) result[i] = condition[i] ? xd[i] : yd[i];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) if (condition[i]) dx[i] += gd[i];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(result, new int[]{n}, List.of(self), backwardFn, "where"));
    }

    // ==================== Cumulative operations ====================

    @Override
    public IDiffVector cumsum() {
        int n = (int) tensor.value().totalSize();
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[n];
        y[0] = fwd[0];
        for (int i = 1; i < n; i++) y[i] = y[i - 1] + fwd[i];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[n];
            double running = 0;
            for (int i = n - 1; i >= 0; i--) {
                running += go[i];
                dx[i] = running;
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{n}, List.of(self), backwardFn, "cumsum"));
    }

    @Override
    public IDiffVector cumprod() {
        int n = (int) tensor.value().totalSize();
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[n];
        y[0] = fwd[0];
        for (int i = 1; i < n; i++) y[i] = y[i - 1] * fwd[i];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[n];
            for (int j = 0; j < n; j++) {
                for (int k = j; k < n; k++) {
                    if (Math.abs(fwd[j]) > 1e-15) {
                        dx[j] += go[k] * y[k] / fwd[j];
                    }
                }
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{n}, List.of(self), backwardFn, "cumprod"));
    }

    @Override
    public IDiffVector diff() {
        int n = (int) tensor.value().totalSize();
        double[] fwd = tensor.value().toDoubleArray();
        double[] y = new double[n - 1];
        for (int i = 0; i < n - 1; i++) y[i] = fwd[i + 1] - fwd[i];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[n];
            dx[0] = -go[0];
            for (int i = 1; i < n - 1; i++) dx[i] = go[i - 1] - go[i];
            dx[n - 1] = go[n - 2];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "diff"));
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
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        Arrays.sort(indices, (a, b) -> Double.compare(fwd[a], fwd[b]));
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = indices[i];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) y[i] = fwd[perm[i]];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) dx[perm[i]] = go[i];
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{n}, List.of(self), backwardFn, "sort"));
    }

    // ==================== Concat (deprecated) ====================

    @Override
    @Deprecated
    public IDiffVector concat(IVector<Double> other) {
        double[] fwd = tensor.value().toDoubleArray();
        double[] od = other.toDoubleArray();
        int selfLen = fwd.length;
        double[] y = new double[selfLen + od.length];
        System.arraycopy(fwd, 0, y, 0, selfLen);
        System.arraycopy(od, 0, y, selfLen, od.length);
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] gd = gradOut.gradData();
            double[] dx = new double[selfLen];
            System.arraycopy(gd, 0, dx, 0, selfLen);
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{y.length}, List.of(self), backwardFn, "concat"));
    }

    // ==================== Statistics ====================

    @Override
    public IDiffVector min() {
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        int minIdx = 0;
        for (int i = 1; i < n; i++) if (fwd[i] < fwd[minIdx]) minIdx = i;
        final int mi = minIdx;
        double minVal = fwd[mi];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double g = gradOut.gradData()[0];
            double[] dx = new double[n];
            dx[mi] = g;
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(new double[]{minVal}, new int[]{1}, List.of(self), backwardFn, "min"));
    }

    @Override
    public IDiffVector max() {
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        int maxIdx = 0;
        for (int i = 1; i < n; i++) if (fwd[i] > fwd[maxIdx]) maxIdx = i;
        final int mi = maxIdx;
        double maxVal = fwd[mi];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double g = gradOut.gradData()[0];
            double[] dx = new double[n];
            dx[mi] = g;
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(new double[]{maxVal}, new int[]{1}, List.of(self), backwardFn, "max"));
    }

    @Override
    public IDiffVector prod() {
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        double totalProd = 1.0;
        for (int i = 0; i < n; i++) totalProd *= fwd[i];
        final double tp = totalProd;
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double g = gradOut.gradData()[0];
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) {
                if (Math.abs(fwd[i]) > 1e-15) dx[i] = g * tp / fwd[i];
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(new double[]{tp}, new int[]{1}, List.of(self), backwardFn, "prod"));
    }

    @Override
    public IDiffVector norm2() {
        double[] fwd = tensor.value().toDoubleArray();
        double normSq = 0;
        for (double v : fwd) normSq += v * v;
        double norm = Math.sqrt(normSq);
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double g = gradOut.gradData()[0];
            double[] dx = new double[fwd.length];
            if (norm > 1e-15) {
                for (int i = 0; i < fwd.length; i++) dx[i] = g * fwd[i] / norm;
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(new double[]{norm}, new int[]{1}, List.of(self), backwardFn, "norm2"));
    }

    @Override
    public IDiffVector norm1() {
        double[] fwd = tensor.value().toDoubleArray();
        double norm = 0;
        for (double v : fwd) norm += Math.abs(v);
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double g = gradOut.gradData()[0];
            double[] dx = new double[fwd.length];
            for (int i = 0; i < fwd.length; i++) dx[i] = g * Math.signum(fwd[i]);
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(new double[]{norm}, new int[]{1}, List.of(self), backwardFn, "norm1"));
    }

    @Override
    public IDiffVector ptp() {
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        int maxIdx = 0, minIdx = 0;
        for (int i = 1; i < n; i++) {
            if (fwd[i] > fwd[maxIdx]) maxIdx = i;
            if (fwd[i] < fwd[minIdx]) minIdx = i;
        }
        final int mi = maxIdx;
        final int ni = minIdx;
        double range = fwd[mi] - fwd[ni];
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double g = gradOut.gradData()[0];
            double[] dx = new double[n];
            dx[mi] = g;
            dx[ni] = -g;
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(new double[]{range}, new int[]{1}, List.of(self), backwardFn, "ptp"));
    }

    @Override
    public IDiffVector normalize() {
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        double normSq = 0;
        for (double v : fwd) normSq += v * v;
        double norm = Math.sqrt(normSq);
        double invNorm = norm > 1e-15 ? 1.0 / norm : 0;
        double invNorm3 = invNorm * invNorm * invNorm;
        double[] y = new double[n];
        for (int i = 0; i < n; i++) y[i] = fwd[i] * invNorm;
        RereDiffTensor self = this.tensor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double[] go = gradOut.gradData();
            double[] dx = new double[n];
            double dot = 0;
            for (int i = 0; i < n; i++) dot += go[i] * fwd[i];
            for (int i = 0; i < n; i++) dx[i] = go[i] * invNorm - dot * fwd[i] * invNorm3;
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(y, new int[]{n}, List.of(self), backwardFn, "normalize"));
    }

    @Override
    public IDiffVector std() { return std(0); }

    @Override
    public IDiffVector std(int ddof) {
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        double mean = 0;
        for (double v : fwd) mean += v;
        mean /= n;
        double var = 0;
        for (double v : fwd) var += (v - mean) * (v - mean);
        double divisor = n - ddof;
        double stdev = Math.sqrt(var / divisor);
        RereDiffTensor self = this.tensor;
        double m = mean, s = stdev, d = divisor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double g = gradOut.gradData()[0];
            double[] dx = new double[n];
            if (s > 1e-15) {
                for (int i = 0; i < n; i++) dx[i] = g * (fwd[i] - m) / (d * s);
            }
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(new double[]{stdev}, new int[]{1}, List.of(self), backwardFn, "std"));
    }

    @Override
    public IDiffVector var() { return var(0); }

    @Override
    public IDiffVector var(int ddof) {
        double[] fwd = tensor.value().toDoubleArray();
        int n = fwd.length;
        double mean = 0;
        for (double v : fwd) mean += v;
        mean /= n;
        double var = 0;
        for (double v : fwd) var += (v - mean) * (v - mean);
        double divisor = n - ddof;
        double variance = var / divisor;
        RereDiffTensor self = this.tensor;
        double m = mean, d = divisor;
        Consumer<RereDiffTensor> backwardFn = (gradOut) -> {
            double g = gradOut.gradData()[0];
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) dx[i] = 2.0 * g * (fwd[i] - m) / d;
            self.accGrad(dx);
        };
        return wrap(new RereDiffTensor(new double[]{variance}, new int[]{1}, List.of(self), backwardFn, "var"));
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
