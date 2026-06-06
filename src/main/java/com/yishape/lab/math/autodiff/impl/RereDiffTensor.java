package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.DoubleBinaryOperator;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.AutodiffBufferPool;
import com.yishape.lab.math.compute.DoubleFlatGemm;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.tensor.EinsumParser;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.linalg.tensor.TensorShape;

/**
 * Reverse-mode differentiable tensor with unified storage (PyTorch-style).
 *
 * <p>{@code RereDiffTensor} owns the computation graph directly. The underlying
 * {@link RereDoubleTensor} provides stride-aware storage, shape, and views.
 * Element-wise operations iterate via {@code linearGet/linearSet} for correct
 * access on non-contiguous views. View operations share the underlying data
 * array and use scatter-backward to map view positions back to parent positions.
 *
 * <p>Gradient storage ({@code grad}) is a flat {@code double[]} of length
 * {@code value.totalSize()} — the number of logical elements. This differs from
 * {@code value.getStorageData().length} for views whose storage is larger than
 * their logical shape.
 */
public class RereDiffTensor implements IDiffTensor {

    // ==================== Fields ====================

    public RereDoubleTensor value;
    public double[] grad;
    public List<RereDiffTensor> inputs;
    public Consumer<RereDiffTensor> backwardFn;
    public String opTag;
    public boolean requiresGrad = true;
    public boolean isLeaf;
    public double scalarParam = Double.NaN;
    public double scalarParam2 = Double.NaN;

    /**
     * Override shape in JSON export. When non-null, used instead of the tensor's own shape
     * for GPU/HPC graph export (e.g. fused pattern nodes where the logical shape differs).
     */
    public int[] exportShape;

    /**
     * Auxiliary backward data exported to GPU/HPC backends (e.g. MaxPool2d argmax indices).
     * When non-null, included in the binary/JSON graph serialization.
     */
    public int[] backwardIndices;

    /**
     * Symbolic backward function for higher-order AD.
     * Takes the output gradient (IDiffTensor) and returns gradients for each input.
     * When non-null, enables {@code AD.grad(output, inputs)} to build a new
     * computation graph whose nodes are themselves differentiable.
     */
    public java.util.function.Function<IDiffTensor, IDiffTensor[]> symbolicBackwardFn;

    /**
     * Link back to the vector-graph node that produced this tensor.
     * Set by {@link IDiffTensor#fromDiffVector} when wrapping a RereDiffVector.
     * Used during backward to bridge gradients from the tensor graph back into
     * the vector graph (e.g. Conv2d → ReLU → Linear parameter chain).
     *
     * @deprecated Tensor-native graph nodes (via CustomOp.tensorApply) don't need
     *             the dual-graph bridge. Once all layers are migrated, this field
     *             and {@link #triggerVectorBackward()} will be removed.
     */
    @Deprecated
    public com.yishape.lab.math.autodiff.impl.RereDiffVector vectorSource;

    // ==================== ThreadLocal ====================

    private static final ThreadLocal<ArrayList<RereDiffTensor>> TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffTensor>> TOPO_SET =
        ThreadLocal.withInitial(HashSet::new);

    // ==================== Constructors ====================

    /** Leaf node from raw data and shape. Data is NOT copied — caller should not mutate it. */
    public RereDiffTensor(double[] data, int... shape) {
        this.value = new RereDoubleTensor(data, shape);
        this.isLeaf = true;
        this.inputs = new ArrayList<>();
    }

    /** Leaf node from an existing tensor (shares storage, no copy). */
    public RereDiffTensor(RereDoubleTensor tensor) {
        this.value = tensor;
        this.isLeaf = true;
        this.inputs = new ArrayList<>();
    }

    /** Intermediate computation node. */
    public RereDiffTensor(double[] data, int[] shape, List<RereDiffTensor> inputs,
                          Consumer<RereDiffTensor> backwardFn, String opTag) {
        this.value = new RereDoubleTensor(data, shape);
        this.isLeaf = false;
        this.inputs = inputs;
        this.backwardFn = backwardFn;
        this.opTag = opTag;
    }

    /** Intermediate node with scalar param. */
    RereDiffTensor(double[] data, int[] shape, List<RereDiffTensor> inputs,
                   Consumer<RereDiffTensor> backwardFn, String opTag, double scalarParam) {
        this(data, shape, inputs, backwardFn, opTag);
        this.scalarParam = scalarParam;
    }

    /** Intermediate node with two scalar params. */
    RereDiffTensor(double[] data, int[] shape, List<RereDiffTensor> inputs,
                   Consumer<RereDiffTensor> backwardFn, String opTag,
                   double scalarParam, double scalarParam2) {
        this(data, shape, inputs, backwardFn, opTag);
        this.scalarParam = scalarParam;
        this.scalarParam2 = scalarParam2;
    }

    /** View node — value is a pre-built RereDoubleTensor view sharing parent storage. */
    RereDiffTensor(RereDoubleTensor value, List<RereDiffTensor> inputs,
                   Consumer<RereDiffTensor> backwardFn, String opTag) {
        this.value = value;
        this.isLeaf = false;
        this.inputs = inputs;
        this.backwardFn = backwardFn;
        this.opTag = opTag;
    }

    // ==================== Shape delegation ====================

    @Override public int rank() { return value.rank(); }
    @Override public int[] shape() { return value.shape(); }
    @Override public int dim(int axis) { return value.dim(axis); }
    @Override public long totalSize() { return value.totalSize(); }
    @Override public int[] strides() { return value.strides(); }
    @Override public int stride(int axis) { return value.stride(axis); }
    @Override public int offset() { return value.offset(); }
    @Override public boolean isContiguous() { return value.isContiguous(); }
    @Override public double item() { return value.item(); }
    @Override public double get(int... indices) { return value.get(indices); }
    @Override public ITensor set(double val, int... indices) { value.set(val, indices); return this; }
    @Override public ITensor fill(double val) { value.fill(val); return this; }
    @Override public IDoubleTensor copy() { return new RereDoubleTensor(value.toDoubleArray(), shape()); }

    // ==================== Gradient infrastructure ====================

    @Override
    public void backward() {
        if (!requiresGrad) return;
        int n = (int) value.totalSize();
        double[] ones = new double[n];
        Arrays.fill(ones, 1.0);
        this.grad = ones;
        backwardImpl();
    }

    @Override
    public void backward(IDoubleTensor gradient) {
        if (!requiresGrad) return;
        this.grad = gradient.toDoubleArray();
        backwardImpl();
    }

    private void backwardImpl() {
        ArrayList<RereDiffTensor> order = TOPO_LIST.get();
        order.clear();
        HashSet<RereDiffTensor> visited = TOPO_SET.get();
        visited.clear();
        buildTopo(order, visited);
        try {
            // Zero intermediate gradients before accumulation
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffTensor v = order.get(i);
                if (v != this) v.grad = null;
            }
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffTensor v = order.get(i);
                if (v.grad != null && v.backwardFn != null) {
                    v.backwardFn.accept(v);
                }
            }
            // Bridge gradients from leaf tensors into the vector graph.
            // Leaf tensors with vectorSource were produced by CustomOp layers
            // (Linear, Conv2d, etc.) — their gradient must flow back through
            // the vector graph to reach parameter leaves.
            for (int i = 0; i < order.size(); i++) {
                RereDiffTensor v = order.get(i);
                if (v.vectorSource != null && v.grad != null) {
                    v.triggerVectorBackward();
                }
            }
        } finally {
            order.clear();
            visited.clear();
        }
    }

    /**
     * Propagate gradient backward through this tensor's sub-graph without zeroing
     * this tensor's own gradient. Uses local collections (not ThreadLocal) so it
     * can be safely called from within RereDiffVector.backward().
     *
     * <p>Called by {@link CustomOp}'s backwardFn to bridge gradients from the
     * vector graph into the tensor graph, enabling gradient flow through layers
     * that use CustomOp (Conv2d, Linear, MaxPool2d, etc.).</p>
     *
     * @deprecated Tensor-native CustomOp.tensorApply() creates pure tensor graph nodes
     *             that propagate gradients through the tensor graph directly — no bridge needed.
     */
    @Deprecated
    public void propagateGrad() {
        if (this.grad == null || !requiresGrad) return;

        ArrayList<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        buildTopo(order, visited);

        // Zero intermediate gradients (skip root — its grad was set by the caller)
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffTensor v = order.get(i);
            if (v != this) v.grad = null;
        }
        // Propagate backward through sub-graph INCLUDING the root.
        // The root's backwardFn must be called to flow gradient to its inputs
        // (e.g. flattenOutTensor.backwardFn → reluOutTensor → convOutTensor).
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffTensor v = order.get(i);
            if (v.grad != null && v.backwardFn != null) {
                v.backwardFn.accept(v);
            }
        }
        // Bridge gradients from leaf tensors into the vector graph.
        // Leaf tensors with vectorSource were produced by CustomOp layers
        // (Conv2d, Linear, etc.) — their gradient must flow back through
        // the vector graph to reach parameter leaves.
        for (int i = 0; i < order.size(); i++) {
            RereDiffTensor v = order.get(i);
            if (v.vectorSource != null && v.grad != null) {
                v.triggerVectorBackward();
            }
        }
    }

    /**
     * Bridge gradient from this tensor into the vector graph via vectorSource.
     * Called during tensor backward propagation when a leaf tensor was produced
     * by a CustomOp (e.g. Linear, Conv2d output wrapped via IDiffTensor.fromDiffVector)
     * or by {@link com.yishape.lab.dl.nn.Parameter#asLeaf() Parameter.asLeaf()}.
     *
     * <p>For leaf vectors (no backwardFn chain), accumulates gradient directly
     * via {@code accGrad()} so {@code syncGradient()} can read it. For non-leaf
     * vectors (CustomOp outputs), traverses the vector backward chain via
     * {@code backwardNested()} and then restores the root gradient.</p>
     * @deprecated Not needed with tensor-native graph nodes (use Parameter.asLeafTensor()).
     */
    @Deprecated
    void triggerVectorBackward() {
        if (vectorSource != null && grad != null) {
            if (vectorSource.isLeaf()) {
                // Raw leaf (e.g. Parameter.asLeafVector()): just accumulate gradient.
                // backwardNested would set and then clear it, losing the gradient.
                vectorSource.accGrad(IDoubleVector.of(grad));
            } else {
                // Non-leaf (CustomOp output): traverse vector backward chain.
                // backwardNested sets grad, runs backwardFn chain, then clears root.
                // Restore root gradient afterward so syncGradient can read it.
                vectorSource.backwardNested(IDoubleVector.of(grad));
                vectorSource.accGrad(IDoubleVector.of(grad));
            }
        }
    }

    /** Iterative DFS post-order topological sort. */
    public void buildTopo(List<RereDiffTensor> order, HashSet<RereDiffTensor> visited) {
        java.util.ArrayDeque<Object[]> stack = new java.util.ArrayDeque<>();
        stack.push(new Object[]{this, Boolean.TRUE});
        while (!stack.isEmpty()) {
            Object[] entry = stack.peek();
            RereDiffTensor node = (RereDiffTensor) entry[0];
            boolean childrenNotPushed = (Boolean) entry[1];
            if (childrenNotPushed) {
                entry[1] = Boolean.FALSE;
                if (!visited.add(node)) { stack.pop(); continue; }
                for (int i = node.inputs.size() - 1; i >= 0; i--) {
                    RereDiffTensor inp = node.inputs.get(i);
                    if (!visited.contains(inp)) stack.push(new Object[]{inp, Boolean.TRUE});
                }
            } else {
                stack.pop();
                order.add(node);
            }
        }
    }

    @Override
    public void zeroGradient() { this.grad = null; }

    @Override
    public IDiffVector flattenGrad() {
        if (grad == null) return null;
        return new TensorBackedDiffVector(this, true);
    }

    @Override
    public IDiffVector flattenValue() {
        return new TensorBackedDiffVector(this, false);
    }

    @Override
    public IDoubleTensor detach() { return new RereDoubleTensor(value.toDoubleArray(), shape()); }

    @Override
    public IDiffTensor clone() {
        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(value.toDoubleArray(), shape()));
        RereDiffTensor cloned = new RereDiffTensor(value.toDoubleArray(), shape());
        cloned.requiresGrad = true;
        return cloned;
    }

    @Override
    public boolean requiresGrad() { return requiresGrad; }

    @Override
    public IDiffTensor setRequiresGrad(boolean requiresGrad) {
        this.requiresGrad = requiresGrad;
        if (!requiresGrad && isLeaf) this.grad = null;
        return this;
    }

    @Override
    public IDoubleTensor grad() {
        if (grad == null) return null;
        return new RereDoubleTensor(grad, shape());
    }

    /** Accumulate gradient into this node. Takes ownership of the array on first call. */
    public void accGrad(double[] incoming) {
        if (grad == null) {
            grad = incoming;
        } else {
            for (int i = 0; i < grad.length; i++) grad[i] += incoming[i];
        }
    }

    /**
     * Accumulate gradient from a pooled buffer, then release the buffer.
     * On first call, copies the buffer into a new owned array and releases the pooled buffer.
     * On subsequent calls, accumulates in-place and releases the pooled buffer.
     * Pattern: acquire from pool → fill → accGradFromPooled → auto release.
     */
    public void accGradFromPooled(double[] pooledBuf, int n) {
        if (grad != null) {
            double[] g = grad;
            for (int i = 0; i < n; i++) g[i] += pooledBuf[i];
            AutodiffBufferPool.release(pooledBuf);
        } else {
            double[] dx = new double[n];
            System.arraycopy(pooledBuf, 0, dx, 0, n);
            AutodiffBufferPool.release(pooledBuf);
            accGrad(dx);
        }
    }

    // ==================== toNonDiff ====================

    private IDiffTensor toNonDiff(IDoubleTensor t) {
        if (t instanceof RereDiffTensor rdt) return rdt;
        if (t instanceof ConstantDiffTensor cdt) return cdt;
        if (t instanceof RereDoubleTensor rdt) return new ConstantDiffTensor(rdt);
        return new ConstantDiffTensor(new RereDoubleTensor(t.toDoubleArray(), t.shape()));
    }

    // ==================== sum() — override default to avoid flattenValue cycle ====================

    @Override
    public IDiffTensor sum() {
        if (!requiresGrad) {
            IDoubleTensor r = value;
            for (int d = rank() - 1; d >= 0; d--) r = r.sum(d, false);
            return toNonDiff(r);
        }
        // Pattern fusion: detect common unaryOp + sum patterns
        IDiffTensor fused = tryFuseSum();
        if (fused != null) return fused;

        long n = value.totalSize();
        double total = 0;
        for (long i = 0; i < n; i++) total += value.linearGet(i);
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            Arrays.fill(inGrad, self.grad[0]);
            input.accGradFromPooled(inGrad, m);
        };
        RereDiffTensor result = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(this), bw, "sum");
        result.symbolicBackwardFn = sumGradFn(shape());
        return result;
    }

    /** Try to fuse unaryOp + sum into a single node. Returns null if no fusion pattern matches. */
    private IDiffTensor tryFuseSum() {
        if (inputs.size() != 1) return null;
        RereDiffTensor x = inputs.get(0);
        if (x == null || !x.requiresGrad) return null;
        int m = (int) x.value.totalSize();
        double total = 0;
        for (int i = 0; i < m; i++) total += value.linearGet(i);
        double[] xData = x.value.toDoubleArray();

        // square().sum()
        if ("square".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = 2.0 * g * xData[i];
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "squareSum");
            r.exportShape = x.shape();
            double[] squareFactor = new double[m];
            for (int i = 0; i < m; i++) squareFactor[i] = 2.0 * xData[i];
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), squareFactor);
            return r;
        }
        // relu().sum()
        if ("relu".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = xData[i] > 0 ? g : 0;
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "reluSum");
            r.exportShape = x.shape();
            double[] reluFactor = new double[m];
            for (int i = 0; i < m; i++) reluFactor[i] = xData[i] > 0 ? 1.0 : 0.0;
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), reluFactor);
            return r;
        }
        // exp().sum()
        if ("exp".equals(opTag)) {
            double[] eData = value.toDoubleArray(); // already computed exp(x)
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g * eData[i];
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "expSum");
            r.exportShape = x.shape();
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), eData);
            return r;
        }
        // sigmoid().sum()
        if ("sigmoid".equals(opTag)) {
            double[] sData = value.toDoubleArray(); // sigmoid(x)
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g * sData[i] * (1.0 - sData[i]);
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "sigmoidSum");
            r.exportShape = x.shape();
            double[] sigmoidFactor = new double[m];
            for (int i = 0; i < m; i++) { double s = sData[i]; sigmoidFactor[i] = s * (1.0 - s); }
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), sigmoidFactor);
            return r;
        }
        // abs().sum()
        if ("abs".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = xData[i] >= 0 ? g : -g;
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "absSum");
            r.exportShape = x.shape();
            double[] absFactor = new double[m];
            for (int i = 0; i < m; i++) absFactor[i] = xData[i] >= 0 ? 1.0 : -1.0;
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), absFactor);
            return r;
        }
        // tanh().sum()
        if ("tanh".equals(opTag)) {
            double[] tData = value.toDoubleArray(); // tanh(x)
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g * (1.0 - tData[i] * tData[i]);
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "tanhSum");
            r.exportShape = x.shape();
            double[] tanhFactor = new double[m];
            for (int i = 0; i < m; i++) tanhFactor[i] = 1.0 - tData[i] * tData[i];
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), tanhFactor);
            return r;
        }
        // silu().sum()
        if ("silu".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) {
                    double xi = xData[i];
                    double sig = 1.0 / (1.0 + Math.exp(-xi));
                    dxBuf[i] = g * (sig + xi * sig * (1.0 - sig));
                }
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "siluSum");
            r.exportShape = x.shape();
            double[] siluFactor = new double[m];
            for (int i = 0; i < m; i++) {
                double xi = xData[i];
                double sig = 1.0 / (1.0 + Math.exp(-xi));
                siluFactor[i] = sig + xi * sig * (1.0 - sig);
            }
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), siluFactor);
            return r;
        }
        // log().sum()
        if ("log".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g / xData[i];
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "logSum");
            r.exportShape = x.shape();
            double[] logFactor = new double[m];
            for (int i = 0; i < m; i++) logFactor[i] = 1.0 / xData[i];
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), logFactor);
            return r;
        }
        // pow(n).sum()
        double scalarP = scalarParam; // capture scalarParam (n)
        if ("pow".equals(opTag) && !Double.isNaN(scalarP)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g * scalarP * Math.pow(xData[i], scalarP - 1);
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "powSum");
            r.exportShape = x.shape();
            r.scalarParam = scalarP;
            double[] powFactor = new double[m];
            for (int i = 0; i < m; i++) powFactor[i] = scalarP * Math.pow(xData[i], scalarP - 1);
            r.symbolicBackwardFn = broadcastGradFn(x.shape(), powFactor);
            return r;
        }
        return null;
    }

    /** Try to fuse unaryOp + sum(dim) into a single fused node. Returns null if no pattern matches. */
    private IDiffTensor tryFuseSumDim(int dim, boolean keepdim) {
        if (inputs.size() != 1) return null;
        RereDiffTensor x = inputs.get(0);
        if (x == null || !x.requiresGrad) return null;
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= s[i];
        int reduce = s[dim];
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= s[i];
        int total = outer * reduce * inner;
        int resultLen = outer * inner;
        double[] xData = x.value.toDoubleArray();
        double[] vals = value.toDoubleArray();

        int fOuter = outer, fReduce = reduce, fInner = inner;
        int[] resultShape = reducedShape(dim, keepdim);

        // Compute actual reduce-sum result from current value (already computed in forward)
        double[] result = new double[resultLen];
        for (int o = 0; o < fOuter; o++) {
            for (int i = 0; i < fInner; i++) {
                double sum = 0;
                for (int r = 0; r < fReduce; r++)
                    sum += vals[(o * fReduce + r) * fInner + i];
                result[o * fInner + i] = sum;
            }
        }

        // -- square().sum(dim) --
        if ("square".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "squareSum",
                (g, xv) -> g * 2.0 * xv, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- relu().sum(dim) --
        if ("relu".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "reluSum",
                (g, xv) -> xv > 0 ? g : 0, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- exp().sum(dim) --
        if ("exp".equals(opTag)) {
            double[] eData = value.toDoubleArray();
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "expSum",
                (g, xv) -> g * Math.exp(xv), xData, fOuter, fReduce, fInner, total);
            // Override: use the exp(x) values for backward factor
            double[] expFactor = new double[total];
            for (int i = 0; i < total; i++) expFactor[i] = eData[i];
            r.symbolicBackwardFn = dimSumGradFn(x.shape(), dim, expFactor);
            return r;
        }
        // -- sigmoid().sum(dim) --
        if ("sigmoid".equals(opTag)) {
            double[] sigData = value.toDoubleArray();
            RereDiffTensor rt = buildFusedSumDim(x, result, resultShape, dim, keepdim, "sigmoidSum",
                (g, xv) -> { double sv = 1.0/(1.0+Math.exp(-xv)); return g * sv * (1-sv); },
                xData, fOuter, fReduce, fInner, total);
            double[] sigFactor = new double[total];
            for (int i = 0; i < total; i++) { double sv = sigData[i]; sigFactor[i] = sv * (1-sv); }
            rt.symbolicBackwardFn = dimSumGradFn(x.shape(), dim, sigFactor);
            return rt;
        }
        // -- abs().sum(dim) --
        if ("abs".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "absSum",
                (g, xv) -> xv >= 0 ? g : -g, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- tanh().sum(dim) --
        if ("tanh".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "tanhSum",
                (g, xv) -> { double t = Math.tanh(xv); return g * (1.0 - t*t); },
                xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- silu().sum(dim) --
        if ("silu".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "siluSum",
                (g, xv) -> { double sig = 1.0/(1.0+Math.exp(-xv)); return g * (sig + xv * sig * (1-sig)); },
                xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- log().sum(dim) --
        if ("log".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "logSum",
                (g, xv) -> g / xv, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- pow(n).sum(dim) --
        double sp = scalarParam;
        if ("pow".equals(opTag) && !Double.isNaN(sp)) {
            double param = sp;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "powSum",
                (g, xv) -> g * param * Math.pow(xv, param - 1), xData, fOuter, fReduce, fInner, total);
            r.scalarParam = param;
            return r;
        }
        return null;
    }

    /** Build a fused unaryOp+sum(dim) graph node. Shared across all 9 patterns. */
    private RereDiffTensor buildFusedSumDim(RereDiffTensor x, double[] result, int[] resultShape,
                                             int dim, boolean keepdim, String tag,
                                             DoubleBinaryOperator gradFn, double[] xData,
                                             int outer, int reduce, int inner, int total) {
        double[] dxBuf = AutodiffBufferPool.acquire(total);
        Consumer<RereDiffTensor> bw = self -> {
            for (int o = 0; o < outer; o++) {
                for (int r = 0; r < reduce; r++) {
                    for (int i = 0; i < inner; i++) {
                        int idx = (o * reduce + r) * inner + i;
                        dxBuf[idx] = gradFn.applyAsDouble(self.grad[o * inner + i], xData[idx]);
                    }
                }
            }
            x.accGradFromPooled(dxBuf, total);
        };
        RereDiffTensor node = new RereDiffTensor(result, resultShape, List.of(x), bw, tag);
        node.exportShape = x.shape();
        node.scalarParam = dim;
        node.scalarParam2 = keepdim ? 1.0 : 0.0;
        // Build symbolic backward factor for higher-order AD
        double[] factor = new double[total];
        for (int o = 0; o < outer; o++) {
            for (int k = 0; k < reduce; k++) {
                for (int i = 0; i < inner; i++) {
                    int idx = (o * reduce + k) * inner + i;
                    factor[idx] = gradFn.applyAsDouble(1.0, xData[idx]);
                }
            }
        }
        node.symbolicBackwardFn = dimSumGradFn(x.shape(), dim, factor);
        return node;
    }

    /** symbolicBackwardFn for sum(dim)-fused ops: broadcast g along dim, multiply by factor. */
    private static java.util.function.Function<IDiffTensor, IDiffTensor[]> dimSumGradFn(
            int[] inputShape, int dim, double[] factor) {
        int[] shapeCopy = inputShape.clone();
        double[] factorCopy = factor.clone();
        int dimCopy = dim;
        return g -> {
            IDiffTensor expanded = g.unsqueeze(dimCopy);
            IDiffTensor factorTensor = IDiffTensor.constantTensor(factorCopy, shapeCopy);
            return new IDiffTensor[]{ expanded.mul(factorTensor) };
        };
    }

    /** Add symbolicBackwardFn that broadcasts g to inputShape, multiplied by element-wise factor. */
    private static java.util.function.Function<IDiffTensor, IDiffTensor[]> broadcastGradFn(int[] inputShape, double[] factor) {
        int[] shapeCopy = inputShape.clone();
        double[] factorCopy = factor.clone();
        return g -> new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(factorCopy, shapeCopy)) };
    }

    /** Add symbolicBackwardFn for sum: broadcast scalar g to inputShape. */
    private static java.util.function.Function<IDiffTensor, IDiffTensor[]> sumGradFn(int[] inputShape) {
        int[] shapeCopy = inputShape.clone();
        long n = 1;
        for (int d : inputShape) n *= d;
        long totalN = n;
        return g -> {
            double[] ones = new double[(int) totalN];
            Arrays.fill(ones, 1.0);
            return new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(ones, shapeCopy)) };
        };
    }

    /** Add symbolicBackwardFn for scalar op: g * backward(1, x). */
    private static java.util.function.Function<IDiffTensor, IDiffTensor[]> scalarOpGradFn(
            DoubleBinaryOperator backward, double[] xData, int[] shape) {
        double[] xCopy = xData.clone();
        int[] shapeCopy = shape.clone();
        return g -> {
            int n = xCopy.length;
            double[] factor = new double[n];
            for (int i = 0; i < n; i++) factor[i] = backward.applyAsDouble(1.0, xCopy[i]);
            return new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(factor, shapeCopy)) };
        };
    }

    /** Symbolic backward for same-shape binary ops: returns gradient for each requiresGrad input. */
    private java.util.function.Function<IDiffTensor, IDiffTensor[]> binarySameSymbolicFn(
            int n, BinaryBackward gradA, BinaryBackward gradB,
            double[] aData, double[] bData, int[] shape,
            boolean hasA, boolean hasB) {
        double[] aCopy = aData.clone();
        double[] bCopy = bData.clone();
        int[] shapeCopy = shape.clone();
        return g -> {
            double[] factA = new double[n];
            double[] factB = new double[n];
            for (int i = 0; i < n; i++) {
                factA[i] = gradA.apply(1.0, aCopy[i], bCopy[i]);
                factB[i] = gradB.apply(1.0, aCopy[i], bCopy[i]);
            }
            IDiffTensor tA = g.mul(IDiffTensor.constantTensor(factA, shapeCopy));
            IDiffTensor tB = g.mul(IDiffTensor.constantTensor(factB, shapeCopy));
            if (hasA && hasB) return new IDiffTensor[]{ tA, tB };
            return new IDiffTensor[]{ hasA ? tA : tB };
        };
    }

    /** Symbolic backward for broadcast binary ops: scatter-reduce gradient factor to original shape. */
    private java.util.function.Function<IDiffTensor, IDiffTensor[]> binaryBroadcastSymbolicFn(
            int n, BinaryBackward gradA, BinaryBackward gradB,
            double[] bcA, double[] bcB,
            int[] sA, int[] sB, int[] resultShape,
            boolean hasA, boolean hasB) {
        double[] bcACopy = bcA.clone();
        double[] bcBCopy = bcB.clone();
        int[] sACopy = sA.clone();
        int[] sBCopy = sB.clone();
        int[] rShapeCopy = resultShape.clone();
        return g -> {
            if (hasA && hasB) {
                int aTotal = (int) computeSize(sACopy);
                int bTotal = (int) computeSize(sBCopy);
                double[] factA = new double[aTotal];
                double[] factB = new double[bTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    if (hasA) factA[flatIndexFromBroadcast(idx, sACopy, rShapeCopy)] += gradA.apply(1.0, bcACopy[i], bcBCopy[i]);
                    if (hasB) factB[flatIndexFromBroadcast(idx, sBCopy, rShapeCopy)] += gradB.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(factA, sACopy)),
                                          g.mul(IDiffTensor.constantTensor(factB, sBCopy)) };
            } else if (hasA) {
                int aTotal = (int) computeSize(sACopy);
                double[] factA = new double[aTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    factA[flatIndexFromBroadcast(idx, sACopy, rShapeCopy)] += gradA.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(factA, sACopy)) };
            } else {
                int bTotal = (int) computeSize(sBCopy);
                double[] factB = new double[bTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    factB[flatIndexFromBroadcast(idx, sBCopy, rShapeCopy)] += gradB.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(factB, sBCopy)) };
            }
        };
    }

    // ==================== Element-wise unary ops ====================

    @Override public IDiffTensor neg() { return unaryOp(x -> -x, (g, x) -> -g, "neg"); }
    @Override public IDiffTensor abs() { return unaryOp(Math::abs, (g, x) -> x >= 0 ? g : -g, "abs"); }
    @Override public IDiffTensor sqrt() { return unaryOp(Math::sqrt, (g, x) -> g / (2.0 * Math.sqrt(x)), "sqrt"); }
    @Override public IDiffTensor exp() { return unaryOp(Math::exp, (g, x) -> g * Math.exp(x), "exp"); }
    @Override public IDiffTensor log() { return unaryOp(Math::log, (g, x) -> g / x, "log"); }
    @Override public IDiffTensor sin() { return unaryOp(Math::sin, (g, x) -> g * Math.cos(x), "sin"); }
    @Override public IDiffTensor cos() { return unaryOp(Math::cos, (g, x) -> -g * Math.sin(x), "cos"); }
    @Override public IDiffTensor tan() { return unaryOp(Math::tan, (g, x) -> { double c = Math.cos(x); return g / (c * c); }, "tan"); }
    @Override public IDiffTensor sigmoid() { return unaryOp(x -> 1.0 / (1.0 + Math.exp(-x)), (g, x) -> { double s = 1.0 / (1.0 + Math.exp(-x)); return g * s * (1.0 - s); }, "sigmoid"); }
    @Override public IDiffTensor relu() { return unaryOp(x -> x > 0 ? x : 0, (g, x) -> x > 0 ? g : 0, "relu"); }

    @Override
    public IDiffTensor square() {
        return unaryOp(x -> x * x, (g, x) -> g * 2.0 * x, "square");
    }

    @Override
    public IDiffTensor pow(double n) {
        if (!requiresGrad) {
            double[] data = value.toDoubleArray();
            for (int i = 0; i < data.length; i++) data[i] = Math.pow(data[i], n);
            return toNonDiff(new RereDoubleTensor(data, shape()));
        }
        int total = (int) value.totalSize();
        double[] out = new double[total];
        for (int i = 0; i < total; i++) out[i] = Math.pow(value.linearGet(i), n);
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < total; i++) {
                double x = input.value.linearGet(i);
                inGrad[i] = self.grad[i] * n * Math.pow(x, n - 1);
            }
            input.accGradFromPooled(inGrad, m);
        };
        RereDiffTensor powResult = new RereDiffTensor(out, shape(), List.of(this), bw, "pow", n);
        // pow symbolic backward: d/dx x^n = n * x^(n-1)
        double[] xData = value.toDoubleArray();
        double scalarN = n;
        powResult.symbolicBackwardFn = scalarOpGradFn(
            (g, v) -> g * scalarN * Math.pow(v, scalarN - 1), xData, shape());
        return powResult;
    }

    @Override
    public IDiffTensor clamp(double min, double max) {
        if (!requiresGrad) return toNonDiff(value.clamp(min, max));
        int total = (int) value.totalSize();
        double[] out = new double[total];
        for (int i = 0; i < total; i++) {
            double x = value.linearGet(i);
            out[i] = x < min ? min : x > max ? max : x;
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < total; i++) {
                double x = input.value.linearGet(i);
                inGrad[i] = (x > min && x < max) ? self.grad[i] : 0;
            }
            input.accGradFromPooled(inGrad, m);
        };
        return new RereDiffTensor(out, shape(), List.of(this), bw, "clamp", min, max);
    }

    @Override public IDiffTensor tanh() { return unaryOp(Math::tanh, (g, x) -> { double t = Math.tanh(x); return g * (1.0 - t * t); }, "tanh"); }

    @Override
    public IDiffTensor silu() {
        return unaryOp(x -> x / (1.0 + Math.exp(-x)),
            (g, x) -> { double s = 1.0 / (1.0 + Math.exp(-x)); return g * (s + x * s * (1.0 - s)); }, "silu");
    }

    @Override
    public IDiffTensor gelu() {
        return unaryOp(x -> {
            double cdf = 0.5 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
            return x * cdf;
        }, (g, x) -> {
            double x3 = x * x * x;
            double inner = Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x3);
            double tanhInner = Math.tanh(inner);
            double sech2 = 1.0 - tanhInner * tanhInner;
            double cdf = 0.5 * (1.0 + tanhInner);
            double pdf = 0.5 * Math.sqrt(2.0 / Math.PI) * (1.0 + 0.134145 * x * x) * sech2;
            return g * (cdf + x * pdf);
        }, "gelu");
    }

    @Override
    public IDiffTensor softplus(double beta) {
        return unaryOp(x -> { double bx = beta * x; return bx > 20 ? x : Math.log(1.0 + Math.exp(bx)) / beta; },
            (g, x) -> { double bx = beta * x; return bx > 20 ? g : g / (1.0 + Math.exp(-bx)); }, "softplus", beta);
    }

    @Override
    public IDiffTensor mish() {
        return unaryOp(x -> x * Math.tanh(Math.log(1.0 + Math.exp(x))),
            (g, x) -> {
                double sp = Math.log(1.0 + Math.exp(x));
                double th = Math.tanh(sp);
                double sig = 1.0 / (1.0 + Math.exp(-x));
                return g * (th + x * sig * (1.0 - th * th));
            }, "mish");
    }

    @Override
    public IDiffTensor elu(double alpha) {
        return unaryOp(x -> x >= 0 ? x : alpha * (Math.exp(x) - 1),
            (g, x) -> x >= 0 ? g : g * alpha * Math.exp(x), "elu", alpha);
    }

    @Override
    public IDiffTensor leakyRelu(double alpha) {
        return unaryOp(x -> x >= 0 ? x : alpha * x, (g, x) -> x >= 0 ? g : g * alpha, "leakyRelu", alpha);
    }

    @Override
    public IDiffTensor selu() {
        double alpha = 1.6732632423543772, scale = 1.0507009873554804;
        return unaryOp(x -> scale * (x >= 0 ? x : alpha * (Math.exp(x) - 1)),
            (g, x) -> x >= 0 ? g * scale : g * scale * alpha * Math.exp(x), "selu");
    }

    @Override
    public IDiffTensor hardtanh(double minVal, double maxVal) {
        return unaryOp(x -> Math.min(Math.max(x, minVal), maxVal),
            (g, x) -> (x > minVal && x < maxVal) ? g : 0, "hardtanh", minVal, maxVal);
    }

    @Override
    public IDiffTensor dropout(double p) {
        if (!requiresGrad) return toNonDiff(value.clone());
        int total = (int) value.totalSize();
        long seed = RereDiffVector.DROPOUT_SEED_COUNTER.incrementAndGet();
        Random rng = new Random(seed);
        double[] mask = new double[total];
        double[] out = new double[total];
        double scale = 1.0 / (1.0 - p);
        for (int i = 0; i < total; i++) {
            mask[i] = rng.nextDouble() > p ? scale : 0.0;
            out[i] = value.linearGet(i) * mask[i];
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < m; i++) {
                inGrad[i] = self.grad[i] * mask[i];
            }
            input.accGradFromPooled(inGrad, m);
        };
        return new RereDiffTensor(out, shape(), List.of(this), bw, "dropout", p,
                Double.longBitsToDouble(seed));
    }

    // ---- SIMD-accelerated unary op helpers ----

    /** Map common element-wise op tags to {@link UniversalOperation} for SIMD forward acceleration. */
    private static UniversalOperation tagToUniversalOp(String tag) {
        return switch (tag) {
            case "exp"     -> UniversalOperation.EXP;
            case "log"     -> UniversalOperation.LOG;
            case "sqrt"    -> UniversalOperation.SQRT;
            case "relu"    -> UniversalOperation.RELU;
            case "sigmoid" -> UniversalOperation.SIGMOID;
            case "tanh"    -> UniversalOperation.TANH;
            case "abs"     -> UniversalOperation.ABS;
            case "sin"     -> UniversalOperation.SIN;
            case "cos"     -> UniversalOperation.COS;
            case "gelu"    -> UniversalOperation.GELU;
            default        -> null;
        };
    }

    // ---- unary op helper ----

    private IDiffTensor unaryOp(java.util.function.DoubleUnaryOperator forward,
                                 java.util.function.DoubleBinaryOperator backward,
                                 String tag) {
        if (!requiresGrad) {
            UniversalOperation uop = tagToUniversalOp(tag);
            if (uop != null && value instanceof RereDoubleTensor rdt) {
                double[] result = rdt.universalOp(uop, 0.0);
                return toNonDiff(new RereDoubleTensor(result, shape()));
            }
            double[] data = value.toDoubleArray();
            for (int i = 0; i < data.length; i++) data[i] = forward.applyAsDouble(data[i]);
            return toNonDiff(new RereDoubleTensor(data, shape()));
        }
        int n = (int) value.totalSize();
        UniversalOperation uop = tagToUniversalOp(tag);
        double[] out;
        if (uop != null && value instanceof RereDoubleTensor rdt) {
            out = rdt.universalOp(uop, 0.0);
        } else {
            out = new double[n];
            for (int i = 0; i < n; i++) out[i] = forward.applyAsDouble(value.linearGet(i));
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                inGrad[i] = backward.applyAsDouble(self.grad[i], input.value.linearGet(i));
            }
            input.accGradFromPooled(inGrad, n);
        };
        RereDiffTensor result = new RereDiffTensor(out, shape(), List.of(this), bw, tag);
        // Set symbolic backward for higher-order AD: dX = grad * backward(1, x)
        // Captures raw input data as double[] constant for the derivative factor.
        // For full second-order support, this should reference the original input
        // node rather than a constant array – deferred optimization.
        double[] xData = value.toDoubleArray();
        result.symbolicBackwardFn = g -> {
            double[] factor = new double[n];
            for (int i = 0; i < n; i++) factor[i] = backward.applyAsDouble(1.0, xData[i]);
            return new IDiffTensor[]{ g.mul(IDiffTensor.constantTensor(factor, shape())) };
        };
        return result;
    }

    private IDiffTensor unaryOp(java.util.function.DoubleUnaryOperator forward,
                                 java.util.function.DoubleBinaryOperator backward,
                                 String tag, double scalarParam) {
        IDiffTensor result = unaryOp(forward, backward, tag);
        if (result instanceof RereDiffTensor rt) rt.scalarParam = scalarParam;
        return result;
    }

    private IDiffTensor unaryOp(java.util.function.DoubleUnaryOperator forward,
                                 java.util.function.DoubleBinaryOperator backward,
                                 String tag, double scalarParam, double scalarParam2) {
        IDiffTensor result = unaryOp(forward, backward, tag);
        if (result instanceof RereDiffTensor rt) { rt.scalarParam = scalarParam; rt.scalarParam2 = scalarParam2; }
        return result;
    }

    // ==================== Element-wise binary ops — scalar ====================

    @Override public IDiffTensor add(double scalar) { return scalarOp(scalar, (a, b) -> a + b, (g, v) -> g, "add"); }
    @Override public IDiffTensor sub(double scalar) { return scalarOp(scalar, (a, b) -> a - b, (g, v) -> g, "sub"); }
    @Override public IDiffTensor mul(double scalar) { return scalarOp(scalar, (a, b) -> a * b, (g, v) -> g * scalar, "mul"); }
    @Override public IDiffTensor div(double scalar) { return scalarOp(scalar, (a, b) -> a / b, (g, v) -> g / scalar, "div"); }

    @Override
    public IDiffTensor rsub(double scalar) {
        if (!requiresGrad) {
            double[] data = value.toDoubleArray();
            for (int i = 0; i < data.length; i++) data[i] = scalar - data[i];
            return toNonDiff(new RereDoubleTensor(data, shape()));
        }
        int n = (int) value.totalSize();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = scalar - value.linearGet(i);
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) inGrad[i] = -self.grad[i];
            input.accGradFromPooled(inGrad, n);
        };
        RereDiffTensor r = new RereDiffTensor(out, shape(), List.of(this), bw, "rsub", scalar);
        double[] rsubFactor = new double[n];
        Arrays.fill(rsubFactor, -1.0);
        r.symbolicBackwardFn = broadcastGradFn(shape(), rsubFactor);
        return r;
    }

    @Override
    public IDiffTensor rdiv(double scalar) {
        if (!requiresGrad) {
            double[] data = value.toDoubleArray();
            for (int i = 0; i < data.length; i++) data[i] = scalar / data[i];
            return toNonDiff(new RereDoubleTensor(data, shape()));
        }
        int n = (int) value.totalSize();
        double[] out = new double[n];
        double[] xd = value.toDoubleArray();
        for (int i = 0; i < n; i++) out[i] = scalar / xd[i];
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < m; i++) inGrad[i] = -self.grad[i] * scalar / (xd[i] * xd[i]);
            input.accGradFromPooled(inGrad, m);
        };
        RereDiffTensor r = new RereDiffTensor(out, shape(), List.of(this), bw, "rdiv", scalar);
        r.symbolicBackwardFn = scalarOpGradFn((g, v) -> -g * scalar / (v * v), xd, shape());
        return r;
    }

    @Override
    public IDiffTensor reciprocal() {
        if (!requiresGrad) {
            double[] data = value.toDoubleArray();
            for (int i = 0; i < data.length; i++) data[i] = 1.0 / data[i];
            return toNonDiff(new RereDoubleTensor(data, shape()));
        }
        int n = (int) value.totalSize();
        double[] out = new double[n];
        double[] xd = value.toDoubleArray();
        for (int i = 0; i < n; i++) out[i] = 1.0 / xd[i];
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int m = (int) input.value.totalSize();
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < m; i++) inGrad[i] = -self.grad[i] / (xd[i] * xd[i]);
            input.accGradFromPooled(inGrad, m);
        };
        RereDiffTensor r = new RereDiffTensor(out, shape(), List.of(this), bw, "reciprocal");
        r.symbolicBackwardFn = scalarOpGradFn((g, v) -> -g / (v * v), xd, shape());
        return r;
    }

    private IDiffTensor scalarOp(double scalar, DoubleBinaryOperator forward,
                                  DoubleBinaryOperator backward, String tag) {
        if (!requiresGrad) {
            double[] data = value.toDoubleArray();
            for (int i = 0; i < data.length; i++) data[i] = forward.applyAsDouble(data[i], scalar);
            return toNonDiff(new RereDoubleTensor(data, shape()));
        }
        int n = (int) value.totalSize();
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = forward.applyAsDouble(value.linearGet(i), scalar);
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                inGrad[i] = backward.applyAsDouble(self.grad[i], input.value.linearGet(i));
            }
            input.accGradFromPooled(inGrad, n);
        };
        RereDiffTensor result = new RereDiffTensor(out, shape(), List.of(this), bw, tag, scalar);
        result.symbolicBackwardFn = scalarOpGradFn(backward, value.toDoubleArray(), shape());
        return result;
    }

    // ==================== Element-wise binary ops — tensor ====================

    @Override public IDiffTensor add(IDoubleTensor other) { return binaryTensorOp(other, (a,b)->a+b, (g,a,b)->g, (g,a,b)->g, "add"); }
    @Override public IDiffTensor sub(IDoubleTensor other) { return binaryTensorOp(other, (a,b)->a-b, (g,a,b)->g, (g,a,b)->-g, "sub"); }
    @Override public IDiffTensor mul(IDoubleTensor other) { return binaryTensorOp(other, (a,b)->a*b, (g,a,b)->g*b, (g,a,b)->g*a, "mul"); }
    @Override public IDiffTensor div(IDoubleTensor other) { return binaryTensorOp(other, (a,b)->a/b, (g,a,b)->g/b, (g,a,b)->-g*a/(b*b), "div"); }

    private IDiffTensor binaryTensorOp(IDoubleTensor other,
                                        DoubleBinaryOperator forward,
                                        BinaryBackward gradA,
                                        BinaryBackward gradB,
                                        String tag) {
        // Detach if needed
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        boolean otherDiff = other instanceof IDiffTensor && ((IDiffTensor) other).requiresGrad();

        if (Arrays.equals(shape(), other.shape())) {
            // Same shape
            if (!requiresGrad && !otherDiff) {
                double[] aData = value.toDoubleArray();
                double[] bData = detOther.toDoubleArray();
                double[] out = new double[aData.length];
                for (int i = 0; i < out.length; i++) out[i] = forward.applyAsDouble(aData[i], bData[i]);
                return toNonDiff(new RereDoubleTensor(out, shape()));
            }
            int n = (int) value.totalSize();
            double[] out = new double[n];
            double[] aData = value.toDoubleArray();
            double[] bData = detOther.toDoubleArray();
            double[] savedA = requiresGrad ? aData.clone() : null;
            double[] savedB = (requiresGrad || otherDiff) ? bData.clone() : null;
            for (int i = 0; i < n; i++) out[i] = forward.applyAsDouble(aData[i], bData[i]);

            List<RereDiffTensor> inputs = new ArrayList<>();
            if (requiresGrad) inputs.add(this);
            RereDiffTensor otherNode = (other instanceof RereDiffTensor rt && rt.requiresGrad) ? rt : null;
            if (otherNode != null) inputs.add(otherNode);

            Consumer<RereDiffTensor> bw = self -> {
                int idx = 0;
                if (requiresGrad) {
                    RereDiffTensor inpA = self.inputs.get(idx++);
                    double[] dA = AutodiffBufferPool.acquire(n);
                    for (int i = 0; i < n; i++) dA[i] = gradA.apply(self.grad[i], savedA[i], savedB[i]);
                    inpA.accGradFromPooled(dA, n);
                }
                if (otherNode != null) {
                    RereDiffTensor inpB = self.inputs.get(idx);
                    double[] dB = AutodiffBufferPool.acquire(n);
                    for (int i = 0; i < n; i++) dB[i] = gradB.apply(self.grad[i], savedA[i], savedB[i]);
                    inpB.accGradFromPooled(dB, n);
                }
            };
            RereDiffTensor result = new RereDiffTensor(out, shape(), inputs, bw, tag);
            boolean hasA = requiresGrad;
            boolean hasB = otherNode != null;
            result.symbolicBackwardFn = binarySameSymbolicFn(n, gradA, gradB, aData, bData, shape(), hasA, hasB);
            return result;
        }

        // Broadcast case
        int[] sA = shape();
        int[] sB = other.shape();
        int[] resultShape = TensorShape.broadcastShape(sA, sB);
        long total = 1;
        for (int d : resultShape) total *= d;
        int n = (int) total;

        double[] aData = value.toDoubleArray();
        double[] bData = detOther.toDoubleArray();
        double[] out = new double[n];
        double[] bcA = new double[n];
        double[] bcB = new double[n];

        for (int i = 0; i < n; i++) {
            int[] idx = unlinearizeInt(i, resultShape);
            int flatA = flatIndexFromBroadcast(idx, sA, resultShape);
            int flatB = flatIndexFromBroadcast(idx, sB, resultShape);
            bcA[i] = aData[flatA];
            bcB[i] = bData[flatB];
            out[i] = forward.applyAsDouble(bcA[i], bcB[i]);
        }

        if (!requiresGrad && !otherDiff) return toNonDiff(new RereDoubleTensor(out, resultShape));

        List<RereDiffTensor> inputs = new ArrayList<>();
        if (requiresGrad) inputs.add(this);
        RereDiffTensor otherNode = (other instanceof RereDiffTensor rt && rt.requiresGrad) ? rt : null;
        if (otherNode != null) inputs.add(otherNode);

        Consumer<RereDiffTensor> bw = self -> {
            int idx = 0;
            if (requiresGrad) {
                RereDiffTensor inpA = self.inputs.get(idx++);
                int aTotal = (int) computeSize(sA);
                double[] dA = AutodiffBufferPool.acquire(aTotal);
                for (int i = 0; i < n; i++) {
                    int flatA = flatIndexFromBroadcast(unlinearizeInt(i, resultShape), sA, resultShape);
                    dA[flatA] += gradA.apply(self.grad[i], bcA[i], bcB[i]);
                }
                inpA.accGradFromPooled(dA, aTotal);
            }
            if (otherNode != null) {
                RereDiffTensor inpB = self.inputs.get(idx);
                int bTotal = (int) computeSize(sB);
                double[] dB = AutodiffBufferPool.acquire(bTotal);
                for (int i = 0; i < n; i++) {
                    int flatB = flatIndexFromBroadcast(unlinearizeInt(i, resultShape), sB, resultShape);
                    dB[flatB] += gradB.apply(self.grad[i], bcA[i], bcB[i]);
                }
                inpB.accGradFromPooled(dB, bTotal);
            }
        };
        RereDiffTensor result = new RereDiffTensor(out, resultShape, inputs, bw, tag);
        boolean bHasA = requiresGrad;
        boolean bHasB = otherNode != null;
        result.symbolicBackwardFn = binaryBroadcastSymbolicFn(
            n, gradA, gradB, bcA, bcB, sA, sB, resultShape, bHasA, bHasB);
        return result;
    }

    @FunctionalInterface
    private interface BinaryBackward { double apply(double grad, double aVal, double bVal); }

    // ==================== View ops ====================

    @Override
    public IDiffTensor select(int dim, long index) {
        RereDoubleTensor view = (RereDoubleTensor) value.select(dim, index);
        if (!requiresGrad) return toNonDiff(view);
        int[] parentShape = shape();
        int d = (dim < 0 ? dim + rank() : dim);
        int[] viewShape = view.shape();
        int viewTotal = (int) view.totalSize();
        // Precompute parent flat index for each view position
        int[] parentIdx = new int[viewTotal];
        for (int i = 0; i < viewTotal; i++) {
            int[] vIdx = unlinearizeInt(i, viewShape);
            int[] pIdx = new int[parentShape.length];
            int vi = 0;
            for (int j = 0; j < parentShape.length; j++) {
                pIdx[j] = (j == d) ? (int) index : vIdx[vi++];
            }
            parentIdx[i] = flatIndex(pIdx, parentShape);
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor parent = self.inputs.get(0);
            int pt = (int) parent.value.totalSize();
            double[] pGrad = AutodiffBufferPool.acquire(pt);
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
            parent.accGradFromPooled(pGrad, pt);
        };
        return new RereDiffTensor(view, List.of(this), bw, "select");
    }

    @Override
    public IDiffTensor slice(int dim, long start, long end) {
        RereDoubleTensor view = (RereDoubleTensor) value.slice(dim, start, end);
        if (!requiresGrad) return toNonDiff(view);
        int[] parentShape = shape();
        int d = (dim < 0 ? dim + rank() : dim);
        int[] viewShape = view.shape();
        int viewTotal = (int) view.totalSize();
        int[] parentIdx = new int[viewTotal];
        for (int i = 0; i < viewTotal; i++) {
            int[] vIdx = unlinearizeInt(i, viewShape);
            int[] pIdx = vIdx.clone();
            pIdx[d] += (int) start;
            parentIdx[i] = flatIndex(pIdx, parentShape);
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor parent = self.inputs.get(0);
            int pt = (int) parent.value.totalSize();
            double[] pGrad = AutodiffBufferPool.acquire(pt);
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
            parent.accGradFromPooled(pGrad, pt);
        };
        return new RereDiffTensor(view, List.of(this), bw, "slice");
    }

    @Override
    public IDiffTensor narrow(int dim, long start, long length) {
        // narrow is same as slice(dim, start, start+length)
        return slice(dim, start, start + length);
    }

    @Override
    public IDiffTensor permute(int... dims) {
        RereDoubleTensor view = (RereDoubleTensor) value.permute(dims);
        if (!requiresGrad) return toNonDiff(view);
        int[] parentShape = shape();
        int[] viewShape = view.shape();
        int viewTotal = (int) view.totalSize();
        // Compute inverse permutation
        int[] invDims = new int[dims.length];
        for (int i = 0; i < dims.length; i++) invDims[dims[i]] = i;
        int[] parentIdx = new int[viewTotal];
        for (int i = 0; i < viewTotal; i++) {
            int[] vIdx = unlinearizeInt(i, viewShape);
            int[] pIdx = new int[parentShape.length];
            for (int j = 0; j < dims.length; j++) pIdx[dims[j]] = vIdx[j];
            parentIdx[i] = flatIndex(pIdx, parentShape);
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor parent = self.inputs.get(0);
            int pt = (int) parent.value.totalSize();
            double[] pGrad = AutodiffBufferPool.acquire(pt);
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
            parent.accGradFromPooled(pGrad, pt);
        };
        RereDiffTensor result = new RereDiffTensor(view, List.of(this), bw, "permute");
        result.backwardIndices = dims;  // axis permutation for HPC/GPU backend
        return result;
    }

    @Override
    public IDiffTensor transpose(int dim0, int dim1) {
        int r = rank();
        int[] perm = new int[r];
        for (int i = 0; i < r; i++) perm[i] = i;
        perm[dim0] = dim1;
        perm[dim1] = dim0;
        return permute(perm);
    }

    @Override
    public IDiffTensor transpose() {
        if (rank() != 2) throw new IllegalStateException("transpose() requires rank 2, got " + rank());
        return transpose(0, 1);
    }

    @Override
    public IDiffTensor squeeze(int... dims) {
        RereDoubleTensor view = (RereDoubleTensor) value.squeeze(dims);
        if (!requiresGrad) return toNonDiff(view);
        int[] parentShape = shape();
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
            int[] vIdx = unlinearizeInt(i, viewShape);
            int[] pIdx = new int[parentShape.length];
            for (int j = 0; j < parentShape.length; j++) pIdx[j] = 0; // squeezed dims stay 0
            for (int j = 0; j < viewShape.length; j++) pIdx[srcDim[j]] = vIdx[j];
            parentIdx[i] = flatIndex(pIdx, parentShape);
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor parent = self.inputs.get(0);
            int pt = (int) parent.value.totalSize();
            double[] pGrad = AutodiffBufferPool.acquire(pt);
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
            parent.accGradFromPooled(pGrad, pt);
        };
        return new RereDiffTensor(view, List.of(this), bw, "squeeze");
    }

    @Override
    public IDiffTensor unsqueeze(int dim) {
        RereDoubleTensor view = (RereDoubleTensor) value.unsqueeze(dim);
        if (!requiresGrad) return toNonDiff(view);
        int d = (dim < 0 ? dim + rank() + 1 : dim);
        int[] parentShape = shape();
        int[] viewShape = view.shape();
        int viewTotal = (int) view.totalSize();
        int[] parentIdx = new int[viewTotal];
        for (int i = 0; i < viewTotal; i++) {
            int[] vIdx = unlinearizeInt(i, viewShape);
            int[] pIdx = new int[parentShape.length];
            int pi = 0;
            for (int j = 0; j < viewShape.length; j++) {
                if (j != d) pIdx[pi++] = vIdx[j];
            }
            parentIdx[i] = flatIndex(pIdx, parentShape);
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor parent = self.inputs.get(0);
            int pt = (int) parent.value.totalSize();
            double[] pGrad = AutodiffBufferPool.acquire(pt);
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
            parent.accGradFromPooled(pGrad, pt);
        };
        return new RereDiffTensor(view, List.of(this), bw, "unsqueeze");
    }

    @Override
    public IDiffTensor flatten(int startDim, int endDim) {
        RereDoubleTensor view = (RereDoubleTensor) value.flatten(startDim, endDim);
        if (!requiresGrad) return toNonDiff(view);
        int[] parentShape = shape();
        int[] viewShape = view.shape();
        int viewTotal = (int) view.totalSize();
        // flatten merges dims [start, end] into one. parentShape -> viewShape mapping:
        int pre = startDim, post = parentShape.length - endDim - 1;
        int midParent = 1;
        for (int j = startDim; j <= endDim; j++) midParent *= parentShape[j];
        int[] parentIdx = new int[viewTotal];
        for (int i = 0; i < viewTotal; i++) {
            int[] vIdx = unlinearizeInt(i, viewShape);
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
            parentIdx[i] = flatIndex(pIdx, parentShape);
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor parent = self.inputs.get(0);
            int pt = (int) parent.value.totalSize();
            double[] pGrad = AutodiffBufferPool.acquire(pt);
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
            parent.accGradFromPooled(pGrad, pt);
        };
        return new RereDiffTensor(view, List.of(this), bw, "flatten");
    }

    @Override
    public IDiffTensor expand(int... targetShape) {
        // expand repeats data along dimensions where parent shape is 1
        RereDoubleTensor view = (RereDoubleTensor) value.expand(targetShape);
        if (!requiresGrad) return toNonDiff(view);
        int[] parentShape = shape();
        int[] viewShape = view.shape();
        int viewTotal = (int) view.totalSize();
        int[] parentIdx = new int[viewTotal];
        for (int i = 0; i < viewTotal; i++) {
            int[] vIdx = unlinearizeInt(i, viewShape);
            int[] pIdx = new int[parentShape.length];
            for (int j = 0; j < parentShape.length; j++) {
                pIdx[j] = (parentShape[j] == 1) ? 0 : vIdx[j];
            }
            parentIdx[i] = flatIndex(pIdx, parentShape);
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor parent = self.inputs.get(0);
            int pt = (int) parent.value.totalSize();
            double[] pGrad = AutodiffBufferPool.acquire(pt);
            for (int i = 0; i < viewTotal; i++) pGrad[parentIdx[i]] += self.grad[i];
            parent.accGradFromPooled(pGrad, pt);
        };
        RereDiffTensor result = new RereDiffTensor(view, List.of(this), bw, "expand");
        result.backwardIndices = parentShape;  // source shape for HPC/GPU backward (reduce-sum over broadcast dims)
        return result;
    }

    @Override
    public IDiffTensor broadcastTo(int... targetShape) {
        return expand(targetShape);
    }

    @Override
    public IDiffTensor contiguous() {
        if (value.isContiguous()) return this;
        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(value.toDoubleArray(), shape()));
        int[] s = shape();
        int n = (int) value.totalSize();
        double[] contigData = new double[n];
        // Precompute source storage positions
        int[] srcIdx = new int[n];
        for (int i = 0; i < n; i++) {
            contigData[i] = value.linearGet(i);
            srcIdx[i] = i; // view logical position → parent logical position
        }
        // For contiguous(), the view IS the same logical elements, just reordered in storage.
        // Since grad is indexed by logical position, the backward is just identity.
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            input.accGrad(self.grad.clone());
        };
        return new RereDiffTensor(new RereDoubleTensor(contigData, s), List.of(this), bw, "contiguous");
    }

    @Override
    public IDiffTensor reshape(int... newShape) {
        if (isContiguous() && offset() == 0) {
            RereDoubleTensor view = (RereDoubleTensor) value.reshape(newShape);
            if (!requiresGrad) return toNonDiff(view);
            // Reshape doesn't change data order for contiguous tensors
            Consumer<RereDiffTensor> bw = self -> {
                RereDiffTensor input = self.inputs.get(0);
                input.accGrad(self.grad.clone());
            };
            return new RereDiffTensor(view, List.of(this), bw, "reshape");
        }
        return this.contiguous().reshape(newShape);
    }

    @Override
    public IDiffTensor tile(int... repeats) {
        int r = rank();
        int[] rep = new int[r];
        for (int i = 0; i < r; i++) rep[i] = (i < repeats.length) ? repeats[i] : 1;
        int[] inShape = shape();
        int[] outShape = new int[r];
        for (int i = 0; i < r; i++) outShape[i] = inShape[i] * rep[i];
        int outTotal = 1;
        for (int d : outShape) outTotal *= d;

        if (!requiresGrad) return toNonDiff(value.tile(repeats));

        double[] inData = value.toDoubleArray();
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
        return new RereDiffTensor(outData, outShape, List.of(this), bw, "tile");
    }

    // ==================== Reduction ops ====================

    @Override
    public IDiffTensor sum(int dim, boolean keepdim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.sum(d, keepdim));

        // Pattern fusion: detect unaryOp + sum(dim) patterns
        IDiffTensor fused = tryFuseSumDim(d, keepdim);
        if (fused != null) return fused;

        return sumDimImpl(d, keepdim);
    }

    /** Regular sum(dim) implementation (no fusion). */
    private IDiffTensor sumDimImpl(int d, boolean keepdim) {
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] vals = value.toDoubleArray();
        int[] resultShape = reducedShape(d, keepdim);
        int resultLen = outer * inner;
        double[] result = new double[resultLen];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double sum = 0;
                for (int r = 0; r < reduce; r++) {
                    sum += vals[(o * reduce + r) * inner + i];
                }
                result[o * inner + i] = sum;
            }
        }
        int fOuter = outer, fReduce = reduce, fInner = inner;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int r = 0; r < fReduce; r++) {
                    for (int i = 0; i < fInner; i++) {
                        inGrad[(o * fReduce + r) * fInner + i] = self.grad[o * fInner + i];
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(result, resultShape, List.of(this), bw, "sum");
    }

    @Override
    public IDiffTensor mean(int dim, boolean keepdim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.mean(d, keepdim));
        IDiffTensor s = sum(d, keepdim);
        double scale = 1.0 / dim(d);
        return s.mul(scale);
    }

    @Override
    public IDiffTensor max(int dim, boolean keepdim) { return minMax(dim, keepdim, true); }
    @Override
    public IDiffTensor min(int dim, boolean keepdim) { return minMax(dim, keepdim, false); }

    private IDiffTensor minMax(int dim, boolean keepdim, boolean isMax) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) {
            return toNonDiff(isMax ? value.max(d, keepdim) : value.min(d, keepdim));
        }
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] vals = value.toDoubleArray();
        double[] result = new double[outer * inner];
        int[] argIdx = new int[outer * inner];
        double init = isMax ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
        Arrays.fill(result, init);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                for (int r = 0; r < reduce; r++) {
                    double v = vals[(o * reduce + r) * inner + i];
                    if ((isMax && v > result[o * inner + i]) || (!isMax && v < result[o * inner + i])) {
                        result[o * inner + i] = v;
                        argIdx[o * inner + i] = r;
                    }
                }
            }
        }
        int fOuter = outer, fReduce = reduce, fInner = inner;
        int[] fArg = argIdx;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    int ri = fArg[o * fInner + i];
                    inGrad[(o * fReduce + ri) * fInner + i] = self.grad[o * fInner + i];
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(result, reducedShape(d, keepdim), List.of(this), bw, isMax ? "max" : "min");
    }

    @Override
    public IDiffTensor prod(int dim, boolean keepdim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.prod(d, keepdim));
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] vals = value.toDoubleArray();
        double[] result = new double[outer * inner];
        Arrays.fill(result, 1.0);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                for (int r = 0; r < reduce; r++) {
                    result[o * inner + i] *= vals[(o * reduce + r) * inner + i];
                }
            }
        }
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] savedVals = vals;
        double[] savedResult = result;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double prodVal = savedResult[o * fInner + i];
                    double gi = self.grad[o * fInner + i];
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        double xi = savedVals[idx];
                        inGrad[idx] = (xi != 0.0) ? gi * prodVal / xi : 0.0;
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(result, reducedShape(d, keepdim), List.of(this), bw, "prod");
    }

    @Override
    public IDiffTensor std(int dim, boolean keepdim) {
        return var(dim, keepdim).sqrt();
    }

    @Override
    public IDiffTensor var(int dim, boolean keepdim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.var(d, keepdim));
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] vals = value.toDoubleArray();
        // Compute means
        double[] means = new double[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double sum = 0;
                for (int r = 0; r < reduce; r++) sum += vals[(o * reduce + r) * inner + i];
                means[o * inner + i] = sum / reduce;
            }
        }
        double[] varData = new double[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double sumSq = 0;
                for (int r = 0; r < reduce; r++) {
                    double diff = vals[(o * reduce + r) * inner + i] - means[o * inner + i];
                    sumSq += diff * diff;
                }
                varData[o * inner + i] = sumSq / reduce;
            }
        }
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] fMeans = means;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double m = fMeans[o * fInner + i];
                    double scale = 2.0 * self.grad[o * fInner + i] / fReduce;
                    for (int r = 0; r < fReduce; r++) {
                        inGrad[(o * fReduce + r) * fInner + i] = scale * (vals[(o * fReduce + r) * fInner + i] - m);
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(varData, reducedShape(d, keepdim), List.of(this), bw, "var");
    }

    // ==================== Full reductions ====================

    @Override public double sumAll() { return value.sumAll(); }
    @Override public double meanAll() { return value.meanAll(); }
    @Override public double maxAll() { return value.maxAll(); }
    @Override public double minAll() { return value.minAll(); }
    @Override public double prodAll() { return value.prodAll(); }

    // ==================== Softmax / logSoftmax ====================

    @Override
    public IDiffTensor softmax(int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.softmax(d));
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] inData = value.toDoubleArray();
        double[] out = new double[inData.length];
        // Compute softmax: out = exp(x - max) / sum(exp(x - max))
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double maxVal = Double.NEGATIVE_INFINITY;
                for (int r = 0; r < reduce; r++) {
                    double v = inData[(o * reduce + r) * inner + i];
                    if (v > maxVal) maxVal = v;
                }
                double sumExp = 0;
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    out[idx] = Math.exp(inData[idx] - maxVal);
                    sumExp += out[idx];
                }
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    out[idx] /= sumExp;
                }
            }
        }
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] sm = out.clone();
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double dot = 0;
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        dot += self.grad[idx] * sm[idx];
                    }
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        inGrad[idx] = sm[idx] * (self.grad[idx] - dot);
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(out, s, List.of(this), bw, "softmax");
    }

    @Override
    public IDiffTensor logSoftmax(int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.logSoftmax(d));
        IDiffTensor sm = softmax(d);
        if (!(sm instanceof RereDiffTensor rsm)) return toNonDiff(value.logSoftmax(d));

        // Compute log(sm) forward, with proper backward
        double[] smData = rsm.value.toDoubleArray();
        double[] logData = new double[smData.length];
        for (int i = 0; i < smData.length; i++) logData[i] = Math.log(smData[i]);

        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] fSmData = smData;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double gSum = 0;
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        gSum += self.grad[idx];
                    }
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        inGrad[idx] = self.grad[idx] - fSmData[idx] * gSum;
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(logData, s, List.of(this), bw, "logSoftmax");
    }

    @Override
    public IDiffTensor softmaxCrossEntropy(IDoubleTensor labels, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int[] s = shape();
        int r = rank();
        int outerSize = 1;
        for (int i = 0; i < d; i++) outerSize *= s[i];
        int classSize = s[d];
        int innerSize = 1;
        for (int i = d + 1; i < r; i++) innerSize *= s[i];
        int totalSamples = outerSize * innerSize;

        double[] logits = value.toDoubleArray();
        double[] labelData = labels instanceof RereDiffTensor rl ? rl.value.toDoubleArray() : labels.toDoubleArray();

        double[] softmax = new double[logits.length];
        double totalLoss = 0;
        for (int o = 0; o < outerSize; o++) {
            for (int in = 0; in < innerSize; in++) {
                int base = (o * classSize) * innerSize + in;
                double mx = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < classSize; c++) {
                    double v = logits[base + c * innerSize];
                    if (v > mx) mx = v;
                }
                double sumExp = 0;
                for (int c = 0; c < classSize; c++) {
                    double ex = Math.exp(logits[base + c * innerSize] - mx);
                    softmax[base + c * innerSize] = ex;
                    sumExp += ex;
                }
                double invSum = 1.0 / sumExp;
                for (int c = 0; c < classSize; c++) {
                    softmax[base + c * innerSize] *= invSum;
                    double p = softmax[base + c * innerSize];
                    double y = labelData[base + c * innerSize];
                    totalLoss += -y * Math.log(p);
                }
            }
        }
        double meanLoss = totalLoss / totalSamples;
        double[] resultData = new double[] { meanLoss };
        int[] resultShape = new int[] { 1 };

        if (!requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(resultData, resultShape));

        int fOuter = outerSize, fClassSize = classSize, fInner = innerSize, fTotal = totalSamples;
        double[] fSoftmax = softmax;
        double[] fLabelData = labelData;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double gradScale = self.grad[0] / fTotal;
            int m = fSoftmax.length;
            double[] inGrad = AutodiffBufferPool.acquire(m);
            for (int i = 0; i < m; i++) {
                inGrad[i] = gradScale * (fSoftmax[i] - fLabelData[i]);
            }
            input.accGradFromPooled(inGrad, m);
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this), bw, "softmaxCrossEntropy");
    }

    // ==================== cumsum / cumprod ====================

    @Override
    public IDiffTensor cumsum(int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.cumsum(d));
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] vals = value.toDoubleArray();
        double[] result = new double[vals.length];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double sum = 0;
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    sum += vals[idx];
                    result[idx] = sum;
                }
            }
        }
        int fOuter = outer, fReduce = reduce, fInner = inner;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double cum = 0;
                    for (int r = fReduce - 1; r >= 0; r--) {
                        int idx = (o * fReduce + r) * fInner + i;
                        cum += self.grad[idx];
                        inGrad[idx] = cum;
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(result, s, List.of(this), bw, "cumsum");
    }

    @Override
    public IDiffTensor cumprod(int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.cumprod(d));
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] vals = value.toDoubleArray();
        double[] result = new double[vals.length];
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double prod = 1;
                for (int r = 0; r < reduce; r++) {
                    int idx = (o * reduce + r) * inner + i;
                    prod *= vals[idx];
                    result[idx] = prod;
                }
            }
        }
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] savedVals = vals;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int i = 0; i < fInner; i++) {
                    double[] cp = new double[fReduce];
                    double p = 1;
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        p *= savedVals[idx];
                        cp[r] = p;
                    }
                    double[] q = new double[fReduce];
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + i;
                        double xi = savedVals[idx];
                        q[r] = (xi != 0.0) ? self.grad[idx] * cp[r] / xi : 0.0;
                    }
                    double cum = 0;
                    for (int r = fReduce - 1; r >= 0; r--) {
                        cum += q[r];
                        inGrad[(o * fReduce + r) * fInner + i] = cum;
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(result, s, List.of(this), bw, "cumprod");
    }

    // ==================== argmax / argmin ====================

    @Override
    public IDiffTensor argmax(int dim) {
        IDoubleTensor r = value.argmax(dim);
        return toNonDiff(r);
    }

    @Override
    public IDiffTensor argmin(int dim) {
        IDoubleTensor r = value.argmin(dim);
        return toNonDiff(r);
    }

    // ==================== Matrix ops ====================

    @Override
    public IDiffTensor mmul(IDoubleTensor other) {
        if (!(other instanceof IDiffTensor otherDiff) || !requiresGrad || !otherDiff.requiresGrad()) {
            IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
            return toNonDiff(value.mmul(detOther));
        }
        if (rank() != 2 || other.rank() != 2) {
            return toNonDiff(value.mmul(((IDiffTensor) other).detach()));
        }
        int M = dim(0), K = dim(1), N = other.dim(1);
        if (K != other.dim(0)) {
            throw new IllegalArgumentException("mmul: shape mismatch " + M + "x" + K + " @ " + other.dim(0) + "x" + N);
        }
        double[] aData = value.toDoubleArray();
        double[] bData = ((RereDiffTensor) otherDiff).value.toDoubleArray();
        double[] resultData = DoubleFlatGemm.flatMmul(aData, M, K, bData, N);
        int[] resultShape = {M, N};

        double[] bT = DoubleFlatGemm.flatTranspose(bData, K, N);
        double[] aT = DoubleFlatGemm.flatTranspose(aData, M, K);
        RereDiffTensor otherNode = (RereDiffTensor) otherDiff;
        boolean selfContig = isContiguous();
        boolean otherContig = otherDiff.isContiguous();
        int[] selfStrides = selfContig ? null : strides();
        int[] otherStrides = otherContig ? null : ((RereDiffTensor) otherDiff).strides();

        int fM = M, fK = K, fN = N;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpA = self.inputs.get(0);
            RereDiffTensor inpB = self.inputs.get(1);
            // dA = dC @ B^T
            double[] dA = DoubleFlatGemm.flatMmul(self.grad, fM, fN, bT, fK);
            if (selfContig) {
                inpA.accGrad(dA);
            } else {
                double[] dARemap = new double[fM * fK];
                for (int i = 0; i < fM; i++) {
                    for (int k = 0; k < fK; k++) {
                        dARemap[i * selfStrides[0] + k * selfStrides[1]] = dA[i * fK + k];
                    }
                }
                inpA.accGrad(dARemap);
            }
            // dB = A^T @ dC
            double[] dB = DoubleFlatGemm.flatMmul(aT, fK, fM, self.grad, fN);
            if (otherContig) {
                inpB.accGrad(dB);
            } else {
                double[] dBRemap = new double[fK * fN];
                for (int k = 0; k < fK; k++) {
                    for (int n = 0; n < fN; n++) {
                        dBRemap[k * otherStrides[0] + n * otherStrides[1]] = dB[k * fN + n];
                    }
                }
                inpB.accGrad(dBRemap);
            }
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this, otherNode), bw, "mmul");
    }

    @Override
    public IDiffTensor bmm(IDoubleTensor other) {
        if (!(other instanceof IDiffTensor otherDiff) || !requiresGrad || !otherDiff.requiresGrad()) {
            IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
            return toNonDiff(value.bmm(detOther));
        }
        if (rank() != 3 || other.rank() != 3) return toNonDiff(value.bmm(((IDiffTensor) other).detach()));
        int B = dim(0), M = dim(1), K = dim(2);
        int B2 = other.dim(0), K2 = other.dim(1), N = other.dim(2);
        if (B != B2 || K != K2) {
            throw new IllegalArgumentException("bmm: shape mismatch");
        }
        int[] resultShape = {B, M, N};
        double[] aData = value.toDoubleArray();
        double[] bData = ((RereDiffTensor) otherDiff).value.toDoubleArray();
        double[] resultData = DoubleFlatGemm.flatMmulBatched(aData, bData, B, M, K, N);

        double[][] bT_slices = new double[B][];
        double[][] aT_slices = new double[B][];
        for (int bi = 0; bi < B; bi++) {
            int aOff = bi * M * K, bOff = bi * K * N;
            double[] bSlice = Arrays.copyOfRange(bData, bOff, bOff + K * N);
            double[] aSlice = Arrays.copyOfRange(aData, aOff, aOff + M * K);
            bT_slices[bi] = DoubleFlatGemm.flatTranspose(bSlice, K, N);
            aT_slices[bi] = DoubleFlatGemm.flatTranspose(aSlice, M, K);
        }
        RereDiffTensor otherNode = (RereDiffTensor) otherDiff;
        int fB = B, fM = M, fK = K, fN = N;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpA = self.inputs.get(0);
            RereDiffTensor inpB = self.inputs.get(1);
            int aStride = fM * fK, bStride = fK * fN, gStride = fM * fN;
            double[] dA = new double[fB * aStride];
            double[] dB = new double[fB * bStride];
            for (int bi = 0; bi < fB; bi++) {
                int aOff = bi * aStride, bOff = bi * bStride, gOff = bi * gStride;
                double[] dASlice = DoubleFlatGemm.flatMmul(self.grad, gOff, fM, fN, bT_slices[bi], 0, fK);
                System.arraycopy(dASlice, 0, dA, aOff, aStride);
                double[] dBSlice = DoubleFlatGemm.flatMmul(aT_slices[bi], 0, fK, fM, self.grad, gOff, fN);
                System.arraycopy(dBSlice, 0, dB, bOff, bStride);
            }
            inpA.accGrad(dA);
            inpB.accGrad(dB);
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this, otherNode), bw, "bmm");
    }

    @Override
    public IDiffTensor einsum(String subscript, IDoubleTensor... others) {
        if (others.length == 0) {
            return einsumSingle(subscript);
        }
        // Non-differentiable path: delegate to linalg einsum
        if (!requiresGrad || !(others[0] instanceof IDiffTensor od && od.requiresGrad())) {
            IDoubleTensor[] detOthers = new IDoubleTensor[others.length];
            for (int i = 0; i < others.length; i++) {
                detOthers[i] = (others[i] instanceof IDiffTensor dt) ? dt.detach() : others[i];
            }
            return toNonDiff(value.einsum(subscript, detOthers));
        }
        IDiffTensor other = (IDiffTensor) others[0];
        if (others.length > 1) {
            throw new UnsupportedOperationException(
                "einsum with >2 inputs not yet supported: " + subscript);
        }
        return einsumPair(subscript, other);
    }

    private IDiffTensor einsumPair(String subscript, IDiffTensor other) {
        EinsumParser.EinsumSpec spec = EinsumParser.parse(subscript, shape(), other.shape());

        if (spec.contractAxes.isEmpty()) {
            // No contraction: treat as element-wise multiply with broadcasting
            return mul(other);
        }

        // Compositional approach: permute → reshape → bmm → reshape
        IDiffTensor aP = permute(spec.permuteA());
        IDiffTensor bP = other.permute(spec.permuteB());

        IDiffTensor aR = aP.reshape(spec.reshapeTo3D(0, aP.shape()));
        IDiffTensor bR = bP.reshape(spec.reshapeTo3D(1, bP.shape()));

        IDiffTensor result = aR.bmm(bR);

        int[] outShape = spec.outputShape(shape(), other.shape());
        return result.reshape(outShape);
    }

    private IDiffTensor einsumSingle(String subscript) {
        EinsumParser.EinsumSpec spec = EinsumParser.parse(subscript, shape());

        if (!spec.contractAxes.isEmpty()) {
            // Summation or trace: sum over contract dims
            IDiffTensor result = this;
            // Sum over contract axes (highest dim first to avoid index shifts)
            int[] sortedDims = spec.contractAxes.stream()
                .mapToInt(c -> spec.inputLabels[0].indexOf(c))
                .sorted().toArray();
            for (int i = sortedDims.length - 1; i >= 0; i--) {
                result = result.sum(sortedDims[i], spec.outputLabels.indexOf(spec.inputLabels[0].charAt(sortedDims[i])) >= 0);
            }
            // If output is scalar (empty), squeeze remaining dims
            if (spec.outputLabels.isEmpty()) {
                while (result.rank() > 1) result = result.sum(0, false);
            }
            return result;
        }

        // Pure permutation
        int[] perm = new int[spec.inputLabels[0].length()];
        for (int i = 0; i < spec.outputLabels.length(); i++) {
            char c = spec.outputLabels.charAt(i);
            perm[i] = spec.inputLabels[0].indexOf(c);
        }
        return permute(perm);
    }

    // ==================== Advanced ops ====================

    @Override
    public IDiffTensor gather(int dim, IDoubleTensor index) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.gather(d, index instanceof IDiffTensor ? ((IDiffTensor) index).detach() : index));
        int[] s = shape();
        int r = rank();
        int idxRank = index.rank();
        int[] idxShape = index.shape();
        int trailingRank = r - d - 1;
        int[] resultShape = new int[idxRank + trailingRank];
        System.arraycopy(idxShape, 0, resultShape, 0, idxRank);
        for (int i = 0; i < trailingRank; i++) resultShape[idxRank + i] = s[d + 1 + i];
        int resultTotal = (int) computeSize(resultShape);

        double[] inData = value.toDoubleArray();
        double[] resultData = new double[resultTotal];
        int[] gatherIndices = new int[resultTotal];
        for (int i = 0; i < resultTotal; i++) {
            int[] outIdx = unlinearizeInt(i, resultShape);
            int[] idxIdx = new int[idxRank];
            System.arraycopy(outIdx, 0, idxIdx, 0, idxRank);
            int gatherIdx = (int) index.get(idxIdx);
            gatherIndices[i] = gatherIdx;
            int[] srcIdx = new int[r];
            for (int j = 0; j < d; j++) srcIdx[j] = outIdx[j];
            srcIdx[d] = gatherIdx;
            for (int j = 0; j < trailingRank; j++) srcIdx[d + 1 + j] = outIdx[idxRank + j];
            resultData[i] = inData[flatIndex(srcIdx, s)];
        }
        int fD = d, fIdxRank = idxRank, fTrailingRank = trailingRank;
        int[] fSrcShape = s, fResultShape = resultShape;
        int[] fGatherIndices = gatherIndices;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = new double[(int) computeSize(fSrcShape)];
            for (int i = 0; i < self.grad.length; i++) {
                int[] outIdx = unlinearizeInt(i, fResultShape);
                int[] srcIdx = new int[fSrcShape.length];
                for (int j = 0; j < fD; j++) srcIdx[j] = outIdx[j];
                srcIdx[fD] = fGatherIndices[i];
                for (int j = 0; j < fTrailingRank; j++) srcIdx[fD + 1 + j] = outIdx[fIdxRank + j];
                inGrad[flatIndex(srcIdx, fSrcShape)] += self.grad[i];
            }
            input.accGrad(inGrad);
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this), bw, "gather");
    }

    @Override
    public IDiffTensor indexSelect(int dim, IDoubleTensor index) {
        return gather(dim, index);
    }

    @Override
    public IDiffTensor argsort(int dim, boolean descending) {
        // Non-differentiable
        int d = (dim < 0 ? dim + rank() : dim);
        int[] s = shape();
        int r = rank();
        int dimSize = s[d];
        int outerTotal = 1;
        for (int i = 0; i < d; i++) outerTotal *= s[i];
        int innerTotal = 1;
        for (int i = d + 1; i < r; i++) innerTotal *= s[i];

        double[] inData = value.toDoubleArray();
        double[] outData = new double[inData.length];
        for (int outer = 0; outer < outerTotal; outer++) {
            for (int inner = 0; inner < innerTotal; inner++) {
                double[] sliceVals = new double[dimSize];
                int[] origIdx = new int[dimSize];
                for (int i = 0; i < dimSize; i++) {
                    sliceVals[i] = inData[(outer * dimSize + i) * innerTotal + inner];
                    origIdx[i] = i;
                }
                for (int i = 1; i < dimSize; i++) {
                    int keyIdx = origIdx[i];
                    double keyVal = sliceVals[i];
                    int j = i - 1;
                    while (j >= 0 && (descending ? sliceVals[j] < keyVal : sliceVals[j] > keyVal)) {
                        origIdx[j + 1] = origIdx[j];
                        sliceVals[j + 1] = sliceVals[j];
                        j--;
                    }
                    origIdx[j + 1] = keyIdx;
                    sliceVals[j + 1] = keyVal;
                }
                for (int i = 0; i < dimSize; i++) {
                    outData[(outer * dimSize + i) * innerTotal + inner] = origIdx[i];
                }
            }
        }
        return toNonDiff(new RereDoubleTensor(outData, s));
    }

    @Override
    public IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) {
            IDoubleTensor detSrc = (source instanceof IDiffTensor ds) ? ds.detach() : source;
            IDoubleTensor detIdx = (index instanceof IDiffTensor di) ? di.detach() : index;
            return toNonDiff(value.scatter(d, detIdx, detSrc));
        }
        int[] resultShape = shape();
        int r = rank();
        double[] resultData = value.toDoubleArray();
        double[] srcData = source instanceof RereDiffTensor rdt ? rdt.value.toDoubleArray() : source.toDoubleArray();
        int[] srcShape = source.shape();
        int idxRank = index.rank();
        int[] idxShape = index.shape();
        int idxTotal = (int) computeSize(idxShape);

        int[] scatterSrcFlat = new int[idxTotal];
        int[] scatterTgtFlat = new int[idxTotal];
        for (int i = 0; i < idxTotal; i++) {
            int[] idx = unlinearizeInt(i, idxShape);
            int scatterIdx = (int) index.get(idx);
            int[] tgtIdx = new int[r];
            for (int j = 0; j < r; j++) tgtIdx[j] = j == d ? scatterIdx : idx[j];
            int tgtFlat = flatIndex(tgtIdx, resultShape);
            int srcFlat = flatIndex(idx, srcShape);
            resultData[tgtFlat] = srcData[srcFlat];
            scatterSrcFlat[i] = srcFlat;
            scatterTgtFlat[i] = tgtFlat;
        }

        List<RereDiffTensor> inputs = new ArrayList<>();
        inputs.add(this);
        RereDiffTensor srcNode = (source instanceof RereDiffTensor rdt2 && rdt2.requiresGrad) ? rdt2 : null;
        if (srcNode != null) inputs.add(srcNode);

        int fIdxTotal = idxTotal;
        int[] fScatterSrcFlat = scatterSrcFlat;
        int[] fScatterTgtFlat = scatterTgtFlat;
        int[] fSrcShape = srcShape;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpSelf = self.inputs.get(0);
            // Self gradient: pass through except at overwritten positions
            double[] dxSelf = self.grad.clone();
            for (int i = 0; i < fIdxTotal; i++) dxSelf[fScatterTgtFlat[i]] = 0.0;
            inpSelf.accGrad(dxSelf);
            // Source gradient
            if (srcNode != null) {
                RereDiffTensor inpSrc = self.inputs.get(1);
                double[] dxSrc = new double[(int) computeSize(fSrcShape)];
                for (int i = 0; i < fIdxTotal; i++) dxSrc[fScatterSrcFlat[i]] += self.grad[fScatterTgtFlat[i]];
                inpSrc.accGrad(dxSrc);
            }
        };
        return new RereDiffTensor(resultData, resultShape, inputs, bw, "scatter");
    }

    @Override
    public IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) {
            IDoubleTensor detSrc = (source instanceof IDiffTensor ds) ? ds.detach() : source;
            IDoubleTensor detIdx = (index instanceof IDiffTensor di) ? di.detach() : index;
            return toNonDiff(value.scatterAdd(d, detIdx, detSrc));
        }
        int[] resultShape = shape();
        int r = rank();
        double[] resultData = value.toDoubleArray();
        double[] srcData = source instanceof RereDiffTensor rdt ? rdt.value.toDoubleArray() : source.toDoubleArray();
        int[] srcShape = source.shape();
        int idxRank = index.rank();
        int[] idxShape = index.shape();
        int idxTotal = (int) computeSize(idxShape);

        int[] scatterSrcFlat = new int[idxTotal];
        int[] scatterTgtFlat = new int[idxTotal];
        for (int i = 0; i < idxTotal; i++) {
            int[] idx = unlinearizeInt(i, idxShape);
            int scatterIdx = (int) index.get(idx);
            int[] tgtIdx = new int[r];
            for (int j = 0; j < r; j++) tgtIdx[j] = j == d ? scatterIdx : idx[j];
            int tgtFlat = flatIndex(tgtIdx, resultShape);
            int srcFlat = flatIndex(idx, srcShape);
            resultData[tgtFlat] += srcData[srcFlat];
            scatterSrcFlat[i] = srcFlat;
            scatterTgtFlat[i] = tgtFlat;
        }

        List<RereDiffTensor> inputs = new ArrayList<>();
        inputs.add(this);
        RereDiffTensor srcNode = (source instanceof RereDiffTensor rdt2 && rdt2.requiresGrad) ? rdt2 : null;
        if (srcNode != null) inputs.add(srcNode);

        int fIdxTotal = idxTotal;
        int[] fScatterSrcFlat = scatterSrcFlat;
        int[] fScatterTgtFlat = scatterTgtFlat;
        int[] fSrcShape = srcShape;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpSelf = self.inputs.get(0);
            inpSelf.accGrad(self.grad.clone());
            if (srcNode != null) {
                RereDiffTensor inpSrc = self.inputs.get(1);
                double[] dxSrc = new double[(int) computeSize(fSrcShape)];
                for (int i = 0; i < fIdxTotal; i++) dxSrc[fScatterSrcFlat[i]] += self.grad[fScatterTgtFlat[i]];
                inpSrc.accGrad(dxSrc);
            }
        };
        return new RereDiffTensor(resultData, resultShape, inputs, bw, "scatterAdd");
    }

    @Override
    public IDiffTensor where(IDoubleTensor condition, IDoubleTensor other) {
        if (!requiresGrad) {
            IDoubleTensor detCond = (condition instanceof IDiffTensor dc) ? dc.detach() : condition;
            IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
            return toNonDiff(value.where(detCond, detOther));
        }
        IDoubleTensor detCond = (condition instanceof IDiffTensor dc) ? dc.detach() : condition;
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        int[] sSelf = shape(), sOther = detOther.shape(), sCond = detCond.shape();
        int[] resultShape = TensorShape.broadcastShape(sSelf, TensorShape.broadcastShape(sOther, sCond));
        long resultTotal = 1;
        for (int d : resultShape) resultTotal *= d;
        int n = (int) resultTotal;

        double[] aData = value.toDoubleArray();
        double[] bData = detOther.toDoubleArray();
        double[] condData = detCond.toDoubleArray();
        double[] resultData = new double[n];
        boolean[] condMask = new boolean[n];
        for (int i = 0; i < n; i++) {
            int[] idx = unlinearizeInt(i, resultShape);
            double condVal = broadcastGetFlat(idx, condData, sCond, resultShape);
            condMask[i] = condVal > 0.5;
            resultData[i] = condMask[i] ? broadcastGetFlat(idx, aData, sSelf, resultShape)
                                        : broadcastGetFlat(idx, bData, sOther, resultShape);
        }

        List<RereDiffTensor> inputs = new ArrayList<>();
        inputs.add(this);
        RereDiffTensor otherNode = (other instanceof RereDiffTensor rt && rt.requiresGrad) ? rt : null;
        if (otherNode != null) inputs.add(otherNode);

        int[] fResultShape = resultShape;
        int[] fSSelf = sSelf, fSOther = sOther;
        boolean[] fCondMask = condMask;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpSelf = self.inputs.get(0);
            double[] dxSelf = new double[(int) computeSize(fSSelf)];
            double[] dxOther = null;
            RereDiffTensor inpOther = null;
            if (otherNode != null) {
                inpOther = self.inputs.get(1);
                dxOther = new double[(int) computeSize(fSOther)];
            }
            for (int i = 0; i < n; i++) {
                int[] idx = unlinearizeInt(i, fResultShape);
                if (fCondMask[i]) {
                    dxSelf[flatIndexFromBroadcast(idx, fSSelf, fResultShape)] += self.grad[i];
                } else if (dxOther != null) {
                    dxOther[flatIndexFromBroadcast(idx, fSOther, fResultShape)] += self.grad[i];
                }
            }
            inpSelf.accGrad(dxSelf);
            if (inpOther != null) inpOther.accGrad(dxOther);
        };
        return new RereDiffTensor(resultData, resultShape, inputs, bw, "where");
    }

    @Override
    public IDiffTensor topk(int k, int dim, boolean largest) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.topk(k, d, largest));
        int[] s = shape();
        int n = s[d];
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];
        int[] resultShape = s.clone();
        resultShape[d] = k;
        int resultSize = outer * k * inner;
        double[] resultData = new double[resultSize];
        int[] argIdx = new int[resultSize];

        double[] inData = value.toDoubleArray();
        for (int o = 0; o < outer; o++) {
            for (int ii = 0; ii < inner; ii++) {
                double[] vals = new double[n];
                for (int r = 0; r < n; r++) vals[r] = inData[(o * n + r) * inner + ii];
                Integer[] idxs = new Integer[n];
                for (int r = 0; r < n; r++) idxs[r] = r;
                final boolean l = largest;
                Arrays.sort(idxs, (a, b) -> l ? Double.compare(vals[b], vals[a]) : Double.compare(vals[a], vals[b]));
                for (int r = 0; r < k; r++) {
                    int flatIdx = (o * k + r) * inner + ii;
                    resultData[flatIdx] = vals[idxs[r]];
                    argIdx[flatIdx] = idxs[r];
                }
            }
        }
        int fOuter = outer, fReduce = n, fInner = inner, fK = k;
        int[] fArgIdx = argIdx;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = new double[fOuter * fReduce * fInner];
            for (int o = 0; o < fOuter; o++) {
                for (int r = 0; r < fK; r++) {
                    for (int ii = 0; ii < fInner; ii++) {
                        int outIdx = (o * fK + r) * fInner + ii;
                        int origR = fArgIdx[outIdx];
                        inGrad[(o * fReduce + origR) * fInner + ii] += self.grad[outIdx];
                    }
                }
            }
            input.accGrad(inGrad);
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this), bw, "topk");
    }

    @Override
    public IDiffTensor pad(int[][] padding, String mode, double padValue) {
        if (!requiresGrad) return toNonDiff(this.value.pad(padding, mode, padValue));
        int[] s = shape();
        int r = rank();
        int[] resultShape = new int[r];
        for (int i = 0; i < r; i++) resultShape[i] = s[i] + padding[i][0] + padding[i][1];
        long total = 1;
        for (int d : resultShape) total *= d;
        int n = (int) total;
        double[] resultData = new double[n];
        Arrays.fill(resultData, padValue);
        double[] inData = this.value.toDoubleArray();
        for (int i = 0; i < inData.length; i++) {
            int[] srcIdx = unlinearizeInt(i, s);
            int[] tgtIdx = new int[r];
            for (int j = 0; j < r; j++) tgtIdx[j] = srcIdx[j] + padding[j][0];
            resultData[flatIndex(tgtIdx, resultShape)] = inData[i];
        }
        int[] fResultShape = resultShape;
        int[] fOrigShape = s;
        int[][] fPadding = padding;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int origTotal = (int) computeSize(fOrigShape);
            double[] inGrad = new double[origTotal];
            for (int i = 0; i < origTotal; i++) {
                int[] srcIdx = unlinearizeInt(i, fOrigShape);
                int[] paddedIdx = new int[r];
                for (int j = 0; j < r; j++) paddedIdx[j] = srcIdx[j] + fPadding[j][0];
                inGrad[i] = self.grad[flatIndex(paddedIdx, fResultShape)];
            }
            input.accGrad(inGrad);
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this), bw, "pad");
    }

    @Override
    public IDiffTensor unfold(int dim, int size, int stride, int dilation) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.unfold(d, size, stride, dilation));
        int[] s = shape();
        int dimSize = s[d];
        int numPatches = (dimSize - dilation * (size - 1) - 1) / stride + 1;
        int outerElems = 1;
        for (int i = 0; i < d; i++) outerElems *= s[i];
        int innerElems = 1;
        for (int i = d + 1; i < rank(); i++) innerElems *= s[i];

        int[] resultShape = new int[rank() + 1];
        for (int i = 0; i < d; i++) resultShape[i] = s[i];
        resultShape[d] = numPatches;
        for (int i = d + 1; i < rank(); i++) resultShape[i] = s[i];
        resultShape[rank()] = size;

        double[] vals = value.toDoubleArray();
        double[] result = new double[outerElems * numPatches * innerElems * size];
        for (int o = 0; o < outerElems; o++) {
            for (int p = 0; p < numPatches; p++) {
                for (int ii = 0; ii < innerElems; ii++) {
                    for (int k = 0; k < size; k++) {
                        int srcIdx = (o * dimSize + (p * stride + k * dilation)) * innerElems + ii;
                        int dstIdx = ((o * numPatches + p) * innerElems + ii) * size + k;
                        result[dstIdx] = vals[srcIdx];
                    }
                }
            }
        }
        int fNumPatches = numPatches, fSize = size, fStride = stride, fDilation = dilation;
        int fOuterElems = outerElems, fDimSize = dimSize, fInnerElems = innerElems;
        int total = (int) value.totalSize();
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = new double[total];
            for (int o = 0; o < fOuterElems; o++) {
                for (int p = 0; p < fNumPatches; p++) {
                    for (int ii = 0; ii < fInnerElems; ii++) {
                        for (int k = 0; k < fSize; k++) {
                            int dstIdx = ((o * fNumPatches + p) * fInnerElems + ii) * fSize + k;
                            int srcIdx = (o * fDimSize + (p * fStride + k * fDilation)) * fInnerElems + ii;
                            inGrad[srcIdx] += self.grad[dstIdx];
                        }
                    }
                }
            }
            input.accGrad(inGrad);
        };
        return new RereDiffTensor(result, resultShape, List.of(this), bw, "unfold");
    }

    @Override public IDiffTensor nonzero() {
        IDoubleTensor r = value.nonzero();
        return toNonDiff(r);
    }

    @Override
    public IDiffTensor maskedSelect(IDoubleTensor mask) {
        if (!requiresGrad) return toNonDiff(value.maskedSelect(mask));
        int[] s = shape();
        int total = (int) value.totalSize();
        double[] inData = value.toDoubleArray();
        ArrayList<Double> selected = new ArrayList<>();
        ArrayList<Integer> selectedIndices = new ArrayList<>();
        for (int i = 0; i < total; i++) {
            int[] idx = unlinearizeInt(i, s);
            if (mask.get(idx) > 0.5) {
                selected.add(inData[i]);
                selectedIndices.add(i);
            }
        }
        int outLen = selected.size();
        double[] resultData = new double[outLen];
        for (int i = 0; i < outLen; i++) resultData[i] = selected.get(i);
        int[] fSelectedIdx = new int[outLen];
        for (int i = 0; i < outLen; i++) fSelectedIdx[i] = selectedIndices.get(i);
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = new double[total];
            for (int i = 0; i < self.grad.length; i++) inGrad[fSelectedIdx[i]] += self.grad[i];
            input.accGrad(inGrad);
        };
        return new RereDiffTensor(resultData, new int[]{outLen}, List.of(this), bw, "maskedSelect");
    }

    @Override
    public IDiffTensor maskedFill(IDoubleTensor mask, double fillValue) {
        if (!requiresGrad) return toNonDiff(value.maskedFill(mask, fillValue));
        int[] s = shape();
        int total = (int) value.totalSize();
        double[] inData = value.toDoubleArray();
        boolean[] maskArr = new boolean[total];
        for (int i = 0; i < total; i++) {
            int[] idx = unlinearizeInt(i, s);
            maskArr[i] = mask.get(idx) > 0.5;
        }
        double[] resultData = new double[total];
        for (int i = 0; i < total; i++) resultData[i] = maskArr[i] ? fillValue : inData[i];
        boolean[] fMask = maskArr;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = new double[self.grad.length];
            for (int i = 0; i < self.grad.length; i++) inGrad[i] = fMask[i] ? 0.0 : self.grad[i];
            input.accGrad(inGrad);
        };
        return new RereDiffTensor(resultData, s, List.of(this), bw, "maskedFill");
    }

    @Override
    public IDiffTensor cat(int dim, IDoubleTensor... others) {
        if (dim < 0) dim += rank();
        int d = dim;
        if (!requiresGrad) {
            IDoubleTensor[] detached = new IDoubleTensor[others.length];
            for (int i = 0; i < others.length; i++) detached[i] = others[i] instanceof IDiffTensor dt ? dt.detach() : others[i];
            return toNonDiff(value.cat(d, detached));
        }
        int[] resultShape = shape().clone();
        int[] sizes = new int[1 + others.length];
        sizes[0] = dim(d);
        for (int i = 0; i < others.length; i++) { sizes[i + 1] = others[i].dim(d); resultShape[d] += sizes[i + 1]; }

        long total = 1;
        for (int rs : resultShape) total *= rs;
        int totalSize = (int) total;
        double[] resultData = new double[totalSize];
        int[] shapeA = shape();
        double[] aData = value.toDoubleArray();
        int r = rank();
        int innerSize = 1;
        for (int j = d + 1; j < r; j++) innerSize *= shapeA[j];
        int aBlockSize = shapeA[d] * innerSize;
        int outerGroups = aData.length / aBlockSize;
        int resultBlockSize = resultShape[d] * innerSize;
        // Copy self data in bulk contiguous blocks
        for (int g = 0; g < outerGroups; g++) {
            System.arraycopy(aData, g * aBlockSize, resultData, g * resultBlockSize, aBlockSize);
        }
        // Copy other tensors in bulk
        int offsetInBlock = shapeA[d] * innerSize;
        for (int ti = 0; ti < others.length; ti++) {
            IDoubleTensor t = others[ti];
            double[] tData = t instanceof RereDiffTensor rdt3 ? rdt3.value.toDoubleArray() : t.toDoubleArray();
            int tBlockSize = t.dim(d) * innerSize;
            for (int g = 0; g < outerGroups; g++) {
                System.arraycopy(tData, g * tBlockSize, resultData, g * resultBlockSize + offsetInBlock, tBlockSize);
            }
            offsetInBlock += tBlockSize;
        }

        // Build gradient inputs
        List<RereDiffTensor> allInputs = new java.util.ArrayList<>();
        allInputs.add(this);
        List<int[]> allShapes = new java.util.ArrayList<>();
        allShapes.add(shape());
        for (IDoubleTensor other : others) {
            if (other instanceof RereDiffTensor rdt4 && rdt4.requiresGrad) {
                allInputs.add(rdt4);
                allShapes.add(other.shape());
            }
        }

        int[] fResultShape = resultShape;
        int[] cumOffset = new int[allInputs.size()];
        cumOffset[0] = 0;
        int off = sizes[0];
        int inpIdx = 1;
        for (int i = 0; i < others.length; i++) {
            if (others[i] instanceof RereDiffTensor rdt4 && rdt4.requiresGrad) {
                cumOffset[inpIdx] = off;
                inpIdx++;
            }
            off += sizes[i + 1];
        }

        int fDim = d;
        int[][] fAllShapes = allShapes.toArray(new int[0][]);
        int fInnerSize = innerSize;
        int fResultBlockSize = resultBlockSize;
        int fOuterGroups = outerGroups;
        Consumer<RereDiffTensor> bw = self -> {
            for (int ai = 0; ai < allInputs.size(); ai++) {
                RereDiffTensor inp = allInputs.get(ai);
                int[] tShape = fAllShapes[ai];
                int tFlat = (int) computeSize(tShape);
                int startOffset = cumOffset[ai];
                double[] subGrad = new double[tFlat];
                int tBlockSize = tShape[fDim] * fInnerSize;
                for (int g = 0; g < fOuterGroups; g++) {
                    System.arraycopy(self.grad, g * fResultBlockSize + startOffset,
                            subGrad, g * tBlockSize, tBlockSize);
                }
                inp.accGrad(subGrad);
            }
        };
        return new RereDiffTensor(resultData, resultShape, allInputs, bw, "cat");
    }

    @Override
    public IDiffTensor stack(int dim, IDoubleTensor... others) {
        if (!requiresGrad) {
            IDoubleTensor[] detached = new IDoubleTensor[others.length];
            for (int i = 0; i < others.length; i++) detached[i] = others[i] instanceof IDiffTensor dt ? dt.detach() : others[i];
            return toNonDiff(value.stack(dim, detached));
        }
        int d = (dim < 0 ? rank() + 1 + dim : dim);
        IDiffTensor[] all = new IDiffTensor[1 + others.length];
        all[0] = this.unsqueeze(d);
        for (int i = 0; i < others.length; i++) {
            if (others[i] instanceof IDiffTensor dt && !dt.requiresGrad()) {
                all[i + 1] = IDiffTensor.constantTensor(others[i].toDoubleArray(), others[i].shape()).unsqueeze(d);
            } else if (others[i] instanceof IDiffTensor dt) {
                all[i + 1] = dt.unsqueeze(d);
            } else {
                all[i + 1] = IDiffTensor.constantTensor(others[i].toDoubleArray(), others[i].shape()).unsqueeze(d);
            }
        }
        return all[0].cat(d, java.util.Arrays.copyOfRange(all, 1, all.length));
    }

    @Override
    public List<IDoubleTensor> unstack(int dim) {
        return value.unstack(dim);
    }

    @Override
    public IDiffTensor normalize(double p, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) return toNonDiff(value.normalize(p, d));
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] inData = value.toDoubleArray();
        double[] resultData = new double[inData.length];
        double[] normVals = new double[outer * inner];
        for (int o = 0; o < outer; o++) {
            for (int ii = 0; ii < inner; ii++) {
                double normP = 0;
                for (int r = 0; r < reduce; r++) normP += Math.pow(Math.abs(inData[(o * reduce + r) * inner + ii]), p);
                double norm = Math.pow(normP, 1.0 / p);
                normVals[o * inner + ii] = norm;
                if (norm > 0) {
                    for (int r = 0; r < reduce; r++) {
                        int idx = (o * reduce + r) * inner + ii;
                        resultData[idx] = inData[idx] / norm;
                    }
                }
            }
        }
        int fOuter = outer, fReduce = reduce, fInner = inner;
        double fP = p;
        double[] savedNorms = normVals;
        double[] savedIn = inData;
        double[] savedResult = resultData;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            double[] inGrad = new double[self.grad.length];
            for (int o = 0; o < fOuter; o++) {
                for (int ii = 0; ii < fInner; ii++) {
                    double norm = savedNorms[o * fInner + ii];
                    if (norm == 0) continue;
                    double dot = 0;
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + ii;
                        dot += self.grad[idx] * savedResult[idx];
                    }
                    for (int r = 0; r < fReduce; r++) {
                        int idx = (o * fReduce + r) * fInner + ii;
                        double xi = savedIn[idx];
                        if (fP == 2.0) {
                            inGrad[idx] = (self.grad[idx] - xi * dot / norm) / norm;
                        } else {
                            double signX = xi >= 0 ? 1.0 : -1.0;
                            inGrad[idx] = (self.grad[idx] - signX * Math.pow(Math.abs(xi), fP - 1) * dot / Math.pow(norm, fP - 1)) / norm;
                        }
                    }
                }
            }
            input.accGrad(inGrad);
        };
        return new RereDiffTensor(resultData, s, List.of(this), bw, "normalize");
    }

    // ==================== In-place ops ====================

    @Override
    public IDiffTensor add_(IDoubleTensor other) {
        return inPlaceBinaryOp(other, (a, b) -> a + b);
    }

    @Override
    public IDiffTensor sub_(IDoubleTensor other) {
        return inPlaceBinaryOp(other, (a, b) -> a - b);
    }

    @Override
    public IDiffTensor mul_(IDoubleTensor other) {
        return inPlaceBinaryOp(other, (a, b) -> a * b);
    }

    @Override
    public IDiffTensor div_(IDoubleTensor other) {
        return inPlaceBinaryOp(other, (a, b) -> a / b);
    }

    private IDiffTensor inPlaceBinaryOp(IDoubleTensor other, DoubleBinaryOperator op) {
        if (!isLeaf) throw new IllegalStateException("In-place ops only allowed on leaf tensors");
        IDoubleTensor detOther = (other instanceof IDiffTensor dt) ? dt.detach() : other;
        if (Arrays.equals(shape(), other.shape())) {
            long n = value.totalSize();
            double[] oData = detOther.toDoubleArray();
            for (long i = 0; i < n; i++) {
                value.linearSet(i, op.applyAsDouble(value.linearGet(i), oData[(int) i]));
            }
        } else {
            // Broadcast in-place
            int[] bc = TensorShape.broadcastShape(shape(), other.shape());
            long n = 1;
            for (int d : bc) n *= d;
            double[] oData = detOther.toDoubleArray();
            for (long i = 0; i < n; i++) {
                int[] bcIdx = unlinearizeInt((int) i, bc);
                int flatSelf = flatIndexFromBroadcast(bcIdx, shape(), bc);
                int flatOther = flatIndexFromBroadcast(bcIdx, other.shape(), bc);
                value.linearSet(flatSelf, op.applyAsDouble(value.linearGet(flatSelf), oData[flatOther]));
            }
        }
        this.grad = null;
        return this;
    }

    @Override
    public IDiffTensor fill_(double val) {
        if (!isLeaf) throw new IllegalStateException("fill_ only allowed on leaf tensors");
        value.fill(val);
        this.grad = null;
        return this;
    }

    @Override
    public IDiffTensor copy_(IDoubleTensor src) {
        if (!isLeaf) throw new IllegalStateException("copy_ only allowed on leaf tensors");
        IDoubleTensor detSrc = (src instanceof IDiffTensor dt) ? dt.detach() : src;
        double[] sData = detSrc.toDoubleArray();
        long n = value.totalSize();
        for (long i = 0; i < n && i < sData.length; i++) value.linearSet(i, sData[(int) i]);
        this.grad = null;
        return this;
    }

    // ==================== Conversion ====================

    @Override public double[] toDoubleArray() { return value.toDoubleArray(); }
    @Override public IMatrix toMatrix() { return value.toMatrix(); }
    @Override public IDoubleVector toVector() { return value.toVector(); }
    @Override public IDoubleVector toVectorCopy() { return value.toVectorCopy(); }

    @Override
    public String toString() {
        return "RereDiffTensor(shape=" + Arrays.toString(shape()) + ", requiresGrad=" + requiresGrad + ")";
    }

    // ==================== Utility helpers ====================

    private int[] reducedShape(int dim, boolean keepdim) {
        if (keepdim) {
            int[] r = shape().clone();
            r[dim] = 1;
            return r;
        }
        if (rank() == 1) return new int[]{1};
        int[] r = new int[rank() - 1];
        int idx = 0;
        for (int i = 0; i < rank(); i++) if (i != dim) r[idx++] = dim(i);
        return r;
    }

    private static long computeSize(int[] shape) {
        long size = 1;
        for (int d : shape) size *= d;
        return size;
    }

    private static int[] unlinearizeInt(int flat, int[] shape) {
        int[] idx = new int[shape.length];
        int remaining = flat;
        for (int j = shape.length - 1; j >= 0; j--) {
            idx[j] = remaining % shape[j];
            remaining /= shape[j];
        }
        return idx;
    }

    private static int flatIndex(int[] indices, int[] shape) {
        int idx = 0;
        int stride = 1;
        for (int j = shape.length - 1; j >= 0; j--) {
            idx += indices[j] * stride;
            stride *= shape[j];
        }
        return idx;
    }

    private static double broadcastGetFlat(int[] resultIdx, double[] srcData, int[] srcShape, int[] resultShape) {
        int diff = resultShape.length - srcShape.length;
        int srcFlat = 0;
        int srcStride = 1;
        for (int j = srcShape.length - 1; j >= 0; j--) {
            int coord = (diff + j < resultShape.length) ? resultIdx[diff + j] : 0;
            int srcCoord = (srcShape[j] == 1) ? 0 : coord;
            srcFlat += srcCoord * srcStride;
            srcStride *= srcShape[j];
        }
        return srcData[srcFlat];
    }

    private static int flatIndexFromBroadcast(int[] resultIdx, int[] srcShape, int[] resultShape) {
        int diff = resultShape.length - srcShape.length;
        int srcFlat = 0;
        int srcStride = 1;
        for (int j = srcShape.length - 1; j >= 0; j--) {
            int coord = (diff + j < resultShape.length) ? resultIdx[diff + j] : 0;
            int srcCoord = (srcShape[j] == 1) ? 0 : coord;
            srcFlat += srcCoord * srcStride;
            srcStride *= srcShape[j];
        }
        return srcFlat;
    }

    // ==================== Layer/Batch Normalization ====================

    @Override
    public IDiffTensor layerNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        RereDiffTensor gr = (RereDiffTensor) gamma;
        RereDiffTensor br = (RereDiffTensor) beta;
        long totalSize = value.totalSize();
        int features = value.dim(rank() - 1);
        if (totalSize % features != 0) {
            throw new IllegalArgumentException(
                "Input size (" + totalSize + ") not divisible by features (" + features + ")");
        }
        int batch = (int) (totalSize / features);

        double[] xd = value.toDoubleArray();
        double[] gd = gr.value.toDoubleArray();
        double[] bd = br.value.toDoubleArray();

        double[] y = new double[(int) totalSize];
        double[] xHat = new double[(int) totalSize];
        double[] means = new double[batch];
        double[] sigmas = new double[batch];

        for (int p = 0; p < batch; p++) {
            int off = p * features;
            double mean = 0;
            for (int j = 0; j < features; j++) mean += xd[off + j];
            mean /= features;
            means[p] = mean;
            double var = 0;
            for (int j = 0; j < features; j++) { double d = xd[off + j] - mean; var += d * d; }
            var /= features;
            double sigma = Math.sqrt(var + eps);
            sigmas[p] = sigma;
            for (int j = 0; j < features; j++) {
                xHat[off + j] = (xd[off + j] - mean) / sigma;
                y[off + j] = gd[j] * xHat[off + j] + bd[j];
            }
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpG = self.inputs.get(1);
            RereDiffTensor inpB = self.inputs.get(2);
            double[] g = self.grad;
            double[] cg = gr.value.toDoubleArray();
            int m = (int) inpX.value.totalSize();
            double[] dx = AutodiffBufferPool.acquire(m);
            double[] dGamma = AutodiffBufferPool.acquire(features);
            double[] dBeta = AutodiffBufferPool.acquire(features);
            for (int p = 0; p < batch; p++) {
                int off = p * features;
                double sigma = sigmas[p];
                double sumGT = 0, sumGTXH = 0;
                for (int j = 0; j < features; j++) {
                    double gt = g[off + j] * cg[j];
                    sumGT += gt;
                    sumGTXH += gt * xHat[off + j];
                }
                double invFS = 1.0 / (features * sigma);
                for (int j = 0; j < features; j++) {
                    double gt = g[off + j] * cg[j];
                    dx[off + j] = (features * gt - sumGT - xHat[off + j] * sumGTXH) * invFS;
                }
                for (int j = 0; j < features; j++) {
                    dGamma[j] += g[off + j] * xHat[off + j];
                    dBeta[j] += g[off + j];
                }
            }
            inpX.accGradFromPooled(dx, m);
            inpG.accGradFromPooled(dGamma, features);
            inpB.accGradFromPooled(dBeta, features);
        };
        RereDiffTensor result = new RereDiffTensor(y, shape(), List.of(this, gr, br), bw, "layerNorm");
        result.scalarParam = eps;
        return result;
    }

    @Override
    public IDiffTensor batchNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        RereDiffTensor gr = (RereDiffTensor) gamma;
        RereDiffTensor br = (RereDiffTensor) beta;
        long totalSize = value.totalSize();
        int features = value.dim(rank() - 1);
        if (totalSize % features != 0) {
            throw new IllegalArgumentException(
                "Input size (" + totalSize + ") not divisible by features (" + features + ")");
        }
        int batch = (int) (totalSize / features);

        double[] xd = value.toDoubleArray();
        double[] gd = gr.value.toDoubleArray();
        double[] bd = br.value.toDoubleArray();

        double[] y = new double[(int) totalSize];
        double[] means = new double[features];
        double[] invSigmas = new double[features];

        for (int j = 0; j < features; j++) {
            double mean = 0;
            for (int i = 0; i < batch; i++) mean += xd[i * features + j];
            mean /= batch;
            means[j] = mean;
            double var = 0;
            for (int i = 0; i < batch; i++) { double d = xd[i * features + j] - mean; var += d * d; }
            var /= batch;
            double invSigma = 1.0 / Math.sqrt(var + eps);
            invSigmas[j] = invSigma;
            for (int i = 0; i < batch; i++) {
                int idx = i * features + j;
                double xHat = (xd[idx] - mean) * invSigma;
                y[idx] = gd[j] * xHat + bd[j];
            }
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpG = self.inputs.get(1);
            RereDiffTensor inpB = self.inputs.get(2);
            double[] g = self.grad;
            double[] cg = gr.value.toDoubleArray();
            int m = (int) inpX.value.totalSize();
            double[] dx = AutodiffBufferPool.acquire(m);
            double[] dGamma = AutodiffBufferPool.acquire(features);
            double[] dBeta = AutodiffBufferPool.acquire(features);
            for (int j = 0; j < features; j++) {
                double invSig = invSigmas[j];
                double mean = means[j];
                double dg = 0, db = 0;
                for (int i = 0; i < batch; i++) {
                    int idx = i * features + j;
                    double xHat = (xd[idx] - mean) * invSig;
                    dg += g[idx] * xHat;
                    db += g[idx];
                }
                dGamma[j] = dg;
                dBeta[j] = db;
                double sumG = 0, sumGXHat = 0;
                for (int i = 0; i < batch; i++) {
                    int idx = i * features + j;
                    double xHat = (xd[idx] - mean) * invSig;
                    sumG += g[idx];
                    sumGXHat += g[idx] * xHat;
                }
                double scale = cg[j] * invSig / batch;
                for (int i = 0; i < batch; i++) {
                    int idx = i * features + j;
                    double xHat = (xd[idx] - mean) * invSig;
                    dx[idx] = scale * (batch * g[idx] - sumG - xHat * sumGXHat);
                }
            }
            inpX.accGradFromPooled(dx, m);
            inpG.accGradFromPooled(dGamma, features);
            inpB.accGradFromPooled(dBeta, features);
        };
        RereDiffTensor result = new RereDiffTensor(y, shape(), List.of(this, gr, br), bw, "batchNorm");
        result.scalarParam = eps;
        return result;
    }

    // ==================== Lightweight constant tensor ====================

    /**
     * Zero-overhead non-differentiable tensor wrapping a {@link RereDoubleTensor}.
     * All ops delegate to the underlying value; {@link #backward()} is a no-op.
     * Used by {@link #toNonDiff} to avoid allocating full graph nodes during eval/inference.
     */
    static final class ConstantDiffTensor implements IDiffTensor {
        final RereDoubleTensor value;

        ConstantDiffTensor(RereDoubleTensor value) { this.value = value; }

        // Shape delegation
        @Override public int rank() { return value.rank(); }
        @Override public int[] shape() { return value.shape(); }
        @Override public int dim(int axis) { return value.dim(axis); }
        @Override public long totalSize() { return value.totalSize(); }
        @Override public int[] strides() { return value.strides(); }
        @Override public int stride(int axis) { return value.stride(axis); }
        @Override public int offset() { return value.offset(); }
        @Override public boolean isContiguous() { return value.isContiguous(); }
        @Override public double item() { return value.item(); }
        @Override public double get(int... indices) { return value.get(indices); }
        @Override public ITensor set(double val, int... indices) { value.set(val, indices); return this; }
        @Override public ITensor fill(double val) { value.fill(val); return this; }
        @Override public IDoubleTensor copy() { return new RereDoubleTensor(value.toDoubleArray(), shape()); }
        @Override public double[] toDoubleArray() { return value.toDoubleArray(); }
        @Override public IMatrix toMatrix() { return value.toMatrix(); }
        @Override public IDoubleVector toVector() { return value.toVector(); }
        @Override public IDoubleVector toVectorCopy() { return value.toVectorCopy(); }

        // Non-diff ops delegate to value, wrap result
        @Override public IDiffTensor add(IDoubleTensor other) { return wrap(value.add(detachOther(other))); }
        @Override public IDiffTensor sub(IDoubleTensor other) { return wrap(value.sub(detachOther(other))); }
        @Override public IDiffTensor mul(IDoubleTensor other) { return wrap(value.mul(detachOther(other))); }
        @Override public IDiffTensor div(IDoubleTensor other) { return wrap(value.div(detachOther(other))); }
        @Override public IDiffTensor add(double scalar) { return wrap(value.add(scalar)); }
        @Override public IDiffTensor sub(double scalar) { return wrap(value.sub(scalar)); }
        @Override public IDiffTensor mul(double scalar) { return wrap(value.mul(scalar)); }
        @Override public IDiffTensor div(double scalar) { return wrap(value.div(scalar)); }
        @Override public IDiffTensor neg() { return wrap(value.neg()); }
        @Override public IDiffTensor abs() { return wrap(value.abs()); }
        @Override public IDiffTensor sqrt() { return wrap(value.sqrt()); }
        @Override public IDiffTensor exp() { return wrap(value.exp()); }
        @Override public IDiffTensor log() { return wrap(value.log()); }
        @Override public IDiffTensor sigmoid() { return wrap(value.sigmoid()); }
        @Override public IDiffTensor relu() { return wrap(value.relu()); }
        @Override public IDiffTensor square() { return wrap(value.square()); }
        @Override public IDiffTensor pow(double n) { return wrap(value.pow(n)); }
        @Override public IDiffTensor clamp(double min, double max) { return wrap(value.clamp(min, max)); }
        @Override public IDiffTensor sin() { return wrap(value.sin()); }
        @Override public IDiffTensor cos() { return wrap(value.cos()); }
        @Override public IDiffTensor tan() { return wrap(value.tan()); }
        @Override public IDiffTensor tanh() { return wrap(value.tanh()); }
        @Override public IDiffTensor silu() { return wrap(value.silu()); }
        @Override public IDiffTensor gelu() { return wrap(value.gelu()); }
        @Override public IDiffTensor softplus(double beta) { return wrap(value.softplus(beta)); }
        @Override public IDiffTensor mish() { return wrap(value.mish()); }
        @Override public IDiffTensor elu(double alpha) { return wrap(value.elu(alpha)); }
        @Override public IDiffTensor leakyRelu(double alpha) { return wrap(value.leakyRelu(alpha)); }
        @Override public IDiffTensor selu() { return wrap(value.selu()); }
        @Override public IDiffTensor hardtanh(double minVal, double maxVal) { return wrap(value.hardtanh(minVal, maxVal)); }
        @Override public IDiffTensor rsub(double scalar) { return wrap(value.neg().add(scalar)); }
        @Override public IDiffTensor rdiv(double scalar) { return wrap(value.pow(-1).mul(scalar)); }
        @Override public IDiffTensor reciprocal() { return wrap(value.pow(-1)); }
        @Override public IDiffTensor dropout(double p) { return this; }
        @Override public IDiffTensor sum(int dim, boolean keepdim) { return wrap(value.sum(dim, keepdim)); }
        @Override public IDiffTensor mean(int dim, boolean keepdim) { return wrap(value.mean(dim, keepdim)); }
        @Override public IDiffTensor max(int dim, boolean keepdim) { return wrap(value.max(dim, keepdim)); }
        @Override public IDiffTensor min(int dim, boolean keepdim) { return wrap(value.min(dim, keepdim)); }
        @Override public IDiffTensor prod(int dim, boolean keepdim) { return wrap(value.prod(dim, keepdim)); }
        @Override public IDiffTensor cumsum(int dim) { return wrap(value.cumsum(dim)); }
        @Override public IDiffTensor cumprod(int dim) { return wrap(value.cumprod(dim)); }
        @Override public IDiffTensor argmax(int dim) { return wrap(value.argmax(dim)); }
        @Override public IDiffTensor argmin(int dim) { return wrap(value.argmin(dim)); }
        @Override public IDiffTensor std(int dim, boolean keepdim) { return wrap(value.std(dim, keepdim)); }
        @Override public IDiffTensor var(int dim, boolean keepdim) { return wrap(value.var(dim, keepdim)); }
        @Override public IDiffTensor softmax(int dim) { return wrap(value.softmax(dim)); }
        @Override public IDiffTensor logSoftmax(int dim) { return wrap(value.logSoftmax(dim)); }
        @Override public IDiffTensor softmaxCrossEntropy(IDoubleTensor labels, int dim) {
            int d = (dim < 0 ? dim + value.rank() : dim);
            IDoubleTensor sm = value.softmax(d);
            double[] smData = sm.toDoubleArray();
            double[] lbData = labels.toDoubleArray();
            double totalLoss = 0;
            for (int i = 0; i < smData.length; i++) totalLoss += -lbData[i] * Math.log(smData[i]);
            int totalSamples = smData.length / value.dim(d);
            return wrap(new RereDoubleTensor(new double[]{totalLoss / totalSamples}, new int[]{1}));
        }
        @Override public IDiffTensor layerNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return computeNorm(gamma, beta, eps, true); }
        @Override public IDiffTensor batchNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return computeNorm(gamma, beta, eps, false); }
        private IDiffTensor computeNorm(IDiffTensor gamma, IDiffTensor beta, double eps, boolean overLastDim) {
            int features = overLastDim ? value.dim(value.rank() - 1) : value.dim(value.rank() - 1);
            long totalSize = value.totalSize();
            int batch = (int) (totalSize / features);
            double[] xd = value.toDoubleArray();
            double[] gd = gamma.toDoubleArray();
            double[] bd = beta.toDoubleArray();
            double[] y = new double[(int) totalSize];
            for (int p = 0; p < batch; p++) {
                int off = p * features;
                double mean = 0;
                for (int j = 0; j < features; j++) mean += xd[off + j];
                mean /= features;
                double var = 0;
                for (int j = 0; j < features; j++) { double d = xd[off + j] - mean; var += d * d; }
                var /= features;
                double inv = 1.0 / Math.sqrt(var + eps);
                for (int j = 0; j < features; j++) {
                    y[off + j] = gd[j] * (xd[off + j] - mean) * inv + bd[j];
                }
            }
            return wrap(new RereDoubleTensor(y, shape()));
        }
        @Override public IDiffTensor mmul(IDoubleTensor other) { return wrap(value.mmul(detachOther(other))); }
        @Override public IDiffTensor bmm(IDoubleTensor other) { return wrap(value.bmm(detachOther(other))); }
        @Override public IDiffTensor einsum(String subscript, IDoubleTensor... others) {
            IDoubleTensor[] detOthers = new IDoubleTensor[others.length];
            for (int i = 0; i < others.length; i++) {
                detOthers[i] = others[i] instanceof IDiffTensor dt ? dt.detach() : others[i];
            }
            return wrap(value.einsum(subscript, detOthers));
        }
        @Override public IDiffTensor reshape(int... newShape) { return wrap(value.reshape(newShape)); }
        @Override public IDiffTensor permute(int... dims) { return wrap(value.permute(dims)); }
        @Override public IDiffTensor transpose(int dim0, int dim1) { return wrap(value.transpose(dim0, dim1)); }
        @Override public IDiffTensor transpose() { return wrap(value.transpose()); }
        @Override public IDiffTensor squeeze(int... dims) { return wrap(value.squeeze(dims)); }
        @Override public IDiffTensor unsqueeze(int dim) { return wrap(value.unsqueeze(dim)); }
        @Override public IDiffTensor select(int dim, long index) { return wrap(value.select(dim, index)); }
        @Override public IDiffTensor flatten(int startDim, int endDim) { return wrap(value.flatten(startDim, endDim)); }
        @Override public IDiffTensor slice(int dim, long start, long end) { return wrap(value.slice(dim, start, end)); }
        @Override public IDiffTensor narrow(int dim, long start, long length) { return wrap(value.narrow(dim, start, length)); }
        @Override public IDiffTensor expand(int... shape) { return wrap(value.expand(shape)); }
        @Override public IDiffTensor contiguous() { return wrap(value.contiguous()); }
        @Override public IDiffTensor tile(int... repeats) { return wrap(value.tile(repeats)); }
        @Override public IDiffTensor broadcastTo(int... shape) { return wrap(value.broadcastTo(shape)); }
        @Override public IDiffTensor clone() { return new ConstantDiffTensor(new RereDoubleTensor(value.toDoubleArray(), shape())); }
        @Override public IDiffTensor gather(int dim, IDoubleTensor index) { return wrap(value.gather(dim, detachOther(index))); }
        @Override public IDiffTensor indexSelect(int dim, IDoubleTensor index) { return wrap(value.indexSelect(dim, detachOther(index))); }
        @Override public IDiffTensor argsort(int dim, boolean descending) { return wrap(value.argsort(dim, descending)); }
        @Override public IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) { return wrap(value.scatter(dim, detachOther(index), detachOther(source))); }
        @Override public IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) { return wrap(value.scatterAdd(dim, detachOther(index), detachOther(source))); }
        @Override public IDiffTensor where(IDoubleTensor condition, IDoubleTensor other) { return wrap(value.where(detachOther(condition), detachOther(other))); }
        @Override public IDiffTensor topk(int k, int dim, boolean largest) { return wrap(value.topk(k, dim, largest)); }
        @Override public IDiffTensor pad(int[][] padding, String mode, double padValue) { return wrap(value.pad(padding, mode, padValue)); }
        @Override public IDiffTensor unfold(int dim, int size, int stride, int dilation) { return wrap(value.unfold(dim, size, stride, dilation)); }
        @Override public IDiffTensor nonzero() { return wrap(value.nonzero()); }
        @Override public IDiffTensor maskedSelect(IDoubleTensor mask) { return wrap(value.maskedSelect(detachOther(mask))); }
        @Override public IDiffTensor maskedFill(IDoubleTensor mask, double fillValue) { return wrap(value.maskedFill(detachOther(mask), fillValue)); }
        @Override public IDiffTensor cat(int dim, IDoubleTensor... others) {
            IDoubleTensor[] d = new IDoubleTensor[others.length];
            for (int i = 0; i < others.length; i++) d[i] = detachOther(others[i]);
            return wrap(value.cat(dim, d));
        }
        @Override public IDiffTensor stack(int dim, IDoubleTensor... others) {
            IDoubleTensor[] d = new IDoubleTensor[others.length];
            for (int i = 0; i < others.length; i++) d[i] = detachOther(others[i]);
            return wrap(value.stack(dim, d));
        }
        @Override public List<IDoubleTensor> unstack(int dim) { return value.unstack(dim); }
        @Override public IDiffTensor normalize(double p, int dim) { return wrap(value.normalize(p, dim)); }

        // In-place ops
        @Override public IDiffTensor add_(IDoubleTensor other) { value.add_(detachOther(other)); return this; }
        @Override public IDiffTensor sub_(IDoubleTensor other) { value.sub_(detachOther(other)); return this; }
        @Override public IDiffTensor mul_(IDoubleTensor other) { value.mul_(detachOther(other)); return this; }
        @Override public IDiffTensor div_(IDoubleTensor other) { value.div_(detachOther(other)); return this; }
        @Override public IDiffTensor fill_(double val) { value.fill(val); return this; }
        @Override public IDiffTensor copy_(IDoubleTensor src) { value.copy_(detachOther(src)); return this; }

        // Non-diff — full reductions
        @Override public double sumAll() { return value.sumAll(); }
        @Override public double meanAll() { return value.meanAll(); }
        @Override public double maxAll() { return value.maxAll(); }
        @Override public double minAll() { return value.minAll(); }
        @Override public double prodAll() { return value.prodAll(); }

        // Gradient ops: all no-ops
        @Override public void backward() {}
        @Override public void backward(IDoubleTensor gradient) {}
        @Override public void zeroGradient() {}
        @Override public IDiffVector flattenGrad() { return null; }
        @Override public IDiffVector flattenValue() {
            return new RereDiffVector(IDoubleVector.of(value.toDoubleArray()));
        }
        @Override public IDoubleTensor detach() { return new RereDoubleTensor(value.toDoubleArray(), shape()); }
        @Override public boolean requiresGrad() { return false; }
        @Override public IDiffTensor setRequiresGrad(boolean rg) { return this; }
        @Override public IDoubleTensor grad() { return null; }

        @Override
        public IDiffTensor sum() {
            double total = value.sumAll();
            return new ConstantDiffTensor(new RereDoubleTensor(new double[]{total}, new int[]{1}));
        }

        private static IDiffTensor wrap(IDoubleTensor t) {
            if (t instanceof RereDoubleTensor rdt) return new ConstantDiffTensor(rdt);
            return new ConstantDiffTensor(new RereDoubleTensor(t.toDoubleArray(), t.shape()));
        }
        private static IDoubleTensor detachOther(IDoubleTensor t) {
            return (t instanceof IDiffTensor dt) ? dt.detach() : t;
        }
    }
}
