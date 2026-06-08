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
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.gpu.GpuActivation;
import com.yishape.lab.math.compute.gpu.GpuReduce;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
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

    private RereDoubleTensor value;
    private double[] grad;
    private List<RereDiffTensor> inputs;
    private Consumer<RereDiffTensor> backwardFn;
    private String opTag;
    private boolean requiresGrad = true;
    private boolean isLeaf;
    /**
     * Op-specific scalar parameter. Semantics depend on {@link #opTag()}:
     * <ul>
     *   <li>{@code pow/powSum/powMean} — exponent</li>
     *   <li>{@code rsubScalar/rdivScalar} — the scalar operand</li>
     *   <li>{@code div/mean} — divisor n</li>
     *   <li>activation variants — alpha/hardness parameter</li>
     * </ul>
     * <b>WARNING:</b> This is NOT batch size. Executors must NOT divide loss
     * or gradients by this value. Each GPU/HPC op must produce correctly-scaled results.
     */
    private double scalarParam = Double.NaN;
    /** Secondary scalar parameter (e.g. beta in activation variants). */
    private double scalarParam2 = Double.NaN;

    /**
     * Override shape in JSON export. When non-null, used instead of the tensor's own shape
     * for GPU/HPC graph export (e.g. fused pattern nodes where the logical shape differs).
     */
    private int[] exportShape;

    /**
     * Auxiliary backward data exported to GPU/HPC backends (e.g. MaxPool2d argmax indices).
     * When non-null, included in the binary/JSON graph serialization.
     */
    private int[] backwardIndices;

    /**
     * Symbolic backward function for higher-order AD.
     * Takes the output gradient (IDiffTensor) and returns gradients for each input.
     * When non-null, enables {@code AD.grad(output, inputs)} to build a new
     * computation graph whose nodes are themselves differentiable.
     */
    private java.util.function.Function<IDiffTensor, IDiffTensor[]> symbolicBackwardFn;

    // ==================== Accessors ====================

    public RereDoubleTensor value() { return value; }
    public void setValue(RereDoubleTensor v) { this.value = v; }
    public double[] gradData() { return grad; }
    public void setGradData(double[] g) { this.grad = g; }
    public List<RereDiffTensor> inputs() { return inputs; }
    public void setInputs(List<RereDiffTensor> ins) { this.inputs = ins; }
    public Consumer<RereDiffTensor> backwardFn() { return backwardFn; }
    public void setBackwardFn(Consumer<RereDiffTensor> fn) { this.backwardFn = fn; }
    public String opTag() { return opTag; }
    public void setOpTag(String tag) { this.opTag = tag; }
    public boolean isLeaf() { return isLeaf; }
    public void setIsLeaf(boolean leaf) { this.isLeaf = leaf; }
    public double scalarParam() { return scalarParam; }
    public void setScalarParam(double sp) { this.scalarParam = sp; }
    public double scalarParam2() { return scalarParam2; }
    public void setScalarParam2(double sp2) { this.scalarParam2 = sp2; }
    public int[] exportShape() { return exportShape; }
    public void setExportShape(int[] es) { this.exportShape = es; }

    /**
     * Returns the shape for backend serialization (JSON / binary protocol).
     *
     * <p>Prefers {@link #exportShape()} when it carries extra metadata dimensions
     * beyond the mathematical output shape (e.g. maxpool2d uses
     * {@code [B,C,inH,inW,outH,outW]} 6D while {@link #shape()} is
     * {@code [B,C,outH,outW]} 4D). Backend executors need the extra dimensions
     * to correctly resolve input layout when stride doesn't divide evenly.</p>
     *
     * <p>All graph serializers ({@code TensorGraphExporter},
     * {@code TensorBinaryProtocol}, {@code ExportShapeValidator}) MUST use this
     * method — never bare {@link #shape()} — to avoid dimension-derivation bugs
     * in HPC/GPU backends.</p>
     *
     * @return the complete shape for backend execution (never null)
     */
    public int[] serializationShape() {
        int[] raw = shape();
        if (exportShape != null && exportShape.length > raw.length) {
            return exportShape;
        }
        return raw;
    }
    public int[] backwardIndices() { return backwardIndices; }
    public void setBackwardIndices(int[] bi) { this.backwardIndices = bi; }
    public java.util.function.Function<IDiffTensor, IDiffTensor[]> symbolicBackwardFn() { return symbolicBackwardFn; }
    public void setSymbolicBackwardFn(java.util.function.Function<IDiffTensor, IDiffTensor[]> fn) { this.symbolicBackwardFn = fn; }

    // ==================== ThreadLocal ====================

    private static final ThreadLocal<ArrayList<RereDiffTensor>> TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffTensor>> TOPO_SET =
        ThreadLocal.withInitial(HashSet::new);

    /** Guards against reentrant backwardImpl() — e.g., ODE adjoint computeVJP triggers
     *  nested backward() during the outer backwardImpl() loop. Reentrant calls use
     *  local collections to avoid corrupting the outer iteration. */
    private static final ThreadLocal<Boolean> IN_BACKWARD_IMPL =
        ThreadLocal.withInitial(() -> false);

    /**
     * Reset all ThreadLocal resources held by the autodiff subsystem.
     * Safe to call from web container shutdown hooks to prevent ClassLoader leaks.
     */
    public static void resetThreadLocals() {
        TOPO_LIST.remove();
        TOPO_SET.remove();
        IN_BACKWARD_IMPL.remove();
        AutodiffBufferPool.resetThreadLocals();
    }

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

    void backwardImpl() {
        if (IN_BACKWARD_IMPL.get()) {
            backwardImplLocal();
            return;
        }
        IN_BACKWARD_IMPL.set(true);
        try {
            ArrayList<RereDiffTensor> order = TOPO_LIST.get();
            order.clear();
            HashSet<RereDiffTensor> visited = TOPO_SET.get();
            visited.clear();
            buildTopo(order, visited);
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
        } finally {
            IN_BACKWARD_IMPL.set(false);
            TOPO_LIST.get().clear();
            TOPO_SET.get().clear();
        }
    }

    /** Reentrant-safe backward using local collections — avoids corrupting the outer
     *  backwardImpl()'s ThreadLocal TOPO_LIST/TOPO_SET during nested backward(). */
    private void backwardImplLocal() {
        ArrayList<RereDiffTensor> order = new ArrayList<>();
        HashSet<RereDiffTensor> visited = new HashSet<>();
        buildTopo(order, visited);
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
    }

    /**
     * Propagate gradient backward through this tensor's sub-graph without zeroing
     * this tensor's own gradient. Uses local collections (not ThreadLocal).
     *
     * <p>Called by {@link ODEDiffTensor} during adjoint sensitivity computation.
     */
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
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffTensor v = order.get(i);
            if (v.grad != null && v.backwardFn != null) {
                v.backwardFn.accept(v);
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
    public void clipGradNorm(double maxNorm) {
        if (grad == null || !requiresGrad || maxNorm <= 0) return;
        double norm = 0;
        double[] g = grad;
        for (int i = 0; i < g.length; i++) norm += g[i] * g[i];
        norm = Math.sqrt(norm);
        if (norm > maxNorm) {
            double scale = maxNorm / norm;
            double[] scaled = COMPUTER.binaryOperate(g, scale, BinaryOperation.MULTIPLY);
            System.arraycopy(scaled, 0, g, 0, g.length);
        }
    }

    @Override
    public void clipGradValue(double maxValue) {
        if (grad == null || !requiresGrad || maxValue <= 0) return;
        double[] g = grad;
        for (int i = 0; i < g.length; i++) {
            if (g[i] > maxValue) g[i] = maxValue;
            else if (g[i] < -maxValue) g[i] = -maxValue;
        }
    }

    private static final DoubleVectorComputer COMPUTER = new DoubleVectorComputer();

    @Override
    public IDiffVector flattenGrad() {
        if (grad == null) return null;
        double[] g = grad.clone();
        return new RereDiffVector(new RereDiffTensor(g, new int[]{g.length}));
    }

    @Override
    public IDiffVector flattenValue() {
        return new RereDiffVector(this);
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
            // Scalar sum for non-differentiable tensors — avoids empty-shape bug
            // in RereDoubleTensor.sum(dim, false) for rank-1 tensors
            double total = 0;
            long n = value.totalSize();
            for (long i = 0; i < n; i++) total += value.linearGet(i);
            IDoubleTensor r = new RereDoubleTensor(new double[]{total}, new int[]{1});
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
        // tape-of-tape: d²(sum)/dx² = 0. Connect to input via mul(0) so
        // MixedMode.hvp() gradients can flow back (as zero) to primal variables.
        RereDiffTensor xRefSumResult = this;
        int[] sumShape = shape().clone();
        result.symbolicBackwardFn = g -> {
            double[] ones = new double[(int) xRefSumResult.value().totalSize()];
            java.util.Arrays.fill(ones, 1.0);
            return new IDiffTensor[]{
                g.mul(xRefSumResult.mul(0.0).add(IDiffTensor.constantTensor(ones, sumShape)))
            };
        };
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
            // Symbolic backward for tape-of-tape: d/dx 2*x = 2, use x_ref for
            // chain-rule propagation. 2.0 * g broadcast via scalar-op mul.
            RereDiffTensor xRefSq = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(2.0).mul(xRefSq)};
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
            // tape-of-tape: d²relu/dx² = 0. Connect factor to x via mul(0)
            // so gradient can flow (as zero) back to primal variables.
            double[] rfCopy = reluFactor;
            int[] rfShape = x.shape().clone();
            RereDiffTensor xRefRelu = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{
                g.mul(xRefRelu.mul(0.0).add(IDiffTensor.constantTensor(rfCopy, rfShape)))
            };
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
            // tape-of-tape: d²(exp(x)·sum)/dx² = exp(x). Use x.exp() to preserve
            // graph connection for MixedMode.hvp().
            RereDiffTensor xRefExp = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{ g.mul(xRefExp.exp()) };
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
            // tape-of-tape: d²σ(x)/dx² = σ(x)(1-σ(x))(1-2σ(x)).
            // Use tensor ops on x so MixedMode.hvp() gradients flow back.
            RereDiffTensor xRefSig = x;
            r.symbolicBackwardFn = g -> {
                IDiffTensor s = xRefSig.sigmoid();
                return new IDiffTensor[]{ g.mul(s).mul(s.neg().add(1.0)) };
            };
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
            // tape-of-tape: d²abs/dx² = 0. Connect factor to x via mul(0).
            double[] afCopy = absFactor;
            int[] afShape = x.shape().clone();
            RereDiffTensor xRefAbs = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{
                g.mul(xRefAbs.mul(0.0).add(IDiffTensor.constantTensor(afCopy, afShape)))
            };
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
            // tape-of-tape: d²tanh(x)/dx² = -2·tanh(x)·(1-tanh²(x)).
            // Use tensor ops on x so MixedMode.hvp() gradients flow back.
            RereDiffTensor xRefTanh = x;
            r.symbolicBackwardFn = g -> {
                IDiffTensor t = xRefTanh.tanh();
                return new IDiffTensor[]{ g.mul(t.square().neg().add(1.0)) };
            };
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
            // tape-of-tape: use tensor ops on x so MixedMode.hvp() gradients flow back.
            RereDiffTensor xRefSilu = x;
            r.symbolicBackwardFn = g -> {
                IDiffTensor s = xRefSilu.sigmoid();
                return new IDiffTensor[]{ g.mul(s.add(xRefSilu.mul(s).mul(s.neg().add(1.0)))) };
            };
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
            // tape-of-tape: d²log(x)/dx² = -1/x². Use x.reciprocal() so
            // MixedMode.hvp() gradients flow back through the symbolic graph.
            RereDiffTensor xRefLog = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{ g.mul(xRefLog.reciprocal()) };
            return r;
        }
        // pow(n).sum()
        // scalarParam = exponent n for pow, forwarded to GPU/HPC via JSON "scalar" field.
        // Backward: d/dx n · x^(n-1) per element, broadcast to input shape.
        double scalarP = scalarParam; // capture scalarParam (exponent n)
        if ("pow".equals(opTag) && !Double.isNaN(scalarP)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g * scalarP * Math.pow(xData[i], scalarP - 1);
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "powSum");
            r.exportShape = x.shape();
            r.scalarParam = scalarP;  // op parameter (exponent), NOT batchSize
            // Symbolic backward for tape-of-tape: d/dx n*x^(n-1)
            // Use the original input tensor x_ref so second derivatives flow back.
            RereDiffTensor xRefPw = x;
            double nCaptured = scalarP;
            if (nCaptured == 1.0) {
                r.symbolicBackwardFn = g -> new IDiffTensor[]{g};
            } else if (nCaptured == 2.0) {
                r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(2.0).mul(xRefPw)};
            } else if (nCaptured == 3.0) {
                r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(3.0).mul(xRefPw.pow(2))};
            } else {
                r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(nCaptured).mul(xRefPw.pow(nCaptured - 1))};
            }
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
            r.symbolicBackwardFn = dimSumGradFn(x.shape(), dim, expFactor, x);
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
            rt.symbolicBackwardFn = dimSumGradFn(x.shape(), dim, sigFactor, x);
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
        node.symbolicBackwardFn = dimSumGradFn(x.shape(), dim, factor, x);
        return node;
    }

    /** symbolicBackwardFn for sum(dim)-fused ops: broadcast g along dim, multiply by factor.
     *  Connects factor to xRef via mul(0) so tape-of-tape gradients can flow back. */
    private static java.util.function.Function<IDiffTensor, IDiffTensor[]> dimSumGradFn(
            int[] inputShape, int dim, double[] factor, RereDiffTensor xRef) {
        int[] shapeCopy = inputShape.clone();
        double[] factorCopy = factor.clone();
        int dimCopy = dim;
        RereDiffTensor xRefCopy = xRef;
        return g -> {
            IDiffTensor expanded = g.unsqueeze(dimCopy);
            IDiffTensor factorTensor = xRefCopy.mul(0.0)
                .add(IDiffTensor.constantTensor(factorCopy, shapeCopy));
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

    /** Symbolic backward for same-shape binary ops: returns gradient for each requiresGrad input.
     *  Connects factors to tensor references via mul(0) so tape-of-tape gradients flow back. */
    private java.util.function.Function<IDiffTensor, IDiffTensor[]> binarySameSymbolicFn(
            int n, BinaryBackward gradA, BinaryBackward gradB,
            double[] aData, double[] bData, int[] shape,
            boolean hasA, boolean hasB,
            RereDiffTensor aRef, RereDiffTensor bRef) {
        double[] aCopy = aData.clone();
        double[] bCopy = bData.clone();
        int[] shapeCopy = shape.clone();
        RereDiffTensor aRefCopy = aRef;
        RereDiffTensor bRefCopy = bRef;
        return g -> {
            double[] factA = new double[n];
            double[] factB = new double[n];
            for (int i = 0; i < n; i++) {
                factA[i] = gradA.apply(1.0, aCopy[i], bCopy[i]);
                factB[i] = gradB.apply(1.0, aCopy[i], bCopy[i]);
            }
            // Connect to input tensors via mul(0) so gradient can flow (as zero for add/sub).
            IDiffTensor tA = hasA ? g.mul(aRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factA, shapeCopy))) : null;
            IDiffTensor tB = hasB ? g.mul(bRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factB, shapeCopy))) : null;
            if (hasA && hasB) return new IDiffTensor[]{ tA, tB };
            return new IDiffTensor[]{ hasA ? tA : tB };
        };
    }

    /** Symbolic backward for broadcast binary ops: scatter-reduce gradient factor to original shape.
     *  Connects factors to tensor references via mul(0) so tape-of-tape gradients flow back. */
    private java.util.function.Function<IDiffTensor, IDiffTensor[]> binaryBroadcastSymbolicFn(
            int n, BinaryBackward gradA, BinaryBackward gradB,
            double[] bcA, double[] bcB,
            int[] sA, int[] sB, int[] resultShape,
            boolean hasA, boolean hasB,
            RereDiffTensor aRef, RereDiffTensor bRef) {
        double[] bcACopy = bcA.clone();
        double[] bcBCopy = bcB.clone();
        int[] sACopy = sA.clone();
        int[] sBCopy = sB.clone();
        int[] rShapeCopy = resultShape.clone();
        RereDiffTensor aRefCopy = aRef;
        RereDiffTensor bRefCopy = bRef;
        return g -> {
            if (hasA && hasB) {
                int aTotal = (int) computeSize(sACopy);
                int bTotal = (int) computeSize(sBCopy);
                double[] factA = new double[aTotal];
                double[] factB = new double[bTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    factA[flatIndexFromBroadcast(idx, sACopy, rShapeCopy)] += gradA.apply(1.0, bcACopy[i], bcBCopy[i]);
                    factB[flatIndexFromBroadcast(idx, sBCopy, rShapeCopy)] += gradB.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{
                    g.mul(aRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factA, sACopy))),
                    g.mul(bRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factB, sBCopy)))
                };
            } else if (hasA) {
                int aTotal = (int) computeSize(sACopy);
                double[] factA = new double[aTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    factA[flatIndexFromBroadcast(idx, sACopy, rShapeCopy)] += gradA.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{
                    g.mul(aRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factA, sACopy)))
                };
            } else {
                int bTotal = (int) computeSize(sBCopy);
                double[] factB = new double[bTotal];
                for (int i = 0; i < n; i++) {
                    int[] idx = unlinearizeInt(i, rShapeCopy);
                    factB[flatIndexFromBroadcast(idx, sBCopy, rShapeCopy)] += gradB.apply(1.0, bcACopy[i], bcBCopy[i]);
                }
                return new IDiffTensor[]{
                    g.mul(bRefCopy.mul(0.0).add(IDiffTensor.constantTensor(factB, sBCopy)))
                };
            }
        };
    }

    // ==================== Element-wise unary ops ====================

    @Override public IDiffTensor neg() { return unaryOp(x -> -x, (g, x) -> -g, "neg"); }
    @Override public IDiffTensor abs() { return unaryOp(Math::abs, (g, x) -> x >= 0 ? g : -g, "abs"); }
    @Override public IDiffTensor sqrt() { return unaryOp(Math::sqrt, (g, x) -> g / (2.0 * Math.sqrt(x)), "sqrt"); }
    @Override public IDiffTensor exp() { return unaryOpSelf(Math::exp, (g, y) -> g * y, "exp"); }
    @Override public IDiffTensor log() { return unaryOp(Math::log, (g, x) -> g / x, "log"); }
    @Override public IDiffTensor sin() { return unaryOp(Math::sin, (g, x) -> g * Math.cos(x), "sin"); }
    @Override public IDiffTensor cos() { return unaryOp(Math::cos, (g, x) -> -g * Math.sin(x), "cos"); }
    @Override public IDiffTensor tan() { return unaryOp(Math::tan, (g, x) -> { double c = Math.cos(x); return g / (c * c); }, "tan"); }
    @Override public IDiffTensor sigmoid() { return unaryOpSelf(x -> 1.0 / (1.0 + Math.exp(-x)), (g, y) -> g * y * (1.0 - y), "sigmoid"); }
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
        // Use the original input tensor (this) so tape-of-tape AD can compute
        // second derivatives through the pow node.
        RereDiffTensor xRef = this;
        double scalarN_captured = n;
        powResult.symbolicBackwardFn = g -> {
            if (scalarN_captured == 0.0) {
                // d/dx x^0 = 0, gradient is null
                return new IDiffTensor[]{
                    new RereDiffTensor(new double[(int)xRef.value.totalSize()], xRef.shape()).fill_(0)
                };
            } else if (scalarN_captured == 1.0) {
                // d/dx x^1 = 1, gradient is g
                return new IDiffTensor[]{g};
            } else if (scalarN_captured == 2.0) {
                // d/dx x^2 = 2x, gradient is g * 2 * x
                return new IDiffTensor[]{g.mul(2.0).mul(xRef)};
            } else if (scalarN_captured == 3.0) {
                // d/dx x^3 = 3x^2
                return new IDiffTensor[]{g.mul(3.0).mul(xRef.pow(2))};
            } else {
                // d/dx x^n = n * x^(n-1)
                return new IDiffTensor[]{g.mul(scalarN_captured).mul(xRef.pow(scalarN_captured - 1))};
            }
        };
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

    @Override public IDiffTensor tanh() { return unaryOpSelf(Math::tanh, (g, y) -> g * (1.0 - y * y), "tanh"); }

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
            case "tan"     -> UniversalOperation.TAN;
            // square, silu, mish, elu, leakyRelu, selu, softplus, hardtanh, clamp
            // have no corresponding UniversalOperation enum entry yet; fall back to scalar
            default        -> null;
        };
    }

    // ---- symbolic unary factor (tape-of-tape support) ----

    /**
     * Creates a symbolic backward factor for unary ops, expressed as tensor operations
     * on {@code xRef} so that tape-of-tape AD (MixedMode.hvp) can propagate gradients
     * back to the primal variables.
     *
     * <p>For ops with non-zero second derivative (exp, log, sigmoid, etc.), returns a
     * proper tensor expression like {@code xRef.exp()}. For ops with zero second
     * derivative (relu, abs, etc.) or complex ops (gelu, mish), returns a tensor
     * that is numerically correct but connected to xRef via {@code xRef.mul(0).add(…)}
     * so backward() reaches xRef with zero gradient.</p>
     *
     * @return a differentiable tensor factor, never null
     */
    private static IDiffTensor symbolicUnaryFactor(String tag, RereDiffTensor xRef,
                                                    java.util.function.DoubleBinaryOperator backward,
                                                    double[] xData, double scalarParam) {
        int n = xData.length;
        int[] shape = xRef.shape();
        IDiffTensor factor = switch (tag) {
            case "exp"     -> xRef.exp();
            case "log"     -> xRef.reciprocal();
            case "sqrt"    -> xRef.pow(-0.5).mul(0.5);
            case "sin"     -> xRef.cos();
            case "cos"     -> xRef.sin().neg();
            case "tan"     -> { IDiffTensor c = xRef.cos(); yield c.pow(2).reciprocal(); }
            case "sigmoid" -> { IDiffTensor s = xRef.sigmoid(); yield s.mul(s.neg().add(1.0)); }
            case "tanh"    -> { IDiffTensor t = xRef.tanh(); yield t.square().neg().add(1.0); }
            case "square"  -> xRef.mul(2.0);
            case "silu"    -> {
                IDiffTensor s = xRef.sigmoid();
                yield s.add(xRef.mul(s).mul(s.neg().add(1.0)));
            }
            case "softplus" -> {
                double beta = Double.isNaN(scalarParam) ? 1.0 : scalarParam;
                yield xRef.mul(beta).sigmoid();
            }
            // Ops where second derivative is zero or not practically expressible:
            // relu, leakyRelu, elu, selu, gelu, mish, abs, neg, clamp, hardtanh
            default        -> null;
        };
        if (factor != null) return factor;
        // Fallback: constant factor connected to xRef via mul(0) so gradient can flow
        // (as zero) back to xRef. Correct for ops with zero second derivative.
        double[] f = new double[n];
        for (int i = 0; i < n; i++) f[i] = backward.applyAsDouble(1.0, xData[i]);
        return xRef.mul(0.0).add(IDiffTensor.constantTensor(f, shape));
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
        // Symbolic backward for tape-of-tape AD: d/dx = grad * f'(x).
        // Uses tensor operations on the original input so MixedMode.hvp() gradients
        // can flow back through the symbolic graph to primal variables.
        RereDiffTensor xRef = this;
        double[] xData = value.toDoubleArray();
        result.symbolicBackwardFn = g -> new IDiffTensor[]{
            g.mul(symbolicUnaryFactor(tag, xRef, backward, xData, Double.NaN))
        };
        return result;
    }

    /**
     * Unary op variant where backward uses the forward <em>output</em> value
     * rather than recomputing from the input. Reduces redundant computation
     * for ops like {@code exp} (d/dx = y), {@code sigmoid} (d/dx = y·(1−y)),
     * and {@code tanh} (d/dx = 1−y²).
     */
    private IDiffTensor unaryOpSelf(java.util.function.DoubleUnaryOperator forward,
                                     java.util.function.DoubleBinaryOperator backwardUsingOutput,
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
                inGrad[i] = backwardUsingOutput.applyAsDouble(self.grad[i], out[i]);
            }
            input.accGradFromPooled(inGrad, n);
        };
        RereDiffTensor result = new RereDiffTensor(out, shape(), List.of(this), bw, tag);
        // Symbolic backward for tape-of-tape AD: d/dx = grad * f'(forward(x)).
        // For exp/sigmoid/tanh, f'(y) = y resp. y(1-y) resp. 1-y².
        // These are expressed using the same symbolicUnaryFactor as unaryOp
        // (which reconstructs from xRef) for consistency.
        RereDiffTensor xRef = this;
        double[] xData = value.toDoubleArray();
        result.symbolicBackwardFn = g -> new IDiffTensor[]{
            g.mul(symbolicUnaryFactor(tag, xRef, backwardUsingOutput, xData, Double.NaN))
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

    @Override public IDiffTensor add(double scalar) { return scalarOp(scalar, (a, b) -> a + b, (g, v) -> g, "addScalar"); }
    @Override public IDiffTensor sub(double scalar) { return scalarOp(scalar, (a, b) -> a - b, (g, v) -> g, "subScalar"); }
    @Override public IDiffTensor mul(double scalar) { return scalarOp(scalar, (a, b) -> a * b, (g, v) -> g * scalar, "mulScalar"); }
    @Override public IDiffTensor div(double scalar) { return scalarOp(scalar, (a, b) -> a / b, (g, v) -> g / scalar, "divScalar"); }

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
        // tape-of-tape: d²(rsub)/dx² = 0. Connect to input via mul(0).
        RereDiffTensor xRefRsub = this;
        int[] rsubShape = shape().clone();
        double[] rsubFactor = new double[n];
        Arrays.fill(rsubFactor, -1.0);
        r.symbolicBackwardFn = g -> new IDiffTensor[]{
            g.mul(xRefRsub.mul(0.0).add(IDiffTensor.constantTensor(rsubFactor, rsubShape)))
        };
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
        // tape-of-tape: d²(scalar/x)/dx² = 2*scalar/x³ ≠ 0.
        // Use tensor ops on xRef so MixedMode.hvp() gradients flow back.
        RereDiffTensor xRefRdiv = this;
        double rdivScalar = scalar;
        r.symbolicBackwardFn = g -> new IDiffTensor[]{
            g.mul(xRefRdiv.pow(2).reciprocal()).mul(-rdivScalar)
        };
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
        // tape-of-tape: d²(1/x)/dx² = 2/x³ ≠ 0.
        // Use tensor ops on xRef so MixedMode.hvp() gradients flow back.
        RereDiffTensor xRefRecip = this;
        r.symbolicBackwardFn = g -> new IDiffTensor[]{
            g.mul(xRefRecip.pow(2).reciprocal()).neg()
        };
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
        // tape-of-tape: d²(scalar_op)/dx² = 0 (scalar ops don't change curvature).
        // Connect factor to input via mul(0) so gradient can flow (as zero).
        RereDiffTensor xRefScalar = this;
        int[] scShape = shape().clone();
        result.symbolicBackwardFn = g -> {
            double[] f = new double[n];
            for (int i = 0; i < n; i++) f[i] = backward.applyAsDouble(1.0, 0.0); // same for all x
            return new IDiffTensor[]{
                g.mul(xRefScalar.mul(0.0).add(IDiffTensor.constantTensor(f, scShape)))
            };
        };
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
            // Always include the other tensor in inputs for GPU/HPC graph serialization,
            // even if it doesn't require gradients. The backward function below only
            // propagates to inputs that require grad (controlled by otherNode flag).
            if (other instanceof RereDiffTensor) inputs.add((RereDiffTensor) other);

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
            // Per-op symbolic backward: for mul/div, use original tensor references
            // so that tape-of-tape AD (MixedMode.hvp) can flow gradients back
            // to the primal variables. For add/sub, constant factors are fine.
            if ("mul".equals(tag)) {
                RereDiffTensor aRef = this;
                RereDiffTensor bRef = otherNode;
                result.symbolicBackwardFn = g -> {
                    if (hasA && hasB) {
                        return new IDiffTensor[]{g.mul(bRef), g.mul(aRef)};
                    }
                    IDiffTensor gradAT = (bRef != null) ? g.mul(bRef)
                        : g.mul(IDiffTensor.constantTensor(bData, shape()));
                    IDiffTensor gradBT = g.mul(aRef);
                    return new IDiffTensor[]{hasA ? gradAT : gradBT};
                };
            } else if ("div".equals(tag)) {
                RereDiffTensor aRef = this;
                RereDiffTensor bRef = otherNode;
                result.symbolicBackwardFn = g -> {
                    if (hasA && hasB) {
                        return new IDiffTensor[]{g.div(bRef),
                            g.neg().mul(aRef).div(bRef.mul(bRef))};
                    } else if (hasA) {
                        IDiffTensor bDiv = (bRef != null) ? bRef
                            : IDiffTensor.constantTensor(bData, shape());
                        return new IDiffTensor[]{g.div(bDiv)};
                    } else {
                        IDiffTensor aConst = IDiffTensor.constantTensor(aData, shape());
                        return new IDiffTensor[]{g.neg().mul(aConst).div(bRef.mul(bRef))};
                    }
                };
            } else {
                result.symbolicBackwardFn = binarySameSymbolicFn(n, gradA, gradB, aData, bData, shape(),
                    hasA, hasB, this, otherNode);
            }
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
        // Always include the other tensor in inputs for GPU/HPC graph serialization,
        // even if it doesn't require gradients. The backward function below only
        // propagates to inputs that require grad (controlled by otherNode flag).
        if (other instanceof RereDiffTensor) inputs.add((RereDiffTensor) other);

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
        // Per-op symbolic backward for broadcast: use original tensor references
        // for mul/div to preserve tape-of-tape connections.
        if ("mul".equals(tag)) {
            RereDiffTensor bRefBc = otherNode;
            result.symbolicBackwardFn = g -> {
                if (bHasA && bHasB) {
                    return new IDiffTensor[]{g.mul(bRefBc), g.mul(this)};
                }
                IDiffTensor gradAT = (bRefBc != null) ? g.mul(bRefBc)
                    : g.mul(IDiffTensor.constantTensor(bData, sB));
                return new IDiffTensor[]{bHasA ? gradAT : g.mul(this)};
            };
        } else if ("div".equals(tag)) {
            RereDiffTensor bRefBc = otherNode;
            result.symbolicBackwardFn = g -> {
                if (bHasA && bHasB) {
                    return new IDiffTensor[]{g.div(bRefBc),
                        g.neg().mul(this).div(bRefBc.mul(bRefBc))};
                } else if (bHasA) {
                    IDiffTensor bDiv = (bRefBc != null) ? bRefBc
                        : IDiffTensor.constantTensor(bData, sB);
                    return new IDiffTensor[]{g.div(bDiv)};
                } else {
                    IDiffTensor aConst = IDiffTensor.constantTensor(aData, sA);
                    return new IDiffTensor[]{g.neg().mul(aConst).div(bRefBc.mul(bRefBc))};
                }
            };
        } else {
            result.symbolicBackwardFn = binaryBroadcastSymbolicFn(
                n, gradA, gradB, bcA, bcB, sA, sB, resultShape, bHasA, bHasB,
                this, otherNode);
        }
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
        RereDiffTensor result = new RereDiffTensor(view, List.of(this), bw, "select");
        result.backwardIndices = bi;
        return result;
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

    /**
     * Returns a contiguous copy of this tensor, or {@code this} if it is already
     * both logically contiguous AND has zero offset.
     *
     * <p><b>PITFALL — offset ≠ 0 causes infinite recursion:</b>
     * {@link #reshape(int...)} calls {@code this.contiguous().reshape(newShape)}.
     * If {@code contiguous()} returns {@code this} (because strides are C-order)
     * but {@code offset() != 0}, the {@code reshape()} guard {@code isContiguous() && offset() == 0}
     * fails, and it calls {@code this.contiguous().reshape()} again → StackOverflowError.
     *
     * <p>This happens with tensor views like {@code select(dim, idx)} on a multi-row
     * tensor: row 1 of [2, N] has shape [N], contiguous strides [1], but offset = N.
     * Always check {@code offset() == 0} alongside {@code isContiguous()}.</p>
     */
    @Override
    public IDiffTensor contiguous() {
        if (value.isContiguous() && offset() == 0) return this;
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
        RereDiffTensor rd = new RereDiffTensor(result, resultShape, List.of(this), bw, "sum");
        // ╔══════════════════════════════════════════════════════════════════════════╗
        // ║  ☠️ STRIDE ENCODING PITFALL — ROOT CAUSE                              ║
        // ║                                                                         ║
        // ║  THE BUG (2026-06-08): GPU sum(dim) always reduced ALL elements to 1    ║
        // ║  value because the Rust graph executor's sum/mean dispatch used         ║
        // ║  outer=1 (flat reduce) for ALL cases. It didn't know the stride.        ║
        // ║                                                                         ║
        // ║  ROOT CAUSE: This method added dimension-specific sum with proper       ║
        // ║  stride calculation (inner = product of dims after reduced dim), but     ║
        // ║  the Rust GPU graph executor (`graph.rs`) was NOT updated to read       ║
        // ║  the stride from scalarParam. It kept using outer=1 (flat sum) for      ║
        // ║  every "sum" op node, ignoring the stride entirely.                     ║
        // ║                                                                         ║
        // ║  DEFENSE: This `inner` value (= stride = product of dims after the      ║
        // ║  reduced dimension) is encoded as scalarParam on the sum tensor node.   ║
        // ║  The Rust graph executor reads scalarParam as the stride parameter:     ║
        // ║    n==1 (flat): outer=1, inner=in_size, stride=1                        ║
        // ║    n>1  (dim):  outer=n, inner=in_size/n, stride=scalarParam            ║
        // ║                                                                         ║
        // ║  If scalarParam is NaN (sum() with no dim argument → flat), the Rust    ║
        // ║  side defaults to stride=1 (contiguous flat reduction).                 ║
        // ║                                                                         ║
        // ║  RULE: Any modification to this encoding MUST also update:              ║
        // ║  1. graph.rs → reduce::dispatch call (outer/inner/stride logic)         ║
        // ║  2. reduce.wgsl → WGSL access formula                                   ║
        // ║  3. reduce.rs → dispatch signature                                      ║
        // ║  4. broadcast.wgsl → strided backward broadcast                         ║
        // ║  5. broadcast.rs → dispatch_strided function                            ║
        // ║  6. Test: SumDimGpuDiagnostic.java                                      ║
        // ╚══════════════════════════════════════════════════════════════════════════╝
        rd.setScalarParam((double) inner);
        return rd;
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
                    totalLoss += -y * Math.log(Math.max(p, 1e-30));
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
        boolean otherNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();
        if (!requiresGrad && !otherNeedsGrad) {
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
        double[] bData = ((RereDiffTensor) other).value.toDoubleArray();
        double[] resultData = DoubleFlatGemm.flatMmul(aData, M, K, bData, N);
        int[] resultShape = {M, N};

        // Lazy transpose: only allocate when the corresponding operand needs gradients
        boolean aNeedsGrad = requiresGrad;
        boolean bNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();
        double[] bT = aNeedsGrad ? DoubleFlatGemm.flatTranspose(bData, K, N) : null;
        double[] aT = bNeedsGrad ? DoubleFlatGemm.flatTranspose(aData, M, K) : null;

        // Dynamic inputs list: only include tensors that require gradients
        List<RereDiffTensor> inputs = new ArrayList<>();
        if (aNeedsGrad) inputs.add(this);
        RereDiffTensor otherNode = bNeedsGrad ? (RereDiffTensor) other : null;
        if (otherNode != null) inputs.add(otherNode);

        int fM = M, fK = K, fN = N;
        Consumer<RereDiffTensor> bw = self -> {
            int idx = 0;
            if (aNeedsGrad) {
                RereDiffTensor inpA = self.inputs.get(idx++);
                // Compute bT lazily on first backward if not pre-computed
                double[] bt = bT != null ? bT : DoubleFlatGemm.flatTranspose(bData, fK, fN);
                double[] dA = DoubleFlatGemm.flatMmul(self.grad, fM, fN, bt, fK);
                inpA.accGrad(dA);
            }
            if (bNeedsGrad) {
                RereDiffTensor inpB = self.inputs.get(idx);
                // Compute aT lazily on first backward if not pre-computed
                double[] at = aT != null ? aT : DoubleFlatGemm.flatTranspose(aData, fM, fK);
                double[] dB = DoubleFlatGemm.flatMmul(at, fK, fM, self.grad, fN);
                inpB.accGrad(dB);
            }
        };
        return new RereDiffTensor(resultData, resultShape, inputs, bw, "mmul");
    }

    @Override
    public IDiffTensor bmm(IDoubleTensor other) {
        boolean otherNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();
        if (!requiresGrad && !otherNeedsGrad) {
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
        double[] bData = ((RereDiffTensor) other).value.toDoubleArray();
        double[] resultData = DoubleFlatGemm.flatMmulBatched(aData, bData, B, M, K, N);

        // Lazy transpose: only pre-allocate slices for operands that need gradients;
        // compute each slice's transpose on-the-fly in the backward closure
        // to avoid storing all B transposes simultaneously in memory (PERF-3).
        boolean aNeedsGrad = requiresGrad;
        boolean bNeedsGrad = other instanceof IDiffTensor od && od.requiresGrad();

        // Dynamic inputs list
        List<RereDiffTensor> inputs = new ArrayList<>();
        if (aNeedsGrad) inputs.add(this);
        RereDiffTensor otherNode = bNeedsGrad ? (RereDiffTensor) other : null;
        if (otherNode != null) inputs.add(otherNode);

        int fB = B, fM = M, fK = K, fN = N;
        Consumer<RereDiffTensor> bw = self -> {
            int idx = 0;
            int aStride = fM * fK, bStride = fK * fN, gStride = fM * fN;
            if (aNeedsGrad) {
                RereDiffTensor inpA = self.inputs.get(idx++);
                double[] dA = new double[fB * aStride];
                for (int bi = 0; bi < fB; bi++) {
                    int aOff = bi * aStride, bOff = bi * bStride, gOff = bi * gStride;
                    // Compute bT on-the-fly for this batch element
                    double[] bSlice = Arrays.copyOfRange(bData, bOff, bOff + bStride);
                    double[] bT = DoubleFlatGemm.flatTranspose(bSlice, fK, fN);
                    double[] dASlice = DoubleFlatGemm.flatMmul(self.grad, gOff, fM, fN, bT, 0, fK);
                    System.arraycopy(dASlice, 0, dA, aOff, aStride);
                }
                inpA.accGrad(dA);
            }
            if (bNeedsGrad) {
                RereDiffTensor inpB = self.inputs.get(idx);
                double[] dB = new double[fB * bStride];
                for (int bi = 0; bi < fB; bi++) {
                    int aOff = bi * aStride, bOff = bi * bStride, gOff = bi * gStride;
                    // Compute aT on-the-fly for this batch element
                    double[] aSlice = Arrays.copyOfRange(aData, aOff, aOff + aStride);
                    double[] aT = DoubleFlatGemm.flatTranspose(aSlice, fM, fK);
                    double[] dBSlice = DoubleFlatGemm.flatMmul(aT, 0, fK, fM, self.grad, gOff, fN);
                    System.arraycopy(dBSlice, 0, dB, bOff, bStride);
                }
                inpB.accGrad(dB);
            }
        };
        return new RereDiffTensor(resultData, resultShape, inputs, bw, "bmm");
    }

    @Override
    public IDiffTensor einsum(String subscript, IDoubleTensor... others) {
        if (others.length == 0) {
            return einsumSingle(subscript);
        }
        // Non-differentiable path: delegate to linalg einsum
        boolean otherNeedsGrad = others[0] instanceof IDiffTensor od && od.requiresGrad();
        if (!requiresGrad && !otherNeedsGrad) {
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
        int[] flatSourcePos = new int[resultTotal];
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
            flatSourcePos[i] = flatIndex(srcIdx, s);
            resultData[i] = inData[flatSourcePos[i]];
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
        // Build inputs for GPU/HPC graph execution.
        // The GPU gather kernel treats the source as [vocab_size, embedding_dim]
        // where embedding_dim = product of source dims after the gather dim.
        // It expects "row indices" (= flat_source_position / embedding_dim) rather
        // than dimension-relative index values (e.g. raw class labels 0-9).
        // We pre-compute row indices so the GPU can do:
        //   output[row*dim+j] = weight[rowIdx[row] * dim + j]
        // The CPU backward uses closure-captured fGatherIndices and is unaffected.
        List<RereDiffTensor> gatherInputs;
        if (index instanceof RereDiffTensor rt) {
            int trailingProduct = 1;
            for (int i = d + 1; i < r; i++) trailingProduct *= s[i];
            int numOutputRows = resultTotal / trailingProduct;
            double[] rowIdxData = new double[numOutputRows];
            for (int row = 0; row < numOutputRows; row++) {
                rowIdxData[row] = (double) (flatSourcePos[row * trailingProduct] / trailingProduct);
            }
            RereDiffTensor rowIdxTensor = new RereDiffTensor(rowIdxData, numOutputRows);
            rowIdxTensor.setRequiresGrad(false);
            gatherInputs = List.of(this, rowIdxTensor);
        } else {
            gatherInputs = List.of(this);
        }
        return new RereDiffTensor(resultData, resultShape, gatherInputs, bw, "gather");
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
    public IDiffTensor tril(int diagonal) {
        int r = rank();
        if (r < 2) return this; // scalar/vector: no-op
        if (!requiresGrad) return toNonDiff(value.tril(diagonal));

        int[] s = shape();
        int M = s[r - 2];
        int N = s[r - 1];
        double[] resultData = value.toDoubleArray().clone();
        int batchStride = M * N;
        int batchCount = resultData.length / batchStride;

        for (int b = 0; b < batchCount; b++) {
            int base = b * batchStride;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    if (j > i + diagonal) {
                        resultData[base + i * N + j] = 0.0;
                    }
                }
            }
        }

        int fDiagonal = diagonal;
        int[] fShape = s;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int bStride = fShape[fShape.length - 2] * fShape[fShape.length - 1];
            int bCount = input.grad.length / bStride;
            // gradient flows only for non-zeroed elements (col <= row + diagonal)
            for (int b = 0; b < bCount; b++) {
                int base = b * bStride;
                for (int i = 0; i < fShape[fShape.length - 2]; i++) {
                    for (int j = 0; j < fShape[fShape.length - 1]; j++) {
                        if (j > i + fDiagonal) {
                            self.grad[base + i * fShape[fShape.length - 1] + j] = 0.0;
                        }
                    }
                }
            }
            input.accGrad(self.grad);
        };
        return new RereDiffTensor(resultData, s, List.of(this), bw, "tril");
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
                double dg = 0, db = 0, sumG = 0, sumGXHat = 0;
                double[] xHatCache = new double[batch];
                for (int i = 0; i < batch; i++) {
                    int idx = i * features + j;
                    double xHat = (xd[idx] - mean) * invSig;
                    xHatCache[i] = xHat;
                    dg += g[idx] * xHat;
                    db += g[idx];
                    sumG += g[idx];
                    sumGXHat += g[idx] * xHat;
                }
                dGamma[j] = dg;
                dBeta[j] = db;
                double scale = cg[j] * invSig / batch;
                for (int i = 0; i < batch; i++) {
                    int idx = i * features + j;
                    dx[idx] = scale * (batch * g[idx] - sumG - xHatCache[i] * sumGXHat);
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

    @Override
    public IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
                               int stride, int padding, int dilation) {
        RereDiffTensor w = (RereDiffTensor) weight;
        RereDiffTensor b = (RereDiffTensor) bias;
        int[] inShape = shape();
        if (inShape.length != 4) {
            throw new IllegalArgumentException("conv2d: input must be 4-D [N,C,H,W], got rank=" + inShape.length);
        }
        int N = inShape[0], C = inShape[1], H = inShape[2], W_in = inShape[3];
        int[] wShape = w.shape();
        int outC = wShape[0], inC = wShape[1], kH = wShape[2], kW = wShape[3];
        if (inC != C) {
            throw new IllegalArgumentException("conv2d: weight inCh=" + inC + " != input C=" + C);
        }
        int outH = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
        int outW = (W_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;
        if (outH <= 0 || outW <= 0) {
            throw new IllegalArgumentException("conv2d: output size invalid outH=" + outH + " outW=" + outW);
        }
        int outHW = outH * outW;
        long M = (long) N * outHW;
        if (M > Integer.MAX_VALUE / 8) {
            throw new IllegalArgumentException("conv2d: output too large");
        }
        int mM = (int) M;
        int Kcol = C * kH * kW;

        double[] xd = value.toDoubleArray();
        double[] wd = w.value.toDoubleArray();
        double[] bd = b != null ? b.value.toDoubleArray() : null;

        // im2col: [N*outH*outW, C*kH*kW], HPC→SISD fallback
        double[] col = new double[mM * Kcol];
        int kH_kW = kH * kW;
        if (HpcIm2col.tryBatchIm2col(xd, N, C, H, W_in, kH, kW, stride, padding, dilation, col)) {
            // HPC succeeded
        } else {
            // SISD fallback
            int H_W = H * W_in;
            for (int n = 0; n < N; n++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        int colRow = n * outHW + oh * outW + ow;
                        int colBase = colRow * Kcol;
                        for (int c = 0; c < C; c++) {
                            int cOff = c * kH_kW;
                            for (int kh = 0; kh < kH; kh++) {
                                int ih = oh * stride + kh * dilation - padding;
                                int khOff = kh * kW;
                                for (int kw = 0; kw < kW; kw++) {
                                    int iw = ow * stride + kw * dilation - padding;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W_in) {
                                        col[colBase + cOff + khOff + kw] = xd[n * C * H_W + c * H_W + ih * W_in + iw];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // gemm: col @ w^T → [N*outH*outW, outC]
        // w is [outC, C*kH*kW], we need w^T = [C*kH*kW, outC]
        double[] wT = DoubleFlatGemm.flatTranspose(wd, outC, Kcol);
        double[] outCol = DoubleFlatGemm.flatMmul(col, mM, Kcol, wT, outC);

        // Add bias via acceleration chain (tile bias vector across rows)
        if (bd != null) {
            double[] biasTiled = new double[mM * outC];
            for (int i = 0; i < mM; i++) {
                System.arraycopy(bd, 0, biasTiled, i * outC, outC);
            }
            outCol = new DoubleVectorComputer().binaryOperate(outCol, biasTiled, BinaryOperation.ADD);
        }

        int[] outShape = {N, outC, outH, outW};
        // Reshape output: [N*outH*outW, outC] → [N, outC, outH, outW]
        double[] y = new double[(int) ((long) N * outC * outHW)];
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int colRow = n * outHW + oh * outW + ow;
                    int outBase = n * outC * outHW + oh * outW + ow;
                    for (int oc = 0; oc < outC; oc++) {
                        y[outBase + oc * outHW] = outCol[colRow * outC + oc];
                    }
                }
            }
        }

        // Build input list
        List<RereDiffTensor> inputs = new ArrayList<>();
        inputs.add(this);
        inputs.add(w);
        if (b != null && b.requiresGrad()) inputs.add(b);

        // Capture for backward
        double[] savedCol = col;
        double[] savedWd = wd;
        int fN = N, fC = C, fH = H, fW = W_in, fOutC = outC, fOutH = outH, fOutW = outW;
        int fKH = kH, fKW = kW, fStride = stride, fPad = padding, fDil = dilation;

        Consumer<RereDiffTensor> bw = self -> {
            int inpIdx = 0;
            RereDiffTensor inpX = self.inputs.get(inpIdx++);
            RereDiffTensor inpW = self.inputs.get(inpIdx++);
            RereDiffTensor inpB = (b != null && b.requiresGrad()) ? self.inputs.get(inpIdx) : null;

            // Reshape grad from [N, outC, outH, outW] → [N*outH*outW, outC]
            int fOutHW = fOutH * fOutW;
            int fM = fN * fOutHW;
            double[] dOutCol = new double[fM * fOutC];
            for (int n2 = 0; n2 < fN; n2++) {
                for (int oh = 0; oh < fOutH; oh++) {
                    for (int ow = 0; ow < fOutW; ow++) {
                        int colRow = n2 * fOutHW + oh * fOutW + ow;
                        int gradBase = n2 * fOutC * fOutHW + oh * fOutW + ow;
                        for (int oc = 0; oc < fOutC; oc++) {
                            dOutCol[colRow * fOutC + oc] = self.grad[gradBase + oc * fOutHW];
                        }
                    }
                }
            }

            // d_weight = dOutCol^T @ col → [outC, C*kH*kW]
            int fKcol = fC * fKH * fKW;
            double[] dOutT = DoubleFlatGemm.flatTranspose(dOutCol, fM, fOutC);
            double[] dW = DoubleFlatGemm.flatMmul(dOutT, fOutC, fM, savedCol, fKcol);
            inpW.accGrad(dW);

            // d_bias = sum over batch+spatial (column-wise sum of [fM, fOutC])
            // Reuses dOutT ([fOutC, fM]) computed above for d_weight; reduce over last dim via GPU→SISD chain
            if (inpB != null) {
                double[] dB = GpuReduce.tryReduce(GpuReduce.SUM, dOutT, fOutC, fM);
                if (dB == null) {
                    dB = new double[fOutC];
                    DoubleVectorComputer bwVc2 = new DoubleVectorComputer();
                    for (int oc = 0; oc < fOutC; oc++) {
                        dB[oc] = bwVc2.reduceOperate(
                            java.util.Arrays.copyOfRange(dOutT, oc * fM, (oc + 1) * fM),
                            ReduceOperation.SUM);
                    }
                }
                inpB.accGrad(dB);
            }

            // d_input: dOutCol @ w → [N*outH*outW, C*kH*kW] → col2im, HPC→SISD fallback
            double[] dCol = DoubleFlatGemm.flatMmul(dOutCol, fM, fOutC, savedWd, fKcol);
            int dXsize = fN * fC * fH * fW;
            double[] dX = new double[dXsize];
            if (HpcIm2col.tryBatchCol2im(dCol, fN, fC, fH, fW, fKH, fKW, fStride, fPad, fDil, dX)) {
                // HPC succeeded
            } else {
                // SISD fallback
                int fHW = fH * fW;
                for (int n2 = 0; n2 < fN; n2++) {
                    for (int oh = 0; oh < fOutH; oh++) {
                        for (int ow = 0; ow < fOutW; ow++) {
                            int colRow = n2 * fOutHW + oh * fOutW + ow;
                            int colBase = colRow * fKcol;
                            for (int c = 0; c < fC; c++) {
                                int cOff = c * fKH * fKW;
                                for (int kh = 0; kh < fKH; kh++) {
                                    int ih = oh * fStride + kh * fDil - fPad;
                                    int khOff = kh * fKW;
                                    for (int kw = 0; kw < fKW; kw++) {
                                        int iw = ow * fStride + kw * fDil - fPad;
                                        if (ih >= 0 && ih < fH && iw >= 0 && iw < fW) {
                                            int idx = n2 * fC * fHW + c * fHW + ih * fW + iw;
                                            dX[idx] += dCol[colBase + cOff + khOff + kw];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            inpX.accGrad(dX);
        };

        RereDiffTensor result = new RereDiffTensor(y, outShape, inputs, bw, "conv2d");
        result.scalarParam = Double.longBitsToDouble(
            ((long) stride << 32) | ((long) padding & 0xFFFFFFFFL));
        result.scalarParam2 = Double.longBitsToDouble(
            ((long) dilation << 32));
        return result;
    }

    /**
     * Stable softmax over the last dimension of a flat 2-D matrix.
     * Uses GPU→SIMD→SISD acceleration chain for row-reduce (max/sum),
     * element-wise exp, and per-row scalar arithmetic.
     *
     * @param scores  flat row-major array [rows, cols], modified in-place
     * @param rows    number of rows
     * @param cols    number of columns (softmax dim)
     * @return the same {@code scores} array, now containing row-wise softmax probabilities
     */
    private static double[] softmaxRowsStable(double[] scores, int rows, int cols) {
        int total = rows * cols;
        DoubleVectorComputer vc = new DoubleVectorComputer();

        // Step 1: Row-wise max via GPU reduce → SISD fallback
        double[] rowMax = GpuReduce.tryReduce(GpuReduce.MAX, scores, rows, cols);
        if (rowMax == null) {
            rowMax = new double[rows];
            for (int r = 0; r < rows; r++) {
                double[] rowSlice = java.util.Arrays.copyOfRange(scores, r * cols, r * cols + cols);
                rowMax[r] = vc.reduceOperate(rowSlice, ReduceOperation.MAX);
            }
        }

        // Step 2: Subtract row max (per-row scalar add) → collect into shifted for batch exp
        double[] shifted = new double[total];
        for (int r = 0; r < rows; r++) {
            int rowOff = r * cols;
            double[] rowSlice = java.util.Arrays.copyOfRange(scores, rowOff, rowOff + cols);
            double[] shiftRow = vc.binaryOperate(rowSlice, -rowMax[r], BinaryOperation.ADD);
            System.arraycopy(shiftRow, 0, shifted, rowOff, cols);
        }

        // Step 3: Batch exp via GPU → SIMD/SISD fallback
        double[] exped = GpuActivation.tryExp(shifted);
        if (exped == null) {
            exped = vc.universalOperate(shifted, UniversalOperation.EXP, 0);
        }

        // Step 4: Row-wise sum via GPU reduce → SISD fallback
        double[] rowSum = GpuReduce.tryReduce(GpuReduce.SUM, exped, rows, cols);
        if (rowSum == null) {
            rowSum = new double[rows];
            for (int r = 0; r < rows; r++) {
                double[] rowSlice = java.util.Arrays.copyOfRange(exped, r * cols, r * cols + cols);
                rowSum[r] = vc.reduceOperate(rowSlice, ReduceOperation.SUM);
            }
        }

        // Step 5: Normalize (per-row scalar multiply) into scores
        for (int r = 0; r < rows; r++) {
            int rowOff = r * cols;
            double[] rowSlice = java.util.Arrays.copyOfRange(exped, rowOff, rowOff + cols);
            double[] normRow = vc.binaryOperate(rowSlice, 1.0 / rowSum[r], BinaryOperation.MULTIPLY);
            System.arraycopy(normRow, 0, scores, rowOff, cols);
        }
        return scores;
    }

    @Override
    public IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
                                                  IDiffTensor mask, double dropout) {
        RereDiffTensor k = (RereDiffTensor) key;
        RereDiffTensor v = (RereDiffTensor) vTensor;
        RereDiffTensor m = (RereDiffTensor) mask;
        int[] qShape = shape();
        int[] kShape = k.shape();
        int[] vShape = v.shape();
        if (qShape.length != 3 || kShape.length != 3 || vShape.length != 3) {
            throw new IllegalArgumentException(
                "scaledDotProductAttention: inputs must be 3-D [batch,seq,dim]");
        }
        int batch = qShape[0], seqQ = qShape[1], dk = qShape[2];
        int seqK = kShape[1], dk2 = kShape[2];
        int seqV = vShape[1], dv = vShape[2];
        if (batch != kShape[0] || batch != vShape[0]) {
            throw new IllegalArgumentException("scaledDotProductAttention: batch mismatch");
        }
        if (dk != dk2) {
            throw new IllegalArgumentException("scaledDotProductAttention: Q.d_k=" + dk + " != K.d_k=" + dk2);
        }
        if (seqK != seqV) {
            throw new IllegalArgumentException("scaledDotProductAttention: K.seq=" + seqK + " != V.seq=" + seqV);
        }

        double[] qd = this.value.toDoubleArray();
        double[] kd = k.value.toDoubleArray();
        double[] vd = v.value.toDoubleArray();
        double[] md = m != null ? m.value.toDoubleArray() : null;

        double scale = 1.0 / Math.sqrt(dk);

        int qStride = seqQ * dk;
        int kStride = seqK * dk;
        int vStride = seqK * dv;
        int scoresStride = seqQ * seqK;
        double[] attnWeights = new double[batch * scoresStride];

        DoubleVectorComputer vc = new DoubleVectorComputer();
        for (int b = 0; b < batch; b++) {
            // scores = Q @ K^T / sqrt(dk)
            double[] qSlice = java.util.Arrays.copyOfRange(qd, b * qStride, b * qStride + qStride);
            double[] kSlice = java.util.Arrays.copyOfRange(kd, b * kStride, b * kStride + kStride);
            double[] kT = DoubleFlatGemm.flatTranspose(kSlice, seqK, dk);
            // GEMM + scale via acceleration chain
            double[] rawScores = DoubleFlatGemm.flatMmul(qSlice, seqQ, dk, kT, seqK);
            double[] scaledScores = vc.binaryOperate(rawScores, scale, BinaryOperation.MULTIPLY);

            // Add mask per-batch via acceleration chain (handles broadcast internally)
            if (md != null) {
                if (md.length == scoresStride) {
                    scaledScores = vc.binaryOperate(scaledScores, md, BinaryOperation.ADD);
                } else {
                    // Tile mask to scores shape using data-movement primitives, then accelerate the add
                    double[] maskTiled = new double[scoresStride];
                    if (md.length == 1) {
                        Arrays.fill(maskTiled, md[0]);
                    } else {
                        int fullReps = scoresStride / md.length;
                        for (int r = 0; r < fullReps; r++) {
                            System.arraycopy(md, 0, maskTiled, r * md.length, md.length);
                        }
                        int rem = scoresStride % md.length;
                        if (rem > 0) {
                            System.arraycopy(md, 0, maskTiled, fullReps * md.length, rem);
                        }
                    }
                    scaledScores = vc.binaryOperate(scaledScores, maskTiled, BinaryOperation.ADD);
                }
            }

            // Stable softmax via acceleration-eligible helper
            double[] attnB = softmaxRowsStable(scaledScores, seqQ, seqK);
            System.arraycopy(attnB, 0, attnWeights, b * scoresStride, scoresStride);
        }

        // Dropout
        double[] attnOut = attnWeights;
        double dropoutScale = 1.0;
        if (dropout > 0 && dropout < 1) {
            dropoutScale = 1.0 / (1.0 - dropout);
            attnOut = new double[batch * scoresStride];
            java.util.Random rng = new java.util.Random(42L);
            for (int i = 0; i < batch * scoresStride; i++) {
                if (rng.nextDouble() > dropout) {
                    attnOut[i] = attnWeights[i] * dropoutScale;
                }
            }
        }

        // output = attn @ V → [batch, seqQ, dv]
        int outStride = seqQ * dv;
        double[] y = new double[batch * outStride];
        for (int b = 0; b < batch; b++) {
            double[] outB = DoubleFlatGemm.flatMmul(
                attnOut, b * scoresStride, seqQ, seqK, vd, b * vStride, dv);
            System.arraycopy(outB, 0, y, b * outStride, outStride);
        }

        // Build inputs — fused backward handles all three
        List<RereDiffTensor> inputs = new ArrayList<>();
        inputs.add(this);
        inputs.add(k);
        inputs.add(v);

        // Capture for backward
        DoubleVectorComputer fvc = vc;
        double[] savedAttn = attnWeights;
        double[] savedQd = qd;
        double[] savedKd = kd;
        double[] savedVd = vd;
        double fScale = scale;
        double fDropoutScale = dropoutScale;
        int fBatch = batch, fSeqQ = seqQ, fSeqK = seqK, fDk = dk, fDv = dv;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpQ = self.inputs.get(0);
            RereDiffTensor inpK = self.inputs.get(1);
            RereDiffTensor inpV = self.inputs.get(2);

            int fSoStride = fSeqQ * fSeqK;
            int fQStride = fSeqQ * fDk;
            int fKStride = fSeqK * fDk;
            int fVStride = fSeqK * fDv;
            int fOutStride = fSeqQ * fDv;

            int dQsize = fBatch * fQStride;
            int dKsize = fBatch * fKStride;
            int dVsize = fBatch * fVStride;
            double[] dQ = new double[dQsize];
            double[] dK = new double[dKsize];
            double[] dV = new double[dVsize];

            for (int b = 0; b < fBatch; b++) {
                int gOff = b * fOutStride;
                int attnOff = b * fSoStride;

                // Extract slices
                double[] attnSlice = java.util.Arrays.copyOfRange(savedAttn, attnOff, attnOff + fSoStride);
                double[] vSlice = java.util.Arrays.copyOfRange(savedVd, b * fVStride, b * fVStride + fVStride);
                double[] kSlice = java.util.Arrays.copyOfRange(savedKd, b * fKStride, b * fKStride + fKStride);
                double[] qSlice = java.util.Arrays.copyOfRange(savedQd, b * fQStride, b * fQStride + fQStride);
                double[] gSlice = java.util.Arrays.copyOfRange(self.grad, gOff, gOff + fOutStride);

                // dV = attn^T @ d_output → [seqK, dv]
                double[] attnT = DoubleFlatGemm.flatTranspose(attnSlice, fSeqQ, fSeqK);
                double[] dVB = DoubleFlatGemm.flatMmul(attnT, fSeqK, fSeqQ, gSlice, fDv);
                System.arraycopy(dVB, 0, dV, b * fVStride, fVStride);

                // d_attn = d_output @ V^T → [seqQ, seqK]
                double[] vT = DoubleFlatGemm.flatTranspose(vSlice, fSeqK, fDv);
                double[] dAttnB = DoubleFlatGemm.flatMmul(gSlice, fSeqQ, fDv, vT, fSeqK);

                if (fDropoutScale != 1.0) {
                    dAttnB = fvc.binaryOperate(dAttnB, fDropoutScale, BinaryOperation.MULTIPLY);
                }

                // Softmax backward: ds_i = p_i * (dp_i - sum_j(p_j * dp_j))
                double[] dScoresB = new double[fSeqQ * fSeqK];
                for (int q = 0; q < fSeqQ; q++) {
                    int rowOff = q * fSeqK;
                    double dot = 0;
                    for (int j = 0; j < fSeqK; j++) {
                        dot += attnSlice[rowOff + j] * dAttnB[rowOff + j];
                    }
                    for (int j = 0; j < fSeqK; j++) {
                        double p = attnSlice[rowOff + j];
                        dScoresB[rowOff + j] = p * (dAttnB[rowOff + j] - dot) * fScale;
                    }
                }

                // dQ = dScores @ K → [seqQ, dk]
                double[] dQB = DoubleFlatGemm.flatMmul(dScoresB, seqQ, fSeqK, kSlice, fDk);
                System.arraycopy(dQB, 0, dQ, b * fQStride, fQStride);

                // dK = dScores^T @ Q → [seqK, dk]
                double[] dScoresT = DoubleFlatGemm.flatTranspose(dScoresB, fSeqQ, fSeqK);
                double[] dKB = DoubleFlatGemm.flatMmul(dScoresT, fSeqK, fSeqQ, qSlice, fDk);
                System.arraycopy(dKB, 0, dK, b * fKStride, fKStride);
            }

            inpQ.accGrad(dQ);
            inpK.accGrad(dK);
            inpV.accGrad(dV);
        };

        RereDiffTensor result = new RereDiffTensor(y, new int[]{batch, seqQ, dv}, inputs, bw,
            "scaledDotProductAttention");
        result.scalarParam = dropout;
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
        @Override public IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
                int stride, int padding, int dilation) {
            throw new UnsupportedOperationException("conv2d is not available on constant tensors");
        }
        @Override public IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
                IDiffTensor mask, double dropout) {
            throw new UnsupportedOperationException("scaledDotProductAttention is not available on constant tensors");
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
        @Override public IDiffTensor tril(int diagonal) { return wrap(value.tril(diagonal)); }
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
        @Override public void clipGradNorm(double maxNorm) {}
        @Override public void clipGradValue(double maxValue) {}
        @Override public IDiffVector flattenGrad() { return null; }
        @Override public IDiffVector flattenValue() {
            return new RereDiffVector(value.toDoubleArray());
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
