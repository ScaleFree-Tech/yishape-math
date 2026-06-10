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
import com.yishape.lab.math.compute.gpu.GpuGroupNorm;
import com.yishape.lab.math.compute.gpu.GpuReduce;
import com.yishape.lab.math.compute.hpc.HpcCross;
import com.yishape.lab.math.compute.hpc.HpcGridSample;
import com.yishape.lab.math.compute.hpc.HpcIm2col;
import com.yishape.lab.math.compute.hpc.HpcLoss;
import com.yishape.lab.math.compute.hpc.HpcTrapezoidalScan;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
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
        // Primitive stack eliminates Object[] + Boolean boxing (was ~2-3µs per call for 55 nodes).
        // Initial capacity 64 covers most graphs without resize.
        RereDiffTensor[] stack = new RereDiffTensor[64];
        byte[] state = new byte[64];  // 0=pending, 1=children_pushed
        int sp = 0;  // stack pointer
        stack[sp] = this;
        state[sp] = 0;
        sp++;
        while (sp > 0) {
            RereDiffTensor node = stack[sp - 1];
            if (state[sp - 1] == 0) {
                state[sp - 1] = 1;
                if (!visited.add(node)) { sp--; continue; }
                var inputs = node.inputs;
                for (int i = inputs.size() - 1; i >= 0; i--) {
                    RereDiffTensor inp = inputs.get(i);
                    if (!visited.contains(inp)) {
                        if (sp >= stack.length) {
                            stack = java.util.Arrays.copyOf(stack, stack.length * 2);
                            state = java.util.Arrays.copyOf(state, state.length * 2);
                        }
                        stack[sp] = inp;
                        state[sp] = 0;
                        sp++;
                    }
                }
            } else {
                sp--;
                order.add(node);
            }
        }
    }

    @Override
    public void zeroGradient() { this.grad = null; }

    @Override
    public void clipGradNorm(double maxNorm) {
        if (grad == null || !requiresGrad || maxNorm <= 0) return;
        double[] g = grad;
        // Compute L2 norm: sum of squares → sqrt
        double[] sq = COMPUTER.binaryOperate(g, g, BinaryOperation.MULTIPLY);
        double norm = Math.sqrt(COMPUTER.reduceOperate(sq, ReduceOperation.SUM));
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
            grad = COMPUTER.binaryOperate(grad, incoming, BinaryOperation.ADD);
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
            grad = COMPUTER.binaryOperate(g, pooledBuf, BinaryOperation.ADD);
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
        // -- gelu().sum() --
        if ("gelu".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) {
                    double xi = xData[i];
                    double c = 0.7978845608028654; // sqrt(2/PI)
                    double arg = c * (xi + 0.044715 * xi * xi * xi);
                    double t = Math.tanh(arg);
                    double dt = 1.0 - t * t;
                    double darg = c * (1.0 + 3.0 * 0.044715 * xi * xi);
                    dxBuf[i] = g * (0.5 * (1.0 + t) + 0.5 * xi * dt * darg);
                }
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "geluSum");
            r.exportShape = x.shape();
            int[] geluShape = x.shape().clone();
            double[] geluFactor = new double[m];
            for (int i = 0; i < m; i++) {
                double xi = xData[i];
                double c = 0.7978845608028654;
                double arg = c * (xi + 0.044715 * xi * xi * xi);
                double t = Math.tanh(arg);
                double dt = 1.0 - t * t;
                double darg = c * (1.0 + 3.0 * 0.044715 * xi * xi);
                geluFactor[i] = 0.5 * (1.0 + t) + 0.5 * xi * dt * darg;
            }
            RereDiffTensor xRefGelu = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefGelu.mul(0.0).add(IDiffTensor.constantTensor(geluFactor, geluShape)))};
            return r;
        }
        // -- mish().sum() --
        if ("mish".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) {
                    double xi = xData[i];
                    double sp = Math.log1p(Math.exp(xi));
                    double t = Math.tanh(sp);
                    double sig = 1.0 / (1.0 + Math.exp(-xi));
                    dxBuf[i] = g * (t + xi * (1.0 - t * t) * sig);
                }
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "mishSum");
            r.exportShape = x.shape();
            int[] mishShape = x.shape().clone();
            double[] mishFactor = new double[m];
            for (int i = 0; i < m; i++) {
                double xi = xData[i];
                double sp = Math.log1p(Math.exp(xi));
                double t = Math.tanh(sp);
                double sig = 1.0 / (1.0 + Math.exp(-xi));
                mishFactor[i] = t + xi * (1.0 - t * t) * sig;
            }
            RereDiffTensor xRefMish = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefMish.mul(0.0).add(IDiffTensor.constantTensor(mishFactor, mishShape)))};
            return r;
        }
        // -- sin().sum() --
        if ("sin".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g * Math.cos(xData[i]);
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "sinSum");
            r.exportShape = x.shape();
            int[] sinShape = x.shape().clone();
            double[] sinFactor = new double[m];
            for (int i = 0; i < m; i++) sinFactor[i] = Math.cos(xData[i]);
            RereDiffTensor xRefSin = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefSin.mul(0.0).add(IDiffTensor.constantTensor(sinFactor, sinShape)))};
            return r;
        }
        // -- cos().sum() --
        if ("cos".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g * (-Math.sin(xData[i]));
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "cosSum");
            r.exportShape = x.shape();
            int[] cosShape = x.shape().clone();
            double[] cosFactor = new double[m];
            for (int i = 0; i < m; i++) cosFactor[i] = -Math.sin(xData[i]);
            RereDiffTensor xRefCos = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefCos.mul(0.0).add(IDiffTensor.constantTensor(cosFactor, cosShape)))};
            return r;
        }
        // -- leakyRelu().sum() --
        if ("leakyRelu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 0.01 : scalarParam;
            double a = alpha;
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = xData[i] > 0 ? g : g * a;
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "leakyReluSum");
            r.exportShape = x.shape();
            int[] lrShape = x.shape().clone();
            double[] lrFactor = new double[m];
            for (int i = 0; i < m; i++) lrFactor[i] = xData[i] > 0 ? 1.0 : a;
            RereDiffTensor xRefLR = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefLR.mul(0.0).add(IDiffTensor.constantTensor(lrFactor, lrShape)))};
            return r;
        }
        // -- elu().sum() --
        if ("elu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 1.0 : scalarParam;
            double a = alpha;
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = xData[i] > 0 ? g : g * a * Math.exp(xData[i]);
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "eluSum");
            r.exportShape = x.shape();
            int[] eluShape = x.shape().clone();
            double[] eluFactor = new double[m];
            for (int i = 0; i < m; i++) eluFactor[i] = xData[i] > 0 ? 1.0 : a * Math.exp(xData[i]);
            RereDiffTensor xRefElu = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefElu.mul(0.0).add(IDiffTensor.constantTensor(eluFactor, eluShape)))};
            return r;
        }
        // -- selu().sum() --
        if ("selu".equals(opTag)) {
            final double seluLambda = 1.0507009873554804934193349852946;
            final double seluAlpha = 1.6732632423543772848170429916717;
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++)
                    dxBuf[i] = xData[i] > 0 ? g * seluLambda : g * seluLambda * seluAlpha * Math.exp(xData[i]);
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "seluSum");
            r.exportShape = x.shape();
            int[] seluShape = x.shape().clone();
            double[] seluFactor = new double[m];
            for (int i = 0; i < m; i++)
                seluFactor[i] = xData[i] > 0 ? seluLambda : seluLambda * seluAlpha * Math.exp(xData[i]);
            RereDiffTensor xRefSelu = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefSelu.mul(0.0).add(IDiffTensor.constantTensor(seluFactor, seluShape)))};
            return r;
        }
        // -- softplus().sum() --
        if ("softplus".equals(opTag)) {
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = g / (1.0 + Math.exp(-xData[i]));
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "softplusSum");
            r.exportShape = x.shape();
            int[] spShape = x.shape().clone();
            double[] spFactor = new double[m];
            for (int i = 0; i < m; i++) spFactor[i] = 1.0 / (1.0 + Math.exp(-xData[i]));
            RereDiffTensor xRefSP = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefSP.mul(0.0).add(IDiffTensor.constantTensor(spFactor, spShape)))};
            return r;
        }
        // -- hardtanh().sum() --
        if ("hardtanh".equals(opTag)) {
            double hmin = Double.isNaN(scalarParam) ? -1.0 : scalarParam;
            double hmax = Double.isNaN(scalarParam2) ? 1.0 : scalarParam2;
            double mn = hmin, mx = hmax;
            double[] dxBuf = AutodiffBufferPool.acquire(m);
            Consumer<RereDiffTensor> bw = self -> {
                double g = self.grad[0];
                for (int i = 0; i < m; i++) dxBuf[i] = (xData[i] > mn && xData[i] < mx) ? g : 0;
                x.accGradFromPooled(dxBuf, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, "hardtanhSum");
            r.exportShape = x.shape();
            int[] htShape = x.shape().clone();
            double[] htFactor = new double[m];
            for (int i = 0; i < m; i++) htFactor[i] = (xData[i] > mn && xData[i] < mx) ? 1.0 : 0.0;
            RereDiffTensor xRefHT = x;
            r.symbolicBackwardFn = g -> new IDiffTensor[]{g.mul(xRefHT.mul(0.0).add(IDiffTensor.constantTensor(htFactor, htShape)))};
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
        // -- gelu().sum(dim) --
        if ("gelu".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "geluSum",
                (g, xv) -> {
                    double c = 0.7978845608028654;
                    double arg = c * (xv + 0.044715 * xv * xv * xv);
                    double t = Math.tanh(arg);
                    double dt = 1.0 - t * t;
                    double darg = c * (1.0 + 3.0 * 0.044715 * xv * xv);
                    return g * (0.5 * (1.0 + t) + 0.5 * xv * dt * darg);
                }, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- mish().sum(dim) --
        if ("mish".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "mishSum",
                (g, xv) -> {
                    double sp_m = Math.log1p(Math.exp(xv));
                    double t = Math.tanh(sp_m);
                    double sig = 1.0 / (1.0 + Math.exp(-xv));
                    return g * (t + xv * (1.0 - t * t) * sig);
                }, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- sin().sum(dim) --
        if ("sin".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "sinSum",
                (g, xv) -> g * Math.cos(xv), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- cos().sum(dim) --
        if ("cos".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "cosSum",
                (g, xv) -> g * (-Math.sin(xv)), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- leakyRelu().sum(dim) --
        if ("leakyRelu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 0.01 : scalarParam;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "leakyReluSum",
                (g, xv) -> xv > 0 ? g : g * alpha, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- elu().sum(dim) --
        if ("elu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 1.0 : scalarParam;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "eluSum",
                (g, xv) -> xv > 0 ? g : g * alpha * Math.exp(xv), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- selu().sum(dim) --
        if ("selu".equals(opTag)) {
            final double seluL = 1.0507009873554804934193349852946;
            final double seluA = 1.6732632423543772848170429916717;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "seluSum",
                (g, xv) -> xv > 0 ? g * seluL : g * seluL * seluA * Math.exp(xv), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- softplus().sum(dim) --
        if ("softplus".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "softplusSum",
                (g, xv) -> g / (1.0 + Math.exp(-xv)), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- hardtanh().sum(dim) --
        if ("hardtanh".equals(opTag)) {
            double hmin = Double.isNaN(scalarParam) ? -1.0 : scalarParam;
            double hmax = Double.isNaN(scalarParam2) ? 1.0 : scalarParam2;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "hardtanhSum",
                (g, xv) -> (xv > hmin && xv < hmax) ? g : 0, xData, fOuter, fReduce, fInner, total);
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

    /** Try to fuse unaryOp + mean(dim) into a single fused node. Returns null if no pattern matches.
     *  Mirrors {@link #tryFuseSumDim(int, boolean)} but produces "reluMean" etc. tags
     *  and scales the backward gradient by 1/reduce. */
    private IDiffTensor tryFuseMeanDim(int dim, boolean keepdim) {
        if (inputs.size() != 1) return null;
        RereDiffTensor x = inputs.get(0);
        if (x == null || !x.requiresGrad) return null;
        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < dim; i++) outer *= s[i];
        int reduce = s[dim];
        int inner = 1;
        for (int i = dim + 1; i < rank(); i++) inner *= s[i];
        double invR = 1.0 / reduce;
        double[] vals = value.toDoubleArray();
        double[] sumResult = new double[outer * inner];
        for (int o = 0; o < outer; o++)
            for (int i = 0; i < inner; i++) {
                double sum = 0;
                for (int r = 0; r < reduce; r++)
                    sum += vals[(o * reduce + r) * inner + i];
                sumResult[o * inner + i] = sum;
            }
        int[] resultShape = reducedShape(dim, keepdim);
        int total = outer * reduce * inner;
        double[] xData = x.value.toDoubleArray();
        double[] result = new double[sumResult.length];
        for (int i = 0; i < result.length; i++) result[i] = sumResult[i] * invR;

        // -- square().mean(dim) --
        if ("square".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "squareMean",
                (g, xv) -> g * invR * 2.0 * xv, xData, outer, reduce, inner, total);
        }
        // -- relu().mean(dim) --
        if ("relu".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "reluMean",
                (g, xv) -> xv > 0 ? g * invR : 0, xData, outer, reduce, inner, total);
        }
        // -- exp().mean(dim) --
        if ("exp".equals(opTag)) {
            double[] eData = value.toDoubleArray();
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "expMean",
                (g, xv) -> g * invR * Math.exp(xv), xData, outer, reduce, inner, total);
            double[] expFactor = new double[total];
            for (int i = 0; i < total; i++) expFactor[i] = eData[i] * invR;
            r.symbolicBackwardFn = dimSumGradFn(x.shape(), dim, expFactor, x);
            return r;
        }
        // -- sigmoid().mean(dim) --
        if ("sigmoid".equals(opTag)) {
            double[] sigData = value.toDoubleArray();
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "sigmoidMean",
                (g, xv) -> { double sv = 1.0/(1.0+Math.exp(-xv)); return g * invR * sv * (1-sv); },
                xData, outer, reduce, inner, total);
            double[] sigFactor = new double[total];
            for (int i = 0; i < total; i++) { double sv = sigData[i]; sigFactor[i] = sv * (1-sv) * invR; }
            r.symbolicBackwardFn = dimSumGradFn(x.shape(), dim, sigFactor, x);
            return r;
        }
        // -- abs().mean(dim) --
        if ("abs".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "absMean",
                (g, xv) -> xv >= 0 ? g * invR : -g * invR, xData, outer, reduce, inner, total);
        }
        // -- tanh().mean(dim) --
        if ("tanh".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "tanhMean",
                (g, xv) -> { double t = Math.tanh(xv); return g * invR * (1.0 - t*t); },
                xData, outer, reduce, inner, total);
        }
        // -- silu().mean(dim) --
        if ("silu".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "siluMean",
                (g, xv) -> { double sig = 1.0/(1.0+Math.exp(-xv)); return g * invR * (sig + xv * sig * (1-sig)); },
                xData, outer, reduce, inner, total);
        }
        // -- log().mean(dim) --
        if ("log".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "logMean",
                (g, xv) -> g * invR / xv, xData, outer, reduce, inner, total);
        }
        // -- pow(n).mean(dim) --
        double spp = scalarParam;
        if ("pow".equals(opTag) && !Double.isNaN(spp)) {
            double param = spp;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, "powMean",
                (g, xv) -> g * invR * param * Math.pow(xv, param - 1), xData, outer, reduce, inner, total);
            r.scalarParam = param;
            return r;
        }
        // -- gelu().mean(dim) --
        if ("gelu".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "geluMean",
                (g, xv) -> {
                    double c = 0.7978845608028654;
                    double arg = c * (xv + 0.044715 * xv * xv * xv);
                    double t = Math.tanh(arg);
                    double dt = 1.0 - t * t;
                    double darg = c * (1.0 + 3.0 * 0.044715 * xv * xv);
                    return g * invR * (0.5 * (1.0 + t) + 0.5 * xv * dt * darg);
                }, xData, outer, reduce, inner, total);
        }
        // -- mish().mean(dim) --
        if ("mish".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "mishMean",
                (g, xv) -> {
                    double sp_m = Math.log1p(Math.exp(xv));
                    double t = Math.tanh(sp_m);
                    double sig = 1.0 / (1.0 + Math.exp(-xv));
                    return g * invR * (t + xv * (1.0 - t * t) * sig);
                }, xData, outer, reduce, inner, total);
        }
        // -- sin().mean(dim) --
        if ("sin".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "sinMean",
                (g, xv) -> g * invR * Math.cos(xv), xData, outer, reduce, inner, total);
        }
        // -- cos().mean(dim) --
        if ("cos".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "cosMean",
                (g, xv) -> g * invR * (-Math.sin(xv)), xData, outer, reduce, inner, total);
        }
        // -- leakyRelu().mean(dim) --
        if ("leakyRelu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 0.01 : scalarParam;
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "leakyReluMean",
                (g, xv) -> xv > 0 ? g * invR : g * invR * alpha, xData, outer, reduce, inner, total);
        }
        // -- elu().mean(dim) --
        if ("elu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 1.0 : scalarParam;
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "eluMean",
                (g, xv) -> xv > 0 ? g * invR : g * invR * alpha * Math.exp(xv), xData, outer, reduce, inner, total);
        }
        // -- selu().mean(dim) --
        if ("selu".equals(opTag)) {
            final double seluL = 1.0507009873554804934193349852946;
            final double seluA = 1.6732632423543772848170429916717;
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "seluMean",
                (g, xv) -> xv > 0 ? g * invR * seluL : g * invR * seluL * seluA * Math.exp(xv),
                xData, outer, reduce, inner, total);
        }
        // -- softplus().mean(dim) --
        if ("softplus".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "softplusMean",
                (g, xv) -> g * invR / (1.0 + Math.exp(-xv)), xData, outer, reduce, inner, total);
        }
        // -- hardtanh().mean(dim) --
        if ("hardtanh".equals(opTag)) {
            double hmin = Double.isNaN(scalarParam) ? -1.0 : scalarParam;
            double hmax = Double.isNaN(scalarParam2) ? 1.0 : scalarParam2;
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, "hardtanhMean",
                (g, xv) -> (xv > hmin && xv < hmax) ? g * invR : 0, xData, outer, reduce, inner, total);
        }
        return null;
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
        RereDiffTensor result = new RereDiffTensor(view, List.of(this), bw, "slice");
        result.setBackwardIndices(bi);
        return result;
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
        // Pattern fusion: detect unaryOp + mean(dim) → single fused node (e.g. "reluMean")
        IDiffTensor fused = tryFuseMeanDim(d, keepdim);
        if (fused != null) return fused;
        IDiffTensor s = sum(d, keepdim);
        double scale = 1.0 / dim(d);
        return s.mul(scale);
    }

    @Override
    public IDiffTensor logSumExp(int dim, boolean keepdim) {
        int d = (dim < 0 ? dim + rank() : dim);
        if (!requiresGrad) {
            double[] vals = value.toDoubleArray();
            int[] s = shape();
            int outer = 1;
            for (int i = 0; i < d; i++) outer *= s[i];
            int reduce = s[d];
            int inner = 1;
            for (int i = d + 1; i < rank(); i++) inner *= s[i];
            double[] result = new double[outer * inner];
            for (int o = 0; o < outer; o++) {
                for (int i = 0; i < inner; i++) {
                    // Find max for numerical stability
                    double max = vals[o * reduce * inner + i];
                    for (int r = 1; r < reduce; r++) {
                        double v = vals[(o * reduce + r) * inner + i];
                        if (v > max) max = v;
                    }
                    double sumExp = 0;
                    for (int r = 0; r < reduce; r++) {
                        sumExp += Math.exp(vals[(o * reduce + r) * inner + i] - max);
                    }
                    result[o * inner + i] = Math.log(sumExp) + max;
                }
            }
            return toNonDiff(new RereDoubleTensor(result, reducedShape(d, keepdim)));
        }

        int[] s = shape();
        int outer = 1;
        for (int i = 0; i < d; i++) outer *= s[i];
        int reduce = s[d];
        int inner = 1;
        for (int i = d + 1; i < rank(); i++) inner *= s[i];

        double[] vals = value.toDoubleArray();
        int[] resultShape = reducedShape(d, keepdim);
        int resultLen = outer * inner;

        // Forward: max → exp(x-max) → sum → log + max
        double[] maxVals = new double[resultLen];
        double[] sumExpVals = new double[resultLen];
        double[] result = new double[resultLen];

        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < inner; i++) {
                double max = vals[o * reduce * inner + i];
                for (int r = 1; r < reduce; r++) {
                    double v = vals[(o * reduce + r) * inner + i];
                    if (v > max) max = v;
                }
                double sumExp = 0;
                for (int r = 0; r < reduce; r++) {
                    sumExp += Math.exp(vals[(o * reduce + r) * inner + i] - max);
                }
                maxVals[o * inner + i] = max;
                sumExpVals[o * inner + i] = sumExp;
                result[o * inner + i] = Math.log(sumExp) + max;
            }
        }

        int fOuter = outer, fReduce = reduce, fInner = inner;
        double[] fMaxVals = maxVals;
        double[] fSumExpVals = sumExpVals;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fOuter * fReduce * fInner;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int r = 0; r < fReduce; r++) {
                    for (int i = 0; i < fInner; i++) {
                        int flatIdx = (o * fReduce + r) * fInner + i;
                        int gradIdx = o * fInner + i;
                        // d(lse)/dx = exp(x - max) / sumExp
                        double weight = Math.exp(vals[flatIdx] - fMaxVals[gradIdx]) / fSumExpVals[gradIdx];
                        inGrad[flatIdx] = self.grad[gradIdx] * weight;
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        RereDiffTensor rd = new RereDiffTensor(result, resultShape, List.of(this), bw, "logSumExp");
        rd.setScalarParam((double) fInner);
        return rd;
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

    /**
     * Fused softmax + cross-entropy with sparse integer labels (class indices).
     * Memory: O(B*C) for softmax probabilities (required for backward), but avoids
     * allocating a [B, C] one-hot tensor. Saves 5 intermediate graph nodes vs
     * the manual logSumExp→sub→gather→sum→div→neg chain.
     *
     * @param labels integer class indices, length = outerSize * innerSize, each in [0, classSize)
     * @param dim    the class dimension
     * @return scalar loss = mean(-log(softmax[target]))
     */
    public IDiffTensor softmaxCrossEntropySparse(int[] labels, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int[] s = shape();
        int r = rank();
        int outerSize = 1;
        for (int i = 0; i < d; i++) outerSize *= s[i];
        int classSize = s[d];
        int innerSize = 1;
        for (int i = d + 1; i < r; i++) innerSize *= s[i];
        int totalSamples = outerSize * innerSize;
        if (labels.length != totalSamples) {
            throw new IllegalArgumentException(
                "labels length " + labels.length + " != totalSamples " + totalSamples);
        }

        double[] logits = value.toDoubleArray();

        // Forward: softmax → -log(softmax[target]) → mean
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
                int t = labels[o * innerSize + in];
                if (t < 0 || t >= classSize) {
                    throw new IllegalArgumentException(
                        "Label " + t + " out of range [0, " + classSize + ") at sample " + (o * innerSize + in));
                }
                for (int c = 0; c < classSize; c++) {
                    softmax[base + c * innerSize] *= invSum;
                }
                double pt = softmax[base + t * innerSize];
                totalLoss += -Math.log(Math.max(pt, 1e-30));
            }
        }
        double meanLoss = totalLoss / totalSamples;
        double[] resultData = new double[] { meanLoss };
        int[] resultShape = new int[] { 1 };

        if (!requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(resultData, resultShape));

        // Convert int[] labels to double[] leaf tensor for graph export (GPU/HPC need labels as input data).
        // This replaces the old closure-only pattern that made labels invisible to graph serialization.
        double[] labelData = new double[labels.length];
        for (int i = 0; i < labels.length; i++) labelData[i] = labels[i];
        RereDiffTensor labelTensor = new RereDiffTensor(labelData, new int[]{totalSamples});
        labelTensor.requiresGrad = false;

        int fOuter = outerSize, fClassSize = classSize, fInner = innerSize, fTotal = totalSamples;
        double[] fSoftmax = softmax;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            RereDiffTensor labelInput = self.inputs.get(1);
            double[] labelVals = labelInput.value.toDoubleArray();
            double gradScale = self.grad[0] / fTotal;
            int m = fSoftmax.length;
            double[] inGrad = AutodiffBufferPool.acquire(m);
            System.arraycopy(fSoftmax, 0, inGrad, 0, m);
            // Subtract 1 at target positions: grad = (softmax - oneHot) / totalSamples
            for (int o = 0; o < fOuter; o++) {
                for (int in = 0; in < fInner; in++) {
                    int t = (int) Math.round(labelVals[o * fInner + in]);
                    int idx = (o * fClassSize + t) * fInner + in;
                    inGrad[idx] -= 1.0;
                }
            }
            for (int i = 0; i < m; i++) inGrad[i] *= gradScale;
            input.accGradFromPooled(inGrad, m);
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this, labelTensor), bw, "softmaxCrossEntropySparse");
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
        //
        // Always compute rowIdxTensor: flatSourcePos is already computed for forward,
        // and index may be a ConstantDiffTensor (from toNonDiff() reshape on
        // requiresGrad=false indices). The instanceof RereDiffTensor gate was
        // incorrect — it skipped rowIdx creation for non-RereDiffTensor indices,
        // causing "missing inputs" GPU worker errors.
        int trailingProduct = 1;
        for (int i = d + 1; i < r; i++) trailingProduct *= s[i];
        int numOutputRows = resultTotal / trailingProduct;
        double[] rowIdxData = new double[numOutputRows];
        for (int row = 0; row < numOutputRows; row++) {
            rowIdxData[row] = (double) (flatSourcePos[row * trailingProduct] / trailingProduct);
        }
        RereDiffTensor rowIdxTensor = new RereDiffTensor(rowIdxData, numOutputRows);
        rowIdxTensor.setRequiresGrad(false);
        List<RereDiffTensor> gatherInputs = List.of(this, rowIdxTensor);
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
        int fTotalElements = (int) computeSize(s);
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int bStride = fShape[fShape.length - 2] * fShape[fShape.length - 1];
            int bCount = fTotalElements / bStride;
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
    public IDiffTensor triu(int diagonal) {
        int r = rank();
        if (r < 2) return this; // scalar/vector: no-op
        if (!requiresGrad) {
            // Manual triu: clone and zero lower triangle
            double[] d = value.toDoubleArray().clone();
            int M = value.dim(r - 2);
            int N = value.dim(r - 1);
            int batchStride = M * N;
            int batchCount = d.length / batchStride;
            for (int b = 0; b < batchCount; b++) {
                int base = b * batchStride;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        if (j < i + diagonal) {
                            d[base + i * N + j] = 0.0;
                        }
                    }
                }
            }
            return toNonDiff(new RereDoubleTensor(d, shape()));
        }

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
                    if (j < i + diagonal) {
                        resultData[base + i * N + j] = 0.0;
                    }
                }
            }
        }

        int fDiagonal = diagonal;
        int[] fShape = s;
        int fTotalElements = (int) computeSize(s);
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int bStride = fShape[fShape.length - 2] * fShape[fShape.length - 1];
            int bCount = fTotalElements / bStride;
            // gradient flows only for non-zeroed elements (col >= row + diagonal)
            for (int b = 0; b < bCount; b++) {
                int base = b * bStride;
                for (int i = 0; i < fShape[fShape.length - 2]; i++) {
                    for (int j = 0; j < fShape[fShape.length - 1]; j++) {
                        if (j < i + fDiagonal) {
                            self.grad[base + i * fShape[fShape.length - 1] + j] = 0.0;
                        }
                    }
                }
            }
            input.accGrad(self.grad);
        };
        return new RereDiffTensor(resultData, s, List.of(this), bw, "triu");
    }

    @Override
    public IDiffTensor diag() {
        int r = rank();
        if (r < 2) return this;
        int[] s = shape();
        int M = s[r - 2];
        int N = s[r - 1];
        if (M != N) {
            throw new IllegalArgumentException("diag() requires a square matrix (last two dims equal), got " + M + "x" + N);
        }
        // Batch support: leading dims treated as batch
        int batchDim = 1;
        for (int i = 0; i < r - 2; i++) batchDim *= s[i];
        int n = (int) Math.min(M, N);

        double[] vals = value.toDoubleArray();
        int totalDiagSize = batchDim * n;
        double[] resultData = new double[totalDiagSize];

        for (int b = 0; b < batchDim; b++) {
            int base = b * M * N;
            for (int i = 0; i < n; i++) {
                resultData[b * n + i] = vals[base + i * N + i];
            }
        }

        int[] resultShape;
        if (r == 2) {
            resultShape = new int[]{n};
        } else {
            resultShape = new int[r - 1];
            for (int i = 0; i < r - 2; i++) resultShape[i] = s[i];
            resultShape[r - 2] = n;
        }

        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(resultData, resultShape));

        int fM = M, fN = N, fBatchDim = batchDim;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = fBatchDim * fM * fN;
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int b = 0; b < fBatchDim; b++) {
                int base = b * fM * fN;
                for (int i = 0; i < n; i++) {
                    inGrad[base + i * fN + i] = self.grad[b * n + i];
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this), bw, "diag");
    }

    @Override
    public IDiffTensor diagonal(int offset, int dim1, int dim2) {
        int r = rank();
        int fDim1 = (dim1 < 0 ? dim1 + r : dim1);
        int fDim2 = (dim2 < 0 ? dim2 + r : dim2);
        int[] s = shape();
        int size = (int) Math.min(s[fDim1], s[fDim2]);
        int effectiveSize = offset >= 0
            ? Math.max(0, Math.min(s[fDim1] - offset, s[fDim2]))
            : Math.max(0, Math.min(s[fDim1], s[fDim2] + offset));
        if (effectiveSize == 0) {
            throw new IllegalArgumentException("diagonal: offset " + offset + " yields empty diagonal for shape " + java.util.Arrays.toString(s));
        }
        int minDim = Math.min(fDim1, fDim2);
        int maxDim = Math.max(fDim1, fDim2);

        // Flatten leading and trailing dims
        int outer = 1;
        for (int i = 0; i < minDim; i++) outer *= s[i];
        int inner = 1;
        for (int i = maxDim + 1; i < r; i++) inner *= s[i];

        double[] vals = value.toDoubleArray();
        double[] resultData = new double[outer * effectiveSize * inner];

        // Compute diagonal extraction
        for (int o = 0; o < outer; o++) {
            for (int k = 0; k < effectiveSize; k++) {
                for (int i = 0; i < inner; i++) {
                    int[] idx = new int[r];
                    int remaining = o;
                    for (int j = minDim - 1; j >= 0; j--) {
                        idx[j] = remaining % s[j];
                        remaining /= s[j];
                    }
                    // Positive offset shifts toward dim2 (away from dim1).
                    // PyTorch/Numpy convention: diagonal(x, offset=1) gives super-diagonal.
                    idx[fDim1] = offset >= 0 ? k : k - offset;
                    idx[fDim2] = offset >= 0 ? k + offset : k;
                    remaining = i;
                    for (int j = r - 1; j > maxDim; j--) {
                        idx[j] = remaining % s[j];
                        remaining /= s[j];
                    }
                    resultData[(o * effectiveSize + k) * inner + i] = vals[flatIndex(idx, s)];
                }
            }
        }

        int[] resultShape;
        if (r == 2) {
            resultShape = new int[]{effectiveSize};
        } else {
            resultShape = new int[r - 1];
            int pos = 0;
            for (int j = 0; j < fDim1; j++) resultShape[pos++] = s[j];
            for (int j = fDim1 + 1; j < fDim2; j++) resultShape[pos++] = s[j];
            for (int j = fDim2 + 1; j < r; j++) resultShape[pos++] = s[j];
            resultShape[minDim] = effectiveSize;
        }

        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(resultData, resultShape));

        int fOffset = offset, fEffSize = effectiveSize, fOuter = outer, fInner = inner;
        int[] fShape = s;
        int ffDim1 = fDim1, ffDim2 = fDim2;
        int fMinDim = minDim, fMaxDim = maxDim;
        int fR = r;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor input = self.inputs.get(0);
            int total = (int) computeSize(fShape);
            double[] inGrad = AutodiffBufferPool.acquire(total);
            for (int o = 0; o < fOuter; o++) {
                for (int k = 0; k < fEffSize; k++) {
                    for (int i = 0; i < fInner; i++) {
                        int[] idx = new int[fR];
                        int remaining = o;
                        for (int j = fMinDim - 1; j >= 0; j--) {
                            idx[j] = remaining % fShape[j];
                            remaining /= fShape[j];
                        }
                        idx[ffDim1] = fOffset >= 0 ? k : k - fOffset;
                        idx[ffDim2] = fOffset >= 0 ? k + fOffset : k;
                        remaining = i;
                        for (int j = fR - 1; j > fMaxDim; j--) {
                            idx[j] = remaining % fShape[j];
                            remaining /= fShape[j];
                        }
                        inGrad[flatIndex(idx, fShape)] = self.grad[(o * fEffSize + k) * fInner + i];
                    }
                }
            }
            input.accGradFromPooled(inGrad, total);
        };
        return new RereDiffTensor(resultData, resultShape, List.of(this), bw, "diagonal");
    }

    @Override
    public IDiffTensor trace() {
        return diag().sum();
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
        // ✓ offsetInBlock = dimOffset * innerSize (flat units, used as flat array offset in arraycopy)
        // This IS correct: dimension offset × innerSize = flat element offset.
        // The backward code MUST do the same (see ⚠️ below).
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
        // ⚠️ cumOffset is in dimension units (count along concat dim), NOT flat array offset.
        // When used in System.arraycopy below, MUST multiply by fInnerSize.
        // For cat([1,H], dim=0) + [1,H]: cumOffset[1]=1 means "1 row offset" = 1*H flat elements.
        // Without *fInnerSize, non-first inputs read shifted gradients (dc = dh[1:] instead of dc).
        // See git history for the bug this comment prevents.
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
                // ⚠️ startOffset is in dimension units (e.g., row count). cumOffset[1]=1 means "row 1".
                // flatArrayPos = startOffset * fInnerSize — do NOT use startOffset alone.
                int startOffset = cumOffset[ai];
                double[] subGrad = new double[tFlat];
                int tBlockSize = tShape[fDim] * fInnerSize;
                for (int g = 0; g < fOuterGroups; g++) {
                    // ⚠️ CRITICAL: startOffset is in dim-units, fResultBlockSize/tBlockSize are flat-units.
                    // Both must use the same unit: multiply startOffset by fInnerSize to convert to flat units.
                    // Bug history: missing *fInnerSize caused dc to read dh[1:] (offset=1 vs offset=1*H).
                    System.arraycopy(self.grad, g * fResultBlockSize + startOffset * fInnerSize,
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
    public IDiffTensor[] split(int splitSize, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int dimSize = dim(d);
        if (splitSize <= 0) throw new IllegalArgumentException("splitSize must be positive, got " + splitSize);
        java.util.ArrayList<Integer> sizes = new java.util.ArrayList<>();
        int remaining = dimSize;
        while (remaining > 0) {
            sizes.add(Math.min(splitSize, remaining));
            remaining -= splitSize;
        }
        int[] sizeArray = new int[sizes.size()];
        for (int i = 0; i < sizes.size(); i++) sizeArray[i] = sizes.get(i);
        return split(sizeArray, d);
    }

    @Override
    public IDiffTensor[] split(int[] splitSizes, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int dimSize = dim(d);
        int sumSizes = 0;
        for (int sz : splitSizes) sumSizes += sz;
        if (sumSizes != dimSize) {
            throw new IllegalArgumentException("split sizes sum to " + sumSizes + " but dim " + dim + " has size " + dimSize);
        }
        int n = splitSizes.length;
        IDiffTensor[] result = new IDiffTensor[n];
        int offset = 0;
        for (int i = 0; i < n; i++) {
            result[i] = narrow(d, offset, splitSizes[i]);
            offset += splitSizes[i];
        }
        return result;
    }

    @Override
    public IDiffTensor[] chunk(int chunks, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int dimSize = dim(d);
        int chunkSize = (dimSize + chunks - 1) / chunks; // ceil division
        // ensure at most chunks pieces
        java.util.ArrayList<Integer> sizes = new java.util.ArrayList<>();
        int remaining = dimSize;
        for (int i = 0; i < chunks && remaining > 0; i++) {
            sizes.add(Math.min(chunkSize, remaining));
            remaining -= chunkSize;
        }
        int[] sizeArray = new int[sizes.size()];
        for (int i = 0; i < sizes.size(); i++) sizeArray[i] = sizes.get(i);
        return split(sizeArray, d);
    }

    @Override
    public IDiffTensor[] unbind(int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int dimSize = dim(d);
        IDiffTensor[] result = new IDiffTensor[dimSize];
        for (int i = 0; i < dimSize; i++) {
            // slice(d, i, i+1) removes the dim, so squeeze back
            IDiffTensor sliced = narrow(d, i, 1);
            result[i] = sliced.squeeze(d);
        }
        return result;
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

    // ==================== Phase 2: flip, roll, repeatInterleave ====================

    @Override
    public IDiffTensor flip(int... dims) {
        if (!requiresGrad) {
            // Inline flip for constant tensors (RereDoubleTensor has no flip method)
            int[] s = shape();
            double[] xd = value.toDoubleArray();
            double[] y = new double[xd.length];
            int[] fDims = new int[dims.length];
            for (int i = 0; i < dims.length; i++) fDims[i] = (dims[i] < 0 ? dims[i] + rank() : dims[i]);
            long n = value.totalSize();
            for (long flatIdx = 0; flatIdx < n; flatIdx++) {
                long dstIdx = 0;
                long srcIdx2 = flatIdx;
                long stride = 1;
                for (int d = rank() - 1; d >= 0; d--) {
                    long coord = srcIdx2 % s[d];
                    srcIdx2 /= s[d];
                    boolean match = false;
                    for (int fd : fDims) { if (fd == d) { match = true; break; } }
                    if (match) coord = s[d] - 1 - coord;
                    dstIdx += coord * stride;
                    stride *= s[d];
                }
                y[(int) dstIdx] = xd[(int) flatIdx];
            }
            return toNonDiff(new RereDoubleTensor(y, s));
        }
        int[] s = shape();
        double[] xd = value.toDoubleArray();
        double[] y = new double[xd.length];
        int[] flippedDims = new int[dims.length];
        for (int i = 0; i < dims.length; i++) flippedDims[i] = (dims[i] < 0 ? dims[i] + rank() : dims[i]);
        // Forward: copy data flipping along each dim
        long n = value.totalSize();
        for (long flatIdx = 0; flatIdx < n; flatIdx++) {
            long srcIdx = flatIdx;
            long dstIdx = 0;
            long stride = 1;
            for (int d = rank() - 1; d >= 0; d--) {
                long coord = srcIdx % s[d];
                srcIdx /= s[d];
                boolean flipDim = false;
                for (int fd : flippedDims) { if (fd == d) { flipDim = true; break; } }
                if (flipDim) coord = s[d] - 1 - coord;
                dstIdx += coord * stride;
                stride *= s[d];
            }
            y[(int) dstIdx] = xd[(int) flatIdx];
        }
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[g.length];
            // flip is self-inverse: apply same coordinate mapping to gradient
            for (long flatIdx = 0; flatIdx < n; flatIdx++) {
                long dstIdx2 = 0;
                long srcIdx2 = flatIdx;
                long stride2 = 1;
                for (int d = rank() - 1; d >= 0; d--) {
                    long coord = srcIdx2 % s[d];
                    srcIdx2 /= s[d];
                    boolean flipDim2 = false;
                    for (int fd : flippedDims) { if (fd == d) { flipDim2 = true; break; } }
                    if (flipDim2) coord = s[d] - 1 - coord;
                    dstIdx2 += coord * stride2;
                    stride2 *= s[d];
                }
                dx[(int) flatIdx] = g[(int) dstIdx2];
            }
            inp.accGrad(dx);
        };
        return new RereDiffTensor(y, s, List.of(this), bw, "flip");
    }

    @Override
    public IDiffTensor roll(int[] shifts, int[] dims) {
        if (!requiresGrad) {
            // Use split+cat for constant tensors
            IDiffTensor result = this;
            for (int i = 0; i < shifts.length; i++) {
                int d = (dims[i] < 0 ? dims[i] + rank() : dims[i]);
                int dimSize = dim(d);
                int shift = ((shifts[i] % dimSize) + dimSize) % dimSize;
                if (shift == 0) continue;
                IDiffTensor[] parts = { result.narrow(d, dimSize - shift, shift),
                                        result.narrow(d, 0, dimSize - shift) };
                result = parts[0].cat(d, parts[1]);
            }
            return result;
        }
        // For differentiable tensors, also use composition (each sub-op is differentiable)
        IDiffTensor result = this;
        for (int i = 0; i < shifts.length; i++) {
            int d = (dims[i] < 0 ? dims[i] + rank() : dims[i]);
            int dimSize = dim(d);
            int shift = ((shifts[i] % dimSize) + dimSize) % dimSize;
            if (shift == 0) continue;
            IDiffTensor[] parts = { result.narrow(d, dimSize - shift, shift),
                                    result.narrow(d, 0, dimSize - shift) };
            result = parts[0].cat(d, parts[1]);
        }
        return result;
    }

    @Override
    public IDiffTensor repeatInterleave(int repeats, int dim) {
        int d = (dim < 0 ? dim + rank() : dim);
        int dimSize = dim(d);
        // Build index array: [0,0,...,0, 1,1,...,1, ...] each repeated `repeats` times
        double[] idxData = new double[dimSize * repeats];
        for (int i = 0; i < dimSize; i++) {
            for (int r = 0; r < repeats; r++) {
                idxData[i * repeats + r] = i;
            }
        }
        IDiffTensor indices = new RereDiffTensor(idxData, new int[]{dimSize * repeats});
        indices.setRequiresGrad(false);
        IDiffTensor result = this.indexSelect(d, indices);
        if (result instanceof RereDiffTensor rt && requiresGrad) {
            rt.setOpTag("repeatInterleave");
        }
        return result;
    }

    // ==================== Phase 2: groupNorm ====================

    @Override
    public IDiffTensor groupNorm(int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps) {
        RereDiffTensor gr = (RereDiffTensor) gamma;
        long totalSize = value.totalSize();
        int[] s = shape();
        int rank = rank();
        if (rank < 2) throw new IllegalArgumentException("groupNorm requires rank >= 2, got " + rank);
        int C = s[rank - 2]; // channels — second to last dim for [N,C,H,W] or [N,C,L]
        if (C % numGroups != 0) throw new IllegalArgumentException("Channels (" + C + ") must be divisible by numGroups (" + numGroups + ")");
        int groupCh = C / numGroups; // channels per group
        // Compute outer dims product (batch or batch*spatial prefix) and spatial dims product
        int outer = 1;
        for (int i = 0; i < rank - 2; i++) outer *= s[i];
        int spatialPerSample = 1;
        for (int i = rank - 1; i < rank; i++) spatialPerSample *= s[i];
        int N = outer;

        double[] xd = value.toDoubleArray();
        double[] gd = gr.value.toDoubleArray();
        double[] bd = (beta != null) ? beta.toDoubleArray() : null;
        double[] y = new double[(int) totalSize];

        // Saved for backward
        double[] means = new double[N * numGroups];
        double[] sigmas = new double[N * numGroups];
        double[] xHat = new double[(int) totalSize];

        // Forward: normalize within each group
        int groupSize = groupCh * spatialPerSample;
        // Try HPC forward first
        boolean hpcFwdOk = com.yishape.lab.math.compute.hpc.HpcGroupNorm.tryForward(
                xd, gd, bd, C, numGroups, spatialPerSample, 1, eps, y);

        if (!hpcFwdOk) {
            // SISD fallback
            for (int n = 0; n < N; n++) {
                for (int g = 0; g < numGroups; g++) {
                    int groupIdx = n * numGroups + g;
                    double mean = 0;
                    int count = 0;
                    for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                        for (int sp = 0; sp < spatialPerSample; sp++) {
                            int idx = n * C * spatialPerSample + c * spatialPerSample + sp;
                            mean += xd[idx];
                            count++;
                        }
                    }
                    mean /= count;
                    means[groupIdx] = mean;
                    double var = 0;
                    for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                        for (int sp = 0; sp < spatialPerSample; sp++) {
                            int idx = n * C * spatialPerSample + c * spatialPerSample + sp;
                            double d = xd[idx] - mean;
                            var += d * d;
                        }
                    }
                    var /= count;
                    double sigma = Math.sqrt(var + eps);
                    sigmas[groupIdx] = sigma;
                    double invSigma = 1.0 / sigma;
                    for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                        for (int sp = 0; sp < spatialPerSample; sp++) {
                            int idx = n * C * spatialPerSample + c * spatialPerSample + sp;
                            double xh = (xd[idx] - mean) * invSigma;
                            xHat[idx] = xh;
                            y[idx] = xh * gd[c] + (bd != null ? bd[c] : 0);
                        }
                    }
                }
            }
        }

        // Try GPU backward first
        final int fGroupCh = groupCh;
        final int fHW = spatialPerSample;
        final int fN = N;
        final int fC = C;
        final int fNumGroups = numGroups;
        final double fEps = eps;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpG = self.inputs.get(1);
            RereDiffTensor inpB = (beta != null) ? self.inputs.get(2) : null;
            double[] g = self.grad;
            long m = inpX.value.totalSize();

            // Try GPU accelerated backward
            double[][] gpuResult = GpuGroupNorm.tryGroupNormBackward(
                xd, gd, g, fNumGroups, fGroupCh, fHW, fEps);
            if (gpuResult != null) {
                inpX.accGrad(gpuResult[0]);
                inpG.accGrad(gpuResult[1]);
                if (inpB != null) inpB.accGrad(gpuResult[2]);
                return;
            }

            // Try HPC accelerated backward
            double[] dxHpc = new double[(int) m];
            double[] dGammaHpc = new double[fC];
            double[] dBetaHpc = (bd != null) ? new double[fC] : new double[fC];
            if (com.yishape.lab.math.compute.hpc.HpcGroupNorm.tryBackward(
                    xd, gd, g, fC, fNumGroups, fHW, 1, fEps, dxHpc, dGammaHpc, dBetaHpc)) {
                inpX.accGrad(dxHpc);
                inpG.accGrad(dGammaHpc);
                if (inpB != null) inpB.accGrad(bd != null ? dBetaHpc : new double[fC]);
                return;
            }

            // CPU fallback
            double[] dx = AutodiffBufferPool.acquire((int) m);
            double[] dGamma = new double[fC];
            double[] dBeta = (bd != null) ? new double[fC] : null;
            int grpSize = fGroupCh * fHW;

            for (int n = 0; n < fN; n++) {
                for (int gIdx = 0; gIdx < fNumGroups; gIdx++) {
                    int groupIdx = n * fNumGroups + gIdx;
                    double sigma = sigmas[groupIdx];
                    double invSigma = 1.0 / sigma;
                    double invSigma2 = invSigma * invSigma;

                    // Compute sumG and sumGXH for this group
                    double sumG = 0, sumGXH = 0;
                    for (int c = gIdx * fGroupCh; c < (gIdx + 1) * fGroupCh; c++) {
                        for (int sp = 0; sp < fHW; sp++) {
                            int idx = n * fC * fHW + c * fHW + sp;
                            double gScaled = g[idx] * gd[c];
                            sumG += gScaled;
                            sumGXH += gScaled * xHat[idx];
                        }
                    }
                    double invGS = 1.0 / groupSize;

                    for (int c = gIdx * fGroupCh; c < (gIdx + 1) * fGroupCh; c++) {
                        for (int sp = 0; sp < fHW; sp++) {
                            int idx = n * fC * fHW + c * fHW + sp;
                            double gScaled = g[idx] * gd[c];
                            // Standard GroupNorm backward
                            dx[idx] = (gScaled - sumG * invGS - xHat[idx] * sumGXH * invGS) * invSigma;
                            dGamma[c] += g[idx] * xHat[idx];
                            if (dBeta != null) dBeta[c] += g[idx];
                        }
                    }
                }
            }
            inpX.accGradFromPooled(dx, (int) m);
            inpG.accGrad(dGamma);
            if (inpB != null && dBeta != null) inpB.accGrad(dBeta);
        };

        List<RereDiffTensor> inputs = (beta != null)
            ? List.of(this, gr, (RereDiffTensor) beta)
            : List.of(this, gr);
        RereDiffTensor result = new RereDiffTensor(y, s, inputs, bw, "groupNorm");
        result.scalarParam = eps;
        return result;
    }

    // ==================== Phase 3: Loss Functions ====================

    @Override
    public IDiffTensor smoothL1Loss(IDiffTensor target, double beta) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = value.totalSize();
        double[] loss = new double[1];
        double[] diff = new double[(int) n];
        double totalLoss = 0;
        double halfBeta = 0.5 * beta;

        for (int i = 0; i < n; i++) {
            double d = xd[i] - td[i];
            diff[i] = d;
            double absD = Math.abs(d);
            if (absD <= beta) {
                totalLoss += 0.5 * d * d / beta;
            } else {
                totalLoss += absD - halfBeta;
            }
        }
        loss[0] = totalLoss / n;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double[] g = self.grad;
            double scale = g[0] / n;
            double[] dx = new double[(int) n];
            double[] dt = new double[(int) n];
            for (int i = 0; i < n; i++) {
                double d = diff[i];
                double absD = Math.abs(d);
                double gradVal;
                if (absD <= beta) {
                    gradVal = d / beta;
                } else {
                    gradVal = Math.signum(d);
                }
                dx[i] = scale * gradVal;
                dt[i] = -scale * gradVal;
            }
            inpX.accGrad(dx);
            inpT.accGrad(dt);
        };
        return new RereDiffTensor(loss, new int[]{1}, List.of(this, tgt), bw, "smoothL1Loss");
    }
    @Override
    public IDiffTensor bceLoss(IDiffTensor target) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = value.totalSize();
        double[] loss = new double[1];
        double[] clamped = new double[(int) n];
        double totalLoss = 0;
        final double eps = 1e-7;
        for (int i = 0; i < n; i++) {
            double p = Math.max(eps, Math.min(1.0 - eps, xd[i]));
            clamped[i] = p;
            double y = td[i];
            totalLoss += -y * Math.log(p) - (1.0 - y) * Math.log(1.0 - p);
        }
        loss[0] = totalLoss / n;
        if (!requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double[] g = self.grad;
            double scale = g[0] / n;
            double[] dx = new double[(int) n];
            double[] dt = new double[(int) n];
            for (int i = 0; i < n; i++) {
                double p = clamped[i];
                double y = td[i];
                dx[i] = scale * (p - y) / (p * (1.0 - p));
                dt[i] = scale * (Math.log(1.0 - p) - Math.log(p));
            }
            inpX.accGrad(dx);
            inpT.accGrad(dt);
        };
        return new RereDiffTensor(loss, new int[]{1}, List.of(this, tgt), bw, "bceLoss");
    }

    @Override
    public IDiffTensor focalLoss(IDiffTensor target, double alpha, double gamma) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = value.totalSize();
        double[] loss = new double[1];
        double[] clamped = new double[(int) n];
        double totalLoss = 0;
        final double eps = 1e-7;
        double oneMinusAlpha = 1.0 - alpha;
        for (int i = 0; i < n; i++) {
            double p = Math.max(eps, Math.min(1.0 - eps, xd[i]));
            clamped[i] = p;
            double y = td[i];
            double pT = (y > 0.5) ? p : 1.0 - p;
            double aT = (y > 0.5) ? alpha : oneMinusAlpha;
            double focalWeight = Math.pow(1.0 - pT, gamma);
            totalLoss += aT * focalWeight * (-Math.log(pT));
        }
        loss[0] = totalLoss / n;
        if (!requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double[] g = self.grad;
            double scale = g[0] / n;
            double[] dx = new double[(int) n];
            double[] dt = new double[(int) n];
            for (int i = 0; i < n; i++) {
                double p = clamped[i];
                double y = td[i];
                if (y > 0.5) {
                    double oneMinusP = 1.0 - p;
                    double term1 = gamma * Math.pow(oneMinusP, gamma - 1.0) * (-Math.log(p));
                    double term2 = Math.pow(oneMinusP, gamma) / p;
                    dx[i] = scale * alpha * (term1 - term2);
                    dt[i] = scale * alpha * Math.pow(oneMinusP, gamma) * Math.log(p);
                } else {
                    double term1 = gamma * Math.pow(p, gamma - 1.0) * (-Math.log(1.0 - p));
                    double term2 = Math.pow(p, gamma) / (1.0 - p);
                    dx[i] = scale * oneMinusAlpha * (term1 + term2);
                    dt[i] = scale * oneMinusAlpha * Math.pow(p, gamma) * Math.log(1.0 - p);
                }
            }
            inpX.accGrad(dx);
            inpT.accGrad(dt);
        };
        return new RereDiffTensor(loss, new int[]{1}, List.of(this, tgt), bw, "focalLoss");
    }

    @Override
    public IDiffTensor diceLoss(IDiffTensor target, double smooth) {
        RereDiffTensor tgt = (RereDiffTensor) target;
        double[] xd = value.toDoubleArray();
        double[] td = tgt.value.toDoubleArray();
        long n = value.totalSize();
        double[] loss = new double[1];
        double I = 0, Sp = 0, St = 0;
        for (int i = 0; i < n; i++) {
            I += xd[i] * td[i];
            Sp += xd[i];
            St += td[i];
        }
        double denom = Sp + St + smooth;
        double dice = (2.0 * I + smooth) / denom;
        final double If = I;
        final double denomf = denom;
        loss[0] = 1.0 - dice;
        if (!requiresGrad) return new ConstantDiffTensor(new RereDoubleTensor(loss, new int[]{1}));
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpT = self.inputs.get(1);
            double g = self.grad[0];
            double invDenom2 = 1.0 / (denomf * denomf);
            double twoIplusSmooth = 2.0 * If + smooth;
            double[] dx = new double[(int) n];
            double[] dt = new double[(int) n];
            for (int i = 0; i < n; i++) {
                dx[i] = g * (twoIplusSmooth - 2.0 * td[i] * denomf) * invDenom2;
                dt[i] = g * (twoIplusSmooth - 2.0 * xd[i] * denomf) * invDenom2;
            }
            inpX.accGrad(dx);
            inpT.accGrad(dt);
        };
        return new RereDiffTensor(loss, new int[]{1}, List.of(this, tgt), bw, "diceLoss");
    }

    @Override
    public IDiffTensor nllLoss(IDiffTensor target, int classDim) {
        // Input is log-probabilities. Compute -mean(gather(logProbs, classDim, target))
        int d = (classDim < 0 ? classDim + rank() : classDim);
        RereDiffTensor tgt = (RereDiffTensor) target;
        // gather along classDim using target as indices
        IDiffTensor gathered = this.gather(d, tgt);
        IDiffTensor loss = gathered.sum().div(gathered.totalSize()).neg();
        if (loss instanceof RereDiffTensor rt) {
            rt.setOpTag("nllLoss");
        }
        return loss;
    }

    // ==================== Phase 3: Pooling ====================

    @Override
    public IDiffTensor maxPool2d(int kH, int kW, int stride, int padding) {
        int[] s = shape();
        if (s.length != 4) throw new IllegalArgumentException("maxPool2d requires 4D input [N,C,H,W], got rank " + s.length);
        int N = s[0], C = s[1], H = s[2], W = s[3];
        int effStride = (stride <= 0) ? kH : stride;
        int outH = (H + 2 * padding - kH) / effStride + 1;
        int outW = (W + 2 * padding - kW) / effStride + 1;
        long outElements = (long) N * C * outH * outW;
        int inElements = (int) value.totalSize();

        double[] xd = value.toDoubleArray();
        double[] y = new double[(int) outElements];
        int[] argmax = new int[(int) outElements];

        // Try HPC accelerated path first
        boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcPool.tryMaxPool2dForward(
                xd, N, C, H, W, kH, kW, effStride, padding, y, argmax);

        if (!hpcOk) {
            // SISD fallback: im2col-style pooling
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                            double maxVal = Double.NEGATIVE_INFINITY;
                            int maxPos = -1;
                            for (int kh = 0; kh < kH; kh++) {
                                int hIdx = oh * effStride + kh - padding;
                                if (hIdx < 0 || hIdx >= H) continue;
                                for (int kw = 0; kw < kW; kw++) {
                                    int wIdx = ow * effStride + kw - padding;
                                    if (wIdx < 0 || wIdx >= W) continue;
                                    int inIdx = ((n * C + c) * H + hIdx) * W + wIdx;
                                    if (xd[inIdx] > maxVal) {
                                        maxVal = xd[inIdx];
                                        maxPos = inIdx;
                                    }
                                }
                            }
                            y[outIdx] = maxVal;
                            argmax[outIdx] = maxPos;
                        }
                    }
                }
            }
        }

        int[] outShape = new int[]{N, C, outH, outW};
        int[] savedArgmax = argmax;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[inElements];
            if (!com.yishape.lab.math.compute.hpc.HpcPool.tryMaxPool2dBackward(
                    g, savedArgmax, N, C, H, W, outH, outW, dx)) {
                // SISD fallback
                int outLen = (int) outElements;
                for (int i = 0; i < outLen; i++) {
                    int maxIdx = savedArgmax[i];
                    if (maxIdx >= 0) dx[maxIdx] += g[i];
                }
            }
            inp.accGrad(dx);
        };
        RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(this), bw, "maxpool2d");
        result.scalarParam = Double.longBitsToDouble(((long) kH << 16) | ((long) kW << 8) | (long) stride);
        result.scalarParam2 = Double.longBitsToDouble(((long) padding << 16));
        // 6D exportShape lets HPC/GPU backends use actual input dims directly,
        // avoiding incorrect derivation when stride does not divide evenly.
        result.exportShape = new int[]{N, C, H, W, outH, outW};
        return result;
    }

    @Override
    public IDiffTensor avgPool2d(int kH, int kW, int stride, int padding) {
        int[] s = shape();
        if (s.length != 4) throw new IllegalArgumentException("avgPool2d requires 4D input [N,C,H,W], got rank " + s.length);
        int N = s[0], C = s[1], H = s[2], W = s[3];
        int effStride = (stride <= 0) ? kH : stride;
        int outH = (H + 2 * padding - kH) / effStride + 1;
        int outW = (W + 2 * padding - kW) / effStride + 1;
        long outElements = (long) N * C * outH * outW;
        int inElements = (int) value.totalSize();

        double[] xd = value.toDoubleArray();
        double[] y = new double[(int) outElements];
        // Save counts for backward SISD fallback (HPC backward recomputes internally)
        int[] counts = new int[(int) outElements];

        // Try HPC accelerated path first
        boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcPool.tryAvgPool2dForward(
                xd, N, C, H, W, kH, kW, effStride, padding, y);

        if (!hpcOk) {
            // SISD fallback
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int oh = 0; oh < outH; oh++) {
                        for (int ow = 0; ow < outW; ow++) {
                            int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                            double sum = 0;
                            int count = 0;
                            for (int kh = 0; kh < kH; kh++) {
                                int hIdx = oh * effStride + kh - padding;
                                if (hIdx < 0 || hIdx >= H) continue;
                                for (int kw = 0; kw < kW; kw++) {
                                    int wIdx = ow * effStride + kw - padding;
                                    if (wIdx < 0 || wIdx >= W) continue;
                                    sum += xd[((n * C + c) * H + hIdx) * W + wIdx];
                                    count++;
                                }
                            }
                            y[outIdx] = (count > 0) ? sum / count : 0;
                            counts[outIdx] = count;
                        }
                    }
                }
            }
        }

        int[] outShape = new int[]{N, C, outH, outW};
        int[] savedCounts = counts;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[inElements];
            if (!com.yishape.lab.math.compute.hpc.HpcPool.tryAvgPool2dBackward(
                    g, N, C, H, W, kH, kW, effStride, padding, outH, outW, dx)) {
                // SISD fallback
                int outLen = (int) outElements;
                for (int n2 = 0; n2 < N; n2++) {
                    for (int c2 = 0; c2 < C; c2++) {
                        for (int oh2 = 0; oh2 < outH; oh2++) {
                            for (int ow2 = 0; ow2 < outW; ow2++) {
                                int outIdx2 = ((n2 * C + c2) * outH + oh2) * outW + ow2;
                                int cnt = savedCounts[outIdx2];
                                if (cnt == 0) continue;
                                double gradPer = g[outIdx2] / cnt;
                                for (int kh2 = 0; kh2 < kH; kh2++) {
                                    int hIdx2 = oh2 * effStride + kh2 - padding;
                                    if (hIdx2 < 0 || hIdx2 >= H) continue;
                                    for (int kw2 = 0; kw2 < kW; kw2++) {
                                        int wIdx2 = ow2 * effStride + kw2 - padding;
                                        if (wIdx2 < 0 || wIdx2 >= W) continue;
                                        dx[((n2 * C + c2) * H + hIdx2) * W + wIdx2] += gradPer;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            inp.accGrad(dx);
        };
        RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(this), bw, "avgpool2d");
        result.scalarParam = Double.longBitsToDouble(((long) kH << 16) | ((long) kW << 8) | (long) stride);
        result.scalarParam2 = Double.longBitsToDouble(((long) padding << 16));
        // 6D exportShape lets HPC/GPU backends use actual input dims directly,
        // avoiding incorrect derivation when stride does not divide evenly.
        result.exportShape = new int[]{N, C, H, W, outH, outW};
        return result;
    }

    @Override
    public IDiffTensor adaptiveAvgPool2d(int outH, int outW) {
        int[] s = shape();
        if (s.length < 3) throw new IllegalArgumentException("adaptiveAvgPool2d requires rank >= 3, got " + s.length);
        int N, C, H, W;
        if (s.length == 3) { N = 1; C = s[0]; H = s[1]; W = s[2]; }
        else if (s.length == 4) { N = s[0]; C = s[1]; H = s[2]; W = s[3]; }
        else { N = 1; for (int i = 0; i < s.length - 3; i++) N *= s[i]; C = s[s.length - 3]; H = s[s.length - 2]; W = s[s.length - 1]; }

        // Each output position (oh, ow) maps to a contiguous block in input space
        double[] xd = value.toDoubleArray();
        double[] y = new double[N * C * outH * outW];
        // Save start/end indices for backward (SISD fallback)
        int[] startH = new int[outH + 1];
        int[] startW = new int[outW + 1];
        for (int oh = 0; oh <= outH; oh++) startH[oh] = (int) Math.floor((double) oh * H / outH);
        for (int ow = 0; ow <= outW; ow++) startW[ow] = (int) Math.floor((double) ow * W / outW);

        // Try HPC accelerated path first
        boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcPool.tryAdaptiveAvgPool2dForward(
                xd, N, C, H, W, outH, outW, y);

        if (!hpcOk) {
            // SISD fallback
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int oh = 0; oh < outH; oh++) {
                        int hStart = startH[oh], hEnd = startH[oh + 1];
                        for (int ow = 0; ow < outW; ow++) {
                            int wStart = startW[ow], wEnd = startW[ow + 1];
                            double sum = 0;
                            int cnt = 0;
                            for (int h = hStart; h < hEnd; h++) {
                                for (int w = wStart; w < wEnd; w++) {
                                    sum += xd[((n * C + c) * H + h) * W + w];
                                    cnt++;
                                }
                            }
                            y[((n * C + c) * outH + oh) * outW + ow] = (cnt > 0) ? sum / cnt : 0;
                        }
                    }
                }
            }
        }

        int[] outShape = (s.length == 4) ? new int[]{N, C, outH, outW}
            : (s.length == 3) ? new int[]{C, outH, outW} : buildOutShape(s, N, C, outH, outW);
        int[] savedStartH = startH, savedStartW = startW;
        int fH = H, fW = W, fOutH = outH, fOutW = outW, fN = N, fC = C;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[(int) inp.value.totalSize()];
            if (!com.yishape.lab.math.compute.hpc.HpcPool.tryAdaptiveAvgPool2dBackward(
                    g, fN, fC, fH, fW, fOutH, fOutW, dx)) {
                // SISD fallback
                for (int n2 = 0; n2 < fN; n2++) {
                    for (int c2 = 0; c2 < fC; c2++) {
                        for (int oh2 = 0; oh2 < fOutH; oh2++) {
                            int hStart2 = savedStartH[oh2], hEnd2 = savedStartH[oh2 + 1];
                            for (int ow2 = 0; ow2 < fOutW; ow2++) {
                                int wStart2 = savedStartW[ow2], wEnd2 = savedStartW[ow2 + 1];
                                double gradVal = g[((n2 * fC + c2) * fOutH + oh2) * fOutW + ow2];
                                double pixelCnt = (hEnd2 - hStart2) * (wEnd2 - wStart2);
                                if (pixelCnt == 0) continue;
                                double perPixel = gradVal / pixelCnt;
                                for (int h2 = hStart2; h2 < hEnd2; h2++) {
                                    for (int w2 = wStart2; w2 < wEnd2; w2++) {
                                        dx[((n2 * fC + c2) * fH + h2) * fW + w2] += perPixel;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            inp.accGrad(dx);
        };
        RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(this), bw, "adaptiveAvgPool2d");
        result.scalarParam = H;
        result.scalarParam2 = W;
        return result;
    }

    @Override
    public IDiffTensor oneHot(int numClasses) {
        // Decompose into scatter: create zero tensor, scatter 1 at class indices
        int[] s = shape();
        long n = value.totalSize();
        int[] outShape = new int[s.length + 1];
        System.arraycopy(s, 0, outShape, 0, s.length);
        outShape[s.length] = numClasses;
        // Create zero tensor of output shape
        double[] y = new double[(int) (n * numClasses)];
        double[] xd = value.toDoubleArray();
        int classStride = 1;
        for (int i = 0; i < n; i++) {
            int cls = (int) Math.round(xd[i]);
            if (cls >= 0 && cls < numClasses) {
                y[i * numClasses + cls] = 1.0;
            }
        }
        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(y, outShape));
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[(int) inp.value.totalSize()];
            for (int i = 0; i < n; i++) {
                int cls = (int) Math.round(xd[i]);
                if (cls >= 0 && cls < numClasses) {
                    dx[i] = g[i * numClasses + cls];
                }
            }
            inp.accGrad(dx);
        };
        return new RereDiffTensor(y, outShape, List.of(this), bw, "oneHot");
    }

    /** Build output shape for adaptiveAvgPool2d with non-standard input rank. */
    private static int[] buildOutShape(int[] inShape, int N, int C, int outH, int outW) {
        if (inShape.length <= 3) return new int[]{C, outH, outW};
        int prefixRank = inShape.length - 3;
        int prefix = 1;
        for (int i = 0; i < prefixRank; i++) prefix *= inShape[i];
        if (prefix != N) throw new IllegalStateException("Shape mismatch");
        int[] out = new int[inShape.length];
        for (int i = 0; i < prefixRank; i++) out[i] = inShape[i];
        out[prefixRank] = C;
        out[prefixRank + 1] = outH;
        out[prefixRank + 2] = outW;
        return out;
    }

    // ==================== Phase 5: instanceNorm ====================

    @Override
    public IDiffTensor instanceNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        RereDiffTensor gr = (RereDiffTensor) gamma;
        int[] s = shape();
        int rank = rank();
        if (rank < 2) throw new IllegalArgumentException("instanceNorm requires rank >= 2, got " + rank);
        // Input: [N, C, ...spatial...] or [N, C, H, W]
        int N, C;
        if (rank >= 2) { N = 1; for (int i = 0; i < rank - 1; i++) N *= s[i]; C = s[rank - 1]; }
        N = 1;
        for (int i = 0; i < rank - 2; i++) N *= s[i];
        C = s[rank - 2];
        int spatial = 1;
        for (int i = rank - 1; i < rank; i++) spatial *= s[i];

        double[] xd = value.toDoubleArray();
        double[] gd = gr.value.toDoubleArray();
        double[] bd = (beta != null) ? beta.toDoubleArray() : null;
        double[] y = new double[xd.length];
        double[] means = new double[N * C];
        double[] sigmas = new double[N * C];

        // Try HPC accelerated path first
        boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcNorm.tryInstanceNormForward(
                xd, gd, bd, N, C, spatial, eps, y);

        if (!hpcOk) {
            // SISD fallback
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    int instIdx = n * C + c;
                    double mean = 0;
                    for (int sp = 0; sp < spatial; sp++) {
                        mean += xd[(n * C + c) * spatial + sp];
                    }
                    mean /= spatial;
                    means[instIdx] = mean;
                    double var = 0;
                    for (int sp = 0; sp < spatial; sp++) {
                        double d = xd[(n * C + c) * spatial + sp] - mean;
                        var += d * d;
                    }
                    var /= spatial;
                    double sigma = Math.sqrt(var + eps);
                    sigmas[instIdx] = sigma;
                    double inv = 1.0 / sigma;
                    for (int sp = 0; sp < spatial; sp++) {
                        int idx = (n * C + c) * spatial + sp;
                        y[idx] = (xd[idx] - mean) * inv * gd[c] + (bd != null ? bd[c] : 0);
                    }
                }
            }
        }

        final int fN = N, fC = C, fSpatial = spatial;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpG = self.inputs.get(1);
            RereDiffTensor inpB = (beta != null) ? self.inputs.get(2) : null;
            double[] g = self.grad;
            double[] dx = new double[(int) inpX.value.totalSize()];
            double[] dGamma = new double[fC];
            double[] dBeta = (bd != null) ? new double[fC] : null;

            if (!com.yishape.lab.math.compute.hpc.HpcNorm.tryInstanceNormBackward(
                    xd, gd, g, fN, fC, fSpatial, eps, dx, dGamma, dBeta != null ? dBeta : new double[fC])) {
                // SISD fallback
                for (int n2 = 0; n2 < fN; n2++) {
                    for (int c2 = 0; c2 < fC; c2++) {
                        int instIdx = n2 * fC + c2;
                        double sigma = sigmas[instIdx];
                        double invSigma = 1.0 / sigma;
                        double sumG = 0, sumGXH = 0;
                        for (int sp2 = 0; sp2 < fSpatial; sp2++) {
                            int idx2 = (n2 * fC + c2) * fSpatial + sp2;
                            double gScaled = g[idx2] * gd[c2];
                            sumG += gScaled;
                            sumGXH += gScaled * (xd[idx2] - means[instIdx]) / sigma;
                        }
                        for (int sp2 = 0; sp2 < fSpatial; sp2++) {
                            int idx2 = (n2 * fC + c2) * fSpatial + sp2;
                            dx[idx2] = (g[idx2] * gd[c2] - sumG / fSpatial - (xd[idx2] - means[instIdx]) / sigma * sumGXH / fSpatial) * invSigma;
                            dGamma[c2] += g[idx2] * (xd[idx2] - means[instIdx]) / sigma;
                            if (dBeta != null) dBeta[c2] += g[idx2];
                        }
                    }
                }
            }
            inpX.accGrad(dx);
            inpG.accGrad(dGamma);
            if (inpB != null && dBeta != null) inpB.accGrad(dBeta);
        };

        List<RereDiffTensor> inputs = (beta != null)
            ? List.of(this, gr, (RereDiffTensor) beta)
            : List.of(this, gr);
        RereDiffTensor result = new RereDiffTensor(y, s, inputs, bw, "instanceNorm");
        result.scalarParam = eps;
        return result;
    }

    // ==================== Phase 5: diagEmbed ====================

    @Override
    public IDiffTensor diagEmbed(int offset, int dim1, int dim2) {
        // Input is a 1D vector or batched vectors. Creates a matrix with input on diagonal.
        int[] s = shape();
        long n = value.totalSize();
        int diagLen = (int) n; // Default: all elements go on diagonal

        // For simple 1D input [D], output is [D, D]
        int M = diagLen + Math.abs(offset);
        int outM = M, outN = M;

        double[] xd = value.toDoubleArray();
        double[] y = new double[outM * outN];
        for (int i = 0; i < diagLen; i++) {
            int row = (offset >= 0) ? i : i - offset;
            int col = (offset >= 0) ? i + offset : i;
            if (row >= 0 && row < outM && col >= 0 && col < outN) {
                y[row * outN + col] = xd[i];
            }
        }

        if (!requiresGrad) return toNonDiff(new RereDoubleTensor(y, new int[]{outM, outN}));
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[diagLen];
            for (int i = 0; i < diagLen; i++) {
                int row = (offset >= 0) ? i : i - offset;
                int col = (offset >= 0) ? i + offset : i;
                if (row >= 0 && row < outM && col >= 0 && col < outN) {
                    dx[i] = g[row * outN + col];
                }
            }
            inp.accGrad(dx);
        };
        return new RereDiffTensor(y, new int[]{outM, outN}, List.of(this), bw, "diagEmbed");
    }

    // ==================== Phase 5: dropout2d ====================

    @Override
    public IDiffTensor dropout2d(double p) {
        if (p <= 0) return this;
        if (!requiresGrad) return this;
        int[] s = shape();
        if (s.length < 3) throw new IllegalArgumentException("dropout2d requires rank >= 3, got " + s.length);
        int rankLocal = s.length;
        int nn = 1;
        for (int i = 0; i < rankLocal - 2; i++) nn *= s[i];
        final int N = nn;
        final int C = s[rankLocal - 2];

        double[] xd = value.toDoubleArray();
        final int total = xd.length;
        double[] y = new double[total];
        // Create channel-wise mask: same for all spatial positions
        double scale = 1.0 / (1.0 - p);
        double[] chMask = new double[N * C];
        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < N * C; i++) {
            chMask[i] = (rng.nextDouble() >= p) ? scale : 0;
        }

        final int spatial = total / (N * C);
        for (int nc = 0; nc < N * C; nc++) {
            double m = chMask[nc];
            for (int sp = 0; sp < spatial; sp++) {
                y[nc * spatial + sp] = xd[nc * spatial + sp] * m;
            }
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[total];
            for (int nc2 = 0; nc2 < N * C; nc2++) {
                double m = chMask[nc2];
                for (int sp2 = 0; sp2 < spatial; sp2++) {
                    dx[nc2 * spatial + sp2] = g[nc2 * spatial + sp2] * m;
                }
            }
            inp.accGrad(dx);
        };
        return new RereDiffTensor(y, s, List.of(this), bw, "dropout2d");
    }

    // ==================== Phase 5: depthwiseConv1d ====================

    @Override
    public IDiffTensor depthwiseConv1d(IDiffTensor weight, int stride, int padding) {
        RereDiffTensor w = (RereDiffTensor) weight;
        int[] s = shape();
        if (s.length != 3) throw new IllegalArgumentException("depthwiseConv1d requires 3D input [N,C,L], got rank " + s.length);
        int N = s[0], C = s[1], L = s[2];
        int kSize = w.dim(w.rank() - 1);
        int effStride = (stride <= 0) ? 1 : stride;
        int outL = (L + 2 * padding - kSize) / effStride + 1;

        double[] xd = value.toDoubleArray();
        double[] wd = w.value.toDoubleArray();
        double[] y = new double[N * C * outL];

        // SISD depthwise convolution: per-channel conv1d
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int ol = 0; ol < outL; ol++) {
                    double sum = 0;
                    for (int k = 0; k < kSize; k++) {
                        int inIdx = ol * effStride + k - padding;
                        if (inIdx >= 0 && inIdx < L) {
                            sum += xd[(n * C + c) * L + inIdx] * wd[c * kSize + k];
                        }
                    }
                    y[(n * C + c) * outL + ol] = sum;
                }
            }
        }

        int[] outShape = new int[]{N, C, outL};
        final int fN = N, fC = C, fL = L, fOutL = outL, fK = kSize, fStride = effStride, fPad = padding;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpW = self.inputs.get(1);
            double[] g = self.grad;
            double[] dx = new double[(int) inpX.value.totalSize()];
            double[] dw = new double[(int) inpW.value.totalSize()];

            // Try HPC backward first (the HPC forward is causal-only, so forward stays SISD)
            int perSampleIn = fC * fL;
            int perSampleOut = fC * fOutL;
            for (int n2 = 0; n2 < fN; n2++) {
                double[] xSlice = new double[perSampleIn];
                double[] gSlice = new double[perSampleOut];
                double[] dxSlice = new double[perSampleIn];
                double[] dwSlice = new double[(int) fC * fK];
                System.arraycopy(xd, n2 * perSampleIn, xSlice, 0, perSampleIn);
                System.arraycopy(g, n2 * perSampleOut, gSlice, 0, perSampleOut);

                if (!com.yishape.lab.math.compute.hpc.HpcDepthwiseConv1d.tryBackward(
                        xSlice, wd, gSlice, fL, fC, fK, fStride, fPad, dxSlice, dwSlice)) {
                    // SISD fallback for this sample
                    for (int c2 = 0; c2 < fC; c2++) {
                        for (int ol2 = 0; ol2 < fOutL; ol2++) {
                            double gradVal = gSlice[(c2) * fOutL + ol2];
                            for (int k2 = 0; k2 < fK; k2++) {
                                int inIdx2 = ol2 * fStride + k2 - fPad;
                                if (inIdx2 >= 0 && inIdx2 < fL) {
                                    int xIdx = c2 * fL + inIdx2;
                                    dxSlice[xIdx] += gradVal * wd[c2 * fK + k2];
                                    dwSlice[c2 * fK + k2] += gradVal * xSlice[xIdx];
                                }
                            }
                        }
                    }
                }
                // Accumulate back
                for (int i = 0; i < perSampleIn; i++) dx[n2 * perSampleIn + i] += dxSlice[i];
                for (int i = 0; i < fC * fK; i++) dw[i] += dwSlice[i];
            }
            inpX.accGrad(dx);
            inpW.accGrad(dw);
        };
        RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(this, w), bw, "depthwiseConv1d");
        result.scalarParam = Double.longBitsToDouble(((long) L << 16) | (long) kSize);
        result.scalarParam2 = C;
        return result;
    }

    // ==================== Phase 5: interpolate ====================

    @Override
    public IDiffTensor interpolate(double scaleFactor, String mode) {
        int[] s = shape();
        if (s.length < 3) throw new IllegalArgumentException("interpolate requires rank >= 3, got " + s.length);
        int N, C, H, W;
        if (s.length == 3) { N = 1; C = s[0]; H = s[1]; W = s[2]; }
        else if (s.length == 4) { N = s[0]; C = s[1]; H = s[2]; W = s[3]; }
        else { N = 1; for (int i = 0; i < s.length - 3; i++) N *= s[i]; C = s[s.length - 3]; H = s[s.length - 2]; W = s[s.length - 1]; }

        int outH = (int) Math.floor(H * scaleFactor);
        int outW = (int) Math.floor(W * scaleFactor);
        boolean bilinear = "bilinear".equals(mode);

        double[] xd = value.toDoubleArray();
        double[] y = new double[N * C * outH * outW];

        // Save interpolation weights for backward (SISD fallback)
        double[][] savedWeights = bilinear ? new double[N * C * outH * outW][4] : null;
        int[][] savedIndices = bilinear ? new int[N * C * outH * outW][4] : new int[N * C * outH * outW][1];

        // Try HPC forward first
        boolean hpcFwdOk = false;
        if (bilinear) {
            hpcFwdOk = com.yishape.lab.math.compute.hpc.HpcInterpolate.tryBilinearForward(
                    xd, N, C, H, W, outH, outW, y);
        } else {
            hpcFwdOk = com.yishape.lab.math.compute.hpc.HpcInterpolate.tryNearestForward(
                    xd, N, C, H, W, outH, outW, y);
        }

        if (!hpcFwdOk) {
            // SISD fallback
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int oh = 0; oh < outH; oh++) {
                        double srcH = (oh + 0.5) / scaleFactor - 0.5;
                        int h0 = (int) Math.floor(srcH);
                        int h1 = Math.min(h0 + 1, H - 1);
                        h0 = Math.max(h0, 0);
                        double dh = srcH - h0;
                        for (int ow = 0; ow < outW; ow++) {
                            double srcW = (ow + 0.5) / scaleFactor - 0.5;
                            int w0 = (int) Math.floor(srcW);
                            int w1 = Math.min(w0 + 1, W - 1);
                            w0 = Math.max(w0, 0);
                            double dw = srcW - w0;
                            int outIdx = ((n * C + c) * outH + oh) * outW + ow;

                            if (bilinear) {
                                double v00 = xd[((n * C + c) * H + h0) * W + w0];
                                double v01 = xd[((n * C + c) * H + h0) * W + w1];
                                double v10 = xd[((n * C + c) * H + h1) * W + w0];
                                double v11 = xd[((n * C + c) * H + h1) * W + w1];
                                double out = (1 - dh) * (1 - dw) * v00 + (1 - dh) * dw * v01
                                           + dh * (1 - dw) * v10 + dh * dw * v11;
                                y[outIdx] = out;
                                savedWeights[outIdx] = new double[]{(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw};
                                savedIndices[outIdx] = new int[]{h0 * W + w0, h0 * W + w1, h1 * W + w0, h1 * W + w1};
                            } else {
                                int hNear = (int) Math.round(srcH);
                                int wNear = (int) Math.round(srcW);
                                hNear = Math.max(0, Math.min(H - 1, hNear));
                                wNear = Math.max(0, Math.min(W - 1, wNear));
                                y[outIdx] = xd[((n * C + c) * H + hNear) * W + wNear];
                                savedIndices[outIdx] = new int[]{hNear * W + wNear};
                            }
                        }
                    }
                }
            }
        }

        int[] outShape = (s.length == 4) ? new int[]{N, C, outH, outW}
            : (s.length == 3) ? new int[]{C, outH, outW} : buildOutShape(s, N, C, outH, outW);
        final int fH = H, fW = W, fOutH = outH, fOutW = outW, fN = N, fC = C;
        final boolean fBilinear = bilinear;
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[(int) inp.value.totalSize()];

            // Try HPC backward first
            boolean hpcBwdOk = false;
            if (fBilinear) {
                hpcBwdOk = com.yishape.lab.math.compute.hpc.HpcInterpolate.tryBilinearBackward(
                        g, fN, fC, fH, fW, fOutH, fOutW, dx);
            } else {
                hpcBwdOk = com.yishape.lab.math.compute.hpc.HpcInterpolate.tryNearestBackward(
                        g, fN, fC, fH, fW, fOutH, fOutW, dx);
            }

            if (!hpcBwdOk) {
                // SISD fallback
                for (int n2 = 0; n2 < fN; n2++) {
                    for (int c2 = 0; c2 < fC; c2++) {
                        for (int oh2 = 0; oh2 < fOutH; oh2++) {
                            for (int ow2 = 0; ow2 < fOutW; ow2++) {
                                int outIdx2 = ((n2 * fC + c2) * fOutH + oh2) * fOutW + ow2;
                                double gradVal = g[outIdx2];
                                int baseIn = ((n2 * fC + c2) * fH);
                                if (fBilinear && savedWeights != null) {
                                    double[] wts = savedWeights[outIdx2];
                                    int[] idxs = savedIndices[outIdx2];
                                    for (int k = 0; k < 4; k++) {
                                        dx[baseIn * fW + idxs[k]] += gradVal * wts[k];
                                    }
                                } else {
                                    int[] idxs = savedIndices[outIdx2];
                                    dx[baseIn * fW + idxs[0]] += gradVal;
                                }
                            }
                        }
                    }
                }
            }
            inp.accGrad(dx);
        };
        RereDiffTensor result = new RereDiffTensor(y, outShape, List.of(this), bw, "interpolate");
        result.scalarParam = scaleFactor;
        result.scalarParam2 = "bilinear".equals(mode) ? 0.0 : 1.0;
        return result;
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

    // ==================== Matrix Decomposition Ops ====================

    @Override
    public IDiffTensor logDet() {
        int[] s = shape();
        if (rank() != 2 || s[0] != s[1]) {
            throw new IllegalArgumentException(
                "logDet requires square 2D matrix, got shape " + Arrays.toString(s));
        }
        int n = s[0];
        double[] xd = value.toDoubleArray();
        IMatrix<Double> A = Linalg.fromArray(xd, n, n);
        double logDet;
        try {
            var luDecomp = com.yishape.lab.math.linalg.decomposition.Decomps.createLU();
            luDecomp.decompose(A);
            double det = luDecomp.getDeterminant();
            logDet = Math.log(Math.abs(det));
        } catch (Exception e) {
            // Singular matrix: log|det| = -inf
            logDet = Double.NEGATIVE_INFINITY;
        }
        double[] result = {logDet};
        final int fN = n;
        final double fLogDet = logDet;

        Consumer<RereDiffTensor> bw = self -> {
            if (Double.isInfinite(fLogDet)) return; // no gradient for singular matrix
            RereDiffTensor inp = self.inputs.get(0);
            double gradOut = self.grad[0];
            // ∂log|det(A)|/∂A = A^{-T}
            double[] inpData = inp.value.toDoubleArray();
            IMatrix<Double> Amat = Linalg.fromArray(inpData, fN, fN);
            IMatrix<Double> I = Linalg.eye(fN);
            IMatrix<Double> Ainv = Amat.solve(I);
            IMatrix<Double> AinvT = Ainv.transpose();
            double[] gradA = AinvT.flatten().toDoubleArray();
            for (int i = 0; i < gradA.length; i++) gradA[i] *= gradOut;
            inp.accGrad(gradA);
        };

        return new RereDiffTensor(result, new int[]{1}, List.of(this), bw, "logDet");
    }

    @Override
    public IDiffTensor[] slogDet() {
        int[] s = shape();
        if (rank() != 2 || s[0] != s[1]) {
            throw new IllegalArgumentException(
                "slogDet requires square 2D matrix, got shape " + Arrays.toString(s));
        }
        int n = s[0];
        double[] xd = value.toDoubleArray();
        IMatrix<Double> A = Linalg.fromArray(xd, n, n);
        var luDecomp = com.yishape.lab.math.linalg.decomposition.Decomps.createLU();
        luDecomp.decompose(A);
        double det = luDecomp.getDeterminant();
        double signVal = Math.signum(det);
        double logDet = Math.log(Math.abs(det));
        double[] signArr = {signVal};
        double[] logDetArr = {logDet};
        final int fN = n;

        // sign tensor — no gradient flow (constant)
        RereDiffTensor signTensor = new RereDiffTensor(signArr, new int[]{1});
        signTensor.setRequiresGrad(false);

        // logDet tensor — differentiable
        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double gradOut = self.grad[0];
            double[] inpData = inp.value.toDoubleArray();
            IMatrix<Double> Amat = Linalg.fromArray(inpData, fN, fN);
            IMatrix<Double> I = Linalg.eye(fN);
            IMatrix<Double> Ainv = Amat.solve(I);
            IMatrix<Double> AinvT = Ainv.transpose();
            double[] gradA = AinvT.flatten().toDoubleArray();
            for (int i = 0; i < gradA.length; i++) gradA[i] *= gradOut;
            inp.accGrad(gradA);
        };

        RereDiffTensor logDetTensor = new RereDiffTensor(logDetArr, new int[]{1}, List.of(this), bw, "logDet");
        return new IDiffTensor[]{signTensor, logDetTensor};
    }

    @Override
    public IDiffTensor nuclearNorm() {
        int[] s = shape();
        if (rank() != 2) {
            throw new IllegalArgumentException(
                "nuclearNorm requires 2D matrix, got rank " + rank());
        }
        int m = s[0], nDim = s[1];
        double[] xd = value.toDoubleArray();
        IMatrix<Double> A = Linalg.fromArray(xd, m, nDim);
        var svdResult = A.svd();
        IMatrix<Double> U = svdResult.getFirst();
        var Svec = svdResult.getSecond();
        IMatrix<Double> VT = svdResult.getThird();

        double[] sVals = Svec.toDoubleArray();
        double nuclear = 0;
        for (double sv : sVals) nuclear += sv;
        double[] result = {nuclear};
        final int fM = m;
        final int fN = nDim;
        final int k = sVals.length;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double gradOut = self.grad[0];
            // ∂||A||_* / ∂A = U @ V^T (using thin SVD factors)
            double[][] uData = U.toDoubleArray();    // m×k (2D)
            double[][] vtData = VT.toDoubleArray();   // n×n (2D)
            // Extract V from V^T: V_hat = first k rows of VT (shape k×n)
            double[] vFlat = new double[k * fN];
            for (int i = 0; i < k; i++) {
                System.arraycopy(vtData[i], 0, vFlat, i * fN, fN);
            }
            // Compute U(m×k) @ V_hat(k×n) via matrix multiply
            IMatrix<Double> Umat = Linalg.fromArray(
                U.flatten().toDoubleArray(), fM, k);
            IMatrix<Double> Vmat = Linalg.fromArray(vFlat, k, fN);
            IMatrix<Double> gradMat = Umat.mmul(Vmat);
            double[] gradA = gradMat.flatten().toDoubleArray();
            for (int i = 0; i < gradA.length; i++) gradA[i] *= gradOut;
            inp.accGrad(gradA);
        };

        return new RereDiffTensor(result, new int[]{1}, List.of(this), bw, "nuclearNorm");
    }

    @Override
    public IDiffTensor ctcLoss(IDiffTensor targets, IDiffTensor inputLengths, IDiffTensor targetLengths) {
        int[] s = shape();
        if (rank() != 3) {
            throw new IllegalArgumentException(
                "ctcLoss requires 3D input [T, N, C], got shape " + Arrays.toString(s));
        }
        int T = s[0], N = s[1], C = s[2];
        double[] xd = value.toDoubleArray();          // [T, N, C] row-major
        double[] tgtData = targets.toDoubleArray();    // [N, S] row-major
        double[] inLenData = inputLengths.toDoubleArray(); // [N]
        double[] tgtLenData = targetLengths.toDoubleArray(); // [N]
        int S = targets.shape()[targets.rank() - 1];

        double totalLoss = 0;
        double[][] batchGrads = new double[N][];

        for (int batch = 0; batch < N; batch++) {
            int inLen = (int) inLenData[batch];
            int tgtLen = (int) tgtLenData[batch];
            if (inLen <= 0 || inLen > T) inLen = T;

            // Extract logProbs for this batch: [T, C] row-major → flat [T*C]
            double[] batchLP = new double[T * C];
            for (int t = 0; t < T; t++) {
                for (int c = 0; c < C; c++) {
                    batchLP[t * C + c] = xd[t * (N * C) + batch * C + c];
                }
            }

            // Extract labels for this batch
            int[] labels = new int[tgtLen];
            for (int i = 0; i < tgtLen; i++) {
                labels[i] = (int) tgtData[batch * S + i];
            }

            double[] lossOut = new double[1];
            double[] gradOut = new double[T * C];
            boolean ok = HpcLoss.tryCtcForwardBackward(batchLP, labels, tgtLen, inLen, C, lossOut, gradOut);
            if (!ok) {
                throw new UnsupportedOperationException(
                    "CTC HPC native runtime unavailable. Java CTC fallback not yet implemented.");
            }
            batchGrads[batch] = gradOut;
            totalLoss += lossOut[0];
        }

        double avgLoss = totalLoss / N;
        double[] lossArr = {avgLoss};
        final int fT = T, fN = N, fC = C;
        final double[][] fBatchGrads = batchGrads;
        final double[] fInLenData = inLenData;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double gradOut = self.grad[0];
            double[] dx = new double[(int) inp.value.totalSize()];

            for (int batch = 0; batch < fN; batch++) {
                int inLen = (int) fInLenData[batch];
                if (inLen <= 0 || inLen > fT) inLen = fT;
                double[] g = fBatchGrads[batch];
                if (g != null) {
                    for (int t = 0; t < inLen; t++) {
                        for (int c = 0; c < fC; c++) {
                            dx[t * (fN * fC) + batch * fC + c] += g[t * fC + c] * gradOut / fN;
                        }
                    }
                }
            }
            inp.accGrad(dx);
        };

        List<RereDiffTensor> inputs = new ArrayList<>();
        inputs.add(this);
        inputs.add((RereDiffTensor) targets);
        inputs.add((RereDiffTensor) inputLengths);
        inputs.add((RereDiffTensor) targetLengths);

        return new RereDiffTensor(lossArr, new int[]{1}, inputs, bw, "ctcLoss");
    }

    // ==================== cross — 3D vector cross product ====================

    @Override
    public IDiffTensor cross(IDiffTensor other) {
        RereDiffTensor o = (RereDiffTensor) other;
        int[] sA = shape();
        int[] sB = o.shape();
        if (sA[sA.length - 1] != 3 || sB[sB.length - 1] != 3) {
            throw new IllegalArgumentException("cross requires last dim = 3, got " +
                Arrays.toString(sA) + " and " + Arrays.toString(sB));
        }
        int[] bcShape = TensorShape.broadcastShape(sA, sB);
        long outSize = 1;
        for (int d : bcShape) outSize *= d;
        double[] aData = value.toDoubleArray();
        double[] bData = o.value.toDoubleArray();
        double[] y = new double[(int) outSize];
        long numTriplets = outSize / 3;
        final long fOutSize = outSize;    // effectively-final capture for lambda
        final long fNumTriplets = numTriplets;
        final int[] fBcShape = bcShape;

        // Pre-broadcast data to common shape for HPC / SIMD path
        double[] aBC = new double[(int) outSize];
        double[] bBC = new double[(int) outSize];
        broadcastTo(aData, sA, aBC, bcShape);
        broadcastTo(bData, sB, bBC, bcShape);

        // Try HPC first, fall back to SISD
        boolean hpcUsed = HpcCross.tryCrossForward(aBC, bBC, (int) numTriplets, y);

        if (!hpcUsed) {
            // SISD fallback: element-wise cross product with broadcast indexing
            for (long t = 0; t < numTriplets; t++) {
                long flatIdx = t * 3;
                int[] bcIdx = unlinearizeInt((int) flatIdx, bcShape);
                int ai = flatIndexFromBroadcast(bcIdx, sA, bcShape);
                int bi = flatIndexFromBroadcast(bcIdx, sB, bcShape);
                int aBase = (ai / 3) * 3, bBase = (bi / 3) * 3;
                y[(int) flatIdx]     = aData[aBase + 1] * bData[bBase + 2] - aData[aBase + 2] * bData[bBase + 1];
                y[(int) flatIdx + 1] = aData[aBase + 2] * bData[bBase + 0] - aData[aBase + 0] * bData[bBase + 2];
                y[(int) flatIdx + 2] = aData[aBase + 0] * bData[bBase + 1] - aData[aBase + 1] * bData[bBase + 0];
            }
        }
        final int[] fSA = sA, fSB = sB;
        final boolean fHpcUsed = hpcUsed;
        final double[] fABC = aBC, fBBC = bBC;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpA = self.inputs.get(0);
            RereDiffTensor inpB = self.inputs.get(1);
            double[] g = self.grad;
            double[] da = new double[(int) inpA.value.totalSize()];
            double[] db = new double[(int) inpB.value.totalSize()];

            if (fHpcUsed) {
                // HPC backward on broadcasted shapes, then un-broadcast
                double[] daBC = new double[(int) fOutSize];
                double[] dbBC = new double[(int) fOutSize];
                if (HpcCross.tryCrossBackward(g, fABC, fBBC, (int) fNumTriplets, daBC, dbBC)) {
                    unbroadcastSum(daBC, fBcShape, da, fSA);
                    unbroadcastSum(dbBC, fBcShape, db, fSB);
                    inpA.accGrad(da);
                    inpB.accGrad(db);
                    return;
                }
                // If HPC backward fails, fall through to SISD
            }

            double[] bd = inpB.value.toDoubleArray();
            double[] ad = inpA.value.toDoubleArray();
            for (long t = 0; t < fNumTriplets; t++) {
                long flatIdx = t * 3;
                int[] bcIdx = unlinearizeInt((int) flatIdx, fBcShape);
                int ai = flatIndexFromBroadcast(bcIdx, fSA, fBcShape);
                int bi = flatIndexFromBroadcast(bcIdx, fSB, fBcShape);
                int aBase = (ai / 3) * 3, bBase = (bi / 3) * 3;
                double g0 = g[(int) flatIdx];
                double g1 = g[(int) flatIdx + 1];
                double g2 = g[(int) flatIdx + 2];
                double b0v = bd[bBase], b1v = bd[bBase + 1], b2v = bd[bBase + 2];
                double a0v = ad[aBase], a1v = ad[aBase + 1], a2v = ad[aBase + 2];
                da[aBase]     += b1v * g2 - b2v * g1;
                da[aBase + 1] += b2v * g0 - b0v * g2;
                da[aBase + 2] += b0v * g1 - b1v * g0;
                db[bBase]     += g1 * a2v - g2 * a1v;
                db[bBase + 1] += g2 * a0v - g0 * a2v;
                db[bBase + 2] += g0 * a1v - g1 * a0v;
            }
            inpA.accGrad(da);
            inpB.accGrad(db);
        };

        int[] outShape = bcShape.clone();
        return new RereDiffTensor(y, outShape, List.of(this, o), bw, "cross");
    }

    // ==================== gridSample — differentiable image warp ====================

    @Override
    public IDiffTensor gridSample(IDiffTensor grid, String mode, String paddingMode) {
        int[] s = shape();
        if (rank() != 4) throw new IllegalArgumentException("gridSample input must be [N,C,H,W], got " + Arrays.toString(s));
        int N = s[0], C = s[1], H = s[2], W = s[3];
        int[] gs = grid.shape();
        int outH = gs[1], outW = gs[2];
        double[] xd = value.toDoubleArray();
        double[] gd = grid.toDoubleArray();
        double[] y = new double[N * C * outH * outW];
        boolean bilinear = "bilinear".equals(mode);
        int padModeIdx = switch (paddingMode) {
            case "border" -> 1;
            case "reflection" -> 2;
            default -> 0;
        };
        int modeIdx = bilinear ? 0 : 1;

        // Try HPC first, fall back to SISD
        boolean hpcUsed = HpcGridSample.tryGridSampleForward(
                xd, gd, N, C, H, W, outH, outW, modeIdx, padModeIdx, y);

        if (!hpcUsed) {
            // SISD fallback
            double[][] savedWeights = bilinear ? new double[N * C * outH * outW][] : null;
            int[][] savedIndices = new int[N * C * outH * outW][];
            for (int n = 0; n < N; n++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double gx = gd[((n * outH + oh) * outW + ow) * 2];
                        double gy = gd[((n * outH + oh) * outW + ow) * 2 + 1];
                        double px = (gx + 1.0) * 0.5 * (W - 1);
                        double py = (gy + 1.0) * 0.5 * (H - 1);

                        if (bilinear) {
                            int ix0 = (int) Math.floor(px), iy0 = (int) Math.floor(py);
                            int ix1 = ix0 + 1, iy1 = iy0 + 1;
                            double wx1 = px - ix0, wy1 = py - iy0;
                            double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1;
                            int[][] corners = {{iy0, ix0}, {iy0, ix1}, {iy1, ix0}, {iy1, ix1}};
                            double[] weights = {wx0 * wy0, wx1 * wy0, wx0 * wy1, wx1 * wy1};
                            for (int c = 0; c < C; c++) {
                                int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                                double val = 0;
                                int[] idxs = new int[4];
                                double[] wts = new double[4];
                                for (int k = 0; k < 4; k++) {
                                    int sy = clampCoord(corners[k][0], H, paddingMode);
                                    int sx = clampCoord(corners[k][1], W, paddingMode);
                                    idxs[k] = sy * W + sx;
                                    wts[k] = weights[k];
                                    if (sy >= 0 && sy < H && sx >= 0 && sx < W) {
                                        val += wts[k] * xd[((n * C + c) * H + sy) * W + sx];
                                    }
                                }
                                y[outIdx] = val;
                                savedWeights[outIdx] = wts;
                                savedIndices[outIdx] = idxs;
                            }
                        } else {
                            int ix = (int) Math.round(px), iy = (int) Math.round(py);
                            ix = clampCoord(ix, W, paddingMode);
                            iy = clampCoord(iy, H, paddingMode);
                            for (int c = 0; c < C; c++) {
                                int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    y[outIdx] = xd[((n * C + c) * H + iy) * W + ix];
                                }
                                savedIndices[outIdx] = new int[]{iy * W + ix};
                            }
                        }
                    }
                }
            }
            // Capture saved indices/weights for Java backward
            final double[][] fWeights = savedWeights;
            final int[][] fIndices = savedIndices;
            final boolean fBilinear = bilinear;
            final int fN2 = N, fC2 = C, fH2 = H, fW2 = W, fOutH2 = outH, fOutW2 = outW;

            Consumer<RereDiffTensor> bw = self -> {
                RereDiffTensor inp = self.inputs.get(0);
                double[] g = self.grad;
                double[] dx = new double[(int) inp.value.totalSize()];
                for (int n2 = 0; n2 < fN2; n2++) {
                    for (int c2 = 0; c2 < fC2; c2++) {
                        for (int oh2 = 0; oh2 < fOutH2; oh2++) {
                            for (int ow2 = 0; ow2 < fOutW2; ow2++) {
                                int outIdx = ((n2 * fC2 + c2) * fOutH2 + oh2) * fOutW2 + ow2;
                                double gv = g[outIdx];
                                if (gv == 0) continue;
                                int[] idxs = fIndices[outIdx];
                                int inBase = ((n2 * fC2 + c2) * fH2) * fW2;
                                if (fBilinear && fWeights != null && fWeights[outIdx] != null) {
                                    double[] wts = fWeights[outIdx];
                                    for (int k = 0; k < 4; k++) {
                                        int flat = idxs[k];
                                        if (flat >= 0 && flat < fH2 * fW2)
                                            dx[inBase + flat] += gv * wts[k];
                                    }
                                } else {
                                    int flat = idxs[0];
                                    if (flat >= 0 && flat < fH2 * fW2)
                                        dx[inBase + flat] += gv;
                                }
                            }
                        }
                    }
                }
                inp.accGrad(dx);
            };

            RereDiffTensor result = new RereDiffTensor(y, new int[]{N, C, outH, outW},
                List.of(this, (RereDiffTensor) grid), bw, "gridSample");
            result.scalarParam = Double.longBitsToDouble(((long) H << 16) | (long) W);
            result.scalarParam2 = Double.longBitsToDouble(((long) padModeIdx << 8) | (long) modeIdx);
            return result;
        }

        // HPC path: backward recomputes sampling positions from input+grid
        final int fN = N, fC = C, fH = H, fW = W, fOutH = outH, fOutW = outW;
        final int fModeIdx = modeIdx, fPadModeIdx = padModeIdx;
        final double[] fInputData = xd;
        final double[] fGridData = gd;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inp = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[(int) inp.value.totalSize()];
            if (HpcGridSample.tryGridSampleBackward(g, fInputData, fGridData,
                    fN, fC, fH, fW, fOutH, fOutW, fModeIdx, fPadModeIdx, dx)) {
                inp.accGrad(dx);
                return;
            }
            // Should not happen: if HPC forward succeeded, HPC backward should too.
            // Fallback: leave gradient zero (rare edge case).
        };

        RereDiffTensor result = new RereDiffTensor(y, new int[]{N, C, outH, outW},
            List.of(this, (RereDiffTensor) grid), bw, "gridSample");
        result.scalarParam = Double.longBitsToDouble(((long) H << 16) | (long) W);
        result.scalarParam2 = Double.longBitsToDouble(((long) padModeIdx << 8) | (long) modeIdx);
        return result;
    }

    private static int clampCoord(int coord, int limit, String paddingMode) {
        return switch (paddingMode) {
            case "border" -> Math.clamp(coord, 0, limit - 1);
            case "reflection" -> {
                int r = Math.abs(coord) % (2 * limit);
                yield r >= limit ? 2 * limit - 1 - r : r;
            }
            default -> coord; // "zeros": return out-of-bounds as-is (handled as 0 in sampling)
        };
    }

    // ==================== trapezoidalScan — Mamba SSM ====================

    @Override
    public IDiffTensor trapezoidalScan(IDiffTensor delta, IDiffTensor A, IDiffTensor B,
                                        IDiffTensor C, IDiffTensor D) {
        int[] s = shape(); // U: [B, L, D]
        if (rank() != 3) throw new IllegalArgumentException(
            "trapezoidalScan: U must be [B,L,D], got " + Arrays.toString(s));
        int bSize = s[0], seqLen = s[1], dim = s[2];

        double[] uData = value.toDoubleArray();
        double[] deltaData = delta.toDoubleArray();
        double[] aData = A.toDoubleArray();
        double[] bData = B.toDoubleArray();
        double[] cData = C.toDoubleArray();
        double[] dData = D.toDoubleArray();

        boolean aIsVec = A.rank() == 1;
        boolean dIsScalar = D.totalSize() == 1;
        boolean deltaBroadcast = delta.shape()[delta.rank() - 2] == 1; // [B,1,D]
        int fAIsVec = aIsVec ? 1 : 0;
        int fDIsScalar = dIsScalar ? 1 : 0;
        int fDeltaBroadcast = deltaBroadcast ? 1 : 0;

        double[] y = new double[bSize * seqLen * dim];
        double[] savedH = new double[bSize * seqLen * dim];
        double[] savedABar = new double[bSize * seqLen * dim];
        double[] savedBBarU = new double[bSize * seqLen * dim];

        // Try HPC first, fall back to SISD
        boolean hpcUsed = HpcTrapezoidalScan.tryTrapezoidalScanForward(
                uData, deltaData, aData, bData, cData, dData,
                bSize, seqLen, dim, fAIsVec, fDIsScalar, fDeltaBroadcast,
                y, savedH, savedABar, savedBBarU);

        if (!hpcUsed) {
            // SISD fallback
            for (int b = 0; b < bSize; b++) {
                double[] h = new double[dim]; // current hidden state (init = 0)
                for (int t = 0; t < seqLen; t++) {
                    double[] hNext = new double[dim];
                    for (int d = 0; d < dim; d++) {
                        double dt = deltaBroadcast
                            ? deltaData[(b * 1 + 0) * dim + d]
                            : deltaData[(b * seqLen + t) * dim + d];
                        double aVal = aIsVec ? aData[d] : aData[b * dim + d];
                        double ut = uData[(b * seqLen + t) * dim + d];
                        double bt = bData[(b * seqLen + t) * dim + d];

                        double aBar = Math.exp(dt * aVal);
                        double bBar = dt * bt;
                        hNext[d] = aBar * h[d] + bBar * ut;

                        int flatT = (b * seqLen + t) * dim + d;
                        savedABar[flatT] = aBar;
                        savedBBarU[flatT] = bBar * ut;
                    }
                    for (int d = 0; d < dim; d++) {
                        double ct = cData[(b * seqLen + t) * dim + d];
                        double dtScalar = dIsScalar ? dData[0] : dData[d];
                        double ut = uData[(b * seqLen + t) * dim + d];
                        int flatT = (b * seqLen + t) * dim + d;
                        y[flatT] = ct * hNext[d] + dtScalar * ut;
                        savedH[flatT] = hNext[d];
                    }
                    h = hNext;
                }
            }

            final int fB2 = bSize, fL2 = seqLen, fD2 = dim;
            final double[] fDeltaData2 = deltaData, fUData2 = uData;
            final double[] fAData2 = aData, fBData2 = bData, fCData2 = cData, fDData2 = dData;
            final boolean fAIsVec2 = aIsVec, fDIsScalar2 = dIsScalar, fDeltaBroadcast2 = deltaBroadcast;

            Consumer<RereDiffTensor> bw = self -> {
                RereDiffTensor inpU = self.inputs.get(0);
                RereDiffTensor inpDelta = self.inputs.get(1);
                RereDiffTensor inpA = self.inputs.get(2);
                RereDiffTensor inpB = self.inputs.get(3);
                RereDiffTensor inpC = self.inputs.get(4);
                RereDiffTensor inpD = self.inputs.get(5);
                double[] gy = self.grad;

                double[] dU = new double[(int) inpU.value.totalSize()];
                double[] dDelta = new double[(int) inpDelta.value.totalSize()];
                double[] dA = new double[(int) inpA.value.totalSize()];
                double[] dB = new double[(int) inpB.value.totalSize()];
                double[] dC = new double[(int) inpC.value.totalSize()];
                double[] dD = new double[(int) inpD.value.totalSize()];

                for (int b = 0; b < fB2; b++) {
                    double[] dh = new double[fD2];
                    for (int t = fL2 - 1; t >= 0; t--) {
                        for (int d = 0; d < fD2; d++) {
                            int flatT = (b * fL2 + t) * fD2 + d;
                            double ct = fCData2[(b * fL2 + t) * fD2 + d];
                            double dt = fDeltaBroadcast2
                                ? fDeltaData2[(b * 1 + 0) * fD2 + d]
                                : fDeltaData2[(b * fL2 + t) * fD2 + d];
                            double aVal = fAIsVec2 ? fAData2[d] : fAData2[b * fD2 + d];
                            double aBar = savedABar[flatT];
                            double ut = fUData2[(b * fL2 + t) * fD2 + d];
                            double bt = fBData2[(b * fL2 + t) * fD2 + d];

                            double gy_t = gy[flatT];
                            dC[flatT] += gy_t * savedH[flatT];
                            if (fDIsScalar2) dD[0] += gy_t * ut;
                            else dD[d] += gy_t * ut;

                            dh[d] += gy_t * ct;
                            double bBar = dt * bt;
                            dU[flatT] += dh[d] * bBar;
                            if (fDIsScalar2) dU[flatT] += gy_t * fDData2[0];
                            else dU[flatT] += gy_t * fDData2[d];

                            dB[flatT] += dh[d] * dt * ut;
                            dDelta[fDeltaBroadcast2 ? (b * 1 + 0) * fD2 + d : flatT] +=
                                dh[d] * bt * ut;
                            dDelta[fDeltaBroadcast2 ? (b * 1 + 0) * fD2 + d : flatT] +=
                                dh[d] * aBar * ((t > 0) ? savedH[flatT - fD2] : 0) * aVal;

                            double hPrev = (t > 0) ? savedH[flatT - fD2] : 0;
                            int aOff = fAIsVec2 ? d : (b * fD2 + d);
                            dA[aOff] += dh[d] * aBar * hPrev * dt;
                            dh[d] = dh[d] * aBar;
                        }
                    }
                }
                inpU.accGrad(dU);
                inpDelta.accGrad(dDelta);
                inpA.accGrad(dA);
                inpB.accGrad(dB);
                inpC.accGrad(dC);
                inpD.accGrad(dD);
            };

            return new RereDiffTensor(y, s, List.of(this, (RereDiffTensor) delta,
                (RereDiffTensor) A, (RereDiffTensor) B, (RereDiffTensor) C, (RereDiffTensor) D),
                bw, "trapezoidalScan");
        }

        // HPC path: backward uses HPC backward with saved state arrays
        final int fB = bSize, fL = seqLen, fD = dim;
        final int fAIsVecI = fAIsVec, fDIsScalarI = fDIsScalar, fDeltaBroadcastI = fDeltaBroadcast;
        final double[] fUData = uData, fDeltaData = deltaData;
        final double[] fAData = aData, fBData = bData, fCData = cData, fDData = dData;

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpU = self.inputs.get(0);
            RereDiffTensor inpDelta = self.inputs.get(1);
            RereDiffTensor inpA = self.inputs.get(2);
            RereDiffTensor inpB = self.inputs.get(3);
            RereDiffTensor inpC = self.inputs.get(4);
            RereDiffTensor inpD = self.inputs.get(5);
            double[] gy = self.grad;

            double[] dU = new double[(int) inpU.value.totalSize()];
            double[] dDelta = new double[(int) inpDelta.value.totalSize()];
            double[] dA = new double[(int) inpA.value.totalSize()];
            double[] dB = new double[(int) inpB.value.totalSize()];
            double[] dC = new double[(int) inpC.value.totalSize()];
            double[] dD = new double[(int) inpD.value.totalSize()];

            if (HpcTrapezoidalScan.tryTrapezoidalScanBackward(
                    gy, fUData, fDeltaData, fAData, fBData, fCData, fDData,
                    savedH, savedABar, savedBBarU,
                    fB, fL, fD, fAIsVecI, fDIsScalarI, fDeltaBroadcastI,
                    dU, dDelta, dA, dB, dC, dD)) {
                inpU.accGrad(dU);
                inpDelta.accGrad(dDelta);
                inpA.accGrad(dA);
                inpB.accGrad(dB);
                inpC.accGrad(dC);
                inpD.accGrad(dD);
                return;
            }
            // Should not happen: if HPC forward succeeded, HPC backward should too.
            // Fallback: leave gradients zero.
        };

        return new RereDiffTensor(y, s, List.of(this, (RereDiffTensor) delta,
            (RereDiffTensor) A, (RereDiffTensor) B, (RereDiffTensor) C, (RereDiffTensor) D),
            bw, "trapezoidalScan");
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

    static int flatIndex(int[] indices, int[] shape) {
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

    /**
     * Broadcast src data (in srcShape) to dst array (in dstShape).
     * Standard NumPy broadcast semantics: dimensions of size 1 are stretched.
     */
    private static void broadcastTo(double[] src, int[] srcShape, double[] dst, int[] dstShape) {
        int diff = dstShape.length - srcShape.length;
        long dstSize = 1;
        for (int d : dstShape) dstSize *= d;
        for (long flat = 0; flat < dstSize; flat++) {
            int[] dstIdx = unlinearizeInt((int) flat, dstShape);
            int srcFlat = flatIndexFromBroadcast(dstIdx, srcShape, dstShape);
            dst[(int) flat] = src[srcFlat];
        }
    }

    /**
     * Sum-reduce a broadcasted gradient array back to the original (non-broadcasted) shape.
     * For each element in the broadcasted array, maps it to the corresponding original index
     * and accumulates the gradient.
     */
    private static void unbroadcastSum(double[] gradBC, int[] bcShape, double[] gradOrig, int[] origShape) {
        long bcSize = 1;
        for (int d : bcShape) bcSize *= d;
        for (long bcFlat = 0; bcFlat < bcSize; bcFlat++) {
            int[] bcIdx = unlinearizeInt((int) bcFlat, bcShape);
            int origFlat = flatIndexFromBroadcast(bcIdx, origShape, bcShape);
            gradOrig[origFlat] += gradBC[(int) bcFlat];
        }
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
    public IDiffTensor rmsNorm(IDiffTensor gamma, double eps) {
        RereDiffTensor gr = (RereDiffTensor) gamma;
        long totalSize = value.totalSize();
        int features = value.dim(rank() - 1);
        if (totalSize % features != 0) {
            throw new IllegalArgumentException(
                "Input size (" + totalSize + ") not divisible by features (" + features + ")");
        }
        int batch = (int) (totalSize / features);

        double[] xd = value.toDoubleArray();
        double[] gd = gr.value.toDoubleArray();
        double[] y = new double[(int) totalSize];
        double[] rmsVals = new double[batch];

        // Try HPC accelerated path first
        boolean hpcOk = com.yishape.lab.math.compute.hpc.HpcNorm.tryRMSNormForward(
                xd, gd, batch, features, eps, y, rmsVals);

        if (!hpcOk) {
            // SISD fallback: y = x / sqrt(mean(x^2) + eps) * gamma
            for (int p = 0; p < batch; p++) {
                int off = p * features;
                double sumSq = 0;
                for (int j = 0; j < features; j++) sumSq += xd[off + j] * xd[off + j];
                double rms = Math.sqrt(sumSq / features + eps);
                rmsVals[p] = rms;
                double invRms = 1.0 / rms;
                for (int j = 0; j < features; j++) {
                    y[off + j] = xd[off + j] * invRms * gd[j];
                }
            }
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            RereDiffTensor inpG = self.inputs.get(1);
            double[] g = self.grad;
            int m = (int) inpX.value.totalSize();
            double[] dx = AutodiffBufferPool.acquire(m);
            double[] dGamma = new double[features];

            if (!com.yishape.lab.math.compute.hpc.HpcNorm.tryRMSNormBackward(
                    xd, gd, g, rmsVals, batch, features, eps, dx, dGamma)) {
                // SISD fallback
                for (int p = 0; p < batch; p++) {
                    int off = p * features;
                    double rms = rmsVals[p];
                    double invRms = 1.0 / rms;
                    double invRms3 = invRms * invRms * invRms; // 1/rms^3
                    double sumGX = 0;
                    for (int j = 0; j < features; j++) {
                        sumGX += g[off + j] * xd[off + j];
                    }
                    double scale = sumGX * invRms3 / features;
                    for (int j = 0; j < features; j++) {
                        int idx = off + j;
                        double xi = xd[idx];
                        dx[idx] = g[idx] * invRms * gd[j] - gd[j] * xi * scale;
                    }
                    for (int j = 0; j < features; j++) {
                        int idx = off + j;
                        dGamma[j] += g[idx] * xd[idx] * invRms;
                    }
                }
            }
            inpX.accGradFromPooled(dx, m);
            inpG.accGradFromPooled(dGamma, features);
        };
        RereDiffTensor result = new RereDiffTensor(y, shape(), List.of(this, gr), bw, "rmsNorm");
        result.scalarParam = eps;
        return result;
    }

    @Override
    public IDiffTensor embedding(IDiffTensor indices) {
        // Differentiable embedding lookup: gather rows from this embedding table.
        // this.shape = [vocabSize, embeddingDim], indices can be any integer tensor.
        // Returns [*indices.shape, embeddingDim]
        RereDiffTensor idx = (RereDiffTensor) indices;
        int[] idxShape = idx.shape();
        int embeddingDim = dim(rank() - 1);
        // Flatten indices to 1D, gather, then reshape.
        // NOTE: opTag is deliberately NOT overridden to "embedding" on the reshape node.
        // The previous setOpTag caused GPU graph execution to crash at yishape_math_gpu
        // graph.rs:906 (index out of bounds: len 1, idx 1) because the Rust GPU handler
        // for "embedding" expected 2 direct inputs (weight + indices), but the tag was on
        // the reshape node which has only 1 input (the gather result).
        // Individual indexSelect(gather) + reshape are both fully supported by GPU/HPC.
        IDiffTensor flatIdx = idx.reshape((int) idx.totalSize());
        return this.indexSelect(0, flatIdx).reshape(outShape(idxShape, embeddingDim));
    }

    /** Helper: compute output shape [*idxShape, embeddingDim]. */
    private static int[] outShape(int[] idxShape, int embeddingDim) {
        int[] outShape = new int[idxShape.length + 1];
        System.arraycopy(idxShape, 0, outShape, 0, idxShape.length);
        outShape[idxShape.length] = embeddingDim;
        return outShape;
    }

    @Override
    public IDiffTensor rope(int dim, int maxLen, double base) {
        // Rotary Position Embedding: apply rotation to pairs in the last dimension.
        // dim is typically headDim/2 (half the actual dimension).
        // Uses structural loops over positions (OK per CLAUDE.md exception for structural loops).
        int[] sh = shape();
        int lastDim = sh[rank() - 1];
        long totalSize = value.totalSize();
        int headDim = lastDim;
        if (headDim != dim * 2) {
            // If dim doesn't match half of last dim, use dim as-is
            headDim = dim * 2;
        }

        int fHeadDim = headDim; // final copy for lambda
        double[] xd = value.toDoubleArray();
        double[] y = new double[(int) totalSize];
        int seqLen = (int) (totalSize / fHeadDim);
        int fSeqLen = seqLen; // final copy for lambda

        // Pre-compute sin/cos tables for all positions and all pairs
        // Each position pos and pair i uses angle = pos / base^(2i/dim)
        // where i ranges 0..dim-1 (half-dim)
        int halfDim = dim;
        int fHalfDim = halfDim; // final copy for lambda
        double[] cosTable = new double[seqLen * halfDim];
        double[] sinTable = new double[seqLen * halfDim];
        for (int pos = 0; pos < seqLen; pos++) {
            int baseOff = pos * halfDim;
            for (int i = 0; i < halfDim; i++) {
                double theta = pos / Math.pow(base, 2.0 * i / dim);
                cosTable[baseOff + i] = Math.cos(theta);
                sinTable[baseOff + i] = Math.sin(theta);
            }
        }

        // Apply rotation: for each position pair, [x1, x2] → [x1*c - x2*s, x1*s + x2*c]
        for (int pos = 0; pos < seqLen; pos++) {
            int baseOff = pos * fHeadDim;
            int tblOff = pos * fHalfDim;
            for (int i = 0; i < fHalfDim; i++) {
                int idx2i = baseOff + 2 * i;
                int idx2i1 = idx2i + 1;
                double x1 = xd[idx2i];
                double x2 = xd[idx2i1];
                double c = cosTable[tblOff + i];
                double s = sinTable[tblOff + i];
                y[idx2i] = x1 * c - x2 * s;
                y[idx2i1] = x1 * s + x2 * c;
            }
        }

        Consumer<RereDiffTensor> bw = self -> {
            RereDiffTensor inpX = self.inputs.get(0);
            double[] g = self.grad;
            double[] dx = new double[g.length];
            for (int pos = 0; pos < fSeqLen; pos++) {
                int baseOff = pos * fHeadDim;
                int tblOff = pos * fHalfDim;
                for (int i = 0; i < fHalfDim; i++) {
                    int idx2i = baseOff + 2 * i;
                    int idx2i1 = idx2i + 1;
                    double dY1 = g[idx2i];
                    double dY2 = g[idx2i1];
                    double c = cosTable[tblOff + i];
                    double s = sinTable[tblOff + i];
                    // Forward: [y1, y2] = [c, -s; s, c] @ [x1, x2]
                    // dL/d[x1,x2] = R^T @ [dY1, dY2] = [c, s; -s, c] @ [dY1, dY2]
                    dx[idx2i] = dY1 * c + dY2 * s;
                    dx[idx2i1] = -dY1 * s + dY2 * c;
                }
            }
            inpX.accGrad(dx);
        };
        RereDiffTensor result = new RereDiffTensor(y, shape(), List.of(this), bw, "rope");
        result.scalarParam = dim;
        result.scalarParam2 = base;
        return result;
    }

    @Override
    public IDiffTensor[] lstmCell(IDiffTensor x, IDiffTensor hPrev, IDiffTensor cPrev,
                                   IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
        // gates = x @ W_i^T + hPrev @ W_h^T + bias
        // x: [batch, inputSize], wInput: [4H, inputSize] → wInput^T: [inputSize, 4H]
        // hPrev: [batch, hiddenSize], wHidden: [4H, hiddenSize] → wHidden^T: [hiddenSize, 4H]
        IDiffTensor gates = x.mmul(wInput.transpose()).add(hPrev.mmul(wHidden.transpose()));
        if (bias != null) gates = gates.add(bias);
        IDiffTensor[] splitGates = gates.chunk(4, gates.rank() - 1);
        IDiffTensor i = splitGates[0].sigmoid();
        IDiffTensor f = splitGates[1].sigmoid();
        IDiffTensor o = splitGates[2].sigmoid();
        IDiffTensor g = splitGates[3].tanh();
        IDiffTensor c = f.mul(cPrev).add(i.mul(g));
        IDiffTensor h = o.mul(c.tanh());
        return new IDiffTensor[]{h, c};
    }

    @Override
    public IDiffTensor gruCell(IDiffTensor x, IDiffTensor hPrev,
                               IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
        // gates = x @ W_i^T + hPrev @ W_h^T + bias
        // x: [batch, inputSize], wInput: [3H, inputSize]
        // hPrev: [batch, hiddenSize], wHidden: [3H, hiddenSize]
        // GRU: z/r/n with r gating hidden part of n
        IDiffTensor xGates = x.mmul(wInput.transpose());
        IDiffTensor hGates = hPrev.mmul(wHidden.transpose());
        IDiffTensor[] xParts = xGates.chunk(3, xGates.rank() - 1);
        IDiffTensor[] hParts = hGates.chunk(3, hGates.rank() - 1);
        IDiffTensor z = xParts[0].add(hParts[0]).sigmoid();
        IDiffTensor r = xParts[1].add(hParts[1]).sigmoid();
        IDiffTensor nPre = xParts[2];
        if (bias != null) {
            IDiffTensor[] biasParts = bias.chunk(3, bias.rank() - 1);
            nPre = nPre.add(biasParts[2]);
        }
        IDiffTensor n = nPre.add(r.mul(hParts[2])).tanh();
        IDiffTensor h = z.neg().add(1.0).mul(n).add(z.mul(hPrev));
        return h;
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
        @Override public IDiffTensor logSumExp(int dim, boolean keepdim) {
            int d = (dim < 0 ? dim + value.rank() : dim);
            double[] vals = value.toDoubleArray();
            int[] s = shape();
            int outer = 1;
            for (int i = 0; i < d; i++) outer *= s[i];
            int reduce = s[d];
            int inner = 1;
            for (int i = d + 1; i < rank(); i++) inner *= s[i];
            double[] result = new double[outer * inner];
            for (int o = 0; o < outer; o++) {
                for (int i = 0; i < inner; i++) {
                    double max = vals[o * reduce * inner + i];
                    for (int r = 1; r < reduce; r++) {
                        double v = vals[(o * reduce + r) * inner + i];
                        if (v > max) max = v;
                    }
                    double sumExp = 0;
                    for (int r = 0; r < reduce; r++) {
                        sumExp += Math.exp(vals[(o * reduce + r) * inner + i] - max);
                    }
                    result[o * inner + i] = Math.log(sumExp) + max;
                }
            }
            int[] reducedShape = new int[s.length];
            System.arraycopy(s, 0, reducedShape, 0, s.length);
            reducedShape[d] = keepdim ? 1 : 0;
            if (!keepdim) {
                int[] trimmed = new int[s.length - 1];
                int pos = 0;
                for (int j = 0; j < s.length; j++) if (j != d) trimmed[pos++] = reducedShape[j];
                reducedShape = trimmed;
            } else {
                reducedShape[d] = 1;
            }
            return wrap(new RereDoubleTensor(result, reducedShape));
        }
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
        @Override public IDiffTensor softmaxCrossEntropySparse(int[] labels, int dim) {
            int d = (dim < 0 ? dim + value.rank() : dim);
            IDoubleTensor sm = value.softmax(d);
            double[] smData = sm.toDoubleArray();
            int outer = 1;
            for (int i = 0; i < d; i++) outer *= value.dim(i);
            int C = value.dim(d);
            int inner = 1;
            for (int i = d + 1; i < value.rank(); i++) inner *= value.dim(i);
            double totalLoss = 0;
            for (int o = 0; o < outer; o++) {
                for (int in = 0; in < inner; in++) {
                    int t = labels[o * inner + in];
                    int idx = (o * C + t) * inner + in;
                    totalLoss += -Math.log(Math.max(smData[idx], 1e-30));
                }
            }
            int totalSamples = outer * inner;
            return wrap(new RereDoubleTensor(new double[]{totalLoss / totalSamples}, new int[]{1}));
        }
        @Override public IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
                int stride, int padding, int dilation) {
            throw new UnsupportedOperationException("conv2d is not available on constant tensors");
        }
        @Override public IDiffTensor[] lstmCell(IDiffTensor x, IDiffTensor hPrev, IDiffTensor cPrev,
                IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
            throw new UnsupportedOperationException("lstmCell not available on constant tensors");
        }
        @Override public IDiffTensor gruCell(IDiffTensor x, IDiffTensor hPrev,
                IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
            throw new UnsupportedOperationException("gruCell not available on constant tensors");
        }
        @Override public IDiffTensor groupNorm(int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps) {
            return computeGroupNorm(numGroups, gamma, beta, eps);
        }
        @Override public IDiffTensor flip(int... dims) {
            int[] s = shape();
            double[] xd = value.toDoubleArray();
            double[] y = new double[xd.length];
            int[] fd = new int[dims.length];
            for (int i = 0; i < dims.length; i++) fd[i] = (dims[i] < 0 ? dims[i] + value.rank() : dims[i]);
            long n = value.totalSize();
            for (long flatIdx = 0; flatIdx < n; flatIdx++) {
                long dstIdx = 0, srcIdx = flatIdx, stride = 1;
                for (int d = value.rank() - 1; d >= 0; d--) {
                    long coord = srcIdx % s[d]; srcIdx /= s[d];
                    boolean match = false;
                    for (int f : fd) { if (f == d) { match = true; break; } }
                    if (match) coord = s[d] - 1 - coord;
                    dstIdx += coord * stride;
                    stride *= s[d];
                }
                y[(int) dstIdx] = xd[(int) flatIdx];
            }
            return wrap(new RereDoubleTensor(y, s));
        }
        @Override public IDiffTensor roll(int[] shifts, int[] dims) {
            IDoubleTensor result = value;
            for (int i = 0; i < shifts.length; i++) {
                int d = (dims[i] < 0 ? dims[i] + value.rank() : dims[i]);
                int dimSize = value.dim(d);
                int shift = ((shifts[i] % dimSize) + dimSize) % dimSize;
                if (shift == 0) continue;
                IDoubleTensor[] parts = { result.narrow(d, dimSize - shift, shift),
                                          result.narrow(d, 0, dimSize - shift) };
                result = parts[0].cat(d, parts[1]);
            }
            return wrap(result);
        }
        @Override public IDiffTensor repeatInterleave(int repeats, int dim) {
            int d = (dim < 0 ? dim + value.rank() : dim);
            int dimSize = value.dim(d);
            double[] idxData = new double[dimSize * repeats];
            for (int i = 0; i < dimSize; i++) {
                for (int r = 0; r < repeats; r++) idxData[i * repeats + r] = i;
            }
            return wrap(value.indexSelect(d, new RereDoubleTensor(idxData, new int[]{dimSize * repeats})));
        }
        @Override public IDiffTensor smoothL1Loss(IDiffTensor target, double beta) {
            double[] xd = value.toDoubleArray();
            double[] td = target.toDoubleArray();
            double total = 0;
            double halfBeta = 0.5 * beta;
            for (int i = 0; i < xd.length; i++) {
                double d = Math.abs(xd[i] - td[i]);
                total += (d <= beta) ? 0.5 * d * d / beta : d - halfBeta;
            }
            return wrap(new RereDoubleTensor(new double[]{total / xd.length}, new int[]{1}));
        }
	        @Override public IDiffTensor bceLoss(IDiffTensor target) {
	            double[] xd = value.toDoubleArray();
	            double[] td = target.toDoubleArray();
	            double total = 0;
	            final double eps = 1e-7;
	            for (int i = 0; i < xd.length; i++) {
	                double p = Math.max(eps, Math.min(1.0 - eps, xd[i]));
	                double y = td[i];
	                total += -y * Math.log(p) - (1.0 - y) * Math.log(1.0 - p);
	            }
	            return wrap(new RereDoubleTensor(new double[]{total / xd.length}, new int[]{1}));
	        }
	        @Override public IDiffTensor focalLoss(IDiffTensor target, double alpha, double gamma) {
	            double[] xd = value.toDoubleArray();
	            double[] td = target.toDoubleArray();
	            double total = 0;
	            final double eps = 1e-7;
	            double oma = 1.0 - alpha;
	            for (int i = 0; i < xd.length; i++) {
	                double p = Math.max(eps, Math.min(1.0 - eps, xd[i]));
	                double y = td[i];
	                double pT = (y > 0.5) ? p : 1.0 - p;
	                double aT = (y > 0.5) ? alpha : oma;
	                total += aT * Math.pow(1.0 - pT, gamma) * (-Math.log(pT));
	            }
	            return wrap(new RereDoubleTensor(new double[]{total / xd.length}, new int[]{1}));
	        }
	        @Override public IDiffTensor diceLoss(IDiffTensor target, double smooth) {
	            double[] xd = value.toDoubleArray();
	            double[] td = target.toDoubleArray();
	            double I = 0, Sp = 0, St = 0;
	            for (int i = 0; i < xd.length; i++) {
	                I += xd[i] * td[i];
	                Sp += xd[i];
	                St += td[i];
	            }
	            double denom = Sp + St + smooth;
	            double dice = (2.0 * I + smooth) / denom;
        final double If = I;
        final double denomf = denom;
	            return wrap(new RereDoubleTensor(new double[]{1.0 - dice}, new int[]{1}));
	        }
        @Override public IDiffTensor nllLoss(IDiffTensor target, int classDim) {
            int d = (classDim < 0 ? classDim + value.rank() : classDim);
            IDoubleTensor gathered = value.gather(d, target);
            double total = 0;
            double[] gd = gathered.toDoubleArray();
            for (double v : gd) total += v;
            return wrap(new RereDoubleTensor(new double[]{-total / gd.length}, new int[]{1}));
        }
        @Override public IDiffTensor maxPool2d(int kH, int kW, int stride, int padding) {
            throw new UnsupportedOperationException("maxPool2d is not available on constant tensors");
        }
        @Override public IDiffTensor avgPool2d(int kH, int kW, int stride, int padding) {
            throw new UnsupportedOperationException("avgPool2d is not available on constant tensors");
        }
        @Override public IDiffTensor adaptiveAvgPool2d(int outH, int outW) {
            int[] s = shape();
            int rank = value.rank();
            int N, C, H, W;
            if (rank == 3) { N = 1; C = s[0]; H = s[1]; W = s[2]; }
            else if (rank == 4) { N = s[0]; C = s[1]; H = s[2]; W = s[3]; }
            else { N = 1; for (int i = 0; i < rank - 3; i++) N *= s[i]; C = s[rank - 3]; H = s[rank - 2]; W = s[rank - 1]; }
            double[] xd = value.toDoubleArray();
            double[] y = new double[N * C * outH * outW];
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    for (int oh = 0; oh < outH; oh++) {
                        int hStart = (int) Math.floor((double) oh * H / outH);
                        int hEnd = (int) Math.floor((double) (oh + 1) * H / outH);
                        for (int ow = 0; ow < outW; ow++) {
                            int wStart = (int) Math.floor((double) ow * W / outW);
                            int wEnd = (int) Math.floor((double) (ow + 1) * W / outW);
                            double sum = 0;
                            for (int h = hStart; h < hEnd; h++)
                                for (int w = wStart; w < wEnd; w++)
                                    sum += xd[((n * C + c) * H + h) * W + w];
                            y[((n * C + c) * outH + oh) * outW + ow] = sum / ((hEnd - hStart) * (wEnd - wStart));
                        }
                    }
                }
            }
            int[] outShape = (rank == 4) ? new int[]{N, C, outH, outW} : new int[]{C, outH, outW};
            return wrap(new RereDoubleTensor(y, outShape));
        }
        @Override public IDiffTensor oneHot(int numClasses) {
            int[] s = shape();
            int[] outShape = new int[s.length + 1];
            System.arraycopy(s, 0, outShape, 0, s.length);
            outShape[s.length] = numClasses;
            long n = value.totalSize();
            double[] y = new double[(int) (n * numClasses)];
            double[] xd = value.toDoubleArray();
            for (int i = 0; i < n; i++) {
                int cls = (int) Math.round(xd[i]);
                if (cls >= 0 && cls < numClasses) y[i * numClasses + cls] = 1.0;
            }
            return wrap(new RereDoubleTensor(y, outShape));
        }
        @Override public IDiffTensor instanceNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
            return computeInstanceNorm(gamma, beta, eps);
        }
        @Override public IDiffTensor diagEmbed(int offset, int dim1, int dim2) {
            long n = value.totalSize();
            int M = (int) n + Math.abs(offset);
            double[] xd = value.toDoubleArray();
            double[] y = new double[M * M];
            for (int i = 0; i < n; i++) {
                int row = (offset >= 0) ? i : i - offset;
                int col = (offset >= 0) ? i + offset : i;
                if (row >= 0 && row < M && col >= 0 && col < M) y[row * M + col] = xd[i];
            }
            return wrap(new RereDoubleTensor(y, new int[]{M, M}));
        }
        @Override public IDiffTensor dropout2d(double p) {
            if (p <= 0) return this;
            int[] s = shape();
            int rank = value.rank();
            int N, C;
            if (rank == 3) { N = 1; C = s[0]; }
            else if (rank == 4) { N = s[0]; C = s[1]; }
            else { N = 1; for (int i = 0; i < rank - 2; i++) N *= s[i]; C = s[rank - 2]; }
            double[] xd = value.toDoubleArray();
            double[] y = new double[xd.length];
            double scale = 1.0 / (1.0 - p);
            int spatial = xd.length / (N * C);
            java.util.Random rng = new java.util.Random();
            for (int nc = 0; nc < N * C; nc++) {
                double m = (rng.nextDouble() >= p) ? scale : 0;
                for (int sp = 0; sp < spatial; sp++) y[nc * spatial + sp] = xd[nc * spatial + sp] * m;
            }
            return wrap(new RereDoubleTensor(y, s));
        }
        @Override public IDiffTensor depthwiseConv1d(IDiffTensor weight, int stride, int padding) {
            throw new UnsupportedOperationException("depthwiseConv1d is not available on constant tensors");
        }
        @Override public IDiffTensor interpolate(double scaleFactor, String mode) {
            throw new UnsupportedOperationException("interpolate is not available on constant tensors");
        }
        @Override public IDiffTensor logDet() {
            int[] s = shape();
            if (s.length != 2 || s[0] != s[1])
                throw new UnsupportedOperationException("logDet on constant tensor requires square 2D");
            double[] xd = value.toDoubleArray();
            int n = s[0];
            double logDet;
            try {
                IMatrix<Double> A = Linalg.fromArray(xd, n, n);
                var luDecomp = com.yishape.lab.math.linalg.decomposition.Decomps.createLU();
                luDecomp.decompose(A);
                logDet = Math.log(Math.abs(luDecomp.getDeterminant()));
            } catch (Exception e) {
                logDet = Double.NEGATIVE_INFINITY;
            }
            return wrap(new RereDoubleTensor(new double[]{logDet}, 1));
        }
        @Override public IDiffTensor[] slogDet() {
            int[] s = shape();
            if (s.length != 2 || s[0] != s[1])
                throw new UnsupportedOperationException("slogDet on constant tensor requires square 2D");
            double[] xd = value.toDoubleArray();
            int n = s[0];
            IMatrix<Double> A = Linalg.fromArray(xd, n, n);
            var luDecomp = com.yishape.lab.math.linalg.decomposition.Decomps.createLU();
            luDecomp.decompose(A);
            double det = luDecomp.getDeterminant();
            IDiffTensor signT = wrap(new RereDoubleTensor(new double[]{Math.signum(det)}, 1));
            IDiffTensor ldT = wrap(new RereDoubleTensor(new double[]{Math.log(Math.abs(det))}, 1));
            return new IDiffTensor[]{signT, ldT};
        }
        @Override public IDiffTensor nuclearNorm() {
            // Constant tensor path: SVD is expensive, just compute Frobenius-like bound
            double[] xd = value.toDoubleArray();
            long n = value.totalSize();
            if (value.rank() == 2) {
                IMatrix<Double> A = Linalg.fromArray(xd, value.dim(0), value.dim(1));
                var svdResult = A.svd();
                var sVec = svdResult.getSecond();
                double[] sv = sVec.toDoubleArray();
                double sum = 0;
                for (double v : sv) sum += v;
                return wrap(new RereDoubleTensor(new double[]{sum}, 1));
            }
            throw new UnsupportedOperationException("nuclearNorm on constant tensor requires 2D matrix");
        }
        @Override public IDiffTensor ctcLoss(IDiffTensor targets, IDiffTensor inputLengths, IDiffTensor targetLengths) {
            throw new UnsupportedOperationException("ctcLoss is not available on constant tensors");
        }
        @Override public IDiffTensor cross(IDiffTensor other) {
            // SISD: constant-tensor cross product
            int[] sA = shape();
            int[] sB = other.shape();
            if (sA[sA.length - 1] != 3 || sB[sB.length - 1] != 3) {
                throw new IllegalArgumentException("cross requires last dim = 3");
            }
            int[] bcShape = TensorShape.broadcastShape(sA, sB);
            long outSize = 1;
            for (int d : bcShape) outSize *= d;
            double[] aData = value.toDoubleArray();
            double[] bData = other.toDoubleArray();
            double[] y = new double[(int) outSize];
            long numTriplets = outSize / 3;
            for (long t = 0; t < numTriplets; t++) {
                long flatIdx = t * 3;
                int[] bcIdx = unlinearizeInt((int) flatIdx, bcShape);
                int ai = flatIndexFromBroadcast(bcIdx, sA, bcShape);
                int bi = flatIndexFromBroadcast(bcIdx, sB, bcShape);
                int aBase = (ai / 3) * 3, bBase = (bi / 3) * 3;
                double a0 = aData[aBase], a1 = aData[aBase + 1], a2 = aData[aBase + 2];
                double b0 = bData[bBase], b1 = bData[bBase + 1], b2 = bData[bBase + 2];
                y[(int) flatIdx]     = a1 * b2 - a2 * b1;
                y[(int) flatIdx + 1] = a2 * b0 - a0 * b2;
                y[(int) flatIdx + 2] = a0 * b1 - a1 * b0;
            }
            return wrap(new RereDoubleTensor(y, bcShape));
        }
        @Override public IDiffTensor gridSample(IDiffTensor grid, String mode, String paddingMode) {
            // SISD fallback for constant grid_sample
            int[] s = shape();
            int N = s[0], C = s[1], H = s[2], W = s[3];
            int[] gs = grid.shape();
            int outH = gs[1], outW = gs[2];
            double[] xd = value.toDoubleArray();
            double[] gd = grid.toDoubleArray();
            double[] y = new double[N * C * outH * outW];
            boolean bilinear = "bilinear".equals(mode);
            for (int n = 0; n < N; n++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double gx = gd[((n * outH + oh) * outW + ow) * 2];
                        double gy = gd[((n * outH + oh) * outW + ow) * 2 + 1];
                        double px = (gx + 1.0) * 0.5 * (W - 1);
                        double py = (gy + 1.0) * 0.5 * (H - 1);
                        if (bilinear) {
                            int ix0 = (int) Math.floor(px), iy0 = (int) Math.floor(py);
                            int ix1 = ix0 + 1, iy1 = iy0 + 1;
                            double wx1 = px - ix0, wy1 = py - iy0;
                            double wx0 = 1.0 - wx1, wy0 = 1.0 - wy1;
                            for (int c = 0; c < C; c++) {
                                double val = 0;
                                int inBase = ((n * C + c) * H);
                                val += sampleBorder(xd, inBase, H, W, iy0, ix0, paddingMode) * wx0 * wy0;
                                val += sampleBorder(xd, inBase, H, W, iy0, ix1, paddingMode) * wx1 * wy0;
                                val += sampleBorder(xd, inBase, H, W, iy1, ix0, paddingMode) * wx0 * wy1;
                                val += sampleBorder(xd, inBase, H, W, iy1, ix1, paddingMode) * wx1 * wy1;
                                y[((n * C + c) * outH + oh) * outW + ow] = val;
                            }
                        } else {
                            int ix = (int) Math.round(px), iy = (int) Math.round(py);
                            for (int c = 0; c < C; c++) {
                                int inBase = ((n * C + c) * H);
                                y[((n * C + c) * outH + oh) * outW + ow] =
                                    sampleBorder(xd, inBase, H, W, iy, ix, paddingMode);
                            }
                        }
                    }
                }
            }
            return wrap(new RereDoubleTensor(y, new int[]{N, C, outH, outW}));
        }
        private static double sampleBorder(double[] data, int inBase, int H, int W, int y, int x, String pad) {
            int cy = switch (pad) {
                case "border" -> Math.clamp(y, 0, H - 1);
                case "reflection" -> { int r = Math.abs(y) % (2 * H); yield r >= H ? 2 * H - 1 - r : r; }
                default -> y;
            };
            int cx = switch (pad) {
                case "border" -> Math.clamp(x, 0, W - 1);
                case "reflection" -> { int r = Math.abs(x) % (2 * W); yield r >= W ? 2 * W - 1 - r : r; }
                default -> x;
            };
            if (cy < 0 || cy >= H || cx < 0 || cx >= W) return 0;
            return data[inBase + cy * W + cx];
        }
        @Override public IDiffTensor trapezoidalScan(IDiffTensor delta, IDiffTensor A, IDiffTensor B,
                                                      IDiffTensor C, IDiffTensor D) {
            throw new UnsupportedOperationException("trapezoidalScan is not available on constant tensors");
        }
        @Override public IDiffTensor embedding(IDiffTensor indices) {
            int[] idxShape = indices.shape();
            int embeddingDim = value.dim(value.rank() - 1);
            IDoubleTensor flatIdx = indices.reshape((int) indices.totalSize());
            IDoubleTensor gathered = value.indexSelect(0, flatIdx);
            int[] outShape = new int[idxShape.length + 1];
            System.arraycopy(idxShape, 0, outShape, 0, idxShape.length);
            outShape[idxShape.length] = embeddingDim;
            return wrap(gathered.reshape(outShape));
        }
        @Override public IDiffTensor rope(int dim, int maxLen, double base) {
            double[] xd = value.toDoubleArray();
            int headDim = value.dim(value.rank() - 1);
            long totalSize = value.totalSize();
            int seqLen = (int) (totalSize / headDim);
            int numPairs = headDim / 2;
            double[] y = new double[xd.length];
            for (int pos = 0; pos < seqLen; pos++) {
                int baseOff = pos * headDim;
                for (int i = 0; i < numPairs; i++) {
                    double theta = pos / Math.pow(base, 2.0 * i / dim);
                    double c = Math.cos(theta), s = Math.sin(theta);
                    int idx2i = baseOff + 2 * i, idx2i1 = idx2i + 1;
                    double x1 = xd[idx2i], x2 = xd[idx2i1];
                    y[idx2i] = x1 * c - x2 * s;
                    y[idx2i1] = x1 * s + x2 * c;
                }
            }
            return wrap(new RereDoubleTensor(y, shape()));
        }
        @Override public IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
                IDiffTensor mask, double dropout) {
            throw new UnsupportedOperationException("scaledDotProductAttention is not available on constant tensors");
        }
        @Override public IDiffTensor layerNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return computeNorm(gamma, beta, eps, true); }
        @Override public IDiffTensor batchNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return computeNorm(gamma, beta, eps, false); }
        @Override public IDiffTensor rmsNorm(IDiffTensor gamma, double eps) {
            int features = value.dim(value.rank() - 1);
            long totalSize = value.totalSize();
            int batch = (int) (totalSize / features);
            double[] xd = value.toDoubleArray();
            double[] gd = gamma.toDoubleArray();
            double[] y = new double[(int) totalSize];
            for (int p = 0; p < batch; p++) {
                int off = p * features;
                double sumSq = 0;
                for (int j = 0; j < features; j++) sumSq += xd[off + j] * xd[off + j];
                double invRms = 1.0 / Math.sqrt(sumSq / features + eps);
                for (int j = 0; j < features; j++) {
                    y[off + j] = xd[off + j] * invRms * gd[j];
                }
            }
            return wrap(new RereDoubleTensor(y, shape()));
        }
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
        private IDiffTensor computeGroupNorm(int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps) {
            int[] s = shape();
            int rank = value.rank();
            int C = s[rank - 2];
            int groupCh = C / numGroups;
            int outer = 1;
            for (int i = 0; i < rank - 2; i++) outer *= s[i];
            int spatial = (int) value.totalSize() / (outer * C);
            double[] xd = value.toDoubleArray();
            double[] gd = gamma.toDoubleArray();
            double[] bd = (beta != null) ? beta.toDoubleArray() : null;
            double[] y = new double[xd.length];
            for (int n = 0; n < outer; n++) {
                for (int g = 0; g < numGroups; g++) {
                    int count = 0;
                    double mean = 0;
                    for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                        for (int sp = 0; sp < spatial; sp++) {
                            mean += xd[n * C * spatial + c * spatial + sp];
                            count++;
                        }
                    }
                    mean /= count;
                    double var = 0;
                    for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                        for (int sp = 0; sp < spatial; sp++) {
                            double d = xd[n * C * spatial + c * spatial + sp] - mean;
                            var += d * d;
                        }
                    }
                    var /= count;
                    double inv = 1.0 / Math.sqrt(var + eps);
                    for (int c = g * groupCh; c < (g + 1) * groupCh; c++) {
                        for (int sp = 0; sp < spatial; sp++) {
                            int idx = n * C * spatial + c * spatial + sp;
                            y[idx] = (xd[idx] - mean) * inv * gd[c] + (bd != null ? bd[c] : 0);
                        }
                    }
                }
            }
            return wrap(new RereDoubleTensor(y, s));
        }
        private IDiffTensor computeInstanceNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
            int[] s = shape();
            int rank = value.rank();
            int N = 1;
            for (int i = 0; i < rank - 2; i++) N *= s[i];
            int C = s[rank - 2];
            int spatial = 1;
            for (int i = rank - 1; i < rank; i++) spatial *= s[i];
            double[] xd = value.toDoubleArray();
            double[] gd = gamma.toDoubleArray();
            double[] bd = (beta != null) ? beta.toDoubleArray() : null;
            double[] y = new double[xd.length];
            for (int n = 0; n < N; n++) {
                for (int c = 0; c < C; c++) {
                    double mean = 0;
                    for (int sp = 0; sp < spatial; sp++) mean += xd[(n * C + c) * spatial + sp];
                    mean /= spatial;
                    double var = 0;
                    for (int sp = 0; sp < spatial; sp++) { double d = xd[(n * C + c) * spatial + sp] - mean; var += d * d; }
                    var /= spatial;
                    double inv = 1.0 / Math.sqrt(var + eps);
                    for (int sp = 0; sp < spatial; sp++) {
                        int idx = (n * C + c) * spatial + sp;
                        y[idx] = (xd[idx] - mean) * inv * gd[c] + (bd != null ? bd[c] : 0);
                    }
                }
            }
            return wrap(new RereDoubleTensor(y, s));
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
        @Override public IDiffTensor triu(int diagonal) {
            // Manual triu: clone and zero lower triangle
            double[] d = value.toDoubleArray().clone();
            int r = rank();
            int M = value.dim(r - 2);
            int N = value.dim(r - 1);
            int batchStride = M * N;
            int batchCount = d.length / batchStride;
            for (int b = 0; b < batchCount; b++) {
                int base = b * batchStride;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        if (j < i + diagonal) d[base + i * N + j] = 0.0;
                    }
                }
            }
            return wrap(new RereDoubleTensor(d, shape()));
        }
        @Override public IDiffTensor diag() {
            int r = rank();
            int[] s = shape();
            int M = (r >= 2) ? value.dim(r - 2) : 1;
            int N = (r >= 2) ? value.dim(r - 1) : 1;
            if (r < 2) return this;
            if (M != N) throw new IllegalArgumentException("diag() requires square matrix, got " + M + "x" + N);
            int batchDim = 1;
            for (int i = 0; i < r - 2; i++) batchDim *= s[i];
            double[] vals = value.toDoubleArray();
            double[] resultData = new double[batchDim * M];
            for (int b = 0; b < batchDim; b++) {
                int base = b * M * N;
                for (int i = 0; i < M; i++) resultData[b * M + i] = vals[base + i * N + i];
            }
            int[] resultShape = (r == 2) ? new int[]{M} : java.util.Arrays.copyOf(s, r - 1);
            if (r > 2) resultShape[r - 2] = M;
            return wrap(new RereDoubleTensor(resultData, resultShape));
        }
        @Override public IDiffTensor diagonal(int offset, int dim1, int dim2) {
            // For constant tensors, just compute via diag path
            int r = rank();
            int[] s = shape();
            if (dim1 < 0) dim1 += r;
            if (dim2 < 0) dim2 += r;
            int size = (int) Math.min(s[dim1], s[dim2]);
            int effSize = offset >= 0
                ? Math.max(0, Math.min(s[dim1] - offset, s[dim2]))
                : Math.max(0, Math.min(s[dim1], s[dim2] + offset));
            double[] vals = value.toDoubleArray();
            int outer = 1;
            for (int i = 0; i < Math.min(dim1, dim2); i++) outer *= s[i];
            int inner = 1;
            for (int i = Math.max(dim1, dim2) + 1; i < r; i++) inner *= s[i];
            double[] resultData = new double[outer * effSize * inner];
            for (int o = 0; o < outer; o++) {
                for (int k = 0; k < effSize; k++) {
                    for (int i = 0; i < inner; i++) {
                        int[] idx = new int[r];
                        int remaining = o;
                        for (int j = Math.min(dim1, dim2) - 1; j >= 0; j--) {
                            idx[j] = remaining % s[j]; remaining /= s[j];
                        }
                        idx[dim1] = offset >= 0 ? k : k - offset;
                        idx[dim2] = offset >= 0 ? k + offset : k;
                        remaining = i;
                        for (int j = r - 1; j > Math.max(dim1, dim2); j--) {
                            idx[j] = remaining % s[j]; remaining /= s[j];
                        }
                        resultData[(o * effSize + k) * inner + i] = vals[flatIndex(idx, s)];
                    }
                }
            }
            int[] resultShape;
            if (r == 2) { resultShape = new int[]{effSize}; }
            else {
                resultShape = new int[r - 1];
                int pos = 0;
                for (int j = 0; j < dim1; j++) resultShape[pos++] = s[j];
                for (int j = dim1 + 1; j < dim2; j++) resultShape[pos++] = s[j];
                for (int j = dim2 + 1; j < r; j++) resultShape[pos++] = s[j];
                resultShape[Math.min(dim1, dim2)] = effSize;
            }
            return wrap(new RereDoubleTensor(resultData, resultShape));
        }
        @Override public IDiffTensor trace() { return diag().sum(); }
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

        @Override public IDiffTensor[] split(int splitSize, int dim) {
            int d = (dim < 0 ? dim + rank() : dim);
            int dimSize = dim(d);
            java.util.ArrayList<Integer> sizes = new java.util.ArrayList<>();
            int remaining = dimSize;
            while (remaining > 0) { sizes.add(Math.min(splitSize, remaining)); remaining -= splitSize; }
            int[] sizeArray = new int[sizes.size()];
            for (int i = 0; i < sizes.size(); i++) sizeArray[i] = sizes.get(i);
            return split(sizeArray, d);
        }
        @Override public IDiffTensor[] split(int[] splitSizes, int dim) {
            int d = (dim < 0 ? dim + rank() : dim);
            int n = splitSizes.length;
            IDiffTensor[] result = new IDiffTensor[n];
            int offset = 0;
            for (int i = 0; i < n; i++) {
                result[i] = narrow(d, offset, splitSizes[i]);
                offset += splitSizes[i];
            }
            return result;
        }
        @Override public IDiffTensor[] chunk(int chunks, int dim) {
            int d = (dim < 0 ? dim + rank() : dim);
            int dimSize = dim(d);
            int chunkSize = (dimSize + chunks - 1) / chunks;
            java.util.ArrayList<Integer> sizes = new java.util.ArrayList<>();
            int remaining = dimSize;
            for (int i = 0; i < chunks && remaining > 0; i++) {
                sizes.add(Math.min(chunkSize, remaining));
                remaining -= chunkSize;
            }
            int[] sizeArray = new int[sizes.size()];
            for (int i = 0; i < sizes.size(); i++) sizeArray[i] = sizes.get(i);
            return split(sizeArray, d);
        }
        @Override public IDiffTensor[] unbind(int dim) {
            int d = (dim < 0 ? dim + rank() : dim);
            int dimSize = dim(d);
            IDiffTensor[] result = new IDiffTensor[dimSize];
            for (int i = 0; i < dimSize; i++) {
                result[i] = narrow(d, i, 1).squeeze(d);
            }
            return result;
        }

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
