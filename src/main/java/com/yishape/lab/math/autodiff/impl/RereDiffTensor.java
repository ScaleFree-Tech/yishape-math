package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.graph.GraphOpSchema;
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
import com.yishape.lab.math.autodiff.impl.DiffTensorUtil;
import com.yishape.lab.math.autodiff.impl.DiffTensorUtil.BinaryBackward;
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

    RereDoubleTensor value;
    double[] grad;
    List<RereDiffTensor> inputs;
    Consumer<RereDiffTensor> backwardFn;
    String opTag;
    boolean requiresGrad = true;
    boolean isLeaf;
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
    double scalarParam = Double.NaN;
    /** Secondary scalar parameter (e.g. beta in activation variants). */
    double scalarParam2 = Double.NaN;

    /**
     * Override shape in JSON export. When non-null, used instead of the tensor's own shape
     * for GPU/HPC graph export (e.g. fused pattern nodes where the logical shape differs).
     */
    int[] exportShape;

    /**
     * Auxiliary backward data exported to GPU/HPC backends (e.g. MaxPool2d argmax indices).
     * When non-null, included in the binary/JSON graph serialization.
     */
    int[] backwardIndices;

    /**
     * Symbolic backward function for higher-order AD.
     * Takes the output gradient (IDiffTensor) and returns gradients for each input.
     * When non-null, enables {@code AD.grad(output, inputs)} to build a new
     * computation graph whose nodes are themselves differentiable.
     */
    java.util.function.Function<IDiffTensor, IDiffTensor[]> symbolicBackwardFn;

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
            // Release graph references so GC can reclaim intermediate nodes immediately.
            // After backward, only gradient values matter — graph edges are stale.
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffTensor v = order.get(i);
                if (v != this && !v.isLeaf) {
                    v.inputs = null;
                    v.backwardFn = null;
                    v.symbolicBackwardFn = null;
                }
            }
        } finally {
            IN_BACKWARD_IMPL.set(false);
            TOPO_LIST.get().clear();
            TOPO_SET.get().clear();
            AutodiffBufferPool.cleanupThread();
        }
    }

    /** Reentrant-safe backward using local collections — avoids corrupting the outer
     *  backwardImpl()'s ThreadLocal TOPO_LIST/TOPO_SET during nested backward(). */
    private void backwardImplLocal() {
        try {
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
            // Release graph references to prevent cross-iteration memory accumulation.
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffTensor v = order.get(i);
                if (v != this && !v.isLeaf) {
                    v.inputs = null;
                    v.backwardFn = null;
                    v.symbolicBackwardFn = null;
                }
            }
        } finally {
            AutodiffBufferPool.cleanupThread();
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
        // Release graph references to prevent cross-iteration memory accumulation.
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffTensor v = order.get(i);
            if (v != this && !v.isLeaf) {
                v.inputs = null;
                v.backwardFn = null;
                v.symbolicBackwardFn = null;
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
                // Guard: inputs may be null if the graph was released after backward()
                // (see backwardImpl() lines 285-292). Nodes with null inputs are terminal
                // — they have no predecessors to traverse.
                if (inputs != null) {
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
                }
            } else {
                sp--;
                order.add(node);
            }
        }
    }

    @Override
    public void zeroGradient() {
        this.grad = null;
        // Also release graph references so GC can reclaim intermediate nodes.
        if (!this.isLeaf) {
            this.inputs = null;
            this.backwardFn = null;
            this.symbolicBackwardFn = null;
        }
    }

    /**
     * Detach this tensor and all reachable non-leaf nodes from the computation graph.
     *
     * <p>After calling this method, {@link #backward()} will have no effect because all
     * graph edges ({@code inputs}, {@code backwardFn}, {@code symbolicBackwardFn}) have
     * been released. Use this to explicitly free graph memory before GC runs.</p>
     */
    public void detachGraph() {
        java.util.ArrayDeque<RereDiffTensor> queue = new java.util.ArrayDeque<>();
        java.util.HashSet<RereDiffTensor> seen = new java.util.HashSet<>();
        queue.add(this);
        while (!queue.isEmpty()) {
            RereDiffTensor node = queue.poll();
            if (!seen.add(node)) continue;
            if (node.inputs != null && !node.isLeaf) {
                for (RereDiffTensor inp : node.inputs) {
                    if (inp != null && !seen.contains(inp)) queue.add(inp);
                }
                node.inputs = null;
            }
            node.backwardFn = null;
            node.symbolicBackwardFn = null;
        }
    }

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
            // D3: avoid copyOf when buffer is already exact size; only slice when slab-aligned pool oversizes
            double[] sliced = (pooledBuf.length == n) ? pooledBuf : Arrays.copyOf(pooledBuf, n);
            AutodiffBufferPool.release(pooledBuf);
            grad = COMPUTER.binaryOperate(g, sliced, BinaryOperation.ADD);
        } else {
            double[] dx = new double[n];
            System.arraycopy(pooledBuf, 0, dx, 0, n);
            AutodiffBufferPool.release(pooledBuf);
            accGrad(dx);
        }
    }

    // ==================== toNonDiff ====================

    IDiffTensor toNonDiff(IDoubleTensor t) {
        if (t instanceof RereDiffTensor rdt) return rdt;
        if (t instanceof ConstantDiffTensor cdt) return cdt;
        if (t instanceof RereDoubleTensor rdt) return new ConstantDiffTensor(rdt);
        return new ConstantDiffTensor(new RereDoubleTensor(t.toDoubleArray(), t.shape()));
    }

    // ==================== sum() — override default to avoid flattenValue cycle ====================

    @Override
    public IDiffTensor sum() {
        if (!requiresGrad) {
            // Scalar sum for non-differentiable tensors — delegate to value's sumAll
            double total = value.sumAll();
            return toNonDiff(new RereDoubleTensor(new double[]{total}, new int[]{1}));
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
        if (x == null) return null;
        if (!x.requiresGrad) return null;
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("square", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("relu", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("exp", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("sigmoid", "sum"));
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
                // sign(x): +1 for x>0, -1 for x<0, 0 for x=0 (subgradient at 0).
                // Use UniversalOperation.SIGN routed through GPU→SIMD→SISD.
                double[] s = COMPUTER.universalOperate(xData, UniversalOperation.SIGN, 0);
                double[] grad = COMPUTER.binaryOperate(s, g, BinaryOperation.MULTIPLY);
                x.accGradFromPooled(grad, m);
            };
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("abs", "sum"));
            r.exportShape = x.shape();
            // d²abs/dx² = 0 (Dirac delta at x=0, zero everywhere else).
            // Zeroed via mul(0); factor values are irrelevant except x=0 convention.
            double[] absFactor = COMPUTER.universalOperate(xData, UniversalOperation.SIGN, 0);
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("tanh", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("silu", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("log", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("pow", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("gelu", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("mish", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("sin", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("cos", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("leakyRelu", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("elu", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("selu", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("softplus", "sum"));
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
            RereDiffTensor r = new RereDiffTensor(new double[]{total}, new int[]{1}, List.of(x), bw, GraphOpSchema.FusedTag.of("hardtanh", "sum"));
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
    IDiffTensor tryFuseSumDim(int dim, boolean keepdim) {
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
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("square", "sum"),
                (g, xv) -> g * 2.0 * xv, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- relu().sum(dim) --
        if ("relu".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("relu", "sum"),
                (g, xv) -> xv > 0 ? g : 0, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- exp().sum(dim) --
        if ("exp".equals(opTag)) {
            double[] eData = value.toDoubleArray();
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("exp", "sum"),
                (g, xv) -> g * Math.exp(xv), xData, fOuter, fReduce, fInner, total);
            // Override: use the exp(x) values for backward factor
            double[] expFactor = new double[total];
            for (int i = 0; i < total; i++) expFactor[i] = eData[i];
            r.symbolicBackwardFn = DiffTensorUtil.dimSumGradFn(x.shape(), dim, expFactor, x);
            return r;
        }
        // -- sigmoid().sum(dim) --
        if ("sigmoid".equals(opTag)) {
            double[] sigData = value.toDoubleArray();
            RereDiffTensor rt = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("sigmoid", "sum"),
                (g, xv) -> { double sv = 1.0/(1.0+Math.exp(-xv)); return g * sv * (1-sv); },
                xData, fOuter, fReduce, fInner, total);
            double[] sigFactor = new double[total];
            for (int i = 0; i < total; i++) { double sv = sigData[i]; sigFactor[i] = sv * (1-sv); }
            rt.symbolicBackwardFn = DiffTensorUtil.dimSumGradFn(x.shape(), dim, sigFactor, x);
            return rt;
        }
        // -- abs().sum(dim) --
        if ("abs".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("abs", "sum"),
                (g, xv) -> xv >= 0 ? g : -g, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- tanh().sum(dim) --
        if ("tanh".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("tanh", "sum"),
                (g, xv) -> { double t = Math.tanh(xv); return g * (1.0 - t*t); },
                xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- silu().sum(dim) --
        if ("silu".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("silu", "sum"),
                (g, xv) -> { double sig = 1.0/(1.0+Math.exp(-xv)); return g * (sig + xv * sig * (1-sig)); },
                xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- log().sum(dim) --
        if ("log".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("log", "sum"),
                (g, xv) -> g / xv, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- pow(n).sum(dim) --
        double sp = scalarParam;
        if ("pow".equals(opTag) && !Double.isNaN(sp)) {
            double param = sp;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("pow", "sum"),
                (g, xv) -> g * param * Math.pow(xv, param - 1), xData, fOuter, fReduce, fInner, total);
            r.scalarParam = param;
            return r;
        }
        // -- gelu().sum(dim) --
        if ("gelu".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("gelu", "sum"),
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
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("mish", "sum"),
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
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("sin", "sum"),
                (g, xv) -> g * Math.cos(xv), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- cos().sum(dim) --
        if ("cos".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("cos", "sum"),
                (g, xv) -> g * (-Math.sin(xv)), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- leakyRelu().sum(dim) --
        if ("leakyRelu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 0.01 : scalarParam;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("leakyRelu", "sum"),
                (g, xv) -> xv > 0 ? g : g * alpha, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- elu().sum(dim) --
        if ("elu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 1.0 : scalarParam;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("elu", "sum"),
                (g, xv) -> xv > 0 ? g : g * alpha * Math.exp(xv), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- selu().sum(dim) --
        if ("selu".equals(opTag)) {
            final double seluL = 1.0507009873554804934193349852946;
            final double seluA = 1.6732632423543772848170429916717;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("selu", "sum"),
                (g, xv) -> xv > 0 ? g * seluL : g * seluL * seluA * Math.exp(xv), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- softplus().sum(dim) --
        if ("softplus".equals(opTag)) {
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("softplus", "sum"),
                (g, xv) -> g / (1.0 + Math.exp(-xv)), xData, fOuter, fReduce, fInner, total);
            return r;
        }
        // -- hardtanh().sum(dim) --
        if ("hardtanh".equals(opTag)) {
            double hmin = Double.isNaN(scalarParam) ? -1.0 : scalarParam;
            double hmax = Double.isNaN(scalarParam2) ? 1.0 : scalarParam2;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("hardtanh", "sum"),
                (g, xv) -> (xv > hmin && xv < hmax) ? g : 0, xData, fOuter, fReduce, fInner, total);
            return r;
        }
        return null;
    }

    /** Build a fused unaryOp+sum(dim) graph node. Shared across all 9 patterns. */
    RereDiffTensor buildFusedSumDim(RereDiffTensor x, double[] result, int[] resultShape,
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
        // exportShape = resultShape (the actual reduced shape), NOT x.shape().
        // Using x.shape() causes backends to allocate wrong buffer sizes for non-contiguous reductions.
        node.exportShape = resultShape;
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
        node.symbolicBackwardFn = DiffTensorUtil.dimSumGradFn(x.shape(), dim, factor, x);
        return node;
    }

    /** Try to fuse unaryOp + mean(dim) into a single fused node. Returns null if no pattern matches.
     *  Mirrors {@link #tryFuseSumDim(int, boolean)} but produces GraphOpSchema.FusedTag.of("relu", "mean") etc. tags
     *  and scales the backward gradient by 1/reduce. */
    IDiffTensor tryFuseMeanDim(int dim, boolean keepdim) {
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
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("square", "mean"),
                (g, xv) -> g * invR * 2.0 * xv, xData, outer, reduce, inner, total);
        }
        // -- relu().mean(dim) --
        if ("relu".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("relu", "mean"),
                (g, xv) -> xv > 0 ? g * invR : 0, xData, outer, reduce, inner, total);
        }
        // -- exp().mean(dim) --
        if ("exp".equals(opTag)) {
            double[] eData = value.toDoubleArray();
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("exp", "mean"),
                (g, xv) -> g * invR * Math.exp(xv), xData, outer, reduce, inner, total);
            double[] expFactor = new double[total];
            for (int i = 0; i < total; i++) expFactor[i] = eData[i] * invR;
            r.symbolicBackwardFn = DiffTensorUtil.dimSumGradFn(x.shape(), dim, expFactor, x);
            return r;
        }
        // -- sigmoid().mean(dim) --
        if ("sigmoid".equals(opTag)) {
            double[] sigData = value.toDoubleArray();
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("sigmoid", "mean"),
                (g, xv) -> { double sv = 1.0/(1.0+Math.exp(-xv)); return g * invR * sv * (1-sv); },
                xData, outer, reduce, inner, total);
            double[] sigFactor = new double[total];
            for (int i = 0; i < total; i++) { double sv = sigData[i]; sigFactor[i] = sv * (1-sv) * invR; }
            r.symbolicBackwardFn = DiffTensorUtil.dimSumGradFn(x.shape(), dim, sigFactor, x);
            return r;
        }
        // -- abs().mean(dim) --
        if ("abs".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("abs", "mean"),
                (g, xv) -> xv >= 0 ? g * invR : -g * invR, xData, outer, reduce, inner, total);
        }
        // -- tanh().mean(dim) --
        if ("tanh".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("tanh", "mean"),
                (g, xv) -> { double t = Math.tanh(xv); return g * invR * (1.0 - t*t); },
                xData, outer, reduce, inner, total);
        }
        // -- silu().mean(dim) --
        if ("silu".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("silu", "mean"),
                (g, xv) -> { double sig = 1.0/(1.0+Math.exp(-xv)); return g * invR * (sig + xv * sig * (1-sig)); },
                xData, outer, reduce, inner, total);
        }
        // -- log().mean(dim) --
        if ("log".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("log", "mean"),
                (g, xv) -> g * invR / Math.max(Math.abs(xv), 1e-15), xData, outer, reduce, inner, total);
        }
        // -- pow(n).mean(dim) --
        double spp = scalarParam;
        if ("pow".equals(opTag) && !Double.isNaN(spp)) {
            double param = spp;
            RereDiffTensor r = buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("pow", "mean"),
                (g, xv) -> g * invR * param * Math.pow(xv, param - 1), xData, outer, reduce, inner, total);
            r.scalarParam = param;
            return r;
        }
        // -- gelu().mean(dim) --
        if ("gelu".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("gelu", "mean"),
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
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("mish", "mean"),
                (g, xv) -> {
                    double sp_m = Math.log1p(Math.exp(xv));
                    double t = Math.tanh(sp_m);
                    double sig = 1.0 / (1.0 + Math.exp(-xv));
                    return g * invR * (t + xv * (1.0 - t * t) * sig);
                }, xData, outer, reduce, inner, total);
        }
        // -- sin().mean(dim) --
        if ("sin".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("sin", "mean"),
                (g, xv) -> g * invR * Math.cos(xv), xData, outer, reduce, inner, total);
        }
        // -- cos().mean(dim) --
        if ("cos".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("cos", "mean"),
                (g, xv) -> g * invR * (-Math.sin(xv)), xData, outer, reduce, inner, total);
        }
        // -- leakyRelu().mean(dim) --
        if ("leakyRelu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 0.01 : scalarParam;
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("leakyRelu", "mean"),
                (g, xv) -> xv > 0 ? g * invR : g * invR * alpha, xData, outer, reduce, inner, total);
        }
        // -- elu().mean(dim) --
        if ("elu".equals(opTag)) {
            double alpha = Double.isNaN(scalarParam) ? 1.0 : scalarParam;
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("elu", "mean"),
                (g, xv) -> xv > 0 ? g * invR : g * invR * alpha * Math.exp(xv), xData, outer, reduce, inner, total);
        }
        // -- selu().mean(dim) --
        if ("selu".equals(opTag)) {
            final double seluL = 1.0507009873554804934193349852946;
            final double seluA = 1.6732632423543772848170429916717;
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("selu", "mean"),
                (g, xv) -> xv > 0 ? g * invR * seluL : g * invR * seluL * seluA * Math.exp(xv),
                xData, outer, reduce, inner, total);
        }
        // -- softplus().mean(dim) --
        if ("softplus".equals(opTag)) {
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("softplus", "mean"),
                (g, xv) -> g * invR / (1.0 + Math.exp(-xv)), xData, outer, reduce, inner, total);
        }
        // -- hardtanh().mean(dim) --
        if ("hardtanh".equals(opTag)) {
            double hmin = Double.isNaN(scalarParam) ? -1.0 : scalarParam;
            double hmax = Double.isNaN(scalarParam2) ? 1.0 : scalarParam2;
            return buildFusedSumDim(x, result, resultShape, dim, keepdim, GraphOpSchema.FusedTag.of("hardtanh", "mean"),
                (g, xv) -> (xv > hmin && xv < hmax) ? g * invR : 0, xData, outer, reduce, inner, total);
        }
        return null;
    }

    // ==================== Element-wise unary ops ====================

    @Override public IDiffTensor neg() { return DiffTensorUnary.neg(this); }
    @Override public IDiffTensor abs() { return DiffTensorUnary.abs(this); }
    @Override public IDiffTensor sqrt() { return DiffTensorUnary.sqrt(this); }
    @Override public IDiffTensor exp() { return DiffTensorUnary.exp(this); }
    @Override public IDiffTensor log() { return DiffTensorUnary.log(this); }
    @Override public IDiffTensor sin() { return DiffTensorUnary.sin(this); }
    @Override public IDiffTensor cos() { return DiffTensorUnary.cos(this); }
    @Override public IDiffTensor tan() { return DiffTensorUnary.tan(this); }
    @Override public IDiffTensor sigmoid() { return DiffTensorUnary.sigmoid(this); }
    @Override public IDiffTensor relu() { return DiffTensorUnary.relu(this); }
    @Override public IDiffTensor square() { return DiffTensorUnary.square(this); }
    @Override public IDiffTensor pow(double n) { return DiffTensorUnary.pow(this, n); }
    @Override public IDiffTensor clamp(double min, double max) { return DiffTensorUnary.clamp(this, min, max); }
    @Override public IDiffTensor tanh() { return DiffTensorUnary.tanh(this); }
    @Override public IDiffTensor silu() { return DiffTensorUnary.silu(this); }
    @Override public IDiffTensor gelu() { return DiffTensorUnary.gelu(this); }
    @Override public IDiffTensor softplus(double beta) { return DiffTensorUnary.softplus(this, beta); }
    @Override public IDiffTensor mish() { return DiffTensorUnary.mish(this); }
    @Override public IDiffTensor elu(double alpha) { return DiffTensorUnary.elu(this, alpha); }
    @Override public IDiffTensor leakyRelu(double alpha) { return DiffTensorUnary.leakyRelu(this, alpha); }
    @Override public IDiffTensor selu() { return DiffTensorUnary.selu(this); }
    @Override public IDiffTensor hardtanh(double minVal, double maxVal) { return DiffTensorUnary.hardtanh(this, minVal, maxVal); }
    @Override public IDiffTensor dropout(double p) { return DiffTensorUnary.dropout(this, p); }

    // ==================== Element-wise binary ops — scalar ====================

    @Override public IDiffTensor add(double scalar) { return DiffTensorBinary.add(this, scalar); }
    @Override public IDiffTensor sub(double scalar) { return DiffTensorBinary.sub(this, scalar); }
    @Override public IDiffTensor mul(double scalar) { return DiffTensorBinary.mul(this, scalar); }
    @Override public IDiffTensor div(double scalar) { return DiffTensorBinary.div(this, scalar); }
    @Override public IDiffTensor rsub(double scalar) { return DiffTensorBinary.rsub(this, scalar); }
    @Override public IDiffTensor rdiv(double scalar) { return DiffTensorBinary.rdiv(this, scalar); }
    @Override public IDiffTensor reciprocal() { return DiffTensorBinary.reciprocal(this); }

    // ==================== Element-wise binary ops — tensor ====================

    @Override public IDiffTensor add(IDoubleTensor other) { return DiffTensorBinary.add(this, other); }
    @Override public IDiffTensor sub(IDoubleTensor other) { return DiffTensorBinary.sub(this, other); }
    @Override public IDiffTensor mul(IDoubleTensor other) { return DiffTensorBinary.mul(this, other); }
    @Override public IDiffTensor div(IDoubleTensor other) { return DiffTensorBinary.div(this, other); }


    // ==================== View ops ====================

    @Override public IDiffTensor select(int dim, long index) { return DiffTensorView.select(this, dim, index); }
    @Override public IDiffTensor slice(int dim, long start, long end) { return DiffTensorView.slice(this, dim, start, end); }
    @Override public IDiffTensor narrow(int dim, long start, long length) { return slice(dim, start, start + length); }
    @Override public IDiffTensor permute(int... dims) { return DiffTensorView.permute(this, dims); }
    @Override public IDiffTensor transpose(int dim0, int dim1) { return DiffTensorView.transpose(this, dim0, dim1); }
    @Override public IDiffTensor transpose() { return DiffTensorView.transpose(this); }
    @Override public IDiffTensor squeeze(int... dims) { return DiffTensorView.squeeze(this, dims); }
    @Override public IDiffTensor unsqueeze(int dim) { return DiffTensorView.unsqueeze(this, dim); }
    @Override public IDiffTensor flatten(int startDim, int endDim) { return DiffTensorView.flatten(this, startDim, endDim); }
    @Override public IDiffTensor expand(int... targetShape) { return DiffTensorView.expand(this, targetShape); }
    @Override public IDiffTensor broadcastTo(int... targetShape) { return expand(targetShape); }
    @Override public IDiffTensor contiguous() { return DiffTensorView.contiguous(this); }
    @Override public IDiffTensor reshape(int... newShape) { return DiffTensorView.reshape(this, newShape); }
    @Override public IDiffTensor tile(int... repeats) { return DiffTensorView.tile(this, repeats); }

    // ==================== Reduction ops ====================

    @Override public IDiffTensor sum(int dim, boolean keepdim) { return DiffTensorReduce.sum(this, dim, keepdim); }
    @Override public IDiffTensor mean(int dim, boolean keepdim) { return DiffTensorReduce.mean(this, dim, keepdim); }
    @Override public IDiffTensor logSumExp(int dim, boolean keepdim) { return DiffTensorReduce.logSumExp(this, dim, keepdim); }
    @Override public IDiffTensor max(int dim, boolean keepdim) { return DiffTensorReduce.max(this, dim, keepdim); }
    @Override public IDiffTensor min(int dim, boolean keepdim) { return DiffTensorReduce.min(this, dim, keepdim); }
    @Override public IDiffTensor prod(int dim, boolean keepdim) { return DiffTensorReduce.prod(this, dim, keepdim); }
    @Override public IDiffTensor std(int dim, boolean keepdim) { return DiffTensorReduce.std(this, dim, keepdim); }
    @Override public IDiffTensor var(int dim, boolean keepdim) { return DiffTensorReduce.var(this, dim, keepdim); }

    // ==================== Full reductions ====================

    @Override public double sumAll() { return value.sumAll(); }
    @Override public double meanAll() { return value.meanAll(); }
    @Override public double maxAll() { return value.maxAll(); }
    @Override public double minAll() { return value.minAll(); }
    @Override public double prodAll() { return value.prodAll(); }

    // ==================== Softmax / logSoftmax ====================

    @Override public IDiffTensor softmax(int dim) { return DiffTensorSoftmax.softmax(this, dim); }

    @Override public IDiffTensor logSoftmax(int dim) { return DiffTensorSoftmax.logSoftmax(this, dim); }

    @Override public IDiffTensor softmaxCrossEntropy(IDoubleTensor labels, int dim) { return DiffTensorSoftmax.softmaxCrossEntropy(this, labels, dim); }

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
    @Override public IDiffTensor softmaxCrossEntropySparse(int[] labels, int dim) { return DiffTensorSoftmax.softmaxCrossEntropySparse(this, labels, dim); }

    // ==================== cumsum / cumprod ====================

    @Override public IDiffTensor cumsum(int dim) { return DiffTensorSoftmax.cumsum(this, dim); }

    @Override public IDiffTensor cumprod(int dim) { return DiffTensorSoftmax.cumprod(this, dim); }

    // ==================== argmax / argmin ====================

    @Override public IDiffTensor argmax(int dim) { return DiffTensorSoftmax.argmax(this, dim); }

    @Override public IDiffTensor argmin(int dim) { return DiffTensorSoftmax.argmin(this, dim); }

    // ==================== Matrix ops ====================

    @Override public IDiffTensor mmul(IDoubleTensor other) { return DiffTensorMatrix.mmul(this, other); }

    @Override public IDiffTensor bmm(IDoubleTensor other) { return DiffTensorMatrix.bmm(this, other); }

    @Override public IDiffTensor einsum(String subscript, IDoubleTensor... others) { return DiffTensorMatrix.einsum(this, subscript, others); }

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
            // C4: single fused sum over all dims instead of sequential per-dim sum nodes
            if (spec.outputLabels.isEmpty()) {
                result = result.sum();
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

    @Override public IDiffTensor gather(int dim, IDoubleTensor index) { return DiffTensorAdvanced.gather(this, dim, index); }

    @Override public IDiffTensor indexSelect(int dim, IDoubleTensor index) { return DiffTensorAdvanced.indexSelect(this, dim, index); }

    @Override public IDiffTensor argsort(int dim, boolean descending) { return DiffTensorAdvanced.argsort(this, dim, descending); }

    @Override public IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) { return DiffTensorAdvanced.scatter(this, dim, index, source); }

    @Override public IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) { return DiffTensorAdvanced.scatterAdd(this, dim, index, source); }

    @Override public IDiffTensor where(IDoubleTensor condition, IDoubleTensor other) { return DiffTensorAdvanced.where(this, condition, other); }

    @Override public IDiffTensor topk(int k, int dim, boolean largest) { return DiffTensorAdvanced.topk(this, k, dim, largest); }

    @Override public IDiffTensor pad(int[][] padding, String mode, double padValue) { return DiffTensorAdvanced.pad(this, padding, mode, padValue); }

    @Override public IDiffTensor tril(int diagonal) { return DiffTensorAdvanced.tril(this, diagonal); }

    @Override public IDiffTensor triu(int diagonal) { return DiffTensorAdvanced.triu(this, diagonal); }

    @Override public IDiffTensor diag() { return DiffTensorAdvanced.diag(this); }

    @Override public IDiffTensor diagonal(int offset, int dim1, int dim2) { return DiffTensorAdvanced.diagonal(this, offset, dim1, dim2); }

    @Override public IDiffTensor trace() { return DiffTensorAdvanced.trace(this); }

    @Override public IDiffTensor unfold(int dim, int size, int stride, int dilation) { return DiffTensorAdvanced.unfold(this, dim, size, stride, dilation); }

    @Override public IDiffTensor nonzero() { return DiffTensorAdvanced.nonzero(this); }

    @Override public IDiffTensor maskedSelect(IDoubleTensor mask) { return DiffTensorAdvanced.maskedSelect(this, mask); }

    @Override public IDiffTensor maskedFill(IDoubleTensor mask, double fillValue) { return DiffTensorAdvanced.maskedFill(this, mask, fillValue); }

    @Override public IDiffTensor cat(int dim, IDoubleTensor... others) { return DiffTensorAdvanced.cat(this, dim, others); }

    @Override public IDiffTensor stack(int dim, IDoubleTensor... others) { return DiffTensorAdvanced.stack(this, dim, others); }

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

    @Override public IDiffTensor normalize(double p, int dim) { return DiffTensorAdvanced.normalize(this, p, dim); }

    // ==================== Phase 2: flip, roll, repeatInterleave ====================

    @Override public IDiffTensor flip(int... dims) { return DiffTensorTransform.flip(this, dims); }

    @Override public IDiffTensor roll(int[] shifts, int[] dims) { return DiffTensorTransform.roll(this, shifts, dims); }

    @Override public IDiffTensor repeatInterleave(int repeats, int dim) { return DiffTensorTransform.repeatInterleave(this, repeats, dim); }

    // ==================== Phase 2: groupNorm ====================

    @Override public IDiffTensor groupNorm(int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps) { return DiffTensorTransform.groupNorm(this, numGroups, gamma, beta, eps); }

    // ==================== Phase 3: Loss Functions ====================

    @Override public IDiffTensor smoothL1Loss(IDiffTensor target, double beta) { return DiffTensorLoss.smoothL1Loss(this, target, beta); }
    @Override public IDiffTensor bceLoss(IDiffTensor target) { return DiffTensorLoss.bceLoss(this, target); }
    @Override public IDiffTensor bceWithLogitsLoss(IDiffTensor target) { return DiffTensorLoss.bceWithLogitsLoss(this, target); }

    @Override public IDiffTensor focalLoss(IDiffTensor target, double alpha, double gamma) { return DiffTensorLoss.focalLoss(this, target, alpha, gamma); }

    @Override public IDiffTensor diceLoss(IDiffTensor target, double smooth) { return DiffTensorLoss.diceLoss(this, target, smooth); }

    @Override public IDiffTensor nllLoss(IDiffTensor target, int classDim) { return DiffTensorLoss.nllLoss(this, target, classDim); }

    // ==================== Phase 3: Pooling ====================

    @Override public IDiffTensor maxPool2d(int kH, int kW, int stride, int padding) { return DiffTensorPooling.maxPool2d(this, kH, kW, stride, padding); }

    @Override public IDiffTensor avgPool2d(int kH, int kW, int stride, int padding) { return DiffTensorPooling.avgPool2d(this, kH, kW, stride, padding); }

    @Override public IDiffTensor adaptiveAvgPool2d(int outH, int outW) { return DiffTensorPooling.adaptiveAvgPool2d(this, outH, outW); }

    @Override public IDiffTensor oneHot(int numClasses) { return DiffTensorPooling.oneHot(this, numClasses); }

    /** Build output shape for adaptiveAvgPool2d with non-standard input rank. */
    static int[] buildOutShape(int[] inShape, int N, int C, int outH, int outW) {
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

    @Override public IDiffTensor instanceNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return DiffTensorExtOps.instanceNorm(this, gamma, beta, eps); }

    // ==================== Phase 5: diagEmbed ====================

    @Override public IDiffTensor diagEmbed(int offset, int dim1, int dim2) { return DiffTensorExtOps.diagEmbed(this, offset, dim1, dim2); }

    // ==================== Phase 5: dropout2d ====================

    @Override public IDiffTensor dropout2d(double p) { return DiffTensorExtOps.dropout2d(this, p); }

    // ==================== Phase 5: depthwiseConv1d ====================

    @Override public IDiffTensor depthwiseConv1d(IDiffTensor weight, int stride, int padding) { return DiffTensorExtOps.depthwiseConv1d(this, weight, stride, padding); }

    // ==================== Phase 5: interpolate ====================

    @Override public IDiffTensor interpolate(double scaleFactor, String mode) { return DiffTensorExtOps.interpolate(this, scaleFactor, mode); }

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
            // C1: guard against (int) index truncation for large tensors (>2^31 elements)
            if (n > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("copy_ broadcast too large: " + n + " elements");
            }
            double[] oData = detOther.toDoubleArray();
            for (long i = 0; i < n; i++) {
                int[] bcIdx = DiffTensorUtil.unlinearizeInt((int) i, bc);
                int flatSelf = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, shape(), bc);
                int flatOther = DiffTensorUtil.flatIndexFromBroadcast(bcIdx, other.shape(), bc);
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
        if (sData.length != n) {
            throw new IllegalArgumentException(
                "copy_ source length " + sData.length + " != target size " + n);
        }
        for (long i = 0; i < n && i < sData.length; i++) value.linearSet(i, sData[(int) i]);
        this.grad = null;
        return this;
    }

    // ==================== Matrix Decomposition Ops ====================

    @Override public IDiffTensor logDet() { return DiffTensorDecomp.logDet(this); }

    @Override public IDiffTensor[] slogDet() { return DiffTensorDecomp.slogDet(this); }

    @Override public IDiffTensor nuclearNorm() { return DiffTensorDecomp.nuclearNorm(this); }

    @Override public IDiffTensor ctcLoss(IDiffTensor targets, IDiffTensor inputLengths, IDiffTensor targetLengths) { return DiffTensorDecomp.ctcLoss(this, targets, inputLengths, targetLengths); }

    // ==================== cross — 3D vector cross product ====================

    @Override public IDiffTensor cross(IDiffTensor other) { return DiffTensorSpatial.cross(this, other); }

    // ==================== gridSample — differentiable image warp ====================

    @Override public IDiffTensor gridSample(IDiffTensor grid, String mode, String paddingMode) { return DiffTensorSpatial.gridSample(this, grid, mode, paddingMode); }

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

    @Override public IDiffTensor trapezoidalScan(IDiffTensor delta, IDiffTensor A, IDiffTensor B,
                                        IDiffTensor C, IDiffTensor D) { return DiffTensorSpatial.trapezoidalScan(this, delta, A, B, C, D); }

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

    int[] reducedShape(int dim, boolean keepdim) {
        return DiffTensorUtil.reducedShape(shape(), dim, keepdim);
    }

    // ==================== Layer/Batch Normalization ====================

    @Override public IDiffTensor layerNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return DiffTensorNormNN.layerNorm(this, gamma, beta, eps); }

    @Override public IDiffTensor batchNorm(IDiffTensor gamma, IDiffTensor beta, double eps) { return DiffTensorNormNN.batchNorm(this, gamma, beta, eps); }

    @Override public IDiffTensor rmsNorm(IDiffTensor gamma, double eps) { return DiffTensorNormNN.rmsNorm(this, gamma, eps); }

    @Override public IDiffTensor embedding(IDiffTensor indices) { return DiffTensorNormNN.embedding(this, indices); }

    /** Helper: compute output shape [*idxShape, embeddingDim]. */
    private static int[] outShape(int[] idxShape, int embeddingDim) {
        int[] outShape = new int[idxShape.length + 1];
        System.arraycopy(idxShape, 0, outShape, 0, idxShape.length);
        outShape[idxShape.length] = embeddingDim;
        return outShape;
    }

    @Override public IDiffTensor rope(int dim, int maxLen, double base) { return DiffTensorNormNN.rope(this, dim, maxLen, base); }

    @Override public IDiffTensor[] lstmCell(IDiffTensor x, IDiffTensor hPrev, IDiffTensor cPrev,
                                   IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) { return DiffTensorNormNN.lstmCell(this, x, hPrev, cPrev, wInput, wHidden, bias); }

    @Override public IDiffTensor gruCell(IDiffTensor x, IDiffTensor hPrev,
                               IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) { return DiffTensorNormNN.gruCell(this, x, hPrev, wInput, wHidden, bias); }

    @Override public IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
                               int stride, int padding, int dilation) { return DiffTensorNormNN.conv2d(this, weight, bias, stride, padding, dilation); }

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
            double[] rowBuf = new double[cols]; // C9: reuse buffer, avoid copyOfRange per row
            for (int r = 0; r < rows; r++) {
                System.arraycopy(scores, r * cols, rowBuf, 0, cols);
                rowMax[r] = vc.reduceOperate(rowBuf, ReduceOperation.MAX);
            }
        }

        // Step 2: Subtract row max (per-row scalar add)
        double[] shifted = new double[total];
        double[] shiftBuf = new double[cols];
        for (int r = 0; r < rows; r++) {
            int rowOff = r * cols;
            System.arraycopy(scores, rowOff, shiftBuf, 0, cols);
            double[] shiftRow = vc.binaryOperate(shiftBuf, -rowMax[r], BinaryOperation.ADD);
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
            double[] sumBuf = new double[cols];
            for (int r = 0; r < rows; r++) {
                System.arraycopy(exped, r * cols, sumBuf, 0, cols);
                rowSum[r] = vc.reduceOperate(sumBuf, ReduceOperation.SUM);
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

    @Override public IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
                                                  IDiffTensor mask, double dropout) { return DiffTensorNormNN.scaledDotProductAttention(this, key, vTensor, mask, dropout); }

}
