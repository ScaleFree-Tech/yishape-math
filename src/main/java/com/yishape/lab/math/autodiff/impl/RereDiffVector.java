package com.yishape.lab.math.autodiff.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Random;
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
import com.yishape.lab.math.compute.gpu.GpuConfig;
import com.yishape.lab.math.compute.gpu.GpuOptionalRuntime;

/**
 * Default reverse-mode AD implementation for {@link IDiffVector}.
 * {@link IDiffVector} 的默认反向模式自动微分实现。
 *
 * <p>Maintains a DAG of {@link #inputs} and {@link #backwardFn}; topological backward pass
 * accumulates gradients via {@link #accGrad}.
 * 通过输入节点与反向函数构成计算图，拓扑序反向传播并用 {@link #accGrad} 累加梯度。</p>
 */
public class RereDiffVector implements IDiffVector, Serializable {

    private static final long serialVersionUID = 4L;
    private static final YishapeLogger log = YishapeLogger.getLogger(RereDiffVector.class);
    /** Global seed counter for deterministic dropout (reproducible across Java/Rust). */
    static final AtomicLong DROPOUT_SEED_COUNTER = new AtomicLong(0);

    private static final ThreadLocal<ArrayList<RereDiffVector>> TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffVector>> TOPO_SET =
        ThreadLocal.withInitial(HashSet::new);

    public IDoubleVector value;
    public IDoubleVector gradient;
    public boolean isLeaf;
    public List<RereDiffVector> inputs;
    public Consumer<IDoubleVector> backwardFn;
    public Function<IDiffVector, IDiffVector[]> symbolicBackwardFn;
    /** Operation tag for graph export and pattern fusion. / 图导出和模式融合用的操作标记。 */
    public String opTag;
    /** Scalar parameter for scalar operations (NaN if not applicable). / 标量操作参数（不适用时为 NaN）。 */
    public double scalarParam = Double.NaN;
    /** Second scalar parameter for dual-param ops like hardtanh/clamp (NaN if not applicable). */
    public double scalarParam2 = Double.NaN;
    /**
     * Override shape in JSON export. When null, GraphExporter serializes shape as [value.size()].
     * When non-null, serializes the specified N-D shape (e.g. [B, C] for softmaxCrossEntropy).
     */
    public int[] exportShape;
    /**
     * Auxiliary backward data exported to JSON for GPU/HPC backends.
     * For MaxPool2d: argmax indices [B*C*outH*outW], used directly instead of recomputing.
     * When non-null, GraphExporter includes an "indices" array in the node JSON.
     */
    public int[] backwardIndices;

    public RereDiffVector(IDoubleVector value) {
        this.value = value;
        this.isLeaf = true;
        this.inputs = new ArrayList<>();
    }

    RereDiffVector(IDoubleVector value, List<RereDiffVector> inputs, Consumer<IDoubleVector> backwardFn) {
        this(value, inputs, backwardFn, null);
    }

    RereDiffVector(IDoubleVector value, List<RereDiffVector> inputs, Consumer<IDoubleVector> backwardFn,
            Function<IDiffVector, IDiffVector[]> symbolicBackwardFn) {
        this.value = value;
        this.isLeaf = false;
        this.inputs = inputs;
        this.backwardFn = backwardFn;
        this.symbolicBackwardFn = symbolicBackwardFn;
    }

    /**
     * Creates a non-leaf computation node with an embedded backward function.
     * Intended for use by {@link com.yishape.lab.math.autodiff.CustomOp} and similar
     * abstractions that want to build graph nodes without touching the global registry.
     */
    public static IDiffVector createOpNode(IDoubleVector value, List<RereDiffVector> inputs,
                                            Consumer<IDoubleVector> backwardFn) {
        return new RereDiffVector(value, inputs, backwardFn);
    }

    // ---- value / gradient access ----

    @Override
    public IDoubleVector getValue() {
        return value;
    }

    @Override
    public IDoubleVector getGradient() {
        return gradient;
    }

    @Override
    public boolean isLeaf() {
        return isLeaf;
    }

    // ---- gradient operations ----

    @Override
    public void backward() {
        backward(IDoubleVector.ones(value.size()));
    }

    @Override
    public void backward(IDoubleVector initialGradient) {
        this.gradient = initialGradient;

        ArrayList<RereDiffVector> order = TOPO_LIST.get();
        order.clear();
        HashSet<RereDiffVector> visited = TOPO_SET.get();
        visited.clear();
        buildTopo(order, visited);

        try {
            // Zero all intermediate/leaf gradients before accumulation to prevent
            // double-counting when backward() is called multiple times.
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffVector v = order.get(i);
                if (v != this) {
                    v.gradient = null;
                }
            }

            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffVector v = order.get(i);
                if (v.gradient != null && v.backwardFn != null) {
                    v.backwardFn.accept(v.gradient);
                }
            }
        } finally {
            // Release graph node references promptly for GC, even on exception
            order.clear();
            visited.clear();
        }
    }

    /**
     * Nested backward for use from tensor-graph bridge (triggerVectorBackward).
     * Uses LOCAL collections to avoid corrupting the outer backward's ThreadLocal
     * topo list, which would cause IndexOutOfBoundsException on re-entry.
     */
    public void backwardNested(IDoubleVector initialGradient) {
        this.gradient = initialGradient;

        ArrayList<RereDiffVector> order = new ArrayList<>();
        HashSet<RereDiffVector> visited = new HashSet<>();
        buildTopo(order, visited);

        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffVector v = order.get(i);
            if (v != this) {
                v.gradient = null;
            }
        }
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffVector v = order.get(i);
            if (v.gradient != null && v.backwardFn != null) {
                v.backwardFn.accept(v.gradient);
            }
        }
        // Clear root gradient to prevent double-backward when the outer
        // backward (loss.backward()) visits this node in its topo order.
        this.gradient = null;
    }

    /** Iterative DFS post-order topological sort for backward traversal. / 迭代式深度优先后序拓扑排序，避免深层图 StackOverflow。 */
    public void buildTopo(List<RereDiffVector> order, Set<RereDiffVector> visited) {
        // Stack entries: (node, childrenNotPushed). When childrenNotPushed is true,
        // we push children and flip to false; when false, all children are done → add to order.
        java.util.ArrayDeque<Object[]> stack = new java.util.ArrayDeque<>();
        stack.push(new Object[]{this, Boolean.TRUE});
        while (!stack.isEmpty()) {
            Object[] entry = stack.peek();
            RereDiffVector node = (RereDiffVector) entry[0];
            boolean childrenNotPushed = (Boolean) entry[1];
            if (childrenNotPushed) {
                entry[1] = Boolean.FALSE;
                if (!visited.add(node)) {
                    stack.pop();
                    continue;
                }
                for (int i = node.inputs.size() - 1; i >= 0; i--) {
                    RereDiffVector inp = node.inputs.get(i);
                    if (!visited.contains(inp)) {
                        stack.push(new Object[]{inp, Boolean.TRUE});
                    }
                }
            } else {
                stack.pop();
                order.add(node);
            }
        }
    }

    @Override
    public void zeroGradient() {
        this.gradient = null;
    }

    @Override
    public IDiffVector grad() {
        if (this.gradient == null) {
            throw new IllegalStateException("Gradient is null — call backward() first");
        }
        return new RereDiffVector(this.gradient.copy());
    }

    /** Accumulates (sums) gradient into this leaf/intermediate node in-place. / 将梯度就地累加到本节点（无额外分配）。 */
    public void accGrad(IDoubleVector grad) {
        if (gradient == null) {
            gradient = grad.copy();
        } else {
            double[] gData = gradient.getData();
            double[] gradData = grad.getData();
            if (gData.length != gradData.length) {
                throw new IllegalArgumentException(
                    "Gradient length mismatch: existing=" + gData.length + " incoming=" + gradData.length);
            }
            for (int i = 0; i < gData.length; i++) {
                gData[i] += gradData[i];
            }
        }
    }

    /** Takes ownership of a freshly-allocated array, avoiding copy on first gradient accumulation. */
    public void accGradDirect(double[] data) {
        if (gradient == null) {
            gradient = IDoubleVector.of(data);
        } else {
            double[] gData = gradient.getData();
            for (int i = 0; i < gData.length; i++) {
                gData[i] += data[i];
            }
        }
    }

    /** Updates the leaf value in-place. Only valid on leaf nodes. Used by CachedModule for graph-node reuse. */
    public void updateData(double[] newData) {
        if (!isLeaf) {
            throw new IllegalStateException("updateData() is only allowed on leaf nodes");
        }
        double[] dest = value.getData();
        System.arraycopy(newData, 0, dest, 0, Math.min(newData.length, dest.length));
        this.gradient = null;  // Clear stale gradient to prevent double-counting
    }

    static IDiffVector constant(IDoubleVector value) {
        RereDiffVector node = new RereDiffVector(value.copy());
        node.opTag = "constant";
        return node;
    }

    private static RereDiffVector withTag(RereDiffVector node, String tag) {
        node.opTag = tag;
        return node;
    }

    private static RereDiffVector withTag(RereDiffVector node, String tag, double scalarParam) {
        node.opTag = tag;
        node.scalarParam = scalarParam;
        return node;
    }

    @Override
    public IDiffVector broadcast(int n) {
        if (this.value.size() != 1) {
            throw new IllegalArgumentException("broadcast requires scalar (size=1) input, got size=" + this.value.size());
        }
        double scalar = this.value.get(0);
        IDoubleVector resultVal = IDoubleVector.ones(n).multiplyByScalar(scalar);
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGrad(gradOut.sum());
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
            return new IDiffVector[] { gradOut.sum() };
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "broadcast");
    }

    /**
     * Symbolic reverse over the DAG using {@link #symbolicBackwardFn} (tape-of-tape / higher-order AD).
     * 基于 {@link #symbolicBackwardFn} 的符号反向（高阶微分 / tape-of-tape）。
     */
    public static IDiffVector[] grad(IDiffVector output, IDiffVector... inputs) {
        RereDiffVector out = (RereDiffVector) output;

        List<RereDiffVector> order = new ArrayList<>();
        Set<RereDiffVector> visited = new HashSet<>();
        out.buildTopo(order, visited);

        Map<RereDiffVector, IDiffVector> nodeGrads = new HashMap<>();
        nodeGrads.put(out, new RereDiffVector(IDoubleVector.ones(out.value.size())));

        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffVector node = order.get(i);
            IDiffVector nodeGrad = nodeGrads.get(node);
            if (nodeGrad != null && node.symbolicBackwardFn != null) {
                IDiffVector[] localGrads = node.symbolicBackwardFn.apply(nodeGrad);
                List<RereDiffVector> nodeInputs = node.inputs;
                for (int j = 0; j < nodeInputs.size(); j++) {
                    RereDiffVector inputNode = nodeInputs.get(j);
                    IDiffVector existing = nodeGrads.get(inputNode);
                    if (existing == null) {
                        nodeGrads.put(inputNode, localGrads[j]);
                    } else {
                        nodeGrads.put(inputNode, existing.add(localGrads[j]));
                    }
                }
            }
        }

        IDiffVector[] result = new IDiffVector[inputs.length];
        for (int k = 0; k < inputs.length; k++) {
            RereDiffVector in = (RereDiffVector) inputs[k];
            IDiffVector g = nodeGrads.get(in);
            result[k] = g != null ? g : new RereDiffVector(IDoubleVector.zeros(in.value.size()));
        }
        return result;
    }

    // ---- element-wise vector division helper (IDoubleVector.divide returns IVector) ----

    private static IDoubleVector div(IDoubleVector a, IDoubleVector b) {
        return a.multiply(b.reciprocal());
    }

    // ---- zero-allocation backward helpers ----
    // These accumulate gradients in-place into the existing gradient array,
    // avoiding wrapper object allocation. For the first gradient (gradient == null),
    // a correctly-sized array is allocated and wrapped.

    /** accGrad(⊙): this.gradient += gradData * otherData */
    private void accGradMulDirect(double[] gradData, double[] otherData, int n) {
        if (gradient != null) {
            double[] g = gradient.getData();
            for (int i = 0; i < n; i++) g[i] += gradData[i] * otherData[i];
        } else {
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) dx[i] = gradData[i] * otherData[i];
            accGradDirect(dx);
        }
    }

    /** accGrad(scalar⊙): this.gradient += scalar * gradData */
    private void accGradScalarMulDirect(double[] gradData, double scalar, int n) {
        if (gradient != null) {
            double[] g = gradient.getData();
            for (int i = 0; i < n; i++) g[i] += gradData[i] * scalar;
        } else {
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) dx[i] = gradData[i] * scalar;
            accGradDirect(dx);
        }
    }

    /** accGrad(⊙/): this.gradient += gradData / denomData */
    private void accGradDivDirect(double[] gradData, double[] denomData, int n) {
        if (gradient != null) {
            double[] g = gradient.getData();
            for (int i = 0; i < n; i++) g[i] += gradData[i] / denomData[i];
        } else {
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) dx[i] = gradData[i] / denomData[i];
            accGradDirect(dx);
        }
    }

    /**
     * Accumulate from a pooled buffer into this node's gradient.
     * If gradient exists: accumulates in-place (zero alloc), then releases buffer.
     * If gradient is null: allocates correctly-sized copy, accumulates, then releases buffer.
     */
    void accGradFromPooled(double[] pooledBuf, int n) {
        if (gradient != null) {
            double[] g = gradient.getData();
            for (int i = 0; i < n; i++) g[i] += pooledBuf[i];
            AutodiffBufferPool.release(pooledBuf);
        } else {
            double[] dx = new double[n];
            System.arraycopy(pooledBuf, 0, dx, 0, n);
            AutodiffBufferPool.release(pooledBuf);
            accGradDirect(dx);
        }
    }

    // ---- arithmetic with variables ----

    @Override
    public IDiffVector add(IDiffVector other) {
        RereDiffVector o = (RereDiffVector) other;
        IDoubleVector resultVal = this.value.add(o.value);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
            o.accGrad(gradOut);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut, gradOut };
        return withTag(new RereDiffVector(resultVal, List.of(this, o), backwardFn, symFn), "add");
    }

    @Override
    public IDiffVector sub(IDiffVector other) {
        RereDiffVector o = (RereDiffVector) other;
        IDoubleVector resultVal = this.value.sub(o.value);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            int n = gd.length;
            this.accGradDirect(gd);
            o.accGradScalarMulDirect(gd, -1.0, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut, gradOut.mul(-1) };
        return withTag(new RereDiffVector(resultVal, List.of(this, o), backwardFn, symFn), "sub");
    }

    @Override
    public IDiffVector mul(IDiffVector other) {
        RereDiffVector o = (RereDiffVector) other;
        IDoubleVector resultVal = this.value.multiply(o.value);
        IDoubleVector thisVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector otherVal = o.isLeaf ? o.value.copy() : o.value;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] od = otherVal.getData();
            double[] td = thisVal.getData();
            int n = gd.length;
            this.accGradMulDirect(gd, od, n);
            o.accGradMulDirect(gd, td, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(other), gradOut.mul(this) };
        return withTag(new RereDiffVector(resultVal, List.of(this, o), backwardFn, symFn), "mul");
    }

    @Override
    public IDiffVector div(IDiffVector other) {
        RereDiffVector o = (RereDiffVector) other;
        IDoubleVector resultVal = div(this.value, o.value);
        IDoubleVector thisVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector otherVal = o.isLeaf ? o.value.copy() : o.value;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] od = otherVal.getData();
            double[] td = thisVal.getData();
            int n = gd.length;
            // this.grad += gradOut / otherVal
            this.accGradDivDirect(gd, od, n);
            // o.grad += (-gradOut * thisVal) / otherVal^2
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) buf[i] = -gd[i] * td[i] / (od[i] * od[i]);
            o.accGradFromPooled(buf, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] {
                gradOut.div(other), gradOut.mul(-1).mul(this).div(other.square()) };
        return withTag(new RereDiffVector(resultVal, List.of(this, o), backwardFn, symFn), "div");
    }

    // ---- arithmetic with scalars ----

    @Override
    public IDiffVector add(double scalar) {
        if (scalar == 0.0) return this;
        IDoubleVector resultVal = this.value.addScalar(scalar);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "addScalar", scalar);
    }

    @Override
    public IDiffVector sub(double scalar) {
        if (scalar == 0.0) return this;
        IDoubleVector resultVal = this.value.subScalar(scalar);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "subScalar", scalar);
    }

    @Override
    public IDiffVector mul(double scalar) {
        if (scalar == 1.0) return this;
        if (scalar == 0.0) return constant(IDoubleVector.zeros(this.value.size()));
        IDoubleVector resultVal = this.value.multiplyByScalar(scalar);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            this.accGradScalarMulDirect(gd, scalar, gd.length);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(scalar) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "mulScalar", scalar);
    }

    @Override
    public IDiffVector div(double scalar) {
        if (scalar == 1.0) return this;
        IDoubleVector resultVal = this.value.divideByScalar(scalar);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.divideByScalar(scalar));
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.div(scalar) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "divScalar", scalar);
    }

    @Override
    public IDiffVector rsub(double scalar) {
        IDoubleVector resultVal = this.value.map(v -> scalar - v);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            this.accGradScalarMulDirect(gd, -1.0, gd.length);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(-1) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "rsubScalar", scalar);
    }

    @Override
    public IDiffVector rdiv(double scalar) {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.map(v -> scalar / v);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] xd = xVal.getData();
            int n = gd.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) buf[i] = -scalar * gd[i] / (xd[i] * xd[i]);
            this.accGradFromPooled(buf, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] {
                gradOut.mul(-scalar).div(this.square()) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "rdivScalar", scalar);
    }

    // ---- unary ----

    @Override
    public IDiffVector neg() {
        IDoubleVector resultVal = this.value.multiplyByScalar(-1.0);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            this.accGradScalarMulDirect(gd, -1.0, gd.length);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(-1) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "neg");
    }

    @Override
    public IDiffVector pow(double n) {
        if (n == 1.0) return this;
        if (n == 0.0) return constant(IDoubleVector.ones(this.value.size()));
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.pow(n);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] xd = xVal.getData();
            int len = gd.length;
            double[] buf = AutodiffBufferPool.acquire(len);
            for (int i = 0; i < len; i++) buf[i] = gd[i] * n * Math.pow(xd[i], n - 1);
            this.accGradFromPooled(buf, len);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(n).mul(this.pow(n - 1)) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "pow", n);
    }

    // ---- element-wise math ----

    @Override
    public IDiffVector exp() {
        IDoubleVector resultVal = this.value.exp();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] zd = resultVal.getData();
            this.accGradMulDirect(gd, zd, gd.length);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(this.exp()) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "exp");
    }

    @Override
    public IDiffVector log() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.log();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] xd = xVal.getData();
            this.accGradDivDirect(gd, xd, gd.length);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.div(this) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "log");
    }

    @Override
    public IDiffVector sin() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.sin();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] xd = xVal.getData();
            int n = gd.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) buf[i] = gd[i] * Math.cos(xd[i]);
            this.accGradFromPooled(buf, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(this.cos()) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "sin");
    }

    @Override
    public IDiffVector cos() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.cos();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] xd = xVal.getData();
            int n = gd.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) buf[i] = -gd[i] * Math.sin(xd[i]);
            this.accGradFromPooled(buf, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(-1).mul(this.sin()) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "cos");
    }

    @Override
    public IDiffVector tan() {
        IDoubleVector resultVal = this.value.tan();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] zd = resultVal.getData();
            int n = gd.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double t = zd[i];
                buf[i] = gd[i] * (t * t + 1.0);
            }
            this.accGradFromPooled(buf, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] {
                gradOut.mul(this.tan().square().add(1)) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "tan");
    }

    @Override
    public IDiffVector tanh() {
        IDoubleVector resultVal = this.value.tanh();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] zd = resultVal.getData();
            int n = gd.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double t = zd[i];
                buf[i] = gd[i] * (1.0 - t * t);
            }
            this.accGradFromPooled(buf, n);
        };
        IDiffVector constGrad = constant(resultVal.map(v -> 1.0 - v * v));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constGrad) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "tanh");
    }

    @Override
    public IDiffVector sigmoid() {
        IDoubleVector resultVal = this.value.sigmoid();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] zd = resultVal.getData();
            int n = gd.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double s = zd[i];
                buf[i] = gd[i] * s * (1.0 - s);
            }
            this.accGradFromPooled(buf, n);
        };
        IDiffVector constGrad = constant(resultVal.map(v -> v * (1.0 - v)));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constGrad) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "sigmoid");
    }

    @Override
    public IDiffVector relu() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.relu();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gradData = gradOut.getData();
            double[] xData = xVal.getData();
            int n = gradData.length;
            double[] dx = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                dx[i] = gradData[i] * (xData[i] > 0.0 ? 1.0 : 0.0);
            }
            this.accGradFromPooled(dx, n);
        };
        double[] mask = new double[xVal.size()];
        double[] xd = xVal.getData();
        for (int i = 0; i < mask.length; i++) {
            mask[i] = xd[i] > 0.0 ? 1.0 : 0.0;
        }
        IDiffVector constMask = constant(IDoubleVector.of(mask));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constMask) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "relu");
    }

    @Override
    public IDiffVector softmax() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        // Try GPU fused softmax (single-row: rows=1, cols=n)
        double[] y = GpuConfig.allowAttempts() ? GpuOptionalRuntime.trySoftmax(xd, 1, n) : null;
        if (y == null) {
            // CPU fallback
            double maxVal = xd[0];
            for (int i = 1; i < n; i++) {
                if (xd[i] > maxVal) maxVal = xd[i];
            }
            double[] expVals = new double[n];
            double sumExp = 0;
            for (int i = 0; i < n; i++) {
                expVals[i] = Math.exp(xd[i] - maxVal);
                sumExp += expVals[i];
            }
            y = new double[n];
            for (int i = 0; i < n; i++) {
                y[i] = expVals[i] / sumExp;
            }
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] yd = resultVal.getData();
            double dot = 0;
            for (int j = 0; j < n; j++) dot += gd[j] * yd[j];
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) buf[i] = yd[i] * (gd[i] - dot);
            this.accGradFromPooled(buf, n);
        };
        IDiffVector yConst = constant(resultVal);
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
            IDiffVector dotTerm = gradOut.mul(yConst).sum();
            return new IDiffVector[] { yConst.mul(gradOut.sub(dotTerm.broadcast(n))) };
        };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(this), backwardFn, symFn);
        node.opTag = "softmax";
        return node;
    }

    @Override
    public IDiffVector logSoftmax() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        // Try GPU fused logSoftmax (single-row); softmax values needed for backward
        double[] y = GpuConfig.allowAttempts() ? GpuOptionalRuntime.tryLogSoftmax(xd, 1, n) : null;
        double[] sm;
        if (y != null) {
            sm = new double[n];
            for (int i = 0; i < n; i++) sm[i] = Math.exp(y[i]);
        } else {
            // CPU fallback
            double maxVal = xd[0];
            for (int i = 1; i < n; i++) {
                if (xd[i] > maxVal) maxVal = xd[i];
            }
            double sumExp = 0;
            double[] expVals = new double[n];
            for (int i = 0; i < n; i++) {
                expVals[i] = Math.exp(xd[i] - maxVal);
                sumExp += expVals[i];
            }
            double logSumExp = Math.log(sumExp) + maxVal;
            y = new double[n];
            sm = new double[n];
            for (int i = 0; i < n; i++) {
                sm[i] = expVals[i] / sumExp;
                y[i] = xd[i] - logSumExp;
            }
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        IDoubleVector smVec = IDoubleVector.of(sm);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double sumGrad = 0;
            for (int j = 0; j < n; j++) sumGrad += gd[j];
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                buf[i] = gd[i] - sm[i] * sumGrad;
            }
            this.accGradFromPooled(buf, n);
        };
        IDiffVector smConst = constant(smVec);
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
            IDiffVector sumGrad = gradOut.sum();
            return new IDiffVector[] { gradOut.sub(smConst.mul(sumGrad.broadcast(n))) };
        };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(this), backwardFn, symFn);
        node.opTag = "logSoftmax";
        return node;
    }

    @Override
    public IDiffVector gelu() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
        double g = 0.044715;
        double[] y = new double[n];
        double[] tanhVals = new double[n];
        double[] innerVals = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            double inner = sqrt2OverPi * (x + g * x * x * x);
            innerVals[i] = inner;
            double tanhInner = Math.tanh(inner);
            tanhVals[i] = tanhInner;
            y[i] = 0.5 * x * (1.0 + tanhInner);
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double x = xd[i];
                double tanhI = tanhVals[i];
                double inner = innerVals[i];
                double sechSq = 1.0 - tanhI * tanhI;
                double din_dx = sqrt2OverPi * (1.0 + 3.0 * g * x * x);
                buf[i] = gd[i] * (0.5 * (1.0 + tanhI) + 0.5 * x * sechSq * din_dx);
            }
            this.accGradFromPooled(buf, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
            IDiffVector x = constant(xVal);
            double sg = sqrt2OverPi;
            double gg = g;
            IDiffVector inner = x.mul(gg).mul(x).mul(x).add(x).mul(sg);
            IDiffVector tanhInner = inner.tanh();
            IDiffVector gradTerm = tanhInner.add(1).mul(0.5)
                .add(x.mul(0.5).mul(constant(IDoubleVector.ones(n)).sub(tanhInner.square())).mul(
                    constant(IDoubleVector.of(new double[]{sg})).broadcast(n).mul(
                        x.square().mul(3 * gg).add(1))));
            return new IDiffVector[] { gradOut.mul(gradTerm) };
        };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(this), backwardFn, symFn);
        node.opTag = "gelu";
        return node;
    }

    @Override
    public IDiffVector leakyRelu(double alpha) {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            y[i] = x > 0 ? x : alpha * x;
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                buf[i] = gd[i] * (xd[i] > 0 ? 1.0 : alpha);
            }
            this.accGradFromPooled(buf, n);
        };
        double[] mask = new double[n];
        for (int i = 0; i < n; i++) mask[i] = xd[i] > 0 ? 1.0 : alpha;
        IDiffVector constMask = constant(IDoubleVector.of(mask));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constMask) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "leakyRelu", alpha);
    }

    @Override
    public IDiffVector elu(double alpha) {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double[] y = new double[n];
        double[] expVal = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            if (x >= 0) {
                y[i] = x;
                expVal[i] = 0;
            } else {
                double ex = Math.exp(x);
                expVal[i] = ex;
                y[i] = alpha * (ex - 1.0);
            }
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                if (xd[i] >= 0) {
                    buf[i] = gd[i];
                } else {
                    buf[i] = gd[i] * alpha * expVal[i];
                }
            }
            this.accGradFromPooled(buf, n);
        };
        double[] eluGradData = new double[n];
        for (int i = 0; i < n; i++) {
            eluGradData[i] = xd[i] >= 0 ? 1.0 : alpha * expVal[i];
        }
        IDiffVector constGrad = constant(IDoubleVector.of(eluGradData));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constGrad) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "elu", alpha);
    }

    @Override
    public IDiffVector selu() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double alpha = 1.6732632423543772848170429916717;
        double scale = 1.0507009873554804934193349852946;
        double[] y = new double[n];
        double[] expVal = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            if (x >= 0) {
                y[i] = scale * x;
                expVal[i] = 0;
            } else {
                double ex = Math.exp(x);
                expVal[i] = ex;
                y[i] = scale * alpha * (ex - 1.0);
            }
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                if (xd[i] >= 0) {
                    buf[i] = gd[i] * scale;
                } else {
                    buf[i] = gd[i] * scale * alpha * expVal[i];
                }
            }
            this.accGradFromPooled(buf, n);
        };
        double[] seluGradData = new double[n];
        for (int i = 0; i < n; i++) {
            seluGradData[i] = xd[i] >= 0 ? scale : scale * alpha * expVal[i];
        }
        IDiffVector constGrad = constant(IDoubleVector.of(seluGradData));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constGrad) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "selu");
    }

    @Override
    public IDiffVector silu() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double[] y = new double[n];
        double[] sigVals = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            double s = 1.0 / (1.0 + Math.exp(-x));
            sigVals[i] = s;
            y[i] = x * s;
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double s = sigVals[i];
                buf[i] = gd[i] * (s + xd[i] * s * (1.0 - s));
            }
            this.accGradFromPooled(buf, n);
        };
        double[] siluGradData = new double[n];
        for (int i = 0; i < n; i++) {
            double s = sigVals[i];
            siluGradData[i] = s + xd[i] * s * (1.0 - s);
        }
        IDiffVector constGrad = constant(IDoubleVector.of(siluGradData));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constGrad) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "silu");
    }

    @Override
    public IDiffVector mish() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double[] y = new double[n];
        double[] spVals = new double[n];
        double[] tanhVals = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            double sp = Math.log(1.0 + Math.exp(x));
            spVals[i] = sp;
            double th = Math.tanh(sp);
            tanhVals[i] = th;
            y[i] = x * th;
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double x = xd[i];
                double th = tanhVals[i];
                double sig = 1.0 / (1.0 + Math.exp(-x));
                double dTanh = 1.0 - th * th;
                buf[i] = gd[i] * (th + x * dTanh * sig);
            }
            this.accGradFromPooled(buf, n);
        };
        double[] mishGradData = new double[n];
        for (int i = 0; i < n; i++) {
            double th = tanhVals[i];
            double sig = 1.0 / (1.0 + Math.exp(-xd[i]));
            mishGradData[i] = th + xd[i] * (1.0 - th * th) * sig;
        }
        IDiffVector constGrad = constant(IDoubleVector.of(mishGradData));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constGrad) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "mish");
    }

    @Override
    public IDiffVector softplus(double beta) {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double bx = beta * xd[i];
            y[i] = bx > 100 ? xd[i] : (1.0 / beta) * Math.log(1.0 + Math.exp(bx));
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double bx = beta * xd[i];
                double sig = bx > 100 ? 1.0 : 1.0 / (1.0 + Math.exp(-bx));
                buf[i] = gd[i] * sig;
            }
            this.accGradFromPooled(buf, n);
        };
        double[] sigVals = new double[n];
        for (int i = 0; i < n; i++) {
            double bx = beta * xd[i];
            sigVals[i] = bx > 100 ? 1.0 : 1.0 / (1.0 + Math.exp(-bx));
        }
        IDiffVector constGrad = constant(IDoubleVector.of(sigVals));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constGrad) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "softplus", beta);
    }

    @Override
    public IDiffVector hardtanh(double minVal, double maxVal) {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double v = xd[i];
            y[i] = v < minVal ? minVal : (v > maxVal ? maxVal : v);
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double v = xd[i];
                buf[i] = gd[i] * (v > minVal && v < maxVal ? 1.0 : 0.0);
            }
            this.accGradFromPooled(buf, n);
        };
        double[] mask = new double[n];
        for (int i = 0; i < n; i++) {
            double v = xd[i];
            mask[i] = (v > minVal && v < maxVal) ? 1.0 : 0.0;
        }
        IDiffVector constMask = constant(IDoubleVector.of(mask));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constMask) };
        RereDiffVector hardtanhNode = new RereDiffVector(resultVal, List.of(this), backwardFn, symFn);
        hardtanhNode.opTag = "hardtanh";
        hardtanhNode.scalarParam = minVal;
        hardtanhNode.scalarParam2 = maxVal;
        return hardtanhNode;
    }

    @Override
    public IDiffVector clamp(double min, double max) {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        int n = xVal.size();
        double[] xd = xVal.getData();
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double v = xd[i];
            y[i] = v < min ? min : (v > max ? max : v);
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                double v = xd[i];
                buf[i] = gd[i] * (v >= min && v <= max ? 1.0 : 0.0);
            }
            this.accGradFromPooled(buf, n);
        };
        double[] mask = new double[n];
        for (int i = 0; i < n; i++) {
            double v = xd[i];
            mask[i] = (v >= min && v <= max) ? 1.0 : 0.0;
        }
        IDiffVector constMask = constant(IDoubleVector.of(mask));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constMask) };
        RereDiffVector clampNode = new RereDiffVector(resultVal, List.of(this), backwardFn, symFn);
        clampNode.opTag = "clamp";
        clampNode.scalarParam = min;
        clampNode.scalarParam2 = max;
        return clampNode;
    }

    @Override
    public IDiffVector layerNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector gr = (RereDiffVector) gamma;
        RereDiffVector br = (RereDiffVector) beta;
        int n = this.value.size();
        int features = gr.value.size();
        int batch = n / features;
        if (n % features != 0) {
            throw new IllegalArgumentException(
                "Input size (" + n + ") not divisible by features (" + features + ")");
        }
        double[] xd = this.value.getData();
        double[] gd = gr.value.getData();
        double[] bd = br.value.getData();

        double[] y = new double[n];
        double[] xHat = new double[n];
        double[] means = new double[batch];
        double[] sigmas = new double[batch];

        // Forward: for each position, normalize over features
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

        Consumer<IDoubleVector> backwardFn = (gradOutput) -> {
            double[] g = ((RereDoubleVector) gradOutput).getData();
            double[] cg = gr.value.getData();
            double[] dx = AutodiffBufferPool.acquire(n);
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

            this.accGradFromPooled(dx, n);
            gr.accGradFromPooled(dGamma, features);
            br.accGradFromPooled(dBeta, features);
        };

        RereDiffVector node = new RereDiffVector(IDoubleVector.of(y), List.of(this, gr, br), backwardFn);
        node.opTag = "layerNorm";
        node.scalarParam = eps;
        return node;
    }

    @Override
    public IDiffVector batchNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector gr = (RereDiffVector) gamma;
        RereDiffVector br = (RereDiffVector) beta;
        int n = this.value.size();
        int features = gr.value.size();
        int batch = n / features;
        if (n % features != 0) {
            throw new IllegalArgumentException(
                "Input size (" + n + ") not divisible by features (" + features + ")");
        }
        double[] xd = this.value.getData();
        double[] gd = gr.value.getData();
        double[] bd = br.value.getData();

        double[] y = new double[n];
        double[] means = new double[features];
        double[] invSigmas = new double[features];

        // Forward: for each feature, normalize over batch dimension
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

        Consumer<IDoubleVector> backwardFn = (gradOutput) -> {
            double[] g = ((RereDoubleVector) gradOutput).getData();
            double[] cg = gr.value.getData();
            double[] dx = AutodiffBufferPool.acquire(n);
            double[] dGamma = AutodiffBufferPool.acquire(features);
            double[] dBeta = AutodiffBufferPool.acquire(features);

            for (int j = 0; j < features; j++) {
                double invSig = invSigmas[j];
                double mean = means[j];
                // Compute dGamma and dBeta
                double dg = 0, db = 0;
                for (int i = 0; i < batch; i++) {
                    int idx = i * features + j;
                    double xHat = (xd[idx] - mean) * invSig;
                    dg += g[idx] * xHat;
                    db += g[idx];
                }
                dGamma[j] = dg;
                dBeta[j] = db;

                // Compute dx using batch norm backward formula:
                // dx = (gamma / (batch * sigma)) * (batch * g - sum(g) - xHat * sum(g * xHat))
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

            this.accGradFromPooled(dx, n);
            gr.accGradFromPooled(dGamma, features);
            br.accGradFromPooled(dBeta, features);
        };

        RereDiffVector node = new RereDiffVector(IDoubleVector.of(y), List.of(this, gr, br), backwardFn);
        node.opTag = "batchNorm";
        node.scalarParam = eps;
        return node;
    }

    @Override
    public IDiffVector dropout(double p) {
        IDoubleVector xVal = this.value.copy();
        int n = xVal.size();
        double scale = 1.0 / (1.0 - p);
        long seed = DROPOUT_SEED_COUNTER.incrementAndGet();
        Random rng = new Random(seed);
        double[] mask = new double[n];
        double[] y = new double[n];
        double[] xd = xVal.getData();
        for (int i = 0; i < n; i++) {
            mask[i] = rng.nextDouble() > p ? scale : 0.0;
            y[i] = xd[i] * mask[i];
        }
        IDoubleVector resultVal = IDoubleVector.of(y);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                buf[i] = gd[i] * mask[i];
            }
            this.accGradFromPooled(buf, n);
        };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(this), backwardFn);
        node.opTag = "dropout";
        node.scalarParam = p;
        node.scalarParam2 = Double.longBitsToDouble(seed);
        return node;
    }

    @Override
    public IDiffVector abs() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.abs();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // Subgradient convention: sign(x) = 1 at x=0 (consistent with relu)
            double[] xd = xVal.getData();
            double[] gd = ((RereDoubleVector) gradOut).getData();
            int n = gd.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                buf[i] = gd[i] * (xd[i] >= 0.0 ? 1.0 : -1.0);
            }
            this.accGradFromPooled(buf, n);
        };
        // Pre-compute sign for symbolic backward (also uses subgradient at 0)
        double[] signData = xVal.getData().clone();
        for (int i = 0; i < signData.length; i++) {
            signData[i] = signData[i] >= 0.0 ? 1.0 : -1.0;
        }
        IDiffVector constSign = constant(IDoubleVector.of(signData));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(constSign) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "abs");
    }

    @Override
    public IDiffVector sqrt() {
        IDoubleVector resultVal = this.value.sqrt();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            double[] zd = resultVal.getData();
            int n = gd.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) buf[i] = gd[i] / (2.0 * zd[i]);
            this.accGradFromPooled(buf, n);
        };
        IDiffVector constDenom = constant(resultVal.multiplyByScalar(2.0));
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.div(constDenom) };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "sqrt");
    }

    @Override
    public IDiffVector square() {
        IDoubleVector xVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.square();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gData = ((RereDoubleVector) gradOut).getData();
            double[] xData = xVal.getData();
            int n = gData.length;
            double[] buf = AutodiffBufferPool.acquire(n);
            for (int i = 0; i < n; i++) {
                buf[i] = 2.0 * gData[i] * xData[i];
            }
            this.accGradFromPooled(buf, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(2).mul(this) };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(this), backwardFn, symFn);
        node.opTag = "square";
        return node;
    }

    // ---- reductions ----

    @Override
    public IDiffVector sum() {
        // Pattern fusion: square().sum() → single fused node
        if ("square".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] buf = AutodiffBufferPool.acquire(len);
                for (int i = 0; i < len; i++) {
                    buf[i] = 2.0 * g * xData[i];
                }
                x.accGradFromPooled(buf, len);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(2).mul(x) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "squareSum";
            return node;
        }
        // Pattern fusion: exp().sum() → single fused node
        if ("exp".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] buf = AutodiffBufferPool.acquire(len);
                for (int i = 0; i < len; i++) {
                    buf[i] = g * Math.exp(xData[i]);
                }
                x.accGradFromPooled(buf, len);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(x.exp()) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "expSum";
            return node;
        }
        // Pattern fusion: pow(N).sum() → single fused node
        if ("pow".equals(this.opTag) && this.inputs.size() == 1) {
            double n_pow = this.scalarParam;
            if (!Double.isNaN(n_pow)) {
                RereDiffVector x = this.inputs.get(0);
                IDoubleVector xVal = x.value.copy();
                double s = this.value.sumValue();
                IDoubleVector resultVal = IDoubleVector.of(s);
                Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                    double g = gradOut.get(0);
                    double[] xData = xVal.getData();
                    int len = xData.length;
                    double[] buf = AutodiffBufferPool.acquire(len);
                    for (int i = 0; i < len; i++) {
                        buf[i] = g * n_pow * Math.pow(xData[i], n_pow - 1.0);
                    }
                    x.accGradFromPooled(buf, len);
                };
                Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                    new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(n_pow).mul(x.pow(n_pow - 1)) };
                RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
                node.opTag = "powSum";
                return node;
            }
        }
        // Pattern fusion: abs().sum() → single fused node (L1 norm)
        if ("abs".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] buf = AutodiffBufferPool.acquire(len);
                for (int i = 0; i < len; i++) {
                    buf[i] = g * Math.signum(xData[i]);
                }
                x.accGradFromPooled(buf, len);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
                double[] signData = new double[xVal.size()];
                double[] xd = xVal.getData();
                for (int i = 0; i < signData.length; i++) signData[i] = Math.signum(xd[i]);
                return new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(new RereDiffVector(IDoubleVector.of(signData))) };
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "absSum";
            return node;
        }
        // Pattern fusion: relu().sum() → single fused node
        if ("relu".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] buf = AutodiffBufferPool.acquire(len);
                for (int i = 0; i < len; i++) {
                    buf[i] = xData[i] > 0 ? g : 0;
                }
                x.accGradFromPooled(buf, len);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
                double[] maskData = new double[xVal.size()];
                double[] xd = xVal.getData();
                for (int i = 0; i < maskData.length; i++) maskData[i] = xd[i] > 0 ? 1.0 : 0.0;
                return new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(new RereDiffVector(IDoubleVector.of(maskData))) };
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "reluSum";
            return node;
        }
        // Pattern fusion: log().sum() → single fused node (log-likelihood)
        if ("log".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    fused[i] = g / xData[i];
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.broadcast(xVal.size()).div(x) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "logSum";
            return node;
        }
        // Pattern fusion: sigmoid().sum() → single fused node
        if ("sigmoid".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            IDoubleVector outVal = this.value.copy(); // sigmoid(x)
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] outData = outVal.getData();
                int len = outData.length;
                double[] buf = AutodiffBufferPool.acquire(len);
                for (int i = 0; i < len; i++) {
                    buf[i] = g * outData[i] * (1.0 - outData[i]);
                }
                x.accGradFromPooled(buf, len);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(x.sigmoid().mul(x.sigmoid().rsub(1.0))) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "sigmoidSum";
            return node;
        }
        // Pattern fusion: tanh().sum() → single fused node
        if ("tanh".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            IDoubleVector outVal = this.value.copy(); // tanh(x)
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] outData = outVal.getData();
                int len = outData.length;
                double[] buf = AutodiffBufferPool.acquire(len);
                for (int i = 0; i < len; i++) {
                    buf[i] = g * (1.0 - outData[i] * outData[i]);
                }
                x.accGradFromPooled(buf, len);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(x.tanh().square().rsub(1.0)) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "tanhSum";
            return node;
        }
        // Pattern fusion: silu().sum() → single fused node
        if ("silu".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            IDoubleVector outVal = this.value.copy(); // silu(x)
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] xData = xVal.getData();
                double[] outData = outVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    double sig = 1.0 / (1.0 + Math.exp(-xData[i]));
                    fused[i] = g * (sig + xData[i] * sig * (1.0 - sig));
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
                IDiffVector sig = x.sigmoid();
                return new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(sig.add(x.mul(sig).mul(sig.rsub(1.0)))) };
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "siluSum";
            return node;
        }
        // Pattern fusion: mish().sum() → single fused node
        if ("mish".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            IDoubleVector outVal = this.value.copy(); // mish(x)
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    double sp = Math.log(1.0 + Math.exp(xData[i]));
                    double th = Math.tanh(sp);
                    double sig = 1.0 / (1.0 + Math.exp(-xData[i]));
                    fused[i] = g * (th + xData[i] * (1.0 - th * th) * sig);
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
                IDiffVector sp = x.exp().add(1.0).log();
                IDiffVector th = sp.tanh();
                IDiffVector sig = x.sigmoid();
                return new IDiffVector[] { gradOut.broadcast(xVal.size()).mul(th.add(x.mul(th.square().rsub(1.0)).mul(sig))) };
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "mishSum";
            return node;
        }
        // Pattern fusion: mul(x).sum() → single fused node (dot-like)
        if ("mul".equals(this.opTag) && this.inputs.size() == 2) {
            RereDiffVector a = this.inputs.get(0);
            RereDiffVector b = this.inputs.get(1);
            IDoubleVector aVal = a.value.copy();
            IDoubleVector bVal = b.value.copy();
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                int len = aVal.size();
                double[] gradA = new double[len];
                double[] gradB = new double[len];
                double[] aData = aVal.getData();
                double[] bData = bVal.getData();
                for (int i = 0; i < len; i++) {
                    gradA[i] = g * bData[i];
                    gradB[i] = g * aData[i];
                }
                a.accGradDirect(gradA);
                b.accGradDirect(gradB);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] {
                    gradOut.broadcast(aVal.size()).mul(b),
                    gradOut.broadcast(bVal.size()).mul(a)
                };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(a, b), backwardFn, symFn);
            node.opTag = "mulSum";
            return node;
        }
        double s = this.value.sumValue();
        IDoubleVector resultVal = IDoubleVector.of(s);
        int n = this.value.size();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            if (this.gradient == null) {
                double[] dx = new double[n];
                java.util.Arrays.fill(dx, g);
                this.gradient = IDoubleVector.of(dx);
            } else {
                double[] gData = this.gradient.getData();
                for (int i = 0; i < n; i++) {
                    gData[i] += g;
                }
            }
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.broadcast(n) };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(this), backwardFn, symFn);
        node.opTag = "sum";
        return node;
    }

    @Override
    public IDiffVector mean() {
        // Pattern fusion: square().mean() → single fused node
        if ("square".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    fused[i] = 2.0 * g * xData[i];
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.div(n).broadcast(n).mul(2).mul(x) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "squareMean";
            return node;
        }
        // Pattern fusion: exp().mean() → single fused node
        if ("exp".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    fused[i] = g * Math.exp(xData[i]);
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.div(n).broadcast(n).mul(x.exp()) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "expMean";
            return node;
        }
        // Pattern fusion: pow(N).mean() → single fused node
        if ("pow".equals(this.opTag) && this.inputs.size() == 1) {
            double n_pow = this.scalarParam;
            if (!Double.isNaN(n_pow)) {
                RereDiffVector x = this.inputs.get(0);
                IDoubleVector xVal = x.value.copy();
                int n = xVal.size();
                double m = this.value.meanValue();
                IDoubleVector resultVal = IDoubleVector.of(m);
                Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                    double g = gradOut.get(0) / n;
                    double[] xData = xVal.getData();
                    int len = xData.length;
                    double[] fused = new double[len];
                    for (int i = 0; i < len; i++) {
                        fused[i] = g * n_pow * Math.pow(xData[i], n_pow - 1.0);
                    }
                    x.accGradDirect(fused);
                };
                Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                    new IDiffVector[] { gradOut.div(n).broadcast(n).mul(n_pow).mul(x.pow(n_pow - 1)) };
                RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
                node.opTag = "powMean";
                return node;
            }
        }
        // Pattern fusion: abs().mean() → single fused node
        if ("abs".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    fused[i] = g * Math.signum(xData[i]);
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
                double[] signData = new double[xVal.size()];
                double[] xd = xVal.getData();
                for (int i = 0; i < signData.length; i++) signData[i] = Math.signum(xd[i]);
                return new IDiffVector[] { gradOut.div(n).broadcast(n).mul(new RereDiffVector(IDoubleVector.of(signData))) };
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "absMean";
            return node;
        }
        // Pattern fusion: relu().mean() → single fused node
        if ("relu".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    fused[i] = xData[i] > 0 ? g : 0;
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
                double[] maskData = new double[xVal.size()];
                double[] xd = xVal.getData();
                for (int i = 0; i < maskData.length; i++) maskData[i] = xd[i] > 0 ? 1.0 : 0.0;
                return new IDiffVector[] { gradOut.div(n).broadcast(n).mul(new RereDiffVector(IDoubleVector.of(maskData))) };
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "reluMean";
            return node;
        }
        // Pattern fusion: log().mean() → single fused node
        if ("log".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    fused[i] = g / xData[i];
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.div(n).broadcast(n).div(x) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "logMean";
            return node;
        }
        // Pattern fusion: sigmoid().mean() → single fused node
        if ("sigmoid".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            IDoubleVector outVal = this.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] outData = outVal.getData();
                int len = outData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    fused[i] = g * outData[i] * (1.0 - outData[i]);
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.div(n).broadcast(n).mul(x.sigmoid().mul(x.sigmoid().rsub(1.0))) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "sigmoidMean";
            return node;
        }
        // Pattern fusion: tanh().mean() → single fused node
        if ("tanh".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            IDoubleVector outVal = this.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] outData = outVal.getData();
                int len = outData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    fused[i] = g * (1.0 - outData[i] * outData[i]);
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) ->
                new IDiffVector[] { gradOut.div(n).broadcast(n).mul(x.tanh().square().rsub(1.0)) };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "tanhMean";
            return node;
        }
        // Pattern fusion: silu().mean() → single fused node
        if ("silu".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            IDoubleVector outVal = this.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    double sig = 1.0 / (1.0 + Math.exp(-xData[i]));
                    fused[i] = g * (sig + xData[i] * sig * (1.0 - sig));
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
                IDiffVector sig = x.sigmoid();
                return new IDiffVector[] { gradOut.div(n).broadcast(n).mul(sig.add(x.mul(sig).mul(sig.rsub(1.0)))) };
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "siluMean";
            return node;
        }
        // Pattern fusion: mish().mean() → single fused node
        if ("mish".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffVector x = this.inputs.get(0);
            IDoubleVector xVal = x.value.copy();
            IDoubleVector outVal = this.value.copy();
            int n = xVal.size();
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[] xData = xVal.getData();
                int len = xData.length;
                double[] fused = new double[len];
                for (int i = 0; i < len; i++) {
                    double sp = Math.log(1.0 + Math.exp(xData[i]));
                    double th = Math.tanh(sp);
                    double sig = 1.0 / (1.0 + Math.exp(-xData[i]));
                    fused[i] = g * (th + xData[i] * (1.0 - th * th) * sig);
                }
                x.accGradDirect(fused);
            };
            Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
                IDiffVector sp = x.exp().add(1.0).log();
                IDiffVector th = sp.tanh();
                IDiffVector sig = x.sigmoid();
                return new IDiffVector[] { gradOut.div(n).broadcast(n).mul(th.add(x.mul(th.square().rsub(1.0)).mul(sig))) };
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(x), backwardFn, symFn);
            node.opTag = "mishMean";
            return node;
        }
        double m = this.value.meanValue();
        IDoubleVector resultVal = IDoubleVector.of(m);
        int n = this.value.size();
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0) / n;
            if (this.gradient == null) {
                double[] dx = new double[n];
                java.util.Arrays.fill(dx, g);
                this.gradient = IDoubleVector.of(dx);
            } else {
                double[] gData = this.gradient.getData();
                for (int i = 0; i < n; i++) {
                    gData[i] += g;
                }
            }
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.div(n).broadcast(n) };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(this), backwardFn, symFn);
        node.opTag = "mean";
        return node;
    }

    // ---- in-place operations ----

    @Override
    public IDiffVector addInPlace(IDiffVector other) {
        // Delegate to out-of-place add() to avoid graph corruption.
        // Mutating isLeaf in-place breaks Parameter leaf tracking.
        return this.add(other);
    }

    @Override
    public IDiffVector mulInPlace(double scalar) {
        if (!this.isLeaf) {
            throw new IllegalStateException("mulInPlace only allowed on leaf variables");
        }
        RereDiffVector self = this;
        this.value = this.value.multiplyByScalar(scalar);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGrad(gradOut.multiplyByScalar(scalar));
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut.mul(scalar) };
        RereDiffVector node = new RereDiffVector(this.value, List.of(this), backwardFn, symFn);
        this.isLeaf = false;
        this.inputs = List.of(this);
        this.backwardFn = node.backwardFn;
        this.symbolicBackwardFn = node.symbolicBackwardFn;
        return this;
    }

    // ---- vector operations ----

    @Override
    public IDiffVector dot(IDiffVector other) {
        RereDiffVector o = (RereDiffVector) other;
        double dotVal = this.value.dotValue(o.value);
        IDoubleVector resultVal = IDoubleVector.of(dotVal);
        IDoubleVector thisVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector otherVal = o.isLeaf ? o.value.copy() : o.value;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            double[] od = otherVal.getData();
            double[] td = thisVal.getData();
            int n = od.length;
            this.accGradScalarMulDirect(od, g, n);
            o.accGradScalarMulDirect(td, g, n);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] {
                other.mul(gradOut.broadcast(other.getValue().size())),
                this.mul(gradOut.broadcast(this.getValue().size())) };
        return withTag(new RereDiffVector(resultVal, List.of(this, o), backwardFn, symFn), "dot");
    }

    // ---- slice ----

    @Override
    public IDiffVector slice(int start, int end) {
        if (start < 0 || end > this.value.size()) {
            throw new IllegalArgumentException(
                "slice(" + start + ", " + end + ") out of bounds for size " + this.value.size());
        }
        IDoubleVector resultVal = this.value.slice(start, end);
        int len = end - start;
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = ((RereDoubleVector) gradOut).getData();
            int fullLen = self.value.size();
            double[] buf = AutodiffBufferPool.acquire(fullLen);
            System.arraycopy(gd, 0, buf, start, len);
            self.accGradFromPooled(buf, fullLen);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> {
            // Scatter: pad gradient with zeros outside the slice range
            double[] gd = gradOut.getData();
            double[] fullGrad = new double[self.value.size()];
            System.arraycopy(gd, 0, fullGrad, start, len);
            return new IDiffVector[] { (IDiffVector) IDoubleVector.of(fullGrad) };
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn, symFn), "slice");
    }

    // ---- cat ----

    @Override
    public IDiffVector cat(IDiffVector... others) {
        int n = 1 + others.length;
        IDiffVector[] all = new IDiffVector[n];
        all[0] = this;
        System.arraycopy(others, 0, all, 1, others.length);

        // 1. Build concatenated value
        int totalLen = 0;
        for (IDiffVector v : all) totalLen += v.getValue().size();
        double[] catData = new double[totalLen];
        int pos = 0;
        for (IDiffVector v : all) {
            double[] d = v.getValue().getData();
            System.arraycopy(d, 0, catData, pos, d.length);
            pos += d.length;
        }

        // 2. Collect inputs and save offsets for backward
        List<RereDiffVector> inputs = new ArrayList<>();
        for (IDiffVector v : all) inputs.add((RereDiffVector) v);

        int[] offsets = new int[n];
        int off = 0;
        for (int i = 0; i < n; i++) {
            offsets[i] = off;
            off += inputs.get(i).value.size();
        }

        // 3. Backward: split gradient by offset, accumulate into each input
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = gradOut.getData();
            for (int i = 0; i < n; i++) {
                int len = inputs.get(i).value.size();
                double[] buf = AutodiffBufferPool.acquire(len);
                System.arraycopy(gd, offsets[i], buf, 0, len);
                inputs.get(i).accGradFromPooled(buf, len);
            }
        };

        return withTag(new RereDiffVector(IDoubleVector.of(catData), inputs, backwardFn), "cat");
    }

    // ---- reshape ----

    @Override
    public IDiffMatrix reshape(int rows, int cols) {
        int origSize = this.value.size();
        if (rows * cols != origSize) {
            throw new IllegalArgumentException(
                "reshape dimensions " + rows + "x" + cols + " must match size " + origSize);
        }
        IDoubleMatrix resultVal = IDoubleMatrix.fromArray(this.value.getData().clone(), rows, cols);
        RereDiffVector self = this;
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            IDoubleVector flatGrad = (IDoubleVector) gradOut.flatten();
            self.accGrad(flatGrad);
            self.propagateGradient();
        };
        Function<IDiffVector, IDiffVector[]> symbolicBackwardFn = (matrixGrad) ->
            new IDiffVector[] { matrixGrad };
        // self (RereDiffVector) cannot be in List<RereDiffMatrix> inputs,
        // so propagateGradient() is needed to propagate through the vector graph.
        RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(), backwardFn);
        node.opTag = "reshape";
        node.symbolicBackwardFn = symbolicBackwardFn;
        return node;
    }

    // ======================================================================
    // Covariant overrides from IDoubleVector / IVector
    // ======================================================================

    @Override
    public IDiffVector copy() {
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGradDirect(gradOut.getData());
        };
        return withTag(new RereDiffVector(
            this.value.copy(), List.of(this), backwardFn), "copy");
    }

    @Override
    public IDiffVector divideInPlace(double alpha) {
        if (alpha == 1.0) return this;
        return this.mulInPlace(1.0 / alpha);
    }

    @Override
    public IDiffVector addScalarInPlace(double p) {
        if (p == 0.0) return this;
        double[] data = this.getValue().getData();
        for (int i = 0; i < data.length; i++) {
            data[i] += p;
        }
        return this;
    }

    @Override
    public IDiffVector subScalarInPlace(double p) {
        if (p == 0.0) return this;
        double[] data = this.getValue().getData();
        for (int i = 0; i < data.length; i++) {
            data[i] -= p;
        }
        return this;
    }

    @Override
    public IDiffVector multiplyByScalarInPlace(double p) {
        double[] data = this.getValue().getData();
        for (int i = 0; i < data.length; i++) {
            data[i] *= p;
        }
        return this;
    }

    @Override
    public IDiffVector addInPlace(IVector<Double> vec) {
        double[] data = this.getValue().getData();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) {
            data[i] += other[i];
        }
        return this;
    }

    @Override
    public IDiffVector subInPlace(IVector<Double> vec) {
        double[] data = this.getValue().getData();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) {
            data[i] -= other[i];
        }
        return this;
    }

    @Override
    public IDiffVector multiplyInPlace(IVector<Double> vec) {
        double[] data = this.getValue().getData();
        double[] other = (vec instanceof RereDoubleVector rdv) ? rdv.getData() : vec.toDoubleArray();
        for (int i = 0; i < data.length; i++) {
            data[i] *= other[i];
        }
        return this;
    }

    @Override
    public IDiffVector negInPlace() {
        double[] data = this.getValue().getData();
        for (int i = 0; i < data.length; i++) {
            data[i] = -data[i];
        }
        return this;
    }

    @Override
    public IDiffVector add(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) {
            return this.add(dv);
        }
        return this.add(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector sub(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) {
            return this.sub(dv);
        }
        return this.sub(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector multiply(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) {
            return this.mul(dv);
        }
        return this.mul(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector divide(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) {
            return this.div(dv);
        }
        return this.div(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector dot(IVector<Double> vec) {
        if (vec instanceof IDiffVector dv) {
            return this.dot(dv);
        }
        return this.dot(new RereDiffVector((IDoubleVector) vec.copy()));
    }

    @Override
    public IDiffVector innerProduct(IVector<Double> vec) {
        return this.dot(vec);
    }

    @Override
    public double dtw(IVector<Double> other) {
        return this.value.dtw(other);
    }

    @Override
    public double normInf() {
        return this.value.normInf();
    }

    // ---- IDoubleVector bridge methods with proper AD graph ----

    // ===== rounding: straight-through estimator (gradient = 1) =====

    @Override
    public IDiffVector round() {
        IDoubleVector resultVal = this.value.round();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGradDirect(gradOut.getData());
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "round");
    }

    @Override
    public IDiffVector floor() {
        IDoubleVector resultVal = this.value.floor();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGradDirect(gradOut.getData());
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "floor");
    }

    @Override
    public IDiffVector ceil() {
        IDoubleVector resultVal = this.value.ceil();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGradDirect(gradOut.getData());
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "ceil");
    }

    @Override
    public IDiffVector trunc() {
        IDoubleVector resultVal = this.value.trunc();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGradDirect(gradOut.getData());
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "trunc");
    }

    @Override
    public IDiffVector sign() {
        IDoubleVector resultVal = this.value.sign();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // sign is flat everywhere → zero gradient
            self.accGradDirect(new double[self.value.size()]);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "sign");
    }

    // ===== element-wise with standard derivative formula =====

    @Override
    public IDiffVector log10() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.log10();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // dx = gradOut / (fwd * ln10) — uses SIMD/HPC/GPU through IDoubleVector ops
            self.accGrad((IDoubleVector) gradOut.divide(fwdVal.multiplyByScalar(Math.log(10))));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "log10");
    }

    @Override
    public IDiffVector reciprocal() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.reciprocal();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // dx = -gradOut * fwd^2 — uses SIMD/HPC/GPU through IDoubleVector ops
            self.accGrad((IDoubleVector) gradOut.multiply(fwdVal.square().neg()));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "reciprocal");
    }

    // ===== inverse trigonometric and hyperbolic =====

    @Override
    public IDiffVector arcsin() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.arcsin();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // dx = gradOut / sqrt(1 - fwd^2) — uses SIMD/HPC/GPU through IDoubleVector ops
            IDoubleVector denom = (IDoubleVector) fwdVal.square().neg().addScalar(1.0).sqrt();
            self.accGrad((IDoubleVector) gradOut.divide(denom));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "arcsin");
    }

    @Override
    public IDiffVector arccos() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.arccos();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // dx = -gradOut / sqrt(1 - fwd^2) — uses SIMD/HPC/GPU through IDoubleVector ops
            IDoubleVector denom = (IDoubleVector) fwdVal.square().neg().addScalar(1.0).sqrt();
            self.accGrad((IDoubleVector) gradOut.divide(denom).neg());
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "arccos");
    }

    @Override
    public IDiffVector arctan() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.arctan();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // dx = gradOut / (1 + fwd^2) — uses SIMD/HPC/GPU through IDoubleVector ops
            self.accGrad((IDoubleVector) gradOut.divide(fwdVal.square().addScalar(1.0)));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "arctan");
    }

    @Override
    public IDiffVector sinh() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.sinh();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // dx = gradOut * cosh(fwd) — uses SIMD/HPC/GPU through IDoubleVector ops
            self.accGrad((IDoubleVector) gradOut.multiply(fwdVal.cosh()));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "sinh");
    }

    @Override
    public IDiffVector cosh() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.cosh();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            // dx = gradOut * sinh(fwd) — uses SIMD/HPC/GPU through IDoubleVector ops
            self.accGrad((IDoubleVector) gradOut.multiply(fwdVal.sinh()));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "cosh");
    }

    // ===== remainder: straight-through estimator =====

    @Override
    public IDiffVector remainder(Double value) {
        IDoubleVector resultVal = (IDoubleVector) this.value.remainder(value);
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGradDirect(gradOut.getData());
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "remainder", value);
    }

    // ===== reverse: simple permute backward =====

    @Override
    public IDiffVector reverse() {
        int n = this.value.size();
        IDoubleVector resultVal = this.value.reverse();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) {
                dx[n - 1 - i] = go[i];
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "reverse");
    }

    // ===== tile / repeat: backward reduces over repeated axis =====

    @Override
    public IDiffVector tile(int reps) {
        int n = this.value.size();
        IDoubleVector resultVal = this.value.tile(reps);
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[n];
            for (int i = 0; i < go.length; i++) {
                dx[i % n] += go[i];
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "tile", reps);
    }

    @Override
    public IDiffVector repeat(int repeats) {
        int n = this.value.size();
        IDoubleVector resultVal = this.value.repeat(repeats);
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[n];
            for (int i = 0; i < go.length; i++) {
                dx[i / repeats] += go[i];
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "repeat", repeats);
    }

    // ===== indexed slicing with step =====

    @Override
    public IDiffVector slice(int start, int end, int step) {
        IDoubleVector resultVal = this.value.slice(start, end, step);
        int len = resultVal.size();
        int fullLen = this.value.size();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = gradOut.getData();
            double[] buf = AutodiffBufferPool.acquire(fullLen);
            for (int i = 0, pos = start; pos < end && i < len; i++, pos += step) {
                buf[pos] = gd[i];
            }
            self.accGradFromPooled(buf, fullLen);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "slice");
    }

    @Override
    public IDiffVector slice(String sliceExpression) {
        // Parse "start:end:step" or "start:end" format
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

    // ===== fancy indexing / boolean masking with scatter-add backward =====

    @Override
    public IDiffVector fancyGet(int[] positions) {
        IDoubleVector resultVal = this.value.fancyGet(positions);
        int selfLen = this.value.size();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = gradOut.getData();
            double[] dx = new double[selfLen];
            for (int i = 0; i < positions.length; i++) {
                dx[positions[i]] += gd[i];
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "fancyGet");
    }

    @Override
    public IDiffVector booleanGet(boolean[] booleanIndex) {
        IDoubleVector resultVal = this.value.booleanGet(booleanIndex);
        int selfLen = this.value.size();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = gradOut.getData();
            double[] dx = new double[selfLen];
            for (int i = 0, outIdx = 0; i < selfLen; i++) {
                if (booleanIndex[i]) {
                    dx[i] += gd[outIdx++];
                }
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "booleanGet");
    }

    // ===== matrix operations with non-differentiable other operand =====

    @Override
    public IDiffVector mmul(IMatrix<Double> other) {
        IDoubleVector resultVal = this.value.mmul(other);
        int rows = other.rows();
        int cols = other.cols();
        double[][] matData = other.toDoubleArray();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[cols];
            for (int j = 0; j < cols; j++) {
                double sum = 0;
                for (int i = 0; i < rows; i++) {
                    sum += go[i] * matData[i][j];
                }
                dx[j] = sum;
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "mmul");
    }

    @Override
    public IDiffVector dot(IMatrix<Double> m) {
        IDoubleVector resultVal = this.value.dot(m);
        int rows = m.rows();
        int cols = m.cols();
        double[][] matData = m.toDoubleArray();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[cols];
            for (int j = 0; j < cols; j++) {
                double sum = 0;
                for (int i = 0; i < rows; i++) {
                    sum += go[i] * matData[i][j];
                }
                dx[j] = sum;
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "dotMat");
    }

    @Override
    public IDiffVector cross(IVector<Double> other) {
        IDoubleVector resultVal = this.value.cross(other);
        double[] oData = other.toDoubleArray();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            // d/da (a × b) = go × b = [go1*b2 - go2*b1, go2*b0 - go0*b2, go0*b1 - go1*b0]
            double[] dx = new double[]{
                go[1] * oData[2] - go[2] * oData[1],
                go[2] * oData[0] - go[0] * oData[2],
                go[0] * oData[1] - go[1] * oData[0]
            };
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "cross");
    }

    // ===== where: self is a template, values come from condition-selected sources =====

    @Override
    public IDiffVector where(boolean[] condition, Double x, Double y) {
        IDoubleVector resultVal = this.value.where(condition, x, y);
        // self does not contribute values → zero gradient to self
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGradDirect(new double[self.value.size()]);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "where");
    }

    @Override
    public IDiffVector where(boolean[] condition, IVector<Double> x, IVector<Double> y) {
        IDoubleVector resultVal = this.value.where(condition, x, y);
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGradDirect(new double[self.value.size()]);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "where");
    }

    // ===== cumulative operations =====

    @Override
    public IDiffVector cumsum() {
        int n = this.value.size();
        IDoubleVector resultVal = this.value.cumsum();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            // backward of cumsum: reverse cumsum of reversed gradient
            double[] dx = new double[n];
            double running = 0;
            for (int i = n - 1; i >= 0; i--) {
                running += go[i];
                dx[i] = running;
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "cumsum");
    }

    @Override
    public IDiffVector cumprod() {
        int n = this.value.size();
        double[] fwd = this.value.getData();
        IDoubleVector resultVal = this.value.cumprod();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[n];
            // d(cumprod)/dx_j: for k >= j, product_{i!=j, i<=k} x_i
            // Efficient: compute cumulative products and reversed cumulative products
            double[] cumProd = new double[n];
            cumProd[0] = fwd[0];
            for (int i = 1; i < n; i++) cumProd[i] = cumProd[i - 1] * fwd[i];
            double[] revCumProd = new double[n];
            revCumProd[n - 1] = fwd[n - 1];
            for (int i = n - 2; i >= 0; i--) revCumProd[i] = revCumProd[i + 1] * fwd[i];
            for (int j = 0; j < n; j++) {
                for (int k = j; k < n; k++) {
                    double prodExcludingJ = 1.0;
                    if (j > 0) prodExcludingJ *= cumProd[j - 1];
                    if (k > 0) prodExcludingJ *= revCumProd[j + 1 < n ? j + 1 : 0];
                    // Actually: product_{i<=k, i!=j} fwd[i] = cumProd[k] / fwd[j]
                    if (Math.abs(fwd[j]) > 1e-15) {
                        dx[j] += go[k] * cumProd[k] / fwd[j];
                    }
                }
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "cumprod");
    }

    @Override
    public IDiffVector diff() {
        int n = this.value.size();
        IDoubleVector resultVal = this.value.diff();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[n];
            // d(diff(x))_i = x_{i+1} - x_i
            // ∂result_k / ∂x_i: k=i → -1, k=i-1 → +1
            dx[0] = -go[0];
            for (int i = 1; i < n - 1; i++) {
                dx[i] = go[i - 1] - go[i];
            }
            dx[n - 1] = go[n - 2];
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "diff");
    }

    @Override
    public IDiffVector diff(int order) {
        if (order == 1) return diff();
        IDiffVector result = this;
        for (int i = 0; i < order; i++) {
            result = result.diff();
        }
        return result;
    }

    // ===== sort: gradient flows through permutation (reverse of sort indices) =====

    @Override
    public IDiffVector sort() {
        double[] fwd = this.value.getData();
        int n = fwd.length;
        // Compute sort permutation (argsort)
        Integer[] indices = new Integer[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        java.util.Arrays.sort(indices, (a, b) -> Double.compare(fwd[a], fwd[b]));
        int[] perm = new int[n];
        for (int i = 0; i < n; i++) perm[i] = indices[i];

        IDoubleVector resultVal = this.value.sort();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[n];
            for (int i = 0; i < n; i++) {
                dx[perm[i]] = go[i];
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "sort");
    }

    // ===== concat with non-differentiable operand (deprecated; use cat()) =====

    @Override
    @Deprecated
    public IDiffVector concat(IVector<Double> other) {
        IDoubleVector resultVal = this.value.concat(other);
        int selfLen = this.value.size();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = gradOut.getData();
            double[] dx = new double[selfLen];
            System.arraycopy(gd, 0, dx, 0, selfLen);
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "concat");
    }

    // ===== statistics (return 1-element vector with gradient) =====

    @Override
    public IDiffVector min() {
        double[] fwd = this.value.getData();
        int n = fwd.length;
        int minIdxVal = 0;
        for (int i = 1; i < n; i++) if (fwd[i] < fwd[minIdxVal]) minIdxVal = i;
        final int minIdx = minIdxVal;
        IDoubleVector resultVal = this.value.min();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            double[] dx = new double[n];
            dx[minIdx] = g;
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "min");
    }

    @Override
    public IDiffVector max() {
        double[] fwd = this.value.getData();
        int n = fwd.length;
        int maxIdxVal = 0;
        for (int i = 1; i < n; i++) if (fwd[i] > fwd[maxIdxVal]) maxIdxVal = i;
        final int maxIdx = maxIdxVal;
        IDoubleVector resultVal = this.value.max();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            double[] dx = new double[n];
            dx[maxIdx] = g;
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "max");
    }

    @Override
    public IDiffVector prod() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        double[] fwd = fwdVal.getData();
        int n = fwd.length;
        double totalProd = 1.0;
        for (int i = 0; i < n; i++) totalProd *= fwd[i];
        final double tp = totalProd;
        IDoubleVector resultVal = this.value.prod();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            self.accGrad(fwdVal.reciprocal().multiplyByScalar(g * tp));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "prod");
    }

    @Override
    public IDiffVector norm2() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.norm2();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            double norm = resultVal.get(0);
            if (norm > 1e-15) {
                self.accGrad(fwdVal.multiplyByScalar(g / norm));
            } else {
                self.accGrad(fwdVal.multiplyByScalar(0));
            }
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "norm2");
    }

    @Override
    public IDiffVector norm1() {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        IDoubleVector resultVal = this.value.norm1();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            self.accGrad(fwdVal.sign().multiplyByScalar(g));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "norm1");
    }

    @Override
    public IDiffVector ptp() {
        double[] fwd = this.value.getData();
        int n = fwd.length;
        int maxIdxVal = 0, minIdxVal = 0;
        for (int i = 1; i < n; i++) {
            if (fwd[i] > fwd[maxIdxVal]) maxIdxVal = i;
            if (fwd[i] < fwd[minIdxVal]) minIdxVal = i;
        }
        final int maxIdx = maxIdxVal;
        final int minIdx = minIdxVal;
        IDoubleVector resultVal = this.value.ptp();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            double[] dx = new double[n];
            dx[maxIdx] = g;
            dx[minIdx] = -g;
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "ptp");
    }

    @Override
    public IDiffVector normalize() {
        double[] fwd = this.value.getData();
        int n = fwd.length;
        double normSq = 0;
        for (int i = 0; i < n; i++) normSq += fwd[i] * fwd[i];
        final double norm = Math.sqrt(normSq);
        final double invNorm = norm > 1e-15 ? 1.0 / norm : 0;
        final double invNorm3 = invNorm * invNorm * invNorm;
        IDoubleVector resultVal = this.value.normalize();
        RereDiffVector self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] go = gradOut.getData();
            double[] dx = new double[n];
            double dot = 0;
            for (int i = 0; i < n; i++) dot += go[i] * fwd[i];
            for (int i = 0; i < n; i++) {
                dx[i] = go[i] * invNorm - dot * fwd[i] * invNorm3;
            }
            self.accGradDirect(dx);
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "normalize");
    }

    @Override
    public IDiffVector std() {
        return std(0);
    }

    @Override
    public IDiffVector std(int ddof) {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        double[] fwd = fwdVal.getData();
        int n = fwd.length;
        double mean = 0;
        for (int i = 0; i < n; i++) mean += fwd[i];
        mean /= n;
        double var = 0;
        for (int i = 0; i < n; i++) var += (fwd[i] - mean) * (fwd[i] - mean);
        double divisor = n - ddof;
        double stdev = Math.sqrt(var / divisor);
        IDoubleVector resultVal = this.value.std(ddof);
        RereDiffVector self = this;
        double m = mean;
        double s = stdev;
        double d = divisor;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            if (s > 1e-15) {
                self.accGrad(fwdVal.addScalar(-m).multiplyByScalar(g / (d * s)));
            } else {
                self.accGrad(fwdVal.multiplyByScalar(0));
            }
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "std");
    }

    @Override
    public IDiffVector var() {
        return var(0);
    }

    @Override
    public IDiffVector var(int ddof) {
        IDoubleVector fwdVal = this.isLeaf ? this.value.copy() : this.value;
        double[] fwd = fwdVal.getData();
        int n = fwd.length;
        double mean = 0;
        for (int i = 0; i < n; i++) mean += fwd[i];
        mean /= n;
        IDoubleVector resultVal = this.value.var(ddof);
        RereDiffVector self = this;
        double m = mean;
        double divisor = n - ddof;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            self.accGrad(fwdVal.addScalar(-m).multiplyByScalar(2.0 * g / divisor));
        };
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "var");
    }

    // ===== non-differentiable delegates =====

    @Override
    public IDiffVector map(Function<Double, Double> fun) {
        throw new UnsupportedOperationException(
            "map() with arbitrary function cannot be differentiated. Use specific operations instead.");
    }

    @Override
    public IDoubleMatrix asColumnVector() {
        return this.getValue().asColumnVector();
    }

    @Override
    public IDoubleMatrix hessianMatrix() {
        return (IDoubleMatrix) this.getValue().hessianMatrix();
    }

    @Override
    public double cov(IVector<Double> other) {
        return this.getValue().cov(other);
    }

    @Override
    public double corr(IVector<Double> other) {
        return this.getValue().corr(other);
    }

    void propagateGradient() {
        if (this.gradient == null || this.backwardFn == null) return;
        // Use local lists to avoid corrupting the outer backward() loop's ThreadLocal
        ArrayList<RereDiffVector> order = new ArrayList<>();
        HashSet<RereDiffVector> visited = new HashSet<>();
        buildTopo(order, visited);
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffVector v = order.get(i);
            if (v.gradient != null && v.backwardFn != null) {
                IDoubleVector savedGrad = v.gradient;
                v.gradient = null;
                v.backwardFn.accept(savedGrad);
            }
        }
    }
}
