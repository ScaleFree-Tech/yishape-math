package com.yishape.lab.math.autodiff.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
    private void accGradFromPooled(double[] pooledBuf, int n) {
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
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = expVals[i] / sumExp;
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
        double[] y = new double[n];
        double[] sm = new double[n];
        for (int i = 0; i < n; i++) {
            sm[i] = expVals[i] / sumExp;
            y[i] = xd[i] - logSumExp;
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
        double[] mask = new double[n];
        double[] y = new double[n];
        double[] xd = xVal.getData();
        for (int i = 0; i < n; i++) {
            mask[i] = Math.random() > p ? scale : 0.0;
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
        return withTag(new RereDiffVector(resultVal, List.of(this), backwardFn), "dropout", p);
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
        if (!this.isLeaf) {
            throw new IllegalStateException("addInPlace only allowed on leaf variables");
        }
        RereDiffVector o = (RereDiffVector) other;
        RereDiffVector self = this;
        IDoubleVector oldVal = this.value.copy();
        IDoubleVector otherVal = o.value.copy();
        this.value = this.value.add(o.value);
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGrad(gradOut);
            o.accGrad(gradOut);
        };
        Function<IDiffVector, IDiffVector[]> symFn = (gradOut) -> new IDiffVector[] { gradOut, gradOut };
        RereDiffVector node = new RereDiffVector(this.value, List.of(this, o), backwardFn, symFn);
        this.isLeaf = false;
        this.inputs = List.of(this, o);
        this.backwardFn = node.backwardFn;
        this.symbolicBackwardFn = node.symbolicBackwardFn;
        return this;
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
        return RereDiffVector.constant(this.value.copy());
    }

    @Override
    public IDiffVector divideInPlace(double alpha) {
        if (alpha == 1.0) return this;
        return this.mulInPlace(1.0 / alpha);
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

    // ---- IDoubleVector bridge methods (delegated to getValue(), no AD graph) ----

    @Override
    public IDiffVector log10() {
        return new RereDiffVector(this.getValue().log10().copy());
    }

    @Override
    public IDiffVector slice(int start, int end, int step) {
        return new RereDiffVector(this.getValue().slice(start, end, step).copy());
    }

    @Override
    public IDiffVector slice(String sliceExpression) {
        return new RereDiffVector(this.getValue().slice(sliceExpression).copy());
    }

    @Override
    public IDiffVector fancyGet(int[] positions) {
        return new RereDiffVector(this.getValue().fancyGet(positions).copy());
    }

    @Override
    public IDiffVector booleanGet(boolean[] booleanIndex) {
        return new RereDiffVector(this.getValue().booleanGet(booleanIndex).copy());
    }

    @Override
    public IDiffVector mmul(IMatrix<Double> other) {
        return new RereDiffVector(this.getValue().mmul(other).copy());
    }

    @Override
    public IDiffVector dot(IMatrix<Double> m) {
        return new RereDiffVector(this.getValue().dot(m).copy());
    }

    @Override
    public IDoubleMatrix asColumnVector() {
        return this.getValue().asColumnVector();
    }

    @Override
    public IDiffVector arcsin() {
        return new RereDiffVector(this.getValue().arcsin().copy());
    }

    @Override
    public IDiffVector arccos() {
        return new RereDiffVector(this.getValue().arccos().copy());
    }

    @Override
    public IDiffVector arctan() {
        return new RereDiffVector(this.getValue().arctan().copy());
    }

    @Override
    public IDiffVector sinh() {
        return new RereDiffVector(this.getValue().sinh().copy());
    }

    @Override
    public IDiffVector cosh() {
        return new RereDiffVector(this.getValue().cosh().copy());
    }

    @Override
    public IDiffVector round() {
        return new RereDiffVector(this.getValue().round().copy());
    }

    @Override
    public IDiffVector floor() {
        return new RereDiffVector(this.getValue().floor().copy());
    }

    @Override
    public IDiffVector ceil() {
        return new RereDiffVector(this.getValue().ceil().copy());
    }

    @Override
    public IDiffVector trunc() {
        return new RereDiffVector(this.getValue().trunc().copy());
    }

    @Override
    public IDiffVector cumsum() {
        return new RereDiffVector(this.getValue().cumsum().copy());
    }

    @Override
    public IDiffVector cumprod() {
        return new RereDiffVector(this.getValue().cumprod().copy());
    }

    @Override
    public IDiffVector diff() {
        return new RereDiffVector(this.getValue().diff().copy());
    }

    @Override
    public IDiffVector diff(int n) {
        return new RereDiffVector(this.getValue().diff(n).copy());
    }

    @Override
    public IDiffVector sort() {
        return new RereDiffVector(this.getValue().sort().copy());
    }

    @Override
    public IDiffVector reverse() {
        return new RereDiffVector(this.getValue().reverse().copy());
    }

    @Override
    public IDiffVector normalize() {
        return new RereDiffVector(this.getValue().normalize().copy());
    }

    @Override
    public IDiffVector reciprocal() {
        return new RereDiffVector(this.getValue().reciprocal().copy());
    }

    @Override
    public IDiffVector cross(IVector<Double> other) {
        return new RereDiffVector(this.getValue().cross(other).copy());
    }

    @Override
    public IDiffVector where(boolean[] condition, Double x, Double y) {
        return new RereDiffVector(this.getValue().where(condition, x, y).copy());
    }

    @Override
    public IDiffVector where(boolean[] condition, IVector<Double> x, IVector<Double> y) {
        return new RereDiffVector(this.getValue().where(condition, x, y).copy());
    }

    @Override
    public IDiffVector repeat(int repeats) {
        return new RereDiffVector(this.getValue().repeat(repeats).copy());
    }

    @Override
    public IDiffVector tile(int reps) {
        return new RereDiffVector(this.getValue().tile(reps).copy());
    }

    @Override
    public IDiffVector map(Function<Double, Double> fun) {
        return new RereDiffVector(this.getValue().map(fun).copy());
    }

    @Override
    public IDiffVector concat(IVector<Double> other) {
        return new RereDiffVector(this.getValue().concat(other).copy());
    }

    @Override
    public IDiffVector sign() {
        return new RereDiffVector(this.getValue().sign().copy());
    }

    @Override
    public IDiffVector min() {
        return new RereDiffVector(this.getValue().min().copy());
    }

    @Override
    public IDiffVector max() {
        return new RereDiffVector(this.getValue().max().copy());
    }

    @Override
    public IDiffVector std() {
        return new RereDiffVector(this.getValue().std().copy());
    }

    @Override
    public IDiffVector std(int ddof) {
        return new RereDiffVector(this.getValue().std(ddof).copy());
    }

    @Override
    public IDiffVector var() {
        return new RereDiffVector(this.getValue().var().copy());
    }

    @Override
    public IDiffVector var(int ddof) {
        return new RereDiffVector(this.getValue().var(ddof).copy());
    }

    @Override
    public IDiffVector prod() {
        return new RereDiffVector(this.getValue().prod().copy());
    }

    @Override
    public IDiffVector norm2() {
        return new RereDiffVector(this.getValue().norm2().copy());
    }

    @Override
    public IDiffVector norm1() {
        return new RereDiffVector(this.getValue().norm1().copy());
    }

    @Override
    public IDiffVector ptp() {
        return new RereDiffVector(this.getValue().ptp().copy());
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

    @Override
    public IDiffVector remainder(Double value) {
        return new RereDiffVector(((IDoubleVector) this.getValue().remainder(value)).copy());
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
