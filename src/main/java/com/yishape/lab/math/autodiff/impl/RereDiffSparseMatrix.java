package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffSparseMatrix;

/**
 * Reverse-mode AD for sparse matrices.
 * 稀疏矩阵的反向模式自动微分实现。
 */
public class RereDiffSparseMatrix implements IDiffSparseMatrix {

    private static final long serialVersionUID = 1L;

    private static final ThreadLocal<ArrayList<RereDiffSparseMatrix>> TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffSparseMatrix>> TOPO_SET =
        ThreadLocal.withInitial(HashSet::new);

    ISparseMatrix value;
    ISparseMatrix gradient;
    boolean isLeaf;
    List<RereDiffSparseMatrix> inputs;
    Consumer<ISparseMatrix> backwardFn;
    Function<IDiffSparseMatrix, IDiffSparseMatrix[]> symbolicBackwardFn;
    String opTag;

    public RereDiffSparseMatrix(ISparseMatrix value) {
        this.value = value;
        this.isLeaf = true;
        this.inputs = new ArrayList<>();
    }

    RereDiffSparseMatrix(ISparseMatrix value, List<RereDiffSparseMatrix> inputs,
            Consumer<ISparseMatrix> backwardFn) {
        this.value = value;
        this.isLeaf = false;
        this.inputs = inputs;
        this.backwardFn = backwardFn;
    }

    @Override public ISparseMatrix getValue() { return value; }
    @Override public ISparseMatrix getGradient() { return gradient; }
    @Override public boolean isLeaf() { return isLeaf; }

    @Override
    public void backward() {
        backward(ISparseMatrix.ones(value.rows(), value.cols()));
    }

    @Override
    public void backward(ISparseMatrix initialGradient) {
        this.gradient = initialGradient;
        ArrayList<RereDiffSparseMatrix> order = TOPO_LIST.get();
        order.clear();
        HashSet<RereDiffSparseMatrix> visited = TOPO_SET.get();
        visited.clear();
        buildTopo(order, visited);
        try {
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffSparseMatrix v = order.get(i);
                if (v != this) {
                    v.gradient = null;
                }
            }
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffSparseMatrix v = order.get(i);
                if (v.gradient != null && v.backwardFn != null) {
                    v.backwardFn.accept(v.gradient);
                }
            }
        } finally {
            order.clear();
            visited.clear();
        }
    }

    /** Iterative DFS post-order topological sort. / 迭代式深度优先后序拓扑排序，避免深层图 StackOverflow。 */
    private void buildTopo(List<RereDiffSparseMatrix> order, Set<RereDiffSparseMatrix> visited) {
        java.util.ArrayDeque<Object[]> stack = new java.util.ArrayDeque<>();
        stack.push(new Object[]{this, Boolean.TRUE});
        while (!stack.isEmpty()) {
            Object[] entry = stack.peek();
            RereDiffSparseMatrix node = (RereDiffSparseMatrix) entry[0];
            boolean childrenNotPushed = (Boolean) entry[1];
            if (childrenNotPushed) {
                entry[1] = Boolean.FALSE;
                if (!visited.add(node)) {
                    stack.pop();
                    continue;
                }
                // Guard against null inputs (graph may have been released after backward)
                var inpList = node.inputs;
                if (inpList != null) {
                    for (int i = inpList.size() - 1; i >= 0; i--) {
                        RereDiffSparseMatrix inp = inpList.get(i);
                        if (!visited.contains(inp)) {
                            stack.push(new Object[]{inp, Boolean.TRUE});
                        }
                    }
                }
            } else {
                stack.pop();
                order.add(node);
            }
        }
    }

    @Override public void zeroGradient() { this.gradient = null; }

    // ==================== Sparse → Dense bridge ====================

    /**
     * Converts this sparse differentiable node to a dense {@link IDiffTensor}.
     *
     * <p>The returned tensor wraps the sparse data as a flat 1-D tensor
     * (row-major). When {@code backward()} is called on the dense graph,
     * the gradient is mapped back to this sparse matrix's gradient storage
     * and propagated through the sparse graph via {@link #propagateGradient()}.</p>
     *
     * <p>This is the <b>primary bridge</b> for mixed sparse↔dense computation:
     * any sparse subgraph that needs to interact with dense operations
     * (e.g. {@code IDiffTensor.add()}, {@code IDiffTensor.mul()}) should
     * call this method first, then operate on the returned dense tensor.</p>
     *
     * <h4>Usage</h4>
     * <pre>{@code
     * IDiffSparseMatrix sparse = ...;
     * IDiffTensor dense = sparse.asDenseDiffTensor();
     * IDiffTensor result = dense.add(otherDense).relu(); // all dense ops
     * result.backward(); // gradient flows back to sparse
     * }</pre>
     */
    public IDiffTensor asDenseDiffTensor() {
        int rows = value.rows();
        int cols = value.cols();
        double[][] dense = value.toDenseArray();
        double[] flat = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(dense[i], 0, flat, i * cols, cols);

        RereDiffSparseMatrix self = this;
        Consumer<RereDiffTensor> bw = t -> {
            double[] gradData = t.gradData();
            if (gradData == null) return;
            double[][] sparseGrad = new double[rows][cols];
            for (int i = 0; i < rows; i++)
                System.arraycopy(gradData, i * cols, sparseGrad[i], 0, cols);
            self.accGrad(ISparseMatrix.fromDense(sparseGrad));
            self.propagateGradient();
        };
        return new RereDiffTensor(flat, new int[]{rows, cols}, new ArrayList<>(), bw, "sparseBridge");
    }

    /**
     * Propagate this sparse matrix's gradient to its upstream inputs.
     * Called when gradient is accumulated from a vector-graph node
     * (e.g. sum(), mean(), matmul()) to bridge the sparse matrix graph
     * into the vector graph traversal.
     */
    void propagateGradient() {
        if (this.gradient == null || this.backwardFn == null) return;
        ArrayList<RereDiffSparseMatrix> order = new ArrayList<>();
        HashSet<RereDiffSparseMatrix> visited = new HashSet<>();
        buildTopo(order, visited);
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffSparseMatrix v = order.get(i);
            if (v.gradient != null && v.backwardFn != null) {
                ISparseMatrix savedGrad = v.gradient;
                v.gradient = null;
                v.backwardFn.accept(savedGrad);
            }
        }
    }

    @Override
    public IDiffSparseMatrix grad() {
        return new RereDiffSparseMatrix(this.gradient.copy());
    }

    @Override
    public IDiffVector sum() {
        double[][] dense = value.toDenseArray();
        double s = 0;
        int r = value.rows();
        int c = value.cols();
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                s += dense[i][j];
        IDoubleVector resultVal = IDoubleVector.of(s);
        RereDiffSparseMatrix self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            double[][] grad = new double[r][c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    if (Math.abs(dense[i][j]) > 1e-15)
                        grad[i][j] = g;
            self.accGrad(ISparseMatrix.fromDense(grad));
            self.propagateGradient();
        };
        RereDiffVector node = RereDiffVector.createNonLeaf(resultVal.getData(), List.of(), backwardFn);
        node.tensor.setOpTag("sum");
        return node;
    }

    @Override
    public IDiffVector mean() {
        double[][] dense = value.toDenseArray();
        double s = 0;
        int r = value.rows();
        int c = value.cols();
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                s += dense[i][j];
        double m = s / (r * c);
        IDoubleVector resultVal = IDoubleVector.of(m);
        int totalElements = r * c;
        RereDiffSparseMatrix self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0) / totalElements;
            double[][] grad = new double[r][c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    if (Math.abs(dense[i][j]) > 1e-15)
                        grad[i][j] = g;
            self.accGrad(ISparseMatrix.fromDense(grad));
            self.propagateGradient();
        };
        RereDiffVector node = RereDiffVector.createNonLeaf(resultVal.getData(), List.of(), backwardFn);
        node.tensor.setOpTag("mean");
        return node;
    }

    void accGrad(ISparseMatrix grad) {
        if (gradient == null) gradient = grad.copy();
        else gradient = gradient.add(grad);
    }

    @Override
    public IDiffVector matmul(IDiffVector vector) {
        RereDiffVector v = (RereDiffVector) vector;
        IVector result = value.multiply(v.getValue());
        ISparseMatrix aVal = value.copy();
        IDoubleVector xVal = v.getValue().copy();
        RereDiffSparseMatrix self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] gd = gradOut.getData();
            double[] xd = xVal.getData();
            int rows = aVal.rows();
            int cols = aVal.cols();
            double[][] outerData = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    outerData[i][j] = gd[i] * xd[j];
                }
            }
            self.accGrad(ISparseMatrix.fromDense(outerData));
            self.propagateGradient();
            v.accGrad((IDoubleVector) aVal.transpose().multiply(gradOut));
        };
        RereDiffVector node = RereDiffVector.createNonLeaf(((IDoubleVector) result).getData(), List.of(), backwardFn);
        node.tensor.setOpTag("matmul");
        return node;
    }

    @Override
    public IDiffSparseMatrix add(IDiffSparseMatrix other) {
        RereDiffSparseMatrix o = (RereDiffSparseMatrix) other;
        ISparseMatrix resultVal = value.add(o.value);
        Consumer<ISparseMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
            o.accGrad(gradOut);
        };
        Function<IDiffSparseMatrix, IDiffSparseMatrix[]> symFn = (gradOut) ->
            new IDiffSparseMatrix[] { gradOut, gradOut };
        RereDiffSparseMatrix node = new RereDiffSparseMatrix(resultVal, List.of(this, o), backwardFn);
        node.symbolicBackwardFn = symFn;
        node.opTag = "add";
        return node;
    }

    @Override
    public IDiffSparseMatrix sub(IDiffSparseMatrix other) {
        RereDiffSparseMatrix o = (RereDiffSparseMatrix) other;
        ISparseMatrix resultVal = value.sub(o.value);
        Consumer<ISparseMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
            o.accGrad(gradOut.scale(-1.0));
        };
        Function<IDiffSparseMatrix, IDiffSparseMatrix[]> symFn = (gradOut) ->
            new IDiffSparseMatrix[] { gradOut, gradOut.mul(-1.0) };
        RereDiffSparseMatrix node = new RereDiffSparseMatrix(resultVal, List.of(this, o), backwardFn);
        node.symbolicBackwardFn = symFn;
        node.opTag = "sub";
        return node;
    }

    @Override
    public IDiffSparseMatrix mul(double scalar) {
        ISparseMatrix resultVal = value.scale(scalar);
        Consumer<ISparseMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.scale(scalar));
        };
        Function<IDiffSparseMatrix, IDiffSparseMatrix[]> symFn = (gradOut) ->
            new IDiffSparseMatrix[] { gradOut.mul(scalar) };
        RereDiffSparseMatrix node = new RereDiffSparseMatrix(resultVal, List.of(this), backwardFn);
        node.symbolicBackwardFn = symFn;
        node.opTag = "mulScalar";
        return node;
    }

    @Override
    public IDiffSparseMatrix div(double scalar) {
        double inv = 1.0 / scalar;
        return mul(inv);
    }

    @Override
    public IDiffSparseMatrix elementwiseMul(IDiffSparseMatrix other) {
        RereDiffSparseMatrix o = (RereDiffSparseMatrix) other;
        ISparseMatrix aVal = value.copy();
        ISparseMatrix bVal = o.value.copy();
        ISparseMatrix resultVal = aVal.hadamard(bVal);
        Consumer<ISparseMatrix> backwardFn = (gradOut) -> {
            // dA = gradOut * B, dB = gradOut * A
            this.accGrad(gradOut.hadamard(bVal));
            o.accGrad(gradOut.hadamard(aVal));
        };
        RereDiffSparseMatrix node = new RereDiffSparseMatrix(resultVal, List.of(this, o), backwardFn);
        node.opTag = "elementwiseMul";
        return node;
    }

    @Override
    public IDiffSparseMatrix negate() {
        return mul(-1.0);
    }

    @Override
    public IDiffSparseMatrix transpose() {
        ISparseMatrix resultVal = value.transpose();
        Consumer<ISparseMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.transpose());
        };
        Function<IDiffSparseMatrix, IDiffSparseMatrix[]> symFn = (gradOut) ->
            new IDiffSparseMatrix[] { gradOut.transpose() };
        RereDiffSparseMatrix node = new RereDiffSparseMatrix(resultVal, List.of(this), backwardFn);
        node.symbolicBackwardFn = symFn;
        node.opTag = "transpose";
        return node;
    }

    // ==================== Activation functions ====================

    @Override
    public IDiffSparseMatrix relu() {
        return applyUnarySparse("relu", x -> Math.max(0.0, x), x -> x > 0.0 ? 1.0 : 0.0);
    }

    @Override
    public IDiffSparseMatrix sigmoid() {
        return applyUnarySparse("sigmoid",
            x -> 1.0 / (1.0 + Math.exp(-x)),
            x -> { double s = 1.0 / (1.0 + Math.exp(-x)); return s * (1.0 - s); });
    }

    @Override
    public IDiffSparseMatrix tanh() {
        return applyUnarySparse("tanh",
            x -> Math.tanh(x),
            x -> { double t = Math.tanh(x); return 1.0 - t * t; });
    }

    @Override
    public IDiffSparseMatrix abs() {
        return applyUnarySparse("abs",
            x -> Math.abs(x),
            x -> x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0));
    }

    /**
     * Generic unary operation on sparse matrix. Processes only non-zero elements
     * (zero elements map to f(0) which may be non-zero for some ops like sigmoid).
     * For sparsity-preserving ops (relu), zero stays zero.
     */
    private IDiffSparseMatrix applyUnarySparse(String tag,
            java.util.function.DoubleUnaryOperator forward,
            java.util.function.DoubleUnaryOperator backwardGrad) {
        double[][] dense = value.toDenseArray();
        int r = value.rows(), c = value.cols();
        double[][] resultDense = new double[r][c];
        for (int i = 0; i < r; i++)
            for (int j = 0; j < c; j++)
                resultDense[i][j] = forward.applyAsDouble(dense[i][j]);
        ISparseMatrix resultVal = ISparseMatrix.fromDense(resultDense);
        // Save input for backward
        double[][] savedDense = dense;
        Consumer<ISparseMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.toDenseArray();
            double[][] grad = new double[r][c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    grad[i][j] = gd[i][j] * backwardGrad.applyAsDouble(savedDense[i][j]);
            this.accGrad(ISparseMatrix.fromDense(grad));
        };
        RereDiffSparseMatrix node = new RereDiffSparseMatrix(resultVal, List.of(this), backwardFn);
        node.opTag = tag;
        return node;
    }
}
