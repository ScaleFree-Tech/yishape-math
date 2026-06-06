package com.yishape.lab.math.autodiff.impl;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Random;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.RereMathUtil;
import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.util.YishapeLogger;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;

/**
 * Default reverse-mode AD implementation for {@link IDiffMatrix}.
 * {@link IDiffMatrix} 的默认反向模式自动微分实现。
 */
public class RereDiffMatrix implements IDiffMatrix, Serializable {

    private static final long serialVersionUID = 4L;
    private static final YishapeLogger log = YishapeLogger.getLogger(RereDiffMatrix.class);

    private static final ThreadLocal<ArrayList<RereDiffMatrix>> TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffMatrix>> TOPO_SET =
        ThreadLocal.withInitial(HashSet::new);

    public IDoubleMatrix value;
    public IDoubleMatrix gradient;
    public boolean isLeaf;
    public List<RereDiffMatrix> inputs;
    public Consumer<IDoubleMatrix> backwardFn;
    public Function<IDiffVector, IDiffVector[]> symbolicBackwardFn;
    /** Operation tag for graph export and pattern fusion. / 图导出和模式融合用的操作标记。 */
    public String opTag;
    /** Scalar parameter for scalar operations (NaN if not applicable). / 标量操作参数（不适用时为 NaN）。 */
    public double scalarParam = Double.NaN;
    /** Second scalar parameter for dual-param ops like hardtanh/clamp (NaN if not applicable). */
    public double scalarParam2 = Double.NaN;

    public RereDiffMatrix(IDoubleMatrix value) {
        this.value = value;
        this.isLeaf = true;
        this.inputs = new ArrayList<>();
    }

    RereDiffMatrix(IDoubleMatrix value, List<RereDiffMatrix> inputs, Consumer<IDoubleMatrix> backwardFn) {
        this.value = value;
        this.isLeaf = false;
        this.inputs = inputs;
        this.backwardFn = backwardFn;
    }

    // ---- value / gradient access ----

    @Override
    public IDoubleMatrix getValue() {
        return value;
    }

    @Override
    public IDoubleMatrix getGradient() {
        return gradient;
    }

    @Override
    public boolean isLeaf() {
        return isLeaf;
    }

    // ---- gradient operations ----

    @Override
    public void backward() {
        backward(IDoubleMatrix.ones(value.rows(), value.cols()));
    }

    @Override
    public void backward(IDoubleMatrix initialGradient) {
        this.gradient = initialGradient;

        ArrayList<RereDiffMatrix> order = TOPO_LIST.get();
        order.clear();
        HashSet<RereDiffMatrix> visited = TOPO_SET.get();
        visited.clear();
        buildTopo(order, visited);

        try {
            // Zero all intermediate/leaf gradients before accumulation to prevent
            // double-counting when backward() is called multiple times.
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffMatrix v = order.get(i);
                if (v != this) {
                    v.gradient = null;
                }
            }

            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffMatrix v = order.get(i);
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

    /** Iterative DFS post-order topological sort. / 迭代式深度优先后序拓扑排序，避免深层图 StackOverflow。 */
    public void buildTopo(List<RereDiffMatrix> order, Set<RereDiffMatrix> visited) {
        java.util.ArrayDeque<Object[]> stack = new java.util.ArrayDeque<>();
        stack.push(new Object[]{this, Boolean.TRUE});
        while (!stack.isEmpty()) {
            Object[] entry = stack.peek();
            RereDiffMatrix node = (RereDiffMatrix) entry[0];
            boolean childrenNotPushed = (Boolean) entry[1];
            if (childrenNotPushed) {
                entry[1] = Boolean.FALSE;
                if (!visited.add(node)) {
                    stack.pop();
                    continue;
                }
                for (int i = node.inputs.size() - 1; i >= 0; i--) {
                    RereDiffMatrix inp = node.inputs.get(i);
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
    public IDiffMatrix grad() {
        if (this.gradient == null) {
            throw new IllegalStateException("Gradient is null — call backward() first");
        }
        return new RereDiffMatrix(this.gradient.copy());
    }

    @Override
    public IDiffMatrix copy() {
        return new RereDiffMatrix(this.value.copy());
    }

    void accGrad(IDoubleMatrix grad) {
        if (gradient == null) {
            gradient = grad.copy();
        } else {
            double[][] gData = gradient.getData();
            double[][] gradData = grad.getData();
            for (int i = 0; i < gData.length; i++) {
                double[] gRow = gData[i];
                double[] gradRow = gradData[i];
                for (int j = 0; j < gRow.length; j++) {
                    gRow[j] += gradRow[j];
                }
            }
        }
    }

    void accGradDirect(double[][] data) {
        if (gradient == null) {
            gradient = IDoubleMatrix.of(data);
        } else {
            double[][] gData = gradient.getData();
            for (int i = 0; i < gData.length; i++) {
                double[] gRow = gData[i];
                double[] dataRow = data[i];
                for (int j = 0; j < gRow.length; j++) {
                    gRow[j] += dataRow[j];
                }
            }
        }
    }

    private static RereDiffMatrix withTag(RereDiffMatrix node, String tag) {
        node.opTag = tag;
        return node;
    }

    private static RereDiffMatrix withTag(RereDiffMatrix node, String tag, double scalarParam) {
        node.opTag = tag;
        node.scalarParam = scalarParam;
        return node;
    }

    // ---- element-wise matrix division helper ----

    private static IDoubleMatrix div(IDoubleMatrix a, IDoubleMatrix b) {
        return a.divide(b);
    }

    // ---- matrix operations ----

    @Override
    public IDiffMatrix matmul(IDiffMatrix other) {
        RereDiffMatrix o = (RereDiffMatrix) other;
        IDoubleMatrix resultVal = this.value.mmul(o.value);
        IDoubleMatrix thisVal = this.value.copy();
        IDoubleMatrix otherVal = o.value.copy();
        // Pre-compute transposes at forward time to avoid allocation in backward
        double[][] otherValT = transposeFlat(otherVal.getData());
        double[][] thisValT = transposeFlat(thisVal.getData());
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.mmul(IDoubleMatrix.of(otherValT)));
            o.accGrad(IDoubleMatrix.of(thisValT).mmul(gradOut));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this, o), backwardFn), "matmul");
    }

    /** Transpose double[][] returning a new array. */
    private static double[][] transposeFlat(double[][] src) {
        int rows = src.length;
        int cols = src[0].length;
        double[][] dst = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dst[j][i] = src[i][j];
            }
        }
        return dst;
    }

    @Override
    public IDiffVector matmul(IDiffVector vector) {
        RereDiffVector v = (RereDiffVector) vector;
        IDoubleVector resultVal = (IDoubleVector) this.value.mmul(v.value);
        IDoubleMatrix aVal = this.value.copy();
        IDoubleVector xVal = v.value.copy();
        RereDiffMatrix self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            self.accGrad((IDoubleMatrix) gradOut.outer(xVal));
            // Trigger matrix backward chain (e.g. square→matmul→transpose→reshape)
            // since self is not in this node's inputs list.
            self.propagateGradient();
            v.accGrad((IDoubleVector) aVal.transposeNew().mmul(gradOut));
        };
        // Note: self (matrix) is NOT in inputs because inputs is List<RereDiffVector>.
        // propagateGradient() in backwardFn handles matrix graph traversal.
        RereDiffVector node = new RereDiffVector(resultVal, List.of(v), backwardFn);
        node.opTag = "matmul";
        return node;
    }

    @Override
    public IDiffMatrix transpose() {
        IDoubleMatrix resultVal = this.value.transposeNew();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.transposeNew());
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "transpose");
    }

    // ---- element-wise with another matrix ----

    @Override
    public IDiffMatrix add(IDiffMatrix other) {
        RereDiffMatrix o = (RereDiffMatrix) other;
        IDoubleMatrix resultVal = this.value.add(o.value);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
            o.accGrad(gradOut);
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this, o), backwardFn), "add");
    }

    @Override
    public IDiffMatrix sub(IDiffMatrix other) {
        RereDiffMatrix o = (RereDiffMatrix) other;
        IDoubleMatrix resultVal = this.value.sub(o.value);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
            o.accGrad(gradOut.multiplyByScalar(-1.0));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this, o), backwardFn), "sub");
    }

    @Override
    public IDiffMatrix mul(IDiffMatrix other) {
        RereDiffMatrix o = (RereDiffMatrix) other;
        IDoubleMatrix resultVal = this.value.multiply(o.value);
        IDoubleMatrix thisVal = this.value.copy();
        IDoubleMatrix otherVal = o.value.copy();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiply(otherVal));
            o.accGrad(gradOut.multiply(thisVal));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this, o), backwardFn), "mul");
    }

    @Override
    public IDiffMatrix div(IDiffMatrix other) {
        RereDiffMatrix o = (RereDiffMatrix) other;
        IDoubleMatrix resultVal = this.value.divide(o.value);
        IDoubleMatrix thisVal = this.value.copy();
        IDoubleMatrix otherVal = o.value.copy();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(div(gradOut, otherVal));
            o.accGrad(gradOut.multiplyByScalar(-1.0).multiply(thisVal).divide(otherVal.multiply(otherVal)));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this, o), backwardFn), "div");
    }

    // ---- scalar arithmetic ----

    @Override
    public IDiffMatrix add(double scalar) {
        IDoubleMatrix resultVal = this.value.applyMap((Function<Double, Double>) v -> v + scalar);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "addScalar", scalar);
    }

    @Override
    public IDiffMatrix sub(double scalar) {
        IDoubleMatrix resultVal = this.value.sub(scalar);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "subScalar", scalar);
    }

    @Override
    public IDiffMatrix mul(double scalar) {
        IDoubleMatrix resultVal = this.value.multiplyByScalar(scalar);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiplyByScalar(scalar));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "mulScalar", scalar);
    }

    @Override
    public IDiffMatrix div(double scalar) {
        IDoubleMatrix resultVal = this.value.divideByScalar(scalar);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.divideByScalar(scalar));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "divScalar", scalar);
    }

    @Override
    public IDiffMatrix rsub(double scalar) {
        IDoubleMatrix resultVal = this.value.applyMap((Function<Double, Double>) v -> scalar - v);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiplyByScalar(-1.0));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "rsubScalar", scalar);
    }

    @Override
    public IDiffMatrix rdiv(double scalar) {
        IDoubleMatrix xVal = this.value.copy();
        IDoubleMatrix resultVal = this.value.applyMap((Function<Double, Double>) v -> scalar / v);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiplyByScalar(-scalar).divide(xVal.multiply(xVal)));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "rdivScalar", scalar);
    }

    // ---- unary ----

    @Override
    public IDiffMatrix neg() {
        IDoubleMatrix resultVal = this.value.multiplyByScalar(-1.0);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiplyByScalar(-1.0));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "neg");
    }

    @Override
    public IDiffMatrix pow(double n) {
        IDoubleMatrix xVal = this.value.copy();
        IDoubleMatrix resultVal = this.value.pow(n);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiplyByScalar(n).multiply(xVal.pow(n - 1)));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "pow", n);
    }

    // ---- element-wise math ----

    @Override
    public IDiffMatrix exp() {
        IDoubleMatrix resultVal = this.value.exp();
        IDoubleMatrix zVal = resultVal.copy();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiply(zVal));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "exp");
    }

    @Override
    public IDiffMatrix log() {
        IDoubleMatrix xVal = this.value.copy();
        IDoubleMatrix resultVal = this.value.log();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.divide(xVal));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "log");
    }

    @Override
    public IDiffMatrix sigmoid() {
        IDoubleMatrix resultVal = this.value.applyMap((Function<Double, Double>) RereMathUtil::sigmoid);
        IDoubleMatrix zVal = resultVal.copy();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiply(zVal.applyMap((Function<Double, Double>) v -> v * (1.0 - v))));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "sigmoid");
    }

    @Override
    public IDiffMatrix relu() {
        IDoubleMatrix xVal = this.value.copy();
        IDoubleMatrix resultVal = this.value.applyMap((Function<Double, Double>) v -> Math.max(0.0, v));
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gradData = gradOut.getData();
            double[][] xData = xVal.getData();
            int rows = gradData.length;
            int cols = gradData[0].length;
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    dx[i][j] = gradData[i][j] * (xData[i][j] > 0.0 ? 1.0 : 0.0);
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "relu");
    }

    @Override
    public IDiffMatrix tanh() {
        IDoubleMatrix resultVal = this.value.tanh();
        IDoubleMatrix zVal = resultVal.copy();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiply(zVal.applyMap((Function<Double, Double>) v -> 1.0 - v * v)));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "tanh");
    }

    @Override
    public IDiffMatrix sqrt() {
        IDoubleMatrix resultVal = this.value.sqrt();
        IDoubleMatrix zVal = resultVal.copy();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.divide(zVal.multiplyByScalar(2.0)));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "sqrt");
    }

    @Override
    public IDiffMatrix square() {
        IDoubleMatrix xVal = this.value.copy();
        IDoubleMatrix resultVal = this.value.multiply(this.value);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiplyByScalar(2.0).multiply(xVal));
        };
        RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(this), backwardFn);
        node.opTag = "square";
        return node;
    }

    @Override
    public IDiffMatrix abs() {
        IDoubleMatrix xVal = this.value.copy();
        IDoubleMatrix resultVal = this.value.abs();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiply(xVal.sign()));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "abs");
    }

    @Override
    public IDiffMatrix gelu() {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
        double g = 0.044715;
        double[][] y = new double[rows][cols];
        double[][] tanhVals = new double[rows][cols];
        double[][] innerVals = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = xd[i][j];
                double inner = sqrt2OverPi * (x + g * x * x * x);
                innerVals[i][j] = inner;
                double tanhInner = Math.tanh(inner);
                tanhVals[i][j] = tanhInner;
                y[i][j] = 0.5 * x * (1.0 + tanhInner);
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double x = xd[i][j];
                    double tanhI = tanhVals[i][j];
                    double inner = innerVals[i][j];
                    double sechSq = 1.0 - tanhI * tanhI;
                    double din_dx = sqrt2OverPi * (1.0 + 3.0 * g * x * x);
                    dx[i][j] = gd[i][j] * (0.5 * (1.0 + tanhI) + 0.5 * x * sechSq * din_dx);
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "gelu");
    }

    @Override
    public IDiffMatrix leakyRelu(double alpha) {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double[][] y = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = xd[i][j];
                y[i][j] = x > 0 ? x : alpha * x;
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    dx[i][j] = gd[i][j] * (xd[i][j] > 0 ? 1.0 : alpha);
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "leakyRelu", alpha);
    }

    @Override
    public IDiffMatrix elu(double alpha) {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double[][] y = new double[rows][cols];
        double[][] expVal = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = xd[i][j];
                if (x >= 0) {
                    y[i][j] = x;
                    expVal[i][j] = 0;
                } else {
                    double ex = Math.exp(x);
                    expVal[i][j] = ex;
                    y[i][j] = alpha * (ex - 1.0);
                }
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (xd[i][j] >= 0) {
                        dx[i][j] = gd[i][j];
                    } else {
                        dx[i][j] = gd[i][j] * alpha * expVal[i][j];
                    }
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "elu", alpha);
    }

    @Override
    public IDiffMatrix selu() {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double alpha = 1.6732632423543772848170429916717;
        double scale = 1.0507009873554804934193349852946;
        double[][] y = new double[rows][cols];
        double[][] expVal = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = xd[i][j];
                if (x >= 0) {
                    y[i][j] = scale * x;
                    expVal[i][j] = 0;
                } else {
                    double ex = Math.exp(x);
                    expVal[i][j] = ex;
                    y[i][j] = scale * alpha * (ex - 1.0);
                }
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    if (xd[i][j] >= 0) {
                        dx[i][j] = gd[i][j] * scale;
                    } else {
                        dx[i][j] = gd[i][j] * scale * alpha * expVal[i][j];
                    }
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "selu");
    }

    @Override
    public IDiffMatrix silu() {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double[][] y = new double[rows][cols];
        double[][] sigVals = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = xd[i][j];
                double s = 1.0 / (1.0 + Math.exp(-x));
                sigVals[i][j] = s;
                y[i][j] = x * s;
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double s = sigVals[i][j];
                    dx[i][j] = gd[i][j] * (s + xd[i][j] * s * (1.0 - s));
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "silu");
    }

    @Override
    public IDiffMatrix mish() {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double[][] y = new double[rows][cols];
        double[][] spVals = new double[rows][cols];
        double[][] tanhVals = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double x = xd[i][j];
                double sp = Math.log(1.0 + Math.exp(x));
                spVals[i][j] = sp;
                double th = Math.tanh(sp);
                tanhVals[i][j] = th;
                y[i][j] = x * th;
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double x = xd[i][j];
                    double th = tanhVals[i][j];
                    double sig = 1.0 / (1.0 + Math.exp(-x));
                    double dTanh = 1.0 - th * th;
                    dx[i][j] = gd[i][j] * (th + x * dTanh * sig);
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "mish");
    }

    @Override
    public IDiffMatrix softplus(double beta) {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double[][] y = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double bx = beta * xd[i][j];
                y[i][j] = bx > 100 ? xd[i][j] : (1.0 / beta) * Math.log(1.0 + Math.exp(bx));
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double bx = beta * xd[i][j];
                    double sig = bx > 100 ? 1.0 : 1.0 / (1.0 + Math.exp(-bx));
                    dx[i][j] = gd[i][j] * sig;
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "softplus", beta);
    }

    @Override
    public IDiffMatrix hardtanh(double minVal, double maxVal) {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double[][] y = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v = xd[i][j];
                y[i][j] = v < minVal ? minVal : (v > maxVal ? maxVal : v);
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double v = xd[i][j];
                    dx[i][j] = gd[i][j] * (v > minVal && v < maxVal ? 1.0 : 0.0);
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(this), backwardFn);
        node.opTag = "hardtanh";
        node.scalarParam = minVal;
        node.scalarParam2 = maxVal;
        return node;
    }

    @Override
    public IDiffMatrix clamp(double min, double max) {
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        int rows = xd.length;
        int cols = xd[0].length;
        double[][] y = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v = xd[i][j];
                y[i][j] = v < min ? min : (v > max ? max : v);
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double v = xd[i][j];
                    dx[i][j] = gd[i][j] * (v >= min && v <= max ? 1.0 : 0.0);
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(this), backwardFn);
        node.opTag = "clamp";
        node.scalarParam = min;
        node.scalarParam2 = max;
        return node;
    }

    @Override
    public IDiffMatrix layerNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector gr = (RereDiffVector) gamma;
        RereDiffVector br = (RereDiffVector) beta;
        int rows = this.value.rows();
        int cols = this.value.cols();
        int features = gr.value.size();
        if (cols != features) {
            throw new IllegalArgumentException(
                "Matrix cols (" + cols + ") != gamma size (" + features + ")");
        }
        double[][] xd = this.value.getData();
        double[] gd = gr.value.getData();
        double[] bd = br.value.getData();

        double[][] y = new double[rows][cols];
        double[][] xHat = new double[rows][cols];
        double[] means = new double[rows];
        double[] sigmas = new double[rows];

        // Forward: for each row, normalize over columns
        for (int i = 0; i < rows; i++) {
            double mean = 0;
            for (int j = 0; j < cols; j++) mean += xd[i][j];
            mean /= cols;
            means[i] = mean;

            double var = 0;
            for (int j = 0; j < cols; j++) { double d = xd[i][j] - mean; var += d * d; }
            var /= cols;

            double sigma = Math.sqrt(var + eps);
            sigmas[i] = sigma;

            for (int j = 0; j < cols; j++) {
                xHat[i][j] = (xd[i][j] - mean) / sigma;
                y[i][j] = gd[j] * xHat[i][j] + bd[j];
            }
        }

        RereDiffMatrix self = this;
        Consumer<IDoubleMatrix> backwardFn = (gradOutput) -> {
            double[][] g = gradOutput.getData();
            double[] cg = gr.value.getData();
            double[][] dx = new double[rows][cols];
            double[] dGamma = new double[features];
            double[] dBeta = new double[features];

            for (int i = 0; i < rows; i++) {
                double sigma = sigmas[i];
                double sumGT = 0, sumGTXH = 0;
                for (int j = 0; j < cols; j++) {
                    double gt = g[i][j] * cg[j];
                    sumGT += gt;
                    sumGTXH += gt * xHat[i][j];
                }
                double invCS = 1.0 / (cols * sigma);
                for (int j = 0; j < cols; j++) {
                    double gt = g[i][j] * cg[j];
                    dx[i][j] = (cols * gt - sumGT - xHat[i][j] * sumGTXH) * invCS;
                }
                for (int j = 0; j < cols; j++) {
                    dGamma[j] += g[i][j] * xHat[i][j];
                    dBeta[j] += g[i][j];
                }
            }

            self.accGrad(IDoubleMatrix.of(dx));
            gr.accGrad(IDoubleVector.of(dGamma));
            br.accGrad(IDoubleVector.of(dBeta));
        };

        RereDiffMatrix node = new RereDiffMatrix(IDoubleMatrix.of(y), List.of(this), backwardFn);
        node.opTag = "layerNorm";
        node.scalarParam = eps;
        return node;
    }

    @Override
    public IDiffMatrix batchNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector gr = (RereDiffVector) gamma;
        RereDiffVector br = (RereDiffVector) beta;
        int rows = this.value.rows();
        int cols = this.value.cols();
        int features = gr.value.size();
        if (cols != features) {
            throw new IllegalArgumentException(
                "Matrix cols (" + cols + ") != gamma size (" + features + ")");
        }
        double[][] xd = this.value.getData();
        double[] gd = gr.value.getData();
        double[] bd = br.value.getData();

        double[][] y = new double[rows][cols];
        double[] means = new double[cols];
        double[] invSigmas = new double[cols];

        // Forward: for each column (feature), normalize over rows (batch)
        for (int j = 0; j < cols; j++) {
            double mean = 0;
            for (int i = 0; i < rows; i++) mean += xd[i][j];
            mean /= rows;
            means[j] = mean;

            double var = 0;
            for (int i = 0; i < rows; i++) { double d = xd[i][j] - mean; var += d * d; }
            var /= rows;

            double invSigma = 1.0 / Math.sqrt(var + eps);
            invSigmas[j] = invSigma;

            for (int i = 0; i < rows; i++) {
                double xHat = (xd[i][j] - mean) * invSigma;
                y[i][j] = gd[j] * xHat + bd[j];
            }
        }

        RereDiffMatrix self = this;
        Consumer<IDoubleMatrix> backwardFn = (gradOutput) -> {
            double[][] g = gradOutput.getData();
            double[] cg = gr.value.getData();
            double[][] dx = new double[rows][cols];
            double[] dGamma = new double[features];
            double[] dBeta = new double[features];

            for (int j = 0; j < cols; j++) {
                double invSig = invSigmas[j];
                double mean = means[j];
                double dg = 0, db = 0;
                for (int i = 0; i < rows; i++) {
                    double xHat = (xd[i][j] - mean) * invSig;
                    dg += g[i][j] * xHat;
                    db += g[i][j];
                }
                dGamma[j] = dg;
                dBeta[j] = db;

                double sumG = 0, sumGXHat = 0;
                for (int i = 0; i < rows; i++) {
                    double xHat = (xd[i][j] - mean) * invSig;
                    sumG += g[i][j];
                    sumGXHat += g[i][j] * xHat;
                }
                double scale = cg[j] * invSig / rows;
                for (int i = 0; i < rows; i++) {
                    double xHat = (xd[i][j] - mean) * invSig;
                    dx[i][j] = scale * (rows * g[i][j] - sumG - xHat * sumGXHat);
                }
            }

            self.accGrad(IDoubleMatrix.of(dx));
            gr.accGrad(IDoubleVector.of(dGamma));
            br.accGrad(IDoubleVector.of(dBeta));
        };

        RereDiffMatrix node = new RereDiffMatrix(IDoubleMatrix.of(y), List.of(this), backwardFn);
        node.opTag = "batchNorm";
        node.scalarParam = eps;
        return node;
    }

    @Override
    public IDiffMatrix dropout(double p) {
        IDoubleMatrix xVal = this.value.copy();
        int rows = xVal.rows();
        int cols = xVal.cols();
        double scale = 1.0 / (1.0 - p);
        long seed = RereDiffVector.DROPOUT_SEED_COUNTER.incrementAndGet();
        Random rng = new Random(seed);
        double[][] mask = new double[rows][cols];
        double[][] y = new double[rows][cols];
        double[][] xd = xVal.getData();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                mask[i][j] = rng.nextDouble() > p ? scale : 0.0;
                y[i][j] = xd[i][j] * mask[i][j];
            }
        }
        IDoubleMatrix resultVal = IDoubleMatrix.of(y);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gd = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    dx[i][j] = gd[i][j] * mask[i][j];
                }
            }
            this.accGrad(IDoubleMatrix.of(dx));
        };
        RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(this), backwardFn);
        node.opTag = "dropout";
        node.scalarParam = p;
        node.scalarParam2 = Double.longBitsToDouble(seed);
        return node;
    }

    // ---- reductions ----

    @Override
    public IDiffMatrix sum() {
        // Pattern fusion: square().sum() → single fused node
        if ("square".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffMatrix x = this.inputs.get(0);
            IDoubleMatrix xVal = x.value.copy();
            int r = xVal.rows();
            int c = xVal.cols();
            double s = this.value.sumValue();
            IDoubleMatrix resultVal = IDoubleMatrix.of(new double[][] { { s } });
            Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
                double g = gradOut.get(0, 0);
                double[][] xData = xVal.getData();
                double[][] fused = new double[r][c];
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < c; j++) {
                        fused[i][j] = 2.0 * g * xData[i][j];
                    }
                }
                x.accGradDirect(fused);
            };
            RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(x), backwardFn);
            node.opTag = "squareSum";
            return node;
        }
        // Pattern fusion: exp().sum() → single fused node
        if ("exp".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffMatrix x = this.inputs.get(0);
            IDoubleMatrix xVal = x.value.copy();
            int r = xVal.rows();
            int c = xVal.cols();
            double s = this.value.sumValue();
            IDoubleMatrix resultVal = IDoubleMatrix.of(new double[][] { { s } });
            Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
                double g = gradOut.get(0, 0);
                double[][] xData = xVal.getData();
                double[][] fused = new double[r][c];
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < c; j++) {
                        fused[i][j] = g * Math.exp(xData[i][j]);
                    }
                }
                x.accGradDirect(fused);
            };
            RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(x), backwardFn);
            node.opTag = "expSum";
            return node;
        }
        double s = this.value.sumValue();
        IDoubleMatrix resultVal = IDoubleMatrix.of(new double[][] { { s } });
        int r = this.value.rows();
        int c = this.value.cols();
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double g = gradOut.get(0, 0);
            if (this.gradient == null) {
                this.gradient = IDoubleMatrix.ones(r, c).multiplyByScalar(g);
            } else {
                double[][] gData = this.gradient.getData();
                for (int i = 0; i < r; i++) {
                    double[] row = gData[i];
                    for (int j = 0; j < c; j++) {
                        row[j] += g;
                    }
                }
            }
        };
        RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(this), backwardFn);
        node.opTag = "sum";
        return node;
    }

    @Override
    public IDiffMatrix mean() {
        // Pattern fusion: square().mean() → single fused node
        if ("square".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffMatrix x = this.inputs.get(0);
            IDoubleMatrix xVal = x.value.copy();
            int r = xVal.rows();
            int c = xVal.cols();
            int n = r * c;
            double m = this.value.meanValue();
            IDoubleMatrix resultVal = IDoubleMatrix.of(new double[][] { { m } });
            Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
                double g = gradOut.get(0, 0) / n;
                double[][] xData = xVal.getData();
                double[][] fused = new double[r][c];
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < c; j++) {
                        fused[i][j] = 2.0 * g * xData[i][j];
                    }
                }
                x.accGradDirect(fused);
            };
            RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(x), backwardFn);
            node.opTag = "squareMean";
            return node;
        }
        double m = this.value.meanValue();
        IDoubleMatrix resultVal = IDoubleMatrix.of(new double[][] { { m } });
        int r = this.value.rows();
        int c = this.value.cols();
        int n = r * c;
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double g = gradOut.get(0, 0) / n;
            if (this.gradient == null) {
                this.gradient = IDoubleMatrix.ones(r, c).multiplyByScalar(g);
            } else {
                double[][] gData = this.gradient.getData();
                for (int i = 0; i < r; i++) {
                    double[] row = gData[i];
                    for (int j = 0; j < c; j++) {
                        row[j] += g;
                    }
                }
            }
        };
        RereDiffMatrix node = new RereDiffMatrix(resultVal, List.of(this), backwardFn);
        node.opTag = "mean";
        return node;
    }

    @Override
    public IDiffVector sumAsVector() {
        // Pattern fusion: square().sumAsVector() → single fused node
        if ("square".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffMatrix x = this.inputs.get(0);
            IDoubleMatrix xVal = x.value.copy();
            int r = xVal.rows();
            int c = xVal.cols();
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[][] xData = xVal.getData();
                double[][] fused = new double[r][c];
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < c; j++) {
                        fused[i][j] = 2.0 * g * xData[i][j];
                    }
                }
                x.accGradDirect(fused);
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
            node.opTag = "squareSum";
            return node;
        }
        // Pattern fusion: exp().sumAsVector() → single fused node
        if ("exp".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffMatrix x = this.inputs.get(0);
            IDoubleMatrix xVal = x.value.copy();
            int r = xVal.rows();
            int c = xVal.cols();
            double s = this.value.sumValue();
            IDoubleVector resultVal = IDoubleVector.of(s);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0);
                double[][] xData = xVal.getData();
                double[][] fused = new double[r][c];
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < c; j++) {
                        fused[i][j] = g * Math.exp(xData[i][j]);
                    }
                }
                x.accGradDirect(fused);
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
            node.opTag = "expSum";
            return node;
        }
        double s = this.value.sumValue();
        IDoubleVector resultVal = IDoubleVector.of(s);
        int r = this.value.rows();
        int c = this.value.cols();
        RereDiffMatrix self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0);
            if (self.gradient == null) {
                self.gradient = IDoubleMatrix.ones(r, c).multiplyByScalar(g);
            } else {
                double[][] gData = self.gradient.getData();
                for (int i = 0; i < r; i++) {
                    double[] row = gData[i];
                    for (int j = 0; j < c; j++) {
                        row[j] += g;
                    }
                }
            }
            self.propagateGradient();
        };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
        node.opTag = "sum";
        return node;
    }

    @Override
    public IDiffVector meanAsVector() {
        // Pattern fusion: square().meanAsVector() → single fused node
        if ("square".equals(this.opTag) && this.inputs.size() == 1) {
            RereDiffMatrix x = this.inputs.get(0);
            IDoubleMatrix xVal = x.value.copy();
            int r = xVal.rows();
            int c = xVal.cols();
            int n = r * c;
            double m = this.value.meanValue();
            IDoubleVector resultVal = IDoubleVector.of(m);
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double g = gradOut.get(0) / n;
                double[][] xData = xVal.getData();
                double[][] fused = new double[r][c];
                for (int i = 0; i < r; i++) {
                    for (int j = 0; j < c; j++) {
                        fused[i][j] = 2.0 * g * xData[i][j];
                    }
                }
                x.accGradDirect(fused);
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
            node.opTag = "squareMean";
            return node;
        }
        double m = this.value.meanValue();
        IDoubleVector resultVal = IDoubleVector.of(m);
        int r = this.value.rows();
        int c = this.value.cols();
        int n = r * c;
        RereDiffMatrix self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double g = gradOut.get(0) / n;
            if (self.gradient == null) {
                self.gradient = IDoubleMatrix.ones(r, c).multiplyByScalar(g);
            } else {
                double[][] gData = self.gradient.getData();
                for (int i = 0; i < r; i++) {
                    double[] row = gData[i];
                    for (int j = 0; j < c; j++) {
                        row[j] += g;
                    }
                }
            }
            self.propagateGradient();
        };
        RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
        node.opTag = "mean";
        return node;
    }

    // ---- in-place operations ----

    @Override
    public IDiffMatrix addInPlace(IDiffMatrix other) {
        if (!this.isLeaf) {
            throw new IllegalStateException("addInPlace only allowed on leaf variables");
        }
        RereDiffMatrix o = (RereDiffMatrix) other;
        this.value = this.value.add(o.value);
        this.isLeaf = false;
        this.inputs = List.of(o);
        this.backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
            o.accGrad(gradOut);
        };
        return this;
    }

    @Override
    public IDiffMatrix mulInPlace(double scalar) {
        if (!this.isLeaf) {
            throw new IllegalStateException("mulInPlace only allowed on leaf variables");
        }
        this.value = this.value.multiplyByScalar(scalar);
        this.isLeaf = false;
        this.inputs = List.of();
        this.backwardFn = (gradOut) -> {
            this.accGrad(gradOut.multiplyByScalar(scalar));
        };
        return this;
    }

    // ---- reshape ----

    @Override
    public IDiffVector flatten() {
        int origRows = this.value.rows();
        int origCols = this.value.cols();
        IDoubleVector resultVal = (IDoubleVector) this.value.flatten();
        RereDiffMatrix self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            IDoubleMatrix matGrad = IDoubleMatrix.fromArray(gradOut.getData(), origRows, origCols);
            self.accGrad(matGrad);
            self.propagateGradient();
        };
        // self (RereDiffMatrix) cannot be in List<RereDiffVector> inputs,
        // so propagateGradient() is needed to propagate through the matrix graph.
        RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
        node.opTag = "flatten";
        return node;
    }

    void propagateGradient() {
        if (this.gradient == null || this.backwardFn == null) return;
        // Use local lists to avoid corrupting the outer backward() loop's ThreadLocal
        ArrayList<RereDiffMatrix> order = new ArrayList<>();
        HashSet<RereDiffMatrix> visited = new HashSet<>();
        buildTopo(order, visited);
        for (int i = order.size() - 1; i >= 0; i--) {
            RereDiffMatrix v = order.get(i);
            if (v.gradient != null && v.backwardFn != null) {
                IDoubleMatrix savedGrad = v.gradient;
                v.gradient = null;
                v.backwardFn.accept(savedGrad);
            }
        }
    }

    @Override
    public IDiffMatrix reshape(int rows, int cols) {
        int origRows = this.value.rows();
        int origCols = this.value.cols();
        IDoubleMatrix resultVal = this.value.reshape(rows, cols);
        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.reshape(origRows, origCols));
        };
        return withTag(new RereDiffMatrix(resultVal, List.of(this), backwardFn), "reshape");
    }

    // ---- axis-wise reductions ----

    @Override
    public IDiffVector sum(int axis) {
        int rows = this.value.rows();
        int cols = this.value.cols();
        IDoubleVector resultVal;
        Consumer<IDoubleVector> backwardFn;
        RereDiffMatrix self = this;

        if (axis == 0) {
            resultVal = (IDoubleVector) this.value.colSums();
            backwardFn = (gradOut) -> {
                double[] gd = gradOut.getData();
                double[][] dx = new double[rows][cols];
                for (int j = 0; j < cols; j++) {
                    double gj = gd[j];
                    for (int i = 0; i < rows; i++) {
                        dx[i][j] = gj;
                    }
                }
                self.accGrad(IDoubleMatrix.of(dx));
                self.propagateGradient();
            };
        } else if (axis == 1) {
            resultVal = (IDoubleVector) this.value.rowSums();
            backwardFn = (gradOut) -> {
                double[] gd = gradOut.getData();
                double[][] dx = new double[rows][cols];
                for (int i = 0; i < rows; i++) {
                    double gi = gd[i];
                    for (int j = 0; j < cols; j++) {
                        dx[i][j] = gi;
                    }
                }
                self.accGrad(IDoubleMatrix.of(dx));
                self.propagateGradient();
            };
        } else {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
        RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
        node.opTag = "sum";
        return node;
    }

    @Override
    public IDiffVector max(int axis) {
        int rows = this.value.rows();
        int cols = this.value.cols();
        IDoubleMatrix xVal = this.value.copy();
        RereDiffMatrix self = this;

        if (axis == 0) {
            IDoubleVector resultVal = (IDoubleVector) xVal.colMaxs();
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double[][] dx = new double[rows][cols];
                double[][] xd = xVal.getData();
                double[] gd = gradOut.getData();
                for (int j = 0; j < cols; j++) {
                    double maxVal = Double.NEGATIVE_INFINITY;
                    int maxI = 0;
                    for (int i = 0; i < rows; i++) {
                        if (xd[i][j] > maxVal) {
                            maxVal = xd[i][j];
                            maxI = i;
                        }
                    }
                    dx[maxI][j] = gd[j];
                }
                self.accGrad(IDoubleMatrix.of(dx));
                self.propagateGradient();
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
            node.opTag = "max";
            return node;
        } else if (axis == 1) {
            IDoubleVector resultVal = (IDoubleVector) xVal.rowMaxs();
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double[][] dx = new double[rows][cols];
                double[][] xd = xVal.getData();
                double[] gd = gradOut.getData();
                for (int i = 0; i < rows; i++) {
                    double maxVal = Double.NEGATIVE_INFINITY;
                    int maxJ = 0;
                    for (int j = 0; j < cols; j++) {
                        if (xd[i][j] > maxVal) {
                            maxVal = xd[i][j];
                            maxJ = j;
                        }
                    }
                    dx[i][maxJ] = gd[i];
                }
                self.accGrad(IDoubleMatrix.of(dx));
                self.propagateGradient();
            };
            RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
            node.opTag = "max";
            return node;
        } else {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
    }

    // ---- broadcast arithmetic ----

    @Override
    public IDiffMatrix sub(IDiffVector vec, int axis) {
        RereDiffVector v = (RereDiffVector) vec;
        int rows = this.value.rows();
        int cols = this.value.cols();
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        double[] vd = v.value.getData();
        double[][] resultData = new double[rows][cols];

        if (axis == 1) {
            if (vd.length != rows) {
                throw new IllegalArgumentException("axis=1: vec length " + vd.length + " != rows " + rows);
            }
            for (int i = 0; i < rows; i++) {
                double vi = vd[i];
                for (int j = 0; j < cols; j++) {
                    resultData[i][j] = xd[i][j] - vi;
                }
            }
            Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
                this.accGrad(gradOut);
                double[][] gd = gradOut.getData();
                double[] vecGrad = new double[rows];
                for (int i = 0; i < rows; i++) {
                    double sum = 0;
                    for (int j = 0; j < cols; j++) {
                        sum += gd[i][j];
                    }
                    vecGrad[i] = -sum; // d(x - v)/dv = -1, sum over broadcast axis
                }
                v.accGrad(IDoubleVector.of(vecGrad));
            };
            return withTag(new RereDiffMatrix(IDoubleMatrix.of(resultData), List.of(this), backwardFn), "sub");
        } else if (axis == 0) {
            if (vd.length != cols) {
                throw new IllegalArgumentException("axis=0: vec length " + vd.length + " != cols " + cols);
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    resultData[i][j] = xd[i][j] - vd[j];
                }
            }
            Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
                this.accGrad(gradOut);
                double[][] gd = gradOut.getData();
                double[] vecGrad = new double[cols];
                for (int j = 0; j < cols; j++) {
                    double sum = 0;
                    for (int i = 0; i < rows; i++) {
                        sum += gd[i][j];
                    }
                    vecGrad[j] = -sum; // d(x - v)/dv = -1, sum over broadcast axis
                }
                v.accGrad(IDoubleVector.of(vecGrad));
            };
            return withTag(new RereDiffMatrix(IDoubleMatrix.of(resultData), List.of(this), backwardFn), "sub");
        } else {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
    }

    @Override
    public IDiffMatrix div(IDiffVector vec, int axis) {
        RereDiffVector v = (RereDiffVector) vec;
        int rows = this.value.rows();
        int cols = this.value.cols();
        IDoubleMatrix xVal = this.value.copy();
        double[][] xd = xVal.getData();
        double[] vd = v.value.getData();
        double[][] resultData = new double[rows][cols];

        if (axis == 1) {
            if (vd.length != rows) {
                throw new IllegalArgumentException("axis=1: vec length " + vd.length + " != rows " + rows);
            }
            for (int i = 0; i < rows; i++) {
                double vi = vd[i];
                for (int j = 0; j < cols; j++) {
                    resultData[i][j] = xd[i][j] / vi;
                }
            }
            Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
                double[][] gd = gradOut.getData();
                double[][] dx = new double[rows][cols];
                double[] vecGrad = new double[rows];
                for (int i = 0; i < rows; i++) {
                    double vi = vd[i];
                    double invVi = 1.0 / vi;
                    double invViSq = invVi * invVi;
                    double vAcc = 0;
                    for (int j = 0; j < cols; j++) {
                        dx[i][j] = gd[i][j] * invVi;
                        vAcc -= gd[i][j] * xd[i][j] * invViSq;
                    }
                    vecGrad[i] = vAcc;
                }
                this.accGrad(IDoubleMatrix.of(dx));
                v.accGrad(IDoubleVector.of(vecGrad));
            };
            return withTag(new RereDiffMatrix(IDoubleMatrix.of(resultData), List.of(this), backwardFn), "div");
        } else if (axis == 0) {
            if (vd.length != cols) {
                throw new IllegalArgumentException("axis=0: vec length " + vd.length + " != cols " + cols);
            }
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    resultData[i][j] = xd[i][j] / vd[j];
                }
            }
            Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
                double[][] gd = gradOut.getData();
                double[][] dx = new double[rows][cols];
                double[] vecGrad = new double[cols];
                for (int j = 0; j < cols; j++) {
                    double vj = vd[j];
                    double invVj = 1.0 / vj;
                    double invVjSq = invVj * invVj;
                    double vAcc = 0;
                    for (int i = 0; i < rows; i++) {
                        dx[i][j] = gd[i][j] * invVj;
                        vAcc -= gd[i][j] * xd[i][j] * invVjSq;
                    }
                    vecGrad[j] = vAcc;
                }
                this.accGrad(IDoubleMatrix.of(dx));
                v.accGrad(IDoubleVector.of(vecGrad));
            };
            return withTag(new RereDiffMatrix(IDoubleMatrix.of(resultData), List.of(this), backwardFn), "div");
        } else {
            throw new IllegalArgumentException("axis must be 0 or 1, got " + axis);
        }
    }

    // ---- comparison operations ----

    @Override
    public boolean[][] ge(IMatrix<Double> other) {
        return this.value.ge(other);
    }

    // ---- fused softmax cross-entropy ----

    @Override
    public IDiffVector softmaxCrossEntropy(IDiffMatrix oneHotLabels) {
        RereDiffMatrix y = (RereDiffMatrix) oneHotLabels;
        int m = this.value.rows();
        int k = this.value.cols();
        IDoubleMatrix zVal = this.value.copy();
        double[][] zd = zVal.getData();
        double[][] yd = y.value.getData();

        int[] trueClass = new int[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                if (yd[i][j] == 1.0) {
                    trueClass[i] = j;
                    break;
                }
            }
        }

        double[] rowMax = new double[m];
        double[] rowSumExp = new double[m];
        double totalLoss = 0;
        for (int i = 0; i < m; i++) {
            double maxVal = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < k; j++) {
                if (zd[i][j] > maxVal) maxVal = zd[i][j];
            }
            rowMax[i] = maxVal;
            double sumExp = 0;
            for (int j = 0; j < k; j++) {
                sumExp += Math.exp(zd[i][j] - maxVal);
            }
            rowSumExp[i] = sumExp;
            totalLoss += Math.log(sumExp) - zd[i][trueClass[i]] + maxVal;
        }
        double lossVal = totalLoss / m;
        IDoubleVector resultVal = IDoubleVector.of(lossVal);

        RereDiffMatrix self = this;
        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double gScale = gradOut.get(0) / m;
            double[][] dz = new double[m][k];
            for (int i = 0; i < m; i++) {
                double mx = rowMax[i];
                double sumExp = rowSumExp[i];
                double invSum = 1.0 / sumExp;
                for (int j = 0; j < k; j++) {
                    double p = Math.exp(zd[i][j] - mx) * invSum;
                    dz[i][j] = gScale * (p - yd[i][j]);
                }
            }
            self.accGrad(IDoubleMatrix.of(dz));
            self.propagateGradient();
        };
        // self (RereDiffMatrix) cannot be in List<RereDiffVector> inputs,
        // so propagateGradient() is needed to propagate through the matrix graph.
        RereDiffVector node = new RereDiffVector(resultVal, List.of(), backwardFn);
        node.opTag = "softmaxCrossEntropy";
        node.scalarParam = m;
        node.exportShape = new int[]{m, k};
        return node;
    }
}
