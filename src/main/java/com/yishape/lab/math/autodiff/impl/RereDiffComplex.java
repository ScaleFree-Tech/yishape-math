package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;

import com.yishape.lab.math.core.Complex;
import com.yishape.lab.math.linalg.complex.IComplexMatrix;
import com.yishape.lab.math.linalg.complex.IComplexMatrix.IComplexVector;
import com.yishape.lab.math.autodiff.IDiffComplex;

/**
 * Reverse-mode AD for complex vectors (Wirtinger gradients).
 * 复向量的反向模式自动微分（Wirtinger 梯度）。
 */
public class RereDiffComplex implements IDiffComplex {

    private static final long serialVersionUID = 1L;

    private static final ThreadLocal<ArrayList<RereDiffComplex>> TOPO_LIST =
        ThreadLocal.withInitial(ArrayList::new);
    private static final ThreadLocal<HashSet<RereDiffComplex>> TOPO_SET =
        ThreadLocal.withInitial(HashSet::new);

    IComplexVector value;
    IComplexVector gradient;
    boolean isLeaf;
    List<RereDiffComplex> inputs;
    Consumer<IComplexVector> backwardFn;
    Function<IDiffComplex, IDiffComplex[]> symbolicBackwardFn;
    String opTag;

    static IDiffComplex constant(IComplexVector value) {
        RereDiffComplex node = new RereDiffComplex(value.copy());
        node.opTag = "constant";
        return node;
    }

    private static RereDiffComplex withTag(RereDiffComplex node, String tag) {
        node.opTag = tag;
        return node;
    }

    public RereDiffComplex(IComplexVector value) {
        this.value = value;
        this.isLeaf = true;
        this.inputs = new ArrayList<>();
    }

    RereDiffComplex(IComplexVector value, List<RereDiffComplex> inputs,
            Consumer<IComplexVector> backwardFn) {
        this.value = value;
        this.isLeaf = false;
        this.inputs = inputs;
        this.backwardFn = backwardFn;
    }

    @Override public IComplexVector getValue() { return value; }
    @Override public IComplexVector getGradient() { return gradient; }
    @Override public boolean isLeaf() { return isLeaf; }

    @Override
    public void backward() {
        int n = value.length();
        Complex[] ones = new Complex[n];
        for (int i = 0; i < n; i++) ones[i] = Complex.ONE;
        backward(IComplexVector.fromComplex(ones));
    }

    @Override
    public void backward(IComplexVector initialGradient) {
        this.gradient = initialGradient;
        ArrayList<RereDiffComplex> order = TOPO_LIST.get();
        order.clear();
        HashSet<RereDiffComplex> visited = TOPO_SET.get();
        visited.clear();
        buildTopo(order, visited);
        try {
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffComplex v = order.get(i);
                if (v != this) {
                    v.gradient = null;
                }
            }
            for (int i = order.size() - 1; i >= 0; i--) {
                RereDiffComplex v = order.get(i);
                if (v.gradient != null && v.backwardFn != null) {
                    v.backwardFn.accept(v.gradient);
                }
            }
        } finally {
            order.clear();
            visited.clear();
        }
        order.clear();
        visited.clear();
    }

    /** Iterative DFS post-order topological sort. / 迭代式深度优先后序拓扑排序，避免深层图 StackOverflow。 */
    void buildTopo(List<RereDiffComplex> order, Set<RereDiffComplex> visited) {
        java.util.ArrayDeque<Object[]> stack = new java.util.ArrayDeque<>();
        stack.push(new Object[]{this, Boolean.TRUE});
        while (!stack.isEmpty()) {
            Object[] entry = stack.peek();
            RereDiffComplex node = (RereDiffComplex) entry[0];
            boolean childrenNotPushed = (Boolean) entry[1];
            if (childrenNotPushed) {
                entry[1] = Boolean.FALSE;
                if (!visited.add(node)) {
                    stack.pop();
                    continue;
                }
                for (int i = node.inputs.size() - 1; i >= 0; i--) {
                    RereDiffComplex inp = node.inputs.get(i);
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

    @Override public void zeroGradient() { this.gradient = null; }

    @Override
    public IDiffComplex grad() {
        return new RereDiffComplex(this.gradient.copy());
    }

    void accGrad(IComplexVector grad) {
        if (gradient == null) gradient = grad.copy();
        else gradient = gradient.add(grad);
    }

    @Override
    public IDiffComplex add(IDiffComplex other) {
        RereDiffComplex o = (RereDiffComplex) other;
        IComplexVector resultVal = value.add(o.value);
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
            o.accGrad(gradOut);
        };
        return new RereDiffComplex(resultVal, List.of(this, o), backwardFn);
    }

    @Override
    public IDiffComplex sub(IDiffComplex other) {
        RereDiffComplex o = (RereDiffComplex) other;
        IComplexVector resultVal = value.sub(o.value);
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut);
            o.accGrad(gradOut.scale(new Complex(-1, 0)));
        };
        return new RereDiffComplex(resultVal, List.of(this, o), backwardFn);
    }

    @Override
    public IDiffComplex mul(IDiffComplex other) {
        RereDiffComplex o = (RereDiffComplex) other;
        IComplexVector resultVal = value.pointwiseMultiply(o.value);
        IComplexVector aVal = value.copy();
        IComplexVector bVal = o.value.copy();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.pointwiseMultiply(bVal.conjugate()));
            o.accGrad(gradOut.pointwiseMultiply(aVal.conjugate()));
        };
        return new RereDiffComplex(resultVal, List.of(this, o), backwardFn);
    }

    @Override
    public IDiffComplex scale(Complex scalar) {
        IComplexVector resultVal = value.scale(scalar);
        Complex conjScalar = scalar.conjugate();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.scale(conjScalar));
        };
        return new RereDiffComplex(resultVal, List.of(this), backwardFn);
    }

    @Override
    public IDiffComplex conjugate() {
        IComplexVector resultVal = value.conjugate();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.conjugate());
        };
        return new RereDiffComplex(resultVal, List.of(this), backwardFn);
    }

    @Override
    public IDiffComplex exp() {
        Complex[] zData = value.toComplexArray();
        Complex[] expData = new Complex[zData.length];
        for (int i = 0; i < zData.length; i++) expData[i] = zData[i].exp();
        IComplexVector resultVal = IComplexVector.fromComplex(expData);
        IComplexVector expCopy = resultVal.copy();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.pointwiseMultiply(expCopy));
        };
        return new RereDiffComplex(resultVal, List.of(this), backwardFn);
    }

    @Override
    public IDiffComplex log() {
        Complex[] zData = value.toComplexArray();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] grad = new Complex[gd.length];
            for (int i = 0; i < gd.length; i++) {
                grad[i] = gd[i].divide(zData[i]);
            }
            this.accGrad(IComplexVector.fromComplex(grad));
        };
        Complex[] logData = new Complex[zData.length];
        for (int i = 0; i < zData.length; i++) logData[i] = zData[i].log();
        return new RereDiffComplex(IComplexVector.fromComplex(logData), List.of(this), backwardFn);
    }

    @Override
    public IDiffComplex sin() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] sinData = new Complex[n];
        Complex[] cosData = new Complex[n];
        for (int i = 0; i < n; i++) {
            sinData[i] = zData[i].sin();
            cosData[i] = zData[i].cos();
        }
        IComplexVector resultVal = IComplexVector.fromComplex(sinData);
        IComplexVector cosCopy = IComplexVector.fromComplex(cosData);
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.pointwiseMultiply(cosCopy));
        };
        Function<IDiffComplex, IDiffComplex[]> symFn = (gradOut) ->
            new IDiffComplex[] { gradOut.mul(constant(cosCopy)) };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "sin");
    }

    @Override
    public IDiffComplex cos() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] cosData = new Complex[n];
        Complex[] sinData = new Complex[n];
        for (int i = 0; i < n; i++) {
            cosData[i] = zData[i].cos();
            sinData[i] = zData[i].sin();
        }
        IComplexVector resultVal = IComplexVector.fromComplex(cosData);
        IComplexVector sinCopy = IComplexVector.fromComplex(sinData);
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.pointwiseMultiply(sinCopy).scale(new Complex(-1, 0)));
        };
        Function<IDiffComplex, IDiffComplex[]> symFn = (gradOut) ->
            new IDiffComplex[] { gradOut.mul(constant(sinCopy)).scale(new Complex(-1, 0)) };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "cos");
    }

    @Override
    public IDiffComplex tan() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] tanData = new Complex[n];
        for (int i = 0; i < n; i++) tanData[i] = zData[i].sin().divide(zData[i].cos());
        IComplexVector resultVal = IComplexVector.fromComplex(tanData);
        IComplexVector tanCopy = resultVal.copy();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] td = tanCopy.toComplexArray();
            Complex[] dz = new Complex[n];
            for (int i = 0; i < n; i++) {
                dz[i] = gd[i].multiply(td[i].multiply(td[i]).add(Complex.ONE));
            }
            this.accGrad(IComplexVector.fromComplex(dz));
        };
        Function<IDiffComplex, IDiffComplex[]> symFn = (gradOut) ->
            new IDiffComplex[] { gradOut.mul(this.tan().square().add(constant(
                IComplexVector.fromComplex(fillOnes(n))))) };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "tan");
    }

    @Override
    public IDiffComplex tanh() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] tanhData = new Complex[n];
        for (int i = 0; i < n; i++) {
            Complex exp2z = zData[i].scale(2).exp();
            tanhData[i] = exp2z.subtract(Complex.ONE).divide(exp2z.add(Complex.ONE));
        }
        IComplexVector resultVal = IComplexVector.fromComplex(tanhData);
        IComplexVector tanhCopy = resultVal.copy();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] td = tanhCopy.toComplexArray();
            Complex[] dz = new Complex[n];
            for (int i = 0; i < n; i++) {
                dz[i] = gd[i].multiply(new Complex(1, 0).subtract(td[i].multiply(td[i])));
            }
            this.accGrad(IComplexVector.fromComplex(dz));
        };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "tanh");
    }

    @Override
    public IDiffComplex sigmoid() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] sigData = new Complex[n];
        for (int i = 0; i < n; i++) {
            sigData[i] = Complex.ONE.divide(Complex.ONE.add(zData[i].multiply(new Complex(-1, 0)).exp()));
        }
        IComplexVector resultVal = IComplexVector.fromComplex(sigData);
        IComplexVector sigCopy = resultVal.copy();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] sd = sigCopy.toComplexArray();
            Complex[] dz = new Complex[n];
            for (int i = 0; i < n; i++) {
                dz[i] = gd[i].multiply(sd[i]).multiply(new Complex(1, 0).subtract(sd[i]));
            }
            this.accGrad(IComplexVector.fromComplex(dz));
        };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "sigmoid");
    }

    @Override
    public IDiffComplex relu() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] reluData = new Complex[n];
        for (int i = 0; i < n; i++) {
            double re = Math.max(0.0, zData[i].getReal());
            reluData[i] = new Complex(re, zData[i].getImaginary());
        }
        IComplexVector resultVal = IComplexVector.fromComplex(reluData);
        Complex[] zCopy = value.copy().toComplexArray();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] dz = new Complex[n];
            for (int i = 0; i < n; i++) {
                double mask = zCopy[i].getReal() > 0.0 ? 1.0 : 0.0;
                dz[i] = gd[i].multiply(new Complex(mask, 0));
            }
            this.accGrad(IComplexVector.fromComplex(dz));
        };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "relu");
    }

    @Override
    public IDiffComplex abs() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] absData = new Complex[n];
        for (int i = 0; i < n; i++) {
            absData[i] = new Complex(zData[i].abs(), 0);
        }
        IComplexVector resultVal = IComplexVector.fromComplex(absData);
        Complex[] zCopy = value.copy().toComplexArray();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] dz = new Complex[n];
            for (int i = 0; i < n; i++) {
                double mag = zCopy[i].abs();
                if (mag > 0) {
                    dz[i] = gd[i].multiply(zCopy[i].divide(new Complex(mag, 0)));
                } else {
                    dz[i] = Complex.ZERO;
                }
            }
            this.accGrad(IComplexVector.fromComplex(dz));
        };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "abs");
    }

    @Override
    public IDiffComplex sqrt() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] sqrtData = new Complex[n];
        for (int i = 0; i < n; i++) sqrtData[i] = zData[i].sqrt();
        IComplexVector resultVal = IComplexVector.fromComplex(sqrtData);
        IComplexVector sqrtCopy = resultVal.copy();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] sd = sqrtCopy.toComplexArray();
            Complex[] dz = new Complex[n];
            for (int i = 0; i < n; i++) {
                dz[i] = gd[i].divide(sd[i].multiply(new Complex(2, 0)));
            }
            this.accGrad(IComplexVector.fromComplex(dz));
        };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "sqrt");
    }

    @Override
    public IDiffComplex square() {
        Complex[] zData = value.toComplexArray();
        int n = zData.length;
        Complex[] sqData = new Complex[n];
        for (int i = 0; i < n; i++) sqData[i] = zData[i].multiply(zData[i]);
        IComplexVector resultVal = IComplexVector.fromComplex(sqData);
        IComplexVector zCopy = value.copy();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.pointwiseMultiply(zCopy).scale(new Complex(2, 0)));
        };
        Function<IDiffComplex, IDiffComplex[]> symFn = (gradOut) ->
            new IDiffComplex[] { gradOut.mul(this).scale(new Complex(2, 0)) };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "square");
    }

    @Override
    public IDiffComplex neg() {
        IComplexVector resultVal = value.scale(new Complex(-1, 0));
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            this.accGrad(gradOut.scale(new Complex(-1, 0)));
        };
        Function<IDiffComplex, IDiffComplex[]> symFn = (gradOut) ->
            new IDiffComplex[] { gradOut.scale(new Complex(-1, 0)) };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "neg");
    }

    @Override
    public IDiffComplex pow(double n) {
        Complex[] zData = value.toComplexArray();
        int len = zData.length;
        Complex[] powData = new Complex[len];
        for (int i = 0; i < len; i++) {
            if (zData[i].abs() < 1e-15 && n > 0) {
                powData[i] = Complex.ZERO;
            } else {
                powData[i] = zData[i].log().scale(n).exp();
            }
        }
        IComplexVector resultVal = IComplexVector.fromComplex(powData);
        Complex[] zCopy = value.copy().toComplexArray();
        Complex nComplex = new Complex(n, 0);
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] dz = new Complex[len];
            for (int i = 0; i < len; i++) {
                Complex zn1;
                if (zCopy[i].abs() < 1e-15 && n > 1) {
                    zn1 = Complex.ZERO;
                } else {
                    zn1 = zCopy[i].log().scale(n - 1).exp();
                }
                dz[i] = gd[i].multiply(nComplex).multiply(zn1);
            }
            this.accGrad(IComplexVector.fromComplex(dz));
        };
        return withTag(new RereDiffComplex(resultVal, List.of(this), backwardFn), "pow");
    }

    @Override
    public IDiffComplex div(IDiffComplex other) {
        RereDiffComplex o = (RereDiffComplex) other;
        Complex[] aData = value.toComplexArray();
        Complex[] bData = o.value.toComplexArray();
        int n = aData.length;
        Complex[] divData = new Complex[n];
        for (int i = 0; i < n; i++) divData[i] = aData[i].divide(bData[i]);
        IComplexVector resultVal = IComplexVector.fromComplex(divData);
        IComplexVector aVal = value.copy();
        IComplexVector bVal = o.value.copy();
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex[] gd = gradOut.toComplexArray();
            Complex[] av = aVal.toComplexArray();
            Complex[] bv = bVal.toComplexArray();
            Complex[] da = new Complex[n];
            Complex[] db = new Complex[n];
            for (int i = 0; i < n; i++) {
                da[i] = gd[i].divide(bv[i]);
                db[i] = gd[i].multiply(new Complex(-1, 0)).multiply(av[i])
                    .divide(bv[i].multiply(bv[i]));
            }
            this.accGrad(IComplexVector.fromComplex(da));
            o.accGrad(IComplexVector.fromComplex(db));
        };
        return withTag(new RereDiffComplex(resultVal, List.of(this, o), backwardFn), "div");
    }

    private static Complex[] fillOnes(int n) {
        Complex[] ones = new Complex[n];
        for (int i = 0; i < n; i++) ones[i] = Complex.ONE;
        return ones;
    }

    @Override
    public IDiffComplex sum() {
        Complex s = value.sum();
        IComplexVector resultVal = IComplexVector.fromComplex(new Complex[]{s});
        int n = value.length();
        RereDiffComplex self = this;
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex g = gradOut.get(0);
            Complex[] grad = new Complex[n];
            for (int i = 0; i < n; i++) grad[i] = g;
            self.accGrad(IComplexVector.fromComplex(grad));
        };
        return new RereDiffComplex(resultVal, List.of(this), backwardFn);
    }

    @Override
    public IDiffComplex innerProduct(IDiffComplex other) {
        RereDiffComplex o = (RereDiffComplex) other;
        Complex ip = value.innerProduct(o.value);
        IComplexVector aVal = value.copy();
        IComplexVector bVal = o.value.copy();
        IComplexVector resultVal = IComplexVector.fromComplex(new Complex[]{ip});
        RereDiffComplex self = this;
        Consumer<IComplexVector> backwardFn = (gradOut) -> {
            Complex g = gradOut.get(0);
            self.accGrad(bVal.scale(g));
            o.accGrad(aVal.conjugate().scale(g));
        };
        return new RereDiffComplex(resultVal, List.of(this, o), backwardFn);
    }
}
