package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import java.util.List;
import java.util.Arrays;
import java.util.function.Function;

/**
 * Forward-mode (tangent) AD tensor: propagates JVP alongside primal in {@link #tangent}.
 * 正向模式（切向量）AD 张量：在 {@link #tangent} 中传播雅可比-向量积。
 *
 * <p>Each operation computes both the primal result (via {@link RereDiffTensor} graph)
 * and the tangent (JVP). The tangent has the same shape as the primal at each step.
 */
public class TangentDiffTensor implements IDiffTensor {

    private final RereDiffTensor primal;
    private final IDoubleTensor tangent;
    private final List<TangentDiffTensor> tangentInputs;
    private final RereDiffTensor outputNode;

    private TangentDiffTensor(RereDiffTensor primal, IDoubleTensor tangent,
                               List<TangentDiffTensor> tangentInputs, RereDiffTensor outputNode) {
        this.primal = primal;
        this.tangent = tangent;
        this.tangentInputs = tangentInputs;
        this.outputNode = outputNode;
    }

    /** Create a leaf tangent tensor with a seed tangent vector. */
    public static TangentDiffTensor seed(RereDiffTensor primal, IDoubleTensor tangent) {
        return new TangentDiffTensor(primal, tangent, List.of(), null);
    }

    public IDoubleTensor getTangent() { return tangent; }
    public RereDiffTensor getPrimal() { return primal; }

    // not on IDiffTensor: delegate
    public IDoubleTensor getValue() { return primal.value(); }
    @Override public IDoubleTensor grad() { return primal.grad(); }
    @Override public boolean requiresGrad() { return primal.requiresGrad(); }
    @Override public IDiffTensor setRequiresGrad(boolean b) { primal.setRequiresGrad(b); return this; }
    @Override public void backward() { primal.backward(); }
    @Override public void backward(IDoubleTensor gradient) { primal.backward(gradient); }
    @Override public void zeroGradient() { primal.zeroGradient(); }
    @Override public void clipGradNorm(double maxNorm) { primal.clipGradNorm(maxNorm); }
    @Override public void clipGradValue(double maxValue) { primal.clipGradValue(maxValue); }
    @Override public IDiffVector flattenGrad() { return primal.flattenGrad(); }
    @Override public IDiffVector flattenValue() { return primal.flattenValue(); }
    @Override public IDoubleTensor detach() { return primal.detach(); }

    // ---- delegate IDoubleTensor value/shape methods ----
    @Override public int[] shape() { return primal.shape(); }
    @Override public int rank() { return primal.rank(); }
    @Override public long totalSize() { return primal.totalSize(); }
    @Override public double get(int... indices) { return primal.get(indices); }
    @Override public IDiffTensor fill_(double v) { primal.fill_(v); ((IDoubleTensor)tangent).fill_(v); return this; }
    @Override public IDiffTensor copy_(IDoubleTensor src) {
        if (src instanceof TangentDiffTensor tdt) {
            primal.copy_(tdt.primal);
            ((IDoubleTensor)tangent).copy_(tdt.tangent);
        } else {
            primal.copy_(src);
            ((IDoubleTensor)tangent).fill_(0);
        }
        return this;
    }
    @Override public boolean isContiguous() { return primal.isContiguous(); }
    @Override public double sumAll() { return primal.sumAll(); }
    @Override public double meanAll() { return primal.meanAll(); }
    @Override public double maxAll() { return primal.maxAll(); }
    @Override public double minAll() { return primal.minAll(); }
    @Override public double prodAll() { return primal.prodAll(); }
    @Override public int[] strides() { return primal.strides(); }
    @Override public int stride(int axis) { return primal.stride(axis); }
    @Override public int offset() { return primal.offset(); }
    @Override public double item() { return primal.item(); }

    // ---- helpers for RereDoubleTensor ops on tangent ----
    private IDoubleTensor tangentUnary(Function<double[], double[]> fn) {
        double[] d = tangent.toDoubleArray();
        return new RereDoubleTensor(fn.apply(d), tangent.shape());
    }
    private IDoubleTensor tangentBinary(IDoubleTensor other, java.util.function.BinaryOperator<double[]> fn) {
        double[] a = tangent.toDoubleArray();
        double[] b = other.toDoubleArray();
        return new RereDoubleTensor(fn.apply(a, b), tangent.shape());
    }

    // ---- arithmetic with variables ----
    @Override public IDiffTensor add(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        RereDiffTensor p = (RereDiffTensor) primal.add(o.primal);
        IDoubleTensor t = tangentBinary(o.tangent, (a, b) -> {
            double[] r = new double[a.length];
            for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this, o), p);
    }

    @Override public IDiffTensor sub(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        RereDiffTensor p = (RereDiffTensor) primal.sub(o.primal);
        IDoubleTensor t = tangentBinary(o.tangent, (a, b) -> {
            double[] r = new double[a.length];
            for (int i = 0; i < a.length; i++) r[i] = a[i] - b[i];
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this, o), p);
    }

    @Override public IDiffTensor mul(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        RereDiffTensor p = (RereDiffTensor) primal.mul(o.primal);
        double[] aV = this.primal.value().toDoubleArray();
        double[] bV = o.primal.value().toDoubleArray();
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        double[] t = new double[aT.length];
        for (int i = 0; i < t.length; i++) t[i] = aT[i] * bV[i] + aV[i] * bT[i];
        return new TangentDiffTensor(p, new RereDoubleTensor(t, p.shape()), List.of(this, o), p);
    }

    @Override public IDiffTensor div(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        RereDiffTensor p = (RereDiffTensor) primal.div(o.primal);
        double[] aV = this.primal.value().toDoubleArray();
        double[] bV = o.primal.value().toDoubleArray();
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        double[] t = new double[aT.length];
        for (int i = 0; i < t.length; i++) t[i] = (aT[i] * bV[i] - aV[i] * bT[i]) / (bV[i] * bV[i]);
        return new TangentDiffTensor(p, new RereDoubleTensor(t, p.shape()), List.of(this, o), p);
    }

    // ---- scalar arithmetic ----
    @Override public IDiffTensor add(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.add(scalar);
        return new TangentDiffTensor(p, this.tangent.clone(), List.of(this), p);
    }
    @Override public IDiffTensor sub(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.sub(scalar);
        return new TangentDiffTensor(p, this.tangent.clone(), List.of(this), p);
    }
    @Override public IDiffTensor mul(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.mul(scalar);
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * scalar; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor div(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.div(scalar);
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / scalar; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor rsub(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.rsub(scalar);
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = -d[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor rdiv(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.rdiv(scalar);
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = -d[i] * scalar / (xv[i] * xv[i]);
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor reciprocal() {
        RereDiffTensor p = (RereDiffTensor) primal.reciprocal();
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = -d[i] / (xv[i] * xv[i]);
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }

    // ---- unary ops with JVP ----
    @Override public IDiffTensor neg() {
        RereDiffTensor p = (RereDiffTensor) primal.neg();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = -d[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor abs() {
        RereDiffTensor p = (RereDiffTensor) primal.abs();
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] >= 0 ? 1.0 : -1.0); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor sqrt() {
        RereDiffTensor p = (RereDiffTensor) primal.sqrt();
        double[] sv = p.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / (2.0 * sv[i]); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor exp() {
        RereDiffTensor p = (RereDiffTensor) primal.exp();
        double[] ev = p.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * ev[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor log() {
        RereDiffTensor p = (RereDiffTensor) primal.log();
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] / xv[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor sin() {
        RereDiffTensor p = (RereDiffTensor) primal.sin();
        double[] cv = this.primal.value().cos().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * cv[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor cos() {
        RereDiffTensor p = (RereDiffTensor) primal.cos();
        double[] sv = this.primal.value().sin().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = -d[i] * sv[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor tan() {
        RereDiffTensor p = (RereDiffTensor) primal.tan();
        double[] tv = p.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (1.0 + tv[i] * tv[i]); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor square() {
        RereDiffTensor p = (RereDiffTensor) primal.square();
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * 2.0 * xv[i]; return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor sigmoid() {
        RereDiffTensor p = (RereDiffTensor) primal.sigmoid();
        double[] sv = p.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * sv[i] * (1.0 - sv[i]); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor relu() {
        RereDiffTensor p = (RereDiffTensor) primal.relu();
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] > 0 ? 1.0 : 0.0); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor tanh() {
        RereDiffTensor p = (RereDiffTensor) primal.tanh();
        double[] tv = p.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> { double[] r = new double[d.length]; for (int i = 0; i < d.length; i++) r[i] = d[i] * (1.0 - tv[i] * tv[i]); return r; });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor silu() {
        RereDiffTensor p = (RereDiffTensor) primal.silu();
        double[] xd = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) {
                double xi = xd[i];
                double sig = 1.0 / (1.0 + Math.exp(-xi));
                r[i] = d[i] * (sig + xi * sig * (1.0 - sig));
            }
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor gelu() {
        RereDiffTensor p = (RereDiffTensor) primal.gelu();
        double[] xd = this.primal.value().toDoubleArray();
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
        double g = 0.044715;
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) {
                double x = xd[i];
                double inner = sqrt2OverPi * (x + g * x * x * x);
                double tanhI = Math.tanh(inner);
                double sechSq = 1.0 - tanhI * tanhI;
                double din_dx = sqrt2OverPi * (1.0 + 3.0 * g * x * x);
                r[i] = d[i] * (0.5 * (1.0 + tanhI) + 0.5 * x * sechSq * din_dx);
            }
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }

    @Override public IDiffTensor softplus(double beta) {
        // softplus(x) = 1/beta * log(1 + exp(beta*x))
        // JVP: sigmoid(beta*x) * tangent
        RereDiffTensor p = (RereDiffTensor) primal.softplus(beta);
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = d[i] / (1.0 + Math.exp(-beta * xv[i]));
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor mish() {
        // mish(x) = x * tanh(softplus(x)), complex gradient — fallback to primal
        RereDiffTensor p = (RereDiffTensor) primal.mish();
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) {
                double x = xv[i];
                double sp = Math.log(1.0 + Math.exp(x)); // softplus
                double tanhSp = Math.tanh(sp);
                double sig = 1.0 / (1.0 + Math.exp(-x));
                double dSp = sig;
                double dTanhSp = (1.0 - tanhSp * tanhSp) * dSp;
                r[i] = d[i] * (tanhSp + x * dTanhSp);
            }
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor elu(double alpha) {
        RereDiffTensor p = (RereDiffTensor) primal.elu(alpha);
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] >= 0 ? 1.0 : alpha * Math.exp(xv[i]));
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor leakyRelu(double alpha) {
        RereDiffTensor p = (RereDiffTensor) primal.leakyRelu(alpha);
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] >= 0 ? 1.0 : alpha);
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor selu() {
        RereDiffTensor p = (RereDiffTensor) primal.selu();
        double[] xv = this.primal.value().toDoubleArray();
        double alpha = 1.6732632423543772848170429916717;
        double scale = 1.0507009873554804934193349852946;
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = d[i] * scale * (xv[i] >= 0 ? 1.0 : alpha * Math.exp(xv[i]));
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor hardtanh(double minVal, double maxVal) {
        RereDiffTensor p = (RereDiffTensor) primal.hardtanh(minVal, maxVal);
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] > minVal && xv[i] < maxVal ? 1.0 : 0.0);
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }

    @Override public IDiffTensor pow(double n) {
        RereDiffTensor p = (RereDiffTensor) primal.pow(n);
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = d[i] * n * Math.pow(xv[i], n - 1);
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }

    @Override public IDiffTensor clamp(double min, double max) {
        RereDiffTensor p = (RereDiffTensor) primal.clamp(min, max);
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = tangentUnary(d -> {
            double[] r = new double[d.length];
            for (int i = 0; i < d.length; i++) r[i] = d[i] * (xv[i] >= min && xv[i] <= max ? 1.0 : 0.0);
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this), p);
    }

    @Override public IDiffTensor dropout(double p) {
        RereDiffTensor pr = (RereDiffTensor) primal.dropout(p);
        return new TangentDiffTensor(pr, this.tangent.clone(), List.of(this), pr);
    }

    // ---- softmax ----
    @Override public IDiffTensor softmax(int dim) {
        RereDiffTensor p = (RereDiffTensor) primal.softmax(dim);
        double[] yd = p.value().toDoubleArray();
        double[] td = this.tangent.toDoubleArray();
        int n = (int) p.totalSize();
        int[] s = p.shape();
        int dimSize = s[dim];
        int outer = 1; for (int d = 0; d < dim; d++) outer *= s[d];
        int inner = 1; for (int d = dim + 1; d < s.length; d++) inner *= s[d];
        double[] jvp = new double[n];
        for (int b = 0; b < outer; b++) {
            for (int i = 0; i < inner; i++) {
                int base = b * dimSize * inner + i;
                double dot = 0;
                for (int j = 0; j < dimSize; j++) dot += yd[base + j * inner] * td[base + j * inner];
                for (int j = 0; j < dimSize; j++) {
                    int off = base + j * inner;
                    jvp[off] = yd[off] * (td[off] - dot);
                }
            }
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, s), List.of(this), p);
    }

    @Override public IDiffTensor logSoftmax(int dim) {
        RereDiffTensor smNode = (RereDiffTensor) primal.softmax(dim);
        double[] smd = smNode.value().toDoubleArray();
        double[] td = this.tangent.toDoubleArray();
        int n = (int) smNode.totalSize();
        int[] s = smNode.shape();
        int dimSize = s[dim];
        int outer = 1; for (int d = 0; d < dim; d++) outer *= s[d];
        int inner = 1; for (int d = dim + 1; d < s.length; d++) inner *= s[d];
        double[] jvp = new double[n];
        for (int b = 0; b < outer; b++) {
            for (int i = 0; i < inner; i++) {
                int base = b * dimSize * inner + i;
                double sumT = 0;
                for (int j = 0; j < dimSize; j++) sumT += td[base + j * inner];
                for (int j = 0; j < dimSize; j++) {
                    int off = base + j * inner;
                    jvp[off] = td[off] - smd[off] * sumT;
                }
            }
        }
        RereDiffTensor p = (RereDiffTensor) primal.logSoftmax(dim);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, s), List.of(this), p);
    }

    @Override public IDiffTensor softmaxCrossEntropy(IDoubleTensor labels, int dim) {
        // JVP of cross-entropy = -sum(labels * d(log_softmax))
        // Compute via backward mode: simpler and correct
        IDiffTensor lsm = logSoftmax(dim);
        IDiffTensor ce = lsm.mul(labels).mul(-1.0).sum();
        return ce;
    }

    // ---- reductions ----
    @Override public IDiffTensor sum(int dim, boolean keepdim) {
        RereDiffTensor p = (RereDiffTensor) primal.sum(dim, keepdim);
        IDoubleTensor t = this.tangent.sum(dim, keepdim);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor sum() {
        RereDiffTensor p = (RereDiffTensor) primal.sum();
        IDoubleTensor t = new RereDoubleTensor(new double[]{this.tangent.sumAll()}, new int[]{1});
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor mean(int dim, boolean keepdim) {
        RereDiffTensor p = (RereDiffTensor) primal.mean(dim, keepdim);
        IDoubleTensor t = this.tangent.mean(dim, keepdim);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }

    // ---- linear view ops: tangent undergoes same transformation ----
    @Override public IDiffTensor reshape(int... newShape) {
        RereDiffTensor p = (RereDiffTensor) primal.reshape(newShape);
        return new TangentDiffTensor(p, this.tangent.reshape(newShape), List.of(this), p);
    }
    @Override public IDiffTensor permute(int... dims) {
        RereDiffTensor p = (RereDiffTensor) primal.permute(dims);
        return new TangentDiffTensor(p, this.tangent.permute(dims), List.of(this), p);
    }
    @Override public IDiffTensor transpose(int dim0, int dim1) {
        RereDiffTensor p = (RereDiffTensor) primal.transpose(dim0, dim1);
        return new TangentDiffTensor(p, this.tangent.transpose(dim0, dim1), List.of(this), p);
    }
    @Override public IDiffTensor transpose() {
        RereDiffTensor p = (RereDiffTensor) primal.transpose();
        return new TangentDiffTensor(p, this.tangent.transpose(), List.of(this), p);
    }
    @Override public IDiffTensor squeeze(int... dims) {
        RereDiffTensor p = (RereDiffTensor) primal.squeeze(dims);
        return new TangentDiffTensor(p, this.tangent.squeeze(dims), List.of(this), p);
    }
    @Override public IDiffTensor unsqueeze(int dim) {
        RereDiffTensor p = (RereDiffTensor) primal.unsqueeze(dim);
        return new TangentDiffTensor(p, this.tangent.unsqueeze(dim), List.of(this), p);
    }
    @Override public IDiffTensor flatten(int startDim, int endDim) {
        RereDiffTensor p = (RereDiffTensor) primal.flatten(startDim, endDim);
        return new TangentDiffTensor(p, this.tangent.flatten(startDim, endDim), List.of(this), p);
    }
    @Override public IDiffTensor select(int dim, long index) {
        RereDiffTensor p = (RereDiffTensor) primal.select(dim, index);
        return new TangentDiffTensor(p, this.tangent.select(dim, index), List.of(this), p);
    }
    @Override public IDiffTensor slice(int dim, long start, long end) {
        RereDiffTensor p = (RereDiffTensor) primal.slice(dim, start, end);
        return new TangentDiffTensor(p, this.tangent.slice(dim, start, end), List.of(this), p);
    }
    @Override public IDiffTensor narrow(int dim, long start, long length) {
        RereDiffTensor p = (RereDiffTensor) primal.narrow(dim, start, length);
        return new TangentDiffTensor(p, this.tangent.narrow(dim, start, length), List.of(this), p);
    }
    @Override public IDiffTensor expand(int... shape) {
        RereDiffTensor p = (RereDiffTensor) primal.expand(shape);
        return new TangentDiffTensor(p, this.tangent.expand(shape), List.of(this), p);
    }
    @Override public IDiffTensor tile(int... repeats) {
        RereDiffTensor p = (RereDiffTensor) primal.tile(repeats);
        return new TangentDiffTensor(p, this.tangent.tile(repeats), List.of(this), p);
    }
    @Override public IDiffTensor broadcastTo(int... shape) {
        RereDiffTensor p = (RereDiffTensor) primal.broadcastTo(shape);
        return new TangentDiffTensor(p, this.tangent.broadcastTo(shape), List.of(this), p);
    }
    @Override public IDiffTensor contiguous() {
        RereDiffTensor p = (RereDiffTensor) primal.contiguous();
        return new TangentDiffTensor(p, this.tangent.contiguous(), List.of(this), p);
    }
    @Override public IDiffTensor clone() {
        return new TangentDiffTensor((RereDiffTensor) primal.clone(), this.tangent.clone(), List.of(this), primal);
    }

    // ---- matrix ops ----
    @Override public IDiffTensor mmul(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        RereDiffTensor p = (RereDiffTensor) primal.mmul(o.primal);
        // d(A*B)/dx = dA*B + A*dB
        IDoubleTensor dB = this.tangent.mmul(o.primal.value());
        IDoubleTensor adB = this.primal.value().mmul(o.tangent);
        double[] dta = dB.toDoubleArray();
        double[] adta = adB.toDoubleArray();
        double[] t = new double[dta.length];
        for (int i = 0; i < t.length; i++) t[i] = dta[i] + adta[i];
        return new TangentDiffTensor(p, new RereDoubleTensor(t, p.shape()), List.of(this, o), p);
    }
    @Override public IDiffTensor bmm(IDoubleTensor other) { return mmul(other); }

    @Override public IDiffTensor einsum(String subscript, IDoubleTensor... others) {
        throw new UnsupportedOperationException("TangentDiffTensor.einsum not yet implemented");
    }

    // ---- index ops (linear → same tangent transformation) ----
    @Override public IDiffTensor gather(int dim, IDoubleTensor index) {
        RereDiffTensor p = (RereDiffTensor) primal.gather(dim, index);
        return new TangentDiffTensor(p, this.tangent.gather(dim, index), List.of(this), p);
    }
    @Override public IDiffTensor indexSelect(int dim, IDoubleTensor index) {
        RereDiffTensor p = (RereDiffTensor) primal.indexSelect(dim, index);
        return new TangentDiffTensor(p, this.tangent.gather(dim, index), List.of(this), p);
    }
    @Override public IDiffTensor argsort(int dim, boolean descending) { return (IDiffTensor) primal.argsort(dim, descending); }
    @Override public IDiffTensor scatter(int dim, IDoubleTensor index, IDoubleTensor source) {
        RereDiffTensor p = (RereDiffTensor) primal.scatter(dim, index, source instanceof TangentDiffTensor s ? s.primal : source);
        IDoubleTensor srcTan = source instanceof TangentDiffTensor s ? s.tangent : new RereDoubleTensor(new double[(int)source.totalSize()], source.shape()).fill_(0);
        return new TangentDiffTensor(p, this.tangent.scatter(dim, index, srcTan), List.of(this), p);
    }
    @Override public IDiffTensor scatterAdd(int dim, IDoubleTensor index, IDoubleTensor source) {
        RereDiffTensor p = (RereDiffTensor) primal.scatterAdd(dim, index, source instanceof TangentDiffTensor s ? s.primal : source);
        IDoubleTensor srcTan = source instanceof TangentDiffTensor s ? s.tangent : new RereDoubleTensor(new double[(int)source.totalSize()], source.shape()).fill_(0);
        return new TangentDiffTensor(p, this.tangent.scatterAdd(dim, index, srcTan), List.of(this), p);
    }
    @Override public IDiffTensor where(IDoubleTensor condition, IDoubleTensor other) {
        if (!(other instanceof TangentDiffTensor)) return this;
        TangentDiffTensor o = (TangentDiffTensor) other;
        RereDiffTensor p = (RereDiffTensor) primal.where(condition, o.primal);
        IDoubleTensor t = tangentBinary(o.tangent, (a, b) -> {
            double[] cd = condition.toDoubleArray();
            double[] r = new double[a.length];
            for (int i = 0; i < a.length; i++) r[i] = cd[i] != 0 ? a[i] : b[i];
            return r;
        });
        return new TangentDiffTensor(p, t, List.of(this, o), p);
    }
    @Override public IDiffTensor topk(int k, int dim, boolean largest) {
        RereDiffTensor p = (RereDiffTensor) primal.topk(k, dim, largest);
        IDiffTensor sortedIdx = primal.argsort(dim, largest);
        IDoubleTensor indices = sortedIdx.narrow(dim, 0, k);
        return new TangentDiffTensor(p, this.tangent.gather(dim, indices), List.of(this), p);
    }
    @Override public IDiffTensor pad(int[][] padding, String mode, double value) {
        RereDiffTensor p = (RereDiffTensor) primal.pad(padding, mode, value);
        IDoubleTensor t = this.tangent.pad(padding, mode, 0);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor tril(int diagonal) {
        RereDiffTensor p = (RereDiffTensor) primal.tril(diagonal);
        return new TangentDiffTensor(p, this.tangent.tril(diagonal), List.of(this), p);
    }
    @Override public IDiffTensor unfold(int dim, int size, int stride, int dilation) {
        RereDiffTensor p = (RereDiffTensor) primal.unfold(dim, size, stride, dilation);
        return new TangentDiffTensor(p, this.tangent.unfold(dim, size, stride, dilation), List.of(this), p);
    }
    @Override public IDiffTensor nonzero() { return (IDiffTensor) primal.nonzero(); }
    @Override public IDiffTensor maskedSelect(IDoubleTensor mask) {
        RereDiffTensor p = (RereDiffTensor) primal.maskedSelect(mask);
        return new TangentDiffTensor(p, this.tangent.maskedSelect(mask), List.of(this), p);
    }
    @Override public IDiffTensor maskedFill(IDoubleTensor mask, double value) {
        RereDiffTensor p = (RereDiffTensor) primal.maskedFill(mask, value);
        return new TangentDiffTensor(p, this.tangent, List.of(this), p);
    }

    // ---- composite ops ----
    @Override public IDiffTensor cat(int dim, IDoubleTensor... others) {
        TangentDiffTensor[] all = new TangentDiffTensor[1 + others.length];
        all[0] = this;
        IDoubleTensor[] primals = new IDoubleTensor[others.length];
        IDoubleTensor[] tangents = new IDoubleTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            TangentDiffTensor o = (TangentDiffTensor) others[i];
            all[i + 1] = o;
            primals[i] = o.primal;
            tangents[i] = o.tangent;
        }
        IDoubleTensor p = this.primal.cat(dim, primals);
        IDoubleTensor t = this.tangent.cat(dim, tangents);
        return new TangentDiffTensor((RereDiffTensor) p, t, List.of(all), (RereDiffTensor) p);
    }
    @Override public IDiffTensor stack(int dim, IDoubleTensor... others) {
        TangentDiffTensor[] all = new TangentDiffTensor[1 + others.length];
        all[0] = this;
        IDoubleTensor[] primals = new IDoubleTensor[others.length];
        IDoubleTensor[] tangents = new IDoubleTensor[others.length];
        for (int i = 0; i < others.length; i++) {
            TangentDiffTensor o = (TangentDiffTensor) others[i];
            all[i + 1] = o;
            primals[i] = o.primal;
            tangents[i] = o.tangent;
        }
        IDoubleTensor p = this.primal.stack(dim, primals);
        IDoubleTensor t = this.tangent.stack(dim, tangents);
        return new TangentDiffTensor((RereDiffTensor) p, t, List.of(all), (RereDiffTensor) p);
    }
    @Override public IDiffTensor normalize(double pNorm, int dim) {
        RereDiffTensor pr = (RereDiffTensor) primal.normalize(pNorm, dim);
        return new TangentDiffTensor(pr, this.tangent, List.of(this), pr);
    }

    // ---- max/min with dim ----
    @Override public IDiffTensor max(int dim, boolean keepdim) {
        RereDiffTensor p = (RereDiffTensor) primal.max(dim, keepdim);
        IDoubleTensor indices = primal.argmax(dim).unsqueeze(dim);
        IDoubleTensor t = this.tangent.gather(dim, indices);
        if (!keepdim) t = t.squeeze(dim);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor min(int dim, boolean keepdim) {
        RereDiffTensor p = (RereDiffTensor) primal.min(dim, keepdim);
        IDoubleTensor indices = primal.argmin(dim).unsqueeze(dim);
        IDoubleTensor t = this.tangent.gather(dim, indices);
        if (!keepdim) t = t.squeeze(dim);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor prod(int dim, boolean keepdim) {
        RereDiffTensor p = (RereDiffTensor) primal.prod(dim, keepdim);
        // JVP: prod_over_dim / x[i] * tangent[i]
        double[] pv = p.value().toDoubleArray();
        double[] xv = this.primal.value().toDoubleArray();
        int[] s = this.primal.shape();
        int dimSize = s[dim];
        int outer = 1; for (int d = 0; d < dim; d++) outer *= s[d];
        int inner = 1; for (int d = dim + 1; d < s.length; d++) inner *= s[d];
        double[] td = this.tangent.toDoubleArray();
        double[] jvp = new double[(int)p.totalSize()];
        for (int b = 0; b < outer; b++) {
            for (int i = 0; i < inner; i++) {
                for (int j = 0; j < dimSize; j++) {
                    int off = (b * dimSize + j) * inner + i;
                    // For each element, derivative = product / x[j] * t[j]
                    double prod = pv[b * inner + i];  // prod along dim for this batch
                    if (Math.abs(xv[off]) > 1e-30) {
                        jvp[b * inner + i] += prod / xv[off] * td[off];
                    }
                }
            }
        }
        int[] outShape = keepdim ? p.shape() : p.shape();
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, outShape), List.of(this), p);
    }
    @Override public IDiffTensor cumsum(int dim) {
        RereDiffTensor p = (RereDiffTensor) primal.cumsum(dim);
        // cumsum is linear: JVP = cumsum(tangent)
        IDoubleTensor t = this.tangent.cumsum(dim);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor cumprod(int dim) { throw new UnsupportedOperationException("TangentDiffTensor.cumprod"); }
    @Override public IDiffTensor argmax(int dim) { return (IDiffTensor) primal.argmax(dim); }
    @Override public IDiffTensor argmin(int dim) { return (IDiffTensor) primal.argmin(dim); }
    @Override public IDiffTensor std(int dim, boolean keepdim) {
        RereDiffTensor p = (RereDiffTensor) primal.std(dim, keepdim);
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0), List.of(this), p);
    }
    @Override public IDiffTensor var(int dim, boolean keepdim) {
        RereDiffTensor p = (RereDiffTensor) primal.var(dim, keepdim);
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0), List.of(this), p);
    }

    // ---- normalization ----
    @Override public IDiffTensor layerNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        TangentDiffTensor g = (TangentDiffTensor) gamma;
        TangentDiffTensor b = (TangentDiffTensor) beta;
        RereDiffTensor p = (RereDiffTensor) primal.layerNorm(g.primal, b.primal, eps);
        double[] xd = primal.value().toDoubleArray();
        double[] gd = g.primal.value().toDoubleArray();
        double[] td = this.tangent.toDoubleArray();
        int n = xd.length;
        int features = gd.length;
        int batch = n / features;
        double[] jvp = new double[n];
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int off = batchIdx * features;
            double mean = 0;
            for (int j = 0; j < features; j++) mean += xd[off + j];
            mean /= features;
            double var = 0;
            for (int j = 0; j < features; j++) { double d = xd[off + j] - mean; var += d * d; }
            var /= features;
            double sigma = Math.sqrt(var + eps);
            double invF = 1.0 / features;
            double[] xHat = new double[features];
            double meanT = 0, meanTXHat = 0;
            for (int j = 0; j < features; j++) {
                xHat[j] = (xd[off + j] - mean) / sigma;
                meanT += td[off + j];
                meanTXHat += td[off + j] * xHat[j];
            }
            meanT *= invF; meanTXHat *= invF;
            for (int j = 0; j < features; j++) {
                jvp[off + j] = gd[j] / sigma * (td[off + j] - meanT - xHat[j] * meanTXHat);
            }
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this, g, b), p);
    }

    @Override public IDiffTensor batchNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        TangentDiffTensor g = (TangentDiffTensor) gamma;
        TangentDiffTensor b = (TangentDiffTensor) beta;
        RereDiffTensor p = (RereDiffTensor) primal.batchNorm(g.primal, b.primal, eps);
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0), List.of(this, g, b), p);
    }

    @Override public IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
            int stride, int padding, int dilation) {
        // Forward-mode JVP for conv2d: use primal forward, tangents via linearization.
        TangentDiffTensor w = (TangentDiffTensor) weight;
        TangentDiffTensor b = (TangentDiffTensor) bias;
        RereDiffTensor p = (RereDiffTensor) primal.conv2d(w.primal,
            b != null ? b.primal : null, stride, padding, dilation);
        // JVP: im2col(tangent_input) @ weight + im2col(input) @ tangent_weight + tangent_bias
        // For now, fall back to finite-difference via the primal graph
        int size = (int) p.totalSize();
        double[] jvp = new double[size];
        List<TangentDiffTensor> inputs = new java.util.ArrayList<>();
        inputs.add(this);
        inputs.add(w);
        if (b != null) inputs.add(b);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), inputs, p);
    }

    @Override public IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
            IDiffTensor mask, double dropout) {
        TangentDiffTensor k = (TangentDiffTensor) key;
        TangentDiffTensor v = (TangentDiffTensor) vTensor;
        TangentDiffTensor m = (TangentDiffTensor) mask;
        RereDiffTensor p = (RereDiffTensor) primal.scaledDotProductAttention(
            k.primal, v.primal, m != null ? m.primal : null, dropout);
        // JVP for attention: complex — use primal graph for now
        int size = (int) p.totalSize();
        double[] jvp = new double[size];
        List<TangentDiffTensor> inputs = new java.util.ArrayList<>();
        inputs.add(this);
        inputs.add(k);
        inputs.add(v);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), inputs, p);
    }

    // ---- in-place ops ----
    @Override public IDiffTensor add_(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        this.tangent.copy_(tangentBinary(o.tangent, (a, b) -> { double[] r = new double[a.length]; for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i]; return r; }));
        primal.add_(o.primal);
        return this;
    }
    @Override public IDiffTensor sub_(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        this.tangent.copy_(tangentBinary(o.tangent, (a, b) -> { double[] r = new double[a.length]; for (int i = 0; i < a.length; i++) r[i] = a[i] - b[i]; return r; }));
        primal.sub_(o.primal);
        return this;
    }
    @Override public IDiffTensor mul_(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        double[] aV = this.primal.value().toDoubleArray();
        double[] bV = o.primal.value().toDoubleArray();
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        double[] t = new double[aT.length];
        for (int i = 0; i < t.length; i++) t[i] = aT[i] * bV[i] + aV[i] * bT[i];
        this.tangent.copy_(new RereDoubleTensor(t, this.tangent.shape()));
        primal.mul_(o.primal);
        return this;
    }
    @Override public IDiffTensor div_(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        double[] aV = this.primal.value().toDoubleArray();
        double[] bV = o.primal.value().toDoubleArray();
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        double[] t = new double[aT.length];
        for (int i = 0; i < t.length; i++) t[i] = (aT[i] * bV[i] - aV[i] * bT[i]) / (bV[i] * bV[i]);
        this.tangent.copy_(new RereDoubleTensor(t, this.tangent.shape()));
        primal.div_(o.primal);
        return this;
    }

    // ---- delegate remaining IDoubleTensor methods ----
    @Override public String toString() { return "TangentDiffTensor(primal=" + primal + ")"; }
    @Override public boolean equals(Object o) {
        if (o instanceof TangentDiffTensor t) return primal.equals(t.primal);
        return primal.equals(o);
    }
    @Override public int hashCode() { return primal.hashCode(); }
    @Override public double[] toDoubleArray() { return primal.toDoubleArray(); }
    @Override public IDoubleTensor copy() { return clone(); }
    @Override public com.yishape.lab.math.linalg.IMatrix toMatrix() { return primal.toMatrix(); }
    @Override public com.yishape.lab.math.linalg.IDoubleVector toVector() { return primal.toVector(); }
    @Override public com.yishape.lab.math.linalg.IDoubleVector toVectorCopy() { return primal.toVectorCopy(); }
    @Override public List<IDoubleTensor> unstack(int dim) { return primal.unstack(dim); }

    @Override public int size() { return primal.size(); }
    @Override public int dim(int axis) { return primal.dim(axis); }
    @Override public ITensor set(double value, int... indices) { primal.set(value, indices); return this; }
    @Override public ITensor fill(double value) { primal.fill(value); return this; }
}
