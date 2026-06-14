package com.yishape.lab.math.autodiff.impl;

import com.yishape.lab.math.autodiff.IDiffTensor;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.tensor.IDoubleTensor;
import com.yishape.lab.math.linalg.tensor.RereDoubleTensor;
import com.yishape.lab.math.linalg.tensor.ITensor;
import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.gpu.GpuActivation;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.UniversalOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.compute.DoubleFlatGemm;
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
    private static final DoubleVectorComputer COMPUTER = new DoubleVectorComputer();

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
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        IDoubleTensor t = new RereDoubleTensor(COMPUTER.binaryOperate(aT, bT, BinaryOperation.ADD), p.shape());
        return new TangentDiffTensor(p, t, List.of(this, o), p);
    }

    @Override public IDiffTensor sub(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        RereDiffTensor p = (RereDiffTensor) primal.sub(o.primal);
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        double[] negBT = COMPUTER.binaryOperate(bT, -1.0, BinaryOperation.MULTIPLY);
        IDoubleTensor t = new RereDiffTensor(COMPUTER.binaryOperate(aT, negBT, BinaryOperation.ADD), p.shape());
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
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), scalar, BinaryOperation.MULTIPLY), tangent.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor div(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.div(scalar);
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), scalar, BinaryOperation.DIVIDE), tangent.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor rsub(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.rsub(scalar);
        double[] zero = COMPUTER.fill((int) tangent.totalSize(), 0.0);
        double[] negT = COMPUTER.binaryOperate(zero, tangent.toDoubleArray(), BinaryOperation.SUBTRACT);
        return new TangentDiffTensor(p, new RereDoubleTensor(negT, p.shape()), List.of(this), p);
    }
    @Override public IDiffTensor rdiv(double scalar) {
        RereDiffTensor p = (RereDiffTensor) primal.rdiv(scalar);
        double[] tv = tangent.toDoubleArray();
        double[] xv = primal.value().toDoubleArray();
        // Guard: JVP = -tangent * scalar / x²; clamp near-zero x to prevent Infinity.
        // Compute via -scalar * exp(-log(x²+eps)): no accelerated reciprocal primitive exists.
        double[] xvSq = COMPUTER.binaryOperate(xv, xv, BinaryOperation.MULTIPLY);
        double[] denom = COMPUTER.binaryOperate(xvSq, 1e-15, BinaryOperation.ADD);
        double[] logD = COMPUTER.universalOperate(denom, UniversalOperation.LOG, 0);
        double[] zero = COMPUTER.fill(logD.length, 0.0);
        double[] negLogD = COMPUTER.binaryOperate(zero, logD, BinaryOperation.SUBTRACT);
        double[] invD = COMPUTER.universalOperate(negLogD, UniversalOperation.EXP, 0);
        double[] scaledInv = COMPUTER.binaryOperate(invD, -scalar, BinaryOperation.MULTIPLY);
        double[] r = COMPUTER.binaryOperate(tv, scaledInv, BinaryOperation.MULTIPLY);
        // JVP of scalar/x is -scalar*tangent/x²: negate the result
        double[] rNeg = COMPUTER.binaryOperate(zero, r, BinaryOperation.SUBTRACT);
        return new TangentDiffTensor(p, new RereDoubleTensor(rNeg, p.shape()), List.of(this), p);
    }
    @Override public IDiffTensor reciprocal() {
        RereDiffTensor p = (RereDiffTensor) primal.reciprocal();
        double[] tv = tangent.toDoubleArray();
        double[] xv = primal.value().toDoubleArray();
        // Guard: JVP = -tangent / x²; clamp near-zero x to prevent Infinity.
        // Compute via -exp(-log(x²+eps)): no accelerated reciprocal primitive exists.
        double[] xvSq = COMPUTER.binaryOperate(xv, xv, BinaryOperation.MULTIPLY);
        double[] denom = COMPUTER.binaryOperate(xvSq, 1e-15, BinaryOperation.ADD);
        double[] logD = COMPUTER.universalOperate(denom, UniversalOperation.LOG, 0);
        double[] zero = COMPUTER.fill(logD.length, 0.0);
        double[] negLogD = COMPUTER.binaryOperate(zero, logD, BinaryOperation.SUBTRACT);
        double[] invD = COMPUTER.universalOperate(negLogD, UniversalOperation.EXP, 0);
        double[] r = COMPUTER.binaryOperate(tv, invD, BinaryOperation.MULTIPLY);
        // JVP of 1/x is -tangent/x²: negate the result
        double[] rNeg = COMPUTER.binaryOperate(zero, r, BinaryOperation.SUBTRACT);
        return new TangentDiffTensor(p, new RereDoubleTensor(rNeg, p.shape()), List.of(this), p);
    }

    // ---- unary ops with JVP ----
    @Override public IDiffTensor neg() {
        RereDiffTensor p = (RereDiffTensor) primal.neg();
        double[] zero = COMPUTER.fill((int) tangent.totalSize(), 0.0);
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(zero, tangent.toDoubleArray(), BinaryOperation.SUBTRACT), p.shape());
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
        double[] tv = tangent.toDoubleArray();
        // Guard: JVP = tangent / (2*sv); clamp sv near zero to prevent Infinity.
        // JVP = tv / (2*sv + eps) = tv * exp(-log(2*sv + eps))
        double[] twoSv = COMPUTER.binaryOperate(sv, 2.0, BinaryOperation.MULTIPLY);
        twoSv = COMPUTER.binaryOperate(twoSv, 1e-15, BinaryOperation.ADD);
        double[] logD = COMPUTER.universalOperate(twoSv, UniversalOperation.LOG, 0);
        double[] zero = COMPUTER.fill(logD.length, 0.0);
        double[] negLogD = COMPUTER.binaryOperate(zero, logD, BinaryOperation.SUBTRACT);
        double[] invD = COMPUTER.universalOperate(negLogD, UniversalOperation.EXP, 0);
        double[] r = COMPUTER.binaryOperate(tv, invD, BinaryOperation.MULTIPLY);
        return new TangentDiffTensor(p, new RereDoubleTensor(r, p.shape()), List.of(this), p);
    }
    @Override public IDiffTensor exp() {
        RereDiffTensor p = (RereDiffTensor) primal.exp();
        double[] ev = p.value().toDoubleArray();
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), ev, BinaryOperation.MULTIPLY), p.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor log() {
        RereDiffTensor p = (RereDiffTensor) primal.log();
        double[] xv = this.primal.value().toDoubleArray();
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), xv, BinaryOperation.DIVIDE), p.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor sin() {
        RereDiffTensor p = (RereDiffTensor) primal.sin();
        double[] cv = this.primal.value().cos().toDoubleArray();
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), cv, BinaryOperation.MULTIPLY), p.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor cos() {
        RereDiffTensor p = (RereDiffTensor) primal.cos();
        double[] sv = this.primal.value().sin().toDoubleArray();
        double[] negSv = COMPUTER.binaryOperate(sv, -1.0, BinaryOperation.MULTIPLY);
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), negSv, BinaryOperation.MULTIPLY), p.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor tan() {
        RereDiffTensor p = (RereDiffTensor) primal.tan();
        double[] tv = p.value().toDoubleArray();
        double[] tvSq = COMPUTER.binaryOperate(tv, tv, BinaryOperation.MULTIPLY);
        double[] onePlus = COMPUTER.binaryOperate(tvSq, 1.0, BinaryOperation.ADD);
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), onePlus, BinaryOperation.MULTIPLY), p.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor square() {
        RereDiffTensor p = (RereDiffTensor) primal.square();
        double[] xv = this.primal.value().toDoubleArray();
        double[] twoXv = COMPUTER.binaryOperate(xv, 2.0, BinaryOperation.MULTIPLY);
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), twoXv, BinaryOperation.MULTIPLY), p.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor sigmoid() {
        RereDiffTensor p = (RereDiffTensor) primal.sigmoid();
        double[] sv = p.value().toDoubleArray();
        double[] oneMinusSv = COMPUTER.binaryOperate(sv, -1.0, BinaryOperation.MULTIPLY);
        oneMinusSv = COMPUTER.binaryOperate(oneMinusSv, 1.0, BinaryOperation.ADD); // 1 - sv
        double[] svFactor = COMPUTER.binaryOperate(sv, oneMinusSv, BinaryOperation.MULTIPLY); // sv * (1 - sv)
        IDoubleTensor t = new RereDoubleTensor(
            COMPUTER.binaryOperate(tangent.toDoubleArray(), svFactor, BinaryOperation.MULTIPLY), p.shape());
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
        double[] tvSq = COMPUTER.binaryOperate(tv, tv, BinaryOperation.MULTIPLY);
        double[] oneMinusTvSq = COMPUTER.binaryOperate(tvSq, -1.0, BinaryOperation.MULTIPLY);
        oneMinusTvSq = COMPUTER.binaryOperate(oneMinusTvSq, 1.0, BinaryOperation.ADD);
        IDoubleTensor t = new RereDoubleTensor(COMPUTER.binaryOperate(tangent.toDoubleArray(), oneMinusTvSq, BinaryOperation.MULTIPLY), p.shape());
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
    @Override public IDiffTensor softmaxCrossEntropySparse(int[] labels, int dim) {
        // Delegate to primal for forward pass; JVP: tangent of SCE = 0 (scalar loss)
        return ((RereDiffTensor) primal).softmaxCrossEntropySparse(labels, dim);
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
        // std = sqrt(var). JVP = 1/(2*std) * JVP_var, then apply reduction along dim.
        // d(var)/dX_i = 2 * (X_i - mean(X)) / N * dX_i
        IDoubleTensor x = primal.value();
        int rank = x.rank();
        int d = dim < 0 ? dim + rank : dim;
        long N = x.dim(d);
        IDoubleTensor meanX = x.mean(d, true);
        IDoubleTensor jvpVar = x.sub(meanX).mul(this.tangent).mul(2.0 / N);
        // Reduce along dim: sum of JVP_var contributions along the reduction dimension
        IDoubleTensor jvpVarReduced = jvpVar.sum(d, keepdim);
        // JVP_std = JVP_var / (2 * std)
        IDoubleTensor jvp = jvpVarReduced.div(p.value().mul(2.0));
        return new TangentDiffTensor(p, jvp, List.of(this), p);
    }
    @Override public IDiffTensor var(int dim, boolean keepdim) {
        RereDiffTensor p = (RereDiffTensor) primal.var(dim, keepdim);
        // var(X) = E[(X - E[X])^2]. JVP: d(var)/dX_i = 2 * (X_i - mean(X)) / N * dX_i
        IDoubleTensor x = primal.value();
        int rank = x.rank();
        int d = dim < 0 ? dim + rank : dim;
        long N = x.dim(d);
        IDoubleTensor meanX = x.mean(d, true);
        IDoubleTensor jvp = x.sub(meanX).mul(this.tangent).mul(2.0 / N);
        // Reduce along dim: sum of JVP contributions
        jvp = jvp.sum(d, keepdim);
        return new TangentDiffTensor(p, jvp, List.of(this), p);
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
        // JVP via RereDoubleTensor tensor ops (COMPUTER-accelerated internally).
        // batchNorm treats last dim as features, normalizes over all other dims.
        double[] xd = primal.value().toDoubleArray();
        double[] td = this.tangent.toDoubleArray();
        double[] gd = g.primal.value().toDoubleArray();
        int total = (int) primal.totalSize();
        int features = primal.shape()[primal.rank() - 1];
        int batch = total / features;
        int[] flatShape = new int[]{batch, features};
        RereDoubleTensor xr = new RereDoubleTensor(xd, flatShape);
        RereDoubleTensor tr = new RereDoubleTensor(td, flatShape);
        RereDoubleTensor gr = new RereDoubleTensor(gd, new int[]{1, features});
        // Per-feature statistics via dim-0 reduction (COMPUTER-accelerated reduce/sum)
        IDoubleTensor mx = xr.mean(0, true);
        IDoubleTensor centered = xr.sub(mx);
        IDoubleTensor is = centered.pow(2).sum(0, true).div(batch).add(eps).sqrt().pow(-1.0);
        IDoubleTensor xh = centered.mul(is);  // x_hat
        IDoubleTensor mt = tr.mean(0, true);
        IDoubleTensor mxh = tr.mul(xh).mean(0, true);
        // jvp = gamma / sigma * (t - meanT - xH * meanTXH)
        double[] jvp = tr.sub(mt).sub(xh.mul(mxh)).mul(is).mul(gr).toDoubleArray();
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this, g, b), p);
    }

    @Override public IDiffTensor rmsNorm(IDiffTensor gamma, double eps) {
        TangentDiffTensor g = (TangentDiffTensor) gamma;
        RereDiffTensor p = (RereDiffTensor) primal.rmsNorm(g.primal, eps);
        double[] xd = primal.value().toDoubleArray();
        double[] gd = g.primal.value().toDoubleArray();
        double[] td = this.tangent.toDoubleArray();
        int n = xd.length;
        int features = gd.length;
        int batch = n / features;
        double[] jvp = new double[n];
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int off = batchIdx * features;
            double sumSq = 0, sumXT = 0;
            for (int j = 0; j < features; j++) {
                sumSq += xd[off + j] * xd[off + j];
                sumXT += td[off + j] * xd[off + j];
            }
            double rms = Math.sqrt(sumSq / features + eps);
            double invRms = 1.0 / rms;
            double scale = sumXT / (features * rms * rms); // sumXT / (N * rms^2)
            for (int j = 0; j < features; j++) {
                // jvp = gamma / rms * (td - x * sumXT / (N * rms^2)) + x / rms * gamma_tangent
                jvp[off + j] = gd[j] * invRms * (td[off + j] - xd[off + j] * scale);
            }
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this, g), p);
    }

    @Override public IDiffTensor embedding(IDiffTensor indices) {
        TangentDiffTensor idx = (TangentDiffTensor) indices;
        RereDiffTensor p = (RereDiffTensor) primal.embedding(idx.primal);
        // JVP: gather tangent rows indexed by indices
        double[] tData = this.tangent.toDoubleArray();
        double[] idxData = idx.primal.value().toDoubleArray();
        int embeddingDim = primal.dim(primal.rank() - 1);
        int n = (int) idx.primal.totalSize();
        double[] jvp = new double[n * embeddingDim];
        for (int i = 0; i < n; i++) {
            int row = (int) idxData[i];
            System.arraycopy(tData, row * embeddingDim, jvp, i * embeddingDim, embeddingDim);
        }
        int[] idxShape = idx.primal.shape();
        int[] outShape = new int[idxShape.length + 1];
        System.arraycopy(idxShape, 0, outShape, 0, idxShape.length);
        outShape[idxShape.length] = embeddingDim;
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, outShape), List.of(this, idx), p);
    }

    @Override public IDiffTensor rope(int dim, int maxLen, double base) {
        RereDiffTensor p = (RereDiffTensor) primal.rope(dim, maxLen, base);
        // JVP: same rotation applied to tangents (linear operation)
        double[] td = this.tangent.toDoubleArray();
        int headDim = p.dim(p.rank() - 1);
        long totalSize = primal.totalSize();
        int seqLen = (int) (totalSize / headDim);
        int numPairs = headDim / 2;
        double[] jvp = new double[td.length];
        for (int pos = 0; pos < seqLen; pos++) {
            int baseOff = pos * headDim;
            for (int i = 0; i < numPairs; i++) {
                double theta = pos / Math.pow(base, 2.0 * i / dim);
                double c = Math.cos(theta), s = Math.sin(theta);
                int idx2i = baseOff + 2 * i;
                int idx2i1 = idx2i + 1;
                double t1 = td[idx2i], t2 = td[idx2i1];
                jvp[idx2i] = t1 * c - t2 * s;
                jvp[idx2i1] = t1 * s + t2 * c;
            }
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this), p);
    }

    @Override public IDiffTensor[] lstmCell(IDiffTensor x, IDiffTensor hPrev, IDiffTensor cPrev,
            IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
        TangentDiffTensor tx = (TangentDiffTensor) x;
        TangentDiffTensor th = (TangentDiffTensor) hPrev;
        TangentDiffTensor tc = (TangentDiffTensor) cPrev;
        TangentDiffTensor twi = (TangentDiffTensor) wInput;
        TangentDiffTensor twh = (TangentDiffTensor) wHidden;
        RereDiffTensor p = (RereDiffTensor) primal.lstmCell(tx.primal, th.primal, tc.primal,
            twi.primal, twh.primal, bias != null ? ((TangentDiffTensor)bias).primal : null)[0];
        return new IDiffTensor[]{
            new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),
                List.of(this, tx, th, tc, twi, twh), p)
        };
    }

    @Override public IDiffTensor gruCell(IDiffTensor x, IDiffTensor hPrev,
            IDiffTensor wInput, IDiffTensor wHidden, IDiffTensor bias) {
        TangentDiffTensor tx = (TangentDiffTensor) x;
        TangentDiffTensor th = (TangentDiffTensor) hPrev;
        TangentDiffTensor twi = (TangentDiffTensor) wInput;
        TangentDiffTensor twh = (TangentDiffTensor) wHidden;
        RereDiffTensor p = (RereDiffTensor) primal.gruCell(tx.primal, th.primal,
            twi.primal, twh.primal, bias != null ? ((TangentDiffTensor)bias).primal : null);
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),
            List.of(this, tx, th, twi, twh), p);
    }

    @Override public IDiffTensor groupNorm(int numGroups, IDiffTensor gamma, IDiffTensor beta, double eps) {
        TangentDiffTensor g = (TangentDiffTensor) gamma;
        TangentDiffTensor b = (TangentDiffTensor) beta;
        RereDiffTensor p = (RereDiffTensor) primal.groupNorm(numGroups, g.primal,
            b != null ? b.primal : null, eps);
        // groupNorm JVP via tensor ops: normalize per (N, group) over (groupCh * spatial).
        int[] s = primal.shape();
        int rank = s.length;
        int C = s[rank - 2];
        int outer = 1;
        for (int i = 0; i < rank - 2; i++) outer *= s[i];
        int spatial = 1;
        for (int i = rank - 1; i < rank; i++) spatial *= s[i];
        if (C % numGroups != 0) return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),
            b != null ? List.of(this, g, b) : List.of(this, g), p);
        int groupCh = C / numGroups;
        int groupSize = groupCh * spatial;
        double[] xd = primal.value().toDoubleArray();
        double[] td = this.tangent.toDoubleArray();
        double[] gd = g.primal.value().toDoubleArray();
        int nGroups = outer * numGroups;
        RereDoubleTensor xr = new RereDoubleTensor(xd, new int[]{nGroups, groupSize});
        RereDoubleTensor tr = new RereDoubleTensor(td, new int[]{nGroups, groupSize});
        IDoubleTensor mx = xr.mean(0, true);
        IDoubleTensor centered = xr.sub(mx);
        IDoubleTensor is = centered.pow(2).sum(0, true).div(groupSize).add(eps).sqrt().pow(-1.0);
        IDoubleTensor xh = centered.mul(is);
        IDoubleTensor mt = tr.mean(0, true);
        IDoubleTensor mxh = tr.mul(xh).mean(0, true);
        double[] jvp = tr.sub(mt).sub(xh.mul(mxh)).mul(is).toDoubleArray();
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()),
            b != null ? List.of(this, g, b) : List.of(this, g), p);
    }

    @Override public IDiffTensor flip(int... dims) {
        RereDiffTensor p = (RereDiffTensor) primal.flip(dims);
        double[] jvp = new double[(int) p.totalSize()];
        // flip is linear: apply same flip to tangents
        double[] td = this.tangent.toDoubleArray();
        int[] s = p.shape();
        for (long flatIdx = 0; flatIdx < jvp.length; flatIdx++) {
            long dstIdx = 0;
            long srcIdx = flatIdx;
            long stride = 1;
            for (int d = p.rank() - 1; d >= 0; d--) {
                long coord = srcIdx % s[d];
                srcIdx /= s[d];
                for (int fd : dims) { if ((fd < 0 ? fd + p.rank() : fd) == d) { coord = s[d] - 1 - coord; break; } }
                dstIdx += coord * stride;
                stride *= s[d];
            }
            jvp[(int) dstIdx] = td[(int) flatIdx];
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this), p);
    }

    @Override public IDiffTensor roll(int[] shifts, int[] dims) {
        // roll decomposes into split/cat — linear, so apply same roll to tangents
        IDiffTensor primalRolled = primal.roll(java.util.Arrays.copyOf(shifts, shifts.length),
            java.util.Arrays.copyOf(dims, dims.length));
        RereDiffTensor p = (RereDiffTensor) primalRolled;
        // Apply roll to tangents via decomposition
        IDoubleTensor tResult = this.tangent;
        for (int i = 0; i < shifts.length; i++) {
            int d = (dims[i] < 0 ? dims[i] + tResult.rank() : dims[i]);
            int dimSize = tResult.dim(d);
            int shift = ((shifts[i] % dimSize) + dimSize) % dimSize;
            if (shift == 0) continue;
            IDoubleTensor[] parts = { tResult.narrow(d, dimSize - shift, shift),
                                      tResult.narrow(d, 0, dimSize - shift) };
            tResult = parts[0].cat(d, parts[1]);
        }
        double[] jvp = tResult.toDoubleArray();
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this), p);
    }

    @Override public IDiffTensor repeatInterleave(int repeats, int dim) {
        RereDiffTensor p = (RereDiffTensor) primal.repeatInterleave(repeats, dim);
        // repeatInterleave on tangent: same index pattern
        int d = (dim < 0 ? dim + this.tangent.rank() : dim);
        int dimSize = this.tangent.dim(d);
        double[] td = this.tangent.toDoubleArray();
        int[] ts = this.tangent.shape();
        int[] js = new int[ts.length];
        js[d] = dimSize * repeats;
        for (int i = 0; i < ts.length; i++) if (i != d) js[i] = ts[i];
        double[] jvp = new double[(int) p.totalSize()];
        int repeatStride = 1;
        for (int i = d + 1; i < ts.length; i++) repeatStride *= ts[i];
        int outerStride = repeatStride * dimSize;
        int outerCount = (int) this.tangent.totalSize() / (dimSize * repeatStride);
        for (int o = 0; o < outerCount; o++) {
            for (int i = 0; i < dimSize; i++) {
                for (int r = 0; r < repeats; r++) {
                    for (int s = 0; s < repeatStride; s++) {
                        int src = o * outerStride + i * repeatStride + s;
                        int dst = o * outerStride * repeats + (i * repeats + r) * repeatStride + s;
                        jvp[dst] = td[src];
                    }
                }
            }
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this), p);
    }


    /**
     * Compute scalar JVP for loss functions: JVP = dot(grad_input, tangent_input) + dot(grad_target, tangent_target).
     * Backpropagates through the loss node p, then computes dot product of input gradients with their tangents.
     */
    private static double[] lossJVP(RereDiffTensor lossNode, TangentDiffTensor input, TangentDiffTensor target) {
        lossNode.backward();
        DoubleVectorComputer comp = COMPUTER;
        double jvp = 0;
        IDoubleTensor gx = input.primal.grad();
        if (gx != null) {
            double[] gxd = gx.toDoubleArray();
            double[] tx = input.tangent.toDoubleArray();
            double[] prod = comp.binaryOperate(gxd, tx, BinaryOperation.MULTIPLY);
            jvp += comp.reduceOperate(prod, ReduceOperation.SUM);
        }
        IDoubleTensor gt = target.primal.grad();
        if (gt != null) {
            double[] gtd = gt.toDoubleArray();
            double[] ty = target.tangent.toDoubleArray();
            double[] prod = comp.binaryOperate(gtd, ty, BinaryOperation.MULTIPLY);
            jvp += comp.reduceOperate(prod, ReduceOperation.SUM);
        }
        return new double[]{jvp};
    }
    @Override public IDiffTensor smoothL1Loss(IDiffTensor target, double beta) {
        TangentDiffTensor t = (TangentDiffTensor) target;
        RereDiffTensor p = (RereDiffTensor) primal.smoothL1Loss(t.primal, beta);
        double[] jvpVal = lossJVP(p, this, t);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);
    }
    @Override public IDiffTensor bceLoss(IDiffTensor target) {
        TangentDiffTensor t = (TangentDiffTensor) target;
        RereDiffTensor p = (RereDiffTensor) primal.bceLoss(t.primal);
        double[] jvpVal = lossJVP(p, this, t);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);
    }
    @Override public IDiffTensor focalLoss(IDiffTensor target, double alpha, double gamma) {
        TangentDiffTensor t = (TangentDiffTensor) target;
        RereDiffTensor p = (RereDiffTensor) primal.focalLoss(t.primal, alpha, gamma);
        double[] jvpVal = lossJVP(p, this, t);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);
    }
    @Override public IDiffTensor diceLoss(IDiffTensor target, double smooth) {
        TangentDiffTensor t = (TangentDiffTensor) target;
        RereDiffTensor p = (RereDiffTensor) primal.diceLoss(t.primal, smooth);
        double[] jvpVal = lossJVP(p, this, t);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);
    }

    @Override public IDiffTensor nllLoss(IDiffTensor target, int classDim) {
        TangentDiffTensor t = (TangentDiffTensor) target;
        RereDiffTensor p = (RereDiffTensor) primal.nllLoss(t.primal, classDim);
        double[] jvpVal = lossJVP(p, this, t);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvpVal, 1), List.of(this, t), p);
    }

    @Override public IDiffTensor maxPool2d(int kH, int kW, int stride, int padding) {
        RereDiffTensor p = (RereDiffTensor) primal.maxPool2d(kH, kW, stride, padding);
        // maxPool2d JVP: scatter tangent values using argmax indices from primal input
        double[] xd = primal.value().toDoubleArray();
        double[] td = this.tangent.toDoubleArray();
        int[] s = primal.shape();
        int N = s[0], C = s[1], H = s[2], W = s[3];
        int outH = (H + 2 * padding - kH) / stride + 1;
        int outW = (W + 2 * padding - kW) / stride + 1;
        double[] jvp = new double[N * C * outH * outW];
        // Structural loop: window-based pooling (not element-wise, cannot use vector computer)
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        int outIdx = ((n * C + c) * outH + oh) * outW + ow;
                        double maxVal = Double.NEGATIVE_INFINITY;
                        int maxPos = -1;
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= W) continue;
                                int inIdx = ((n * C + c) * H + ih) * W + iw;
                                if (xd[inIdx] > maxVal) { maxVal = xd[inIdx]; maxPos = inIdx; }
                            }
                        }
                        jvp[outIdx] = (maxPos >= 0) ? td[maxPos] : 0;
                    }
                }
            }
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this), p);
    }

    @Override public IDiffTensor avgPool2d(int kH, int kW, int stride, int padding) {
        RereDiffTensor p = (RereDiffTensor) primal.avgPool2d(kH, kW, stride, padding);
        // avgPool2d is linear: JVP = avgPool2d(tangent) — same averaging applied to tangent
        double[] td = this.tangent.toDoubleArray();
        int[] s = primal.shape();
        int N = s[0], C = s[1], H = s[2], W = s[3];
        int outH = (H + 2 * padding - kH) / stride + 1;
        int outW = (W + 2 * padding - kW) / stride + 1;
        double[] jvp = new double[N * C * outH * outW];
        // Structural loop: window-based pooling (not element-wise, cannot use vector computer)
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        double sum = 0;
                        int count = 0;
                        for (int kh = 0; kh < kH; kh++) {
                            int ih = oh * stride + kh - padding;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < kW; kw++) {
                                int iw = ow * stride + kw - padding;
                                if (iw < 0 || iw >= W) continue;
                                sum += td[((n * C + c) * H + ih) * W + iw];
                                count++;
                            }
                        }
                        jvp[((n * C + c) * outH + oh) * outW + ow] = sum / count;
                    }
                }
            }
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this), p);
    }

    @Override public IDiffTensor adaptiveAvgPool2d(int outH, int outW) {
        RereDiffTensor p = (RereDiffTensor) primal.adaptiveAvgPool2d(outH, outW);
        // adaptiveAvgPool2d is linear: JVP = adaptiveAvgPool2d(tangent)
        double[] td = this.tangent.toDoubleArray();
        int[] s = primal.shape();
        int N = s[0], C = s[1], H = s[2], W = s[3];
        double[] jvp = new double[N * C * outH * outW];
        // Structural loop: adaptive pooling (not element-wise, cannot use vector computer)
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int oh = 0; oh < outH; oh++) {
                    int hStart = (int) Math.floor((double) oh * H / outH);
                    int hEnd = (int) Math.ceil((double) (oh + 1) * H / outH);
                    for (int ow = 0; ow < outW; ow++) {
                        int wStart = (int) Math.floor((double) ow * W / outW);
                        int wEnd = (int) Math.ceil((double) (ow + 1) * W / outW);
                        double sum = 0;
                        int count = 0;
                        for (int ih = hStart; ih < hEnd; ih++) {
                            for (int iw = wStart; iw < wEnd; iw++) {
                                sum += td[((n * C + c) * H + ih) * W + iw];
                                count++;
                            }
                        }
                        jvp[((n * C + c) * outH + oh) * outW + ow] = sum / count;
                    }
                }
            }
        }
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), List.of(this), p);
    }

    @Override public IDiffTensor oneHot(int numClasses) {
        RereDiffTensor p = (RereDiffTensor) primal.oneHot(numClasses);
        // oneHot is piecewise constant: the output is 0 or 1 and does not change
        // with small perturbations of the input, so JVP = 0 (mathematically correct).
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),
            List.of(this), p);
    }

    @Override public IDiffTensor instanceNorm(IDiffTensor gamma, IDiffTensor beta, double eps) {
        TangentDiffTensor g = (TangentDiffTensor) gamma;
        TangentDiffTensor b = (TangentDiffTensor) beta;
        RereDiffTensor p = (RereDiffTensor) primal.instanceNorm(g.primal,
            b != null ? b.primal : null, eps);
        // instanceNorm JVP via tensor ops: normalize per (N, C) over spatial dims.
        // Layout: reshape [N, C, H, W] → [N*C, H*W], then use dim-0 reduction.
        int[] s = primal.shape();
        int rank = s.length;
        if (rank < 2) return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),
            b != null ? List.of(this, g, b) : List.of(this, g), p);
        int N = 1;
        for (int i = 0; i < rank - 2; i++) N *= s[i];
        int C = s[rank - 2];
        int spatial = 1;
        for (int i = rank - 1; i < rank; i++) spatial *= s[i];
        int nInst = N * C;
        double[] xd = primal.value().toDoubleArray();
        double[] td = this.tangent.toDoubleArray();
        RereDoubleTensor xr = new RereDoubleTensor(xd, new int[]{nInst, spatial});
        RereDoubleTensor tr = new RereDoubleTensor(td, new int[]{nInst, spatial});
        IDoubleTensor mx = xr.mean(0, true);
        IDoubleTensor centered = xr.sub(mx);
        IDoubleTensor is = centered.pow(2).sum(0, true).div(spatial).add(eps).sqrt().pow(-1.0);
        IDoubleTensor xh = centered.mul(is);
        IDoubleTensor mt = tr.mean(0, true);
        IDoubleTensor mxh = tr.mul(xh).mean(0, true);
        // JVP = sigma⁻¹ * (t - meanT - xH * meanTXH) per (N,C) instance
        double[] jvp = tr.sub(mt).sub(xh.mul(mxh)).mul(is).toDoubleArray();
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()),
            b != null ? List.of(this, g, b) : List.of(this, g), p);
    }

    @Override public IDiffTensor diagEmbed(int offset, int dim1, int dim2) {
        RereDiffTensor p = (RereDiffTensor) primal.diagEmbed(offset, dim1, dim2);
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int)p.totalSize()], p.shape()).fill_(0),
            List.of(this), p);
    }

    @Override public IDiffTensor dropout2d(double p) {
        RereDiffTensor pt = (RereDiffTensor) primal.dropout2d(p);
        // JVP for dropout2d should apply the same mask: JVP = mask * tangent.
        // The dropout mask is generated internally during primal forward and not exposed.
        // TODO: Save mask from primal forward and apply here.
        return new TangentDiffTensor(pt, new RereDoubleTensor(new double[(int)pt.totalSize()], pt.shape()).fill_(0),
            List.of(this), pt);
    }

    @Override public IDiffTensor depthwiseConv1d(IDiffTensor weight, int stride, int padding) {
        TangentDiffTensor w = (TangentDiffTensor) weight;
        RereDiffTensor p = (RereDiffTensor) primal.depthwiseConv1d(w.primal, stride, padding);
        // depthwiseConv1d is bilinear: JVP = conv1d(tangent_input, weight) + conv1d(input, tangent_weight)
        double[] td = this.tangent.toDoubleArray();
        double[] wd = w.tangent.toDoubleArray();
        RereDiffTensor tInput = new RereDiffTensor(td, primal.shape());
        RereDiffTensor tWeight = new RereDiffTensor(wd, w.primal.shape());
        IDiffTensor jvpInput = tInput.depthwiseConv1d(w.primal, stride, padding);
        IDiffTensor jvpWeight = primal.depthwiseConv1d(tWeight, stride, padding);
        IDiffTensor jvp = jvpInput.add(jvpWeight);
        return new TangentDiffTensor(p, jvp, List.of(this, w), p);
    }

    @Override public IDiffTensor interpolate(double scaleFactor, String mode) {
        RereDiffTensor p = (RereDiffTensor) primal.interpolate(scaleFactor, mode);
        // interpolate is linear in input: JVP = interpolate(tangent, scaleFactor, mode)
        double[] td = this.tangent.toDoubleArray();
        RereDiffTensor tInput = new RereDiffTensor(td, primal.shape());
        IDiffTensor jvpTerm = tInput.interpolate(scaleFactor, mode);
        return new TangentDiffTensor(p, jvpTerm, List.of(this), p);
    }

    @Override public IDiffTensor logDet() {
        RereDiffTensor p = (RereDiffTensor) primal.logDet();
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[1], 1).fill_(0),
            List.of(this), p);
    }

    @Override public IDiffTensor[] slogDet() {
        IDiffTensor[] pres = primal.slogDet();
        RereDiffTensor p0 = (RereDiffTensor) pres[0];
        RereDiffTensor p1 = (RereDiffTensor) pres[1];
        return new IDiffTensor[]{
            new TangentDiffTensor(p0, new RereDoubleTensor(new double[1], 1).fill_(0), List.of(this), p0),
            new TangentDiffTensor(p1, new RereDoubleTensor(new double[1], 1).fill_(0), List.of(this), p1)
        };
    }

    @Override public IDiffTensor nuclearNorm() {
        RereDiffTensor p = (RereDiffTensor) primal.nuclearNorm();
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[1], 1).fill_(0),
            List.of(this), p);
    }

    @Override public IDiffTensor ctcLoss(IDiffTensor targets, IDiffTensor inputLengths, IDiffTensor targetLengths) {
        RereDiffTensor p = (RereDiffTensor) primal.ctcLoss(targets, inputLengths, targetLengths);
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[1], 1).fill_(0),
            List.of(this), p);
    }

    @Override public IDiffTensor cross(IDiffTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        RereDiffTensor p = (RereDiffTensor) primal.cross(o.primal);
        // cross(a,b) is bilinear: JVP = cross(tangent_a, b) + cross(a, tangent_b)
        // Use RereDoubleTensor ops to compute cross products without raw loops.
        IDoubleTensor ta = this.tangent;
        IDoubleTensor tb = o.tangent;
        IDoubleTensor xa = primal.value();
        IDoubleTensor xb = o.primal.value();
        int rank = ta.rank();
        // Extract components from last dim (size 3) via narrow
        IDoubleTensor a0 = ta.narrow(rank - 1, 0, 1);
        IDoubleTensor a1 = ta.narrow(rank - 1, 1, 1);
        IDoubleTensor a2 = ta.narrow(rank - 1, 2, 1);
        IDoubleTensor b0 = tb.narrow(rank - 1, 0, 1);
        IDoubleTensor b1 = tb.narrow(rank - 1, 1, 1);
        IDoubleTensor b2 = tb.narrow(rank - 1, 2, 1);
        IDoubleTensor ya0 = xa.narrow(rank - 1, 0, 1);
        IDoubleTensor ya1 = xa.narrow(rank - 1, 1, 1);
        IDoubleTensor ya2 = xa.narrow(rank - 1, 2, 1);
        IDoubleTensor yb0 = xb.narrow(rank - 1, 0, 1);
        IDoubleTensor yb1 = xb.narrow(rank - 1, 1, 1);
        IDoubleTensor yb2 = xb.narrow(rank - 1, 2, 1);
        // cross(tangent_a, b) + cross(a, tangent_b)
        IDoubleTensor c0 = a1.mul(yb2).sub(a2.mul(yb1)).add(ya1.mul(b2)).sub(ya2.mul(b1));
        IDoubleTensor c1 = a2.mul(yb0).sub(a0.mul(yb2)).add(ya2.mul(b0)).sub(ya0.mul(b2));
        IDoubleTensor c2 = a0.mul(yb1).sub(a1.mul(yb0)).add(ya0.mul(b1)).sub(ya1.mul(b0));
        // Concatenate along last dim to restore [..., 3] shape
        IDoubleTensor jvp = c0.cat(rank - 1, c1, c2);
        return new TangentDiffTensor(p, jvp, List.of(this, o), p);
    }

    @Override public IDiffTensor gridSample(IDiffTensor grid, String mode, String paddingMode) {
        RereDiffTensor p = (RereDiffTensor) primal.gridSample(grid, mode, paddingMode);
        // gridSample is linear in input: JVP = gridSample(tangent, grid, mode, paddingMode)
        double[] td = this.tangent.toDoubleArray();
        RereDiffTensor tInput = new RereDiffTensor(td, primal.shape());
        IDiffTensor jvpTerm = tInput.gridSample(grid, mode, paddingMode);
        return new TangentDiffTensor(p, jvpTerm, List.of(this), p);
    }

    @Override public IDiffTensor trapezoidalScan(IDiffTensor delta, IDiffTensor A, IDiffTensor B,
                                                  IDiffTensor C, IDiffTensor D) {
        RereDiffTensor p = (RereDiffTensor) primal.trapezoidalScan(delta, A, B, C, D);
        return new TangentDiffTensor(p, new RereDoubleTensor(new double[(int) p.totalSize()], p.shape()).fill_(0),
            List.of(this), p);
    }

    @Override public IDiffTensor conv2d(IDiffTensor weight, IDiffTensor bias,
            int stride, int padding, int dilation) {
        // Forward-mode JVP for conv2d: bilinear = conv2d(tangent_input, weight, bias) + conv2d(input, tangent_weight, tangent_bias)
        TangentDiffTensor w = (TangentDiffTensor) weight;
        TangentDiffTensor b = (TangentDiffTensor) bias;
        RereDiffTensor p = (RereDiffTensor) primal.conv2d(w.primal,
            b != null ? b.primal : null, stride, padding, dilation);
        double[] td = this.tangent.toDoubleArray();
        double[] wd = w.tangent.toDoubleArray();
        RereDiffTensor tInput = new RereDiffTensor(td, primal.shape());
        RereDiffTensor tWeight = new RereDiffTensor(wd, w.primal.shape());
        IDiffTensor jvpInput = tInput.conv2d(w.primal, b != null ? b.primal : null, stride, padding, dilation);
        IDiffTensor jvpWeight = primal.conv2d(tWeight, b != null ? b.primal : null, stride, padding, dilation);
        IDiffTensor jvp = jvpInput.add(jvpWeight);
        if (b != null) {
            // Bias JVP: tangent_bias [outC] broadcast to output [N, outC, outH, outW]
            // unsqueeze to [1, outC, 1, 1] then expand for broadcast-add
            RereDiffTensor tBias = new RereDiffTensor(b.tangent.toDoubleArray(), b.primal.shape());
            IDiffTensor biasTerm = tBias.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(p.shape());
            jvp = jvp.add(biasTerm);
        }
        return new TangentDiffTensor(p, jvp, List.of(this, w, b).stream().filter(java.util.Objects::nonNull).toList(), p);
    }

    @Override public IDiffTensor scaledDotProductAttention(IDiffTensor key, IDiffTensor vTensor,
            IDiffTensor mask, double dropout) {
        TangentDiffTensor k = (TangentDiffTensor) key;
        TangentDiffTensor v = (TangentDiffTensor) vTensor;
        TangentDiffTensor m = (TangentDiffTensor) mask;
        RereDiffTensor p = (RereDiffTensor) primal.scaledDotProductAttention(
            k.primal, v.primal, m != null ? m.primal : null, dropout);
        // JVP for attention: Q and K tangents are zero → softmax derivative term = 0.
        // JVP = attn_weights @ tangent_V where attn_weights = softmax(Q @ K^T / sqrt(d_k))
        // Use raw arrays via DoubleFlatGemm + DoubleVectorComputer for acceleration.
        int[] qShape = primal.shape();
        int[] kShape = k.primal.shape();
        int[] vShape = v.primal.shape();
        int batch = qShape[0], seqQ = qShape[1], dk = qShape[2];
        int seqK = kShape[1], dv = vShape[2];
        double[] qd = primal.value().toDoubleArray();
        double[] kd = k.primal.value().toDoubleArray();
        double scale = 1.0 / Math.sqrt(dk);
        int qStride = seqQ * dk;
        int kStride = seqK * dk;
        int scoresStride = seqQ * seqK;
        DoubleVectorComputer vc = COMPUTER;
        double[] scoresFlat = new double[batch * scoresStride];
        for (int b = 0; b < batch; b++) {
            double[] qSlice = java.util.Arrays.copyOfRange(qd, b * qStride, b * qStride + qStride);
            double[] kSlice = java.util.Arrays.copyOfRange(kd, b * kStride, b * kStride + kStride);
            double[] kT = DoubleFlatGemm.flatTranspose(kSlice, seqK, dk);
            double[] rawScores = DoubleFlatGemm.flatMmul(qSlice, seqQ, dk, kT, seqK);
            double[] scaled = vc.binaryOperate(rawScores, scale, BinaryOperation.MULTIPLY);
            // Row-wise softmax via DoubleVectorComputer (GPU→SIMD→SISD)
            double[] ones = vc.fill(seqK, 1.0);
            for (int r = 0; r < seqQ; r++) {
                int offset = r * seqK;
                double[] row = java.util.Arrays.copyOfRange(scaled, offset, offset + seqK);
                double rowMax = vc.reduceOperate(row, ReduceOperation.MAX);
                double[] shifted = vc.binaryOperate(row, rowMax, BinaryOperation.SUBTRACT);
                double[] expVals = vc.universalOperate(shifted, UniversalOperation.EXP, 0.0);
                double sumExp = vc.reduceOperate(expVals, ReduceOperation.SUM);
                double[] invSum = vc.binaryOperate(ones, sumExp, BinaryOperation.DIVIDE);
                double[] probs = vc.binaryOperate(expVals, invSum, BinaryOperation.MULTIPLY);
                System.arraycopy(probs, 0, scoresFlat, b * scoresStride + offset, seqK);
            }
        }
        // JVP = attn @ tangent_V
        double[] vd = v.tangent.toDoubleArray();
        int vStride = seqK * dv;
        double[] jvp = new double[batch * seqQ * dv];
        for (int b = 0; b < batch; b++) {
            double[] attnSlice = java.util.Arrays.copyOfRange(scoresFlat, b * scoresStride, b * scoresStride + scoresStride);
            double[] vSlice = java.util.Arrays.copyOfRange(vd, b * vStride, b * vStride + vStride);
            double[] vT = DoubleFlatGemm.flatTranspose(vSlice, seqK, dv);
            double[] result = DoubleFlatGemm.flatMmul(attnSlice, seqQ, seqK, vT, dv);
            System.arraycopy(result, 0, jvp, b * seqQ * dv, seqQ * dv);
        }
        List<TangentDiffTensor> inputs = new java.util.ArrayList<>();
        inputs.add(this); inputs.add(k); inputs.add(v);
        return new TangentDiffTensor(p, new RereDoubleTensor(jvp, p.shape()), inputs, p);
    }

    // ---- in-place ops ----
    @Override public IDiffTensor add_(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        this.tangent.copy_(new RereDoubleTensor(COMPUTER.binaryOperate(aT, bT, BinaryOperation.ADD), this.tangent.shape()));
        primal.add_(o.primal);
        return this;
    }
    @Override public IDiffTensor sub_(IDoubleTensor other) {
        TangentDiffTensor o = (TangentDiffTensor) other;
        double[] aT = this.tangent.toDoubleArray();
        double[] bT = o.tangent.toDoubleArray();
        this.tangent.copy_(new RereDoubleTensor(COMPUTER.binaryOperate(aT, bT, BinaryOperation.SUBTRACT), this.tangent.shape()));
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

    // ---- Phase 0: triu / diag / diagonal / trace / logSumExp / split / chunk / unbind ----

    @Override public IDiffTensor triu(int diagonal) {
        RereDiffTensor p = (RereDiffTensor) primal.triu(diagonal);
        // Tangent: apply same triu mask to tangent data (linear operation)
        double[] tData = this.tangent.toDoubleArray().clone();
        int r = p.rank();
        int M = p.dim(r - 2);
        int N = p.dim(r - 1);
        int batchStride = M * N;
        int batchCount = tData.length / batchStride;
        for (int b = 0; b < batchCount; b++) {
            int base = b * batchStride;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    if (j < i + diagonal) tData[base + i * N + j] = 0.0;
                }
            }
        }
        IDoubleTensor t = new RereDoubleTensor(tData, p.shape());
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor diag() {
        RereDiffTensor p = (RereDiffTensor) primal.diag();
        // Tangent: extract diagonal of tangent
        double[] tData = this.tangent.toDoubleArray();
        int[] s = this.tangent.shape();
        int r = s.length;
        int M = s[r - 2];
        int batchDim = 1;
        for (int i = 0; i < r - 2; i++) batchDim *= s[i];
        double[] diagData = new double[batchDim * M];
        for (int b = 0; b < batchDim; b++) {
            int base = b * M * M;
            for (int i = 0; i < M; i++) diagData[b * M + i] = tData[base + i * M + i];
        }
        int[] resultShape = (r == 2) ? new int[]{M} : java.util.Arrays.copyOf(s, r - 1);
        if (r > 2) resultShape[r - 2] = M;
        IDoubleTensor t = new RereDoubleTensor(diagData, resultShape);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor diagonal(int offset, int dim1, int dim2) {
        RereDiffTensor p = (RereDiffTensor) primal.diagonal(offset, dim1, dim2);
        // Tangent: forward-mode AD for diagonal is identity operator on tangent
        // (diagonal is linear). Extract same diagonal from tangent.
        // Reuse primal's diagonal result shape, get tangent's data
        IDoubleTensor t = diagonalOnTensor(this.tangent, offset, dim1, dim2);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    /** Helper: apply diagonal extraction to IDoubleTensor (not differentiable). */
    private static IDoubleTensor diagonalOnTensor(IDoubleTensor src, int offset, int dim1, int dim2) {
        int r = src.rank();
        if (dim1 < 0) dim1 += r;
        if (dim2 < 0) dim2 += r;
        int[] s = src.shape();
        int effSize = offset >= 0
            ? Math.max(0, Math.min(s[dim1] - offset, s[dim2]))
            : Math.max(0, Math.min(s[dim1], s[dim2] + offset));
        if (effSize == 0) throw new IllegalArgumentException("diagonal: offset " + offset + " yields empty diagonal");
        int outer = 1;
        for (int i = 0; i < Math.min(dim1, dim2); i++) outer *= s[i];
        int inner = 1;
        for (int i = Math.max(dim1, dim2) + 1; i < r; i++) inner *= s[i];
        double[] vals = src.toDoubleArray();
        double[] resultData = new double[outer * effSize * inner];
        for (int o = 0; o < outer; o++) {
            for (int k = 0; k < effSize; k++) {
                for (int i = 0; i < inner; i++) {
                    int[] idx = new int[r];
                    int remaining = o;
                    for (int j = Math.min(dim1, dim2) - 1; j >= 0; j--) {
                        idx[j] = remaining % s[j]; remaining /= s[j];
                    }
                    idx[dim1] = offset >= 0 ? k + offset : k;
                    idx[dim2] = offset >= 0 ? k : k - offset;
                    remaining = i;
                    for (int j = r - 1; j > Math.max(dim1, dim2); j--) {
                        idx[j] = remaining % s[j]; remaining /= s[j];
                    }
                    resultData[(o * effSize + k) * inner + i] = vals[DiffTensorUtil.flatIndex(idx, s)];
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
        return new RereDoubleTensor(resultData, resultShape);
    }
    @Override public IDiffTensor trace() {
        RereDiffTensor p = (RereDiffTensor) primal.trace();
        // Tangent: trace(tangent)
        IDiffTensor diagT = (IDiffTensor) ((RereDiffTensor) primal.diag()).sum();
        // Actually trace of tangent = diag(tangent).sumAll()
        double diagSum = 0;
        double[] tDiag = computeDiag(tangent);
        for (double v : tDiag) diagSum += v;
        IDoubleTensor t = new RereDoubleTensor(new double[]{diagSum}, new int[]{1});
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    /** Extract diagonal elements from an IDoubleTensor matrix. */
    private static double[] computeDiag(IDoubleTensor src) {
        int[] s = src.shape();
        int r = s.length;
        int M = s[r - 2];
        int N = s[r - 1];
        int n = Math.min(M, N);
        int batchDim = 1;
        for (int i = 0; i < r - 2; i++) batchDim *= s[i];
        double[] vals = src.toDoubleArray();
        double[] result = new double[batchDim * n];
        for (int b = 0; b < batchDim; b++) {
            int base = b * M * N;
            for (int i = 0; i < n; i++) result[b * n + i] = vals[base + i * N + i];
        }
        return result;
    }
    @Override public IDiffTensor logSumExp(int dim, boolean keepdim) {
        // Forward-mode: JVP = sum(d_tangent * softmax(x), dim)
        int d = (dim < 0 ? dim + primal.rank() : dim);
        RereDiffTensor p = (RereDiffTensor) primal.logSumExp(dim, keepdim);
        IDoubleTensor sm = primal.softmax(d);
        IDoubleTensor t = this.tangent.mul(sm).sum(d, keepdim);
        return new TangentDiffTensor(p, t, List.of(this), p);
    }
    @Override public IDiffTensor[] split(int splitSize, int dim) {
        IDiffTensor[] primals = primal.split(splitSize, dim);
        IDoubleTensor[] tangents = null; // compute below via narrow
        IDiffTensor[] result = new IDiffTensor[primals.length];
        int offset = 0;
        int d = (dim < 0 ? dim + primal.rank() : dim);
        for (int i = 0; i < primals.length; i++) {
            int segSize = primals[i].dim(d < primals[i].rank() ? d : 0);
            IDoubleTensor tSeg = this.tangent.narrow(d < this.tangent.rank() ? d : 0, offset, segSize);
            offset += segSize;
            result[i] = new TangentDiffTensor(
                (RereDiffTensor) primals[i], tSeg, List.of(this), (RereDiffTensor) primals[i]);
        }
        return result;
    }
    @Override public IDiffTensor[] split(int[] splitSizes, int dim) {
        int d = (dim < 0 ? dim + primal.rank() : dim);
        int n = splitSizes.length;
        IDiffTensor[] result = new IDiffTensor[n];
        int offset = 0;
        for (int i = 0; i < n; i++) {
            int sz = splitSizes[i];
            RereDiffTensor p = (RereDiffTensor) primal.narrow(d, offset, sz);
            IDoubleTensor t = this.tangent.narrow(d, offset, sz);
            offset += sz;
            result[i] = new TangentDiffTensor(p, t, List.of(this), p);
        }
        return result;
    }
    @Override public IDiffTensor[] chunk(int chunks, int dim) {
        int d = (dim < 0 ? dim + primal.rank() : dim);
        int dimSize = primal.dim(d);
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
        int d = (dim < 0 ? dim + primal.rank() : dim);
        int dimSize = primal.dim(d);
        IDiffTensor[] result = new IDiffTensor[dimSize];
        for (int i = 0; i < dimSize; i++) {
            RereDiffTensor p = (RereDiffTensor) primal.narrow(d, i, 1);
            IDoubleTensor t = this.tangent.narrow(d, i, 1);
            RereDiffTensor pS = (RereDiffTensor) p.squeeze(d);
            result[i] = new TangentDiffTensor(pS, t.squeeze(d), List.of(this), pS);
        }
        return result;
    }

    @Override public int size() { return primal.size(); }
    @Override public int dim(int axis) { return primal.dim(axis); }
    @Override public ITensor set(double value, int... indices) { primal.set(value, indices); return this; }
    @Override public ITensor fill(double value) { primal.fill(value); return this; }
}
