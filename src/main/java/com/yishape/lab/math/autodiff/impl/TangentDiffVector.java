package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.List;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;
import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * Forward-mode (tangent) AD wrapper: propagates JVP alongside primal in {@link #tangent}.
 * 正向模式（切向量）AD 包装：在 {@link #tangent} 中传播雅可比-向量积。
 *
 * <p>Used by {@link com.yishape.lab.math.optimize.autodiff.AD#jacobian}.
 * 供 {@link com.yishape.lab.math.optimize.autodiff.AD#jacobian} 使用。</p>
 */
public class TangentDiffVector implements IDiffVector {

    private static final long serialVersionUID = 1L;

    private final RereDiffVector primal;
    private final IDoubleVector tangent;
    private final List<TangentDiffVector> tangentInputs;
    private final RereDiffVector outputNode;

    private TangentDiffVector(RereDiffVector primal, IDoubleVector tangent,
            List<TangentDiffVector> tangentInputs, RereDiffVector outputNode) {
        this.primal = primal;
        this.tangent = tangent;
        this.tangentInputs = tangentInputs;
        this.outputNode = outputNode;
    }

    public static TangentDiffVector seed(RereDiffVector primal, IDoubleVector tangent) {
        return new TangentDiffVector(primal, tangent, List.of(), null);
    }

    public IDoubleVector getTangent() {
        return tangent;
    }

    public RereDiffVector getPrimal() {
        return primal;
    }

    @Override public IDoubleVector getValue() { return primal.getValue(); }
    @Override public IDoubleVector getGradient() { return primal.getGradient(); }
    @Override public boolean isLeaf() { return primal.isLeaf(); }

    @Override public void backward() { primal.backward(); }
    @Override public void backward(IDoubleVector g) { primal.backward(g); }
    @Override public void zeroGradient() { primal.zeroGradient(); }
    @Override public IDiffVector grad() { return primal.grad(); }

    @Override public IDiffVector broadcast(int n) {
        RereDiffVector p = (RereDiffVector) primal.broadcast(n);
        IDoubleVector t;
        if (this.tangent.size() == 1) {
            t = IDoubleVector.ones(n).multiplyByScalar(this.tangent.get(0));
        } else if (this.tangent.size() == n) {
            t = this.tangent;
        } else {
            throw new IllegalArgumentException(
                "broadcast tangent size mismatch: tangent size " + this.tangent.size()
                + " != target size " + n + " and != 1");
        }
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    // -- arithmetic with variables --

    @Override public IDiffVector add(IDiffVector other) {
        TangentDiffVector o = (TangentDiffVector) other;
        RereDiffVector p = (RereDiffVector) primal.add(o.primal);
        IDoubleVector t = this.tangent.add(o.tangent);
        return new TangentDiffVector(p, t, List.of(this, o), p);
    }

    @Override public IDiffVector sub(IDiffVector other) {
        TangentDiffVector o = (TangentDiffVector) other;
        RereDiffVector p = (RereDiffVector) primal.sub(o.primal);
        IDoubleVector t = this.tangent.sub(o.tangent);
        return new TangentDiffVector(p, t, List.of(this, o), p);
    }

    @Override public IDiffVector mul(IDiffVector other) {
        TangentDiffVector o = (TangentDiffVector) other;
        RereDiffVector p = (RereDiffVector) primal.mul(o.primal);
        IDoubleVector t = this.tangent.multiply(o.primal.getValue()).add(o.tangent.multiply(this.primal.getValue()));
        return new TangentDiffVector(p, t, List.of(this, o), p);
    }

    @Override public IDiffVector div(IDiffVector other) {
        TangentDiffVector o = (TangentDiffVector) other;
        RereDiffVector p = (RereDiffVector) primal.div(o.primal);
        IDoubleVector a = this.primal.getValue();
        IDoubleVector b = o.primal.getValue();
        IDoubleVector t = this.tangent.divide(b).sub(
                o.tangent.multiply(a).divide(b.square()));
        return new TangentDiffVector(p, t, List.of(this, o), p);
    }

    // -- scalar arithmetic --

    @Override public IDiffVector add(double scalar) {
        RereDiffVector p = (RereDiffVector) primal.add(scalar);
        return new TangentDiffVector(p, this.tangent.copy(), List.of(this), p);
    }

    @Override public IDiffVector sub(double scalar) {
        RereDiffVector p = (RereDiffVector) primal.sub(scalar);
        return new TangentDiffVector(p, this.tangent.copy(), List.of(this), p);
    }

    @Override public IDiffVector mul(double scalar) {
        RereDiffVector p = (RereDiffVector) primal.mul(scalar);
        IDoubleVector t = this.tangent.multiplyByScalar(scalar);
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    @Override public IDiffVector div(double scalar) {
        RereDiffVector p = (RereDiffVector) primal.div(scalar);
        IDoubleVector t = this.tangent.divideByScalar(scalar);
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    @Override public IDiffVector rsub(double scalar) {
        RereDiffVector p = (RereDiffVector) primal.rsub(scalar);
        IDoubleVector t = this.tangent.multiplyByScalar(-1.0);
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    @Override public IDiffVector rdiv(double scalar) {
        RereDiffVector p = (RereDiffVector) primal.rdiv(scalar);
        IDoubleVector a = this.primal.getValue();
        IDoubleVector t = this.tangent.multiplyByScalar(-scalar).divide(a.square());
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    // -- unary --

    @Override public IDiffVector neg() {
        RereDiffVector p = (RereDiffVector) primal.neg();
        IDoubleVector t = this.tangent.multiplyByScalar(-1.0);
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    @Override public IDiffVector pow(double n) {
        RereDiffVector p = (RereDiffVector) primal.pow(n);
        IDoubleVector xVal = this.primal.getValue();
        IDoubleVector t = this.tangent.multiplyByScalar(n).multiply(xVal.pow(n - 1));
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    // -- element-wise math --

    @Override public IDiffVector exp() { return unaryWithTangent(v -> v.exp(), t -> t.multiply(primal.getValue().exp())); }
    @Override public IDiffVector log() { return unaryWithTangent(v -> v.log(), t -> t.divide(primal.getValue())); }
    @Override public IDiffVector sin() { return unaryWithTangent(v -> v.sin(), t -> t.multiply(primal.getValue().cos())); }
    @Override public IDiffVector cos() { return unaryWithTangent(v -> v.cos(), t -> t.multiplyByScalar(-1.0).multiply(primal.getValue().sin())); }
    @Override public IDiffVector tan() { IDoubleVector z = primal.getValue().tan(); return unaryWithTangent(v -> z, t -> t.multiply(z.square().addScalar(1.0))); }
    @Override public IDiffVector tanh() { IDoubleVector z = primal.getValue().tanh(); return unaryWithTangent(v -> z, t -> t.multiply(z.map(x -> 1.0 - x * x))); }

    @Override public IDiffVector sigmoid() {
        IDoubleVector z = primal.getValue().sigmoid();
        return unaryWithTangent(v -> z, t -> t.multiply(z.map(x -> x * (1.0 - x))));
    }

    @Override public IDiffVector relu() {
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.relu();
        IDoubleVector t = IDoubleVector.of(new double[xVal.size()]);
        double[] tData = t.getData();
        double[] xData = xVal.getData();
        double[] inTData = this.tangent.getData();
        for (int i = 0; i < tData.length; i++) {
            tData[i] = inTData[i] * (xData[i] > 0.0 ? 1.0 : 0.0);
        }
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    @Override public IDiffVector softmax() {
        RereDiffVector p = (RereDiffVector) primal.softmax();
        IDoubleVector y = p.getValue();
        double[] yd = y.getData();
        double[] tData = this.tangent.getData();
        int n = yd.length;
        double dot = 0;
        for (int i = 0; i < n; i++) dot += yd[i] * tData[i];
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) jvp[i] = yd[i] * (tData[i] - dot);
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector logSoftmax() {
        RereDiffVector p = (RereDiffVector) primal.logSoftmax();
        IDoubleVector sm = primal.softmax().getValue();
        double[] smd = sm.getData();
        double[] tData = this.tangent.getData();
        int n = smd.length;
        double sumT = 0;
        for (int i = 0; i < n; i++) sumT += tData[i];
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) jvp[i] = tData[i] - smd[i] * sumT;
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector gelu() {
        RereDiffVector p = (RereDiffVector) primal.gelu();
        IDoubleVector xVal = primal.getValue();
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
        double g = 0.044715;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            double inner = sqrt2OverPi * (x + g * x * x * x);
            double tanhI = Math.tanh(inner);
            double sechSq = 1.0 - tanhI * tanhI;
            double din_dx = sqrt2OverPi * (1.0 + 3.0 * g * x * x);
            jvp[i] = tData[i] * (0.5 * (1.0 + tanhI) + 0.5 * x * sechSq * din_dx);
        }
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector clamp(double min, double max) {
        IDoubleVector xVal = primal.getValue();
        // Use primal.clamp() to maintain graph connectivity (not a disconnected leaf)
        RereDiffVector p = (RereDiffVector) primal.clamp(min, max);
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) {
            double v = xd[i];
            jvp[i] = tData[i] * (v >= min && v <= max ? 1.0 : 0.0);
        }
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector layerNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector p = (RereDiffVector) primal.layerNorm(gamma, beta, eps);
        IDoubleVector xVal = primal.getValue();
        double[] xd = xVal.getData();
        double[] gd = gamma.getValue().getData();
        double[] td = this.tangent.getData();
        int n = xd.length;
        int features = gd.length;
        int batch = n / features;
        double[] jvp = new double[n];
        // Per-batch-position normalization, matching RereDiffVector.layerNorm()
        for (int b = 0; b < batch; b++) {
            int off = b * features;
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
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector batchNorm(IDiffVector gamma, IDiffVector beta, double eps) {
        RereDiffVector p = (RereDiffVector) primal.batchNorm(gamma, beta, eps);
        IDoubleVector xVal = primal.getValue();
        double[] xd = xVal.getData();
        double[] gd = gamma.getValue().getData();
        double[] td = this.tangent.getData();
        int n = xd.length;
        int features = gd.length;
        int batch = n / features;
        // For each feature, compute JVP over batch dimension
        double[] jvp = new double[n];
        for (int j = 0; j < features; j++) {
            double mean = 0;
            for (int i = 0; i < batch; i++) mean += xd[i * features + j];
            mean /= batch;
            double var = 0;
            for (int i = 0; i < batch; i++) { double d = xd[i * features + j] - mean; var += d * d; }
            var /= batch;
            double sigma = Math.sqrt(var + eps);
            double invSig = 1.0 / sigma;
            // xHat and tangent stats
            double meanT = 0, meanTXHat = 0;
            double[] xHat = new double[batch];
            for (int i = 0; i < batch; i++) {
                int idx = i * features + j;
                xHat[i] = (xd[idx] - mean) * invSig;
                meanT += td[idx];
                meanTXHat += td[idx] * xHat[i];
            }
            meanT /= batch; meanTXHat /= batch;
            for (int i = 0; i < batch; i++) {
                int idx = i * features + j;
                jvp[idx] = gd[j] * invSig * (td[idx] - meanT - xHat[i] * meanTXHat);
            }
        }
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector dropout(double p) {
        throw new UnsupportedOperationException("dropout not supported in forward mode");
    }

    @Override public IDiffVector leakyRelu(double alpha) {
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.leakyRelu(alpha);
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) jvp[i] = tData[i] * (xd[i] > 0 ? 1.0 : alpha);
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector elu(double alpha) {
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.elu(alpha);
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) jvp[i] = tData[i] * (xd[i] >= 0 ? 1.0 : alpha * Math.exp(xd[i]));
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector selu() {
        double scale = 1.0507009873554804934193349852946;
        double alpha = 1.6732632423543772848170429916717;
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.selu();
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) jvp[i] = tData[i] * scale * (xd[i] >= 0 ? 1.0 : alpha * Math.exp(xd[i]));
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector silu() {
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.silu();
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            double s = 1.0 / (1.0 + Math.exp(-x));
            jvp[i] = tData[i] * (s + x * s * (1.0 - s));
        }
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector mish() {
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.mish();
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) {
            double x = xd[i];
            double sp = Math.log(1.0 + Math.exp(x));
            double th = Math.tanh(sp);
            double sig = 1.0 / (1.0 + Math.exp(-x));
            double thSq = th * th;
            jvp[i] = tData[i] * (th + x * (1.0 - thSq) * sig);
        }
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector softplus(double beta) {
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.softplus(beta);
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) {
            double bx = beta * xd[i];
            jvp[i] = tData[i] * (bx > 100 ? 1.0 : 1.0 / (1.0 + Math.exp(-bx)));
        }
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector hardtanh(double minVal, double maxVal) {
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.hardtanh(minVal, maxVal);
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        int n = xd.length;
        double[] jvp = new double[n];
        for (int i = 0; i < n; i++) {
            double v = xd[i];
            jvp[i] = tData[i] * (v >= minVal && v <= maxVal ? 1.0 : 0.0);
        }
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector slice(int start, int end) {
        // Forward: extract sub-vector. JVP: extract sub-tangent.
        RereDiffVector p = (RereDiffVector) primal.slice(start, end);
        IDoubleVector t = this.tangent.slice(start, end);
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    @Override public IDiffVector cat(IDiffVector... others) {
        // Forward: cat primals. JVP: cat tangents (cat is linear).
        IDiffVector[] primals = new IDiffVector[others.length];
        IDoubleVector[] tangents = new IDoubleVector[others.length];
        for (int i = 0; i < others.length; i++) {
            TangentDiffVector td = (TangentDiffVector) others[i];
            primals[i] = td.primal;
            tangents[i] = td.tangent;
        }
        RereDiffVector p = (RereDiffVector) this.primal.cat(primals);

        // Concatenate tangents
        int totalLen = this.tangent.size();
        for (IDoubleVector tv : tangents) totalLen += tv.size();
        double[] tData = new double[totalLen];
        int pos = 0;
        System.arraycopy(this.tangent.getData(), 0, tData, pos, this.tangent.size());
        pos += this.tangent.size();
        for (IDoubleVector tv : tangents) {
            System.arraycopy(tv.getData(), 0, tData, pos, tv.size());
            pos += tv.size();
        }

        java.util.List<TangentDiffVector> inputTangents = new java.util.ArrayList<>();
        inputTangents.add(this);
        for (IDiffVector o : others) inputTangents.add((TangentDiffVector) o);

        return new TangentDiffVector(p, IDoubleVector.of(tData), inputTangents, p);
    }

    @Override public IDiffMatrix reshape(int rows, int cols) {
        throw new UnsupportedOperationException(
            "reshape not supported in forward mode (TangentDiffMatrix not available)");
    }

    @Override public IDiffVector abs() {
        IDoubleVector xVal = primal.getValue();
        RereDiffVector p = (RereDiffVector) primal.abs();
        // Use subgradient convention: sign(x) = 1 at x=0 (consistent with relu)
        double[] xd = xVal.getData();
        double[] tData = this.tangent.getData();
        double[] jvp = new double[xd.length];
        for (int i = 0; i < xd.length; i++) {
            jvp[i] = tData[i] * (xd[i] >= 0.0 ? 1.0 : -1.0);
        }
        return new TangentDiffVector(p, IDoubleVector.of(jvp), List.of(this), p);
    }

    @Override public IDiffVector sqrt() {
        IDoubleVector z = primal.getValue().sqrt();
        return unaryWithTangent(v -> z, t -> t.divide(z.multiplyByScalar(2.0)));
    }

    @Override public IDiffVector square() {
        RereDiffVector p = (RereDiffVector) primal.square();
        IDoubleVector t = this.tangent.multiplyByScalar(2.0).multiply(primal.getValue());
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    // -- reductions --

    @Override public IDiffVector sum() {
        RereDiffVector p = (RereDiffVector) primal.sum();
        double s = this.tangent.sumValue();
        IDoubleVector t = IDoubleVector.of(s);
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    @Override public IDiffVector mean() {
        RereDiffVector p = (RereDiffVector) primal.mean();
        double m = this.tangent.meanValue();
        IDoubleVector t = IDoubleVector.of(m);
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    @Override public IDiffVector cumsum() {
        RereDiffVector p = (RereDiffVector) primal.cumsum();
        return new TangentDiffVector(p, this.tangent.cumsum(), List.of(this), p);
    }

    @Override public IDiffVector cumprod() {
        RereDiffVector p = (RereDiffVector) primal.cumprod();
        int n = primal.getValue().size();
        double[] x = primal.getValue().getData();
        double[] v = this.tangent.getData();
        double[] tc = new double[n];
        double[] fc = new double[n];
        fc[0] = x[0];
        tc[0] = v[0];
        for (int k = 1; k < n; k++) {
            tc[k] = tc[k - 1] * x[k] + fc[k - 1] * v[k];
            fc[k] = fc[k - 1] * x[k];
        }
        IDoubleVector t = IDoubleVector.of(tc);
        return new TangentDiffVector(p, t, List.of(this), p);
    }

    // -- vector ops --

    @Override public IDiffVector dot(IDiffVector other) {
        TangentDiffVector o = (TangentDiffVector) other;
        RereDiffVector p = (RereDiffVector) primal.dot(o.primal);
        IDoubleVector a = this.primal.getValue();
        IDoubleVector b = o.primal.getValue();
        double d = this.tangent.dotValue(b) + a.dotValue(o.tangent);
        IDoubleVector t = IDoubleVector.of(d);
        return new TangentDiffVector(p, t, List.of(this, o), p);
    }

    // -- covariant overrides from IDoubleVector / IVector --

    @Override public IDiffVector copy() {
        // Note: copy creates a fresh primal leaf node. This is correct for forward-mode AD
        // where copy() snapshots the current value+tangent state.
        return new TangentDiffVector((RereDiffVector) primal.copy(), tangent.copy(), List.of(), null);
    }

    @Override public IDiffVector detach() { return primal.detach(); }
    @Override public IDiffVector divideInPlace(double alpha) { throw new UnsupportedOperationException(); }
    @Override public IDiffVector addScalarInPlace(double p) { throw new UnsupportedOperationException(); }
    @Override public IDiffVector subScalarInPlace(double p) { throw new UnsupportedOperationException(); }
    @Override public IDiffVector multiplyByScalarInPlace(double p) { throw new UnsupportedOperationException(); }
    @Override public IDiffVector addInPlace(IVector<Double> vec) { throw new UnsupportedOperationException(); }
    @Override public IDiffVector subInPlace(IVector<Double> vec) { throw new UnsupportedOperationException(); }
    @Override public IDiffVector multiplyInPlace(IVector<Double> vec) { throw new UnsupportedOperationException(); }
    @Override public IDiffVector negInPlace() { throw new UnsupportedOperationException(); }

    @Override public IDiffVector add(IVector<Double> vec) { return this.add((IDiffVector) vec); }
    @Override public IDiffVector sub(IVector<Double> vec) { return this.sub((IDiffVector) vec); }
    @Override public IDiffVector multiply(IVector<Double> vec) { return this.mul((IDiffVector) vec); }
    @Override public IDiffVector divide(IVector<Double> vec) { return this.div((IDiffVector) vec); }
    @Override public IDiffVector dot(IVector<Double> vec) { return this.dot((IDiffVector) vec); }
    @Override public IDiffVector innerProduct(IVector<Double> vec) { return this.dot((IDiffVector) vec); }

    @Override public double dtw(IVector<Double> other) { return primal.dtw(other); }
    @Override public double normInf() { return primal.normInf(); }

    // ---- IDoubleVector bridge methods (forward-mode AD) ----

    @Override
    public IDiffVector log10() {
        IDoubleVector xVal = primal.getValue();
        return unaryWithTangent(
            v -> xVal.log10(),
            t -> t.divide(xVal.multiplyByScalar(Math.log(10)))
        );
    }

    @Override
    public IDiffVector arcsin() {
        IDoubleVector xVal = primal.getValue();
        IDoubleVector denom = xVal.map(v -> Math.sqrt(1.0 - v * v));
        return unaryWithTangent(v -> xVal.arcsin(), t -> t.divide(denom));
    }

    @Override
    public IDiffVector arccos() {
        IDoubleVector xVal = primal.getValue();
        IDoubleVector denom = xVal.map(v -> Math.sqrt(1.0 - v * v));
        return unaryWithTangent(v -> xVal.arccos(), t -> t.divide(denom).multiplyByScalar(-1.0));
    }

    @Override
    public IDiffVector arctan() {
        IDoubleVector xVal = primal.getValue();
        return unaryWithTangent(v -> xVal.arctan(), t -> t.divide(xVal.square().addScalar(1.0)));
    }

    @Override
    public IDiffVector sinh() {
        IDoubleVector xVal = primal.getValue();
        return unaryWithTangent(v -> xVal.sinh(), t -> t.multiply(xVal.cosh()));
    }

    @Override
    public IDiffVector cosh() {
        IDoubleVector xVal = primal.getValue();
        return unaryWithTangent(v -> xVal.cosh(), t -> t.multiply(xVal.sinh()));
    }

    @Override
    public IDiffVector reciprocal() {
        IDoubleVector xVal = primal.getValue();
        return unaryWithTangent(
            v -> xVal.reciprocal(),
            t -> t.divide(xVal.square()).multiplyByScalar(-1.0)
        );
    }

    // -- rounding ops (zero gradient, tangent = 0) --

    @Override
    public IDiffVector round() {
        return unaryWithTangent(v -> v.round(), t -> IDoubleVector.zeros(t.size()));
    }

    @Override
    public IDiffVector floor() {
        return unaryWithTangent(v -> v.floor(), t -> IDoubleVector.zeros(t.size()));
    }

    @Override
    public IDiffVector ceil() {
        return unaryWithTangent(v -> v.ceil(), t -> IDoubleVector.zeros(t.size()));
    }

    @Override
    public IDiffVector trunc() {
        return unaryWithTangent(v -> v.trunc(), t -> IDoubleVector.zeros(t.size()));
    }

    @Override
    public IDiffVector sign() {
        return unaryWithTangent(v -> v.sign(), t -> IDoubleVector.zeros(t.size()));
    }

    // -- in-place (unsupported in forward mode) --

    @Override public IDiffVector addInPlace(IDiffVector other) { throw new UnsupportedOperationException(); }
    @Override public IDiffVector mulInPlace(double scalar) { throw new UnsupportedOperationException(); }

    // -- helper --

    /**
     * Creates a forward-mode AD node with both primal and tangent propagation.
     *
     * <p><b>IMPORTANT:</b> The primal created here is a leaf node (disconnected from the
     * backward graph). Forward-mode AD only needs JVP, not reverse-mode gradient flow.
     * Calling {@link #backward()} on the primal returned by this method will NOT
     * propagate gradients to upstream parameters.</p>
     *
     * <p>For mixed-mode usage (HVP via {@link MixedMode}), use the original primal
     * (not the TangentDiffVector's primal) for backward pass.</p>
     *
     * @see com.yishape.lab.math.autodiff.MixedMode#hvp(Function, IDiffVector, IDiffVector)
     */
    private IDiffVector unaryWithTangent(java.util.function.Function<IDoubleVector, IDoubleVector> forward,
            java.util.function.Function<IDoubleVector, IDoubleVector> tangentFn) {
        IDoubleVector resultVal = forward.apply(primal.getValue());
        RereDiffVector p = new RereDiffVector(resultVal);
        p.tensor.setIsLeaf(true); // explicitly leaf — see Javadoc above
        IDoubleVector t = tangentFn.apply(this.tangent);
        return new TangentDiffVector(p, t, List.of(this), p);
    }
}
