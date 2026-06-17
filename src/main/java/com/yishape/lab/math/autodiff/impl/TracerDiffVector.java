package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.List;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.autodiff.IDiffMatrix;
import com.yishape.lab.math.autodiff.IDiffVector;

/**
 * Tracing proxy that records fusible element-wise ops; unsupported ops trigger fallback.
 * 追踪代理：记录可融合逐元素运算；遇到不支持运算则触发回退。
 *
 * @see com.yishape.lab.math.optimize.autodiff.AD#elementwise
 */
public class TracerDiffVector implements IDiffVector {

    private static final long serialVersionUID = 5L;

    private final RereDiffVector original;
    final List<FusedOps.FusedOp> ops = new ArrayList<>();
    boolean fusible = true;

    public TracerDiffVector(RereDiffVector original) {
        this.original = original;
    }

    public boolean isFusible() {
        return fusible && !ops.isEmpty();
    }

    public IDiffVector buildFused() {
        return new FusedOps(original, ops).compute();
    }

    // ---- unsupported: throw to trigger fallback / 不支持则抛异常以触发回退 ----

    private void bail() {
        fusible = false;
        throw new RuntimeException("non-fusible");
    }

    @Override public IDoubleVector getValue() { bail(); return null; }
    @Override public IDoubleVector getGradient() { bail(); return null; }
    @Override public boolean isLeaf() { bail(); return false; }
    @Override public void backward() { bail(); }
    @Override public void backward(IDoubleVector g) { bail(); }
    @Override public void zeroGradient() { bail(); }
    @Override public IDiffVector grad() { bail(); return null; }
    @Override public IDiffVector sum() { bail(); return null; }
    @Override public IDiffVector mean() { bail(); return null; }
    @Override public IDiffVector dot(IDiffVector other) { bail(); return null; }
    @Override public IDiffVector sin() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SIN, 0)); return this; }
    @Override public IDiffVector cos() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.COS, 0)); return this; }
    @Override public IDiffVector tan() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.TAN, 0)); return this; }
    @Override public IDiffVector broadcast(int n) { bail(); return null; }

    // ---- supported unary element-wise ----

    @Override public IDiffVector exp()  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.EXP, 0)); return this; }
    @Override public IDiffVector log()  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.LOG, 0)); return this; }
    @Override public IDiffVector sqrt() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SQRT, 0)); return this; }
    @Override public IDiffVector square() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SQUARE, 0)); return this; }
    @Override public IDiffVector sigmoid() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SIGMOID, 0)); return this; }
    @Override public IDiffVector tanh() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.TANH, 0)); return this; }
    @Override public IDiffVector relu() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.RELU, 0)); return this; }
    @Override public IDiffVector abs()  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.ABS, 0)); return this; }
    @Override public IDiffVector neg()  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.NEG, 0)); return this; }

    // ---- supported scalar parameter ----

    @Override public IDiffVector pow(double n) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.POW, n)); return this; }

    // ---- supported binary variable: only fuse when other is a real variable ----

    @Override
    public IDiffVector add(IDiffVector other) {
        if (other instanceof RereDiffVector rv) {
            ops.add(new FusedOps.FusedOp(FusedOps.OpType.ADD_V, 0, rv));
            return this;
        }
        bail(); return null;
    }

    @Override
    public IDiffVector sub(IDiffVector other) {
        if (other instanceof RereDiffVector rv) {
            ops.add(new FusedOps.FusedOp(FusedOps.OpType.SUB_V, 0, rv));
            return this;
        }
        bail(); return null;
    }

    @Override
    public IDiffVector mul(IDiffVector other) {
        if (other instanceof RereDiffVector rv) {
            ops.add(new FusedOps.FusedOp(FusedOps.OpType.MUL_V, 0, rv));
            return this;
        }
        bail(); return null;
    }

    @Override
    public IDiffVector div(IDiffVector other) {
        if (other instanceof RereDiffVector rv) {
            ops.add(new FusedOps.FusedOp(FusedOps.OpType.DIV_V, 0, rv));
            return this;
        }
        bail(); return null;
    }

    // ---- scalar arithmetic overrides ----

    @Override public IDiffVector add(double scalar) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.ADD_C, scalar)); return this; }
    @Override public IDiffVector sub(double scalar) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SUB_C, scalar)); return this; }
    @Override public IDiffVector mul(double scalar) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.MUL_C, scalar)); return this; }
    @Override public IDiffVector div(double scalar) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.DIV_C, scalar)); return this; }

    @Override public IDiffVector rsub(double scalar) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.RSUB_C, scalar)); return this; }
    @Override public IDiffVector rdiv(double scalar) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.RDIV_C, scalar)); return this; }
    @Override public IDiffVector softmax() { bail(); return null; }
    @Override public IDiffVector logSoftmax() { bail(); return null; }
    @Override public IDiffVector gelu() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.GELU, 0)); return this; }
    @Override public IDiffVector clamp(double min, double max) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.CLAMP, min, max, null)); return this; }
    @Override public IDiffVector dropout(double p) { bail(); return null; }

    // ---- new activations: fused element-wise ----
    @Override public IDiffVector leakyRelu(double alpha) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.LEAKY_RELU, alpha)); return this; }
    @Override public IDiffVector elu(double alpha) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.ELU, alpha)); return this; }
    @Override public IDiffVector selu() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SELU, 0)); return this; }
    @Override public IDiffVector silu() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SILU, 0)); return this; }
    @Override public IDiffVector mish() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.MISH, 0)); return this; }
    @Override public IDiffVector softplus(double beta) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SOFTPLUS, beta)); return this; }
    @Override public IDiffVector hardtanh(double minVal, double maxVal) { ops.add(new FusedOps.FusedOp(FusedOps.OpType.HARDTANH, minVal, maxVal, null)); return this; }
    @Override public IDiffVector layerNorm(IDiffVector gamma, IDiffVector beta, double eps) { bail(); return null; }
    @Override public IDiffVector batchNorm(IDiffVector gamma, IDiffVector beta, double eps) { bail(); return null; }
    @Override public IDiffVector slice(int start, int length) { bail(); return null; }
    @Override public IDiffVector cat(IDiffVector... others) { bail(); return null; }

    @Override public IDiffMatrix reshape(int rows, int cols) { bail(); return null; }

    @Override public IDiffVector addInPlace(IDiffVector other) { bail(); return null; }
    @Override public IDiffVector mulInPlace(double scalar) { bail(); return null; }

    // -- covariant overrides from IDoubleVector / IVector --

    @Override public IDiffVector copy() {
        // Preserve accumulated ops list so the copy reflects the same fusion chain
        TracerDiffVector c = new TracerDiffVector(original);
        c.ops.addAll(this.ops);
        c.fusible = this.fusible;
        return c;
    }

    @Override public IDiffVector detach() { return original.detach(); }
    @Override public IDiffVector divideInPlace(double alpha) { bail(); return null; }
    @Override public IDiffVector addScalarInPlace(double p) { bail(); return null; }
    @Override public IDiffVector subScalarInPlace(double p) { bail(); return null; }
    @Override public IDiffVector multiplyByScalarInPlace(double p) { bail(); return null; }
    @Override public IDiffVector addInPlace(IVector<Double> vec) { bail(); return null; }
    @Override public IDiffVector subInPlace(IVector<Double> vec) { bail(); return null; }
    @Override public IDiffVector multiplyInPlace(IVector<Double> vec) { bail(); return null; }
    @Override public IDiffVector negInPlace() { bail(); return null; }
    @Override public IDiffVector add(IVector<Double> vec) { return this.add((IDiffVector) vec); }
    @Override public IDiffVector sub(IVector<Double> vec) { return this.sub((IDiffVector) vec); }
    @Override public IDiffVector multiply(IVector<Double> vec) { return this.mul((IDiffVector) vec); }
    @Override public IDiffVector divide(IVector<Double> vec) { return this.div((IDiffVector) vec); }
    @Override public IDiffVector dot(IVector<Double> vec) { return this.dot((IDiffVector) vec); }
    @Override public IDiffVector innerProduct(IVector<Double> vec) { return this.dot((IDiffVector) vec); }
    @Override public double dtw(IVector<Double> other) { bail(); return 0; }
    @Override public double normInf() { bail(); return 0; }
}
