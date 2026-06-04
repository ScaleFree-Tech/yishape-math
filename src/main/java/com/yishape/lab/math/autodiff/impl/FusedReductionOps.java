package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.linalg.IDoubleVector;

/**
 * Fused element-wise + reduction operator chain.
 * Applies a sequence of element-wise ops followed by a reduction terminator
 * (softmax, normalize, layerNorm, sum, mean) in a single forward/backward kernel,
 * avoiding intermediate tensor allocations.
 *
 * <p>Built via {@code AD.fuseReduce(x).exp().relu().softmax()} or accessed directly.
 * The element-wise ops reuse the same {@link FusedOps.OpType} kernel logic.
 * Reduction-specific intermediate values (softmax output, norm values, etc.)
 * are saved for the backward pass.</p>
 */
public class FusedReductionOps {

    private final RereDiffVector x;
    private final List<FusedOps.FusedOp> elementOps = new ArrayList<>();

    /** Reduction terminator kind. */
    enum ReduceOp {
        SOFTMAX, LOG_SOFTMAX, NORMALIZE, LAYER_NORM, SUM, MEAN
    }

    private ReduceOp reduceOp;
    private double reduceParam = Double.NaN;
    private double reduceParam2 = Double.NaN;

    public FusedReductionOps(RereDiffVector x) {
        this.x = x;
    }

    // ---- Element-wise builder methods (delegate to same FusedOp types) ----

    public FusedReductionOps exp()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.EXP, 0)); return this; }
    public FusedReductionOps log()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.LOG, 0)); return this; }
    public FusedReductionOps sqrt()   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.SQRT, 0)); return this; }
    public FusedReductionOps square() { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.SQUARE, 0)); return this; }
    public FusedReductionOps sigmoid() { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.SIGMOID, 0)); return this; }
    public FusedReductionOps tanh()   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.TANH, 0)); return this; }
    public FusedReductionOps relu()   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.RELU, 0)); return this; }
    public FusedReductionOps abs()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.ABS, 0)); return this; }
    public FusedReductionOps neg()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.NEG, 0)); return this; }
    public FusedReductionOps sin()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.SIN, 0)); return this; }
    public FusedReductionOps cos()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.COS, 0)); return this; }
    public FusedReductionOps tan()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.TAN, 0)); return this; }
    public FusedReductionOps gelu()   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.GELU, 0)); return this; }
    public FusedReductionOps reciprocal() { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.RECIPROCAL, 0)); return this; }
    public FusedReductionOps pow(double n)   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.POW, n)); return this; }
    public FusedReductionOps add(double c)   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.ADD_C, c)); return this; }
    public FusedReductionOps sub(double c)   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.SUB_C, c)); return this; }
    public FusedReductionOps mul(double c)   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.MUL_C, c)); return this; }
    public FusedReductionOps div(double c)   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.DIV_C, c)); return this; }
    public FusedReductionOps leakyRelu(double a) { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.LEAKY_RELU, a)); return this; }
    public FusedReductionOps elu(double a)   { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.ELU, a)); return this; }
    public FusedReductionOps selu()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.SELU, 0)); return this; }
    public FusedReductionOps silu()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.SILU, 0)); return this; }
    public FusedReductionOps mish()    { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.MISH, 0)); return this; }
    public FusedReductionOps softplus(double b) { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.SOFTPLUS, b)); return this; }
    public FusedReductionOps clamp(double min, double max) { elementOps.add(new FusedOps.FusedOp(FusedOps.OpType.CLAMP, min, max, null)); return this; }

    // ---- Reduction terminators ----

    public IDiffVector softmax() {
        this.reduceOp = ReduceOp.SOFTMAX;
        return compute();
    }

    public IDiffVector logSoftmax() {
        this.reduceOp = ReduceOp.LOG_SOFTMAX;
        return compute();
    }

    public IDiffVector normalize(double p) {
        this.reduceOp = ReduceOp.NORMALIZE;
        this.reduceParam = p;
        return compute();
    }

    public IDiffVector sum() {
        this.reduceOp = ReduceOp.SUM;
        return compute();
    }

    public IDiffVector mean() {
        this.reduceOp = ReduceOp.MEAN;
        return compute();
    }

    public IDiffVector layerNorm(double eps) {
        this.reduceOp = ReduceOp.LAYER_NORM;
        this.reduceParam = eps;
        return compute();
    }

    // ---- Compute ----

    private IDiffVector compute() {
        if (reduceOp == null) {
            // No reduction — fall back to plain FusedOps
            if (elementOps.isEmpty()) return x;
            FusedOps fo = new FusedOps(x, elementOps);
            return fo.compute();
        }

        int n = x.value.size();
        double[] xData = x.value.getData();

        // Apply element-wise chain
        double[] mid = forwardElementOps(xData, n);
        int numElem = elementOps.size();
        double[][] saved = null;
        if (numElem > 0) {
            saved = new double[numElem][];
            for (int j = 0; j < numElem; j++) saved[j] = AutodiffBufferPool.acquire(n);
            forwardElementOpsWithSaved(xData, n, saved);
            System.arraycopy(mid, 0, saved[numElem - 1], 0, n); // last saved is input to reduction
        }

        // Apply reduction
        double[] result = null;
        double[] reduceSaved = null;

        switch (reduceOp) {
            case SOFTMAX -> {
                result = softmaxForward(mid, n);
                reduceSaved = result.clone(); // save softmax output for backward
            }
            case LOG_SOFTMAX -> {
                double[] sm = new double[n];
                result = logSoftmaxForward(mid, n, sm);
                reduceSaved = sm; // save softmax output for backward
            }
            case NORMALIZE -> {
                double norm = normalizeForward(mid, n, reduceParam);
                result = mid; // in-place, mid already normalized
                reduceSaved = new double[]{norm};
            }
            case LAYER_NORM -> {
                double[] stats = new double[2]; // [mean, invStd]
                result = layerNormForward(mid, n, reduceParam, stats);
                reduceSaved = stats;
            }
            case SUM -> {
                double s = 0;
                for (int i = 0; i < n; i++) s += mid[i];
                result = new double[]{s};
            }
            case MEAN -> {
                double s = 0;
                for (int i = 0; i < n; i++) s += mid[i];
                result = new double[]{s / n};
            }
        }

        IDoubleVector resultVal = IDoubleVector.of(result);
        RereDiffVector self = this.x;

        // Capture for backward
        final int fn = n;
        final int fNumElem = numElem;
        final double[][] fSaved = saved;
        final double[] fReduceSaved = reduceSaved;
        final double[] fResult = result;
        final double[] fMid = (reduceOp == ReduceOp.NORMALIZE) ? mid.clone() : mid;
        final double[] fXData = (numElem > 0) ? xData : null;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] buf = AutodiffBufferPool.acquire(fn);
            double[] gd = gradOut.getData();

            // Step 1: Reduction backward → produce gradient w.r.t. mid
            switch (reduceOp) {
                case SOFTMAX -> reductionBackwardSoftmax(gd, fReduceSaved, buf, fn);
                case LOG_SOFTMAX -> reductionBackwardLogSoftmax(gd, fReduceSaved, buf, fn);
                case NORMALIZE -> reductionBackwardNormalize(gd, fMid, fReduceSaved, buf, fn, reduceParam);
                case LAYER_NORM -> reductionBackwardLayerNorm(gd, fMid, fReduceSaved, buf, fn, reduceParam);
                case SUM -> {
                    double g = gd[0];
                    for (int i = 0; i < fn; i++) buf[i] = g;
                }
                case MEAN -> {
                    double g = gd[0] / fn;
                    for (int i = 0; i < fn; i++) buf[i] = g;
                }
            }

            // Step 2: Element-wise backward chain (reverse order)
            if (fNumElem > 0) {
                for (int j = fNumElem - 1; j >= 0; j--) {
                    FusedOps.FusedOp op = elementOps.get(j);
                    double[] inputVals = fSaved[j];
                    for (int i = 0; i < fn; i++) {
                        double inputV = inputVals[i];
                        buf[i] = applyElementGradient(op, buf[i], inputV);
                    }
                }
            }

            self.accGradFromPooled(buf, fn);

            // Release pooled buffers
            if (fSaved != null) {
                for (int j = 0; j < fNumElem; j++) {
                    AutodiffBufferPool.release(fSaved[j]);
                }
            }
        };

        List<RereDiffVector> inputs = new ArrayList<>();
        inputs.add(self);
        return new RereDiffVector(resultVal, inputs, backwardFn);
    }

    // ---- Element-wise forward helpers ----

    private double[] forwardElementOps(double[] xData, int n) {
        if (elementOps.isEmpty()) return xData;
        double[] cur = AutodiffBufferPool.acquire(n);
        System.arraycopy(xData, 0, cur, 0, n);
        double[] next = AutodiffBufferPool.acquire(n);
        for (FusedOps.FusedOp op : elementOps) {
            for (int i = 0; i < n; i++) {
                next[i] = applyElementForward(op, cur[i]);
            }
            double[] tmp = cur; cur = next; next = tmp;
        }
        AutodiffBufferPool.release(next);
        return cur;
    }

    private void forwardElementOpsWithSaved(double[] xData, int n, double[][] saved) {
        double[] cur = AutodiffBufferPool.acquire(n);
        System.arraycopy(xData, 0, cur, 0, n);
        double[] next = AutodiffBufferPool.acquire(n);
        for (int j = 0; j < elementOps.size(); j++) {
            System.arraycopy(cur, 0, saved[j], 0, n); // save input to this op
            FusedOps.FusedOp op = elementOps.get(j);
            for (int i = 0; i < n; i++) {
                next[i] = applyElementForward(op, cur[i]);
            }
            double[] tmp = cur; cur = next; next = tmp;
        }
        // cur now holds the final element-wise output
        System.arraycopy(cur, 0, saved[elementOps.size() - 1], 0, n);
        AutodiffBufferPool.release(cur);
        AutodiffBufferPool.release(next);
    }

    // ---- Reduction forward helpers ----

    static double[] softmaxForward(double[] x, int n) {
        double maxVal = x[0];
        for (int i = 1; i < n; i++) { if (x[i] > maxVal) maxVal = x[i]; }
        double[] result = new double[n];
        double sumExp = 0;
        for (int i = 0; i < n; i++) {
            result[i] = Math.exp(x[i] - maxVal);
            sumExp += result[i];
        }
        for (int i = 0; i < n; i++) result[i] /= sumExp;
        return result;
    }

    static double[] logSoftmaxForward(double[] x, int n, double[] smOut) {
        double maxVal = x[0];
        for (int i = 1; i < n; i++) { if (x[i] > maxVal) maxVal = x[i]; }
        double sumExp = 0;
        double[] expVals = new double[n];
        for (int i = 0; i < n; i++) {
            expVals[i] = Math.exp(x[i] - maxVal);
            sumExp += expVals[i];
        }
        double logSumExp = Math.log(sumExp) + maxVal;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            smOut[i] = expVals[i] / sumExp;
            result[i] = x[i] - logSumExp;
        }
        return result;
    }

    static double normalizeForward(double[] x, int n, double p) {
        double normP = 0;
        for (int i = 0; i < n; i++) {
            normP += Math.pow(Math.abs(x[i]), p);
        }
        double norm = Math.pow(normP, 1.0 / p);
        if (norm > 1e-12) {
            for (int i = 0; i < n; i++) x[i] /= norm;
        }
        return norm;
    }

    static double[] layerNormForward(double[] x, int n, double eps, double[] stats) {
        double mean = 0;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        double var = 0;
        for (int i = 0; i < n; i++) {
            double d = x[i] - mean;
            var += d * d;
        }
        var /= n;
        double invStd = 1.0 / Math.sqrt(var + eps);
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = (x[i] - mean) * invStd;
        }
        stats[0] = mean;
        stats[1] = invStd;
        return result;
    }

    // ---- Reduction backward helpers ----

    static void reductionBackwardSoftmax(double[] grad, double[] y, double[] buf, int n) {
        double dot = 0;
        for (int i = 0; i < n; i++) dot += grad[i] * y[i];
        for (int i = 0; i < n; i++) buf[i] = y[i] * (grad[i] - dot);
    }

    static void reductionBackwardLogSoftmax(double[] grad, double[] sm, double[] buf, int n) {
        double sumGrad = 0;
        for (int i = 0; i < n; i++) sumGrad += grad[i];
        for (int i = 0; i < n; i++) buf[i] = grad[i] - sm[i] * sumGrad;
    }

    static void reductionBackwardNormalize(double[] grad, double[] y, double[] saved, double[] buf, int n, double p) {
        double norm = saved[0];
        if (norm < 1e-12) {
            for (int i = 0; i < n; i++) buf[i] = grad[i];
            return;
        }
        // y-based formula: dx_orig[i] = (grad[i] - sign(y[i]) * |y[i]|^(p-1) * sum_j(grad[j]*y[j])) / N
        double dot = 0;
        for (int i = 0; i < n; i++) dot += grad[i] * y[i];
        for (int i = 0; i < n; i++) {
            double yi = y[i];
            double signY = Math.signum(yi);
            buf[i] = (grad[i] - signY * Math.pow(Math.abs(yi), p - 1) * dot) / norm;
        }
    }

    static void reductionBackwardLayerNorm(double[] grad, double[] x, double[] stats, double[] buf, int n, double eps) {
        double mean = stats[0];
        double invStd = stats[1];
        double sumGrad = 0;
        double sumGradXhat = 0;
        for (int i = 0; i < n; i++) {
            sumGrad += grad[i];
            sumGradXhat += grad[i] * (x[i] - mean) * invStd;
        }
        for (int i = 0; i < n; i++) {
            double xhat = (x[i] - mean) * invStd;
            buf[i] = invStd * (grad[i] - (sumGrad + xhat * sumGradXhat) / n);
        }
    }

    // ---- Element-wise kernel helpers ----

    static double applyElementForward(FusedOps.FusedOp op, double x) {
        return switch (op.type) {
            case EXP     -> Math.exp(x);
            case LOG     -> Math.log(x);
            case SQRT    -> Math.sqrt(x);
            case SQUARE  -> x * x;
            case SIGMOID -> 1.0 / (1.0 + Math.exp(-x));
            case TANH    -> Math.tanh(x);
            case RELU    -> Math.max(0, x);
            case ABS     -> Math.abs(x);
            case NEG     -> -x;
            case SIN     -> Math.sin(x);
            case COS     -> Math.cos(x);
            case TAN     -> Math.tan(x);
            case GELU    -> x * 0.5 * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
            case RECIPROCAL -> 1.0 / x;
            case POW     -> Math.pow(x, op.param);
            case ADD_C   -> x + op.param;
            case SUB_C   -> x - op.param;
            case MUL_C   -> x * op.param;
            case DIV_C   -> x / op.param;
            case RSUB_C  -> op.param - x;
            case RDIV_C  -> op.param / x;
            case CLAMP   -> Math.max(op.param, Math.min(op.param2, x));
            case LEAKY_RELU -> x >= 0 ? x : op.param * x;
            case ELU     -> x >= 0 ? x : op.param * (Math.exp(x) - 1);
            case SELU    -> { double a=1.6732632423543778, s=1.0507009873554804; yield s * (x >= 0 ? x : a * (Math.exp(x) - 1)); }
            case SILU    -> x / (1.0 + Math.exp(-x));
            case MISH    -> x * Math.tanh(Math.log1p(Math.exp(x)));
            case SOFTPLUS -> Math.log1p(Math.exp(op.param * x)) / op.param;
            case HARDTANH -> Math.max(op.param, Math.min(op.param2, x));
            default -> x;
        };
    }

    static double applyElementGradient(FusedOps.FusedOp op, double grad, double input) {
        return switch (op.type) {
            case EXP     -> grad * Math.exp(input);
            case LOG     -> grad / input;
            case SQRT    -> grad / (2.0 * Math.sqrt(input));
            case SQUARE  -> grad * 2.0 * input;
            case SIGMOID -> { double s = 1.0/(1.0+Math.exp(-input)); yield grad * s * (1.0 - s); }
            case TANH    -> { double t = Math.tanh(input); yield grad * (1.0 - t * t); }
            case RELU    -> input > 0 ? grad : 0;
            case ABS     -> grad * Math.signum(input);
            case NEG     -> -grad;
            case SIN     -> grad * Math.cos(input);
            case COS     -> -grad * Math.sin(input);
            case TAN     -> { double c = Math.cos(input); yield grad / (c * c); }
            case GELU    -> {
                double c = Math.sqrt(2.0 / Math.PI);
                double t = Math.tanh(c * (input + 0.044715 * input * input * input));
                double dt = 1.0 - t * t;
                double dx = c * (1.0 + 3.0 * 0.044715 * input * input);
                yield grad * (0.5 * (1.0 + t) + 0.5 * input * dt * dx);
            }
            case RECIPROCAL -> -grad / (input * input);
            case POW     -> grad * op.param * Math.pow(input, op.param - 1);
            case ADD_C   -> grad;
            case SUB_C   -> grad;
            case MUL_C   -> grad * op.param;
            case DIV_C   -> grad / op.param;
            case RSUB_C  -> -grad;
            case RDIV_C  -> -grad * op.param / (input * input);
            case CLAMP   -> (input > op.param && input < op.param2) ? grad : 0;
            case LEAKY_RELU -> input >= 0 ? grad : grad * op.param;
            case ELU     -> input >= 0 ? grad : grad * op.param * Math.exp(input);
            case SELU    -> {
                double a=1.6732632423543778, s=1.0507009873554804;
                yield s * (input >= 0 ? grad : grad * a * Math.exp(input));
            }
            case SILU    -> {
                double s = 1.0/(1.0+Math.exp(-input));
                yield grad * (s + input * s * (1.0 - s));
            }
            case MISH    -> {
                double sp = Math.log1p(Math.exp(input));
                double t = Math.tanh(sp);
                double sig = 1.0/(1.0+Math.exp(-input));
                yield grad * (t + input * (1.0 - t*t) * sig);
            }
            case SOFTPLUS -> {
                double e = Math.exp(op.param * input);
                yield grad * e / (1.0 + e);
            }
            case HARDTANH -> (input > op.param && input < op.param2) ? grad : 0;
            default -> grad;
        };
    }
}
