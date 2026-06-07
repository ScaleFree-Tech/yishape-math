package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import com.yishape.lab.math.linalg.IDoubleVector;
import com.yishape.lab.math.autodiff.IDiffVector;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Fused element-wise operator chain on a single vector (single forward/backward kernel).
 * 单向量融合逐元素算子链（单次前向/反向内核）。
 *
 * <p>Built via {@link com.yishape.lab.math.optimize.autodiff.AD#fuse(IDiffVector)} or
 * {@link TracerDiffVector} auto-fusion.
 * 通过 {@link com.yishape.lab.math.optimize.autodiff.AD#fuse(IDiffVector)} 或
 * {@link TracerDiffVector} 自动融合构建。</p>
 *
 * @deprecated Use {@link TensorFusedOps} for tensor-native fused operations.
 */
@Deprecated
public class FusedOps {

    private final RereDiffVector x;
    private final List<FusedOp> ops = new ArrayList<>();

    public FusedOps(IDiffVector x) {
        this.x = (RereDiffVector) x;
    }

    FusedOps(RereDiffVector x, List<FusedOp> ops) {
        this.x = x;
        this.ops.addAll(ops);
    }

    /** Supported fused unary/binary op kinds. / 可融合的逐元素算子类型。 */
    enum OpType {
        EXP, LOG, SQRT, SQUARE, SIGMOID, TANH, RELU, ABS, NEG,
        SIN, COS, TAN, GELU, ERF, RECIPROCAL,
        POW, ADD_C, SUB_C, MUL_C, DIV_C, RSUB_C, RDIV_C, CLAMP,
        ADD_V, SUB_V, MUL_V, DIV_V, MAX_V, MIN_V,
        LEAKY_RELU, ELU, SELU, SILU, MISH, SOFTPLUS, HARDTANH
    }

    /** One recorded op in the fusion chain. / 融合链中的单条运算记录。 */
    static class FusedOp {
        final OpType type;
        final double param;
        final double param2;
        final Object other;

        FusedOp(OpType type, double param) {
            this(type, param, Double.NaN, null);
        }

        FusedOp(OpType type, double param, Object other) {
            this(type, param, Double.NaN, other);
        }

        FusedOp(OpType type, double param, double param2, Object other) {
            this.type = type;
            this.param = param;
            this.param2 = param2;
            this.other = other;
        }

        boolean isBinary() {
            return other != null;
        }
    }

    // ---- builder methods (unary / scalar param) ----

    public FusedOps exp()    { ops.add(new FusedOp(OpType.EXP, 0)); return this; }
    public FusedOps log()    { ops.add(new FusedOp(OpType.LOG, 0)); return this; }
    public FusedOps sqrt()   { ops.add(new FusedOp(OpType.SQRT, 0)); return this; }
    public FusedOps square() { ops.add(new FusedOp(OpType.SQUARE, 0)); return this; }
    public FusedOps sigmoid() { ops.add(new FusedOp(OpType.SIGMOID, 0)); return this; }
    public FusedOps tanh()   { ops.add(new FusedOp(OpType.TANH, 0)); return this; }
    public FusedOps relu()   { ops.add(new FusedOp(OpType.RELU, 0)); return this; }
    public FusedOps abs()    { ops.add(new FusedOp(OpType.ABS, 0)); return this; }
    public FusedOps neg()    { ops.add(new FusedOp(OpType.NEG, 0)); return this; }
    public FusedOps sin()    { ops.add(new FusedOp(OpType.SIN, 0)); return this; }
    public FusedOps cos()    { ops.add(new FusedOp(OpType.COS, 0)); return this; }
    public FusedOps tan()    { ops.add(new FusedOp(OpType.TAN, 0)); return this; }
    public FusedOps gelu()   { ops.add(new FusedOp(OpType.GELU, 0)); return this; }
    public FusedOps erf()    { ops.add(new FusedOp(OpType.ERF, 0)); return this; }
    public FusedOps reciprocal() { ops.add(new FusedOp(OpType.RECIPROCAL, 0)); return this; }
    public FusedOps pow(double n)  { ops.add(new FusedOp(OpType.POW, n)); return this; }
    public FusedOps add(double c)  { ops.add(new FusedOp(OpType.ADD_C, c)); return this; }
    public FusedOps sub(double c)  { ops.add(new FusedOp(OpType.SUB_C, c)); return this; }
    public FusedOps mul(double c)  { ops.add(new FusedOp(OpType.MUL_C, c)); return this; }
    public FusedOps div(double c)  { ops.add(new FusedOp(OpType.DIV_C, c)); return this; }
    public FusedOps rsub(double c) { ops.add(new FusedOp(OpType.RSUB_C, c)); return this; }
    public FusedOps rdiv(double c) { ops.add(new FusedOp(OpType.RDIV_C, c)); return this; }
    public FusedOps clamp(double min, double max) { ops.add(new FusedOp(OpType.CLAMP, min, max, null)); return this; }
    public FusedOps leakyRelu(double alpha) { ops.add(new FusedOp(OpType.LEAKY_RELU, alpha)); return this; }
    public FusedOps elu(double alpha) { ops.add(new FusedOp(OpType.ELU, alpha)); return this; }
    public FusedOps selu() { ops.add(new FusedOp(OpType.SELU, 0)); return this; }
    public FusedOps silu() { ops.add(new FusedOp(OpType.SILU, 0)); return this; }
    public FusedOps mish() { ops.add(new FusedOp(OpType.MISH, 0)); return this; }
    public FusedOps softplus(double beta) { ops.add(new FusedOp(OpType.SOFTPLUS, beta)); return this; }
    public FusedOps hardtanh(double minVal, double maxVal) { ops.add(new FusedOp(OpType.HARDTANH, minVal, maxVal, null)); return this; }

    // ---- builder methods (binary variable) ----

    public FusedOps add(IDiffVector other) { ops.add(new FusedOp(OpType.ADD_V, 0, (RereDiffVector) other)); return this; }
    public FusedOps sub(IDiffVector other) { ops.add(new FusedOp(OpType.SUB_V, 0, (RereDiffVector) other)); return this; }
    public FusedOps mul(IDiffVector other) { ops.add(new FusedOp(OpType.MUL_V, 0, (RereDiffVector) other)); return this; }
    public FusedOps div(IDiffVector other) { ops.add(new FusedOp(OpType.DIV_V, 0, (RereDiffVector) other)); return this; }
    public FusedOps maximum(IDiffVector other) { ops.add(new FusedOp(OpType.MAX_V, 0, (RereDiffVector) other)); return this; }
    public FusedOps minimum(IDiffVector other) { ops.add(new FusedOp(OpType.MIN_V, 0, (RereDiffVector) other)); return this; }

    // ---- SIMD species ----
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    // ---- compute ----

    public IDiffVector compute() {
        if (ops.isEmpty()) {
            return x;
        }

        int n = x.getValue().size();
        double[] xData = x.getValue().getData();

        // --- Fast path: single SIMD-friendly op → vectorized directly ---
        if (ops.size() == 1 && !ops.get(0).isBinary() && isSimdFriendlyUnary(ops.get(0).type)) {
            FusedOp singleOp = ops.get(0);
            double[] result = new double[n];
            double[] saved0 = AutodiffBufferPool.acquire(n);
            System.arraycopy(xData, 0, saved0, 0, n);

            // SIMD forward
            int vl = SPECIES.length();
            int i = 0;
            for (; i + vl <= n; i += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, xData, i);
                applySimdUnary(singleOp, v).intoArray(result, i);
            }
            for (; i < n; i++) {
                result[i] = applyForwardUnary(singleOp, xData[i]);
            }

            IDoubleVector resultVal = IDoubleVector.of(result);
            RereDiffVector self = this.x;
            Consumer<IDoubleVector> backwardFn = (gradOut) -> {
                double[] dx = AutodiffBufferPool.acquire(n);
                double[] gradData = gradOut.getData();
                for (int bi = 0; bi < n; bi++) {
                    dx[bi] = applyGradientUnary(singleOp, gradData[bi], saved0[bi], result[bi]);
                }
                self.accGradFromPooled(dx, n);
                AutodiffBufferPool.release(saved0);
            };
            List<RereDiffVector> inputs = new ArrayList<>();
            inputs.add(this.x);
            return RereDiffVector.createNonLeaf(result, inputs, backwardFn);
        }

        // --- Multi-op path: SIMD vectorized forward, scalar backward ---
        double[] result = new double[n];
        double[][] saved = new double[ops.size()][];
        for (int j = 0; j < ops.size(); j++) saved[j] = AutodiffBufferPool.acquire(n);

        int binaryCount = 0;
        for (FusedOp op : ops) { if (op.isBinary()) binaryCount++; }
        double[][] otherGrads = binaryCount > 0 ? new double[ops.size()][] : null;
        if (binaryCount > 0) {
            for (int j = 0; j < ops.size(); j++) {
                if (ops.get(j).isBinary()) otherGrads[j] = new double[n];
            }
        }

        // Prepare binary operand data arrays
        double[][] otherData = new double[ops.size()][];
        for (int j = 0; j < ops.size(); j++) {
            if (ops.get(j).isBinary()) {
                otherData[j] = ((RereDiffVector) ops.get(j).other).getValue().getData();
            }
        }

        // SIMD forward pass
        int vl = SPECIES.length();
        int i = 0;
        boolean allSimdFriendly = true;
        for (FusedOp op : ops) {
            if (op.isBinary()) {
                switch (op.type) {
                    case ADD_V, SUB_V, MUL_V, DIV_V, MAX_V, MIN_V -> {}
                    default -> { allSimdFriendly = false; break; }
                }
            } else {
                if (!isSimdFriendlyUnary(op.type)) { allSimdFriendly = false; break; }
            }
        }

        if (allSimdFriendly && n >= vl) {
            for (; i + vl <= n; i += vl) {
                DoubleVector v = DoubleVector.fromArray(SPECIES, xData, i);
                for (int j = 0; j < ops.size(); j++) {
                    v.intoArray(saved[j], i);
                    FusedOp op = ops.get(j);
                    if (op.isBinary()) {
                        DoubleVector ov = DoubleVector.fromArray(SPECIES, otherData[j], i);
                        v = applySimdBinary(op.type, v, ov);
                    } else {
                        v = applySimdUnary(op, v);
                    }
                }
                v.intoArray(result, i);
            }
        }

        // Scalar tail (or full scalar if not all SIMD-friendly)
        for (; i < n; i++) {
            double v = xData[i];
            for (int j = 0; j < ops.size(); j++) {
                FusedOp op = ops.get(j);
                saved[j][i] = v;
                if (op.isBinary()) {
                    v = applyForwardBinary(op, v, otherData[j][i]);
                } else {
                    v = applyForwardUnary(op, v);
                }
            }
            result[i] = v;
        }

        IDoubleVector resultVal = IDoubleVector.of(result);
        RereDiffVector self = this.x;

        // Capture for backward
        final double[][] fSaved = saved;
        final double[][] fOtherGrads = otherGrads;
        final double[] fResult = result;

        Consumer<IDoubleVector> backwardFn = (gradOut) -> {
            double[] dx = AutodiffBufferPool.acquire(n);
            double[] gradData = gradOut.getData();
            for (int bi = 0; bi < n; bi++) {
                double g = gradData[bi];
                for (int j = ops.size() - 1; j >= 0; j--) {
                    FusedOp op = ops.get(j);
                    double inputV = fSaved[j][bi];
                    double outputV = (j == ops.size() - 1) ? fResult[bi] : fSaved[j + 1][bi];
                    if (op.isBinary()) {
                        double otherV = otherData[j][bi];
                        fOtherGrads[j][bi] = applyGradientOther(op, g, inputV, otherV, outputV);
                        g = applyGradientSelf(op, g, inputV, otherV, outputV);
                    } else {
                        g = applyGradientUnary(op, g, inputV, outputV);
                    }
                }
                dx[bi] = g;
            }
            self.accGradFromPooled(dx, n);

            for (int j = 0; j < ops.size(); j++) {
                if (ops.get(j).isBinary()) {
                    double[] og = fOtherGrads[j];
                    ((RereDiffVector) ops.get(j).other).accGradDirect(og);
                }
            }

            // Release pooled buffers
            for (int j = 0; j < ops.size(); j++) AutodiffBufferPool.release(fSaved[j]);
        };

        List<RereDiffVector> inputs = new ArrayList<>();
        inputs.add(this.x);
        for (FusedOp op : ops) {
            if (op.isBinary()) {
                RereDiffVector ov = (RereDiffVector) op.other;
                if (!inputs.contains(ov)) {
                    inputs.add(ov);
                }
            }
        }
        return RereDiffVector.createNonLeaf(result, inputs, backwardFn);
    }

    static boolean isSimdFriendlyUnary(OpType type) {
        return switch (type) {
            case EXP, LOG, SQRT, SQUARE, SIGMOID, TANH, RELU, ABS, NEG,
                 SIN, COS, TAN, GELU, RECIPROCAL,
                 POW, ADD_C, SUB_C, MUL_C, DIV_C, RSUB_C, RDIV_C -> true;
            default -> false;
        };
    }

    // ---- forward kernels ----

    private static double applyForwardUnary(FusedOp op, double x) {
        return switch (op.type) {
            case EXP     -> Math.exp(x);
            case LOG     -> Math.log(x);
            case SQRT    -> Math.sqrt(x);
            case SQUARE  -> x * x;
            case SIGMOID -> 1.0 / (1.0 + Math.exp(-x));
            case TANH    -> Math.tanh(x);
            case RELU    -> Math.max(0.0, x);
            case ABS     -> Math.abs(x);
            case NEG     -> -x;
            case POW     -> Math.pow(x, op.param);
            case CLAMP   -> x < op.param ? op.param : (x > op.param2 ? op.param2 : x);
            case LEAKY_RELU -> x > 0 ? x : op.param * x;
            case ELU -> x >= 0 ? x : op.param * (Math.exp(x) - 1.0);
            case SELU -> { double sa = 1.0507009873554804, al = 1.6732632423543772; yield x >= 0 ? sa * x : sa * al * (Math.exp(x) - 1.0); }
            case SILU -> x / (1.0 + Math.exp(-x));
            case MISH -> { double sp = Math.log(1.0 + Math.exp(x)); yield x * Math.tanh(sp); }
            case SOFTPLUS -> { double bx = op.param * x; yield bx > 100 ? x : (1.0 / op.param) * Math.log(1.0 + Math.exp(bx)); }
            case HARDTANH -> x < op.param ? op.param : (x > op.param2 ? op.param2 : x);
            case SIN      -> Math.sin(x);
            case COS      -> Math.cos(x);
            case TAN      -> Math.tan(x);
            case GELU     -> 0.5 * x * (1.0 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            case ERF      -> erf(x);
            case RECIPROCAL -> 1.0 / x;
            case ADD_C   -> x + op.param;
            case SUB_C   -> x - op.param;
            case MUL_C   -> x * op.param;
            case DIV_C   -> x / op.param;
            case RSUB_C  -> op.param - x;
            case RDIV_C  -> op.param / x;
            default      -> x;
        };
    }

    private static double applyForwardBinary(FusedOp op, double x, double other) {
        return switch (op.type) {
            case ADD_V -> x + other;
            case SUB_V -> x - other;
            case MUL_V -> x * other;
            case DIV_V -> x / other;
            case MAX_V -> Math.max(x, other);
            case MIN_V -> Math.min(x, other);
            default    -> x;
        };
    }

    // ---- SIMD forward kernels ----

    static DoubleVector applySimdUnary(FusedOp op, DoubleVector a) {
        return switch (op.type) {
            case EXP -> a.lanewise(VectorOperators.EXP);
            case LOG -> a.lanewise(VectorOperators.LOG);
            case SQRT -> a.lanewise(VectorOperators.SQRT);
            case SQUARE -> a.mul(a);
            case SIN -> a.lanewise(VectorOperators.SIN);
            case COS -> a.lanewise(VectorOperators.COS);
            case TAN -> a.lanewise(VectorOperators.TAN);
            case TANH -> a.lanewise(VectorOperators.TANH);
            case ABS -> a.abs();
            case NEG -> a.neg();
            case RECIPROCAL -> DoubleVector.broadcast(SPECIES, 1.0).div(a);
            case POW -> a.lanewise(VectorOperators.POW, op.param);
            case ADD_C -> a.add(op.param);
            case SUB_C -> a.sub(op.param);
            case MUL_C -> a.mul(op.param);
            case DIV_C -> a.div(op.param);
            case RSUB_C -> DoubleVector.broadcast(SPECIES, op.param).sub(a);
            case RDIV_C -> DoubleVector.broadcast(SPECIES, op.param).div(a);
            case RELU -> a.max(0.0);
            case SIGMOID -> a.mul(0.5).lanewise(VectorOperators.TANH).mul(0.5).add(0.5);
            case GELU -> {
                DoubleVector xCubed = a.mul(a).mul(a);
                DoubleVector inner = a.add(xCubed.mul(0.044715)).mul(0.7978845608028654);
                DoubleVector tanhV = inner.lanewise(VectorOperators.TANH);
                yield a.mul(0.5).mul(tanhV.add(1.0));
            }
            default -> a; // fallback: pass through (should not reach here if allSimdFriendly)
        };
    }

    static DoubleVector applySimdBinary(OpType type, DoubleVector a, DoubleVector b) {
        return switch (type) {
            case ADD_V -> a.add(b);
            case SUB_V -> a.sub(b);
            case MUL_V -> a.mul(b);
            case DIV_V -> a.div(b);
            case MAX_V -> a.max(b);
            case MIN_V -> a.min(b);
            default -> a;
        };
    }

    // ---- backward kernels (unary) ----

    private static double applyGradientUnary(FusedOp op, double grad, double inputV, double outputV) {
        return switch (op.type) {
            case EXP     -> grad * outputV;
            case LOG     -> grad / inputV;
            case SQRT    -> grad / (2.0 * outputV);
            case SQUARE  -> grad * 2.0 * inputV;
            case SIGMOID -> grad * outputV * (1.0 - outputV);
            case TANH    -> grad * (1.0 - outputV * outputV);
            case RELU    -> grad * (inputV > 0.0 ? 1.0 : 0.0);
            case ABS     -> grad * (inputV > 0.0 ? 1.0 : (inputV < 0.0 ? -1.0 : 0.0));
            case NEG     -> -grad;
            case POW     -> grad * op.param * Math.pow(inputV, op.param - 1.0);
            case LEAKY_RELU -> grad * (inputV > 0 ? 1.0 : op.param);
            case ELU -> grad * (inputV >= 0 ? 1.0 : op.param * Math.exp(inputV));
            case SELU -> { double sa = 1.0507009873554804, al = 1.6732632423543772; yield grad * (inputV >= 0 ? sa : sa * al * Math.exp(inputV)); }
            case SILU -> { double s = 1.0 / (1.0 + Math.exp(-inputV)); yield grad * (s + inputV * s * (1.0 - s)); }
            case MISH -> { double sp = Math.log(1.0 + Math.exp(inputV)); double th = Math.tanh(sp); double sig = 1.0 / (1.0 + Math.exp(-inputV)); yield grad * (th + inputV * (1.0 - th*th) * sig); }
            case SOFTPLUS -> { double bx = op.param * inputV; double sig = bx > 100 ? 1.0 : 1.0 / (1.0 + Math.exp(-bx)); yield grad * sig; }
            case HARDTANH -> grad * (inputV > op.param && inputV < op.param2 ? 1.0 : 0.0);
            case CLAMP   -> grad * (inputV >= op.param && inputV <= op.param2 ? 1.0 : 0.0);
            case SIN      -> grad * Math.cos(inputV);
            case COS      -> -grad * Math.sin(inputV);
            case TAN      -> { double s = 1.0 / Math.cos(inputV); yield grad * s * s; }
            case GELU     -> { double cdf = 0.5 * (1.0 + Math.tanh(0.7978845608028654 * (inputV + 0.044715 * inputV * inputV * inputV))); double pdf = 0.3989422804014327 * Math.exp(-0.5 * inputV * inputV); yield grad * (cdf + inputV * pdf); }
            case ERF      -> grad * 1.1283791670955126 * Math.exp(-inputV * inputV);
            case RECIPROCAL -> -grad / (inputV * inputV);
            case ADD_C   -> grad;
            case SUB_C   -> grad;
            case MUL_C   -> grad * op.param;
            case DIV_C   -> grad / op.param;
            case RSUB_C  -> -grad;
            case RDIV_C  -> -grad * op.param / (inputV * inputV);
            default      -> grad;
        };
    }

    // ---- backward kernels (binary) ----

    private static double applyGradientSelf(FusedOp op, double grad, double inputV, double otherV, double outputV) {
        return switch (op.type) {
            case ADD_V -> grad;
            case SUB_V -> grad;
            case MUL_V -> grad * otherV;
            case DIV_V -> grad / otherV;
            case MAX_V -> grad * (inputV >= otherV ? 1.0 : 0.0);
            case MIN_V -> grad * (inputV <= otherV ? 1.0 : 0.0);
            default    -> grad;
        };
    }

    private static double applyGradientOther(FusedOp op, double grad, double inputV, double otherV, double outputV) {
        return switch (op.type) {
            case ADD_V -> grad;
            case SUB_V -> -grad;
            case MUL_V -> grad * inputV;
            case DIV_V -> -grad * inputV / (otherV * otherV);
            case MAX_V -> grad * (otherV >= inputV ? 1.0 : 0.0);
            case MIN_V -> grad * (otherV <= inputV ? 1.0 : 0.0);
            default    -> 0;
        };
    }

    // ---- erf helper (Abramowitz & Stegun approximation, max error ~1.5e-7) ----

    private static double erf(double x) {
        double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        double a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        double sign = x >= 0 ? 1.0 : -1.0;
        double ax = Math.abs(x);
        double t = 1.0 / (1.0 + p * ax);
        double y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
        return sign * y;
    }
}
