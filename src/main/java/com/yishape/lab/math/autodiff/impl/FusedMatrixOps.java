package com.yishape.lab.math.autodiff.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.autodiff.IDiffMatrix;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Fused element-wise operator chain on a matrix (same idea as {@link FusedOps}).
 * 矩阵版融合逐元素算子链，语义同 {@link FusedOps}。
 * Includes SIMD acceleration for forward pass on SIMD-friendly op chains.
 */
public class FusedMatrixOps {

    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private final RereDiffMatrix x;
    private final List<FusedOps.FusedOp> ops = new ArrayList<>();

    public FusedMatrixOps(IDiffMatrix x) {
        this.x = (RereDiffMatrix) x;
    }

    // ---- builder methods (unary / scalar param) ----

    public FusedMatrixOps exp()    { ops.add(new FusedOps.FusedOp(FusedOps.OpType.EXP, 0)); return this; }
    public FusedMatrixOps log()    { ops.add(new FusedOps.FusedOp(FusedOps.OpType.LOG, 0)); return this; }
    public FusedMatrixOps sqrt()   { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SQRT, 0)); return this; }
    public FusedMatrixOps square() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SQUARE, 0)); return this; }
    public FusedMatrixOps sigmoid() { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SIGMOID, 0)); return this; }
    public FusedMatrixOps tanh()   { ops.add(new FusedOps.FusedOp(FusedOps.OpType.TANH, 0)); return this; }
    public FusedMatrixOps relu()   { ops.add(new FusedOps.FusedOp(FusedOps.OpType.RELU, 0)); return this; }
    public FusedMatrixOps abs()    { ops.add(new FusedOps.FusedOp(FusedOps.OpType.ABS, 0)); return this; }
    public FusedMatrixOps neg()    { ops.add(new FusedOps.FusedOp(FusedOps.OpType.NEG, 0)); return this; }
    public FusedMatrixOps pow(double n)  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.POW, n)); return this; }
    public FusedMatrixOps add(double c)  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.ADD_C, c)); return this; }
    public FusedMatrixOps sub(double c)  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.SUB_C, c)); return this; }
    public FusedMatrixOps mul(double c)  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.MUL_C, c)); return this; }
    public FusedMatrixOps div(double c)  { ops.add(new FusedOps.FusedOp(FusedOps.OpType.DIV_C, c)); return this; }

    // ---- builder methods (binary variable) ----

    public FusedMatrixOps add(IDiffMatrix other) {
        ops.add(new FusedOps.FusedOp(FusedOps.OpType.ADD_V, 0, (RereDiffMatrix) other)); return this;
    }
    public FusedMatrixOps sub(IDiffMatrix other) {
        ops.add(new FusedOps.FusedOp(FusedOps.OpType.SUB_V, 0, (RereDiffMatrix) other)); return this;
    }
    public FusedMatrixOps mul(IDiffMatrix other) {
        ops.add(new FusedOps.FusedOp(FusedOps.OpType.MUL_V, 0, (RereDiffMatrix) other)); return this;
    }
    public FusedMatrixOps div(IDiffMatrix other) {
        ops.add(new FusedOps.FusedOp(FusedOps.OpType.DIV_V, 0, (RereDiffMatrix) other)); return this;
    }

    // ---- compute ----

    public IDiffMatrix compute() {
        if (ops.isEmpty()) {
            return x;
        }

        int rows = x.value.rows();
        int cols = x.value.cols();
        int n = rows * cols;
        double[][] xData = x.value.getData();
        double[][] result = new double[rows][cols];

        // Check if all ops are SIMD-friendly for forward pass
        boolean allSimdFriendly = true;
        for (FusedOps.FusedOp op : ops) {
            if (op.isBinary()) {
                FusedOps.OpType t = op.type;
                if (t != FusedOps.OpType.ADD_V && t != FusedOps.OpType.SUB_V &&
                    t != FusedOps.OpType.MUL_V && t != FusedOps.OpType.DIV_V &&
                    t != FusedOps.OpType.MAX_V && t != FusedOps.OpType.MIN_V) {
                    allSimdFriendly = false;
                    break;
                }
            } else if (!FusedOps.isSimdFriendlyUnary(op.type)) {
                allSimdFriendly = false;
                break;
            }
        }

        // Allocate saved arrays (flat per op, pooled)
        int opCount = ops.size();
        double[][] saved = new double[opCount][];
        for (int k = 0; k < opCount; k++) {
            saved[k] = AutodiffBufferPool.acquire(n);
        }

        int binaryCount = 0;
        for (FusedOps.FusedOp op : ops) {
            if (op.isBinary()) binaryCount++;
        }
        double[][][] otherGrads = binaryCount > 0 ? new double[opCount][rows][cols] : null;

        if (allSimdFriendly) {
            // SIMD forward pass — row by row, SIMD within each row
            int vl = SPECIES.length();
            for (int i = 0; i < rows; i++) {
                double[] row = xData[i];
                int rowBase = i * cols;
                int j = 0;
                for (; j + vl <= cols; j += vl) {
                    DoubleVector v = DoubleVector.fromArray(SPECIES, row, j);
                    for (int k = 0; k < opCount; k++) {
                        FusedOps.FusedOp op = ops.get(k);
                        v.intoArray(saved[k], rowBase + j);
                        if (op.isBinary()) {
                            double[] otherRow = ((RereDiffMatrix) op.other).value.getData()[i];
                            DoubleVector ov = DoubleVector.fromArray(SPECIES, otherRow, j);
                            v = FusedOps.applySimdBinary(op.type, v, ov);
                        } else {
                            v = FusedOps.applySimdUnary(op, v);
                        }
                    }
                    v.intoArray(result[i], j);
                }
                // Scalar tail
                for (; j < cols; j++) {
                    double v = row[j];
                    int idx = rowBase + j;
                    for (int k = 0; k < opCount; k++) {
                        FusedOps.FusedOp op = ops.get(k);
                        saved[k][idx] = v;
                        if (op.isBinary()) {
                            double otherV = ((RereDiffMatrix) op.other).value.getData()[i][j];
                            v = applyForwardBinary(op, v, otherV);
                        } else {
                            v = applyForwardUnary(op, v);
                        }
                    }
                    result[i][j] = v;
                }
            }
        } else {
            // Scalar fallback
            for (int i = 0; i < rows; i++) {
                int rowBase = i * cols;
                for (int j = 0; j < cols; j++) {
                    double v = xData[i][j];
                    int idx = rowBase + j;
                    for (int k = 0; k < opCount; k++) {
                        FusedOps.FusedOp op = ops.get(k);
                        saved[k][idx] = v;
                        if (op.isBinary()) {
                            double otherV = ((RereDiffMatrix) op.other).value.getData()[i][j];
                            v = applyForwardBinary(op, v, otherV);
                        } else {
                            v = applyForwardUnary(op, v);
                        }
                    }
                    result[i][j] = v;
                }
            }
        }

        IDoubleMatrix resultVal = IDoubleMatrix.of(result);
        RereDiffMatrix self = this.x;

        Consumer<IDoubleMatrix> backwardFn = (gradOut) -> {
            double[][] gradData = gradOut.getData();
            double[][] dx = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                int rowBase = i * cols;
                for (int j = 0; j < cols; j++) {
                    double g = gradData[i][j];
                    int idx = rowBase + j;
                    for (int k = opCount - 1; k >= 0; k--) {
                        FusedOps.FusedOp op = ops.get(k);
                        double inputV = saved[k][idx];
                        double outputV = (k == opCount - 1) ? result[i][j] : saved[k + 1][idx];
                        if (op.isBinary()) {
                            double otherV = ((RereDiffMatrix) op.other).value.getData()[i][j];
                            otherGrads[k][i][j] = applyGradientOther(op, g, inputV, otherV, outputV);
                            g = applyGradientSelf(op, g, inputV, otherV, outputV);
                        } else {
                            g = applyGradientUnary(op, g, inputV, outputV);
                        }
                    }
                    dx[i][j] = g;
                }
            }
            // Release saved buffers
            for (int k = 0; k < opCount; k++) {
                AutodiffBufferPool.release(saved[k]);
            }
            self.accGradDirect(dx);

            for (int k = 0; k < opCount; k++) {
                if (ops.get(k).isBinary()) {
                    ((RereDiffMatrix) ops.get(k).other).accGradDirect(otherGrads[k]);
                }
            }
        };

        List<RereDiffMatrix> inputs = new ArrayList<>();
        inputs.add(this.x);
        for (FusedOps.FusedOp op : ops) {
            if (op.isBinary()) {
                RereDiffMatrix otherMat = (RereDiffMatrix) op.other;
                if (!inputs.contains(otherMat)) {
                    inputs.add(otherMat);
                }
            }
        }
        return new RereDiffMatrix(resultVal, inputs, backwardFn);
    }

    // ---- forward/backward kernels (scalar) ----

    private static double applyForwardUnary(FusedOps.FusedOp op, double x) {
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
            case ADD_C   -> x + op.param;
            case SUB_C   -> x - op.param;
            case MUL_C   -> x * op.param;
            case DIV_C   -> x / op.param;
            case RSUB_C  -> op.param - x;
            case RDIV_C  -> op.param / x;
            case GELU    -> x * 0.5 * (1.0 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
            case LEAKY_RELU -> x >= 0 ? x : op.param * x;
            case ELU     -> x >= 0 ? x : op.param * (Math.exp(x) - 1.0);
            case SELU    -> x >= 0 ? 1.0507009873554805 * x : 1.0507009873554805 * 1.6732632423543772 * (Math.exp(x) - 1.0);
            case SILU    -> x / (1.0 + Math.exp(-x));
            case MISH    -> x * Math.tanh(Math.log(1.0 + Math.exp(x)));
            case SOFTPLUS -> Math.log(1.0 + Math.exp(op.param * x)) / op.param;
            case HARDTANH -> Math.max(op.param, Math.min(op.param2, x));
            case CLAMP   -> Math.max(op.param, Math.min(op.param2, x));
            case SIN     -> Math.sin(x);
            case COS     -> Math.cos(x);
            case TAN     -> Math.tan(x);
            case RECIPROCAL -> 1.0 / x;
            default -> throw new UnsupportedOperationException("Unsupported unary op in FusedMatrixOps: " + op.type);
        };
    }

    private static double applyForwardBinary(FusedOps.FusedOp op, double x, double other) {
        return switch (op.type) {
            case ADD_V -> x + other;
            case SUB_V -> x - other;
            case MUL_V -> x * other;
            case DIV_V -> x / other;
            case MAX_V -> Math.max(x, other);
            case MIN_V -> Math.min(x, other);
            default -> throw new UnsupportedOperationException("Unsupported binary op in FusedMatrixOps: " + op.type);
        };
    }

    private static double applyGradientUnary(FusedOps.FusedOp op, double grad, double inputV, double outputV) {
        return switch (op.type) {
            case EXP     -> grad * outputV;
            case LOG     -> grad / inputV;
            case SQRT    -> grad / (2.0 * outputV);
            case SQUARE  -> grad * 2.0 * inputV;
            case SIGMOID -> grad * outputV * (1.0 - outputV);
            case TANH    -> grad * (1.0 - outputV * outputV);
            case RELU    -> grad * (inputV > 0.0 ? 1.0 : 0.0);
            case ABS     -> grad * (inputV >= 0.0 ? 1.0 : -1.0);
            case NEG     -> -grad;
            case POW     -> grad * op.param * Math.pow(inputV, op.param - 1.0);
            case ADD_C   -> grad;
            case SUB_C   -> grad;
            case MUL_C   -> grad * op.param;
            case DIV_C   -> grad / op.param;
            case RSUB_C  -> -grad;
            case RDIV_C  -> -grad * op.param / (inputV * inputV);
            case GELU -> {
                double sqrt2OverPi = 0.7978845608028654;
                double g = 0.044715;
                double inner = sqrt2OverPi * (inputV + g * inputV * inputV * inputV);
                double tanhI = Math.tanh(inner);
                double sechSq = 1.0 - tanhI * tanhI;
                double din_dx = sqrt2OverPi * (1.0 + 3.0 * g * inputV * inputV);
                yield grad * (0.5 * (1.0 + tanhI) + 0.5 * inputV * sechSq * din_dx);
            }
            case LEAKY_RELU -> grad * (inputV >= 0 ? 1.0 : op.param);
            case ELU     -> grad * (inputV >= 0 ? 1.0 : op.param * Math.exp(inputV));
            case SELU    -> grad * (inputV >= 0 ? 1.0507009873554805 : 1.0507009873554805 * 1.6732632423543772 * Math.exp(inputV));
            case SILU -> {
                double sig = 1.0 / (1.0 + Math.exp(-inputV));
                yield grad * (sig + inputV * sig * (1.0 - sig));
            }
            case MISH -> {
                double sp = Math.log(1.0 + Math.exp(inputV));
                double tanhSP = Math.tanh(sp);
                double sig = 1.0 / (1.0 + Math.exp(-inputV));
                yield grad * (tanhSP + inputV * sig * (1.0 - tanhSP * tanhSP));
            }
            case SOFTPLUS -> {
                double sig = 1.0 / (1.0 + Math.exp(-op.param * inputV));
                yield grad * sig;
            }
            case HARDTANH -> grad * (inputV >= op.param && inputV <= op.param2 ? 1.0 : 0.0);
            case CLAMP   -> grad * (inputV >= op.param && inputV <= op.param2 ? 1.0 : 0.0);
            case SIN     -> grad * Math.cos(inputV);
            case COS     -> -grad * Math.sin(inputV);
            case TAN     -> grad * (1.0 + outputV * outputV);
            case RECIPROCAL -> -grad / (inputV * inputV);
            default      -> grad;
        };
    }

    private static double applyGradientSelf(FusedOps.FusedOp op, double grad, double inputV, double otherV, double outputV) {
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

    private static double applyGradientOther(FusedOps.FusedOp op, double grad, double inputV, double otherV, double outputV) {
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
}
