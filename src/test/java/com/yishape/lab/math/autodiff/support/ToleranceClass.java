package com.yishape.lab.math.autodiff.support;

/**
 * Standardized tolerance classes for cross-backend numerical comparison.
 *
 * <p>Each class defines two tolerance levels:
 * <ul>
 *   <li>{@code absTol} — absolute tolerance (used when both values are near zero)</li>
 *   <li>{@code relTol} — relative tolerance (used when values are large)</li>
 * </ul>
 *
 * <p>The effective tolerance for comparing two values (a, b) is:
 * <pre>max(absTol, max(|a|,|b|) * relTol)</pre>
 *
 * <p>F32 variants apply when one backend uses float precision (GPU).
 *
 * @since 0.9.0
 */
public enum ToleranceClass {

    // ── f64 precision (CPU, HPC) ──────────────────────────────────────

    /**
     * Algebraically exact operations: add, sub, mul, neg.
     * Rounding error only in the last bit.
     */
    EXACT(1e-14, 1e-7),

    /**
     * Operations involving division: div, sigmoid, tanh, gelu.
     */
    DIVISION(1e-13, 1e-7),

    /**
     * Reductions: sum, mean, matmul.
     * Error accumulates with element count.
     */
    REDUCTION(1e-12, 1e-5),

    /**
     * Complex chained operations: conv2d, attention, softmax, layerNorm.
     */
    CHAIN(1e-10, 1e-4),

    // ── f32 precision (GPU vs CPU/HPC) ────────────────────────────────

    /** GPU vs CPU for exact operations. */
    EXACT_F32(1e-6, 1e-4),

    /** GPU vs CPU for division-based ops. */
    DIVISION_F32(1e-5, 1e-3),

    /** GPU vs CPU for reduction ops. */
    REDUCTION_F32(1e-4, 1e-2),

    /** GPU vs CPU for complex chained ops. */
    CHAIN_F32(1e-3, 1e-1);

    /** Absolute tolerance. */
    public final double absTol;

    /** Relative tolerance. */
    public final double relTol;

    ToleranceClass(double absTol, double relTol) {
        this.absTol = absTol;
        this.relTol = relTol;
    }

    /**
     * Returns true if |a - b| is within tolerance.
     *
     * @param a first value
     * @param b second value
     * @return true if within tolerance
     */
    public boolean within(double a, double b) {
        double maxAbs = Math.max(Math.abs(a), Math.abs(b));
        double effective = Math.max(absTol, maxAbs * relTol);
        return Math.abs(a - b) <= effective;
    }

    /**
     * Returns the effective tolerance for comparing two values.
     *
     * @param a first value
     * @param b second value
     * @return effective tolerance = max(absTol, max(|a|,|b|) * relTol)
     */
    public double effective(double a, double b) {
        double maxAbs = Math.max(Math.abs(a), Math.abs(b));
        return Math.max(absTol, maxAbs * relTol);
    }

    /**
     * Select the appropriate tolerance class for a given operation tag and backend pair.
     *
     * @param opTag    operation tag (e.g. "add", "mul", "mmul", "softmax")
     * @param hasF32   true if one backend uses float precision (GPU involved)
     * @return the matching tolerance class
     */
    public static ToleranceClass forOp(String opTag, boolean hasF32) {
        return select(hasF32, classifyOp(opTag));
    }

    private static ToleranceClass classifyOp(String opTag) {
        if (opTag == null) return CHAIN;

        return switch (opTag) {
            case "add", "sub", "neg", "abs", "square",
                 "mul", "rsub", "rdiv" -> EXACT;
            case "div", "sigmoid", "tanh", "tan",
                 "sin", "cos", "gelu", "mish",
                 "silu", "elu", "selu", "leakyRelu",
                 "softplus", "hardtanh", "clamp",
                 "sqrt", "reciprocal" -> DIVISION;
            case "sum", "mean", "max", "min",
                 "mmul", "bmm", "dot",
                 "var", "std", "prod" -> REDUCTION;
            case "softmax", "logSoftmax",
                 "layerNorm", "batchNorm", "rmsNorm",
                 "groupNorm", "softmaxCrossEntropy",
                 "conv2d", "scaledDotProductAttention",
                 "logSumExp", "dropout" -> CHAIN;
            default -> CHAIN;
        };
    }

    private static ToleranceClass select(boolean hasF32, ToleranceClass base) {
        if (!hasF32) return base;
        return switch (base) {
            case EXACT -> EXACT_F32;
            case DIVISION -> DIVISION_F32;
            case REDUCTION -> REDUCTION_F32;
            case CHAIN -> CHAIN_F32;
            default -> CHAIN_F32;
        };
    }
}
