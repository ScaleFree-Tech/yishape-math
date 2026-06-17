package com.yishape.lab.math.codegen;

import java.util.*;

/**
 * SINGLE SOURCE OF TRUTH for all compute operation definitions.
 *
 * <p>Every op that can appear in the autodiff computation graph is defined here.
 * The {@link CodegenTool} reads this registry and generates:
 * <ul>
 *   <li>{@code GraphOpSchema.Gpu.BASE} and {@code GraphOpSchema.Hpc.BASE} — backend-supported op sets</li>
 *   <li>{@code GraphOpSchema.FusedTag} — fusion pattern components</li>
 *   <li>{@code GraphOpSchema.describeTag()} — human-readable op descriptions</li>
 *   <li>{@code GraphOpSchema.FusedTagRegistry} — backend-specific fused subsets</li>
 *   <li>Rust dispatch match arms (future: {@code graph_ops_dispatch.rs})</li>
 *   <li>Documentation ({@code op_schema.md})</li>
 * </ul>
 *
 * <p><b>Modification rules:</b>
 * <ol>
 *   <li>Adding a new op → add it here in the ALL_OPS list</li>
 *   <li>Enabling GPU/HPC for an op → flip its {@code gpu}/{@code hpc} flags</li>
 *   <li>Adding fused variants → update the op's {@code fusedTags} set</li>
 *   <li>Run {@code CodegenTool.main()} to regenerate all derived files</li>
 * </ol>
 */
public final class OpRegistry {

    private OpRegistry() {}

    // ========================================================================
    // Fused tag patterns — single authority for {unary}{Reduce} naming
    // ========================================================================

    /** Canonical fused tag builder: fuseTag("square", "mean") → "squareMean". */
    public static String fuseTag(String unaryTag, String reduceTag) {
        return unaryTag + Character.toUpperCase(reduceTag.charAt(0)) + reduceTag.substring(1);
    }

    /** Builds all {unary}{reduce} fused tags for a given unary. */
    public static Set<String> buildFusedTags(String unaryTag) {
        return Set.of(fuseTag(unaryTag, "sum"), fuseTag(unaryTag, "mean"));
    }

    // ========================================================================
    // ALL OPERATIONS — single source of truth
    // ========================================================================

    /** Complete registry of all compute operations, in canonical order. */
    public static final List<OpDefinition> ALL_OPS = List.of(
        // ── Binary element-wise ──
        new OpDefinition("add",     OpCategory.BINARY,  2, true,  true,  "Element-wise Addition",     Set.of(), null),
        new OpDefinition("sub",     OpCategory.BINARY,  2, true,  true,  "Element-wise Subtraction",  Set.of(), null),
        new OpDefinition("mul",     OpCategory.BINARY,  2, true,  true,  "Element-wise Multiplication", buildFusedTags("mul"), null),
        new OpDefinition("div",     OpCategory.BINARY,  2, true,  true,  "Element-wise Division",     Set.of(), null),

        // ── Binary scalar ──
        new OpDefinition("addScalar",  OpCategory.BINARY_SCALAR, 1, true, true, "Add Scalar",   Set.of(), null),
        new OpDefinition("subScalar",  OpCategory.BINARY_SCALAR, 1, true, true, "Subtract Scalar", Set.of(), null),
        new OpDefinition("mulScalar",  OpCategory.BINARY_SCALAR, 1, true, true, "Multiply by Scalar", Set.of(), null),
        new OpDefinition("divScalar",  OpCategory.BINARY_SCALAR, 1, true, true, "Divide by Scalar", Set.of(), null),
        new OpDefinition("rsubScalar", OpCategory.BINARY_SCALAR, 1, true, true, "Reverse Subtract Scalar", Set.of(), null),
        new OpDefinition("rdivScalar", OpCategory.BINARY_SCALAR, 1, true, true, "Reverse Divide Scalar", Set.of(), null),

        // ── Unary ──
        new OpDefinition("neg",  OpCategory.UNARY, 1, true, true, "Negate", Set.of(), null),
        new OpDefinition("abs",  OpCategory.UNARY, 1, true, true, "Absolute Value", buildFusedTags("abs"), null),
        new OpDefinition("sqrt", OpCategory.UNARY, 1, true, true, "Square Root", Set.of(), null),
        new OpDefinition("square", OpCategory.UNARY, 1, true, true, "Square", buildFusedTags("square"), null),
        new OpDefinition("pow",  OpCategory.UNARY, 1, true, true, "Power", buildFusedTags("pow"), "exponent"),
        new OpDefinition("reciprocal", OpCategory.UNARY, 1, false, true, "Reciprocal (1/x)", Set.of(), null),

        // ── Activation ──
        new OpDefinition("exp",    OpCategory.ACTIVATION, 1, true, true, "Exponential", buildFusedTags("exp"), null),
        new OpDefinition("log",    OpCategory.ACTIVATION, 1, true, true, "Natural Logarithm", buildFusedTags("log"), null),
        new OpDefinition("relu",   OpCategory.ACTIVATION, 1, true, true, "ReLU Activation", buildFusedTags("relu"), null),
        new OpDefinition("sigmoid", OpCategory.ACTIVATION, 1, true, true, "Sigmoid Activation", buildFusedTags("sigmoid"), null),
        new OpDefinition("tanh",   OpCategory.ACTIVATION, 1, true, true, "Tanh Activation", buildFusedTags("tanh"), null),
        new OpDefinition("gelu",   OpCategory.ACTIVATION, 1, true, true, "GELU Activation", buildFusedTags("gelu"), null),
        new OpDefinition("silu",   OpCategory.ACTIVATION, 1, true, true, "SiLU/Swish Activation", buildFusedTags("silu"), null),
        new OpDefinition("mish",   OpCategory.ACTIVATION, 1, true, true, "Mish Activation", buildFusedTags("mish"), null),
        new OpDefinition("leakyRelu", OpCategory.ACTIVATION, 1, true, true, "Leaky ReLU Activation", buildFusedTags("leakyRelu"), "alpha"),
        new OpDefinition("elu",    OpCategory.ACTIVATION, 1, true, true, "ELU Activation", buildFusedTags("elu"), "alpha"),
        new OpDefinition("selu",   OpCategory.ACTIVATION, 1, true, true, "SELU Activation", buildFusedTags("selu"), null),
        new OpDefinition("softplus", OpCategory.ACTIVATION, 1, true, true, "Softplus Activation", buildFusedTags("softplus"), "beta"),
        new OpDefinition("hardtanh", OpCategory.ACTIVATION, 1, true, true, "HardTanh Activation", buildFusedTags("hardtanh"), null),
        new OpDefinition("clamp",  OpCategory.ACTIVATION, 1, true, true, "Clamp", Set.of(), null),
        new OpDefinition("sin",    OpCategory.ACTIVATION, 1, true, true, "Sine", buildFusedTags("sin"), null),
        new OpDefinition("cos",    OpCategory.ACTIVATION, 1, true, true, "Cosine", buildFusedTags("cos"), null),
        new OpDefinition("tan",    OpCategory.ACTIVATION, 1, true, true, "Tangent", Set.of(), null),
        new OpDefinition("normalize", OpCategory.ACTIVATION, 1, true, true, "L2 Normalize", Set.of(), null),

        // ── Reduce ──
        new OpDefinition("sum",  OpCategory.REDUCE, 1, true, true, "Sum Reduction", Set.of(), null),
        new OpDefinition("mean", OpCategory.REDUCE, 1, true, true, "Mean Reduction", Set.of(), null),

        // ── Linalg ──
        new OpDefinition("matmul", OpCategory.LINALG, 2, true, true, "Matrix Multiplication", Set.of(), null),
        new OpDefinition("mmul",   OpCategory.LINALG, 2, true, true, "Matrix Multiply (alias)", Set.of(), null),
        new OpDefinition("dot",    OpCategory.LINALG, 2, true, true, "Dot Product", Set.of(), null),
        new OpDefinition("bmm",    OpCategory.LINALG, 2, false, true, "Batch Matrix Multiply", Set.of(), null),
        new OpDefinition("cross",  OpCategory.LINALG, 2, true, true, "Cross Product", Set.of(), null),

        // ── View / data movement ──
        new OpDefinition("broadcast", OpCategory.VIEW, 1, true, true, "Broadcast", Set.of(), null),
        new OpDefinition("expand",    OpCategory.VIEW, 1, true, true, "Expand", Set.of(), null),
        new OpDefinition("transpose", OpCategory.VIEW, 1, true, true, "Transpose", Set.of(), null),
        new OpDefinition("reshape",   OpCategory.VIEW, 1, true, true, "Reshape", Set.of(), null),
        new OpDefinition("flatten",   OpCategory.VIEW, 1, true, true, "Flatten", Set.of(), null),
        new OpDefinition("squeeze",   OpCategory.VIEW, 1, true, true, "Squeeze", Set.of(), null),
        new OpDefinition("unsqueeze", OpCategory.VIEW, 1, true, true, "Unsqueeze", Set.of(), null),
        new OpDefinition("permute",   OpCategory.VIEW, 1, true, true, "Permute", Set.of(), null),
        new OpDefinition("cat",       OpCategory.VIEW, 2, true, true, "Concatenate", Set.of(), null),
        new OpDefinition("gather",    OpCategory.VIEW, 2, true, true, "Gather", Set.of(), null),
        new OpDefinition("scatter",   OpCategory.VIEW, 2, false, true, "Scatter", Set.of(), null),
        new OpDefinition("select",    OpCategory.VIEW, 2, true, true, "Select", Set.of(), null),
        new OpDefinition("slice",     OpCategory.VIEW, 1, true, true, "Slice", Set.of(), null),
        new OpDefinition("contiguous", OpCategory.VIEW, 1, true, true, "Contiguous", Set.of(), null),
        new OpDefinition("interpolate", OpCategory.VIEW, 1, true, true, "Interpolate", Set.of(), null),
        new OpDefinition("gridSample",  OpCategory.VIEW, 1, true, true, "Grid Sample", Set.of(), null),

        // ── Random ──
        new OpDefinition("dropout", OpCategory.RANDOM, 1, true, true, "Dropout", Set.of(), null),

        // ── Normalization ──
        new OpDefinition("layerNorm",  OpCategory.NORMALIZATION, 3, true, true, "Layer Normalization", Set.of(), "epsilon"),
        new OpDefinition("batchNorm",  OpCategory.NORMALIZATION, 3, false, true, "Batch Normalization 1D", Set.of(), null),
        new OpDefinition("batchNorm2d", OpCategory.NORMALIZATION, 3, true, true, "Batch Normalization 2D", Set.of(), null),
        new OpDefinition("rmsNorm",    OpCategory.NORMALIZATION, 2, true, true, "RMS Normalization", Set.of(), "epsilon"),
        new OpDefinition("groupNorm",  OpCategory.NORMALIZATION, 3, true, true, "Group Normalization", Set.of(), null),
        new OpDefinition("instanceNorm", OpCategory.NORMALIZATION, 3, true, true, "Instance Normalization", Set.of(), null),

        // ── Deep Learning ──
        new OpDefinition("conv2d",          OpCategory.DL, 3, true, true, "2D Convolution", Set.of(), null),
        new OpDefinition("convTranspose2d", OpCategory.DL, 3, true, true, "2D Transposed Convolution", Set.of(), null),
        new OpDefinition("maxpool2d",       OpCategory.DL, 1, true, true, "Max Pooling 2D", Set.of(), null),
        new OpDefinition("avgpool2d",       OpCategory.DL, 1, true, true, "Average Pooling 2D", Set.of(), null),
        new OpDefinition("adaptiveAvgPool2d", OpCategory.DL, 1, true, true, "Adaptive Average Pooling 2D", Set.of(), null),
        new OpDefinition("depthwiseConv1d", OpCategory.DL, 3, true, true, "Depthwise 1D Convolution", Set.of(), null),

        // ── Attention ──
        new OpDefinition("mha", OpCategory.ATTENTION, 3, true, true, "Multi-Head Attention", Set.of(), null),
        new OpDefinition("scaledDotProductAttention", OpCategory.ATTENTION, 4, true, true, "Scaled Dot-Product Attention", Set.of(), null),

        // ── SSM ──
        new OpDefinition("selectiveScan",  OpCategory.SSM, 6, true, true, "Selective Scan (Mamba SSM)", Set.of(), null),
        new OpDefinition("selectiveScan2", OpCategory.SSM, 6, true, true, "Selective Scan 2 (Chunked)", Set.of(), null),
        new OpDefinition("trapezoidalScan", OpCategory.SSM, 6, true, true, "Trapezoidal Scan (Mamba SSM)", Set.of(), null),

        // ── Embedding ──
        new OpDefinition("embedding", OpCategory.EMBEDDING, 2, true, true, "Embedding Lookup", Set.of(), null),

        // ── RNN ──
        new OpDefinition("lstmStep", OpCategory.DL, 5, true, true, "LSTM Timestep", Set.of(), null),

        // ── Softmax family ──
        new OpDefinition("softmax",    OpCategory.ACTIVATION, 1, true, true, "Softmax", Set.of(), null),
        new OpDefinition("logSoftmax", OpCategory.ACTIVATION, 1, true, true, "Log Softmax", Set.of(), null),

        // ── Fused loss / compound ──
        new OpDefinition("logSumExp", OpCategory.REDUCE, 1, true, true, "Log-Sum-Exp", Set.of(), null),
        new OpDefinition("softmaxCrossEntropy", OpCategory.LOSS, 2, true, true, "Softmax Cross-Entropy Loss", Set.of(), null),
        new OpDefinition("softmaxCrossEntropySparse", OpCategory.LOSS, 2, true, true, "Sparse Softmax Cross-Entropy Loss", Set.of(), null),
        new OpDefinition("bceLoss",   OpCategory.LOSS, 2, true, true, "Binary Cross Entropy Loss", Set.of(), null),
        new OpDefinition("focalLoss", OpCategory.LOSS, 2, false, true, "Focal Loss", Set.of(), null),
        new OpDefinition("diceLoss",  OpCategory.LOSS, 2, false, true, "Dice Loss", Set.of(), null),

        // ── Graph structure (arity 0 = leaf/constant, no backend execution) ──
        new OpDefinition("leaf",     OpCategory.GRAPH, 0, true, true, "Leaf Variable", Set.of(), null),
        new OpDefinition("constant", OpCategory.GRAPH, 0, true, true, "Constant", Set.of(), null),

        // ── Compound DL ──
        new OpDefinition("linear", OpCategory.DL, 3, true, true, "Linear (Fully Connected)", Set.of(), null)
    );

    // ========================================================================
    // Derived views — computed from ALL_OPS
    // ========================================================================

    /** All ops with GPU support (including leaves/constants). */
    public static List<OpDefinition> gpuOps() {
        return ALL_OPS.stream().filter(OpDefinition::gpu).toList();
    }

    /** All ops with HPC support. */
    public static List<OpDefinition> hpcOps() {
        return ALL_OPS.stream().filter(OpDefinition::hpc).toList();
    }

    /** Unary ops that participate in {unary}{Reduce} fusion. */
    public static List<OpDefinition> fusionUnaryOps() {
        return ALL_OPS.stream().filter(OpDefinition::isFusionBase).toList();
    }

    /** Ops with a human-readable description (for describeTag()). */
    public static List<OpDefinition> describedOps() {
        return ALL_OPS.stream()
            .filter(o -> !o.description().equals(o.tag()))
            .toList();
    }

    /** Lookup by tag. */
    public static Optional<OpDefinition> findByTag(String tag) {
        return ALL_OPS.stream().filter(o -> o.tag().equals(tag)).findFirst();
    }

    /** Total count. */
    public static int size() { return ALL_OPS.size(); }

    // ========================================================================
    // Compound specials — ops NOT in {unary}{Reduce} pattern
    // ========================================================================

    private static final Set<String> COMPOUND_SPECIAL_TAGS = Set.of(
        "logSumExp", "softmaxCrossEntropy", "softmaxCrossEntropySparse",
        "bceLoss", "focalLoss", "diceLoss"
    );

    /** Check if a tag is a compound special. */
    public static boolean isCompoundSpecial(String tag) {
        return COMPOUND_SPECIAL_TAGS.contains(tag);
    }

    /** Compound special ops (all 6). */
    public static List<OpDefinition> compoundSpecials() {
        return ALL_OPS.stream()
            .filter(o -> COMPOUND_SPECIAL_TAGS.contains(o.tag()))
            .toList();
    }

    /** GPU-supported compound special tags. */
    public static Set<String> gpuCompoundSpecials() {
        Set<String> s = new LinkedHashSet<>();
        for (OpDefinition op : compoundSpecials())
            if (op.gpu()) s.add(op.tag());
        return s;
    }

    /** HPC-supported compound special tags. */
    public static Set<String> hpcCompoundSpecials() {
        Set<String> s = new LinkedHashSet<>();
        for (OpDefinition op : compoundSpecials())
            if (op.hpc()) s.add(op.tag());
        return s;
    }
}
