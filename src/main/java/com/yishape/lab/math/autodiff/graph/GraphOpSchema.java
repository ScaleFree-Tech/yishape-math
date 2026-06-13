package com.yishape.lab.math.autodiff.graph;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Canonical input/output index schema for every {@code graphOpTag} used in
 * cross-backend graph execution (Java → JSON → HPC/GPU).
 *
 * <h2>⚠️⚠️⚠️  CRITICAL: SINGLE SOURCE OF TRUTH  ⚠️⚠️⚠️</h2>
 *
 * <p>This class is the <b>single source of truth</b> in code form. The
 * {@code yishape-dl/private_docs/CUSTOM_OP_CONTRACT.md} document provides the
 * human-readable version.</p>
 *
 * <p><b style="color:red">CHANGING ANY CONSTANT IN THIS FILE WITHOUT UPDATING
 * ALL 3 MIRROR LOCATIONS WILL CAUSE SILENT INCORRECT RESULTS.</b>
 * The graph executors in HPC (Rust/faer) and GPU (Rust/wgpu) read inputs
 * <b>by array index</b> — an index mismatch produces garbage output with
 * zero error messages. This is the #1 cause of "loss looks normal but
 * accuracy is random" bugs.</p>
 *
 * <h3>Mandatory change checklist</h3>
 * <p>Changes to any op's input ordering, scalar bit layout, or backward grad
 * order <b>MUST</b> update ALL of:</p>
 * <ol>
 *   <li><b>This class</b> (yishape-math) — you are here</li>
 *   <li><b>{@code yishape_math_rust/src/op_schema.rs}</b> — HPC faer backend</li>
 *   <li><b>{@code yishape_math_gpu/src/ops/op_schema.rs}</b> — GPU wgpu backend</li>
 *   <li><b>{@code yishape-dl/private_docs/CUSTOM_OP_CONTRACT.md}</b> — human-readable spec</li>
 *   <li><b>All yishape-dl CustomOp forward/backward methods</b> that use the changed op</li>
 * </ol>
 *
 * <h3>What happens if you don't sync</h3>
 * <ul>
 *   <li>Java CustomOp reads input[2] as bias, but HPC graph executor reads input[2] as weight — bias values become weight values</li>
 *   <li>Backward returns grads in wrong order — wrong parameter gets updated</li>
 *   <li>No exception, no error message, loss decreases normally, model never converges</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * // In yishape-dl CustomOp:
 * op.tensorApply(shape,
 *     input,                    // GraphOpSchema.MHA.X
 *     wQkvT,                    // GraphOpSchema.MHA.W_QKV
 *     qkvBias.asLeafTensor(),   // GraphOpSchema.MHA.QKV_BIAS
 *     wOutT,                    // GraphOpSchema.MHA.woIdx(hasBias)
 *     outBias.asLeafTensor()    // GraphOpSchema.MHA.OUT_BIAS (when hasBias)
 * );
 * }</pre>
 *
 * <h3>Conditional inputs</h3>
 * Some ops have optional inputs (e.g. bias). When the optional input is absent,
 * subsequent indices shift. Always use the helper methods
 * ({@code woIdx(hasBias)}, etc.) rather than hardcoded numbers.
 *
 * @see <a href="file:../../../yishape-dl/private_docs/CUSTOM_OP_CONTRACT.md">CUSTOM_OP_CONTRACT.md</a>
 */
public final class GraphOpSchema {

    private GraphOpSchema() { /* constants only */ }

    // ========================================================================
    // 1. Multi-Head Attention
    // ========================================================================

    /**
     * Multi-Head Self/Cross Attention with combined QKV projection.
     *
     * @implNote This is the most complex op in terms of conditional input
     *           indexing. When hasBias=false, inputs.length=3 and WO shifts from
     *           index 3 to index 2. Both Java CustomOp.forward() and Rust
     *           graph executors must handle this shift identically.
     *           Use {@link #woIdx(boolean)} rather than hardcoded indices.
     */
    public static final class MHA {
        public static final String TAG = "mha";

        // --- Forward inputs (no bias: len=3, with bias: len=5) ---
        /** Input sequence x, shape [seqLen * dModel] */
        public static final int X = 0;
        /** Combined QKV weight, shape [qkvWidth * dModel] */
        public static final int W_QKV = 1;
        /** QKV bias, shape [qkvWidth] (only when hasBias=true) */
        public static final int QKV_BIAS = 2;
        /** Output projection weight index when bias IS present */
        public static final int WO_WITH_BIAS = 3;
        /** Output projection weight index when bias is NOT present */
        public static final int WO_NO_BIAS = 2;
        /** Output bias, shape [dModel] (only when hasBias=true) */
        public static final int OUT_BIAS = 4;

        /** Output projection weight index given hasBias flag. */
        public static int woIdx(boolean hasBias) { return hasBias ? WO_WITH_BIAS : WO_NO_BIAS; }

        // --- Scalar bit packing ---
        /** scalarParam bit ranges: [63-29 reserved] [28-25 numHeads] [24-21 numKVHeads] [20-0 dModel]
         *  All values fit in f64 53-bit mantissa: max packed = (255<<28)|(15<<24)|1048575 < 2^53 */
        public static final int SCALAR_NUMHEADS_SHIFT = 28;
        public static final long SCALAR_NUMHEADS_MASK = 0xFL;
        public static final int SCALAR_NUMKVHEADS_SHIFT = 24;
        public static final long SCALAR_NUMKVHEADS_MASK = 0xFL;
        public static final int SCALAR_DMODEL_SHIFT = 0;
        public static final long SCALAR_DMODEL_MASK = 0x1FFFFFL;

        /** scalarParam2 bit ranges: [63-36 reserved] [35-19 seqLen] [18-2 reserved] [1 causal] [0 hasBias]
         *  Max seqLen=131071 at bit 19: 131071<<19 = 68_719_476_736 < 2^53 */
        public static final int SCALAR2_SEQLEN_SHIFT = 19;
        public static final long SCALAR2_SEQLEN_MASK = 0x1FFFFL;
        public static final int SCALAR2_CAUSAL_BIT = 1;
        public static final int SCALAR2_HASBIAS_BIT = 0;

        // --- Backward grad return indices ---
        /** d_x gradient (always returned) */
        public static final int GRAD_DX = 0;
        /** d_wQkv gradient (always returned) */
        public static final int GRAD_DWQKV = 1;
        /** d_qkvBias gradient (only when hasBias) */
        public static final int GRAD_DQKVBIAS = 2;
        /** d_wo gradient (always returned) */
        public static final int GRAD_DWO = 3;
        /** d_outBias gradient (only when hasBias) */
        public static final int GRAD_DOUTBIAS = 4;
    }

    // ========================================================================
    // 2. Fully Connected (Linear)
    // ========================================================================

    public static final class Linear {
        public static final String TAG = "linear";

        /** Input x, shape [batch * inFeatures] */
        public static final int X = 0;
        /** Weight matrix, shape [outFeatures * inFeatures] (row-major) */
        public static final int WEIGHT = 1;
        /** Bias vector, shape [outFeatures] (only when hasBias, index=2) */
        public static final int BIAS = 2;

        /** Total inputs without bias */
        public static final int LEN_NO_BIAS = 2;
        /** Total inputs with bias */
        public static final int LEN_WITH_BIAS = 3;

        // Scalar packing: [63-48 reserved] [47-32 batchSize] [31-16 outFeatures] [15-0 inFeatures]
        public static final int SCALAR_BATCH_SHIFT = 32;
        public static final long SCALAR_BATCH_MASK = 0xFFFFL;
        public static final int SCALAR_OUT_SHIFT = 16;
        public static final long SCALAR_OUT_MASK = 0xFFFFL;
        public static final int SCALAR_IN_SHIFT = 0;
        public static final long SCALAR_IN_MASK = 0xFFFFL;

        // Backward
        public static final int GRAD_DX = 0;
        public static final int GRAD_DWEIGHT = 1;
        public static final int GRAD_DBIAS = 2; // conditional
    }

    // ========================================================================
    // 3. 2D Convolution
    // ========================================================================

    public static final class Conv2d {
        public static final String TAG = "conv2d";

        public static final int X = 0;
        public static final int WEIGHT = 1; // [outCh * C * kH * kW]
        public static final int BIAS = 2;   // conditional

        public static final int LEN_NO_BIAS = 2;
        public static final int LEN_WITH_BIAS = 3;

        // Backward
        public static final int GRAD_DX = 0;
        public static final int GRAD_DWEIGHT = 1;
        public static final int GRAD_DBIAS = 2; // conditional
    }

    // ========================================================================
    // 3b. 2D Transposed Convolution
    // ========================================================================

    public static final class ConvTranspose2d {
        public static final String TAG = "convTranspose2d";

        public static final int X = 0;      // [inCh * H * W]
        public static final int WEIGHT = 1; // [outCh * inCh * kH * kW]
        public static final int BIAS = 2;   // conditional, [outCh]

        public static final int LEN_NO_BIAS = 2;
        public static final int LEN_WITH_BIAS = 3;

        // Backward
        public static final int GRAD_DX = 0;
        public static final int GRAD_DWEIGHT = 1;
        public static final int GRAD_DBIAS = 2; // conditional
    }

    // ========================================================================
    // 4. Batch Normalization 2D (training mode)
    // ========================================================================

    public static final class BatchNorm2d {
        public static final String TAG = "batchNorm2d";

        public static final int X = 0;     // [N * C * H * W]
        public static final int GAMMA = 1; // [C]
        public static final int BETA = 2;  // [C]
        public static final int LEN = 3;

        // Scalar: [63-48 reserved] [47-32 N] [31-16 C] [15-0 H*W]
        public static final int SCALAR_N_SHIFT = 32;
        public static final long SCALAR_N_MASK = 0xFFFFL;
        public static final int SCALAR_C_SHIFT = 16;
        public static final long SCALAR_C_MASK = 0xFFFFL;
        public static final int SCALAR_HW_SHIFT = 0;
        public static final long SCALAR_HW_MASK = 0xFFFFL;

        // Backward
        public static final int GRAD_DX = 0;
        public static final int GRAD_DGAMMA = 1;
        public static final int GRAD_DBETA = 2;
    }

    // ========================================================================
    // 5. Layer Normalization
    // ========================================================================

    public static final class LayerNorm {
        public static final String TAG = "layerNorm";

        public static final int X = 0;     // [batch * features]
        public static final int GAMMA = 1; // [features]
        public static final int BETA = 2;  // [features]
        public static final int LEN = 3;

        // Scalar: [63-48 reserved] [47-32 batchSize] [31-16 features] [15-0 eps_bits]
        public static final int SCALAR_BATCH_SHIFT = 32;
        public static final long SCALAR_BATCH_MASK = 0xFFFFL;
        public static final int SCALAR_FEATURES_SHIFT = 16;
        public static final long SCALAR_FEATURES_MASK = 0xFFFFL;

        // Backward
        public static final int GRAD_DX = 0;
        public static final int GRAD_DGAMMA = 1;
        public static final int GRAD_DBETA = 2;
    }

    // ========================================================================
    // 6. RMS Normalization
    // ========================================================================

    public static final class RMSNorm {
        public static final String TAG = "rmsNorm";

        public static final int X = 0;     // [batch * features]
        public static final int GAMMA = 1; // [features]
        public static final int LEN = 2;

        // Backward
        public static final int GRAD_DX = 0;
        public static final int GRAD_DGAMMA = 1;
    }

    // ========================================================================
    // 7. LSTM Single Timestep
    // ========================================================================

    /**
     * LSTM single-timestep with fused gate computation.
     *
     * @implNote Conditional bias inputs ({@link #BIAS_I}, {@link #BIAS_H}) shift
     *           from indices 5,6 (hasBias=true, len=7) to absent (hasBias=false,
     *           len=5). Backward grad return length changes accordingly.
     *           Java and both Rust backends must handle this shift identically.
     */
    public static final class LSTMStep {
        public static final String TAG = "lstmStep";

        // Without bias (len=5)
        public static final int X = 0;     // [inputSize]
        public static final int H_PREV = 1; // [hiddenSize]
        public static final int C_PREV = 2; // [hiddenSize]
        public static final int WI = 3;     // [4H * inputSize]
        public static final int WH = 4;     // [4H * hiddenSize]

        // With bias (len=7)
        public static final int BIAS_I = 5; // [4H]
        public static final int BIAS_H = 6; // [4H]

        public static final int LEN_NO_BIAS = 5;
        public static final int LEN_WITH_BIAS = 7;

        // Backward
        public static final int GRAD_DX = 0;
        public static final int GRAD_DHPREV = 1;
        public static final int GRAD_DCPREV = 2;
        public static final int GRAD_DWI = 3;
        public static final int GRAD_DWH = 4;
        public static final int GRAD_DBIASI = 5; // conditional
        public static final int GRAD_DBIASH = 6; // conditional
    }

    // ========================================================================
    // 8. Embedding Lookup
    // ========================================================================

    public static final class Embedding {
        public static final String TAG = "embedding";

        public static final int WEIGHT = 0;  // [vocabSize * embeddingDim]
        public static final int INDICES = 1; // [numIndices] (stored as f64)
        public static final int LEN = 2;

        // Backward
        public static final int GRAD_DWEIGHT = 0;
    }

    // ========================================================================
    // 9. Max Pooling 2D
    // ========================================================================

    public static final class MaxPool2d {
        public static final String TAG = "maxpool2d";

        public static final int X = 0; // [B * C * H * W]
        public static final int LEN = 1;

        // Backward
        public static final int GRAD_DX = 0;
    }

    // ========================================================================
    // 10. Depthwise 1D Convolution
    // ========================================================================

    public static final class DepthwiseConv1d {
        public static final String TAG = "depthwiseConv1d";

        public static final int X = 0;      // [B * C * L]
        public static final int WEIGHT = 1; // [C * kernelSize]
        public static final int LEN = 2;

        // Backward
        public static final int GRAD_DX = 0;
        public static final int GRAD_DWEIGHT = 1;
    }

    // ========================================================================
    // 11. Selective Scan (Mamba SSM)
    // ========================================================================

    public static final class SelectiveScan {
        public static final String TAG = "selectiveScan";

        public static final int U = 0;     // [seqLen * dim]
        public static final int DELTA = 1; // [seqLen * dim]
        public static final int A = 2;     // [dim * stateSize]
        public static final int B = 3;     // [seqLen * stateSize]
        public static final int C = 4;     // [seqLen * stateSize]
        public static final int D = 5;     // [dim] (optional)
        public static final int LEN = 6;
        public static final int LEN_NO_D = 5;

        // Backward
        public static final int GRAD_DU = 0;
        public static final int GRAD_DDELTA = 1;
        public static final int GRAD_DA = 2;
        public static final int GRAD_DB = 3;
        public static final int GRAD_DC = 4;
        public static final int GRAD_DD = 5; // conditional
    }

    /** Selective Scan variant 2 (chunked parallel scan). */
    public static final class SelectiveScan2 {
        public static final String TAG = "selectiveScan2";
        // Same input layout as SelectiveScan; delta semantics differ
        public static final int U = SelectiveScan.U;
        public static final int DELTA = SelectiveScan.DELTA;
        public static final int A = SelectiveScan.A;
        public static final int B = SelectiveScan.B;
        public static final int C = SelectiveScan.C;
        public static final int D = SelectiveScan.D;
    }

    /** Selective Scan variant: trapezoidal rule discretization. */
    public static final class TrapezoidalScan {
        public static final String TAG = "trapezoidalScan";
        public static final int U = SelectiveScan.U;
        public static final int DELTA = SelectiveScan.DELTA;
        public static final int A = SelectiveScan.A;
        public static final int B = SelectiveScan.B;
        public static final int C = SelectiveScan.C;
        public static final int D = SelectiveScan.D;
    }

    // ========================================================================
    // Tag lookup — maps TAG string to the human-readable op name
    // ========================================================================

    /** Returns a human-readable name for a graphOpTag, or the tag itself if unknown. */
    public static String describeTag(String graphOpTag) {
        return switch (graphOpTag) {
            case "mha"               -> "Multi-Head Attention";
            case "linear"            -> "Linear (Fully Connected)";
            case "conv2d"            -> "2D Convolution";
            case "convTranspose2d"   -> "2D Transposed Convolution";
            case "batchNorm2d"       -> "Batch Normalization 2D";
            case "layerNorm"         -> "Layer Normalization";
            case "rmsNorm"           -> "RMS Normalization";
            case "lstmStep"          -> "LSTM Timestep";
            case "embedding"         -> "Embedding Lookup";
            case "maxpool2d"         -> "Max Pooling 2D";
            case "depthwiseConv1d"   -> "Depthwise 1D Convolution";
            case "selectiveScan"     -> "Selective Scan (Mamba SSM)";
            case "selectiveScan2"    -> "Selective Scan 2 (Chunked)";
            case "trapezoidalScan"   -> "Trapezoidal Scan (Mamba SSM)";
            case "bceLoss"           -> "Binary Cross Entropy Loss";
            case "focalLoss"         -> "Focal Loss";
            case "diceLoss"          -> "Dice Loss";
            default                  -> graphOpTag;
        };
    }

    // ========================================================================
    // Fused Tag Naming — single authority for fused op tag convention
    // ========================================================================

    /**
     * Canonical naming convention for fused operation tags.
     *
     * <h2>Pattern</h2>
     * Simple fused: {@code {unaryTag}{ReduceTag}} in camelCase, e.g.
     * "square" + "mean" → "squareMean".
     *
     * <h2>Multi-step chains</h2>
     * Arbitrary element-wise chain + reduction terminator, e.g.
     * ["square","addScalar","sqrt"] + "mean" → "squareAddScalarSqrtMean".
     * Use {@link #ofChain(List, String)} for chains, {@link #of(String, String)} for single unary.
     *
     * <h2>Usage</h2>
     * All fused tag generation MUST use these methods — never hand-concatenate.
     */
    public static final class FusedTag {

        /** Builds a simple fused tag: of("square", "mean") → "squareMean". */
        public static String of(String unaryTag, String reduceTag) {
            return unaryTag + Character.toUpperCase(reduceTag.charAt(0))
                   + reduceTag.substring(1);
        }

        /** Builds a multi-step chain tag: ofChain(["square","addScalar","sqrt"], "mean") → "squareAddScalarSqrtMean". */
        public static String ofChain(List<String> opTags, String reduceTag) {
            if (opTags.isEmpty()) return reduceTag;
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < opTags.size(); i++) {
                String tag = opTags.get(i);
                if (i == 0) sb.append(tag);
                else sb.append(Character.toUpperCase(tag.charAt(0))).append(tag, 1, tag.length());
            }
            sb.append(Character.toUpperCase(reduceTag.charAt(0)));
            if (reduceTag.length() > 1) sb.append(reduceTag, 1, reduceTag.length());
            return sb.toString();
        }

        // ---- Component sets ----

        /** 19 unary ops that participate in {unary}{Reduce} fusion. */
        public static final Set<String> UNARY_TAGS = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "square", "relu", "exp", "abs", "log", "sigmoid", "tanh",
            "silu", "pow", "gelu", "mish", "sin", "cos",
            "leakyRelu", "elu", "selu", "softplus", "hardtanh", "mul"
        )));

        /** Reduction terminators for {unary}{Reduce} fusion. */
        public static final Set<String> REDUCE_TAGS = Set.of("sum", "mean");

        /** Compound special fused tags (non-{unary}{Reduce} patterned). */
        public static final Set<String> COMPOUND_SPECIALS = Set.of(
            "logSumExp", "softmaxCrossEntropy", "softmaxCrossEntropySparse",
            "bceLoss", "focalLoss", "diceLoss"
        );
    }

    // ========================================================================
    // Fused Tag Registry — which combinations are implemented per backend
    // ========================================================================

    /**
     * Registry of which fused op tags have native implementations in each backend.
     *
     * <p>All pattern-generated tags derive from {@link FusedTag#UNARY_TAGS} ×
     * {@link FusedTag#REDUCE_TAGS}. Backend-specific subsets ({@link #GPU_PATTERN},
     * {@link #HPC_PATTERN}) record which combinations are actually implemented.
     * Compound specials are tracked separately.
     *
     * <p>When adding a new fused op:
     * <ol>
     *   <li>Ensure the unary tag is in {@link FusedTag#UNARY_TAGS}</li>
     *   <li>Implement in Rust backend</li>
     *   <li>Add the tag string to {@link #GPU_PATTERN} and/or {@link #HPC_PATTERN}</li>
     * </ol>
     */
    public static final class FusedTagRegistry {

        /** All {unary}{Reduce} pattern tags (cartesian product: 19×2 = 38). */
        public static final Set<String> ALL_PATTERN;
        static {
            Set<String> all = new HashSet<>();
            for (String u : FusedTag.UNARY_TAGS)
                for (String r : FusedTag.REDUCE_TAGS)
                    all.add(FusedTag.of(u, r));
            ALL_PATTERN = Collections.unmodifiableSet(all);
        }

        // ---- GPU-supported fused subsets ----

        /** {unary}{Reduce} tags with GPU WGSL implementations. */
        public static final Set<String> GPU_PATTERN = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "squareSum", "squareMean", "reluSum", "reluMean", "expSum", "expMean",
            "absSum", "absMean", "logSum", "logMean", "sigmoidSum", "sigmoidMean",
            "tanhSum", "tanhMean", "siluSum", "siluMean", "mishSum", "mishMean",
            "geluSum", "geluMean", "sinSum", "sinMean", "cosSum", "cosMean",
            "leakyReluSum", "leakyReluMean", "eluSum", "eluMean", "seluSum", "seluMean",
            "softplusSum", "softplusMean", "hardtanhSum", "hardtanhMean",
            "mulSum", "mulMean", "powSum", "powMean"
        )));

        /** Compound specials with GPU implementations. */
        public static final Set<String> GPU_COMPOUND = Set.of(
            "logSumExp", "softmaxCrossEntropy", "softmaxCrossEntropySparse", "bceLoss"
        );

        /** All GPU-supported fused tags (pattern + compound). */
        public static final Set<String> GPU_ALL;
        static {
            Set<String> s = new HashSet<>(GPU_PATTERN);
            s.addAll(GPU_COMPOUND);
            GPU_ALL = Collections.unmodifiableSet(s);
        }

        // ---- HPC-supported fused subsets ----

        /** {unary}{Reduce} tags with HPC faer implementations. */
        public static final Set<String> HPC_PATTERN = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "squareSum", "squareMean", "reluSum", "reluMean", "expSum", "expMean",
            "absSum", "absMean", "logSum", "logMean", "sigmoidSum", "sigmoidMean",
            "tanhSum", "tanhMean", "siluSum", "siluMean", "mishSum", "mishMean",
            "mulSum", "mulMean", "powSum", "powMean",
            "geluSum", "geluMean", "sinSum", "sinMean", "cosSum", "cosMean",
            "leakyReluSum", "leakyReluMean", "eluSum", "eluMean",
            "seluSum", "seluMean", "softplusSum", "softplusMean",
            "hardtanhSum", "hardtanhMean"
        )));

        /** Compound specials with HPC implementations. */
        public static final Set<String> HPC_COMPOUND = Set.of(
            "logSumExp", "softmaxCrossEntropy", "softmaxCrossEntropySparse",
            "bceLoss", "focalLoss", "diceLoss"
        );

        /** All HPC-supported fused tags (pattern + compound). */
        public static final Set<String> HPC_ALL;
        static {
            Set<String> s = new HashSet<>(HPC_PATTERN);
            s.addAll(HPC_COMPOUND);
            HPC_ALL = Collections.unmodifiableSet(s);
        }
    }

    // ========================================================================
    // Supported operation sets — single source of truth
    // ========================================================================

    /** All ops that have GPU WGSL shader implementations (tensor-native path). */
    public static final class Gpu {
        static final Set<String> BASE = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "add", "sub", "mul", "div",
            "addScalar", "subScalar", "mulScalar", "divScalar", "rsubScalar", "rdivScalar",
            "neg", "pow", "exp", "log", "sin", "cos", "tan",
            "sigmoid", "tanh", "relu", "abs", "sqrt", "square", "dropout",
            "sum", "mean", "dot", "matmul",
            "gelu", "softmax", "logSoftmax", "silu", "mish",
            "leakyRelu", "elu", "selu", "softplus", "hardtanh", "clamp",
            "normalize", "layerNorm",
            "broadcast", "transpose", "reshape", "flatten",
            "squeeze", "unsqueeze",
            "mmul",
            "leaf", "constant",
            "linear", "conv2d", "convTranspose2d", "maxpool2d", "avgpool2d", "adaptiveAvgPool2d",
            "batchNorm2d", "embedding", "mha", "lstmStep",
            "selectiveScan", "selectiveScan2", "depthwiseConv1d",
            "scaledDotProductAttention",
            "rmsNorm", "groupNorm", "instanceNorm",
            "interpolate", "gridSample",
            "cross", "trapezoidalScan", "cat",
            "gather", "select", "contiguous",
            "permute", "slice"
        )));

        public static final Set<String> SUPPORTED;
        static {
            Set<String> s = new HashSet<>(BASE);
            s.addAll(FusedTagRegistry.GPU_ALL);
            SUPPORTED = Collections.unmodifiableSet(s);
        }
    }

    /** All ops that have HPC faer-based implementations (tensor-native path). */
    public static final class Hpc {
        static final Set<String> BASE = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            "add", "sub", "mul", "div",
            "addScalar", "subScalar", "mulScalar", "divScalar", "rsubScalar", "rdivScalar",
            "neg", "pow", "exp", "log", "sin", "cos", "tan",
            "sigmoid", "tanh", "relu", "abs", "sqrt", "square", "dropout",
            "sum", "mean", "dot", "matmul",
            "gelu", "softmax", "logSoftmax", "silu", "mish",
            "leakyRelu", "elu", "selu", "softplus", "hardtanh", "clamp",
            "layerNorm",
            "broadcast", "transpose", "reshape", "flatten",
            "squeeze", "unsqueeze",
            "mmul",
            "leaf", "constant",
            "linear", "conv2d", "convTranspose2d", "maxpool2d", "avgpool2d", "batchNorm2d",
            "embedding", "mha", "lstmStep",
            "selectiveScan", "selectiveScan2", "depthwiseConv1d",
            "permute", "expand", "reciprocal", "rsub", "rdiv",
            "gather", "scatter", "select", "slice", "narrow",
            "cat", "contiguous",
            "batchNorm", "groupNorm"
        )));

        public static final Set<String> SUPPORTED;
        static {
            Set<String> s = new HashSet<>(BASE);
            s.addAll(FusedTagRegistry.HPC_ALL);
            SUPPORTED = Collections.unmodifiableSet(s);
        }
    }

    /**
     * Union of GPU + HPC tensor-supported ops.
     * Used by GraphIntegrityChecker for graph integrity validation.
     * An op must appear in GPU_TENSOR or HPC_TENSOR to pass the check.
     */
    public static final Set<String> COMMON_TENSOR;
    static {
        HashSet<String> u = new HashSet<>(Gpu.SUPPORTED);
        u.addAll(Hpc.SUPPORTED);
        COMMON_TENSOR = Collections.unmodifiableSet(u);
    }
}
