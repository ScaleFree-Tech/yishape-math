package com.yishape.lab.math.autodiff.graph;

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
        /** scalarParam bit ranges: [63-48 reserved] [47-32 numHeads] [31-16 numKVHeads] [15-0 dModel] */
        public static final int SCALAR_NUMHEADS_SHIFT = 32;
        public static final long SCALAR_NUMHEADS_MASK = 0xFFFFL;
        public static final int SCALAR_NUMKVHEADS_SHIFT = 16;
        public static final long SCALAR_NUMKVHEADS_MASK = 0xFFFFL;
        public static final int SCALAR_DMODEL_SHIFT = 0;
        public static final long SCALAR_DMODEL_MASK = 0xFFFFL;

        /** scalarParam2 bit ranges: [63-48 reserved] [47-32 seqLen] [31-2 reserved] [1 causal] [0 hasBias] */
        public static final int SCALAR2_SEQLEN_SHIFT = 32;
        public static final long SCALAR2_SEQLEN_MASK = 0xFFFFL;
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
            default                  -> graphOpTag;
        };
    }
}
