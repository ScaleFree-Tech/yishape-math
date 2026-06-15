package com.yishape.lab.math.autodiff.graph;

import java.util.*;
import java.util.stream.*;

/**
 * Validates and exports {@link GraphOpSchema} as JSON for cross-backend
 * consistency checking.
 *
 * <h3>Purpose</h3>
 * The {@code graphOpTag} schema is mirrored across 5 locations (Java, Rust HPC,
 * Rust GPU, {@code CUSTOM_OP_CONTRACT.md}, and yishape-dl CustomOp call sites).
 * This validator catches internal inconsistencies in the Java source of truth
 * and provides a JSON dump for cross-backend comparison.
 *
 * <h3>Usage (Java-side internal consistency)</h3>
 * <pre>{@code
 * GraphOpSchemaValidator.Result r = GraphOpSchemaValidator.validateAll();
 * if (r.hasErrors()) {
 *     throw new AssertionError("Schema inconsistency:\n" + r.toString());
 * }
 * }</pre>
 *
 * <h3>Usage (cross-backend export)</h3>
 * <pre>{@code
 * String json = GraphOpSchemaValidator.exportJson();
 * // Save json for comparison with Rust HPC/GPU op_schema.rs
 * }</pre>
 *
 * @see GraphOpSchema
 * @see <a href="file:../../../yishape-dl/private_docs/CUSTOM_OP_CONTRACT.md">CUSTOM_OP_CONTRACT.md</a>
 */
public final class GraphOpSchemaValidator {

    private GraphOpSchemaValidator() {}

    // ========================================================================
    // Schema descriptors — canonical encoding of every op
    // ========================================================================

    /** Descriptor for one op's schema. */
    public record OpSchema(
            String tag,
            List<InputDef> inputsNoBias,
            List<InputDef> inputsWithBias,
            List<ScalarField> scalarFields,
            List<ScalarField> scalar2Fields,
            List<GradDef> backwardGrads,
            int lenNoBias,
            int lenWithBias,
            String description) {}

    public record InputDef(String name, int index, String shape, String description) {}
    public record ScalarField(String name, int shift, long mask, String description) {}
    public record GradDef(String name, int index, String condition) {}

    public static final class Result {
        private final List<String> errors = new ArrayList<>();
        private final List<String> warnings = new ArrayList<>();

        public void addError(String msg) { errors.add(msg); }
        public void addWarning(String msg) { warnings.add(msg); }
        public boolean hasErrors() { return !errors.isEmpty(); }
        public boolean hasWarnings() { return !warnings.isEmpty(); }
        public List<String> errors() { return errors; }
        public List<String> warnings() { return warnings; }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            if (!errors.isEmpty()) {
                sb.append("ERRORS:\n");
                for (String e : errors) sb.append("  - ").append(e).append('\n');
            }
            if (!warnings.isEmpty()) {
                sb.append("WARNINGS:\n");
                for (String w : warnings) sb.append("  - ").append(w).append('\n');
            }
            if (!errors.isEmpty() && warnings.isEmpty()) sb.append("(no warnings)");
            return sb.toString();
        }
    }

    // ========================================================================
    // Op definitions — one per supported op tag
    // ========================================================================

    /** Returns the canonical list of all op schemas. */
    public static List<OpSchema> allOps() {
        return List.of(
            mha(),
            linear(),
            conv2d(),
            batchNorm2d(),
            layerNorm(),
            rmsNorm(),
            lstmStep(),
            embedding(),
            maxPool2d(),
            avgPool2d(),
            adaptiveAvgPool2d(),
            scaledDotProductAttention(),
            groupNorm(),
            instanceNorm(),
            depthwiseConv1d(),
            selectiveScan()
        );
    }

    // ---- 1. MHA ----

    private static OpSchema mha() {
        return new OpSchema(
            "mha",
            List.of(
                new InputDef("x", 0, "[seqLen * dModel]", "Input sequence (flat)"),
                new InputDef("wqkv", 1, "[qkvWidth * dModel]", "Combined QKV weight (row-major)"),
                new InputDef("wo", 2, "[dModel * dModel]", "Output projection weight (row-major)")
            ),
            List.of(
                new InputDef("x", 0, "[seqLen * dModel]", "Input sequence (flat)"),
                new InputDef("wqkv", 1, "[qkvWidth * dModel]", "Combined QKV weight (row-major)"),
                new InputDef("qkvBias", 2, "[qkvWidth]", "QKV bias"),
                new InputDef("wo", 3, "[dModel * dModel]", "Output projection weight (row-major)"),
                new InputDef("outBias", 4, "[dModel]", "Output bias")
            ),
            List.of(
                new ScalarField("numHeads", 32, 0xFFFFL, "Number of attention heads"),
                new ScalarField("numKVHeads", 16, 0xFFFFL, "Number of KV heads (GQA)"),
                new ScalarField("dModel", 0, 0xFFFFL, "Model dimension")
            ),
            List.of(
                new ScalarField("seqLen", 32, 0xFFFFL, "Sequence length"),
                new ScalarField("causal", 1, 1L, "Causal mask flag (bit)"),
                new ScalarField("hasBias", 0, 1L, "Has bias flag (bit)")
            ),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_wqkv", 1, "always"),
                new GradDef("d_qkvBias", 2, "hasBias"),
                new GradDef("d_wo", 3, "always"),
                new GradDef("d_outBias", 4, "hasBias")
            ),
            3, 5, "Multi-Head Self/Cross Attention with combined QKV projection"
        );
    }

    // ---- 2. Linear ----

    private static OpSchema linear() {
        return new OpSchema(
            "linear",
            List.of(
                new InputDef("x", 0, "[batch * inFeatures]", "Input (flat)"),
                new InputDef("weight", 1, "[outFeatures * inFeatures]", "Weight matrix (row-major)")
            ),
            List.of(
                new InputDef("x", 0, "[batch * inFeatures]", "Input (flat)"),
                new InputDef("weight", 1, "[outFeatures * inFeatures]", "Weight matrix (row-major)"),
                new InputDef("bias", 2, "[outFeatures]", "Bias vector")
            ),
            List.of(
                new ScalarField("batchSize", 32, 0xFFFFL, "Batch size"),
                new ScalarField("outFeatures", 16, 0xFFFFL, "Output features"),
                new ScalarField("inFeatures", 0, 0xFFFFL, "Input features")
            ),
            List.of(), // no scalar2
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_weight", 1, "always"),
                new GradDef("d_bias", 2, "hasBias")
            ),
            2, 3, "Fully Connected (Linear) layer"
        );
    }

    // ---- 3. Conv2d ----

    private static OpSchema conv2d() {
        return new OpSchema(
            "conv2d",
            List.of(
                new InputDef("x", 0, "[B * C * H * W]", "Input (NCHW flat)"),
                new InputDef("weight", 1, "[outCh * C * kH * kW]", "Weight (row-major)")
            ),
            List.of(
                new InputDef("x", 0, "[B * C * H * W]", "Input (NCHW flat)"),
                new InputDef("weight", 1, "[outCh * C * kH * kW]", "Weight (row-major)"),
                new InputDef("bias", 2, "[outCh]", "Bias vector")
            ),
            List.of(
                new ScalarField("packedShape", 0, 0xFFFFFFFFFFFFFFFFL, "Shape params (B,C,H,W,outCh,kH,kW,stride,pad)")
            ),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_weight", 1, "always"),
                new GradDef("d_bias", 2, "hasBias")
            ),
            2, 3, "2D Convolution"
        );
    }

    // ---- 4. BatchNorm2d ----

    private static OpSchema batchNorm2d() {
        return new OpSchema(
            "batchNorm2d",
            List.of(
                new InputDef("x", 0, "[N * C * H * W]", "Input (flat)"),
                new InputDef("gamma", 1, "[C]", "Scale parameter"),
                new InputDef("beta", 2, "[C]", "Shift parameter")
            ),
            List.of(
                new InputDef("x", 0, "[N * C * H * W]", "Input (flat)"),
                new InputDef("gamma", 1, "[C]", "Scale parameter"),
                new InputDef("beta", 2, "[C]", "Shift parameter")
            ),
            List.of(
                new ScalarField("N", 32, 0xFFFFL, "Batch size"),
                new ScalarField("C", 16, 0xFFFFL, "Channels"),
                new ScalarField("HW", 0, 0xFFFFL, "Spatial size (H*W)")
            ),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_gamma", 1, "always"),
                new GradDef("d_beta", 2, "always")
            ),
            3, 3, "Batch Normalization 2D (training mode)"
        );
    }

    // ---- 5. LayerNorm ----

    private static OpSchema layerNorm() {
        return new OpSchema(
            "layerNorm",
            List.of(
                new InputDef("x", 0, "[batch * features]", "Input (flat)"),
                new InputDef("gamma", 1, "[features]", "Scale parameter"),
                new InputDef("beta", 2, "[features]", "Shift parameter")
            ),
            List.of(
                new InputDef("x", 0, "[batch * features]", "Input (flat)"),
                new InputDef("gamma", 1, "[features]", "Scale parameter"),
                new InputDef("beta", 2, "[features]", "Shift parameter")
            ),
            List.of(
                new ScalarField("batchSize", 32, 0xFFFFL, "Batch size"),
                new ScalarField("features", 16, 0xFFFFL, "Feature dimension"),
                new ScalarField("eps_bits", 0, 0xFFFFL, "Epsilon bits")
            ),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_gamma", 1, "always"),
                new GradDef("d_beta", 2, "always")
            ),
            3, 3, "Layer Normalization"
        );
    }

    // ---- 6. RMSNorm ----

    private static OpSchema rmsNorm() {
        return new OpSchema(
            "rmsNorm",
            List.of(
                new InputDef("x", 0, "[batch * features]", "Input (flat)"),
                new InputDef("gamma", 1, "[features]", "Scale parameter")
            ),
            List.of(
                new InputDef("x", 0, "[batch * features]", "Input (flat)"),
                new InputDef("gamma", 1, "[features]", "Scale parameter")
            ),
            List.of(),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_gamma", 1, "always")
            ),
            2, 2, "RMS Normalization (no re-centering)"
        );
    }

    // ---- 7. LSTMStep ----

    private static OpSchema lstmStep() {
        return new OpSchema(
            "lstmStep",
            List.of(
                new InputDef("x", 0, "[inputSize]", "Input at current timestep"),
                new InputDef("h_prev", 1, "[hiddenSize]", "Previous hidden state"),
                new InputDef("c_prev", 2, "[hiddenSize]", "Previous cell state"),
                new InputDef("wI", 3, "[4H * inputSize]", "Input-to-hidden weight"),
                new InputDef("wH", 4, "[4H * hiddenSize]", "Hidden-to-hidden weight")
            ),
            List.of(
                new InputDef("x", 0, "[inputSize]", "Input at current timestep"),
                new InputDef("h_prev", 1, "[hiddenSize]", "Previous hidden state"),
                new InputDef("c_prev", 2, "[hiddenSize]", "Previous cell state"),
                new InputDef("wI", 3, "[4H * inputSize]", "Input-to-hidden weight"),
                new InputDef("wH", 4, "[4H * hiddenSize]", "Hidden-to-hidden weight"),
                new InputDef("biasI", 5, "[4H]", "Input bias"),
                new InputDef("biasH", 6, "[4H]", "Hidden bias")
            ),
            List.of(
                new ScalarField("hiddenSize", 0, 0xFFFFFFFFL, "Hidden size (as plain double)")
            ),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_hPrev", 1, "always"),
                new GradDef("d_cPrev", 2, "always"),
                new GradDef("d_wI", 3, "always"),
                new GradDef("d_wH", 4, "always"),
                new GradDef("d_biasI", 5, "hasBias"),
                new GradDef("d_biasH", 6, "hasBias")
            ),
            5, 7, "LSTM Single Timestep with fused gates"
        );
    }

    // ---- 8. Embedding ----

    private static OpSchema embedding() {
        return new OpSchema(
            "embedding",
            List.of(
                new InputDef("weight", 0, "[vocabSize * embeddingDim]", "Embedding table (row-major)"),
                new InputDef("indices", 1, "[numIndices]", "Token indices (as f64)")
            ),
            List.of(
                new InputDef("weight", 0, "[vocabSize * embeddingDim]", "Embedding table (row-major)"),
                new InputDef("indices", 1, "[numIndices]", "Token indices (as f64)")
            ),
            List.of(),
            List.of(),
            List.of(
                new GradDef("d_weight", 0, "always")
            ),
            2, 2, "Embedding Lookup (row-wise)"
        );
    }

    // ---- 9. MaxPool2d ----

    private static OpSchema maxPool2d() {
        return new OpSchema(
            "maxpool2d",
            List.of(
                new InputDef("x", 0, "[B * C * H * W]", "Input (flat)")
            ),
            List.of(
                new InputDef("x", 0, "[B * C * H * W]", "Input (flat)")
            ),
            List.of(
                new ScalarField("packed", 0, 0xFFFFFFFFFFFFFFFFL, "kernelH, kernelW, stride, padding")
            ),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always")
            ),
            1, 1, "Max Pooling 2D"
        );
    }

    // ---- 10. DepthwiseConv1d ----

    private static OpSchema depthwiseConv1d() {
        return new OpSchema(
            "depthwiseConv1d",
            List.of(
                new InputDef("x", 0, "[B * C * L]", "Input (BLC flat)"),
                new InputDef("weight", 1, "[C * kernelSize]", "Per-channel conv weight")
            ),
            List.of(
                new InputDef("x", 0, "[B * C * L]", "Input (BLC flat)"),
                new InputDef("weight", 1, "[C * kernelSize]", "Per-channel conv weight")
            ),
            List.of(
                new ScalarField("packed", 0, 0xFFFFFFFFFFFFFFFFL, "B, C, L, kernelSize, padding")
            ),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_weight", 1, "always")
            ),
            2, 2, "Depthwise 1D Convolution (Mamba SSM)"
        );
    }

    // ---- 11. SelectiveScan ----

    private static OpSchema selectiveScan() {
        return new OpSchema(
            "selectiveScan",
            List.of(
                new InputDef("u", 0, "[seqLen * dim]", "Input sequence (flat)"),
                new InputDef("delta", 1, "[seqLen * dim]", "Delta (time step)"),
                new InputDef("A", 2, "[dim * stateSize]", "State matrix A"),
                new InputDef("B", 3, "[seqLen * stateSize]", "Input projection B"),
                new InputDef("C", 4, "[seqLen * stateSize]", "Output projection C")
            ),
            List.of(
                new InputDef("u", 0, "[seqLen * dim]", "Input sequence (flat)"),
                new InputDef("delta", 1, "[seqLen * dim]", "Delta (time step)"),
                new InputDef("A", 2, "[dim * stateSize]", "State matrix A"),
                new InputDef("B", 3, "[seqLen * stateSize]", "Input projection B"),
                new InputDef("C", 4, "[seqLen * stateSize]", "Output projection C"),
                new InputDef("D", 5, "[dim]", "Skip connection D (optional)")
            ),
            List.of(
                new ScalarField("packed", 0, 0xFFFFFFFFFFFFFFFFL, "seqLen, dim, stateSize")
            ),
            List.of(),
            List.of(
                new GradDef("d_u", 0, "always"),
                new GradDef("d_delta", 1, "always"),
                new GradDef("d_A", 2, "always"),
                new GradDef("d_B", 3, "always"),
                new GradDef("d_C", 4, "always"),
                new GradDef("d_D", 5, "hasD")
            ),
            5, 6, "Selective Scan (Mamba SSM)"
        );
    }

    // ---- 12. AvgPool2d ----

    private static OpSchema avgPool2d() {
        return new OpSchema(
            "avgpool2d",
            List.of(new InputDef("x", 0, "[B * C * H * W]", "Input (flat)")),
            List.of(new InputDef("x", 0, "[B * C * H * W]", "Input (flat)")),
            List.of(new ScalarField("kH_kW_stride", 0, 0xFFFFFFFFFFFFFFFFL, "kH<<16 | kW<<8 | stride")),
            List.of(new ScalarField("padding", 0, 0xFFFFFFFFFFFFFFFFL, "padding<<16")),
            List.of(new GradDef("d_x", 0, "always")),
            1, 1, "Average Pooling 2D"
        );
    }

    // ---- 13. AdaptiveAvgPool2d ----

    private static OpSchema adaptiveAvgPool2d() {
        return new OpSchema(
            "adaptiveAvgPool2d",
            List.of(new InputDef("x", 0, "[B * C * H * W]", "Input (flat)")),
            List.of(new InputDef("x", 0, "[B * C * H * W]", "Input (flat)")),
            List.of(new ScalarField("outH_outW", 0, 0xFFFFFFFFFFFFFFFFL, "outH<<16 | outW")),
            List.of(),
            List.of(new GradDef("d_x", 0, "always")),
            1, 1, "Adaptive Average Pooling 2D"
        );
    }

    // ---- 14. ScaledDotProductAttention ----

    private static OpSchema scaledDotProductAttention() {
        return new OpSchema(
            "scaledDotProductAttention",
            List.of(
                new InputDef("q", 0, "[batch * seqQ * dk]", "Query (flat)"),
                new InputDef("k", 1, "[batch * seqK * dk]", "Key (flat)"),
                new InputDef("v", 2, "[batch * seqK * dv]", "Value (flat)")
            ),
            List.of(
                new InputDef("q", 0, "[batch * seqQ * dk]", "Query (flat)"),
                new InputDef("k", 1, "[batch * seqK * dk]", "Key (flat)"),
                new InputDef("v", 2, "[batch * seqK * dv]", "Value (flat)")
            ),
            List.of(new ScalarField("dropout", 0, 0xFFFFFFFFFFFFFFFFL, "Dropout rate (as f64)")),
            List.of(),
            List.of(
                new GradDef("d_q", 0, "always"),
                new GradDef("d_k", 1, "always"),
                new GradDef("d_v", 2, "always")
            ),
            3, 3, "Scaled Dot-Product Attention (Q·K^T + softmax + V)"
        );
    }

    // ---- 15. GroupNorm ----

    private static OpSchema groupNorm() {
        return new OpSchema(
            "groupNorm",
            List.of(
                new InputDef("x", 0, "[B * C * H * W]", "Input (flat)"),
                new InputDef("gamma", 1, "[C]", "Scale per channel"),
                new InputDef("beta", 2, "[C]", "Shift per channel")
            ),
            List.of(
                new InputDef("x", 0, "[B * C * H * W]", "Input (flat)"),
                new InputDef("gamma", 1, "[C]", "Scale per channel"),
                new InputDef("beta", 2, "[C]", "Shift per channel")
            ),
            List.of(new ScalarField("numGroups", 0, 0xFFFFFFFFFFFFFFFFL, "Number of groups")),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_gamma", 1, "always"),
                new GradDef("d_beta", 2, "always")
            ),
            3, 3, "Group Normalization"
        );
    }

    // ---- 16. InstanceNorm ----

    private static OpSchema instanceNorm() {
        return new OpSchema(
            "instanceNorm",
            List.of(
                new InputDef("x", 0, "[N * C * H * W]", "Input (flat)"),
                new InputDef("gamma", 1, "[C]", "Scale per channel (optional)"),
                new InputDef("beta", 2, "[C]", "Shift per channel (optional)")
            ),
            List.of(
                new InputDef("x", 0, "[N * C * H * W]", "Input (flat)"),
                new InputDef("gamma", 1, "[C]", "Scale per channel (optional)"),
                new InputDef("beta", 2, "[C]", "Shift per channel (optional)")
            ),
            List.of(new ScalarField("features", 0, 0xFFFFFFFFFFFFFFFFL, "Number of features (C)")),
            List.of(),
            List.of(
                new GradDef("d_x", 0, "always"),
                new GradDef("d_gamma", 1, "always"),
                new GradDef("d_beta", 2, "always")
            ),
            3, 3, "Instance Normalization (per-sample, per-channel)"
        );
    }

    // ========================================================================
    // Validation
    // ========================================================================

    /**
     * Validates all op schemas for internal consistency.
     */
    public static Result validateAll() {
        Result r = new Result();
        Set<String> seenTags = new HashSet<>();

        for (OpSchema op : allOps()) {
            // Check for duplicate tags
            if (!seenTags.add(op.tag())) {
                r.addError("Duplicate tag: " + op.tag());
            }

            // Validate no-bias inputs
            validateInputs(r, op.tag(), op.inputsNoBias(), "noBias", op.lenNoBias());

            // Validate with-bias inputs (when applicable)
            if (op.lenWithBias() > op.lenNoBias()) {
                validateInputs(r, op.tag(), op.inputsWithBias(), "withBias", op.lenWithBias());
            }

            // Validate backward grads
            validateBackwardGrads(r, op.tag(), op.backwardGrads());

            // Validate scalar fields
            validateScalars(r, op.tag(), op.scalarFields(), "scalar");
            validateScalars(r, op.tag(), op.scalar2Fields(), "scalar2");

            // Consistent len: withBias should have >= noBias
            if (op.lenWithBias() < op.lenNoBias()) {
                r.addError(String.format("%s: lenWithBias (%d) < lenNoBias (%d)",
                    op.tag(), op.lenWithBias(), op.lenNoBias()));
            }
        }

        return r;
    }

    private static void validateInputs(Result r, String tag, List<InputDef> inputs, String variant, int expectedLen) {
        if (inputs == null) {
            r.addError(String.format("%s %s: inputs is null", tag, variant));
            return;
        }
        // Check for non-negative indices
        for (InputDef in : inputs) {
            if (in.index() < 0) {
                r.addError(String.format("%s %s: negative index %d for '%s'",
                    tag, variant, in.index(), in.name()));
            }
        }
        // Check for duplicate indices
        Set<Integer> indices = new HashSet<>();
        for (InputDef in : inputs) {
            if (!indices.add(in.index())) {
                r.addError(String.format("%s %s: duplicate index %d for '%s'",
                    tag, variant, in.index(), in.name()));
            }
        }
        // Check LEN matches input count
        if (inputs.size() != expectedLen) {
            r.addError(String.format("%s %s: expected %d inputs, got %d",
                tag, variant, expectedLen, inputs.size()));
        }
    }

    private static void validateBackwardGrads(Result r, String tag, List<GradDef> grads) {
        if (grads == null) {
            r.addError(tag + ": backwardGrads is null");
            return;
        }
        Set<Integer> indices = new HashSet<>();
        for (GradDef g : grads) {
            if (g.index() < 0) {
                r.addError(String.format("%s backward: negative index %d for '%s'",
                    tag, g.index(), g.name()));
            }
            if (!indices.add(g.index())) {
                r.addError(String.format("%s backward: duplicate index %d for '%s'",
                    tag, g.index(), g.name()));
            }
        }
    }

    private static void validateScalars(Result r, String tag, List<ScalarField> fields, String scalarName) {
        if (fields == null) return;
        Set<Integer> shifts = new HashSet<>();
        for (ScalarField f : fields) {
            if (f.shift() < 0) {
                r.addWarning(String.format("%s %s: negative shift %d for '%s'",
                    tag, scalarName, f.shift(), f.name()));
            }
            if (!shifts.add(f.shift())) {
                r.addWarning(String.format("%s %s: duplicate shift %d for '%s'",
                    tag, scalarName, f.shift(), f.name()));
            }
        }
    }

    // ========================================================================
    // JSON export
    // ========================================================================

    /**
     * Exports all op schemas as a JSON array string.
     */
    public static String exportJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");
        List<OpSchema> ops = allOps();
        for (int i = 0; i < ops.size(); i++) {
            sb.append(toJson(ops.get(i)));
            if (i < ops.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("]\n");
        return sb.toString();
    }

    private static String toJson(OpSchema op) {
        StringBuilder sb = new StringBuilder();
        sb.append("  {\n");
        jsonField(sb, 1, "tag", op.tag());
        sb.append(",\n");
        jsonField(sb, 1, "description", op.description());
        sb.append(",\n");
        jsonArray(sb, 1, "inputsNoBias", op.inputsNoBias());
        sb.append(",\n");
        jsonArray(sb, 1, "inputsWithBias", op.inputsWithBias());
        sb.append(",\n");
        jsonScalarArray(sb, 1, "scalarFields", op.scalarFields());
        sb.append(",\n");
        jsonScalarArray(sb, 1, "scalar2Fields", op.scalar2Fields());
        sb.append(",\n");
        jsonGradArray(sb, 1, "backwardGrads", op.backwardGrads());
        sb.append(",\n");
        jsonField(sb, 1, "lenNoBias", String.valueOf(op.lenNoBias()));
        sb.append(",\n");
        jsonField(sb, 1, "lenWithBias", String.valueOf(op.lenWithBias()));
        sb.append('\n');
        sb.append("  }");
        return sb.toString();
    }

    private static void jsonField(StringBuilder sb, int indent, String key, String value) {
        sb.append("    ".repeat(Math.max(0, indent)));
        sb.append('"').append(key).append("\": \"").append(escapeJson(value)).append('"');
    }

    private static void jsonArray(StringBuilder sb, int indent, String key, List<InputDef> items) {
        sb.append("    ".repeat(Math.max(0, indent)));
        sb.append('"').append(key).append("\": [\n");
        for (int i = 0; i < items.size(); i++) {
            InputDef in = items.get(i);
            sb.append("    ".repeat(Math.max(0, indent + 1)));
            sb.append("{\"name\": \"").append(in.name())
              .append("\", \"index\": ").append(in.index())
              .append(", \"shape\": \"").append(escapeJson(in.shape()))
              .append("\", \"description\": \"").append(escapeJson(in.description()))
              .append("\"}");
            if (i < items.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("    ".repeat(Math.max(0, indent))).append(']');
    }

    private static void jsonScalarArray(StringBuilder sb, int indent, String key, List<ScalarField> items) {
        sb.append("    ".repeat(Math.max(0, indent)));
        sb.append('"').append(key).append("\": [\n");
        for (int i = 0; i < items.size(); i++) {
            ScalarField f = items.get(i);
            sb.append("    ".repeat(Math.max(0, indent + 1)));
            sb.append("{\"name\": \"").append(f.name())
              .append("\", \"shift\": ").append(f.shift())
              .append(", \"mask\": ").append(f.mask())
              .append(", \"description\": \"").append(escapeJson(f.description()))
              .append("\"}");
            if (i < items.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("    ".repeat(Math.max(0, indent))).append(']');
    }

    private static void jsonGradArray(StringBuilder sb, int indent, String key, List<GradDef> items) {
        sb.append("    ".repeat(Math.max(0, indent)));
        sb.append('"').append(key).append("\": [\n");
        for (int i = 0; i < items.size(); i++) {
            GradDef g = items.get(i);
            sb.append("    ".repeat(Math.max(0, indent + 1)));
            sb.append("{\"name\": \"").append(g.name())
              .append("\", \"index\": ").append(g.index())
              .append(", \"condition\": \"").append(escapeJson(g.condition()))
              .append("\"}");
            if (i < items.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("    ".repeat(Math.max(0, indent))).append(']');
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }
}
