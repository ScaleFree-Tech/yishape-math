package com.yishape.lab.math.autodiff.graph;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;

/**
 * CI contract test: Java GraphOpSchema ↔ Rust backend op implementations.
 *
 * <p>Validates that every op declared as SUPPORTED in the Java schema
 * has a corresponding implementation in the Rust GPU (wgpu) and HPC (faer)
 * backends. The Rust-supported sets below are curated from:
 * <ul>
 *   <li>{@code yishape_math_gpu/src/ops/graph.rs} — forward_dispatch / backward_dispatch</li>
 *   <li>{@code yishape_math_rust/src/graph.rs} — forward_node / backward_fn</li>
 * </ul>
 *
 * <p>When adding a new op to any Rust backend, update the corresponding
 * set below AND the Java {@link GraphOpSchema}. When removing an op,
 * update both sides.</p>
 */
public class SchemaContractTest {

    // ========================================================================
    // Rust GPU (wgpu) — curated from yishape_math_gpu/src/ops/graph.rs
    // Last updated: 2026-06-17
    // ========================================================================

    /** Ops with forward AND backward WGSL shader implementations in Rust GPU. */
    private static final Set<String> RUST_GPU_FORWARD_BACKWARD = Collections.unmodifiableSet(
        new HashSet<>(Arrays.asList(
            // Binary element-wise
            "add", "sub", "mul", "div",
            "addScalar", "subScalar", "mulScalar", "divScalar", "rsubScalar", "rdivScalar",
            // Unary
            "neg", "abs", "exp", "log", "sqrt", "square", "dropout",
            "pow", "sin", "cos", "tan",
            // Activations
            "relu", "gelu", "sigmoid", "tanh", "leakyRelu", "elu", "selu",
            "silu", "mish", "softplus", "hardtanh", "clamp",
            // Reduce
            "sum", "mean",
            // BLAS
            "dot", "matmul", "mmul",
            // Normalization
            "normalize", "layerNorm", "rmsNorm", "groupNorm", "instanceNorm",
            "batchNorm2d",
            // Shape ops
            "broadcast", "transpose", "reshape", "flatten", "squeeze", "unsqueeze",
            "expand", "permute", "slice", "contiguous",
            // Indexing
            "select", "gather", "cat",
            // DL ops
            "softmax", "logSoftmax",
            "linear", "conv2d", "convTranspose2d", "maxpool2d", "avgpool2d",
            "adaptiveAvgPool2d",
            "embedding", "mha", "lstmStep",
            "selectiveScan", "selectiveScan2", "depthwiseConv1d",
            "scaledDotProductAttention",
            "interpolate", "gridSample", "cross", "trapezoidalScan",
            // Loss ops
            "softmaxCrossEntropy", "softmaxCrossEntropySparse", "bceLoss",
            // Fused (forward + backward)
            "sinSum", "sinMean", "cosSum", "cosMean",
            "geluSum", "geluMean",
            "mulSum", "mulMean",
            "powSum", "powMean",
            "logSumExp",
            // Meta
            "leaf", "constant"
        )));

    /** Ops present in Rust GPU forward_dispatch but without separate backward dispatch
     *  (backward is handled by generic graph traversal, not op-specific WGSL shaders).
     *  These should NOT be in GPU SUPPORTED because the backward path is incomplete. */
    private static final Set<String> RUST_GPU_FORWARD_ONLY = Collections.unmodifiableSet(
        new HashSet<>(Arrays.asList(
            // batchNorm — GPU only has batchNorm2d; batchNorm (1D) is not in Rust GPU
        )));

    // ========================================================================
    // Rust HPC (faer) — curated from yishape_math_rust/src/graph.rs
    // Last updated: 2026-06-17
    // ========================================================================

    /** Ops with forward AND backward implementations in Rust HPC. */
    private static final Set<String> RUST_HPC_FORWARD_BACKWARD = Collections.unmodifiableSet(
        new HashSet<>(Arrays.asList(
            // Binary element-wise
            "add", "sub", "mul", "div",
            "addScalar", "subScalar", "mulScalar", "divScalar", "rsubScalar", "rdivScalar",
            // Unary
            "neg", "pow", "exp", "log", "sin", "cos", "tan",
            "sigmoid", "tanh", "relu", "abs", "sqrt", "square", "dropout",
            "reciprocal",
            // Normalization
            "normalize", "groupNorm",
            // Activations
            "gelu", "softmax", "logSoftmax", "leakyRelu", "elu", "selu",
            "silu", "mish", "softplus", "hardtanh", "clamp",
            // Reduce
            "sum", "mean", "logSumExp",
            // BLAS
            "dot", "matmul", "mmul", "bmm",
            // DL ops
            "linear", "conv2d", "convTranspose2d", "maxpool2d", "avgpool2d",
            "batchNorm2d", "batchNorm",
            "embedding", "mha", "lstmStep",
            "selectiveScan", "selectiveScan2", "depthwiseConv1d",
            "scaledDotProductAttention",
            // Normalization (compound)
            "layerNorm", "rmsNorm", "instanceNorm",
            // Shape ops
            "broadcast", "transpose", "reshape", "flatten", "squeeze", "unsqueeze",
            "expand", "permute", "contiguous",
            // Indexing
            "select", "slice", "scatter", "gather", "cat",
            // Loss ops
            "softmaxCrossEntropy", "softmaxCrossEntropySparse", "bceLoss",
            "focalLoss", "diceLoss",
            // DL more
            "adaptiveAvgPool2d", "interpolate", "gridSample", "cross", "trapezoidalScan",
            // Fused (forward + backward)
            "squareSum", "expSum", "powSum", "mulSum",
            "squareMean", "absSum", "absMean",
            "reluSum", "reluMean", "logSum", "logMean",
            "sigmoidSum", "sigmoidMean", "tanhSum", "tanhMean",
            "siluSum", "siluMean", "mishSum", "mishMean",
            "expMean", "powMean", "mulMean",
            "geluSum", "geluMean", "sinSum", "sinMean", "cosSum", "cosMean",
            "leakyReluSum", "leakyReluMean", "eluSum", "eluMean",
            "seluSum", "seluMean", "softplusSum", "softplusMean",
            "hardtanhSum", "hardtanhMean",
            // Meta
            "leaf", "constant"
        )));

    // ========================================================================
    // Tests
    // ========================================================================

    @Test
    void testGpuSchemaSubsetOfRustGpu() {
        Set<String> javaGpuOps = new HashSet<>(GraphOpSchema.Gpu.BASE);
        Set<String> rustGpuCapable = new HashSet<>(RUST_GPU_FORWARD_BACKWARD);

        Set<String> declaredButNotImplemented = new HashSet<>(javaGpuOps);
        declaredButNotImplemented.removeAll(rustGpuCapable);

        if (!declaredButNotImplemented.isEmpty()) {
            fail(String.format(
                "GPU Schema BASE declares %d ops NOT implemented in Rust GPU (wgpu): %s%n"
                + "Either remove them from GraphOpSchema.Gpu.BASE or implement in yishape_math_gpu/src/ops/graph.rs.",
                declaredButNotImplemented.size(), declaredButNotImplemented));
        }
    }

    @Test
    void testHpcSchemaSubsetOfRustHpc() {
        Set<String> javaHpcOps = new HashSet<>(GraphOpSchema.Hpc.BASE);
        Set<String> rustHpcCapable = new HashSet<>(RUST_HPC_FORWARD_BACKWARD);

        Set<String> declaredButNotImplemented = new HashSet<>(javaHpcOps);
        declaredButNotImplemented.removeAll(rustHpcCapable);

        if (!declaredButNotImplemented.isEmpty()) {
            fail(String.format(
                "HPC Schema BASE declares %d ops NOT implemented in Rust HPC (faer): %s%n"
                + "Either remove them from GraphOpSchema.Hpc.BASE or implement in yishape_math_rust/src/graph.rs.",
                declaredButNotImplemented.size(), declaredButNotImplemented));
        }
    }

    @Test
    void testRustGpuOpsNotForgottenInJava() {
        // Use SUPPORTED (BASE + fused ops from FusedTagRegistry) as the full Java surface
        Set<String> javaGpuOps = new HashSet<>(GraphOpSchema.Gpu.SUPPORTED);
        Set<String> rustGpuCapable = new HashSet<>(RUST_GPU_FORWARD_BACKWARD);

        Set<String> implementedButNotDeclared = new HashSet<>(rustGpuCapable);
        implementedButNotDeclared.removeAll(javaGpuOps);
        // Meta ops are always implicit
        implementedButNotDeclared.remove("leaf");
        implementedButNotDeclared.remove("constant");

        if (!implementedButNotDeclared.isEmpty()) {
            fail(String.format(
                "Rust GPU implements %d ops NOT in GraphOpSchema.Gpu.SUPPORTED (BASE + fused): %s%n"
                + "Add to GraphOpSchema.Gpu.BASE or FusedTagRegistry.GPU_ALL, or update this curated set.",
                implementedButNotDeclared.size(), implementedButNotDeclared));
        }
    }

    @Test
    void testRustHpcOpsNotForgottenInJava() {
        // Use SUPPORTED (BASE + fused ops from FusedTagRegistry) as the full Java surface
        Set<String> javaHpcOps = new HashSet<>(GraphOpSchema.Hpc.SUPPORTED);
        Set<String> rustHpcCapable = new HashSet<>(RUST_HPC_FORWARD_BACKWARD);

        Set<String> implementedButNotDeclared = new HashSet<>(rustHpcCapable);
        implementedButNotDeclared.removeAll(javaHpcOps);
        // Meta ops are always implicit
        implementedButNotDeclared.remove("leaf");
        implementedButNotDeclared.remove("constant");

        if (!implementedButNotDeclared.isEmpty()) {
            fail(String.format(
                "Rust HPC implements %d ops NOT in GraphOpSchema.Hpc.SUPPORTED (BASE + fused): %s%n"
                + "Add to GraphOpSchema.Hpc.BASE or FusedTagRegistry.HPC_ALL, or update this curated set.",
                implementedButNotDeclared.size(), implementedButNotDeclared));
        }
    }

    @Test
    void testGpuNotDeclaringForwardOnlyOps() {
        Set<String> javaGpuOps = new HashSet<>(GraphOpSchema.Gpu.BASE);

        for (String op : RUST_GPU_FORWARD_ONLY) {
            if (javaGpuOps.contains(op)) {
                fail(String.format(
                    "GPU Schema declares '%s' but Rust GPU only has forward (no backward shader).%n"
                    + "Remove from GraphOpSchema.Gpu.BASE or implement backward in Rust GPU.",
                    op));
            }
        }
    }

    @Test
    void testBatchNormGpuDivergence() {
        // batchNorm is in GPU schema but Rust GPU only implements batchNorm2d
        Set<String> javaGpuOps = new HashSet<>(GraphOpSchema.Gpu.BASE);
        assertFalse(javaGpuOps.contains("batchNorm"),
            "batchNorm (1D) should NOT be in GPU SUPPORTED — Rust GPU only has batchNorm2d. "
            + "Remove 'batchNorm' from GraphOpSchema.Gpu.BASE.");
    }

    @Test
    void testBatchNormHpcCorrect() {
        // batchNorm IS implemented in HPC Rust
        Set<String> javaHpcOps = new HashSet<>(GraphOpSchema.Hpc.BASE);
        assertTrue(javaHpcOps.contains("batchNorm"),
            "batchNorm should be in HPC SUPPORTED — Rust HPC implements it.");
    }

    @Test
    void testRsubRdivInGpu() {
        // Verify that rsubScalar and rdivScalar ARE now in GPU (they were added recently)
        Set<String> javaGpuOps = new HashSet<>(GraphOpSchema.Gpu.BASE);
        assertTrue(javaGpuOps.contains("rsubScalar"),
            "rsubScalar should be in GPU SUPPORTED (Rust GPU graph.rs L547)");
        assertTrue(javaGpuOps.contains("rdivScalar"),
            "rdivScalar should be in GPU SUPPORTED (Rust GPU graph.rs L551)");
    }

    @Test
    void testReciprocalNotInGpu() {
        // reciprocal is HPC-only, correctly excluded from GPU
        Set<String> javaGpuOps = new HashSet<>(GraphOpSchema.Gpu.BASE);
        assertFalse(javaGpuOps.contains("reciprocal"),
            "reciprocal should NOT be in GPU SUPPORTED — only HPC implements it.");
    }

    @Test
    void testScatterNotInGpu() {
        // scatter is HPC-only, correctly excluded from GPU
        Set<String> javaGpuOps = new HashSet<>(GraphOpSchema.Gpu.BASE);
        assertFalse(javaGpuOps.contains("scatter"),
            "scatter should NOT be in GPU SUPPORTED — only HPC implements it.");
    }
}
