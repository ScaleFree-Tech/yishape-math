package com.yishape.lab.math.autodiff.graph;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Schema conformance test: validates GraphOpSchema internal consistency.
 *
 * <p>Checks that the GPU and HPC op sets are well-defined, FusedTagRegistry
 * entries are valid, and compound specials are documented. Runs without any
 * native code — pure metadata validation.</p>
 */
public class SchemaConformanceTest {

    // ═══════════════════════════════════════════════════════════════
    // Op set consistency
    // ═══════════════════════════════════════════════════════════════

    @Test void testGpuOps_nonEmpty() {
        Set<String> gpu = GraphOpSchema.Gpu.BASE;
        assertNotNull(gpu, "Gpu.BASE op set must not be null");
        assertFalse(gpu.isEmpty(), "Gpu.BASE op set must not be empty");
    }

    @Test void testHpcOps_nonEmpty() {
        Set<String> hpc = GraphOpSchema.Hpc.BASE;
        assertNotNull(hpc, "Hpc.BASE op set must not be null");
        assertFalse(hpc.isEmpty(), "Hpc.BASE op set must not be empty");
    }

    @Test void testOpNames_wellFormed() {
        // All op names should be lowercase alphanumeric with underscores
        for (String op : GraphOpSchema.Gpu.BASE) {
            assertTrue(op.matches("[a-zA-Z][a-zA-Z0-9_]*"),
                () -> "GPU op '" + op + "' has invalid format");
        }
        for (String op : GraphOpSchema.Hpc.BASE) {
            assertTrue(op.matches("[a-zA-Z][a-zA-Z0-9_]*"),
                () -> "HPC op '" + op + "' has invalid format");
        }
    }

    @Test void testGpuAndHpc_overlap() {
        // Compute shared ops between GPU and HPC
        Set<String> gpu = new HashSet<>(GraphOpSchema.Gpu.BASE);
        Set<String> hpc = new HashSet<>(GraphOpSchema.Hpc.BASE);

        Set<String> shared = new HashSet<>(gpu);
        shared.retainAll(hpc);

        // Most common ops should be in both
        assertTrue(shared.contains("add"), "add should be in both GPU and HPC");
        assertTrue(shared.contains("mul"), "mul should be in both GPU and HPC");
        assertTrue(shared.contains("relu"), "relu should be in both GPU and HPC");
        assertTrue(shared.contains("sum"), "sum should be in both GPU and HPC");
        assertTrue(shared.contains("matmul"), "matmul should be in both GPU and HPC");

        // Log unique ops for documentation
        Set<String> gpuOnly = new HashSet<>(gpu);
        gpuOnly.removeAll(hpc);
        Set<String> hpcOnly = new HashSet<>(hpc);
        hpcOnly.removeAll(gpu);

        System.out.println("[Schema] Shared ops: " + shared.size());
        if (!gpuOnly.isEmpty()) System.out.println("[Schema] GPU-only: " + gpuOnly);
        if (!hpcOnly.isEmpty()) System.out.println("[Schema] HPC-only: " + hpcOnly);
    }

    @Test void testGpuTensor_allIncludesFused() {
        // GPU_TENSOR.ALL should be a superset of BASE
        Set<String> all = GraphOpSchema.Gpu.SUPPORTED;
        Set<String> base = GraphOpSchema.Gpu.BASE;
        assertTrue(all.containsAll(base),
            () -> "GPU_TENSOR.ALL missing " + diff(base, all) + " base ops");
        // Should also include fused tags
        assertTrue(all.containsAll(GraphOpSchema.FusedTagRegistry.GPU_ALL),
            "GPU_TENSOR.ALL should include all GPU fused tags");
    }

    @Test void testHpcTensor_allIncludesFused() {
        Set<String> all = GraphOpSchema.Hpc.SUPPORTED;
        Set<String> base = GraphOpSchema.Hpc.BASE;
        assertTrue(all.containsAll(base),
            () -> "HPC_TENSOR.ALL missing " + diff(base, all) + " base ops");
        assertTrue(all.containsAll(GraphOpSchema.FusedTagRegistry.HPC_ALL),
            "HPC_TENSOR.ALL should include all HPC fused tags");
    }

    // ═══════════════════════════════════════════════════════════════
    // Fused tag registry
    // ═══════════════════════════════════════════════════════════════

    @Test void testFusedTagRegistry_allPattern_nonEmpty() {
        Set<String> all = GraphOpSchema.FusedTagRegistry.ALL_PATTERN;
        assertNotNull(all, "ALL_PATTERN must not be null");
        assertFalse(all.isEmpty(), "ALL_PATTERN must not be empty");
    }

    @Test void testFusedTagRegistry_gpuSubsetOfAll() {
        assertTrue(GraphOpSchema.FusedTagRegistry.ALL_PATTERN
                .containsAll(GraphOpSchema.FusedTagRegistry.GPU_PATTERN),
            "GPU_PATTERN should be a subset of ALL_PATTERN");
    }

    @Test void testFusedTagRegistry_hpcSubsetOfAll() {
        assertTrue(GraphOpSchema.FusedTagRegistry.ALL_PATTERN
                .containsAll(GraphOpSchema.FusedTagRegistry.HPC_PATTERN),
            "HPC_PATTERN should be a subset of ALL_PATTERN");
    }

    @Test void testFusedTagRegistry_gpuAllConsistent() {
        Set<String> gpuAll = GraphOpSchema.FusedTagRegistry.GPU_ALL;
        assertTrue(gpuAll.containsAll(GraphOpSchema.FusedTagRegistry.GPU_PATTERN),
            "GPU_ALL should contain GPU_PATTERN");
        assertTrue(gpuAll.containsAll(GraphOpSchema.FusedTagRegistry.GPU_COMPOUND),
            "GPU_ALL should contain GPU_COMPOUND");
    }

    @Test void testFusedTagRegistry_hpcAllConsistent() {
        Set<String> hpcAll = GraphOpSchema.FusedTagRegistry.HPC_ALL;
        assertTrue(hpcAll.containsAll(GraphOpSchema.FusedTagRegistry.HPC_PATTERN),
            "HPC_ALL should contain HPC_PATTERN");
        assertTrue(hpcAll.containsAll(GraphOpSchema.FusedTagRegistry.HPC_COMPOUND),
            "HPC_ALL should contain HPC_COMPOUND");
    }

    @Test void testFusedTagRegistry_compoundSpecials_describable() {
        // Each compound special should be describable via describeTag
        Set<String> gpuCompounds = GraphOpSchema.FusedTagRegistry.GPU_COMPOUND;
        Set<String> hpcCompounds = GraphOpSchema.FusedTagRegistry.HPC_COMPOUND;

        Set<String> allCompounds = new HashSet<>();
        allCompounds.addAll(gpuCompounds);
        allCompounds.addAll(hpcCompounds);

        System.out.println("[Schema] GPU compounds: " + gpuCompounds);
        System.out.println("[Schema] HPC compounds: " + hpcCompounds);
        System.out.println("[Schema] All compounds: " + allCompounds.size());

        for (String tag : allCompounds) {
            String desc = GraphOpSchema.describeTag(tag);
            assertNotNull(desc, () -> "Compound '" + tag + "' has null description");
            assertFalse(desc.isEmpty(),
                () -> "Compound '" + tag + "' has empty description");
        }
    }

    @Test void testFusedTagRegistry_noCaseInsensitiveDuplicates() {
        // GPU and HPC pattern + compound tags should not have case-insensitive duplicates
        Set<String> seen = new HashSet<>();
        for (String tag : GraphOpSchema.FusedTagRegistry.GPU_ALL) {
            assertTrue(seen.add(tag.toLowerCase()),
                () -> "Duplicate case-insensitive GPU fused tag: " + tag);
        }
        seen.clear();
        for (String tag : GraphOpSchema.FusedTagRegistry.HPC_ALL) {
            assertTrue(seen.add(tag.toLowerCase()),
                () -> "Duplicate case-insensitive HPC fused tag: " + tag);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Cross-reference: describeTag coverage
    // ═══════════════════════════════════════════════════════════════

    @Test void testDescribeTag_knownOps_returnNonEmpty() {
        // describeTag should return non-empty for known ops
        for (String op : Arrays.asList("add", "mul", "relu", "sum", "matmul",
                "softmax", "conv2d", "linear", "pow", "exp", "tanh", "sigmoid")) {
            String desc = GraphOpSchema.describeTag(op);
            assertNotNull(desc, () -> "describeTag('" + op + "') returned null");
            assertFalse(desc.isEmpty(),
                () -> "describeTag('" + op + "') returned empty");
        }
    }

    @Test void testDescribeTag_unknownOp_returnsGracefully() {
        // describeTag for unknown ops should not throw
        String desc = GraphOpSchema.describeTag("nonexistent_op_xyz");
        // May return null or "unknown" — either is acceptable
        if (desc != null) {
            System.out.println("[Schema] Unknown op description: " + desc);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Log summary
    // ═══════════════════════════════════════════════════════════════

    @Test void testLogSummary() {
        System.out.println("[Schema] GPU base ops: " + GraphOpSchema.Gpu.BASE.size());
        System.out.println("[Schema] HPC base ops: " + GraphOpSchema.Hpc.BASE.size());
        System.out.println("[Schema] GPU fused all: " + GraphOpSchema.FusedTagRegistry.GPU_ALL.size());
        System.out.println("[Schema] HPC fused all: " + GraphOpSchema.FusedTagRegistry.HPC_ALL.size());
        System.out.println("[Schema] GPU tensor all: " + GraphOpSchema.Gpu.SUPPORTED.size());
        System.out.println("[Schema] HPC tensor all: " + GraphOpSchema.Hpc.SUPPORTED.size());
        System.out.println("[Schema] Fused pattern all: " + GraphOpSchema.FusedTagRegistry.ALL_PATTERN.size());
    }

    private static Set<String> diff(Set<String> subset, Set<String> superset) {
        Set<String> d = new HashSet<>(subset);
        d.removeAll(superset);
        return d;
    }
}
