package com.yishape.lab.math.codegen;

import static org.junit.jupiter.api.Assertions.*;

import com.yishape.lab.math.autodiff.graph.GraphOpSchema;
import org.junit.jupiter.api.Test;
import java.util.*;

/**
 * Verifies that OpRegistry (the single source of truth) is consistent with the
 * current manually-maintained lists in GraphOpSchema.
 *
 * <p>When OpRegistry is fully adopted, GraphOpSchema's lists will be generated
 * from OpRegistry, making this test the authority check.
 */
class OpRegistryConsistencyTest {

    @Test
    void gpuOps_matchGraphOpSchema() {
        Set<String> fromRegistry = collectTags(OpRegistry.gpuOps());
        // Add all fused tags derived from registry ops
        for (OpDefinition op : OpRegistry.fusionUnaryOps()) {
            fromRegistry.addAll(op.fusedTags());
        }
        // Add compound specials that are GPU-supported
        fromRegistry.addAll(GraphOpSchema.FusedTagRegistry.GPU_COMPOUND);

        Set<String> fromSchema = GraphOpSchema.Gpu.SUPPORTED;

        Set<String> missingFromSchema = new HashSet<>(fromRegistry);
        missingFromSchema.remove("leaf");
        missingFromSchema.remove("constant");
        missingFromSchema.removeAll(fromSchema);
        assertTrue(missingFromSchema.isEmpty(),
            "OpRegistry GPU ops not in GraphOpSchema.Gpu.SUPPORTED: " + missingFromSchema);

        Set<String> missingFromRegistry = new HashSet<>(fromSchema);
        missingFromRegistry.removeAll(fromRegistry);
        assertTrue(missingFromRegistry.isEmpty(),
            "GraphOpSchema.Gpu.SUPPORTED ops not in OpRegistry: " + missingFromRegistry);
    }

    @Test
    void hpcOps_matchGraphOpSchema() {
        Set<String> fromRegistry = collectTags(OpRegistry.hpcOps());
        // Add all fused tags derived from registry ops
        for (OpDefinition op : OpRegistry.fusionUnaryOps()) {
            fromRegistry.addAll(op.fusedTags());
        }
        // Add compound specials
        fromRegistry.addAll(GraphOpSchema.FusedTagRegistry.HPC_COMPOUND);

        Set<String> fromSchema = GraphOpSchema.Hpc.SUPPORTED;

        Set<String> missingFromSchema = new HashSet<>(fromRegistry);
        missingFromSchema.remove("leaf");
        missingFromSchema.remove("constant");
        missingFromSchema.removeAll(fromSchema);
        assertTrue(missingFromSchema.isEmpty(),
            "OpRegistry HPC ops not in GraphOpSchema.Hpc.SUPPORTED: " + missingFromSchema);

        Set<String> missingFromRegistry = new HashSet<>(fromSchema);
        missingFromRegistry.removeAll(fromRegistry);
        assertTrue(missingFromRegistry.isEmpty(),
            "GraphOpSchema.Hpc.SUPPORTED ops not in OpRegistry: " + missingFromRegistry);
    }

    @Test
    void fusionUnaryTags_matchFusedTagUnary() {
        Set<String> fromRegistry = new HashSet<>();
        for (OpDefinition op : OpRegistry.fusionUnaryOps()) {
            fromRegistry.add(op.tag());
        }
        Set<String> fromSchema = new HashSet<>(GraphOpSchema.FusedTag.UNARY_TAGS);

        Set<String> missingFromSchema = new HashSet<>(fromRegistry);
        missingFromSchema.removeAll(fromSchema);
        assertTrue(missingFromSchema.isEmpty(),
            "OpRegistry fusion unary tags not in FusedTag.UNARY_TAGS: " + missingFromSchema);

        Set<String> missingFromRegistry = new HashSet<>(fromSchema);
        missingFromRegistry.removeAll(fromRegistry);
        assertTrue(missingFromRegistry.isEmpty(),
            "FusedTag.UNARY_TAGS not in OpRegistry: " + missingFromRegistry);
    }

    @Test
    void noDuplicateTags() {
        Set<String> seen = new HashSet<>();
        List<String> dupes = new ArrayList<>();
        for (OpDefinition op : OpRegistry.ALL_OPS) {
            if (!seen.add(op.tag())) {
                dupes.add(op.tag());
            }
        }
        assertTrue(dupes.isEmpty(), "Duplicate op tags in OpRegistry: " + dupes);
    }

    @Test
    void describeTag_coverage() {
        // Only high-level compound ops have describeTag entries (not element-wise math ops).
        // These match the switch cases in GraphOpSchema.describeTag().
        Set<String> expectedTags = Set.of(
            "mha", "linear", "conv2d", "convTranspose2d", "batchNorm2d",
            "layerNorm", "rmsNorm", "lstmStep", "embedding",
            "maxpool2d", "depthwiseConv1d", "selectiveScan", "selectiveScan2",
            "trapezoidalScan", "bceLoss", "focalLoss", "diceLoss"
        );

        for (String tag : expectedTags) {
            String desc = GraphOpSchema.describeTag(tag);
            assertNotNull(desc);
            assertNotEquals(tag, desc,
                "describeTag falls through to default for: " + tag);
        }

        // Verify OpRegistry has these tags
        for (String tag : expectedTags) {
            assertTrue(OpRegistry.findByTag(tag).isPresent(),
                "OpRegistry missing describeTag-covered op: " + tag);
        }
    }

    private static Set<String> collectTags(List<OpDefinition> ops) {
        Set<String> tags = new LinkedHashSet<>();
        for (OpDefinition op : ops) {
            tags.add(op.tag());
        }
        return tags;
    }
}
