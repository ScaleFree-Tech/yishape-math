package com.yishape.lab.math.autodiff.graph;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;

/**
 * Validates that the {@link GraphOpSchema.FusedTagRegistry} and
 * {@link GraphOpSchema.FusedTag} naming convention is internally
 * consistent and matches the GPU/HPC SUPPORTED sets.
 */
class FusedTagRegistryConsistencyTest {

    // ── 1. FusedTag.of() round-trip ──

    @Test
    void of_roundTripsForAllKnownPatterns() {
        for (String utag : GraphOpSchema.FusedTag.UNARY_TAGS) {
            for (String rtag : GraphOpSchema.FusedTag.REDUCE_TAGS) {
                String tag = GraphOpSchema.FusedTag.of(utag, rtag);
                // tag must appear in GPU_PATTERN or HPC_PATTERN (or documented as unsupported)
                boolean inGpu = GraphOpSchema.FusedTagRegistry.GPU_PATTERN.contains(tag);
                boolean inHpc = GraphOpSchema.FusedTagRegistry.HPC_PATTERN.contains(tag);
                assertTrue(inGpu || inHpc,
                    "Tag '" + tag + "' generated from UNARY=" + utag + " REDUCE=" + rtag +
                    " is not in GPU_PATTERN or HPC_PATTERN. " +
                    "Either implement it or document as unsupported.");
            }
        }
    }

    // ── 2. GPU_PATTERN membership validation ──

    @Test
    void gpuPattern_allEntriesValidPattern() {
        for (String tag : GraphOpSchema.FusedTagRegistry.GPU_PATTERN) {
            assertTrue(isValidPatternTag(tag),
                "GPU_PATTERN entry '" + tag + "' does not match {unary}{Reduce} pattern. " +
                "Add to COMPOUND_SPECIALS or fix the tag name.");
        }
    }

    // ── 3. HPC_PATTERN membership validation ──

    @Test
    void hpcPattern_allEntriesValidPattern() {
        for (String tag : GraphOpSchema.FusedTagRegistry.HPC_PATTERN) {
            assertTrue(isValidPatternTag(tag),
                "HPC_PATTERN entry '" + tag + "' does not match {unary}{Reduce} pattern. " +
                "Add to COMPOUND_SPECIALS or fix the tag name.");
        }
    }

    // ── 4. SUPPORTED ∋ BASE ∪ FUSED ──

    @Test
    void gpuSupported_containsBaseAndFused() {
        Set<String> expected = new HashSet<>(GraphOpSchema.Gpu.BASE);
        expected.addAll(GraphOpSchema.FusedTagRegistry.GPU_ALL);
        assertEquals(expected, GraphOpSchema.Gpu.SUPPORTED,
            "GPU SUPPORTED must equal BASE ∪ GPU_ALL_FUSED");
    }

    @Test
    void hpcSupported_containsBaseAndFused() {
        Set<String> expected = new HashSet<>(GraphOpSchema.Hpc.BASE);
        expected.addAll(GraphOpSchema.FusedTagRegistry.HPC_ALL);
        assertEquals(expected, GraphOpSchema.Hpc.SUPPORTED,
            "HPC SUPPORTED must equal BASE ∪ HPC_ALL_FUSED");
    }

    // ── 5. No overlap between BASE and FUSED ──

    @Test
    void gpu_baseAndFused_disjoint() {
        Set<String> intersection = new HashSet<>(GraphOpSchema.Gpu.BASE);
        intersection.retainAll(GraphOpSchema.FusedTagRegistry.GPU_ALL);
        assertTrue(intersection.isEmpty(),
            "GPU BASE and GPU_ALL_FUSED must be disjoint. Overlap: " + intersection);
    }

    @Test
    void hpc_baseAndFused_disjoint() {
        Set<String> intersection = new HashSet<>(GraphOpSchema.Hpc.BASE);
        intersection.retainAll(GraphOpSchema.FusedTagRegistry.HPC_ALL);
        assertTrue(intersection.isEmpty(),
            "HPC BASE and HPC_ALL_FUSED must be disjoint. Overlap: " + intersection);
    }

    // ── 6. ofChain idempotent with single-op chain ──

    @Test
    void ofChain_singleOpEqualsOf() {
        for (String utag : GraphOpSchema.FusedTag.UNARY_TAGS) {
            for (String rtag : GraphOpSchema.FusedTag.REDUCE_TAGS) {
                String fromOf = GraphOpSchema.FusedTag.of(utag, rtag);
                String fromChain = GraphOpSchema.FusedTag.ofChain(
                    java.util.List.of(utag), rtag);
                assertEquals(fromOf, fromChain,
                    "of(\"" + utag + "\", \"" + rtag + "\") must equal ofChain([\"" + utag + "\"], \"" + rtag + "\")");
            }
        }
    }

    // ── 7. GPU/HPC compound entries are in COMPOUND_SPECIALS ──

    @Test
    void gpuCompound_areKnownSpecials() {
        for (String tag : GraphOpSchema.FusedTagRegistry.GPU_COMPOUND) {
            assertTrue(GraphOpSchema.FusedTag.COMPOUND_SPECIALS.contains(tag),
                "GPU_COMPOUND entry '" + tag + "' not in COMPOUND_SPECIALS");
        }
    }

    @Test
    void hpcCompound_areKnownSpecials() {
        for (String tag : GraphOpSchema.FusedTagRegistry.HPC_COMPOUND) {
            assertTrue(GraphOpSchema.FusedTag.COMPOUND_SPECIALS.contains(tag),
                "HPC_COMPOUND entry '" + tag + "' not in COMPOUND_SPECIALS");
        }
    }

    // ── 8. ALL_PATTERN has correct size ──

    @Test
    void allPattern_hasCorrectCardinality() {
        int expected = GraphOpSchema.FusedTag.UNARY_TAGS.size()
                     * GraphOpSchema.FusedTag.REDUCE_TAGS.size();
        assertEquals(expected, GraphOpSchema.FusedTagRegistry.ALL_PATTERN.size(),
            "ALL_PATTERN = |UNARY_TAGS| × |REDUCE_TAGS| = " + expected);
    }

    // ── helpers ──

    /** Checks if tag looks like a {unary}{Reduce} pattern, e.g. "squareMean" or "leakyReluSum". */
    private static boolean isValidPatternTag(String tag) {
        if (tag == null || tag.isEmpty()) return false;
        // Try every uppercase position as the reduce boundary.
        // For multi-word unary tags like "leakyReluSum", the unary part "leakyRelu"
        // already contains uppercase, so we try all split points.
        for (int i = 1; i < tag.length(); i++) {
            if (!Character.isUpperCase(tag.charAt(i))) continue;
            String unary = tag.substring(0, i);
            String reduce = Character.toLowerCase(tag.charAt(i)) + tag.substring(i + 1);
            if (GraphOpSchema.FusedTag.UNARY_TAGS.contains(unary)
                && GraphOpSchema.FusedTag.REDUCE_TAGS.contains(reduce)
                && tag.equals(GraphOpSchema.FusedTag.of(unary, reduce))) {
                return true;
            }
        }
        return false;
    }
}
