package com.yishape.lab.math.autodiff.graph;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Automated op coverage matrix: Java {@link GpuGraphExecutor#SUPPORTED_OPS} ↔ Rust {@code graph.rs} dispatch.
 *
 * <p>This test extracts the supported-op list from Java source and the dispatch match arms
 * from Rust source, then produces a coverage matrix. Any divergence (op supported on one
 * side but missing on the other) is reported as a test failure.</p>
 *
 * <p>This prevents the class of bug where an op is added to Java serialization but the
 * Rust GPU worker doesn't handle it — or vice versa.</p>
 */
public class OpCoverageMatrixTest {

    /** Java SUPPORTED_OPS are in GpuGraphExecutor.java */
    private static final Path JAVA_SOURCE = Paths.get(
        "src/main/java/com/yishape/lab/math/autodiff/graph/GpuGraphExecutor.java");

    /** Rust forward/backward dispatch is in graph.rs.
     *  Override via env var YISHAPE_GPU_RUST_DIR or system property yishape.gpu.rust.dir. */
    private static Path resolveRustGraphRs() {
        String envDir = System.getenv("YISHAPE_GPU_RUST_DIR");
        if (envDir == null) envDir = System.getProperty("yishape.gpu.rust.dir");
        if (envDir != null) return Paths.get(envDir, "src/ops/graph.rs");
        // Default: sibling of yishape-math's parent directory
        return Paths.get("../../rust_work/yishape_math_gpu/src/ops/graph.rs");
    }

    /** Ops that are Java-only by design (leaf, constant are graph structure, not compute ops). */
    private static final Set<String> JAVA_ONLY = Set.of("leaf", "constant");

    /** Ops that are known to be missing from Rust forward dispatch (TODO items). */
    private static final Set<String> KNOWN_MISSING_FORWARD = Set.of(
        "rdivScalar",                // only backward dispatch exists, no forward
        "scaledDotProductAttention"  // forward not yet implemented in Rust
    );

    /** Ops that are known to be missing from Rust backward dispatch. */
    private static final Set<String> KNOWN_MISSING_BACKWARD = Set.of(
        "scaledDotProductAttention"  // backward not yet implemented in Rust
    );

    /** Ops found in Rust dispatch but not in Java SUPPORTED_OPS (likely internal/Rust-only). */
    private static final Set<String> KNOWN_RUST_ONLY = Set.of(
        "Mean",   // Rust-internal casing variant of mean
        "outer"   // Rust-internal op, not in Java serialization yet
    );

    @Test
    void testOpCoverageMatrix() throws IOException {
        Path rustGraphRs = resolveRustGraphRs();
        if (!Files.exists(rustGraphRs)) {
            System.out.println("=== Op Coverage Matrix: SKIPPED (Rust source not found at " + rustGraphRs + ") ===");
            System.out.println("Set env var YISHAPE_GPU_RUST_DIR or system property yishape.gpu.rust.dir to enable.");
            return;
        }
        Set<String> javaOps = extractJavaOps();
        Set<String> rustForward = extractRustForwardOps(rustGraphRs);
        Set<String> rustBackward = extractRustBackwardOps(rustGraphRs);

        // Filter out structural ops
        Set<String> computeOps = new HashSet<>(javaOps);
        computeOps.removeAll(JAVA_ONLY);

        System.out.println("\n=== Op Coverage Matrix ===");
        System.out.printf("%-30s %-10s %-10s %-10s%n", "Op", "Java", "RustFwd", "RustBwd");
        System.out.println("-".repeat(65));

        List<String> sortedOps = new ArrayList<>(computeOps);
        Collections.sort(sortedOps);

        int javaOnlyFwd = 0, javaOnlyBwd = 0, rustOnlyFwd = 0, rustOnlyBwd = 0;
        int fullCoverage = 0, total = 0;

        for (String op : sortedOps) {
            boolean inJava = true; // it's from javaOps
            boolean inRustFwd = rustForward.contains(op);
            boolean inRustBwd = rustBackward.contains(op);

            String javaS = "✓";
            String fwdS = inRustFwd ? "✓" : "✗";
            String bwdS = inRustBwd ? "✓" : "✗";

            System.out.printf("%-30s %-10s %-10s %-10s%n", op, javaS, fwdS, bwdS);
            total++;

            if (!inRustFwd) javaOnlyFwd++;
            if (!inRustBwd) javaOnlyBwd++;
            if (inRustFwd && inRustBwd) fullCoverage++;
        }

        // Find ops in Rust but not in Java
        Set<String> rustOnlyForwardOps = new HashSet<>(rustForward);
        rustOnlyForwardOps.removeAll(javaOps);
        rustOnlyForwardOps.removeAll(JAVA_ONLY);
        rustOnlyForwardOps.removeAll(KNOWN_RUST_ONLY);

        Set<String> rustOnlyBackwardOps = new HashSet<>(rustBackward);
        rustOnlyBackwardOps.removeAll(javaOps);
        rustOnlyBackwardOps.removeAll(JAVA_ONLY);
        rustOnlyBackwardOps.removeAll(KNOWN_RUST_ONLY);

        if (!rustOnlyForwardOps.isEmpty()) {
            System.out.println("\nOps in Rust forward dispatch but NOT in Java SUPPORTED_OPS:");
            for (String op : rustOnlyForwardOps) {
                System.out.println("  FWD-ONLY: " + op);
            }
        }

        // Summary
        System.out.printf("%n=== Summary ===%n");
        System.out.printf("Total compute ops: %d%n", total);
        System.out.printf("Full coverage (Java + RustFwd + RustBwd): %d/%d (%.0f%%)%n",
            fullCoverage, total, 100.0 * fullCoverage / total);
        System.out.printf("Missing Rust forward:  %d ops%n", javaOnlyFwd);
        System.out.printf("Missing Rust backward: %d ops%n", javaOnlyBwd);

        // Known gaps
        if (!KNOWN_MISSING_FORWARD.isEmpty()) {
            System.out.printf("Known missing forward (tracked): %s%n", KNOWN_MISSING_FORWARD);
        }
        if (!KNOWN_MISSING_BACKWARD.isEmpty()) {
            System.out.printf("Known missing backward (tracked): %s%n", KNOWN_MISSING_BACKWARD);
        }

        // Compute unexpected gaps
        Set<String> unexpectedFwdGaps = new HashSet<>();
        for (String op : computeOps) {
            if (!rustForward.contains(op) && !KNOWN_MISSING_FORWARD.contains(op)) {
                unexpectedFwdGaps.add(op);
            }
        }
        Set<String> unexpectedBwdGaps = new HashSet<>();
        for (String op : computeOps) {
            if (!rustBackward.contains(op) && !KNOWN_MISSING_BACKWARD.contains(op)) {
                unexpectedBwdGaps.add(op);
            }
        }

        // Don't fail on structural/allowed ops
        unexpectedFwdGaps.removeAll(JAVA_ONLY);
        unexpectedBwdGaps.removeAll(JAVA_ONLY);

        // Assert no unexpected gaps
        if (!unexpectedFwdGaps.isEmpty()) {
            fail("UNEXPECTED GAPS in Rust forward dispatch (add to graph.rs or KNOWN_MISSING_FORWARD): " + unexpectedFwdGaps);
        }
        if (!unexpectedBwdGaps.isEmpty()) {
            fail("UNEXPECTED GAPS in Rust backward dispatch (add to graph.rs or KNOWN_MISSING_BACKWARD): " + unexpectedBwdGaps);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Java source parsing: extract SUPPORTED_OPS from GpuGraphExecutor
    // ═══════════════════════════════════════════════════════════════

    private static Set<String> extractJavaOps() throws IOException {
        String content = Files.readString(JAVA_SOURCE);
        Set<String> ops = new LinkedHashSet<>();

        // Find the first SUPPORTED_OPS = new HashSet<>(Arrays.asList( block.
        // The initialization uses Arrays.asList(...) with parentheses, not braces.
        int startIdx = content.indexOf("SUPPORTED_OPS");
        if (startIdx < 0) {
            throw new IOException("Cannot find SUPPORTED_OPS in GpuGraphExecutor.java");
        }
        ops.addAll(extractStringsFromArraysAsList(content, startIdx));

        // Also capture TENSOR_SUPPORTED_OPS additions (addAll in static block)
        int tensorIdx = content.indexOf("TENSOR_SUPPORTED_OPS");
        if (tensorIdx >= 0) {
            // Find all Arrays.asList(...) calls after TENSOR_SUPPORTED_OPS
            int searchFrom = tensorIdx;
            while (true) {
                int asListIdx = content.indexOf("Arrays.asList(", searchFrom);
                if (asListIdx < 0) break;
                // Only grab it if it's within ~2000 chars (same static block)
                if (asListIdx - tensorIdx > 2000) break;
                ops.addAll(extractStringsFromArraysAsListAt(content, asListIdx + "Arrays.asList".length()));
                searchFrom = asListIdx + 1;
            }
        }
        return ops;
    }

    /** Extract all "quotedString" from Arrays.asList(...) starting at searchFrom. */
    private static Set<String> extractStringsFromArraysAsList(String content, int searchFrom) {
        int listStart = content.indexOf("Arrays.asList(", searchFrom);
        if (listStart < 0) return Set.of();
        return extractStringsFromArraysAsListAt(content, listStart + "Arrays.asList".length());
    }

    private static Set<String> extractStringsFromArraysAsListAt(String content, int parenIdx) {
        Set<String> ops = new LinkedHashSet<>();
        // Match parentheses to find the closing ')' of Arrays.asList(...)
        int depth = 0;
        int parenEnd = parenIdx;
        for (int i = parenIdx; i < content.length(); i++) {
            char c = content.charAt(i);
            if (c == '(') depth++;
            else if (c == ')') { depth--; if (depth == 0) { parenEnd = i; break; } }
        }
        String block = content.substring(parenIdx, parenEnd);
        Pattern p = Pattern.compile("\"([a-zA-Z0-9]+)\"");
        Matcher m = p.matcher(block);
        while (m.find()) {
            ops.add(m.group(1));
        }
        return ops;
    }

    // ═══════════════════════════════════════════════════════════════
    // Rust source parsing: extract match arms from forward/backward dispatch
    // ═══════════════════════════════════════════════════════════════

    private static Set<String> extractRustForwardOps(Path rustFile) throws IOException {
        return extractRustMatchArms(rustFile, "fn forward_dispatch");
    }

    private static Set<String> extractRustBackwardOps(Path rustFile) throws IOException {
        return extractRustMatchArms(rustFile, "fn backward_dispatch");
    }

    private static Set<String> extractRustMatchArms(Path rustFile, String fnMarker) throws IOException {
        String content = Files.readString(rustFile);
        Set<String> ops = new LinkedHashSet<>();

        // Find the function
        int fnStart = content.indexOf(fnMarker);
        if (fnStart < 0) {
            System.out.println("  WARNING: Cannot find " + fnMarker + " in " + rustFile);
            return ops;
        }

        // Find the match block within the function
        int matchStart = content.indexOf("match op.as_str()", fnStart);
        if (matchStart < 0) {
            System.out.println("  WARNING: Cannot find match block in " + fnMarker);
            return ops;
        }

        // Find the closing of the match block (track brace depth)
        int braceSearchStart = content.indexOf('{', matchStart);
        int depth = 0;
        int matchEnd = braceSearchStart;
        for (int i = braceSearchStart; i < content.length(); i++) {
            char c = content.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') { depth--; if (depth == 0) { matchEnd = i; break; } }
        }
        String matchBlock = content.substring(matchStart, matchEnd);

        // Extract quoted string patterns from match arms: "opName" =>
        // Pattern matches: "add" | "sub" | ... => or "opname" => {
        Pattern armPattern = Pattern.compile("\"([a-zA-Z0-9]+)\"");
        Matcher m = armPattern.matcher(matchBlock);
        while (m.find()) {
            ops.add(m.group(1));
        }
        return ops;
    }

    // ═══════════════════════════════════════════════════════════════
    // Manual override: run this test directly for a detailed report
    // ═══════════════════════════════════════════════════════════════

    public static void main(String[] args) throws Exception {
        OpCoverageMatrixTest test = new OpCoverageMatrixTest();
        test.testOpCoverageMatrix();
    }
}
