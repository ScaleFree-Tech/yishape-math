package com.yishape.lab.math.codegen;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Code generation tool for the unified op schema.
 *
 * <p>Reads the single source of truth ({@link OpRegistry#ALL_OPS}) and generates
 * sections of {@code GraphOpSchema.java} delimited by
 * {@code // BEGIN GENERATED:<section>} / {@code // END GENERATED:<section>} markers.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 *   java com.yishape.lab.math.codegen.CodegenTool
 * }</pre>
 *
 * <p>CI enforces: {@code git diff --exit-code} after codegen.
 */
public final class CodegenTool {

    private static final Path PROJECT_ROOT = Paths.get("").toAbsolutePath();
    private static final Path GRAPH_OP_SCHEMA = PROJECT_ROOT.resolve(
        "src/main/java/com/yishape/lab/math/autodiff/graph/GraphOpSchema.java");

    public static void main(String[] args) throws IOException {
        System.out.println("CodegenTool — generating from OpRegistry (" + OpRegistry.size() + " ops)");
        System.out.println("  GPU ops: " + OpRegistry.gpuOps().size());
        System.out.println("  HPC ops: " + OpRegistry.hpcOps().size());

        if (!Files.exists(GRAPH_OP_SCHEMA)) {
            System.out.println("  GraphOpSchema.java not found at: " + GRAPH_OP_SCHEMA);
            System.exit(1);
        }

        processGraphOpSchema();
        RustDispatchGenerator.generate();
        DocGenerator.generate();
        System.out.println("Done.");
    }

    private static void processGraphOpSchema() throws IOException {
        String content = Files.readString(GRAPH_OP_SCHEMA);
        String original = content;

        content = replaceSection(content, "describeTag", generateDescribeTag());
        content = replaceSection(content, "fusedTag", generateFusedTag());
        content = replaceSection(content, "fusedTagRegistry", generateFusedTagRegistry());
        content = replaceSection(content, "gpuOps", generateBackendOps(true));
        content = replaceSection(content, "hpcOps", generateBackendOps(false));

        if (!content.equals(original)) {
            Files.writeString(GRAPH_OP_SCHEMA, content);
            System.out.println("  GraphOpSchema.java: updated");
        } else {
            System.out.println("  GraphOpSchema.java: no changes needed");
        }
    }

    /**
     * Replaces content between {@code // BEGIN GENERATED:<name>} and
     * {@code // END GENERATED:<name>}.  Returns unchanged if markers absent.
     */
    private static String replaceSection(String content, String name, String newContent) {
        String begin = "// BEGIN GENERATED:" + name;
        String end = "// END GENERATED:" + name;
        int bi = content.indexOf(begin);
        int ei = content.indexOf(end);
        if (bi < 0 || ei < 0 || ei <= bi) {
            System.out.println("  Warning: section '" + name + "' markers not found");
            return content;
        }
        int insertStart = content.indexOf('\n', bi) + 1;
        int insertEnd = content.lastIndexOf('\n', ei);
        if (insertEnd < insertStart) insertEnd = ei;
        return content.substring(0, insertStart)
            + newContent
            + content.substring(insertEnd);
    }

    // ═══════════════════════════════════════════════════════════════
    // Generators
    // ═══════════════════════════════════════════════════════════════

    /** Generates the body of describeTag() — one case per described op. */
    private static String generateDescribeTag() {
        StringBuilder sb = new StringBuilder();
        for (OpDefinition op : OpRegistry.describedOps()) {
            sb.append("            case \"")
              .append(op.tag()).append("\" -> \"")
              .append(esc(op.description())).append("\";\n");
        }
        // Default is OUTSIDE the generated section in GraphOpSchema.java
        return sb.toString();
    }

    /** Generates the complete FusedTag inner class. */
    private static String generateFusedTag() {
        StringBuilder sb = new StringBuilder();
        sb.append("""
            /** Fused tag naming convention. All fused tags MUST use these methods. */
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
                        else sb.append(Character.toUpperCase(tag.charAt(0)))
                               .append(tag, 1, tag.length());
                    }
                    sb.append(Character.toUpperCase(reduceTag.charAt(0)));
                    if (reduceTag.length() > 1) sb.append(reduceTag, 1, reduceTag.length());
                    return sb.toString();
                }

            """);

        // UNARY_TAGS
        List<OpDefinition> fusionUnary = OpRegistry.fusionUnaryOps();
        sb.append("        /** ").append(fusionUnary.size())
          .append(" unary ops for {unary}{Reduce} fusion. */\n");
        sb.append("        public static final Set<String> UNARY_TAGS = ")
          .append("Collections.unmodifiableSet(new HashSet<>(Arrays.asList(\n");
        for (int i = 0; i < fusionUnary.size(); i++) {
            sb.append("            \"").append(fusionUnary.get(i).tag()).append('"');
            if (i < fusionUnary.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("        )));\n\n");

        // REDUCE_TAGS
        sb.append("        /** Reduction terminators. */\n");
        sb.append("        public static final Set<String> REDUCE_TAGS = Set.of(\"sum\", \"mean\");\n\n");

        // COMPOUND_SPECIALS
        List<OpDefinition> compound = OpRegistry.compoundSpecials();
        sb.append("        /** Compound specials (non-{unary}{Reduce} patterned). */\n");
        sb.append("        public static final Set<String> COMPOUND_SPECIALS = Set.of(\n");
        for (int i = 0; i < compound.size(); i++) {
            sb.append("            \"").append(compound.get(i).tag()).append('"');
            if (i < compound.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("        );\n");
        sb.append("    }\n");
        return sb.toString();
    }

    /** Generates the complete FusedTagRegistry inner class. */
    private static String generateFusedTagRegistry() {
        StringBuilder sb = new StringBuilder();
        sb.append("""
            /** Registry of which fused op tags have native implementations per backend. */
            public static final class FusedTagRegistry {

                /** All {unary}{Reduce} pattern tags (cartesian product). */
                public static final Set<String> ALL_PATTERN;
                static {
                    Set<String> all = new HashSet<>();
                    for (String u : FusedTag.UNARY_TAGS)
                        for (String r : FusedTag.REDUCE_TAGS)
                            all.add(FusedTag.of(u, r));
                    ALL_PATTERN = Collections.unmodifiableSet(all);
                }

                // ---- GPU ----

                /** {unary}{Reduce} tags with GPU WGSL implementations. */
                public static final Set<String> GPU_PATTERN = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            """);

        // GPU pattern: all fusion unary ops that have GPU support
        List<OpDefinition> fusionUnary = OpRegistry.fusionUnaryOps();
        List<String> gpuPattern = new ArrayList<>();
        for (OpDefinition op : fusionUnary) {
            if (op.gpu()) {
                gpuPattern.add(OpRegistry.fuseTag(op.tag(), "sum"));
                gpuPattern.add(OpRegistry.fuseTag(op.tag(), "mean"));
            }
        }
        for (int i = 0; i < gpuPattern.size(); i++) {
            sb.append("            \"").append(gpuPattern.get(i)).append('"');
            if (i < gpuPattern.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("        )));\n\n");

        // GPU compound
        Set<String> gpuCompound = OpRegistry.gpuCompoundSpecials();
        sb.append("        /** Compound specials with GPU implementations. */\n");
        sb.append("        public static final Set<String> GPU_COMPOUND = Set.of(");
        sb.append(gpuCompound.stream().map(t -> "\"" + t + "\"").collect(Collectors.joining(", ")));
        sb.append(");\n\n");

        sb.append("""
                /** All GPU-supported fused tags. */
                public static final Set<String> GPU_ALL;
                static {
                    Set<String> s = new HashSet<>(GPU_PATTERN);
                    s.addAll(GPU_COMPOUND);
                    GPU_ALL = Collections.unmodifiableSet(s);
                }

                // ---- HPC ----

                /** {unary}{Reduce} tags with HPC faer implementations. */
                public static final Set<String> HPC_PATTERN = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
            """);

        // HPC pattern: all fusion unary ops that have HPC support
        List<String> hpcPattern = new ArrayList<>();
        for (OpDefinition op : fusionUnary) {
            if (op.hpc()) {
                hpcPattern.add(OpRegistry.fuseTag(op.tag(), "sum"));
                hpcPattern.add(OpRegistry.fuseTag(op.tag(), "mean"));
            }
        }
        for (int i = 0; i < hpcPattern.size(); i++) {
            sb.append("            \"").append(hpcPattern.get(i)).append('"');
            if (i < hpcPattern.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("        )));\n\n");

        // HPC compound
        Set<String> hpcCompound = OpRegistry.hpcCompoundSpecials();
        sb.append("        /** Compound specials with HPC implementations. */\n");
        sb.append("        public static final Set<String> HPC_COMPOUND = Set.of(");
        sb.append(hpcCompound.stream().map(t -> "\"" + t + "\"").collect(Collectors.joining(", ")));
        sb.append(");\n\n");

        sb.append("""
                /** All HPC-supported fused tags. */
                public static final Set<String> HPC_ALL;
                static {
                    Set<String> s = new HashSet<>(HPC_PATTERN);
                    s.addAll(HPC_COMPOUND);
                    HPC_ALL = Collections.unmodifiableSet(s);
                }
            }

            """);
        return sb.toString();
    }

    /** Generates the complete Gpu or Hpc inner class. */
    private static String generateBackendOps(boolean gpu) {
        String className = gpu ? "Gpu" : "Hpc";
        String nameConst = gpu ? "GPU" : "HPC";
        String desc = gpu ? "GPU WGSL shader" : "HPC faer-based";

        // Collect base ops (sorted for deterministic output).
        // Compound specials are excluded from BASE — they enter SUPPORTED via
        // FusedTagRegistry.<NAME>_COMPOUND addition, keeping BASE ∩ FUSED = ∅.
        TreeSet<String> baseTags = new TreeSet<>();
        for (OpDefinition op : OpRegistry.ALL_OPS) {
            boolean supported = gpu ? op.gpu() : op.hpc();
            if (supported && !op.isLeaf() && !OpRegistry.isCompoundSpecial(op.tag())) {
                baseTags.add(op.tag());
            }
        }
        baseTags.add("leaf");
        baseTags.add("constant");

        StringBuilder sb = new StringBuilder();
        sb.append("    /** All ops with ").append(desc).append(" implementations. */\n");
        sb.append("    public static final class ").append(className).append(" {\n");

        sb.append("        static final Set<String> BASE = ")
          .append("Collections.unmodifiableSet(new HashSet<>(Arrays.asList(\n");
        List<String> sorted = new ArrayList<>(baseTags);
        for (int i = 0; i < sorted.size(); i++) {
            sb.append("            \"").append(sorted.get(i)).append('"');
            if (i < sorted.size() - 1) sb.append(',');
            sb.append('\n');
        }
        sb.append("        )));\n\n");

        sb.append("        public static final Set<String> SUPPORTED;\n");
        sb.append("        static {\n");
        sb.append("            Set<String> s = new HashSet<>(BASE);\n");
        sb.append("            s.addAll(FusedTagRegistry.").append(nameConst).append("_PATTERN);\n");

        Set<String> compound = gpu ? OpRegistry.gpuCompoundSpecials() : OpRegistry.hpcCompoundSpecials();
        for (String tag : compound) {
            sb.append("            s.add(\"").append(tag).append("\");\n");
        }

        sb.append("            SUPPORTED = Collections.unmodifiableSet(s);\n");
        sb.append("        }\n");
        sb.append("    }\n");
        return sb.toString();
    }

    private static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
