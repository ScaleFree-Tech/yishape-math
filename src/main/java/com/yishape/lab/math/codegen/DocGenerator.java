package com.yishape.lab.math.codegen;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Generates {@code docs/op_schema.md} — the human-readable operation matrix —
 * from the single source of truth ({@link OpRegistry#ALL_OPS}).
 *
 * <p>Invoked from {@link CodegenTool#main} so the CI {@code git diff --exit-code}
 * check keeps the doc in sync with the registry. Never edit {@code op_schema.md}
 * by hand; edit {@link OpRegistry} and re-run codegen.
 *
 * <h2>Output structure</h2>
 * <ul>
 *   <li>Header + auto-generated banner + summary statistics</li>
 *   <li>Operation matrix grouped by {@link OpCategory}</li>
 *   <li>Fusion pattern reference ({@code {unary}{Reduce}} + compound specials)</li>
 *   <li>Backend coverage summary (GPU / HPC)</li>
 * </ul>
 */
public final class DocGenerator {

    private static final Path PROJECT_ROOT = Paths.get("").toAbsolutePath();
    private static final Path OP_SCHEMA_MD = PROJECT_ROOT.resolve("docs/op_schema.md");

    private DocGenerator() {}

    /** Generates (or refreshes) {@code docs/op_schema.md}. */
    public static void generate() throws IOException {
        String markdown = buildMarkdown();
        Files.createDirectories(OP_SCHEMA_MD.getParent());
        if (!Files.exists(OP_SCHEMA_MD) || !Files.readString(OP_SCHEMA_MD).equals(markdown)) {
            Files.writeString(OP_SCHEMA_MD, markdown);
            System.out.println("  docs/op_schema.md: updated");
        } else {
            System.out.println("  docs/op_schema.md: no changes needed");
        }
    }

    private static String buildMarkdown() {
        StringBuilder sb = new StringBuilder();

        sb.append("# Autodiff Operation Schema\n\n");
        sb.append("> **AUTO-GENERATED from `OpRegistry.ALL_OPS`.** Do not edit by hand —\n");
        sb.append("> edit `src/main/java/com/yishape/lab/math/codegen/OpRegistry.java` and re-run\n");
        sb.append("> `com.yishape.lab.math.codegen.CodegenTool`. CI rejects an out-of-sync doc.\n\n");

        // ── Summary statistics ──
        int total = OpRegistry.size();
        long gpuCount = OpRegistry.gpuOps().size();
        long hpcCount = OpRegistry.hpcOps().size();
        long fusionBases = OpRegistry.fusionUnaryOps().size();
        long compoundCount = OpRegistry.compoundSpecials().size();
        long leafCount = OpRegistry.ALL_OPS.stream().filter(OpDefinition::isLeaf).count();

        sb.append("## Summary\n\n");
        sb.append("| Metric | Value |\n|---|---|\n");
        sb.append("| Total ops (registry) | ").append(total).append(" |\n");
        sb.append("| Leaf / constant nodes (arity 0) | ").append(leafCount).append(" |\n");
        sb.append("| Executable ops (arity ≥ 1) | ").append(total - leafCount).append(" |\n");
        sb.append("| GPU-supported | ").append(gpuCount).append(" |\n");
        sb.append("| HPC-supported | ").append(hpcCount).append(" |\n");
        sb.append("| Fusion base ops (`{unary}{Reduce}`) | ").append(fusionBases).append(" |\n");
        sb.append("| Compound specials | ").append(compoundCount).append(" |\n\n");

        // ── Operation matrix grouped by category ──
        sb.append("## Operation Matrix\n\n");
        sb.append("Columns: **Arity** = tensor inputs consumed (0 = leaf/constant, no execution); ");
        sb.append("**GPU** / **HPC** = backend implements the op; **Fused** = derived `{unary}{Reduce}` ");
        sb.append("tags; **Param** = optional scalar parameter name.\n\n");

        Map<OpCategory, List<OpDefinition>> byCategory = new EnumMap<>(OpCategory.class);
        for (OpDefinition op : OpRegistry.ALL_OPS) {
            byCategory.computeIfAbsent(op.category(), k -> new ArrayList<>()).add(op);
        }

        for (OpCategory cat : OpCategory.values()) {
            List<OpDefinition> ops = byCategory.get(cat);
            if (ops == null || ops.isEmpty()) continue;
            sb.append("### ").append(pretty(cat)).append("\n\n");
            sb.append("| Tag | Arity | GPU | HPC | Fused variants | Param | Description |\n");
            sb.append("|---|---:|:---:|:---:|---|---|---|\n");
            // Sort alphabetically within category for stable diffs.
            List<OpDefinition> sorted = new ArrayList<>(ops);
            sorted.sort(Comparator.comparing(OpDefinition::tag));
            for (OpDefinition op : sorted) {
                sb.append("| `").append(op.tag()).append("` ")
                  .append(arityBadge(op)).append(" | ")
                  .append(op.arity()).append(" | ")
                  .append(op.gpu() ? "✅" : "—").append(" | ")
                  .append(op.hpc() ? "✅" : "—").append(" | ")
                  .append(fusedCell(op.fusedTags())).append(" | ")
                  .append(op.scalarParam() == null ? "" : "`" + op.scalarParam() + "`").append(" | ")
                  .append(op.description()).append(" |\n");
            }
            sb.append('\n');
        }

        // ── Fusion pattern reference ──
        sb.append("## Fusion Patterns\n\n");
        sb.append("Two classes of fused op carry native GPU/HPC implementations beyond the base ops:\n\n");
        sb.append("### `{unary}{Reduce}` pattern\n\n");
        sb.append("Any fusion-base unary op composed with `sum` or `mean` reduces to a single fused ");
        sb.append("node named by concatenation (e.g. `square().sum()` → `squareSum`, ");
        sb.append("`exp().mean()` → `expMean`). Built by `OpRegistry.fuseTag(unary, reduce)`.\n\n");
        sb.append("Fusion-base unary ops (").append(fusionBases).append("):\n\n");
        sb.append("| Unary tag | Fused tags |\n|---|---|\n");
        List<OpDefinition> fusionUnary = OpRegistry.fusionUnaryOps();
        for (OpDefinition op : fusionUnary) {
            sb.append("| `").append(op.tag()).append("` | ")
              .append(fusedCell(op.fusedTags())).append(" |\n");
        }
        sb.append('\n');

        sb.append("### Compound specials\n\n");
        sb.append("Loss / reduce compounds that do not follow the `{unary}{Reduce}` pattern. ");
        sb.append("They are excluded from `BASE` and enter `SUPPORTED` via ");
        sb.append("`FusedTagRegistry.<BACKEND>_COMPOUND`.\n\n");
        sb.append("| Tag | GPU | HPC | Description |\n|---|:---:|:---:|---|\n");
        for (OpDefinition op : OpRegistry.compoundSpecials()) {
            sb.append("| `").append(op.tag()).append("` | ")
              .append(op.gpu() ? "✅" : "—").append(" | ")
              .append(op.hpc() ? "✅" : "—").append(" | ")
              .append(op.description()).append(" |\n");
        }
        sb.append('\n');

        // ── Backend coverage ──
        sb.append("## Backend Coverage\n\n");
        sb.append("`SUPPORTED = BASE ∪ <BACKEND>_PATTERN ∪ <BACKEND>_COMPOUND`. ");
        sb.append("`BASE ∩ FUSED = ∅` by construction (compound specials are kept out of BASE).\n\n");
        sb.append("| Backend | BASE ops | Pattern fused | Compound | Total SUPPORTED |\n|---|---:|---:|---:|---:|\n");
        sb.append(backendRow(true));
        sb.append(backendRow(false));
        sb.append('\n');

        sb.append("---\n");
        sb.append("*Generated by `com.yishape.lab.math.codegen.DocGenerator` from `OpRegistry` (")
          .append(total).append(" ops).*\n");

        return sb.toString();
    }

    /** Inline badge marking leaf/constant nodes (arity 0). */
    private static String arityBadge(OpDefinition op) {
        return op.isLeaf() ? "🍃" : "";
    }

    /** Human-friendly category label: BINARY_SCALAR → "Binary (scalar)". */
    private static String pretty(OpCategory cat) {
        String s = cat.name().replace('_', ' ');
        return Character.toUpperCase(s.charAt(0)) + s.substring(1).toLowerCase();
    }

    private static String fusedCell(Set<String> fusedTags) {
        if (fusedTags == null || fusedTags.isEmpty()) return "—";
        TreeSet<String> sorted = new TreeSet<>(fusedTags);
        StringBuilder b = new StringBuilder();
        for (String t : sorted) {
            if (b.length() > 0) b.append(", ");
            b.append("`").append(t).append("`");
        }
        return b.toString();
    }

    private static String backendRow(boolean gpu) {
        String name = gpu ? "GPU" : "HPC";
        // BASE = executable non-compound ops + leaf/constant.
        long base = OpRegistry.ALL_OPS.stream()
            .filter(o -> (gpu ? o.gpu() : o.hpc()))
            .filter(o -> o.isLeaf() || !OpRegistry.isCompoundSpecial(o.tag()))
            .count();
        long pattern = OpRegistry.fusionUnaryOps().stream()
            .filter(o -> gpu ? o.gpu() : o.hpc())
            .count() * 2L;  // each base yields {tag}Sum + {tag}Mean
        long compound = (gpu ? OpRegistry.gpuCompoundSpecials() : OpRegistry.hpcCompoundSpecials()).size();
        long supported = base + pattern + compound;
        return "| " + name + " | " + base + " | " + pattern + " | " + compound
            + " | " + supported + " |\n";
    }
}
