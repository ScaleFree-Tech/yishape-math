package com.yishape.lab.math.codegen;

import java.util.List;
import java.util.Set;

/**
 * Single-source-of-truth definition for a compute operation.
 *
 * <p>One OpDefinition captures everything needed to generate:
 * <ul>
 *   <li>Java: GraphOpSchema.Gpu.BASE / Hpc.BASE membership, FusedTag entries, describeTag()</li>
 *   <li>Rust: forward_dispatch / backward_dispatch match arms, SUPPORTED_OPS array</li>
 *   <li>Docs: op_schema.md operation matrix row</li>
 * </ul>
 *
 * @param tag         canonical op tag string (e.g. "add", "matmul", "relu")
 * @param category    classification for docs and validation
 * @param arity       number of tensor inputs this op consumes (0 for leaf/constant)
 * @param gpu         whether GPU backend implements this op
 * @param hpc         whether HPC backend implements this op
 * @param description human-readable name for describeTag() and docs (e.g. "Element-wise Addition")
 * @param fusedTags   set of fused variant tags derived from this op (e.g. ["addSum", "addMean"])
 * @param scalarParam name of optional scalar parameter (e.g. "exponent" for pow, null for most ops)
 */
public record OpDefinition(
    String tag,
    OpCategory category,
    int arity,
    boolean gpu,
    boolean hpc,
    String description,
    Set<String> fusedTags,
    String scalarParam
) {
    public OpDefinition {
        if (tag == null || tag.isBlank()) throw new IllegalArgumentException("tag is required");
        if (category == null) throw new IllegalArgumentException("category is required for " + tag);
        if (arity < 0) throw new IllegalArgumentException("arity must be >= 0 for " + tag);
        if (description == null || description.isBlank()) {
            description = tag; // fallback: use tag as description
        }
        if (fusedTags == null) fusedTags = Set.of();
        if (scalarParam != null && scalarParam.isBlank()) scalarParam = null;
    }

    /** Convenience: op with arity 1, no fused tags. */
    public OpDefinition(String tag, OpCategory category, boolean gpu, boolean hpc, String description) {
        this(tag, category, 1, gpu, hpc, description, Set.of(), null);
    }

    /** Convenience: op with arity 1, no scalar param. */
    public OpDefinition(String tag, OpCategory category, int arity, boolean gpu, boolean hpc, String description) {
        this(tag, category, arity, gpu, hpc, description, Set.of(), null);
    }

    public boolean isLeaf() { return arity == 0; }

    /** Returns true if this op participates in the {unary}{reduce} fusion pattern. */
    public boolean isFusionBase() {
        return !fusedTags.isEmpty();
    }
}
