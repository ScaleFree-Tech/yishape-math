package com.yishape.lab.math.codegen;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 4.2 Step 4.1 — smoke test for {@link DocGenerator}.
 *
 * <p>Regenerates {@code docs/op_schema.md} from {@link OpRegistry} and asserts
 * the output is well-formed and in sync with the registry. Because
 * {@link DocGenerator#generate} is idempotent (it only writes when content
 * changes), running this test leaves the working tree clean iff the committed
 * doc is up to date — which CI's {@code git diff} check also enforces.
 */
public class DocGeneratorTest {

    private static final Path OP_SCHEMA_MD =
        Paths.get("docs/op_schema.md").toAbsolutePath();

    @Test
    void regenerateProducesInSyncDoc() throws Exception {
        DocGenerator.generate();
        assertTrue(Files.exists(OP_SCHEMA_MD), "op_schema.md must be generated");
    }

    @Test
    void docContainsEveryRegistryOp() throws Exception {
        DocGenerator.generate();
        String md = Files.readString(OP_SCHEMA_MD);

        // Every op tag in the registry must appear as a backticked code span.
        for (OpDefinition op : OpRegistry.ALL_OPS) {
            assertTrue(md.contains("`" + op.tag() + "`"),
                "op_schema.md missing tag: " + op.tag());
        }
    }

    @Test
    void docHasExpectedSections() throws Exception {
        DocGenerator.generate();
        String md = Files.readString(OP_SCHEMA_MD);
        assertAll("required sections present",
            () -> assertTrue(md.contains("# Autodiff Operation Schema"), "title"),
            () -> assertTrue(md.contains("## Summary"), "summary section"),
            () -> assertTrue(md.contains("## Operation Matrix"), "matrix section"),
            () -> assertTrue(md.contains("## Fusion Patterns"), "fusion section"),
            () -> assertTrue(md.contains("## Backend Coverage"), "backend section"),
            () -> assertTrue(md.contains("AUTO-GENERATED"), "auto-gen banner")
        );
    }

    @Test
    void summaryCountsMatchRegistry() throws Exception {
        DocGenerator.generate();
        String md = Files.readString(OP_SCHEMA_MD);

        // Total ops line must cite the registry size.
        assertTrue(md.contains("| Total ops (registry) | " + OpRegistry.size() + " |"),
            "total op count must match OpRegistry.size()");
        assertTrue(md.contains("| GPU-supported | " + OpRegistry.gpuOps().size() + " |"),
            "GPU count must match registry");
        assertTrue(md.contains("| HPC-supported | " + OpRegistry.hpcOps().size() + " |"),
            "HPC count must match registry");
    }
}
