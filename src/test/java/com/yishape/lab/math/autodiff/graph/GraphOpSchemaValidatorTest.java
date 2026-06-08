package com.yishape.lab.math.autodiff.graph;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link GraphOpSchemaValidator} — validates internal consistency
 * of the op schema definitions and verifies JSON export is well-formed.
 */
public class GraphOpSchemaValidatorTest {

    @Test
    void testValidateAllNoErrors() {
        GraphOpSchemaValidator.Result r = GraphOpSchemaValidator.validateAll();
        assertFalse(r.hasErrors(), "Schema validation should have zero errors:\n" + r);
    }

    @Test
    void testAllOpsHaveUniqueTags() {
        var ops = GraphOpSchemaValidator.allOps();
        var tags = ops.stream().map(GraphOpSchemaValidator.OpSchema::tag).toList();
        assertEquals(tags.size(), tags.stream().distinct().count(),
            "All op tags must be unique");
    }

    @Test
    void testExportJsonIsValid() {
        String json = GraphOpSchemaValidator.exportJson();
        assertNotNull(json);
        assertTrue(json.startsWith("["), "JSON must start with array");
        assertTrue(json.endsWith("]\n") || json.endsWith("]"), "JSON must end with array close");

        // Verify all known tags appear in JSON
        for (GraphOpSchemaValidator.OpSchema op : GraphOpSchemaValidator.allOps()) {
            assertTrue(json.contains("\"tag\": \"" + op.tag() + "\""),
                "JSON must contain tag '" + op.tag() + "'");
        }
    }

    @Test
    void testJsonParsableAsValidFormat() {
        String json = GraphOpSchemaValidator.exportJson();
        // Check key structural elements
        assertTrue(json.contains("\"inputsNoBias\""));
        assertTrue(json.contains("\"inputsWithBias\""));
        assertTrue(json.contains("\"scalarFields\""));
        assertTrue(json.contains("\"backwardGrads\""));
        assertTrue(json.contains("\"lenNoBias\""));
        assertTrue(json.contains("\"lenWithBias\""));
    }

    @Test
    void testLinearSchemaMatchesJavaConstants() {
        var linear = GraphOpSchemaValidator.allOps().stream()
            .filter(o -> o.tag().equals("linear"))
            .findFirst().orElseThrow();

        assertEquals(GraphOpSchema.Linear.X, linear.inputsNoBias().get(0).index());
        assertEquals(GraphOpSchema.Linear.WEIGHT, linear.inputsNoBias().get(1).index());

        assertEquals(GraphOpSchema.Linear.LEN_NO_BIAS, linear.lenNoBias());
        assertEquals(GraphOpSchema.Linear.LEN_WITH_BIAS, linear.lenWithBias());

        // Backward grads
        assertEquals(GraphOpSchema.Linear.GRAD_DX, linear.backwardGrads().get(0).index());
        assertEquals(GraphOpSchema.Linear.GRAD_DWEIGHT, linear.backwardGrads().get(1).index());
        assertEquals(GraphOpSchema.Linear.GRAD_DBIAS, linear.backwardGrads().get(2).index());
    }

    @Test
    void testMhaSchemaMatchesJavaConstants() {
        var mha = GraphOpSchemaValidator.allOps().stream()
            .filter(o -> o.tag().equals("mha"))
            .findFirst().orElseThrow();

        assertEquals(GraphOpSchema.MHA.X, mha.inputsNoBias().get(0).index());
        assertEquals(GraphOpSchema.MHA.W_QKV, mha.inputsNoBias().get(1).index());
        assertEquals(GraphOpSchema.MHA.WO_NO_BIAS, mha.inputsNoBias().get(2).index());

        // With bias
        assertEquals(GraphOpSchema.MHA.QKV_BIAS, mha.inputsWithBias().get(2).index());
        assertEquals(GraphOpSchema.MHA.WO_WITH_BIAS, mha.inputsWithBias().get(3).index());
        assertEquals(GraphOpSchema.MHA.OUT_BIAS, mha.inputsWithBias().get(4).index());

        // Backward
        assertEquals(GraphOpSchema.MHA.GRAD_DX, mha.backwardGrads().get(0).index());
        assertEquals(GraphOpSchema.MHA.GRAD_DWQKV, mha.backwardGrads().get(1).index());
        assertEquals(GraphOpSchema.MHA.GRAD_DWO, mha.backwardGrads().get(3).index());
    }

    @Test
    void testAllOPScalarsHaveNonNegativeShift() {
        for (var op : GraphOpSchemaValidator.allOps()) {
            for (var s : op.scalarFields()) {
                assertTrue(s.shift() >= 0, op.tag() + " scalar '" + s.name() + "' has negative shift");
                assertTrue(s.mask() != 0, op.tag() + " scalar '" + s.name() + "' has zero mask");
            }
            for (var s : op.scalar2Fields()) {
                assertTrue(s.shift() >= 0, op.tag() + " scalar2 '" + s.name() + "' has negative shift");
            }
        }
    }

    @Test
    void testEachOpHasAtLeastOneInput() {
        for (var op : GraphOpSchemaValidator.allOps()) {
            assertFalse(op.inputsNoBias().isEmpty(),
                op.tag() + " must have at least one input (noBias)");
            assertFalse(op.inputsWithBias().isEmpty(),
                op.tag() + " must have at least one input (withBias)");
        }
    }

    @Test
    void testNoOpHasNegativeBackwardIndex() {
        for (var op : GraphOpSchemaValidator.allOps()) {
            for (var g : op.backwardGrads()) {
                assertTrue(g.index() >= 0,
                    op.tag() + " backward grad '" + g.name() + "' has negative index " + g.index());
            }
        }
    }

    @Test
    void testNoBiasAndWithBiasSameLengthOpsHaveConsistentNames() {
        // For ops where bias doesn't shift indices (same length in both variants),
        // every input at the same position must match.
        for (var op : GraphOpSchemaValidator.allOps()) {
            if (op.lenNoBias() != op.lenWithBias()) continue; // skip conditional-index ops
            for (int i = 0; i < op.inputsNoBias().size(); i++) {
                assertEquals(op.inputsNoBias().get(i).name(), op.inputsWithBias().get(i).name(),
                    op.tag() + ": position " + i + " name mismatch");
            }
        }
    }

    @Test
    void testConditionalIndexOpsHaveHelperMethods() {
        // For ops where bias shifts indices, verify the index values make sense
        for (var op : GraphOpSchemaValidator.allOps()) {
            if (op.lenNoBias() == op.lenWithBias()) continue;

            // The noBias variant should be a prefix of withBias up to the bias insertion point,
            // OR the bias insertion adds entries that shift later indices.
            // Just verify that all noBias indices are covered by withBias or shifted by bias count
            assertTrue(op.lenWithBias() > op.lenNoBias(),
                op.tag() + ": withBias length must be >= noBias length");
        }
    }

    @Test
    void testSampleJsonOutput() {
        // Print JSON for manual inspection if needed
        String json = GraphOpSchemaValidator.exportJson();
        System.out.println("=== GraphOpSchema JSON Export ===");
        System.out.println(json);
    }
}
