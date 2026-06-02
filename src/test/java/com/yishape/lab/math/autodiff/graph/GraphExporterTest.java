package com.yishape.lab.math.autodiff.graph;

import com.yishape.lab.math.autodiff.AD;
import com.yishape.lab.math.autodiff.IDiffVector;
import com.yishape.lab.math.autodiff.impl.RereDiffVector;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link GraphExporter} — validates JSON format matches the Rust GPU parser contract.
 * <p>
 * Expected JSON format per node:
 * <pre>
 * {"id":N,"shape":[N],"op":"...","data":[...],"scalar":N,"param2":N,"inputs":[N,N]}
 * </pre>
 */
public class GraphExporterTest {

    // ==================== Basic Export ====================

    @Test
    void testSingleLeafExport() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        String json = GraphExporter.toJson((RereDiffVector) x);
        assertNotNull(json);
        assertTrue(json.contains("\"nodes\":["));
        assertTrue(json.contains("\"op\":\"leaf\""));
        assertTrue(json.contains("\"data\":[1.0,2.0,3.0]"));
        assertTrue(json.contains("\"shape\":[3]"));
        assertTrue(json.contains("\"id\":0"));
    }

    @Test
    void testSimpleBinaryOpExport() {
        IDiffVector a = AD.vector(new double[]{1, 2});
        IDiffVector b = AD.vector(new double[]{3, 4});
        IDiffVector loss = a.add(b).sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertNotNull(json);
        // Should have: leaf(a), leaf(b), add, sum = 4 nodes
        assertTrue(json.contains("\"op\":\"leaf\""));
        assertTrue(json.contains("\"op\":\"add\""));
        assertTrue(json.contains("\"op\":\"sum\""));
        // add node should reference inputs
        assertTrue(json.contains("\"inputs\":[0,1]"));
        // sum node should reference add
        assertTrue(json.contains("\"inputs\":[2]"));
    }

    // ==================== Op Tags ====================

    @Test
    void testPowOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        // pow(2).abs().sum() — abs() barrier prevents powSum fusion so we see separate "pow" op
        IDiffVector loss = x.pow(2).abs().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"pow\""));
        assertTrue(json.contains("\"scalar\":2.0"));
    }

    @Test
    void testMulScalarOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        IDiffVector loss = x.mul(3.0).sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"mulScalar\""));
        assertTrue(json.contains("\"scalar\":3.0"));
    }

    @Test
    void testExpOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2});
        // exp().abs().sum() — abs() barrier prevents expSum fusion
        IDiffVector loss = x.exp().abs().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"exp\""));
    }

    @Test
    void testSigmoidOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2});
        IDiffVector loss = x.sigmoid().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        // sigmoid().sum() now fuses into a single "sigmoidSum" node
        assertTrue(json.contains("\"op\":\"sigmoidSum\"") || json.contains("\"op\":\"sigmoid\""));
    }

    @Test
    void testTanhOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2});
        IDiffVector loss = x.tanh().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        // tanh().sum() now fuses into a single "tanhSum" node
        assertTrue(json.contains("\"op\":\"tanhSum\"") || json.contains("\"op\":\"tanh\""));
    }

    @Test
    void testReluOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2});
        IDiffVector loss = x.relu().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        // relu().sum() now fuses into a single "reluSum" node
        assertTrue(json.contains("\"op\":\"reluSum\"") || json.contains("\"op\":\"relu\""));
    }

    @Test
    void testGeluOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2});
        IDiffVector loss = x.gelu().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"gelu\""));
    }

    @Test
    void testDotOpTag() {
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        IDiffVector loss = a.dot(b);
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"dot\""));
    }

    @Test
    void testMeanOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3, 4});
        IDiffVector loss = x.mean();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"mean\""));
    }

    @Test
    void testNegOpTag() {
        IDiffVector x = AD.vector(new double[]{1, 2});
        IDiffVector loss = x.neg().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"neg\""));
    }

    @Test
    void testSubOpTag() {
        IDiffVector a = AD.vector(new double[]{1, 2});
        IDiffVector b = AD.vector(new double[]{3, 4});
        IDiffVector loss = a.sub(b).sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"sub\""));
    }

    @Test
    void testDivOpTag() {
        IDiffVector a = AD.vector(new double[]{6, 8});
        IDiffVector b = AD.vector(new double[]{2, 4});
        IDiffVector loss = a.div(b).sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"op\":\"div\""));
    }

    // ==================== Scalar Params ====================

    @Test
    void testScalarParamIncluded() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        // pow(3).abs().sum() — abs() barrier prevents powSum fusion
        IDiffVector loss = x.pow(3).abs().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"scalar\":3.0"));
    }

    @Test
    void testNoScalarParamForBinaryOps() {
        IDiffVector a = AD.vector(new double[]{1, 2});
        IDiffVector b = AD.vector(new double[]{3, 4});
        // a.mul(b).abs().sum() — abs() barrier prevents mulSum fusion so we see separate "mul" node
        IDiffVector loss = a.mul(b).abs().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        // Find the mul node
        int mulIdx = json.indexOf("\"op\":\"mul\"");
        assertTrue(mulIdx > 0);
        // Check there's no "scalar" before the next node
        int nextNode = json.indexOf("{\"id\":", mulIdx + 1);
        String mulNode = json.substring(mulIdx, nextNode > 0 ? nextNode : json.length());
        assertFalse(mulNode.contains("\"scalar\""), "mul node should not have scalar: " + mulNode);
    }

    @Test
    void testParam2IncludedForClamp() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3});
        IDiffVector loss = x.clamp(-1.0, 2.0).sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertTrue(json.contains("\"param2\":"));
    }

    // ==================== Shape Arrays ====================

    @Test
    void testShapeArrayCorrect() {
        IDiffVector x = AD.vector(new double[]{1, 2, 3, 4, 5});
        String json = GraphExporter.toJson((RereDiffVector) x);
        assertTrue(json.contains("\"shape\":[5]"));
    }

    // ==================== Node IDs ====================

    @Test
    void testNodeIdsAreSequential() {
        IDiffVector a = AD.vector(new double[]{1, 2});
        IDiffVector b = AD.vector(new double[]{3, 4});
        // a.mul(b).abs().sum() — abs() barrier prevents mulSum fusion
        IDiffVector loss = a.mul(b).abs().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        // Should have id:0..4 (a, b, mul, abs, sum)
        assertTrue(json.contains("\"id\":0"));
        assertTrue(json.contains("\"id\":1"));
        assertTrue(json.contains("\"id\":2"));
    }

    // ==================== Graph Structure ====================

    @Test
    void testMultiLeafGraph() {
        IDiffVector a = AD.vector(new double[]{1, 2});
        IDiffVector b = AD.vector(new double[]{3, 4});
        IDiffVector c = AD.vector(new double[]{5, 6});
        // a.add(b).mul(c).abs().sum() — abs().sum() now fuses into absSum
        IDiffVector loss = a.add(b).mul(c).abs().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertNotNull(json);
        assertTrue(json.contains("\"op\":\"leaf\""));
        assertTrue(json.contains("\"op\":\"add\""));
        assertTrue(json.contains("\"op\":\"mul\""));
        assertTrue(json.contains("\"op\":\"absSum\"") || json.contains("\"op\":\"sum\""));
    }

    @Test
    void testDeepChainExport() {
        // x -> exp -> log -> sigmoid -> sum (sigmoid().sum() fuses into sigmoidSum)
        IDiffVector x = AD.vector(new double[]{0.5, 1.0});
        IDiffVector loss = x.exp().log().sigmoid().sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);
        assertNotNull(json);
        assertTrue(json.contains("\"op\":\"exp\""));
        assertTrue(json.contains("\"op\":\"log\""));
        assertTrue(json.contains("\"op\":\"sigmoidSum\"") || json.contains("\"op\":\"sigmoid\""));
    }

    // ==================== Valid JSON Structure ====================

    @Test
    void testExportedJsonIsValidStructure() {
        IDiffVector a = AD.vector(new double[]{1, 2, 3});
        IDiffVector b = AD.vector(new double[]{4, 5, 6});
        IDiffVector loss = a.dot(b);
        String json = GraphExporter.toJson((RereDiffVector) loss);

        // Basic structural checks
        assertTrue(json.startsWith("{\"nodes\":["));
        assertTrue(json.endsWith("]}"));
        // All nodes should have required fields
        assertTrue(json.contains("\"id\":"));
        assertTrue(json.contains("\"shape\":"));
        assertTrue(json.contains("\"op\":"));
    }

    // ==================== Leaf Data ====================

    @Test
    void testLeafDataPresent() {
        IDiffVector x = AD.vector(new double[]{10, 20, 30});
        String json = GraphExporter.toJson((RereDiffVector) x);
        assertTrue(json.contains("\"data\":[10.0,20.0,30.0]"));
    }

    @Test
    void testNonLeafNodeNoData() {
        IDiffVector a = AD.vector(new double[]{1, 2});
        IDiffVector b = AD.vector(new double[]{3, 4});
        IDiffVector loss = a.add(b).sum();
        String json = GraphExporter.toJson((RereDiffVector) loss);

        // Find the add node — it should NOT have data
        int addIdx = json.indexOf("\"op\":\"add\"");
        assertTrue(addIdx > 0);
        int nextNode = json.indexOf("{\"id\":", addIdx + 1);
        String addNode = json.substring(addIdx, nextNode > 0 ? nextNode : json.length());
        assertFalse(addNode.contains("\"data\""), "add node should not have data: " + addNode);
    }
}
