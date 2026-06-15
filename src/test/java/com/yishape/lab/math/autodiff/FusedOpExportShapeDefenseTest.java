package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.graph.GraphOpSchema;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Defensive regression test for fused-op exportShape.
 *
 * The bug: a prior refactor changed r.exportShape from x.shape() (input shape)
 * to new int[]{1} (output scalar shape) across ALL 17 fusion branches.
 * This broke GPU/HPC export because the executor needs the input shape to
 * reconstruct the intermediate tensor layout.
 *
 * Every fused unaryOp+sum pattern is tested — if a new fusion is added,
 * it must be added here too.
 */
public class FusedOpExportShapeDefenseTest {

    private static void assertExportShapeEqualsInput(String opTag, RereDiffTensor x, IDiffTensor fused) {
        assertTrue(fused instanceof RereDiffTensor,
            opTag + "+sum should produce RereDiffTensor");
        RereDiffTensor rf = (RereDiffTensor) fused;
        assertArrayEquals(x.shape(), rf.exportShape(),
            opTag + "+sum exportShape should match input shape "
            + java.util.Arrays.toString(x.shape()));
    }

    // Helper: create input, apply op, sum, verify exportShape
    private void verifyFusedExport(String opTag, int[] shape, java.util.function.Function<RereDiffTensor, IDiffTensor> op) {
        int total = 1;
        for (int s : shape) total *= s;
        double[] data = new double[total];
        // Fill with values that avoid degenerate gradients (no all-zeros, avoid relu death)
        java.util.Arrays.fill(data, 1.5);
        RereDiffTensor x = new RereDiffTensor(data, shape);
        x.setRequiresGrad(true);
        IDiffTensor activated = op.apply(x);
        IDiffTensor fused = activated.sum();
        assertExportShapeEqualsInput(opTag, x, fused);
    }

    private static final int[] SHAPE = {2, 3};

    @Test void testSquareSumExportShape()  { verifyFusedExport("square",  SHAPE, x -> x.square()); }
    @Test void testReluSumExportShape()    { verifyFusedExport("relu",    SHAPE, x -> x.relu()); }
    @Test void testExpSumExportShape()     { verifyFusedExport("exp",     SHAPE, x -> x.exp()); }
    @Test void testSigmoidSumExportShape() { verifyFusedExport("sigmoid", SHAPE, x -> x.sigmoid()); }
    @Test void testAbsSumExportShape()     { verifyFusedExport("abs",     SHAPE, x -> x.abs()); }
    @Test void testTanhSumExportShape()    { verifyFusedExport("tanh",    SHAPE, x -> x.tanh()); }
    @Test void testSiluSumExportShape()    { verifyFusedExport("silu",    SHAPE, x -> x.silu()); }
    @Test void testLogSumExportShape()     {
        // log requires positive input; don't use the 1.5-fill helper
        RereDiffTensor x = new RereDiffTensor(new double[]{2, 2, 2, 2, 2, 2}, 2, 3);
        x.setRequiresGrad(true);
        IDiffTensor fused = x.log().sum();
        assertExportShapeEqualsInput("log", x, fused);
    }
    @Test void testSinSumExportShape()     { verifyFusedExport("sin",     SHAPE, x -> x.sin()); }
    @Test void testCosSumExportShape()     { verifyFusedExport("cos",     SHAPE, x -> x.cos()); }
    @Test void testLeakyReluSumExportShape() { verifyFusedExport("leakyRelu", SHAPE, x -> x.leakyRelu(0.01)); }
    @Test void testEluSumExportShape()     { verifyFusedExport("elu",     SHAPE, x -> x.elu(1.0)); }
    @Test void testSeluSumExportShape()    { verifyFusedExport("selu",    SHAPE, x -> x.selu()); }
    @Test void testSoftplusSumExportShape(){ verifyFusedExport("softplus",SHAPE, x -> x.softplus(1.0)); }
    @Test void testHardtanhSumExportShape(){ verifyFusedExport("hardtanh",SHAPE, x -> x.hardtanh(-1, 1)); }
    @Test void testMishSumExportShape()    { verifyFusedExport("mish",    SHAPE, x -> x.mish()); }
    @Test void testGeluSumExportShape()    { verifyFusedExport("gelu",    SHAPE, x -> x.gelu()); }

    @Test void testPowSumExportShape() {
        double[] d = new double[6];
        java.util.Arrays.fill(d, 2.0);
        RereDiffTensor x = new RereDiffTensor(d, SHAPE);
        x.setRequiresGrad(true);
        IDiffTensor fused = x.pow(3.0).sum();
        assertExportShapeEqualsInput("pow", x, fused);
    }

    // Verify non-fused sum does NOT have exportShape set
    @Test
    void testNonFusedSumHasNoExportShape() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        x.setRequiresGrad(true);
        // add(1).sum() does NOT match any fusion pattern
        IDiffTensor s = x.add(1).sum();
        assertTrue(s instanceof RereDiffTensor);
        RereDiffTensor rs = (RereDiffTensor) s;
        assertNull(rs.exportShape(), "non-fused sum should have null exportShape");
    }

    // Verify that the fused op tag is correct
    @Test
    void testFusedOpTagIsCorrect() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, 2, 3, 4}, 2, 2);
        x.setRequiresGrad(true);
        IDiffTensor s = x.square().sum();
        assertTrue(s instanceof RereDiffTensor);
        RereDiffTensor rs = (RereDiffTensor) s;
        assertEquals(GraphOpSchema.FusedTag.of("square", "sum"), rs.opTag());
    }
}
