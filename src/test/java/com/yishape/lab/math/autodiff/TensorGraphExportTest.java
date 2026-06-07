package com.yishape.lab.math.autodiff;

import com.yishape.lab.math.autodiff.impl.RereDiffTensor;
import com.yishape.lab.math.autodiff.graph.TensorGraphExporter;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 0 verification: tensor graph backward + JSON export.
 */
public class TensorGraphExportTest {

    @Test
    void testReluSumBackward() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, -2, 3, -4, 5, -6}, 2, 3);
        x.setRequiresGrad(true);

        IDiffTensor r = x.relu();
        IDiffTensor s = r.sum();
        s.backward();

        double[] g = x.gradData();
        assertNotNull(g);
        assertEquals(1.0, g[0], 1e-12);
        assertEquals(0.0, g[1], 1e-12);
        assertEquals(1.0, g[2], 1e-12);
        assertEquals(0.0, g[3], 1e-12);
        assertEquals(1.0, g[4], 1e-12);
        assertEquals(0.0, g[5], 1e-12);
    }

    @Test
    void testSigmoidGradient() {
        RereDiffTensor x = new RereDiffTensor(new double[]{0, 1, -1, 2}, 2, 2);
        x.setRequiresGrad(true);

        IDiffTensor s = x.sigmoid();
        IDiffTensor m = s.sum();
        m.backward();

        // sigmoid gradient: s * (1-s)
        double[] xd = {0, 1, -1, 2};
        double[] expected = new double[4];
        for (int i = 0; i < 4; i++) {
            double si = 1.0 / (1.0 + Math.exp(-xd[i]));
            expected[i] = si * (1 - si);
        }
        double[] g = x.gradData();
        assertNotNull(g);
        for (int i = 0; i < 4; i++) {
            assertEquals(expected[i], g[i], 1e-10);
        }
    }

    @Test
    void testExportJsonFormat() {
        RereDiffTensor x = new RereDiffTensor(new double[]{1, -2, 3}, 1, 3);
        x.setRequiresGrad(true);
        x.setOpTag("leaf");
        x.setScalarParam(0.5);

        IDiffTensor r = x.relu();
        IDiffTensor s = r.sum();

        String json = TensorGraphExporter.toJson((RereDiffTensor) s);
        assertTrue(json.contains("\"nodes\""), "Should contain nodes array");
        // relu().sum() is fused into a single reluSum node
        assertTrue(json.contains("\"op\":\"reluSum\""), "Should contain reluSum fused op");
        assertTrue(json.contains("\"shape\""), "Should contain shape");
        assertTrue(json.contains("\"inputs\""), "Should contain input references");
    }

    @Test
    void testAddGradient() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        a.setRequiresGrad(true);
        RereDiffTensor b = new RereDiffTensor(new double[]{4, 5, 6}, 3);
        b.setRequiresGrad(true);

        IDiffTensor c = a.add(b);
        IDiffTensor s = c.sum();
        s.backward();

        // d/dx_i (x_i + y_i) = 1
        assertArrayEquals(new double[]{1, 1, 1}, a.gradData(), 1e-12);
        assertArrayEquals(new double[]{1, 1, 1}, b.gradData(), 1e-12);
    }

    @Test
    void testMulGradient() {
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3}, 3);
        a.setRequiresGrad(true);
        RereDiffTensor b = new RereDiffTensor(new double[]{4, 5, 6}, 3);
        b.setRequiresGrad(true);

        IDiffTensor c = a.mul(b);
        IDiffTensor s = c.sum();
        s.backward();

        // d/dx_i (x_i * y_i) = y_i
        assertArrayEquals(new double[]{4, 5, 6}, a.gradData(), 1e-12);
        assertArrayEquals(new double[]{1, 2, 3}, b.gradData(), 1e-12);
    }

    @Test
    void testBroadcastAddGradient() {
        // a: [2,3], b: [3] → broadcast add → sum
        RereDiffTensor a = new RereDiffTensor(new double[]{1, 2, 3, 4, 5, 6}, 2, 3);
        a.setRequiresGrad(true);
        RereDiffTensor b = new RereDiffTensor(new double[]{10, 20, 30}, 3);
        b.setRequiresGrad(true);

        IDiffTensor c = a.add(b);
        IDiffTensor s = c.sum();
        s.backward();

        // d/db_i = sum over broadcast dimension (rows): each b_i contributes to 2 rows
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.gradData(), 1e-12);
        assertArrayEquals(new double[]{2, 2, 2}, b.gradData(), 1e-12);
    }
}
