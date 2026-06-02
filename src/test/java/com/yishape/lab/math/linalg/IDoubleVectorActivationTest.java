package com.yishape.lab.math.linalg;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class IDoubleVectorActivationTest {

    private static final double EPS = 1e-12;

    @Test
    void testSoftmax() {
        IDoubleVector x = IDoubleVector.of(new double[]{2.0, 1.0, 0.1});
        IDoubleVector sm = x.softmax();
        assertEquals(1.0, sm.sumValue(), EPS, "softmax should sum to 1");
        assertTrue(sm.get(0) > sm.get(1), "softmax should preserve order");
        assertTrue(sm.get(0) > 0 && sm.get(0) < 1, "softmax values in (0,1)");
    }

    @Test
    void testLogSoftmax() {
        IDoubleVector x = IDoubleVector.of(new double[]{2.0, 1.0, 0.1});
        IDoubleVector sm = x.softmax();
        IDoubleVector lsm = x.logSoftmax();
        assertEquals(Math.log(sm.get(0)), lsm.get(0), EPS);
        assertEquals(Math.log(sm.get(1)), lsm.get(1), EPS);
    }

    @Test
    void testGelu() {
        IDoubleVector x = IDoubleVector.of(new double[]{0.0, 1.0, -1.0});
        IDoubleVector g = x.gelu();
        assertEquals(0.0, g.get(0), 1e-10, "GELU(0) should be 0");
        assertTrue(g.get(1) > 0, "GELU(1) should be positive");
        assertTrue(g.get(2) < 0, "GELU(-1) should be negative");
    }

    @Test
    void testSilu() {
        IDoubleVector x = IDoubleVector.of(new double[]{0.0, 2.0});
        IDoubleVector s = x.silu();
        assertEquals(0.0, s.get(0), EPS, "SiLU(0) = 0");
        double expected = 2.0 / (1.0 + Math.exp(-2.0));
        assertEquals(expected, s.get(1), 1e-10);
    }

    @Test
    void testLeakyRelu() {
        IDoubleVector x = IDoubleVector.of(new double[]{-2.0, -1.0, 0.0, 1.0, 2.0});
        IDoubleVector lr = x.leakyRelu(0.01);
        assertEquals(-0.02, lr.get(0), EPS);
        assertEquals(-0.01, lr.get(1), EPS);
        assertEquals(0.0, lr.get(2), EPS);
        assertEquals(1.0, lr.get(3), EPS);
        assertEquals(2.0, lr.get(4), EPS);
    }

    @Test
    void testElu() {
        IDoubleVector x = IDoubleVector.of(new double[]{-2.0, 0.0, 2.0});
        IDoubleVector e = x.elu(1.0);
        assertEquals(Math.exp(-2) - 1, e.get(0), EPS);
        assertEquals(0.0, e.get(1), EPS);
        assertEquals(2.0, e.get(2), EPS);
    }

    @Test
    void testSelu() {
        IDoubleVector x = IDoubleVector.of(new double[]{-2.0, 0.0, 2.0});
        IDoubleVector s = x.selu();
        final double SCALE = 1.0507009873554804934193349852946;
        final double ALPHA = 1.6732632423543772848170429916717;
        assertEquals(SCALE * ALPHA * (Math.exp(-2) - 1), s.get(0), 1e-10);
        assertEquals(0.0, s.get(1), EPS, "SELU(0) = 0");
        assertEquals(SCALE * 2.0, s.get(2), EPS);
    }

    @Test
    void testMish() {
        IDoubleVector x = IDoubleVector.of(new double[]{0.0, 1.0});
        IDoubleVector m = x.mish();
        assertEquals(0.0, m.get(0), EPS, "Mish(0) = 0");
        assertTrue(m.get(1) > 0);
    }

    @Test
    void testSoftplus() {
        IDoubleVector x = IDoubleVector.of(new double[]{-1.0, 0.0, 1.0});
        IDoubleVector sp = x.softplus(1.0);
        assertEquals(Math.log(1 + Math.exp(-1)), sp.get(0), 1e-10);
        assertEquals(Math.log(2), sp.get(1), EPS);
        assertEquals(Math.log(1 + Math.exp(1)), sp.get(2), 1e-10);
    }

    @Test
    void testHardtanh() {
        IDoubleVector x = IDoubleVector.of(new double[]{-2.0, -0.5, 0.0, 0.5, 2.0});
        IDoubleVector h = x.hardtanh(-0.5, 0.5);
        assertEquals(-0.5, h.get(0), EPS);
        assertEquals(-0.5, h.get(1), EPS);
        assertEquals(0.0, h.get(2), EPS);
        assertEquals(0.5, h.get(3), EPS);
        assertEquals(0.5, h.get(4), EPS);
    }

    @Test
    void testClamp() {
        IDoubleVector x = IDoubleVector.of(new double[]{-1.0, 0.0, 1.0});
        IDoubleVector c = x.clamp(-0.5, 0.5);
        assertEquals(-0.5, c.get(0), EPS);
        assertEquals(0.0, c.get(1), EPS);
        assertEquals(0.5, c.get(2), EPS);
    }

    @Test
    void testNeg() {
        IDoubleVector x = IDoubleVector.of(new double[]{2.0, -3.0, 0.0});
        IDoubleVector n = x.neg();
        assertEquals(-2.0, n.get(0), EPS);
        assertEquals(3.0, n.get(1), EPS);
        assertEquals(0.0, n.get(2), EPS);
    }

    @Test
    void testRsub() {
        IDoubleVector x = IDoubleVector.of(new double[]{2.0, 3.0, 5.0});
        IDoubleVector r = x.rsub(10.0);
        assertEquals(8.0, r.get(0), EPS);
        assertEquals(7.0, r.get(1), EPS);
        assertEquals(5.0, r.get(2), EPS);
    }

    @Test
    void testRdiv() {
        IDoubleVector x = IDoubleVector.of(new double[]{2.0, 4.0, 5.0});
        IDoubleVector r = x.rdiv(10.0);
        assertEquals(5.0, r.get(0), EPS);
        assertEquals(2.5, r.get(1), EPS);
        assertEquals(2.0, r.get(2), EPS);
    }

    @Test
    void testLayerNorm() {
        IDoubleVector x = IDoubleVector.of(new double[]{1.0, 2.0, 3.0});
        IDoubleVector gamma = IDoubleVector.of(new double[]{1.0, 1.0, 1.0});
        IDoubleVector beta = IDoubleVector.of(new double[]{0.0, 0.0, 0.0});
        IDoubleVector ln = x.layerNorm(gamma, beta, 1e-5);
        double mean = x.meanValue();
        double std = x.stdValue();
        assertEquals((1.0 - mean) / (std + 1e-5), ln.get(0), EPS);
        assertEquals((3.0 - mean) / (std + 1e-5), ln.get(2), EPS);
    }

    @Test
    void testDropout() {
        IDoubleVector x = IDoubleVector.of(new double[]{1.0, 2.0, 3.0});
        IDoubleVector d = x.dropout(0.5);
        assertEquals(1.0, d.get(0), EPS);
        assertEquals(2.0, d.get(1), EPS);
        assertEquals(3.0, d.get(2), EPS);
    }
}
