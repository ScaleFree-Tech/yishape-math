package com.yishape.lab.math.stats.distribution;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link BetaDistribution}.
 */
class BetaDistributionTest {

    private static final double EPS = 1e-4;

    // ==================== Constructor ====================

    @Test
    void constructor_validParams() {
        BetaDistribution d = new BetaDistribution(2, 3);
        assertEquals(2, d.getAlpha());
        assertEquals(3, d.getBeta());
    }

    @Test
    void constructor_zeroAlpha_throws() {
        assertThrows(IllegalArgumentException.class, () -> new BetaDistribution(0, 1));
    }

    @Test
    void constructor_negativeBeta_throws() {
        assertThrows(IllegalArgumentException.class, () -> new BetaDistribution(1, -1));
    }

    // ==================== Moments ====================

    @Test
    void mean() {
        BetaDistribution d = new BetaDistribution(2, 3);
        assertEquals(0.4, d.mean(), EPS);
    }

    @Test
    void var() {
        BetaDistribution d = new BetaDistribution(2, 3);
        double expected = 2.0 * 3.0 / (25.0 * 6.0);
        assertEquals(expected, d.var(), EPS);
    }

    @Test
    void symmetricDistribution() {
        BetaDistribution d = new BetaDistribution(5, 5);
        assertEquals(0.5, d.mean(), EPS);
    }

    // ==================== PDF ====================

    @Test
    void pdf_atZero_alphaLessThan1_infinity() {
        BetaDistribution d = new BetaDistribution(0.5, 1);
        assertTrue(Double.isInfinite(d.pdf(0)));
    }

    @Test
    void pdf_atOne_betaLessThan1_infinity() {
        BetaDistribution d = new BetaDistribution(1, 0.5);
        assertTrue(Double.isInfinite(d.pdf(1)));
    }

    @Test
    void pdf_atHalf_symmetric() {
        BetaDistribution d = new BetaDistribution(5, 5);
        assertEquals(d.pdf(0.3), d.pdf(0.7), EPS);
    }

    @Test
    void pdf_positive() {
        BetaDistribution d = new BetaDistribution(2, 3);
        assertTrue(d.pdf(0.5) > 0);
    }

    // ==================== CDF ====================

    @Test
    void cdf_atZero_isZero() {
        BetaDistribution d = new BetaDistribution(2, 3);
        assertEquals(0, d.cdf(0), EPS);
    }

    @Test
    void cdf_atOne_isOne() {
        BetaDistribution d = new BetaDistribution(2, 3);
        assertEquals(1, d.cdf(1), EPS);
    }

    @Test
    void cdf_symmetric() {
        BetaDistribution d = new BetaDistribution(5, 5);
        assertEquals(d.cdf(0.3), 1 - d.cdf(0.7), EPS);
    }

    // ==================== PPF ====================

    @Test
    void ppf_atHalf_isMedian() {
        BetaDistribution d = new BetaDistribution(2, 3);
        assertEquals(d.median(), d.ppf(0.5), 1e-3);
    }

    @Test
    void ppf_roundtrip() {
        BetaDistribution d = new BetaDistribution(3, 4);
        for (double p = 0.1; p < 1; p += 0.2) {
            assertEquals(p, d.cdf(d.ppf(p)), 1e-3);
        }
    }

    @Test
    void ppf_rangeIsZeroToOne() {
        BetaDistribution d = new BetaDistribution(2, 3);
        assertTrue(d.ppf(0.01) >= 0);
        assertTrue(d.ppf(0.99) <= 1);
    }

    // ==================== Sampling ====================

    @Test
    void sample_inRange() {
        BetaDistribution d = new BetaDistribution(2, 3);
        double[] samples = d.sample(500);
        for (double s : samples) {
            assertTrue(s >= 0 && s <= 1, "Beta samples should be in [0,1]");
        }
    }

    @Test
    void sample_meanApproximates() {
        BetaDistribution d = new BetaDistribution(2, 3);
        double[] samples = d.sample(10000);
        double mean = 0;
        for (double s : samples) mean += s;
        mean /= samples.length;
        assertEquals(d.mean(), mean, 0.05);
    }
}
