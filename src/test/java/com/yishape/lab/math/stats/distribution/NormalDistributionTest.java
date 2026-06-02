package com.yishape.lab.math.stats.distribution;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link NormalDistribution}.
 */
class NormalDistributionTest {

    private static final double EPS = 1e-6;

    // ==================== Constructor ====================

    @Test
    void standardNormal() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(0, d.mean(), EPS);
        assertEquals(1, d.var(), EPS);
        assertEquals(1, d.std(), EPS);
    }

    @Test
    void customNormal() {
        NormalDistribution d = new NormalDistribution(5, 2);
        assertEquals(5, d.mean(), EPS);
        assertEquals(4, d.var(), EPS);
        assertEquals(2, d.std(), EPS);
    }

    @Test
    void negativeStdDev_throws() {
        assertThrows(IllegalArgumentException.class, () -> new NormalDistribution(0, -1));
    }

    @Test
    void zeroStdDev_throws() {
        assertThrows(IllegalArgumentException.class, () -> new NormalDistribution(0, 0));
    }

    // ==================== PDF ====================

    @Test
    void pdf_atMean() {
        NormalDistribution d = new NormalDistribution();
        // 1/sqrt(2*pi)
        assertEquals(1.0 / Math.sqrt(2 * Math.PI), d.pdf(0), EPS);
    }

    @Test
    void pdf_atZero_standardNormal() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(0.3989422804014327, d.pdf(0), EPS);
    }

    @Test
    void pdf_symmetric() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(d.pdf(1), d.pdf(-1), EPS);
    }

    @Test
    void pdf_scaled() {
        NormalDistribution d = new NormalDistribution(0, 2);
        // At x=0: 1/(2*sqrt(2*pi))
        assertEquals(1.0 / (2 * Math.sqrt(2 * Math.PI)), d.pdf(0), EPS);
    }

    // ==================== CDF ====================

    @Test
    void cdf_atMean_isHalf() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(0.5, d.cdf(0), EPS);
    }

    @Test
    void cdf_atMinusInfinity_isZero() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(0, d.cdf(-10), 1e-6);
    }

    @Test
    void cdf_atPlusInfinity_isOne() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(1, d.cdf(10), 1e-6);
    }

    @Test
    void cdf_symmetric() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(d.cdf(1), 1 - d.cdf(-1), EPS);
    }

    // ==================== PPF (Quantile) ====================

    @Test
    void ppf_atHalf_isMean() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(0, d.ppf(0.5), EPS);
    }

    @Test
    void ppf_roundtrip() {
        NormalDistribution d = new NormalDistribution();
        for (double p = 0.01; p < 1; p += 0.1) {
            assertEquals(p, d.cdf(d.ppf(p)), 1e-4);
        }
    }

    @Test
    void ppf_outOfRange_throws() {
        NormalDistribution d = new NormalDistribution();
        assertThrows(IllegalArgumentException.class, () -> d.ppf(-0.1));
        assertThrows(IllegalArgumentException.class, () -> d.ppf(1.1));
    }

    // ==================== SF & ISF ====================

    @Test
    void sf_isOneMinusCdf() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(1 - d.cdf(1), d.sf(1), EPS);
    }

    @Test
    void isf_roundtrip() {
        NormalDistribution d = new NormalDistribution();
        // isf(0.5) = ppf(1-0.5) = ppf(0.5) = 0
        assertEquals(0, d.isf(0.5), 1e-6);
    }

    // ==================== Moments ====================

    @Test
    void median_equalsMean() {
        NormalDistribution d = new NormalDistribution(3, 1);
        assertEquals(d.mean(), d.median(), EPS);
    }

    @Test
    void mode_equalsMean() {
        NormalDistribution d = new NormalDistribution(3, 1);
        assertEquals(d.mean(), d.mode(), EPS);
    }

    @Test
    void skewness_isZero() {
        assertEquals(0, new NormalDistribution().skewness(), EPS);
    }

    @Test
    void kurtosis_isZero() {
        assertEquals(0, new NormalDistribution().kurtosis(), EPS);
    }

    @Test
    void q1() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(-0.6745, d.q1(), 1e-3);
    }

    @Test
    void q3() {
        NormalDistribution d = new NormalDistribution();
        assertEquals(0.6745, d.q3(), 1e-3);
    }

    // ==================== Sampling ====================

    @Test
    void sample_returnsValues() {
        NormalDistribution d = new NormalDistribution();
        double s = d.sample();
        assertTrue(Double.isFinite(s));
    }

    @Test
    void sample_array() {
        NormalDistribution d = new NormalDistribution();
        double[] samples = d.sample(100);
        assertEquals(100, samples.length);
        for (double s : samples) {
            assertTrue(Double.isFinite(s));
        }
    }

    @Test
    void sample_meanApproximatesPopulationMean() {
        NormalDistribution d = new NormalDistribution(5, 1);
        double[] samples = d.sample(10000);
        double mean = 0;
        for (double s : samples) mean += s;
        mean /= samples.length;
        assertEquals(5, mean, 0.2);
    }
}
