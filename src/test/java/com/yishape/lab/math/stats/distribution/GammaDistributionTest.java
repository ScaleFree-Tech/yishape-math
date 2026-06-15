package com.yishape.lab.math.stats.distribution;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link GammaDistribution}.
 */
class GammaDistributionTest {

    private static final double EPS = 1e-4;

    // ==================== Constructor ====================

    @Test
    void constructor_validParams() {
        GammaDistribution d = new GammaDistribution(2, 3);
        assertEquals(2, d.getAlpha());
        assertEquals(3, d.getBeta());
    }

    @Test
    void constructor_zeroAlpha_throws() {
        assertThrows(IllegalArgumentException.class, () -> new GammaDistribution(0, 1));
    }

    @Test
    void constructor_negativeBeta_throws() {
        assertThrows(IllegalArgumentException.class, () -> new GammaDistribution(1, -1));
    }

    // ==================== Moments ====================

    @Test
    void mean() {
        GammaDistribution d = new GammaDistribution(3, 2);
        // mean = alpha * beta = 3 * 2 = 6 (beta is scale, not rate)
        assertEquals(6.0, d.mean(), EPS);
    }

    @Test
    void var() {
        GammaDistribution d = new GammaDistribution(3, 2);
        // variance = alpha * beta^2 = 3 * 4 = 12
        assertEquals(12.0, d.var(), EPS);
    }

    @Test
    void std() {
        GammaDistribution d = new GammaDistribution(3, 2);
        // std = sqrt(alpha * beta^2) = sqrt(12)
        assertEquals(Math.sqrt(12.0), d.std(), EPS);
    }

    @Test
    void skewness() {
        GammaDistribution d = new GammaDistribution(4, 1);
        assertEquals(1.0, d.skewness(), EPS);
    }

    @Test
    void kurtosis() {
        GammaDistribution d = new GammaDistribution(3, 1);
        assertEquals(2.0, d.kurtosis(), EPS);
    }

    @Test
    void mode_alphaLessThan1() {
        GammaDistribution d = new GammaDistribution(0.5, 1);
        assertEquals(0.0, d.mode(), EPS);
    }

    @Test
    void mode_alphaGreaterThan1() {
        GammaDistribution d = new GammaDistribution(4, 2);
        assertEquals(1.5, d.mode(), EPS);
    }

    // ==================== PDF ====================

    @Test
    void pdf_atZero_alphaLessThan1_infinity() {
        GammaDistribution d = new GammaDistribution(0.5, 1);
        assertTrue(Double.isInfinite(d.pdf(0)));
    }

    @Test
    void pdf_atZero_alphaEquals1_equalsBeta() {
        GammaDistribution d = new GammaDistribution(1, 3);
        assertEquals(3.0, d.pdf(0), EPS);
    }

    @Test
    void pdf_atZero_alphaGreaterThan1_isZero() {
        GammaDistribution d = new GammaDistribution(2, 1);
        assertEquals(0.0, d.pdf(0), EPS);
    }

    @Test
    void pdf_positive() {
        GammaDistribution d = new GammaDistribution(2, 1);
        assertTrue(d.pdf(1) > 0);
    }

    // ==================== CDF ====================

    @Test
    void cdf_atZero_isZero() {
        GammaDistribution d = new GammaDistribution(2, 1);
        assertEquals(0, d.cdf(0), EPS);
    }

    @Test
    void cdf_atInfinity_isOne() {
        GammaDistribution d = new GammaDistribution(2, 1);
        assertEquals(1, d.cdf(100), 1e-4);
    }

    @Test
    void cdf_monotonicallyIncreasing() {
        GammaDistribution d = new GammaDistribution(3, 1);
        double prev = 0;
        for (double x = 0.5; x < 10; x += 0.5) {
            double curr = d.cdf(x);
            assertTrue(curr >= prev, "CDF should be monotonically increasing");
            prev = curr;
        }
    }

    // ==================== PPF ====================

    @Test
    void ppf_atHalf_isMedian() {
        GammaDistribution d = new GammaDistribution(3, 1);
        assertEquals(d.median(), d.ppf(0.5), 1e-3);
    }

    @Test
    void ppf_roundtrip() {
        GammaDistribution d = new GammaDistribution(2, 1);
        for (double p = 0.1; p < 1; p += 0.2) {
            assertEquals(p, d.cdf(d.ppf(p)), 1e-3);
        }
    }

    // ==================== Sampling ====================

    @Test
    void sample_array() {
        GammaDistribution d = new GammaDistribution(2, 1);
        double[] samples = d.sample(500);
        assertEquals(500, samples.length);
        for (double s : samples) {
            assertTrue(s >= 0, "Gamma samples should be non-negative");
        }
    }

    @Test
    void sample_meanApproximates() {
        GammaDistribution d = new GammaDistribution(3, 2);
        double[] samples = d.sample(10000);
        double mean = 0;
        for (double s : samples) mean += s;
        mean /= samples.length;
        assertEquals(d.mean(), mean, 0.2);
    }
}
