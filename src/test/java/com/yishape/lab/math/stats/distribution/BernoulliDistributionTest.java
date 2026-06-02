package com.yishape.lab.math.stats.distribution;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link BernoulliDistribution}.
 */
class BernoulliDistributionTest {

    private static final double EPS = 1e-9;

    // ==================== Constructor ====================

    @Test
    void constructor_valid() {
        BernoulliDistribution d = new BernoulliDistribution(0.3);
        assertEquals(0.3, d.getProbability(), EPS);
    }

    @Test
    void constructor_negative_throws() {
        assertThrows(IllegalArgumentException.class, () -> new BernoulliDistribution(-0.1));
    }

    @Test
    void constructor_aboveOne_throws() {
        assertThrows(IllegalArgumentException.class, () -> new BernoulliDistribution(1.1));
    }

    // ==================== Moments ====================

    @Test
    void mean() {
        assertEquals(0.3, new BernoulliDistribution(0.3).mean(), EPS);
    }

    @Test
    void var() {
        assertEquals(0.21, new BernoulliDistribution(0.3).var(), EPS);
    }

    @Test
    void std() {
        assertEquals(Math.sqrt(0.21), new BernoulliDistribution(0.3).std(), 1e-4);
    }

    @Test
    void median_lessThanHalf() {
        assertEquals(0, new BernoulliDistribution(0.3).median(), EPS);
    }

    @Test
    void median_greaterThanHalf() {
        assertEquals(1, new BernoulliDistribution(0.7).median(), EPS);
    }

    @Test
    void median_equalHalf() {
        assertEquals(0.5, new BernoulliDistribution(0.5).median(), EPS);
    }

    @Test
    void mode_lessThanHalf() {
        assertEquals(0, new BernoulliDistribution(0.3).mode(), EPS);
    }

    @Test
    void mode_greaterThanHalf() {
        assertEquals(1, new BernoulliDistribution(0.7).mode(), EPS);
    }

    // ==================== PMF & CDF ====================

    @Test
    void pmf_at1() {
        assertEquals(0.3, new BernoulliDistribution(0.3).pmf(1), EPS);
    }

    @Test
    void pmf_at0() {
        assertEquals(0.7, new BernoulliDistribution(0.3).pmf(0), EPS);
    }

    @Test
    void pmf_outOfSupport() {
        assertEquals(0, new BernoulliDistribution(0.3).pmf(2), EPS);
    }

    @Test
    void cdf_atNeg() {
        assertEquals(0, new BernoulliDistribution(0.3).cdf(-1), EPS);
    }

    @Test
    void cdf_at0() {
        assertEquals(0.7, new BernoulliDistribution(0.3).cdf(0), EPS);
    }

    @Test
    void cdf_at1() {
        assertEquals(1, new BernoulliDistribution(0.3).cdf(1), EPS);
    }

    // ==================== PPF & ISF ====================

    @Test
    void ppf_belowQ_returns0() {
        assertEquals(0, new BernoulliDistribution(0.3).ppf(0.5), EPS);
    }

    @Test
    void ppf_aboveQ_returns1() {
        assertEquals(1, new BernoulliDistribution(0.3).ppf(0.8), EPS);
    }

    @Test
    void isf() {
        assertEquals(1, new BernoulliDistribution(0.3).isf(0.2), EPS);
        assertEquals(0, new BernoulliDistribution(0.3).isf(0.9), EPS);
    }

    // ==================== Support ====================

    @Test
    void minSupport() {
        assertEquals(0, new BernoulliDistribution(0.5).getMinSupport());
    }

    @Test
    void maxSupport() {
        assertEquals(1, new BernoulliDistribution(0.5).getMaxSupport());
    }

    @Test
    void isInSupport() {
        BernoulliDistribution d = new BernoulliDistribution(0.5);
        assertTrue(d.isInSupport(0));
        assertTrue(d.isInSupport(1));
        assertFalse(d.isInSupport(2));
    }

    @Test
    void isBounded() {
        assertTrue(new BernoulliDistribution(0.5).isBounded());
    }

    // ==================== Entropy ====================

    @Test
    void entropy_maxAtHalf() {
        double h = new BernoulliDistribution(0.5).entropy();
        assertEquals(Math.log(2), h, 1e-4);
    }

    @Test
    void entropy_zeroAtExtremes() {
        assertEquals(0, new BernoulliDistribution(0).entropy(), EPS);
        assertEquals(0, new BernoulliDistribution(1).entropy(), EPS);
    }

    // ==================== Divergences ====================

    @Test
    void klDivergence_sameDistribution() {
        BernoulliDistribution p = new BernoulliDistribution(0.3);
        assertEquals(0, p.klDivergence(p), EPS);
    }

    @Test
    void jsDivergence_symmetric() {
        BernoulliDistribution p = new BernoulliDistribution(0.3);
        BernoulliDistribution q = new BernoulliDistribution(0.7);
        assertEquals(p.jsDivergence(q), q.jsDivergence(p), EPS);
    }

    @Test
    void wassersteinDistance() {
        BernoulliDistribution p = new BernoulliDistribution(0.3);
        BernoulliDistribution q = new BernoulliDistribution(0.7);
        assertEquals(0.4, p.wassersteinDistance(q), EPS);
    }

    // ==================== Symmetry & Memoryless ====================

    @Test
    void isSymmetric_atHalf() {
        assertTrue(new BernoulliDistribution(0.5).isSymmetric());
    }

    @Test
    void isSymmetric_notAtHalf() {
        assertFalse(new BernoulliDistribution(0.3).isSymmetric());
    }

    @Test
    void isMemoryless() {
        assertFalse(new BernoulliDistribution(0.5).isMemoryless());
    }

    // ==================== Sampling ====================

    @Test
    void sample_only0And1() {
        BernoulliDistribution d = new BernoulliDistribution(0.5);
        int[] samples = d.sample(1000);
        for (int s : samples) {
            assertTrue(s == 0 || s == 1);
        }
    }

    @Test
    void sample_meanApproximatesP() {
        BernoulliDistribution d = new BernoulliDistribution(0.3);
        int[] samples = d.sample(10000);
        double sum = 0;
        for (int s : samples) sum += s;
        assertEquals(0.3, sum / samples.length, 0.05);
    }

    // ==================== Identity ====================

    @Test
    void equals_sameP() {
        assertEquals(new BernoulliDistribution(0.3), new BernoulliDistribution(0.3));
    }

    @Test
    void equals_differentP() {
        assertNotEquals(new BernoulliDistribution(0.3), new BernoulliDistribution(0.7));
    }

    @Test
    void hashCode_same() {
        assertEquals(new BernoulliDistribution(0.3).hashCode(), new BernoulliDistribution(0.3).hashCode());
    }

    @Test
    void name() {
        assertEquals("Bernoulli", new BernoulliDistribution(0.5).getDistributionName());
    }

    @Test
    void parameterInfo() {
        assertNotNull(new BernoulliDistribution(0.5).getParameterInfo());
    }
}
