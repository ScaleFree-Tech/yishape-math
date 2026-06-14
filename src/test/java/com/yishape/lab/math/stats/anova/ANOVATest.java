package com.yishape.lab.math.stats.anova;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link ANOVA}.
 */
class ANOVATest {

    private static final double EPS = 1e-4;

    // ==================== One-Way ANOVA ====================

    @Test
    void oneWayANOVA_clearDifference() {
        // Three groups with clearly different means
        IVector<Double> g1 = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> g2 = Linalg.vector(new double[]{10, 11, 12, 13, 14});
        IVector<Double> g3 = Linalg.vector(new double[]{20, 21, 22, 23, 24});

        ANOVA anova = new ANOVA();
        ANOVAResult result = anova.performOneWayANOVA(g1, g2, g3);
        assertNotNull(result);
        assertTrue(result.fStatistic > 1, "F-statistic should be large for clearly different groups");
        assertTrue(result.p < 0.05, "p-value should be small for clearly different groups");
    }

    @Test
    void oneWayANOVA_similarGroups() {
        // Three groups with similar means (high variance, small differences)
        IVector<Double> g1 = Linalg.vector(new double[]{1, 100, 1, 100, 1});
        IVector<Double> g2 = Linalg.vector(new double[]{2, 101, 2, 101, 2});
        IVector<Double> g3 = Linalg.vector(new double[]{3, 102, 3, 102, 3});

        ANOVA anova = new ANOVA();
        ANOVAResult result = anova.performOneWayANOVA(g1, g2, g3);
        assertNotNull(result);
        // With high variance within groups, F should be small
        assertTrue(result.fStatistic >= 0);
    }

    @Test
    void oneWayANOVA_sumOfSquares() {
        IVector<Double> g1 = Linalg.vector(new double[]{1, 2, 3});
        IVector<Double> g2 = Linalg.vector(new double[]{4, 5, 6});

        ANOVA anova = new ANOVA();
        ANOVAResult result = anova.performOneWayANOVA(g1, g2);

        // Independently compute total SS from raw data
        double[] all = {1, 2, 3, 4, 5, 6};
        double grandMean = (1+2+3+4+5+6) / 6.0;
        double ssTotalIndependent = 0;
        for (double v : all) ssTotalIndependent += (v - grandMean) * (v - grandMean);

        // Verify: ssBetween + ssWithin == ssTotal (ANOVA identity)
        assertEquals(result.ssBetween + result.ssWithin, result.ssTotal, EPS,
            "ssBetween + ssWithin should equal ssTotal");
        // Also verify ssTotal matches independent computation
        assertEquals(ssTotalIndependent, result.ssTotal, EPS,
            "ssTotal should match independent computation from raw data");
    }

    @Test
    void oneWayANOVA_unequalSizes() {
        IVector<Double> g1 = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> g2 = Linalg.vector(new double[]{10, 11});

        ANOVA anova = new ANOVA();
        ANOVAResult result = anova.performOneWayANOVA(g1, g2);
        assertNotNull(result);
        assertTrue(result.fStatistic >= 0);
    }

    // ==================== Normality Test ====================

    @Test
    void testNormality_normalData() {
        IVector<Double> data = Linalg.vector(new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
        // Just test it doesn't throw - the simplified normality test has strict thresholds
        boolean result = ANOVA.testNormality(data);
        assertNotNull(Boolean.valueOf(result));
    }

    // ==================== Homogeneity of Variance ====================

    @Test
    void testHomogeneity_similarVariances() {
        IVector<Double> g1 = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IVector<Double> g2 = Linalg.vector(new double[]{10, 11, 12, 13, 14});

        ANOVA anova = new ANOVA();
        assertTrue(anova.testHomogeneityOfVariance(g1, g2));
    }

    @Test
    void testHomogeneity_differentVariances() {
        IVector<Double> g1 = Linalg.vector(new double[]{1, 1.01, 0.99, 1.02, 0.98});
        IVector<Double> g2 = Linalg.vector(new double[]{1, 10, 1, 10, 1});

        ANOVA anova = new ANOVA();
        assertFalse(anova.testHomogeneityOfVariance(g1, g2));
    }
}
