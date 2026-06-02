package com.yishape.lab.math.stats.testing;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link HypothesisTesting}.
 */
class HypothesisTestingTest {

    private final HypothesisTesting ht = new HypothesisTesting();

    // ==================== One-Sample T-Test ====================

    @Test
    void tTest_sampleFromNullHypothesis() {
        IVector<Double> sample = Linalg.vector(new double[]{4.8, 5.1, 5.0, 4.9, 5.2, 5.0, 4.7, 5.3, 5.1, 4.9});
        TestingResult result = ht.testMeanEqualWithT(5.0, sample, 0.95);
        assertNotNull(result);
        assertNotNull(result.pass);
        assertTrue(result.p > 0.05, "Should not reject H0 when sample comes from null hypothesis");
    }

    @Test
    void tTest_sampleClearlyDifferent() {
        IVector<Double> sample = Linalg.vector(new double[]{9.8, 10.1, 10.0, 9.9, 10.2, 10.0, 9.7, 10.3, 10.1, 9.9});
        TestingResult result = ht.testMeanEqualWithT(5.0, sample, 0.95);
        assertNotNull(result);
        assertFalse(result.pass, "Should reject H0 when sample mean is clearly different");
    }

    @Test
    void tTest_returnsPValue() {
        IVector<Double> sample = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        TestingResult result = ht.testMeanEqualWithT(3.0, sample, 0.95);
        assertTrue(result.p >= 0 && result.p <= 1, "p-value should be in [0,1]");
    }

    @Test
    void tTest_returnsConfidenceInterval() {
        IVector<Double> sample = Linalg.vector(new double[]{1.0, 2.0, 3.0, 4.0, 5.0});
        TestingResult result = ht.testMeanEqualWithT(3.0, sample, 0.95);
        assertNotNull(result.criticalInteval);
        assertTrue(result.criticalInteval._1 < result.criticalInteval._2,
            "CI lower should be less than upper");
    }

    // ==================== Chi-Square Variance Test ====================

    @Test
    void chi2Test_variance() {
        IVector<Double> sample = Linalg.vector(new double[]{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
        TestingResult result = ht.testVarEqualWithChi2(0.001, sample, 0.95);
        assertNotNull(result);
        assertNotNull(result.pass);
    }

    @Test
    void chi2Test_returnsPValue() {
        IVector<Double> sample = Linalg.vector(new double[]{2.0, 3.0, 4.0, 5.0, 6.0});
        TestingResult result = ht.testVarEqualWithChi2(2.0, sample, 0.95);
        assertTrue(result.p >= 0 && result.p <= 1);
    }
}
