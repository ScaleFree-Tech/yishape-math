package com.yishape.lab.math.stats.distribution;

import com.yishape.lab.math.RereMathUtil;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Gamma 分布 PDF 在 x=0 及非有限处的边界语义（与形状参数 α 的极限一致）。
 */
class GammaDistributionPdfBoundaryTest {

    @Test
    void pdfAtZero_alphaLessThanOne_isPositiveInfinity() {
        GammaDistribution g = new GammaDistribution(0.5, 1.0);
        assertTrue(Double.isInfinite(g.pdf(0.0)) && g.pdf(0.0) > 0);
    }

    @Test
    void pdfAtZero_alphaEqualsOne_equalsRateBeta() {
        assertEquals(2.0, new GammaDistribution(1.0, 2.0).pdf(0.0), 1e-12);
        assertEquals(0.5, new GammaDistribution(1.0, 0.5).pdf(0.0), 1e-12);
    }

    @Test
    void pdfAtZero_alphaGreaterThanOne_isZero() {
        assertEquals(0.0, new GammaDistribution(2.0, 1.0).pdf(0.0), 0.0);
        assertEquals(0.0, new GammaDistribution(3.0, 3.0).pdf(0.0), 0.0);
    }

    @Test
    void pdfNegative_isZero() {
        assertEquals(0.0, new GammaDistribution(2.0, 1.0).pdf(-1.0), 0.0);
    }

    @Test
    void pdfPositiveInfinity_isZero() {
        assertEquals(0.0, new GammaDistribution(2.0, 1.0).pdf(Double.POSITIVE_INFINITY), 0.0);
    }

    @Test
    void pdfNaN_isNaN() {
        assertTrue(Double.isNaN(new GammaDistribution(2.0, 1.0).pdf(Double.NaN)));
    }

    @Test
    void pdfSmallPositive_matchesExpLogFormula() {
        GammaDistribution g = new GammaDistribution(2.0, 1.5);
        double x = 0.01;
        double logPdf = 2.0 * Math.log(1.5) - Math.log(RereMathUtil.gamma(2.0))
                + Math.log(x) - 1.5 * x;
        assertEquals(Math.exp(logPdf), g.pdf(x), 1e-10);
    }
}
