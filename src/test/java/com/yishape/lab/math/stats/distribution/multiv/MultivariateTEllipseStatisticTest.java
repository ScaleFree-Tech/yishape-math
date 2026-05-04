package com.yishape.lab.math.stats.distribution.multiv;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 二元边际多元 t 置信椭圆：Mahalanobis² 阈值应为 2·F_{2,ν−2}^{-1}，ν→∞ 时趋于 χ²₂。
 */
class MultivariateTEllipseStatisticTest {

    @Test
    void largeNuMahalanobisSquaredApproachesChiSquare2() {
        double conf = 0.95;
        // 真 χ²₂(0.95) ≈ 5.991；Wilson–Hilferty 的 chiSquareQuantile 略偏低，故与闭式 F/Beta 阈值比对文献值。
        double chi2Exact = 5.991464547107983;
        double mahalT = MultivariateDistributionMath.mahalanobisSquaredBivariateTMarginal(conf, 400);
        assertEquals(chi2Exact, mahalT, 0.05);
    }

    @Test
    void moderateNuDiffersFromChiSquare() {
        double conf = 0.95;
        double chi2 = MultivariateDistributionMath.chiSquareQuantile(conf, 2);
        double mahalT = MultivariateDistributionMath.mahalanobisSquaredBivariateTMarginal(conf, 10);
        assertTrue(Math.abs(mahalT - chi2) > 0.5, "finite ν should inflate Mahalanobis² vs χ²₂");
    }
}
