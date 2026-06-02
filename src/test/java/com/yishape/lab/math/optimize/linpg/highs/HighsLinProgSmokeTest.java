package com.yishape.lab.math.optimize.linpg.highs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.RereIntegerProg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;

/**
 * 在原生可用时校验 HiGHS 路径；不可用时 {@link HighsLinProgSolver} / {@link HighsIntegerProg} 应回退且仍给出可行结果。
 */
class HighsLinProgSmokeTest {

    @Test
    void highsLinProg_agreesWithFallbackOrFallbackSucceeds() {
        var c = Linalg.vector(new double[] { 1.0 });
        var aEq = Linalg.matrix(new double[][] { { 1.0 } });
        var bEq = Linalg.vector(new double[] { 1.0 });
        var init = Linalg.vector(new double[] { 1.0 });

        OptResult base = new RereSimplexLinProgSolver().solve(c, null, null, aEq, bEq, init);
        OptResult highs = new HighsLinProgSolver().solve(c, null, null, aEq, bEq, init);

        assertTrue(base.isConverged());
        assertTrue(highs.isConverged());
        assertEquals(base.getOptimalValue(), highs.getOptimalValue(), 1e-6);
        assertEquals((Double) base.getOptimalPoint().get(0), (Double) highs.getOptimalPoint().get(0), 1e-6);
    }

    @Test
    void highsIntegerProg_matchesPureBnBOrFallback() {
        var c = Linalg.vector(new double[] { 1.0, 1.0 });
        var aUb = Linalg.matrix(new double[][] { { 1.0, 1.0 } });
        var bUb = Linalg.vector(new double[] { 2.0 });

        var ref = new RereIntegerProg();
        ref.setAllVariablesInteger();

        var hi = new HighsIntegerProg();
        hi.setAllVariablesInteger();

        OptResult r0 = ref.solve(c, aUb, bUb, null, null);
        OptResult r1 = hi.solve(c, aUb, bUb, null, null);
        assertTrue(r0.isConverged());
        assertTrue(r1.isConverged());
        assertEquals(r0.getOptimalValue(), r1.getOptimalValue(), 1e-4);
    }
}
