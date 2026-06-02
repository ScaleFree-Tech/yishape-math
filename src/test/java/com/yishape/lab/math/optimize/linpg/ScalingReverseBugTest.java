package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.highs.HighsLinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the reverse exponent scaling bug.
 *
 * The bug: column scaling multiplies coefficients by 2^exp (via updateExponent).
 * This means x_original = x_tableau * 2^exp.
 * But reverseExponentScaling computes x_tableau * 2^(-exp) — the reciprocal.
 *
 * Seed=42 reliably triggers column scaling, exposing this bug.
 */
public class ScalingReverseBugTest {

    @Test
    @DisplayName("SRB1: Seed=42 EQ form — scaling ON fails, scaling OFF works")
    void testSeed42EqFormScalingOnVsOff() {
        int n = 5, m = 3;
        Random rand = new Random(42);

        double[] cArr = new double[n + m];
        for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
        for (int i = n; i < n + m; i++) cArr[i] = 0;

        double[][] aArr = new double[m][n + m];
        double[] bArr = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) aArr[i][j] = rand.nextDouble() * 10 + 0.1;
            for (int j = 0; j < m; j++) aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
            bArr[i] = rand.nextDouble() * 50 + 10;
        }
        IMatrix A_eq = Linalg.matrix(aArr);
        IVector b_eq = Linalg.vector(bArr);
        IVector c = Linalg.vector(cArr);

        // Get HiGHS reference
        HighsLinProgSolver highs = new HighsLinProgSolver();
        OptResult hr = highs.solve(c, null, null, A_eq, b_eq);

        // Case A: Scaling OFF
        RereSimplexLinProgSolver solverOff = new RereSimplexLinProgSolver();
        solverOff.setUseNumericalScaling(false);
        OptResult rOff = solverOff.solve(c, null, null, A_eq, b_eq);

        // Case B: Scaling ON (default)
        RereSimplexLinProgSolver solverOn = new RereSimplexLinProgSolver();
        solverOn.setUseNumericalScaling(true);
        OptResult rOn = solverOn.solve(c, null, null, A_eq, b_eq);

        double relDiffOff = Math.abs(hr.getOptimalValue() - rOff.getOptimalValue()) / Math.abs(hr.getOptimalValue());
        double relDiffOn = Math.abs(hr.getOptimalValue() - rOn.getOptimalValue()) / Math.abs(hr.getOptimalValue());

        System.out.println("HiGHS:  " + hr.getOptimalValue());
        System.out.println("Java OFF: " + rOff.getOptimalValue() + " (relDiff=" + (relDiffOff * 100) + "%)");
        System.out.println("Java ON:  " + rOn.getOptimalValue() + " (relDiff=" + (relDiffOn * 100) + "%)");

        // Scaling OFF should match HiGHS
        assertTrue(relDiffOff < 1e-6, "Without scaling, should match HiGHS (relDiff=" + relDiffOff + ")");

        // After fix: scaling ON should also match HiGHS
        assertTrue(relDiffOn < 1e-6,
            "After fix, scaling ON should also match HiGHS. relDiff=" + relDiffOn);

        // Check feasibility with scaling ON
        if (rOn.isConverged()) {
            IVector sol = rOn.getOptimalPoint();
            boolean feasible = true;
            for (int i = 0; i < m && feasible; i++) {
                double lhs = 0;
                for (int j = 0; j < sol.length(); j++) lhs += A_eq.get(i, j) * sol.get(j);
                if (Math.abs(lhs - b_eq.get(i)) > 1e-4) feasible = false;
            }
            assertTrue(feasible, "Solution should be feasible with scaling ON after fix");
        }
    }

    @Test
    @DisplayName("SRB2: Seed=42 LEQ form — scaling ON fails, scaling OFF works")
    void testSeed42LeqFormScalingOnVsOff() {
        int n = 5, m = 3;
        Random rand = new Random(42);

        double[] cArr = new double[n];
        for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
        IVector c = Linalg.vector(cArr);

        double[][] aArr = new double[m][n];
        double[] bArr = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) aArr[i][j] = rand.nextDouble() * 10 + 0.1;
            bArr[i] = rand.nextDouble() * 50 + 10;
        }
        IMatrix A_ub = Linalg.matrix(aArr);
        IVector b_ub = Linalg.vector(bArr);

        HighsLinProgSolver highs = new HighsLinProgSolver();
        OptResult hr = highs.solve(c, A_ub, b_ub);

        RereSimplexLinProgSolver solverOff = new RereSimplexLinProgSolver();
        solverOff.setUseNumericalScaling(false);
        OptResult rOff = solverOff.solve(c, A_ub, b_ub);

        RereSimplexLinProgSolver solverOn = new RereSimplexLinProgSolver();
        solverOn.setUseNumericalScaling(true);
        OptResult rOn = solverOn.solve(c, A_ub, b_ub);

        double relDiffOff = Math.abs(hr.getOptimalValue() - rOff.getOptimalValue()) / Math.abs(hr.getOptimalValue());
        double relDiffOn = Math.abs(hr.getOptimalValue() - rOn.getOptimalValue()) / Math.abs(hr.getOptimalValue());

        System.out.println("HiGHS:  " + hr.getOptimalValue());
        System.out.println("Java OFF: " + rOff.getOptimalValue() + " (relDiff=" + (relDiffOff * 100) + "%)");
        System.out.println("Java ON:  " + rOn.getOptimalValue() + " (relDiff=" + (relDiffOn * 100) + "%)");

        assertTrue(relDiffOff < 1e-6, "Without scaling, should match HiGHS");
        assertTrue(relDiffOn < 1e-6,
            "After fix, scaling ON should also match HiGHS. relDiff=" + relDiffOn);
    }

    @Test
    @DisplayName("SRB3: Seed=123 should work fine with or without scaling")
    void testSeed123WorksWithScaling() {
        int n = 10, m = 5;
        Random rand = new Random(123);

        double[] cArr = new double[n + m];
        for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
        for (int i = n; i < n + m; i++) cArr[i] = 0;

        double[][] aArr = new double[m][n + m];
        double[] bArr = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) aArr[i][j] = rand.nextDouble() * 10 + 0.1;
            for (int j = 0; j < m; j++) aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
            bArr[i] = rand.nextDouble() * 50 + 10;
        }
        IMatrix A_eq = Linalg.matrix(aArr);
        IVector b_eq = Linalg.vector(bArr);
        IVector c = Linalg.vector(cArr);

        HighsLinProgSolver highs = new HighsLinProgSolver();
        OptResult hr = highs.solve(c, null, null, A_eq, b_eq);

        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setUseNumericalScaling(true);
        OptResult javaResult = solver.solve(c, null, null, A_eq, b_eq);

        double relDiff = Math.abs(hr.getOptimalValue() - javaResult.getOptimalValue()) / Math.abs(hr.getOptimalValue());
        System.out.println("Seed 123: HiGHS=" + hr.getOptimalValue() + " Java=" + javaResult.getOptimalValue() +
                           " relDiff=" + (relDiff * 100) + "%");

        // Seed 123 works fine even with scaling, because scaling is not triggered
        // for this seed's coefficient distribution
        assertTrue(relDiff < 0.01, "Seed 123 should work fine (relDiff=" + relDiff + ")");
    }
}
