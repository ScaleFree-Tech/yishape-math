package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive tests for InteriorPointLinProgSolver and LangMultiplierLinProgSolver.
 * Covers edge cases, null handling, larger problems, convergence behavior, and performance.
 */
public class InteriorAndMultiplierComprehensiveTest {

    private static final double TOL = 1e-4;
    private static final double TOL_LOOSE = 1e-2;

    // =========================================================================
    // InteriorPointLinProgSolver - Edge Cases & Correctness
    // =========================================================================

    @Test
    void testInteriorPointNullInitX() {
        InteriorPointLinProgSolver solver = new InteriorPointLinProgSolver();
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});

        // Passing null initX should NOT throw NPE — solver should create defaults
        OptResult result = solver.solve(c, null, null, A_eq, b_eq, null);
        assertNotNull(result);
        assertNotNull(result.getOptimalPoint());
        assertEquals(2.0, result.getOptimalValue(), TOL_LOOSE);
    }

    @Test
    void testInteriorPointConvergedFlag() {
        InteriorPointLinProgSolver solver = new InteriorPointLinProgSolver();
        IVector c = Linalg.vector(new double[]{2, 3});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 2}, {2, 1}});
        IVector b_eq = Linalg.vector(new double[]{4, 5});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        assertNotNull(result);
        assertTrue(result.isConverged(), "Solver should converge on well-conditioned problem");
        assertTrue(result.getIterations() > 0, "Should use at least 1 iteration");
        assertTrue(result.getIterations() <= 100, "Should not exceed max iterations");
        assertTrue(result.getExecutionTimeMs() >= 0);
    }

    @Test
    void testInteriorPointConstraintSatisfactionTight() {
        InteriorPointLinProgSolver solver = new InteriorPointLinProgSolver();
        // x1 + x2 = 3, minimize x1 + 2*x2  => optimum at x1=3, x2=0, value=3
        IVector c = Linalg.vector(new double[]{1, 2});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{3});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        assertNotNull(result);
        IVector sol = result.getOptimalPoint();

        // Constraint satisfaction
        double sum = ((Number) sol.get(0)).doubleValue() + ((Number) sol.get(1)).doubleValue();
        assertEquals(3.0, sum, TOL, "Equality constraint must be satisfied tightly");

        // Non-negativity
        assertTrue(((Number) sol.get(0)).doubleValue() >= -1e-8, "x1 >= 0");
        assertTrue(((Number) sol.get(1)).doubleValue() >= -1e-8, "x2 >= 0");

        // Objective value should be close to 3 (x1=3, x2=0 is optimal)
        assertTrue(result.getOptimalValue() <= 3.0 + TOL_LOOSE, "Objective should be <= 3");
    }

    @Test
    void testInteriorPointLargerProblem() {
        InteriorPointLinProgSolver solver = new InteriorPointLinProgSolver();
        // 5 variables, 3 equality constraints
        // minimize x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5
        // s.t.
        //   x1 + x2 + x3 = 6
        //   x2 + x3 + x4 = 9
        //   x3 + x4 + x5 = 12
        // all xi >= 0
        // Optimal: x1 = 0, x2 = 0, x3 = 6, x4 = 3, x5 = 3, value = 3*6 + 4*3 + 5*3 = 18+12+15 = 45
        IVector c = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1, 1, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 1, 1, 1}
        });
        IVector b_eq = Linalg.vector(new double[]{6, 9, 12});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        assertNotNull(result);
        assertNotNull(result.getOptimalPoint());

        IVector sol = result.getOptimalPoint();
        // Verify constraints
        double c1 = ((Number) sol.get(0)).doubleValue() + ((Number) sol.get(1)).doubleValue() + ((Number) sol.get(2)).doubleValue();
        double c2 = ((Number) sol.get(1)).doubleValue() + ((Number) sol.get(2)).doubleValue() + ((Number) sol.get(3)).doubleValue();
        double c3 = ((Number) sol.get(2)).doubleValue() + ((Number) sol.get(3)).doubleValue() + ((Number) sol.get(4)).doubleValue();
        assertEquals(6.0, c1, TOL_LOOSE);
        assertEquals(9.0, c2, TOL_LOOSE);
        assertEquals(12.0, c3, TOL_LOOSE);

        // Non-negativity
        for (int i = 0; i < 5; i++) {
            assertTrue(((Number) sol.get(i)).doubleValue() >= -1e-8, "x" + i + " >= 0");
        }

        // Optimal value should be close to 45
        double expectedOpt = 45.0;
        assertTrue(Math.abs(result.getOptimalValue() - expectedOpt) < TOL_LOOSE * 50,
            "Optimal value near 45, got " + result.getOptimalValue());
    }

    @Test
    void testInteriorPointConvergenceSpeed() {
        InteriorPointLinProgSolver solver = new InteriorPointLinProgSolver();
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});

        long start = System.nanoTime();
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;

        assertNotNull(result);
        assertTrue(result.isConverged());
        // Simple 2-var problem should finish quickly
        assertTrue(elapsedMs < 5000, "Simple problem should finish within 5s, took " + elapsedMs + "ms");
        assertTrue(result.getIterations() < 50, "Should converge in fewer than 50 barrier iterations");
    }

    // =========================================================================
    // LangMultiplierLinProgSolver - Edge Cases & Correctness
    // =========================================================================

    @Test
    void testLangMultiplierNullInitX() {
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});

        // Passing null initX — was NPE before fix, should work now
        OptResult result = solver.solve(c, null, null, A_eq, b_eq, null);
        assertNotNull(result);
        assertNotNull(result.getOptimalPoint());
        assertEquals(2.0, result.getOptimalValue(), TOL_LOOSE);
    }

    @Test
    void testLangMultiplierConvergedFlag() {
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        IVector c = Linalg.vector(new double[]{2, 3});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 2}, {2, 1}});
        IVector b_eq = Linalg.vector(new double[]{4, 5});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        assertNotNull(result);
        assertTrue(result.getIterations() > 0, "Should use at least 1 iteration");
        // MAX_ITERATIONS=100 should bound this
        assertTrue(result.getIterations() <= 100, "Iterations should be bounded by MAX_ITERATIONS");
    }

    @Test
    void testLangMultiplierNonNegativityEnforcement() {
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        // minimize -x1 - x2 (i.e., maximize x1+x2) s.t. x1+x2=2, x1,x2>=0
        // Optimum at boundary: x1=2, x2=0 or vice versa, value=-2
        IVector c = Linalg.vector(new double[]{-1, -1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        assertNotNull(result);
        IVector sol = result.getOptimalPoint();

        // Non-negativity must hold
        for (int i = 0; i < sol.length(); i++) {
            assertTrue(((Number) sol.get(i)).doubleValue() >= -1e-8,
                "Variable " + i + " must be non-negative, got " + sol.get(i));
        }

        // Equality constraint must hold
        double sum = ((Number) sol.get(0)).doubleValue() + ((Number) sol.get(1)).doubleValue();
        assertEquals(2.0, sum, TOL);
    }

    @Test
    void testLangMultiplierLargerProblem() {
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        // Same 5-var, 3-constraint problem
        IVector c = Linalg.vector(new double[]{1, 2, 3, 4, 5});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1, 1, 1, 0, 0},
            {0, 1, 1, 1, 0},
            {0, 0, 1, 1, 1}
        });
        IVector b_eq = Linalg.vector(new double[]{6, 9, 12});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        assertNotNull(result);
        assertNotNull(result.getOptimalPoint());

        IVector sol = result.getOptimalPoint();
        // Verify constraints
        double c1 = ((Number) sol.get(0)).doubleValue() + ((Number) sol.get(1)).doubleValue() + ((Number) sol.get(2)).doubleValue();
        double c2 = ((Number) sol.get(1)).doubleValue() + ((Number) sol.get(2)).doubleValue() + ((Number) sol.get(3)).doubleValue();
        double c3 = ((Number) sol.get(2)).doubleValue() + ((Number) sol.get(3)).doubleValue() + ((Number) sol.get(4)).doubleValue();
        assertEquals(6.0, c1, TOL_LOOSE);
        assertEquals(9.0, c2, TOL_LOOSE);
        assertEquals(12.0, c3, TOL_LOOSE);

        // Non-negativity
        for (int i = 0; i < 5; i++) {
            assertTrue(((Number) sol.get(i)).doubleValue() >= -1e-8, "x" + i + " >= 0");
        }

        double expectedOpt = 45.0;
        assertTrue(Math.abs(result.getOptimalValue() - expectedOpt) < TOL_LOOSE * 50,
            "Optimal value near 45, got " + result.getOptimalValue());
    }

    @Test
    void testLangMultiplierConvergenceSpeed() {
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});

        long start = System.nanoTime();
        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;

        assertNotNull(result);
        assertTrue(result.isConverged(), "Should converge or reach iteration limit");
        assertTrue(elapsedMs < 30000, "Should finish within 30s, took " + elapsedMs + "ms");
    }

    // =========================================================================
    // Cross-Solver Consistency Tests
    // =========================================================================

    @Test
    void testBothSolversAgreeOnWellConditionedProblem() {
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{2});

        InteriorPointLinProgSolver ipSolver = new InteriorPointLinProgSolver();
        LangMultiplierLinProgSolver lmSolver = new LangMultiplierLinProgSolver();

        OptResult ipResult = ipSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        OptResult lmResult = lmSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);

        assertNotNull(ipResult);
        assertNotNull(lmResult);

        double ipVal = ipResult.getOptimalValue();
        double lmVal = lmResult.getOptimalValue();

        assertEquals(ipVal, lmVal, TOL_LOOSE,
            "Both solvers should agree: interior=" + ipVal + ", lagrange=" + lmVal);

        // Both should satisfy constraints
        IVector ipSol = ipResult.getOptimalPoint();
        IVector lmSol = lmResult.getOptimalPoint();
        double ipSum = ((Number) ipSol.get(0)).doubleValue() + ((Number) ipSol.get(1)).doubleValue();
        double lmSum = ((Number) lmSol.get(0)).doubleValue() + ((Number) lmSol.get(1)).doubleValue();
        assertEquals(2.0, ipSum, TOL_LOOSE);
        assertEquals(2.0, lmSum, TOL_LOOSE);
    }

    @Test
    void testBothSolversAgreeOnComplexProblem() {
        // minimize 2x + 3y s.t. x+2y=4, 2x+y=5, x,y>=0
        // Optimal: x=2, y=1, value=7
        IVector c = Linalg.vector(new double[]{2, 3});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 2}, {2, 1}});
        IVector b_eq = Linalg.vector(new double[]{4, 5});

        InteriorPointLinProgSolver ipSolver = new InteriorPointLinProgSolver();
        LangMultiplierLinProgSolver lmSolver = new LangMultiplierLinProgSolver();

        OptResult ipResult = ipSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        OptResult lmResult = lmSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);

        assertNotNull(ipResult);
        assertNotNull(lmResult);

        assertEquals(7.0, ipResult.getOptimalValue(), TOL_LOOSE, "Interior point should be close to 7");
        assertEquals(7.0, lmResult.getOptimalValue(), TOL_LOOSE, "Lagrange multiplier should be close to 7");

        assertEquals(ipResult.getOptimalValue(), lmResult.getOptimalValue(), TOL_LOOSE * 10,
            "Both solvers should produce similar optimal values");
    }

    // =========================================================================
    // Performance Benchmark
    // =========================================================================

    @Test
    void testPerformance10VariableProblem() {
        // 10 variables, 4 constraints — a moderate-sized LP
        int n = 10;
        int m = 4;

        double[] cArr = new double[n];
        for (int i = 0; i < n; i++) {
            cArr[i] = i + 1; // 1, 2, 3, ..., 10
        }
        IVector c = Linalg.vector(cArr);

        // Random-ish but well-conditioned constraint matrix
        double[][] aArr = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                aArr[i][j] = ((i * 7 + j * 3 + 1) % 5);
            }
        }
        IMatrix A_eq = Linalg.matrix(aArr);
        IVector b_eq = Linalg.vector(new double[]{10, 15, 20, 12});

        // Interior point
        InteriorPointLinProgSolver ipSolver = new InteriorPointLinProgSolver();
        long ipStart = System.nanoTime();
        OptResult ipResult = ipSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        long ipElapsed = (System.nanoTime() - ipStart) / 1_000_000;

        assertNotNull(ipResult);
        assertNotNull(ipResult.getOptimalPoint());

        // Verify constraints
        IVector ipSol = ipResult.getOptimalPoint();
        for (int i = 0; i < n; i++) {
            assertTrue(((Number) ipSol.get(i)).doubleValue() >= -1e-8, "IP x" + i + " >= 0");
        }

        System.out.println("[Perf] InteriorPoint 10x4: " + ipElapsed + "ms, " +
            ipResult.getIterations() + " barrier iters, obj=" + ipResult.getOptimalValue());

        // Lagrange multiplier
        LangMultiplierLinProgSolver lmSolver = new LangMultiplierLinProgSolver();
        long lmStart = System.nanoTime();
        OptResult lmResult = lmSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        long lmElapsed = (System.nanoTime() - lmStart) / 1_000_000;

        assertNotNull(lmResult);
        assertNotNull(lmResult.getOptimalPoint());

        // Verify constraints
        IVector lmSol = lmResult.getOptimalPoint();
        for (int i = 0; i < n; i++) {
            assertTrue(((Number) lmSol.get(i)).doubleValue() >= -1e-8, "LM x" + i + " >= 0");
        }

        System.out.println("[Perf] LangMultiplier 10x4: " + lmElapsed + "ms, " +
            lmResult.getIterations() + " barrier iters, obj=" + lmResult.getOptimalValue());

        // Both should be within reasonable time
        assertTrue(ipElapsed < 30000, "Interior point 10-var problem should finish within 30s");
        assertTrue(lmElapsed < 60000, "Lagrange multiplier 10-var problem should finish within 60s");

        // Optimal values should be reasonably close
        double ipObj = ipResult.getOptimalValue();
        double lmObj = lmResult.getOptimalValue();
        double relDiff = Math.abs(ipObj - lmObj) / Math.max(Math.abs(ipObj), Math.abs(lmObj));
        assertTrue(relDiff < 0.2 || Math.abs(ipObj - lmObj) < 1.0,
            "Solvers should agree within 20% relative or 1.0 absolute: " +
            "IP=" + ipObj + ", LM=" + lmObj);
    }

    // =========================================================================
    // Degenerate Problem Tests
    // =========================================================================

    @Test
    void testDegenerateProblemBothSolvers() {
        // Degenerate: redundant constraints
        // x1 + x2 = 2, 2*x1 + 2*x2 = 4, minimize x1 + x2
        // Any point on x1+x2=2 with xi>=0 is optimal, value=2
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}, {2, 2}});
        IVector b_eq = Linalg.vector(new double[]{2, 4});

        InteriorPointLinProgSolver ipSolver = new InteriorPointLinProgSolver();
        LangMultiplierLinProgSolver lmSolver = new LangMultiplierLinProgSolver();

        OptResult ipResult = ipSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        OptResult lmResult = lmSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);

        assertNotNull(ipResult);
        assertNotNull(lmResult);

        // Should both get objective ≈ 2
        assertEquals(2.0, ipResult.getOptimalValue(), TOL_LOOSE * 5,
            "IP degenerate: " + ipResult.getOptimalValue());
        assertEquals(2.0, lmResult.getOptimalValue(), TOL_LOOSE * 5,
            "LM degenerate: " + lmResult.getOptimalValue());

        // Non-negativity should hold
        IVector ipSol = ipResult.getOptimalPoint();
        IVector lmSol = lmResult.getOptimalPoint();
        assertTrue(((Number) ipSol.get(0)).doubleValue() >= -1e-8);
        assertTrue(((Number) ipSol.get(1)).doubleValue() >= -1e-8);
        assertTrue(((Number) lmSol.get(0)).doubleValue() >= -1e-8);
        assertTrue(((Number) lmSol.get(1)).doubleValue() >= -1e-8);
    }

    // =========================================================================
    // InitX Warm Start Tests
    // =========================================================================

    @Test
    void testInteriorPointWarmStart() {
        InteriorPointLinProgSolver solver = new InteriorPointLinProgSolver();
        IVector c = Linalg.vector(new double[]{2, 3});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 2}, {2, 1}});
        IVector b_eq = Linalg.vector(new double[]{4, 5});

        // Warm start near optimal (x1=1.9, x2=1.05)
        IVector initX = Linalg.vector(new double[]{1.9, 1.05});

        OptResult warmResult = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, initX);
        assertNotNull(warmResult);
        assertTrue(warmResult.isConverged());

        // Solution should be near (2, 1)
        IVector sol = warmResult.getOptimalPoint();
        assertEquals(2.0, ((Number) sol.get(0)).doubleValue(), TOL_LOOSE);
        assertEquals(1.0, ((Number) sol.get(1)).doubleValue(), TOL_LOOSE);
    }

    @Test
    void testLangMultiplierWarmStart() {
        LangMultiplierLinProgSolver solver = new LangMultiplierLinProgSolver();
        IVector c = Linalg.vector(new double[]{2, 3});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 2}, {2, 1}});
        IVector b_eq = Linalg.vector(new double[]{4, 5});

        // Warm start near optimal
        IVector initX = Linalg.vector(new double[]{1.9, 1.05});

        OptResult warmResult = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, initX);
        assertNotNull(warmResult);

        IVector sol = warmResult.getOptimalPoint();
        assertEquals(2.0, ((Number) sol.get(0)).doubleValue(), TOL_LOOSE);
        assertEquals(1.0, ((Number) sol.get(1)).doubleValue(), TOL_LOOSE);
    }

    // =========================================================================
    // Boundary: All variables at optimum are zero
    // =========================================================================

    @Test
    void testAllZeroOptimum() {
        // minimize x1 + x2 s.t. x1 + x2 = 0, x1,x2 >= 0
        // Only feasible point is (0,0), value=0
        IVector c = Linalg.vector(new double[]{1, 1});
        IMatrix A_eq = Linalg.matrix(new double[][]{{1, 1}});
        IVector b_eq = Linalg.vector(new double[]{0});

        InteriorPointLinProgSolver ipSolver = new InteriorPointLinProgSolver();
        OptResult ipResult = ipSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        assertNotNull(ipResult);
        assertTrue(ipResult.getOptimalValue() >= -TOL_LOOSE, "IP obj should be >= 0");
        assertTrue(ipResult.getOptimalValue() < 1.0, "IP obj should be small");

        LangMultiplierLinProgSolver lmSolver = new LangMultiplierLinProgSolver();
        OptResult lmResult = lmSolver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq);
        assertNotNull(lmResult);
        assertTrue(lmResult.getOptimalValue() >= -TOL_LOOSE, "LM obj should be >= 0");
        assertTrue(lmResult.getOptimalValue() < 1.0, "LM obj should be small");
    }
}
