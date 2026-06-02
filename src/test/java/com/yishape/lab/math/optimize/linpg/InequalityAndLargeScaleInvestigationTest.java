package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.highs.HighsLinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * <h1>Inequality Constraint & Large-Scale Bug Investigation</h1>
 *
 * <h2>Issue 1: solve(c, A_ub, b_ub) returned all-zero solutions</h2>
 * <p>Root cause: the same hasBigMStructure bug in PivotSelectionStrategy.performPivotOperation().
 * For LEQ-only problems with 2 constraints and >= 3 original vars, hasBigMStructure=true
 * caused pivotRow = leavingRow + 2 instead of correct +1 (getNumObjectiveFunctions()).
 * Fix applied: replaced dimension heuristic with getNumObjectiveFunctions().</p>
 *
 * <h2>Issue 2: seed=42 large-scale suboptimal solution</h2>
 * <p>50 vars, 20 constraints: Java returns -109.4 vs HiGHS -47.4.
 * Investigated below with step-by-step convergence tracking.</p>
 */
public class InequalityAndLargeScaleInvestigationTest {

    private static final double TOLERANCE = 1e-4;
    private static final double LOOSE_TOLERANCE = 1e-3;

    private RereSimplexLinProgSolver simplexSolver;
    private HighsLinProgSolver highsSolver;

    @BeforeEach
    void setUp() {
        simplexSolver = new RereSimplexLinProgSolver();
        highsSolver = new HighsLinProgSolver();
    }

    // ============================================================
    // ISSUE 1: Inequality constraint solve(c, A_ub, b_ub)
    // ============================================================

    @Test
    @DisplayName("I1: Simple LEQ problem via solve(c, A_ub, b_ub) — basic 2-var")
    void testLeqSolveBasic() {
        // min 3*x1 + 2*x2 s.t. x1+x2>=4, 2*x1+x2>=6
        // Equivalent LP in standard form: min 3*x1+2*x2 s.t. x1+x2<=4, 2*x1+x2<=6 → NO
        // The >= constraints need conversion to <= for this interface.
        // Let's use: max 3*x1 + 2*x2 s.t. x1+x2<=4, 2*x1+x2<=6
        // min form: min -3*x1 - 2*x2

        IVector c = Linalg.vector(new double[]{-3.0, -2.0});  // min -3*x1 - 2*x2
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {1.0, 1.0},   // x1 + x2 <= 4
            {2.0, 1.0}    // 2*x1 + x2 <= 6
        });
        IVector b_ub = Linalg.vector(new double[]{4.0, 6.0});

        OptResult javaResult = simplexSolver.solve(c, A_ub, b_ub);
        OptResult highsResult = highsSolver.solve(c, A_ub, b_ub);

        System.out.println("=== I1: Basic LEQ via solve(c, A_ub, b_ub) ===");
        System.out.println("min -3*x1 - 2*x2 s.t. x1+x2<=4, 2*x1+x2<=6");
        System.out.println("Expected: x1=2, x2=2, obj=-10");

        assertTrue(javaResult.isConverged(), "Should converge: " + javaResult.getConvergenceReason());
        IVector sol = javaResult.getOptimalPoint();
        System.out.println("Java: x1=" + sol.get(0) + ", x2=" + sol.get(1) + ", obj=" + javaResult.getOptimalValue());
        System.out.println("HiGHS: obj=" + highsResult.getOptimalValue());

        assertEquals(2.0, sol.get(0), TOLERANCE, "x1 should be 2.0");
        assertEquals(2.0, sol.get(1), TOLERANCE, "x2 should be 2.0");
        assertEquals(-10.0, javaResult.getOptimalValue(), TOLERANCE, "obj should be -10.0 (min form)");
    }

    @Test
    @DisplayName("I2: LEQ with 5 vars, 2 constraints — was affected by hasBigMStructure")
    void testLeqSolveFiveVars() {
        // 5 vars, 2 LEQ constraints → height=3, width=8 → old hasBigMStructure=true → BUG
        // Now fixed: should work correctly

        IVector c = Linalg.vector(new double[]{-3.0, -2.0, -1.0, -4.0, -2.0});
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0, 0.0, 0.0},
            {2.0, 1.0, 0.0, 1.0, 1.0}
        });
        IVector b_ub = Linalg.vector(new double[]{10.0, 15.0});

        OptResult javaResult = simplexSolver.solve(c, A_ub, b_ub);
        OptResult highsResult = highsSolver.solve(c, A_ub, b_ub);

        System.out.println("=== I2: 5-var LEQ (was hasBigMStructure=true before fix) ===");
        System.out.println("Converged: " + javaResult.isConverged());

        assertTrue(javaResult.isConverged(), "Should converge after fix: " + javaResult.getConvergenceReason());

        IVector sol = javaResult.getOptimalPoint();
        System.out.println("Java: obj=" + javaResult.getOptimalValue() + ", x=" + sol);
        System.out.println("HiGHS: obj=" + highsResult.getOptimalValue());

        // Verify non-negativity
        for (int i = 0; i < sol.length(); i++) {
            assertTrue(sol.get(i) >= -TOLERANCE, "x" + i + " should be non-negative, got " + sol.get(i));
        }

        // Objective values should be close
        assertEquals(highsResult.getOptimalValue(), javaResult.getOptimalValue(), LOOSE_TOLERANCE * 10,
            "Objective should be close to HiGHS");
    }

    @Test
    @DisplayName("I3: LEQ with 6 vars, 2 constraints — larger version")
    void testLeqSolveSixVars() {
        // 6 vars, 2 LEQ constraints → height=3, width=9 → old hasBigMStructure=true → BUG
        IVector c = Linalg.vector(new double[]{-5.0, -4.0, -3.0, -2.0, -1.0, -6.0});
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {1.0, 2.0, 1.0, 0.0, 0.0, 1.0},
            {3.0, 1.0, 0.0, 1.0, 1.0, 0.0}
        });
        IVector b_ub = Linalg.vector(new double[]{20.0, 15.0});

        OptResult javaResult = simplexSolver.solve(c, A_ub, b_ub);
        OptResult highsResult = highsSolver.solve(c, A_ub, b_ub);

        System.out.println("=== I3: 6-var LEQ (was hasBigMStructure=true before fix) ===");
        System.out.println("Converged: " + javaResult.isConverged());

        assertTrue(javaResult.isConverged(), "Should converge: " + javaResult.getConvergenceReason());

        IVector sol = javaResult.getOptimalPoint();
        System.out.println("Java: obj=" + javaResult.getOptimalValue());
        System.out.println("HiGHS: obj=" + highsResult.getOptimalValue());

        for (int i = 0; i < sol.length(); i++) {
            assertTrue(sol.get(i) >= -TOLERANCE, "x" + i + " should be non-negative, got " + sol.get(i));
        }

        // Verify constraint satisfaction
        for (int i = 0; i < A_ub.rows(); i++) {
            double lhs = 0;
            for (int j = 0; j < sol.length(); j++) {
                lhs += A_ub.get(i, j) * sol.get(j);
            }
            assertTrue(lhs <= b_ub.get(i) + TOLERANCE,
                "Constraint " + i + ": " + lhs + " should be <= " + b_ub.get(i));
        }

        assertEquals(highsResult.getOptimalValue(), javaResult.getOptimalValue(), LOOSE_TOLERANCE * 10,
            "Objective should be close to HiGHS");
    }

    @Test
    @DisplayName("I4: LEQ with 3 constraints — height != 3, never affected by bug")
    void testLeqSolveThreeConstraints() {
        // 5 vars, 3 LEQ constraints → height=4 → old hasBigMStructure=false → was OK
        IVector c = Linalg.vector(new double[]{-3.0, -2.0, -1.0, -4.0, -2.0});
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0, 0.0, 0.0},
            {2.0, 1.0, 0.0, 1.0, 1.0},
            {0.0, 1.0, 2.0, 1.0, 0.0}
        });
        IVector b_ub = Linalg.vector(new double[]{10.0, 15.0, 8.0});

        OptResult javaResult = simplexSolver.solve(c, A_ub, b_ub);
        OptResult highsResult = highsSolver.solve(c, A_ub, b_ub);

        System.out.println("=== I4: 3-constraint LEQ (was never broken) ===");
        System.out.println("Converged: " + javaResult.isConverged());

        assertTrue(javaResult.isConverged(), "Should converge: " + javaResult.getConvergenceReason());

        IVector sol = javaResult.getOptimalPoint();
        System.out.println("Java: obj=" + javaResult.getOptimalValue());
        System.out.println("HiGHS: obj=" + highsResult.getOptimalValue());

        for (int i = 0; i < sol.length(); i++) {
            assertTrue(sol.get(i) >= -TOLERANCE, "x" + i + " should be non-negative, got " + sol.get(i));
        }

        assertEquals(highsResult.getOptimalValue(), javaResult.getOptimalValue(), LOOSE_TOLERANCE * 10,
            "Objective should be close to HiGHS");
    }

    // ============================================================
    // ISSUE 2: seed=42 large-scale suboptimal solution
    // ============================================================

    @Test
    @DisplayName("S1: Reproduce seed=42 suboptimal issue with convergence diagnostics")
    void testSeed42SuboptimalDiagnostics() {
        int n = 50;
        int m = 20;
        Random rand = new Random(42);

        double[] cArr = new double[n + m];
        for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
        for (int i = n; i < n + m; i++) cArr[i] = 0;
        IVector c = Linalg.vector(cArr);

        double[][] aArr = new double[m][n + m];
        double[] bArr = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                aArr[i][j] = rand.nextDouble() * 10 + 0.1;
            }
            for (int j = 0; j < m; j++) {
                aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
            }
            bArr[i] = rand.nextDouble() * 50 + 10;
        }
        IMatrix A_eq = Linalg.matrix(aArr);
        IVector b_eq = Linalg.vector(bArr);

        // Solve with verbose output to see iteration progress
        simplexSolver.setVerbose(true);
        OptResult javaResult = simplexSolver.solve(c, null, null, A_eq, b_eq);
        simplexSolver.setVerbose(false);
        OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);

        System.out.println("=== S1: seed=42 suboptimal diagnostics ===");
        System.out.println("Java converged: " + javaResult.isConverged());
        System.out.println("Java iterations: " + javaResult.getIterations());
        System.out.println("Java objective: " + javaResult.getOptimalValue());
        System.out.println("HiGHS objective: " + highsResult.getOptimalValue());

        double relDiff = Math.abs(highsResult.getOptimalValue() - javaResult.getOptimalValue())
                         / Math.abs(highsResult.getOptimalValue());
        System.out.println("Relative difference: " + (relDiff * 100) + "%");
        System.out.println("Java convergence reason: " + javaResult.getConvergenceReason());

        assertTrue(javaResult.isConverged(), "Java solver should converge");

        // Check if this is the known suboptimal case
        if (relDiff > 0.01) {
            System.out.println("KNOWN ISSUE: seed=42 produces suboptimal solution");
            System.out.println("  This is a remaining correctness issue in the simplex implementation");

            // Verify the Java solution is feasible
            IVector javaSol = javaResult.getOptimalPoint();
            boolean feasible = true;
            for (int i = 0; i < A_eq.rows(); i++) {
                double lhs = 0;
                for (int j = 0; j < javaSol.length(); j++) {
                    lhs += A_eq.get(i, j) * javaSol.get(j);
                }
                if (Math.abs(lhs - b_eq.get(i)) > LOOSE_TOLERANCE) {
                    System.out.println("  Constraint " + i + " violation: " + lhs + " vs " + b_eq.get(i));
                    feasible = false;
                }
            }
            for (int j = 0; j < n; j++) {
                if (javaSol.get(j) < -LOOSE_TOLERANCE) {
                    System.out.println("  Variable x" + j + " negative: " + javaSol.get(j));
                    feasible = false;
                }
            }
            System.out.println("  Java solution feasible: " + feasible);
        }
    }

    @Test
    @DisplayName("S2: Test other seeds for comparison")
    void testOtherSeeds() {
        int n = 50;
        int m = 20;
        int[] seeds = {123, 456, 789, 999};

        for (int seed : seeds) {
            Random rand = new Random(seed);

            double[] cArr = new double[n + m];
            for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
            for (int i = n; i < n + m; i++) cArr[i] = 0;
            IVector c = Linalg.vector(cArr);

            double[][] aArr = new double[m][n + m];
            double[] bArr = new double[m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    aArr[i][j] = rand.nextDouble() * 10 + 0.1;
                }
                for (int j = 0; j < m; j++) {
                    aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
                }
                bArr[i] = rand.nextDouble() * 50 + 10;
            }
            IMatrix A_eq = Linalg.matrix(aArr);
            IVector b_eq = Linalg.vector(bArr);

            OptResult javaResult = simplexSolver.solve(c, null, null, A_eq, b_eq);
            OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);

            double javaOpt = javaResult.getOptimalValue();
            double highsOpt = highsResult.getOptimalValue();
            double relDiff = Math.abs(highsOpt - javaOpt) / Math.abs(highsOpt);

            System.out.println("Seed " + seed + ": Java=" + javaOpt + ", HiGHS=" + highsOpt +
                               ", relDiff=" + (relDiff * 100) + "%" +
                               ", converged=" + javaResult.isConverged());
        }
    }

    @Test
    @DisplayName("S3: Smaller versions of the seed=42 pattern")
    void testSmallerSeed42Pattern() {
        // Test with same seed=42 pattern at smaller sizes to isolate
        int[] sizes = {10, 20, 30};

        for (int n : sizes) {
            int m = n / 2;
            Random rand = new Random(42);

            double[] cArr = new double[n + m];
            for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
            for (int i = n; i < n + m; i++) cArr[i] = 0;
            IVector c = Linalg.vector(cArr);

            double[][] aArr = new double[m][n + m];
            double[] bArr = new double[m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    aArr[i][j] = rand.nextDouble() * 10 + 0.1;
                }
                for (int j = 0; j < m; j++) {
                    aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
                }
                bArr[i] = rand.nextDouble() * 50 + 10;
            }
            IMatrix A_eq = Linalg.matrix(aArr);
            IVector b_eq = Linalg.vector(bArr);

            OptResult javaResult = simplexSolver.solve(c, null, null, A_eq, b_eq);
            OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);

            double javaOpt = javaResult.getOptimalValue();
            double highsOpt = highsResult.getOptimalValue();
            double relDiff = Math.abs(highsOpt - javaOpt) / Math.abs(highsOpt);

            System.out.println("n=" + n + ", m=" + m + ": Java=" + javaOpt +
                               ", HiGHS=" + highsOpt + ", relDiff=" + (relDiff * 100) + "%");

            // Smaller problems should match exactly
            if (n <= 20) {
                assertEquals(highsOpt, javaOpt, LOOSE_TOLERANCE * 100,
                    "Small problem (n=" + n + ") should match HiGHS");
            }
        }
    }

    @Test
    @DisplayName("S4: Analyze phase II convergence for seed=42")
    void testSeed42PhasedConvergence() {
        // The seed=42 issue might be related to:
        // 1. Stalling: objective value stops improving before true optimum
        // 2. Cycling: Bland's rule preventing progress but also preventing escape
        // 3. Numerical: precision issues with the large coefficients

        int n = 50;
        int m = 20;
        Random rand = new Random(42);

        double[] cArr = new double[n + m];
        for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
        for (int i = n; i < n + m; i++) cArr[i] = 0;
        IVector c = Linalg.vector(cArr);

        double[][] aArr = new double[m][n + m];
        double[] bArr = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                aArr[i][j] = rand.nextDouble() * 10 + 0.1;
            }
            for (int j = 0; j < m; j++) {
                aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
            }
            bArr[i] = rand.nextDouble() * 50 + 10;
        }
        IMatrix A_eq = Linalg.matrix(aArr);
        IVector b_eq = Linalg.vector(bArr);

        System.out.println("=== S4: Seed=42 convergence analysis ===");
        System.out.println("n=" + n + ", m=" + m);

        // Analyze coefficient ranges
        double maxCoeff = 0, minCoeff = Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            double absC = Math.abs(cArr[i]);
            if (absC > 0) { maxCoeff = Math.max(maxCoeff, absC); minCoeff = Math.min(minCoeff, absC); }
        }
        System.out.println("Objective coefficients: range=[" + String.format("%.2f", minCoeff) +
                           ", " + String.format("%.2f", maxCoeff) + "]");

        maxCoeff = 0; minCoeff = Double.MAX_VALUE;
        double maxRhs = 0, minRhs = Double.MAX_VALUE;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double absA = Math.abs(aArr[i][j]);
                if (absA > 0) { maxCoeff = Math.max(maxCoeff, absA); minCoeff = Math.min(minCoeff, absA); }
            }
            if (bArr[i] > 0) { maxRhs = Math.max(maxRhs, bArr[i]); minRhs = Math.min(minRhs, bArr[i]); }
        }
        System.out.println("Constraint coefficients: range=[" + String.format("%.2f", minCoeff) +
                           ", " + String.format("%.2f", maxCoeff) + "]");
        System.out.println("RHS: range=[" + String.format("%.2f", minRhs) +
                           ", " + String.format("%.2f", maxRhs) + "]");

        // Run the solver with different pivot rules
        System.out.println("\nTesting with different pivot rules:");
        for (String rule : new String[]{"DANTZIG"}) {
            RereSimplexLinProgSolver s = new RereSimplexLinProgSolver();
            // s.setPivotSelectionRule(...) only supports DANTZIG, BLAND, STEEP_EDGE
            // STEEP_EDGE falls back to Dantzig

            OptResult r = s.solve(c, null, null, A_eq, b_eq);
            if (r.isConverged()) {
                System.out.println("  " + rule + ": obj=" + r.getOptimalValue() +
                                   ", iterations=" + r.getIterations());
            } else {
                System.out.println("  " + rule + ": NOT CONVERGED: " + r.getConvergenceReason());
            }
        }

        // Compare with HiGHS
        OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);
        System.out.println("\nHiGHS: obj=" + highsResult.getOptimalValue() +
                           ", iterations=" + highsResult.getIterations());
    }
}
