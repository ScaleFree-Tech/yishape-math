package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.OptResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInfo;

import static org.junit.jupiter.api.Assertions.*;

/**
 * <h1>Phase II Bug Diagnostic Test Suite</h1>
 *
 * <h2>Bug Summary</h2>
 * <p>
 * The root cause is in {@code PivotSelectionStrategy.performPivotOperation()} (line ~150-165):
 * it uses a dimension-based heuristic {@code hasBigMStructure = (height==3 && width>=7)}
 * to decide the pivot row offset. After {@code RereSimplexTableau.dropPhase1Objective()}
 * removes the W row and artificial variable columns, the resulting Phase II tableau has:
 * </p>
 * <ul>
 *   <li>height = 3 (1 objective row + 2 constraint rows)</li>
 *   <li>width = originalWidth - 1(W col) - numArtificialCols</li>
 * </ul>
 * <p>
 * When the problem has 2 EQ constraints and >= 5 original decision variables,
 * width >= 7 after drop, triggering {@code hasBigMStructure = true}. This causes
 * {@code pivotRow = leavingRow + 2} instead of the correct {@code leavingRow + getNumObjectiveFunctions()}
 * (= leavingRow + 1). Every pivot in Phase II operates on the WRONG constraint row,
 * corrupting the tableau and preventing convergence (iterates to MAX_ITERATIONS=10000).
 * </p>
 *
 * <h2>Bug Trigger Conditions</h2>
 * <ul>
 *   <li>Exactly 2 EQ constraints (→ height=3 after drop)</li>
 *   <li>At least 5 original decision variables (→ width >= 7 after drop)</li>
 *   <li>Solver goes through Phase I then Phase II (artificial variables needed)</li>
 * </ul>
 *
 * <h2>Why It Wasn't Caught Before</h2>
 * <ul>
 *   <li>Problems with <= 4 variables: width < 7 after drop → hasBigMStructure = false ✓</li>
 *   <li>Problems with != 2 constraints: height != 3 after drop → hasBigMStructure = false ✓</li>
 *   <li>LEQ-only problems skip Phase I entirely</li>
 * </ul>
 */
public class PhaseIIBugDiagnosticTest {

    private static final double TOLERANCE = 1e-6;

    // ============================================================
    // TEST 1: Reproduce the original failure with diagnostic output
    // ============================================================

    @Test
    @DisplayName("Reproduce: 6-var Big-M EQ problem — triggers hasBigMStructure bug in Phase II")
    void testReproduceOriginalFailure() {
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);

        double bigM = 10000.0;
        // 6 original variables: [x1, x2, s1, s2, a1, a2]
        // post-dropPhase1Objective: width = 8 (Z + 6 vars + RHS), height = 3
        // → hasBigMStructure = (3==3 && 8>=7) = TRUE ← BUG TRIGGERED
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, bigM, bigM});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 1.0, 0.0},
            {2.0, 1.0, 0.0, -1.0, 0.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);

        System.out.println("=== TEST 1: Original 6-var problem ===");
        System.out.println("Converged: " + result.isConverged());
        System.out.println("Reason: " + result.getConvergenceReason());
        System.out.println("Iterations: " + result.getIterations());

        // This test WILL fail with current code — demonstrates the bug
        if (result.isConverged()) {
            IVector sol = result.getOptimalPoint();
            System.out.println("x1=" + sol.get(0) + ", x2=" + sol.get(1));
            System.out.println("obj=" + result.getOptimalValue());
        } else {
            System.out.println("BUG CONFIRMED: Phase II reached iteration limit (10000)");
        }
    }

    // ============================================================
    // TEST 2: Same problem structure, different variable count — shows the threshold
    // ============================================================

    @Test
    @DisplayName("Threshold analysis: 4-var problem — width < 7 after drop → NO bug triggered")
    void testFourVariableProblemNoBug() {
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(false);

        double bigM = 10000.0;
        // 4 variables: [x1, x2, s1, s2] — surplus vars embedded, no pre-embedded artificials
        // But we need a feasible problem. Let's use: x1+x2>=4, 2*x1+x2>=6
        // Tableau will add 2 artificial vars for the 2 EQ constraints.
        // After dropPhase1Objective: width = Z(1) + 4 vars + RHS(1) = 6 → hasBigMStructure = (3==3 && 6>=7) = FALSE
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0},
            {2.0, 1.0, 0.0, -1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);

        System.out.println("=== TEST 2: 4-var problem (width=6 after drop → hasBigMStructure=false) ===");
        System.out.println("Converged: " + result.isConverged());
        System.out.println("Reason: " + result.getConvergenceReason());
        System.out.println("Iterations: " + result.getIterations());

        assertTrue(result.isConverged(), "4-var problem should converge (width < 7, bug does not trigger)");

        IVector sol = result.getOptimalPoint();
        System.out.println("x1=" + sol.get(0) + ", x2=" + sol.get(1) +
                           ", s1=" + sol.get(2) + ", s2=" + sol.get(3));
        System.out.println("obj=" + result.getOptimalValue());

        // Expected: x1=2, x2=2 → obj = 10 (minimization → should be ~10)
        assertEquals(2.0, sol.get(0), TOLERANCE, "x1 should be 2.0");
        assertEquals(2.0, sol.get(1), TOLERANCE, "x2 should be 2.0");
    }

    // ============================================================
    // TEST 3: 6-var problem with LEQ constraints — no Phase I → no bug
    // ============================================================

    @Test
    @DisplayName("LEQ-only 6-var problem — skips Phase I, no dropPhase1Objective → NO bug")
    void testLeqOnlyNoBug() {
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(false);

        // max 3*x1 + 2*x2 s.t. x1+x2<=4, 2*x1+x2<=6, x1,x2>=0
        // Expected: x1=2, x2=2, obj=10
        IVector c = Linalg.vector(new double[]{3.0, 2.0});
        IMatrix A_ub = Linalg.matrix(new double[][]{
            {1.0, 1.0},
            {2.0, 1.0}
        });
        IVector b_ub = Linalg.vector(new double[]{4.0, 6.0});

        // Use solve interface for minimization, or maximize directly
        OptResult result = solver.maximize(c, A_ub, b_ub, null, null, null);

        System.out.println("=== TEST 3: LEQ-only problem (no Phase I, no drop) ===");
        System.out.println("Converged: " + result.isConverged());
        System.out.println("Iterations: " + result.getIterations());

        assertTrue(result.isConverged(), "LEQ-only should converge");

        IVector sol = result.getOptimalPoint();
        System.out.println("x1=" + sol.get(0) + ", x2=" + sol.get(1));
        System.out.println("obj=" + result.getOptimalValue());

        assertEquals(2.0, sol.get(0), TOLERANCE);
        assertEquals(2.0, sol.get(1), TOLERANCE);
    }

    // ============================================================
    // TEST 4: 5-var EQ problem — border case where width = 7 exactly
    // ============================================================

    @Test
    @DisplayName("Borderline: 5-var EQ problem — width=7 after drop → hasBigMStructure=true → BUG")
    void testFiveVariableBugTriggered() {
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(false);

        double bigM = 10000.0;
        // 5 variables: [x1, x2, x3, s1, s2]
        // Problem: min 3*x1 + 2*x2 + 0*x3 s.t. x1+x2+x3>=4, 2*x1+x2>=6
        // After drop: width = Z(1) + 5 vars + RHS(1) = 7 → hasBigMStructure = TRUE
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, 0.0});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, 1.0, -1.0, 0.0},
            {2.0, 1.0, 0.0, 0.0, -1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);

        System.out.println("=== TEST 4: 5-var EQ problem (width=7 → hasBigMStructure=true) ===");
        System.out.println("Converged: " + result.isConverged());
        System.out.println("Reason: " + result.getConvergenceReason());
        System.out.println("Iterations: " + result.getIterations());

        // This SHOULD fail because width=7 triggers the bug
        if (!result.isConverged()) {
            System.out.println("BUG CONFIRMED at borderline (width=7): Phase II fails to converge");
        }
    }

    // ============================================================
    // TEST 5: 1-constraint EQ problem — height != 3 → no bug
    // ============================================================

    @Test
    @DisplayName("1-constraint 6-var EQ — height=2 after drop → hasBigMStructure=false → NO bug")
    void testOneConstraintNoBug() {
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(false);

        double bigM = 10000.0;
        // 1 EQ constraint → height=3 before drop, height=2 after drop
        // hasBigMStructure = (2==3) = FALSE
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, bigM, bigM});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 1.0, 0.0}
        });
        IVector b_eq = Linalg.vector(new double[]{4.0});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);

        System.out.println("=== TEST 5: 1-constraint 6-var EQ (height=2 → no bug) ===");
        System.out.println("Converged: " + result.isConverged());
        System.out.println("Iterations: " + result.getIterations());

        assertTrue(result.isConverged(), "1-constraint should converge (height != 3)");
    }

    // ============================================================
    // TEST 6: Exact hasBigMStructure dimension analysis
    // ============================================================

    @Test
    @DisplayName("Dimension analysis: verify the hasBigMStructure trigger formula")
    void testHasBigMStructureTriggerFormula() {
        System.out.println("=== TEST 6: hasBigMStructure Trigger Formula ===");
        System.out.println();
        System.out.println("Bug formula in PivotSelectionStrategy.performPivotOperation():");
        System.out.println("  hasBigMStructure = (tableau.getHeight() == 3) && (tableau.getWidth() >= 7)");
        System.out.println();
        System.out.println("After dropPhase1Objective():");
        System.out.println("  newHeight = constraints + 1  (1 objective row + constraints)");
        System.out.println("  newWidth = originalWidth - 1(W col) - numArtificialVariables");
        System.out.println("  originalWidth = 2(obj funcs) + nVars + nSlack + nArt + 1(RHS)");
        System.out.println("  For EQ constraints: nSlack=0, nArt=nConstraints");
        System.out.println("  newWidth = (2 + nVars + 0 + nConstraints + 1) - 1 - nConstraints");
        System.out.println("           = nVars + 2");
        System.out.println();
        System.out.println("  hasBigMStructure = (nConstraints + 1 == 3) && (nVars + 2 >= 7)");
        System.out.println("                   = (nConstraints == 2) && (nVars >= 5)");
        System.out.println();

        // Verify with actual solver behavior
        int[] varCounts = {2, 3, 4, 5, 6, 10};
        for (int nVars : varCounts) {
            boolean triggers = (nVars >= 5);
            int newWidth = nVars + 2;
            System.out.println("  nVars=" + nVars + " → newWidth=" + newWidth +
                               " → hasBigMStructure=" + (newWidth >= 7) +
                               (triggers ? " ← BUG TRIGGERED" : ""));
        }
        System.out.println();
        System.out.println("CONCLUSION: Bug triggers ONLY when nConstraints==2 AND nVars>=5");
        System.out.println("This is a narrow condition that evaded simpler test cases.");
    }

    // ============================================================
    // TEST 7: Direct proof — 6-var LEQ works, 6-var EQ fails
    // ============================================================

    @Test
    @DisplayName("Direct comparison: LEQ works, EQ fails for same 6-var problem")
    void testLeqWorksEqFails() {
        System.out.println("=== TEST 7: Same problem → LEQ path works, EQ path fails ===");

        // --- LEQ version (no Phase I) ---
        RereSimplexLinProgSolver leqSolver = new RereSimplexLinProgSolver();
        leqSolver.setVerbose(false);
        IVector cLeq = Linalg.vector(new double[]{3.0, 2.0});
        IMatrix A_ub = Linalg.matrix(new double[][]{{1.0, 1.0}, {2.0, 1.0}});
        IVector b_ub = Linalg.vector(new double[]{4.0, 6.0});
        OptResult leqResult = leqSolver.maximize(cLeq, A_ub, b_ub, null, null, null);

        System.out.println("LEQ path: converged=" + leqResult.isConverged() +
                           ", iterations=" + leqResult.getIterations());
        assertTrue(leqResult.isConverged(), "LEQ path should converge");

        // --- EQ version (triggers Phase I → Phase II → bug) ---
        RereSimplexLinProgSolver eqSolver = new RereSimplexLinProgSolver();
        eqSolver.setVerbose(false);
        double bigM = 10000.0;
        IVector cEq = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, bigM, bigM});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 1.0, 0.0},
            {2.0, 1.0, 0.0, -1.0, 0.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0});
        OptResult eqResult = eqSolver.solveWithNonNegativeEqualConstraints(cEq, A_eq, b_eq, null);

        System.out.println("EQ path: converged=" + eqResult.isConverged() +
                           ", iterations=" + eqResult.getIterations() +
                           ", reason=" + eqResult.getConvergenceReason());

        // EQ path should fail with current code
        if (!eqResult.isConverged()) {
            System.out.println("BUG DEMONSTRATED: EQ path fails while LEQ path succeeds");
            System.out.println("Root cause: hasBigMStructure=true in performPivotOperation");
            System.out.println("  after dropPhase1Objective → wrong pivotRow offset (+2 instead of +1)");
        }
    }

    // ============================================================
    // TEST 8: 3-constraint EQ — height != 3 → no bug
    // ============================================================

    @Test
    @DisplayName("3-constraint 6-var EQ — height=4 after drop → hasBigMStructure=false → NO bug")
    void testThreeConstraintsNoBug() {
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(false);

        double bigM = 10000.0;
        // 3 constraints → height=5 before drop, height=4 after drop → hasBigMStructure=false
        IVector c = Linalg.vector(new double[]{3.0, 2.0, 0.0, 0.0, 0.0, bigM, bigM, bigM});
        IMatrix A_eq = Linalg.matrix(new double[][]{
            {1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0},
            {2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0},
            {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0}
        });
        IVector b_eq = Linalg.vector(new double[]{4.0, 6.0, 1.0});

        OptResult result = solver.solveWithNonNegativeEqualConstraints(c, A_eq, b_eq, null);

        System.out.println("=== TEST 8: 3-constraint EQ (height=4 → no bug) ===");
        System.out.println("Converged: " + result.isConverged());
        System.out.println("Iterations: " + result.getIterations());

        assertTrue(result.isConverged(), "3-constraint should converge (height != 3)");
    }
}
