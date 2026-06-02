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
 * Targeted tests to isolate the seed=42 infeasibility root cause.
 */
public class Seed42RootCauseIsolationTest {

    @Test
    @DisplayName("RC1: Test seed=42 with scaling disabled")
    void testSeed42NoScaling() {
        int n = 5, m = 3;
        seed42Problem(n, m, false, true, "scaling=OFF, degeneracy=ON");
        seed42Problem(n, m, false, false, "scaling=OFF, degeneracy=OFF");
        seed42Problem(n, m, true, false, "scaling=ON, degeneracy=OFF");
        seed42Problem(n, m, true, true, "scaling=ON, degeneracy=ON");
    }

    private void seed42Problem(int n, int m, boolean scaling, boolean degeneracy, String label) {
        Random rand = new Random(42);
        double[] cArr = new double[n + m];
        for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
        for (int i = n; i < n + m; i++) cArr[i] = 0;
        IVector c = Linalg.vector(cArr);

        double[][] aArr = new double[m][n + m];
        double[] bArr = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) aArr[i][j] = rand.nextDouble() * 10 + 0.1;
            for (int j = 0; j < m; j++) aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
            bArr[i] = rand.nextDouble() * 50 + 10;
        }
        IMatrix A_eq = Linalg.matrix(aArr);
        IVector b_eq = Linalg.vector(bArr);

        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setUseNumericalScaling(scaling);
        solver.setUseAdvancedDegeneracyHandling(degeneracy);
        OptResult r = solver.solve(c, null, null, A_eq, b_eq);

        HighsLinProgSolver highs = new HighsLinProgSolver();
        OptResult hr = highs.solve(c, null, null, A_eq, b_eq);

        boolean feasible = true;
        if (r.isConverged()) {
            IVector sol = r.getOptimalPoint();
            for (int i = 0; i < m && feasible; i++) {
                double lhs = 0;
                for (int j = 0; j < sol.length(); j++) lhs += A_eq.get(i, j) * sol.get(j);
                if (Math.abs(lhs - b_eq.get(i)) > 1e-4) feasible = false;
            }
        } else {
            feasible = false;
        }
        double relDiff = Math.abs(hr.getOptimalValue() - r.getOptimalValue()) / Math.max(1e-10, Math.abs(hr.getOptimalValue()));
        System.out.println("  " + label + ": obj=" + r.getOptimalValue() + " vs HiGHS=" + hr.getOptimalValue() +
                           ", relDiff=" + (relDiff*100) + "%, feasible=" + feasible + ", iters=" + r.getIterations());
    }

    @Test
    @DisplayName("RC2: Test with LEQ form — compare scaling on/off")
    void testLeqFormSeed42ScalingVariants() {
        System.out.println("=== RC2: LEQ form, seed=42, n=5, m=3 ===");
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

        for (boolean scaling : new boolean[]{true, false}) {
            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            solver.setUseNumericalScaling(scaling);
            OptResult r = solver.solve(c, A_ub, b_ub);

            HighsLinProgSolver highs = new HighsLinProgSolver();
            OptResult hr = highs.solve(c, A_ub, b_ub);

            double relDiff = Math.abs(hr.getOptimalValue() - r.getOptimalValue()) / Math.max(1e-10, Math.abs(hr.getOptimalValue()));
            System.out.println("  scaling=" + scaling + ": obj=" + r.getOptimalValue() +
                               " vs HiGHS=" + hr.getOptimalValue() +
                               ", relDiff=" + (relDiff*100) + "%, iters=" + r.getIterations());
        }
    }

    @Test
    @DisplayName("RC3: Print the exact objective coefficients and A matrix for seed=42 n=5")
    void testPrintSeed42Coefficients() {
        System.out.println("=== RC3: Exact coefficients, seed=42, n=5, m=3 ===");
        int n = 5, m = 3;
        Random rand = new Random(42);

        System.out.println("Objective coefficients (minimization):");
        for (int i = 0; i < n + m; i++) {
            double val;
            if (i < n) {
                val = -(rand.nextDouble() * 10 + 1);
            } else {
                val = 0;
            }
            System.out.println("  c[" + i + "] = " + val);
        }

        rand = new Random(42);
        double[] cArr = new double[n + m];
        for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
        for (int i = n; i < n + m; i++) cArr[i] = 0;

        System.out.println("\nConstraint matrix A_eq and RHS b_eq:");
        for (int i = 0; i < m; i++) {
            System.out.print("  Row " + i + ": [");
            for (int j = 0; j < n; j++) {
                double a = rand.nextDouble() * 10 + 0.1;
                System.out.print(String.format("%.15f", a));
                if (j < n - 1) System.out.print(", ");
            }
            System.out.print(" |");
            for (int j = 0; j < m; j++) {
                System.out.print(String.format(" %.1f", (i == j) ? 1.0 : 0.0));
            }
            double b = rand.nextDouble() * 50 + 10;
            System.out.println("] = " + b);
        }

        // Verify: solve via maximize directly to see tableau-level behavior
        System.out.println("\nSolving via maximize directly (c negated for maximize):");
        IVector c = Linalg.vector(cArr);
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        OptResult r = solver.maximize(c.multiplyByScalar(-1.0), null, null,
            Linalg.matrix(new double[m][n+m]), Linalg.vector(new double[m]), null);
        // This won't work because we need the actual A_eq matrix
    }

    @Test
    @DisplayName("RC4: Compare LEQ form with EQ form for seed=42 — shows problem is in EQ form only?")
    void testLeqVsEqSameProblem() {
        System.out.println("=== RC4: Compare LEQ vs EQ for seed=42, n=5, m=3 ===");

        // LEQ form: only decision vars in A, solver adds slacks
        int n = 5, m = 3;
        Random rand = new Random(42);

        double[] cLeq = new double[n];
        for (int i = 0; i < n; i++) cLeq[i] = -(rand.nextDouble() * 10 + 1);

        double[][] aLeq = new double[m][n];
        double[] bLeq = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) aLeq[i][j] = rand.nextDouble() * 10 + 0.1;
            bLeq[i] = rand.nextDouble() * 50 + 10;
        }

        RereSimplexLinProgSolver leqSolver = new RereSimplexLinProgSolver();
        OptResult leqR = leqSolver.solve(Linalg.vector(cLeq), Linalg.matrix(aLeq), Linalg.vector(bLeq));
        HighsLinProgSolver highs = new HighsLinProgSolver();
        OptResult leqH = highs.solve(Linalg.vector(cLeq), Linalg.matrix(aLeq), Linalg.vector(bLeq));

        System.out.println("LEQ form: Java=" + leqR.getOptimalValue() + " (iters=" + leqR.getIterations() +
                           "), HiGHS=" + leqH.getOptimalValue());

        // EQ form: same problem but with embedded slacks
        rand = new Random(42);
        double[] cEq = new double[n + m];
        for (int i = 0; i < n; i++) cEq[i] = -(rand.nextDouble() * 10 + 1);
        for (int i = n; i < n + m; i++) cEq[i] = 0;

        double[][] aEq = new double[m][n + m];
        double[] bEq = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) aEq[i][j] = rand.nextDouble() * 10 + 0.1;
            for (int j = 0; j < m; j++) aEq[i][n + j] = (i == j) ? 1.0 : 0.0;
            bEq[i] = rand.nextDouble() * 50 + 10;
        }

        RereSimplexLinProgSolver eqSolver = new RereSimplexLinProgSolver();
        OptResult eqR = eqSolver.solve(Linalg.vector(cEq), null, null, Linalg.matrix(aEq), Linalg.vector(bEq));
        OptResult eqH = highs.solve(Linalg.vector(cEq), null, null, Linalg.matrix(aEq), Linalg.vector(bEq));

        System.out.println("EQ form: Java=" + eqR.getOptimalValue() + " (iters=" + eqR.getIterations() +
                           "), HiGHS=" + eqH.getOptimalValue());
        System.out.println("LEQ HiGHS=" + leqH.getOptimalValue() + " vs EQ HiGHS=" + eqH.getOptimalValue() +
                           " (should be same problem)");

        // Check: are LEQ and EQ really the same problem?
        // LEQ: min c^T x s.t. A*x <= b (solver adds slacks internally)
        // EQ:  min c^T x s.t. A*x + s = b, x,s >= 0
        // These ARE the same problem! HiGHS should give the same answer.
        assertEquals(leqH.getOptimalValue(), eqH.getOptimalValue(), 1e-2,
            "LEQ and EQ forms should be equivalent for HiGHS");
    }
}
