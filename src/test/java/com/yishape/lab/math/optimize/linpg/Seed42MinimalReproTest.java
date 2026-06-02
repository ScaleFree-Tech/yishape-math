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

/**
 * Minimal reproduction of seed=42 infeasibility — n=5, m=3 (smallest failing case).
 */
public class Seed42MinimalReproTest {

    @Test
    @DisplayName("Minimal seed=42 reproduction: n=5, m=3, full verbose trace")
    void testMinimalReproVerbose() {
        int n = 5; // Small but >=5 to trigger orig bug pattern
        int m = 3; // m>=3 triggers seed=42 infeasibility
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

        System.out.println("=== Minimal Repro: n=" + n + ", m=" + m + " ===");

        // Show constraint matrix
        System.out.println("A_eq:");
        for (int i = 0; i < m; i++) {
            System.out.print("  Row " + i + ": ");
            for (int j = 0; j < n + m; j++) {
                System.out.print(String.format("%8.3f", A_eq.get(i, j)));
            }
            System.out.println(" | " + String.format("%8.3f", b_eq.get(i)));
        }

        // Check that the trivial solution (x=0, s=b) works
        System.out.println("\nTrivial feasible solution check:");
        for (int i = 0; i < m; i++) {
            double lhs = 0;
            for (int j = 0; j < n; j++) lhs += A_eq.get(i, j) * 0;
            for (int j = 0; j < m; j++) lhs += A_eq.get(i, n+j) * b_eq.get(j);
            // This should give: lhs = b[i] (since s_i is the only non-zero for row i)
            System.out.println("  Row " + i + ": with x=0,s=b: lhs=" + lhs + " expected=" + b_eq.get(i));
        }

        // Solve with full verbose
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        OptResult result = solver.solve(c, null, null, A_eq, b_eq);

        System.out.println("\nResult: converged=" + result.isConverged() +
                           ", obj=" + result.getOptimalValue() +
                           ", iters=" + result.getIterations() +
                           ", reason=" + result.getConvergenceReason());

        if (result.isConverged()) {
            IVector sol = result.getOptimalPoint();
            System.out.println("Solution:");
            for (int j = 0; j < sol.length(); j++) {
                if (Math.abs(sol.get(j)) > 1e-6) {
                    System.out.println("  [" + j + "] = " + sol.get(j));
                }
            }

            System.out.println("Constraint check:");
            int violations = 0;
            for (int i = 0; i < m; i++) {
                double lhs = 0;
                for (int j = 0; j < sol.length(); j++) {
                    lhs += A_eq.get(i, j) * sol.get(j);
                }
                double diff = Math.abs(lhs - b_eq.get(i));
                if (diff > 1e-4) {
                    System.out.println("  Row " + i + ": VIOLATION lhs=" + lhs + " expected=" + b_eq.get(i));
                    violations++;
                }
            }
            System.out.println("Violations: " + violations + "/" + m);

            // Also test with LEQ form — should work
            System.out.println("\n--- Testing same problem via LEQ form ---");
            // Convert A_eq to A_ub: need the constraint coeffs only for original vars
            // A_ub should be m×n (only decision variable coefficients, no slacks)
            IMatrix A_ub = Linalg.matrix(new double[m][n]);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    A_ub.set(i, j, A_eq.get(i, j));
                }
            }
            // For LEQ: we need x_i >= 0. But our problem has x >= 0 and s >= 0.
            // LEQ form: A_ub * x <= b_eq means the solvers adds slacks internally.
            // This is DIFFERENT from the EQ form which has pre-embedded slacks.

            RereSimplexLinProgSolver leqSolver = new RereSimplexLinProgSolver();
            IVector cOnly = Linalg.vector(new double[n]);
            for (int j = 0; j < n; j++) cOnly.set(j, c.get(j));
            OptResult leqResult = leqSolver.solve(cOnly, A_ub, b_eq);

            System.out.println("LEQ result: converged=" + leqResult.isConverged() +
                               ", obj=" + leqResult.getOptimalValue());
            if (leqResult.isConverged()) {
                IVector leqSol = leqResult.getOptimalPoint();
                System.out.println("LEQ solution: " + leqSol);
            }
        }
    }
}
