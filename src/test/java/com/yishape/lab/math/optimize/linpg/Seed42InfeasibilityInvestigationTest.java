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
 * Deep investigation of seed=42 infeasibility bug.
 * The solver converges but produces solutions that violate constraints heavily.
 * Only specific seeds trigger this — most seeds produce perfect results.
 */
public class Seed42InfeasibilityInvestigationTest {

    private static final double TOLERANCE = 1e-4;

    @Test
    @DisplayName("D1: n=10, m=5 with verbose output — trace Phase I → Phase II")
    void testN10M5Seed42Verbose() {
        int n = 10;
        int m = 5;
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

        System.out.println("=== D1: n=10, m=5, seed=42 VERBOSE ===");
        System.out.println("A_eq dims: " + A_eq.rows() + "x" + A_eq.cols());
        System.out.println("c dims: " + c.length());
        System.out.println("b_eq: " + b_eq);

        // Solve with verbose
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        OptResult javaResult = solver.solve(c, null, null, A_eq, b_eq);

        HighsLinProgSolver highsSolver = new HighsLinProgSolver();
        OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);

        System.out.println("\n=== Results ===");
        System.out.println("Java converged: " + javaResult.isConverged());
        System.out.println("Java iterations: " + javaResult.getIterations());
        System.out.println("Java objective: " + javaResult.getOptimalValue());
        System.out.println("HiGHS objective: " + highsResult.getOptimalValue());

        IVector javaSol = javaResult.getOptimalPoint();
        if (javaResult.isConverged()) {
            // Check constraint satisfaction for each constraint
            System.out.println("\nConstraint check (A_eq * x vs b_eq):");
            int feasibleCount = 0;
            int violatedCount = 0;
            for (int i = 0; i < A_eq.rows(); i++) {
                double lhs = 0;
                for (int j = 0; j < javaSol.length(); j++) {
                    lhs += A_eq.get(i, j) * javaSol.get(j);
                }
                double expected = b_eq.get(i);
                double diff = Math.abs(lhs - expected);
                if (diff > TOLERANCE) {
                    System.out.println("  Row " + i + ": LHS=" + lhs + " vs RHS=" + expected + " (diff=" + diff + ") ← VIOLATION");
                    violatedCount++;
                } else {
                    feasibleCount++;
                }
            }

            System.out.println("\nStatus: " + feasibleCount + " constraints OK, " + violatedCount + " violated");

            // Show non-zero variables
            System.out.println("\nNon-zero variables in solution:");
            int nonZeroCount = 0;
            for (int j = 0; j < javaSol.length(); j++) {
                if (Math.abs(javaSol.get(j)) > TOLERANCE) {
                    System.out.println("  x[" + j + "] = " + javaSol.get(j));
                    nonZeroCount++;
                }
            }
            System.out.println("Total non-zero vars: " + nonZeroCount + " (expected <= " + m + " basic vars)");
        }
    }

    @Test
    @DisplayName("D2: Test seed=42 vs other seeds for n=10, m=5 — pattern analysis")
    void testSeed42VsNearbySeeds() {
        System.out.println("=== D2: Seed=42 vs nearby seeds, n=10, m=5 ===");

        int n = 10, m = 5;
        java.util.List<Integer> problematicSeeds = new java.util.ArrayList<>();

        for (int seed = 40; seed <= 50; seed++) {
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

            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            OptResult javaResult = solver.solve(c, null, null, A_eq, b_eq);

            HighsLinProgSolver highsSolver = new HighsLinProgSolver();
            OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);

            double javaOpt = javaResult.getOptimalValue();
            double highsOpt = highsResult.getOptimalValue();
            double relDiff = Math.abs(highsOpt - javaOpt) / Math.abs(highsOpt);

            // Check feasibility
            boolean feasible = true;
            if (javaResult.isConverged()) {
                IVector sol = javaResult.getOptimalPoint();
                for (int i = 0; i < m && feasible; i++) {
                    double lhs = 0;
                    for (int j = 0; j < sol.length(); j++) {
                        lhs += A_eq.get(i, j) * sol.get(j);
                    }
                    if (Math.abs(lhs - b_eq.get(i)) > TOLERANCE) {
                        feasible = false;
                    }
                }
            } else {
                feasible = false;
            }

            String status = feasible ? "OK" : "INFEASIBLE";
            System.out.println("Seed " + seed + ": Java=" + String.format("%.4f", javaOpt) +
                               ", HiGHS=" + String.format("%.4f", highsOpt) +
                               ", relDiff=" + String.format("%.2f", relDiff*100) + "%" +
                               ", " + status);

            if (!feasible || relDiff > 0.01) {
                problematicSeeds.add(seed);
            }
        }

        System.out.println("\nProblematic seeds: " + problematicSeeds);
    }

    @Test
    @DisplayName("D3: Vary problem dimensions with seed=42")
    void testDimensionSensitivity() {
        System.out.println("=== D3: Dimension sensitivity, seed=42 ===");

        int[][] configs = {{5,2}, {5,3}, {5,4}, {8,3}, {8,4}, {10,3}, {10,4}, {10,5}, {15,5}, {15,8}};

        for (int[] config : configs) {
            int n = config[0];
            int m = config[1];
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

            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            OptResult javaResult = solver.solve(c, null, null, A_eq, b_eq);

            HighsLinProgSolver highsSolver = new HighsLinProgSolver();
            OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);

            double javaOpt = javaResult.getOptimalValue();
            double highsOpt = highsResult.getOptimalValue();
            double relDiff = Math.abs(highsOpt - javaOpt) / Math.max(1e-10, Math.abs(highsOpt));

            // Check feasibility
            boolean feasible = true;
            if (javaResult.isConverged()) {
                IVector sol = javaResult.getOptimalPoint();
                for (int i = 0; i < m && feasible; i++) {
                    double lhs = 0;
                    for (int j = 0; j < sol.length(); j++) {
                        lhs += A_eq.get(i, j) * sol.get(j);
                    }
                    if (Math.abs(lhs - b_eq.get(i)) > TOLERANCE) {
                        feasible = false;
                    }
                }
            }

            String status = feasible ? "OK" : "INFEASIBLE";
            System.out.println("n=" + n + ", m=" + m + ": Java=" + String.format("%.4f", javaOpt) +
                               ", HiGHS=" + String.format("%.4f", highsOpt) +
                               ", relDiff=" + String.format("%.2f", relDiff*100) + "%" +
                               ", iters=" + javaResult.getIterations() +
                               ", " + status);
        }
    }

    @Test
    @DisplayName("D4: Check if problem is the seed or the generated coefficients")
    void testSeed42WithDifferentRandoms() {
        System.out.println("=== D4: Seed=42 with different random generation patterns ===");

        int n = 10, m = 5;

        // Pattern 1: Original pattern (c: negative, A: positive)
        {
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
            OptResult r = solver.solve(c, null, null, A_eq, b_eq);
            System.out.println("Pattern 1 (original neg c): obj=" + r.getOptimalValue() + ", converged=" + r.isConverged());
        }

        // Pattern 2: Positive c (minimization directly), A has positive coefficients
        {
            Random rand = new Random(42);
            double[] cArr = new double[n + m];
            for (int i = 0; i < n; i++) cArr[i] = rand.nextDouble() * 10 + 1; // positive
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
            OptResult r = solver.solve(c, null, null, A_eq, b_eq);
            System.out.println("Pattern 2 (positive c): obj=" + r.getOptimalValue() + ", converged=" + r.isConverged());
        }

        // Pattern 3: All coefficients from 0-1 range
        {
            Random rand = new Random(42);
            double[] cArr = new double[n + m];
            for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() + 0.1);
            for (int i = n; i < n + m; i++) cArr[i] = 0;
            IVector c = Linalg.vector(cArr);
            double[][] aArr = new double[m][n + m];
            double[] bArr = new double[m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) aArr[i][j] = rand.nextDouble() + 0.1;
                for (int j = 0; j < m; j++) aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
                bArr[i] = rand.nextDouble() * 5 + 1;
            }
            IMatrix A_eq = Linalg.matrix(aArr);
            IVector b_eq = Linalg.vector(bArr);

            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            OptResult r = solver.solve(c, null, null, A_eq, b_eq);
            System.out.println("Pattern 3 (0-1 range): obj=" + r.getOptimalValue() + ", converged=" + r.isConverged());
        }

        // Pattern 4: Integer coefficients
        {
            Random rand = new Random(42);
            double[] cArr = new double[n + m];
            for (int i = 0; i < n; i++) cArr[i] = -(rand.nextInt(10) + 1);
            for (int i = n; i < n + m; i++) cArr[i] = 0;
            IVector c = Linalg.vector(cArr);
            double[][] aArr = new double[m][n + m];
            double[] bArr = new double[m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) aArr[i][j] = rand.nextInt(10) + 1;
                for (int j = 0; j < m; j++) aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
                bArr[i] = rand.nextInt(50) + 10;
            }
            IMatrix A_eq = Linalg.matrix(aArr);
            IVector b_eq = Linalg.vector(bArr);

            RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
            OptResult r = solver.solve(c, null, null, A_eq, b_eq);
            System.out.println("Pattern 4 (integer): obj=" + r.getOptimalValue() + ", converged=" + r.isConverged());
        }
    }

    @Test
    @DisplayName("D5: Phase I diagnostic — does Phase I correctly find feasible solution?")
    void testPhaseIFeasibilityForSeed42() {
        System.out.println("=== D5: Phase I diagnostic for seed=42, n=10, m=5 ===");

        int n = 10, m = 5;
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

        // The problem has positive RHS and all constraints have at least one positive coefficient.
        // Adding slack variables with identity matrix should make this trivially feasible
        // (set original vars to 0, slack vars to RHS values).
        System.out.println("Trivial feasible solution: x[0.." + (n-1) + "]=0, s[0.." + (m-1) + "]=b_eq");
        System.out.println("b_eq: " + b_eq);

        // Use maximize directly (bypass solve's negate) to see raw behavior
        IVector cMax = c.multiplyByScalar(-1.0);
        RereSimplexLinProgSolver solver = new RereSimplexLinProgSolver();
        solver.setVerbose(true);
        OptResult maxResult = solver.maximize(cMax, null, null, A_eq, b_eq, null);

        System.out.println("\nMaximize result:");
        System.out.println("Converged: " + maxResult.isConverged());
        System.out.println("Objective: " + maxResult.getOptimalValue());
        System.out.println("Iterations: " + maxResult.getIterations());

        if (maxResult.isConverged()) {
            IVector sol = maxResult.getOptimalPoint();
            System.out.println("Solution non-zero entries:");
            for (int j = 0; j < sol.length(); j++) {
                if (Math.abs(sol.get(j)) > TOLERANCE) {
                    System.out.println("  [" + j + "] = " + sol.get(j));
                }
            }

            System.out.println("\nConstraint verification:");
            for (int i = 0; i < m; i++) {
                double lhs = 0;
                for (int j = 0; j < sol.length(); j++) {
                    lhs += A_eq.get(i, j) * sol.get(j);
                }
                System.out.println("  Row " + i + ": " + lhs + " vs " + b_eq.get(i) +
                                   " (diff=" + Math.abs(lhs - b_eq.get(i)) + ")");
            }
        }
    }
}
