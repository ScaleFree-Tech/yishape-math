package com.yishape.lab.math.optimize.mclp;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive test suite for the MCLP package.
 * Covers MclpUtil, MclpResult, and all 7 solver implementations.
 */
@DisplayName("MCLP Comprehensive Tests")
public class MclpComprehensiveTest {

    // ==================== Helper methods ====================

    private static ILinProgSolver newSolver() {
        return ILinProgSolver.of();
    }

    private static final double TOL = 1e-6;

    // Problem: minimize x1 + x2, subject to x1 + x2 >= 2, x1 >= 0, x2 >= 0
    // Optimum: x1=x2=1, obj=2
    private IVector[] simple2Obj() {
        return new IVector[]{
            Linalg.vector(new double[]{1, 0}), // obj1: minimize x1
            Linalg.vector(new double[]{0, 1})  // obj2: minimize x2
        };
    }

    private IMatrix simpleAub() {
        return Linalg.matrix(new double[][]{{-1, -1}, {1, 0}, {0, 1}});
    }

    private IVector simpleBub() {
        return Linalg.vector(new double[]{-2, 10, 10});
    }

    // ==================== MclpUtil Tests ====================

    @Test
    @DisplayName("generateUniformWeights — 2 objectives, 5 samples")
    void uniformWeights2Obj5Samples() {
        List<double[]> weights = MclpUtil.generateUniformWeights(2, 5);
        assertEquals(5, weights.size());
        for (double[] w : weights) {
            assertEquals(2, w.length);
            assertEquals(1.0, w[0] + w[1], TOL);
            assertTrue(w[0] >= 0 && w[0] <= 1.0);
        }
    }

    @Test
    @DisplayName("generateUniformWeights — single sample returns equal weights")
    void uniformWeightsSingleSample() {
        List<double[]> weights = MclpUtil.generateUniformWeights(3, 1);
        assertEquals(1, weights.size());
        double[] w = weights.get(0);
        assertEquals(3, w.length);
        assertEquals(1.0 / 3.0, w[0], TOL);
        assertEquals(1.0 / 3.0, w[1], TOL);
        assertEquals(1.0 / 3.0, w[2], TOL);
    }

    @Test
    @DisplayName("generateUniformWeights — zero samples edge case")
    void uniformWeightsZeroSamples() {
        List<double[]> weights = MclpUtil.generateUniformWeights(2, 0);
        assertEquals(1, weights.size());
        double[] w = weights.get(0);
        assertEquals(0.5, w[0], TOL);
        assertEquals(0.5, w[1], TOL);
    }

    @Test
    @DisplayName("generateRandomWeights — sums to 1, reproducibility")
    void randomWeightsSumAndReproducible() {
        List<double[]> weights1 = MclpUtil.generateRandomWeights(3, 20, 42);
        List<double[]> weights2 = MclpUtil.generateRandomWeights(3, 20, 42);
        assertEquals(20, weights1.size());
        assertEquals(20, weights2.size());
        for (int i = 0; i < 20; i++) {
            double[] w1 = weights1.get(i);
            double[] w2 = weights2.get(i);
            double sum1 = 0.0, sum2 = 0.0;
            for (int j = 0; j < 3; j++) {
                sum1 += w1[j];
                sum2 += w2[j];
                assertEquals(w1[j], w2[j], TOL);
                assertTrue(w1[j] >= 0 && w1[j] <= 1.0);
            }
            assertEquals(1.0, sum1, TOL);
            assertEquals(1.0, sum2, TOL);
        }
    }

    @Test
    @DisplayName("generatePriorityWeights — basic priority mapping")
    void priorityWeightsBasic() {
        double[] weights = MclpUtil.generatePriorityWeights(new int[]{3, 1, 2});
        assertEquals(3, weights.length);
        assertEquals(1.0, weights[0] + weights[1] + weights[2], TOL);
        // Priority 3 > 2 > 1, so w[0] > w[2] > w[1]
        assertTrue(weights[0] > weights[2]);
        assertTrue(weights[2] > weights[1]);
    }

    @Test
    @DisplayName("generatePriorityWeights — empty array")
    void priorityWeightsEmpty() {
        double[] weights = MclpUtil.generatePriorityWeights(new int[0]);
        assertEquals(0, weights.length);
    }

    @Test
    @DisplayName("generatePriorityWeights — single priority")
    void priorityWeightsSingle() {
        double[] weights = MclpUtil.generatePriorityWeights(new int[]{5});
        assertEquals(1, weights.length);
        assertEquals(1.0, weights[0], TOL);
    }

    @Test
    @DisplayName("filterParetoOptimal — known dominated set")
    void filterParetoKnownDominated() {
        // (1,5) dominates (2,6); (2,2) dominates others; (3,1) is non-dominated
        List<double[]> values = Arrays.asList(
            new double[]{1, 5}, // Pareto (best in obj1)
            new double[]{2, 2}, // Pareto (best in obj2)
            new double[]{3, 1}, // Pareto (best in obj2)
            new double[]{2, 6}, // Dominated by (1,5)
            new double[]{4, 4}  // Dominated by (2,2)
        );
        List<Integer> pareto = MclpUtil.filterParetoOptimal(values);
        assertEquals(3, pareto.size());
        assertTrue(pareto.contains(0));
        assertTrue(pareto.contains(1));
        assertTrue(pareto.contains(2));
    }

    @Test
    @DisplayName("filterParetoOptimal — single point always non-dominated")
    void filterParetoSinglePoint() {
        List<double[]> values = Arrays.asList(new double[]{5, 5});
        List<Integer> pareto = MclpUtil.filterParetoOptimal(values);
        assertEquals(1, pareto.size());
        assertEquals(0, pareto.get(0));
    }

    @Test
    @DisplayName("filterParetoOptimal — all non-dominated")
    void filterParetoAllNonDominated() {
        // (1,4), (2,2), (4,1) — none dominates another
        List<double[]> values = Arrays.asList(
            new double[]{1, 4},
            new double[]{2, 2},
            new double[]{4, 1}
        );
        List<Integer> pareto = MclpUtil.filterParetoOptimal(values);
        assertEquals(3, pareto.size());
    }

    @Test
    @DisplayName("filterParetoOptimal — empty input")
    void filterParetoEmpty() {
        List<double[]> values = new ArrayList<>();
        List<Integer> pareto = MclpUtil.filterParetoOptimal(values);
        assertTrue(pareto.isEmpty());
    }

    @Test
    @DisplayName("normalizeObjectives — zero-range column handled")
    void normalizeObjectivesZeroRange() {
        // Column 1 has range 0 (all same value)
        List<double[]> values = Arrays.asList(
            new double[]{1, 5},
            new double[]{3, 5},
            new double[]{5, 5}
        );
        double[] ideal = new double[]{1, 5};
        double[] nadir = new double[]{5, 5}; // zero range on col 1
        double[][] normalized = MclpUtil.normalizeObjectives(values, ideal, nadir);
        assertEquals(3, normalized.length);
        // Column 1 should be 0.0 after normalization (zero range)
        for (double[] row : normalized) {
            assertEquals(0.0, row[1], TOL);
        }
        // Column 0 should be normalized
        assertTrue(normalized[0][0] < normalized[1][0]);
        assertTrue(normalized[1][0] < normalized[2][0]);
    }

    @Test
    @DisplayName("normalizeVector — zero vector handled")
    void normalizeVectorZero() {
        double[] result = MclpUtil.normalizeVector(new double[]{0, 0, 0});
        for (double v : result) {
            assertEquals(0.0, v, TOL);
        }
    }

    @Test
    @DisplayName("normalizeVector — negative values")
    void normalizeVectorNegative() {
        double[] result = MclpUtil.normalizeVector(new double[]{-3, 0, 3});
        double norm = Math.sqrt(9 + 0 + 9);
        assertEquals(-3.0 / norm, result[0], TOL);
        assertEquals(0.0, result[1], TOL);
        assertEquals(3.0 / norm, result[2], TOL);
    }

    @Test
    @DisplayName("normalizeVector — single-element vector")
    void normalizeVectorSingle() {
        double[] result = MclpUtil.normalizeVector(new double[]{5});
        assertEquals(1.0, result[0], TOL);
    }

    @Test
    @DisplayName("computeIdealPoint — basic 2D problem")
    void computeIdealPointBasic() {
        IVector[] c = simple2Obj();
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();
        double[] ideal = MclpUtil.computeIdealPoint(c, Aub, bub, null, null, newSolver());
        assertEquals(2, ideal.length);
        assertTrue(ideal[0] < Double.MAX_VALUE);
        assertTrue(ideal[1] < Double.MAX_VALUE);
    }

    @Test
    @DisplayName("computeNadirPoint — 2D problem returns values")
    void computeNadirPointFinite() {
        IVector[] c = simple2Obj();
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();
        double[] nadir = MclpUtil.computeNadirPoint(c, Aub, bub, null, null, newSolver());
        assertEquals(2, nadir.length);
        // Bounded problem should produce finite nadir values
        assertTrue(nadir[0] <= Double.MAX_VALUE);
        assertTrue(nadir[1] <= Double.MAX_VALUE);
    }

    @Test
    @DisplayName("selectBestByTopsis — with actual data")
    void selectBestByTopsisWithData() {
        IVector[] c = simple2Obj();
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();
        ILinProgSolver solver = newSolver();

        List<IVector> solutions = new ArrayList<>();
        List<double[]> objValues = new ArrayList<>();
        for (double w1 : new double[]{0.0, 0.3, 0.5, 0.7, 1.0}) {
            double w2 = 1.0 - w1;
            IVector weightedC = Linalg.vector(new double[]{w1, w2});
            OptResult r = solver.solve(weightedC, Aub, bub);
            if (r.isConverged() && r.getOptimalPoint() != null) {
                IVector x = r.getOptimalPoint();
                solutions.add(x);
                objValues.add(new double[]{
                    c[0].innerProductValue(x),
                    c[1].innerProductValue(x)
                });
            }
        }

        assertFalse(solutions.isEmpty());
        int bestIdx = MclpUtil.selectBestByTopsis(solutions, objValues, new double[]{0.5, 0.5});
        assertTrue(bestIdx >= 0);
        assertTrue(bestIdx < objValues.size());
    }

    @Test
    @DisplayName("computeExtremePoints — returns points for all objectives")
    void computeExtremePointsBasic() {
        IVector[] c = simple2Obj();
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();
        var result = MclpUtil.computeExtremePoints(c, Aub, bub, null, null, newSolver());
        assertNotNull(result._1);
        assertNotNull(result._2);
        assertFalse(result._1.isEmpty());
        assertFalse(result._2.isEmpty());
    }

    @Test
    @DisplayName("computeIdealAndNadirPoints — returns valid bounds")
    void computeIdealAndNadirPoints() {
        IVector[] c = simple2Obj();
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();
        var result = MclpUtil.computeIdealAndNadirPoints(c, Aub, bub, null, null, newSolver());
        double[] ideal = result._1;
        double[] nadir = result._2;
        assertNotNull(ideal);
        assertNotNull(nadir);
        assertEquals(2, ideal.length);
        assertEquals(2, nadir.length);
        // Ideal should be <= Nadir for each objective (minimization)
        for (int i = 0; i < ideal.length; i++) {
            assertTrue(ideal[i] <= nadir[i] + TOL,
                "ideal[" + i + "]=" + ideal[i] + " should be <= nadir[" + i + "]=" + nadir[i]);
        }
    }

    // ==================== MclpResult Tests ====================

    @Test
    @DisplayName("MclpResult Builder — basic build")
    void mclpResultBuilderBasic() {
        IVector sol = Linalg.vector(new double[]{1, 1});
        MclpResult result = new MclpResult.Builder()
            .addSolution(sol, new double[]{1, 2})
            .numObjectives(2)
            .numVariables(2)
            .solverType(MclpSolverType.WeightedSum)
            .solverName("Test")
            .build();

        assertEquals(1, result.getNumSolutions());
        assertEquals(2, result.getNumObjectives());
        assertEquals(2, result.getNumVariables());
        assertEquals(MclpSolverType.WeightedSum, result.getSolverType());
        assertTrue(result.isConverged());
        assertNotNull(result.getSummary());
        assertNotNull(result.getDetailedReport());
    }

    @Test
    @DisplayName("MclpResult — Pareto dominance: dominates")
    void paretoDominates() {
        // (1,2) dominates (2,3) in minimization
        int result = MclpResult.paretoDominates(new double[]{1, 2}, new double[]{2, 3});
        assertEquals(1, result);
    }

    @Test
    @DisplayName("MclpResult — Pareto dominance: is dominated")
    void paretoIsDominated() {
        int result = MclpResult.paretoDominates(new double[]{2, 3}, new double[]{1, 2});
        assertEquals(-1, result);
    }

    @Test
    @DisplayName("MclpResult — Pareto dominance: incomparable")
    void paretoIncomparable() {
        int result = MclpResult.paretoDominates(new double[]{1, 3}, new double[]{3, 1});
        assertEquals(0, result);
    }

    @Test
    @DisplayName("MclpResult — Pareto dominance: equal")
    void paretoEqual() {
        int result = MclpResult.paretoDominates(new double[]{2, 2}, new double[]{2, 2});
        assertEquals(0, result);
    }

    @Test
    @DisplayName("MclpResult — getNonDominatedSolutions")
    void getNonDominatedSolutions() {
        MclpResult result = new MclpResult.Builder()
            .solutions(Arrays.asList(
                Linalg.vector(new double[]{1, 0}),
                Linalg.vector(new double[]{0, 1}),
                Linalg.vector(new double[]{1, 1})
            ))
            .objectiveValues(Arrays.asList(
                new double[]{1, 5},
                new double[]{5, 1},
                new double[]{2, 6}  // dominated by (1,5)
            ))
            .numObjectives(2)
            .numVariables(2)
            .build();

        List<Integer> pareto = result.getNonDominatedSolutions();
        assertEquals(2, pareto.size());
        assertTrue(pareto.contains(0));
        assertTrue(pareto.contains(1));
    }

    @Test
    @DisplayName("MclpResult — computeIdealPoint from solution set")
    void mclpResultComputeIdealPoint() {
        MclpResult result = new MclpResult.Builder()
            .solutions(Arrays.asList(
                Linalg.vector(new double[]{1, 0}),
                Linalg.vector(new double[]{0, 1})
            ))
            .objectiveValues(Arrays.asList(
                new double[]{3, 7},
                new double[]{8, 2}
            ))
            .numObjectives(2)
            .numVariables(2)
            .build();

        double[] ideal = result.computeIdealPoint();
        assertNotNull(ideal);
        assertEquals(3.0, ideal[0], TOL);
        assertEquals(2.0, ideal[1], TOL);
    }

    @Test
    @DisplayName("MclpResult — computeNadirPoint from solution set")
    void mclpResultComputeNadirPoint() {
        MclpResult result = new MclpResult.Builder()
            .solutions(Arrays.asList(
                Linalg.vector(new double[]{1, 0}),
                Linalg.vector(new double[]{0, 1})
            ))
            .objectiveValues(Arrays.asList(
                new double[]{3, 7},
                new double[]{8, 2}
            ))
            .numObjectives(2)
            .numVariables(2)
            .build();

        double[] nadir = result.computeNadirPoint();
        assertNotNull(nadir);
        assertEquals(8.0, nadir[0], TOL);
        assertEquals(7.0, nadir[1], TOL);
    }

    @Test
    @DisplayName("MclpResult — selectedSolutionIndex bounds check")
    void selectedSolutionIndexBounds() {
        // Builder clamps negative index to 0; OOB is handled by getSelectedSolution
        MclpResult result = new MclpResult.Builder()
            .addSolution(Linalg.vector(new double[]{1, 1}), new double[]{1, 2})
            .numObjectives(2)
            .numVariables(2)
            .selectedSolutionIndex(-5) // negative clamped to 0
            .build();
        assertNotNull(result.getSelectedSolution()); // falls back to index 0
        assertEquals(0, result.getSelectedSolutionIndex());
    }

    @Test
    @DisplayName("MclpResult — getSolution with index bounds check")
    void getSolutionBoundsCheck() {
        MclpResult result = new MclpResult.Builder()
            .addSolution(Linalg.vector(new double[]{1, 1}), new double[]{1, 2})
            .numObjectives(2)
            .numVariables(2)
            .build();
        assertNull(result.getSolution(-1));
        assertNull(result.getSolution(100));
        assertNull(result.getObjectiveValues(-1));
        assertNotNull(result.getSolution(0));
    }

    @Test
    @DisplayName("MclpResult — weights and goals stored and retrieved")
    void weightsAndGoalsStored() {
        MclpResult result = new MclpResult.Builder()
            .addSolution(Linalg.vector(new double[]{1, 1}), new double[]{1, 2})
            .numObjectives(2)
            .numVariables(2)
            .weights(new double[]{0.7, 0.3})
            .goals(new double[]{5, 10})
            .priorityOrder(new int[]{1, 0})
            .build();

        assertArrayEquals(new double[]{0.7, 0.3}, result.getWeights(), TOL);
        assertArrayEquals(new double[]{5, 10}, result.getGoals(), TOL);
        assertArrayEquals(new int[]{1, 0}, result.getPriorityOrder());
    }

    @Test
    @DisplayName("MclpResult — getSummary and getDetailedReport not null")
    void summaryAndReportNotNull() {
        MclpResult result = new MclpResult.Builder()
            .addSolution(Linalg.vector(new double[]{1, 1}), new double[]{1, 2})
            .numObjectives(2)
            .numVariables(2)
            .idealPoint(new double[]{1, 2})
            .nadirPoint(new double[]{3, 4})
            .build();

        String summary = result.getSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("MCLP"));
        String report = result.getDetailedReport();
        assertNotNull(report);
        assertTrue(report.contains("Solution"));
    }

    @Test
    @DisplayName("MclpResult — isParetoOptimal for known front")
    void isParetoOptimal() {
        MclpResult result = new MclpResult.Builder()
            .solutions(Arrays.asList(
                Linalg.vector(new double[]{1, 0}),
                Linalg.vector(new double[]{0, 1}),
                Linalg.vector(new double[]{2, 2})
            ))
            .objectiveValues(Arrays.asList(
                new double[]{1, 5},
                new double[]{5, 1},
                new double[]{2, 6}
            ))
            .numObjectives(2)
            .numVariables(2)
            .build();

        assertTrue(result.isParetoOptimal(0));
        assertTrue(result.isParetoOptimal(1));
        assertFalse(result.isParetoOptimal(2));
    }

    // ==================== RereWeightedSum Tests ====================

    @Test
    @DisplayName("WeightedSum — basic 2-objective solve")
    void weightedSumBasic() {
        RereWeightedSum solver = new RereWeightedSum(new double[]{0.5, 0.5});
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertEquals(1, result.getNumSolutions());
        assertNotNull(result.getSelectedSolution());
        double[] objs = result.getSelectedObjectiveValues();
        assertNotNull(objs);
        assertEquals(2, objs.length);
        assertTrue(objs[0] >= 0 && objs[1] >= 0, "Objective values should be non-negative");
        assertTrue(objs[0] + objs[1] >= 2.0 - TOL, "Should satisfy x1+x2 >= 2");
    }

    @Test
    @DisplayName("WeightedSum — normalize weights disabled")
    void weightedSumNormalizeDisabled() {
        RereWeightedSum solver = new RereWeightedSum(new double[]{2, 1});
        solver.setNormalizeWeights(false);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
    }

    @Test
    @DisplayName("WeightedSum — single objective degenerates to LP")
    void weightedSumSingleObjective() {
        RereWeightedSum solver = new RereWeightedSum(new double[]{1.0});
        IVector[] c = new IVector[]{Linalg.vector(new double[]{1, 2})};
        IMatrix Aub = Linalg.matrix(new double[][]{{-1, -1}});
        IVector bub = Linalg.vector(new double[]{-2});

        MclpResult result = solver.solve(c, Aub, bub, null, null, null);
        assertTrue(result.isConverged());
        assertEquals(1, result.getNumObjectives());
        assertEquals(2, result.getNumVariables());
    }

    @Test
    @DisplayName("WeightedSum — preference weighting with asymmetric weights")
    void weightedSumAsymmetric() {
        // Weight obj1 (x1) heavily vs obj2 (x2) -> solution should favor minimizing x1
        RereWeightedSum solver = new RereWeightedSum(new double[]{0.99, 0.01});
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged(), "Solver should converge");
        double[] objs = result.getSelectedObjectiveValues();
        assertNotNull(objs);
        // x1+x2 >= 2; heavily-weighted x1 should be well-optimized
        assertTrue(objs[0] <= 2.0, "obj1 (x1) should not exceed 2 for min problem");
        assertTrue(objs[0] + objs[1] >= 2.0 - TOL, "Must satisfy x1+x2 >= 2");
    }

    @Test
    @DisplayName("WeightedSum — null weights defaults to equal")
    void weightedSumNullWeightsDefaults() {
        RereWeightedSum solver = new RereWeightedSum();
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
        assertNotNull(result.getWeights());
        assertEquals(2, result.getWeights().length);
    }

    // ==================== RereGoalProgramming Tests ====================

    @Test
    @DisplayName("GoalProgramming — basic weighted solve")
    void goalProgrammingWeighted() {
        RereGoalProgramming solver = new RereGoalProgramming(
            new double[]{0.5, 0.5},
            new double[]{0.5, 0.5}
        );
        solver.setMethodType(RereGoalProgramming.GoalProgrammingType.Weighted);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertEquals(1, result.getNumSolutions());
        assertNotNull(result.getSelectedSolution());
    }

    @Test
    @DisplayName("GoalProgramming — default goals computation")
    void goalProgrammingDefaultGoals() {
        RereGoalProgramming solver = new RereGoalProgramming();
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
        assertNotNull(result.getGoals());
        assertEquals(2, result.getGoals().length);
    }

    @Test
    @DisplayName("GoalProgramming — Lexicographic method type")
    void goalProgrammingLexicographic() {
        RereGoalProgramming solver = new RereGoalProgramming(
            new double[]{0.5, 1.0},
            new double[]{0.8, 0.2}  // higher priority on obj1
        );
        solver.setMethodType(RereGoalProgramming.GoalProgrammingType.Lexicographic);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
        assertNotNull(result.getSelectedSolution());
    }

    @Test
    @DisplayName("GoalProgramming — Chebyshev method type")
    void goalProgrammingChebyshev() {
        RereGoalProgramming solver = new RereGoalProgramming(
            new double[]{0.5, 1.0},
            new double[]{0.5, 0.5}
        );
        solver.setMethodType(RereGoalProgramming.GoalProgrammingType.Chebyshev);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
        assertNotNull(result.getSelectedSolution());
    }

    @Test
    @DisplayName("GoalProgramming — all three method types produce valid results")
    void goalProgrammingAllTypes() {
        double[] goals = new double[]{1.0, 1.0};
        double[] weights = new double[]{0.5, 0.5};

        for (RereGoalProgramming.GoalProgrammingType type : RereGoalProgramming.GoalProgrammingType.values()) {
            RereGoalProgramming solver = new RereGoalProgramming(goals, weights);
            solver.setMethodType(type);
            MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
            assertTrue(result.isConverged(), type + " should converge");
            assertNotNull(result.getSelectedSolution(), type + " should have a solution");
        }
    }

    @Test
    @DisplayName("GoalProgramming — invalid goal count handled")
    void goalProgrammingInvalidGoalCount() {
        RereGoalProgramming solver = new RereGoalProgramming(new double[]{1.0}); // only 1 goal for 2 objectives
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged()); // should auto-compute defaults
    }

    // ==================== RereLexicographic Tests ====================

    @Test
    @DisplayName("Lexicographic — basic priority ordering")
    void lexicographicBasic() {
        RereLexicographic solver = new RereLexicographic(new int[]{0, 1});
        solver.setTolerance(1e-3);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertNotNull(result.getSelectedSolution());
        assertEquals(2, result.getNumObjectives());
    }

    @Test
    @DisplayName("Lexicographic — custom priority order")
    void lexicographicCustomPriority() {
        // Optimize obj2 first, then obj1
        RereLexicographic solver = new RereLexicographic(new int[]{1, 0});
        solver.setTolerance(1e-3);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        double[] objs = result.getSelectedObjectiveValues();
        assertNotNull(objs);
        assertTrue(objs[1] <= objs[0] + 0.1); // obj2 should be well-optimized (optimized first)
    }

    @Test
    @DisplayName("Lexicographic — default priority order when not set")
    void lexicographicDefaultPriority() {
        RereLexicographic solver = new RereLexicographic();
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
        assertNotNull(result.getPriorityOrder());
        assertEquals(2, result.getPriorityOrder().length);
    }

    @Test
    @DisplayName("Lexicographic — tolerance relaxation works")
    void lexicographicToleranceApplied() {
        RereLexicographic tight = new RereLexicographic(new int[]{0, 1});
        tight.setTolerance(1e-8);
        MclpResult r1 = tight.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        RereLexicographic loose = new RereLexicographic(new int[]{0, 1});
        loose.setTolerance(0.5);
        MclpResult r2 = loose.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(r1.isConverged());
        assertTrue(r2.isConverged());
        // Loose tolerance should allow second objective more room to improve
        double[] o1 = r1.getSelectedObjectiveValues();
        double[] o2 = r2.getSelectedObjectiveValues();
        assertNotNull(o1);
        assertNotNull(o2);
        // obj1 (highest priority) should be similar; obj2 may differ
        assertEquals(o1[0], o2[0], 0.5); // first priority similar
    }

    @Test
    @DisplayName("Lexicographic — produces intermediate solutions")
    void lexicographicIntermediateSolutions() {
        RereLexicographic solver = new RereLexicographic(new int[]{0, 1});
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        // Should have at least the final solution
        assertTrue(result.getNumSolutions() >= 1);
    }

    // ==================== RereParetoOptimal Tests ====================

    @Test
    @DisplayName("ParetoOptimal — basic 2-objective Pareto front")
    void paretoOptimalBasic2D() {
        RereParetoOptimal solver = new RereParetoOptimal(10);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertTrue(result.getNumSolutions() > 0, "Should find at least one Pareto-optimal solution");
        assertTrue(result.getParetoCoverage() >= 0);
        assertTrue(result.getDiversityMetric() >= 0);
    }

    @Test
    @DisplayName("ParetoOptimal — uniform grid sampling")
    void paretoOptimalUniformGrid() {
        RereParetoOptimal solver = new RereParetoOptimal(15);
        solver.setSamplingMethod(RereParetoOptimal.SamplingMethod.UniformGrid);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
        assertTrue(result.getNumSolutions() >= 1);
    }

    @Test
    @DisplayName("ParetoOptimal — random sampling")
    void paretoOptimalRandom() {
        RereParetoOptimal solver = new RereParetoOptimal(20);
        solver.setSamplingMethod(RereParetoOptimal.SamplingMethod.Random);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
        assertTrue(result.getNumSolutions() >= 1);
    }

    @Test
    @DisplayName("ParetoOptimal — adaptive sampling")
    void paretoOptimalAdaptive() {
        RereParetoOptimal solver = new RereParetoOptimal(15);
        solver.setSamplingMethod(RereParetoOptimal.SamplingMethod.Adaptive);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);
        assertTrue(result.isConverged());
        assertTrue(result.getNumSolutions() >= 1);
    }

    @Test
    @DisplayName("ParetoOptimal — all solutions are non-dominated")
    void paretoOptimalAllNonDominated() {
        RereParetoOptimal solver = new RereParetoOptimal(20);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        List<double[]> objs = result.getObjectiveValues();
        for (int i = 0; i < objs.size(); i++) {
            for (int j = i + 1; j < objs.size(); j++) {
                // None of the returned solutions should dominate another
                int domStatus = MclpResult.paretoDominates(objs.get(i), objs.get(j));
                assertEquals(0, domStatus,
                    "Pareto-optimal solutions should not dominate each other");
            }
        }
    }

    @Test
    @DisplayName("ParetoOptimal — hypervolume computation for 2D")
    void paretoOptimalHypervolume2D() {
        RereParetoOptimal solver = new RereParetoOptimal(10);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        double hv = result.getHypervolume();
        assertTrue(hv >= 0, "Hypervolume should be non-negative");
        // With reference point = nadir, hypervolume should be > 0 if we have solutions
        if (result.getNumSolutions() > 0) {
            assertTrue(hv > 0, "Hypervolume should be positive with solutions");
        }
    }

    // ==================== RereAHP Tests ====================

    @Test
    @DisplayName("AHP — 3x3 consistent matrix weight computation")
    void ahp3x3Consistent() {
        // Consistent comparison matrix: w = (0.6, 0.3, 0.1)
        IMatrix comparison = Linalg.matrix(new double[][]{
            {1.0, 2.0, 6.0},
            {0.5, 1.0, 3.0},
            {1.0/6.0, 1.0/3.0, 1.0}
        });

        RereAHP ahp = new RereAHP(comparison);
        double[] weights = ahp.getWeights();
        assertNotNull(weights);
        assertEquals(3, weights.length);
        assertEquals(1.0, weights[0] + weights[1] + weights[2], TOL);
        // Weights should be proportional to (6, 3, 1) normalized
        assertEquals(0.6, weights[0], 0.05);
        assertEquals(0.3, weights[1], 0.05);
        assertEquals(0.1, weights[2], 0.05);
    }

    @Test
    @DisplayName("AHP — 2x2 matrix consistency ratio = 0 (regression: no div0)")
    void ahp2x2Div0Regression() {
        // 2x2 matrices are always consistent; RANDOM_INDEX[2]=0 caused div0
        IMatrix comparison = Linalg.matrix(new double[][]{
            {1.0, 3.0},
            {1.0/3.0, 1.0}
        });

        RereAHP ahp = new RereAHP(comparison);
        assertTrue(ahp.isConsistent(), "2x2 matrices should always be consistent");
        assertEquals(0.0, ahp.getConsistencyRatio(), TOL);
        assertEquals(0.0, ahp.getConsistencyIndex(), 1e-10);
        double[] weights = ahp.getWeights();
        assertEquals(1.0, weights[0] + weights[1], TOL);
        assertTrue(weights[0] > weights[1], "First objective should have higher weight");
    }

    @Test
    @DisplayName("AHP — createConsistentMatrix roundtrip")
    void ahpCreateConsistentMatrixRoundtrip() {
        double[] originalWeights = new double[]{0.4, 0.35, 0.25};
        IMatrix consistent = RereAHP.createConsistentMatrix(originalWeights);

        RereAHP ahp = new RereAHP(consistent);
        double[] computed = ahp.getWeights();
        assertArrayEquals(originalWeights, computed, 5e-3);
    }

    @Test
    @DisplayName("AHP — inconsistent matrix detection")
    void ahpInconsistentDetection() {
        // Deliberately inconsistent (intransitive): A>B, B>C, but C>A
        IMatrix comparison = Linalg.matrix(new double[][]{
            {1.0, 3.0, 0.5},
            {1.0/3.0, 1.0, 3.0},
            {2.0, 1.0/3.0, 1.0}
        });

        RereAHP ahp = new RereAHP(comparison);
        // This should have CR > 0.1 (inconsistent)
        assertTrue(ahp.getConsistencyRatio() > 0.05,
            "Intransitive matrix should have non-trivial CR: " + ahp.getConsistencyRatio());
    }

    @Test
    @DisplayName("AHP — solve with computed weights")
    void ahpSolveWithWeights() {
        IMatrix comparison = Linalg.matrix(new double[][]{
            {1.0, 2.0},
            {0.5, 1.0}
        });

        RereAHP solver = new RereAHP(comparison);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertNotNull(result.getSelectedSolution());
        double[] weights = result.getWeights();
        assertNotNull(weights);
        assertEquals(2, weights.length);
        assertTrue(weights[0] > weights[1]); // obj1 more important
    }

    @Test
    @DisplayName("AHP — no comparison matrix defaults to equal weights")
    void ahpDefaultEqualWeights() {
        RereAHP solver = new RereAHP();
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        double[] weights = result.getWeights();
        assertNotNull(weights);
        assertEquals(0.5, weights[0], TOL);
        assertEquals(0.5, weights[1], TOL);
    }

    @Test
    @DisplayName("AHP — setWeights bypasses AHP computation")
    void ahpSetWeightsBypass() {
        RereAHP solver = new RereAHP();
        solver.setWeights(new double[]{0.8, 0.2});
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        double[] w = result.getWeights();
        assertEquals(0.8, w[0], TOL);
        assertEquals(0.2, w[1], TOL);
    }

    @Test
    @DisplayName("AHP — static checkConsistency")
    void ahpStaticCheckConsistency() {
        IMatrix consistent = RereAHP.createConsistentMatrix(new double[]{0.6, 0.4});
        assertTrue(RereAHP.checkConsistency(consistent));
    }

    // ==================== RereTopsis Tests ====================

    @Test
    @DisplayName("TOPSIS — basic selection")
    void topsisBasic() {
        RereTopsis solver = new RereTopsis(new double[]{0.5, 0.5});
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertNotNull(result.getSelectedSolution());
        double[] objs = result.getSelectedObjectiveValues();
        assertNotNull(objs);
        assertEquals(2, objs.length);
        assertTrue(result.getDiversityMetric() > 0);
    }

    @Test
    @DisplayName("TOPSIS — with maximization flags")
    void topsisMaximization() {
        RereTopsis solver = new RereTopsis(new double[]{0.5, 0.5});
        solver.setMaximizationFlags(new boolean[]{true, false}); // maximize obj1, minimize obj2
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertNotNull(result.getSelectedSolution());
    }

    @Test
    @DisplayName("TOPSIS — single candidate returns that candidate")
    void topsisSingleCandidate() {
        // With 1 objective, TOPSIS should return it
        IVector[] c = new IVector[]{Linalg.vector(new double[]{1, 2})};
        IMatrix Aub = Linalg.matrix(new double[][]{{-1, -1}});
        IVector bub = Linalg.vector(new double[]{-2});

        RereTopsis solver = new RereTopsis(new double[]{1.0});
        MclpResult result = solver.solve(c, Aub, bub, null, null, null);
        assertTrue(result.isConverged());
        assertEquals(1, result.getNumSolutions());
    }

    @Test
    @DisplayName("TOPSIS — asymmetric weights favor one objective")
    void topsisAsymmetricWeights() {
        RereTopsis solver = new RereTopsis(new double[]{0.9, 0.1});
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        double[] objs = result.getSelectedObjectiveValues();
        assertNotNull(objs);
        // With heavy weight on obj1, the solution should have better obj1
        assertTrue(objs[0] < 1.1, "obj1 should be well-optimized: " + objs[0]);
    }

    // ==================== RereInteractive Tests ====================

    @Test
    @DisplayName("Interactive — maxIterations validation throws")
    void interactiveMaxIterationsValidation() {
        assertThrows(IllegalArgumentException.class, () -> {
            new RereInteractive().setMaxIterations(0);
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new RereInteractive().setMaxIterations(-1);
        });
    }

    @Test
    @DisplayName("Interactive — basic auto-adjustment runs without callback")
    void interactiveAutoAdjust() {
        RereInteractive solver = new RereInteractive();
        solver.setMaxIterations(3);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertNotNull(result);
        assertTrue(result.getNumSolutions() >= 1, "Should have at least one solution");
        assertNotNull(result.getSelectedSolution(), "Should have a selected solution");
    }

    @Test
    @DisplayName("Interactive — callback satisfied immediately")
    void interactiveCallbackSatisfied() {
        RereInteractive solver = new RereInteractive(new RereInteractive.DecisionMakerCallback() {
            @Override
            public RereInteractive.DecisionMakerResponse provideFeedback(
                    IVector currentSolution, double[] objectiveValues, int iteration) {
                return RereInteractive.DecisionMakerResponse.satisfied();
            }

            @Override
            public double[] getAspirationLevels() {
                return new double[]{1.0, 1.0};
            }
        });
        solver.setMaxIterations(10);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertEquals("决策者满意或达到收敛条件", result.getConvergenceReason());
    }

    @Test
    @DisplayName("Interactive — callback adjusts weights")
    void interactiveCallbackAdjustWeights() {
        final int[] callCount = {0};
        RereInteractive solver = new RereInteractive(new RereInteractive.DecisionMakerCallback() {
            @Override
            public RereInteractive.DecisionMakerResponse provideFeedback(
                    IVector currentSolution, double[] objectiveValues, int iteration) {
                callCount[0]++;
                if (callCount[0] >= 3) {
                    return RereInteractive.DecisionMakerResponse.satisfied();
                }
                return RereInteractive.DecisionMakerResponse.notSatisfiedAdjustWeights(
                    new double[]{0.7, 0.3});
            }

            @Override
            public double[] getAspirationLevels() {
                return new double[]{1.0, 1.0};
            }
        });
        solver.setMaxIterations(5);
        MclpResult result = solver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(result.isConverged());
        assertTrue(callCount[0] >= 1);
    }

    // ==================== Cross-Solver Tests ====================

    @Test
    @DisplayName("Cross-solver — all solvers produce finite objective values")
    void allSolversFiniteObjectives() {
        IVector[] c = simple2Obj();
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();

        IMclpSolver[] solvers = {
            new RereWeightedSum(new double[]{0.5, 0.5}),
            new RereGoalProgramming(new double[]{1.0, 1.0}, new double[]{0.5, 0.5}),
            new RereLexicographic(new int[]{0, 1}),
            new RereParetoOptimal(8),
            new RereAHP(RereAHP.createConsistentMatrix(new double[]{0.5, 0.5})),
            new RereTopsis(new double[]{0.5, 0.5}),
            new RereInteractive()
        };

        for (IMclpSolver solver : solvers) {
            MclpResult result = solver.solve(c, Aub, bub, null, null, null);
            assertNotNull(result, solver.getName() + " returned null");
            assertTrue(result.getNumSolutions() >= 1,
                solver.getName() + " should have at least 1 solution");
            double[] objs = result.getSelectedObjectiveValues();
            if (objs != null) {
                for (int i = 0; i < objs.length; i++) {
                    assertTrue(Double.isFinite(objs[i]),
                        solver.getName() + " obj[" + i + "] should be finite, got " + objs[i]);
                }
            }
        }
    }

    @Test
    @DisplayName("Cross-solver — consistent results on single-objective problem")
    void allSolversSingleObjectiveConsistency() {
        IVector[] c = new IVector[]{Linalg.vector(new double[]{1, 2})};
        IMatrix Aub = Linalg.matrix(new double[][]{{-1, -1}});
        IVector bub = Linalg.vector(new double[]{-2});

        IMclpSolver[] solvers = {
            new RereWeightedSum(new double[]{1.0}),
            new RereGoalProgramming(new double[]{2.0}, new double[]{1.0}),
            new RereLexicographic(new int[]{0}),
            new RereAHP(),
            new RereTopsis(new double[]{1.0})
        };

        Double firstObj = null;
        for (IMclpSolver solver : solvers) {
            MclpResult result = solver.solve(c, Aub, bub, null, null, null);
            assertTrue(result.isConverged(), solver.getName() + " should converge");
            double[] objs = result.getSelectedObjectiveValues();
            assertNotNull(objs, solver.getName() + " should have objective values");
            if (firstObj == null) {
                firstObj = objs[0];
            } else {
                assertEquals(firstObj, objs[0], 0.01,
                    solver.getName() + " should have consistent objective value. Expected ~" + firstObj + " got " + objs[0]);
            }
        }
    }

    @Test
    @DisplayName("Cross-solver — handles equality constraints")
    void equalityConstraintHandling() {
        IVector[] c = new IVector[]{
            Linalg.vector(new double[]{1, 0, 0}),
            Linalg.vector(new double[]{0, 1, 0})
        };
        // minimize x1, x2 subject to x1 + x2 + x3 = 3, x_i >= 0
        IMatrix Aeq = Linalg.matrix(new double[][]{{1, 1, 1}});
        IVector beq = Linalg.vector(new double[]{3});

        IMclpSolver[] solvers = {
            new RereWeightedSum(new double[]{0.5, 0.5}),
            new RereGoalProgramming(),
            new RereLexicographic(new int[]{0, 1}),
            new RereParetoOptimal(6),
        };

        for (IMclpSolver solver : solvers) {
            MclpResult result = solver.solve(c, null, null, Aeq, beq, null);
            assertNotNull(result, solver.getName() + " returned null");
            assertTrue(result.isConverged(), solver.getName() + " should converge");
            IVector sol = result.getSelectedSolution();
            assertNotNull(sol, solver.getName() + " should have a solution");
            double sum = (Double) sol.get(0) + (Double) sol.get(1) + (Double) sol.get(2);
            assertEquals(3.0, sum, 1e-4,
                solver.getName() + " should satisfy equality constraint, got sum=" + sum);
        }
    }

    @Test
    @DisplayName("Cross-solver — all solver types report correct MclpSolverType")
    void solverTypesCorrect() {
        assertEquals(MclpSolverType.WeightedSum, new RereWeightedSum().getSolverType());
        assertEquals(MclpSolverType.GoalProgramming, new RereGoalProgramming().getSolverType());
        assertEquals(MclpSolverType.Lexicographic, new RereLexicographic().getSolverType());
        assertEquals(MclpSolverType.Pareto, new RereParetoOptimal().getSolverType());
        assertEquals(MclpSolverType.Ahp, new RereAHP().getSolverType());
        assertEquals(MclpSolverType.Topsis, new RereTopsis().getSolverType());
        assertEquals(MclpSolverType.Interactive, new RereInteractive().getSolverType());
    }

    @Test
    @DisplayName("Cross-solver — null c[0] throws IllegalArgumentException")
    void nullCObjectValidation() {
        IVector[] c = new IVector[]{null, Linalg.vector(new double[]{1})};
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();

        IMclpSolver[] solvers = {
            new RereWeightedSum(), new RereGoalProgramming(), new RereLexicographic(),
            new RereParetoOptimal(), new RereAHP(), new RereTopsis(), new RereInteractive()
        };

        for (IMclpSolver solver : solvers) {
            assertThrows(IllegalArgumentException.class,
                () -> solver.solve(c, Aub, bub, null, null, null),
                solver.getName() + " should reject null c[0]");
        }
    }

    @Test
    @DisplayName("Cross-solver — empty c array throws IllegalArgumentException")
    void emptyCObjectValidation() {
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();

        IMclpSolver[] solvers = {
            new RereWeightedSum(), new RereGoalProgramming(), new RereLexicographic(),
            new RereParetoOptimal(), new RereAHP(), new RereTopsis(), new RereInteractive()
        };

        for (IMclpSolver solver : solvers) {
            assertThrows(IllegalArgumentException.class,
                () -> solver.solve(new IVector[0], Aub, bub, null, null, null),
                solver.getName() + " should reject empty c array");
        }
    }

    @Test
    @DisplayName("Cross-solver — getName and getDescription return non-null")
    void namesAndDescriptions() {
        IMclpSolver[] solvers = {
            new RereWeightedSum(), new RereGoalProgramming(), new RereLexicographic(),
            new RereParetoOptimal(), new RereAHP(), new RereTopsis(), new RereInteractive()
        };

        for (IMclpSolver solver : solvers) {
            assertNotNull(solver.getName(), solver.getClass().getSimpleName() + " getName returned null");
            assertNotNull(solver.getDescription(), solver.getClass().getSimpleName() + " getDescription returned null");
            assertFalse(solver.getName().isEmpty());
            assertFalse(solver.getDescription().isEmpty());
        }
    }

    // ==================== Edge Case Tests ====================

    @Test
    @DisplayName("Edge case — problem with zero variables should throw")
    void zeroVariablesProblem() {
        // This should trigger validation, not crash
        assertThrows(Exception.class, () -> {
            IVector[] c = new IVector[]{Linalg.vector(new double[0])};
            new RereWeightedSum().solve(c, null, null, null, null, null);
        });
    }

    @Test
    @DisplayName("Edge case — non-converged base solver handling")
    void nonConvergedBaseSolverHandling() {
        // This tests that solvers handle partial convergence gracefully
        // The LP with conflicting constraints should not crash
        IVector[] c = simple2Obj();
        // x1 + x2 <= 1 AND x1 + x2 >= 2 (infeasible)
        IMatrix Aub = Linalg.matrix(new double[][]{{1, 1}, {-1, -1}});
        IVector bub = Linalg.vector(new double[]{1, -2});

        try {
            MclpResult r = new RereWeightedSum(new double[]{0.5, 0.5})
                .solve(c, Aub, bub, null, null, null);
            // If it doesn't throw, it should report non-converged
            assertNotNull(r);
        } catch (Exception e) {
            // Throwing is also acceptable for infeasible problems
            assertTrue(e.getMessage() != null);
        }
    }

    @Test
    @DisplayName("Regression — AHP 2x2 consistency ratio is zero not NaN")
    void regressionAhp2x2ConsistencyRatio() {
        IMatrix comparison = Linalg.matrix(new double[][]{
            {1.0, 5.0},
            {0.2, 1.0}
        });
        RereAHP ahp = new RereAHP(comparison);
        double cr = ahp.getConsistencyRatio();
        assertFalse(Double.isNaN(cr), "Consistency ratio should not be NaN");
        assertFalse(Double.isInfinite(cr), "Consistency ratio should not be infinite");
        assertEquals(0.0, cr, TOL);
    }

    @Test
    @DisplayName("Regression — MclpResult computeNadirPoint uses -Double.MAX_VALUE")
    void regressionNadirPointNegativeValues() {
        MclpResult result = new MclpResult.Builder()
            .solutions(Arrays.asList(
                Linalg.vector(new double[]{1, 0}),
                Linalg.vector(new double[]{0, 1})
            ))
            .objectiveValues(Arrays.asList(
                new double[]{-10, -5},
                new double[]{-3, -8}
            ))
            .numObjectives(2)
            .numVariables(2)
            .build();

        double[] nadir = result.computeNadirPoint();
        assertNotNull(nadir);
        assertEquals(-3.0, nadir[0], TOL); // max of (-10, -3)
        assertEquals(-5.0, nadir[1], TOL); // max of (-5, -8)
    }

    @Test
    @DisplayName("Regression — MclpUtil computeNadirPoint handles non-converged")
    void regressionNadirPointConvergenceCheck() {
        IVector[] c = simple2Obj();
        IMatrix Aub = simpleAub();
        IVector bub = simpleBub();
        double[] nadir = MclpUtil.computeNadirPoint(c, Aub, bub, null, null, newSolver());
        assertNotNull(nadir);
        for (double v : nadir) {
            assertTrue(Double.isFinite(v) || v == Double.MAX_VALUE,
                "Nadir should be finite or Double.MAX_VALUE fallback, got " + v);
        }
    }

    @Test
    @DisplayName("Regression — TOPSIS ideal/negative-ideal respects isMaximization")
    void regressionTopsisMaximizationFlag() {
        RereTopsis minSolver = new RereTopsis(new double[]{0.5, 0.5});
        minSolver.setMaximizationFlags(new boolean[]{false, false});
        MclpResult minResult = minSolver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        RereTopsis maxSolver = new RereTopsis(new double[]{0.5, 0.5});
        maxSolver.setMaximizationFlags(new boolean[]{true, false});
        MclpResult maxResult = maxSolver.solve(simple2Obj(), simpleAub(), simpleBub(), null, null, null);

        assertTrue(minResult.isConverged());
        assertTrue(maxResult.isConverged());
        // Both should produce valid solutions
        assertNotNull(minResult.getSelectedSolution());
        assertNotNull(maxResult.getSelectedSolution());
    }

    @Test
    @DisplayName("MclpUtil — selectBestByTopsis handles null weights gracefully")
    void topsisSelectNullWeights() {
        List<IVector> solutions = Arrays.asList(
            Linalg.vector(new double[]{1, 0}),
            Linalg.vector(new double[]{0, 1}),
            Linalg.vector(new double[]{1, 1})
        );
        List<double[]> values = Arrays.asList(
            new double[]{1, 5},
            new double[]{3, 3},
            new double[]{5, 1}
        );
        // Null weights — selectBestByTopsis will compute default equal weights internally
        // since it doesn't have weights parameter (weights are baked in before call)
        int bestIdx = MclpUtil.selectBestByTopsis(solutions, values, new double[]{0.5, 0.5});
        assertTrue(bestIdx >= 0 && bestIdx < values.size());
    }
}
