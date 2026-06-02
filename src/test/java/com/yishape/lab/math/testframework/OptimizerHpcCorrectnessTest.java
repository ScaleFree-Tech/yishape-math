package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.*;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.newton.RustLBFGS;
import com.yishape.lab.math.optimize.newton.RustOWLQN;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness validation for Rust/HPC-accelerated L-BFGS and OWL-QN optimizers.
 * Each test runs RustLBFGS, RustOWLQN (orthantwiseC=0), and RereLBFGS as reference,
 * verifying that all three produce equivalent results.
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
public class OptimizerHpcCorrectnessTest {

    private static final double EPS = 1e-3;
    private static final double LOOSE_EPS = 1e-2;
    private static TestResult.Recorder recorder;

    @BeforeAll
    static void init() {
        recorder = new TestResult.Recorder("optimizer_hpc", "test_docs/results");
    }

    @AfterAll
    static void teardown() {
        recorder.writeToFile();
        System.out.println("\n=== OPTIMIZER HPC CORRECTNESS TEST SUMMARY ===");
        System.out.println("Total:  " + recorder.getResults().size());
        System.out.println("Passed: " + recorder.getPassed());
        System.out.println("Failed: " + recorder.getFailed());
    }

    /**
     * Run all three optimizers on a problem and record results.
     */
    private static void runAllOptimizers(TestResult r, String subTest,
                                          IObjectiveFunction obj, IGradientFunction grad,
                                          IVector initX, double expectedValue, double[] expectedPoint,
                                          int maxIterations) {
        double[][] allPoints = new double[3][];
        double[] allValues = new double[3];
        String[] labels = {"RustLBFGS", "RustOWLQN_c0", "RereLBFGS"};
        IOptimizer[] optimizers = {
            new RustLBFGS(10, 1e-6, maxIterations),
            new RustOWLQN(10, 1e-6, maxIterations, 0.0),
            new RereLBFGS(10, 1e-6, maxIterations)
        };

        boolean allOk = true;
        StringBuilder detail = new StringBuilder();

        for (int i = 0; i < optimizers.length; i++) {
            OptResult result = optimizers[i].optimize(initX, obj, grad);
            allPoints[i] = new double[expectedPoint.length];
            allValues[i] = result.getOptimalValue();

            detail.append(String.format("\n  %s: value=%.8e, iters=%d, converged=%s",
                labels[i], result.getOptimalValue(), result.getIterations(), result.isConverged()));

            if (result.getOptimalPoint() != null) {
                for (int j = 0; j < Math.min(result.getOptimalPoint().size(), expectedPoint.length); j++) {
                    allPoints[i][j] = result.getOptimalPoint().get(j);
                }
                detail.append(", point=").append(result.getOptimalPoint());
            }

            if (Double.isNaN(result.getOptimalValue()) || Double.isInfinite(result.getOptimalValue())) {
                allOk = false;
            }
        }

        if (!allOk) {
            r.fail("One or more optimizers produced NaN/Infinity" + detail.toString());
            return;
        }

        // Check value against expected
        double valueError = Math.abs(allValues[0] - expectedValue);
        double pointError = 0.0;
        for (int j = 0; j < expectedPoint.length; j++) {
            pointError += Math.abs(allPoints[0][j] - expectedPoint[j]);
        }

        // Check consistency across optimizers
        for (int i = 1; i < optimizers.length; i++) {
            double valDiff = Math.abs(allValues[i] - allValues[0]);
            if (valDiff > 1e-4) {
                allOk = false;
                detail.append(String.format("\n  ** MISMATCH %s vs %s: valDiff=%.2e", labels[i], labels[0], valDiff));
            }
        }

        double tol = expectedPoint.length > 2 ? LOOSE_EPS : EPS;
        if (allOk && valueError < tol && pointError < tol * expectedPoint.length) {
            r.pass("all three optimizers consistent" + detail.toString());
        } else {
            String msg = String.format("valueErr=%.2e, pointErr=%.2e", valueError, pointError);
            r.fail(msg + detail.toString(), allValues[0], expectedValue);
        }
    }

    /**
     * Run optimizers and check value only (for multi-minima problems).
     */
    private static void runAllOptimizersValueOnly(TestResult r, String subTest,
                                                   IObjectiveFunction obj, IGradientFunction grad,
                                                   IVector initX, double expectedValue,
                                                   int maxIterations) {
        String[] labels = {"RustLBFGS", "RustOWLQN_c0", "RereLBFGS"};
        IOptimizer[] optimizers = {
            new RustLBFGS(10, 1e-6, maxIterations),
            new RustOWLQN(10, 1e-6, maxIterations, 0.0),
            new RereLBFGS(10, 1e-6, maxIterations)
        };

        boolean allOk = true;
        StringBuilder detail = new StringBuilder();

        for (int i = 0; i < optimizers.length; i++) {
            OptResult result = optimizers[i].optimize(initX, obj, grad);

            detail.append(String.format("\n  %s: value=%.8e, iters=%d, converged=%s",
                labels[i], result.getOptimalValue(), result.getIterations(), result.isConverged()));

            if (Double.isNaN(result.getOptimalValue()) || Double.isInfinite(result.getOptimalValue())) {
                allOk = false;
            }
        }

        if (!allOk) {
            r.fail("One or more optimizers produced NaN/Infinity" + detail.toString());
            return;
        }

        // Use RustLBFGS value
        OptResult firstResult = optimizers[0].optimize(initX, obj, grad);
        double valueError = Math.abs(firstResult.getOptimalValue() - expectedValue);

        if (valueError < LOOSE_EPS) {
            r.pass("converged to near-zero" + detail.toString());
        } else {
            r.fail("value error too large" + detail.toString(),
                   firstResult.getOptimalValue(), expectedValue);
        }
    }

    // ========================================================================
    // 1. Quadratic 2D: f(x) = ||x - target||^2, minimum at target
    // ========================================================================

    @Test
    @DisplayName("1.1 Quadratic 2D")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testQuadratic2D() {
        double[] target = {3.0, -2.0};
        IObjectiveFunction obj = x -> {
            double d1 = x.get(0) - target[0];
            double d2 = x.get(1) - target[1];
            return d1 * d1 + d2 * d2;
        };
        IGradientFunction grad = x -> Linalg.vector(new double[]{
            2.0 * (x.get(0) - target[0]),
            2.0 * (x.get(1) - target[1])
        });
        IVector initX = Linalg.vector(new double[]{5.0, 5.0});
        TestResult r = recorder.record("rust_lbfgs", "quadratic_2d");
        runAllOptimizers(r, "quadratic_2d", obj, grad, initX, 0.0, target, 5000);
    }

    // ========================================================================
    // 2. Quadratic 10D: f(x) = ||x - target||^2
    // ========================================================================

    @Test
    @DisplayName("2.1 Quadratic 10D")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testQuadratic10D() {
        double[] target = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        IObjectiveFunction obj = x -> {
            double sum = 0.0;
            for (int i = 0; i < x.size(); i++) {
                double d = x.get(i) - target[i];
                sum += d * d;
            }
            return sum;
        };
        IGradientFunction grad = x -> {
            double[] g = new double[x.size()];
            for (int i = 0; i < x.size(); i++) {
                g[i] = 2.0 * (x.get(i) - target[i]);
            }
            return Linalg.vector(g);
        };
        IVector initX = Linalg.vector(new double[]{-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
        TestResult r = recorder.record("rust_lbfgs", "quadratic_10d");
        runAllOptimizers(r, "quadratic_10d", obj, grad, initX, 0.0, target, 5000);
    }

    // ========================================================================
    // 3. Quadratic 100D: f(x) = ||x - target||^2
    // ========================================================================

    @Test
    @DisplayName("3.1 Quadratic 100D")
    @Timeout(value = 60, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testQuadratic100D() {
        double[] target = new double[100];
        for (int i = 0; i < 100; i++) {
            target[i] = i * 0.1;
        }
        IObjectiveFunction obj = x -> {
            double sum = 0.0;
            for (int i = 0; i < x.size(); i++) {
                double d = x.get(i) - target[i];
                sum += d * d;
            }
            return sum;
        };
        IGradientFunction grad = x -> {
            double[] g = new double[x.size()];
            for (int i = 0; i < x.size(); i++) {
                g[i] = 2.0 * (x.get(i) - target[i]);
            }
            return Linalg.vector(g);
        };
        double[] init = new double[100];
        for (int i = 0; i < 100; i++) {
            init[i] = target[i] + 5.0;
        }
        IVector initX = Linalg.vector(init);
        TestResult r = recorder.record("rust_lbfgs", "quadratic_100d");
        runAllOptimizers(r, "quadratic_100d", obj, grad, initX, 0.0, target, 5000);
    }

    // ========================================================================
    // 4. Rosenbrock 2D: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1)
    // ========================================================================

    @Test
    @DisplayName("4.1 Rosenbrock 2D")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testRosenbrock() {
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double a = 1.0 - v1;
            double b = v2 - v1 * v1;
            return a * a + 100.0 * b * b;
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double dx = -2.0 * (1.0 - v1) - 400.0 * v1 * (v2 - v1 * v1);
            double dy = 200.0 * (v2 - v1 * v1);
            return Linalg.vector(new double[]{dx, dy});
        };
        IVector initX = Linalg.vector(new double[]{-1.5, 0.5});
        TestResult r = recorder.record("rust_lbfgs", "rosenbrock_2d");
        runAllOptimizers(r, "rosenbrock_2d", obj, grad, initX, 0.0, new double[]{1.0, 1.0}, 10000);
    }

    // ========================================================================
    // 5. Himmelblau 2D: f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
    //    Multiple minima at (3,2), (-2.805,3.131), (-3.779,-3.283), (3.584,-1.848)
    // ========================================================================

    @Test
    @DisplayName("5.1 Himmelblau 2D")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testHimmelblau() {
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double a = v1 * v1 + v2 - 11.0;
            double b = v1 + v2 * v2 - 7.0;
            return a * a + b * b;
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double a = v1 * v1 + v2 - 11.0;
            double b = v1 + v2 * v2 - 7.0;
            double dx = 4.0 * v1 * a + 2.0 * b;
            double dy = 2.0 * a + 4.0 * v2 * b;
            return Linalg.vector(new double[]{dx, dy});
        };
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        TestResult r = recorder.record("rust_lbfgs", "himmelblau_2d");
        runAllOptimizersValueOnly(r, "himmelblau_2d", obj, grad, initX, 0.0, 5000);
    }

    // ========================================================================
    // 6. Beale 2D: f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
    //    Minimum at (3, 0.5), f=0
    // ========================================================================

    @Test
    @DisplayName("6.1 Beale 2D")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testBeale() {
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double a = 1.5 - v1 + v1 * v2;
            double b = 2.25 - v1 + v1 * v2 * v2;
            double c = 2.625 - v1 + v1 * v2 * v2 * v2;
            return a * a + b * b + c * c;
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double a = 1.5 - v1 + v1 * v2;
            double b = 2.25 - v1 + v1 * v2 * v2;
            double c = 2.625 - v1 + v1 * v2 * v2 * v2;
            double da_dx = -1.0 + v2;
            double db_dx = -1.0 + v2 * v2;
            double dc_dx = -1.0 + v2 * v2 * v2;
            double da_dy = v1;
            double db_dy = 2.0 * v1 * v2;
            double dc_dy = 3.0 * v1 * v2 * v2;
            double dx = 2.0 * a * da_dx + 2.0 * b * db_dx + 2.0 * c * dc_dx;
            double dy = 2.0 * a * da_dy + 2.0 * b * db_dy + 2.0 * c * dc_dy;
            return Linalg.vector(new double[]{dx, dy});
        };
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        TestResult r = recorder.record("rust_lbfgs", "beale_2d");
        runAllOptimizers(r, "beale_2d", obj, grad, initX, 0.0, new double[]{3.0, 0.5}, 5000);
    }

    // ========================================================================
    // 7. Matyas 2D: f(x,y) = 0.26*(x^2+y^2) - 0.48*x*y, minimum at (0,0)
    // ========================================================================

    @Test
    @DisplayName("7.1 Matyas 2D")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testMatyas() {
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            return 0.26 * (v1 * v1 + v2 * v2) - 0.48 * v1 * v2;
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double dx = 0.52 * v1 - 0.48 * v2;
            double dy = 0.52 * v2 - 0.48 * v1;
            return Linalg.vector(new double[]{dx, dy});
        };
        IVector initX = Linalg.vector(new double[]{3.0, -2.0});
        TestResult r = recorder.record("rust_lbfgs", "matyas_2d");
        runAllOptimizers(r, "matyas_2d", obj, grad, initX, 0.0, new double[]{0.0, 0.0}, 5000);
    }

    // ========================================================================
    // 8. Starting at optimum (boundary condition: x0 = solution)
    // ========================================================================

    @Test
    @DisplayName("8.1 Starting at optimum")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testStartAtOptimum() {
        double[] optimum = {2.0, -3.0};
        IObjectiveFunction obj = x -> {
            double d1 = x.get(0) - optimum[0];
            double d2 = x.get(1) - optimum[1];
            return d1 * d1 + d2 * d2;
        };
        IGradientFunction grad = x -> Linalg.vector(new double[]{
            2.0 * (x.get(0) - optimum[0]),
            2.0 * (x.get(1) - optimum[1])
        });
        IVector initX = Linalg.vector(new double[]{2.0, -3.0});

        TestResult r = recorder.record("rust_lbfgs", "start_at_optimum");
        String[] labels = {"RustLBFGS", "RustOWLQN_c0", "RereLBFGS"};
        IOptimizer[] optimizers = {
            new RustLBFGS(),
            new RustOWLQN(),
            new RereLBFGS()
        };

        boolean allOk = true;
        StringBuilder detail = new StringBuilder();
        for (int i = 0; i < optimizers.length; i++) {
            OptResult result = optimizers[i].optimize(initX, obj, grad);
            detail.append(String.format("\n  %s: value=%.8e, iters=%d, converged=%s",
                labels[i], result.getOptimalValue(), result.getIterations(), result.isConverged()));
            if (Double.isNaN(result.getOptimalValue()) || Double.isInfinite(result.getOptimalValue())) {
                allOk = false;
            }
        }

        if (allOk) {
            r.pass("all optimizers handle start-at-optimum" + detail.toString());
        } else {
            r.fail("one or more optimizers failed at optimum" + detail.toString());
        }
    }

    // ========================================================================
    // 9. Null / exception handling
    // ========================================================================

    @Test
    @DisplayName("9.1 Null parameter handling")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testNullParameters() {
        IObjectiveFunction obj = x -> 0.0;
        IGradientFunction grad = x -> Linalg.vector(new double[]{0.0});

        TestResult r = recorder.record("rust_lbfgs", "null_params");

        assertThrows(IllegalArgumentException.class,
            () -> new RustLBFGS().optimize(null, obj, grad));
        assertThrows(IllegalArgumentException.class,
            () -> new RustLBFGS().optimize(Linalg.vector(new double[]{1.0}), null, grad));
        assertThrows(IllegalArgumentException.class,
            () -> new RustLBFGS().optimize(Linalg.vector(new double[]{1.0}), obj, null));

        assertThrows(IllegalArgumentException.class,
            () -> new RustOWLQN().optimize(null, obj, grad));
        assertThrows(IllegalArgumentException.class,
            () -> new RustOWLQN().optimize(Linalg.vector(new double[]{1.0}), null, grad));
        assertThrows(IllegalArgumentException.class,
            () -> new RustOWLQN().optimize(Linalg.vector(new double[]{1.0}), obj, null));

        r.pass("null parameter exceptions thrown correctly");
    }

    // ========================================================================
    // 10. OWL-QN L1 sparsity: f(x) = ||x - target||^2 + c * ||x||_1
    //     With high L1 weight, solution should be sparse (many zeros)
    // ========================================================================

    @Test
    @DisplayName("10.1 OWL-QN L1 sparsity")
    @Timeout(value = 60, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOwlqnL1Sparsity() {
        double[] target = {0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5};
        double orthantwiseC = 5.0;

        IObjectiveFunction obj = x -> {
            double sumSq = 0.0;
            double sumL1 = 0.0;
            for (int i = 0; i < x.size(); i++) {
                double d = x.get(i) - target[i];
                sumSq += d * d;
                sumL1 += Math.abs(x.get(i));
            }
            return sumSq + orthantwiseC * sumL1;
        };
        IGradientFunction grad = x -> {
            double[] g = new double[x.size()];
            for (int i = 0; i < x.size(); i++) {
                double xi = x.get(i);
                g[i] = 2.0 * (xi - target[i]) + orthantwiseC * Math.signum(xi);
            }
            return Linalg.vector(g);
        };

        IVector initX = Linalg.vector(new double[]{2.0, 2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0});

        TestResult r = recorder.record("rust_owlqn", "l1_sparsity");
        RustOWLQN owlqn = new RustOWLQN(10, 1e-7, 10000, orthantwiseC);
        OptResult result = owlqn.optimize(initX, obj, grad);

        if (result == null || result.getOptimalPoint() == null) {
            r.fail("OWL-QN returned null result");
            return;
        }

        int zeroCount = 0;
        double totalAbs = 0.0;
        for (int i = 0; i < result.getOptimalPoint().size(); i++) {
            double v = result.getOptimalPoint().get(i);
            totalAbs += Math.abs(v);
            if (Math.abs(v) < 1e-4) {
                zeroCount++;
            }
        }

        String detail = String.format("  value=%.8e, zeros=%d/%d, L1_norm=%.6f, iters=%d",
            result.getOptimalValue(), zeroCount, result.getOptimalPoint().size(),
            totalAbs, result.getIterations());

        // With high L1 weight, expect at least some variables driven to zero
        if (zeroCount >= 3 && totalAbs < 5.0) {
            r.pass("OWL-QN produced sparse solution" + detail);
        } else if (totalAbs < 0.5) {
            // Solution near zero — also acceptable (shrinkage effect)
            r.pass("OWL-QN produced near-zero solution (strong shrinkage)" + detail);
        } else {
            r.fail("OWL-QN did not produce sparse solution" + detail);
        }
    }

    // ========================================================================
    // 11. OWL-QN zero regularization: orthantwiseC = 0 must match L-BFGS
    // ========================================================================

    @Test
    @DisplayName("11.1 OWL-QN zero regularization matches L-BFGS")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOwlqnZeroRegularization() {
        double[] target = {1.5, -2.5, 3.0};
        IObjectiveFunction obj = x -> {
            double sum = 0.0;
            for (int i = 0; i < x.size(); i++) {
                double d = x.get(i) - target[i];
                sum += d * d;
            }
            return sum;
        };
        IGradientFunction grad = x -> {
            double[] g = new double[x.size()];
            for (int i = 0; i < x.size(); i++) {
                g[i] = 2.0 * (x.get(i) - target[i]);
            }
            return Linalg.vector(g);
        };

        IVector initX = Linalg.vector(new double[]{5.0, -5.0, 5.0});

        RustLBFGS lbfgs = new RustLBFGS(10, 1e-8, 5000);
        RustOWLQN owlqnC0 = new RustOWLQN(10, 1e-8, 5000, 0.0);

        OptResult lbfgsResult = lbfgs.optimize(initX, obj, grad);
        OptResult owlqnResult = owlqnC0.optimize(initX, obj, grad);

        TestResult r = recorder.record("rust_owlqn", "zero_reg_matches_lbfgs");

        if (lbfgsResult == null || owlqnResult == null) {
            r.fail("one or both optimizers returned null");
            return;
        }

        double valDiff = Math.abs(lbfgsResult.getOptimalValue() - owlqnResult.getOptimalValue());
        double pointDiff = 0.0;
        if (lbfgsResult.getOptimalPoint() != null && owlqnResult.getOptimalPoint() != null) {
            int dim = Math.min(lbfgsResult.getOptimalPoint().size(), owlqnResult.getOptimalPoint().size());
            for (int i = 0; i < dim; i++) {
                pointDiff += Math.abs(
                    lbfgsResult.getOptimalPoint().get(i) -
                    owlqnResult.getOptimalPoint().get(i));
            }
        }

        String detail = String.format(
            "\n  RustLBFGS: value=%.8e, iters=%d\n  RustOWLQN(c=0): value=%.8e, iters=%d\n  valDiff=%.2e, pointDiff=%.2e",
            lbfgsResult.getOptimalValue(), lbfgsResult.getIterations(),
            owlqnResult.getOptimalValue(), owlqnResult.getIterations(),
            valDiff, pointDiff);

        if (valDiff < 1e-6 && pointDiff < 1e-6) {
            r.pass("OWL-QN(c=0) matches L-BFGS" + detail);
        } else if (valDiff < 1e-4) {
            r.pass("OWL-QN(c=0) close to L-BFGS" + detail);
        } else {
            r.fail("OWL-QN(c=0) differs from L-BFGS" + detail,
                   lbfgsResult.getOptimalValue(), owlqnResult.getOptimalValue());
        }
    }
}
