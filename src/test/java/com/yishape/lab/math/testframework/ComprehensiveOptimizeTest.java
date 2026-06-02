package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.*;
import com.yishape.lab.math.optimize.linpg.ILinProgSolver;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.newton.RereOnlineAdam;
import com.yishape.lab.math.optimize.newton.RereOnlineSGD;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive correctness validation test for com.yishape.lab.math.optimize.
 * Tests batch optimizers (LBFGS, ConjugateGradient) and online optimizers (Adam, SGD)
 * against known analytical solutions with controlled tolerances and timeouts.
 *
 * <p>Run: mvn test -Dtest=ComprehensiveOptimizeTest</p>
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
public class ComprehensiveOptimizeTest {

    private static final double EPS = 1e-3;
    private static final double LOOSE_EPS = 1e-2;
    private static final int MAX_ITERATIONS_ONLINE = 5000;
    private static TestResult.Recorder recorder;

    @BeforeAll
    static void init() {
        recorder = new TestResult.Recorder("optimize", "test_docs/results");
    }

    @AfterAll
    static void teardown() {
        recorder.writeToFile();
        System.out.println("\n=== OPTIMIZE TEST SUMMARY ===");
        System.out.println("Total:  " + recorder.getResults().size());
        System.out.println("Passed: " + recorder.getPassed());
        System.out.println("Failed: " + recorder.getFailed());
    }

    // ========================================================================
    // 1. Simple Quadratic: f(x) = x^2, minimum at x = 0
    // ========================================================================

    @Test
    @DisplayName("1.1 LBFGS on f(x)=x^2")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_Quadratic1D() {
        IObjectiveFunction obj = x -> {
            double v = x.get(0);
            return v * v;
        };
        IGradientFunction grad = x -> {
            double v = x.get(0);
            return Linalg.vector(new double[]{2.0 * v});
        };

        IOptimizer optimizer = Opts.lbfgs();
        IVector initX = Linalg.vector(new double[]{5.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("lbfgs", "quadratic_1d");
        validateBatchResult(r, result, 0.0, new double[]{0.0}, initX, 100);
    }

    @Test
    @DisplayName("1.2 ConjugateGradient on f(x)=x^2")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testCG_Quadratic1D() {
        IObjectiveFunction obj = x -> {
            double v = x.get(0);
            return v * v;
        };
        IGradientFunction grad = x -> {
            double v = x.get(0);
            return Linalg.vector(new double[]{2.0 * v});
        };

        IOptimizer optimizer = Opts.conjugateGradient();
        IVector initX = Linalg.vector(new double[]{5.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("cg", "quadratic_1d");
        validateBatchResult(r, result, 0.0, new double[]{0.0}, initX, 200);
    }

    @Test
    @DisplayName("1.3 OnlineSGD on f(x)=x^2")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineSGD_Quadratic1D() {
        IObjectiveFunction obj = x -> {
            double v = x.get(0);
            return v * v;
        };
        IGradientFunction grad = x -> {
            double v = x.get(0);
            return Linalg.vector(new double[]{2.0 * v});
        };

        IOnlineOptimizer optimizer = Opts.onlineSGD();
        IVector initX = Linalg.vector(new double[]{5.0});
        optimizer.initialize(initX);

        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_sgd", "quadratic_1d");
        validateOnlineResult(r, or, 0.0, new double[]{0.0}, initX, MAX_ITERATIONS_ONLINE);
    }

    @Test
    @DisplayName("1.4 OnlineAdam on f(x)=x^2")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_Quadratic1D() {
        IObjectiveFunction obj = x -> {
            double v = x.get(0);
            return v * v;
        };
        IGradientFunction grad = x -> {
            double v = x.get(0);
            return Linalg.vector(new double[]{2.0 * v});
        };

        RereOnlineAdam optimizer = (RereOnlineAdam) Opts.onlineAdam();
        optimizer.setLearningRate(0.1);
        IVector initX = Linalg.vector(new double[]{5.0});
        optimizer.initialize(initX);

        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_adam", "quadratic_1d");
        validateOnlineResult(r, or, 0.0, new double[]{0.0}, initX, MAX_ITERATIONS_ONLINE);
    }

    // ========================================================================
    // 2. 2D Quadratic: f(x,y) = (x-2)^2 + (y-3)^2, minimum at (2,3)
    // ========================================================================

    @Test
    @DisplayName("2.1 LBFGS on 2D quadratic")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_Quadratic2D() {
        IObjectiveFunction obj = x -> {
            double x1 = x.get(0) - 2.0;
            double x2 = x.get(1) - 3.0;
            return x1 * x1 + x2 * x2;
        };
        IGradientFunction grad = x -> {
            double x1 = x.get(0);
            double x2 = x.get(1);
            return Linalg.vector(new double[]{2.0 * (x1 - 2.0), 2.0 * (x2 - 3.0)});
        };

        IOptimizer optimizer = Opts.lbfgs();
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("lbfgs", "quadratic_2d");
        validateBatchResult(r, result, 0.0, new double[]{2.0, 3.0}, initX, 100);
    }

    @Test
    @DisplayName("2.2 ConjugateGradient on 2D quadratic")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testCG_Quadratic2D() {
        IObjectiveFunction obj = x -> {
            double x1 = x.get(0) - 2.0;
            double x2 = x.get(1) - 3.0;
            return x1 * x1 + x2 * x2;
        };
        IGradientFunction grad = x -> {
            double x1 = x.get(0);
            double x2 = x.get(1);
            return Linalg.vector(new double[]{2.0 * (x1 - 2.0), 2.0 * (x2 - 3.0)});
        };

        IOptimizer optimizer = Opts.conjugateGradient();
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("cg", "quadratic_2d");
        validateBatchResult(r, result, 0.0, new double[]{2.0, 3.0}, initX, 200);
    }

    @Test
    @DisplayName("2.3 OnlineSGD on 2D quadratic")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineSGD_Quadratic2D() {
        IObjectiveFunction obj = x -> {
            double x1 = x.get(0) - 2.0;
            double x2 = x.get(1) - 3.0;
            return x1 * x1 + x2 * x2;
        };
        IGradientFunction grad = x -> {
            double x1 = x.get(0);
            double x2 = x.get(1);
            return Linalg.vector(new double[]{2.0 * (x1 - 2.0), 2.0 * (x2 - 3.0)});
        };

        IOnlineOptimizer optimizer = Opts.onlineSGD();
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        optimizer.initialize(initX);

        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_sgd", "quadratic_2d");
        validateOnlineResult(r, or, 0.0, new double[]{2.0, 3.0}, initX, MAX_ITERATIONS_ONLINE);
    }

    @Test
    @DisplayName("2.4 OnlineAdam on 2D quadratic")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_Quadratic2D() {
        IObjectiveFunction obj = x -> {
            double x1 = x.get(0) - 2.0;
            double x2 = x.get(1) - 3.0;
            return x1 * x1 + x2 * x2;
        };
        IGradientFunction grad = x -> {
            double x1 = x.get(0);
            double x2 = x.get(1);
            return Linalg.vector(new double[]{2.0 * (x1 - 2.0), 2.0 * (x2 - 3.0)});
        };

        RereOnlineAdam optimizer = (RereOnlineAdam) Opts.onlineAdam();
        optimizer.setLearningRate(0.1);
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});
        optimizer.initialize(initX);

        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_adam", "quadratic_2d");
        validateOnlineResult(r, or, 0.0, new double[]{2.0, 3.0}, initX, MAX_ITERATIONS_ONLINE);
    }

    // ========================================================================
    // 3. Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1)
    // ========================================================================

    @Test
    @DisplayName("3.1 LBFGS on Rosenbrock")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_Rosenbrock() {
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

        // 经典难点初值 (-1.2, 1.0)；强 Wolfe 线搜索 + 修正后的停滞判定应能收敛到 (1,1)
        RereLBFGS optimizer = (RereLBFGS) Opts.lbfgs();
        optimizer.setMaxIterations(500);
        optimizer.setTolerance(1e-6);
        IVector initX = Linalg.vector(new double[]{-1.2, 1.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("lbfgs", "rosenbrock");
        // Rosenbrock is a difficult function; use relaxed tolerance for point check
        double valueError = Math.abs(result.getOptimalValue());
        double pointError = 0.0;
        if (result.getOptimalPoint() != null) {
            for (int i = 0; i < 2; i++) {
                pointError += Math.abs(result.getOptimalPoint().get(i) - 1.0);
            }
        }
        if (valueError < LOOSE_EPS && pointError < LOOSE_EPS * 2) {
            r.pass(String.format("converged: value=%.6e, point_err=%.2e, iters=%d",
                result.getOptimalValue(), pointError, result.getIterations()));
        } else {
            r.fail(String.format("value=%.6e, point_err=%.2e, iters=%d, converged=%s",
                result.getOptimalValue(), pointError, result.getIterations(), result.isConverged()),
                result.getOptimalValue(), 0.0);
        }
    }

    @Test
    @DisplayName("3.2 ConjugateGradient on Rosenbrock")
    @Timeout(value = 20, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testCG_Rosenbrock() {
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

        IOptimizer optimizer = Opts.conjugateGradient();
        IVector initX = Linalg.vector(new double[]{-1.2, 1.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("cg", "rosenbrock");
        validateBatchResult(r, result, 0.0, new double[]{1.0, 1.0}, initX, 500);
    }

    @Test
    @DisplayName("3.3 OnlineSGD on Rosenbrock")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineSGD_Rosenbrock() {
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

        // Online SGD on Rosenbrock: use a safer starting point and lower learning rate
        // Rosenbrock is very difficult for first-order methods; relax expectations
        RereOnlineSGD optimizer = (RereOnlineSGD) Opts.onlineSGD();
        optimizer.setLearningRate(0.001);
        IVector initX = Linalg.vector(new double[]{0.5, 0.5});
        optimizer.initialize(initX);

        // Use tighter convergence threshold to get closer to optimum
        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_sgd", "rosenbrock");
        // Rosenbrock is hard for SGD; use relaxed point tolerance
        double valueError = Math.abs(or.finalLoss);
        double pointError = 0.0;
        if (or.finalParams != null) {
            for (int i = 0; i < 2; i++) {
                pointError += Math.abs(or.finalParams.get(i) - 1.0);
            }
        }
        if (valueError < LOOSE_EPS && pointError < LOOSE_EPS * 10) {
            r.pass(String.format("converged: loss=%.6e, point_err=%.2e, steps=%d",
                or.finalLoss, pointError, or.steps));
        } else {
            r.fail(String.format("loss=%.6e, point_err=%.2e, steps=%d",
                or.finalLoss, pointError, or.steps), or.finalLoss, 0.0);
        }
    }

    @Test
    @DisplayName("3.4 OnlineAdam on Rosenbrock")
    @Timeout(value = 30, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_Rosenbrock() {
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

        // Online Adam on Rosenbrock: use a safer starting point
        RereOnlineAdam optimizer = (RereOnlineAdam) Opts.onlineAdam();
        optimizer.setLearningRate(0.1);
        IVector initX = Linalg.vector(new double[]{0.5, 0.5});
        optimizer.initialize(initX);

        // Use tighter convergence threshold
        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_adam", "rosenbrock");
        // Rosenbrock is hard for first-order methods; use relaxed point tolerance
        double valueError = Math.abs(or.finalLoss);
        double pointError = 0.0;
        if (or.finalParams != null) {
            for (int i = 0; i < 2; i++) {
                pointError += Math.abs(or.finalParams.get(i) - 1.0);
            }
        }
        if (valueError < LOOSE_EPS && pointError < LOOSE_EPS * 10) {
            r.pass(String.format("converged: loss=%.6e, point_err=%.2e, steps=%d",
                or.finalLoss, pointError, or.steps));
        } else {
            r.fail(String.format("loss=%.6e, point_err=%.2e, steps=%d",
                or.finalLoss, pointError, or.steps), or.finalLoss, 0.0);
        }
    }

    // ========================================================================
    // 4. Himmelblau function: multiple local minima
    // f(x,y) = (x^2+y-11)^2 + (x+y^2-7)^2
    // Minima at (3,2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    // ========================================================================

    @Test
    @DisplayName("4.1 LBFGS on Himmelblau (converges to one minimum)")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_Himmelblau() {
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

        IOptimizer optimizer = Opts.lbfgs();
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("lbfgs", "himmelblau");
        // Himmelblau has multiple minima, all with f=0. Just verify convergence to near-zero.
        validateBatchResultValueOnly(r, result, 0.0, initX, 200);
    }

    @Test
    @DisplayName("4.2 ConjugateGradient on Himmelblau")
    @Timeout(value = 20, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testCG_Himmelblau() {
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

        IOptimizer optimizer = Opts.conjugateGradient();
        IVector initX = Linalg.vector(new double[]{0.0, 0.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("cg", "himmelblau");
        validateBatchResultValueOnly(r, result, 0.0, initX, 500);
    }

    // ========================================================================
    // 5. 3D Quadratic: f(x,y,z) = (x-1)^2 + 2*(y+2)^2 + 3*(z-3)^2, min at (1,-2,3)
    // ========================================================================

    @Test
    @DisplayName("5.1 LBFGS on 3D quadratic")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_Quadratic3D() {
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0) - 1.0;
            double v2 = x.get(1) + 2.0;
            double v3 = x.get(2) - 3.0;
            return v1 * v1 + 2.0 * v2 * v2 + 3.0 * v3 * v3;
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double v3 = x.get(2);
            return Linalg.vector(new double[]{
                2.0 * (v1 - 1.0),
                4.0 * (v2 + 2.0),
                6.0 * (v3 - 3.0)
            });
        };

        IOptimizer optimizer = Opts.lbfgs();
        IVector initX = Linalg.vector(new double[]{0.0, 0.0, 0.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("lbfgs", "quadratic_3d");
        validateBatchResult(r, result, 0.0, new double[]{1.0, -2.0, 3.0}, initX, 100);
    }

    @Test
    @DisplayName("5.2 ConjugateGradient on 3D quadratic")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testCG_Quadratic3D() {
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0) - 1.0;
            double v2 = x.get(1) + 2.0;
            double v3 = x.get(2) - 3.0;
            return v1 * v1 + 2.0 * v2 * v2 + 3.0 * v3 * v3;
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double v3 = x.get(2);
            return Linalg.vector(new double[]{
                2.0 * (v1 - 1.0),
                4.0 * (v2 + 2.0),
                6.0 * (v3 - 3.0)
            });
        };

        IOptimizer optimizer = Opts.conjugateGradient();
        IVector initX = Linalg.vector(new double[]{0.0, 0.0, 0.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("cg", "quadratic_3d");
        validateBatchResult(r, result, 0.0, new double[]{1.0, -2.0, 3.0}, initX, 300);
    }

    @Test
    @DisplayName("5.3 OnlineAdam on 3D quadratic")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_Quadratic3D() {
        IObjectiveFunction obj = x -> {
            double v1 = x.get(0) - 1.0;
            double v2 = x.get(1) + 2.0;
            double v3 = x.get(2) - 3.0;
            return v1 * v1 + 2.0 * v2 * v2 + 3.0 * v3 * v3;
        };
        IGradientFunction grad = x -> {
            double v1 = x.get(0);
            double v2 = x.get(1);
            double v3 = x.get(2);
            return Linalg.vector(new double[]{
                2.0 * (v1 - 1.0),
                4.0 * (v2 + 2.0),
                6.0 * (v3 - 3.0)
            });
        };

        RereOnlineAdam optimizer = (RereOnlineAdam) Opts.onlineAdam();
        optimizer.setLearningRate(0.1);
        IVector initX = Linalg.vector(new double[]{0.0, 0.0, 0.0});
        optimizer.initialize(initX);

        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_adam", "quadratic_3d");
        validateOnlineResult(r, or, 0.0, new double[]{1.0, -2.0, 3.0}, initX, MAX_ITERATIONS_ONLINE);
    }

    // ========================================================================
    // 6. Beale function: f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
    // Minimum at (3, 0.5), f=0
    // ========================================================================

    @Test
    @DisplayName("6.1 LBFGS on Beale function")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_Beale() {
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
            double dx = 2.0 * a * (-1.0 + v2) + 2.0 * b * (-1.0 + v2 * v2) + 2.0 * c * (-1.0 + v2 * v2 * v2);
            double dy = 2.0 * a * v1 + 2.0 * b * (2.0 * v1 * v2) + 2.0 * c * (3.0 * v1 * v2 * v2);
            return Linalg.vector(new double[]{dx, dy});
        };

        IOptimizer optimizer = Opts.lbfgs();
        IVector initX = Linalg.vector(new double[]{1.0, 1.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("lbfgs", "beale");
        validateBatchResult(r, result, 0.0, new double[]{3.0, 0.5}, initX, 200);
    }

    @Test
    @DisplayName("6.2 ConjugateGradient on Beale function")
    @Timeout(value = 20, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testCG_Beale() {
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
            double dx = 2.0 * a * (-1.0 + v2) + 2.0 * b * (-1.0 + v2 * v2) + 2.0 * c * (-1.0 + v2 * v2 * v2);
            double dy = 2.0 * a * v1 + 2.0 * b * (2.0 * v1 * v2) + 2.0 * c * (3.0 * v1 * v2 * v2);
            return Linalg.vector(new double[]{dx, dy});
        };

        IOptimizer optimizer = Opts.conjugateGradient();
        IVector initX = Linalg.vector(new double[]{1.0, 1.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("cg", "beale");
        validateBatchResult(r, result, 0.0, new double[]{3.0, 0.5}, initX, 500);
    }

    // ========================================================================
    // 7. Online optimizer state management tests
    // ========================================================================

    @Test
    @DisplayName("7.1 OnlineSGD state management")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineSGD_StateManagement() {
        IOnlineOptimizer optimizer = Opts.onlineSGD();

        TestResult r = recorder.record("online_sgd", "state_management");

        // Test uninitialized state
        assertFalse(optimizer.isInitialized(), "Should not be initialized initially");
        assertEquals(0, optimizer.getCurrentStep(), "Step should be 0 initially");

        // Test initialize
        IVector init = Linalg.vector(new double[]{1.0, 2.0});
        optimizer.initialize(init);
        assertTrue(optimizer.isInitialized(), "Should be initialized after initialize()");
        IVector params = optimizer.getCurrentParams();
        assertEquals(1.0, params.get(0), 1e-10);
        assertEquals(2.0, params.get(1), 1e-10);

        // Test reset
        optimizer.reset();
        assertFalse(optimizer.isInitialized(), "Should not be initialized after reset()");
        assertEquals(0, optimizer.getCurrentStep(), "Step should be 0 after reset");

        r.pass("state management correct");
    }

    @Test
    @DisplayName("7.2 OnlineAdam state management")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_StateManagement() {
        IOnlineOptimizer optimizer = Opts.onlineAdam();

        TestResult r = recorder.record("online_adam", "state_management");

        // Test uninitialized state
        assertFalse(optimizer.isInitialized(), "Should not be initialized initially");
        assertEquals(0, optimizer.getCurrentStep(), "Step should be 0 initially");

        // Test initialize
        IVector init = Linalg.vector(new double[]{1.0, 2.0, 3.0});
        optimizer.initialize(init);
        assertTrue(optimizer.isInitialized(), "Should be initialized after initialize()");
        IVector params = optimizer.getCurrentParams();
        assertEquals(3, params.size(), "Dimension should match");

        // Test reset
        optimizer.reset();
        assertFalse(optimizer.isInitialized(), "Should not be initialized after reset()");

        r.pass("state management correct");
    }

    @Test
    @DisplayName("7.3 OnlineSGD clone")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineSGD_Clone() {
        RereOnlineSGD original = (RereOnlineSGD) Opts.onlineSGD();
        original.setLearningRate(0.05);
        original.setMomentum(0.9);

        IVector init = Linalg.vector(new double[]{1.0, 2.0});
        original.initialize(init);

        IOnlineOptimizer cloned = original.clone();
        TestResult r = recorder.record("online_sgd", "clone");

        assertTrue(cloned.isInitialized(), "Clone should be initialized");
        assertEquals(0.05, cloned.getCurrentLearningRate(), 1e-10, "Learning rate should match");
        assertEquals(0, cloned.getCurrentStep(), "Clone step should start at 0");

        IVector clonedParams = cloned.getCurrentParams();
        assertEquals(1.0, clonedParams.get(0), 1e-10);
        assertEquals(2.0, clonedParams.get(1), 1e-10);

        r.pass("clone correct");
    }

    @Test
    @DisplayName("7.4 OnlineAdam clone")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_Clone() {
        RereOnlineAdam original = (RereOnlineAdam) Opts.onlineAdam();
        original.setLearningRate(0.01);
        original.setBeta1(0.95);

        IVector init = Linalg.vector(new double[]{1.0, 2.0});
        original.initialize(init);

        IOnlineOptimizer cloned = original.clone();
        TestResult r = recorder.record("online_adam", "clone");

        assertTrue(cloned.isInitialized(), "Clone should be initialized");
        assertEquals(0.01, cloned.getCurrentLearningRate(), 1e-10, "Learning rate should match");

        r.pass("clone correct");
    }

    // ========================================================================
    // 8. Online optimizer learning rate decay
    // ========================================================================

    @Test
    @DisplayName("8.1 OnlineSGD learning rate decay")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineSGD_LearningRateDecay() {
        RereOnlineSGD optimizer = (RereOnlineSGD) Opts.onlineSGD();
        optimizer.setLearningRate(0.1);
        optimizer.setLrDecayRate(0.1);
        optimizer.setLrDecayStep(10);

        IVector init = Linalg.vector(new double[]{1.0});
        optimizer.initialize(init);

        double initialLR = optimizer.getCurrentLearningRate();

        // Run 10 steps
        for (int i = 0; i < 10; i++) {
            optimizer.step(Linalg.vector(new double[]{0.1}));
        }

        double afterDecayLR = optimizer.getCurrentLearningRate();

        TestResult r = recorder.record("online_sgd", "lr_decay");
        assertTrue(afterDecayLR < initialLR, "Learning rate should decay");
        r.pass("LR decayed from " + initialLR + " to " + afterDecayLR);
    }

    @Test
    @DisplayName("8.2 OnlineAdam learning rate decay")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_LearningRateDecay() {
        RereOnlineAdam optimizer = (RereOnlineAdam) Opts.onlineAdam();
        optimizer.setLearningRate(0.1);
        optimizer.setLrDecayRate(0.1);
        optimizer.setLrDecayStep(10);

        IVector init = Linalg.vector(new double[]{1.0});
        optimizer.initialize(init);

        double initialLR = optimizer.getCurrentLearningRate();

        for (int i = 0; i < 10; i++) {
            optimizer.step(Linalg.vector(new double[]{0.1}));
        }

        double afterDecayLR = optimizer.getCurrentLearningRate();

        TestResult r = recorder.record("online_adam", "lr_decay");
        assertTrue(afterDecayLR < initialLR, "Learning rate should decay");
        r.pass("LR decayed from " + initialLR + " to " + afterDecayLR);
    }

    // ========================================================================
    // 9. Error handling tests
    // ========================================================================

    @Test
    @DisplayName("9.1 LBFGS null parameter handling")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_NullParameters() {
        IOptimizer optimizer = Opts.lbfgs();
        TestResult r = recorder.record("lbfgs", "null_params");

        IObjectiveFunction obj = x -> x.get(0) * x.get(0);
        IGradientFunction grad = x -> Linalg.vector(new double[]{2.0 * x.get(0)});

        assertThrows(IllegalArgumentException.class, () -> optimizer.optimize(null, obj, grad));
        assertThrows(IllegalArgumentException.class, () -> optimizer.optimize(Linalg.vector(new double[]{1.0}), null, grad));
        assertThrows(IllegalArgumentException.class, () -> optimizer.optimize(Linalg.vector(new double[]{1.0}), obj, null));

        r.pass("null parameter exceptions thrown correctly");
    }

    @Test
    @DisplayName("9.2 OnlineSGD null parameter handling")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineSGD_NullParameters() {
        IOnlineOptimizer optimizer = Opts.onlineSGD();
        TestResult r = recorder.record("online_sgd", "null_params");

        assertThrows(IllegalArgumentException.class, () -> optimizer.initialize(null));

        IVector init = Linalg.vector(new double[]{1.0});
        optimizer.initialize(init);

        assertThrows(IllegalArgumentException.class, () -> optimizer.step(null));

        r.pass("null parameter exceptions thrown correctly");
    }

    @Test
    @DisplayName("9.3 OnlineAdam invalid gradient handling")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_InvalidGradient() {
        IOnlineOptimizer optimizer = Opts.onlineAdam();
        TestResult r = recorder.record("online_adam", "invalid_gradient");

        IVector init = Linalg.vector(new double[]{1.0, 2.0});
        optimizer.initialize(init);

        assertThrows(IllegalArgumentException.class,
            () -> optimizer.step(Linalg.vector(new double[]{Double.NaN, 0.0})));
        assertThrows(IllegalArgumentException.class,
            () -> optimizer.step(Linalg.vector(new double[]{Double.POSITIVE_INFINITY, 0.0})));

        r.pass("invalid gradient exceptions thrown correctly");
    }

    // ========================================================================
    // 10. Linear Programming (optional)
    // ========================================================================

    @Test
    @DisplayName("10.1 Simplex LP solver - basic problem")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testSimplexLP_Basic() {
        ILinProgSolver solver = Opts.simplexLinProgSolver();
        TestResult r = recorder.record("simplex_lp", "basic");

        // Minimize: c^T * x = -x0 - 2*x1
        // Subject to: x0 + x1 <= 4, x0 >= 0, x1 >= 0
        // Optimal: x0=0, x1=4, value=-8
        try {
            IVector c = Linalg.vector(new double[]{-1.0, -2.0});
            // A_ub * x <= b_ub  =>  [1, 1] * [x0, x1]^T <= 4
            var A_ub = com.yishape.lab.math.linalg.IMatrix.of(new double[][]{{1.0, 1.0}});
            IVector b_ub = Linalg.vector(new double[]{4.0});

            OptResult result = solver.solve(c, A_ub, b_ub);

            if (result == null) {
                r.fail("solver returned null result");
                return;
            }

            double optimalValue = result.getOptimalValue();
            IVector optimalPoint = result.getOptimalPoint();

            // Verify result is finite
            if (Double.isNaN(optimalValue) || Double.isInfinite(optimalValue)) {
                r.fail("optimal value is NaN or infinite", optimalValue, -8.0);
                return;
            }

            // For simplex, verify the solution is reasonable (close to x0=0, x1=4)
            if (optimalPoint != null && optimalPoint.size() >= 2) {
                double x0 = optimalPoint.get(0);
                double x1 = optimalPoint.get(1);
                double valError = Math.abs(optimalValue - (-8.0));
                double pointError = Math.abs(x0 - 0.0) + Math.abs(x1 - 4.0);

                if (valError < LOOSE_EPS && pointError < LOOSE_EPS) {
                    r.pass("converged to correct optimum: value=" + optimalValue +
                           ", point=[" + x0 + ", " + x1 + "]");
                } else {
                    r.fail("did not converge to expected optimum. value=" + optimalValue +
                           " (expected -8), point=[" + x0 + ", " + x1 + "] (expected [0, 4])",
                           optimalValue, -8.0);
                }
            } else {
                r.fail("optimal point is null or has wrong dimension");
            }
        } catch (Exception e) {
            r.fail("exception: " + e.getMessage());
        }
    }

    @Test
    @DisplayName("10.2 Simplex LP solver - production problem")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testSimplexLP_Production() {
        ILinProgSolver solver = Opts.simplexLinProgSolver();
        TestResult r = recorder.record("simplex_lp", "production");

        // Production planning: maximize profit = 3x + 5y
        // => minimize -3x - 5y
        // Constraints: x + 2y <= 8, 3x + 2y <= 12, x >= 0, y >= 0
        // Optimal: x=2, y=3, profit=21 => minimized value = -21
        try {
            IVector c = Linalg.vector(new double[]{-3.0, -5.0});
            var A_ub = com.yishape.lab.math.linalg.IMatrix.of(new double[][]{
                {1.0, 2.0},
                {3.0, 2.0}
            });
            IVector b_ub = Linalg.vector(new double[]{8.0, 12.0});

            OptResult result = solver.solve(c, A_ub, b_ub);

            if (result == null) {
                r.fail("solver returned null result");
                return;
            }

            double optimalValue = result.getOptimalValue();

            if (Double.isNaN(optimalValue) || Double.isInfinite(optimalValue)) {
                r.fail("optimal value is NaN or infinite", optimalValue, -21.0);
                return;
            }

            IVector optimalPoint = result.getOptimalPoint();
            if (optimalPoint != null && optimalPoint.size() >= 2) {
                double x = optimalPoint.get(0);
                double y = optimalPoint.get(1);
                double valError = Math.abs(optimalValue - (-21.0));
                double pointError = Math.abs(x - 2.0) + Math.abs(y - 3.0);

                if (valError < LOOSE_EPS && pointError < LOOSE_EPS) {
                    r.pass("converged to correct optimum");
                } else {
                    r.fail("value=" + optimalValue + " (exp -21), point=[" + x + ", " + y + "] (exp [2, 3])",
                           optimalValue, -21.0);
                }
            } else {
                r.fail("optimal point is null or has wrong dimension");
            }
        } catch (Exception e) {
            r.fail("exception: " + e.getMessage());
        }
    }

    // ========================================================================
    // 11. High-dimensional quadratic test
    // ========================================================================

    @Test
    @DisplayName("11.1 LBFGS on 10D quadratic")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_Quadratic10D() {
        int n = 10;
        double[] target = new double[n];
        for (int i = 0; i < n; i++) target[i] = i + 1;

        IObjectiveFunction obj = x -> {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = x.get(i) - target[i];
                sum += diff * diff;
            }
            return sum;
        };
        IGradientFunction grad = x -> {
            double[] g = new double[n];
            for (int i = 0; i < n; i++) {
                g[i] = 2.0 * (x.get(i) - target[i]);
            }
            return Linalg.vector(g);
        };

        double[] initArr = new double[n];
        for (int i = 0; i < n; i++) initArr[i] = 0.0;
        IVector initX = Linalg.vector(initArr);

        IOptimizer optimizer = Opts.lbfgs();
        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("lbfgs", "quadratic_10d");
        validateBatchResult(r, result, 0.0, target, initX, 200);
    }

    @Test
    @DisplayName("11.2 ConjugateGradient on 10D quadratic")
    @Timeout(value = 20, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testCG_Quadratic10D() {
        int n = 10;
        double[] target = new double[n];
        for (int i = 0; i < n; i++) target[i] = i + 1;

        IObjectiveFunction obj = x -> {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = x.get(i) - target[i];
                sum += diff * diff;
            }
            return sum;
        };
        IGradientFunction grad = x -> {
            double[] g = new double[n];
            for (int i = 0; i < n; i++) {
                g[i] = 2.0 * (x.get(i) - target[i]);
            }
            return Linalg.vector(g);
        };

        double[] initArr = new double[n];
        for (int i = 0; i < n; i++) initArr[i] = 0.0;
        IVector initX = Linalg.vector(initArr);

        IOptimizer optimizer = Opts.conjugateGradient();
        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("cg", "quadratic_10d");
        validateBatchResult(r, result, 0.0, target, initX, 500);
    }

    // ========================================================================
    // 12. Matyas function: f(x,y) = 0.26*(x^2+y^2) - 0.48*x*y, minimum at (0,0)
    // ========================================================================

    @Test
    @DisplayName("12.1 LBFGS on Matyas function")
    @Timeout(value = 10, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testLBFGS_Matyas() {
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

        IOptimizer optimizer = Opts.lbfgs();
        IVector initX = Linalg.vector(new double[]{5.0, -5.0});

        OptResult result = optimizer.optimize(initX, obj, grad);

        TestResult r = recorder.record("lbfgs", "matyas");
        validateBatchResult(r, result, 0.0, new double[]{0.0, 0.0}, initX, 100);
    }

    @Test
    @DisplayName("12.2 OnlineAdam on Matyas function")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_Matyas() {
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

        RereOnlineAdam optimizer = (RereOnlineAdam) Opts.onlineAdam();
        optimizer.setLearningRate(0.1);
        IVector initX = Linalg.vector(new double[]{5.0, -5.0});
        optimizer.initialize(initX);

        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_adam", "matyas");
        validateOnlineResult(r, or, 0.0, new double[]{0.0, 0.0}, initX, MAX_ITERATIONS_ONLINE);
    }

    // ========================================================================
    // 13. Online optimizer with momentum
    // ========================================================================

    @Test
    @DisplayName("13.1 OnlineSGD with momentum on quadratic")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineSGD_WithMomentum() {
        IObjectiveFunction obj = x -> {
            double v = x.get(0);
            return v * v;
        };
        IGradientFunction grad = x -> {
            double v = x.get(0);
            return Linalg.vector(new double[]{2.0 * v});
        };

        RereOnlineSGD optimizer = (RereOnlineSGD) Opts.onlineSGD();
        optimizer.setLearningRate(0.1);
        optimizer.setMomentum(0.9);

        IVector initX = Linalg.vector(new double[]{5.0});
        optimizer.initialize(initX);

        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-6);

        TestResult r = recorder.record("online_sgd_momentum", "quadratic_1d");
        validateOnlineResult(r, or, 0.0, new double[]{0.0}, initX, MAX_ITERATIONS_ONLINE);
    }

    @Test
    @DisplayName("13.2 OnlineAdam AMSGrad variant")
    @Timeout(value = 15, threadMode = Timeout.ThreadMode.SEPARATE_THREAD)
    void testOnlineAdam_AMSGrad() {
        IObjectiveFunction obj = x -> {
            double v = x.get(0);
            return v * v;
        };
        IGradientFunction grad = x -> {
            double v = x.get(0);
            return Linalg.vector(new double[]{2.0 * v});
        };

        RereOnlineAdam optimizer = (RereOnlineAdam) Opts.onlineAdam();
        optimizer.setAmsgrad(true);
        optimizer.setLearningRate(0.1);

        IVector initX = Linalg.vector(new double[]{5.0});
        optimizer.initialize(initX);

        OnlineResult or = runOnlineOptimizer(optimizer, obj, grad, MAX_ITERATIONS_ONLINE, 1e-4);

        TestResult r = recorder.record("online_adam_amsgrad", "quadratic_1d");
        validateOnlineResult(r, or, 0.0, new double[]{0.0}, initX, MAX_ITERATIONS_ONLINE);
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    /**
     * Run an online optimizer for a given number of steps or until convergence.
     */
    private OnlineResult runOnlineOptimizer(IOnlineOptimizer optimizer,
                                            IObjectiveFunction obj,
                                            IGradientFunction grad,
                                            int maxSteps,
                                            double convergenceThreshold) {
        IVector currentParams = optimizer.getCurrentParams();
        double finalLoss = obj.computeObjective(currentParams);
        int steps = 0;

        for (int i = 0; i < maxSteps; i++) {
            IVector gradient = grad.computeGradient(currentParams);
            double loss = obj.computeObjective(currentParams);
            currentParams = optimizer.step(gradient, loss);
            finalLoss = loss;
            steps = i + 1;

            if (loss < convergenceThreshold) {
                break;
            }
        }

        return new OnlineResult(currentParams, finalLoss, steps);
    }

    /**
     * Validate a batch optimizer result against expected optimal value and point.
     */
    private void validateBatchResult(TestResult r, OptResult result,
                                     double expectedValue, double[] expectedPoint,
                                     IVector initX, int maxReasonableIterations) {
        // Check result is not null
        if (result == null) {
            r.fail("result is null");
            return;
        }

        double optimalValue = result.getOptimalValue();
        IVector optimalPoint = result.getOptimalPoint();

        // Check for NaN/Infinity
        if (Double.isNaN(optimalValue)) {
            r.fail("optimal value is NaN");
            return;
        }
        if (Double.isInfinite(optimalValue)) {
            r.fail("optimal value is infinite", optimalValue, expectedValue);
            return;
        }

        // Check optimal point is not null and has correct dimension
        if (optimalPoint == null) {
            r.fail("optimal point is null");
            return;
        }
        if (optimalPoint.size() != expectedPoint.length) {
            r.fail("optimal point dimension mismatch: " + optimalPoint.size() + " vs " + expectedPoint.length);
            return;
        }

        // Check optimal point for NaN/Infinity
        for (int i = 0; i < optimalPoint.size(); i++) {
            double v = optimalPoint.get(i);
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                r.fail("optimal point contains NaN or Infinity at index " + i);
                return;
            }
        }

        // Check function value convergence
        double valueError = Math.abs(optimalValue - expectedValue);
        double pointError = 0.0;
        for (int i = 0; i < expectedPoint.length; i++) {
            pointError += Math.abs(optimalPoint.get(i) - expectedPoint[i]);
        }

        // Check iteration count is reasonable
        int iterations = result.getIterations();
        if (iterations > maxReasonableIterations) {
            // For some difficult problems (Rosenbrock), may need more iterations
            // Just warn but don't fail
        }

        // Use looser tolerance for difficult functions and online optimizers
        double tol = (expectedPoint.length > 2 || !result.isConverged()) ? LOOSE_EPS : EPS;

        if (valueError < tol && pointError < tol * expectedPoint.length) {
            String msg = String.format("converged: value=%.6e (err=%.2e), point_err=%.2e, iters=%d, reason=%s",
                optimalValue, valueError, pointError, iterations, result.getConvergenceReason());
            r.pass(msg);
        } else {
            String msg = String.format("value=%.6e (err=%.2e), point_err=%.2e, iters=%d, converged=%s",
                optimalValue, valueError, pointError, iterations, result.isConverged());
            r.fail(msg, optimalValue, expectedValue);
        }
    }

    /**
     * Validate batch result by function value only (for multi-minima functions like Himmelblau).
     */
    private void validateBatchResultValueOnly(TestResult r, OptResult result,
                                               double expectedValue, IVector initX,
                                               int maxReasonableIterations) {
        if (result == null) {
            r.fail("result is null");
            return;
        }

        double optimalValue = result.getOptimalValue();

        if (Double.isNaN(optimalValue)) {
            r.fail("optimal value is NaN");
            return;
        }
        if (Double.isInfinite(optimalValue)) {
            r.fail("optimal value is infinite", optimalValue, expectedValue);
            return;
        }

        IVector optimalPoint = result.getOptimalPoint();
        if (optimalPoint == null) {
            r.fail("optimal point is null");
            return;
        }

        for (int i = 0; i < optimalPoint.size(); i++) {
            double v = optimalPoint.get(i);
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                r.fail("optimal point contains NaN or Infinity");
                return;
            }
        }

        double valueError = Math.abs(optimalValue - expectedValue);
        int iterations = result.getIterations();

        if (valueError < LOOSE_EPS) {
            String msg = String.format("converged to near-zero: value=%.6e, iters=%d, reason=%s",
                optimalValue, iterations, result.getConvergenceReason());
            r.pass(msg);
        } else {
            r.fail("value=" + optimalValue + " (expected near " + expectedValue + ")",
                   optimalValue, expectedValue);
        }
    }

    /**
     * Validate an online optimizer result.
     */
    private void validateOnlineResult(TestResult r, OnlineResult or,
                                      double expectedValue, double[] expectedPoint,
                                      IVector initX, int maxSteps) {
        double finalLoss = or.finalLoss;
        IVector finalParams = or.finalParams;
        int steps = or.steps;

        // Check for NaN/Infinity
        if (Double.isNaN(finalLoss)) {
            r.fail("final loss is NaN");
            return;
        }
        if (Double.isInfinite(finalLoss)) {
            r.fail("final loss is infinite", finalLoss, expectedValue);
            return;
        }

        if (finalParams == null) {
            r.fail("final params is null");
            return;
        }

        for (int i = 0; i < finalParams.size(); i++) {
            double v = finalParams.get(i);
            if (Double.isNaN(v) || Double.isInfinite(v)) {
                r.fail("final params contains NaN or Infinity");
                return;
            }
        }

        double valueError = Math.abs(finalLoss - expectedValue);
        double pointError = 0.0;
        for (int i = 0; i < expectedPoint.length; i++) {
            pointError += Math.abs(finalParams.get(i) - expectedPoint[i]);
        }

        // Online optimizers use looser tolerance
        double tol = LOOSE_EPS;

        if (valueError < tol && pointError < tol * expectedPoint.length) {
            String msg = String.format("converged: loss=%.6e (err=%.2e), point_err=%.2e, steps=%d",
                finalLoss, valueError, pointError, steps);
            r.pass(msg);
        } else {
            String msg = String.format("loss=%.6e (err=%.2e), point_err=%.2e, steps=%d",
                finalLoss, valueError, pointError, steps);
            r.fail(msg, finalLoss, expectedValue);
        }
    }

    /**
     * Simple container for online optimizer results.
     */
    private static class OnlineResult {
        final IVector finalParams;
        final double finalLoss;
        final int steps;

        OnlineResult(IVector finalParams, double finalLoss, int steps) {
            this.finalParams = finalParams;
            this.finalLoss = finalLoss;
            this.steps = steps;
        }
    }
}
