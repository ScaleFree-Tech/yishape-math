package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.*;
import com.yishape.lab.math.optimize.newton.RereConjugateGradient;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.newton.RustLBFGS;
import com.yishape.lab.math.optimize.newton.RustOWLQN;
import org.junit.jupiter.api.*;

/**
 * 大规模性能基准测试：Rust/HPC vs PureJava 优化器。
 * 覆盖简单二次（超大规模）、扩展 Rosenbrock（病态）、三角函数（多局部极小）。
 *
 * <p>BENCHMARK 行格式: BENCHMARK|optimize|{optimizer}|{test}|{dim}|{timeMs}|{iterations}|{fx}</p>
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
@Disabled("优化器性能基准测试，耗时长，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
public class OptimizerHpcBenchmarkTest {

    private static final long FIXED_SEED = 42L;

    private static void benchmark(String optimizer, String test, int dim, long timeMs, int iterations, double fx) {
        System.out.printf("BENCHMARK|optimize|%s|%s|%d|%d|%d|%.8e%n",
            optimizer, test, dim, timeMs, iterations, fx);
    }

    private static void warmup(Runnable task, int times) {
        for (int i = 0; i < times; i++) {
            try { task.run(); } catch (Exception ignored) { }
        }
    }

    private static long timeMillis(Runnable task) {
        long start = System.nanoTime();
        task.run();
        return (System.nanoTime() - start) / 1_000_000;
    }

    private static long runBenchmark(Runnable warmupTask, Runnable benchmarkTask,
                                     int warmupTimes, int runs) {
        warmup(warmupTask, warmupTimes);
        long total = 0;
        for (int i = 0; i < runs; i++) {
            total += timeMillis(benchmarkTask);
        }
        return total / runs;
    }

    // ====================================================================
    // Test problems
    // ====================================================================

    /** 简单二次：f(x) = ||x - target||^2 */
    private static IObjectiveFunction quadraticObj(double[] target) {
        return x -> {
            double sum = 0.0;
            for (int i = 0; i < x.size(); i++) {
                double d = x.get(i) - target[i];
                sum += d * d;
            }
            return sum;
        };
    }

    private static IGradientFunction quadraticGrad(double[] target) {
        return x -> {
            double[] g = new double[x.size()];
            for (int i = 0; i < x.size(); i++) {
                g[i] = 2.0 * (x.get(i) - target[i]);
            }
            return Linalg.vector(g);
        };
    }

    /** 扩展 Rosenbrock（链式 2D 块）：f(x) = Σ[(1-x_{2i})² + 100·(x_{2i+1}-x_{2i}²)²] */
    private static IObjectiveFunction extendedRosenbrockObj(int dim) {
        int pairs = dim / 2;
        return x -> {
            double sum = 0.0;
            for (int p = 0; p < pairs; p++) {
                double v1 = x.get(2 * p);
                double v2 = x.get(2 * p + 1);
                double a = 1.0 - v1;
                double b = v2 - v1 * v1;
                sum += a * a + 100.0 * b * b;
            }
            return sum;
        };
    }

    private static IGradientFunction extendedRosenbrockGrad(int dim) {
        int pairs = dim / 2;
        return x -> {
            double[] g = new double[dim];
            for (int p = 0; p < pairs; p++) {
                double v1 = x.get(2 * p);
                double v2 = x.get(2 * p + 1);
                int i = 2 * p;
                g[i] = -2.0 * (1.0 - v1) - 400.0 * v1 * (v2 - v1 * v1);
                g[i + 1] = 200.0 * (v2 - v1 * v1);
            }
            return Linalg.vector(g);
        };
    }

    /** 三角函数：f(x) = Σ[i·cos(x_i) - sin(x_i)] + 非线性耦合 */
    private static IObjectiveFunction trigObj(int dim) {
        return x -> {
            double sum = 0.0;
            for (int i = 0; i < x.size(); i++) {
                double xi = x.get(i);
                sum += (i + 1) * Math.cos(xi) - Math.sin(xi);
            }
            return sum;
        };
    }

    private static IGradientFunction trigGrad(int dim) {
        return x -> {
            double[] g = new double[dim];
            for (int i = 0; i < dim; i++) {
                double xi = x.get(i);
                g[i] = -(i + 1) * Math.sin(xi) - Math.cos(xi);
            }
            return Linalg.vector(g);
        };
    }

    // ====================================================================
    // Benchmark runners
    // ====================================================================

    private static void runOptimizers(String label, int dim,
                                       IObjectiveFunction obj, IGradientFunction grad,
                                       IVector initX, int maxIter, int runs) {
        IOptimizer[] opts = {
            new RustLBFGS(10, 1e-6, maxIter),
            new RereLBFGS(10, 1e-6, maxIter),
            new RereConjugateGradient(1e-6, maxIter, 0.5),
            new RustOWLQN(10, 1e-6, maxIter, 0.0)
        };
        String[] names = {"RustLBFGS", "RereLBFGS", "RereConjugateGradient", "RustOWLQN"};
        int warmupIters = Math.min(2, Math.max(1, maxIter / 1000));

        System.out.printf("\n--- %s (dim=%d, runs=%d, maxIter=%d) ---%n", label, dim, runs, maxIter);
        System.out.printf("%-28s %10s %10s %18s %8s%n", "Optimizer", "Time(ms)", "Iters", "f(x)", "Convg");
        System.out.println("-".repeat(80));

        for (int o = 0; o < opts.length; o++) {
            int finalO = o;
            OptResult singleResult = opts[o].optimize(initX, obj, grad);

            long avgTimeMs = runBenchmark(
                () -> timeOptimizer(opts[finalO], obj, grad, initX),
                () -> timeOptimizer(opts[finalO], obj, grad, initX),
                warmupIters, runs
            );

            benchmark(names[o], label, dim, avgTimeMs, singleResult.getIterations(), singleResult.getOptimalValue());
            System.out.printf("%-28s %10d %10d %18.8e %8s%n",
                names[o], avgTimeMs, singleResult.getIterations(), singleResult.getOptimalValue(),
                singleResult.isConverged() ? "yes" : "no");
        }
    }

    private static Object[] timeOptimizer(IOptimizer optimizer,
                                           IObjectiveFunction obj, IGradientFunction grad,
                                           IVector initX) {
        long start = System.nanoTime();
        OptResult result = optimizer.optimize(initX, obj, grad);
        long elapsed = (System.nanoTime() - start) / 1_000_000;
        return new Object[]{elapsed, result.getIterations(), result.getOptimalValue()};
    }

    // ====================================================================
    // Test 1: 简单二次 — 小中大规模全覆盖
    // ====================================================================

    @Test
    @DisplayName("1. Quadratic ||x-target||²")
    @Timeout(value = 300)
    void benchmarkQuadratic() {
        System.out.println("\n=== 1. Quadratic ||x - target||² ===");
        int[] dims = {2, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000};

        for (int dim : dims) {
            double[] target = new double[dim];
            for (int i = 0; i < dim; i++) target[i] = (i % 5) * 1.0;
            double[] init = new double[dim];
            for (int i = 0; i < dim; i++) init[i] = target[i] + 10.0;

            int runs = (dim >= 5000) ? 1 : (dim >= 1000 ? 2 : 3);
            runOptimizers("quadratic", dim,
                quadraticObj(target), quadraticGrad(target),
                Linalg.vector(init), 5000, runs);
        }
    }

    // ====================================================================
    // Test 2: 扩展 Rosenbrock — 病态、高迭代次数
    // ====================================================================

    @Test
    @DisplayName("2. Extended Rosenbrock")
    @Timeout(value = 600)
    void benchmarkExtendedRosenbrock() {
        System.out.println("\n=== 2. Extended Rosenbrock Σ[(1-x₂ᵢ)² + 100·(x₂ᵢ₊₁-x₂ᵢ²)²] ===");
        int[] dims = {10, 100, 500, 1000, 2000};

        for (int dim : dims) {
            double[] init = new double[dim];
            for (int i = 0; i < dim; i++) {
                init[i] = (i % 2 == 0) ? -1.5 : 0.5;
            }

            int runs = (dim >= 1000) ? 1 : 3;
            runOptimizers("rosenbrock", dim,
                extendedRosenbrockObj(dim), extendedRosenbrockGrad(dim),
                Linalg.vector(init), Math.max(10000, dim * 50), runs);
        }
    }

    // ====================================================================
    // Test 3: 三角函数 — 多局部极小、中等难度
    // ====================================================================

    @Test
    @DisplayName("3. Trigonometric Σ[i·cos(x_i) - sin(x_i)]")
    @Timeout(value = 600)
    void benchmarkTrigonometric() {
        System.out.println("\n=== 3. Trigonometric Σ[i·cos(x_i) - sin(x_i)] ===");
        int[] dims = {10, 50, 100, 500, 1000};

        for (int dim : dims) {
            double[] init = new double[dim];
            // 分布更广的起点
            for (int i = 0; i < dim; i++) init[i] = (i % 3 - 1) * 3.0;

            int runs = (dim >= 500) ? 1 : 3;
            runOptimizers("trig", dim,
                trigObj(dim), trigGrad(dim),
                Linalg.vector(init), 20000, runs);
        }
    }
}
