package com.yishape.lab.math.testframework;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.IDoubleVectorComputer;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.ml.clf.lr.RereLogisticRegression;
import com.yishape.lab.math.ml.clf.tree.RereDecisionTree;
import com.yishape.lab.math.ml.clf.tree.RereRandomForest;
import com.yishape.lab.math.ml.clu.KMeansPlusPlus;
import com.yishape.lab.math.ml.dr.RerePCA;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IOnlineOptimizer;
import com.yishape.lab.math.optimize.Opts;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.signal.Signals;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereDCT;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.signal.filter.ButterworthFilter;
import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.stats.distribution.GammaDistribution;
import com.yishape.lab.math.stats.distribution.NormalDistribution;
import com.yishape.lab.math.timeseries.model.UnifiedARIMAModel;

import org.junit.jupiter.api.*;
import org.junit.jupiter.api.Timeout;

import java.util.Random;

/**
 * 全面性能基准测试
 * Comprehensive Performance Benchmark Test
 *
 * 输出格式: BENCHMARK|模块|操作|规模|时间_ms
 * Output format: BENCHMARK|module|operation|size|time_ms
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
@Disabled("性能基准测试，耗时长，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
public class PerformanceBenchmarkTest {

    private static final long FIXED_SEED = 42L;
    private static final Random RAND = new Random(FIXED_SEED);

    // ============ 辅助方法 ============

    private static void benchmark(String module, String operation, String size, long timeMs) {
        System.out.println("BENCHMARK|" + module + "|" + operation + "|" + size + "|" + timeMs);
    }

    private static void warmup(Runnable task, int times) {
        for (int i = 0; i < times; i++) {
            try {
                task.run();
            } catch (Exception e) {
                // warmup failures are ignored
            }
        }
    }

    private static long timeMillis(Runnable task) {
        long start = System.nanoTime();
        task.run();
        return (System.nanoTime() - start) / 1_000_000;
    }

    private static long runBenchmark(Runnable warmupTask, Runnable benchmarkTask, int warmupTimes, int runs) {
        warmup(warmupTask, warmupTimes);
        long total = 0;
        for (int i = 0; i < runs; i++) {
            total += timeMillis(benchmarkTask);
        }
        return total / runs;
    }

    // ============ 1. 线性代数性能 ============

    @Test
    @DisplayName("1.1 矩阵乘法")
    @Timeout(value = 300)
    void benchmarkMatrixMultiplication() {
        System.out.println("\n=== Benchmark: Matrix Multiplication ===");
        for (int n : new int[]{100, 500, 1000}) {
            IMatrix<Double> A = IMatrix.randn(n, n, FIXED_SEED);
            IMatrix<Double> B = IMatrix.randn(n, n, FIXED_SEED);
            int runs = (n >= 1000) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.mmul(B),
                () -> A.mmul(B),
                2, runs
            );
            benchmark("linalg", "mmul", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.2 SVD分解")
    @Timeout(value = 300)
    void benchmarkSVD() {
        System.out.println("\n=== Benchmark: SVD Decomposition ===");
        for (int n : new int[]{50, 100, 200}) {
            IMatrix<Double> A = IMatrix.randn(n, n, FIXED_SEED);
            int runs = (n >= 200) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.svd(),
                () -> A.svd(),
                2, runs
            );
            benchmark("linalg", "svd", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.3 QR分解")
    @Timeout(value = 300)
    void benchmarkQR() {
        System.out.println("\n=== Benchmark: QR Decomposition ===");
        for (int n : new int[]{50, 100, 200, 500}) {
            IMatrix<Double> A = IMatrix.randn(n, n, FIXED_SEED);
            int runs = (n >= 500) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.qr(),
                () -> A.qr(),
                2, runs
            );
            benchmark("linalg", "qr", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.4 Cholesky分解")
    @Timeout(value = 300)
    void benchmarkCholesky() {
        System.out.println("\n=== Benchmark: Cholesky Decomposition ===");
        for (int n : new int[]{50, 100, 200, 500}) {
            IMatrix<Double> base = IMatrix.randn(n, n, FIXED_SEED);
            IMatrix<Double> A = base.mmul(base.transpose()).add(IMatrix.<Double>eye(n).multiplyByScalar((double) n));
            // Explicitly symmetrize to avoid floating-point asymmetry triggering NonSymmetricMatrixException
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double avg = (A.get(i, j) + A.get(j, i)) / 2.0;
                    A.put(i, j, avg);
                    A.put(j, i, avg);
                }
            }
            final IMatrix<Double> Af = A;
            int runs = (n >= 500) ? 1 : 3;
            long ms = runBenchmark(
                () -> Af.cholesky(),
                () -> Af.cholesky(),
                2, runs
            );
            benchmark("linalg", "cholesky", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.5 LU分解")
    @Timeout(value = 300)
    void benchmarkLU() {
        System.out.println("\n=== Benchmark: LU Decomposition ===");
        for (int n : new int[]{50, 100, 200, 500}) {
            IMatrix<Double> A = IMatrix.randn(n, n, FIXED_SEED);
            int runs = (n >= 500) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.lu(),
                () -> A.lu(),
                2, runs
            );
            benchmark("linalg", "lu", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.6 线性求解 Ax=b")
    @Timeout(value = 300)
    void benchmarkLinearSolve() {
        System.out.println("\n=== Benchmark: Linear Solve Ax=b ===");
        for (int n : new int[]{100, 500, 1000}) {
            IMatrix<Double> A = IMatrix.randn(n, n, FIXED_SEED);
            IVector<Double> b = IVector.of(new double[n]);
            for (int i = 0; i < n; i++) {
                b.set(i, RAND.nextDouble());
            }
            int runs = (n >= 1000) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.solve(b),
                () -> A.solve(b),
                2, runs
            );
            benchmark("linalg", "solve", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.7 矩阵求逆")
    @Timeout(value = 300)
    void benchmarkMatrixInverse() {
        System.out.println("\n=== Benchmark: Matrix Inverse ===");
        for (int n : new int[]{50, 100, 200}) {
            IMatrix<Double> A = IMatrix.randn(n, n, FIXED_SEED);
            int runs = (n >= 200) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.inv(),
                () -> A.inv(),
                2, runs
            );
            benchmark("linalg", "inv", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.8 特征值分解（非对称）")
    @Timeout(value = 300)
    void benchmarkEigenDecomposition() {
        System.out.println("\n=== Benchmark: Eigen Decomposition (general) ===");
        for (int n : new int[]{50, 100, 200}) {
            IMatrix<Double> A = IMatrix.randn(n, n, FIXED_SEED);
            int runs = (n >= 200) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.eigen(),
                () -> A.eigen(),
                2, runs
            );
            benchmark("linalg", "eigen", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.8b 特征值分解（对称/HPC 加速）")
    @Timeout(value = 300)
    void benchmarkSymmetricEigenDecomposition() {
        System.out.println("\n=== Benchmark: Eigen Decomposition (symmetric, HPC-capable) ===");
        for (int n : new int[]{50, 100, 200}) {
            // 构造对称矩阵 A^T * A 以保证正定对称
            IMatrix<Double> R = IMatrix.randn(n, n, FIXED_SEED + 1);
            IMatrix<Double> Rt = R.transpose();
            IMatrix<Double> A = Rt.mmul(R);
            int runs = (n >= 200) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.eigen(),
                () -> A.eigen(),
                2, runs
            );
            benchmark("linalg", "eigen_sym", n + "x" + n, ms);
        }
    }

    @Test
    @DisplayName("1.9 伪逆")
    @Timeout(value = 300)
    void benchmarkPseudoInverse() {
        System.out.println("\n=== Benchmark: Pseudo Inverse ===");
        for (int m : new int[]{100, 500}) {
            int n = m / 2;
            IMatrix<Double> A = IMatrix.randn(m, n, FIXED_SEED);
            int runs = (m >= 500) ? 1 : 3;
            long ms = runBenchmark(
                () -> A.pinv(),
                () -> A.pinv(),
                2, runs
            );
            benchmark("linalg", "pinv", m + "x" + n, ms);
        }
    }

    // ============ 2. 统计性能 ============

    @Test
    @DisplayName("2.1 正态分布PDF/CDF计算")
    @Timeout(value = 300)
    void benchmarkNormalPDF() {
        System.out.println("\n=== Benchmark: Normal Distribution PDF ===");
        for (int n : new int[]{10000, 100000, 1000000}) {
            NormalDistribution dist = Stats.norm();
            int runs = (n >= 1000000) ? 1 : 3;
            long ms = runBenchmark(
                () -> {
                    for (int i = 0; i < n; i++) dist.pdf(i * 0.001);
                },
                () -> {
                    for (int i = 0; i < n; i++) dist.pdf(i * 0.001);
                },
                2, runs
            );
            benchmark("stats", "normal_pdf", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("2.2 正态分布采样")
    @Timeout(value = 300)
    void benchmarkNormalSampling() {
        System.out.println("\n=== Benchmark: Normal Distribution Sampling ===");
        for (int n : new int[]{10000, 100000, 1000000}) {
            NormalDistribution dist = Stats.norm();
            int runs = (n >= 1000000) ? 1 : 3;
            long ms = runBenchmark(
                () -> dist.sample(n),
                () -> dist.sample(n),
                2, runs
            );
            benchmark("stats", "normal_sample", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("2.3 Gamma分布采样")
    @Timeout(value = 300)
    void benchmarkGammaSampling() {
        System.out.println("\n=== Benchmark: Gamma Distribution Sampling ===");
        for (int n : new int[]{10000, 100000}) {
            GammaDistribution dist = Stats.gamma(2, 1);
            int runs = (n >= 100000) ? 1 : 3;
            long ms = runBenchmark(
                () -> dist.sample(n),
                () -> dist.sample(n),
                2, runs
            );
            benchmark("stats", "gamma_sample", String.valueOf(n), ms);
        }
    }

    // ============ 3. ML性能 ============

    @Test
    @DisplayName("3.1 逻辑回归训练")
    @Timeout(value = 300)
    void benchmarkLogisticRegression() {
        System.out.println("\n=== Benchmark: Logistic Regression Training ===");
        for (int n : new int[]{100, 500, 1000}) {
            IMatrix<Double> features = IMatrix.randn(n, 2, FIXED_SEED);
            String[] labels = new String[n];
            for (int i = 0; i < n; i++) {
                double sum = features.get(i, 0) + features.get(i, 1);
                labels[i] = sum > 0 ? "positive" : "negative";
            }
            RereLogisticRegression lr = new RereLogisticRegression();
            int runs = (n >= 1000) ? 1 : 3;
            long ms = runBenchmark(
                () -> lr.fit(features, labels),
                () -> lr.fit(features, labels),
                2, runs
            );
            benchmark("ml", "logistic_regression", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("3.2 决策树训练")
    @Timeout(value = 300)
    void benchmarkDecisionTree() {
        System.out.println("\n=== Benchmark: Decision Tree Training ===");
        for (int n : new int[]{100, 500, 1000, 5000}) {
            IMatrix<Double> features = IMatrix.randn(n, 2, FIXED_SEED);
            String[] labels = new String[n];
            for (int i = 0; i < n; i++) {
                double sum = features.get(i, 0) + features.get(i, 1);
                labels[i] = sum > 0 ? "positive" : "negative";
            }
            RereDecisionTree dt = new RereDecisionTree();
            int runs = (n >= 1000) ? 1 : 3;
            long ms = runBenchmark(
                () -> dt.fit(features, labels),
                () -> dt.fit(features, labels),
                2, runs
            );
            benchmark("ml", "decision_tree", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("3.3 随机森林训练")
    @Timeout(value = 300)
    void benchmarkRandomForest() {
        System.out.println("\n=== Benchmark: Random Forest Training ===");
        for (int n : new int[]{100, 500, 1000}) {
            IMatrix<Double> features = IMatrix.randn(n, 2, FIXED_SEED);
            String[] labels = new String[n];
            for (int i = 0; i < n; i++) {
                double sum = features.get(i, 0) + features.get(i, 1);
                labels[i] = sum > 0 ? "positive" : "negative";
            }
            RereRandomForest rf = new RereRandomForest();
            int runs = (n >= 1000) ? 1 : 3;
            long ms = runBenchmark(
                () -> rf.fit(features, labels),
                () -> rf.fit(features, labels),
                2, runs
            );
            benchmark("ml", "random_forest", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("3.4 KMeans聚类")
    @Timeout(value = 300)
    void benchmarkKMeans() {
        System.out.println("\n=== Benchmark: KMeans Clustering ===");
        for (int n : new int[]{100, 500, 1000}) {
            IMatrix<Double> data = IMatrix.randn(n, 2, FIXED_SEED);
            KMeansPlusPlus kmeans = new KMeansPlusPlus(3);
            int runs = (n >= 1000) ? 1 : 3;
            long ms = runBenchmark(
                () -> kmeans.fit(data),
                () -> kmeans.fit(data),
                2, runs
            );
            benchmark("ml", "kmeans", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("3.5 PCA降维")
    @Timeout(value = 300)
    void benchmarkPCA() {
        System.out.println("\n=== Benchmark: PCA Dimension Reduction ===");
        for (int n : new int[]{100, 500, 1000}) {
            IMatrix<Double> data = IMatrix.randn(n, 10, FIXED_SEED);
            RerePCA pca = new RerePCA().setNComponents(3);
            int runs = (n >= 1000) ? 1 : 3;
            long ms = runBenchmark(
                () -> pca.fitTransform(data),
                () -> pca.fitTransform(data),
                2, runs
            );
            benchmark("ml", "pca", String.valueOf(n), ms);
        }
    }

    // ============ 4. 优化器性能 ============

    private static final IObjectiveFunction rosenbrockObj = new IObjectiveFunction() {
        @Override
        public double computeObjective(IVector x) {
            double x0 = x.get(0);
            double x1 = x.get(1);
            double a = 1.0 - x0;
            double b = x1 - x0 * x0;
            return a * a + 100.0 * b * b;
        }
    };

    private static final IGradientFunction rosenbrockGrad = new IGradientFunction() {
        @Override
        public IVector computeGradient(IVector x) {
            double x0 = x.get(0);
            double x1 = x.get(1);
            double g0 = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0);
            double g1 = 200.0 * (x1 - x0 * x0);
            return IVector.of(new double[]{g0, g1});
        }
    };

    @Test
    @DisplayName("4.1 LBFGS优化")
    @Timeout(value = 300)
    void benchmarkLBFGS() {
        System.out.println("\n=== Benchmark: LBFGS Optimization ===");
        IVector<Double> init = IVector.of(new double[]{0.0, 0.0});
        IOptimizer lbfgs = Opts.lbfgs();
        // warmup
        lbfgs.optimize(init, rosenbrockObj, rosenbrockGrad);
        long start = System.nanoTime();
        lbfgs.optimize(init, rosenbrockObj, rosenbrockGrad);
        long ms = (System.nanoTime() - start) / 1_000_000;
        benchmark("optimize", "lbfgs", "rosenbrock_2d", ms);
    }

    @Test
    @DisplayName("4.2 共轭梯度优化")
    @Timeout(value = 300)
    void benchmarkConjugateGradient() {
        System.out.println("\n=== Benchmark: Conjugate Gradient Optimization ===");
        IVector<Double> init = IVector.of(new double[]{0.0, 0.0});
        IOptimizer cg = Opts.conjugateGradient();
        // warmup
        cg.optimize(init, rosenbrockObj, rosenbrockGrad);
        long start = System.nanoTime();
        cg.optimize(init, rosenbrockObj, rosenbrockGrad);
        long ms = (System.nanoTime() - start) / 1_000_000;
        benchmark("optimize", "cg", "rosenbrock_2d", ms);
    }

    @Test
    @DisplayName("4.3 在线Adam优化")
    @Timeout(value = 300)
    void benchmarkOnlineAdam() {
        System.out.println("\n=== Benchmark: Online Adam (1000 steps) ===");
        IVector<Double> init = IVector.of(new double[]{0.0, 0.0});
        IOnlineOptimizer adam = Opts.onlineAdam();
        adam.initialize(init);
        // warmup
        for (int i = 0; i < 100; i++) {
            adam.step(rosenbrockGrad.computeGradient(adam.getCurrentParams()));
        }
        adam.initialize(init);
        long start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            adam.step(rosenbrockGrad.computeGradient(adam.getCurrentParams()));
        }
        long ms = (System.nanoTime() - start) / 1_000_000;
        benchmark("optimize", "online_adam", "1000_steps", ms);
    }

    @Test
    @DisplayName("4.4 在线SGD优化")
    @Timeout(value = 300)
    void benchmarkOnlineSGD() {
        System.out.println("\n=== Benchmark: Online SGD (1000 steps) ===");
        // Use a simpler quadratic objective for SGD to avoid NaN with Rosenbrock
        IVector<Double> init = IVector.of(new double[]{1.0, 1.0});
        IOnlineOptimizer sgd = Opts.onlineSGD();
        sgd.initialize(init);
        IGradientFunction simpleGrad = new IGradientFunction() {
            @Override
            public IVector computeGradient(IVector x) {
                double x0 = x.get(0);
                double x1 = x.get(1);
                return IVector.of(new double[]{2.0 * (x0 - 1.0), 2.0 * (x1 - 1.0)});
            }
        };
        // warmup
        for (int i = 0; i < 100; i++) {
            sgd.step(simpleGrad.computeGradient(sgd.getCurrentParams()));
        }
        sgd.initialize(init);
        long start = System.nanoTime();
        for (int i = 0; i < 1000; i++) {
            sgd.step(simpleGrad.computeGradient(sgd.getCurrentParams()));
        }
        long ms = (System.nanoTime() - start) / 1_000_000;
        benchmark("optimize", "online_sgd", "1000_steps", ms);
    }

    // ============ 5. 信号处理性能 ============

    @Test
    @DisplayName("5.1 FFT")
    @Timeout(value = 300)
    void benchmarkFFT() {
        System.out.println("\n=== Benchmark: FFT ===");
        for (int n : new int[]{1024, 4096, 16384, 65536}) {
            Complex[] signal = new Complex[n];
            for (int i = 0; i < n; i++) {
                signal[i] = new Complex(Math.sin(2 * Math.PI * i / n), 0);
            }
            int runs = (n >= 65536) ? 1 : 3;
            long ms = runBenchmark(
                () -> RereFFT.fft(signal),
                () -> RereFFT.fft(signal),
                2, runs
            );
            benchmark("signal", "fft", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("5.2 DCT")
    @Timeout(value = 300)
    void benchmarkDCT() {
        System.out.println("\n=== Benchmark: DCT ===");
        for (int n : new int[]{1024, 4096, 16384}) {
            double[] data = new double[n];
            for (int i = 0; i < n; i++) {
                data[i] = Math.sin(2 * Math.PI * i / n);
            }
            IVector<Double> signal = IVector.of(data);
            int runs = (n >= 16384) ? 1 : 3;
            long ms = runBenchmark(
                () -> RereDCT.dct2(signal),
                () -> RereDCT.dct2(signal),
                2, runs
            );
            benchmark("signal", "dct", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("5.3 Butterworth滤波器")
    @Timeout(value = 300)
    void benchmarkButterworthFilter() throws Exception {
        System.out.println("\n=== Benchmark: Butterworth Filter ===");
        for (int n : new int[]{1000, 10000, 100000}) {
            IVector<Double> signal = Signals.sineWave(n, 10, 1000, 1, 0);
            int runs = (n >= 100000) ? 1 : 3;
            long ms = runBenchmark(
                () -> {
                    try {
                        ButterworthFilter filter = new ButterworthFilter(4, 100, 1000);
                        filter.filter(signal);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                },
                () -> {
                    try {
                        ButterworthFilter filter = new ButterworthFilter(4, 100, 1000);
                        filter.filter(signal);
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                },
                2, runs
            );
            benchmark("signal", "butterworth_filter", String.valueOf(n), ms);
        }
    }

    // ============ 6. 时间序列性能 ============

    @Test
    @DisplayName("6.1 ARIMA拟合")
    @Timeout(value = 300)
    void benchmarkARIMA() {
        System.out.println("\n=== Benchmark: ARIMA Fitting ===");
        for (int n : new int[]{100, 500, 1000}) {
            IVector<Double> data = Signals.sineWave(n, 1, 100, 1, 0);
            int runs = (n >= 1000) ? 1 : 3;
            long ms = runBenchmark(
                () -> UnifiedARIMAModel.fit(data, 2, 0, 2),
                () -> UnifiedARIMAModel.fit(data, 2, 0, 2),
                2, runs
            );
            benchmark("timeseries", "arima_fit", String.valueOf(n), ms);
        }
    }

    // ============ 7. 向量运算性能 ============

    @Test
    @DisplayName("7.1 大向量逐元素运算")
    @Timeout(value = 300)
    void benchmarkVectorElementWise() {
        System.out.println("\n=== Benchmark: Vector Element-wise Operations ===");
        DoubleVectorComputer computer = new DoubleVectorComputer();
        for (int n : new int[]{10000, 100000, 1000000, 10000000}) {
            double[] a = new double[n];
            double[] b = new double[n];
            for (int i = 0; i < n; i++) {
                a[i] = RAND.nextDouble();
                b[i] = RAND.nextDouble();
            }
            int runs = (n >= 1000000) ? 1 : 3;
            long ms = runBenchmark(
                () -> computer.binaryOperate(a, b, IDoubleVectorComputer.BinaryOperation.ADD),
                () -> computer.binaryOperate(a, b, IDoubleVectorComputer.BinaryOperation.ADD),
                2, runs
            );
            benchmark("compute", "vector_add", String.valueOf(n), ms);
        }
    }

    @Test
    @DisplayName("7.2 大向量归约运算")
    @Timeout(value = 300)
    void benchmarkVectorReduction() {
        System.out.println("\n=== Benchmark: Vector Reduction Operations ===");
        DoubleVectorComputer computer = new DoubleVectorComputer();
        for (int n : new int[]{10000, 100000, 1000000, 10000000}) {
            double[] a = new double[n];
            for (int i = 0; i < n; i++) {
                a[i] = RAND.nextDouble();
            }
            int runs = (n >= 1000000) ? 1 : 3;
            long msSum = runBenchmark(
                () -> computer.reduceOperate(a, IDoubleVectorComputer.ReduceOperation.SUM),
                () -> computer.reduceOperate(a, IDoubleVectorComputer.ReduceOperation.SUM),
                2, runs
            );
            benchmark("compute", "vector_sum", String.valueOf(n), msSum);

            long msMean = runBenchmark(
                () -> computer.reduceOperate(a, IDoubleVectorComputer.ReduceOperation.MEAN),
                () -> computer.reduceOperate(a, IDoubleVectorComputer.ReduceOperation.MEAN),
                2, runs
            );
            benchmark("compute", "vector_mean", String.valueOf(n), msMean);
        }
    }
}
