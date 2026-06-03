package com.yishape.lab.math.bench;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import com.yishape.lab.math.compute.DoubleVectorComputer;
import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.compute.ops.BinaryOperation;
import com.yishape.lab.math.compute.ops.ReduceOperation;
import com.yishape.lab.math.hpc.YishapeHpc;
import com.yishape.lab.math.hpc.YishapeHpcStatus;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.IGradientFunction;
import com.yishape.lab.math.optimize.IOptimizer;
import com.yishape.lab.math.optimize.IObjectiveFunction;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import com.yishape.lab.math.optimize.newton.RereConjugateGradient;
import com.yishape.lab.math.optimize.newton.RereLBFGS;
import com.yishape.lab.math.optimize.newton.RustLBFGS;
import com.yishape.lab.math.optimize.newton.RustOWLQN;
import com.yishape.lab.math.signal.core.Complex;
import com.yishape.lab.math.signal.core.RereDCT;
import com.yishape.lab.math.signal.core.RereFFT;
import com.yishape.lab.math.stats.Stats;
import com.yishape.lab.math.stats.distribution.GammaDistribution;
import com.yishape.lab.math.stats.distribution.NormalDistribution;

/**
 * Unified benchmark: Pure Java vs HPC at identical sizes for every operation.
 * Output: benchmarks/out/unified_java.csv
 *
 * <p>Run with HPC JAR on classpath for HPC data; toggles via system property.</p>
 */
public final class UnifiedBenchmarkMain {

    private static final int WARMUP = 2;
    private static final int REPEAT = 5;
    private static final long SEED = 42L;
    private static final Path OUT = Path.of("benchmarks/out/unified_java.csv");

    private UnifiedBenchmarkMain() {}

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUT.getParent());
        boolean nativeOk = YishapeHpc.isNativeRuntimeAvailable();
        System.out.println("META_JSON {\"java\":\"" + Runtime.version() + "\",\"hpcNative\":" + nativeOk + "}");

        List<Row> rows = new ArrayList<>();

        // Run both pure Java and HPC
        for (boolean hpc : new boolean[]{false, true}) {
            setHpcMode(hpc);
            String backend = hpc ? "java_hpc" : "java_pure";
            String tag = hpc ? "HPC" : "Pure Java";
            System.out.println("\n========== " + tag + " ==========");

            benchGemm(rows, backend);
            benchSvd(rows, backend);
            benchQr(rows, backend);
            benchCholesky(rows, backend);
            benchLu(rows, backend);
            benchSolve(rows, backend);
            benchInverse(rows, backend);
            benchEigenGeneral(rows, backend);
            benchEigenSymmetric(rows, backend, hpc, nativeOk);
            benchPseudoinverse(rows, backend);
            benchHilbertSolve(rows, backend);
            benchDetRank(rows, backend);

            // Optimizers (run for both modes, but implementations differ)
            benchOptimizers(rows, backend, hpc);

            // Stats, signal, vector — HPC doesn't affect these, but run both for completeness
            benchStats(rows, backend);
            benchSignal(rows, backend);
            benchVector(rows, backend);
        }

        // LP benchmarks (HiGHS vs Simplex, each in its own mode)
        benchLpGrid(rows, nativeOk);
        benchLpSimplex(rows);

        writeCsv(rows);
        System.out.println("WROTE " + OUT.toAbsolutePath() + " (" + rows.size() + " rows)");
    }

    // ========== HPC mode control ==========

    private static void setHpcMode(boolean enable) {
        System.setProperty(HpcConfig.PROP_ENABLE, Boolean.toString(enable));
        if (enable) {
            System.setProperty(HpcConfig.PROP_GEMM_MIN_FLOPS, "0");
            System.setProperty(HpcConfig.PROP_CHOLESKY_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_SVD_MIN_TOTAL_ELEMENTS, "1");
            System.setProperty(HpcConfig.PROP_SOLVE_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_LU_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_EIGEN_MIN_DIM, "32");
        }
    }

    // ========== 1. GEMM ==========

    private static void benchGemm(List<Row> rows, String backend) {
        for (int n : new int[]{100, 500, 1000, 1500}) {
            IMatrix<Double> a = Linalg.rand(n, n, SEED);
            IMatrix<Double> b = Linalg.rand(n, n, SEED + 1);
            int runs = (n >= 1000) ? 1 : REPEAT;
            double ms = benchMs(() -> a.mmul(b), runs);
            rows.add(new Row("java", "linalg", "mmul", n + "x" + n, backend, ms));
        }
    }

    // ========== 2. SVD ==========

    private static void benchSvd(List<Row> rows, String backend) {
        for (int n : new int[]{50, 100, 200, 500, 1000}) {
            IMatrix<Double> a = Linalg.rand(n, n, SEED + 2 + n);
            int runs = (n >= 200) ? 1 : REPEAT;
            double ms = benchMs(() -> a.svd(), runs);
            rows.add(new Row("java", "linalg", "svd", n + "x" + n, backend, ms));
        }
    }

    // ========== 3. QR ==========

    private static void benchQr(List<Row> rows, String backend) {
        for (int n : new int[]{50, 100, 200, 500, 1000}) {
            IMatrix<Double> a = Linalg.rand(n, n, SEED + 3 + n);
            int runs = (n >= 500) ? 1 : REPEAT;
            double ms = benchMs(() -> a.qr(), runs);
            rows.add(new Row("java", "linalg", "qr", n + "x" + n, backend, ms));
        }
    }

    // ========== 4. Cholesky ==========

    private static void benchCholesky(List<Row> rows, String backend) {
        for (int n : new int[]{50, 100, 200, 500, 1000, 1500}) {
            IMatrix<Double> spd = spdMatrix(n, SEED + 4);
            int runs = (n >= 500) ? 1 : REPEAT;
            double ms = benchMs(() -> spd.cholesky(), runs);
            rows.add(new Row("java", "linalg", "cholesky", n + "x" + n, backend, ms));
        }
    }

    // ========== 5. LU ==========

    private static void benchLu(List<Row> rows, String backend) {
        for (int n : new int[]{50, 100, 200, 500, 1000, 1500}) {
            IMatrix<Double> a = Linalg.rand(n, n, SEED + 5 + n);
            int runs = (n >= 500) ? 1 : REPEAT;
            double ms = benchMs(() -> a.lu(), runs);
            rows.add(new Row("java", "linalg", "lu", n + "x" + n, backend, ms));
        }
    }

    // ========== 6. Solve ==========

    private static void benchSolve(List<Row> rows, String backend) {
        for (int n : new int[]{100, 500, 1000, 1500}) {
            IMatrix<Double> a = wellConditionedMatrix(n, SEED + 6);
            IVector<Double> b = randVector(n, SEED + 7);
            int runs = (n >= 1000) ? 1 : REPEAT;
            double ms = benchMs(() -> a.solve(b), runs);
            rows.add(new Row("java", "linalg", "solve", "n=" + n, backend, ms));
        }
    }

    // ========== 7. Inverse ==========

    private static void benchInverse(List<Row> rows, String backend) {
        for (int n : new int[]{50, 100, 200, 500}) {
            IMatrix<Double> a = wellConditionedMatrix(n, SEED + 8);
            int runs = (n >= 200) ? 1 : REPEAT;
            double ms = benchMs(() -> a.inv(), runs);
            rows.add(new Row("java", "linalg", "inverse", n + "x" + n, backend, ms));
        }
    }

    // ========== 8. Eigen general ==========

    private static void benchEigenGeneral(List<Row> rows, String backend) {
        for (int n : new int[]{50, 100, 200}) {
            IMatrix<Double> a = Linalg.rand(n, n, SEED + 9 + n);
            int runs = (n >= 200) ? 1 : REPEAT;
            double ms = benchMs(() -> a.eigen(), runs);
            rows.add(new Row("java", "linalg", "eigen_general", n + "x" + n, backend, ms));
        }
    }

    // ========== 9. Eigen symmetric ==========

    private static void benchEigenSymmetric(List<Row> rows, String backend, boolean hpc, boolean nativeOk) {
        for (int n : new int[]{50, 100, 200, 500}) {
            IMatrix<Double> sym = symMatrix(n, SEED + 10);
            int runs = (n >= 200) ? 1 : REPEAT;
            double ms = benchMs(() -> sym.eigen(), runs);
            rows.add(new Row("java", "linalg", "eigen_symmetric", n + "x" + n, backend, ms));
        }
    }

    // ========== 10. Pseudoinverse ==========

    private static void benchPseudoinverse(List<Row> rows, String backend) {
        for (int[] mn : new int[][]{{100, 50}, {500, 250}}) {
            int m = mn[0], n = mn[1];
            IMatrix<Double> a = Linalg.rand(m, n, SEED + 11);
            double ms = benchMs(() -> a.pinv(), REPEAT);
            rows.add(new Row("java", "linalg", "pseudoinverse", m + "x" + n, backend, ms));
        }
    }

    // ========== 11. Hilbert solve ==========

    private static void benchHilbertSolve(List<Row> rows, String backend) {
        for (int n : new int[]{5, 10, 15}) {
            IMatrix<Double> h = hilbertMatrix(n);
            IVector<Double> ones = Linalg.vector(fill(n, 1.0));
            IVector<Double> b = h.mmul(ones);
            double ms = benchMs(() -> h.solve(b), REPEAT);
            rows.add(new Row("java", "linalg", "hilbert_solve", "n=" + n, backend, ms));
        }
    }

    // ========== 12. Determinant & Rank ==========

    private static void benchDetRank(List<Row> rows, String backend) {
        for (int n : new int[]{50, 100, 500}) {
            IMatrix<Double> a = wellConditionedMatrix(n, SEED + 12);
            double msDet = benchMs(() -> a.det(), 1);
            rows.add(new Row("java", "linalg", "det", "n=" + n, backend, msDet));
            double msRank = benchMs(() -> a.rank(), 1);
            rows.add(new Row("java", "linalg", "rank", "n=" + n, backend, msRank));
        }
    }

    // ========== 13. Optimizers ==========

    private static void benchOptimizers(List<Row> rows, String backend, boolean hpc) {
        // 2D Rosenbrock — LBFGS
        IVector<Double> init2d = Linalg.vector(new double[]{0.0, 0.0});

        IOptimizer rustLbfgs = new RustLBFGS(10, 1e-12, 200);
        IOptimizer rereLbfgs = new RereLBFGS(10, 1e-12, 200);
        IOptimizer cg = new RereConjugateGradient(1e-12, 200, 0.5);
        IOptimizer owlqn = new RustOWLQN(10, 1e-12, 200, 0.0);

        // Warmup all
        rustLbfgs.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        rereLbfgs.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        cg.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);
        owlqn.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD);

        // Benchmark Rosenbrock 2D
        double msRust = benchMs(() -> rustLbfgs.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        rows.add(new Row("java", "optimize", "lbfgs_rosenbrock", "2d", backend + "_rustlbfgs", msRust));

        double msRere = benchMs(() -> rereLbfgs.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        rows.add(new Row("java", "optimize", "lbfgs_rosenbrock", "2d", backend + "_rerelbfgs", msRere));

        double msCg = benchMs(() -> cg.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        rows.add(new Row("java", "optimize", "cg_rosenbrock", "2d", backend, msCg));

        double msOwl = benchMs(() -> owlqn.optimize(init2d, ROSENBROCK_OBJ, ROSENBROCK_GRAD), REPEAT);
        rows.add(new Row("java", "optimize", "owlqn_rosenbrock", "2d", backend, msOwl));

        // Quadratic ||x - target||²
        for (int dim : new int[]{10, 100, 500, 1000}) {
            double[] targetArr = fill(dim, 1.0);
            double[] initArr = fill(dim, 0.0);
            IVector<Double> target = Linalg.vector(targetArr);
            IVector<Double> initX = Linalg.vector(initArr);
            IObjectiveFunction obj = quadraticObj(target);
            IGradientFunction grad = quadraticGrad(target);

            int runs = (dim >= 1000) ? 1 : REPEAT;

            IOptimizer qRust = new RustLBFGS(10, 1e-12, 100);
            IOptimizer qRere = new RereLBFGS(10, 1e-12, 100);
            qRust.optimize(initX, obj, grad);
            qRere.optimize(initX, obj, grad);

            double msQRust = benchMs(() -> qRust.optimize(initX, obj, grad), runs);
            rows.add(new Row("java", "optimize", "lbfgs_quadratic", "d=" + dim, backend + "_rustlbfgs", msQRust));

            double msQRere = benchMs(() -> qRere.optimize(initX, obj, grad), runs);
            rows.add(new Row("java", "optimize", "lbfgs_quadratic", "d=" + dim, backend + "_rerelbfgs", msQRere));
        }

        // Extended Rosenbrock
        for (int dim : new int[]{10, 100, 500}) {
            double[] initArr = new double[dim];
            for (int i = 0; i < dim; i++) {
                initArr[i] = (i % 2 == 0) ? -1.5 : 0.5;
            }
            IVector<Double> initX = Linalg.vector(initArr);
            IObjectiveFunction obj = extendedRosenbrockObj(dim);
            IGradientFunction grad = extendedRosenbrockGrad(dim);

            int runs = (dim >= 500) ? 1 : REPEAT;

            IOptimizer eRust = new RustLBFGS(10, 1e-12, Math.max(10000, dim * 50));
            IOptimizer eRere = new RereLBFGS(10, 1e-12, Math.max(10000, dim * 50));
            eRust.optimize(initX, obj, grad);
            eRere.optimize(initX, obj, grad);

            double msERust = benchMs(() -> eRust.optimize(initX, obj, grad), runs);
            rows.add(new Row("java", "optimize", "lbfgs_ext_rosenbrock", "d=" + dim, backend + "_rustlbfgs", msERust));

            double msERere = benchMs(() -> eRere.optimize(initX, obj, grad), runs);
            rows.add(new Row("java", "optimize", "lbfgs_ext_rosenbrock", "d=" + dim, backend + "_rerelbfgs", msERere));
        }
    }

    // ========== 14. LP Grid ==========

    private static void benchLpGrid(List<Row> rows, boolean nativeOk) {
        int[] ns = {250, 500, 1000, 2000, 3500};
        int[] mMults = {1, 2};

        // HPC HiGHS
        setHpcMode(true);
        if (nativeOk) {
            for (int mult : mMults) {
                for (int n : ns) {
                    int mLe = n * mult;
                    LpProblem p = buildLp(n, mLe, 424242L + n * 31L + mult);
                    double ms = benchMs(() -> {
                        var r = YishapeHpc.lpNonnegative(p.c, p.aUb, p.bUb, null, null);
                        if (r.status() != YishapeHpcStatus.OK) {
                            throw new IllegalStateException("highs " + r.status());
                        }
                    }, REPEAT);
                    rows.add(new Row("java", "linprog", "lp_grid", "n=" + n + "_m=" + mLe, "yishape_hpc_highs", ms));
                }
            }
        }

        // Pure Java Simplex (skip large ones)
        setHpcMode(false);
        for (int mult : mMults) {
            for (int n : ns) {
                int mLe = n * mult;
                long coeffs = (long) n * mLe;
                if (coeffs > 400_000L) continue;
                LpProblem p = buildLp(n, mLe, 424242L + n * 31L + mult);
                IVector<Double> cj = Linalg.vector(p.c);
                IMatrix<Double> aUb = Linalg.matrix(p.aUb);
                IVector<Double> bUb = Linalg.vector(p.bUb);
                try {
                    double ms = benchMs(() -> {
                        OptResult r = new RereSimplexLinProgSolver().solve(cj, aUb, bUb, null, null);
                        if (!r.isConverged()) throw new IllegalStateException("not converged");
                    }, REPEAT);
                    rows.add(new Row("java", "linprog", "lp_grid", "n=" + n + "_m=" + mLe, "rere_simplex", ms));
                } catch (Exception e) {
                    rows.add(new Row("java", "linprog", "lp_grid", "n=" + n + "_m=" + mLe, "rere_simplex", -1));
                }
            }
        }
    }

    private static void benchLpSimplex(List<Row> rows) {
        setHpcMode(false);
        for (int n : new int[]{80, 200}) {
            double[] c = new double[n];
            double[][] aEq = new double[1][n];
            for (int j = 0; j < n; j++) {
                c[j] = (j * 17 + 41) % 100 / 100.0 + 0.01;
                aEq[0][j] = 1.0;
            }
            double[] bEq = {1.0};
            IVector<Double> cj = Linalg.vector(c);
            IMatrix<Double> aEqM = Linalg.matrix(aEq);
            IVector<Double> bEqV = Linalg.vector(bEq);
            double ms = benchMs(() -> {
                OptResult r = new RereSimplexLinProgSolver().solve(cj, null, null, aEqM, bEqV);
                if (!r.isConverged()) throw new IllegalStateException("not converged");
            }, REPEAT);
            rows.add(new Row("java", "linprog", "lp_nonnegative_simplex", "n=" + n, "rere_simplex", ms));
        }
    }

    // ========== 15. Stats ==========

    private static void benchStats(List<Row> rows, String backend) {
        for (int n : new int[]{10000, 100000, 1000000}) {
            NormalDistribution norm = Stats.norm();
            int runs = (n >= 1000000) ? 1 : REPEAT;

            final int nn = n;
            double msPdf = benchMs(() -> {
                for (int i = 0; i < nn; i++) norm.pdf(i * 0.001);
            }, runs);
            rows.add(new Row("java", "stats", "normal_pdf", String.valueOf(n), backend, msPdf));

            double msSample = benchMs(() -> norm.sample(nn), runs);
            rows.add(new Row("java", "stats", "normal_sample", String.valueOf(n), backend, msSample));
        }

        for (int n : new int[]{10000, 100000}) {
            GammaDistribution gamma = Stats.gamma(2, 1);
            int runs = (n >= 100000) ? 1 : REPEAT;
            final int nn = n;
            double ms = benchMs(() -> gamma.sample(nn), runs);
            rows.add(new Row("java", "stats", "gamma_sample", String.valueOf(n), backend, ms));
        }
    }

    // ========== 16. Signal ==========

    private static void benchSignal(List<Row> rows, String backend) {
        for (int n : new int[]{1024, 4096, 16384, 65536}) {
            Complex[] signal = new Complex[n];
            for (int i = 0; i < n; i++) {
                signal[i] = new Complex(Math.sin(2 * Math.PI * i / n), 0);
            }
            int runs = (n >= 65536) ? 1 : REPEAT;
            double ms = benchMs(() -> RereFFT.fft(signal), runs);
            rows.add(new Row("java", "signal", "fft", String.valueOf(n), backend, ms));
        }

        for (int n : new int[]{1024, 4096, 16384}) {
            double[] data = new double[n];
            for (int i = 0; i < n; i++) {
                data[i] = Math.sin(2 * Math.PI * i / n);
            }
            IVector<Double> signal = Linalg.vector(data);
            int runs = (n >= 16384) ? 1 : REPEAT;
            double ms = benchMs(() -> RereDCT.dct2(signal), runs);
            rows.add(new Row("java", "signal", "dct", String.valueOf(n), backend, ms));
        }
    }

    // ========== 17. Vector ops ==========

    private static void benchVector(List<Row> rows, String backend) {
        DoubleVectorComputer computer = new DoubleVectorComputer();
        java.util.Random rng = new java.util.Random(SEED);

        for (int n : new int[]{10000, 100000, 1000000, 10000000}) {
            double[] a = new double[n];
            double[] b = new double[n];
            for (int i = 0; i < n; i++) {
                a[i] = rng.nextDouble();
                b[i] = rng.nextDouble();
            }
            int runs = (n >= 1000000) ? 1 : REPEAT;

            double msAdd = benchMs(() -> computer.binaryOperate(a, b, BinaryOperation.ADD), runs);
            rows.add(new Row("java", "compute", "vector_add", String.valueOf(n), backend, msAdd));

            double msSum = benchMs(() -> computer.reduceOperate(a, ReduceOperation.SUM), runs);
            rows.add(new Row("java", "compute", "vector_sum", String.valueOf(n), backend, msSum));

            double msMean = benchMs(() -> computer.reduceOperate(a, ReduceOperation.MEAN), runs);
            rows.add(new Row("java", "compute", "vector_mean", String.valueOf(n), backend, msMean));
        }
    }

    // ========== Helper: matrix constructors ==========

    private static IMatrix<Double> spdMatrix(int n, long seed) {
        IMatrix<Double> r = Linalg.rand(n, n, seed);
        IMatrix<Double> a = r.mmul(r.transpose()).add(Linalg.eye(n).multiplyByScalar((double) n));
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double avg = (a.get(i, j) + a.get(j, i)) / 2.0;
                a.put(i, j, avg);
                a.put(j, i, avg);
            }
        }
        return a;
    }

    private static IMatrix<Double> wellConditionedMatrix(int n, long seed) {
        return spdMatrix(n, seed);
    }

    private static IMatrix<Double> symMatrix(int n, long seed) {
        IMatrix<Double> m = Linalg.rand(n, n, seed);
        return m.add(m.transpose()).multiplyByScalar(0.5);
    }

    private static IMatrix<Double> hilbertMatrix(int n) {
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                data[i][j] = 1.0 / (i + j + 1);
            }
        }
        return Linalg.matrix(data);
    }

    private static IVector<Double> randVector(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        double[] data = new double[n];
        for (int i = 0; i < n; i++) data[i] = rng.nextDouble();
        return Linalg.vector(data);
    }

    private static double[] fill(int n, double v) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = v;
        return a;
    }

    // ========== Optimizer problem definitions ==========

    private static IObjectiveFunction quadraticObj(IVector target) {
        return x -> {
            double sum = 0.0;
            for (int i = 0; i < x.size(); i++) {
                double d = x.get(i) - target.get(i);
                sum += d * d;
            }
            return sum;
        };
    }

    private static IGradientFunction quadraticGrad(IVector target) {
        return x -> {
            double[] g = new double[x.size()];
            for (int i = 0; i < x.size(); i++) {
                g[i] = 2.0 * (x.get(i) - target.get(i));
            }
            return Linalg.vector(g);
        };
    }

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

    private static final IObjectiveFunction ROSENBROCK_OBJ = x -> {
        double x0 = x.get(0), x1 = x.get(1);
        double a = 1.0 - x0, b = x1 - x0 * x0;
        return a * a + 100.0 * b * b;
    };

    private static final IGradientFunction ROSENBROCK_GRAD = x -> {
        double x0 = x.get(0), x1 = x.get(1);
        return Linalg.vector(new double[]{
            -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0),
            200.0 * (x1 - x0 * x0)
        });
    };

    // ========== LP helpers ==========

    private static LpProblem buildLp(int n, int mLe, long seed) {
        double[] c = new double[n];
        for (int j = 0; j < n; j++) c[j] = 1.0 + (j % 11) * 0.003;
        double[][] aUb = new double[mLe][n];
        double[] bUb = new double[mLe];
        for (int i = 0; i < n; i++) { aUb[i][i] = -1.0; bUb[i] = -1.0; }
        java.util.Random rng = new java.util.Random(seed);
        for (int i = n; i < mLe; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) { double v = rng.nextDouble(); aUb[i][j] = v; sum += v; }
            bUb[i] = sum * 25.0 + 2.0;
        }
        return new LpProblem(c, aUb, bUb);
    }

    record LpProblem(double[] c, double[][] aUb, double[] bUb) {}

    // ========== Benchmark infrastructure ==========

    private static double benchMs(Runnable op, int runs) {
        for (int i = 0; i < WARMUP; i++) op.run();
        long[] ns = new long[runs];
        for (int i = 0; i < runs; i++) {
            long t0 = System.nanoTime();
            op.run();
            ns[i] = System.nanoTime() - t0;
        }
        java.util.Arrays.sort(ns);
        return ns[ns.length / 2] / 1_000_000.0;
    }

    record Row(String suite, String module, String operation, String size, String backend, double ms) {}

    private static void writeCsv(List<Row> rows) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("suite,module,operation,size,backend,ms\n");
        for (Row r : rows) {
            sb.append(r.suite).append(',').append(r.module).append(',').append(r.operation).append(',')
              .append(r.size).append(',').append(r.backend).append(',')
              .append(String.format(Locale.ROOT, "%.6f", r.ms)).append('\n');
        }
        Files.writeString(OUT, sb.toString(), StandardCharsets.UTF_8,
            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
}
