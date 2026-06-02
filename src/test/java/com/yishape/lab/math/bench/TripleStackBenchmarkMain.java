package com.yishape.lab.math.bench;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;

import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.hpc.SymmetricEigenResult;
import com.yishape.lab.math.hpc.YishapeHpc;
import com.yishape.lab.math.hpc.YishapeHpcStatus;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;

/**
 * 矩阵乘、稠密求解、分解、LP 等：在 JVM 内对比「关闭 hpc」与「开启 hpc（原生可用时）」。
 * 结果写入 {@code benchmarks/out/java_perf.csv}（自项目根运行 Maven）。
 */
public final class TripleStackBenchmarkMain {

    private static final int WARMUP = 2;
    private static final int REPEAT = 5;
    private static final Path OUT = Path.of("benchmarks/out/java_perf.csv");

    private TripleStackBenchmarkMain() {
    }

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUT.getParent());
        List<Row> rows = new ArrayList<>();
        boolean nativeOk = YishapeHpc.isNativeRuntimeAvailable();
        System.out.println("META_JSON {\"java\":\"" + Runtime.version() + "\",\"hpcNative\":"
                + nativeOk + ",\"hpcAbi\":" + YishapeHpc.abiVersion() + "}");

        for (boolean hpc : new boolean[] { false, true }) {
            setHpcMode(hpc);
            String backend = hpc ? "java_hpc" : "java_pure";
            for (int n : new int[] { 256, 512, 1024, 1536 }) {
                IMatrix<Double> a = Linalg.rand(n, n, 42L + n);
                IMatrix<Double> b = Linalg.rand(n, n, 43L + n);
                double med = benchMs(() -> a.mmul(b));
                rows.add(new Row("java", "gemm", n + "x" + n + "x" + n, backend, med));
            }

            for (int n : new int[] { 256, 512, 1024 }) {
                IMatrix<Double> spd = spdMatrix(n, 44L);
                double[] bd = new double[n];
                for (int i = 0; i < n; i++) {
                    bd[i] = Math.sin(i * 0.13 + n);
                }
                IVector<Double> rhs = Linalg.vector(bd);
                double medJv = benchMs(() -> spd.solve(rhs));
                rows.add(new Row("java", "solve_dense", "n=" + n, backend, medJv));

                if (hpc && nativeOk) {
                    double[][] ad = spd.toDoubleArray();
                    double[] br = rhs.toDoubleArray();
                    double medN = benchMs(() -> {
                        var r = YishapeHpc.solveSquare(ad, br);
                        if (!r.ok()) {
                            throw new IllegalStateException("solveSquare rc=" + r.status());
                        }
                    });
                    rows.add(new Row("java", "solve_dense", "n=" + n, "yishape_hpc_ffi", medN));
                }
            }

            for (int n : new int[] { 256, 512, 1024 }) {
                IMatrix<Double> spd = spdMatrix(n, 45L);
                double med = benchMs(() -> spd.cholesky());
                rows.add(new Row("java", "cholesky", "n=" + n, backend, med));
            }

            for (int n : new int[] { 192, 256, 384 }) {
                IMatrix<Double> m = Linalg.rand(n, n, 46L + n);
                double med = benchMs(() -> m.svd());
                rows.add(new Row("java", "svd", n + "x" + n, backend, med));
            }

            for (int n : new int[] { 256, 384, 512 }) {
                IMatrix<Double> sym = symMatrix(n, 47L);
                double medJ = benchMs(() -> sym.eigen());
                rows.add(new Row("java", "eigh", "n=" + n, backend, medJ));

                if (hpc && nativeOk) {
                    double[][] sd = sym.toDoubleArray();
                    double medN = benchMs(() -> {
                        SymmetricEigenResult r = YishapeHpc.eigenSymmetric(sd);
                        if (r.status() != YishapeHpcStatus.OK) {
                            throw new IllegalStateException("eigenSymmetric rc=" + r.status());
                        }
                    });
                    rows.add(new Row("java", "eigh", "n=" + n, "yishape_hpc_ffi", medN));
                }
            }
        }

        setHpcMode(true);
        if (nativeOk) {
            for (int n : new int[] { 80, 200 }) {
                double[] c = new double[n];
                double[][] aEq = new double[1][n];
                for (int j = 0; j < n; j++) {
                    c[j] = (j * 17 + 41) % 100 / 100.0 + 0.01;
                    aEq[0][j] = 1.0;
                }
                double[] bEq = new double[] { 1.0 };
                double med = benchMs(() -> {
                    var lp = YishapeHpc.lpNonnegative(c, null, null, aEq, bEq);
                    if (lp.status() != YishapeHpcStatus.OK) {
                        throw new IllegalStateException("lp rc=" + lp.status());
                    }
                });
                rows.add(new Row("java", "lp_nonneg_eq", "n=" + n, "yishape_hpc_highs", med));
            }
        }

        setHpcMode(false);
        for (int n : new int[] { 80, 200 }) {
            IVector<Double> c = Linalg.vector(rowC(n));
            IMatrix<Double> aEq = Linalg.matrix(rowAEq(n));
            IVector<Double> bEq = Linalg.vector(new double[] { 1.0 });
            double med = benchMs(() -> new RereSimplexLinProgSolver().solve(c, null, null, aEq, bEq));
            rows.add(new Row("java", "lp_nonneg_eq", "n=" + n, "rere_simplex", med));
        }

        writeCsv(rows);
        System.out.println("WROTE " + OUT.toAbsolutePath());
    }

    private static double[] rowC(int n) {
        double[] c = new double[n];
        for (int j = 0; j < n; j++) {
            c[j] = (j * 17 + 41) % 100 / 100.0 + 0.01;
        }
        return c;
    }

    private static double[][] rowAEq(int n) {
        double[][] a = new double[1][n];
        for (int j = 0; j < n; j++) {
            a[0][j] = 1.0;
        }
        return a;
    }

    private static void setHpcMode(boolean enable) {
        System.setProperty(HpcConfig.PROP_ENABLE, Boolean.toString(enable));
        if (enable) {
            System.setProperty(HpcConfig.PROP_GEMM_MIN_FLOPS, "0");
            System.setProperty(HpcConfig.PROP_CHOLESKY_MIN_DIM, "32");
            System.setProperty(HpcConfig.PROP_SVD_MIN_TOTAL_ELEMENTS, "1");
        }
    }

    private static IMatrix<Double> spdMatrix(int n, long seed) {
        IMatrix<Double> m = Linalg.rand(n, n, seed);
        IMatrix<Double> mt = m.transpose();
        return m.mmul(mt).add(Linalg.eye(n).multiplyByScalar((double) n));
    }

    private static IMatrix<Double> symMatrix(int n, long seed) {
        IMatrix<Double> m = Linalg.rand(n, n, seed);
        return m.add(m.transpose()).multiplyByScalar(0.5);
    }

    private static double benchMs(Runnable op) {
        for (int i = 0; i < WARMUP; i++) {
            op.run();
        }
        long[] ns = new long[REPEAT];
        for (int i = 0; i < REPEAT; i++) {
            long t0 = System.nanoTime();
            op.run();
            ns[i] = System.nanoTime() - t0;
        }
        java.util.Arrays.sort(ns);
        return ns[ns.length / 2] / 1_000_000.0;
    }

    private record Row(String suite, String task, String size, String backend, double ms) {
    }

    private static void writeCsv(List<Row> rows) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("suite,task,size,backend,ms\n");
        for (Row r : rows) {
            sb.append(r.suite).append(',').append(r.task).append(',').append(r.size).append(',')
                    .append(r.backend).append(',').append(String.format(java.util.Locale.ROOT, "%.6f", r.ms))
                    .append('\n');
        }
        Files.writeString(OUT, sb.toString(), StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
    }
}
