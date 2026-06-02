package com.yishape.lab.math.bench;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import com.yishape.lab.math.compute.hpc.HpcConfig;
import com.yishape.lab.math.hpc.YishapeHpc;
import com.yishape.lab.math.hpc.YishapeHpcStatus;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;

/**
 * 大规模稠密 LP 网格：最小化 {@code c'x}，约束 {@code A_ub x <= b_ub}、{@code x >= 0}；
 * 前 {@code n} 行为 {@code x_j >= 1}（即 {@code -x_j <= -1}），其余行为随机非负系数与充足右端项以保证可行。
 * 输出 {@code benchmarks/out/lp_grid_java.csv}。
 */
public final class LargeLpGridBenchmarkMain {

    private static final int WARMUP = 1;
    private static final int REPEAT = 3;
    private static final Path OUT = Path.of("benchmarks/out/lp_grid_java.csv");
    /**
     * 超过则<strong>基准不再调用</strong>单纯形（避免过久）；CSV 记为 {@code skip_size}，不是求解器抛错。
     */
    private static final long SIMPLEX_MAX_COEFFS = 400_000L;

    private LargeLpGridBenchmarkMain() {
    }

    public static void main(String[] args) throws Exception {
        Files.createDirectories(OUT.getParent());
        System.setProperty(HpcConfig.PROP_ENABLE, "true");
        boolean nativeOk = YishapeHpc.isNativeRuntimeAvailable();
        System.out.println("META_JSON {\"hpcNative\":" + nativeOk + "}");

        List<String> lines = new ArrayList<>();
        lines.add("suite,task,n,m_le,backend,ms,ok_or_rc");

        int[] ns = { 250, 500, 1000, 2000, 3500 };
        int[] mMults = { 1, 2 }; // m_le = n * mult（mult=1 仅下界行；2 为下界 + 额外 n 条随机 <=）

        for (int mult : mMults) {
            for (int n : ns) {
                int mLe = n * mult;
                DenseLpProblem p = buildFeasibleLeOnly(n, mLe, 424242L + n * 31L + mult);

                if (nativeOk) {
                    double t = benchMs(() -> {
                        var r = YishapeHpc.lpNonnegative(p.c, p.aUb, p.bUb, null, null);
                        if (r.status() != YishapeHpcStatus.OK) {
                            throw new IllegalStateException("highs " + r.status());
                        }
                    });
                    lines.add(String.format(Locale.ROOT, "java,lp_grid_dense,%d,%d,yishape_hpc_highs,%.6f,1",
                            n, mLe, t));
                }

                long coeffs = (long) n * mLe;
                if (coeffs <= SIMPLEX_MAX_COEFFS) {
                    IVector<Double> cj = Linalg.vector(p.c);
                    IMatrix<Double> aUb = Linalg.matrix(p.aUb);
                    IVector<Double> bUb = Linalg.vector(p.bUb);
                    try {
                        double t = benchMs(() -> {
                            OptResult r = new RereSimplexLinProgSolver().solve(cj, aUb, bUb, null, null);
                            if (!r.isConverged()) {
                                throw new IllegalStateException(
                                        "RereSimplex not converged: " + r.getConvergenceReason());
                            }
                        });
                        lines.add(String.format(Locale.ROOT, "java,lp_grid_dense,%d,%d,rere_simplex,%.6f,1",
                                n, mLe, t));
                    } catch (Throwable e) {
                        lines.add(String.format(Locale.ROOT, "java,lp_grid_dense,%d,%d,rere_simplex,,0:%s",
                                n, mLe, escape(e.getMessage())));
                    }
                } else {
                    lines.add(String.format(Locale.ROOT, "java,lp_grid_dense,%d,%d,rere_simplex,,skip_size",
                            n, mLe));
                }
            }
        }

        Files.write(OUT, lines, StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        System.out.println("WROTE " + OUT.toAbsolutePath());
    }

    private static String escape(String m) {
        if (m == null) {
            return "";
        }
        return m.replace(',', ';').replace('\n', ' ');
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

    static final class DenseLpProblem {
        final double[] c;
        final double[][] aUb;
        final double[] bUb;

        DenseLpProblem(double[] c, double[][] aUb, double[] bUb) {
            this.c = c;
            this.aUb = aUb;
            this.bUb = bUb;
        }
    }

    static DenseLpProblem buildFeasibleLeOnly(int n, int mLe, long seed) {
        if (mLe < n) {
            throw new IllegalArgumentException("need mLe >= n for lower-bound rows");
        }
        double[] c = new double[n];
        for (int j = 0; j < n; j++) {
            c[j] = 1.0 + (j % 11) * 0.003;
        }
        double[][] aUb = new double[mLe][n];
        double[] bUb = new double[mLe];
        for (int i = 0; i < n; i++) {
            aUb[i][i] = -1.0;
            bUb[i] = -1.0;
        }
        java.util.Random rng = new java.util.Random(seed);
        for (int i = n; i < mLe; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                double v = rng.nextDouble();
                aUb[i][j] = v;
                sum += v;
            }
            bUb[i] = sum * 25.0 + 2.0;
        }
        return new DenseLpProblem(c, aUb, bUb);
    }
}
