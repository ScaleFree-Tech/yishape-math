package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.highs.HighsLinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 线性规划 / 整数规划系统化与大规模回归测试。
 * <p>
 * 默认档位：中等规模、可复现随机实例，与 HiGHS（或回退路径）对照；0-1 背包与小规模整数规划与暴力最优对照。<br>
 * 压力档位：在 JVM 上增加变量与约束规模，需显式开启：
 * {@code mvn test -Dtest=LinProgIntegerProgLargeScaleSystematicTest -Dyishape.linpg.stress=true}
 * </p>
 */
@DisplayName("LP / MILP 系统化大规模测试")
@Disabled("LP/MILP 大规模系统化测试，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
class LinProgIntegerProgLargeScaleSystematicTest {

    private static final double LP_OBJ_TOL = 1e-4;
    private static final double FEAS_TOL = 1e-5;
    private static final double IP_OBJ_TOL = 2e-3;

    /** 供 {@link EnabledIf} 反射调用；必须为 public。 */
    @SuppressWarnings("unused")
    public static boolean stressEnabled() {
        return Boolean.getBoolean("yishape.linpg.stress");
    }

    static void assertFeasibleLeq(IMatrix aUb, IVector bUb, IVector x) {
        assertNotNull(aUb);
        assertNotNull(bUb);
        assertNotNull(x);
        IVector ax = aUb.mmul(x);
        for (int i = 0; i < ax.length(); i++) {
            final int row = i;
            double lhs = ax.get(i);
            double rhs = bUb.get(i);
            assertTrue(lhs <= rhs + 1e-7 + FEAS_TOL * (1 + Math.abs(rhs)),
                    () -> String.format("行 %d: A_ub x = %.8g > b %.8g", row, lhs, rhs));
        }
    }

    static void assertFeasibleEq(IMatrix aEq, IVector bEq, IVector x) {
        assertNotNull(aEq);
        assertNotNull(bEq);
        IVector ax = aEq.mmul(x);
        for (int i = 0; i < ax.length(); i++) {
            final int row = i;
            double lhs = ax.get(i);
            double rhs = bEq.get(i);
            assertEquals(rhs, lhs, FEAS_TOL * (1 + Math.abs(rhs)) + 1e-7,
                    () -> "等式行 " + row);
        }
    }

    static void assertNonNegative(IVector x) {
        for (int i = 0; i < x.length(); i++) {
            final int idx = i;
            assertTrue(x.get(i) >= -1e-7, () -> "x[" + idx + "] < 0");
        }
    }

    /**
     * 构造 minimize c'x s.t. A_ub x &lt;= b_ub, x &gt;= 0 的严格可行内点 x0（用于定 b）。
     */
    static class LeqInstance {
        final IVector c;
        final IMatrix aUb;
        final IVector bUb;

        LeqInstance(IVector c, IMatrix aUb, IVector bUb) {
            this.c = c;
            this.aUb = aUb;
            this.bUb = bUb;
        }
    }

    static LeqInstance randomFeasibleLeq(int n, int m, long seed) {
        Random rnd = new Random(seed);
        double[] x0 = new double[n];
        for (int j = 0; j < n; j++) {
            x0[j] = 0.05 + rnd.nextDouble() * 2.0;
        }
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = rnd.nextDouble() * 1.5;
            }
        }
        double[] b = new double[m];
        for (int i = 0; i < m; i++) {
            double s = 0;
            for (int j = 0; j < n; j++) {
                s += a[i][j] * x0[j];
            }
            b[i] = s + 0.1 + rnd.nextDouble() * 1.5;
        }
        double[] c = new double[n];
        for (int j = 0; j < n; j++) {
            c[j] = rnd.nextGaussian() * 0.8;
        }
        return new LeqInstance(Linalg.vector(c), Linalg.matrix(a), Linalg.vector(b));
    }

    static double bruteMax01Knapsack(double[] w, double[] p, double cap) {
        int n = w.length;
        double best = Double.NEGATIVE_INFINITY;
        int total = 1 << n;
        for (int mask = 0; mask < total; mask++) {
            double ww = 0;
            double pp = 0;
            for (int i = 0; i < n; i++) {
                if (((mask >> i) & 1) != 0) {
                    ww += w[i];
                    pp += p[i];
                }
            }
            if (ww <= cap + 1e-9) {
                best = Math.max(best, pp);
            }
        }
        return best;
    }

    @Nested
    @DisplayName("LP：随机可行（<=）与 HiGHS 对照")
    class LpRandomFeasibleVsHighs {

        @Test
        @DisplayName("标准：30 个实例 n=12,m=18")
        void standardManyInstances() {
            var simp = new RereSimplexLinProgSolver();
            var hi = new HighsLinProgSolver();
            for (int k = 0; k < 30; k++) {
                LeqInstance P = randomFeasibleLeq(12, 18, 10_000L + k);
                IVector init = Linalg.ones(12);
                OptResult rs = simp.solve(P.c, P.aUb, P.bUb, null, null, init);
                OptResult rh = hi.solve(P.c, P.aUb, P.bUb, null, null, init);
                assertNotNull(rs, "simplex k=" + k);
                assertNotNull(rh, "highs k=" + k);
                assertTrue(rs.isConverged(), "simplex converged k=" + k);
                assertTrue(rh.isConverged(), "highs converged k=" + k);
                assertFeasibleLeq(P.aUb, P.bUb, rs.getOptimalPoint());
                assertNonNegative(rs.getOptimalPoint());
                assertEquals(rh.getOptimalValue(), rs.getOptimalValue(),
                        LP_OBJ_TOL * (1 + Math.abs(rh.getOptimalValue())),
                        "objective k=" + k);
            }
        }

        @Test
        @DisplayName("中等：20 个实例 n=28,m=36")
        void mediumInstances() {
            var simp = new RereSimplexLinProgSolver();
            var hi = new HighsLinProgSolver();
            for (int k = 0; k < 20; k++) {
                LeqInstance P = randomFeasibleLeq(28, 36, 20_000L + k);
                IVector init = Linalg.ones(28);
                OptResult rs = simp.solve(P.c, P.aUb, P.bUb, null, null, init);
                OptResult rh = hi.solve(P.c, P.aUb, P.bUb, null, null, init);
                assertTrue(rs.isConverged());
                assertTrue(rh.isConverged());
                assertFeasibleLeq(P.aUb, P.bUb, rs.getOptimalPoint());
                assertEquals(rh.getOptimalValue(), rs.getOptimalValue(),
                        LP_OBJ_TOL * (1 + Math.abs(rh.getOptimalValue())));
            }
        }
    }

    @Nested
    @DisplayName("LP：等式 + 非负 批量")
    class LpEqualityBatch {

        @Test
        @DisplayName("多组等式 min c'x（与 HiGHS 对照）")
        void manyEqualityProblems() {
            var simp = new RereSimplexLinProgSolver();
            var hi = new HighsLinProgSolver();
            long base = 777L;
            for (int k = 0; k < 25; k++) {
                Random rnd = new Random(base + k);
                int n = 10;
                int m = 5;
                double[][] a = new double[m][n];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        a[i][j] = (int) (rnd.nextDouble() * 6) - 1;
                    }
                }
                double[] xfeas = new double[n];
                for (int j = 0; j < n; j++) {
                    xfeas[j] = rnd.nextDouble() * 1.5;
                }
                double[] b = new double[m];
                for (int i = 0; i < m; i++) {
                    double s = 0;
                    for (int j = 0; j < n; j++) {
                        s += a[i][j] * xfeas[j];
                    }
                    b[i] = s;
                }
                double[] c = new double[n];
                for (int j = 0; j < n; j++) {
                    c[j] = rnd.nextGaussian();
                }
                IVector cV = Linalg.vector(c);
                IMatrix aEq = Linalg.matrix(a);
                IVector bV = Linalg.vector(b);
                IVector init = Linalg.vector(xfeas);
                OptResult rs = simp.solve(cV, null, null, aEq, bV, init);
                OptResult rh = hi.solve(cV, null, null, aEq, bV, init);
                assertTrue(rs.isConverged(), "k=" + k);
                assertTrue(rh.isConverged(), "k=" + k);
                assertFeasibleEq(aEq, bV, rs.getOptimalPoint());
                assertNonNegative(rs.getOptimalPoint());
                assertEquals(rh.getOptimalValue(), rs.getOptimalValue(),
                        LP_OBJ_TOL * (1 + Math.abs(rh.getOptimalValue())), "k=" + k);
            }
        }
    }

    @Nested
    @DisplayName("IP：0-1 背包 vs 暴力（标准规模）")
    class IpKnapsackBrute {

        @Test
        @DisplayName("40 个随机背包 n=14")
        void randomKnapsackMany() {
            Random rnd = new Random(42);
            for (int t = 0; t < 40; t++) {
                int n = 14;
                double[] w = new double[n];
                double[] p = new double[n];
                for (int i = 0; i < n; i++) {
                    w[i] = 1 + rnd.nextInt(8);
                    p[i] = 1 + rnd.nextInt(20);
                }
                double cap = 5 + rnd.nextInt(25);
                double brute = bruteMax01Knapsack(w, p, cap);

                IVector c = Linalg.zeros(n);
                for (int i = 0; i < n; i++) {
                    c.set(i, -p[i]);
                }
                double[][] row = new double[1][n];
                System.arraycopy(w, 0, row[0], 0, n);
                IMatrix aUb = Linalg.matrix(row);
                IVector bUb = Linalg.vector(cap);

                RereIntegerProg mip = new RereIntegerProg();
                mip.setMaxIterations(200_000);
                mip.setGapTolerance(1e-10);
                mip.addBinaryVariables(java.util.stream.IntStream.range(0, n).toArray());
                OptResult r = mip.solve(c, aUb, bUb, null, null);
                assertTrue(r.isConverged(), "t=" + t);
                IVector x = r.getOptimalPoint();
                assertEquals(n, x.length());
                double wx = 0;
                for (int i = 0; i < n; i++) {
                    double xi = x.get(i);
                    assertTrue(xi >= -1e-6 && xi <= 1 + 1e-6);
                    wx += w[i] * xi;
                }
                assertTrue(wx <= cap + 1e-5);
                assertEquals(-brute, r.getOptimalValue(), IP_OBJ_TOL * (1 + Math.abs(brute)) + 1e-6, "t=" + t);
            }
        }

        @Test
        @DisplayName("纯整数等式多组（与已知最优值区间一致）")
        void integerEqualityDiophantineBatch() {
            Random rnd = new Random(99);
            for (int k = 0; k < 15; k++) {
                int sumrhs = 4 + rnd.nextInt(8);
                int n = 3 + rnd.nextInt(4);
                double[] c = new double[n];
                for (int j = 0; j < n; j++) {
                    c[j] = rnd.nextInt(5) + 1;
                }
                double[] arow = new double[n];
                for (int j = 0; j < n; j++) {
                    arow[j] = 1;
                }
                IVector cV = Linalg.vector(c);
                IMatrix aEq = Linalg.matrix(new double[][] { arow });
                IVector bEq = Linalg.vector((double) sumrhs);

                RereIntegerProg mip = new RereIntegerProg();
                mip.addIntegerVariables(java.util.stream.IntStream.range(0, n).toArray());
                OptResult r = mip.solveWithNonNegativeEqualConstraints(cV, aEq, bEq);
                assertTrue(r.isConverged(), "k=" + k);
                IVector x = r.getOptimalPoint();
                int s = 0;
                double obj = 0;
                for (int j = 0; j < n; j++) {
                    double v = x.get(j);
                    assertEquals(Math.rint(v), v, 1e-5);
                    assertTrue(v >= -1e-6);
                    s += (int) Math.round(v);
                    obj += c[j] * v;
                }
                assertEquals(sumrhs, s, "k=" + k);
                assertEquals(obj, r.getOptimalValue(), 1e-5);
            }
        }
    }

    @Nested
    @DisplayName("压力：大规模 LP / MILP（-Dyishape.linpg.stress=true）")
    @EnabledIf("com.yishape.lab.math.optimize.linpg.LinProgIntegerProgLargeScaleSystematicTest#stressEnabled")
    class StressSuite {

        @Test
        @DisplayName("LP n=85,m=110 随机可行 12 例 vs HiGHS")
        void lpLargeRandom() {
            var simp = new RereSimplexLinProgSolver();
            var hi = new HighsLinProgSolver();
            for (int k = 0; k < 12; k++) {
                LeqInstance P = randomFeasibleLeq(85, 110, 100_000L + k);
                IVector init = Linalg.ones(85);
                OptResult rs = simp.solve(P.c, P.aUb, P.bUb, null, null, init);
                OptResult rh = hi.solve(P.c, P.aUb, P.bUb, null, null, init);
                assertTrue(rs.isConverged(), "k=" + k);
                assertTrue(rh.isConverged(), "k=" + k);
                assertFeasibleLeq(P.aUb, P.bUb, rs.getOptimalPoint());
                assertEquals(rh.getOptimalValue(), rs.getOptimalValue(),
                        2e-3 * (1 + Math.abs(rh.getOptimalValue())), "k=" + k);
            }
        }

        @Test
        @DisplayName("0-1 背包 n=20 暴力对照 25 例")
        void knapsackLargerBrute() {
            Random rnd = new Random(12345);
            for (int t = 0; t < 25; t++) {
                int n = 20;
                double[] w = new double[n];
                double[] p = new double[n];
                for (int i = 0; i < n; i++) {
                    w[i] = 1 + rnd.nextInt(10);
                    p[i] = 1 + rnd.nextInt(25);
                }
                double cap = 8 + rnd.nextInt(40);
                double brute = bruteMax01Knapsack(w, p, cap);

                IVector c = Linalg.zeros(n);
                for (int i = 0; i < n; i++) {
                    c.set(i, -p[i]);
                }
                double[][] row = new double[1][n];
                System.arraycopy(w, 0, row[0], 0, n);
                IMatrix aUb = Linalg.matrix(row);
                IVector bUb = Linalg.vector(cap);

                RereIntegerProg mip = new RereIntegerProg();
                mip.setMaxIterations(200_000);
                mip.setGapTolerance(1e-10);
                mip.addBinaryVariables(java.util.stream.IntStream.range(0, n).toArray());
                OptResult r = mip.solve(c, aUb, bUb, null, null);
                assertTrue(r.isConverged(), "t=" + t);
                assertEquals(-brute, r.getOptimalValue(), 0.02 * (1 + Math.abs(brute)), "t=" + t);
            }
        }

        @Test
        @DisplayName("等式+整数 n=8 变量 多约束 20 例（可行性+收敛）")
        void stressIntegerDense() {
            Random rnd = new Random(321);
            for (int k = 0; k < 20; k++) {
                int n = 8;
                int m = 5;
                double[][] a = new double[m][n];
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) {
                        a[i][j] = rnd.nextInt(4);
                    }
                }
                double[] x0 = new double[n];
                for (int j = 0; j < n; j++) {
                    x0[j] = rnd.nextInt(3);
                }
                double[] b = new double[m];
                for (int i = 0; i < m; i++) {
                    double s = 0;
                    for (int j = 0; j < n; j++) {
                        s += a[i][j] * x0[j];
                    }
                    b[i] = s;
                }
                double[] c = new double[n];
                for (int j = 0; j < n; j++) {
                    c[j] = rnd.nextInt(4) + 1;
                }
                IVector cV = Linalg.vector(c);
                IMatrix aEq = Linalg.matrix(a);
                IVector bV = Linalg.vector(b);

                RereIntegerProg mip = new RereIntegerProg();
                mip.addIntegerVariables(0, 1, 2, 3, 4, 5, 6, 7);
                OptResult r = mip.solveWithNonNegativeEqualConstraints(cV, aEq, bV);
                assertTrue(r.isConverged(), "k=" + k);
                assertFeasibleEq(aEq, bV, r.getOptimalPoint());
                for (int j = 0; j < n; j++) {
                    double v = r.getOptimalPoint().get(j);
                    assertEquals(Math.rint(v), v, 5e-4, "k=" + k + " j=" + j);
                }
            }
        }
    }
}
