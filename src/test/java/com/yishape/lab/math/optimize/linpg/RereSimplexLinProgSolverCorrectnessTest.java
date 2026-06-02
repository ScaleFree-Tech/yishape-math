package com.yishape.lab.math.optimize.linpg;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.optimize.OptResult;
import com.yishape.lab.math.optimize.linpg.highs.HighsLinProgSolver;
import com.yishape.lab.math.optimize.linpg.simplex.RereSimplexLinProgSolver;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.MethodOrderer;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * RereSimplexLinProgSolver 大规模正确性与性能测试
 */
@TestMethodOrder(MethodOrderer.DisplayName.class)
@DisplayName("RereSimplexLinProgSolver 大规模测试")
public class RereSimplexLinProgSolverCorrectnessTest {

    private static final double TOLERANCE = 1e-4;
    private static final double LOOSE_TOLERANCE = 1e-3;

    private RereSimplexLinProgSolver javaSolver;
    private HighsLinProgSolver highsSolver;

    @BeforeEach
    void setUp() {
        javaSolver = new RereSimplexLinProgSolver();
        highsSolver = new HighsLinProgSolver();
    }

    /**
     * 验证解满足非负约束和等式约束
     */
    private boolean verifySolution(IVector sol, IMatrix A_eq, IVector b_eq, double tol) {
        for (int i = 0; i < sol.length(); i++) {
            if (sol.get(i) < -tol) return false;
        }
        for (int i = 0; i < A_eq.rows(); i++) {
            double lhs = 0;
            for (int j = 0; j < sol.length(); j++) {
                lhs += A_eq.get(i, j) * sol.get(j);
            }
            if (Math.abs(lhs - b_eq.get(i)) > tol) return false;
        }
        return true;
    }

    // ==================== 大规模测试 ====================

    @Test
    @DisplayName("大规模测试: 50变量")
    void testLargeScale50() {
        runLargeScaleTest(50, 20, 42, 1.0);
    }

    @Test
    @DisplayName("大规模测试: 100变量")
    void testLargeScale100() {
        runLargeScaleTest(100, 40, 123, 1.0);
    }

    @Test
    @DisplayName("大规模测试: 200变量")
    void testLargeScale200() {
        runLargeScaleTest(200, 80, 456, 1.0);
    }

    @Test
    @DisplayName("大规模测试: 500变量")
    void testLargeScale500() {
        runLargeScaleTest(500, 150, 789, 2.0);
    }

    @Test
    @DisplayName("大规模测试: 1000变量")
    void testLargeScale1000() {
        runLargeScaleTest(1000, 300, 999, 5.0);
    }

    @Test
    @DisplayName("随机批次测试: 20次")
    void testRandomBatch20() {
        int n = 200;
        int m = 80;
        int numTests = 20;
        Random rand = new Random(42);

        int successCount = 0;
        int constraintFailCount = 0;
        double maxRelDiff = 0;

        System.out.println("\n========== 随机批次测试(20次) ==========");
        System.out.println("配置: n=" + n + ", m=" + m);

        for (int t = 0; t < numTests; t++) {
            double[] cArr = new double[n + m];
            for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
            for (int i = n; i < n + m; i++) cArr[i] = 0;
            IVector c = Linalg.vector(cArr);

            double[][] aArr = new double[m][n + m];
            double[] bArr = new double[m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    aArr[i][j] = rand.nextDouble() * 5 + 0.1;
                }
                for (int j = 0; j < m; j++) {
                    aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
                }
                bArr[i] = rand.nextDouble() * 50 + 10;
            }
            IMatrix A_eq = Linalg.matrix(aArr);
            IVector b_eq = Linalg.vector(bArr);

            OptResult javaResult = javaSolver.solve(c, null, null, A_eq, b_eq);
            OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);

            boolean javaOk = javaResult.isConverged();
            boolean highsOk = highsResult.isConverged();

            if (javaOk && highsOk) {
                IVector javaSol = javaResult.getOptimalPoint();
                boolean constraintsOk = verifySolution(javaSol, A_eq, b_eq, LOOSE_TOLERANCE);

                double javaOpt = javaResult.getOptimalValue();
                double highsOpt = highsResult.getOptimalValue();
                double relDiff = Math.abs(highsOpt - javaOpt) / Math.abs(highsOpt);
                maxRelDiff = Math.max(maxRelDiff, relDiff);

                if (constraintsOk) {
                    successCount++;
                    if (t < 5 || t >= numTests - 3) {
                        System.out.printf("  测试%2d: OK - Java=%12.4f, HiGHS=%12.4f, relDiff=%.2e%n",
                                t + 1, javaOpt, highsOpt, relDiff);
                    } else if (t == 5) {
                        System.out.println("  ... ...");
                    }
                } else {
                    constraintFailCount++;
                    System.out.println("  测试" + (t+1) + ": FAIL - 约束不满足, relDiff=" + relDiff);
                }
            } else {
                System.out.println("  测试" + (t+1) + ": " +
                        (javaOk ? "OK" : "Java未收敛") + ", " +
                        (highsOk ? "OK" : "HiGHS未收敛"));
            }
        }

        System.out.println("\n========== 随机批次结果 ==========");
        System.out.println("成功: " + successCount + "/" + numTests);
        System.out.println("约束失败: " + constraintFailCount);
        System.out.println("最大相对差异: " + maxRelDiff);

        assertTrue(successCount >= numTests * 0.95,
                "至少95%测试应通过: " + successCount + "/" + numTests);
    }

    @Test
    @DisplayName("大规模性能测试: 200变量 5次运行")
    void testPerformance200x5() {
        int n = 200;
        int m = 80;
        int runs = 5;

        System.out.println("\n========== 性能测试: 200变量 x 5次 ==========");

        for (int r = 0; r < runs; r++) {
            Random rand = new Random(1000 + r);

            double[] cArr = new double[n + m];
            for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
            for (int i = n; i < n + m; i++) cArr[i] = 0;
            IVector c = Linalg.vector(cArr);

            double[][] aArr = new double[m][n + m];
            double[] bArr = new double[m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    aArr[i][j] = rand.nextDouble() * 5 + 0.1;
                }
                for (int j = 0; j < m; j++) {
                    aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
                }
                bArr[i] = rand.nextDouble() * 50 + 10;
            }
            IMatrix A_eq = Linalg.matrix(aArr);
            IVector b_eq = Linalg.vector(bArr);

            long start = System.nanoTime();
            OptResult javaResult = javaSolver.solve(c, null, null, A_eq, b_eq);
            long javaTime = System.nanoTime() - start;

            start = System.nanoTime();
            OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);
            long highsTime = System.nanoTime() - start;

            if (javaResult.isConverged() && highsResult.isConverged()) {
                double javaOpt = javaResult.getOptimalValue();
                double highsOpt = highsResult.getOptimalValue();
                double relDiff = Math.abs(highsOpt - javaOpt) / Math.abs(highsOpt);

                System.out.printf("  运行%d: Java=%12.4f (%6.2fms), HiGHS=%12.4f (%6.2fms), relDiff=%.2e%n",
                        r + 1, javaOpt, javaTime/1e6, highsOpt, highsTime/1e6, relDiff);
            } else {
                System.out.println("  运行" + (r+1) + ": 收敛失败");
            }
        }
    }

    private void runLargeScaleTest(int n, int m, int seed, double timeLimitSeconds) {
        Random rand = new Random(seed);

        double[] cArr = new double[n + m];
        for (int i = 0; i < n; i++) cArr[i] = -(rand.nextDouble() * 10 + 1);
        for (int i = n; i < n + m; i++) cArr[i] = 0;
        IVector c = Linalg.vector(cArr);

        double[][] aArr = new double[m][n + m];
        double[] bArr = new double[m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                aArr[i][j] = rand.nextDouble() * 5 + 0.1;
            }
            for (int j = 0; j < m; j++) {
                aArr[i][n + j] = (i == j) ? 1.0 : 0.0;
            }
            bArr[i] = rand.nextDouble() * 50 + 10;
        }
        IMatrix A_eq = Linalg.matrix(aArr);
        IVector b_eq = Linalg.vector(bArr);

        System.out.println("\n========== 大规模测试: " + n + "变量 ==========");

        // Java求解
        long start = System.nanoTime();
        OptResult javaResult = javaSolver.solve(c, null, null, A_eq, b_eq);
        long javaTime = System.nanoTime() - start;

        // HiGHS求解
        start = System.nanoTime();
        OptResult highsResult = highsSolver.solve(c, null, null, A_eq, b_eq);
        long highsTime = System.nanoTime() - start;

        assertTrue(javaResult.isConverged(), "Java求解器应收敛: " + javaResult.getConvergenceReason());
        assertTrue(highsResult.isConverged(), "HiGHS求解器应收敛");

        IVector javaSol = javaResult.getOptimalPoint();
        boolean constraintsOk = verifySolution(javaSol, A_eq, b_eq, LOOSE_TOLERANCE);

        double javaOpt = javaResult.getOptimalValue();
        double highsOpt = highsResult.getOptimalValue();
        double relDiff = Math.abs(highsOpt - javaOpt) / Math.abs(highsOpt);
        double absDiff = Math.abs(highsOpt - javaOpt);

        System.out.println("约束满足: " + (constraintsOk ? "✅" : "❌"));
        System.out.printf("Java: opt=%12.4f, 时间=%8.2fms%n", javaOpt, javaTime/1e6);
        System.out.printf("HiGHS: opt=%12.4f, 时间=%8.2fms%n", highsOpt, highsTime/1e6);
        System.out.printf("相对差异: %.2e (%.2e)%n", relDiff, absDiff);
        System.out.printf("性能比: HiGHS/Java = %.2fx%n", (double)highsTime/javaTime);

        assertTrue(constraintsOk, "解应满足约束");
        assertEquals(highsOpt, javaOpt, Math.max(Math.abs(highsOpt) * 0.01, 0.1),
                "目标值应接近 (relDiff=" + (relDiff*100) + "%)");
    }

    @Test
    @DisplayName("测试总结")
    void testSummary() {
        System.out.println("\n========== RereSimplexLinProgSolver 大规模测试总结 ==========");
        System.out.println("大规模测试: 50, 100, 200, 500, 1000 变量");
        System.out.println("随机批次: 20次 x 200变量");
        System.out.println("性能测试: 200变量 x 5次");
        System.out.println("==========================================================");
    }
}
