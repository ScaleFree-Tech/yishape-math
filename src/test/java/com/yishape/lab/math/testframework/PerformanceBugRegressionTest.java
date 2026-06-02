package com.yishape.lab.math.testframework;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import com.yishape.lab.math.ml.clf.lr.RereLogisticRegression;
import com.yishape.lab.math.ml.clf.tree.RereDecisionTree;
import com.yishape.lab.math.ml.clf.tree.RereRandomForest;
import com.yishape.lab.math.signal.core.RereDCT;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Performance Bug Regression Test Suite.
 * Validates 7 previously fixed performance-related bugs using completely new test data.
 *
 * Framework: JUnit 5, direct assertions (no TestResult Recorder).
 * Timing: System.nanoTime() with BENCHMARK output format.
 */
@Disabled("性能回归基准，耗时长，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
public class PerformanceBugRegressionTest {

    private static final long FIXED_SEED = 42L;

    // ========== Helper methods ==========

    private static void benchmark(String module, String operation, String size, long timeNs) {
        long timeMs = timeNs / 1_000_000;
        System.out.println("BENCHMARK|" + module + "|" + operation + "|" + size + "|" + timeMs);
    }

    private static long timeNanos(Runnable task) {
        long start = System.nanoTime();
        task.run();
        return System.nanoTime() - start;
    }

    // ========== 1. SVD Decomposition Correctness ==========

    @Test
    @DisplayName("1. SVD decomposition correctness - 5x4 well-conditioned matrix")
    @Timeout(value = 60)
    void testSVDReconstruction_5x4_WellConditioned() {
        // Well-conditioned 5x4 matrix (distinct from previous 2x3 and 3x4 tests)
        double[][] data = {
            {2.5, 1.0, 3.0, 0.5},
            {1.0, 4.0, 2.0, 3.0},
            {3.0, 2.0, 5.0, 1.0},
            {0.5, 3.0, 1.0, 4.0},
            {1.5, 0.5, 2.5, 1.0}
        };
        IMatrix<Double> A = Linalg.matrix(data);

        long timeNs = timeNanos(() -> {
            ISVDDecomposition svd = Decomps.createSVD();
            Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result = svd.decompose(A);

            IMatrix<Double> U = result._1;
            IVector<Double> S = result._2;
            IMatrix<Double> VT = result._3;

            int m = A.rows();
            int n = A.cols();
            int k = Math.min(m, n);

            // 1. Reconstruct: U * diag(S) * VT
            IMatrix<Double> reconstructed = Linalg.zeros(m, n);
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double sum = 0.0;
                    for (int s = 0; s < k; s++) {
                        sum += U.get(i, s) * S.get(s) * VT.get(s, j);
                    }
                    reconstructed.put(i, j, sum);
                }
            }

            double maxDiff = 0.0;
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    maxDiff = Math.max(maxDiff, Math.abs(A.get(i, j) - reconstructed.get(i, j)));
                }
            }
            assertTrue(maxDiff < 1e-6, "SVD reconstruction error too large: " + maxDiff);

            // 2. Singular values must be positive and descending
            for (int i = 0; i < S.length(); i++) {
                assertTrue(S.get(i) >= 0, "Singular value " + i + " must be non-negative: " + S.get(i));
                if (i > 0) {
                    assertTrue(S.get(i) <= S.get(i - 1),
                        "Singular values must be descending: S[" + i + "]=" + S.get(i) + " > S[" + (i-1) + "]=" + S.get(i-1));
                }
            }

            // 3. U columns orthonormal: U^T * U = I (for thin SVD, U is m x k)
            IMatrix<Double> UTU = U.transpose().mmul(U);
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    double actual = UTU.get(i, j);
                    assertTrue(Math.abs(actual - expected) < 1e-6,
                        "U orthogonality violation at (" + i + "," + j + "): " + actual);
                }
            }

            // 4. VT rows orthonormal: VT * VT^T = I (VT is k x n)
            IMatrix<Double> VTV = VT.mmul(VT.transpose());
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    double actual = VTV.get(i, j);
                    assertTrue(Math.abs(actual - expected) < 1e-6,
                        "V^T orthogonality violation at (" + i + "," + j + "): " + actual);
                }
            }
        });

        benchmark("linalg", "svd", "5x4", timeNs);
    }

    // ========== 2. Linear Solve Correctness ==========

    @Test
    @DisplayName("2a. Linear solve correctness - 5x5 diagonally dominant")
    @Timeout(value = 60)
    void testLinearSolve_5x5_DiagonallyDominant() {
        double[][] Adata = {
            {5, 1, 2, 3, 1},
            {1, 6, 1, 2, 1},
            {2, 1, 7, 1, 2},
            {3, 2, 1, 8, 1},
            {1, 1, 2, 1, 9}
        };
        double[] bdata = {15, 20, 25, 30, 35};
        IMatrix<Double> A = Linalg.matrix(Adata);
        IVector<Double> b = IVector.of(bdata);

        long timeNs = timeNanos(() -> {
            IVector<Double> x = A.solve(b);

            // Verify A*x = b
            IVector<Double> Ax = A.mmul(x);
            double maxDiff = 0.0;
            for (int i = 0; i < bdata.length; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(Ax.get(i) - b.get(i)));
            }
            assertTrue(maxDiff < 1e-8, "Linear solve error too large: " + maxDiff);
        });

        benchmark("linalg", "solve", "5x5", timeNs);
    }

    @Test
    @DisplayName("2b. Linear solve correctness - 100x100 well-conditioned Hilbert variant")
    @Timeout(value = 120)
    void testLinearSolve_100x100_HilbertVariant() {
        int n = 100;
        double[][] Adata = new double[n][n];
        double[] xExpected = new double[n];
        double[] bdata = new double[n];

        // Hilbert matrix variant: A[i][j] = 1/(i+j+1) + (i==j ? 100 : 0)
        for (int i = 0; i < n; i++) {
            xExpected[i] = 1.0; // solution should be all ones
            for (int j = 0; j < n; j++) {
                Adata[i][j] = 1.0 / (i + j + 1) + (i == j ? 100.0 : 0.0);
            }
        }

        // b = A * ones
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += Adata[i][j] * xExpected[j];
            }
            bdata[i] = sum;
        }

        IMatrix<Double> A = Linalg.matrix(Adata);
        IVector<Double> b = IVector.of(bdata);

        long timeNs = timeNanos(() -> {
            IVector<Double> x = A.solve(b);

            // Verify solution is close to all ones
            double maxDiff = 0.0;
            for (int i = 0; i < n; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(x.get(i) - xExpected[i]));
            }
            assertTrue(maxDiff < 1e-6, "Hilbert variant solve error too large: " + maxDiff);
        });

        benchmark("linalg", "solve", "100x100", timeNs);
    }

    // ========== 3. Matrix Inverse Correctness ==========

    @Test
    @DisplayName("3. Matrix inverse correctness - 3x3")
    @Timeout(value = 60)
    void testMatrixInverse_3x3() {
        double[][] data = {
            {3, 1, 4},
            {1, 5, 9},
            {2, 6, 5}
        };
        IMatrix<Double> A = Linalg.matrix(data);

        long timeNs = timeNanos(() -> {
            IMatrix<Double> invA = A.inv();
            IMatrix<Double> product = invA.mmul(A);

            int n = A.rows();
            double maxDiff = 0.0;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    maxDiff = Math.max(maxDiff, Math.abs(product.get(i, j) - expected));
                }
            }
            assertTrue(maxDiff < 1e-8, "Matrix inverse error too large: " + maxDiff);
        });

        benchmark("linalg", "inv", "3x3", timeNs);
    }

    // ========== 4. Logistic Regression Correctness ==========

    @Test
    @DisplayName("4. Logistic regression correctness - 4-class linearly separable")
    @Timeout(value = 120)
    void testLogisticRegression_4Class() {
        double[][] features = {
            {0.1, 0.1}, {0.2, 0.3}, {0.3, 0.2}, {0.4, 0.5}, {0.5, 0.4},   // A类
            {-0.1, -0.1}, {-0.2, -0.3}, {-0.3, -0.2}, {-0.4, -0.5}, {-0.5, -0.4}, // B类
            {0.1, -0.1}, {0.2, -0.3}, {0.3, -0.2}, {0.4, -0.5}, {0.5, -0.4},   // C类
            {-0.1, 0.1}, {-0.2, 0.3}, {-0.3, 0.2}, {-0.4, 0.5}, {-0.5, 0.4}    // D类
        };
        String[] labels = {
            "A", "A", "A", "A", "A",
            "B", "B", "B", "B", "B",
            "C", "C", "C", "C", "C",
            "D", "D", "D", "D", "D"
        };

        IMatrix<Double> X = Linalg.matrix(features);

        long timeNs = timeNanos(() -> {
            RereLogisticRegression lr = new RereLogisticRegression();
            lr.setStandardizeFeatures(false);
            lr.fit(X, labels);

            String[] predictions = lr.predictBatch(X);
            int correct = 0;
            for (int i = 0; i < labels.length; i++) {
                if (predictions[i].equals(labels[i])) correct++;
            }
            double accuracy = (double) correct / labels.length;
            assertTrue(accuracy >= 0.85, "Logistic regression accuracy too low: " + accuracy);
        });

        benchmark("ml", "logistic_regression", "20", timeNs);
    }

    // ========== 5. Decision Tree Correctness ==========

    @Test
    @DisplayName("5. Decision tree correctness - 100 samples 3D")
    @Timeout(value = 120)
    void testDecisionTree_100_3D() {
        int n = 100;
        double[][] features = new double[n][3];
        String[] labels = new String[n];

        Random rand = new Random(FIXED_SEED);
        for (int i = 0; i < n; i++) {
            features[i][0] = rand.nextDouble() * 4 - 2; // [-2, 2]
            features[i][1] = rand.nextDouble() * 4 - 2;
            features[i][2] = rand.nextDouble() * 4 - 2;

            double sum = features[i][0] + features[i][1] + features[i][2];
            if (sum > 1.0) {
                labels[i] = "pos";
            } else if (sum < -1.0) {
                labels[i] = "neg";
            } else {
                labels[i] = "mid";
            }
        }

        IMatrix<Double> X = Linalg.matrix(features);

        long timeNs = timeNanos(() -> {
            RereDecisionTree dt = new RereDecisionTree();
            dt.fit(X, labels);

            String[] predictions = dt.predictBatch(X);
            int correct = 0;
            for (int i = 0; i < labels.length; i++) {
                if (predictions[i].equals(labels[i])) correct++;
            }
            double accuracy = (double) correct / labels.length;
            assertTrue(accuracy >= 0.80, "Decision tree accuracy too low: " + accuracy);
        });

        benchmark("ml", "decision_tree", "100", timeNs);
    }

    // ========== 6. Random Forest Correctness ==========

    @Test
    @DisplayName("6. Random forest correctness - 100 samples 3D")
    @Timeout(value = 120)
    void testRandomForest_100_3D() {
        int n = 100;
        double[][] features = new double[n][3];
        String[] labels = new String[n];

        Random rand = new Random(FIXED_SEED);
        for (int i = 0; i < n; i++) {
            features[i][0] = rand.nextDouble() * 4 - 2;
            features[i][1] = rand.nextDouble() * 4 - 2;
            features[i][2] = rand.nextDouble() * 4 - 2;

            double sum = features[i][0] + features[i][1] + features[i][2];
            if (sum > 1.0) {
                labels[i] = "pos";
            } else if (sum < -1.0) {
                labels[i] = "neg";
            } else {
                labels[i] = "mid";
            }
        }

        IMatrix<Double> X = Linalg.matrix(features);

        long timeNs = timeNanos(() -> {
            RereRandomForest rf = new RereRandomForest();
            rf.setnEstimators(50);
            rf.setRandomSeed(FIXED_SEED);
            rf.fit(X, labels);

            String[] predictions = rf.predictBatch(X);
            int correct = 0;
            for (int i = 0; i < labels.length; i++) {
                if (predictions[i].equals(labels[i])) correct++;
            }
            double accuracy = (double) correct / labels.length;
            assertTrue(accuracy >= 0.80, "Random forest accuracy too low: " + accuracy);
        });

        benchmark("ml", "random_forest", "100", timeNs);
    }

    // ========== 7. DCT Correctness ==========

    @Test
    @DisplayName("7a. DCT-II/IDCT-II roundtrip - 8-point sequence")
    @Timeout(value = 60)
    void testDCT_Roundtrip_8() {
        double[] data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        IVector<Double> signal = IVector.of(data);

        long timeNs = timeNanos(() -> {
            IVector<Double> dct = RereDCT.dct2(signal);
            IVector<Double> reconstructed = RereDCT.idct2(dct);

            double maxDiff = 0.0;
            for (int i = 0; i < data.length; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(reconstructed.get(i) - data[i]));
            }
            assertTrue(maxDiff < 1e-10, "DCT roundtrip error too large: " + maxDiff);

            // Constant sequence: first coefficient should be sqrt(8)*5, rest 0
            double[] constantData = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
            IVector<Double> constantSignal = IVector.of(constantData);
            IVector<Double> constantDct = RereDCT.dct2(constantSignal);

            double expectedFirst = Math.sqrt(8.0) * 5.0;
            assertTrue(Math.abs(constantDct.get(0) - expectedFirst) < 1e-10,
                "DCT first coefficient mismatch: expected " + expectedFirst + ", got " + constantDct.get(0));

            for (int i = 1; i < 8; i++) {
                assertTrue(Math.abs(constantDct.get(i)) < 1e-10,
                    "DCT coefficient " + i + " should be 0, got " + constantDct.get(i));
            }
        });

        benchmark("signal", "dct", "8", timeNs);
    }

    @Test
    @DisplayName("7b. DCT-II/IDCT-II roundtrip - 16384 random signal")
    @Timeout(value = 120)
    void testDCT_Roundtrip_16384() {
        int n = 16384;
        Random rand = new Random(FIXED_SEED);
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = rand.nextDouble();
        }
        IVector<Double> signal = IVector.of(data);

        long timeNs = timeNanos(() -> {
            IVector<Double> dct = RereDCT.dct2(signal);
            IVector<Double> reconstructed = RereDCT.idct2(dct);

            double maxDiff = 0.0;
            for (int i = 0; i < n; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(reconstructed.get(i) - data[i]));
            }
            assertTrue(maxDiff < 1e-6, "DCT roundtrip error for 16384 too large: " + maxDiff);
        });

        benchmark("signal", "dct", "16384", timeNs);
    }
}
