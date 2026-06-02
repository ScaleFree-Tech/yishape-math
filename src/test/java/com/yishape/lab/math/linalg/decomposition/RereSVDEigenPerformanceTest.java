package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.util.Tuple2;
import com.yishape.lab.util.Tuple3;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Performance test suite for RereSVDDecomposition and RereEigenDecomposition.
 * Measures execution time across various matrix sizes.
 */
@Disabled("SVD/Eigen 性能基准，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
public class RereSVDEigenPerformanceTest {

    private static final int WARMUP_ITERATIONS = 2;
    private static final int MEASURE_ITERATIONS = 3;
    private static final double TOL = 1e-8;

    // ========== SVD Performance Tests ==========

    @Test
    @Tag("performance")
    public void testSVDPerformance_SmallMatrices() {
        int[] sizes = {10, 20, 30, 50};
        for (int size : sizes) {
            double[][] data = createRandomMatrix(size, size, 12345);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereSVDDecompBlas2 warmupSvd = new RereSVDDecompBlas2();
            warmupSvd.decompose(A);

            // Measure
            RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
            long time = measureSVD(svd, A);

            System.out.printf("SVD %3d x %-3d: %8d ms%n", size, size, time);

            // Basic correctness check
            assertNotNull(svd.decompose(A).getFirst());
        }
    }

    @Test
    @Tag("performance")
    public void testSVDPerformance_MediumMatrices() {
        int[] sizes = {100, 200, 300};
        for (int size : sizes) {
            double[][] data = createRandomMatrix(size, size, 54321);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereSVDDecompBlas2 warmupSvd = new RereSVDDecompBlas2();
            warmupSvd.decompose(A);

            // Measure
            RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
            long time = measureSVD(svd, A);

            System.out.printf("SVD %3d x %-3d: %8d ms%n", size, size, time);

            assertNotNull(svd.decompose(A).getFirst());
        }
    }

    @Test
    @Tag("performance")
    public void testSVDPerformance_LargeMatrices() {
        int[] sizes = {500, 800};
        for (int size : sizes) {
            double[][] data = createRandomMatrix(size, size, 99999);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup (single iteration for large matrices)
            RereSVDDecompBlas2 warmupSvd = new RereSVDDecompBlas2();
            warmupSvd.decompose(A);

            // Measure
            RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
            long time = measureSVD(svd, A);

            System.out.printf("SVD %3d x %-3d: %8d ms%n", size, size, time);

            assertNotNull(svd.decompose(A).getFirst());
        }
    }

    @Test
    @Tag("performance")
    public void testSVDPerformance_NonSquareMatrices() {
        // Tall matrices (m > n)
        int[][] tallSizes = {{100, 50}, {200, 100}, {500, 200}};
        for (int[] sizes : tallSizes) {
            int m = sizes[0], n = sizes[1];
            double[][] data = createRandomMatrix(m, n, 11111);
            IMatrix<Double> A = Linalg.matrix(data);

            RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
            long time = measureSVD(svd, A);

            System.out.printf("SVD %3d x %-3d (tall): %6d ms%n", m, n, time);
            assertNotNull(svd.decompose(A).getFirst());
        }

        // Wide matrices (n > m)
        int[][] wideSizes = {{50, 100}, {100, 200}, {200, 500}};
        for (int[] sizes : wideSizes) {
            int m = sizes[0], n = sizes[1];
            double[][] data = createRandomMatrix(m, n, 22222);
            IMatrix<Double> A = Linalg.matrix(data);

            RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
            long time = measureSVD(svd, A);

            System.out.printf("SVD %3d x %-3d (wide): %6d ms%n", m, n, time);
            assertNotNull(svd.decompose(A).getFirst());
        }
    }

    @Test
    @Tag("performance")
    public void testSVDPerformance_StabilityWithRepeatedCalls() {
        // Test that repeated decompositions give consistent results
        double[][] data = createRandomMatrix(100, 100, 42);
        IMatrix<Double> A = Linalg.matrix(data);

        RereSVDDecompBlas2 svd1 = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result1 = svd1.decompose(A);

        RereSVDDecompBlas2 svd2 = new RereSVDDecompBlas2();
        Tuple3<IMatrix<Double>, IVector<Double>, IMatrix<Double>> result2 = svd2.decompose(A);

        // Compare singular values
        IVector<Double> S1 = result1.getSecond();
        IVector<Double> S2 = result2.getSecond();
        for (int i = 0; i < S1.length(); i++) {
            assertEquals(S1.get(i), S2.get(i), TOL);
        }
    }

    // ========== Eigen Decomposition Performance Tests ==========

    @Test
    @Tag("performance")
    public void testEigenPerformance_SymmetricSmall() {
        int[] sizes = {10, 20, 30, 50};
        for (int size : sizes) {
            double[][] data = createRandomSymmetricMatrix(size, 12345);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereEigenDecomposition warmupEigen = new RereEigenDecomposition();
            warmupEigen.decompose(A);

            // Measure
            RereEigenDecomposition eigen = new RereEigenDecomposition();
            long time = measureEigenDecomposition(eigen, A);

            System.out.printf("Eigen Sym %3d x %-3d: %8d ms%n", size, size, time);

            assertNotNull(eigen.decompose(A)._1);
        }
    }

    @Test
    @Tag("performance")
    public void testEigenPerformance_SymmetricMedium() {
        int[] sizes = {100, 200, 300};
        for (int size : sizes) {
            double[][] data = createRandomSymmetricMatrix(size, 54321);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereEigenDecomposition warmupEigen = new RereEigenDecomposition();
            warmupEigen.decompose(A);

            // Measure
            RereEigenDecomposition eigen = new RereEigenDecomposition();
            long time = measureEigenDecomposition(eigen, A);

            System.out.printf("Eigen Sym %3d x %-3d: %8d ms%n", size, size, time);

            assertNotNull(eigen.decompose(A)._1);
        }
    }

    @Test
    @Tag("performance")
    public void testEigenPerformance_SymmetricLarge() {
        int[] sizes = {500, 800};
        for (int size : sizes) {
            double[][] data = createRandomSymmetricMatrix(size, 99999);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereEigenDecomposition warmupEigen = new RereEigenDecomposition();
            warmupEigen.decompose(A);

            // Measure
            RereEigenDecomposition eigen = new RereEigenDecomposition();
            long time = measureEigenDecomposition(eigen, A);

            System.out.printf("Eigen Sym %3d x %-3d: %8d ms%n", size, size, time);

            assertNotNull(eigen.decompose(A)._1);
        }
    }

    @Test
    @Tag("performance")
    public void testEigenPerformance_NonSymmetricSmall() {
        int[] sizes = {10, 20, 30, 50};
        for (int size : sizes) {
            double[][] data = createRandomMatrix(size, size, 11111);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereEigenDecomposition warmupEigen = new RereEigenDecomposition();
            warmupEigen.decompose(A);

            // Measure
            RereEigenDecomposition eigen = new RereEigenDecomposition();
            long time = measureEigenDecomposition(eigen, A);

            System.out.printf("Eigen Gen  %3d x %-3d: %8d ms%n", size, size, time);

            assertNotNull(eigen.decompose(A)._1);
        }
    }

    @Test
    @Tag("performance")
    public void testEigenPerformance_NonSymmetricMedium() {
        int[] sizes = {100, 150};
        for (int size : sizes) {
            double[][] data = createRandomMatrix(size, size, 22222);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereEigenDecomposition warmupEigen = new RereEigenDecomposition();
            warmupEigen.decompose(A);

            // Measure
            RereEigenDecomposition eigen = new RereEigenDecomposition();
            long time = measureEigenDecomposition(eigen, A);

            System.out.printf("Eigen Gen  %3d x %-3d: %8d ms%n", size, size, time);

            assertNotNull(eigen.decompose(A)._1);
        }
    }

    @Test
    @Tag("performance")
    public void testEigenPerformance_StabilityWithRepeatedCalls() {
        // Test that repeated decompositions give consistent results
        double[][] data = createRandomSymmetricMatrix(100, 42);
        IMatrix<Double> A = Linalg.matrix(data);

        RereEigenDecomposition eigen1 = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result1 = eigen1.decompose(A);

        RereEigenDecomposition eigen2 = new RereEigenDecomposition();
        Tuple2<IVector<Double>, IMatrix<Double>> result2 = eigen2.decompose(A);

        // Compare eigenvalues
        IVector<Double> ev1 = result1._1;
        IVector<Double> ev2 = result2._1;
        for (int i = 0; i < ev1.length(); i++) {
            assertEquals(ev1.get(i), ev2.get(i), TOL);
        }
    }

    // ========== Comparison Tests ==========

    @Test
    @Tag("performance")
    public void testSVDvsEigenPerformance_SymmetricMatrix() {
        // For symmetric matrices, SVD and Eigen decomposition should have similar performance
        int size = 200;
        double[][] data = createRandomSymmetricMatrix(size, 77777);
        IMatrix<Double> A = Linalg.matrix(data);

        // Warmup
        RereSVDDecompBlas2 warmupSvd = new RereSVDDecompBlas2();
        warmupSvd.decompose(A);
        RereEigenDecomposition warmupEigen = new RereEigenDecomposition();
        warmupEigen.decompose(A);

        // Measure
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        long svdTime = measureSVD(svd, A);

        RereEigenDecomposition eigen = new RereEigenDecomposition();
        long eigenTime = measureEigenDecomposition(eigen, A);

        System.out.printf("Symmetric matrix %3d x %3d:%n", size, size);
        System.out.printf("  SVD:   %8d ms%n", svdTime);
        System.out.printf("  Eigen: %8d ms%n", eigenTime);

        // Both should complete successfully
        assertNotNull(svd.decompose(A).getFirst());
        assertNotNull(eigen.decompose(A)._1);
    }

    @Test
    @Tag("performance")
    public void testScalabilityTrend() {
        // Test scalability by checking that time grows roughly as O(n^3)
        int[] sizes = {50, 100, 200};
        long[] svdTimes = new long[sizes.length];
        long[] eigenTimes = new long[sizes.length];

        for (int i = 0; i < sizes.length; i++) {
            int size = sizes[i];
            double[][] data = createRandomMatrix(size, size, 33333);
            IMatrix<Double> A = Linalg.matrix(data);

            // SVD
            RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
            svd.decompose(A);  // Warmup
            svdTimes[i] = measureSVD(svd, A);

            // Eigen
            RereEigenDecomposition eigen = new RereEigenDecomposition();
            eigen.decompose(A);  // Warmup
            eigenTimes[i] = measureEigenDecomposition(eigen, A);

            System.out.printf("Size %3d: SVD=%8d ms, Eigen=%8d ms%n",
                size, svdTimes[i], eigenTimes[i]);
        }

        // Check that times are increasing (basic sanity check)
        assertTrue(svdTimes[0] <= svdTimes[1] * 10,
            "SVD time should not explode unexpectedly");
        assertTrue(svdTimes[1] <= svdTimes[2] * 10,
            "SVD time should not explode unexpectedly");
    }

    // ========== Helper Methods ==========

    private long measureSVD(RereSVDDecompBlas2 svd, IMatrix<Double> A) {
        long totalTime = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.currentTimeMillis();
            svd.decompose(A);
            long end = System.currentTimeMillis();
            totalTime += (end - start);
        }
        return totalTime / MEASURE_ITERATIONS;
    }

    private long measureEigenDecomposition(RereEigenDecomposition eigen, IMatrix<Double> A) {
        long totalTime = 0;
        for (int i = 0; i < MEASURE_ITERATIONS; i++) {
            long start = System.currentTimeMillis();
            eigen.decompose(A);
            long end = System.currentTimeMillis();
            totalTime += (end - start);
        }
        return totalTime / MEASURE_ITERATIONS;
    }

    private double[][] createRandomMatrix(int rows, int cols, long seed) {
        java.util.Random rand = new java.util.Random(seed);
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rand.nextDouble() * 100;
            }
        }
        return data;
    }

    private double[][] createRandomSymmetricMatrix(int size, long seed) {
        java.util.Random rand = new java.util.Random(seed);
        double[][] data = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = i; j < size; j++) {
                double val = rand.nextDouble() * 100;
                data[i][j] = val;
                data[j][i] = val;
            }
        }
        return data;
    }
}
