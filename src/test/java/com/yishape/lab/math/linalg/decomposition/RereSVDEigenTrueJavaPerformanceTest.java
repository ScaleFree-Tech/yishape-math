package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.compute.hpc.HpcSwitch;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.RereSVDDecompBlas2;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.AfterAll;

import static org.junit.jupiter.api.Assertions.*;

/**
 * True Pure Java performance test (HPC + SIMD disabled).
 * Uses HpcSwitch to disable native HPC and -Dyishape.math.use.simd=false to disable SIMD.
 */
@Disabled("SVD/Eigen 性能基准，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
public class RereSVDEigenTrueJavaPerformanceTest {

    private static final int WARMUP_ITERATIONS = 1;
    private static final int MEASURE_ITERATIONS = 2;

    @BeforeAll
    public static void setup() {
        System.out.println("\n=== TRUE PURE JAVA MODE (HPC + SIMD DISABLED) ===");
        System.out.println("Before: HpcSwitch.isEnabled() = " + HpcSwitch.isEnabled());

        // Disable HPC
        HpcSwitch.disable();

        // Check SIMD status
        String simdProp = System.getProperty("yishape.math.use.simd", "true");
        System.out.println("System property yishape.math.use.simd = " + simdProp);
        System.out.println("After: HpcSwitch.isEnabled() = " + HpcSwitch.isEnabled());
    }

    @AfterAll
    public static void teardown() {
        HpcSwitch.enable();
        System.out.println("\nHPC re-enabled, HpcSwitch.isEnabled() = " + HpcSwitch.isEnabled());
    }

    // ========== SVD True Pure Java Performance Tests ==========

    @Test
    @Tag("performance")
    public void testSVDPerformance_SquareMatrices() {
        System.out.println("\n--- SVD True Pure Java Performance ---");
        int[] sizes = {10, 20, 30, 50, 100, 200, 300};
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
            assertNotNull(svd.decompose(A).getFirst());
        }
    }

    @Test
    @Tag("performance")
    public void testSVDPerformance_NonSquareMatrices() {
        System.out.println("\n--- SVD Non-Square True Pure Java Performance ---");

        // Tall matrices
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

        // Wide matrices
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

    // ========== Eigen True Pure Java Performance Tests ==========

    @Test
    @Tag("performance")
    public void testEigenPerformance_SymmetricMatrices() {
        System.out.println("\n--- Eigen Symmetric True Pure Java Performance ---");
        int[] sizes = {10, 20, 30, 50, 100, 200, 300};
        for (int size : sizes) {
            double[][] data = createRandomSymmetricMatrix(size, 12345);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereEigenDecomposition warmupEigen = new RereEigenDecomposition();
            warmupEigen.decompose(A);

            // Measure
            RereEigenDecomposition eigen = new RereEigenDecomposition();
            long time = measureEigen(eigen, A);

            System.out.printf("Eigen Sym %3d x %-3d: %8d ms%n", size, size, time);
            assertNotNull(eigen.decompose(A)._1);
        }
    }

    @Test
    @Tag("performance")
    public void testEigenPerformance_NonSymmetricMatrices() {
        System.out.println("\n--- Eigen Non-Symmetric True Pure Java Performance ---");
        int[] sizes = {10, 20, 30, 50, 100};
        for (int size : sizes) {
            double[][] data = createRandomMatrix(size, size, 11111);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            RereEigenDecomposition warmupEigen = new RereEigenDecomposition();
            warmupEigen.decompose(A);

            // Measure
            RereEigenDecomposition eigen = new RereEigenDecomposition();
            long time = measureEigen(eigen, A);

            System.out.printf("Eigen Gen  %3d x %-3d: %8d ms%n", size, size, time);
            assertNotNull(eigen.decompose(A)._1);
        }
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

    private long measureEigen(RereEigenDecomposition eigen, IMatrix<Double> A) {
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
