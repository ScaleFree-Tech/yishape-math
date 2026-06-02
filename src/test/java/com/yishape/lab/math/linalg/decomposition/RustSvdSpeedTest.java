package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.compute.hpc.HpcSwitch;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.AfterAll;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test Rust SVD performance by using matrix.svd() which actually calls HpcLapackDecomps.trySvd()
 */
@Disabled("Rust SVD 速度基准，默认跳过；需要时去掉本注解或单独 -Dtest=... 运行")
public class RustSvdSpeedTest {

    @BeforeAll
    public static void setup() {
        System.out.println("\n=== Testing Rust SVD Path ===");
        System.out.println("HpcSwitch.isEnabled() = " + HpcSwitch.isEnabled());
    }

    @AfterAll
    public static void teardown() {
        HpcSwitch.enable();
    }

    @Test
    @Tag("performance")
    public void testRustSvdSpeed() {
        int[] sizes = {100, 200, 300, 500};

        System.out.println("\n--- SVD via matrix.svd() (uses Rust if threshold met) ---");
        for (int size : sizes) {
            double[][] data = createRandomMatrix(size, size, 12345);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            A.svd();
            A.svd();

            // Measure
            long start = System.currentTimeMillis();
            for (int i = 0; i < 3; i++) {
                A.svd();
            }
            long end = System.currentTimeMillis();
            long time = (end - start) / 3;

            long elements = (long) size * size;
            boolean shouldUseRust = elements >= 10001;

            System.out.printf("  %3dx%-3d: %4d ms (elements=%d, Rust threshold met=%s)%n",
                size, size, time, elements, shouldUseRust);
        }
    }

    @Test
    @Tag("performance")
    public void testRustSvdSpeed_Disabled() {
        System.out.println("\n--- SVD via matrix.svd() with HPC DISABLED ---");
        HpcSwitch.disable();
        System.out.println("HpcSwitch.isEnabled() = " + HpcSwitch.isEnabled());

        int[] sizes = {100, 200, 300, 500};
        for (int size : sizes) {
            double[][] data = createRandomMatrix(size, size, 12345);
            IMatrix<Double> A = Linalg.matrix(data);

            // Warmup
            A.svd();
            A.svd();

            // Measure
            long start = System.currentTimeMillis();
            for (int i = 0; i < 3; i++) {
                A.svd();
            }
            long end = System.currentTimeMillis();
            long time = (end - start) / 3;

            long elements = (long) size * size;
            boolean shouldUseRust = elements >= 10001;

            System.out.printf("  %3dx%-3d: %4d ms (elements=%d, Rust threshold met=%s)%n",
                size, size, time, elements, shouldUseRust);
        }
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
}
