package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.ISVDDecomposition;
import org.junit.jupiter.api.Test;

import java.util.Random;

/**
 * 快速性能基准：对比 FaerStyle BLAS-3 与 legacy BLAS-2（通过 RereSVDDecompBlas2 开关）。
注意：这是临时测试类，仅用于初步评估优化效果。
 */
public class RereSVDBenchmarkQuick {

    private static final int WARMUP = 3;
    private static final int RUNS = 5;

    @Test
    public void benchmarkAll() {
        System.out.println("=== SVD 快速性能基准 ===");
        System.out.printf("%-10s %-12s %-12s %-12s %-10s%n",
                "规模", "BLAS-3 (ms)", "BLAS-2 (ms)", "加速比", "误差");
        System.out.println("-".repeat(60));

        int[] sizes = {50, 100, 200, 300, 500, 800, 1000};
        for (int n : sizes) {
            double[][] data = randomMatrix(n, n, 42 + n);
            double blas3 = benchmarkFaerStyle(data);
            double blas2 = benchmarkLegacy(data);
            double speedup = blas2 / blas3;
            double err = reconstructionError(data, new RereSVDDecompBlas3());
            System.out.printf("%-10s %-12.2f %-12.2f %-12.2f %-10.2e%n",
                    n + "x" + n, blas3, blas2, speedup, err);
        }

        // tall / wide 矩阵
        System.out.println("-".repeat(60));
        System.out.println("矩形矩阵:");
        int[][] shapes = {{500, 100}, {100, 500}, {1000, 200}, {200, 1000}};
        for (int[] shape : shapes) {
            int m = shape[0], n = shape[1];
            double[][] data = randomMatrix(m, n, 99);
            double blas3 = benchmarkFaerStyle(data);
            double blas2 = benchmarkLegacy(data);
            double speedup = blas2 / blas3;
            double err = reconstructionError(data, new RereSVDDecompBlas3());
            System.out.printf("%-10s %-12.2f %-12.2f %-12.2f %-10.2e%n",
                    m + "x" + n, blas3, blas2, speedup, err);
        }
    }

    private double benchmarkFaerStyle(double[][] data) {
        ISVDDecomposition svd = new RereSVDDecompBlas3();
        IMatrix<Double> mat = Linalg.matrix(data);
        for (int i = 0; i < WARMUP; i++) svd.decompose(mat);
        long t0 = System.nanoTime();
        for (int i = 0; i < RUNS; i++) svd.decompose(mat);
        long t1 = System.nanoTime();
        return (t1 - t0) / 1_000_000.0 / RUNS;
    }

    private double benchmarkLegacy(double[][] data) {
        RereSVDDecompBlas2 svd = new RereSVDDecompBlas2();
        svd.setUseFaerStyle(false);
        IMatrix<Double> mat = Linalg.matrix(data);
        for (int i = 0; i < WARMUP; i++) svd.decompose(mat);
        long t0 = System.nanoTime();
        for (int i = 0; i < RUNS; i++) svd.decompose(mat);
        long t1 = System.nanoTime();
        return (t1 - t0) / 1_000_000.0 / RUNS;
    }

    private double reconstructionError(double[][] data, ISVDDecomposition svd) {
        IMatrix<Double> A = Linalg.matrix(data);
        var r = svd.decompose(A);
        IMatrix<Double> U = r.getFirst();
        var S = r.getSecond();
        IMatrix<Double> VT = r.getThird();
        int m = A.rows(), n = A.cols(), k = Math.min(m, n);
        double maxErr = 0.0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += U.get(i, l) * S.get(l) * VT.get(l, j);
                }
                maxErr = Math.max(maxErr, Math.abs(sum - data[i][j]));
            }
        }
        return maxErr;
    }

    private static double[][] randomMatrix(int rows, int cols, long seed) {
        Random rng = new Random(seed);
        double[][] data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rng.nextGaussian();
            }
        }
        return data;
    }
}
