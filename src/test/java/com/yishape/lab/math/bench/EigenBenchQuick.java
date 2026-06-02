package com.yishape.lab.math.bench;

import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.compute.hpc.HpcSwitch;

public class EigenBenchQuick {
    static final int WARMUP = 3;
    static final int REPEAT = 10;
    static final long SEED = 42;

    static IMatrix<Double> genMatrix(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i][j] = rng.nextGaussian();
        return Linalg.matrix(A);
    }

    public static void main(String[] args) {
        HpcSwitch.disable();
        for (int n : new int[]{50, 100, 200}) {
            IMatrix<Double> a = genMatrix(n, SEED + 9 + n);

            // Warmup
            for (int w = 0; w < WARMUP; w++) a.eigen();

            // Timed
            long best = Long.MAX_VALUE;
            for (int r = 0; r < REPEAT; r++) {
                long t0 = System.nanoTime();
                var result = a.eigen();
                long t1 = System.nanoTime();
                best = Math.min(best, t1 - t0);

                // Verify on last run
                if (r == 0) {
                    double trace = 0;
                    double[][] ad = ((IDoubleMatrix)a).getData();
                    for (int i = 0; i < n; i++) trace += ad[i][i];
                    double evSum = 0;
                    for (int i = 0; i < n; i++) evSum += result._1.get(i);
                    System.out.printf("n=%d: trace=%.8e evSum=%.8e diff=%.2e%n", n, trace, evSum, Math.abs(trace-evSum));
                }
            }
            double ms = best / 1_000_000.0;
            System.out.printf("EigenGen n=%d: %.1f ms (best of %d)%n", n, ms, REPEAT);
        }
    }
}
