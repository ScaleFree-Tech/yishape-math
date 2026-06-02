package com.yishape.lab.math.bench;

import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.compute.hpc.HpcSwitch;

public class EigenBenchAll {
    static final int WARMUP = 3;
    static final int REPEAT = 8;
    static final long SEED = 42;

    static IMatrix<Double> genMatrix(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i][j] = rng.nextGaussian();
        return Linalg.matrix(A);
    }

    static double bench(String label, IMatrix<Double> a, int runs) {
        long best = Long.MAX_VALUE;
        for (int r = 0; r < runs; r++) {
            long t0 = System.nanoTime();
            a.eigen();
            long t1 = System.nanoTime();
            best = Math.min(best, t1 - t0);
        }
        double ms = best / 1_000_000.0;
        // Verify
        var result = a.eigen();
        double[][] ad = ((IDoubleMatrix)a).getData();
        double trace = 0;
        for (int i = 0; i < ad.length; i++) trace += ad[i][i];
        double evSum = 0;
        for (int i = 0; i < ad.length; i++) evSum += result._1.get(i);
        System.out.printf("  %s: %.1f ms  trace=%.4e evSum=%.4e diff=%.1e%n",
            label, ms, trace, evSum, Math.abs(trace-evSum));
        return ms;
    }

    public static void main(String[] args) {
        for (int n : new int[]{50, 100, 200}) {
            System.out.printf("n=%d:%n", n);
            IMatrix<Double> a = genMatrix(n, SEED + 9 + n);
            double[][] ad = ((IDoubleMatrix)a).getData();

            // Pure Java
            HpcSwitch.disable();
            for (int w = 0; w < WARMUP; w++) a.eigen();
            double msP = bench("java_pure", a, REPEAT);

            // HPC API (falls back to Java for non-symmetric eigen)
            try {
                java.lang.reflect.Method m = HpcSwitch.class.getMethod("enable");
                m.invoke(null);
            } catch (Exception e) {
                // fallback
            }
            var a2 = Linalg.matrix(ad);
            for (int w = 0; w < WARMUP; w++) a2.eigen();
            double msA = bench("java_hpc_api", a2, REPEAT);
            HpcSwitch.disable();

            System.out.printf("  => pure=%.1fms  api=%.1fms%n%n", msP, msA);
        }
    }
}
