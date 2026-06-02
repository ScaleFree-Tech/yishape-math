package com.yishape.lab.math.bench;
import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.compute.hpc.HpcSwitch;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;

public class EigenIterCount2 {
    static final int WARMUP = 3;
    static final long SEED = 42;

    static IMatrix<Double> gaussian(int n) {
        java.util.Random rng = new java.util.Random(SEED + 9 + n);
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i][j] = rng.nextGaussian();
        return Linalg.matrix(A);
    }

    static IMatrix<Double> uniform(int n) {
        return Linalg.rand(n, n, SEED + 9 + n);
    }

    public static void main(String[] args) {
        HpcSwitch.disable();

        for (int n : new int[]{50, 100, 200}) {
            System.out.printf("--- n=%d ---%n", n);
            for (String type : new String[]{"gaussian", "uniform"}) {
                IMatrix<Double> a = type.equals("gaussian") ? gaussian(n) : uniform(n);

                // Warmup
                for (int w = 0; w < WARMUP; w++) a.eigen();

                // Timed run
                var decomp = new RereEigenDecomposition();
                long t0 = System.nanoTime();
                var result = decomp.decompose(a);
                long t1 = System.nanoTime();
                double ms = (t1 - t0) / 1_000_000.0;
                var ev = result.getFirst();

                // Verify
                double[][] ad = ((com.yishape.lab.math.linalg.IDoubleMatrix)a).getData();
                double trace = 0;
                for (int i = 0; i < n; i++) trace += ad[i][i];
                double evSum = 0;
                for (int i = 0; i < n; i++) evSum += ev.get(i);

                System.out.printf("  %-10s: %.1fms  steps=%d  trace=%.4e evSum=%.4e diff=%.1e%n",
                    type, ms, decomp.lastStepCount, trace, evSum, Math.abs(trace-evSum));
            }
        }
    }
}
