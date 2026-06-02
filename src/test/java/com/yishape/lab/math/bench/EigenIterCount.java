package com.yishape.lab.math.bench;

import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.linalg.decomposition.impl.RereEigenDecomposition;
import com.yishape.lab.math.compute.hpc.HpcSwitch;

public class EigenIterCount {
    static final long SEED = 42;

    static double[][] genMatrix(int n, long seed) {
        java.util.Random rng = new java.util.Random(seed);
        double[][] A = new double[n][n];
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) A[i][j] = rng.nextGaussian();
        return A;
    }

    public static void main(String[] args) {
        HpcSwitch.disable();
        for (int n : new int[]{50, 100, 200}) {
            double[][] A = genMatrix(n, SEED + 9 + n);
            IMatrix<Double> M = Linalg.matrix(A);

            long t0 = System.nanoTime();
            var result = M.eigen();
            long t1 = System.nanoTime();
            double ms = (t1 - t0) / 1_000_000.0;

            double trace = 0;
            for (int i = 0; i < n; i++) trace += A[i][i];
            double evSum = 0;
            for (int i = 0; i < n; i++) evSum += result._1.get(i);

            System.out.printf("n=%d: %.1f ms  trace=%.4e  evSum=%.4e  diff=%.2e%n",
                n, ms, trace, evSum, Math.abs(trace-evSum));
        }
    }
}
