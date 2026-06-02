package com.yishape.lab.math.bench;

import com.yishape.lab.math.linalg.*;
import com.yishape.lab.math.compute.hpc.HpcSwitch;

/**
 * Quick debug test for non-symmetric eigenvalue decomposition.
 */
public class EigenDebug {
    static double trace(double[][] A) {
        double t = 0; for (int i = 0; i < A.length; i++) t += A[i][i]; return t;
    }

    public static void main(String[] args) {
        HpcSwitch.disable();

        // 5x5 test matrix from NumPy rng(42)
        double[][] A = {
            { 1.5235853988e-01, -5.1999205312e-01,  3.7522559790e-01,  4.7028235820e-01, -9.7551759433e-01},
            {-6.5108975343e-01,  6.3920201584e-02, -1.5812129617e-01, -8.4005787521e-03, -4.2652196379e-01},
            { 4.3969898743e-01,  3.8889596771e-01,  3.3015348781e-02,  5.6362060348e-01,  2.3375467113e-01},
            {-4.2964623144e-01,  1.8437539204e-01, -4.7944130041e-01,  4.3922515065e-01, -2.4962955493e-02},
            {-9.2431181773e-02, -3.4046477220e-01,  6.1127066934e-01, -7.7264741034e-02, -2.1416391108e-01}
        };

        System.out.printf("trace(A) = %.10e%n", trace(A));

        IMatrix<Double> M = Linalg.matrix(A);
        var result = M.eigen();
        double[] evals = toArr1D(result._1);
        double sum = 0;
        System.out.print("eigenvalues: ");
        for (double e : evals) { sum += e; System.out.printf("% .8e ", e); }
        System.out.printf("%nsum(evals) = %.10e%n", sum);
        System.out.printf("diff from trace = %.2e%n", Math.abs(sum - trace(A)));
        System.out.printf("Expected sum = %.10e (NumPy)%n", 4.7435532981e-01);
    }

    static double[] toArr1D(IVector<Double> v) {
        double[] a = new double[v.size()];
        for (int i = 0; i < v.size(); i++) a[i] = v.get(i);
        return a;
    }
}
