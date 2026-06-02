package com.yishape.lab.math.timeseries.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

public class LevinsonDurbinSolver {

    private LevinsonDurbinSolver() {}

    public static record YuleWalkerResult(
            IVector<Double> arCoefficients,
            double noiseVariance,
            IVector<Double> reflectionCoefficients,
            double[] aic) {}

    /**
     * Solve Yule-Walker equations for AR(p) coefficients.
     * Returns AR coefficients [a1, ..., ap] and noise variance.
     */
    public static IVector<Double> solveYuleWalker(double[] r, int p) {
        return solveYuleWalkerDetailed(r, p).arCoefficients();
    }

    /**
     * Solve Yule-Walker equations with full diagnostics.
     *
     * @param r autocorrelation values r[0..p]
     * @param p AR order
     * @return AR coefficients, noise variance, reflection coefficients, AIC per order
     */
    public static YuleWalkerResult solveYuleWalkerDetailed(double[] r, int p) {
        if (r.length < p + 1) {
            throw new IllegalArgumentException("Need at least " + (p + 1) + " autocorrelation values, got " + r.length);
        }
        double[] a = new double[p + 1];
        a[0] = 1.0;
        double[] aNew = new double[p + 1];
        double[] refl = new double[p];
        double[] aic = new double[p + 1];
        double E = r[0];
        aic[0] = Double.POSITIVE_INFINITY;

        for (int k = 1; k <= p; k++) {
            double lambda = 0;
            for (int j = 0; j < k; j++) {
                lambda += a[j] * r[k - j];
            }
            lambda = -lambda / E;

            aNew[0] = 1.0;
            for (int j = 1; j < k; j++) {
                aNew[j] = a[j] + lambda * a[k - j];
            }
            aNew[k] = lambda;

            double[] tmp = a;
            a = aNew;
            aNew = tmp;

            refl[k - 1] = lambda;
            E *= (1.0 - lambda * lambda);
            if (E < 1e-30) E = 1e-30;
            aic[k] = Math.log(E) + 2.0 * k / (r.length - 1);
        }

        double[] ar = new double[p];
        System.arraycopy(a, 1, ar, 0, p);
        return new YuleWalkerResult(Linalg.vector(ar), E, Linalg.vector(refl), aic);
    }

    /**
     * Solve symmetric Toeplitz system T x = b using Durbin recursion in O(n²).
     * T[i][j] = column[|i-j|].
     */
    public static IVector<Double> solve(double[] column, double[] b) {
        int n = column.length;
        if (b.length != n) {
            throw new IllegalArgumentException("Dimension mismatch");
        }

        double[] a = new double[n];
        double[] aNext = new double[n];
        a[0] = 1.0;
        double E = column[0];

        double[] x = new double[n];
        x[0] = b[0] / column[0];

        for (int m = 0; m < n - 1; m++) {
            double q = column[m + 1];
            for (int i = 1; i <= m; i++) {
                q += column[m + 1 - i] * a[i];
            }

            double kappa = -q / E;
            E *= (1.0 - kappa * kappa);
            if (E < 1e-30) E = 1e-30;

            aNext[0] = 1.0;
            for (int i = 1; i <= m; i++) {
                aNext[i] = a[i] + kappa * a[m + 1 - i];
            }
            aNext[m + 1] = kappa;

            double[] tmp = a; a = aNext; aNext = tmp;

            double v = b[m + 1];
            for (int i = 0; i <= m; i++) {
                v -= column[m + 1 - i] * x[i];
            }
            double w = v / E;

            double[] xNew = new double[m + 2];
            for (int i = 0; i <= m; i++) {
                xNew[i] = x[i] + w * a[m + 1 - i];
            }
            xNew[m + 1] = w;
            x = xNew;
        }

        return Linalg.vector(x);
    }

    /**
     * Solve symmetric Toeplitz system T x = b.
     */
    public static IVector<Double> solve(IMatrix<Double> T, IVector<Double> b) {
        int n = T.rows();
        if (T.cols() != n || b.length() != n) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        double[] column = new double[n];
        for (int i = 0; i < n; i++) {
            column[i] = T.get(i, 0);
        }
        double[] bArr = new double[n];
        for (int i = 0; i < n; i++) {
            bArr[i] = b.get(i);
        }
        return solve(column, bArr);
    }
}
