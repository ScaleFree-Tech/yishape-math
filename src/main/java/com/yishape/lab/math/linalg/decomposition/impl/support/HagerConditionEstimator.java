package com.yishape.lab.math.linalg.decomposition.impl.support;

/**
 * Hager-Higham condition number estimator (LAPACK xLACON-style).
 * Estimates ||A||_1 and ||A^{-1}||_1 in O(n^2) via power iteration,
 * without computing the inverse explicitly.
 *
 * <p>Algorithm: Hager 1984, improved by Higham 1988 (Algorithm 4.1 in
 * Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed.).</p>
 */
public final class HagerConditionEstimator {

    private static final int MAX_ITER = 5;

    private HagerConditionEstimator() {}

    /**
     * Estimate the 1-norm condition number of a square matrix.
     *
     * @param A square double[][] data (dense row-major)
     * @return estimated kappa_1(A), or Double.POSITIVE_INFINITY if singular
     */
    public static double estimateCondition(double[][] A) {
        int n = A.length;
        double normA = norm1(A);
        if (normA == 0.0) {
            return Double.POSITIVE_INFINITY;
        }
        double normAInv = estimateInverseNorm1(A);
        if (normAInv == 0.0 || Double.isInfinite(normAInv)) {
            return Double.POSITIVE_INFINITY;
        }
        return normA * normAInv;
    }

    /**
     * Estimate ||A^{-1}||_1 using power iteration.
     * Requires solving A*x = b and A^T*x = b at each iteration.
     */
    public static double estimateInverseNorm1(double[][] A) {
        int n = A.length;
        return estimateInverseNorm1(n, (b, transpose) -> {
            double[] x = new double[n];
            if (transpose) {
                solveTranspose(A, b, x);
            } else {
                solveNormal(A, b, x);
            }
            return x;
        });
    }

    /**
     * General form: caller provides solve(A, b, transpose) -&gt; x.
     */
    public static double estimateInverseNorm1(int n,
            java.util.function.BiFunction<double[], Boolean, double[]> solve) {
        double[] x = new double[n];
        double[] y = new double[n];
        double[] z = new double[n];

        // Initial vector: all 1/n
        for (int i = 0; i < n; i++) {
            x[i] = 1.0 / n;
        }

        double estOld = 0.0;
        double est = 0.0;
        int ind = 0;
        boolean indSet = false;

        for (int iter = 0; iter < MAX_ITER; iter++) {
            // y = A^{-1} * x (solve A*y = x)
            double[] yTmp = solve.apply(x, false);
            System.arraycopy(yTmp, 0, y, 0, n);

            // Estimate: ||y||_1
            est = 0.0;
            for (int i = 0; i < n; i++) {
                est += Math.abs(y[i]);
            }

            // Check convergence
            if (iter > 0 && est <= estOld) {
                break;
            }
            estOld = est;

            // Sign pattern: z = sign(y)
            for (int i = 0; i < n; i++) {
                z[i] = (y[i] >= 0) ? 1.0 : -1.0;
            }

            // x = (A^{-1})^T * z (solve A^T * x = z)
            double[] xTmp = solve.apply(z, true);
            System.arraycopy(xTmp, 0, x, 0, n);

            // Find max |x_i|
            double xMax = 0.0;
            indSet = false;
            for (int i = 0; i < n; i++) {
                double absXi = Math.abs(x[i]);
                if (absXi > xMax) {
                    xMax = absXi;
                    ind = i;
                    indSet = true;
                }
            }

            if (!indSet || xMax == 0.0) {
                break;
            }

            // x = e_ind (unit vector at max position)
            for (int i = 0; i < n; i++) {
                x[i] = (i == ind) ? 1.0 : 0.0;
            }
        }

        return est;
    }

    /** Compute ||A||_1 (max column sum). */
    public static double norm1(double[][] A) {
        int n = A.length;
        double maxColSum = 0.0;
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += Math.abs(A[i][j]);
            }
            maxColSum = Math.max(maxColSum, sum);
        }
        return maxColSum;
    }

    /** Solve A*x = b using simple LU decomposition (internal). */
    private static void solveNormal(double[][] A, double[] b, double[] x) {
        int n = A.length;
        double[][] lu = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, lu[i], 0, n);
        }
        int[] pivot = new int[n];
        luDecompose(lu, pivot, n);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = b[pivot[i]];
        }
        luSolve(lu, pivot, y, x, n);
    }

    /** Solve A^T*x = b using simple LU decomposition (internal). */
    private static void solveTranspose(double[][] A, double[] b, double[] x) {
        int n = A.length;
        double[][] lu = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, lu[i], 0, n);
        }
        int[] pivot = new int[n];
        luDecompose(lu, pivot, n);
        luSolveTranspose(lu, pivot, b, x, n);
    }

    /** Simple LU decomposition with partial pivoting (in-place on A). */
    private static void luDecompose(double[][] A, int[] pivot, int n) {
        for (int i = 0; i < n; i++) {
            pivot[i] = i;
        }
        for (int k = 0; k < n; k++) {
            // Find pivot
            int maxRow = k;
            double maxVal = Math.abs(A[k][k]);
            for (int i = k + 1; i < n; i++) {
                double v = Math.abs(A[i][k]);
                if (v > maxVal) {
                    maxVal = v;
                    maxRow = i;
                }
            }
            if (maxVal == 0.0) {
                A[k][k] = 1e-30; // regularization for singular matrices
            }
            // Swap rows
            if (maxRow != k) {
                double[] tmp = A[k];
                A[k] = A[maxRow];
                A[maxRow] = tmp;
                int t = pivot[k];
                pivot[k] = pivot[maxRow];
                pivot[maxRow] = t;
            }
            // Eliminate
            for (int i = k + 1; i < n; i++) {
                A[i][k] /= A[k][k];
                for (int j = k + 1; j < n; j++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
            }
        }
    }

    /** Forward/back substitution for L*U*x = P*b. */
    private static void luSolve(double[][] lu, int[] pivot, double[] b, double[] x, int n) {
        // Forward: L*y = b
        for (int i = 0; i < n; i++) {
            double sum = b[i];
            for (int j = 0; j < i; j++) {
                sum -= lu[i][j] * x[j];
            }
            x[i] = sum;
        }
        // Backward: U*x = y (in-place on x)
        for (int i = n - 1; i >= 0; i--) {
            double sum = x[i];
            for (int j = i + 1; j < n; j++) {
                sum -= lu[i][j] * x[j];
            }
            x[i] = sum / lu[i][i];
        }
    }

    /** Solve L^T*U^T*x = P*b (transposed system). */
    private static void luSolveTranspose(double[][] lu, int[] pivot, double[] b, double[] x, int n) {
        // U^T * y = P*b: forward with U^T
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = b[pivot[i]];
            for (int j = 0; j < i; j++) {
                sum -= lu[j][i] * y[j];
            }
            y[i] = sum / lu[i][i];
        }
        // L^T * x = y: backward with L^T
        for (int i = n - 1; i >= 0; i--) {
            double sum = y[i];
            for (int j = i + 1; j < n; j++) {
                sum -= lu[j][i] * x[j];
            }
            x[i] = sum;
        }
    }
}
