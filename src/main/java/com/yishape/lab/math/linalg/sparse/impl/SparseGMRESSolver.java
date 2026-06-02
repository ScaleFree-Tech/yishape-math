package com.yishape.lab.math.linalg.sparse.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.sparse.ISparseLinearSolver;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.linalg.sparse.ISparsePreconditioner;

public class SparseGMRESSolver implements ISparseLinearSolver {

    private final double tol;
    private final int maxIter;
    private final int restart;
    private final ISparsePreconditioner precond;
    private int iterationCount;
    private double residual;
    private double[] residualHistory;

    public SparseGMRESSolver(double tol, int maxIter, int restart, ISparsePreconditioner precond) {
        this.tol = tol;
        this.maxIter = maxIter;
        this.restart = restart;
        this.precond = precond;
    }

    public SparseGMRESSolver(double tol, int maxIter, int restart) {
        this(tol, maxIter, restart, null);
    }

    public SparseGMRESSolver() {
        this(1e-8, 1000, 30, null);
    }

    public double[] getResidualHistory() {
        return residualHistory != null ? residualHistory.clone() : new double[0];
    }

    @Override
    public IVector<Double> solve(ISparseMatrix A, IVector<Double> b) {
        int n = A.rows();
        return solve(A, b, Linalg.vector(new double[n]));
    }

    @Override
    public IVector<Double> solve(ISparseMatrix A, IVector<Double> b, IVector<Double> x0) {
        int n = A.rows();
        if (n != A.cols() || n != b.length() || n != x0.length()) {
            throw new IllegalArgumentException("Dimension mismatch: A " + A.rows() + "x" + A.cols()
                    + ", b " + b.length() + ", x0 " + x0.length());
        }

        int m = Math.min(restart, n);
        double[][] V = new double[m + 1][n];
        double[][] H = new double[m + 1][m];
        double[] cs = new double[m];
        double[] sn = new double[m];
        double[] g = new double[m + 1];
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = x0.get(i);

        double[] bArr = new double[n];
        for (int i = 0; i < n; i++) bArr[i] = b.get(i);

        int totalIter = 0;
        java.util.ArrayList<Double> resHistory = new java.util.ArrayList<>();

        while (totalIter < maxIter) {
            double[] r = computeResidual(A, x, bArr);
            if (precond != null) {
                r = toArray(precond.apply(Linalg.vector(r)));
            }

            double beta = norm2(r);
            resHistory.add(beta);
            if (beta < tol) {
                iterationCount = totalIter;
                residual = beta;
                residualHistory = toArray(resHistory);
                return Linalg.vector(x);
            }

            for (int i = 0; i < n; i++) V[0][i] = r[i] / beta;
            g[0] = beta;
            for (int i = 1; i <= m; i++) g[i] = 0;

            int krylovDim = 0;
            boolean converged = false;

            for (int j = 0; j < m && totalIter + j < maxIter; j++) {
                // w = A * v_j
                double[] vj = V[j];
                double[] w;
                if (precond != null) {
                    IVector<Double> Av = A.multiply(Linalg.vector(vj));
                    w = toArray(precond.apply(Av));
                } else {
                    w = toArray(A.multiply(Linalg.vector(vj)));
                }

                // 2-pass Modified Gram-Schmidt
                for (int i = 0; i <= j; i++) {
                    H[i][j] = dot(w, V[i]);
                }
                for (int i = 0; i <= j; i++) {
                    double hij = H[i][j];
                    for (int k = 0; k < n; k++) {
                        w[k] -= hij * V[i][k];
                    }
                }

                double hNew = norm2(w);
                H[j + 1][j] = hNew;

                // Apply previous Givens rotations to column j
                for (int i = 0; i < j; i++) {
                    double h1 = H[i][j];
                    double h2 = H[i + 1][j];
                    H[i][j] = cs[i] * h1 + sn[i] * h2;
                    H[i + 1][j] = -sn[i] * h1 + cs[i] * h2;
                }

                // Compute new Givens rotation
                double hjj = H[j][j];
                double hj1j = H[j + 1][j];
                double rho = Math.sqrt(hjj * hjj + hj1j * hj1j);
                if (rho < 1e-30) {
                    // Happy breakdown
                    H[j][j] = hjj;
                    H[j + 1][j] = 0;
                    cs[j] = 1;
                    sn[j] = 0;
                } else {
                    cs[j] = hjj / rho;
                    sn[j] = hj1j / rho;
                    H[j][j] = rho;
                    H[j + 1][j] = 0;
                }

                // Apply rotation to g
                double gj = g[j];
                double gj1 = g[j + 1];
                g[j] = cs[j] * gj + sn[j] * gj1;
                g[j + 1] = -sn[j] * gj + cs[j] * gj1;

                krylovDim = j + 1;
                if (Math.abs(g[j + 1]) < tol) {
                    converged = true;
                    break;
                }

                if (hNew < 1e-30) break;

                for (int i = 0; i < n; i++) V[j + 1][i] = w[i] / hNew;
            }

            // Solve H(1:k, 1:k) y = g(1:k) via back substitution (H is upper triangular from Givens)
            double[] y = new double[krylovDim];
            for (int i = krylovDim - 1; i >= 0; i--) {
                double sum = g[i];
                for (int j = i + 1; j < krylovDim; j++) {
                    sum -= H[i][j] * y[j];
                }
                y[i] = sum / H[i][i];
            }

            // x = x0 + V * y
            for (int i = 0; i < krylovDim; i++) {
                double yi = y[i];
                double[] vi = V[i];
                for (int k = 0; k < n; k++) {
                    x[k] += yi * vi[k];
                }
            }

            totalIter += krylovDim;
            double finalRes = Math.abs(g[krylovDim]);
            if (converged || finalRes < tol) {
                iterationCount = totalIter;
                residual = finalRes;
                residualHistory = toArray(resHistory);
                return Linalg.vector(x);
            }
        }

        iterationCount = totalIter;
        double[] rFinal = computeResidual(A, x, bArr);
        residual = norm2(rFinal);
        residualHistory = toArray(resHistory);
        return Linalg.vector(x);
    }

    private double[] computeResidual(ISparseMatrix A, double[] x, double[] b) {
        int n = x.length;
        IVector<Double> Ax = A.multiply(Linalg.vector(x));
        double[] r = new double[n];
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - Ax.get(i);
        }
        return r;
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static double norm2(double[] v) {
        return Math.sqrt(dot(v, v));
    }

    private static double[] toArray(IVector<Double> v) {
        int n = v.length();
        double[] arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = v.get(i);
        return arr;
    }

    private static double[] toArray(java.util.ArrayList<Double> list) {
        double[] arr = new double[list.size()];
        for (int i = 0; i < list.size(); i++) arr[i] = list.get(i);
        return arr;
    }

    @Override
    public int getIterationCount() {
        return iterationCount;
    }

    @Override
    public double getResidual() {
        return residual;
    }
}
