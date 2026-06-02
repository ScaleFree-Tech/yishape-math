package com.yishape.lab.math.linalg.sparse.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.sparse.ISparseLinearSolver;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.linalg.sparse.ISparsePreconditioner;

public class SparseBICGSTABSolver implements ISparseLinearSolver {

    private final double tol;
    private final int maxIter;
    private final ISparsePreconditioner precond;
    private int iterationCount;
    private double residual;

    public SparseBICGSTABSolver(double tol, int maxIter, ISparsePreconditioner precond) {
        this.tol = tol;
        this.maxIter = maxIter;
        this.precond = precond;
    }

    public SparseBICGSTABSolver(double tol, int maxIter) {
        this(tol, maxIter, null);
    }

    public SparseBICGSTABSolver() {
        this(1e-8, 10000, null);
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
            throw new IllegalArgumentException("Dimension mismatch");
        }

        double[] x = new double[n];
        double[] r = new double[n];
        double[] rHat = new double[n];
        double[] p = new double[n];
        double[] s = new double[n];
        double[] t = new double[n];
        double[] v = new double[n];
        double[] pHat = hasPrecond() ? new double[n] : null;
        double[] sHat = hasPrecond() ? new double[n] : null;

        for (int i = 0; i < n; i++) {
            x[i] = x0.get(i);
        }

        IVector<Double> Ax = A.multiply(Linalg.vector(x));
        double[] AxArr = toArrayDense(Ax);
        for (int i = 0; i < n; i++) {
            r[i] = b.get(i) - AxArr[i];
        }

        System.arraycopy(r, 0, rHat, 0, n);
        System.arraycopy(r, 0, p, 0, n);

        double rhoOld = 1.0;
        double alpha = 1.0;
        double omega = 1.0;

        residual = Math.sqrt(dot(r, r));
        if (residual < tol) {
            iterationCount = 0;
            return Linalg.vector(x);
        }

        for (int iter = 0; iter < maxIter; iter++) {
            double rho = dot(rHat, r);
            if (Math.abs(rho) < 1e-30) {
                iterationCount = iter + 1;
                residual = Math.sqrt(dot(r, r));
                return Linalg.vector(x);
            }

            if (iter == 0) {
                for (int i = 0; i < n; i++) p[i] = r[i];
            } else {
                double beta = (rho / rhoOld) * (alpha / omega);
                for (int i = 0; i < n; i++) {
                    p[i] = r[i] + beta * (p[i] - omega * v[i]);
                }
            }

            double[] pForAv = p;
            if (hasPrecond()) {
                double[] pPre = toArrayDense(precond.apply(Linalg.vector(p)));
                System.arraycopy(pPre, 0, pHat, 0, n);
                pForAv = pHat;
            }

            IVector<Double> vVec = A.multiply(Linalg.vector(pForAv));
            double[] vArr = toArrayDense(vVec);
            System.arraycopy(vArr, 0, v, 0, n);
            double rHatDotV = dot(rHat, v);

            if (Math.abs(rHatDotV) < 1e-30) {
                iterationCount = iter + 1;
                residual = Math.sqrt(dot(r, r));
                return Linalg.vector(x);
            }

            alpha = rho / rHatDotV;
            for (int i = 0; i < n; i++) {
                s[i] = r[i] - alpha * v[i];
            }

            double[] sForAv = s;
            if (hasPrecond()) {
                double[] sPre = toArrayDense(precond.apply(Linalg.vector(s)));
                System.arraycopy(sPre, 0, sHat, 0, n);
                sForAv = sHat;
            }

            IVector<Double> tVec = A.multiply(Linalg.vector(sForAv));
            double[] tArr = toArrayDense(tVec);
            System.arraycopy(tArr, 0, t, 0, n);

            double tDotT = dot(t, t);
            if (tDotT < 1e-30) {
                omega = 0.0;
            } else {
                omega = dot(t, s) / tDotT;
            }

            for (int i = 0; i < n; i++) {
                x[i] += alpha * pForAv[i] + omega * sForAv[i];
                r[i] = s[i] - omega * t[i];
            }

            rhoOld = rho;
            residual = Math.sqrt(dot(r, r));
            if (residual < tol) {
                iterationCount = iter + 1;
                return Linalg.vector(x);
            }
        }

        iterationCount = maxIter;
        return Linalg.vector(x);
    }

    private boolean hasPrecond() {
        return precond != null;
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static double[] toArrayDense(IVector<Double> v) {
        int n = v.length();
        double[] arr = new double[n];
        for (int i = 0; i < n; i++) {
            arr[i] = v.get(i);
        }
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
