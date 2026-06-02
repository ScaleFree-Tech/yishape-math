package com.yishape.lab.math.linalg.sparse.impl;

import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.sparse.ISparseLinearSolver;
import com.yishape.lab.math.linalg.sparse.ISparseMatrix;
import com.yishape.lab.math.linalg.sparse.ISparsePreconditioner;

public class SparseConjugateGradientSolver implements ISparseLinearSolver {

    private final double tol;
    private final int maxIter;
    private final ISparsePreconditioner precond;
    private int iterationCount;
    private double residual;

    public SparseConjugateGradientSolver(double tol, int maxIter, ISparsePreconditioner precond) {
        this.tol = tol;
        this.maxIter = maxIter;
        this.precond = precond;
    }

    public SparseConjugateGradientSolver(double tol, int maxIter, boolean usePreconditioner) {
        this(tol, maxIter, usePreconditioner ? null : null);
    }

    public SparseConjugateGradientSolver() {
        this(1e-8, 10000, (ISparsePreconditioner) null);
    }

    public SparseConjugateGradientSolver(double tol, int maxIter) {
        this(tol, maxIter, (ISparsePreconditioner) null);
    }

    @Override
    public IVector<Double> solve(ISparseMatrix A, IVector<Double> b) {
        int n = A.rows();
        double[] x0 = new double[n];
        return solve(A, b, Linalg.vector(x0));
    }

    @Override
    public IVector<Double> solve(ISparseMatrix A, IVector<Double> b, IVector<Double> x0) {
        int n = A.rows();
        if (n != A.cols() || n != b.length() || n != x0.length()) {
            throw new IllegalArgumentException("Dimension mismatch: A " + A.rows() + "x" + A.cols() + ", b " + b.length() + ", x0 " + x0.length());
        }

        double[] x = new double[n];
        double[] r = new double[n];
        double[] p = new double[n];
        double[] z = new double[n];

        for (int i = 0; i < n; i++) {
            x[i] = x0.get(i);
        }

        IVector<Double> Ax = A.multiply(Linalg.vector(x));
        double[] AxArr = toArray(Ax);
        for (int i = 0; i < n; i++) {
            r[i] = b.get(i) - AxArr[i];
        }

        boolean hasPrecond = (precond != null);
        if (hasPrecond) {
            double[] zArr = toArray(precond.apply(Linalg.vector(r)));
            System.arraycopy(zArr, 0, z, 0, n);
            System.arraycopy(z, 0, p, 0, n);
        } else {
            System.arraycopy(r, 0, p, 0, n);
        }

        double rsOld = hasPrecond ? dot(r, z) : dot(r, r);

        residual = Math.sqrt(dot(r, r));
        if (residual < tol) {
            iterationCount = 0;
            return Linalg.vector(x);
        }

        for (int iter = 0; iter < maxIter; iter++) {
            IVector<Double> Ap = A.multiply(Linalg.vector(p));
            double[] ApArr = toArray(Ap);

            double pAp = dot(p, ApArr);
            if (Math.abs(pAp) < 1e-30) {
                iterationCount = iter + 1;
                residual = Math.sqrt(dot(r, r));
                return Linalg.vector(x);
            }

            double alpha = rsOld / pAp;
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * ApArr[i];
            }

            residual = Math.sqrt(dot(r, r));
            if (residual < tol) {
                iterationCount = iter + 1;
                return Linalg.vector(x);
            }

            double rsNew;
            if (hasPrecond) {
                double[] zNew = toArray(precond.apply(Linalg.vector(r)));
                System.arraycopy(zNew, 0, z, 0, n);
                rsNew = dot(r, z);
            } else {
                rsNew = dot(r, r);
            }

            double beta = rsNew / rsOld;
            for (int i = 0; i < n; i++) {
                p[i] = (hasPrecond ? z[i] : r[i]) + beta * p[i];
            }
            rsOld = rsNew;
        }

        iterationCount = maxIter;
        return Linalg.vector(x);
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static double[] toArray(IVector<Double> v) {
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
