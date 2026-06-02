package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.DecompositionDenseAccess;
import com.yishape.lab.math.linalg.decomposition.IBunchKaufmanDecomposition;
import com.yishape.lab.math.linalg.decomposition.ICholeskyDecomposition;
import com.yishape.lab.math.linalg.decomposition.IMatrixDecomposition;
import com.yishape.lab.math.linalg.decomposition.NonSymmetricMatrixException;
import com.yishape.lab.math.linalg.decomposition.solver.BunchKaufmanDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;

/**
 * 稠密对称不定矩阵：Bunch–Kaufman L·D·Lᵀ（LAPACK DSYTF2/DSYTRS，下三角）。
 */
public class RereBunchKaufmanDecomposition implements IBunchKaufmanDecomposition {

    private double[][] ldlt;
    private int[] ipiv;
    private int info;
    private double symmetryTolerance;
    private double singularityEps;
    private double determinant;
    private boolean decomposed;
    private boolean usable;

    public RereBunchKaufmanDecomposition() {
        this(ICholeskyDecomposition.DEFAULT_RELATIVE_SYMMETRY_THRESHOLD, 1e-12);
    }

    public RereBunchKaufmanDecomposition(double relativeSymmetryTolerance, double singularityEps) {
        this.symmetryTolerance = relativeSymmetryTolerance;
        this.singularityEps = singularityEps;
    }

    @Override
    public IMatrix<Double> decompose(IMatrix<Double> matrix) {
        return decompose(matrix, symmetryTolerance, singularityEps);
    }

    @Override
    public IMatrix<Double> decompose(IMatrix<Double> matrix, double epsilon) {
        return decompose(matrix, epsilon, singularityEps);
    }

    @Override
    public IMatrix<Double> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        return decompose(matrix, epsilon, singularityEps);
    }

    private IMatrix<Double> decompose(IMatrix<Double> matrix, double relativeSymmetryTol, double singularityEpsilon) {
        if (!matrix.isSquare()) {
            throw new IllegalArgumentException("只有方阵才能进行 Bunch–Kaufman 分解");
        }
        this.symmetryTolerance = relativeSymmetryTol;
        this.singularityEps = singularityEpsilon;
        int n = matrix.rows();
        ldlt = new double[n][n];
        DecompositionDenseAccess.copyInto(matrix, ldlt, n, n);

        double maxAsym = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double asym = Math.abs(ldlt[i][j] - ldlt[j][i]);
                maxAsym = Math.max(maxAsym, asym);
                if (asym > relativeSymmetryTol * Math.max(Math.abs(ldlt[i][j]), Math.abs(ldlt[j][i]))) {
                    throw new NonSymmetricMatrixException(
                        "矩阵必须对称（Bunch–Kaufman）",
                        "Bunch–Kaufman分解",
                        "矩阵 " + n + "×" + n,
                        relativeSymmetryTol,
                        maxAsym);
                }
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                ldlt[i][j] = 0.0;
            }
        }

        ipiv = new int[n];
        info = BunchKaufmanLdltLower.dsytf2Lower(ldlt, n, ipiv);
        determinant = BunchKaufmanLdltLower.determinantFromFactor(ldlt, n, ipiv, singularityEpsilon);
        decomposed = true;
        usable = info == 0 && Math.abs(determinant) > singularityEpsilon;
        return getLdltFactor();
    }

    @Override
    public int[] getIpiv() {
        return decomposed && ipiv != null ? ipiv.clone() : null;
    }

    @Override
    public IMatrix<Double> getLdltFactor() {
        if (!decomposed || ldlt == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        int n = ldlt.length;
        double[][] copy = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(ldlt[i], 0, copy[i], 0, n);
        }
        return Linalg.matrix(copy);
    }

    @Override
    public IDecompositionSolver getSolver() {
        if (!decomposed) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        return new BunchKaufmanDecompositionSolver(ldlt, ipiv, ldlt.length, usable);
    }

    @Override
    public double getDeterminant() {
        if (!decomposed) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        return determinant;
    }

    @Override
    public boolean isNonSingular() {
        if (!decomposed) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        return usable;
    }

    @Override
    public double getConditionNumber() {
        if (!decomposed) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        return Double.NaN;
    }

    @Override
    public int getRank() {
        if (!decomposed) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        if (usable) {
            return ldlt.length;
        }
        return rankFromFactor(ldlt, ldlt.length, ipiv, singularityEps);
    }

    private static int rankFromFactor(double[][] a, int n, int[] ipiv, double eps) {
        int r = 0;
        int k = 0;
        while (k < n) {
            if (ipiv[k] >= 0) {
                if (Math.abs(a[k][k]) > eps) {
                    r++;
                }
                k++;
            } else {
                double a11 = a[k][k];
                double a22 = a[k + 1][k + 1];
                double a21 = a[k + 1][k];
                double d2 = a11 * a22 - a21 * a21;
                if (Math.abs(d2) > eps) {
                    r += 2;
                }
                k += 2;
            }
        }
        return r;
    }

    @Override
    public double getEpsilon() {
        return symmetryTolerance;
    }

    @Override
    public int getMaxIterations() {
        return IMatrixDecomposition.DEFAULT_MAX_ITERATIONS;
    }
}
