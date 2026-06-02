package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.solver.DecompositionRhsCopy;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;

/**
 * xGELSY 风格：在已得到的列主元 QR（紧凑存储 {@code qrt}/{@code rDiag}）上，按 {@code rankTolerance}
 * 截断 {@code R} 的主元，对 {@code Q^T B} 做容忍 Householder 应用，再对 leading {@code rank×rank} 块回代，
 * 自由变量置零，得到极小 Frobenius 范数意义下的最小二乘解之一；最后按列主元还原 {@code X = P·Z}。
 */
public class QrcpLeastSquaresTolerantSolver implements IDecompositionSolver {

    private final double[][] qrt;
    private final double[] rDiag;
    private final int[] columnPivot;
    private final double rankTolerance;
    private final double reflectorGuard;
    private final int n;
    private final int m;

    public QrcpLeastSquaresTolerantSolver(
            double[][] qrt,
            double[] rDiag,
            int[] columnPivot,
            double rankTolerance,
            double reflectorThreshold) {
        this.qrt = qrt;
        this.rDiag = rDiag;
        this.columnPivot = columnPivot;
        this.rankTolerance = rankTolerance;
        this.n = qrt.length;
        this.m = qrt[0].length;
        this.reflectorGuard = Math.max(1e-15 * m, reflectorThreshold);
    }

    @Override
    public boolean isNonSingular() {
        return effectiveRank() == n && m >= n;
    }

    private int effectiveRank() {
        int lim = Math.min(m, n);
        int r = 0;
        for (int i = 0; i < lim; i++) {
            if (Math.abs(rDiag[i]) > rankTolerance) {
                r++;
            } else {
                break;
            }
        }
        return r;
    }

    @Override
    public IVector<Double> solve(IVector<Double> b) {
        double[] y = b.toDoubleArray();
        if (y.length != m) {
            throw new IllegalArgumentException("Vector dimension mismatch: expected " + m + ", got " + y.length);
        }
        IMatrix<Double> bMat = Linalg.matrix(new double[][]{y}).transpose();
        return solve(bMat).getColumn(0);
    }

    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        if (b.rows() != m) {
            throw new IllegalArgumentException("Matrix row dimension mismatch: expected " + m + ", got " + b.rows());
        }
        int nrhs = b.cols();
        double[][] bData = DecompositionRhsCopy.mutableRowMajorCopy(b);
        applyQt(bData, nrhs);
        int rank = effectiveRank();
        double[][] z = new double[n][nrhs];
        triangularSolveRankTruncated(bData, z, rank, nrhs);
        return permuteSolutionToNatural(z, nrhs);
    }

    private void applyQt(double[][] bData, int nrhs) {
        final int lim = Math.min(m, n);
        for (int minor = 0; minor < lim; minor++) {
            final double[] qrtMinor = qrt[minor];
            double denom = rDiag[minor] * qrtMinor[minor];
            if (Math.abs(denom) <= reflectorGuard) {
                continue;
            }
            for (int col = 0; col < nrhs; col++) {
                double dot = 0.0;
                for (int row = minor; row < m; row++) {
                    dot += bData[row][col] * qrtMinor[row];
                }
                dot /= denom;
                for (int row = minor; row < m; row++) {
                    bData[row][col] += dot * qrtMinor[row];
                }
            }
        }
    }

    /**
     * 前 {@code rank} 行上三角回代：与 {@link com.yishape.lab.math.linalg.decomposition.solver.QRDecompositionSolver}
     * 同序，仅在列 {@code rank..n-1} 上解为 0。
     */
    private void triangularSolveRankTruncated(double[][] bData, double[][] z, int rank, int nrhs) {
        if (rank == 0) {
            return;
        }
        for (int row = rank - 1; row >= 0; --row) {
            if (Math.abs(rDiag[row]) <= rankTolerance) {
                for (int col = 0; col < nrhs; col++) {
                    z[row][col] = 0.0;
                }
                continue;
            }
            for (int col = 0; col < nrhs; col++) {
                double acc = bData[row][col];
                for (int j = row + 1; j < rank; j++) {
                    acc -= qrt[j][row] * z[j][col];
                }
                double xi = acc / rDiag[row];
                z[row][col] = xi;
            }
        }
    }

    private IMatrix<Double> permuteSolutionToNatural(double[][] z, int nrhs) {
        double[][] x = new double[n][nrhs];
        for (int i = 0; i < n; i++) {
            for (int col = 0; col < nrhs; col++) {
                x[columnPivot[i]][col] = z[i][col];
            }
        }
        return Linalg.matrix(x);
    }

    @Override
    public IMatrix<Double> getInverse() {
        if (m != n || effectiveRank() != n) {
            throw new ArithmeticException("Inverse only defined for full-rank square R; use SVD for rank-deficient case");
        }
        return solve(Linalg.eye(n));
    }
}
