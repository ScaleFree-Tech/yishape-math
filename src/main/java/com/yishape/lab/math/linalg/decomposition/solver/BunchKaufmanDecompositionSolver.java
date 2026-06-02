package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.impl.BunchKaufmanLdltLower;

/**
 * 对 Bunch–Kaufman L·D·Lᵀ 分解（下三角）的线性方程组求解器。
 */
public class BunchKaufmanDecompositionSolver implements IDecompositionSolver {

    private final double[][] ldlt;
    private final int[] ipiv;
    private final int n;
    private final boolean usable;

    public BunchKaufmanDecompositionSolver(double[][] ldlt, int[] ipiv, int n, boolean usable) {
        this.ldlt = ldlt;
        this.ipiv = ipiv;
        this.n = n;
        this.usable = usable;
    }

    @Override
    public boolean isNonSingular() {
        return usable;
    }

    @Override
    public IVector<Double> solve(IVector<Double> b) {
        if (b.size() != n) {
            throw new IllegalArgumentException("Vector dimension mismatch: expected " + n + ", got " + b.size());
        }
        if (!usable) {
            throw new RuntimeException("Matrix is singular");
        }
        IMatrix<Double> bMat = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        return solve(bMat).getColumn(0);
    }

    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        if (b.rows() != n) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        if (!usable) {
            throw new RuntimeException("Matrix is singular");
        }
        int nrhs = b.cols();
        double[][] work = new double[n][nrhs];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < nrhs; j++) {
                work[i][j] = b.get(i, j);
            }
        }
        BunchKaufmanLdltLower.dsytrsLower(ldlt, n, ipiv, work, nrhs);
        return Linalg.matrix(work);
    }

    @Override
    public IMatrix<Double> getInverse() {
        return solve(Linalg.eye(n));
    }
}
