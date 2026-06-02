package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for Cholesky decomposition algorithms.
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class CholeskyDecompositionSolver implements IDecompositionSolver {
    /** Cached lower triangular matrix L. */
    private final IMatrix<Double> cachedL;
    /** Cached upper triangular matrix LT. */
    private final IMatrix<Double> cachedLT;

    /**
     * Build a solver from decomposed matrix.
     *
     * @param lTData row-oriented storage for LT matrix data
     */
    public CholeskyDecompositionSolver(final double[][] lTData) {
        final int m = lTData.length;
        // Build L from LT: L[i][j] = LT[j][i]
        double[][] lValues = new double[m][];
        for (int i = 0; i < m; i++) {
            lValues[i] = new double[i + 1];
            for (int j = 0; j <= i; j++) {
                lValues[i][j] = lTData[j][i];
            }
        }
        this.cachedL = Linalg.lowerTriMatrix(m, lValues);
        this.cachedLT = Linalg.upperTriMatrix(m, lTData);
    }

    /** {@inheritDoc} */
    @Override
    public boolean isNonSingular() {
        // if we get this far, the matrix was positive definite, hence non-singular
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public IVector<Double> solve(IVector<Double> b) {
        final int m = cachedL.rows();
        if (b.size() != m) {
            throw new IllegalArgumentException("Vector dimension mismatch: expected " + m + ", got " + b.size());
        }

        // Convert IVector to matrix form for solving
        IMatrix<Double> bMatrix = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose(); // Convert to column vector

        // Solve using forward and backward substitution
        IMatrix<Double> solution = solve(bMatrix);

        // Convert back to vector
        return solution.getColumn(0);
    }

    /** {@inheritDoc} */
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        final int m = cachedL.rows();
        if (b.rows() != m) {
            throw new IllegalArgumentException("Matrix row dimension mismatch: expected " + m + ", got " + b.rows());
        }

        // Solve LY = b (forward substitution)
        IMatrix<Double> y = Linalg.forwardSolve(cachedL, b);

        // Solve LTX = Y (backward substitution)
        IMatrix<Double> x = Linalg.backwardSolve(cachedLT, y);

        return x;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public IMatrix<Double> getInverse() {
        return solve(Linalg.eye(cachedL.rows()));
    }
}