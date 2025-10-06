package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for Cholesky decomposition algorithms.
 */
public class CholeskyDecompositionSolver implements IDecompositionSolver {
    /** Entries of LT decomposition. */
    private final double[][] lTData;

    /**
     * Build a solver from decomposed matrix.
     *
     * @param lTData row-oriented storage for LT matrix data
     */
    public CholeskyDecompositionSolver(final double[][] lTData) {
        this.lTData = lTData;
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
        final int m = lTData.length;
        if (b.size() != m) {
            throw new IllegalArgumentException("Vector dimension mismatch: expected " + m + ", got " + b.size());
        }

        // Convert IVector to matrix form for solving
        IMatrix<Double> bMatrix = Linalg.matrix(new double[][]{b.toDoubleArray()}).t(); // Convert to column vector

        // Solve using forward and backward substitution
        IMatrix<Double> solution = solve(bMatrix);
        
        // Convert back to vector
        return solution.getColumn(0);
    }

    /** {@inheritDoc} */
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        final int m = lTData.length;
        if (b.rows() != m) {
            throw new IllegalArgumentException("Matrix row dimension mismatch: expected " + m + ", got " + b.rows());
        }

        // Create L and LT matrices from the Cholesky decomposition data
        // L is lower triangular
        // LT is upper triangular (transpose of L)
        double[][] lValues = new double[m][];
        
        for (int i = 0; i < m; i++) {
            lValues[i] = new double[i + 1];
            
            for (int j = 0; j <= i; j++) {
                lValues[i][j] = lTData[j][i]; // Transpose of LT gives us L
            }
        }
        
        // Create L and LT matrices
        IMatrix<Double> L = Linalg.lowerTriMatrix(m, lValues);
        IMatrix<Double> LT = Linalg.upperTriMatrix(m, lTData);

        // Solve LY = b (forward substitution)
        IMatrix<Double> y = Linalg.forwardSolve(L, b);

        // Solve LTX = Y (backward substitution)
        IMatrix<Double> x = Linalg.backwardSolve(LT, y);

        return x;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public IMatrix<Double> getInverse() {
        return solve(Linalg.eye(lTData.length));
    }
}