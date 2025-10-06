package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for LU decomposition algorithms with partial pivoting.
 */
public class LUDecompositionSolver implements IDecompositionSolver {
    /** Entries of LU decomposition. */
    private final double[][] lu;
    /** Pivot permutation associated with LU decomposition. */
    private final int[] pivot;
    /** Singularity indicator. */
    private final boolean singular;
    /** Parity of the permutation. */
    private final boolean even;

    /**
     * Build a solver from decomposed matrix.
     *
     * @param lu entries of LU decomposition
     * @param pivot pivot permutation associated with LU decomposition
     * @param singular singularity indicator
     * @param even parity of the permutation
     */
    public LUDecompositionSolver(final double[][] lu, final int[] pivot, final boolean singular, final boolean even) {
        this.lu = lu;
        this.pivot = pivot;
        this.singular = singular;
        this.even = even;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isNonSingular() {
        return !singular;
    }

    /** {@inheritDoc} */
    @Override
    public IVector<Double> solve(IVector<Double> b) {
        final int m = pivot.length;
        if (b.size() != m) {
            throw new IllegalArgumentException("Vector dimension mismatch: expected " + m + ", got " + b.size());
        }
        if (singular) {
            throw new RuntimeException("Matrix is singular");
        }

        // Convert IVector to double array using toDoubleArray method
        double[] bp = b.toDoubleArray();

        // Apply permutations to b using permutation vector
        double[] x = new double[m];
        for (int row = 0; row < m; row++) {
            x[row] = bp[pivot[row]];
        }

        // Convert to matrix form for solving
        IMatrix<Double> bMatrix = Linalg.matrix(new double[][]{x}).t(); // Convert to column vector

        // Solve using forward and backward substitution
        IMatrix<Double> solution = solveVectorAsMatrix(bMatrix);
        
        // Convert back to vector
        return solution.getColumn(0);
    }

    /** {@inheritDoc} */
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        final int m = pivot.length;
        if (b.rows() != m) {
            throw new IllegalArgumentException("Matrix row dimension mismatch: expected " + m + ", got " + b.rows());
        }
        if (singular) {
            throw new RuntimeException("Matrix is singular");
        }

        // Apply permutations to b using the new permuteRows method
        IMatrix<Double> permutedB = Linalg.permuteRows(b, pivot);

        // Extract L and U matrices from the LU decomposition
        // L is unit lower triangular (diagonal elements are 1, stored implicitly)
        // U is upper triangular
        double[][] lValues = new double[m][];
        double[][] uValues = new double[m][];
        
        for (int i = 0; i < m; i++) {
            lValues[i] = new double[i + 1];
            uValues[i] = new double[m];
            
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    lValues[i][j] = 1.0; // Unit diagonal
                } else {
                    lValues[i][j] = lu[i][j];
                }
            }
            
            for (int j = i; j < m; j++) {
                uValues[i][j] = lu[i][j];
            }
        }
        
        // Create L and U matrices
        IMatrix<Double> L = Linalg.unitLowerTriMatrix(m, lValues);
        IMatrix<Double> U = Linalg.upperTriMatrix(m, uValues);

        // Solve LY = Pb (forward substitution)
        IMatrix<Double> y = Linalg.forwardSolve(L, permutedB);

        // Solve UX = Y (backward substitution)
        IMatrix<Double> x = Linalg.backwardSolve(U, y);

        return x;
    }
    
    /**
     * Solve method for vector that doesn't apply permutation (since it's already applied)
     */
    private IMatrix<Double> solveVectorAsMatrix(IMatrix<Double> b) {
        final int m = pivot.length;
        
        // Extract L and U matrices from the LU decomposition
        // L is unit lower triangular (diagonal elements are 1, stored implicitly)
        // U is upper triangular
        double[][] lValues = new double[m][];
        double[][] uValues = new double[m][];
        
        for (int i = 0; i < m; i++) {
            lValues[i] = new double[i + 1];
            uValues[i] = new double[m];
            
            for (int j = 0; j <= i; j++) {
                if (i == j) {
                    lValues[i][j] = 1.0; // Unit diagonal
                } else {
                    lValues[i][j] = lu[i][j];
                }
            }
            
            for (int j = i; j < m; j++) {
                uValues[i][j] = lu[i][j];
            }
        }
        
        // Create L and U matrices
        IMatrix<Double> L = Linalg.unitLowerTriMatrix(m, lValues);
        IMatrix<Double> U = Linalg.upperTriMatrix(m, uValues);

        // Solve LY = b (forward substitution)
        IMatrix<Double> y = Linalg.forwardSolve(L, b);

        // Solve UX = Y (backward substitution)
        IMatrix<Double> x = Linalg.backwardSolve(U, y);

        return x;
    }

    /**
     * {@inheritDoc}
     * @throws RuntimeException if the decomposed matrix is singular.
     */
    @Override
    public IMatrix<Double> getInverse() {
        return solve(Linalg.eye(pivot.length));
    }
}