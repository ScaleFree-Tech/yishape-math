package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IDoubleMatrix;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;

/**
 * Solver for LU decomposition algorithms with partial pivoting.
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class LUDecompositionSolver implements IDecompositionSolver {

    /**
     * Entries of LU decomposition.
     */
    private final double[][] lu;
    /**
     * Pivot permutation associated with LU decomposition.
     */
    private final int[] pivot;
    /**
     * Singularity indicator.
     */
    private final boolean singular;
    /**
     * Parity of the permutation.
     */
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

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isNonSingular() {
        return !singular;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public IVector<Double> solve(IVector<Double> b) {
        final int m = pivot.length;
        if (b.size() != m) {
            throw new IllegalArgumentException("Vector dimension mismatch: expected " + m + ", got " + b.size());
        }
        if (singular) {
            throw new RuntimeException("Matrix is singular");
        }

        // Convert IVector to column matrix and delegate to matrix solve method
        IMatrix<Double> bMatrix = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        IMatrix<Double> solution = solve(bMatrix);

        // Convert back to vector
        return solution.getColumn(0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public IMatrix<Double> solve(IMatrix<Double> b) {
        final int m = pivot.length;
        final int nColB = b.cols();
        if (b.rows() != m) {
            throw new IllegalArgumentException("Dimension mismatch");
        }
        if (singular) {
            throw new RuntimeException("Matrix is singular");
        }

        // Convert b to array for easier manipulation
        double[][] bData = new double[m][nColB];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < nColB; j++) {
                bData[i][j] = b.get(i, j);
            }
        }

        // For LU decomposition with pivoting: P*A = L*U
        // To solve A*x = b, we have P*A*x = P*b => L*U*x = P*b
        // So we solve: 1. L*y = P*b, 2. U*x = y
        
        // 1. Apply permutation P to b (Pb)
        // pivot[i] = j means original row j is now at position i
        // We need perm[orig_row] = current_row to apply P*b correctly
        // where (P*b)[current_row] = b[orig_row]
        // Apply permutations to b
        double[][] bp = new double[m][nColB];
        for (int row = 0; row < m; row++) {
            final double[] bpRow = bp[row];
            final int pRow = pivot[row];
            for (int col = 0; col < nColB; col++) {
                bpRow[col] = bData[pRow][col];
            }
        }

        // 2. Forward substitution: Solve L*Y = P*b
        // L is unit lower triangular (diagonal is 1's).
        // Operate directly on bp[][] – RereDoubleMatrix.getRow() now returns
        // a clone (immutable pattern), so bpMat.getRow().axpy() would miss.
        for (int col = 0; col < m; col++) {
            final double[] bpCol = bp[col];           // y[col] = Pb[col] (L[col][col]=1)
            for (int i = col + 1; i < m; i++) {
                final double luICol = lu[i][col];
                if (luICol != 0.0) {
                    final double[] bpRow = bp[i];
                    for (int j = 0; j < nColB; j++) {
                        bpRow[j] -= luICol * bpCol[j];
                    }
                }
            }
        }

        // 3. Backward substitution: Solve U*X = Y
        for (int col = m - 1; col >= 0; col--) {
            final double[] bpCol = bp[col];
            final double uDiag = lu[col][col];
            if (Math.abs(uDiag) < 1e-15) {
                throw new RuntimeException("Numerically singular U diagonal at column " + col);
            }
            final double invDiag = 1.0 / uDiag;
            for (int j = 0; j < nColB; j++) {
                bpCol[j] *= invDiag;
            }
            for (int i = 0; i < col; i++) {
                final double luICol = lu[i][col];
                if (luICol != 0.0) {
                    final double[] bpRow = bp[i];
                    for (int j = 0; j < nColB; j++) {
                        bpRow[j] -= luICol * bpCol[j];
                    }
                }
            }
        }

        return Linalg.matrix(bp);
    }

    /**
     * {@inheritDoc}
     *
     * @throws RuntimeException if the decomposed matrix is singular.
     */
    @Override
    public IMatrix<Double> getInverse() {
        return solve(Linalg.eye(pivot.length));
    }
}
