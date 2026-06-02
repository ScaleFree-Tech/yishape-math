package com.yishape.lab.math.linalg.decomposition.impl;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.RereDoubleMatrix;
import com.yishape.lab.math.linalg.decomposition.ILUDecomposition;
import com.yishape.lab.math.linalg.decomposition.IMatrixDecomposition;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.solver.LUDecompositionSolver;
import com.yishape.lab.math.linalg.decomposition.DecompositionDenseAccess;
import com.yishape.lab.math.linalg.decomposition.SingularMatrixException;
import com.yishape.lab.util.Tuple2;

/**
 * Implementation of LU decomposition with partial pivoting for better numerical stability.
 *
 * @author RereMouse
 * @version 1.0
 * @since 2.0
 */
public class RereLUDecomposition implements ILUDecomposition {
    /** Default bound to determine effective singularity in LU decomposition. */
    private static final double DEFAULT_TOO_SMALL = 1e-11;
    
    /** Entries of LU decomposition. */
    private double[][] lu;
    /** Pivot permutation associated with LU decomposition. */
    private int[] pivot;
    /** Parity of the permutation associated with the LU decomposition. */
    private boolean even;
    /** Singularity indicator. */
    private boolean singular;
    /** Singularity threshold. */
    private final double singularityThreshold;
    /** Cached value of L. */
    private IMatrix<Double> cachedL;
    /** Cached value of U. */
    private IMatrix<Double> cachedU;
    /** Cached value of P. */
    private IMatrix<Double> cachedP;

    /**
     * Calculates the LU-decomposition of the given matrix.
     * This constructor uses 1e-11 as default value for the singularity threshold.
     */
    public RereLUDecomposition() {
        this(DEFAULT_TOO_SMALL);
    }

    /**
     * Calculates the LU-decomposition of the given matrix.
     * @param singularityThreshold threshold (based on partial row norm)
     * under which a matrix is considered singular
     */
    public RereLUDecomposition(double singularityThreshold) {
        this.singularityThreshold = singularityThreshold;
        this.lu = null;
        this.pivot = null;
        this.cachedL = null;
        this.cachedU = null;
        this.cachedP = null;
    }

    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix) {
        if (!matrix.isSquare()) {
            throw new IllegalArgumentException("只有方阵才能进行LU分解 / Only square matrices can perform LU decomposition");
        }

        final int m = matrix.cols();
        // Copy into LU workspace without allocating an intermediate full toDoubleArray() copy for dense matrices
        this.lu = new double[m][m];
        DecompositionDenseAccess.copyInto(matrix, this.lu, m, m);
        this.pivot = new int[m];
        this.cachedL = null;
        this.cachedU = null;
        this.cachedP = null;

        // Initialize permutation array to identity and parity
        // pivot[i] = j means original row j is now in position i after all swaps
        for (int row = 0; row < m; row++) {
            pivot[row] = row;
        }
        even = true;
        singular = false;

        // Loop over columns
        for (int col = 0; col < m; col++) {

            if (m < 256) {
                // Small matrices: direct stride-n access. The column fits in L1 cache
                // (256 doubles = 2KB) so the precopy buffer overhead isn't amortized.
                for (int row = 0; row < col; row++) {
                    final double[] luRow = lu[row];
                    double sum = luRow[col];
                    for (int i = 0; i < row; i++) {
                        sum -= luRow[i] * lu[i][col];
                    }
                    luRow[col] = sum;
                }

                int max = col;
                double largest = Double.NEGATIVE_INFINITY;
                for (int row = col; row < m; row++) {
                    final double[] luRow = lu[row];
                    double sum = luRow[col];
                    for (int i = 0; i < col; i++) {
                        sum -= luRow[i] * lu[i][col];
                    }
                    luRow[col] = sum;
                    if (Math.abs(sum) > largest) {
                        largest = Math.abs(sum);
                        max = row;
                    }
                }

                if (Math.abs(lu[max][col]) < singularityThreshold) {
                    singular = true;
                    throw new SingularMatrixException("LU decomposition failed: matrix is singular at column " + col);
                }

                if (max != col) {
                    double[] tmp = lu[col];
                    lu[col] = lu[max];
                    lu[max] = tmp;
                    int temp = pivot[max];
                    pivot[max] = pivot[col];
                    pivot[col] = temp;
                    even = !even;
                }

                final double luDiag = lu[col][col];
                for (int row = col + 1; row < m; row++) {
                    lu[row][col] /= luDiag;
                }
            } else {
                // Large matrices: column pre-extraction enables stride-1 inner loop,
                // amortizing the O(n²) buffer write cost with the O(n³) arithmetic.
                double[] colSlice = new double[m];
                for (int i = 0; i < col; i++) {
                    colSlice[i] = lu[i][col];
                }

                for (int row = 0; row < col; row++) {
                    final double[] luRow = lu[row];
                    double sum = luRow[col];
                    for (int i = 0; i < row; i++) {
                        sum -= luRow[i] * colSlice[i];
                    }
                    luRow[col] = sum;
                    colSlice[row] = sum;
                }

                int max = col;
                double largest = Double.NEGATIVE_INFINITY;
                for (int row = col; row < m; row++) {
                    final double[] luRow = lu[row];
                    double sum = luRow[col];
                    for (int i = 0; i < col; i++) {
                        sum -= luRow[i] * colSlice[i];
                    }
                    luRow[col] = sum;
                    if (Math.abs(sum) > largest) {
                        largest = Math.abs(sum);
                        max = row;
                    }
                }

                if (Math.abs(lu[max][col]) < singularityThreshold) {
                    singular = true;
                    throw new SingularMatrixException("LU decomposition failed: matrix is singular at column " + col);
                }

                if (max != col) {
                    double[] tmp = lu[col];
                    lu[col] = lu[max];
                    lu[max] = tmp;
                    int temp = pivot[max];
                    pivot[max] = pivot[col];
                    pivot[col] = temp;
                    even = !even;
                }

                final double luDiag = lu[col][col];
                for (int row = col + 1; row < m; row++) {
                    lu[row][col] /= luDiag;
                }
            }

        }

        return new Tuple2<>(getL(), getU());
    }

    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decomposeInPlace(IMatrix<Double> matrix) {
        if (!(matrix instanceof RereDoubleMatrix dm)) {
            return decompose(matrix);
        }
        if (!matrix.isSquare()) {
            throw new IllegalArgumentException("Only square matrices can perform LU decomposition");
        }

        final int m = matrix.cols();
        this.lu = dm.getData();
        this.pivot = new int[m];
        this.cachedL = null;
        this.cachedU = null;
        this.cachedP = null;

        for (int row = 0; row < m; row++) {
            pivot[row] = row;
        }
        even = true;
        singular = false;

        for (int col = 0; col < m; col++) {
            if (m < 256) {
                // Direct stride-n access — column fits in L1, buffer overhead not amortized.
                for (int row = 0; row < col; row++) {
                    final double[] luRow = lu[row];
                    double sum = luRow[col];
                    for (int i = 0; i < row; i++) {
                        sum -= luRow[i] * lu[i][col];
                    }
                    luRow[col] = sum;
                }

                int max = col;
                double largest = Double.NEGATIVE_INFINITY;
                for (int row = col; row < m; row++) {
                    final double[] luRow = lu[row];
                    double sum = luRow[col];
                    for (int i = 0; i < col; i++) {
                        sum -= luRow[i] * lu[i][col];
                    }
                    luRow[col] = sum;
                    if (Math.abs(sum) > largest) {
                        largest = Math.abs(sum);
                        max = row;
                    }
                }

                if (Math.abs(largest) < singularityThreshold) {
                    singular = true;
                    throw new SingularMatrixException("LU decomposition failed: matrix is singular at column " + col);
                }

                if (max != col) {
                    double[] tmp = lu[col];
                    lu[col] = lu[max];
                    lu[max] = tmp;
                    int temp = pivot[col];
                    pivot[col] = pivot[max];
                    pivot[max] = temp;
                    even = !even;
                }

                final double luDiag = lu[col][col];
                for (int row = col + 1; row < m; row++) {
                    lu[row][col] /= luDiag;
                }
            } else {
                // Large matrices: column pre-extraction enables stride-1 inner loop.
                double[] colSlice = new double[m];
                for (int i = 0; i < col; i++) {
                    colSlice[i] = lu[i][col];
                }
                for (int row = 0; row < col; row++) {
                    final double[] luRow = lu[row];
                    double sum = luRow[col];
                    for (int i = 0; i < row; i++) {
                        sum -= luRow[i] * colSlice[i];
                    }
                    luRow[col] = sum;
                    colSlice[row] = sum;
                }

                int max = col;
                double largest = Double.NEGATIVE_INFINITY;
                for (int row = col; row < m; row++) {
                    final double[] luRow = lu[row];
                    double sum = luRow[col];
                    for (int i = 0; i < col; i++) {
                        sum -= luRow[i] * colSlice[i];
                    }
                    luRow[col] = sum;
                    if (Math.abs(sum) > largest) {
                        largest = Math.abs(sum);
                        max = row;
                    }
                }

                if (Math.abs(largest) < singularityThreshold) {
                    singular = true;
                    throw new SingularMatrixException("LU decomposition failed: matrix is singular at column " + col);
                }

                if (max != col) {
                    double[] tmp = lu[col];
                    lu[col] = lu[max];
                    lu[max] = tmp;
                    int temp = pivot[col];
                    pivot[col] = pivot[max];
                    pivot[max] = temp;
                    even = !even;
                }

                final double luDiag = lu[col][col];
                for (int row = col + 1; row < m; row++) {
                    lu[row][col] /= luDiag;
                }
            }
        }

        return new Tuple2<>(getL(), getU());
    }

    /**
     * Returns the matrix L of the decomposition.
     * <p>L is a lower-triangular matrix with unit diagonal</p>
     * @return the L matrix (or null if decomposed matrix is singular)
     */
    @Override
    public IMatrix<Double> getL() {
        if (cachedL == null && !singular && lu != null) {
            final int m = pivot.length;
            // Create L matrix - after row swaps, L is stored in lu
            double[][] lData = new double[m][m];
            for (int i = 0; i < m; ++i) {
                final double[] luRow = lu[i];
                for (int j = 0; j < i; ++j) {
                    lData[i][j] = luRow[j];
                }
                // Diagonal elements are 1.0 for L matrix
                lData[i][i] = 1.0;
            }
            cachedL = Linalg.matrix(lData);
        }
        return cachedL;
    }

    /**
     * Returns the matrix U of the decomposition.
     * <p>U is an upper-triangular matrix</p>
     * @return the U matrix (or null if decomposed matrix is singular)
     */
    @Override
    public IMatrix<Double> getU() {
        if (cachedU == null && !singular && lu != null) {
            final int m = pivot.length;
            // Create U matrix - after row swaps, U is stored in lu
            double[][] uData = new double[m][m];
            for (int i = 0; i < m; ++i) {
                final double[] luRow = lu[i];
                for (int j = i; j < m; ++j) {
                    uData[i][j] = luRow[j];
                }
            }
            cachedU = Linalg.matrix(uData);
        }
        return cachedU;
    }

    /**
     * Returns the P rows permutation matrix.
     * <p>P is a sparse matrix with exactly one element set to 1.0 in
     * each row and each column, all other elements being set to 0.0.</p>
     * @return the P rows permutation matrix (or null if decomposed matrix is singular)
     */
    @Override
    public IMatrix<Double> getP() {
        if (cachedP == null && !singular && pivot != null) {
            // Create P matrix using the new Linalg API
            cachedP = Linalg.permutationMatrix(pivot.length, pivot);
        }
        return cachedP;
    }

    /**
     * Returns the pivot permutation vector.
     * @return the pivot permutation vector
     */
    @Override
    public int[] getPivot() {
        return pivot != null ? pivot.clone() : null;
    }

    /**
     * Return the determinant of the matrix.
     * @return determinant of the matrix
     */
    @Override
    public double getDeterminant() {
        if (singular) {
            return 0;
        } else if (lu == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        } else {
            final int m = pivot.length;
            // Logarithmic accumulation avoids overflow/overflow from ad-hoc scaling.
            double logAbsDet = 0;
            int sign = even ? 1 : -1;
            for (int i = 0; i < m; i++) {
                double diag = lu[i][i];
                if (diag == 0) {
                    return 0;
                }
                if (diag < 0) {
                    sign = -sign;
                }
                logAbsDet += Math.log(Math.abs(diag));
            }
            if (logAbsDet > Math.log(Double.MAX_VALUE)) {
                return sign > 0 ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
            }
            return sign * Math.exp(logAbsDet);
        }
    }

    /**
     * Get a solver for finding the A &times; X = B solution in exact linear
     * sense.
     * @return a solver
     */
    @Override
    public IDecompositionSolver getSolver() {
        if (lu == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        return new LUDecompositionSolver(lu, pivot, singular, even);
    }

    /**
     * Get the singularity threshold.
     * @return the singularity threshold
     */
    @Override
    public double getSingularityThreshold() {
        return singularityThreshold;
    }

    /**
     * Get the epsilon value used for numerical comparisons.
     * 
     * @return the epsilon value (not used in LU decomposition)
     */
    @Override
    public double getEpsilon() {
        // LU decomposition doesn't use epsilon, but we need to provide a value
        return DEFAULT_TOO_SMALL;
    }

    /**
     * Perform LU decomposition with a specified epsilon.
     * 
     * @param matrix The matrix to decompose
     * @param epsilon Threshold for considering an element as zero (not used in LU decomposition)
     * @return A tuple containing L and U matrices
     * @throws IllegalArgumentException if the matrix is not square
     */
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon) {
        // Epsilon is not used in LU decomposition, but we maintain the interface contract
        return decompose(matrix);
    }

    /**
     * Perform LU decomposition with configurable parameters.
     * 
     * @param matrix The matrix to decompose
     * @param epsilon Threshold for considering an element as zero (not used in LU decomposition)
     * @param maxIterations Maximum number of iterations (not used in LU decomposition)
     * @return A tuple containing L and U matrices
     * @throws IllegalArgumentException if the matrix is not square
     */
    @Override
    public Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations) {
        // Epsilon and maxIterations are not used in LU decomposition, but we maintain the interface contract
        return decompose(matrix);
    }

    /**
     * Check if the decomposed matrix is non-singular.
     * 
     * @return true if the decomposed matrix is non-singular
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public boolean isNonSingular() {
        if (lu == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        return !singular;
    }

    /**
     * Get the condition number of the matrix.
     * 
     * @return condition number of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public double getConditionNumber() {
        if (lu == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        
        if (singular) {
            return Double.POSITIVE_INFINITY;
        }
        
        // For LU decomposition, we can estimate the condition number
        // Using the fact that ||A|| ≈ ||L|| * ||U||
        IMatrix<Double> l = getL();
        IMatrix<Double> u = getU();
        
        if (l == null || u == null) {
            // Matrix is singular
            return Double.POSITIVE_INFINITY;
        }
        
        // Estimate norms (using Frobenius norm for simplicity)
        double lNorm = 0.0;
        double uNorm = 0.0;
        
        for (int i = 0; i < l.rows(); i++) {
            for (int j = 0; j < l.cols(); j++) {
                double val = l.get(i, j);
                lNorm += val * val;
            }
        }
        lNorm = Math.sqrt(lNorm);
        
        for (int i = 0; i < u.rows(); i++) {
            for (int j = 0; j < u.cols(); j++) {
                double val = u.get(i, j);
                uNorm += val * val;
            }
        }
        uNorm = Math.sqrt(uNorm);
        
        // Condition number estimate
        double normA = lNorm * uNorm;
        
        // For the inverse, we would need to compute it, but we can estimate
        // For now, we'll use a simplified approach
        double det = getDeterminant();
        if (Math.abs(det) < singularityThreshold) {
            return Double.POSITIVE_INFINITY;
        }
        double normInvEstimate = 1.0 / Math.abs(det);
        
        return normA * normInvEstimate;
    }

    /**
     * Get the rank of the matrix.
     *
     * @return rank of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    @Override
    public int getRank() {
        if (lu == null) {
            throw new IllegalStateException("Decomposition not performed yet");
        }
        
        if (singular) {
            // Count non-zero diagonal elements
            int rank = 0;
            for (int i = 0; i < lu.length; i++) {
                if (Math.abs(lu[i][i]) > singularityThreshold) {
                    rank++;
                }
            }
            return rank;
        } else {
            // Full rank
            return lu.length;
        }
    }

    /**
     * Get the maximum number of iterations allowed.
     * 
     * @return the maximum number of iterations (not applicable for LU decomposition)
     */
    @Override
    public int getMaxIterations() {
        // LU decomposition is not iterative, return a default value
        return IMatrixDecomposition.DEFAULT_MAX_ITERATIONS;
    }
}