package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.util.Tuple2;

/**
 * LU decomposition interface with partial pivoting support.
 *
 * <p>
 * LU decomposition decomposes a matrix A into the form A = P^T * L * U where
 * P is a permutation matrix, L is a lower triangular matrix with unit diagonal,
 * and U is an upper triangular matrix. The decomposition with partial pivoting
 * ensures numerical stability by swapping rows to place the largest pivot element
 * in each column.
 * </p>
 *
 * <h3>Key Features</h3>
 * <ul>
 *   <li>Partial pivoting for numerical stability</li>
 *   <li>Efficient solving of linear systems</li>
 *   <li>Comprehensive error handling</li>
 *   <li>Unified interface with other decompositions</li>
 * </ul>
 *
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Press, W. H., Teukolsky, S. A., Vetterling, W. T., &amp; Flannery, B. P. (2007). Numerical recipes: The art of scientific computing (3rd ed.). Cambridge University Press.</li>
 * </ul>
 *
 * @author lteb2
 * @version 1.0
 * @since 2.0
 */
public interface ILUDecomposition extends IMatrixDecomposition<Tuple2<IMatrix<Double>, IMatrix<Double>>> {
    
    /**
     * Perform LUP decomposition on a matrix
     * 
     * @param matrix The matrix to decompose
     * @return A tuple containing L and U matrices
     * @throws IllegalArgumentException if the matrix is not square
     */
    @Override
    Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix);
    
    /**
     * Returns the matrix L of the decomposition.
     * <p>L is a lower-triangular matrix with unit diagonal</p>
     * @return the L matrix (or null if decomposed matrix is singular)
     */
    IMatrix<Double> getL();
    
    /**
     * Returns the matrix U of the decomposition.
     * <p>U is an upper-triangular matrix</p>
     * @return the U matrix (or null if decomposed matrix is singular)
     */
    IMatrix<Double> getU();
    
    /**
     * Returns the P rows permutation matrix.
     * <p>P is a sparse matrix with exactly one element set to 1.0 in
     * each row and each column, all other elements being set to 0.0.</p>
     * @return the P rows permutation matrix (or null if decomposed matrix is singular)
     */
    IMatrix<Double> getP();
    
    /**
     * Returns the pivot permutation vector.
     * @return the pivot permutation vector
     */
    int[] getPivot();
    
    /**
     * Return the determinant of the matrix.
     * @return determinant of the matrix
     */
    double getDeterminant();
    
    /**
     * Get a solver for finding the A &times; X = B solution in exact linear
     * sense.
     * @return a solver
     */
    IDecompositionSolver getSolver();
    
    /**
     * Get the singularity threshold.
     * @return the singularity threshold
     */
    double getSingularityThreshold();

    /**
     * Decompose the matrix IN PLACE. The caller-provided matrix is overwritten
     * with the LU factors. Only supported when the input is a
     * {@link com.yishape.lab.math.linalg.RereDoubleMatrix} (falls back to
     * {@link #decompose(IMatrix)} otherwise).
     *
     * @param matrix will be MUTATED to store L and U factors
     * @return (L, U) tuple (views over the mutated input)
     * @throws IllegalArgumentException if matrix is not square
     */
    Tuple2<IMatrix<Double>, IMatrix<Double>> decomposeInPlace(IMatrix<Double> matrix);
}