package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.util.Tuple2;

/**
 * LU decomposition interface with partial pivoting support
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
}