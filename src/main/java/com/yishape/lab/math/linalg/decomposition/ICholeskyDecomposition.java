package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;

/**
 * Cholesky decomposition interface with enhanced features.
 * <p>
 * The Cholesky decomposition of a symmetric, positive definite matrix A is a
 * decomposition of the form: A = L * L^T where L is a lower triangular matrix
 * with positive diagonal entries.
 * </p>
 *
 * <h3>Key Features</h3>
 * <ul>
 *   <li>Enhanced numerical stability with configurable thresholds</li>
 *   <li>Comprehensive error handling with specific exceptions</li>
 *   <li>Efficient caching of computed results</li>
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
public interface ICholeskyDecomposition extends IMatrixDecomposition<IMatrix<Double>> {
    
    /**
     * Default threshold above which off-diagonal elements are considered too different
     * and matrix not symmetric.
     */
    double DEFAULT_RELATIVE_SYMMETRY_THRESHOLD = 1.0e-15;
    
    /**
     * Default threshold below which diagonal elements are considered null
     * and matrix not positive definite.
     */
    double DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD = 1.0e-10;
    
    /**
     * Perform Cholesky decomposition on a matrix
     * 
     * @param matrix The matrix to decompose (must be symmetric positive definite)
     * @return The lower triangular matrix L such that A = L * L^T
     * @throws NonSymmetricMatrixException if the matrix is not symmetric
     * @throws NonPositiveDefiniteMatrixException if the matrix is not positive definite
     */
    @Override
    IMatrix<Double> decompose(IMatrix<Double> matrix);
    
    /**
     * Perform Cholesky decomposition on a matrix with configurable thresholds
     * 
     * @param matrix The matrix to decompose (must be symmetric positive definite)
     * @param relativeSymmetryThreshold threshold above which off-diagonal
     * elements are considered too different and matrix not symmetric
     * @param absolutePositivityThreshold threshold below which diagonal
     * elements are considered null and matrix not positive definite
     * @return The lower triangular matrix L such that A = L * L^T
     * @throws NonSymmetricMatrixException if the matrix is not symmetric
     * @throws NonPositiveDefiniteMatrixException if the matrix is not positive definite
     */
    IMatrix<Double> decompose(IMatrix<Double> matrix, 
                             double relativeSymmetryThreshold,
                             double absolutePositivityThreshold);
    
    /**
     * Perform Cholesky decomposition on a matrix with unified parameters
     * 
     * @param matrix The matrix to decompose (must be symmetric positive definite)
     * @param epsilon Threshold for considering an element as zero
     * @param maxIterations Maximum number of iterations (not used in Cholesky)
     * @return The lower triangular matrix L such that A = L * L^T
     * @throws NonSymmetricMatrixException if the matrix is not symmetric
     * @throws NonPositiveDefiniteMatrixException if the matrix is not positive definite
     */
    @Override
    IMatrix<Double> decompose(IMatrix<Double> matrix, double epsilon, int maxIterations);
    
    /**
     * Returns the matrix L of the decomposition.
     * <p>L is a lower-triangular matrix</p>
     * @return the L matrix
     */
    IMatrix<Double> getL();
    
    /**
     * Returns the transpose of the matrix L of the decomposition.
     * <p>L<sup>T</sup> is an upper-triangular matrix</p>
     * @return the transpose of the matrix L of the decomposition
     */
    IMatrix<Double> getLT();
    
    /**
     * Return the determinant of the matrix.
     * @return determinant of the matrix
     */
    @Override
    double getDeterminant();
    
    /**
     * Get a solver for finding the A &times; X = B solution in exact linear
     * sense.
     * @return a solver
     */
    @Override
    IDecompositionSolver getSolver();
    
    /**
     * Check if the decomposed matrix is non-singular.
     * @return true if the decomposed matrix is non-singular
     */
    @Override
    boolean isNonSingular();
    
    /**
     * Get the condition number of the matrix.
     * @return condition number of the matrix
     */
    @Override
    double getConditionNumber();
    
    /**
     * Get the rank of the matrix.
     * @return rank of the matrix
     */
    @Override
    int getRank();
    
    /**
     * Get the relative symmetry threshold.
     * @return the relative symmetry threshold
     */
    double getRelativeSymmetryThreshold();
    
    /**
     * Get the absolute positivity threshold.
     * @return the absolute positivity threshold
     */
    double getAbsolutePositivityThreshold();
    
    /**
     * Get the epsilon value used for numerical comparisons.
     * @return the epsilon value
     */
    @Override
    double getEpsilon();
    
    /**
     * Get the maximum number of iterations allowed.
     * @return the maximum number of iterations
     */
    @Override
    int getMaxIterations();
}