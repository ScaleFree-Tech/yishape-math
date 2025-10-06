package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;

/**
 * Unified interface for all matrix decomposition algorithms.
 * <p>
 * This interface provides a common set of methods that all matrix decomposition
 * algorithms should implement to ensure consistency and ease of use.
 * </p>
 * 
 * <h3>Design Principles</h3>
 * <ul>
 *   <li><strong>Consistency</strong>: All decompositions follow the same interface patterns</li>
 *   <li><strong>Flexibility</strong>: Configurable parameters for numerical stability</li>
 *   <li><strong>Performance</strong>: Efficient caching of computed results</li>
 *   <li><strong>Robustness</strong>: Comprehensive error handling with detailed exceptions</li>
 * </ul>
 * 
 * @param <T> The type of result returned by the decomposition
 * @since 2.0
 */
public interface IMatrixDecomposition<T>  {
    
    /**
     * Default threshold for considering an element as zero in numerical computations.
     */
    double DEFAULT_EPSILON = 1e-12;
    
    /**
     * Default maximum number of iterations for iterative algorithms.
     */
    int DEFAULT_MAX_ITERATIONS = 1000;
    
    /**
     * Perform the matrix decomposition with default parameters.
     * 
     * @param matrix The matrix to decompose
     * @return The decomposition result
     * @throws MatrixDecompositionException if the decomposition fails
     */
    T decompose(IMatrix<Double> matrix);
    
    /**
     * Perform the matrix decomposition with configurable epsilon.
     * 
     * @param matrix The matrix to decompose
     * @param epsilon Threshold for considering an element as zero
     * @return The decomposition result
     * @throws MatrixDecompositionException if the decomposition fails
     */
    T decompose(IMatrix<Double> matrix, double epsilon);
    
    /**
     * Perform the matrix decomposition with configurable parameters.
     * 
     * @param matrix The matrix to decompose
     * @param epsilon Threshold for considering an element as zero
     * @param maxIterations Maximum number of iterations for iterative algorithms
     * @return The decomposition result
     * @throws MatrixDecompositionException if the decomposition fails
     */
    T decompose(IMatrix<Double> matrix, double epsilon, int maxIterations);
    
    /**
     * Get a solver for finding the A &times; X = B solution.
     * 
     * @return a solver
     * @throws IllegalStateException if decomposition has not been performed
     */
    IDecompositionSolver getSolver();
    
    /**
     * Calculate the determinant of the matrix.
     * 
     * @return determinant of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    double getDeterminant();
    
    /**
     * Check if the decomposed matrix is non-singular.
     * 
     * @return true if the decomposed matrix is non-singular
     * @throws IllegalStateException if decomposition has not been performed
     */
    boolean isNonSingular();
    
    /**
     * Get the condition number of the matrix.
     * 
     * @return condition number of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    double getConditionNumber();
    
    /**
     * Get the rank of the matrix.
     * 
     * @return rank of the matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    int getRank();
    
    /**
     * Get the epsilon value used for numerical comparisons.
     * 
     * @return the epsilon value
     */
    double getEpsilon();
    
    /**
     * Get the maximum number of iterations allowed.
     * 
     * @return the maximum number of iterations
     */
    int getMaxIterations();
    
    
}