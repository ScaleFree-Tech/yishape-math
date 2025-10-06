package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.util.Tuple2;

/**
 * Hessenberg decomposition interface with enhanced features.
 * <p>
 * Hessenberg decomposition transforms a matrix A into the form A = Q * H * Q^T,
 * where H is a Hessenberg matrix, and Q is an orthogonal matrix.
 * </p>
 * 
 * <h3>Key Features</h3>
 * <ul>
 *   <li>Efficient reduction to Hessenberg form using Householder reflections</li>
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
 * @since 2.0
 */
public interface IHessenbergDecomposition extends IMatrixDecomposition<Tuple2<IMatrix<Double>, IMatrix<Double>>> {
    /**
     * Performs Hessenberg decomposition of the given matrix
     *
     * @param matrix the matrix to decompose
     * @return a tuple containing (H, Q) where H is the Hessenberg matrix
     * @throws NonSquareMatrixException if the matrix is not square
     */
    @Override
    Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix);
    
    /**
     * Performs Hessenberg decomposition of the given matrix with configurable epsilon.
     *
     * @param matrix the matrix to decompose
     * @param epsilon Threshold for considering an element as zero
     * @return a tuple containing (H, Q) where H is the Hessenberg matrix
     * @throws NonSquareMatrixException if the matrix is not square
     */
    @Override
    Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon);
    
    /**
     * Get a solver for finding the A &times; X = B solution.
     * @return a solver
     */
    @Override
    IDecompositionSolver getSolver();
    
    /**
     * Calculate the determinant of the matrix.
     * @return determinant of the matrix
     */
    @Override
    double getDeterminant();
    
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