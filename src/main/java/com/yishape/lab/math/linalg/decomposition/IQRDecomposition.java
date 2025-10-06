package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.util.Tuple2;

/**
 * QR decomposition interface.
 * <p>
 * The QR decomposition of a matrix A is a decomposition of the form: A = Q * R
 * where Q is an orthogonal matrix and R is an upper triangular matrix.
 * </p>
 * 
 * <h3>Key Features</h3>
 * <ul>
 *   <li>Householder reflections for numerical stability</li>
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
public interface IQRDecomposition extends IMatrixDecomposition<Tuple2<IMatrix<Double>, IMatrix<Double>>> {
    
    /**
     * Perform QR decomposition on a matrix
     * 
     * @param matrix The matrix to decompose
     * @return A tuple containing Q and R matrices
     */
    @Override
    Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix);
    
    /**
     * Perform QR decomposition on a matrix with configurable epsilon.
     * 
     * @param matrix The matrix to decompose
     * @param epsilon Threshold for considering an element as zero
     * @return A tuple containing Q and R matrices
     */
    @Override
    Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon);
    
    /**
     * Returns the matrix Q of the decomposition.
     * <p>Q is an orthogonal matrix</p>
     * @return the Q matrix
     */
    IMatrix<Double> getQ();
    
    /**
     * Returns the matrix R of the decomposition.
     * <p>R is an upper-triangular matrix</p>
     * @return the R matrix
     */
    IMatrix<Double> getR();
    
    /**
     * Returns the transpose of the matrix Q of the decomposition.
     * <p>Q is an orthogonal matrix</p>
     * @return the transpose of the Q matrix, Q<sup>T</sup>
     */
    IMatrix<Double> getQT();
    
    /**
     * Get a solver for finding the A &times; X = B solution in least square sense.
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