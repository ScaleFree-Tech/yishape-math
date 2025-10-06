package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.util.Tuple3;

/**
 * Bidiagonal decomposition interface
 * <p>
 * Bidiagonal decomposition transforms a matrix A into the form A = U * B * V^T,
 * where B is a bidiagonal matrix, and U and V are orthogonal matrices.
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
public interface IBidiagonalDecomposition extends IMatrixDecomposition<Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>>> {
    
    /**
     * Performs bidiagonal decomposition of the given matrix
     *
     * @param matrix the matrix to decompose
     * @return a tuple containing (U, B, V) where B is the bidiagonal matrix
     */
    @Override
    Tuple3<IMatrix<Double>, IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix);
    
    /**
     * Returns the orthogonal matrix U of the decomposition.
     * <p>U is an orthogonal matrix</p>
     * @return the U matrix
     */
    IMatrix<Double> getU();
    
    /**
     * Returns the bidiagonal matrix B of the decomposition.
     * <p>B is a bidiagonal matrix</p>
     * @return the B matrix
     */
    IMatrix<Double> getB();
    
    /**
     * Returns the orthogonal matrix V of the decomposition.
     * <p>V is an orthogonal matrix</p>
     * @return the V matrix
     */
    IMatrix<Double> getV();
    
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