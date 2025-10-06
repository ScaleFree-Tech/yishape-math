package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.util.Tuple2;

/**
 * Schur decomposition interface.
 * <p>
 * The Schur decomposition of a real square matrix A is a decomposition of the form:
 * A = U T U^T where:
 * <ul>
 *   <li>U is an orthogonal matrix</li>
 *   <li>T is a quasi-upper triangular matrix (block upper triangular with 1x1 and 2x2 blocks)</li>
 * </ul>
 * </p>
 * 
 * <p>
 * For a complex square matrix, the Schur decomposition is:
 * A = U T U^H where:
 * <ul>
 *   <li>U is a unitary matrix</li>
 *   <li>T is an upper triangular matrix</li>
 *   <li>U^H is the conjugate transpose of U</li>
 * </ul>
 * </p>
 * 
 * <p>
 * The Schur decomposition always exists for any square matrix and is numerically stable.
 * It is particularly useful for computing eigenvalues and eigenvectors, as the eigenvalues
 * of A are the diagonal elements of T.
 * </p>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Horn, R. A., &amp; Johnson, C. R. (2012). Matrix analysis (2nd ed.). Cambridge University Press.</li>
 *   <li>Watkins, D. S. (2007). The matrix eigenvalue problem: GR and Krylov subspace methods. SIAM.</li>
 * </ul>
 * 
 * @since 2.0
 * @see <a href="http://mathworld.wolfram.com/SchurDecomposition.html">Schur Decomposition - MathWorld</a>
 * @see <a href="http://en.wikipedia.org/wiki/Schur_decomposition">Schur Decomposition - Wikipedia</a>
 */
public interface ISchurDecomposition extends IMatrixDecomposition<Tuple2<IMatrix<Double>, IMatrix<Double>>> {
    
    /**
     * Default threshold for considering an element as zero in numerical computations.
     */
    double DEFAULT_EPSILON = 1e-12;
    
    /**
     * Perform Schur decomposition on a matrix.
     * 
     * @param matrix The matrix to decompose (must be square)
     * @return A tuple containing U (orthogonal matrix) and T (quasi-upper triangular matrix)
     * @throws NonSquareMatrixException if the matrix is not square
     * @throws DecompositionFailedException if the decomposition fails to converge
     */
    @Override
    Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix);
    
    /**
     * Perform Schur decomposition on a matrix with a specified epsilon.
     * 
     * @param matrix The matrix to decompose (must be square)
     * @param epsilon Threshold for considering an element as zero
     * @return A tuple containing U (orthogonal matrix) and T (quasi-upper triangular matrix)
     * @throws NonSquareMatrixException if the matrix is not square
     * @throws DecompositionFailedException if the decomposition fails to converge
     */
    Tuple2<IMatrix<Double>, IMatrix<Double>> decompose(IMatrix<Double> matrix, double epsilon);
    
    /**
     * Returns the orthogonal matrix U of the decomposition.
     * <p>U is an orthogonal matrix: U^T U = I</p>
     * 
     * @return the U matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    IMatrix<Double> getU();
    
    /**
     * Returns the quasi-upper triangular matrix T of the decomposition.
     * <p>T is a quasi-upper triangular matrix with 1x1 and 2x2 blocks on the diagonal</p>
     * 
     * @return the T matrix
     * @throws IllegalStateException if decomposition has not been performed
     */
    IMatrix<Double> getT();
    
    /**
     * Returns the transpose of the orthogonal matrix U of the decomposition.
     * <p>U^T is an orthogonal matrix: U U^T = I</p>
     * 
     * @return the transpose of the U matrix, U^T
     * @throws IllegalStateException if decomposition has not been performed
     */
    IMatrix<Double> getUT();
    
    /**
     * Get a solver for finding the A &times; X = B solution.
     * 
     * @return a solver
     * @throws IllegalStateException if decomposition has not been performed
     */
    IDecompositionSolver getSolver();
    
    /**
     * Get the epsilon value used for numerical comparisons.
     * 
     * @return the epsilon value
     */
    double getEpsilon();
}