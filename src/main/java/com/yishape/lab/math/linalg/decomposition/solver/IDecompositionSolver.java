package com.yishape.lab.math.linalg.decomposition.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;

/**
 * Interface handling decomposition algorithms that can solve A &times; X = B.
 * <p>
 * Decomposition algorithms decompose an A matrix as a product of several specific
 * matrices from which they can solve A &times; X = B in least squares sense: they find X
 * such that ||A &times; X - B|| is minimal.
 * <p>
 * Some solvers like {@link ILUDecomposition} can only find the solution for
 * square matrices and when the solution is an exact linear solution, i.e. when
 * ||A &times; X - B|| is exactly 0. Other solvers can also find solutions
 * with non-square matrix A and with non-null minimal norm. If an exact linear
 * solution exists it is also the minimal norm solution.
 * </p>
 * 
 * <h3>Key Features</h3>
 * <ul>
 *   <li>Unified interface for solving linear systems</li>
 *   <li>Support for both vector and matrix right-hand sides</li>
 *   <li>Singularity checking for numerical robustness</li>
 *   <li>Pseudo-inverse computation for non-square systems</li>
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
public interface IDecompositionSolver {

    /**
     * Solve the linear equation A &times; X = B for matrices A.
     * <p>
     * The A matrix is implicit, it is provided by the underlying
     * decomposition algorithm.
     *
     * @param b right-hand side of the equation A &times; X = B
     * @return a vector X that minimizes the two norm of A &times; X - B
     * @throws IllegalArgumentException if the matrices dimensions do not match.
     * @throws RuntimeException if the decomposed matrix is singular.
     */
    IVector<Double> solve(IVector<Double> b);

    /**
     * Solve the linear equation A &times; X = B for matrices A.
     * <p>
     * The A matrix is implicit, it is provided by the underlying
     * decomposition algorithm.
     *
     * @param b right-hand side of the equation A &times; X = B
     * @return a matrix X that minimizes the two norm of A &times; X - B
     * @throws IllegalArgumentException if the matrices dimensions do not match.
     * @throws RuntimeException if the decomposed matrix is singular.
     */
    IMatrix<Double> solve(IMatrix<Double> b);

    /**
     * Check if the decomposed matrix is non-singular.
     * @return true if the decomposed matrix is non-singular.
     */
    boolean isNonSingular();

    /**
     * Get the pseudo-inverse of the decomposed matrix.
     * <p>
     * This is equal to the inverse of the decomposed matrix, if such an inverse exists.
     * <p>
     * If no such inverse exists, then the result has properties that resemble that of an inverse.
     * <p>
     * In particular, in this case, if the decomposed matrix is A, then the system of equations
     * \( A x = b \) may have no solutions, or many. If it has no solutions, then the pseudo-inverse
     * \( A^+ \) gives the "closest" solution \( z = A^+ b \), meaning \( \left \| A z - b \right \|_2 \)
     * is minimized. If there are many solutions, then \( z = A^+ b \) is the smallest solution,
     * meaning \( \left \| z \right \|_2 \) is minimized.
     *
     * @return pseudo-inverse matrix (which is the inverse, if it exists)
     * @throws RuntimeException if the decomposed matrix is singular and the decomposition
     * can not compute a pseudo-inverse
     */
    IMatrix<Double> getInverse();
}