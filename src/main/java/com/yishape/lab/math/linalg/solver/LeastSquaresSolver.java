package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.IVector;
import com.yishape.lab.math.linalg.Linalg;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;
import com.yishape.lab.math.util.RerePrecision;
import com.yishape.lab.util.Tuple2;

/**
 * Unified least squares solver that automatically selects the most appropriate
 * decomposition method based on matrix properties.
 * <p>
 * This solver analyzes the input matrix characteristics and selects the optimal
 * decomposition algorithm for solving the least squares problem min||A × X - B||₂.
 * It provides a unified interface for solving least squares problems regardless
 * of matrix type or condition, automatically choosing the most efficient and
 * numerically stable method available.
 * </p>
 * 
 * <h3>Supported Decomposition Methods</h3>
 * <p>
 * The solver supports all 9 matrix decomposition types available in the system:
 * </p>
 * <ul>
 *   <li><strong>QR Decomposition</strong> - For well-conditioned overdetermined systems</li>
 *   <li><strong>SVD Decomposition</strong> - For ill-conditioned or rank-deficient systems</li>
 *   <li><strong>Cholesky Decomposition</strong> - For normal equations A^T × A when well-conditioned</li>
 *   <li><strong>LU Decomposition</strong> - For normal equations A^T × A when QR/SVD not preferred</li>
 *   <li><strong>Tridiagonal Decomposition</strong> - For symmetric normal equations</li>
 *   <li><strong>Hessenberg Decomposition</strong> - For large normal equations</li>
 *   <li><strong>Schur Decomposition</strong> - For eigenvalue-related normal equations</li>
 *   <li><strong>Eigen Decomposition</strong> - For small normal equations</li>
 *   <li><strong>Bidiagonal Decomposition</strong> - For large matrices and SVD computation</li>
 * </ul>
 * 
 * <h3>Algorithm Selection Strategy</h3>
 * <p>
 * The solver employs a hierarchical selection strategy based on problem characteristics:
 * </p>
 * <ol>
 *   <li><strong>Overdetermined Analysis</strong> - Determine if system is overdetermined (more equations than unknowns)</li>
 *   <li><strong>Condition Assessment</strong> - Evaluate numerical conditioning and rank deficiency</li>
 *   <li><strong>Size Evaluation</strong> - Consider matrix dimensions for performance optimization</li>
 *   <li><strong>Specialized Detection</strong> - Identify if problem requires specific decomposition</li>
 *   <li><strong>Performance Optimization</strong> - Select fastest applicable method</li>
 *   <li><strong>Fallback Mechanism</strong> - Use robust methods (SVD) when others fail</li>
 * </ol>
 * 
 * <h3>Performance Characteristics</h3>
 * <ul>
 *   <li><strong>Speed</strong>: QR > Cholesky > LU > Tridiagonal > Hessenberg > Schur > Eigen > Bidiagonal > SVD</li>
 *   <li><strong>Stability</strong>: SVD > QR ≥ Cholesky > LU > Schur > Hessenberg > Tridiagonal > Eigen > Bidiagonal</li>
 *   <li><strong>Applicability</strong>: SVD (any system) > QR (overdetermined) > Cholesky/LU (normal equations) > others (specialized)</li>
 * </ul>
 * 
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 * // Solve a least squares problem with matrix right-hand side
 * IMatrix<Double> A = Linalg.matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
 * IMatrix<Double> B = Linalg.matrix(new double[][]{{1}, {2}, {3}});
 * IMatrix<Double> X = LeastSquaresSolver.solve(A, B);
 * 
 * // Solve a least squares problem with vector right-hand side
 * IVector<Double> b = Linalg.vector(new double[]{1, 2, 3});
 * IVector<Double> x = LeastSquaresSolver.solve(A, b);
 * 
 * // Solve a least squares problem and get both solution and residual
 * Tuple2<IVector<Double>, Double> result = LeastSquaresSolver.solveWithResidual(A, b);
 * IVector<Double> solution = result.getFirst();
 * Double residual = result.getSecond();
 * }
 * </pre>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Press, W. H., Teukolsky, S. A., Vetterling, W. T., &amp; Flannery, B. P. (2007). Numerical recipes: The art of scientific computing (3rd ed.). Cambridge University Press.</li>
 *   <li>Björck, Å. (1996). Numerical methods for least squares problems. SIAM.</li>
 *   <li>Lawson, C. L., &amp; Hanson, R. J. (1995). Solving least squares problems. SIAM.</li>
 * </ul>
 * 
 * @since 2.0
 * @see Decomps For creating specific decomposition instances
 * @see IDecompositionSolver For individual solver interfaces
 */
public class LeastSquaresSolver {
    
    /** Default epsilon for numerical comparisons. */
    private static final double DEFAULT_EPSILON = 1e-12;
    
    /** Default threshold for condition number to determine ill-conditioning. */
    private static final double CONDITION_THRESHOLD = 1e12;
    
    /** Threshold for matrix size to determine when to use specialized algorithms. */
    private static final int LARGE_MATRIX_THRESHOLD = 100;
    
    /**
     * Solve the least squares problem min||A × x - b||₂ by automatically selecting
     * the most appropriate decomposition method based on matrix properties.
     * 
     * @param A Coefficient matrix of the least squares problem
     * @param b Right-hand side vector of the least squares problem
     * @return Solution vector x that minimizes ||A × x - b||₂
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     */
    public static IVector<Double> solve(IMatrix<Double> A, IVector<Double> b) {
        // Validate input matrices
        if (A == null || b == null) {
            throw new IllegalArgumentException("Input matrix and vector cannot be null");
        }
        
        if (A.rows() != b.size()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() + 
                " rows but b has " + b.size() + " elements");
        }
        
        // Convert vector to matrix form (column vector)
        IMatrix<Double> bMatrix = Linalg.matrix(new double[][]{b.toDoubleArray()}).transpose();
        
        // Solve using the matrix version
        IMatrix<Double> xMatrix = solve(A, bMatrix);
        
        // Convert result back to vector using getColumn method
        return xMatrix.getColumn(0);
    }
    
    /**
     * Solve the least squares problem min||A × X - B||₂ by automatically selecting
     * the most appropriate decomposition method based on matrix properties.
     * 
     * @param A Coefficient matrix of the least squares problem
     * @param B Right-hand side matrix of the least squares problem
     * @return Solution matrix X that minimizes ||A × X - B||₂
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     */
    public static IMatrix<Double> solve(IMatrix<Double> A, IMatrix<Double> B) {
        // Validate input matrices
        if (A == null || B == null) {
            throw new IllegalArgumentException("Input matrices cannot be null");
        }
        
        if (A.rows() != B.rows()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() + 
                " rows but B has " + B.rows() + " rows");
        }
        
        // 1. First check if the system is well-conditioned and overdetermined
        if (isOverdetermined(A) && isWellConditioned(A)) {
            try {
                // Well-conditioned overdetermined system, use QR decomposition
                var qr = Decomps.createQR();
                qr.decompose(A);
                return qr.getSolver().solve(B);
            } catch (Exception e) {
                // If QR fails, fall back to other methods
            }
        }
        
        // 2. Check if the system is ill-conditioned or rank-deficient
        if (isIllConditioned(A) || isRankDeficient(A)) {
            try {
                // Ill-conditioned or rank-deficient system, use SVD decomposition
                var svd = Decomps.createSVD();
                svd.decompose(A);
                return svd.getSolver().solve(B);
            } catch (Exception e) {
                // If SVD fails, fall back to other methods
            }
        }
        
        // 3. For normal equations approach when matrix is not too large
        if (A.rows() <= LARGE_MATRIX_THRESHOLD) {
            try {
                // Form normal equations A^T × A × X = A^T × B
                IMatrix<Double> AtA = A.transpose().mmul(A);
                
                // Check if normal equations matrix is symmetric positive definite
                if (isSymmetricPositiveDefinite(AtA)) {
                    // Use Cholesky decomposition for normal equations
                    var chol = Decomps.createCholesky();
                    chol.decompose(AtA);
                    IMatrix<Double> AtB = A.transpose().mmul(B);
                    return chol.getSolver().solve(AtB);
                } else {
                    // Use LU decomposition for normal equations
                    var lu = Decomps.createLU();
                    lu.decompose(AtA);
                    IMatrix<Double> AtB = A.transpose().mmul(B);
                    return lu.getSolver().solve(AtB);
                }
            } catch (Exception e) {
                // If normal equations approach fails, fall back to other methods
            }
        }
        
        // 4. Check if normal equations matrix is symmetric
        if (isSymmetric(A.transpose().mmul(A))) {
            try {
                // Form normal equations A^T × A × X = A^T × B
                IMatrix<Double> AtA = A.transpose().mmul(A);
                IMatrix<Double> AtB = A.transpose().mmul(B);
                
                // Symmetric normal equations matrix, use Tridiagonal decomposition
                var tri = Decomps.createTridiagonal();
                tri.decompose(AtA);
                return tri.getSolver().solve(AtB);
            } catch (Exception e) {
                // If Tridiagonal fails, fall back to other methods
            }
        }
        
        // 5. Check if normal equations matrix is a large matrix
        if (A.rows() > LARGE_MATRIX_THRESHOLD) {
            try {
                // Form normal equations A^T × A × X = A^T × B
                IMatrix<Double> AtA = A.transpose().mmul(A);
                IMatrix<Double> AtB = A.transpose().mmul(B);
                
                // Large normal equations matrix, use Hessenberg decomposition for efficiency
                var hess = Decomps.createHessenberg();
                hess.decompose(AtA);
                return hess.getSolver().solve(AtB);
            } catch (Exception e) {
                // If Hessenberg fails, fall back to other methods
            }
        }
        
        // 6. For eigenvalue-related normal equations, use Eigen decomposition
        if (isForEigenvalueProblem(A)) {
            try {
                // Form normal equations A^T × A × X = A^T × B
                IMatrix<Double> AtA = A.transpose().mmul(A);
                IMatrix<Double> AtB = A.transpose().mmul(B);
                
                var eigen = Decomps.createEigen();
                eigen.decompose(AtA);
                return eigen.getSolver().solve(AtB);
            } catch (Exception e) {
                // If Eigen fails, fall back to other methods
            }
        }
        
        // 7. For Schur form normal equations, use Schur decomposition
        if (isForSchurProblem(A)) {
            try {
                // Form normal equations A^T × A × X = A^T × B
                IMatrix<Double> AtA = A.transpose().mmul(A);
                IMatrix<Double> AtB = A.transpose().mmul(B);
                
                var schur = Decomps.createSchur();
                schur.decompose(AtA);
                return schur.getSolver().solve(AtB);
            } catch (Exception e) {
                // If Schur fails, fall back to other methods
            }
        }
        
        // 8. For bidiagonal form problems, use Bidiagonal decomposition
        if (isForBidiagonalProblem(A)) {
            try {
                // Form normal equations A^T × A × X = A^T × B
                IMatrix<Double> AtA = A.transpose().mmul(A);
                IMatrix<Double> AtB = A.transpose().mmul(B);
                
                var bidiag = Decomps.createBidiagonal();
                bidiag.decompose(AtA);
                return bidiag.getSolver().solve(AtB);
            } catch (Exception e) {
                // If Bidiagonal fails, fall back to other methods
            }
        }
        
        // 9. For any remaining cases, use SVD as the most robust method
        try {
            var svd = Decomps.createSVD();
            svd.decompose(A);
            return svd.getSolver().solve(B);
        } catch (Exception e) {
            throw new DecompositionFailedException(
                "Failed to solve least squares problem using any decomposition method", 
                "LeastSquaresSolver", 
                "All decomposition methods failed", 
                -1, 
                Double.NaN, 
                e);
        }
    }
    
    /**
     * Solve the least squares problem min||A × x - b||₂ and return both the solution
     * and the residual norm ||A × x - b||₂.
     * 
     * @param A Coefficient matrix of the least squares problem
     * @param b Right-hand side vector of the least squares problem
     * @return Tuple2 containing the solution vector x and the residual norm
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     */
    public static Tuple2<IVector<Double>, Double> solveWithResidual(IMatrix<Double> A, IVector<Double> b) {
        // Validate input matrices
        if (A == null || b == null) {
            throw new IllegalArgumentException("Input matrix and vector cannot be null");
        }
        
        if (A.rows() != b.size()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() + 
                " rows but b has " + b.size() + " elements");
        }
        
        // Solve the least squares problem
        IVector<Double> x = solve(A, b);
        
        // Compute the residual: ||A × x - b||₂
        IVector<Double> residualVector = A.mmul(x).sub(b);
        double residualNorm = RerePrecision.roundToDecimalPlaces(residualVector.norm2(),6);
        
        return new Tuple2<>(x, residualNorm);
    }
    
    /**
     * Solve the least squares problem min||A × X - B||₂ and return both the solution
     * and the residual norm ||A × X - B||₂.
     * 
     * @param A Coefficient matrix of the least squares problem
     * @param B Right-hand side matrix of the least squares problem
     * @return Tuple2 containing the solution matrix X and the residual norm
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrices dimensions do not match
     */
    public static Tuple2<IMatrix<Double>, Double> solveWithResidual(IMatrix<Double> A, IMatrix<Double> B) {
        // Validate input matrices
        if (A == null || B == null) {
            throw new IllegalArgumentException("Input matrices cannot be null");
        }
        
        if (A.rows() != B.rows()) {
            throw new IllegalArgumentException(
                "Matrix row dimension mismatch: A has " + A.rows() + 
                " rows but B has " + B.rows() + " rows");
        }
        
        // Solve the least squares problem
        IMatrix<Double> X = solve(A, B);
        
        // Compute the residual: ||A × X - B||₂
        IMatrix<Double> residualMatrix = A.mmul(X).sub(B);
        double residualNorm = residualMatrix.frobeniusNorm().doubleValue();
        
        return new Tuple2<>(X, residualNorm);
    }
    
    /**
     * Check if a matrix represents an overdetermined system (more rows than columns).
     * 
     * @param matrix The matrix to check
     * @return true if the system is overdetermined, false otherwise
     */
    private static boolean isOverdetermined(IMatrix<Double> matrix) {
        return matrix.rows() > matrix.cols();
    }
    
    /**
     * Check if a matrix is symmetric positive definite.
     * 
     * @param matrix The matrix to check
     * @return true if the matrix is symmetric positive definite, false otherwise
     */
    private static boolean isSymmetricPositiveDefinite(IMatrix<Double> matrix) {
        try {
            // First check if matrix is square
            if (matrix.rows() != matrix.cols()) {
                return false;
            }
            
            // Check if matrix is symmetric
            if (!matrix.isSymmetric()) {
                return false;
            }
            
            // Check if matrix is positive definite
            return matrix.isPositiveDefinite();
        } catch (Exception e) {
            // If any check fails, assume it's not symmetric positive definite
            return false;
        }
    }
    
    /**
     * Check if a matrix is symmetric.
     * 
     * @param matrix The matrix to check
     * @return true if the matrix is symmetric, false otherwise
     */
    private static boolean isSymmetric(IMatrix<Double> matrix) {
        try {
            // First check if matrix is square
            if (matrix.rows() != matrix.cols()) {
                return false;
            }
            
            return matrix.isSymmetric();
        } catch (Exception e) {
            // If the check fails, assume it's not symmetric
            return false;
        }
    }
    
    /**
     * Check if a matrix is well-conditioned.
     * 
     * @param matrix The matrix to check
     * @return true if the matrix is well-conditioned, false otherwise
     */
    private static boolean isWellConditioned(IMatrix<Double> matrix) {
        try {
            // Check condition number
            double conditionNumber = matrix.cond().doubleValue();
            
            // A matrix is well-conditioned if its condition number is not too large
            return conditionNumber < CONDITION_THRESHOLD;
        } catch (Exception e) {
            // If the check fails, assume it's not well-conditioned
            return false;
        }
    }
    
    /**
     * Check if a matrix is ill-conditioned.
     * 
     * @param matrix The matrix to check
     * @return true if the matrix is ill-conditioned, false otherwise
     */
    private static boolean isIllConditioned(IMatrix<Double> matrix) {
        try {
            // Check condition number
            double conditionNumber = matrix.cond().doubleValue();
            
            // A matrix is ill-conditioned if its condition number is too large
            return conditionNumber >= CONDITION_THRESHOLD;
        } catch (Exception e) {
            // If the check fails, assume it's not ill-conditioned
            return false;
        }
    }
    
    /**
     * Check if a matrix is rank-deficient.
     * 
     * @param matrix The matrix to check
     * @return true if the matrix is rank-deficient, false otherwise
     */
    private static boolean isRankDeficient(IMatrix<Double> matrix) {
        try {
            // Get rank of the matrix
            int rank = matrix.rank();
            
            // A matrix is rank-deficient if its rank is less than the minimum dimension
            return rank < Math.min(matrix.rows(), matrix.cols());
        } catch (Exception e) {
            // If the check fails, assume it's not rank-deficient
            return false;
        }
    }
    
    /**
     * Check if the problem is related to eigenvalue computation.
     * This is a heuristic check based on matrix properties.
     * 
     * @param matrix The matrix to check
     * @return true if the problem is likely related to eigenvalue computation
     */
    private static boolean isForEigenvalueProblem(IMatrix<Double> matrix) {
        try {
            // For now, we'll use a simple heuristic
            // In practice, this would be determined by the context of the problem
            return Math.min(matrix.rows(), matrix.cols()) <= 50; // Small matrices
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Check if the problem is related to Schur form computation.
     * This is a heuristic check based on matrix properties.
     * 
     * @param matrix The matrix to check
     * @return true if the problem is likely related to Schur form computation
     */
    private static boolean isForSchurProblem(IMatrix<Double> matrix) {
        try {
            // For now, we'll use a simple heuristic
            // In practice, this would be determined by the context of the problem
            int minDim = Math.min(matrix.rows(), matrix.cols());
            return minDim > 50 && minDim <= LARGE_MATRIX_THRESHOLD; // Medium-sized matrices
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Check if the problem is related to bidiagonal form computation.
     * This is a heuristic check based on matrix properties.
     * 
     * @param matrix The matrix to check
     * @return true if the problem is likely related to bidiagonal form computation
     */
    private static boolean isForBidiagonalProblem(IMatrix<Double> matrix) {
        try {
            // Bidiagonal decomposition is typically used for large matrices
            return Math.min(matrix.rows(), matrix.cols()) > LARGE_MATRIX_THRESHOLD;
        } catch (Exception e) {
            return false;
        }
    }
}