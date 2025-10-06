package com.yishape.lab.math.linalg.decomposition;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.decomposition.solver.IDecompositionSolver;

/**
 * Unified determinant calculator that automatically selects the most appropriate
 * decomposition method based on matrix properties.
 * <p>
 * This calculator analyzes the input matrix characteristics and selects the optimal
 * decomposition algorithm for computing the matrix determinant det(A). It provides
 * a unified interface for determinant calculation regardless of matrix type or condition,
 * automatically choosing the most efficient and numerically stable method available.
 * </p>
 * 
 * <h3>Supported Decomposition Methods</h3>
 * <p>
 * The calculator supports all 9 matrix decomposition types available in the system:
 * </p>
 * <ul>
 *   <li><strong>LU Decomposition</strong> - For general square matrices (most common method)</li>
 *   <li><strong>QR Decomposition</strong> - For well-conditioned square matrices</li>
 *   <li><strong>Cholesky Decomposition</strong> - For symmetric positive definite matrices</li>
 *   <li><strong>SVD Decomposition</strong> - For ill-conditioned or singular matrices</li>
 *   <li><strong>Tridiagonal Decomposition</strong> - For symmetric matrices</li>
 *   <li><strong>Hessenberg Decomposition</strong> - For large square matrices</li>
 *   <li><strong>Schur Decomposition</strong> - For eigenvalue-related problems with medium-sized matrices</li>
 *   <li><strong>Eigen Decomposition</strong> - For eigenvalue-related problems with small matrices</li>
 *   <li><strong>Bidiagonal Decomposition</strong> - For large matrices and SVD computation</li>
 * </ul>
 * 
 * <h3>Algorithm Selection Strategy</h3>
 * <p>
 * The calculator employs a hierarchical selection strategy based on matrix properties:
 * </p>
 * <ol>
 *   <li><strong>Matrix Type Analysis</strong> - Determine if matrix is symmetric, positive definite, etc.</li>
 *   <li><strong>Condition Assessment</strong> - Evaluate numerical conditioning and singularity</li>
 *   <li><strong>Size Evaluation</strong> - Consider matrix dimensions for performance optimization</li>
 *   <li><strong>Specialized Detection</strong> - Identify if matrix requires specific decomposition</li>
 *   <li><strong>Performance Optimization</strong> - Select fastest applicable method</li>
 *   <li><strong>Fallback Mechanism</strong> - Use robust methods (SVD) when others fail</li>
 * </ol>
 * 
 * <h3>Performance Characteristics</h3>
 * <ul>
 *   <li><strong>Speed</strong>: LU > Cholesky > QR > Tridiagonal > Hessenberg > Schur > Eigen > Bidiagonal > SVD</li>
 *   <li><strong>Stability</strong>: SVD > QR ≥ Cholesky > LU > Schur > Hessenberg > Tridiagonal > Eigen > Bidiagonal</li>
 *   <li><strong>Applicability</strong>: LU (any square) > SVD (any matrix) > others (specialized)</li>
 * </ul>
 * 
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 * // Compute the determinant of a matrix
 * IMatrix<Double> A = Linalg.matrix(new double[][]{{2, 1}, {1, 2}});
 * double det = DeterminantSolver.compute(A);
 * }
 * </pre>
 * 
 * <h3>References</h3>
 * <ul>
 *   <li>Golub, G. H., &amp; Van Loan, C. F. (2013). Matrix computations (4th ed.). Johns Hopkins University Press.</li>
 *   <li>Press, W. H., Teukolsky, S. A., Vetterling, W. T., &amp; Flannery, B. P. (2007). Numerical recipes: The art of scientific computing (3rd ed.). Cambridge University Press.</li>
 *   <li>Horn, R. A., &amp; Johnson, C. R. (2012). Matrix analysis (2nd ed.). Cambridge University Press.</li>
 * </ul>
 * 
 * @since 2.0
 * @see Decomps For creating specific decomposition instances
 * @see IDecompositionSolver For individual solver interfaces
 */
public class DeterminantSolver {
    
    /** Default epsilon for numerical comparisons. */
    private static final double DEFAULT_EPSILON = 1e-12;
    
    /** Default threshold for condition number to determine ill-conditioning. */
    private static final double CONDITION_THRESHOLD = 1e12;
    
    /** Threshold for matrix size to determine when to use specialized algorithms. */
    private static final int LARGE_MATRIX_THRESHOLD = 100;
    
    /**
     * Compute the determinant of a matrix by automatically selecting the most
     * appropriate decomposition method based on matrix properties.
     * 
     * @param A The matrix to compute determinant for
     * @return The determinant of matrix A
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     * @throws IllegalArgumentException if matrix is not square
     */
    public static double compute(IMatrix<Double> A) {
        // Validate input matrix
        if (A == null) {
            throw new IllegalArgumentException("Input matrix cannot be null");
        }
        
        // Check if matrix is square
        if (A.rows() != A.cols()) {
            throw new IllegalArgumentException("Matrix must be square to compute determinant");
        }
        
        // 1. First check if matrix is symmetric positive definite
        if (isSymmetricPositiveDefinite(A)) {
            try {
                // Symmetric positive definite matrix, use Cholesky decomposition
                ICholeskyDecomposition chol = Decomps.createCholesky();
                chol.decompose(A);
                return chol.getDeterminant();
            } catch (Exception e) {
                // If Cholesky fails, fall back to other methods
            }
        }
        
        // 2. Check if matrix is symmetric
        if (isSymmetric(A)) {
            try {
                // Symmetric matrix, use Tridiagonal decomposition
                ITridiagonalDecomposition tri = Decomps.createTridiagonal();
                tri.decompose(A);
                return tri.getDeterminant();
            } catch (Exception e) {
                // If Tridiagonal fails, fall back to other methods
            }
        }
        
        // 3. Check if matrix is a large square matrix
        if (A.rows() > LARGE_MATRIX_THRESHOLD) {
            try {
                // Large square matrix, use Hessenberg decomposition for efficiency
                IHessenbergDecomposition hess = Decomps.createHessenberg();
                hess.decompose(A);
                return hess.getDeterminant();
            } catch (Exception e) {
                // If Hessenberg fails, fall back to other methods
            }
        }
        
        // 4. Check if matrix is a square matrix for general purpose determinant calculation
        if (A.rows() == A.cols()) {
            try {
                // Square matrix, try LU decomposition first (common method for determinant)
                ILUDecomposition lu = Decomps.createLU();
                lu.decompose(A);
                return lu.getDeterminant();
            } catch (Exception e) {
                // If LU fails, try QR decomposition
                try {
                    IQRDecomposition qr = Decomps.createQR();
                    qr.decompose(A);
                    return qr.getDeterminant();
                } catch (Exception e2) {
                    // If QR also fails, fall back to other methods
                }
            }
        }
        
        // 5. Check if matrix is well-conditioned
        if (isWellConditioned(A)) {
            try {
                // Well-conditioned square matrix, use QR decomposition
                IQRDecomposition qr = Decomps.createQR();
                qr.decompose(A);
                return qr.getDeterminant();
            } catch (Exception e) {
                // If QR fails, fall back to other methods
            }
        }
        
        // 6. For eigenvalue-related problems, use Eigen decomposition
        if (isForEigenvalueProblem(A)) {
            try {
                IEigenDecomposition eigen = Decomps.createEigen();
                eigen.decompose(A);
                return eigen.getDeterminant();
            } catch (Exception e) {
                // If Eigen fails, fall back to other methods
            }
        }
        
        // 7. For Schur form problems, use Schur decomposition
        if (isForSchurProblem(A)) {
            try {
                ISchurDecomposition schur = Decomps.createSchur();
                schur.decompose(A);
                return schur.getDeterminant();
            } catch (Exception e) {
                // If Schur fails, fall back to other methods
            }
        }
        
        // 8. For bidiagonal form problems, use Bidiagonal decomposition
        if (isForBidiagonalProblem(A)) {
            try {
                IBidiagonalDecomposition bidiag = Decomps.createBidiagonal();
                bidiag.decompose(A);
                return bidiag.getDeterminant();
            } catch (Exception e) {
                // If Bidiagonal fails, fall back to other methods
            }
        }
        
        // 9. For ill-conditioned matrices or when other methods fail, use SVD
        try {
            ISVDDecomposition svd = Decomps.createSVD();
            svd.decompose(A);
            return svd.getDeterminant();
        } catch (Exception e) {
            throw new DecompositionFailedException(
                "Failed to compute matrix determinant using any decomposition method", 
                "DeterminantSolver", 
                "All decomposition methods failed", 
                -1, 
                Double.NaN, 
                e);
        }
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
            // First check if matrix is square
            if (matrix.rows() != matrix.cols()) {
                return false;
            }
            
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
            return matrix.rows() <= 50; // Small matrices are more likely to be used for eigenvalue problems
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
            return matrix.rows() > 50 && matrix.rows() <= LARGE_MATRIX_THRESHOLD; // Medium-sized matrices
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
            return matrix.rows() > LARGE_MATRIX_THRESHOLD;
        } catch (Exception e) {
            return false;
        }
    }
}