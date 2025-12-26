package com.yishape.lab.math.linalg.solver;

import com.yishape.lab.math.linalg.IMatrix;
import com.yishape.lab.math.linalg.decomposition.Decomps;
import com.yishape.lab.math.linalg.decomposition.DecompositionFailedException;
import com.yishape.lab.math.util.RerePrecision;

/**
 * Unified matrix rank calculator that automatically selects the most appropriate
 * decomposition method based on matrix properties.
 * <p>
 * This calculator analyzes the input matrix characteristics and selects the optimal
 * decomposition algorithm for computing the matrix rank rank(A). It provides
 * a unified interface for rank calculation regardless of matrix type or condition,
 * automatically choosing the most efficient and numerically stable method available.
 * </p>
 * 
 * <h3>Supported Decomposition Methods</h3>
 * <p>
 * The calculator supports all 9 matrix decomposition types available in the system:
 * </p>
 * <ul>
 *   <li><strong>SVD Decomposition</strong> - For any matrix (most accurate method)</li>
 *   <li><strong>QR Decomposition</strong> - For well-conditioned square matrices</li>
 *   <li><strong>LU Decomposition</strong> - For general square matrices</li>
 *   <li><strong>Cholesky Decomposition</strong> - For symmetric positive definite matrices</li>
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
 *   <li><strong>Size Evaluation</strong> - Consider matrix dimensions for performance optimization</li>
 *   <li><strong>Specialized Detection</strong> - Identify if matrix requires specific decomposition</li>
 *   <li><strong>Performance Optimization</strong> - Select fastest applicable method</li>
 *   <li><strong>Fallback Mechanism</strong> - Use robust methods (SVD) when others fail</li>
 * </ol>
 * 
 * <h3>Performance Characteristics</h3>
 * <ul>
 *   <li><strong>Accuracy</strong>: SVD > Eigen ≥ Schur > others (approximate)</li>
 *   <li><strong>Speed</strong>: QR > LU > Cholesky > Tridiagonal > Hessenberg > Schur > Eigen > Bidiagonal > SVD</li>
 *   <li><strong>Applicability</strong>: SVD (any matrix) > others (specialized)</li>
 * </ul>
 * 
 * <h3>Usage Examples</h3>
 * <pre>
 * {@code
 * // Compute the rank of a matrix
 * IMatrix<Double> A = Linalg.matrix(new double[][]{{2, 1}, {1, 2}});
 * int rank = RankSolver.compute(A);
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
public class RankSolver {
    
    /** Default epsilon for numerical comparisons. */
    private static final double DEFAULT_EPSILON = 1e-12;
    
    /** Threshold for matrix size to determine when to use specialized algorithms. */
    private static final int LARGE_MATRIX_THRESHOLD = 100;
    
    /**
     * Compute the rank of a matrix by automatically selecting the most
     * appropriate decomposition method based on matrix properties.
     * 
     * @param A The matrix to compute rank for
     * @return The rank of matrix A
     * @throws DecompositionFailedException if no suitable decomposition method can be applied
     */
    public static int compute(IMatrix<Double> A) {
        // Validate input matrix
        if (A == null) {
            throw new IllegalArgumentException("Input matrix cannot be null");
        }
        
        // For rank calculation, SVD is the most accurate method
        try {
            // Try SVD decomposition first as it's the most accurate for rank calculation
            var svd = Decomps.createSVD();
            svd.decompose(A);
            return svd.getRank();
        } catch (Exception e) {
            // If SVD fails, fall back to other methods
        }
        
        // 1. Check if matrix is symmetric positive definite
        if (isSymmetricPositiveDefinite(A)) {
            try {
                // Symmetric positive definite matrix, use Cholesky decomposition
                var chol = Decomps.createCholesky();
                chol.decompose(A);
                return chol.getRank();
            } catch (Exception e) {
                // If Cholesky fails, fall back to other methods
            }
        }
        
        // 2. Check if matrix is symmetric
        if (isSymmetric(A)) {
            try {
                // Symmetric matrix, use Tridiagonal decomposition
                var tri = Decomps.createTridiagonal();
                tri.decompose(A);
                return tri.getRank();
            } catch (Exception e) {
                // If Tridiagonal fails, fall back to other methods
            }
        }
        
        // 3. Check if matrix is a large square matrix
        if (A.rows() == A.cols() && A.rows() > LARGE_MATRIX_THRESHOLD) {
            try {
                // Large square matrix, use Hessenberg decomposition for efficiency
                var hess = Decomps.createHessenberg();
                hess.decompose(A);
                return hess.getRank();
            } catch (Exception e) {
                // If Hessenberg fails, fall back to other methods
            }
        }
        
        // 4. Check if matrix is a square matrix for general purpose rank calculation
        if (A.rows() == A.cols()) {
            try {
                // Square matrix, try LU decomposition first
                var lu = Decomps.createLU();
                lu.decompose(A);
                return lu.getRank();
            } catch (Exception e) {
                // If LU fails, try QR decomposition
                try {
                    var qr = Decomps.createQR();
                    qr.decompose(A);
                    return qr.getRank();
                } catch (Exception e2) {
                    // If QR also fails, fall back to other methods
                }
            }
        }
        
        // 5. For eigenvalue-related problems, use Eigen decomposition
        if (isForEigenvalueProblem(A)) {
            try {
                var eigen = Decomps.createEigen();
                eigen.decompose(A);
                return eigen.getRank();
            } catch (Exception e) {
                // If Eigen fails, fall back to other methods
            }
        }
        
        // 6. For Schur form problems, use Schur decomposition
        if (isForSchurProblem(A)) {
            try {
                var schur = Decomps.createSchur();
                schur.decompose(A);
                return schur.getRank();
            } catch (Exception e) {
                // If Schur fails, fall back to other methods
            }
        }
        
        // 7. For bidiagonal form problems, use Bidiagonal decomposition
        if (isForBidiagonalProblem(A)) {
            try {
                var bidiag = Decomps.createBidiagonal();
                bidiag.decompose(A);
                return bidiag.getRank();
            } catch (Exception e) {
                // If Bidiagonal fails, fall back to other methods
            }
        }
        
        // 8. For ill-conditioned or problematic matrices, try Gaussian elimination
        if (isIllConditioned(A)) {
            try {
                return rankByGaussianElimination(A);
            } catch (Exception e) {
                // If Gaussian elimination fails, continue to other methods
            }
        }
        
        // 9. For any remaining cases, try QR as a last resort before throwing exception
        try {
            var qr = Decomps.createQR();
            qr.decompose(A);
            return qr.getRank();
        } catch (Exception e) {
            throw new DecompositionFailedException(
                "Failed to compute matrix rank using any decomposition method", 
                "RankSolver", 
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
     * Check if a matrix is ill-conditioned.
     * 
     * @param matrix The matrix to check
     * @return true if the matrix is ill-conditioned, false otherwise
     */
    private static boolean isIllConditioned(IMatrix<Double> matrix) {
        try {
            // First check if matrix is square
            if (matrix.rows() != matrix.cols()) {
                return false;
            }
            
            // A matrix is ill-conditioned if its condition number is too large
            // We'll use a threshold of 1e12 for ill-conditioning
            double conditionThreshold = 1e12;
            
            // Try to get condition number from the matrix
            try {
                double conditionNumber = matrix.cond().doubleValue();
                return conditionNumber >= conditionThreshold;
            } catch (Exception e) {
                // If we can't compute condition number directly, assume it's not ill-conditioned
                return false;
            }
        } catch (Exception e) {
            // If any check fails, assume it's not ill-conditioned
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
    
    
    /**
     * 使用高斯消元法计算矩阵秩的辅助方法 / Helper method for computing rank using Gaussian
     * elimination
     */
    private static int rankByGaussianElimination(IMatrix<Double> matrix0) {
        double[][] matrix = matrix0.copy().toDoubleArray();
                int rows = matrix.length;
        int cols = matrix[0].length;

        int rank = 0;
        final double tolerance = 1e-10;

        for (int col = 0; col < cols && rank < rows; col++) {
            // 寻找主元
            int pivotRow = rank;
            for (int row = rank + 1; row < rows; row++) {
                if (!RerePrecision.equalsZero(matrix[row][col], 1e-12) &&
                    (RerePrecision.equalsZero(matrix[pivotRow][col], 1e-12) ||
                     Math.abs(matrix[row][col]) > Math.abs(matrix[pivotRow][col]))) {
                    pivotRow = row;
                }
            }

            if (!RerePrecision.equalsZero(matrix[pivotRow][col], tolerance)) {
                // 交换行
                if (pivotRow != rank) {
                    double[] temp = matrix[rank];
                    matrix[rank] = matrix[pivotRow];
                    matrix[pivotRow] = temp;
                }

                // 消元
                for (int row = rank + 1; row < rows; row++) {
                    double factor = matrix[row][col] / matrix[rank][col];
                    for (int c = col; c < cols; c++) {
                        matrix[row][c] -= factor * matrix[rank][c];
                    }
                }
                rank++;
            }
        }

        return rank;
    }
}